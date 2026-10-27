/* Speculative decoding (TODO S1, docs/qwen38-speculative.md): a small draft model guesses the next tokens
 * greedily, the target model checks them in one batched pass, and the output is exactly plain decoding's.
 *
 * One round, the target sequence at length P, t its last sampled token (not fed yet):
 *   draft   the draft sequence follows the target's tokens (the longest common prefix kept, the rest fed), then t
 *           and its own guesses d1, d2, ... (argmax of its logits; adaptive: it stops at the first guess whose
 *           top-1 probability is < tau), at most k, as many as fit before max_new and the context's end
 *   verify  [t, d1..dn] in one pass of the target (rq_prefill, RQ_LOGITS_ALL; no guess: one decode step)
 *   accept  rq_gen_step on the rows in order: row j holds the logits of the token at P + j, whatever it is; while
 *           the sampled token equals d_{j+1} it is already in the cache and the next row is valid; the first
 *           mismatch's sample (or the bonus sample of row n) is the next t; the target is cut after the last
 *           row used (rq_seq_truncate), the draft at the next round (common prefix)
 * Why exact: the rows of a batched pass are bit-identical to decode steps at those positions (the engine's
 * invariant), each accepted guess is the token plain decoding would have fed, and rq_gen_step draws from the
 * target sequence's RNG exactly as rq_generate does (one draw per row stepped through, none for the draft, which
 * picks by argmax), so tokens, RNG states, callbacks and stops are those of rq_generate. The draft only decides how
 * many rows a pass has.
 * Qwen3.5 (the recurrent state cannot be rewound): a checkpoint after each guess row of the target's pass and of
 * the draft's steps (rq_seq_checkpoint_rows; k + 2 slots: those, the prompt's, one spare), so that both truncations
 * land exactly; one that does not (no checkpoint there) refeeds the missing positions (RqSpecStats.refed). */
#include "internal.h"

static uint64_t vocab_hash(const RqModel *m) {
    // FNV-1a over every token's byte length and bytes (rq_token_bytes: "" past the tokenizer's tokens)
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < m->config.vocab_size; i++) {
        int len;
        const unsigned char *b = (const unsigned char *)rq_token_bytes(m, i, &len);
        for (int j = 0; j < 4; j++)
            h = (h ^ (unsigned char)(len >> 8 * j)) * 1099511628211ull;
        for (int j = 0; j < len; j++)
            h = (h ^ b[j]) * 1099511628211ull;
    }
    return h;
}

int rq_spec_check(const RqModel *target, const RqModel *draft, char *err, size_t errlen) {
    char z[1];
    if (!err) err = z, errlen = sizeof(z);
    err[0] = 0;
    if (!target || !draft) return RQ_ERR_ARG;
    if (!target->has_tok || !draft->has_tok) {
        snprintf(err, errlen, "speculative decoding: a model without its tokenizer");
        return RQ_ERR_STATE;
    }
    if (target->config.vocab_size != draft->config.vocab_size) {
        snprintf(err, errlen, "the draft model's vocabulary has %d tokens, the target's %d: not usable as a draft",
                 draft->config.vocab_size, target->config.vocab_size);
        return RQ_ERR_ARG;
    }
    uint64_t ht = vocab_hash(target), hd = vocab_hash(draft);
    if (ht != hd) {
        snprintf(err, errlen, "the draft model's token strings differ from the target's (vocabulary hash %016llx vs "
                 "%016llx, %d tokens each): not usable as a draft", (unsigned long long)hd, (unsigned long long)ht,
                 target->config.vocab_size);
        return RQ_ERR_ARG;
    }
    return RQ_OK;
}

static int draft_pick(const float *lg, int V, const unsigned char *allow, float tau, float *p1) {
    // the draft's guess: its argmax (the lowest id on ties; among the allowed tokens with a mask); with tau > 0 its
    // probability, 1 / sum exp(l - max), only compared with tau: the sum stops once it shows p1 < tau
    int best = -1;
    float mx = 0;
    for (int i = 0; i < V; i++)
        if ((!allow || allow[i]) && (best < 0 || lg[i] > mx)) best = i, mx = lg[i];
    *p1 = 1;
    if (best >= 0 && tau > 0) {
        float lim = 1 / tau, sum = 0;
        const int B = 4096;
        for (int i0 = 0; i0 < V && sum <= lim; i0 += B) {
            int i1 = i0 + B < V ? i0 + B : V;
            float part = 0;
            for (int i = i0; i < i1; i++)
                part += (!allow || allow[i]) ? expf(lg[i] - mx) : 0;
            sum += part;
        }
        *p1 = sum <= lim ? 1 / sum : 0;
    }
    return best;
}

typedef struct {
    int *v;
    int cap;
} Buf;

static int buf_fit(Buf *b, int n) {
    if (n <= b->cap) return RQ_OK;
    int *v = realloc(b->v, (size_t)n * sizeof(int));
    if (!v) return RQ_ERR_NOMEM;
    b->v = v;
    b->cap = n;
    return RQ_OK;
}

static int draft_follow(RqCtx *dc, RqSeq *ds, const int *toks, int n, int t, int ck_rows, Buf *buf, float **lg,
                        RqSpecStats *st) {
    // the draft sequence := toks[0..n-1] (+ t if t >= 0): its longest common prefix kept, the rest fed (logits of the
    // last row if lg); ck_rows > 0: Qwen3.5 checkpoints after the rows at n.. n + ck_rows - 1 (t and the guesses)
    int L = rq_seq_common_prefix(ds, toks, n), e;
    if (L < rq_seq_len(ds)) { // (Qwen3.5: back to a checkpoint <= L)
        if ((e = rq_seq_truncate(ds, L)) != RQ_OK) return e;
        st->refed += L - rq_seq_len(ds);
    }
    int Ld = rq_seq_len(ds), m = n - Ld + (t >= 0);
    if (ck_rows > 0 && (e = rq_seq_checkpoint_rows(ds, n, ck_rows)) != RQ_OK) return e;
    if (m == 0) return RQ_OK;
    st->draft_steps++;
    if (m == 1) return rq_decode(dc, ds, Ld < n ? toks[Ld] : t, lg);
    if ((e = buf_fit(buf, m)) != RQ_OK) return e;
    memcpy(buf->v, toks + Ld, (size_t)(n - Ld) * sizeof(int));
    if (t >= 0) buf->v[m - 1] = t;
    return rq_prefill(dc, ds, buf->v, m, lg ? RQ_LOGITS_LAST : RQ_LOGITS_NONE, NULL, lg);
}

static void ck_slots(RqSeq *s, int n) {
    // Qwen3.5: at least n checkpoint slots
    if (s->m->hy.on && s->nck < n) rq_seq_set_checkpoints(s, n);
}

int rq_spec_run(RqGen *g, RqCtx *c, float *logits, RqCtx *dc, RqSeq *ds, const RqSpecOpts *o, RqSpecStats *st) {
    RqSeq *s = g->s;
    RqSpecStats z;
    if (!st) st = &z;
    int k = o ? o->k : 0, V = s->m->config.vocab_size, e = RQ_OK;
    float tau = o ? o->tau : 0;
    if (k < 0 || k > RQ_SPEC_K_MAX || (k > 0 && (!dc || !ds || ds == s)) || !logits) return RQ_ERR_ARG;
    if (k > 0 && rq_spec_check(s->m, ds->m, NULL, 0) != RQ_OK) return RQ_ERR_ARG;
    double t0 = rq_now();
    float *out = NULL;
    Buf buf = { NULL, 0 };
    int guess[RQ_SPEC_K_MAX + 1], nck0 = s->nck;
    if (k > 0) {
        if (!(out = malloc((size_t)(k + 1) * V * sizeof(float)))) return RQ_ERR_NOMEM;
        ck_slots(s, k + 2);
        ck_slots(ds, k + 2);
    }
    int r = rq_gen_step(g, c, rq_seq_len(s) - 1, logits);
    while (r > 0) {
        int P = rq_seq_len(s), t = g->token, cap = rq_seq_capacity(s);
        // guesses: at most k, the tokens left before max_new but one (the pass's last row gives one more), the
        // positions left (the pass's rows P..P + n), and the draft's capacity (its rows P..P + n - 1)
        int kk = r == RQ_GEN_LAST ? 0 : k, nd = 0;
        if (g->max_new > 0 && kk > g->max_new - g->n_gen - 1) kk = g->max_new - g->n_gen - 1;
        if (kk > cap - 1 - P) kk = cap - 1 - P;
        if (k > 0 && kk > rq_seq_capacity(ds) - P) kk = rq_seq_capacity(ds) - P;
        if (kk > 0) {
            double td = rq_now();
            float *lg = NULL, p1;
            e = draft_follow(dc, ds, rq_seq_tokens(s), P, t, kk, &buf, &lg, st);
            while (e == RQ_OK) {
                int d = draft_pick(lg, V, g->cfg->allow, tau, &p1);
                if (d < 0 || (tau > 0 && p1 < tau)) break;
                guess[++nd] = d;
                if (nd == kk) break;
                st->draft_steps++;
                e = rq_decode(dc, ds, d, &lg);
            }
            st->t_draft += rq_now() - td;
            if (e != RQ_OK) break;
        }
        // verify [t, d1..dn] (RQ_GEN_LAST: t fed alone, no logits)
        double tv = rq_now();
        float *rows = NULL;
        guess[0] = t;
        if (nd == 0) {
            e = rq_decode(c, s, t, r == RQ_GEN_LAST ? NULL : &rows);
        } else {
            if ((e = rq_seq_checkpoint_rows(s, P, nd)) == RQ_OK)
                e = rq_prefill(c, s, guess, nd + 1, RQ_LOGITS_ALL, out, &rows);
        }
        st->t_verify += rq_now() - tv;
        if (e != RQ_OK || r == RQ_GEN_LAST) break;
        st->passes++;
        st->drafted += nd;
        // accept: step through the rows while the sample is the next row's token
        int keep = P + 1 + nd, fed = 0;
        for (int j = 0; j <= nd; j++) {
            r = rq_gen_step(g, c, P + j, rows + (size_t)j * V);
            if (r < 0) { e = r; break; }
            if (r == RQ_GEN_DONE) { keep = P + 1 + j; break; }
            if (j < nd && g->token == guess[j + 1]) { // (already in the cache at P + j + 1)
                st->accepted++;
                if (r == RQ_GEN_LAST) { keep = P + 2 + j; fed = 1; break; }
                continue;
            }
            keep = P + 1 + j;
            break;
        }
        if (e != RQ_OK) break;
        if (keep < rq_seq_len(s)) { // (Qwen3.5: lands on the checkpoint of row keep - 1 - P)
            if ((e = rq_seq_truncate(s, keep)) != RQ_OK) break;
            int L = rq_seq_len(s);
            if (L < keep) {
                st->refed += keep - L;
                if ((e = rq_prefill(c, s, guess + (L - P), keep - L, RQ_LOGITS_NONE, NULL, NULL)) != RQ_OK) break;
            }
        }
        if (fed) break;
    }
    free(out);
    free(buf.v);
    if (k > 0 && s->m->hy.on) {
        // the target's checkpoints as rq_generate leaves them: the rows' dropped, its slot count back (a later
        // truncation, e.g. a chat's next turn, lands where it would after plain decoding: the same tokens to prefill,
        // the same RNG draws). The draft keeps its own (they only save it refeeding)
        for (int i = 0; i < s->nck; i++)
            if (s->ck[i].rows) {
                s->ck[i].pos = -1;
                s->ck[i].rows = 0;
            }
        if (s->nck > nck0) rq_seq_set_checkpoints(s, nck0);
    }
    st->gen.n_gen = g->n_gen;
    if (e == RQ_OK && r < 0) e = r;
    st->gen.stop = e == RQ_OK ? g->stop : RQ_STOP_END;
    st->gen.t_gen = rq_now() - t0;
    return e;
}

int rq_generate_spec(RqCtx *c, RqSeq *s, RqCtx *dc, RqSeq *ds, const int *tokens, int n, const RqSampling *cfg,
                     int max_new, const RqSpecOpts *o, const RqCallbacks *cb, RqSpecStats *st) {
    RqSpecStats z;
    if (!st) st = &z;
    memset(st, 0, sizeof(*st));
    int k = o ? o->k : 0;
    if (k == 0) return rq_generate(c, s, tokens, n, cfg, max_new, cb, &st->gen);
    if (k < 0 || k > RQ_SPEC_K_MAX || !dc || !ds || ds == s) return RQ_ERR_ARG;
    if (rq_spec_check(s->m, ds->m, NULL, 0) != RQ_OK) return RQ_ERR_ARG;
    double t0 = rq_now();
    float *logits = NULL;
    int e = rq_prefill(c, s, tokens, n, RQ_LOGITS_LAST, NULL, &logits);
    if (e != RQ_OK) return e;
    st->gen.n_prompt = n;
    // the draft follows the prompt (its first round feeds the sampled token; a draft context shorter than the prompt:
    // what fits, the loop guesses nothing past it)
    Buf buf = { NULL, 0 };
    int nd = rq_seq_len(s) < rq_seq_capacity(ds) ? rq_seq_len(s) : rq_seq_capacity(ds);
    e = draft_follow(dc, ds, rq_seq_tokens(s), nd, -1, 0, &buf, NULL, st);
    free(buf.v);
    if (e != RQ_OK) return e;
    st->draft_steps = st->refed = 0; // (the loop's only: a chat's rollback to its checkpoint is not the loop's)
    RqGen g;
    rq_gen_start(&g, s, cfg, max_new, cb, n);
    st->gen.t_prefill = rq_now() - t0;
    RqGenStats gs = st->gen;
    e = rq_spec_run(&g, c, logits, dc, ds, o, st);
    st->gen.n_prompt = gs.n_prompt;
    st->gen.t_prefill = gs.t_prefill;
    return e;
}
