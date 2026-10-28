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
 * land exactly; one that does not (no checkpoint there) refeeds the missing positions (RqSpecStats.refed).
 * Prompt lookup (TODO G6, RqSpecOpts.lookup): before the draft, the guesses may come from the target's own tokens: the
 * longest recent match of its last tokens + t earlier in the sequence (RqLookup, below), the tokens after it; when it
 * finds one (>= lookup_min tokens), the draft model is not run that round (it catches up on the next one it drafts:
 * draft_follow's common prefix). The verify and accept steps are the same, so is the output. */
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

// ---------------------------------------------------------------------------------------------- prompt lookup
// A hash table of chains over the pairs of consecutive tokens (as LZ77 compressors index their matches): head[h] the
// latest position i whose pair (toks[i - 1], toks[i]) hashes to h, prev[i] the one before it with the same hash (-1:
// none). A lookup walks the chain of the pair (last token, next token) from the most recent, at most RQ_LOOKUP_CHAIN
// positions, and keeps the longest match backwards (up to RQ_LOOKUP_LMAX tokens): the same choice as trying the
// longest n-gram first (n = LMAX .. 2, the most recent occurrence of each) while the chain reaches it
struct RqLookup {
    int *head, *prev;
    int hbits, cap, n; // n: tokens indexed (the pairs ending at 1 .. n - 1)
};

static inline unsigned lk_hash(const RqLookup *lk, int a, int b) {
    uint64_t x = ((uint64_t)(uint32_t)a << 32 | (uint32_t)b) * 0x9E3779B97F4A7C15ull;
    return (unsigned)(x >> (64 - lk->hbits));
}

RqLookup *rq_lookup_new(int capacity) {
    if (capacity < 1) capacity = 1;
    RqLookup *lk = calloc(1, sizeof(*lk));
    if (!lk) return NULL;
    lk->hbits = 10;
    while (lk->hbits < 24 && (1 << lk->hbits) < 2 * capacity) lk->hbits++;
    lk->cap = capacity;
    lk->head = malloc(sizeof(int) << lk->hbits);
    lk->prev = malloc((size_t)capacity * sizeof(int));
    if (!lk->head || !lk->prev) {
        rq_lookup_free(lk);
        return NULL;
    }
    rq_lookup_reset(lk);
    return lk;
}

void rq_lookup_free(RqLookup *lk) {
    if (!lk) return;
    free(lk->head);
    free(lk->prev);
    free(lk);
}

void rq_lookup_reset(RqLookup *lk) {
    memset(lk->head, 0xff, sizeof(int) << lk->hbits); // (-1)
    lk->n = 0;
}

int rq_lookup_add(RqLookup *lk, const int *toks, int n) {
    if (n > lk->cap || n < lk->n) return RQ_ERR_ARG;
    for (int i = lk->n > 1 ? lk->n : 1; i < n; i++) {
        unsigned h = lk_hash(lk, toks[i - 1], toks[i]);
        lk->prev[i] = lk->head[h];
        lk->head[h] = i;
    }
    lk->n = n;
    return RQ_OK;
}

int rq_lookup_draft(const RqLookup *lk, const int *toks, int t, int kmax, int min_len, int *guess, int *len) {
    int n = lk->n, best = -1, bl = 0;
    *len = 0;
    if (n < 2 || kmax < 1) return 0;
    int a = toks[n - 1];
    for (int j = lk->head[lk_hash(lk, a, t)], c = 0; j >= 0 && c < RQ_LOOKUP_CHAIN; j = lk->prev[j], c++) {
        // j: the pair (toks[j - 1], toks[j]), 1 <= j <= n - 1; the match: backwards from j - 1 vs from n - 1
        if (toks[j] != t || toks[j - 1] != a) continue; // (another pair, the same hash)
        int l = 2;
        while (l < RQ_LOOKUP_LMAX && j - l >= 0 && toks[j - l] == toks[n - l]) l++;
        if (l > bl) bl = l, best = j;
        if (bl == RQ_LOOKUP_LMAX) break;
    }
    if (best < 0 || bl < min_len) return 0;
    *len = bl;
    // the tokens after it in toks[0..n-1], t: positions best + 1 .. n (n: t)
    int m = n - best < kmax ? n - best : kmax;
    for (int i = 0; i < m; i++)
        guess[i] = best + 1 + i < n ? toks[best + 1 + i] : t;
    return m;
}

static void ck_slots(RqSeq *s, int n) {
    // Qwen3.5: at least n checkpoint slots
    if (s->m->hy.on && s->nck < n) rq_seq_set_checkpoints(s, n);
}

int rq_spec_run(RqGen *g, RqCtx *c, float *logits, RqCtx *dc, RqSeq *ds, const RqSpecOpts *o, RqSpecStats *st) {
    RqSeq *s = g->s;
    RqSpecStats z;
    if (!st) st = &z;
    int k = o ? o->k : 0, kl = o ? o->lookup : 0, V = s->m->config.vocab_size, e = RQ_OK;
    int lmin = o && o->lookup_min > 0 ? o->lookup_min : RQ_LOOKUP_MIN_DEFAULT;
    float tau = o ? o->tau : 0;
    if (k < 0 || k > RQ_SPEC_K_MAX || kl < 0 || kl > RQ_SPEC_K_MAX || (o && o->lookup_min < 0) ||
        (k > 0 && (!dc || !ds || ds == s)) || !logits)
        return RQ_ERR_ARG;
    if (k > 0 && rq_spec_check(s->m, ds->m, NULL, 0) != RQ_OK) return RQ_ERR_ARG;
    double t0 = rq_now();
    float *out = NULL;
    Buf buf = { NULL, 0 };
    RqLookup *lk = NULL;
    int kmax = k > kl ? k : kl, guess[RQ_SPEC_K_MAX + 1], nck0 = s->nck;
    if (kmax > 0) {
        if (!(out = malloc((size_t)(kmax + 1) * V * sizeof(float)))) return RQ_ERR_NOMEM;
        ck_slots(s, kmax + 2);
        if (k > 0) ck_slots(ds, k + 2);
    }
    if (kl > 0 && !(lk = rq_lookup_new(rq_seq_capacity(s)))) {
        free(out);
        return RQ_ERR_NOMEM;
    }
    int r = rq_gen_step(g, c, rq_seq_len(s) - 1, logits);
    while (r > 0) {
        int P = rq_seq_len(s), t = g->token, cap = rq_seq_capacity(s);
        // guesses: at most k (kl from the lookup), the tokens left before max_new but one (the pass's last row gives
        // one more), the positions left (the pass's rows P..P + n), and the draft's capacity (its rows P..P + n - 1)
        int lim = r == RQ_GEN_LAST ? 0 : kmax, nd = 0, src = 0;
        if (g->max_new > 0 && lim > g->max_new - g->n_gen - 1) lim = g->max_new - g->n_gen - 1;
        if (lim > cap - 1 - P) lim = cap - 1 - P;
        if (kl > 0 && lim > 0) { // the lookup first: the sequence's tokens are indexed as they become final
            double td = rq_now();
            int ml;
            if (rq_lookup_add(lk, rq_seq_tokens(s), P) != RQ_OK) { // (a sequence cut below the indexed length)
                rq_lookup_reset(lk);
                rq_lookup_add(lk, rq_seq_tokens(s), P);
            }
            nd = rq_lookup_draft(lk, rq_seq_tokens(s), t, kl < lim ? kl : lim, lmin, guess + 1, &ml);
            if (g->cfg->allow) // (a guess the mask forbids cannot be sampled: the guesses stop before it)
                for (int j = 1; j <= nd; j++)
                    if (!g->cfg->allow[guess[j]]) { nd = j - 1; break; }
            src = nd > 0;
            st->t_draft += rq_now() - td;
        }
        int kk = src ? 0 : k < lim ? k : lim;
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
        st->lookup_passes += src;
        st->lookup_drafted += src ? nd : 0;
        // accept: step through the rows while the sample is the next row's token
        int keep = P + 1 + nd, fed = 0;
        for (int j = 0; j <= nd; j++) {
            r = rq_gen_step(g, c, P + j, rows + (size_t)j * V);
            if (r < 0) { e = r; break; }
            if (r == RQ_GEN_DONE) { keep = P + 1 + j; break; }
            if (j < nd && g->token == guess[j + 1]) { // (already in the cache at P + j + 1)
                st->accepted++;
                st->lookup_accepted += src;
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
    rq_lookup_free(lk);
    if (kmax > 0 && s->m->hy.on) {
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
    int k = o ? o->k : 0, kl = o ? o->lookup : 0;
    if (k == 0 && kl == 0) return rq_generate(c, s, tokens, n, cfg, max_new, cb, &st->gen);
    if (k < 0 || k > RQ_SPEC_K_MAX || kl < 0 || kl > RQ_SPEC_K_MAX || o->lookup_min < 0 ||
        (k > 0 && (!dc || !ds || ds == s)))
        return RQ_ERR_ARG;
    if (k > 0 && rq_spec_check(s->m, ds->m, NULL, 0) != RQ_OK) return RQ_ERR_ARG;
    double t0 = rq_now();
    float *logits = NULL;
    int e = rq_prefill(c, s, tokens, n, RQ_LOGITS_LAST, NULL, &logits);
    if (e != RQ_OK) return e;
    st->gen.n_prompt = n;
    // the draft follows the prompt (its first round feeds the sampled token; a draft context shorter than the prompt:
    // what fits, the loop guesses nothing past it)
    if (k > 0) {
        Buf buf = { NULL, 0 };
        int nd = rq_seq_len(s) < rq_seq_capacity(ds) ? rq_seq_len(s) : rq_seq_capacity(ds);
        e = draft_follow(dc, ds, rq_seq_tokens(s), nd, -1, 0, &buf, NULL, st);
        free(buf.v);
        if (e != RQ_OK) return e;
    }
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
