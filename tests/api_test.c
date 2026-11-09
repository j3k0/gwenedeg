/* Tests of the library API (src/runq.h), run by `make test` (tests/check.py, group api) on each model:
 *   tests/api_test MODEL [--digest | --digest-small [--long]] [--n65] [--kv] [--kv-type TYPE]
 * Prints PASS / FAIL lines, exit status 1 if any failed. The forward-pass checks are bitwise: a
 * sequence's logits must not depend on what else the context ran in between or batched with it
 * (sequences interleaved, several sequences in one rq_forward_batch, a prefill chunk next to decode
 * rows, positions past one attention block of 256), nor on truncating and re-prefilling.
 * Batched decode (section 9): N sequences decoding together, one row each, == each alone (rq_decode);
 * the batched passes also print DIGEST lines (a hash of their logits), which must be the same for any
 * thread count and build (check.py compares runs at other thread counts and of the generic build);
 * --digest: only those passes (no references, no PASS / FAIL lines); --digest-small: only the short
 * ones (for the slow generic build; --long: and the long ones); --n65: also 65 sequences in one pass
 * (references for 48 more sequences: one model is enough); --kv: also a bf16 KV cache and the bf16 dot-product
 * attention (bf16-dot, f16-dot: 3 long sequences); --kv-type: the KV cache type of all the other sequences (default
 * f16; e.g. bf16-dot: every path of the API in the bf16 dot-product attention, its digests across builds).
 *   tests/api_test --two MODEL_A MODEL_B [--gs MODEL_GS]
 * Two models in one process (a draft and a target of speculative decoding: one thread pool, one activation
 * group size): their prefills, decode steps, batched passes (several sequences, a prefill chunk next to decode
 * rows), truncations to checkpoints (Qwen3.5), copies, batched sampling and rq_generate, interleaved step by
 * step, give the logits hashes of each model alone (each run in a fresh process first: fork); then A is freed
 * and B goes on (== alone). Also: a sequence of one model on the other's context is refused, and with --gs a
 * model of another group size (the .bin: 64) is refused cleanly while A or B is loaded (they go on unchanged),
 * and loads once both are freed. DIGEST lines (a hash of each model's work) to compare builds.
 *   tests/api_test --spec TARGET DRAFT [--quick] [--other MODEL]
 * Speculative decoding (rq_generate_spec, TODO S1) == rq_generate: for each case (prompts with thinking off / on and
 * raw text; greedy and seeded sampling with top-p / top-k; stops at an end token, at max_new and at the end of the
 * context inside a verify pass, the draft's context shorter than the target's or than the prompt; a second call after
 * both were cut
 * back, as a chat's next turn; a raw text that copies a block of its prompt) and each draft setting (k = 1 / 3 / 8,
 * adaptive tau 0.5 with k 8; the prompt lookup alone, no draft model: 4 guesses, 30 with matches of >= 4 tokens;
 * combined: lookup 8 (>= 3) + k 3, lookup 3 + k 8 tau 0.5; auto (G7: RqSpecOpts.cost): k 3 + lookup 3 with the cost
 * models calibrated at the start, k 8 + lookup 8 with a cost model where guesses are nearly free; --quick: k 3,
 * adaptive, lookup 4, the first combined and the second auto, fewer cases): the tokens, the RNG state after each, every sampled logits vector's position and hash, the stop, the
 * cache's tokens and one more decode step's logits. Also k 0 == rq_generate, the vocabulary check (the draft itself
 * accepted; --other: a model of another vocabulary refused, loaded tokenizer-only), for Qwen3.5, that every rollback
 * landed on its checkpoint (RqSpecStats.refed 0), the copy's acceptance with the lookup alone (>= 3 tokens per pass)
 * and the lookup index (rq_lookup_*) against a brute-force search on synthetic sequences.
 *   tests/api_test MODEL --sample LOGITS_FILE < PLAN
 * rq_sample on the rows of a -m logits file, with the settings and RNG states of PLAN's lines ("T P K ROW
 * SEED..."): SAMPLE lines for tests/sampler_ref.py (the sampler's exact order replayed in numpy: top-k).
 *   tests/api_test --mtp TARGET [--quick] [--plain MODEL_WITHOUT_BLOCK] [--optional MODEL_WITH_OPTIONAL_NEXTN]
 * The MTP drafter's parts (TODO S9a): hidden_rows: RqSegment.hidden of every row of the target == the decode path's
 * (-b 1), from short segments (the batched decode rows), long chunks (prefill: rows with all / the last / no logits),
 * mixed passes; the logits with and without it equal. mtp_paths: the MTP view (rq_mtp_load) fed the target's hidden
 * rows shifted by one (h_in) gives the same logits and hidden rows batched, alone, through decode and mixed; h_in NULL
 * == zeros; a vocabulary subset == the full vocabulary on its rows, RQ_LOGIT_OFF elsewhere; truncate + refeed == straight.
 * mtp_load: no block / optional nextn tensors / the memory budget refused, a second view, views loaded and freed
 * twice, h_in on the target refused, a view's sequence on the target's context refused, its info / estimate / matrix
 * names. Prints "mtp digest: HEX" (the target's hidden rows and the view's logits and hidden rows of one prompt: the
 * same for any thread count, build and --quick). --quick: fewer variants (the generic build). TODO S9c.1:
 * decode_hidden/{target,view}: rq_decode_hidden == a one-row rq_forward_batch with hidden (h_in on the view) == the
 * decode path, logits and hidden bitwise, also without logits; mtp_follow/*: the view followed by a caller that batches
 * its rows (runq serve): rq_mtp_hin's h rule and count (hin_rule), rq_mtp_final / rq_mtp_keep / rq_mtp_kept
 * (final_keep: also a kept state past the view's end at its capacity), and --mtp-seqcheck's comparison on a saved
 * lockstep pair (seqcheck: passes, with the kept state saved; the view fed zeros for h: fails). TODO S9c.2: seqfile/*:
 * the view's kept state on disk (mtp_h: saved with it as version 2, restored by a whole load only; without it, or kept
 * before the last row: version 1), a corrupted state or extension and a file cut before the state refused
 * (mtp_h_corrupt)
 *   tests/api_test --mtp-seqcheck MODEL TARGET.rqseq VIEW.rqseq
 * A saved target sequence and its MTP view's (runq serve's RUNQ_TEST_MTP_DUMP) against a fresh lockstep run over the
 * target file's tokens (S9b's mtp_lockstep on files): the target's hidden rows of a fresh prefill, the view's prefill
 * over them shifted by one; the view's length and tokens == the target's, its KV cache bitwise over the whole
 * 16-position tiles; a view file with the kept state (version 2): the state == the target's last hidden row bitwise, a
 * probe row with it == the reference's. One PASS / FAIL line (mtp_seqcheck)
 *   tests/api_test --seqfile-out MODEL DIR, tests/api_test --seqfile-load MODEL FILE [--view]
 * A target sequence and its MTP view's (without and with the kept state) saved into DIR; a file's info and whole load
 * (their error codes, the length and the kept state). Built into 915894a's api_test too (tests/make_seqfile_v1ref.sh),
 * for tests/check_models.py's seqfile/v1_same (version 1 files byte for byte 915894a's) and seqfile/v2_on_v1_reader
 *   tests/api_test --mtp-dump MODEL TEXTFILE OUT.bin [-b N] [--vocab N]
 * The MTP view's logits teacher-forced on TEXTFILE's tokens (raw text, as runq -m logits): the target's prefill with
 * hidden rows, then the view's over the same tokens with h shifted by one (row 0: zeros), every row's logits, in
 * runq -m logits' format (int32 n, vocab, ids[n], float32 logits[n][vocab]); -b: the chunk (1: the decode path),
 * --vocab: RqMtpOpts.vocab. For tests/check_models.py's mtp_ref (vs tests/ref_qwen35.py mtp_forward) and mtp_top1 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/wait.h>
#endif
#include "../src/runq.h"

static int failed = 0;

static void report(int ok, const char *name, const char *detail) {
    printf("%s %s%s%s\n", ok ? "PASS" : "FAIL", name, detail && detail[0] ? "  " : "", detail ? detail : "");
    failed |= !ok;
}

static RqModel *M;
static RqModelInfo I;
static int V;

static void die(const char *what, int e) {
    printf("FAIL setup  %s: %s\n", what, rq_strerror(e));
    exit(1);
}

static RqTokens toks(const char *text) {
    RqTokens t = { 0 };
    int e = rq_tokenize(M, text, -1, &t);
    if (e != RQ_OK) die("rq_tokenize", e);
    return t;
}

static int KVT = RQ_KV_DEFAULT; // the main tests' KV cache type (--kv-type: e.g. bf16-dot, the bf16 dot-product attention)

static RqSeq *seq_new(int cap) {
    int e;
    RqSeq *s = rq_seq_new(M, cap, KVT, &e);
    if (!s) die("rq_seq_new", e);
    return s;
}

static int same(const float *a, const float *b, size_t n) { return !memcmp(a, b, n * sizeof(float)); }

static long rss_anon(void) {
    // RssAnon of /proc/self/status in bytes, -1 where unknown (not Linux)
    long kb = -1;
#ifdef __linux__
    FILE *f = fopen("/proc/self/status", "r");
    char line[256];
    while (f && fgets(line, sizeof(line), f))
        if (sscanf(line, "RssAnon: %ld kB", &kb) == 1) break;
    if (f) fclose(f);
#endif
    return kb < 0 ? -1 : kb * 1024;
}

static int seq_from(RqCtx *c, RqSeq *dst, const RqSeq *base, int n) {
    // dst := base's first n positions: rq_seq_copy; a model with linear-attention layers (Qwen3.5) keeps only a
    // checkpointed length there, so its prefix is prefilled again (the same bits as a copy would give)
    if (!I.linear_layers) return rq_seq_copy(dst, base, n);
    rq_seq_reset(dst);
    return n > 0 ? rq_prefill(c, dst, rq_seq_tokens(base), n, RQ_LOGITS_NONE, NULL, NULL) : RQ_OK;
}

static int ck_cont(RqCtx *c, RqSeq *s, int P, const int *u, const float *ref, float *out) {
    // s (on the tokens u) truncated to its checkpoint at P, exactly, and continued with u[P..P+5): every row == ref's
    if (rq_seq_keep(s, P) != P || rq_seq_truncate(s, P) != RQ_OK || rq_seq_len(s) != P) return 0;
    return rq_prefill(c, s, u + P, 5, RQ_LOGITS_ALL, out, NULL) == RQ_OK && same(out, ref + (size_t)P * V, (size_t)5 * V);
}

static float *logits_alone(RqCtx *c, const int *t, int n, const int *dec, int ndec) {
    // reference: a fresh sequence, all-rows prefill of t, then ndec decode steps; rows n + ndec
    float *out = malloc((size_t)(n + ndec) * V * sizeof(float)), *lg;
    RqSeq *s = seq_new(0);
    int e = rq_prefill(c, s, t, n, RQ_LOGITS_ALL, out, NULL);
    for (int i = 0; i < ndec && e == RQ_OK; i++) {
        e = rq_decode(c, s, dec[i], &lg);
        if (e == RQ_OK) memcpy(out + (size_t)(n + i) * V, lg, V * sizeof(float));
    }
    if (e != RQ_OK) die("reference run", e);
    rq_seq_free(s);
    return out;
}

static uint64_t hash_f(const float *x, size_t n) {
    // FNV-1a over the bytes: bitwise comparisons of logits vectors without keeping them
    const unsigned char *b = (const unsigned char *)x;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n * sizeof(float); i++)
        h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

// Batched decode rows. Sequence i of a set: the first plen(i) positions of a base sequence (seq_from:
// prefixes of different lengths, so the rows sit at different positions), then the tokens tk(i, 0),
// tk(i, 1) (sequences i >= BD_TWO: tk(i, 0) only), decoded together with the set's other sequences (one
// row each per rq_forward_batch pass)
#define BD_STEPS 2
#define BD_TWO 9
#define BD_SEQS 65
static int plen(int i, int lo, int span) { return lo + (i * 7) % span; }
static int tk(int i, int t) { return 1000 + 37 * i + 11 * t; }

static void bd_refs(RqCtx *c, RqSeq *base, int lo, int span, int n, int kvt, uint64_t (*h)[BD_STEPS]) {
    // each sequence alone: copy, then rq_decode per token (the single-sequence decode path)
    int e;
    RqSeq *s = rq_seq_new(M, 0, kvt, &e);
    if (!s) die("rq_seq_new", e);
    for (int i = 0; i < n; i++) {
        if ((e = seq_from(c, s, base, plen(i, lo, span))) != RQ_OK) die("rq_seq_copy", e);
        for (int t = 0; t < (i < BD_TWO ? BD_STEPS : 1); t++) {
            float *lg;
            if ((e = rq_decode(c, s, tk(i, t), &lg)) != RQ_OK) die("rq_decode", e);
            h[i][t] = hash_f(lg, V);
        }
    }
    rq_seq_free(s);
}

static uint64_t digest = 0; // (of every batched pass of section 9, in order)
static int bd_long = 0, bd_n65 = 0, bd_kv = 0; // --long, --n65, --kv

static int bd_batch(RqCtx *c, RqSeq **sq, RqSeq *base, int lo, int span, int n, uint64_t (*h)[BD_STEPS], float *out,
                    const char *name) {
    // the n sequences decoded together (sq[0..n-1] reused), compared with the references h (NULL: digest only)
    int ok = 1, tok[BD_SEQS];
    RqSegment sg[BD_SEQS];
    uint64_t dg = 0;
    for (int i = 0; i < n; i++)
        ok &= seq_from(c, sq[i], base, plen(i, lo, span)) == RQ_OK;
    for (int t = 0; t < (n <= BD_TWO ? BD_STEPS : 1) && ok; t++) {
        for (int i = 0; i < n; i++) {
            tok[i] = tk(i, t);
            sg[i] = (RqSegment) { sq[i], &tok[i], 1, RQ_LOGITS_LAST };
        }
        ok &= rq_forward_batch(c, sg, n, out, NULL) == RQ_OK;
        for (int i = 0; i < n && ok; i++) {
            uint64_t x = hash_f(out + (size_t)i * V, V);
            dg = dg * 31 + x;
            ok &= !h || x == h[i][t];
        }
    }
    printf("DIGEST %s %016llx\n", name, (unsigned long long)dg);
    digest = digest * 31 + dg;
    return ok;
}

// Verify segments (TODO S2: a speculative verify pass, the k + 1 rows of one sequence with every row's logits, takes
// the decode-rows path of rq_forward_batch): sequence base's first P positions, then the r tokens T[0..r-1] in one
// segment; references: the same tokens decoded one by one (rq_decode), then T[r] after them
#define VF_MAX 10
static void vf_refs(RqCtx *c, RqSeq *base, int P, const int *T, int r, uint64_t *h) {
    int e;
    RqSeq *s = seq_new(0);
    if ((e = seq_from(c, s, base, P)) != RQ_OK) die("seq_from", e);
    for (int j = 0; j <= r; j++) {
        float *lg;
        if ((e = rq_decode(c, s, T[j], &lg)) != RQ_OK) die("rq_decode", e);
        h[j] = hash_f(lg, V);
    }
    rq_seq_free(s);
}

static int vf_rollback(RqCtx *c, RqSeq *s, int P, const int *T, int r, const uint64_t *h) {
    // after the pass (s at P + r, a checkpoint per row: rq_seq_checkpoint_rows), from the last row back to the first:
    // truncated after row j, exactly (Qwen3.5: onto row j's checkpoint), the next token's logits == the references
    int ok = 1;
    for (int j = r - 1; j >= 0 && ok; j--) {
        float *lg;
        ok &= rq_seq_keep(s, P + 1 + j) == P + 1 + j && rq_seq_truncate(s, P + 1 + j) == RQ_OK && rq_seq_len(s) == P + 1 + j;
        ok &= ok && rq_decode(c, s, T[j + 1], &lg) == RQ_OK && hash_f(lg, V) == h[j + 1];
    }
    return ok;
}

static int on_tok(void *u, int t) {
    RqTokens *r = u;
    rq_tokens_push(r, &t, 1);
    return 0;
}

// rq_generate's callbacks recorded: each sampled logits vector (position, rq_logits_hash) and each reported
// token with the RNG state right after its draw
#define GR_MAX 512
typedef struct {
    RqSeq *s;
    int n, nl, tok[GR_MAX], pos[GR_MAX];
    unsigned long long rng[GR_MAX], lh[GR_MAX];
    int stop_at;  // on_token asks to stop at its stop_at-th token (0: never)
} GenRec;

static void gr_logits(void *u, int pos, const float *lg) {
    GenRec *r = u;
    if (r->nl < GR_MAX) {
        r->pos[r->nl] = pos;
        r->lh[r->nl++] = rq_logits_hash(lg, V);
    }
}

static int gr_token(void *u, int t) {
    GenRec *r = u;
    if (r->n < GR_MAX) {
        r->rng[r->n] = *rq_seq_rng(r->s);
        r->tok[r->n++] = t;
    }
    return r->stop_at > 0 && r->n == r->stop_at;
}

static int gr_same(const GenRec *a, const GenRec *b) {
    return a->n == b->n && a->nl == b->nl && !memcmp(a->tok, b->tok, a->n * sizeof(int)) &&
           !memcmp(a->rng, b->rng, a->n * sizeof(a->rng[0])) && !memcmp(a->pos, b->pos, a->nl * sizeof(int)) &&
           !memcmp(a->lh, b->lh, a->nl * sizeof(a->lh[0]));
}

static int gen_verify_loop(RqCtx *c, RqSeq *s, const int *p, int np, const RqSampling *cfg, int max_new,
                           const RqCallbacks *cb, const GenRec *ref, RqGen *g) {
    // TODO S1's loop with an oracle draft: the reply's tokens as guesses (ref), every 4th one wrong, k = 3: feed
    // [t, d1, d2, d3] in one pass (all rows' logits), rq_gen_step through the rows while the sample equals the
    // guess; at the first mismatch (or the end) the sequence is cut after the stepped row (Qwen3.5: back to the
    // checkpoint taken at the pass's start, then the accepted rows prefilled again) and the sample is the next t
    enum { K = 3 };
    float *out = malloc((size_t)(K + 1) * V * sizeof(float)), *lg;
    int e = rq_prefill(c, s, p, np, RQ_LOGITS_LAST, NULL, &lg);
    if (e != RQ_OK) { free(out); return e; }
    rq_gen_start(g, s, cfg, max_new, cb, np);
    int r = rq_gen_step(g, c, rq_seq_len(s) - 1, lg), done = 0;
    while (r > 0 && !done) {
        int pos = rq_seq_len(s) - 1, seg[K + 1], idx = g->n_gen; // (the next token is the reply's token idx)
        seg[0] = g->token;
        if (r == RQ_GEN_LAST) { e = rq_decode(c, s, seg[0], NULL); break; }
        for (int j = 1; j <= K; j++) {
            int i = idx + j - 1;
            seg[j] = i < ref->n ? ref->tok[i] : 1000 + j;
            if (i % 4 == 2) seg[j] = (seg[j] + 1) % V; // a wrong guess
        }
        if ((e = rq_seq_checkpoint_at(s, pos + 1)) != RQ_OK) break;
        if ((e = rq_prefill(c, s, seg, K + 1, RQ_LOGITS_ALL, out, NULL)) != RQ_OK) break;
        int j = 0, keep = 0;
        for (; j <= K; j++) { // row j: the logits of seg[j], at pos + 1 + j
            r = rq_gen_step(g, c, pos + 1 + j, out + (size_t)j * V);
            if (r < 0 || r == RQ_GEN_DONE) { keep = j + 1; done = 1; break; }
            if (j < K && g->token == seg[j + 1]) { // accepted (already fed)
                if (r == RQ_GEN_LAST) { keep = j + 2; done = 1; break; }
                continue;
            }
            keep = j + 1; // rejected: g->token is the next t (RQ_GEN_LAST: fed alone below)
            break;
        }
        if (r < 0) { e = r; break; }
        if ((e = rq_seq_truncate(s, pos + 1 + keep)) != RQ_OK) break;
        int L = rq_seq_len(s);
        if (L < pos + 1 + keep && (e = rq_prefill(c, s, seg + (L - pos - 1), pos + 1 + keep - L, RQ_LOGITS_NONE, NULL, NULL)) != RQ_OK)
            break;
    }
    free(out);
    return e != RQ_OK ? e : r < 0 ? r : RQ_OK;
}

static void batch_tests(RqCtx *c, RqTokens A, RqTokens B, RqTokens C, const float *refB, int only_digest);
static int two_main(int argc, char **argv);
static int spec_main(int argc, char **argv);
static int mtp_main(int argc, char **argv);
static int mtp_dump_main(int argc, char **argv);
static int mtp_seqcheck_main(int argc, char **argv);

static int sample_main(int argc, char **argv) {
    // api_test MODEL --sample LOGITS_FILE < PLAN: rq_sample on rows of a -m logits file (int32 n, vocab, ids[n],
    // float32 logits[n][vocab]); each PLAN line "T P K ROW SEED..." samples row ROW at temperature T, top-p P,
    // top-k K once per SEED (the RNG state set to it) and prints "SAMPLE ROW token..." (tests/sampler_ref.py
    // replays them in numpy)
    (void)argc;
    char err[512];
    rq_threads_init(1);
    if (!(M = rq_model_load(argv[1], NULL, err, sizeof(err)))) { printf("FAIL setup  %s\n", err); return 1; }
    rq_model_info(M, &I);
    V = I.vocab_size;
    FILE *f = fopen(argv[3], "rb");
    int32_t hd[2];
    if (!f || fread(hd, 4, 2, f) != 2 || hd[1] != V || hd[0] < 1) { printf("FAIL setup  %s: not a logits file of this vocabulary\n", argv[3]); return 1; }
    int n = hd[0], e;
    float *lg = malloc((size_t)n * V * sizeof(float)), *row = malloc((size_t)V * sizeof(float));
    if (fseek(f, 4L * n, SEEK_CUR) || fread(lg, sizeof(float), (size_t)n * V, f) != (size_t)n * V) { printf("FAIL setup  short file\n"); return 1; }
    fclose(f);
    RqCtx *c = rq_ctx_new(M, 1, 0, &e);
    RqSeq *s = rq_seq_new(M, 16, RQ_KV_DEFAULT, &e);
    if (!c || !s) die("rq_ctx_new / rq_seq_new", e);
    RqSampling cfg = { 0 };
    int r, k;
    while (scanf("%f %f %d %d", &cfg.temperature, &cfg.topp, &cfg.top_k, &r) == 4) {
        if (r < 0 || r >= n) { printf("FAIL setup  row %d of %d\n", r, n); return 1; }
        printf("SAMPLE %d", r);
        unsigned long long seed;
        while ((k = getchar()) == ' ' || k == '\t')
            ;
        for (ungetc(k, stdin); k != '\n' && k != EOF && scanf("%llu", &seed) == 1;) {
            memcpy(row, lg + (size_t)r * V, V * sizeof(float));
            rq_seq_seed(s, seed);
            printf(" %d", rq_sample(c, s, row, &cfg));
            while ((k = getchar()) == ' ' || k == '\t')
                ;
            ungetc(k, stdin);
        }
        printf("\n");
    }
    free(lg);
    free(row);
    rq_seq_free(s);
    rq_ctx_free(c);
    rq_model_free(M);
    return 0;
}

// seqfile-out: begin (tests/make_seqfile_v1ref.sh copies this block into 915894a's api_test.c: its API only, plus
// rq_mtp_keep where runq.h has it)
static int sfo_rows(RqCtx *c, RqSeq *s, const int *t, int n, int dim, float *H, const float *hin) {
    // t on s in passes of 64 rows: their hidden rows into H, their h inputs from hin (each NULL: none)
    int e = RQ_OK;
    for (int q = 0; q < n && e == RQ_OK; q += 64) {
        RqSegment g = { s, t + q, n - q < 64 ? n - q : 64, RQ_LOGITS_NONE, H ? H + (size_t)q * dim : NULL, hin ? hin + (size_t)q * dim : NULL };
        e = rq_forward_batch(c, &g, 1, NULL, NULL);
    }
    return e;
}

static int seqfile_out_main(int argc, char **argv) {
    // api_test --seqfile-out MODEL DIR: DIR/target.rqseq (300 positions: two attention blocks; seeded; Qwen3.5: the
    // checkpoints at 40 and 260) and, for a file with an MTP block, its view's over the same tokens (h shifted by one):
    // DIR/view.rqseq (no kept state) and, in a build with rq_mtp_keep, DIR/view-h.rqseq (the target's state at 299 kept).
    // The files of seqfile.c's version 1 must be those 915894a writes (tests/check_models.py's seqfile/v1_same)
    if (argc != 4) {
        fprintf(stderr, "usage: api_test --seqfile-out MODEL DIR\n");
        return 2;
    }
    char err[512], p[1024];
    int n = 300, e, *t = malloc((size_t)n * sizeof(int));
    rq_threads_init(0);
    RqModel *m = rq_model_load(argv[2], NULL, err, sizeof(err));
    if (!m || !t) { fprintf(stderr, "%s\n", m ? "out of memory" : err); return 1; }
    RqModelInfo mi;
    rq_model_info(m, &mi);
    for (int i = 0; i < n; i++) t[i] = 100 + (int)((i * 7919u + 13u) % 3000u);
    float *H = malloc((size_t)n * mi.dim * sizeof(float)), *hin = calloc((size_t)n * mi.dim, sizeof(float));
    RqCtx *c = rq_ctx_new(m, 64, 0, &e);
    RqSeq *s = c ? rq_seq_new(m, 512, RQ_KV_DEFAULT, &e) : NULL;
    if (!H || !hin || !s) { fprintf(stderr, "setup: %s\n", rq_strerror(e)); return 1; }
    rq_seq_seed(s, 0x5eedf11eULL);
    if (mi.linear_layers && (rq_seq_checkpoint_at(s, 40) != RQ_OK || rq_seq_checkpoint_at(s, 260) != RQ_OK)) return 1;
    snprintf(p, sizeof(p), "%s/target.rqseq", argv[3]);
    if ((e = sfo_rows(c, s, t, n, mi.dim, H, NULL)) != RQ_OK || (e = rq_seq_save(s, p)) != RQ_OK) { fprintf(stderr, "%s: %s\n", p, rq_strerror(e)); return 1; }
    printf("seqfile-out: %s\n", p);
    RqMtpOpts mo = { 0, -1 };
    RqModel *v = rq_mtp_load(m, &mo, err, sizeof(err));
    if (v) {
        RqCtx *vc = rq_ctx_new(v, 64, 0, &e);
        RqSeq *vs = vc ? rq_seq_new(v, 512, RQ_KV_DEFAULT, &e) : NULL;
        if (!vs) { fprintf(stderr, "the view: %s\n", rq_strerror(e)); return 1; }
        memcpy(hin + mi.dim, H, (size_t)(n - 1) * mi.dim * sizeof(float));
        snprintf(p, sizeof(p), "%s/view.rqseq", argv[3]);
        if ((e = sfo_rows(vc, vs, t, n, mi.dim, NULL, hin)) != RQ_OK || (e = rq_seq_save(vs, p)) != RQ_OK) { fprintf(stderr, "%s: %s\n", p, rq_strerror(e)); return 1; }
        printf("seqfile-out: %s\n", p);
#ifdef RQ_MTP_VOCAB_DEFAULT // (rq_mtp_keep: TODO S9c.1)
        snprintf(p, sizeof(p), "%s/view-h.rqseq", argv[3]);
        if ((e = rq_mtp_keep(vs, n - 1, H + (size_t)(n - 1) * mi.dim)) != RQ_OK || (e = rq_seq_save(vs, p)) != RQ_OK) {
            fprintf(stderr, "%s: %s\n", p, rq_strerror(e));
            return 1;
        }
        printf("seqfile-out: %s\n", p);
#endif
        rq_seq_free(vs);
        rq_ctx_free(vc);
        rq_model_free(v);
    }
    free(t);
    free(H);
    free(hin);
    rq_seq_free(s);
    rq_ctx_free(c);
    rq_model_free(m);
    return 0;
}

static int seqfile_load_main(int argc, char **argv) {
    // api_test --seqfile-load MODEL FILE [--view]: FILE's rq_seq_file_info and its whole rq_seq_load into a fresh
    // sequence of MODEL (--view: of its MTP view): "seqfile-load: info CODE load CODE len N" (and "kept P" in a build
    // with rq_mtp_kept); exit status 0 if both ran
    if (argc < 4 || argc > 5 || (argc == 5 && strcmp(argv[4], "--view"))) {
        fprintf(stderr, "usage: api_test --seqfile-load MODEL FILE [--view]\n");
        return 2;
    }
    char err[512];
    int e;
    rq_threads_init(0);
    RqModel *m = rq_model_load(argv[2], NULL, err, sizeof(err)), *v = NULL;
    RqMtpOpts mo = { 0, -1 };
    if (m && argc == 5) v = rq_mtp_load(m, &mo, err, sizeof(err));
    if (!m || (argc == 5 && !v)) { fprintf(stderr, "%s\n", err); return 1; }
    RqSeqFile f;
    int ei = rq_seq_file_info(v ? v : m, argv[3], &f, NULL);
    RqSeq *s = rq_seq_new(v ? v : m, 512, RQ_KV_DEFAULT, &e);
    if (!s) { fprintf(stderr, "rq_seq_new: %s\n", rq_strerror(e)); return 1; }
    int el = rq_seq_load(s, argv[3], ei == RQ_OK ? f.len : 1);
    printf("seqfile-load: info %d load %d len %d", ei, el, rq_seq_len(s));
#ifdef RQ_MTP_VOCAB_DEFAULT
    printf(" kept %d", v ? rq_mtp_kept(s, NULL) : -1);
#endif
    printf("\n");
    rq_seq_free(s);
    rq_model_free(v);
    rq_model_free(m);
    return 0;
}
// seqfile-out: end

int main(int argc, char **argv) {
    int only_digest = 0;
    if (argc > 1 && !strcmp(argv[1], "--seqfile-out")) return seqfile_out_main(argc, argv);
    if (argc > 1 && !strcmp(argv[1], "--seqfile-load")) return seqfile_load_main(argc, argv);
    if (argc > 1 && !strcmp(argv[1], "--two")) return two_main(argc, argv);
    if (argc > 1 && !strcmp(argv[1], "--spec")) return spec_main(argc, argv);
    if (argc > 1 && !strcmp(argv[1], "--mtp")) return mtp_main(argc, argv);
    if (argc > 1 && !strcmp(argv[1], "--mtp-dump")) return mtp_dump_main(argc, argv);
    if (argc > 1 && !strcmp(argv[1], "--mtp-seqcheck")) return mtp_seqcheck_main(argc, argv);
    if (argc > 3 && !strcmp(argv[2], "--sample")) return sample_main(argc, argv);
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--digest")) only_digest = 1;
        else if (!strcmp(argv[i], "--digest-small")) only_digest = 2;
        else if (!strcmp(argv[i], "--long")) bd_long = 1;
        else if (!strcmp(argv[i], "--n65")) bd_n65 = 1;
        else if (!strcmp(argv[i], "--kv")) bd_kv = 1;
        else if (!strcmp(argv[i], "--kv-type") && i + 1 < argc && (KVT = rq_kv_parse(argv[i + 1])) >= 0) i++;
        else argc = 0;
    }
    if (argc < 2) {
        fprintf(stderr, "usage: api_test MODEL [--digest | --digest-small [--long]] [--n65] [--kv] [--kv-type TYPE]\n"
                        "       api_test --two MODEL_A MODEL_B [--gs MODEL_OTHER_GROUP_SIZE]\n"
                        "       api_test --spec TARGET DRAFT [--quick] [--other MODEL_OTHER_VOCABULARY]\n"
                        "       api_test --mtp TARGET [--quick] [--plain MODEL_WITHOUT_BLOCK] [--optional MODEL_WITH_OPTIONAL_NEXTN]\n"
                        "       api_test --mtp-dump MODEL TEXTFILE OUT.bin [-b N] [--vocab N]\n"
                        "       api_test --mtp-seqcheck MODEL TARGET.rqseq VIEW.rqseq\n"
                        "       api_test --seqfile-out MODEL DIR\n"
                        "       api_test --seqfile-load MODEL FILE [--view]\n"
                        "       api_test MODEL --sample LOGITS_FILE < PLAN (lines: T P K ROW SEED...)\n");
        return 2;
    }
    char err[512];
    setvbuf(stdout, NULL, _IOLBF, 0);
    rq_threads_init(0);
    RqModel *bad = rq_model_load("models/no-such-file.gguf", NULL, err, sizeof(err));
    report(!bad && strstr(err, "Couldn't open checkpoint"), "load_error", err);
    M = rq_model_load(argv[1], NULL, err, sizeof(err));
    if (!M) {
        printf("FAIL setup  %s\n", err);
        return 1;
    }
    rq_model_info(M, &I);
    V = I.vocab_size;
    int e;
    RqCtx *c = rq_ctx_new(M, 64, 0, &e);
    if (!c) die("rq_ctx_new", e);

    // prompts: A 33 tokens, B 73 (> 64: 2 chunks), C 253 (crosses 256 positions when decoded)
    RqTokens A = toks("The capital of France is Paris, and the capital of Germany is Berlin. Rivers: the Seine, "
                      "the Rhine and the Danube flow through famous cities.");
    RqTokens B = toks("def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n\n# Print the first ten "
                      "Fibonacci numbers, one per line, then their sum, and explain why the recursion is slow for "
                      "large n and how memoization fixes it.\nfor i in range(10):\n    print(fib(i))\n");
    RqTokens C = { 0 };
    while (C.n < 250) {
        RqTokens t = toks("A lighthouse keeper trims the wicks, polishes the lens and writes the weather into the log. ");
        rq_tokens_push(&C, t.v, t.n);
        rq_tokens_free(&t);
    }
    int dA[3] = { 279, 6722, 13 }, dB[3] = { 198, 262, 470 }, dC[24];
    for (int i = 0; i < 24; i++) dC[i] = 1000 + 37 * i;
    if (only_digest) { // the batched passes of section 9 only
        rq_tokens_push(&C, dC, 7);
        batch_tests(c, A, B, C, NULL, only_digest);
        rq_tokens_free(&A);
        rq_tokens_free(&B);
        rq_tokens_free(&C);
        rq_ctx_free(c);
        rq_model_free(M);
        return 0;
    }
    float *refA = logits_alone(c, A.v, A.n, dA, 3), *refB = logits_alone(c, B.v, B.n, dB, 3);
    float *refC = logits_alone(c, C.v, C.n, dC, 24);
    char d[256];

    // 1. interleaved: A and B prefilled and decoded alternately on one context
    {
        RqSeq *sa = seq_new(0), *sb = seq_new(0);
        float *la = malloc((size_t)A.n * V * sizeof(float)), *lb = malloc((size_t)B.n * V * sizeof(float)), *lg;
        int ok = rq_prefill(c, sa, A.v, A.n, RQ_LOGITS_ALL, la, NULL) == RQ_OK;
        ok &= rq_prefill(c, sb, B.v, B.n, RQ_LOGITS_ALL, lb, NULL) == RQ_OK;
        ok &= same(la, refA, (size_t)A.n * V) && same(lb, refB, (size_t)B.n * V);
        for (int i = 0; i < 3 && ok; i++) {
            ok &= rq_decode(c, sb, dB[i], &lg) == RQ_OK && same(lg, refB + (size_t)(B.n + i) * V, V);
            ok &= rq_decode(c, sa, dA[i], &lg) == RQ_OK && same(lg, refA + (size_t)(A.n + i) * V, V);
        }
        report(ok, "interleaved_bitwise", "prefill + 3 decode steps of 2 sequences, alternating");
        free(la);
        free(lb);
        rq_seq_free(sa);
        rq_seq_free(sb);
    }

    // 2. several sequences in one batched pass: prefill chunks of A and B together (all rows), then
    // decode rows of A, B, C together (last rows: the classifier GEMM over 3 rows), then a prefill
    // chunk of C next to decode rows of A and B; C crosses the 256-position attention block
    {
        RqSeq *sa = seq_new(0), *sb = seq_new(0), *sc = seq_new(0);
        float *out = malloc((size_t)64 * V * sizeof(float)), *lg;
        int ok = 1, nb0 = 64 - A.n; // (B's first rows fill the pass, the rest in the next one)
        RqSegment s1[2] = { { sa, A.v, A.n, RQ_LOGITS_ALL }, { sb, B.v, nb0, RQ_LOGITS_ALL } };
        ok &= rq_forward_batch(c, s1, 2, out, &lg) == RQ_OK && lg == out;
        ok &= same(out, refA, (size_t)A.n * V) && same(out + (size_t)A.n * V, refB, (size_t)nb0 * V);
        int c0 = 64 - (B.n - nb0) < 20 ? 64 - (B.n - nb0) : 20; // (C's first rows next to B's last ones: the tokenizer's counts)
        RqSegment s2[2] = { { sb, B.v + nb0, B.n - nb0, RQ_LOGITS_LAST }, { sc, C.v, c0, RQ_LOGITS_NONE } };
        ok &= rq_forward_batch(c, s2, 2, NULL, &lg) == RQ_OK && lg && same(lg, refB + (size_t)(B.n - 1) * V, V);
        for (int i = c0; i < C.n && ok; i += 64) { // the rest of C in 64-row chunks
            int k = C.n - i < 64 ? C.n - i : 64;
            RqSegment s = { sc, C.v + i, k, RQ_LOGITS_LAST };
            ok &= rq_forward_batch(c, &s, 1, NULL, &lg) == RQ_OK;
        }
        ok = ok && lg && same(lg, refC + (size_t)(C.n - 1) * V, V);
        report(ok, "batch_prefill_bitwise", "chunks of 2 sequences in one pass (all rows / last row / none)");
        ok = 1;
        for (int i = 0; i < 3 && ok; i++) { // 3 decode rows of 3 sequences per pass
            RqSegment s3[3] = { { sc, dC + i, 1, RQ_LOGITS_LAST }, { sa, dA + i, 1, RQ_LOGITS_LAST }, { sb, dB + i, 1, RQ_LOGITS_LAST } };
            ok &= rq_forward_batch(c, s3, 3, out, NULL) == RQ_OK;
            ok &= same(out, refC + (size_t)(C.n + i) * V, V) && same(out + V, refA + (size_t)(A.n + i) * V, V) &&
                  same(out + 2 * (size_t)V, refB + (size_t)(B.n + i) * V, V);
        }
        report(ok, "batch_decode_bitwise", "3 sequences, one row each, 3 passes (C at positions 250+)");
        // C's next 21 tokens as one chunk, all rows: positions past the first attention block
        RqSegment s4 = { sc, dC + 3, 21, RQ_LOGITS_ALL };
        ok = rq_forward_batch(c, &s4, 1, out, NULL) == RQ_OK && same(out, refC + (size_t)(C.n + 3) * V, (size_t)21 * V) &&
             rq_seq_len(sc) == C.n + 24 && C.n + 24 > 256;
        snprintf(d, sizeof(d), "positions %d..%d", C.n + 3, C.n + 23);
        report(ok, "batch_past_block_bitwise", d);
        free(out);
        rq_seq_free(sa);
        rq_seq_free(sb);
        rq_seq_free(sc);
    }

    // 3. truncate + re-prefill == fresh prefill (positions overwritten in the cache; Qwen3.5: to a checkpoint)
    {
        RqSeq *s = seq_new(0);
        float *lg;
        int ok = rq_seq_checkpoint_at(s, 20) == RQ_OK; // (a no-op for Qwen3)
        ok &= rq_prefill(c, s, B.v, B.n, RQ_LOGITS_NONE, NULL, NULL) == RQ_OK;
        for (int i = 0; i < 3; i++) ok &= rq_decode(c, s, dB[i], NULL) == RQ_OK;
        ok &= rq_seq_truncate(s, 20) == RQ_OK && rq_seq_len(s) == 20;
        ok &= rq_prefill(c, s, A.v, A.n, RQ_LOGITS_LAST, NULL, &lg) == RQ_OK;
        RqTokens t = { 0 };
        rq_tokens_push(&t, B.v, 20);
        rq_tokens_push(&t, A.v, A.n);
        float *ref = logits_alone(c, t.v, t.n, NULL, 0);
        ok &= same(lg, ref + (size_t)(t.n - 1) * V, V);
        ok &= rq_seq_common_prefix(s, t.v, t.n) == t.n && rq_seq_truncate(s, t.n + 1) == RQ_ERR_ARG;
        report(ok, "truncate_reprefill_bitwise", "B[0:20] kept, A re-prefilled == fresh B[0:20] + A");
        free(ref);
        rq_tokens_free(&t);
        rq_seq_free(s);
    }

    // 3b. Qwen3.5: recurrent-state checkpoints (rq_seq_checkpoint_at) taken by a prefill chunk (mid-chunk), at once,
    // by a decode step and by a batched decode pass; truncate / copy / rq_seq_keep to, above and below them; each
    // continuation == a fresh run of the same tokens, bitwise
    if (I.linear_layers) {
        RqSeq *s = seq_new(0), *t = seq_new(0);
        float *lg;
        int ok = 1, d2[2] = { 279, 13 };
        RqTokens u = { 0 };
        rq_tokens_push(&u, B.v, B.n);
        rq_tokens_push(&u, dB, 3);
        float *ref = logits_alone(c, u.v, u.n, NULL, 0); // B + dB, every row
        ok &= rq_seq_checkpoint_at(s, 40) == RQ_OK && rq_prefill(c, s, B.v, B.n, RQ_LOGITS_NONE, NULL, NULL) == RQ_OK;
        ok &= rq_seq_checkpoint_at(s, B.n) == RQ_OK && rq_seq_checkpoint_at(s, B.n - 1) == RQ_ERR_ARG;
        ok &= rq_seq_checkpoint_at(s, B.n + 2) == RQ_OK;
        for (int i = 0; i < 3; i++)
            ok &= rq_decode(c, s, dB[i], NULL) == RQ_OK; // (takes the one at B.n + 2, replacing the one at 40)
        ok &= rq_seq_keep(s, B.n + 3) == B.n + 3 && rq_seq_keep(s, B.n + 1) == B.n && rq_seq_keep(s, B.n + 2) == B.n + 2;
        ok &= rq_seq_keep(s, 50) == 0 && rq_seq_keep(s, B.n + 4) == RQ_ERR_ARG;
        // back to B.n + 2 (exact), then the last token again
        ok &= rq_seq_truncate(s, B.n + 2) == RQ_OK && rq_seq_len(s) == B.n + 2;
        ok &= rq_decode(c, s, dB[2], &lg) == RQ_OK && same(lg, ref + (size_t)(u.n - 1) * V, V);
        // above B.n (the checkpoint at B.n + 2 is gone with the truncation below it): to B.n, re-prefill 2
        ok &= rq_seq_truncate(s, B.n + 1) == RQ_OK && rq_seq_len(s) == B.n && rq_seq_keep(s, B.n) == B.n;
        ok &= rq_prefill(c, s, dB, 2, RQ_LOGITS_LAST, NULL, &lg) == RQ_OK && same(lg, ref + (size_t)(B.n + 1) * V, V);
        // copy: src's length (its state), below it (the checkpoint at B.n), below every checkpoint (0)
        ok &= rq_seq_copy(t, s, rq_seq_len(s)) == RQ_OK && rq_seq_len(t) == B.n + 2;
        ok &= rq_decode(c, t, dB[2], &lg) == RQ_OK && same(lg, ref + (size_t)(u.n - 1) * V, V);
        ok &= rq_seq_copy(t, s, B.n + 1) == RQ_OK && rq_seq_len(t) == B.n;
        ok &= rq_prefill(c, t, dB, 3, RQ_LOGITS_ALL, NULL, NULL) == RQ_ERR_ARG; // (all rows need an output buffer)
        ok &= rq_prefill(c, t, dB, 3, RQ_LOGITS_LAST, NULL, &lg) == RQ_OK && same(lg, ref + (size_t)(u.n - 1) * V, V);
        ok &= rq_seq_copy(t, s, 30) == RQ_OK && rq_seq_len(t) == 0;
        // below every checkpoint: to 0, re-prefill all
        ok &= rq_seq_truncate(s, 30) == RQ_OK && rq_seq_len(s) == 0;
        ok &= rq_prefill(c, s, u.v, u.n, RQ_LOGITS_LAST, NULL, &lg) == RQ_OK && same(lg, ref + (size_t)(u.n - 1) * V, V);
        // a checkpoint taken by a batched decode pass (2 sequences, one row each; s's next position)
        rq_seq_reset(t);
        ok &= rq_prefill(c, t, A.v, A.n, RQ_LOGITS_NONE, NULL, NULL) == RQ_OK && rq_seq_checkpoint_at(s, u.n + 1) == RQ_OK;
        RqSegment two[2] = { { s, d2, 1, RQ_LOGITS_LAST }, { t, d2 + 1, 1, RQ_LOGITS_LAST } };
        ok &= rq_forward_batch(c, two, 2, NULL, NULL) == RQ_ERR_ARG; // (2 rows of logits: an output buffer)
        float *o2 = malloc((size_t)2 * V * sizeof(float));
        ok &= o2 && rq_forward_batch(c, two, 2, o2, NULL) == RQ_OK && rq_seq_keep(s, u.n + 1) == u.n + 1;
        free(o2);
        ok &= rq_decode(c, s, d2[1], NULL) == RQ_OK && rq_seq_truncate(s, u.n + 1) == RQ_OK && rq_seq_len(s) == u.n + 1;
        RqTokens w = { 0 };
        rq_tokens_push(&w, u.v, u.n);
        rq_tokens_push(&w, d2, 2);
        float *ref2 = logits_alone(c, w.v, w.n, NULL, 0);
        ok &= rq_decode(c, s, d2[1], &lg) == RQ_OK && same(lg, ref2 + (size_t)(w.n - 1) * V, V);
        report(ok, "checkpoints_bitwise", "taken by a prefill chunk / at once / a decode step / a batched decode pass; "
               "truncate, copy, keep to / above / below them == fresh runs");
        free(ref);
        free(ref2);
        rq_tokens_free(&u);
        rq_tokens_free(&w);
        rq_seq_free(s);
        rq_seq_free(t);
    }

    // 3c. Qwen3.5: checkpoint sets (rq_seq_set_checkpoints, several pending positions per pass, rq_seq_checkpoint_rows)
    if (I.linear_layers) {
        int ok = 1, nu = C.n + 24, *u = malloc((size_t)nu * sizeof(int));
        float *out = malloc((size_t)64 * V * sizeof(float));
        memcpy(u, C.v, (size_t)C.n * sizeof(int));
        memcpy(u + C.n, dC, 24 * sizeof(int)); // (refC: every row of u)
        RqSeq *s = seq_new(0), *t = seq_new(0), *w = seq_new(0);
        // the default 2 slots: 3 pending in one prefill chunk, the 2 highest stay
        ok &= rq_seq_checkpoint_at(s, 25) == RQ_OK && rq_seq_checkpoint_at(s, 50) == RQ_OK && rq_seq_checkpoint_at(s, 10) == RQ_OK;
        ok &= rq_prefill(c, s, u, 60, RQ_LOGITS_NONE, NULL, NULL) == RQ_OK;
        ok &= rq_seq_keep(s, 59) == 50 && rq_seq_keep(s, 49) == 25 && rq_seq_keep(s, 24) == 0;
        ok &= ck_cont(c, s, 50, u, refC, out) && ck_cont(c, s, 25, u, refC, out);
        // 6 slots: 3 pending in one chunk, 2 at a chunk boundary (64: the chunk's last row, 65: the next one's first), one
        // in a later chunk; then truncate to each (highest first) and continue
        static const int P6[6] = { 200, 65, 64, 50, 25, 10 };
        ok &= rq_seq_set_checkpoints(s, 6) == RQ_OK;
        rq_seq_reset(s);
        for (int i = 0; i < 6; i++)
            ok &= rq_seq_checkpoint_at(s, P6[i]) == RQ_OK;
        ok &= rq_prefill(c, s, u, C.n, RQ_LOGITS_NONE, NULL, NULL) == RQ_OK;
        for (int i = 0; i < 6; i++)
            ok &= rq_seq_keep(s, P6[i]) == P6[i] && rq_seq_keep(s, P6[i] - 1) == (i < 5 ? P6[i + 1] : 0);
        for (int i = 0; i < 6; i++)
            ok &= ck_cont(c, s, P6[i], u, refC, out);
        // eviction: 7 pending + the checkpoint at 10, 6 slots: the 6 highest stay (10 replaced, then 20 inside the pass)
        static const int P7[7] = { 61, 20, 30, 60, 40, 55, 45 };
        for (int i = 0; i < 7; i++)
            ok &= rq_seq_checkpoint_at(s, P7[i]) == RQ_OK;
        ok &= rq_prefill(c, s, u + 15, 55, RQ_LOGITS_NONE, NULL, NULL) == RQ_OK && rq_seq_len(s) == 70;
        ok &= rq_seq_keep(s, 29) == 0 && rq_seq_keep(s, 39) == 30 && rq_seq_keep(s, 69) == 61;
        // fewer slots: the 3 highest stay
        ok &= rq_seq_set_checkpoints(s, 3) == RQ_OK && rq_seq_keep(s, 54) == 0 && rq_seq_keep(s, 59) == 55;
        ok &= ck_cont(c, s, 61, u, refC, out) && ck_cont(c, s, 60, u, refC, out) && ck_cont(c, s, 55, u, refC, out);
        ok &= rq_seq_set_checkpoints(s, 0) == RQ_ERR_ARG && rq_seq_set_checkpoints(s, RQ_CHECKPOINTS_MAX + 1) == RQ_ERR_ARG;
        // a batched pass of 3 sequences: a 20-row chunk with 4 pending (s, 4 slots), a 5-row chunk with 3 (t, 3 slots), a
        // decode row with 1 (w)
        ok &= rq_seq_set_checkpoints(s, 4) == RQ_OK && rq_seq_set_checkpoints(t, 3) == RQ_OK;
        rq_seq_reset(s);
        ok &= rq_prefill(c, s, u, 30, RQ_LOGITS_NONE, NULL, NULL) == RQ_OK;
        ok &= rq_prefill(c, t, u, 100, RQ_LOGITS_NONE, NULL, NULL) == RQ_OK && rq_prefill(c, w, u, 40, RQ_LOGITS_NONE, NULL, NULL) == RQ_OK;
        static const int PS[4] = { 50, 42, 37, 33 }, PT[3] = { 105, 102, 101 };
        for (int i = 0; i < 4; i++)
            ok &= rq_seq_checkpoint_at(s, PS[i]) == RQ_OK;
        for (int i = 0; i < 3; i++)
            ok &= rq_seq_checkpoint_at(t, PT[i]) == RQ_OK;
        ok &= rq_seq_checkpoint_at(w, 41) == RQ_OK;
        RqSegment three[3] = { { s, u + 30, 20, RQ_LOGITS_NONE }, { t, u + 100, 5, RQ_LOGITS_LAST }, { w, u + 40, 1, RQ_LOGITS_LAST } };
        ok &= rq_forward_batch(c, three, 3, out, NULL) == RQ_OK && same(out, refC + (size_t)104 * V, V) &&
              same(out + V, refC + (size_t)40 * V, V);
        for (int i = 0; i < 4; i++)
            ok &= ck_cont(c, s, PS[i], u, refC, out);
        for (int i = 0; i < 3; i++)
            ok &= ck_cont(c, t, PT[i], u, refC, out);
        ok &= rq_decode(c, w, u[41], NULL) == RQ_OK && ck_cont(c, w, 41, u, refC, out);
        // rq_seq_checkpoint_rows, as a speculative loop uses it: the prompt's checkpoint (at 40) + a verify segment of 4
        // rows per pass (5 slots), truncated to accepted lengths; each segment's rows == the reference, the checkpoint at
        // 40 stays; then a draft's 3 decode steps (one checkpoint each)
        ok &= rq_seq_set_checkpoints(s, 5) == RQ_OK;
        rq_seq_reset(s);
        ok &= rq_seq_checkpoint_at(s, 40) == RQ_OK && rq_prefill(c, s, u, 60, RQ_LOGITS_NONE, NULL, NULL) == RQ_OK;
        static const int acc[5] = { 0, 2, 3, 1, 3 }; // drafts accepted per segment
        for (int i = 0, L = 60; i < 5 && ok; i++) {
            ok &= rq_seq_checkpoint_rows(s, L, 4) == RQ_OK;
            ok &= rq_prefill(c, s, u + L, 4, RQ_LOGITS_ALL, out, NULL) == RQ_OK && same(out, refC + (size_t)L * V, (size_t)4 * V);
            for (int j = 1; j <= 4; j++)
                ok &= rq_seq_keep(s, L + j) == L + j;
            ok &= rq_seq_keep(s, L) == 40; // (the previous segment's checkpoints: replaced)
            L += acc[i] + 1;
            ok &= rq_seq_truncate(s, L) == RQ_OK && rq_seq_len(s) == L;
        }
        int L = rq_seq_len(s);
        ok &= rq_seq_checkpoint_rows(s, L, 3) == RQ_OK;
        for (int i = 0; i < 3; i++)
            ok &= rq_decode(c, s, u[L + i], NULL) == RQ_OK;
        ok &= rq_seq_keep(s, L + 1) == L + 1 && rq_seq_keep(s, L + 2) == L + 2 && ck_cont(c, s, L + 1, u, refC, out);
        ok &= rq_seq_checkpoint_rows(s, rq_seq_len(s) - 1, 2) == RQ_ERR_ARG && rq_seq_checkpoint_rows(s, rq_seq_len(s), 0) == RQ_ERR_ARG;
        ok &= rq_seq_checkpoint_rows(s, rq_seq_len(s), RQ_CHECKPOINTS_MAX + 1) == RQ_ERR_ARG;
        ok &= rq_seq_checkpoint_rows(s, rq_seq_capacity(s) - 1, 2) == RQ_ERR_ARG;
        ok &= ck_cont(c, s, 40, u, refC, out);
        // rq_seq_checkpoints / rq_seq_checkpoint_drop (the server keeps the checkpoints it values from the lowest-first
        // replacement): the list (taken ones only, ascending), a dropped slot taken by the next checkpoint instead of the
        // lowest one, which still continues exactly
        int pos[8];
        rq_seq_reset(s);
        ok &= rq_seq_set_checkpoints(s, 3) == RQ_OK && rq_seq_checkpoints(s, pos, 8) == 0;
        ok &= rq_seq_checkpoint_at(s, 10) == RQ_OK && rq_seq_checkpoint_at(s, 30) == RQ_OK && rq_seq_checkpoint_at(s, 20) == RQ_OK;
        ok &= rq_seq_checkpoints(s, pos, 8) == 0; // (pending)
        ok &= rq_prefill(c, s, u, 40, RQ_LOGITS_NONE, NULL, NULL) == RQ_OK;
        ok &= rq_seq_checkpoints(s, pos, 8) == 3 && pos[0] == 10 && pos[1] == 20 && pos[2] == 30;
        ok &= rq_seq_checkpoints(s, pos, 1) == 3 && pos[0] == 10 && rq_seq_checkpoints(s, NULL, 0) == 3;
        ok &= rq_seq_checkpoint_drop(s, 20) == RQ_OK && rq_seq_checkpoint_drop(s, 20) == RQ_ERR_ARG && rq_seq_checkpoint_drop(s, 15) == RQ_ERR_ARG;
        ok &= rq_seq_checkpoints(s, pos, 8) == 2 && pos[0] == 10 && pos[1] == 30 && rq_seq_keep(s, 25) == 10;
        ok &= rq_seq_checkpoint_at(s, 50) == RQ_OK && rq_prefill(c, s, u + 40, 15, RQ_LOGITS_NONE, NULL, NULL) == RQ_OK;
        ok &= rq_seq_checkpoints(s, pos, 8) == 3 && pos[0] == 10 && pos[1] == 30 && pos[2] == 50;
        ok &= ck_cont(c, s, 50, u, refC, out) && ck_cont(c, s, 10, u, refC, out);
        report(ok, "checkpoint_sets", "3 pending in one prefill chunk (2 slots: the highest 2), 6 across chunks, eviction, fewer "
               "slots, a batched pass of 3 sequences, rq_seq_checkpoint_rows (verify segments, decode steps), "
               "rq_seq_checkpoints / rq_seq_checkpoint_drop: each truncation continues == a fresh run");
        free(u);
        free(out);
        rq_seq_free(s);
        rq_seq_free(t);
        rq_seq_free(w);
    }

    // 3d. sequences on disk (rq_seq_save / rq_seq_file_info / rq_seq_load): C + 24 decode steps (two attention blocks),
    // saved with its RNG (and, Qwen3.5, checkpoints at 40 and 260), loaded whole into a fresh sequence: its tokens and
    // RNG, then 5 more rows == the reference's; loaded to 100 (Qwen3.5: onto 40) and to its 260 and continued == the
    // reference (only the blocks up to there read); its loaded checkpoints usable (truncate to 40). Refused: a file of
    // another KV type (MISMATCH), a corrupted token (MISMATCH), a truncated file (MISMATCH), a missing one (IO), n past the
    // saved length or dst's capacity (ARG), a save into a missing directory (IO, nothing left); an empty sequence (ARG);
    // the refused loads leave dst as it was
#ifndef _WIN32
    {
        int ok = 1, nu = C.n + 24, *u = malloc((size_t)nu * sizeof(int)), *tk = NULL;
        float *out = malloc((size_t)64 * V * sizeof(float));
        memcpy(u, C.v, (size_t)C.n * sizeof(int));
        memcpy(u + C.n, dC, 24 * sizeof(int)); // (refC: every row of u)
        const char *td = getenv("TMPDIR");
        char dir[512], path[600], bad[600];
        snprintf(dir, sizeof(dir), "%s/api_test_seqXXXXXX", td && *td ? td : "/tmp");
        if (!mkdtemp(dir)) die("mkdtemp", RQ_ERR_IO);
        snprintf(path, sizeof(path), "%s/c.rqseq", dir);
        RqSeq *s = seq_new(0), *t = seq_new(0), *small = seq_new(64);
        int P = nu - 5; // (saved at P: 5 rows of u left to continue with)
        rq_seq_seed(s, 0x1234abcdULL);
        ok &= rq_seq_checkpoint_at(s, 40) == RQ_OK && rq_seq_checkpoint_at(s, 260) == RQ_OK;
        ok &= rq_prefill(c, s, u, P, RQ_LOGITS_NONE, NULL, NULL) == RQ_OK;
        unsigned long long rng = *rq_seq_rng(s);
        double t0 = rq_now();
        ok &= rq_seq_save(s, path) == RQ_OK && access(path, F_OK) == 0;
        double ts = rq_now() - t0;
        RqSeqFile f;
        ok &= rq_seq_file_info(M, path, &f, &tk) == RQ_OK && f.len == P && f.kv_type == KVT && tk &&
              !memcmp(tk, u, (size_t)P * sizeof(int)) && f.linear == !!I.linear_layers;
        if (I.linear_layers) ok &= f.n_ck == 2 && f.ck[0] == 40 && f.ck[1] == 260 && rq_seq_file_keep(&f, 100) == 40;
        else ok &= f.n_ck == 0 && rq_seq_file_keep(&f, 100) == 100;
        ok &= rq_seq_file_keep(&f, P) == P && rq_seq_file_keep(&f, P + 9) == P && rq_seq_file_keep(&f, -3) == 0;
        free(tk);
        // whole: the tokens, the RNG, 5 more rows; the source too (saving changed nothing)
        t0 = rq_now();
        ok &= rq_seq_load(t, path, P) == RQ_OK && rq_seq_len(t) == P && *rq_seq_rng(t) == rng;
        double tl = rq_now() - t0;
        ok &= !memcmp(rq_seq_tokens(t), u, (size_t)P * sizeof(int));
        ok &= rq_prefill(c, t, u + P, 5, RQ_LOGITS_ALL, out, NULL) == RQ_OK && same(out, refC + (size_t)P * V, (size_t)5 * V);
        ok &= rq_prefill(c, s, u + P, 5, RQ_LOGITS_ALL, out, NULL) == RQ_OK && same(out, refC + (size_t)P * V, (size_t)5 * V);
        int whole = ok;
        // part: to 100 (Qwen3.5: the checkpoint at 40), to 260 (Qwen3.5: a checkpoint, in the second block)
        int k100 = I.linear_layers ? 40 : 100;
        ok &= rq_seq_load(t, path, 100) == RQ_OK && rq_seq_len(t) == k100 &&
              rq_prefill(c, t, u + k100, 5, RQ_LOGITS_ALL, out, NULL) == RQ_OK && same(out, refC + (size_t)k100 * V, (size_t)5 * V);
        ok &= rq_seq_load(t, path, 260) == RQ_OK && rq_seq_len(t) == 260 &&
              rq_prefill(c, t, u + 260, 5, RQ_LOGITS_ALL, out, NULL) == RQ_OK && same(out, refC + (size_t)260 * V, (size_t)5 * V);
        if (I.linear_layers) { // the loaded checkpoints: kept, and exact
            int pos[4];
            ok &= rq_seq_load(t, path, P) == RQ_OK && rq_seq_checkpoints(t, pos, 4) == 2 && pos[0] == 40 && pos[1] == 260;
            ok &= ck_cont(c, t, 260, u, refC, out) && ck_cont(c, t, 40, u, refC, out);
            ok &= rq_seq_set_checkpoints(t, 1) == RQ_OK && rq_seq_load(t, path, P) == RQ_OK && rq_seq_checkpoints(t, pos, 4) == 1 && pos[0] == 260;
            ok &= rq_seq_load(t, path, 30) == RQ_OK && rq_seq_len(t) == 0;
            rq_seq_set_checkpoints(t, 2);
        }
        int part = ok, tlen = rq_seq_len(t);
        // refused (dst unchanged: the checks come before any transfer)
        int e1, e2, e3, e4, e5, e6, e7;
        int okt = KVT == RQ_KV_F32 ? RQ_KV_F16 : RQ_KV_F32, e;
        RqSeq *o = rq_seq_new(M, 0, okt, &e);
        e1 = o ? rq_seq_load(o, path, 10) : -100;
        rq_seq_free(o);
        e2 = rq_seq_load(t, path, P + 1);
        e3 = rq_seq_load(small, path, 100);
        snprintf(bad, sizeof(bad), "%s/missing.rqseq", dir);
        e4 = rq_seq_load(t, bad, 10);
        snprintf(bad, sizeof(bad), "%s/no-dir/x.rqseq", dir);
        e5 = rq_seq_save(s, bad);
        rq_seq_reset(small);
        e6 = rq_seq_save(small, path);
        // a corrupted token, then a truncated file (copies of the saved one)
        snprintf(bad, sizeof(bad), "%s/bad.rqseq", dir);
        FILE *fi = fopen(path, "rb"), *fo = fopen(bad, "wb");
        long long n = 0;
        int ch;
        while (fi && fo && (ch = fgetc(fi)) != EOF) {
            if (n == 4096 + 4 * 7) ch ^= 1;
            fputc(ch, fo);
            n++;
        }
        if (fi) fclose(fi);
        if (fo) fclose(fo);
        e7 = rq_seq_load(t, bad, 10);
        int trunc = truncate(path, (off_t)(f.bytes - 4096)) == 0 ? rq_seq_load(t, path, 10) : -100;
        ok &= e1 == RQ_ERR_MISMATCH && e2 == RQ_ERR_ARG && e3 == RQ_ERR_ARG && e4 == RQ_ERR_IO && e5 == RQ_ERR_IO &&
              e6 == RQ_ERR_ARG && e7 == RQ_ERR_MISMATCH && trunc == RQ_ERR_MISMATCH && rq_seq_len(t) == tlen;
        snprintf(bad, sizeof(bad), "%s/no-dir/x.rqseq.tmp", dir);
        ok &= access(bad, F_OK) != 0;
        snprintf(d, sizeof(d), "%d positions, %.1f MB: save %.1f ms, load %.1f ms; whole %s, parts %s; errors %d %d %d %d %d %d %d %d",
                 P, f.bytes / 1e6, ts * 1e3, tl * 1e3, whole ? "ok" : "FAIL", part ? "ok" : "FAIL", e1, e2, e3, e4, e5, e6, e7, trunc);
        report(ok, "seq_file_bitwise", d);
        snprintf(bad, sizeof(bad), "%s/bad.rqseq", dir);
        unlink(bad);
        unlink(path);
        rmdir(dir);
        free(u);
        free(out);
        rq_seq_free(s);
        rq_seq_free(t);
        rq_seq_free(small);
    }
#endif

    // 3e. giving memory back (rq_seq_release, rq_seq_resident, rq_seq_kv_bytes): a sequence prefilled to 1,024 positions
    // (4 whole blocks, Qwen3.5: checkpoints at 300 and 768 of 4 slots), cut to keep = 768 (block-aligned), 300
    // (mid-block), 0, then released: its resident estimate is kv_bytes(keep) (+ the state copies it still holds: the
    // free checkpoint buffers go, at length 0 the live state too), RssAnon drops by the estimate's difference (within a
    // page per range; a huge page per range more where THP is always on: the mapping's tail past the last block), and
    // the next rows (4 in a prefill chunk, then a decode step) == a fresh sequence's on the same prefix, bitwise;
    // Qwen3.5: the kept checkpoint still restores (truncate to it again, the same rows). keep below the length: the
    // length is kept. kv_bytes at the capacity == rq_seq_mem's kv; errors. (Windows: no release, hwm stays)
#ifndef _WIN32
    {
        RqSeq *s = seq_new(0), *f = seq_new(0);
        enum { N = 4 * 256, R = 5 };
        int cap = rq_seq_capacity(s);
        if (cap >= N + R) {
            int ok = 1, *u = malloc((size_t)(N + R) * sizeof(int)), nattn = 0;
            float *out = malloc((size_t)R * V * sizeof(float)), *ref = malloc((size_t)R * V * sizeof(float));
            for (int i = 0; i < N + R; i++)
                u[i] = C.v[i % C.n];
            RqSeqMem sm, r0, r1;
            ok &= rq_seq_mem(M, cap, KVT, &sm) == RQ_OK && rq_seq_kv_bytes(s, cap) == sm.kv && rq_seq_kv_bytes(s, cap + 9) == sm.kv;
            ok &= rq_seq_kv_bytes(s, 0) == (size_t)cap * sizeof(int) && rq_seq_kv_bytes(s, -1) == rq_seq_kv_bytes(s, 0);
            ok &= rq_seq_kv_bytes(s, 1) == rq_seq_kv_bytes(s, 256) && rq_seq_kv_bytes(s, 257) > rq_seq_kv_bytes(s, 256);
            ok &= rq_seq_release(s, -1) == RQ_ERR_ARG && rq_seq_release(s, 0) == RQ_OK;
            rq_seq_resident(s, &r0);
            ok &= r0.kv == rq_seq_kv_bytes(s, 0) && r0.total == r0.kv && r0.checkpoints == 0;
            nattn = I.n_layers - I.linear_layers; // (the KV cache's ranges: 2 per attention layer)
            long thp = 0;
#ifdef __linux__
            FILE *tf = fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r");
            char tb[128] = "";
            if (tf) {
                if (!fgets(tb, sizeof(tb), tf)) tb[0] = 0;
                fclose(tf);
            }
            thp = strstr(tb, "[always]") ? 2L << 20 : 0;
#endif
            // below the estimate: a page per range at most (the partial pages kept); above: + a huge page per range
            // (THP faulted past the high-water mark, given back with the whole mapping at keep 0)
            long pg = 4096;
#ifdef __linux__
            pg = sysconf(_SC_PAGESIZE) > 0 ? sysconf(_SC_PAGESIZE) : 4096;
#endif
            long slo = 2L * nattn * pg + (64L << 10), shi = slo + 2L * nattn * thp;
            static const int keep[3] = { 768, 300, 0 };
            char dd[3][96];
            ok &= rq_seq_set_checkpoints(s, 4) == RQ_OK;
            for (int k = 0; k < 3; k++) {
                int P = keep[k], okk = 1, cont = 0;
                // the reference: a fresh sequence on u[0..P), then the R rows
                rq_seq_reset(f);
                okk &= (P == 0 || rq_prefill(c, f, u, P, RQ_LOGITS_NONE, NULL, NULL) == RQ_OK) &&
                       rq_prefill(c, f, u + P, R - 1, RQ_LOGITS_ALL, ref, NULL) == RQ_OK;
                float *lg;
                okk &= rq_decode(c, f, u[P + R - 1], &lg) == RQ_OK;
                if (okk) memcpy(ref + (size_t)(R - 1) * V, lg, V * sizeof(float));
                rq_seq_reset(s);
                okk &= rq_seq_checkpoint_at(s, 300) == RQ_OK && rq_seq_checkpoint_at(s, 768) == RQ_OK;
                okk &= rq_prefill(c, s, u, N, RQ_LOGITS_NONE, NULL, NULL) == RQ_OK;
                okk &= P ? rq_seq_truncate(s, P) == RQ_OK && rq_seq_len(s) == P : (rq_seq_reset(s), 1);
                rq_seq_resident(s, &r0);
                okk &= r0.kv == rq_seq_kv_bytes(s, N);
                long a0 = rss_anon();
                okk &= rq_seq_release(s, P) == RQ_OK;
                long a1 = rss_anon();
                rq_seq_resident(s, &r1);
                int nck = I.linear_layers ? (P >= 768) + (P >= 300) : 0; // (the held ones; the freed: P = 300, 768's)
                okk &= r1.kv == rq_seq_kv_bytes(s, P) && r1.checkpoints == nck &&
                       r1.total == r1.kv + (size_t)(nck + (I.linear_layers && P > 0)) * r1.state;
                long want = (long)(r0.total - r1.total), drop = a0 - a1;
                okk &= a0 < 0 || (drop >= want - slo && drop <= want + shi);
                // the next rows: as the fresh sequence's (Qwen3.5 and P > 0: again from the kept checkpoint)
                for (int again = 0; again < 1 + (I.linear_layers && P > 0) && okk; again++) {
                    okk &= rq_seq_truncate(s, P) == RQ_OK && rq_seq_len(s) == P;
                    okk &= rq_prefill(c, s, u + P, R - 1, RQ_LOGITS_ALL, out, NULL) == RQ_OK &&
                           same(out, ref, (size_t)(R - 1) * V);
                    okk &= rq_decode(c, s, u[P + R - 1], &lg) == RQ_OK && same(lg, ref + (size_t)(R - 1) * V, V);
                    cont += okk;
                }
                snprintf(dd[k], sizeof(dd[k]), "keep %d: RssAnon -%.1f MiB (estimate -%.1f), %d continuation(s) %s", P,
                         (a0 - a1) / 1048576.0, want / 1048576.0, cont, okk ? "ok" : "FAIL");
                ok &= okk;
            }
            // keep below the length: nothing past it goes (len P + R = 5 positions, one block)
            rq_seq_resident(s, &r0);
            ok &= rq_seq_release(s, 0) == RQ_OK && rq_seq_len(s) == R;
            rq_seq_resident(s, &r1);
            ok &= r1.kv == r0.kv && r1.kv == rq_seq_kv_bytes(s, R);
            char dl[3 * 96 + 8];
            snprintf(dl, sizeof(dl), "%s; %s; %s", dd[0], dd[1], dd[2]);
            report(ok, "seq_release", dl);
            free(u);
            free(out);
            free(ref);
        } else
            report(1, "seq_release", "skipped: capacity below 1,029 positions");
        rq_seq_free(s);
        rq_seq_free(f);
    }
#endif

    // 4. rq_generate (greedy) == a hand-written sample / decode loop; stats; the reply limit
    {
        RqSeq *s1 = seq_new(0), *s2 = seq_new(0);
        RqTokens p = { 0 }, r = { 0 };
        rq_chat_user(M, "Name three colours.", 0, &p);
        RqSampling g = { 0, 0.9f, NULL };
        RqCallbacks cb = { on_tok, NULL, &r };
        RqGenStats st;
        int ok = rq_generate(c, s1, p.v, p.n, &g, 12, &cb, &st) == RQ_OK;
        float *lg;
        int n = 0, t = 0;
        ok &= rq_prefill(c, s2, p.v, p.n, RQ_LOGITS_LAST, NULL, &lg) == RQ_OK;
        while (ok && n < 12 && !rq_is_end_token(M, t = rq_sample(c, s2, lg, &g))) {
            ok &= t == r.v[n++];
            ok &= rq_decode(c, s2, t, n < 12 ? &lg : NULL) == RQ_OK;
        }
        ok &= n == r.n && st.n_gen == r.n && st.n_prompt == p.n && rq_seq_len(s1) == p.n + st.n_gen;
        ok &= st.stop == (r.n == 12 ? RQ_STOP_LIMIT : RQ_STOP_END);
        snprintf(d, sizeof(d), "%d tokens, stop %d", r.n, st.stop);
        report(ok, "generate_loop", d);
        rq_tokens_free(&p);
        rq_tokens_free(&r);
        rq_seq_free(s1);
        rq_seq_free(s2);
    }

    // 4b. the generation step (rq_gen_start / rq_gen_step) == rq_generate: a hand loop on it; the RNG pattern
    // written out (n - 1 draws after the prompt, rq_sample per token, rq_rng_f32 on a copy = the next draw); a
    // verify-pass loop (TODO S1's shape, an oracle draft with wrong guesses). Equal: the tokens, the RNG state
    // after each sample, each sampled logits vector's position and hash, the stop, the cache's tokens and the
    // logits of one more decode step. Greedy, seeded top-p and the full distribution; limits 40 and 9
    {
        RqTokens p = { 0 };
        rq_chat_user(M, "Name three colours.", 0, &p);
        RqSampling cf[4] = { { 0, 0.9f, NULL }, { 0.8f, 0.9f, NULL }, { 1.0f, 1.0f, NULL }, { 0.7f, 0.8f, NULL } };
        int lim[4] = { 40, 40, 9, 9 }, ok = 1, peek_ok = 1;
        char dd[192] = "";
        for (int k = 0; k < 4; k++) {
            RqSeq *s[4];
            GenRec rec[4];
            RqGen g[4];
            int stop[4], ng[4];
            unsigned long long hnext[4];
            for (int i = 0; i < 4; i++) {
                s[i] = seq_new(0);
                rq_seq_seed(s[i], 42 + k);
                memset(&rec[i], 0, sizeof(rec[i]));
                rec[i].s = s[i];
            }
            RqCallbacks cb[4];
            for (int i = 0; i < 4; i++) cb[i] = (RqCallbacks) { gr_token, gr_logits, &rec[i] };
            // 0: rq_generate
            RqGenStats st;
            int e = rq_generate(c, s[0], p.v, p.n, &cf[k], lim[k], &cb[0], &st);
            stop[0] = st.stop, ng[0] = st.n_gen;
            // 1: a hand loop on the step
            float *lg;
            if (e == RQ_OK) e = rq_prefill(c, s[1], p.v, p.n, RQ_LOGITS_LAST, NULL, &lg);
            if (e == RQ_OK) rq_gen_start(&g[1], s[1], &cf[k], lim[k], &cb[1], p.n);
            for (int pos = rq_seq_len(s[1]) - 1, r; e == RQ_OK; pos++) {
                if ((r = rq_gen_step(&g[1], c, pos, lg)) <= 0) { e = r < 0 ? r : RQ_OK; break; }
                e = rq_decode(c, s[1], g[1].token, r == RQ_GEN_LAST ? NULL : &lg);
                if (r == RQ_GEN_LAST) break;
            }
            stop[1] = g[1].stop, ng[1] = g[1].n_gen;
            // 2: the pattern written out with rq_sample (no step function)
            if (e == RQ_OK) e = rq_prefill(c, s[2], p.v, p.n, RQ_LOGITS_LAST, NULL, &lg);
            if (cf[k].temperature != 0)
                for (int i = 0; i < p.n - 1; i++) rq_rng_f32(rq_seq_rng(s[2]));
            stop[2] = -1, ng[2] = 0;
            while (e == RQ_OK) {
                gr_logits(&rec[2], rq_seq_len(s[2]) - 1, lg);
                unsigned long long cp = *rq_seq_rng(s[2]);
                float u = rq_rng_f32(&cp); // (the next draw, peeked)
                int t = rq_sample(c, s[2], lg, &cf[k]);
                peek_ok &= cf[k].temperature == 0 ? cp != *rq_seq_rng(s[2]) || u < 0 : cp == *rq_seq_rng(s[2]);
                if (t < 0) { e = t; break; }
                if (rq_is_end_token(M, t)) { stop[2] = RQ_STOP_END; break; }
                gr_token(&rec[2], t);
                int last = ++ng[2] >= lim[k];
                e = rq_decode(c, s[2], t, last ? NULL : &lg);
                if (last) { stop[2] = RQ_STOP_LIMIT; break; }
            }
            // 3: the verify-pass loop
            if (e == RQ_OK) e = gen_verify_loop(c, s[3], p.v, p.n, &cf[k], lim[k], &cb[3], &rec[0], &g[3]);
            stop[3] = g[3].stop, ng[3] = g[3].n_gen;
            for (int i = 0; i < 4 && e == RQ_OK; i++) { // the caches: one more decode step
                if ((e = rq_decode(c, s[i], 1000, &lg)) == RQ_OK) hnext[i] = rq_logits_hash(lg, V);
            }
            int okk = e == RQ_OK;
            for (int i = 1; i < 4 && okk; i++)
                okk &= gr_same(&rec[0], &rec[i]) && *rq_seq_rng(s[0]) == *rq_seq_rng(s[i]) && stop[i] == stop[0] &&
                        ng[i] == ng[0] && rq_seq_len(s[i]) == rq_seq_len(s[0]) && hnext[i] == hnext[0] &&
                        !memcmp(rq_seq_tokens(s[i]), rq_seq_tokens(s[0]), rq_seq_len(s[0]) * sizeof(int));
            okk &= rec[0].n == ng[0] && rec[0].nl == ng[0] + (stop[0] == RQ_STOP_END);
            ok &= okk;
            size_t dl = strlen(dd);
            snprintf(dd + dl, sizeof(dd) - dl, "%s%s %d tok stop %d%s", k ? "; " : "", k == 0 ? "greedy" : "seeded", ng[0], stop[0],
                     okk ? "" : " DIFFER");
            for (int i = 0; i < 4; i++) rq_seq_free(s[i]);
        }
        report(ok && peek_ok, "generate_step", dd);
        rq_tokens_free(&p);
    }

    // 5. errors: context full (nothing prefilled), bad token, a sequence twice in one pass, logits rows
    // without an output buffer, a batch larger than the context's
    {
        RqSeq *s = seq_new(64);
        int ok = rq_prefill(c, s, B.v, B.n, RQ_LOGITS_LAST, NULL, NULL) == RQ_ERR_CONTEXT && rq_seq_len(s) == 0;
        int badtok = V;
        ok &= rq_decode(c, s, badtok, NULL) == RQ_ERR_ARG && rq_seq_len(s) == 0;
        RqSegment two[2] = { { s, A.v, 2, RQ_LOGITS_NONE }, { s, A.v, 2, RQ_LOGITS_NONE } };
        ok &= rq_forward_batch(c, two, 2, NULL, NULL) == RQ_ERR_ARG;
        RqSegment all = { s, A.v, 4, RQ_LOGITS_ALL };
        ok &= rq_forward_batch(c, &all, 1, NULL, NULL) == RQ_ERR_ARG;
        RqSegment big = { s, C.v, 65, RQ_LOGITS_NONE };
        ok &= rq_forward_batch(c, &big, 1, NULL, NULL) == RQ_ERR_ARG && rq_seq_len(s) == 0;
        ok &= rq_prefill(c, s, A.v, A.n, RQ_LOGITS_NONE, NULL, NULL) == RQ_OK && rq_seq_len(s) == A.n;
        report(ok, "errors", rq_ctx_error(c));
        rq_seq_free(s);
    }

    // 6. chat rendering: a message list == the pieces the CLI's chat uses; thinking: earlier replies
    // without their reasoning (Instruct-2507: never thinks, replies verbatim; Thinking-2507: an answered
    // turn without its "<think>\n", rq_chat_user_history)
    {
        RqMessage msgs[4] = { { RQ_ROLE_SYSTEM, "Be brief." }, { RQ_ROLE_USER, "Hi" },
                              { RQ_ROLE_ASSISTANT, "<think>\nhmm\n</think>\n\nHello!" }, { RQ_ROLE_USER, "Bye" } };
        int ok = 1;
        for (int th = 0; th < 2; th++) {
            RqTokens r = { 0 }, x = { 0 };
            int eff = I.chat == RQ_CHAT_INSTRUCT ? 0 : th; // (the mode rendered)
            int q38 = I.chat == RQ_CHAT_QWEN38; // (Qwen3.8: earlier replies keep their reasoning, in both modes)
            ok &= rq_chat_render(M, msgs, 4, th, &r) == RQ_OK;
            rq_chat_system(M, "Be brief.", th, &x);
            rq_chat_user_history(M, "Hi", th, &x);
            RqTokens h = toks(eff && !q38 ? "Hello!" : "<think>\nhmm\n</think>\n\nHello!");
            rq_tokens_push(&x, h.v, h.n);
            rq_chat_end(M, &x);
            rq_chat_user(M, "Bye", th, &x);
            ok &= r.n == x.n && !memcmp(r.v, x.v, r.n * sizeof(int));
            // the history form from the generated tokens equals the one from the text (Qwen3.8: generated after the
            // generation suffix, "<think>\n" / the empty think block, which the history form puts back)
            RqTokens g = toks(!q38 ? "<think>\nhmm\n</think>\n\nHello!" : th ? "hmm\n</think>\n\nHello!" : "Hello!"), hg = { 0 };
            RqTokens hw = q38 && !th ? toks("<think>\n\n</think>\n\nHello!") : h;
            ok &= rq_chat_reply_history(M, g.v, g.n, th, &hg) == RQ_OK && hg.n == hw.n && !memcmp(hg.v, hw.v, hw.n * sizeof(int));
            if (hw.v != h.v) rq_tokens_free(&hw);
            // the generation prompt alone: the end of the user turn's piece
            RqTokens u = { 0 }, ua = { 0 }, ga = { 0 };
            rq_chat_user(M, "Hi", th, &u);
            rq_tokenize(M, "<|im_start|>user\nHi<|im_end|>\n", -1, &ua);
            ok &= rq_chat_assistant(M, th, &ga) == RQ_OK && ua.n + ga.n == u.n && !memcmp(u.v + ua.n, ga.v, ga.n * sizeof(int));
            rq_tokens_free(&u);
            rq_tokens_free(&ua);
            rq_tokens_free(&ga);
            rq_tokens_free(&r);
            rq_tokens_free(&x);
            rq_tokens_free(&h);
            rq_tokens_free(&g);
            rq_tokens_free(&hg);
        }
        RqMessage badm[2] = { { RQ_ROLE_ASSISTANT, "x" }, { RQ_ROLE_USER, "y" } };
        RqTokens r = { 0 };
        ok &= rq_chat_render(M, badm, 2, 0, &r) == RQ_ERR_ARG;
        rq_tokens_free(&r);
        report(ok, "chat_render", I.chat == RQ_CHAT_HYBRID ? "system, user, assistant, user; thinking off / on"
                                  : I.chat == RQ_CHAT_INSTRUCT ? "system, user, assistant, user (instruct-2507: no thinking)"
                                  : I.chat == RQ_CHAT_QWEN35 ? "system, user, assistant, user (qwen3.5); thinking off / on"
                                  : I.chat == RQ_CHAT_QWEN38 ? "system, user, assistant, user (qwen3.8: reasoning kept); thinking off / on"
                                  : "system, user, assistant, user (thinking-2507)");
    }

    // 7. incremental detokenization: the chunks are whole UTF-8 characters and concatenate to the text
    {
        const char *text = "naïve café 東京 🌍🎉 ok\xff end";
        RqTokens t = toks(text);
        RqDetok dt = { { 0 }, 0 };
        char buf[4096], chunk[256];
        size_t n = 0;
        int ok = 1;
        for (int i = 0; i < t.n; i++) {
            int w = rq_detok_push(M, &dt, t.v[i], chunk, sizeof(chunk));
            ok &= w >= 0;
            if (w > 0) { // a chunk never ends inside a (valid) multi-byte character
                unsigned char last = chunk[w - 1];
                int k = w - 1;
                while (k > 0 && (chunk[k] & 0xc0) == 0x80) k--;
                unsigned char lead = chunk[k];
                int need = lead >= 0xf0 ? 4 : lead >= 0xe0 ? 3 : lead >= 0xc0 ? 2 : 1;
                ok &= (last < 0x80 || w - k == need || lead == 0xff);
                memcpy(buf + n, chunk, w);
                n += w;
            }
        }
        int w = rq_detok_flush(&dt, buf + n, sizeof(buf) - n);
        n += w > 0 ? w : 0;
        ok &= n == strlen(text) && !memcmp(buf, text, n);
        snprintf(d, sizeof(d), "%d tokens, %zu bytes", t.n, n);
        report(ok, "detok_utf8", d);
        rq_tokens_free(&t);
    }

    // 8. sampler: a token mask (constrained decoding), greedy and sampled; no allowed token: an error
    {
        RqSeq *s = seq_new(0);
        float *lg, *l2 = malloc((size_t)V * sizeof(float));
        unsigned char *allow = calloc(V, 1);
        int ok = rq_prefill(c, s, A.v, A.n, RQ_LOGITS_LAST, NULL, &lg) == RQ_OK;
        int x = 1234, y = 5678, best = lg[x] >= lg[y] ? x : y;
        allow[x] = allow[y] = 1;
        RqSampling g = { 0, 0.9f, allow }, t = { 1.5f, 1.0f, allow };
        memcpy(l2, lg, V * sizeof(float));
        ok &= rq_sample(c, s, l2, &g) == best;
        rq_seq_seed(s, 7);
        for (int i = 0; i < 20 && ok; i++) {
            memcpy(l2, lg, V * sizeof(float));
            int k = rq_sample(c, s, l2, &t);
            ok &= k == x || k == y;
        }
        memset(allow, 0, V);
        memcpy(l2, lg, V * sizeof(float));
        ok &= rq_sample(c, s, l2, &g) == RQ_ERR_ARG;
        report(ok, "sample_mask", "2 allowed tokens: greedy picks the larger, sampling only those");
        // the file's sampling defaults (general.sampling.*): only the fields it has are written
        RqSampling d0 = { -1, -1, allow, -1 }, d1 = d0;
        int ms = rq_model_sampling(M, &d1);
        int okd = ms >= 0 && ms <= 7 && d1.allow == allow;
        okd &= (ms & RQ_SAMPLING_TEMP) ? d1.temperature >= 0 : d1.temperature == d0.temperature;
        okd &= (ms & RQ_SAMPLING_TOPP) ? d1.topp > 0 && d1.topp <= 1 : d1.topp == d0.topp;
        okd &= (ms & RQ_SAMPLING_TOPK) ? d1.top_k >= 0 : d1.top_k == d0.top_k;
        snprintf(d, sizeof(d), "set %d: temperature %g, top_p %g, top_k %d", ms, ms & 1 ? d1.temperature : 0,
                 ms & 2 ? d1.topp : 0, ms & 4 ? d1.top_k : 0);
        report(okd, "model_sampling", d);
        free(l2);
        free(allow);
        rq_seq_free(s);
    }

    // 9. batched decode rows (above bd_refs / bd_batch); C + 7 tokens: the long base (260 positions)
    rq_tokens_push(&C, dC, 7);
    batch_tests(c, A, B, C, refB, 0);

    free(refA);
    free(refB);
    free(refC);
    rq_tokens_free(&A);
    rq_tokens_free(&B);
    rq_tokens_free(&C);
    rq_ctx_free(c);
    rq_model_free(M);
    return failed;
}

static void batch_tests(RqCtx *c, RqTokens A, RqTokens B, RqTokens C, const float *refB, int only_digest) {
    // short set: prefixes of A (8..32 positions); long set: prefixes of C (248..259 positions: the rows cross
    // the first attention block of 256 at different steps). only_digest 2: the short passes (bd_long: and
    // the long ones)
    static uint64_t hs[BD_SEQS][BD_STEPS], hl[BD_SEQS][BD_STEPS];
    int e, ns = bd_n65 ? BD_SEQS : 17, nl = 9, small = only_digest == 2, lng = !small || bd_long;
    char d[256];
    RqSeq *bs = seq_new(0), *bl = seq_new(0), *sq[BD_SEQS];
    for (int i = 0; i < BD_SEQS; i++) sq[i] = seq_new(0);
    if ((e = rq_prefill(c, bs, A.v, A.n, RQ_LOGITS_NONE, NULL, NULL)) != RQ_OK) die("prefill", e);
    if (lng && (e = rq_prefill(c, bl, C.v, C.n, RQ_LOGITS_NONE, NULL, NULL)) != RQ_OK) die("prefill", e);
    if (!only_digest) {
        bd_refs(c, bs, 8, 25, ns, KVT, hs);
        bd_refs(c, bl, 248, 12, nl, KVT, hl);
    }
    float *out = malloc((size_t)BD_SEQS * V * sizeof(float));
    uint64_t (*rs)[BD_STEPS] = only_digest ? NULL : hs, (*rl)[BD_STEPS] = only_digest ? NULL : hl;

    // N decode rows per pass: the batched decode path (rows of one token each, up to 64), its small-batch
    // kernels (one row: like the GEMV; 2-8 rows: one register tile; 9 and 17: more than one); short and long
    int Ns[] = { 1, 2, 3, 5, 8, 9, 17 }, ok;
    for (unsigned k = 0; k < sizeof(Ns) / sizeof(Ns[0]); k++) {
        if (small && Ns[k] != 3 && Ns[k] != 9) continue;
        snprintf(d, sizeof(d), "batch_decode_n%d", Ns[k]);
        ok = bd_batch(c, sq, bs, 8, 25, Ns[k], rs, out, d);
        if (!only_digest) report(ok, d, "N sequences (8..32 positions), 2 steps together == each alone");
    }
    int Nl[] = { 3, 9 };
    for (unsigned k = 0; k < sizeof(Nl) / sizeof(Nl[0]); k++) {
        if (!lng || (small && Nl[k] != 3)) continue;
        snprintf(d, sizeof(d), "batch_decode_long_n%d", Nl[k]);
        ok = bd_batch(c, sq, bl, 248, 12, Nl[k], rl, out, d);
        if (!only_digest) report(ok, d, "positions 248..260: across the first attention block (split / not)");
    }

    // 65 rows: more than the batched decode path takes, so the prefill path (GEMM tiles) on a wider context
    if (!small && bd_n65) {
        RqCtx *c80 = rq_ctx_new(M, 80, 0, &e);
        if (!c80) die("rq_ctx_new", e);
        ok = bd_batch(c80, sq, bs, 8, 25, 65, rs, out, "batch_decode_n65");
        if (!only_digest) report(ok, "batch_decode_n65", "65 sequences in one pass (the prefill path's GEMM) == each alone");
        rq_ctx_free(c80);
    }

    // mixed passes: a prefill chunk of a new sequence next to decode rows of others (the prefill path):
    // (a) B's first 20 rows (all logits) + 3 long and 2 short decode rows (or 5 short), (b) a 2-row chunk +
    // 2 decode rows (few items: long rows' attention split over its blocks)
    for (int nlr = lng ? 3 : 0; nlr >= 0; nlr -= 3) { // nlr: rows from the long set (else all short); long, then short
        RqSeq *p = seq_new(0);
        int tok[8], okm = 1;
        RqSegment sg[8];
        uint64_t dg = 0;
        for (int i = 0; i < 5; i++) {
            okm &= seq_from(c, sq[i], i < nlr ? bl : bs, i < nlr ? plen(i, 248, 12) : plen(i, 8, 25)) == RQ_OK;
            tok[i] = tk(i, 0);
        }
        sg[0] = (RqSegment) { sq[0], &tok[0], 1, RQ_LOGITS_LAST };
        sg[1] = (RqSegment) { p, B.v, 20, RQ_LOGITS_ALL };
        for (int i = 1; i < 5; i++) sg[i + 1] = (RqSegment) { sq[i], &tok[i], 1, RQ_LOGITS_LAST };
        okm &= rq_forward_batch(c, sg, 6, out, NULL) == RQ_OK;
        for (int r = 0; r < 25 && okm; r++) { // rows: sq[0], B[0..19], sq[1..4]
            uint64_t x = hash_f(out + (size_t)r * V, V);
            dg = dg * 31 + x;
            if (only_digest) continue;
            if (r >= 1 && r <= 20) okm &= !memcmp(out + (size_t)r * V, refB + (size_t)(r - 1) * V, V * sizeof(float));
            else { int i = r == 0 ? 0 : r - 20; okm &= x == (i < nlr ? hl[i][0] : hs[i][0]); }
        }
        // (b): B[20..21] of p + 2 long rows (their second token)
        tok[5] = tk(0, 1);
        tok[6] = tk(1, 1);
        RqSegment sb[3] = { { sq[0], &tok[5], 1, RQ_LOGITS_LAST }, { p, B.v + 20, 2, RQ_LOGITS_LAST }, { sq[1], &tok[6], 1, RQ_LOGITS_LAST } };
        okm &= rq_forward_batch(c, sb, 3, out, NULL) == RQ_OK;
        for (int r = 0; r < 3 && okm; r++) {
            uint64_t x = hash_f(out + (size_t)r * V, V);
            dg = dg * 31 + x;
            if (only_digest) continue;
            if (r == 1) okm &= !memcmp(out + (size_t)r * V, refB + (size_t)21 * V, V * sizeof(float));
            else okm &= x == (nlr ? hl : hs)[r == 0 ? 0 : 1][1];
        }
        printf("DIGEST batch_mixed%s %016llx\n", nlr ? "" : "_short", (unsigned long long)dg);
        digest = digest * 31 + dg;
        if (!only_digest) report(okm, nlr ? "batch_mixed" : "batch_mixed_short",
                                 "a 20-row prefill chunk + 5 decode rows; a 2-row chunk + 2 decode rows");
        rq_seq_free(p);
    }
    // verify segments: (a) 4 rows (k = 3) of one sequence at position 20, all rows' logits, and the same with the
    // last row's only; (b) 9 rows (k = 8) at 250..258, across the first attention block (lng); (c) one pass with the
    // 4-row segment and 2 short decode rows; (d) one with a 6-row segment at 252..257 and a decode row at 257 (lng:
    // few attention items, so the long rows' attention is split over its blocks, in the prefill path the single
    // row's); each row == its sequence decoded token by token, and with a checkpoint per row (rq_seq_checkpoint_rows) each row's state is reachable by
    // rq_seq_truncate and continues exactly. On the default context (batch_decode_worker: segments of at most
    // bd_seg rows) and on one made with RUNQ_BD=64,2,8,1 (the prefill path for any segment of more than one row, as
    // before TODO S2). DIGEST lines: batch_verify (a), _long (b), _mixed (c), _mixed_long (d); + _prefill
    {
        static const int vp[3] = { 20, 250, 252 }, vr[3] = { 4, 9, 6 };
        static const char *vn[2] = { "", "_prefill" };
        int vt[3][VF_MAX + 1], tok2[2], e1;
        uint64_t vh[3][VF_MAX + 1];
        RqSeq *vs[3] = { seq_new(0), seq_new(0), seq_new(0) };
        char *old = getenv("RUNQ_BD") ? strdup(getenv("RUNQ_BD")) : NULL;
        setenv("RUNQ_BD", "64,2,8,1", 1);
        RqCtx *cp = rq_ctx_new(M, rq_ctx_max_batch(c), 0, &e1), *cv[2] = { c, cp };
        if (old) setenv("RUNQ_BD", old, 1); else unsetenv("RUNQ_BD");
        free(old);
        if (!cp) die("rq_ctx_new", e1);
        for (int q = 0; q < 3; q++) {
            for (int j = 0; j <= vr[q]; j++) vt[q][j] = tk(40 + 10 * q + j, 0);
            if (!only_digest && (q == 0 || lng)) vf_refs(c, q ? bl : bs, vp[q], vt[q], vr[q], vh[q]);
        }
        for (int x = 0; x < 2; x++) {
            RqCtx *cx = cv[x];
            int okv = 1, okm = 1;
            for (int q = 0; q < 2; q++) { // (a), (b)
                uint64_t dg = 0;
                if (q && !lng) continue;
                RqSeq *v = vs[q];
                okv &= seq_from(cx, v, q ? bl : bs, vp[q]) == RQ_OK && rq_seq_set_checkpoints(v, vr[q] + 1) == RQ_OK &&
                       rq_seq_checkpoint_rows(v, vp[q], vr[q]) == RQ_OK;
                RqSegment sv = { v, vt[q], vr[q], RQ_LOGITS_ALL };
                okv &= rq_forward_batch(cx, &sv, 1, out, NULL) == RQ_OK && rq_seq_len(v) == vp[q] + vr[q];
                for (int j = 0; j < vr[q] && okv; j++) {
                    uint64_t h = hash_f(out + (size_t)j * V, V);
                    dg = dg * 31 + h;
                    okv &= only_digest || h == vh[q][j];
                }
                if (!only_digest) okv &= vf_rollback(cx, v, vp[q], vt[q], vr[q], vh[q]);
                if (q == 0) { // the last row's logits only (into the context's buffer)
                    float *lg;
                    sv.logits = RQ_LOGITS_LAST;
                    okv &= seq_from(cx, v, bs, vp[0]) == RQ_OK && rq_forward_batch(cx, &sv, 1, NULL, &lg) == RQ_OK;
                    okv &= okv && (only_digest || hash_f(lg, V) == vh[0][vr[0] - 1]);
                }
                printf("DIGEST batch_verify%s%s %016llx\n", q ? "_long" : "", vn[x], (unsigned long long)dg);
                digest = digest * 31 + dg;
            }
            for (int ml = 0; ml <= lng; ml++) { // (c): rows [vs[0]: 4] [sq[2]: 1] [sq[1]: 1]; (d): [vs[2]: 6] [sq[3]: 1 at 257]
                RqSegment sm[4];
                int nsm = 0, row = 0, q = 2 * ml;
                uint64_t dm = 0;
                okm &= seq_from(cx, vs[q], q ? bl : bs, vp[q]) == RQ_OK && rq_seq_set_checkpoints(vs[q], vr[q] + 1) == RQ_OK &&
                       rq_seq_checkpoint_rows(vs[q], vp[q], vr[q]) == RQ_OK;
                sm[nsm++] = (RqSegment) { vs[q], vt[q], vr[q], RQ_LOGITS_ALL };
                for (int i = 0; i < 2 - ml; i++) { // (sq[3] of the long set, or sq[2] and sq[1] of the short one)
                    int k = ml ? 3 : 2 - i;
                    okm &= seq_from(cx, sq[k], ml ? bl : bs, ml ? plen(k, 248, 12) : plen(k, 8, 25)) == RQ_OK;
                    tok2[i] = tk(k, 0);
                    sm[nsm++] = (RqSegment) { sq[k], &tok2[i], 1, RQ_LOGITS_LAST };
                }
                okm &= rq_forward_batch(cx, sm, nsm, out, NULL) == RQ_OK;
                for (int i = 0; i < nsm && okm; i++)
                    for (int j = 0; j < sm[i].n; j++, row++) {
                        uint64_t h = hash_f(out + (size_t)row * V, V);
                        dm = dm * 31 + h;
                        if (only_digest) continue;
                        okm &= i == 0 ? h == vh[q][j] : sm[i].seq == sq[3] ? h == hl[3][0] : h == hs[sm[i].seq == sq[1] ? 1 : 2][0];
                    }
                if (!only_digest) okm &= vf_rollback(cx, vs[q], vp[q], vt[q], vr[q], vh[q]);
                printf("DIGEST batch_verify_mixed%s%s %016llx\n", ml ? "_long" : "", vn[x], (unsigned long long)dm);
                digest = digest * 31 + dm;
            }
            if (!only_digest) {
                char nm[64];
                snprintf(nm, sizeof(nm), "batch_verify%s", vn[x]);
                snprintf(d, sizeof(d), "%s: a verify segment of 4 rows at 20 (all / the last row's logits)%s == decoded one by "
                         "one, each row's checkpoint continues exactly", x ? "prefill path" : "decode-rows path",
                         lng ? ", 9 rows at 250..258" : "");
                report(okv, nm, d);
                snprintf(nm, sizeof(nm), "batch_verify_mixed%s", vn[x]);
                snprintf(d, sizeof(d), "%s: a verify segment of 4 rows + 2 decode rows in one pass%s == each alone, the "
                         "checkpoints too", x ? "prefill path" : "decode-rows path", lng ? "; 6 rows at 252..257 + a decode row "
                         "at 257 (split attention)" : "");
                report(okm, nm, d);
            }
        }
        rq_ctx_free(cp);
        for (int q = 0; q < 3; q++) rq_seq_free(vs[q]);
    }
    printf("DIGEST all%s %016llx\n", small ? "_small" : "", (unsigned long long)digest);

    if (!only_digest) {
        // batched sampling == rq_sample per row (tokens and RNG states), mixed settings per row
        RqSampling cf[5] = { { 0, 0, NULL }, { 0.8f, 0.9f, NULL }, { 1.5f, 0.95f, NULL }, { 1.0f, 1.0f, NULL },
                             { 1.0f, 0.95f, NULL, 20 } }, cfg[9];
        float *l2 = malloc((size_t)9 * V * sizeof(float));
        int t1[9], t2[9], oks = bd_batch(c, sq, bs, 8, 25, 9, NULL, out, "batch_sample_rows");
        unsigned long long rng[9];
        for (int i = 0; i < 9; i++) {
            cfg[i] = cf[i % 5];
            rq_seq_seed(sq[i], 77 + i);
        }
        memcpy(l2, out, (size_t)9 * V * sizeof(float));
        oks &= rq_sample_batch(c, sq, out, 9, cfg, 9, t1) == RQ_OK;
        for (int i = 0; i < 9; i++) {
            rng[i] = *rq_seq_rng(sq[i]);
            rq_seq_seed(sq[i], 77 + i);
            t2[i] = rq_sample(c, sq[i], l2 + (size_t)i * V, &cfg[i]);
            oks &= t1[i] == t2[i] && t1[i] >= 0 && rng[i] == *rq_seq_rng(sq[i]);
        }
        oks &= rq_sample_batch(c, sq, out, 9, cfg, 2, t1) == RQ_ERR_ARG;
        report(oks, "batch_sample", "rq_sample_batch == rq_sample per row (greedy, top-p, full distribution, top-k)");
        free(l2);

        // KV bf16 (f16 above; f32: the plain fp32 tile layout of the same kernels) and the bf16 dot-product attention
        // on both 16-bit caches: 3 long sequences
        int kts[] = { RQ_KV_BF16, RQ_KV_BF16_DOT, RQ_KV_F16_DOT };
        for (unsigned ki = 0; ki < sizeof(kts) / sizeof(kts[0]) && bd_kv; ki++) {
            int kvt = kts[ki];
            if (kvt == KVT) continue;
            RqSeq *b2 = rq_seq_new(M, 0, kvt, &e), *q2[3];
            if (!b2) die("rq_seq_new", e);
            if ((e = rq_prefill(c, b2, C.v, C.n, RQ_LOGITS_NONE, NULL, NULL)) != RQ_OK) die("prefill", e);
            uint64_t h2[3][BD_STEPS];
            bd_refs(c, b2, 248, 12, 3, kvt, h2);
            for (int i = 0; i < 3; i++)
                if (!(q2[i] = rq_seq_new(M, 0, kvt, &e))) die("rq_seq_new", e);
            snprintf(d, sizeof(d), "batch_decode_long_%s", rq_kv_name(kvt));
            ok = bd_batch(c, q2, b2, 248, 12, 3, h2, out, d);
            report(ok, d, "3 long sequences, KV cache type");
            for (int i = 0; i < 3; i++) rq_seq_free(q2[i]);
            rq_seq_free(b2);
        }
        // copies: a copied prefix continues exactly as the original (and errors)
        RqSeq *f32 = rq_seq_new(M, 0, RQ_KV_F32, &e);
        ok = rq_seq_copy(sq[0], bl, rq_seq_len(bl) + 1) == RQ_ERR_ARG && rq_seq_copy(f32, bl, 4) == RQ_ERR_ARG &&
             rq_seq_copy(sq[0], sq[0], 0) == RQ_ERR_ARG && rq_seq_copy(sq[0], bl, 0) == RQ_OK && rq_seq_len(sq[0]) == 0;
        rq_seq_free(f32);
        report(ok, "seq_copy_errors", "past the source length, another KV type, itself");
    }
    free(out);
    for (int i = 0; i < BD_SEQS; i++) rq_seq_free(sq[i]);
    rq_seq_free(bs);
    rq_seq_free(bl);
}

// ------------------------------------------------------------------------------------------- two models
// api_test --two: a script of steps per model (TwoM), run interleaved with the other model's and alone. Each step
// hashes the logits it produces (and the tokens / lengths / RNG states where it has them)

#define TW_MAIN 10 // steps run while both models are loaded
#define TW_ALL 13  // + the steps B runs after A is freed
static const char *tw_name[TW_ALL] = { "prefill_all", "prefill_chunks", "decode", "batch_mixed", "batch_decode", "truncate",
                                       "truncate_batch", "copy", "sample_batch", "generate", "decode_after",
                                       "batch_after", "generate_after" };
static const int tw_tk[8] = { 279, 6722, 13, 198, 262, 470, 1000, 1037 };

typedef struct {
    RqModel *m;
    RqModelInfo info;
    RqCtx *c;
    RqSeq *s[4];
    RqTokens P, Q, R, U; // prompts: P ~33 tokens, Q ~73 (2 chunks of 64), R ~57, U a chat user turn
    float *out;          // max(64, P.n) rows of logits
    uint64_t h[TW_ALL];
    int bad;             // the first step that returned an error + 1 (0: none)
} TwoM;

typedef struct { uint64_t h; int V; } TwGen;

static uint64_t tw_mix(uint64_t h, uint64_t x) { return h * 31 + x; }

static uint64_t tw_rows(const TwoM *w, const float *x, int n) {
    uint64_t h = 0;
    for (int i = 0; i < n; i++)
        h = tw_mix(h, hash_f(x + (size_t)i * w->info.vocab_size, w->info.vocab_size));
    return h;
}

static void tw_on_logits(void *u, int pos, const float *lg) {
    TwGen *g = u;
    g->h = tw_mix(tw_mix(g->h, (uint64_t)pos), hash_f(lg, g->V));
}

static int tw_on_token(void *u, int t) {
    TwGen *g = u;
    g->h = tw_mix(g->h, (uint64_t)t);
    return 0;
}

static void tw_free(TwoM *w) {
    for (int j = 0; j < 4; j++) rq_seq_free(w->s[j]);
    rq_ctx_free(w->c);
    rq_model_free(w->m);
    rq_tokens_free(&w->P);
    rq_tokens_free(&w->Q);
    rq_tokens_free(&w->R);
    rq_tokens_free(&w->U);
    free(w->out);
    memset(w->s, 0, sizeof(w->s));
    w->c = NULL;
    w->m = NULL;
    w->out = NULL;
}

static int tw_load(TwoM *w, const char *path, char *err, size_t errlen) {
    // the model, a context (64 rows per pass), 4 sequences of 512 positions and the prompts; 0 on failure (err)
    int e = RQ_OK;
    memset(w, 0, sizeof(*w));
    if (!(w->m = rq_model_load(path, NULL, err, errlen))) return 0;
    rq_model_info(w->m, &w->info);
    if (!(w->c = rq_ctx_new(w->m, 64, 512, &e))) goto fail;
    for (int j = 0; j < 4; j++)
        if (!(w->s[j] = rq_seq_new(w->m, 512, RQ_KV_DEFAULT, &e))) goto fail;
    e = rq_tokenize(w->m, "The capital of France is Paris, and the capital of Germany is Berlin. Rivers: the Seine, "
                    "the Rhine and the Danube flow through famous cities.", -1, &w->P);
    if (e == RQ_OK)
        e = rq_tokenize(w->m, "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n\n# Print the first ten "
                        "Fibonacci numbers, one per line, then their sum, and explain why the recursion is slow for "
                        "large n and how memoization fixes it.\nfor i in range(10):\n    print(fib(i))\n", -1, &w->Q);
    for (int r = 0; r < 3 && e == RQ_OK; r++)
        e = rq_tokenize(w->m, "A lighthouse keeper trims the wicks, polishes the lens and writes the weather into the log. ",
                        -1, &w->R);
    if (e == RQ_OK) e = rq_chat_user(w->m, "Name three colours.", 0, &w->U);
    if (e != RQ_OK) goto fail;
    if (w->Q.n < 66 || w->R.n < 40 || w->P.n < 8) {
        snprintf(err, errlen, "%s: prompts too short (%d, %d, %d tokens)", path, w->P.n, w->Q.n, w->R.n);
        return 0;
    }
    if (!(w->out = malloc((size_t)(w->P.n > 64 ? w->P.n : 64) * w->info.vocab_size * sizeof(float)))) { e = RQ_ERR_NOMEM; goto fail; }
    return 1;
fail:
    snprintf(err, errlen, "%s: setup: %s", path, rq_strerror(e));
    return 0;
}

static void tw_step(TwoM *w, int i) {
    RqCtx *c = w->c;
    RqSeq **s = w->s;
    int V = w->info.vocab_size, e = RQ_OK;
    uint64_t h = 0;
    float *lg = NULL;
    switch (i) {
    case 0: // s0: P, every row's logits
        e = rq_prefill(c, s[0], w->P.v, w->P.n, RQ_LOGITS_ALL, w->out, NULL);
        h = tw_rows(w, w->out, w->P.n);
        break;
    case 1: // s1: Q in 2 chunks, the last row; Qwen3.5: a checkpoint at 40 (taken inside the first chunk)
        e = rq_seq_checkpoint_at(s[1], 40);
        if (e == RQ_OK) e = rq_prefill(c, s[1], w->Q.v, w->Q.n, RQ_LOGITS_LAST, NULL, &lg);
        break;
    case 2: // s0: a decode step
        e = rq_decode(c, s[0], tw_tk[0], &lg);
        break;
    case 3: { // a prefill chunk of s2 (R[0..19], every row; Qwen3.5: a checkpoint at 12) between decode rows of s0, s1
        RqSegment sg[3] = { { s[0], &tw_tk[1], 1, RQ_LOGITS_LAST }, { s[2], w->R.v, 20, RQ_LOGITS_ALL },
                            { s[1], &tw_tk[2], 1, RQ_LOGITS_LAST } };
        e = rq_seq_checkpoint_at(s[2], 12);
        if (e == RQ_OK) e = rq_forward_batch(c, sg, 3, w->out, NULL);
        h = tw_rows(w, w->out, 22);
        break;
    }
    case 4: { // decode rows of s2, s0, s1 in one pass
        RqSegment sg[3] = { { s[2], w->R.v + 20, 1, RQ_LOGITS_LAST }, { s[0], &tw_tk[3], 1, RQ_LOGITS_LAST },
                            { s[1], &tw_tk[4], 1, RQ_LOGITS_LAST } };
        e = rq_forward_batch(c, sg, 3, w->out, NULL);
        h = tw_rows(w, w->out, 3);
        break;
    }
    case 5: { // s1 back to 41 (Qwen3.5: to its checkpoint at 40), Q's rest prefilled again
        e = rq_seq_truncate(s[1], 41);
        int L = rq_seq_len(s[1]);
        if (e == RQ_OK && L != (w->info.linear_layers ? 40 : 41)) e = RQ_ERR_STATE; // (not where expected)
        if (e == RQ_OK) e = rq_prefill(c, s[1], w->Q.v + L, w->Q.n - L, RQ_LOGITS_LAST, NULL, &lg);
        h = (uint64_t)L;
        break;
    }
    case 6: { // s2 back to 15 (Qwen3.5: 12), R[L..29] again (every row) next to a decode row of s0
        e = rq_seq_truncate(s[2], 15);
        int L = rq_seq_len(s[2]);
        if (e == RQ_OK && L != (w->info.linear_layers ? 12 : 15)) e = RQ_ERR_STATE;
        RqSegment sg[2] = { { s[0], &tw_tk[5], 1, RQ_LOGITS_LAST }, { s[2], w->R.v + L, 30 - L, RQ_LOGITS_ALL } };
        if (e == RQ_OK) e = rq_forward_batch(c, sg, 2, w->out, NULL);
        h = tw_mix((uint64_t)L, tw_rows(w, w->out, 31 - L));
        break;
    }
    case 7: // s3 := s0 (rq_seq_copy, whole), a decode step on the copy
        e = rq_seq_copy(s[3], s[0], rq_seq_len(s[0]));
        if (e == RQ_OK) e = rq_decode(c, s[3], tw_tk[6], &lg);
        h = (uint64_t)rq_seq_len(s[3]);
        break;
    case 8: { // a decode row of each sequence in one pass, then rq_sample_batch (seeded top-p / greedy per row)
        RqSegment sg[4];
        RqSampling cf[2] = { { 0.8f, 0.9f, NULL }, { 0, 0, NULL } }, cfg[4];
        int tok[4] = { tw_tk[7], tw_tk[0], tw_tk[1], tw_tk[2] }, got[4] = { 0 };
        for (int j = 0; j < 4; j++) {
            sg[j] = (RqSegment) { s[j], &tok[j], 1, RQ_LOGITS_LAST };
            cfg[j] = cf[j % 2];
            rq_seq_seed(s[j], 11 + j);
        }
        e = rq_forward_batch(c, sg, 4, w->out, NULL);
        h = tw_rows(w, w->out, 4);
        if (e == RQ_OK) e = rq_sample_batch(c, s, w->out, 4, cfg, 4, got);
        for (int j = 0; j < 4; j++) h = tw_mix(tw_mix(h, (uint64_t)got[j]), *rq_seq_rng(s[j]));
        break;
    }
    case 9: case 12: { // rq_generate on s3 (reset) from a chat turn: greedy 5 tokens; after: seeded 6 tokens
        TwGen g = { 0, V };
        RqCallbacks cb = { tw_on_token, tw_on_logits, &g };
        RqSampling cfg = { i == 9 ? 0 : 0.8f, 0.9f, NULL };
        RqGenStats st;
        rq_seq_reset(s[3]);
        rq_seq_seed(s[3], 3);
        e = rq_generate(c, s[3], w->U.v, w->U.n, &cfg, i == 9 ? 5 : 6, &cb, &st);
        h = tw_mix(tw_mix(g.h, (uint64_t)st.n_gen), (uint64_t)st.stop);
        break;
    }
    case 10: // s1: a decode step
        e = rq_decode(c, s[1], tw_tk[3], &lg);
        break;
    case 11: { // a prefill chunk of s2 (R's next tokens, every row) between decode rows of s0 and s1
        int L = rq_seq_len(s[2]), n = w->R.n - L < 24 ? w->R.n - L : 24;
        RqSegment sg[3] = { { s[0], &tw_tk[4], 1, RQ_LOGITS_LAST }, { s[2], w->R.v + L, n, RQ_LOGITS_ALL },
                            { s[1], &tw_tk[5], 1, RQ_LOGITS_LAST } };
        e = n > 0 ? rq_forward_batch(c, sg, 3, w->out, NULL) : RQ_ERR_ARG;
        h = tw_rows(w, w->out, n + 2);
        break;
    }
    }
    if (lg) h = tw_mix(h, hash_f(lg, V));
    w->h[i] = e == RQ_OK ? h : 0;
    if (e != RQ_OK && !w->bad) w->bad = i + 1;
}

static int tw_alone(const char *path, uint64_t *h, char *err, size_t errlen) {
    // every step of one model, alone: 0, -1 (setup failed: err) or the first failing step + 1
    TwoM w;
    int ok = tw_load(&w, path, err, errlen);
    for (int i = 0; i < TW_ALL && ok; i++)
        tw_step(&w, i);
    memcpy(h, w.h, sizeof(w.h));
    tw_free(&w);
    return ok ? w.bad : -1;
}

static int tw_alone_proc(const char *path, uint64_t *h, char *err, size_t errlen) {
    // tw_alone in a fresh process (fork: nothing is loaded and the pool not started yet), else in this one
#ifndef _WIN32
    struct { int bad; uint64_t h[TW_ALL]; char err[400]; } r;
    int fd[2], st = 0;
    if (pipe(fd)) { snprintf(err, errlen, "pipe failed"); return -1; }
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
        close(fd[0]);
        memset(&r, 0, sizeof(r));
        r.bad = tw_alone(path, r.h, r.err, sizeof(r.err));
        _exit(write(fd[1], &r, sizeof(r)) == (ssize_t)sizeof(r) ? 0 : 1);
    }
    close(fd[1]);
    size_t got = 0;
    while (pid > 0 && got < sizeof(r)) {
        ssize_t k = read(fd[0], (char *)&r + got, sizeof(r) - got);
        if (k <= 0) break;
        got += (size_t)k;
    }
    close(fd[0]);
    if (pid < 0 || waitpid(pid, &st, 0) != pid || !WIFEXITED(st) || WEXITSTATUS(st) || got != sizeof(r)) {
        snprintf(err, errlen, "%s alone: the child process failed (status %d)", path, st);
        return -1;
    }
    memcpy(h, r.h, sizeof(r.h));
    snprintf(err, errlen, "%s", r.err);
    return r.bad;
#else
    return tw_alone(path, h, err, errlen);
#endif
}

static int tw_gs_refused(const char *path, char *d, size_t dn) {
    // loading path (another activation group size than the loaded models') fails with the group size error
    char err[512];
    RqModelOpts o = { 0, 0, 1 };
    RqModel *g = rq_model_load(path, &o, err, sizeof(err));
    if (g) {
        RqModelInfo gi;
        rq_model_info(g, &gi);
        snprintf(d, dn, "%s loaded (group size %d): expected a group size error", path, gi.group_size);
        rq_model_free(g);
        return 0;
    }
    snprintf(d, dn, "%s", err);
    return strstr(err, "group size") != NULL;
}

static int tw_compare(const TwoM *w, const uint64_t *ref, int i0, int i1, char *d, size_t dn) {
    // steps i0..i1-1 of w == ref (the model alone); d: the differing ones
    int ok = !w->bad || w->bad > i1;
    size_t k = strlen(d);
    for (int i = i0; i < i1; i++)
        if (w->h[i] != ref[i] && k + 32 < dn) {
            k += snprintf(d + k, dn - k, "%s%s", ok ? "; differ: " : ", ", tw_name[i]);
            ok = 0;
        }
    if (w->bad && w->bad <= i1) snprintf(d + k, dn - k, "; step %s failed", tw_name[w->bad - 1]);
    return ok;
}

static int two_main(int argc, char **argv) {
    const char *path[2] = { NULL, NULL }, *pg = NULL;
    int bad_args = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--gs") && i + 1 < argc) pg = argv[++i];
        else if (argv[i][0] == '-') bad_args = 1;
        else if (!path[0]) path[0] = argv[i];
        else if (!path[1]) path[1] = argv[i];
        else bad_args = 1;
    }
    if (bad_args || !path[1]) {
        fprintf(stderr, "usage: api_test --two MODEL_A MODEL_B [--gs MODEL_OTHER_GROUP_SIZE]\n");
        return 2;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    rq_threads_init(0);
    char err[512], d[1024], gsd[600] = "";
    uint64_t ref[2][TW_ALL];
    for (int k = 0; k < 2; k++) {
        int r = tw_alone_proc(path[k], ref[k], err, sizeof(err));
        if (r) {
            if (r > 0) snprintf(err, sizeof(err), "%s alone: step %s failed", path[k], tw_name[r - 1]);
            printf("FAIL setup  %s\n", err);
            return 1;
        }
    }
    // A, then (--gs) the other group size refused, B
    TwoM w[2];
    int gs_ok = 1, gs_n = 0, gs_a = 0;
    if (!tw_load(&w[0], path[0], err, sizeof(err))) {
        printf("FAIL setup  %s\n", err);
        return 1;
    }
    gs_a = w[0].info.group_size;
    if (pg) gs_ok &= tw_gs_refused(pg, gsd, sizeof(gsd)), gs_n++;
    if (!tw_load(&w[1], path[1], err, sizeof(err))) {
        printf("FAIL setup  %s\n", err);
        return 1;
    }
    // a sequence of one model on the other's context: refused, nothing run
    {
        float *lg;
        RqSegment sg[2] = { { w[0].s[0], tw_tk, 1, RQ_LOGITS_LAST }, { w[1].s[0], tw_tk, 1, RQ_LOGITS_LAST } };
        int ok = rq_decode(w[0].c, w[1].s[0], tw_tk[0], &lg) == RQ_ERR_ARG;
        ok &= rq_prefill(w[1].c, w[0].s[1], w[0].P.v, 4, RQ_LOGITS_NONE, NULL, NULL) == RQ_ERR_ARG;
        ok &= rq_forward_batch(w[0].c, sg, 2, w[0].out, NULL) == RQ_ERR_ARG;
        ok &= rq_seq_copy(w[0].s[3], w[1].s[0], 0) == RQ_ERR_ARG;
        for (int k = 0; k < 2; k++)
            for (int j = 0; j < 4; j++) ok &= rq_seq_len(w[k].s[j]) == 0;
        report(ok, "two_cross_model", rq_ctx_error(w[0].c));
    }
    // the steps of A and B interleaved (A first on even steps, B on odd ones); the other group size again
    for (int i = 0; i < TW_MAIN; i++) {
        tw_step(&w[i & 1], i);
        tw_step(&w[!(i & 1)], i);
        if (pg && i == 4) gs_ok &= tw_gs_refused(pg, gsd, sizeof(gsd)), gs_n++;
    }
    // A freed, B goes on
    tw_free(&w[0]); // (its hashes kept)
    if (pg) gs_ok &= tw_gs_refused(pg, gsd, sizeof(gsd)), gs_n++;
    for (int i = TW_MAIN; i < TW_ALL; i++)
        tw_step(&w[1], i);
    for (int k = 0; k < 2; k++) {
        d[0] = 0;
        int ok = tw_compare(&w[k], ref[k], 0, TW_MAIN, d, sizeof(d));
        char head[600];
        snprintf(head, sizeof(head), "%d steps interleaved with %s's == %s alone (another process)%s", TW_MAIN,
                 k ? "A" : "B", path[k], d);
        report(ok, k ? "two_interleaved_b" : "two_interleaved_a", head);
    }
    d[0] = 0;
    int ok = tw_compare(&w[1], ref[1], TW_MAIN, TW_ALL, d, sizeof(d));
    char head[600];
    snprintf(head, sizeof(head), "B's %d steps after A was freed (decode, a mixed pass, seeded rq_generate) == alone%s",
             TW_ALL - TW_MAIN, d);
    report(ok, "two_free_other", head);
    for (int k = 0; k < 2; k++) {
        uint64_t dg = 0;
        for (int i = 0; i < (k ? TW_ALL : TW_MAIN); i++) dg = tw_mix(dg, w[k].h[i]);
        printf("DIGEST two_%c %016llx\n", k ? 'b' : 'a', (unsigned long long)dg);
    }
    if (pg) {
        snprintf(head, sizeof(head), "%d loads refused (A loaded; A and B mid-run; B after A freed), both went on: %s", gs_n, gsd);
        report(gs_ok, "two_gs_refused", head);
    }
    tw_free(&w[1]);
    // nothing loaded: the other group size loads now, and runs
    if (pg) {
        RqModelOpts o = { 0, 0, 1 };
        RqModel *g = rq_model_load(pg, &o, err, sizeof(err));
        RqModelInfo gi = { 0 };
        int e = RQ_ERR_LOAD;
        if (g) {
            rq_model_info(g, &gi);
            RqCtx *c = rq_ctx_new(g, 0, 64, &e);
            RqSeq *s = c ? rq_seq_new(g, 64, RQ_KV_DEFAULT, &e) : NULL;
            float *lg = NULL;
            if (s) e = rq_decode(c, s, tw_tk[0], &lg);
            ok = e == RQ_OK && lg && lg[0] == lg[0];
            rq_seq_free(s);
            rq_ctx_free(c);
            rq_model_free(g);
        }
        snprintf(head, sizeof(head), "group size %d after %d's models were freed: a decode step (%s)", gi.group_size, gs_a,
                 g ? rq_strerror(e) : err);
        report(g && ok && gi.group_size != gs_a, "two_gs_after_free", head);
    }
    return failed;
}

// ---------------------------------------------------------------------------------------- api_test --spec
// rq_generate_spec == rq_generate (header). Each run on fresh sequences: the reference with rq_generate, then
// rq_generate_spec per draft setting

// src/internal.h's hooks (the library's, linked from its objects): rq_prefill with every row's final hidden state
// (hidden, n x dim; NULL: none) and an MTP view's h input rows (h_in; NULL: zeros); an MTP view sequence's final rows
// (hvalid), the target's hidden state it keeps (hprev at hpos)
int fw_rows(RqCtx *c, RqSeq *s, const int *tokens, int n, int mode, float *out, float **logits, float *hidden, const float *h_in);
void mtp_seq_state(const RqSeq *s, int *hvalid, int *hpos, const float **hprev);

static int D; // the model's dim

typedef struct {
    const char *name;
    int prompt;          // sp_prompt: 0 a chat turn, thinking off; 1 thinking on; 2 raw text (code)
    RqSampling cfg;
    unsigned long long seed;
    int max_new;
    int cap, dcap;       // the target's / draft's capacity past the prompt (0: the model's; < 0: shorter than it)
    int turn2;           // a second call: both cut back 3 positions (at most to the prompt), then a user turn
    int quick;           // in --quick
    int ck;              // turn2: 1 the cut at the prompt's end with a checkpoint of the target there (the truncation lands
                         // exactly where the kept rows end: the MTP's first row of the second call takes zeros), 2 the
                         // checkpoint 2 positions below it (both fed again: the MTP's h known), 3 no cut (the second
                         // call goes on where the first ended, as a Qwen3.8 chat: the MTP's first row takes the kept h)
    int stop_at;         // on_token stops the reply at its stop_at-th token (inside a verify pass: RQ_STOP_CALLBACK)
} SpCase;

#define SP_CALLS 2
typedef struct {
    GenRec rec;
    int stop[SP_CALLS], ngen[SP_CALLS], len;
    unsigned long long rng, htok, hnext;
    RqSpecStats st;      // summed over the calls (gen: the last call's)
    int *ctok[SP_CALLS], nctok[SP_CALLS]; // the reference's (o NULL): the sequence's tokens after each call
} SpOut;

// RqSpecOpts.guess_hook (rq_spec_run's test hook): the guesses replaced by a pattern against the plain run's tokens
// (right: the token the target samples there; wrong: another one), the draft's own depth-0 picks recorded (with an MTP
// view: the lockstep check compares them with the reference's)
enum { HK_RIGHT, HK_WRONG, HK_ROTATE, HK_MIX, HK_N };
static const char *hk_name[] = { "right", "wrong", "wrong-at-rotating-depth", "mix" };
#define HK_PICKS 4096
typedef struct {
    int pattern, k, call, round;
    const SpOut *ref;   // the plain run (its tokens per call)
    unsigned long long rng;
    int npick, pick_pos[HK_PICKS], pick_tok[HK_PICKS], pick_call[HK_PICKS];
} SpHook;

static int sp_hook(void *u, int pos, int depth, int guess, float p1) {
    SpHook *h = u;
    (void)p1;
    if (depth == 0) {
        h->round++;
        if (h->npick < HK_PICKS)
            h->pick_pos[h->npick] = pos, h->pick_tok[h->npick] = guess, h->pick_call[h->npick++] = h->call;
    }
    const int *rt = h->ref->ctok[h->call];
    int n = h->ref->nctok[h->call], right = pos < n ? rt[pos] : -1, wrong = ((right >= 0 ? right : guess >= 0 ? guess : 0) + 1) % V;
    switch (h->pattern) {
    case HK_RIGHT: return right; // (-1 past the reply's end: no more guesses)
    case HK_WRONG: return wrong;
    case HK_ROTATE: return depth == h->round % h->k ? wrong : right;
    default: {
        h->rng ^= h->rng << 13, h->rng ^= h->rng >> 7, h->rng ^= h->rng << 17;
        int r = (int)(h->rng >> 32) % 10;
        return r < 4 ? right : r < 7 ? guess : r < 9 ? wrong : -1;
    }
    }
}

// the MTP drafter's bookkeeping (TODO S9b, mtp_lockstep): what a run's calls tell the check
typedef struct {
    int approx;            // the rows expected to take zeros (the first row after a cut that lands where the kept rows end)
    int apos[SP_CALLS];    // their positions
    int keep[SP_CALLS];    // per call but the last: the view's rows the next call keeps (its picks below them stay valid)
    int got_approx;        // RqSpecStats.mtp_approx summed
} SpLock;

static RqModel *SD;
static int SD_MTP; // the draft is the target's MTP view (DRAFT mtp)
static RqTokens sp_prompt[4], sp_turn2;

static int sp_lockstep(RqCtx *c, RqCtx *dc, RqSeq *s, RqSeq *ms, const SpLock *lk, const SpHook *hk, int *npicks, char *d, size_t dn);

static int sp_run(RqCtx *c, RqCtx *dc, const SpCase *k, const RqSpecOpts *o, SpOut *out, int *lock, char *ld, size_t ldn) {
    // o NULL: rq_generate; o->k 0: the lookup alone, no draft model (dc, ds NULL). lock (an MTP view as the draft): the
    // lockstep check after the calls (sp_lockstep: 1 passed, 0 failed, -1 not run), its detail in ld
    int e, np = sp_prompt[k->prompt].n;
    RqSeq *s = rq_seq_new(M, k->cap ? np + k->cap : 0, RQ_KV_DEFAULT, &e);
    if (!s) die("rq_seq_new", e);
    RqSeq *ds = o && o->k ? rq_seq_new(SD, k->dcap ? np + k->dcap : 0, RQ_KV_DEFAULT, &e) : NULL;
    if (o && o->k && !ds) die("rq_seq_new (draft)", e);
    if (!ds) dc = NULL;
    SpHook *hk = o && o->guess_hook ? o->hook_user : NULL;
    SpLock lk;
    memset(&lk, 0, sizeof(lk));
    if (lock) *lock = -1;
    rq_seq_seed(s, k->seed);
    memset(out, 0, sizeof(*out));
    out->rec.s = s;
    out->rec.stop_at = k->stop_at;
    RqCallbacks cb = { gr_token, gr_logits, &out->rec };
    RqTokens t = { 0 };
    rq_tokens_push(&t, sp_prompt[k->prompt].v, np);
    if (k->ck && k->ck < 3 && (e = rq_seq_checkpoint_at(s, np - (k->ck == 2 ? 2 : 0))) != RQ_OK) die("rq_seq_checkpoint_at", e);
    for (int call = 0; call <= k->turn2; call++) {
        RqSpecStats st;
        if (hk) hk->call = call;
        if (o) {
            e = rq_generate_spec(c, s, dc, ds, t.v, t.n, &k->cfg, k->max_new, o, &cb, &st);
            out->st.passes += st.passes, out->st.drafted += st.drafted, out->st.accepted += st.accepted;
            out->st.refed += st.refed, out->st.draft_steps += st.draft_steps;
            out->st.lookup_passes += st.lookup_passes, out->st.lookup_drafted += st.lookup_drafted;
            out->st.lookup_accepted += st.lookup_accepted;
            out->st.mtp_rows += st.mtp_rows, out->st.mtp_approx += st.mtp_approx;
        } else {
            e = rq_generate(c, s, t.v, t.n, &k->cfg, k->max_new, &cb, &st.gen);
        }
        if (e != RQ_OK) break;
        out->st.gen = st.gen;
        out->stop[call] = st.gen.stop;
        out->ngen[call] = st.gen.n_gen;
        if (!o) { // (the reference: its tokens after the call, for the hook)
            out->nctok[call] = rq_seq_len(s);
            out->ctok[call] = malloc((size_t)rq_seq_len(s) * sizeof(int) + 1);
            memcpy(out->ctok[call], rq_seq_tokens(s), (size_t)rq_seq_len(s) * sizeof(int));
        }
        if (call < k->turn2) { // (Qwen3.5: the truncation goes back to a checkpoint: what it lost is fed again)
            int L = k->ck == 3 ? rq_seq_len(s) : k->ck ? np : rq_seq_len(s) - 3 < np ? np : rq_seq_len(s) - 3, Lprev = rq_seq_len(s);
            t.n = 0;
            rq_tokens_push(&t, rq_seq_tokens(s), L);
            if ((e = rq_seq_truncate(s, L)) != RQ_OK) break;
            int cut = rq_seq_len(s);
            memmove(t.v, t.v + cut, (size_t)(L - cut) * sizeof(int));
            t.n = L - cut;
            rq_tokens_push(&t, sp_turn2.v, sp_turn2.n);
            if (ds && SD_MTP) {
                // the view's rows the next call keeps: its common prefix with the target's tokens then t (K); its row K
                // takes zeros when the target does not compute K - 1 again (it resumes at K: cut == K) and the call before
                // did not end there (K != Lprev), if row K fits (its capacity)
                const int *mt = rq_seq_tokens(ds), *st0 = rq_seq_tokens(s);
                int K = 0, ml = rq_seq_len(ds);
                while (K < ml && K < cut + t.n && mt[K] == (K < cut ? st0[K] : t.v[K - cut])) K++;
                lk.keep[call] = K;
                if (K > 0 && K < rq_seq_capacity(ds) && cut == K && K != Lprev) lk.apos[lk.approx++] = K;
            }
        }
    }
    out->len = rq_seq_len(s);
    out->htok = hash_f((const float *)rq_seq_tokens(s), (size_t)out->len);
    out->rng = *rq_seq_rng(s);
    if (e == RQ_OK && ds && SD_MTP && lock) { // (before the probe decode step below: the target as the loop left it)
        lk.got_approx = out->st.mtp_approx;
        int np_ = 0;
        *lock = sp_lockstep(c, dc, s, ds, &lk, hk, &np_, ld, ldn);
    }
    float *lg;
    if (e == RQ_OK && out->len < rq_seq_capacity(s) && (e = rq_decode(c, s, 1000, &lg)) == RQ_OK)
        out->hnext = rq_logits_hash(lg, V);
    rq_tokens_free(&t);
    rq_seq_free(s);
    rq_seq_free(ds);
    return e;
}

static void sp_out_free(SpOut *o) {
    for (int i = 0; i < SP_CALLS; i++) free(o->ctok[i]);
}

static int file_same(const char *a, const char *b) {
    // two files byte for byte (both readable)
    FILE *f = fopen(a, "rb"), *g = fopen(b, "rb");
    int ok = f && g, x = 0, y = 0;
    while (ok && (x = fgetc(f)) == (y = fgetc(g)) && x != EOF) {}
    ok &= x == y;
    if (f) fclose(f);
    if (g) fclose(g);
    return ok;
}

static int sp_lockstep(RqCtx *c, RqCtx *dc, RqSeq *s, RqSeq *ms, const SpLock *lk, const SpHook *hk, int *npicks, char *d, size_t dn) {
    // the MTP view's sequence after the loop (rev-math M4) against a reference built from the final tokens alone: the
    // target's hidden rows of a fresh prefill, the view's fresh prefill over them (h shifted by one, zeros at the expected
    // positions). Its length == the target's (to its capacity), mtp_approx == expected, every row final (hvalid), the
    // target's last hidden state kept (hpos); the KV cache bitwise (rq_seq_save of copies of the whole 16-position tiles:
    // a cache's positions past its length hold what was written there before); a probe row at the next position on both
    // (the loop's with its hprev) bitwise; the hook's depth-0 picks (the view's own first guess of each round) == the
    // reference's rq_spec_pick at those rows
    int L = rq_seq_len(s), capm = rq_seq_capacity(ms), ML = rq_seq_len(ms), hv, hp, e, ok = 1;
    const float *hprev;
    mtp_seq_state(ms, &hv, &hp, &hprev);
    int want = L < capm ? L : capm;
    char why[256] = "";
#define LK_FAIL(...) do { if (ok) snprintf(why, sizeof(why), __VA_ARGS__); ok = 0; } while (0)
    if (ML != want) LK_FAIL("length %d, target %d (capacity %d)", ML, L, capm);
    if (lk->got_approx != lk->approx) LK_FAIL("mtp_approx %d, expected %d", lk->got_approx, lk->approx);
    if (hv != ML) LK_FAIL("hvalid %d of %d rows", hv, ML);
    if (hp != L - 1) LK_FAIL("hpos %d, the target's last position %d", hp, L - 1);
    float *H = malloc((size_t)L * D * sizeof(float)), *hin = calloc((size_t)(want + 1) * D, sizeof(float));
    float *out = malloc((size_t)64 * V * sizeof(float));
    int *pick = malloc((size_t)(want + 1) * sizeof(int));
    RqSeq *rs = rq_seq_new(M, rq_seq_capacity(s), RQ_KV_DEFAULT, &e), *rv = rq_seq_new(SD, capm, RQ_KV_DEFAULT, &e);
    if (!H || !hin || !out || !pick || !rs || !rv) die("sp_lockstep", RQ_ERR_NOMEM);
    e = fw_rows(c, rs, rq_seq_tokens(s), L, RQ_LOGITS_NONE, NULL, NULL, H, NULL);
    for (int q = 1; q < want; q++) {
        int z = 0;
        for (int i = 0; i < lk->approx; i++) z |= lk->apos[i] == q;
        if (!z) memcpy(hin + (size_t)q * D, H + (size_t)(q - 1) * D, (size_t)D * sizeof(float));
    }
    for (int q = 0; q < want && e == RQ_OK; q += 64) { // (the reference's picks: its rows' rq_spec_pick)
        int r = want - q < 64 ? want - q : 64;
        e = fw_rows(dc, rv, rq_seq_tokens(s) + q, r, RQ_LOGITS_ALL, out, NULL, NULL, hin + (size_t)q * D);
        for (int i = 0; i < r && e == RQ_OK; i++) {
            float p1;
            pick[q + i] = rq_spec_pick(out + (size_t)i * V, V, NULL, 0, &p1);
        }
    }
    if (e != RQ_OK) LK_FAIL("reference: %s (%s)", rq_strerror(e), rq_ctx_error(dc));
    // the KV cache: copies of the whole tiles below the length (fresh sequences: zeros past them; the kept state cleared:
    // a copy to the length keeps it, which rq_seq_save would add), saved, byte for byte
    int L16 = ML / 16 * 16;
    if (ok && L16 > 0) {
        RqSeq *a = rq_seq_new(SD, capm, RQ_KV_DEFAULT, &e), *b = rq_seq_new(SD, capm, RQ_KV_DEFAULT, &e);
        char pa[512], pb[512];
        const char *tmp = getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp";
        snprintf(pa, sizeof(pa), "%s/api_test_lock_%d_a.rqseq", tmp, (int)getpid());
        snprintf(pb, sizeof(pb), "%s/api_test_lock_%d_b.rqseq", tmp, (int)getpid());
        int okk = a && b && rq_seq_copy(a, ms, L16) == RQ_OK && rq_seq_copy(b, rv, L16) == RQ_OK && rq_mtp_keep(a, 0, NULL) == RQ_OK &&
                  rq_mtp_keep(b, 0, NULL) == RQ_OK && rq_seq_save(a, pa) == RQ_OK && rq_seq_save(b, pb) == RQ_OK && file_same(pa, pb);
        if (!okk) LK_FAIL("the KV cache of rows 0..%d differs from the reference's", L16 - 1);
        remove(pa);
        remove(pb);
        rq_seq_free(a);
        rq_seq_free(b);
    }
    // a probe row at the next position: the loop's sequence with its hprev, the reference with h_{L-1}
    int probe = ok && ML == L && L < capm && hp == L - 1;
    if (probe) {
        float *lg;
        int tok = 1000 % V;
        unsigned long long h1 = 0, h2 = 1;
        if (fw_rows(dc, ms, &tok, 1, RQ_LOGITS_LAST, NULL, &lg, NULL, hprev) == RQ_OK) h1 = rq_logits_hash(lg, V);
        if (fw_rows(dc, rv, &tok, 1, RQ_LOGITS_LAST, NULL, &lg, NULL, H + (size_t)(L - 1) * D) == RQ_OK) h2 = rq_logits_hash(lg, V);
        if (h1 != h2) LK_FAIL("the probe row at %d differs (the kept hidden state or the cache)", L);
    }
    // the depth-0 picks (an earlier call's: rows the next call kept)
    int np = 0;
    for (int i = 0; hk && i < hk->npick && ok; i++) {
        int row = hk->pick_pos[i] - 1, cl = hk->pick_call[i];
        if (cl < hk->call && row >= lk->keep[cl]) continue;
        if (row >= want || hk->pick_tok[i] != pick[row])
            LK_FAIL("the first guess at %d: %d, the reference's %d", row + 1, hk->pick_tok[i], row < want ? pick[row] : -1);
        np++;
    }
    *npicks = np;
    snprintf(d, dn, "%d rows%s%s%s", ML, lk->approx ? " (1 approximate)" : "", probe ? " + probe" : "", ok ? "" : ": ");
    if (!ok) snprintf(d + strlen(d), dn - strlen(d), "%s", why);
#undef LK_FAIL
    free(H);
    free(hin);
    free(out);
    free(pick);
    rq_seq_free(rs);
    rq_seq_free(rv);
    return ok;
}

static int sp_same(const SpOut *a, const SpOut *b) {
    return gr_same(&a->rec, &b->rec) && !memcmp(a->stop, b->stop, sizeof(a->stop)) && !memcmp(a->ngen, b->ngen, sizeof(a->ngen)) &&
           a->len == b->len && a->htok == b->htok && a->rng == b->rng && a->hnext == b->hnext;
}

static int lk_brute(const int *toks, int n, int t, int kmax, int min_len, int *guess, int *len) {
    // rq_lookup_draft's rule, by a scan: the RQ_LOOKUP_CHAIN most recent j (1 <= j <= n - 1) with (toks[j - 1],
    // toks[j]) == (toks[n - 1], t), the longest match backwards (up to RQ_LOOKUP_LMAX; the most recent among equals)
    int best = -1, bl = 0, c = 0;
    *len = 0;
    for (int j = n - 1; j >= 1 && c < RQ_LOOKUP_CHAIN && n >= 2; j--) {
        if (toks[j] != t || toks[j - 1] != toks[n - 1]) continue;
        c++;
        int l = 2;
        while (l < RQ_LOOKUP_LMAX && j - l >= 0 && toks[j - l] == toks[n - l]) l++;
        if (l > bl) bl = l, best = j;
        if (bl == RQ_LOOKUP_LMAX) break;
    }
    if (best < 0 || bl < min_len || kmax < 1) return 0;
    *len = bl;
    int m = n - best < kmax ? n - best : kmax;
    for (int i = 0; i < m; i++) guess[i] = best + 1 + i < n ? toks[best + 1 + i] : t;
    return m;
}

static void lookup_index_test(void) {
    // rq_lookup_* == lk_brute at every position of synthetic sequences (alphabets of 2-300 tokens, repeated blocks with
    // edits, so that chains are long and matches of every length occur), indexed incrementally (1-3 tokens per call) and
    // after a reset; kmax 1 / 5 / 30, min_len 2 / 4
    enum { N = 3000 };
    static int tk[N];
    int g1[RQ_SPEC_K_MAX], g2[RQ_SPEC_K_MAX], bad = 0, found = 0, lmax = 0, checks = 0;
    uint64_t x = 88172645463325252ull;
    const int alpha[] = { 2, 5, 40, 300 };
    for (int ai = 0; ai < 4; ai++) {
        for (int i = 0; i < N; i++) {
            x ^= x << 13, x ^= x >> 7, x ^= x << 17;
            int r = (int)(x % 100);
            tk[i] = i >= 64 && r < 70 ? tk[i - 1 - (int)(x >> 40) % 63] : (int)((x >> 20) % (unsigned)alpha[ai]);
            if (i >= 200 && r >= 70 && r < 85) tk[i] = tk[i - 150]; // (long copies)
        }
        RqLookup *lk = rq_lookup_new(N);
        if (!lk) die("rq_lookup_new", RQ_ERR_NOMEM);
        for (int pass = 0; pass < 2; pass++) {
            rq_lookup_reset(lk);
            for (int n = 0; n < N - 1;) {
                n += 1 + (int)((x >> 33) % 3), x ^= x << 13, x ^= x >> 7, x ^= x << 17;
                if (n > N - 1) n = N - 1;
                if (rq_lookup_add(lk, tk, n) != RQ_OK) bad++;
                const int ks[] = { 1, 5, 30 }, ms[] = { 2, 4 };
                int t = tk[n];
                for (int a = 0; a < 3; a++)
                    for (int b = 0; b < 2; b++) {
                        int l1, l2, n1 = rq_lookup_draft(lk, tk, t, ks[a], ms[b], g1, &l1);
                        int n2 = lk_brute(tk, n, t, ks[a], ms[b], g2, &l2);
                        bad += n1 != n2 || l1 != l2 || memcmp(g1, g2, (size_t)n1 * sizeof(int));
                        found += n1 > 0, lmax = l1 > lmax ? l1 : lmax, checks++;
                    }
            }
        }
        bad += rq_lookup_add(lk, tk, N + 1) != RQ_ERR_ARG; // (past the capacity)
        rq_lookup_free(lk);
    }
    char d[160];
    snprintf(d, sizeof(d), "%d lookups == a brute-force scan (%d found a match, the longest %d tokens)%s", checks, found,
             lmax, bad ? ": DIFFER" : "");
    report(!bad && found > checks / 4 && lmax == RQ_LOOKUP_LMAX, "spec_lookup_index", d);
}

static int spec_main(int argc, char **argv) {
    const char *path[2] = { NULL, NULL }, *other = NULL;
    int quick = 0, bad = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--quick")) quick = 1;
        else if (!strcmp(argv[i], "--other") && i + 1 < argc) other = argv[++i];
        else if (argv[i][0] == '-') bad = 1;
        else if (!path[0]) path[0] = argv[i];
        else if (!path[1]) path[1] = argv[i];
        else bad = 1;
    }
    if (bad || !path[1]) {
        fprintf(stderr, "usage: api_test --spec TARGET DRAFT|mtp [--quick] [--other MODEL_OTHER_VOCABULARY]\n");
        return 2;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    rq_threads_init(0);
    char err[512], d[4096];
    RqMtpOpts mo = { 0, -1 };
    SD_MTP = !strcmp(path[1], "mtp"); // (the target's own MTP block: rq_mtp_load)
    if (!(M = rq_model_load(path[0], NULL, err, sizeof(err))) ||
        !(SD = SD_MTP ? rq_mtp_load(M, &mo, err, sizeof(err)) : rq_model_load(path[1], NULL, err, sizeof(err)))) {
        printf("FAIL setup  %s\n", err);
        return 1;
    }
    rq_model_info(M, &I);
    V = I.vocab_size;
    D = I.dim;
    int e;
    RqCtx *c = rq_ctx_new(M, 64, 0, &e), *dc = c ? rq_ctx_new(SD, 64, 0, &e) : NULL;
    if (!dc) die("rq_ctx_new", e);

    lookup_index_test();
    // the vocabulary check: the draft and the target itself accepted; another vocabulary refused
    {
        double t0 = rq_now();
        int ok = rq_spec_check(M, SD, err, sizeof(err)) == RQ_OK && rq_spec_check(M, M, NULL, 0) == RQ_OK;
        double ms = (rq_now() - t0) * 1e3 / 2;
        snprintf(d, sizeof(d), "the draft and the target itself accepted (%.1f ms per check)", ms);
        if (other) {
            RqModelOpts o = { 0, 1, 0 };
            RqModel *x = rq_model_load(other, &o, err, sizeof(err));
            int r = x ? rq_spec_check(M, x, err, sizeof(err)) : RQ_ERR_LOAD;
            ok &= r == RQ_ERR_ARG && strstr(err, "not usable as a draft");
            size_t dl = strlen(d);
            snprintf(d + dl, sizeof(d) - dl, "; refused: %s", err);
            rq_model_free(x);
        }
        report(ok, "spec_vocab", d);
    }

    // prompts (the chat ones as the CLI renders them; a model without thinking renders its own form)
    rq_chat_user(M, "Name three colours.", 0, &sp_prompt[0]);
    rq_chat_user(M, "Write a C function that reverses a string in place.", 1, &sp_prompt[1]);
    sp_prompt[2] = toks("def is_prime(n):\n    \"\"\"Return True if n is a prime number.\"\"\"\n");
    // (a block of the prompt that the model copies: greedy, both test models do, verbatim)
    sp_prompt[3] = toks("def moving_average(xs, n):\n    out = []\n    for i in range(len(xs)):\n        if i < n - 1:\n"
                        "            out.append(None)\n        else:\n            out.append(sum(xs[i-n+1:i+1]) / n)\n"
                        "    return out\n\n\ndef moving_sum(xs, n):\n    out = []\n    for i in range(len(xs)):\n"
                        "        if i < n - 1:\n            out.append(None)\n        else:\n"
                        "            out.append(sum(xs[i-n+1:i+1]))\n    return out\n\n\n"
                        "# the two functions again, copied verbatim:\n\ndef moving_average(xs, n):\n");
    rq_chat_end(M, &sp_turn2);
    rq_chat_user(M, "And three animals?", 0, &sp_turn2);
    const SpCase cases[] = {
        { "chat_greedy", 0, { 0, 0.9f, NULL, 0 }, 1, 96, 0, 0, 0, 1 },
        { "chat_seeded_topk", 0, { 1.0f, 0.95f, NULL, 20 }, 7, 96, 0, 0, 0, 0 },
        { "think_greedy_limit", 1, { 0, 0.9f, NULL, 0 }, 1, 45, 0, 0, 0, 0 },
        { "think_seeded_limit", 1, { 0.8f, 0.9f, NULL, 0 }, 11, 37, 0, 0, 0, 1 },
        { "text_greedy_context", 2, { 0, 0.9f, NULL, 0 }, 1, 0, 29, 0, 0, 0 },
        { "text_seeded_context_draft_short", 2, { 1.0f, 0.95f, NULL, 20 }, 5, 0, 29, 12, 0, 1 },
        { "text_draft_shorter_than_prompt", 2, { 0, 0.9f, NULL, 0 }, 1, 12, 0, -3, 0, 0 },
        { "chat_turn2_seeded", 0, { 0.7f, 0.9f, NULL, 40 }, 3, 24, 0, 0, 1, 0 },
        { "think_turn2_greedy", 1, { 0, 0.9f, NULL, 0 }, 1, 30, 0, 0, 1, 1 },
        { "text_copy_greedy", 3, { 0, 0.9f, NULL, 0 }, 1, 110, 0, 0, 0, 1 },
        { "text_copy_seeded_turn2", 3, { 0.6f, 0.9f, NULL, 20 }, 13, 70, 0, 0, 1, 0 },
        { "think_turn2_cut_at_checkpoint", 1, { 0, 0.9f, NULL, 0 }, 1, 30, 0, 0, 1, 1, 1 },
        { "chat_turn2_checkpoint_below_cut", 0, { 0.7f, 0.9f, NULL, 40 }, 3, 24, 0, 0, 1, 0, 2 },
        { "chat_turn2_no_cut", 0, { 0, 0.9f, NULL, 0 }, 1, 20, 0, 0, 1, 1, 3 },
        { "chat_callback_stop", 0, { 1.0f, 0.95f, NULL, 20 }, 9, 96, 0, 0, 0, 1, 0, 25 },
        { "chat_seeded_end_token", 0, { 1.0f, 0.95f, NULL, 20 }, 9, 96, 0, 0, 0, 0 }, // (synth-q35-27b-mtp: <|im_end|> 41st)
    };
    // G7, auto: the cost models calibrated here (on empty sequences), and a cheap one (guesses nearly free: every pass
    // guesses, the maximum of either source)
    RqPassCost tcal, dcal, tfree, dfree;
    {
        RqSeq *x = seq_new(0), *y = rq_seq_new(SD, 0, RQ_KV_DEFAULT, &e);
        if (!y || rq_pass_cost_measure(c, x, 600, 2, &tcal) != RQ_OK || rq_pass_cost_measure(dc, y, 600, 2, &dcal) != RQ_OK)
            die("rq_pass_cost_measure", RQ_ERR_STATE);
        rq_seq_free(x);
        rq_seq_free(y);
        tfree = tcal;
        tfree.att = tfree.att_x = 0;
        for (int i = 1; i < tfree.n; i++) tfree.f[i] = 1 + 1e-4 * i;
        dfree = dcal;
        dfree.t_w = 1e-9;
        dfree.att = dfree.att_x = 0;
    }
    const RqSpecOpts opts[] = { { 1, 0, 0, 0 }, { 3, 0, 0, 0 }, { 8, 0, 0, 0 }, { 8, 0.5f, 0, 0 }, { 0, 0, 4, 0 },
                                { 0, 0, 30, 4 }, { 3, 0, 8, 3 }, { 8, 0.5f, 3, 0 }, { 3, 0, 3, 0, &tcal, &dcal },
                                { 8, 0, 8, 0, &tfree, &dfree } };
    const int quick_opts[] = { 1, 3, 4, 6, 9 };
    const char *stop_name[] = { "end", "limit", "context", "callback" };
    // the hook's patterns (RqSpecOpts.guess_hook) per case: with an MTP view at k 3 and k 8 with tau 0.5 (the hook overrides
    // tau; --quick: k 3), with a draft model at k 3 (not --quick)
    const RqSpecOpts hk_opts[] = { { 3, 0, 0, 0 }, { 8, 0.5f, 0, 0 } };
    int nhk = SD_MTP ? (quick ? 1 : 2) : quick ? 0 : 1;
    static SpHook hk;
    int refed = 0;
    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        const SpCase *k = &cases[ci];
        if (quick && !k->quick) continue;
        SpOut ref, got;
        int lock_ok = 1, nlock = 0, lock, ok = (e = sp_run(c, dc, k, NULL, &ref, NULL, NULL, 0)) == RQ_OK;
        char ld[512] = "", lfail[600] = "";
        snprintf(d, sizeof(d), "%d%s tok, stop %s%s%s:", ref.ngen[0] + ref.ngen[1], k->turn2 ? " (2 calls)" : "",
                 stop_name[ref.stop[k->turn2] & 3], k->turn2 ? "/" : "", k->turn2 ? stop_name[ref.stop[1] & 3] : "");
        for (size_t oi = 0; oi < sizeof(opts) / sizeof(opts[0]) && ok; oi++) {
            int qi = 0;
            while (qi < 5 && quick_opts[qi] != (int)oi) qi++;
            if (quick && qi == 5) continue;
            const RqSpecOpts *o = &opts[oi];
            int okk = (e = sp_run(c, dc, k, o, &got, &lock, ld, sizeof(ld))) == RQ_OK && sp_same(&ref, &got);
            if (lock >= 0) {
                nlock++;
                if (!lock && lock_ok) snprintf(lfail, sizeof(lfail), " FAILED at setting %zu: %s", oi, ld);
                lock_ok &= lock;
            }
            okk &= got.st.accepted <= got.st.drafted && (I.linear_layers || got.st.refed == 0);
            okk &= got.st.lookup_accepted <= got.st.lookup_drafted && got.st.lookup_drafted <= got.st.drafted &&
                   (o->lookup || got.st.lookup_passes == 0) && (o->k || got.st.lookup_drafted == got.st.drafted);
            refed += got.st.refed;
            size_t dl = strlen(d);
            char how[48] = "";
            if (o->cost) snprintf(how, sizeof(how), "auto%s:", o->cost == &tfree ? "-free" : "");
            if (o->k) snprintf(how + strlen(how), sizeof(how) - strlen(how), "k%d%s", o->k, o->tau > 0 ? " tau 0.5" : "");
            if (o->lookup) {
                size_t hl = strlen(how);
                snprintf(how + hl, sizeof(how) - hl, "%sL%d%s", hl ? "+" : "", o->lookup, o->lookup_min ? (o->lookup_min == 3 ? ">=3" : ">=4") : "");
            }
            snprintf(d + dl, sizeof(d) - dl, " %s %d/%d in %d passes%s%s", how, got.st.accepted, got.st.drafted, got.st.passes,
                     got.st.refed ? " REFED" : "", okk ? "" : " DIFFER");
            if (o->k && o->lookup) {
                dl = strlen(d);
                snprintf(d + dl, sizeof(d) - dl, " (lookup %d/%d)", got.st.lookup_accepted, got.st.lookup_drafted);
            }
            ok &= okk;
            if (k->prompt == 3 && k->cfg.temperature == 0 && oi == 4 && !SD_MTP) { // the copy, the lookup alone (4 guesses;
                // the model pairs': the synthetic MTP files' main models do not copy)
                double tpp = (double)got.st.gen.n_gen / (got.st.passes > 0 ? got.st.passes : 1);
                char dd[160];
                snprintf(dd, sizeof(dd), "a verbatim copy, lookup 4 alone: %.2f tokens per pass (%d/%d guesses accepted)", tpp,
                         got.st.accepted, got.st.drafted);
                report(okk && tpp >= 3.0, "spec_lookup_copy", dd);
            }
        }
        // the hook's patterns: right (every guess accepted), wrong (none), wrong at a depth rotating per round, a seeded mix
        int npick = 0;
        for (int hi = 0; hi < nhk && ok && e == RQ_OK; hi++)
            for (int p = 0; p < HK_N && ok; p++) {
                RqSpecOpts o = hk_opts[hi];
                memset(&hk, 0, sizeof(hk));
                hk.pattern = p, hk.k = o.k, hk.ref = &ref, hk.rng = 0x9E3779B97F4A7C15ull ^ (ci * 131 + p);
                o.guess_hook = sp_hook;
                o.hook_user = &hk;
                int okk = (e = sp_run(c, dc, k, &o, &got, &lock, ld, sizeof(ld))) == RQ_OK && sp_same(&ref, &got);
                okk &= p != HK_RIGHT || got.st.accepted == got.st.drafted;
                okk &= p != HK_WRONG || got.st.accepted == 0;
                okk &= got.st.accepted <= got.st.drafted && (I.linear_layers || got.st.refed == 0);
                refed += got.st.refed;
                if (lock >= 0) {
                    nlock++;
                    if (!lock && lock_ok) snprintf(lfail, sizeof(lfail), " FAILED at hook %s k%d: %s", hk_name[p], o.k, ld);
                    lock_ok &= lock;
                    npick += hk.npick;
                }
                size_t dl = strlen(d);
                snprintf(d + dl, sizeof(d) - dl, " hook-%s k%d%s %d/%d%s", hk_name[p], o.k, o.tau > 0 ? " tau 0.5" : "",
                         got.st.accepted, got.st.drafted, okk ? "" : " DIFFER");
                ok &= okk;
            }
        if (e != RQ_OK) {
            size_t dl = strlen(d);
            snprintf(d + dl, sizeof(d) - dl, " error %s (%s)", rq_strerror(e), rq_ctx_error(c));
        }
        char name[96];
        snprintf(name, sizeof(name), "spec_%s", k->name);
        report(ok, name, d);
        if (SD_MTP) { // (rev-math M4: the view's rows, hidden states and guesses as a fresh lockstep run over the final tokens gives them)
            snprintf(name, sizeof(name), "spec_%s/mtp_lockstep", k->name);
            snprintf(d, sizeof(d), "%d runs (every setting and hook): the MTP view == a fresh lockstep prefill of the final tokens "
                     "(length, mtp_approx, hvalid, hpos, the KV cache bitwise, a probe row, %d first guesses), last: %s%s", nlock, npick,
                     ld, lfail);
            report(ok && nlock > 0 && lock_ok, name, d);
        }
        sp_out_free(&ref);
        if (ci == 0) { // k 0: rq_generate itself (no draft)
            RqSpecOpts z = { 0, 0 };
            RqSeq *s = seq_new(0);
            rq_seq_seed(s, k->seed);
            memset(&got, 0, sizeof(got));
            got.rec.s = s;
            RqCallbacks cb = { gr_token, gr_logits, &got.rec };
            RqSpecStats st;
            int ok0 = rq_generate_spec(c, s, NULL, NULL, sp_prompt[0].v, sp_prompt[0].n, &k->cfg, k->max_new, &z, &cb, &st) == RQ_OK;
            ok0 &= gr_same(&ref.rec, &got.rec) && st.gen.stop == ref.stop[0] && st.gen.n_gen == ref.ngen[0] && st.passes == 0;
            report(ok0, "spec_k0", "k 0 (no draft): rq_generate");
            rq_seq_free(s);
        }
    }
    // Qwen3.5: every rollback on a checkpoint (both sequences)
    if (I.linear_layers) report(refed == 0, "spec_rollback_exact", refed ? "positions fed again after a rollback" : "no position fed again");
    if (SD_MTP) { // an MTP view's final rows (hvalid): rq_prefill_spec, rq_seq_common_prefix, copy, truncate, save / load, reset
        RqTokens u = toks("The quick brown fox jumps over the lazy dog. The quick brown fox");
        int n = u.n, hv, hp, ok = 1;
        const float *hpv;
        RqSeq *s = seq_new(0), *s2 = seq_new(0), *ms = rq_seq_new(SD, 0, RQ_KV_DEFAULT, &e), *m2 = rq_seq_new(SD, 0, RQ_KV_DEFAULT, &e);
        float *l1 = NULL, *l2 = NULL, *o1 = malloc((size_t)n * V * sizeof(float)), *o2 = malloc((size_t)n * V * sizeof(float));
        // the target's logits: rq_prefill's (all rows, the last); the view in lockstep: every row final, h_{n-2} kept
        ok &= rq_prefill(c, s2, u.v, n - 1, RQ_LOGITS_ALL, o1, NULL) == RQ_OK &&
              rq_prefill_spec(c, s, dc, ms, u.v, n - 1, RQ_LOGITS_ALL, o2, NULL) == RQ_OK && same(o1, o2, (size_t)(n - 1) * V);
        mtp_seq_state(ms, &hv, &hp, &hpv);
        ok &= rq_seq_len(ms) == n - 1 && hv == n - 1 && hp == n - 2 && rq_seq_common_prefix(ms, u.v, n) == n - 1;
        rq_seq_reset(s);
        rq_seq_reset(s2);
        ok &= rq_prefill(c, s2, u.v, n, RQ_LOGITS_LAST, NULL, &l1) == RQ_OK && (memcpy(o1, l1, V * sizeof(float)), 1) &&
              rq_prefill_spec(c, s, dc, ms, u.v, n, RQ_LOGITS_LAST, NULL, &l2) == RQ_OK && same(o1, l2, V);
        mtp_seq_state(ms, &hv, &hp, &hpv);
        ok &= rq_seq_len(ms) == n && hv == n && hp == n - 1; // (its n - 1 rows kept: the same tokens)
        // a row fed outside the loop (provisional: hvalid unchanged) is not kept by the common prefix
        ok &= rq_seq_truncate(ms, n - 1) == RQ_OK && fw_rows(dc, ms, u.v + n - 1, 1, RQ_LOGITS_NONE, NULL, NULL, NULL, NULL) == RQ_OK;
        mtp_seq_state(ms, &hv, &hp, &hpv);
        ok &= rq_seq_len(ms) == n && hv == n - 1 && hp == -1 && rq_seq_common_prefix(ms, u.v, n) == n - 1;
        ok &= rq_seq_copy(m2, ms, n) == RQ_OK && (mtp_seq_state(m2, &hv, &hp, &hpv), hv == n - 1 && hp == -1);
        ok &= rq_seq_truncate(ms, 3) == RQ_OK && (mtp_seq_state(ms, &hv, &hp, &hpv), hv == 3) && rq_seq_common_prefix(ms, u.v, n) == 3;
        char pa[512];
        snprintf(pa, sizeof(pa), "%s/api_test_hvalid_%d.rqseq", getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp", (int)getpid());
        ok &= rq_seq_save(m2, pa) == RQ_OK && rq_seq_load(ms, pa, n) == RQ_OK && (mtp_seq_state(ms, &hv, &hp, &hpv), hv == n && hp == -1);
        remove(pa);
        rq_seq_reset(ms);
        mtp_seq_state(ms, &hv, &hp, &hpv);
        ok &= hv == 0 && hp == -1;
        { // a prompt rq_prefill refuses (past the capacity; a token out of range after the first chunk): the same error,
          // nothing fed (not the first chunk)
            int nb = rq_ctx_max_batch(c) > 1 ? rq_ctx_max_batch(c) : 64, np = nb + 16, *pt = malloc((size_t)np * sizeof(int));
            RqSeq *sc = rq_seq_new(M, nb + 8, RQ_KV_DEFAULT, &e);
            if (!pt || !sc) die("spec_mtp_hvalid", RQ_ERR_NOMEM);
            for (int i = 0; i < np; i++) pt[i] = u.v[i % n];
            rq_seq_reset(ms);
            int e1 = rq_prefill(c, sc, pt, np, RQ_LOGITS_LAST, NULL, &l1), e2 = rq_prefill_spec(c, sc, dc, ms, pt, np, RQ_LOGITS_LAST, NULL, &l2);
            ok &= e1 != RQ_OK && e2 == e1 && rq_seq_len(sc) == 0;
            pt[nb + 2] = V;
            rq_seq_reset(s);
            e1 = rq_prefill(c, s, pt, nb + 4, RQ_LOGITS_LAST, NULL, &l1);
            e2 = rq_prefill_spec(c, s, dc, ms, pt, nb + 4, RQ_LOGITS_LAST, NULL, &l2);
            ok &= e1 == RQ_ERR_ARG && e2 == e1 && rq_seq_len(s) == 0;
            rq_seq_free(sc);
            free(pt);
        }
        report(ok, "spec_mtp_hvalid", "rq_prefill_spec: the target's logits == rq_prefill's, the view's rows final (hvalid) with "
               "the last hidden state kept; a row fed outside the loop not kept by rq_seq_common_prefix; copy / truncate / "
               "save + load / reset; a prompt rq_prefill refuses: its error, nothing fed");
        // the MTP block of another model (a second load of the target file): refused before any pass
        if (!quick) {
            ok = 1;
            RqModel *M2 = rq_model_load(path[0], NULL, err, sizeof(err));
            RqModel *v2 = M2 ? rq_mtp_load(M2, &mo, err, sizeof(err)) : NULL;
            RqSeq *x = v2 ? rq_seq_new(v2, 0, RQ_KV_DEFAULT, &e) : NULL;
            RqCtx *xc = v2 ? rq_ctx_new(v2, 8, 0, &e) : NULL;
            RqSpecOpts so = { 3, 0, 0, 0 };
            rq_seq_reset(s);
            ok &= x && xc && rq_generate_spec(c, s, xc, x, u.v, n, &cases[0].cfg, 4, &so, NULL, NULL) == RQ_ERR_ARG && rq_seq_len(s) == 0 &&
                  rq_prefill_spec(c, s, xc, x, u.v, n, RQ_LOGITS_NONE, NULL, NULL) == RQ_ERR_ARG && rq_seq_len(s) == 0;
            rq_ctx_free(xc);
            rq_seq_free(x);
            rq_model_free(v2);
            rq_model_free(M2);
            report(ok, "spec_mtp_other_model", "the MTP view of another model (a second load): RQ_ERR_ARG before any pass");
        }
        free(o1);
        free(o2);
        rq_seq_free(s);
        rq_seq_free(s2);
        rq_seq_free(ms);
        rq_seq_free(m2);
        rq_tokens_free(&u);
    }
    // a verify row's checkpoint kept (rq_seq_checkpoint_at below the length: the server's </think> inside a verify pass),
    // the rows' dropped (rq_seq_checkpoint_rows_drop: the end of a job's speculation); a truncation there == a fresh prefill
    {
        RqTokens u = toks("The quick brown fox jumps over the lazy dog. The quick brown fox");
        int P = u.n - 4, held[RQ_CHECKPOINTS_MAX], nh, ok = 1;
        float *ref = logits_alone(c, u.v, u.n, NULL, 0), *lg;
        RqSeq *s = seq_new(0);
        rq_seq_set_checkpoints(s, 6);
        ok &= rq_prefill(c, s, u.v, P, RQ_LOGITS_NONE, NULL, NULL) == RQ_OK && rq_seq_checkpoint_rows(s, P, 3) == RQ_OK;
        ok &= rq_prefill(c, s, u.v + P, 4, RQ_LOGITS_ALL, NULL, NULL) == RQ_ERR_ARG; // (all rows need an output buffer)
        float *out = malloc((size_t)4 * V * sizeof(float));
        ok &= rq_prefill(c, s, u.v + P, 4, RQ_LOGITS_ALL, out, NULL) == RQ_OK;
        ok &= rq_seq_checkpoint_at(s, P + 2) == RQ_OK && (rq_seq_checkpoint_at(s, P - 1) == RQ_ERR_ARG) == (I.linear_layers > 0);
        rq_seq_checkpoint_rows_drop(s);
        nh = rq_seq_checkpoints(s, held, RQ_CHECKPOINTS_MAX);
        ok &= I.linear_layers ? nh == 1 && held[0] == P + 2 : nh == 0;
        ok &= rq_seq_truncate(s, P + 2) == RQ_OK && rq_seq_len(s) == P + 2;
        ok &= rq_prefill(c, s, u.v + P + 2, 2, RQ_LOGITS_LAST, NULL, &lg) == RQ_OK && same(lg, ref + (size_t)(u.n - 1) * V, V);
        ok &= rq_seq_checkpoint_rows(s, rq_seq_len(s), 2) == RQ_OK; // (a later rows call keeps it)
        rq_seq_checkpoint_rows_drop(s);
        nh = rq_seq_checkpoints(s, held, RQ_CHECKPOINTS_MAX);
        ok &= I.linear_layers ? nh == 1 && held[0] == P + 2 : nh == 0;
        report(ok, "spec_checkpoint_keep", I.linear_layers ? "a verify row's checkpoint kept, the others dropped; truncated there "
               "== a fresh prefill" : "no-ops (Qwen3)");
        free(out);
        free(ref);
        rq_seq_free(s);
        rq_tokens_free(&u);
    }
    // the cost model (G7): a calibration on an empty sequence (emptied again), its shape; the choice by the costs
    {
        RqPassCost t, dd;
        RqSeq *s = seq_new(0);
        double t0 = rq_now();
        int ok = rq_pass_cost_measure(c, s, 600, 2, &t) == RQ_OK && rq_seq_len(s) == 0 && t.calibrated && t.t_w > 0 && t.att >= 0 &&
                 t.att_x >= 0 && t.att_x <= t.att && t.n >= 2 && t.r[0] == 1 && t.f[0] == 1;
        for (int i = 1; i < t.n; i++) ok &= t.f[i] >= t.f[i - 1] && t.r[i] > t.r[i - 1];
        ok &= rq_pass_cost(&t, 0, 0, 0) == 0 && rq_pass_cost(&t, 1, 0, 0) == t.t_w && rq_pass_cost(&t, 64, 0, 0) > rq_pass_cost(&t, 16, 0, 0);
        ok &= rq_pass_cost_measure(c, s, 20, 1, &dd) == RQ_ERR_ARG; // (too short)
        snprintf(d, sizeof(d), "decode %.2f ms + %.3f / %.3f us per position (first / further rows), f(4) %.2f, f(16) %.2f (%.1f s)",
                 t.t_w * 1e3, t.att * 1e6, t.att_x * 1e6, rq_pass_cost(&t, 4, 0, 0) / t.t_w, rq_pass_cost(&t, 16, 0, 0) / t.t_w, rq_now() - t0);
        // a free draft that is always right: the model; no draft and a long match: the lookup; a huge attention: plain
        // (on a cost model where extra rows are cheap, as on the AVX-512 build: the generic build's are not)
        for (int i = 1; i < t.n; i++) t.f[i] = 1 + 0.05 * (t.r[i] - 1);
        t.att = t.att_x = 1e-9;
        dd = t;
        dd.t_w = 1e-9;
        dd.att = dd.att_x = 0;
        RqSpecChoice x = rq_spec_choose(&t, &dd, 100, 3, 0.99, 3, 0.5);
        ok &= x.src == RQ_SPEC_MODEL && x.n == 3 && x.gain > 1;
        x = rq_spec_choose(&t, NULL, 100, 3, 0.99, 3, 0.94);
        ok &= x.src == RQ_SPEC_LOOKUP && x.gain > 1;
        RqPassCost h = t;
        h.att = h.att_x = 1e-3;
        h.t_w = 1e-6;
        x = rq_spec_choose(&h, &dd, 100000, 3, 0.6, 3, 0.6);
        ok &= x.src == RQ_SPEC_PLAIN && x.n == 0 && x.gain == 1;
        ok &= rq_spec_expect(0.5, 3) > 0.875 - 1e-12 && rq_spec_expect(0.5, 3) < 0.875 + 1e-12 && rq_spec_expect(0.7, 0) == 0 && rq_lookup_prior(16) > rq_lookup_prior(2);
        report(ok, "spec_cost_model", d);
        rq_seq_free(s);
    }
    for (int i = 0; i < 4; i++) rq_tokens_free(&sp_prompt[i]);
    rq_tokens_free(&sp_turn2);
    rq_ctx_free(dc);
    rq_ctx_free(c);
    rq_model_free(SD);
    rq_model_free(M);
    return failed;
}

// ---------------------------------------------------------------------------------------- api_test --mtp
// (fw_rows, D: above, api_test --spec)

static RqSeq *mt_seq(RqModel *m) {
    int e;
    RqSeq *s = rq_seq_new(m, 0, RQ_KV_DEFAULT, &e);
    if (!s) die("rq_seq_new", e);
    return s;
}

static RqCtx *mt_ctx(RqModel *m, int nb) {
    int e;
    RqCtx *c = rq_ctx_new(m, nb, 0, &e);
    if (!c) die("rq_ctx_new", e);
    return c;
}

static int mt_ref(RqCtx *c1, RqSeq *s, const int *t, int n, const float *h_in, float *out, float *hid, uint64_t *lh) {
    // the reference: t on s (emptied) through the decode path (c1: max_batch 1), every row's logits (out, n x V; lh: their
    // hashes) and hidden state
    rq_seq_reset(s);
    int e = fw_rows(c1, s, t, n, RQ_LOGITS_ALL, out, NULL, hid, h_in);
    for (int i = 0; i < n && e == RQ_OK; i++)
        lh[i] = hash_f(out + (size_t)i * V, V);
    return e;
}

static int mt_segs(RqCtx *c, RqSeq *s, const int *t, int n, int k, const int *modes, int nmodes, const float *h_in, float *hid,
                   uint64_t *lh, float *out) {
    // t on s (emptied) in rq_forward_batch passes of k rows, pass j with the logits mode modes[j % nmodes], every row's
    // hidden state into hid (NULL: none), h_in rows (NULL: zeros); lh[r]: row r's logits hash, 0 without logits
    int e = RQ_OK;
    rq_seq_reset(s);
    for (int i = 0, j = 0; i < n && e == RQ_OK; i += k, j++) {
        int r = n - i < k ? n - i : k, mode = modes[j % nmodes];
        RqSegment sg = { s, t + i, r, mode, hid ? hid + (size_t)i * D : NULL, h_in ? h_in + (size_t)i * D : NULL };
        float *lg = NULL;
        e = rq_forward_batch(c, &sg, 1, mode == RQ_LOGITS_ALL ? out : NULL, &lg);
        for (int q = 0; q < r; q++)
            lh[i + q] = e == RQ_OK && mode == RQ_LOGITS_ALL ? hash_f(out + (size_t)q * V, V) : 0;
        if (e == RQ_OK && mode == RQ_LOGITS_LAST) lh[i + r - 1] = hash_f(lg, V);
    }
    return e;
}

static int mt_eq(const uint64_t *ref, const uint64_t *lh, const float *hr, const float *h, int n) {
    // the rows' logits (where computed) and hidden states (h NULL: not asked for) equal the reference's
    for (int i = 0; i < n; i++)
        if (lh[i] && lh[i] != ref[i]) return 0;
    return !h || same(hr, h, (size_t)n * D);
}

static int mt_mixed(RqCtx *c, RqSeq *a, RqSeq *b, const int *t, int pa, int ka, int pb, const float *h_in, float *hid,
                    const uint64_t *lhR, const float *hR, float *out) {
    // one pass with two sequences of t: a (its first pa rows fed before) ka rows with every row's hidden state and the
    // last one's logits, b (pb rows before) one row with its logits, no hidden state; each == the reference's rows
    rq_seq_reset(a);
    rq_seq_reset(b);
    int e = fw_rows(c, a, t, pa, RQ_LOGITS_NONE, NULL, NULL, NULL, h_in);
    if (e == RQ_OK) e = fw_rows(c, b, t, pb, RQ_LOGITS_NONE, NULL, NULL, NULL, h_in);
    RqSegment sg[2] = { { a, t + pa, ka, RQ_LOGITS_LAST, hid, h_in ? h_in + (size_t)pa * D : NULL },
                        { b, t + pb, 1, RQ_LOGITS_LAST, NULL, h_in ? h_in + (size_t)pb * D : NULL } };
    if (e == RQ_OK) e = rq_forward_batch(c, sg, 2, out, NULL);
    return e == RQ_OK && hash_f(out, V) == lhR[pa + ka - 1] && hash_f(out + V, V) == lhR[pb] && same(hid, hR + (size_t)pa * D, (size_t)ka * D);
}

static int mt_paths(RqModel *m, RqCtx *c1, RqCtx *c64, const int *t, int n, const float *h_in, int quick, const uint64_t *lhR,
                    const float *hR, char *d, size_t dn) {
    // the passes of model m (the target, or the view with h_in) == the reference (decode path: lhR, hR)
    RqSeq *s = mt_seq(m), *s2 = mt_seq(m);
    float *hid = malloc((size_t)n * D * sizeof(float)), *out = malloc((size_t)64 * V * sizeof(float));
    uint64_t *lh = malloc((size_t)n * sizeof(uint64_t));
    const int all[] = { RQ_LOGITS_ALL }, last[] = { RQ_LOGITS_LAST }, mix3[] = { RQ_LOGITS_ALL, RQ_LOGITS_LAST, RQ_LOGITS_NONE },
              mix3b[] = { RQ_LOGITS_LAST, RQ_LOGITS_NONE, RQ_LOGITS_ALL };
    struct { const char *name; int k; const int *modes; int nm, hid, q; } cs[] = {
        { "chunks of 64 (all / last / no logits)", 64, mix3, 3, 1, 1 }, { "chunks of 64 without hidden", 64, all, 1, 0, 0 },
        { "chunks of 33 (last / none / all)", 33, mix3b, 3, 1, 0 }, { "segments of 7 (batched decode rows)", 7, mix3b, 3, 1, 1 },
        { "segments of 16", 16, mix3, 3, 1, 0 }, { "rows one at a time", 1, last, 1, 1, 0 } };
    int ok = 1;
    size_t dl = strlen(d);
    for (size_t i = 0; i < sizeof(cs) / sizeof(cs[0]); i++) {
        if (quick && !cs[i].q) continue;
        memset(hid, 0xff, (size_t)n * D * sizeof(float)); // (NaN: every row must be written)
        int e = mt_segs(c64, s, t, n, cs[i].k, cs[i].modes, cs[i].nm, h_in, cs[i].hid ? hid : NULL, lh, out);
        int okk = e == RQ_OK && mt_eq(lhR, lh, hR, cs[i].hid ? hid : NULL, n);
        snprintf(d + dl, dn - dl, "%s%s%s", dl ? ", " : "", cs[i].name, okk ? "" : " DIFFER");
        dl = strlen(d);
        ok &= okk;
    }
    { // the decode path without hidden rows (the logits)
        float *o = malloc((size_t)n * V * sizeof(float));
        rq_seq_reset(s);
        int okk = fw_rows(c1, s, t, n, RQ_LOGITS_ALL, o, NULL, NULL, h_in) == RQ_OK;
        for (int i = 0; i < n && okk; i++) okk &= hash_f(o + (size_t)i * V, V) == lhR[i];
        free(o);
        snprintf(d + dl, dn - dl, ", decode without hidden%s", okk ? "" : " DIFFER");
        dl = strlen(d);
        ok &= okk;
    }
    { // mixed passes: prefill_worker (a 30-row segment next to a decode row), batch_decode_worker (5 rows + 1)
        int okk = mt_mixed(c64, s, s2, t, 40, 30, 60, h_in, hid, lhR, hR, out);
        okk &= mt_mixed(c64, s, s2, t, 21, 5, 70, h_in, hid, lhR, hR, out);
        snprintf(d + dl, dn - dl, ", mixed passes%s", okk ? "" : " DIFFER");
        ok &= okk;
    }
    free(hid);
    free(out);
    free(lh);
    rq_seq_free(s);
    rq_seq_free(s2);
    return ok;
}

static int logit_off(float x) { // (RQ_LOGIT_OFF, by its bits)
    const float off = RQ_LOGIT_OFF;
    return !memcmp(&x, &off, 4);
}

static int mt_subset(RqModel *view, RqCtx *c64, const int *t, int n, const float *h_in, const float *full, int quick, char *d, size_t dn) {
    // a vocabulary subset (RqMtpOpts.vocab 1000): every path's logits == the full vocabulary's (full) on rows [0, 1008)
    // and from the added tokens' block on, RQ_LOGIT_OFF between; the hidden states equal (they do not depend on the subset)
    (void)c64;
    char err[512];
    RqMtpOpts o = { 1000, -1 };
    RqModel *sv = rq_mtp_load(M, &o, err, sizeof(err));
    if (!sv) { snprintf(d, dn, "rq_mtp_load(vocab 1000): %s", err); return 0; }
    RqCtx *s1 = mt_ctx(sv, 1), *s64 = mt_ctx(sv, 64);
    RqSeq *s = mt_seq(sv);
    float *out = malloc((size_t)n * V * sizeof(float)), *ho = malloc((size_t)n * D * sizeof(float)), *hf = malloc((size_t)n * D * sizeof(float));
    int fa = V; // the first added token at or past 1008
    for (int i = V - 1; i >= 1008; i--)
        if (rq_token_is_added(M, i)) fa = i;
    int ok = 1, lo1 = -1;
    struct { RqCtx *c; int k; } runs[] = { { s1, 0 }, { s64, 64 }, { s64, 7 } };
    for (int r = 0; r < (quick ? 2 : 3); r++) {
        rq_seq_reset(s);
        int e = RQ_OK;
        if (!runs[r].k) e = fw_rows(runs[r].c, s, t, n, RQ_LOGITS_ALL, out, NULL, ho, h_in);
        for (int i = 0; runs[r].k && i < n && e == RQ_OK; i += runs[r].k) {
            int k = n - i < runs[r].k ? n - i : runs[r].k;
            RqSegment sg = { s, t + i, k, RQ_LOGITS_ALL, ho + (size_t)i * D, h_in + (size_t)i * D };
            e = rq_forward_batch(runs[r].c, &sg, 1, out + (size_t)i * V, NULL);
        }
        ok &= e == RQ_OK;
        if (r == 0) { // the full view's hidden states, for comparison
            RqSeq *f = mt_seq(view);
            RqCtx *fc = mt_ctx(view, 1);
            ok &= fw_rows(fc, f, t, n, RQ_LOGITS_NONE, NULL, NULL, hf, h_in) == RQ_OK;
            rq_ctx_free(fc);
            rq_seq_free(f);
        }
        ok &= same(ho, hf, (size_t)n * D);
        for (int q = 0; q < n && ok; q++) {
            const float *a = out + (size_t)q * V, *b = full + (size_t)q * V;
            int i = 0;
            while (i < 1008 && a[i] == b[i]) i++;
            ok &= i == 1008;
            while (i < V && logit_off(a[i])) i++;
            if (lo1 < 0) lo1 = i;
            ok &= i == lo1 && lo1 % 16 == 0 && lo1 <= fa && lo1 > fa - 16 && !memcmp(a + i, b + i, (size_t)(V - i) * sizeof(float));
        }
    }
    snprintf(d, dn, "vocab 1000: rows [0, 1008) and [%d, %d) == the full vocabulary's, RQ_LOGIT_OFF (-1e30) between (first added token %d), "
             "decode / chunks of 64%s%s", lo1, V, fa, quick ? "" : " / segments of 7", ok ? "" : " DIFFER");
    free(out);
    free(ho);
    free(hf);
    rq_seq_free(s);
    rq_ctx_free(s1);
    rq_ctx_free(s64);
    rq_model_free(sv);
    return ok;
}

static int mt_dec(RqModel *m, RqCtx *c1, RqCtx *c64, const int *t, int n, const float *h_in, int P, const uint64_t *lhR,
                  const float *hR, char *d, size_t dn) {
    // rq_decode_hidden (alternately on c1 and c64) == a one-row rq_forward_batch with hidden (c64) == the reference (the
    // decode path: lhR, hR), rows P..n-1 one at a time on sequences fed t[0..P) before: logits and hidden bitwise; with
    // logits NULL the same hidden state, and its sequence goes on alike (its next rows compared too)
    RqSeq *a = mt_seq(m), *b = mt_seq(m), *z = mt_seq(m);
    float *la = malloc((size_t)V * sizeof(float)), *ha = malloc((size_t)D * sizeof(float)), *hb = malloc((size_t)D * sizeof(float)),
          *hz = malloc((size_t)D * sizeof(float));
    int e = fw_rows(c64, a, t, P, RQ_LOGITS_NONE, NULL, NULL, NULL, h_in), ok = 1, q = P;
    if (e == RQ_OK) e = fw_rows(c64, b, t, P, RQ_LOGITS_NONE, NULL, NULL, NULL, h_in);
    if (e == RQ_OK) e = fw_rows(c64, z, t, P, RQ_LOGITS_NONE, NULL, NULL, NULL, h_in);
    for (; q < n && ok && e == RQ_OK; q++) {
        const float *hi = h_in ? h_in + (size_t)q * D : NULL;
        float *lg = NULL, *lb = NULL;
        RqSegment sg = { b, t + q, 1, RQ_LOGITS_LAST, hb, hi };
        if ((e = rq_decode_hidden(q % 2 ? c1 : c64, a, t[q], &lg, ha, hi)) != RQ_OK) break;
        memcpy(la, lg, (size_t)V * sizeof(float));
        if ((e = rq_forward_batch(c64, &sg, 1, NULL, &lb)) != RQ_OK) break;
        ok &= same(la, lb, V) && same(ha, hb, D) && hash_f(la, V) == lhR[q] && same(ha, hR + (size_t)q * D, D);
        if ((e = rq_decode_hidden(c64, z, t[q], NULL, hz, hi)) != RQ_OK) break;
        ok &= same(hz, ha, D) && rq_seq_len(z) == q + 1;
    }
    ok &= e == RQ_OK;
    // refused: a token out of range; h_in on a model that is not a view; a full sequence (nothing fed)
    float *lg = (float *)la;
    int L0 = rq_seq_len(a);
    ok &= rq_decode_hidden(c64, a, V, &lg, ha, NULL) == RQ_ERR_ARG && !lg && rq_seq_len(a) == L0;
    if (!h_in) ok &= rq_decode_hidden(c64, a, t[0], NULL, ha, hR) == RQ_ERR_ARG && rq_seq_len(a) == L0;
    RqSeq *f = rq_seq_new(m, 4, RQ_KV_DEFAULT, &e);
    ok &= f && fw_rows(c64, f, t, 4, RQ_LOGITS_NONE, NULL, NULL, NULL, h_in) == RQ_OK &&
          rq_decode_hidden(c1, f, t[4], NULL, ha, h_in ? h_in + 4 * (size_t)D : NULL) == RQ_ERR_CONTEXT && rq_seq_len(f) == 4;
    snprintf(d, dn, "rows %d..%d one at a time (decode / batch-1 contexts alternately)%s: logits and hidden == a one-row "
             "rq_forward_batch with hidden%s == the decode path's; logits NULL: the same hidden state; refused: a token out of "
             "range%s, a full sequence%s", P, n - 1, h_in ? ", h_in the target's rows shifted by one" : "", h_in ? " and h_in" : "",
             h_in ? "" : ", h_in on the target", ok ? "" : " DIFFER");
    rq_seq_free(f);
    rq_seq_free(a);
    rq_seq_free(b);
    rq_seq_free(z);
    free(la);
    free(ha);
    free(hb);
    free(hz);
    return ok;
}

static int mt_hin(RqModel *view, RqCtx *v64, RqSeq *s, const int *t, int n, const float *hR, const float *Hs, const uint64_t *vlhR,
                  char *d, size_t dn) {
    // rq_mtp_hin's rule (runq.h): row q's h from H (positions h0 ..), else the kept one, else zeros (row 0: exact), the
    // count of the zero rows; hR: the target's hidden rows of t (n x D), Hs: them shifted by one (row 0 zeros)
    RqSeq *vs = mt_seq(view);
    float *hin = malloc((size_t)16 * D * sizeof(float)), *z = calloc((size_t)16 * D, sizeof(float)), *lg = NULL;
    int ok = 1, r, bad = 0;
#define HN_CHECK(c) do { if (!(c) && !bad) bad = __LINE__; } while (0)
    r = rq_mtp_hin(vs, hR, 0, 10, 12, hin); // (rows 0..11 of an empty view: 0 exact, 1..10 from H, 11 unknown)
    HN_CHECK(r == 1 && same(hin, Hs, (size_t)11 * D) && same(hin + (size_t)11 * D, z, D));
    r = rq_mtp_hin(vs, NULL, 0, 5, 4, hin); // (H NULL: none, whatever hn says)
    HN_CHECK(r == 3 && same(hin, z, (size_t)4 * D));
    HN_CHECK(fw_rows(v64, vs, t, 5, RQ_LOGITS_NONE, NULL, NULL, NULL, Hs) == RQ_OK && rq_mtp_keep(vs, 4, hR + (size_t)4 * D) == RQ_OK);
    r = rq_mtp_hin(vs, NULL, 0, 0, 3, hin); // (row 5: the kept state at 4; 6, 7 unknown)
    HN_CHECK(r == 2 && same(hin, Hs + (size_t)5 * D, D) && same(hin + D, z, (size_t)2 * D));
    r = rq_mtp_hin(vs, hR + (size_t)10 * D, 4, 2, 3, hin); // (H before the kept state: rows 5, 6 from H; 7 unknown)
    HN_CHECK(r == 1 && same(hin, hR + (size_t)10 * D, (size_t)2 * D) && same(hin + (size_t)2 * D, z, D));
    HN_CHECK(rq_mtp_keep(vs, 2, hR + (size_t)2 * D) == RQ_OK); // (kept at another position: not row 5's)
    r = rq_mtp_hin(vs, NULL, 0, 0, 1, hin);
    HN_CHECK(r == 1 && same(hin, z, D));
    HN_CHECK(rq_mtp_hin(s, hR, 0, 1, 1, hin) == RQ_ERR_ARG); // (the target's sequence)
    // the rule as the server feeds a view: the rest in rows of up to 16 from the target's rows (H: positions 0..n-1)
    for (int q = 5; q < n && !bad; q += 16) {
        int k = n - q < 16 ? n - q : 16;
        HN_CHECK(rq_mtp_hin(vs, hR, 0, n, k, hin) == 0 && fw_rows(v64, vs, t + q, k, RQ_LOGITS_LAST, NULL, &lg, NULL, hin) == RQ_OK);
    }
    HN_CHECK(lg && hash_f(lg, V) == vlhR[n - 1]);
#undef HN_CHECK
    ok = !bad;
    snprintf(d, dn, "H, the kept state, zeros (row 0 exact, the others counted), H NULL, H before the kept one, a state kept at "
             "another position, the target's sequence refused; the rest of the prompt fed with its inputs == the reference%s",
             ok ? "" : " FAILED");
    if (!ok) snprintf(d + strlen(d), dn - strlen(d), " (api_test.c:%d)", bad);
    free(hin);
    free(z);
    rq_seq_free(vs);
    return ok;
}

static int mt_keep(RqModel *view, RqCtx *v64, RqSeq *s, const int *t, int n, const float *hR, const float *Hs, char *d, size_t dn) {
    // rq_mtp_final from hvalid == len0 only; rq_mtp_keep / rq_mtp_kept, also past the view's end at its capacity, cleared
    // with h NULL; the kept state dropped by a truncation below it, kept by a copy that includes it, cleared by a reset
    RqSeq *vs = mt_seq(view), *cp = mt_seq(view);
    float *h = malloc((size_t)D * sizeof(float));
    int bad = 0, e;
#define KP_CHECK(c) do { if (!(c) && !bad) bad = __LINE__; } while (0)
    KP_CHECK(fw_rows(v64, vs, t, 5, RQ_LOGITS_NONE, NULL, NULL, NULL, Hs) == RQ_OK && rq_seq_common_prefix(vs, t, n) == 0);
    rq_mtp_final(vs, 0);
    KP_CHECK(rq_seq_common_prefix(vs, t, n) == 5);
    KP_CHECK(fw_rows(v64, vs, t + 5, 3, RQ_LOGITS_NONE, NULL, NULL, NULL, Hs + (size_t)5 * D) == RQ_OK && rq_seq_common_prefix(vs, t, n) == 5);
    rq_mtp_final(vs, 6); // (hvalid 5: not from there)
    KP_CHECK(rq_seq_common_prefix(vs, t, n) == 5);
    rq_mtp_final(vs, 5);
    KP_CHECK(rq_seq_common_prefix(vs, t, n) == 8);
    KP_CHECK(rq_mtp_kept(vs, h) == -1 && rq_mtp_keep(vs, 7, hR + (size_t)7 * D) == RQ_OK && rq_mtp_kept(vs, NULL) == 7 &&
             rq_mtp_kept(vs, h) == 7 && same(h, hR + (size_t)7 * D, D));
    KP_CHECK(rq_seq_copy(cp, vs, 8) == RQ_OK && rq_mtp_kept(cp, h) == 7 && same(h, hR + (size_t)7 * D, D)); // (the copy includes it)
    KP_CHECK(rq_seq_copy(cp, vs, 7) == RQ_OK && rq_mtp_kept(cp, NULL) == -1);
    KP_CHECK(rq_seq_truncate(vs, 8) == RQ_OK && rq_mtp_kept(vs, NULL) == 7); // (its length: kept)
    KP_CHECK(rq_seq_truncate(vs, 7) == RQ_OK && rq_mtp_kept(vs, NULL) == -1); // (below it: dropped)
    KP_CHECK(rq_mtp_keep(vs, 6, hR + (size_t)6 * D) == RQ_OK && rq_mtp_kept(vs, NULL) == 6);
    rq_seq_reset(vs);
    KP_CHECK(rq_mtp_kept(vs, NULL) == -1);
    // at the capacity: the target's state past the view's end (the CLI's end-of-call state there)
    RqSeq *c8 = rq_seq_new(view, 8, RQ_KV_DEFAULT, &e);
    KP_CHECK(c8 && fw_rows(v64, c8, t, 8, RQ_LOGITS_NONE, NULL, NULL, NULL, Hs) == RQ_OK);
    KP_CHECK(c8 && rq_mtp_keep(c8, 8, hR + (size_t)8 * D) == RQ_OK && rq_mtp_kept(c8, h) == 8 && same(h, hR + (size_t)8 * D, D));
    KP_CHECK(c8 && rq_mtp_keep(c8, 30, hR) == RQ_OK && rq_mtp_kept(c8, NULL) == 30);
    KP_CHECK(c8 && rq_mtp_keep(c8, 5, NULL) == RQ_OK && rq_mtp_kept(c8, NULL) == -1); // (h NULL: cleared, pos ignored)
    KP_CHECK(c8 && rq_mtp_keep(c8, -1, hR) == RQ_ERR_ARG && rq_mtp_keep(c8, -1, NULL) == RQ_OK);
    KP_CHECK(rq_mtp_keep(s, 0, hR) == RQ_ERR_ARG && rq_mtp_keep(s, 0, NULL) == RQ_ERR_ARG && rq_mtp_kept(s, h) == -1); // (the target's)
#undef KP_CHECK
    snprintf(d, dn, "rq_mtp_final from hvalid == len0 only; keep / kept round trip, a copy including it keeps it (one to below "
             "it does not), a truncation to below it drops it (to its length: kept), a reset clears it; at the capacity "
             "positions 8 and 30 past the view's end kept, h NULL clears, pos < 0 with h refused; the target's sequence refused%s",
             bad ? " FAILED" : "");
    if (bad) snprintf(d + strlen(d), dn - strlen(d), " (api_test.c:%d)", bad);
    rq_seq_free(c8);
    rq_seq_free(vs);
    rq_seq_free(cp);
    free(h);
    return !bad;
}

static int mt_seqcheck(RqModel *m, RqModel *view, const char *tp, const char *vp, char *d, size_t dn) {
    // a saved target / MTP view pair (runq serve's RUNQ_TEST_MTP_DUMP) against a fresh lockstep run over the target
    // file's tokens: the target's hidden rows of a fresh prefill (its KV type), the view's prefill over them shifted by one
    // (row 0 zeros; the file's KV type). The view's length == the target's, its tokens the target's, its KV cache == the
    // reference's bitwise over the whole 16-position tiles (copies with the kept state cleared, rq_seq_save, byte for byte);
    // a view file with the kept state (TODO S9c.2: version 2): the state == the target's at L - 1 bitwise, and a probe row
    // at L with it == the reference's with h_{L-1}
    RqSeqFile ti, vi;
    int *tok = NULL, e, ok = 1, kept = -1;
    char why[256] = "";
#define SQ_FAIL(...) do { if (ok) snprintf(why, sizeof(why), __VA_ARGS__); ok = 0; } while (0)
    if ((e = rq_seq_file_info(m, tp, &ti, &tok)) != RQ_OK) {
        snprintf(d, dn, "%s: %s", tp, rq_strerror(e));
        return 0;
    }
    if ((e = rq_seq_file_info(view, vp, &vi, NULL)) != RQ_OK) {
        snprintf(d, dn, "%s: %s", vp, rq_strerror(e));
        free(tok);
        return 0;
    }
    int L = ti.len, Lv = vi.len, cap = L + 16;
    RqCtx *c = mt_ctx(m, 64), *vc = mt_ctx(view, 64);
    RqSeq *rs = rq_seq_new(m, cap, ti.kv_type, &e), *rv = rq_seq_new(view, cap, vi.kv_type, &e), *lv = rq_seq_new(view, cap, vi.kv_type, &e);
    RqSeq *a = rq_seq_new(view, cap, vi.kv_type, &e), *b = rq_seq_new(view, cap, vi.kv_type, &e);
    float *H = malloc((size_t)L * D * sizeof(float)), *hin = calloc((size_t)L * D, sizeof(float));
    if (!rs || !rv || !lv || !a || !b || !H || !hin) die("mt_seqcheck", RQ_ERR_NOMEM);
    if (Lv != L) SQ_FAIL("the view's length %d, the target's %d", Lv, L);
    if (ok && (e = fw_rows(c, rs, tok, L, RQ_LOGITS_NONE, NULL, NULL, H, NULL)) != RQ_OK) SQ_FAIL("the target's prefill: %s", rq_strerror(e));
    if (ok) memcpy(hin + D, H, (size_t)(L - 1) * D * sizeof(float));
    if (ok && (e = fw_rows(vc, rv, tok, Lv, RQ_LOGITS_NONE, NULL, NULL, NULL, hin)) != RQ_OK) SQ_FAIL("the view's prefill: %s", rq_strerror(e));
    if (ok && ((e = rq_seq_load(lv, vp, Lv)) != RQ_OK || rq_seq_len(lv) != Lv)) SQ_FAIL("the view's load: %s", rq_strerror(e));
    if (ok && memcmp(rq_seq_tokens(lv), tok, (size_t)Lv * sizeof(int))) SQ_FAIL("the view's tokens are not the target's");
    int L16 = Lv / 16 * 16;
    if (ok && L16 > 0) {
        char pa[512], pb[512];
        const char *tmp = getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp";
        snprintf(pa, sizeof(pa), "%s/api_test_seqcheck_%d_a.rqseq", tmp, (int)getpid());
        snprintf(pb, sizeof(pb), "%s/api_test_seqcheck_%d_b.rqseq", tmp, (int)getpid());
        int okk = rq_seq_copy(a, lv, L16) == RQ_OK && rq_seq_copy(b, rv, L16) == RQ_OK && rq_mtp_keep(a, 0, NULL) == RQ_OK &&
                  rq_mtp_keep(b, 0, NULL) == RQ_OK && rq_seq_save(a, pa) == RQ_OK && rq_seq_save(b, pb) == RQ_OK && file_same(pa, pb);
        if (!okk) SQ_FAIL("the KV cache of rows 0..%d differs from the reference's", L16 - 1);
        remove(pa);
        remove(pb);
    }
    float *hk = malloc((size_t)D * sizeof(float));
    if (!hk) die("mt_seqcheck", RQ_ERR_NOMEM);
    if (ok && (kept = rq_mtp_kept(lv, hk)) >= 0) {
        float *lg;
        int pt = 1000 % V;
        unsigned long long h1 = 0, h2 = 1;
        if (kept != L - 1) SQ_FAIL("the kept state at %d, the target's last position %d", kept, L - 1);
        else if (!same(hk, H + (size_t)(L - 1) * D, D)) SQ_FAIL("the kept state differs from the target's at %d", L - 1);
        if (ok && fw_rows(vc, lv, &pt, 1, RQ_LOGITS_LAST, NULL, &lg, NULL, hk) == RQ_OK) h1 = rq_logits_hash(lg, V);
        if (ok && fw_rows(vc, rv, &pt, 1, RQ_LOGITS_LAST, NULL, &lg, NULL, H + (size_t)(L - 1) * D) == RQ_OK) h2 = rq_logits_hash(lg, V);
        if (ok && h1 != h2) SQ_FAIL("the probe row at %d differs", L);
    }
    snprintf(d, dn, "%d rows (KV %s), the KV cache over %d rows, %s%s%s", Lv, rq_kv_name(vi.kv_type), L16,
             kept >= 0 ? "the kept state (version 2) and a probe row with it" : "no kept state", ok ? "" : ": ", why);
#undef SQ_FAIL
    free(hk);
    free(tok);
    free(H);
    free(hin);
    rq_seq_free(rs);
    rq_seq_free(rv);
    rq_seq_free(lv);
    rq_seq_free(a);
    rq_seq_free(b);
    rq_ctx_free(c);
    rq_ctx_free(vc);
    return ok;
}

static long long file_size(const char *p) {
    FILE *f = fopen(p, "rb");
    long long n = f && !fseek(f, 0, SEEK_END) ? (long long)ftell(f) : -1;
    if (f) fclose(f);
    return n;
}

static int file_flip(const char *src, const char *dst, long long n, long long off) {
    // dst := the first n bytes of src with the byte at off flipped (off < 0: none)
    FILE *fi = fopen(src, "rb"), *fo = fopen(dst, "wb");
    int ok = fi && fo, ch;
    for (long long i = 0; ok && i < n && (ch = fgetc(fi)) != EOF; i++) ok = fputc(i == off ? ch ^ 0x10 : ch, fo) != EOF;
    if (fi) fclose(fi);
    if (fo && fclose(fo)) ok = 0;
    return ok;
}

static int mt_seqfile(RqModel *view, RqCtx *v64, const int *t, int n, const float *hR, const float *Hs, int corrupt, char *d, size_t dn) {
    // the view's n rows (all final) with the target's state at n - 1 kept, saved (version 2: the file's version field, its
    // size == rq_seq_save_bytes == version 1's + the state's page); rq_seq_file_info reads it; loaded whole: the kept
    // state bitwise, a row after it == the saved sequence's; loaded to n - 5: none kept; saved without a kept state or
    // with one before the last row: version 1, a page smaller, nothing kept by a load. corrupt: a flipped byte in the
    // state, in SeqFileExt, the file cut before the state's page: RQ_ERR_MISMATCH (rq_seq_file_info and rq_seq_load),
    // the destination unchanged
    char p[512], q[512];
    const char *tmp = getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp";
    snprintf(p, sizeof(p), "%s/api_test_sf_%d.draft.rqseq", tmp, (int)getpid());
    snprintf(q, sizeof(q), "%s/api_test_sf_%d_x.draft.rqseq", tmp, (int)getpid());
    RqSeq *w = mt_seq(view), *x = mt_seq(view);
    float *h = malloc((size_t)D * sizeof(float)), *lg;
    int bad = 0, *tk = NULL;
    long long hb = ((long long)D * 4 + 4095) / 4096 * 4096, n2 = -1, n1 = -1;
    unsigned long long h1 = 0, h2 = 1;
    uint32_t ver = 0;
    RqSeqFile f;
#define SF_CHECK(c) do { if (!(c) && !bad) bad = __LINE__; } while (0)
    SF_CHECK(fw_rows(v64, w, t, n, RQ_LOGITS_NONE, NULL, NULL, NULL, Hs) == RQ_OK);
    rq_mtp_final(w, 0);
    SF_CHECK(rq_mtp_keep(w, n - 1, hR + (size_t)(n - 1) * D) == RQ_OK && rq_seq_save(w, p) == RQ_OK);
    FILE *fv = fopen(p, "rb");
    SF_CHECK(fv && !fseek(fv, 8, SEEK_SET) && fread(&ver, 4, 1, fv) == 1 && ver == 2);
    if (fv) fclose(fv);
    SF_CHECK((n2 = file_size(p)) == rq_seq_save_bytes(w));
    if (!corrupt) {
        int pt = 1000 % V;
        SF_CHECK(rq_seq_file_info(view, p, &f, &tk) == RQ_OK && f.len == n && f.bytes == n2 && tk && !memcmp(tk, t, (size_t)n * sizeof(int)));
        SF_CHECK(rq_seq_load(x, p, n) == RQ_OK && rq_seq_len(x) == n && rq_seq_common_prefix(x, t, n) == n && rq_mtp_kept(x, h) == n - 1 &&
                 same(h, hR + (size_t)(n - 1) * D, D));
        SF_CHECK(fw_rows(v64, x, &pt, 1, RQ_LOGITS_LAST, NULL, &lg, NULL, h) == RQ_OK && (h1 = rq_logits_hash(lg, V), 1) &&
                 fw_rows(v64, w, &pt, 1, RQ_LOGITS_LAST, NULL, &lg, NULL, hR + (size_t)(n - 1) * D) == RQ_OK && (h2 = rq_logits_hash(lg, V)) == h1);
        SF_CHECK(rq_seq_truncate(w, n) == RQ_OK && rq_mtp_keep(w, n - 1, hR + (size_t)(n - 1) * D) == RQ_OK); // (as saved)
        SF_CHECK(rq_seq_load(x, p, n - 5) == RQ_OK && rq_seq_len(x) == n - 5 && rq_mtp_kept(x, NULL) == -1);
        // version 1: no kept state, one before the last row
        SF_CHECK(rq_mtp_keep(w, 0, NULL) == RQ_OK && rq_seq_save(w, q) == RQ_OK && (n1 = file_size(q)) == n2 - hb && n1 == rq_seq_save_bytes(w));
        SF_CHECK(rq_seq_load(x, q, n) == RQ_OK && rq_seq_len(x) == n && rq_mtp_kept(x, NULL) == -1);
        SF_CHECK(rq_mtp_keep(w, n - 2, hR + (size_t)(n - 2) * D) == RQ_OK && rq_seq_save(w, q) == RQ_OK && file_size(q) == n1);
        SF_CHECK(rq_seq_load(x, q, n) == RQ_OK && rq_mtp_kept(x, NULL) == -1);
        snprintf(d, dn, "%d rows, the target's state at %d kept: version 2, %lld bytes (version 1's + %lld); its info; loaded whole: "
                 "the state bitwise, a row after it == the saved sequence's; loaded to %d: none kept; without a kept state, or "
                 "one at %d: version 1, nothing kept by a load%s", n, n - 1, n2, hb, n - 5, n - 2, bad ? " FAILED" : "");
    } else {
        int e1, e2, e3, e4, e5;
        SF_CHECK(rq_seq_load(x, p, n) == RQ_OK && rq_seq_truncate(x, 7) == RQ_OK); // (the destination: 7 rows)
        SF_CHECK(file_flip(p, q, n2, n2 - hb + 4 * (D / 2)));
        e1 = rq_seq_file_info(view, q, &f, NULL);
        e2 = rq_seq_load(x, q, n);
        SF_CHECK(file_flip(p, q, n2, 264 + 1)); // (SeqFileExt.hpos)
        e3 = rq_seq_load(x, q, n);
        SF_CHECK(file_flip(p, q, n2 - hb, -1)); // (cut before the state)
        e4 = rq_seq_file_info(view, q, &f, NULL);
        e5 = rq_seq_load(x, q, n);
        SF_CHECK(e1 == RQ_ERR_MISMATCH && e2 == RQ_ERR_MISMATCH && e3 == RQ_ERR_MISMATCH && e4 == RQ_ERR_MISMATCH && e5 == RQ_ERR_MISMATCH);
        SF_CHECK(rq_seq_len(x) == 7);
        snprintf(d, dn, "version 2, %lld bytes: a byte of the state flipped: info %s, load %s; SeqFileExt's: load %s; the file cut "
                 "before the state: info %s, load %s; the destination unchanged%s", n2, rq_strerror(e1), rq_strerror(e2),
                 rq_strerror(e3), rq_strerror(e4), rq_strerror(e5), bad ? " FAILED" : "");
    }
#undef SF_CHECK
    if (bad) snprintf(d + strlen(d), dn - strlen(d), " (api_test.c:%d)", bad);
    remove(p);
    remove(q);
    free(tk);
    free(h);
    rq_seq_free(w);
    rq_seq_free(x);
    return !bad;
}

static int mtp_seqcheck_main(int argc, char **argv) {
    // api_test --mtp-seqcheck MODEL TARGET.rqseq VIEW.rqseq (the head of this file)
    if (argc != 5) {
        fprintf(stderr, "usage: api_test --mtp-seqcheck MODEL TARGET.rqseq VIEW.rqseq\n");
        return 2;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    rq_threads_init(0);
    char err[512], d[512];
    if (!(M = rq_model_load(argv[2], NULL, err, sizeof(err)))) { printf("FAIL mtp_seqcheck  %s\n", err); return 1; }
    rq_model_info(M, &I);
    V = I.vocab_size;
    D = I.dim;
    RqModel *view = rq_mtp_load(M, NULL, err, sizeof(err));
    if (!view) { printf("FAIL mtp_seqcheck  rq_mtp_load: %s\n", err); return 1; }
    report(mt_seqcheck(M, view, argv[3], argv[4], d, sizeof(d)), "mtp_seqcheck", d);
    rq_model_free(view);
    rq_model_free(M);
    return failed;
}

static int mtp_main(int argc, char **argv) {
    const char *path = NULL, *plain = NULL, *optional = NULL;
    int quick = 0, bad = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--quick")) quick = 1;
        else if (!strcmp(argv[i], "--plain") && i + 1 < argc) plain = argv[++i];
        else if (!strcmp(argv[i], "--optional") && i + 1 < argc) optional = argv[++i];
        else if (argv[i][0] == '-' || path) bad = 1;
        else path = argv[i];
    }
    if (bad || !path) {
        fprintf(stderr, "usage: api_test --mtp TARGET [--quick] [--plain MODEL_WITHOUT_BLOCK] [--optional MODEL_WITH_OPTIONAL_NEXTN]\n");
        return 2;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    rq_threads_init(0);
    char err[512], d[1024];
    if (!(M = rq_model_load(path, NULL, err, sizeof(err)))) { printf("FAIL setup  %s\n", err); return 1; }
    rq_model_info(M, &I);
    V = I.vocab_size;
    D = I.dim;
    RqModelMem mem0, mem1;
    rq_model_mem(M, &mem0);
    RqMtpOpts mo = { 0, -1 };
    RqModel *view = rq_mtp_load(M, &mo, err, sizeof(err));
    if (!view) { printf("FAIL setup  rq_mtp_load: %s\n", err); return 1; }
    RqCtx *c1 = mt_ctx(M, 1), *c64 = mt_ctx(M, 64), *v1 = mt_ctx(view, 1), *v64 = mt_ctx(view, 64);
    RqTokens T = toks("The old lighthouse stood at the edge of the cliff, its lamp turning slowly through the fog. Every "
                      "night the keeper climbed the spiral stairs, trimmed the wick, polished the brass and wrote the "
                      "weather into his log: wind from the west, rain, a ship far out heading north. In the morning he "
                      "walked down to the village for bread and news, and the children asked him about the storms.");
    int n = T.n;
    float *hR = malloc((size_t)n * D * sizeof(float)), *Hs = calloc((size_t)n * D, sizeof(float));
    float *vhR = malloc((size_t)n * D * sizeof(float)), *full = malloc((size_t)n * V * sizeof(float));
    uint64_t *lhR = malloc((size_t)n * sizeof(uint64_t)), *vlhR = malloc((size_t)n * sizeof(uint64_t));
    RqSeq *s = mt_seq(M), *vs = mt_seq(view);

    // the digest (every mode, build, thread count): the target's hidden rows (chunks of 64, no logits), then the view
    // over them (h shifted by one): its logits and hidden rows
    {
        uint64_t dg = 0;
        float *out = malloc((size_t)n * V * sizeof(float));
        int e = fw_rows(c64, s, T.v, n, RQ_LOGITS_NONE, NULL, NULL, hR, NULL);
        memcpy(Hs + D, hR, (size_t)(n - 1) * D * sizeof(float));
        if (e == RQ_OK) e = fw_rows(v64, vs, T.v, n, RQ_LOGITS_ALL, out, NULL, vhR, Hs);
        if (e != RQ_OK) die("the digest's passes", e);
        dg = tw_mix(hash_f(hR, (size_t)n * D), hash_f(vhR, (size_t)n * D));
        for (int i = 0; i < n; i++)
            dg = tw_mix(dg, hash_f(out + (size_t)i * V, V));
        printf("mtp digest: %016llx\n", (unsigned long long)dg);
        free(out);
    }

    // hidden_rows: the target's
    {
        float *o = malloc((size_t)n * V * sizeof(float)), *h = malloc((size_t)n * D * sizeof(float));
        int e = mt_ref(c1, s, T.v, n, NULL, o, h, lhR);
        int ok = e == RQ_OK && same(h, hR, (size_t)n * D);
        snprintf(d, sizeof(d), "%d rows: decode == chunks of 64 without logits", n);
        free(o);
        free(h);
        char d2[768] = "";
        ok &= mt_paths(M, c1, c64, T.v, n, NULL, quick, lhR, hR, d2, sizeof(d2));
        size_t dl = strlen(d);
        snprintf(d + dl, sizeof(d) - dl, ", %s", d2);
        report(ok, "hidden_rows", d);
    }
    // mtp_paths: the view fed the target's hidden rows shifted by one
    {
        int e = mt_ref(v1, vs, T.v, n, Hs, full, vhR, vlhR);
        char d2[768] = "";
        int ok = e == RQ_OK && mt_paths(view, v1, v64, T.v, n, Hs, quick, vlhR, vhR, d2, sizeof(d2));
        report(ok, "mtp_paths", d2);
        // h_in NULL == zeros (decode, chunks)
        float *z = calloc((size_t)n * D, sizeof(float)), *o = malloc((size_t)n * V * sizeof(float));
        uint64_t *l0 = malloc((size_t)n * sizeof(uint64_t)), *l1 = malloc((size_t)n * sizeof(uint64_t));
        ok = mt_ref(v1, vs, T.v, n, NULL, o, NULL, l0) == RQ_OK && mt_ref(v1, vs, T.v, n, z, o, NULL, l1) == RQ_OK &&
             !memcmp(l0, l1, (size_t)n * sizeof(uint64_t));
        int mode[] = { RQ_LOGITS_ALL };
        ok &= mt_segs(v64, vs, T.v, n, 64, mode, 1, NULL, NULL, l1, o) == RQ_OK && !memcmp(l0, l1, (size_t)n * sizeof(uint64_t));
        ok &= memcmp(l0, vlhR, (size_t)n * sizeof(uint64_t)) != 0; // (and the h input matters)
        report(ok, "mtp_paths/h_in_null", "h_in NULL == explicit zeros (decode, chunks of 64), != the target's hidden rows");
        // truncate + refeed == straight (a plain KV cache: any length)
        int okt = 1;
        for (int P = 1; P < n; P += quick ? 50 : 23) {
            rq_seq_reset(vs);
            okt &= fw_rows(v64, vs, T.v, n, RQ_LOGITS_NONE, NULL, NULL, NULL, Hs) == RQ_OK && rq_seq_keep(vs, P) == P &&
                   rq_seq_truncate(vs, P) == RQ_OK && rq_seq_len(vs) == P;
            float *h = malloc((size_t)(n - P) * D * sizeof(float));
            okt &= fw_rows(P % 2 ? v64 : v1, vs, T.v + P, n - P, RQ_LOGITS_ALL, o, NULL, h, Hs + (size_t)P * D) == RQ_OK;
            for (int i = 0; i < n - P && okt; i++) okt &= hash_f(o + (size_t)i * V, V) == vlhR[P + i];
            okt &= same(h, vhR + (size_t)P * D, (size_t)(n - P) * D);
            free(h);
        }
        report(okt, "mtp_paths/truncate", quick ? "truncated at lengths 1 and 51, fed again (chunks of 64 / decode) == straight"
                                                 : "truncated at every 23rd length and fed again (chunks of 64 / decode) == straight");
        ok = mt_subset(view, v64, T.v, n, Hs, full, quick, d, sizeof(d));
        report(ok, "mtp_paths/vocab_subset", d);
        free(z);
        free(o);
        free(l0);
        free(l1);
    }
    // decode_hidden (TODO S9c.1): rq_decode_hidden == a one-row rq_forward_batch with hidden (and h_in on the view)
    report(mt_dec(M, c1, c64, T.v, n, NULL, quick ? n - 12 : n / 2, lhR, hR, d, sizeof(d)), "decode_hidden/target", d);
    report(mt_dec(view, v1, v64, T.v, n, Hs, quick ? n - 12 : n / 2, vlhR, vhR, d, sizeof(d)), "decode_hidden/view", d);
    // mtp_follow (TODO S9c.1): the view followed by a caller (runq serve): the h rule, the final rows, the kept state;
    // a saved pair against a fresh lockstep run (--mtp-seqcheck's)
    report(mt_hin(view, v64, s, T.v, n, hR, Hs, vlhR, d, sizeof(d)), "mtp_follow/hin_rule", d);
    report(mt_keep(view, v64, s, T.v, n, hR, Hs, d, sizeof(d)), "mtp_follow/final_keep", d);
    {
        char tp[512], vp[512], d2[512];
        const char *tmp = getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp";
        snprintf(tp, sizeof(tp), "%s/api_test_mtp_%d.rqseq", tmp, (int)getpid());
        snprintf(vp, sizeof(vp), "%s/api_test_mtp_%d.draft.rqseq", tmp, (int)getpid());
        RqSeq *x = mt_seq(M), *w = mt_seq(view);
        int ok = rq_prefill_spec(c64, x, v64, w, T.v, n, RQ_LOGITS_NONE, NULL, NULL) == RQ_OK && rq_seq_save(x, tp) == RQ_OK &&
                 rq_seq_save(w, vp) == RQ_OK && mt_seqcheck(M, view, tp, vp, d, sizeof(d));
        // the view's rows fed with zeros for h (all final): its cache differs
        rq_seq_reset(w);
        ok &= fw_rows(v64, w, T.v, n, RQ_LOGITS_NONE, NULL, NULL, NULL, NULL) == RQ_OK && (rq_mtp_final(w, 0), rq_seq_save(w, vp) == RQ_OK) &&
              !mt_seqcheck(M, view, tp, vp, d2, sizeof(d2)) && strstr(d2, "differs");
        ok &= strstr(d, "(version 2)") != NULL; // (rq_prefill_spec's view keeps the target's last state: saved with it)
        size_t dl = strlen(d);
        snprintf(d + dl, sizeof(d) - dl, " (rq_prefill_spec's pair, saved); the view fed zeros for h: %s", d2);
        report(ok, "mtp_follow/seqcheck", d);
        remove(tp);
        remove(vp);
        rq_seq_free(x);
        rq_seq_free(w);
    }
    // seqfile (TODO S9c.2): the view's kept state on disk (seqfile.c's version 2)
    report(mt_seqfile(view, v64, T.v, n, hR, Hs, 0, d, sizeof(d)), "seqfile/mtp_h", d);
    report(mt_seqfile(view, v64, T.v, n, hR, Hs, 1, d, sizeof(d)), "seqfile/mtp_h_corrupt", d);
    // mtp_load
    {
        RqModelInfo vi;
        rq_model_info(view, &vi);
        RqModelMem vm;
        RqSeqMem sm;
        rq_model_mem(M, &mem1);
        int ok = vi.n_layers == 1 && vi.linear_layers == 0 && vi.fingerprint != I.fingerprint && vi.vocab_size == V && vi.dim == D;
        ok &= rq_model_mem(view, &vm) == RQ_OK && vm.weights > 0 && vm.runtime > 0 && !memcmp(&mem0, &mem1, sizeof(mem0));
        ok &= rq_seq_mem(view, 4096, RQ_KV_F16, &sm) == RQ_OK && sm.state == 0 && sm.checkpoints == 0 &&
              sm.kv == (size_t)2 * 4096 * vi.n_kv_heads * vi.head_dim * 2 + 4096 * 4;
        // what a one-row pass reads (RqModelMem.read, the cost model's): the model's own = its weights; the view's: + the
        // classifier (more than its block on these files), less with a vocabulary subset (the same block)
        RqMtpOpts so = { 1000, -1 };
        RqModel *sv = rq_mtp_load(M, &so, err, sizeof(err));
        RqModelMem svm;
        ok &= mem0.read == mem0.weights && vm.read > 2 * vm.weights && sv && rq_model_mem(sv, &svm) == RQ_OK &&
              svm.weights == vm.weights && svm.read > svm.weights && svm.read < vm.read;
        snprintf(d, sizeof(d), "info (1 layer, no linear ones, a fingerprint of its own), estimate weights %.1f MiB + runtime "
                 "%.1f + %zu bytes per position (f16), the target's unchanged; a pass reads %.1f MiB (vocab 1000: %.1f; the "
                 "model's %.1f = its weights)", vm.weights / 1048576.0, vm.runtime / 1048576.0, (sm.kv - 4096 * 4) / 4096,
                 vm.read / 1048576.0, sv ? svm.read / 1048576.0 : 0.0, mem0.read / 1048576.0);
        rq_model_free(sv);
        report(ok, "mtp_load/info", d);
        // the block's matrices by their names (rq_dequant_rows: from the file's blocks and the repacked layout)
        ok = 1;
        d[0] = 0;
        const char *role[] = { "nextn.eh_proj", "attn_q", "ffn_down" };
        for (int i = 0; i < 3; i++) {
            char nm[96];
            float *rows = NULL, *packed = NULL;
            int nc = 0, np = 0;
            snprintf(nm, sizeof(nm), "blk.%d.%s.weight", I.n_layers, role[i]);
            int e = rq_dequant_rows(view, nm, strlen(nm), 0, 16, &rows, &packed, &nc, &np, err, sizeof(err));
            int okk = e == RQ_OK && nc == (i == 0 ? 2 * D : i == 2 ? I.hidden_dim : D) && (!np || same(rows, packed, (size_t)16 * nc));
            for (int j = 0; okk && j < 16 * nc; j++) okk &= rows[j] == rows[j];
            size_t dl = strlen(d);
            snprintf(d + dl, sizeof(d) - dl, "%s%s %d cols%s%s", dl ? ", " : "", nm, nc, np ? " (== repacked)" : "", okk ? "" : " BAD");
            ok &= okk;
            free(rows);
            free(packed);
        }
        report(ok, "mtp_load/dequant", d);
        // refusals: no block, optional nextn tensors, the memory budget; h_in on the target; a view's sequence on the
        // target's context
        ok = 1;
        d[0] = 0;
        if (plain) {
            RqModel *x = rq_model_load(plain, NULL, err, sizeof(err));
            RqModel *y = x ? rq_mtp_load(x, NULL, err, sizeof(err)) : NULL;
            ok &= x && !y && strstr(err, "no MTP block");
            snprintf(d, sizeof(d), "%s; ", err);
            rq_model_free(y);
            rq_model_free(x);
        }
        if (optional) {
            RqModel *x = rq_model_load(optional, NULL, err, sizeof(err));
            RqModel *y = x ? rq_mtp_load(x, NULL, err, sizeof(err)) : NULL;
            ok &= x && !y && strstr(err, "optional MTP tensors (nextn.embed_tokens / shared_head_head) not supported");
            size_t dl = strlen(d);
            snprintf(d + dl, sizeof(d) - dl, "%s; ", err);
            rq_model_free(y);
            rq_model_free(x);
        }
        RqMtpOpts tiny = { 0, 1e-6 };
        RqModel *y = rq_mtp_load(M, &tiny, err, sizeof(err));
        ok &= !y && strstr(err, "needs ~") && strstr(err, "not loaded");
        size_t dl = strlen(d);
        snprintf(d + dl, sizeof(d) - dl, "%s", err);
        RqModel *z = rq_mtp_load(view, NULL, err, sizeof(err));
        ok &= !z && strstr(err, "no MTP block"); // (a view has none)
        RqSegment sg = { s, T.v, 3, RQ_LOGITS_NONE, NULL, Hs };
        rq_seq_reset(s);
        ok &= rq_forward_batch(c64, &sg, 1, NULL, NULL) == RQ_ERR_ARG && fw_rows(c64, s, T.v, 3, RQ_LOGITS_NONE, NULL, NULL, NULL, Hs) == RQ_ERR_ARG &&
              fw_rows(c1, s, T.v, 3, RQ_LOGITS_NONE, NULL, NULL, NULL, Hs) == RQ_ERR_ARG && rq_seq_len(s) == 0;
        rq_seq_reset(vs);
        ok &= rq_decode(c1, vs, T.v[0], NULL) == RQ_ERR_ARG && rq_decode(v1, s, T.v[0], NULL) == RQ_ERR_ARG;
        report(ok, "mtp_load/refused", d);
        // a second view of the same target, and views loaded and freed twice: the same results; the target's after
        ok = 1;
        RqModel *v2 = rq_mtp_load(M, NULL, err, sizeof(err));
        for (int k = 0; k < 2 && ok; k++) {
            RqModel *w = rq_mtp_load(M, NULL, err, sizeof(err));
            RqCtx *wc = w ? rq_ctx_new(w, 8, 0, NULL) : NULL;
            RqSeq *ws = w ? rq_seq_new(w, 64, RQ_KV_DEFAULT, NULL) : NULL;
            float *lg = NULL;
            ok &= wc && ws && fw_rows(wc, ws, T.v, 5, RQ_LOGITS_LAST, NULL, &lg, NULL, Hs) == RQ_OK && hash_f(lg, V) == vlhR[4];
            rq_ctx_free(wc);
            rq_seq_free(ws);
            rq_model_free(w);
        }
        ok &= v2 != NULL;
        rq_model_free(v2);
        float *lg;
        rq_seq_reset(s);
        ok &= fw_rows(c64, s, T.v, 9, RQ_LOGITS_LAST, NULL, &lg, NULL, NULL) == RQ_OK && hash_f(lg, V) == lhR[8];
        report(ok, "mtp_load/twice", "a second view; views loaded, run and freed twice (== the first view); the target unchanged");
    }
    free(hR);
    free(Hs);
    free(vhR);
    free(full);
    free(lhR);
    free(vlhR);
    rq_tokens_free(&T);
    rq_seq_free(vs);
    rq_seq_free(s);
    rq_ctx_free(v1);
    rq_ctx_free(v64);
    rq_ctx_free(c1);
    rq_ctx_free(c64);
    rq_model_free(view); // (before the target: it shares its mapping, tokenizer and classifier)
    rq_model_free(M);
    return failed;
}

static int mtp_dump_main(int argc, char **argv) {
    // api_test --mtp-dump MODEL TEXTFILE OUT.bin [-b N] [--vocab N] (the head of this file)
    const char *path[3] = { NULL, NULL, NULL };
    int nb = 64, vocab = 0, bad = 0, np = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-b") && i + 1 < argc) nb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vocab") && i + 1 < argc) vocab = atoi(argv[++i]);
        else if (argv[i][0] == '-' || np == 3) bad = 1;
        else path[np++] = argv[i];
    }
    if (bad || np < 3) {
        fprintf(stderr, "usage: api_test --mtp-dump MODEL TEXTFILE OUT.bin [-b N] [--vocab N]\n");
        return 2;
    }
    rq_threads_init(0);
    char err[512];
    if (!(M = rq_model_load(path[0], NULL, err, sizeof(err)))) { fprintf(stderr, "%s\n", err); return 1; }
    rq_model_info(M, &I);
    V = I.vocab_size;
    D = I.dim;
    RqMtpOpts mo = { vocab, -1 };
    RqModel *view = rq_mtp_load(M, &mo, err, sizeof(err));
    if (!view) { fprintf(stderr, "%s\n", err); return 1; }
    FILE *f = fopen(path[1], "rb");
    char *text = NULL;
    long len = 0;
    if (f && !fseek(f, 0, SEEK_END) && (len = ftell(f)) >= 0 && !fseek(f, 0, SEEK_SET) && (text = malloc(len + 1)) &&
        fread(text, 1, len, f) == (size_t)len)
        text[len] = 0;
    else { fprintf(stderr, "can't read %s\n", path[1]); return 1; }
    fclose(f);
    RqTokens t = { 0 };
    if (rq_tokenize(M, text, len, &t) != RQ_OK || t.n < 1) { fprintf(stderr, "no tokens\n"); return 1; }
    int n = t.n, e;
    RqCtx *c = mt_ctx(M, nb), *vc = mt_ctx(view, nb);
    RqSeq *s = rq_seq_new(M, n, RQ_KV_DEFAULT, &e), *vs = s ? rq_seq_new(view, n, RQ_KV_DEFAULT, &e) : NULL;
    float *H = calloc((size_t)(n + 1) * D, sizeof(float)), *out = malloc((size_t)n * V * sizeof(float));
    if (!vs || !H || !out) { fprintf(stderr, "out of memory\n"); return 1; }
    // the target's hidden rows into H + D (row q's at H + (q + 1) D): the view's row q takes h_{q-1} = H + q D, row 0 zeros
    if ((e = fw_rows(c, s, t.v, n, RQ_LOGITS_NONE, NULL, NULL, H + D, NULL)) != RQ_OK ||
        (e = fw_rows(vc, vs, t.v, n, RQ_LOGITS_ALL, out, NULL, NULL, H)) != RQ_OK) {
        fprintf(stderr, "%s: %s\n", rq_strerror(e), rq_ctx_error(vc));
        return 1;
    }
    FILE *o = fopen(path[2], "wb");
    int32_t hd[2] = { n, V };
    int ok = o && fwrite(hd, 4, 2, o) == 2 && fwrite(t.v, 4, n, o) == (size_t)n && fwrite(out, sizeof(float), (size_t)n * V, o) == (size_t)n * V;
    if (!o || fclose(o) != 0 || !ok) { fprintf(stderr, "can't write %s\n", path[2]); return 1; }
    free(H);
    free(out);
    free(text);
    rq_tokens_free(&t);
    rq_seq_free(vs);
    rq_seq_free(s);
    rq_ctx_free(vc);
    rq_ctx_free(c);
    rq_model_free(view);
    rq_model_free(M);
    return 0;
}
