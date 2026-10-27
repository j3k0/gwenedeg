/* Tests of the library API (src/runq.h), run by `make test` (tests/check.py, group api) on each model:
 *   tests/api_test MODEL [--digest | --digest-small [--long]] [--n65] [--kv]
 * Prints PASS / FAIL lines, exit status 1 if any failed. The forward-pass checks are bitwise: a
 * sequence's logits must not depend on what else the context ran in between or batched with it
 * (sequences interleaved, several sequences in one rq_forward_batch, a prefill chunk next to decode
 * rows, positions past one attention block of 256), nor on truncating and re-prefilling.
 * Batched decode (section 9): N sequences decoding together, one row each, == each alone (rq_decode);
 * the batched passes also print DIGEST lines (a hash of their logits), which must be the same for any
 * thread count and build (check.py compares runs at other thread counts and of the generic build);
 * --digest: only those passes (no references, no PASS / FAIL lines); --digest-small: only the short
 * ones (for the slow generic build; --long: and the long ones); --n65: also 65 sequences in one pass
 * (references for 48 more sequences: one model is enough); --kv: also a bf16 KV cache.
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
 * back, as a chat's next turn) and each draft setting (k = 1 / 3 / 8, adaptive tau 0.5 with k 8; --quick: k 3 and
 * adaptive, fewer cases): the tokens, the RNG state after each, every sampled logits vector's position and hash, the
 * stop, the cache's tokens and one more decode step's logits. Also k 0 == rq_generate, the vocabulary check (the draft
 * itself accepted; --other: a model of another vocabulary refused, loaded tokenizer-only) and, for Qwen3.5, that
 * every rollback landed on its checkpoint (RqSpecStats.refed 0).
 *   tests/api_test MODEL --sample LOGITS_FILE < PLAN
 * rq_sample on the rows of a -m logits file, with the settings and RNG states of PLAN's lines ("T P K ROW
 * SEED..."): SAMPLE lines for tests/sampler_ref.py (the sampler's exact order replayed in numpy: top-k). */
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

static RqSeq *seq_new(int cap) {
    int e;
    RqSeq *s = rq_seq_new(M, cap, RQ_KV_DEFAULT, &e);
    if (!s) die("rq_seq_new", e);
    return s;
}

static int same(const float *a, const float *b, size_t n) { return !memcmp(a, b, n * sizeof(float)); }

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
    return 0;
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

int main(int argc, char **argv) {
    int only_digest = 0;
    if (argc > 1 && !strcmp(argv[1], "--two")) return two_main(argc, argv);
    if (argc > 1 && !strcmp(argv[1], "--spec")) return spec_main(argc, argv);
    if (argc > 3 && !strcmp(argv[2], "--sample")) return sample_main(argc, argv);
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--digest")) only_digest = 1;
        else if (!strcmp(argv[i], "--digest-small")) only_digest = 2;
        else if (!strcmp(argv[i], "--long")) bd_long = 1;
        else if (!strcmp(argv[i], "--n65")) bd_n65 = 1;
        else if (!strcmp(argv[i], "--kv")) bd_kv = 1;
        else argc = 0;
    }
    if (argc < 2) {
        fprintf(stderr, "usage: api_test MODEL [--digest | --digest-small [--long]] [--n65] [--kv]\n"
                        "       api_test --two MODEL_A MODEL_B [--gs MODEL_OTHER_GROUP_SIZE]\n"
                        "       api_test --spec TARGET DRAFT [--quick] [--other MODEL_OTHER_VOCABULARY]\n"
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
        bd_refs(c, bs, 8, 25, ns, RQ_KV_DEFAULT, hs);
        bd_refs(c, bl, 248, 12, nl, RQ_KV_DEFAULT, hl);
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

        // KV bf16 (f16 above; f32: the plain fp32 tile layout of the same kernels): 3 long sequences
        for (int kvt = RQ_KV_BF16; kvt <= RQ_KV_BF16 && bd_kv; kvt++) {
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
typedef struct {
    const char *name;
    int prompt;          // sp_prompt: 0 a chat turn, thinking off; 1 thinking on; 2 raw text (code)
    RqSampling cfg;
    unsigned long long seed;
    int max_new;
    int cap, dcap;       // the target's / draft's capacity past the prompt (0: the model's; < 0: shorter than it)
    int turn2;           // a second call: both cut back 3 positions (at most to the prompt), then a user turn
    int quick;           // in --quick
} SpCase;

typedef struct {
    GenRec rec;
    int stop[2], ngen[2], len;
    unsigned long long rng, htok, hnext;
    RqSpecStats st;      // summed over the calls (gen: the last call's)
} SpOut;

static RqModel *SD;
static RqTokens sp_prompt[3], sp_turn2;

static int sp_run(RqCtx *c, RqCtx *dc, const SpCase *k, const RqSpecOpts *o, SpOut *out) {
    // o NULL: rq_generate
    int e, np = sp_prompt[k->prompt].n;
    RqSeq *s = rq_seq_new(M, k->cap ? np + k->cap : 0, RQ_KV_DEFAULT, &e);
    if (!s) die("rq_seq_new", e);
    RqSeq *ds = o ? rq_seq_new(SD, k->dcap ? np + k->dcap : 0, RQ_KV_DEFAULT, &e) : NULL;
    if (o && !ds) die("rq_seq_new (draft)", e);
    rq_seq_seed(s, k->seed);
    memset(out, 0, sizeof(*out));
    out->rec.s = s;
    RqCallbacks cb = { gr_token, gr_logits, &out->rec };
    RqTokens t = { 0 };
    rq_tokens_push(&t, sp_prompt[k->prompt].v, np);
    for (int call = 0; call <= k->turn2; call++) {
        RqSpecStats st;
        if (o) {
            e = rq_generate_spec(c, s, dc, ds, t.v, t.n, &k->cfg, k->max_new, o, &cb, &st);
            out->st.passes += st.passes, out->st.drafted += st.drafted, out->st.accepted += st.accepted;
            out->st.refed += st.refed, out->st.draft_steps += st.draft_steps;
        } else {
            e = rq_generate(c, s, t.v, t.n, &k->cfg, k->max_new, &cb, &st.gen);
        }
        if (e != RQ_OK) break;
        out->st.gen = st.gen;
        out->stop[call] = st.gen.stop;
        out->ngen[call] = st.gen.n_gen;
        if (call < k->turn2) { // (Qwen3.5: the truncation goes back to a checkpoint: what it lost is fed again)
            int L = rq_seq_len(s) - 3 < np ? np : rq_seq_len(s) - 3;
            t.n = 0;
            rq_tokens_push(&t, rq_seq_tokens(s), L);
            if ((e = rq_seq_truncate(s, L)) != RQ_OK) break;
            memmove(t.v, t.v + rq_seq_len(s), (size_t)(L - rq_seq_len(s)) * sizeof(int));
            t.n = L - rq_seq_len(s);
            rq_tokens_push(&t, sp_turn2.v, sp_turn2.n);
        }
    }
    out->len = rq_seq_len(s);
    out->htok = hash_f((const float *)rq_seq_tokens(s), (size_t)out->len);
    out->rng = *rq_seq_rng(s);
    float *lg;
    if (e == RQ_OK && out->len < rq_seq_capacity(s) && (e = rq_decode(c, s, 1000, &lg)) == RQ_OK)
        out->hnext = rq_logits_hash(lg, V);
    rq_tokens_free(&t);
    rq_seq_free(s);
    rq_seq_free(ds);
    return e;
}

static int sp_same(const SpOut *a, const SpOut *b) {
    return gr_same(&a->rec, &b->rec) && !memcmp(a->stop, b->stop, sizeof(a->stop)) && !memcmp(a->ngen, b->ngen, sizeof(a->ngen)) &&
           a->len == b->len && a->htok == b->htok && a->rng == b->rng && a->hnext == b->hnext;
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
        fprintf(stderr, "usage: api_test --spec TARGET DRAFT [--quick] [--other MODEL_OTHER_VOCABULARY]\n");
        return 2;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    rq_threads_init(0);
    char err[512], d[2048];
    if (!(M = rq_model_load(path[0], NULL, err, sizeof(err))) || !(SD = rq_model_load(path[1], NULL, err, sizeof(err)))) {
        printf("FAIL setup  %s\n", err);
        return 1;
    }
    rq_model_info(M, &I);
    V = I.vocab_size;
    int e;
    RqCtx *c = rq_ctx_new(M, 64, 0, &e), *dc = c ? rq_ctx_new(SD, 64, 0, &e) : NULL;
    if (!dc) die("rq_ctx_new", e);

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
    };
    const RqSpecOpts opts[] = { { 1, 0 }, { 3, 0 }, { 8, 0 }, { 8, 0.5f } };
    const char *stop_name[] = { "end", "limit", "context", "callback" };
    int refed = 0;
    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        const SpCase *k = &cases[ci];
        if (quick && !k->quick) continue;
        SpOut ref, got;
        int ok = (e = sp_run(c, dc, k, NULL, &ref)) == RQ_OK;
        snprintf(d, sizeof(d), "%d%s tok, stop %s%s%s:", ref.ngen[0] + ref.ngen[1], k->turn2 ? " (2 calls)" : "",
                 stop_name[ref.stop[k->turn2] & 3], k->turn2 ? "/" : "", k->turn2 ? stop_name[ref.stop[1] & 3] : "");
        for (size_t oi = 0; oi < sizeof(opts) / sizeof(opts[0]) && ok; oi++) {
            if (quick && oi != 1 && oi != 3) continue;
            int okk = (e = sp_run(c, dc, k, &opts[oi], &got)) == RQ_OK && sp_same(&ref, &got);
            okk &= got.st.accepted <= got.st.drafted && (I.linear_layers || got.st.refed == 0);
            refed += got.st.refed;
            size_t dl = strlen(d);
            snprintf(d + dl, sizeof(d) - dl, " k%d%s %d/%d in %d passes%s%s", opts[oi].k, opts[oi].tau > 0 ? " tau 0.5" : "",
                     got.st.accepted, got.st.drafted, got.st.passes, got.st.refed ? " REFED" : "", okk ? "" : " DIFFER");
            ok &= okk;
        }
        if (e != RQ_OK) {
            size_t dl = strlen(d);
            snprintf(d + dl, sizeof(d) - dl, " error %s (%s)", rq_strerror(e), rq_ctx_error(c));
        }
        char name[96];
        snprintf(name, sizeof(name), "spec_%s", k->name);
        report(ok, name, d);
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
    for (int i = 0; i < 3; i++) rq_tokens_free(&sp_prompt[i]);
    rq_tokens_free(&sp_turn2);
    rq_ctx_free(dc);
    rq_ctx_free(c);
    rq_model_free(SD);
    rq_model_free(M);
    return failed;
}
