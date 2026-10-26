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
 * (references for 48 more sequences: one model is enough); --kv: also a bf16 KV cache. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
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

// Batched decode rows. Sequence i of a set: the first plen(i) positions of a base sequence (rq_seq_copy:
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
        if ((e = rq_seq_copy(s, base, plen(i, lo, span))) != RQ_OK) die("rq_seq_copy", e);
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
        ok &= rq_seq_copy(sq[i], base, plen(i, lo, span)) == RQ_OK;
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

static int on_tok(void *u, int t) {
    RqTokens *r = u;
    rq_tokens_push(r, &t, 1);
    return 0;
}

static void batch_tests(RqCtx *c, RqTokens A, RqTokens B, RqTokens C, const float *refB, int only_digest);

int main(int argc, char **argv) {
    int only_digest = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--digest")) only_digest = 1;
        else if (!strcmp(argv[i], "--digest-small")) only_digest = 2;
        else if (!strcmp(argv[i], "--long")) bd_long = 1;
        else if (!strcmp(argv[i], "--n65")) bd_n65 = 1;
        else if (!strcmp(argv[i], "--kv")) bd_kv = 1;
        else argc = 0;
    }
    if (argc < 2) {
        fprintf(stderr, "usage: api_test MODEL [--digest | --digest-small [--long]] [--n65] [--kv]\n");
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
        RqSegment s2[2] = { { sb, B.v + nb0, B.n - nb0, RQ_LOGITS_LAST }, { sc, C.v, 20, RQ_LOGITS_NONE } };
        ok &= rq_forward_batch(c, s2, 2, NULL, &lg) == RQ_OK && lg && same(lg, refB + (size_t)(B.n - 1) * V, V);
        for (int i = 20; i < C.n && ok; i += 64) { // the rest of C in 64-row chunks
            int k = C.n - i < 64 ? C.n - i : 64;
            RqSegment s = { sc, C.v + i, k, RQ_LOGITS_LAST };
            ok &= rq_forward_batch(c, &s, 1, NULL, &lg) == RQ_OK;
        }
        ok &= same(lg, refC + (size_t)(C.n - 1) * V, V);
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

    // 3. truncate + re-prefill == fresh prefill (positions overwritten in the cache)
    {
        RqSeq *s = seq_new(0);
        float *lg;
        int ok = rq_prefill(c, s, B.v, B.n, RQ_LOGITS_NONE, NULL, NULL) == RQ_OK;
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
    // without their reasoning
    {
        RqMessage msgs[4] = { { RQ_ROLE_SYSTEM, "Be brief." }, { RQ_ROLE_USER, "Hi" },
                              { RQ_ROLE_ASSISTANT, "<think>\nhmm\n</think>\n\nHello!" }, { RQ_ROLE_USER, "Bye" } };
        int ok = 1;
        for (int th = 0; th < 2; th++) {
            RqTokens r = { 0 }, x = { 0 };
            ok &= rq_chat_render(M, msgs, 4, th, &r) == RQ_OK;
            rq_chat_system(M, "Be brief.", th, &x);
            rq_chat_user(M, "Hi", th, &x);
            RqTokens h = toks(th ? "Hello!" : "<think>\nhmm\n</think>\n\nHello!");
            rq_tokens_push(&x, h.v, h.n);
            rq_chat_end(M, &x);
            rq_chat_user(M, "Bye", th, &x);
            ok &= r.n == x.n && !memcmp(r.v, x.v, r.n * sizeof(int));
            // the history form from the generated tokens equals the one from the text
            RqTokens g = toks("<think>\nhmm\n</think>\n\nHello!"), hg = { 0 };
            ok &= rq_chat_reply_history(M, g.v, g.n, th, &hg) == RQ_OK && hg.n == h.n && !memcmp(hg.v, h.v, h.n * sizeof(int));
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
        report(ok, "chat_render", "system, user, assistant, user; thinking off / on");
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
            okm &= rq_seq_copy(sq[i], i < nlr ? bl : bs, i < nlr ? plen(i, 248, 12) : plen(i, 8, 25)) == RQ_OK;
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
    printf("DIGEST all%s %016llx\n", small ? "_small" : "", (unsigned long long)digest);

    if (!only_digest) {
        // batched sampling == rq_sample per row (tokens and RNG states), mixed settings per row
        RqSampling cf[4] = { { 0, 0, NULL }, { 0.8f, 0.9f, NULL }, { 1.5f, 0.95f, NULL }, { 1.0f, 1.0f, NULL } }, cfg[9];
        float *l2 = malloc((size_t)9 * V * sizeof(float));
        int t1[9], t2[9], oks = bd_batch(c, sq, bs, 8, 25, 9, NULL, out, "batch_sample_rows");
        unsigned long long rng[9];
        for (int i = 0; i < 9; i++) {
            cfg[i] = cf[i % 4];
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
        report(oks, "batch_sample", "rq_sample_batch == rq_sample per row (greedy, top-p, full distribution)");
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
