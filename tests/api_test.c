/* Tests of the library API (src/runq.h), run by `make test` (tests/check.py, group api) on each model:
 *   tests/api_test MODEL
 * Prints PASS / FAIL lines, exit status 1 if any failed. The forward-pass checks are bitwise: a
 * sequence's logits must not depend on what else the context ran in between or batched with it
 * (sequences interleaved, several sequences in one rq_forward_batch, a prefill chunk next to decode
 * rows, positions past one attention block of 256), nor on truncating and re-prefilling. */
#include <stdio.h>
#include <stdlib.h>
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

static int on_tok(void *u, int t) {
    RqTokens *r = u;
    rq_tokens_push(r, &t, 1);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: api_test MODEL\n");
        return 2;
    }
    char err[512];
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
