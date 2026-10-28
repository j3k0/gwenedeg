/* Teacher-forced draft predictions on long token streams (TODO G0 (c), scripts/lookup_accept.py `draft`): what a
 * draft model would guess at every position of the given replies, given everything before them (an agentic
 * session's context: tool results, earlier replies). `runq -m logits` writes every position's full logits (~1 MB per
 * position for a 248k vocabulary); this keeps only the argmax and its probability at the positions asked for.
 *
 *   build/teacher_force MODEL IN OUT [-b 512] [-k f16]
 * IN (int32, little endian): n_streams, then per stream: n_tokens, the tokens, n_spans, then (start, end) per span
 * (increasing, non-overlapping, 1 <= start < end <= n_tokens). Each stream runs on a fresh sequence of n_tokens
 * positions: the tokens before each span prefilled (no logits, chunks of -b), the span's predicting rows start - 1 ..
 * end - 2 with every row's logits (chunks of 32). OUT: per stream, per span, per position p in [start, end): int32
 * argmax of the logits of row p - 1 (the lowest id on ties), float32 its softmax probability. The tokens after the
 * last span are not fed. Built by `make teacher-force`. Threads: OMP_NUM_THREADS.
 *
 *   build/teacher_force -l IN OUT
 * The prompt lookup instead (rq_lookup_*, src/spec.c; no model): at every span position p, the lookup src/spec.c's loop
 * would run with the tokens before p - 1 indexed and t = token p - 1 (kmax 64, min_len 2). OUT: per position int32
 * match length (0: no match), int32 guesses, int32 how many of them equal the tokens from p on (within the span). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../src/runq.h"

static void die(const char *what, int e, RqCtx *c) {
    fprintf(stderr, "teacher_force: %s: %s %s\n", what, rq_strerror(e), c ? rq_ctx_error(c) : "");
    exit(1);
}

static int rd(FILE *f, int *v, size_t n) { return fread(v, sizeof(int), n, f) == n; }

static int lookup_main(const char *inp, const char *outp) {
    FILE *in = fopen(inp, "rb"), *out = fopen(outp, "wb");
    if (!in || !out) { perror("teacher_force"); return 1; }
    int ns, g[64];
    if (!rd(in, &ns, 1)) die("input", RQ_ERR_ARG, NULL);
    for (int si = 0; si < ns; si++) {
        int n, nsp;
        if (!rd(in, &n, 1) || n < 1) die("input", RQ_ERR_ARG, NULL);
        int *tok = malloc((size_t)n * sizeof(int));
        if (!tok || !rd(in, tok, (size_t)n) || !rd(in, &nsp, 1)) die("input", RQ_ERR_ARG, NULL);
        RqLookup *lk = rq_lookup_new(n);
        if (!lk) die("rq_lookup_new", RQ_ERR_NOMEM, NULL);
        for (int j = 0; j < nsp; j++) {
            int ab[2];
            if (!rd(in, ab, 2) || ab[0] < 1 || ab[1] <= ab[0] || ab[1] > n) die("span", RQ_ERR_ARG, NULL);
            for (int p = ab[0]; p < ab[1]; p++) {
                if (rq_lookup_add(lk, tok, p - 1) != RQ_OK) die("rq_lookup_add", RQ_ERR_ARG, NULL);
                int r[3];
                r[1] = rq_lookup_draft(lk, tok, tok[p - 1], 64, 2, g, &r[0]);
                r[2] = 0;
                while (r[2] < r[1] && p + r[2] < ab[1] && g[r[2]] == tok[p + r[2]]) r[2]++;
                fwrite(r, sizeof(int), 3, out);
            }
        }
        rq_lookup_free(lk);
        free(tok);
    }
    fclose(in);
    return fclose(out) ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[1], "-l")) return lookup_main(argv[2], argv[3]);
    if (argc < 4) {
        fprintf(stderr, "usage: teacher_force MODEL IN OUT [-b 512] [-k f16] | teacher_force -l IN OUT\n");
        return 2;
    }
    int chunk = 512, kv = RQ_KV_DEFAULT;
    for (int i = 4; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "-b")) chunk = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-k")) kv = rq_kv_parse(argv[i + 1]);
        else { fprintf(stderr, "teacher_force: unknown option %s\n", argv[i]); return 2; }
    }
    if (chunk < 32 || kv < 0) { fprintf(stderr, "teacher_force: bad -b / -k\n"); return 2; }
    const char *warn = rq_threads_init(0);
    if (warn) fprintf(stderr, "%s\n", warn);
    char err[512];
    RqModel *m = rq_model_load(argv[1], NULL, err, sizeof(err));
    if (!m) { fprintf(stderr, "teacher_force: %s\n", err); return 1; }
    RqModelInfo info;
    rq_model_info(m, &info);
    int V = info.vocab_size, e;
    RqCtx *c = rq_ctx_new(m, chunk, 0, &e);
    if (!c) die("rq_ctx_new", e, NULL);
    FILE *in = fopen(argv[2], "rb"), *out = fopen(argv[3], "wb");
    if (!in || !out) { perror("teacher_force"); return 1; }
    enum { R = 32 };
    float *lg = malloc((size_t)R * V * sizeof(float));
    int ns;
    if (!lg || !rd(in, &ns, 1)) die("input", RQ_ERR_ARG, NULL);
    double t0 = rq_now();
    long fed = 0, rows = 0;
    for (int si = 0; si < ns; si++) {
        int n, nsp;
        if (!rd(in, &n, 1) || n < 1) die("input", RQ_ERR_ARG, NULL);
        int *tok = malloc((size_t)n * sizeof(int));
        if (!tok || !rd(in, tok, (size_t)n) || !rd(in, &nsp, 1)) die("input", RQ_ERR_ARG, NULL);
        int *sp = malloc((size_t)(2 * nsp + 1) * sizeof(int));
        if (!sp || !rd(in, sp, (size_t)2 * nsp)) die("input", RQ_ERR_ARG, NULL);
        int last = nsp ? sp[2 * nsp - 1] : 0;
        RqSeq *s = rq_seq_new(m, last > 0 ? last : 1, kv, &e);
        if (!s) die("rq_seq_new", e, NULL);
        for (int j = 0; j < nsp; j++) {
            int a = sp[2 * j], b = sp[2 * j + 1], L = rq_seq_len(s);
            if (a < 1 || b <= a || b > n || a - 1 < L) die("span", RQ_ERR_ARG, NULL);
            if (a - 1 > L && (e = rq_prefill(c, s, tok + L, a - 1 - L, RQ_LOGITS_NONE, NULL, NULL)) != RQ_OK)
                die("prefill", e, c);
            fed += a - 1 - L;
            for (int r0 = a - 1; r0 < b - 1; r0 += R) {
                int nr = b - 1 - r0 < R ? b - 1 - r0 : R;
                float *rowsp;
                if ((e = rq_prefill(c, s, tok + r0, nr, RQ_LOGITS_ALL, lg, &rowsp)) != RQ_OK) die("prefill rows", e, c);
                for (int r = 0; r < nr; r++) {
                    const float *x = rowsp + (size_t)r * V;
                    int best = 0;
                    for (int i = 1; i < V; i++)
                        if (x[i] > x[best]) best = i;
                    double sum = 0;
                    for (int i = 0; i < V; i++) sum += exp((double)x[i] - x[best]);
                    float p1 = (float)(1 / sum);
                    fwrite(&best, sizeof(int), 1, out);
                    fwrite(&p1, sizeof(float), 1, out);
                }
                fed += nr;
                rows += nr;
            }
        }
        rq_seq_free(s);
        free(tok);
        free(sp);
        fprintf(stderr, "[teacher_force] stream %d/%d: %d tokens, %d spans; total %ld fed (%ld rows) in %.0f s\n", si + 1, ns,
                n, nsp, fed, rows, rq_now() - t0);
    }
    fclose(in);
    if (fclose(out)) { perror("teacher_force"); return 1; }
    free(lg);
    rq_ctx_free(c);
    rq_model_free(m);
    return 0;
}
