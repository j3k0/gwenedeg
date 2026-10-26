/* Batched decode benchmark (the library API, src/runq.h): N independent sequences decoding together,
 * one row per sequence per rq_forward_batch pass, the logits of every row, greedy (or seeded top-p)
 * sampling per row; aggregate and per-sequence tok/s per (context, N). Built by `make bench-batch`
 * (build/bench_batch); scripts/bench_batch.sh runs it over models and thread counts.
 *
 *   build/bench_batch MODEL [-N 1,2,4,8] [-L 50,500,2000] [-s STEPS] [-r REPS] [-k f16] [-t TEMP]
 *                     [-m MAXGB] [-d] [-q]
 *
 * Per context length L, one sequence is prefilled with L tokens of text and copied into the N
 * sequences (rq_seq_copy); every rep truncates them back to L and decodes STEPS tokens each (the first
 * one different per sequence). Reported: the best and the median rep (the machine is shared). -d adds
 * N = 1 through rq_decode (the single-sequence decode path) as the reference line "dec". -m skips
 * configurations whose KV cache would exceed MAXGB (default 6). A -DRUNQ_PROFILE build also prints
 * the time per phase type and pass of the best rep. Threads: OMP_NUM_THREADS (else the default). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/runq.h"

static int parse_list(const char *s, int *v, int max) {
    int n = 0;
    while (*s && n < max) {
        v[n++] = atoi(s);
        const char *c = strchr(s, ',');
        if (!c) break;
        s = c + 1;
    }
    return n;
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static void die(const char *what, int e, RqCtx *c) {
    fprintf(stderr, "bench_batch: %s: %s %s\n", what, rq_strerror(e), c ? rq_ctx_error(c) : "");
    exit(1);
}

int main(int argc, char **argv) {
    int Ns[64] = { 1, 2, 3, 4, 6, 8, 12, 16, 32, 64 }, nN = 10, Ls[16] = { 50, 500, 2000 }, nL = 3;
    int steps = 32, reps = 3, kvt = RQ_KV_DEFAULT, dec = 0, quiet = 0;
    float temp = 0, maxgb = 6;
    const char *path = NULL;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') { path = argv[i]; continue; }
        if (!strcmp(argv[i], "-d")) { dec = 1; continue; }
        if (!strcmp(argv[i], "-q")) { quiet = 1; continue; }
        if (i + 1 >= argc) goto usage;
        const char *v = argv[++i];
        switch (argv[i - 1][1]) {
            case 'N': nN = parse_list(v, Ns, 64); break;
            case 'L': nL = parse_list(v, Ls, 16); break;
            case 's': steps = atoi(v); break;
            case 'r': reps = atoi(v); break;
            case 'k': kvt = rq_kv_parse(v); break;
            case 't': temp = atof(v); break;
            case 'm': maxgb = atof(v); break;
            default: goto usage;
        }
    }
    if (!path || steps < 1 || reps < 1 || kvt < 0) goto usage;

    const char *w = rq_threads_init(0);
    if (w) fprintf(stderr, "%s\n", w);
    char err[512];
    RqModel *m = rq_model_load(path, NULL, err, sizeof(err));
    if (!m) { fprintf(stderr, "bench_batch: %s\n", err); return 1; }
    RqModelInfo I;
    rq_model_info(m, &I);
    int V = I.vocab_size, nmax = 1, e;
    for (int i = 0; i < nN; i++) nmax = Ns[i] > nmax ? Ns[i] : nmax;
    RqCtx *c = rq_ctx_new(m, nmax > 1 ? (nmax > 64 ? nmax : 64) : 64, 0, &e);
    if (!c) die("rq_ctx_new", e, NULL);
    float *out = malloc((size_t)nmax * V * sizeof(float));
    RqSeq **sq = calloc(nmax, sizeof(RqSeq *));
    RqSegment *sg = calloc(nmax, sizeof(RqSegment));
    int *tok = calloc(nmax, sizeof(int));
    RqSampling smp = { temp, temp > 0 ? 0.9f : 0, NULL };
    const char *pn[16];
    double p0[16], p1[16], pbest[16];
    int np = rq_ctx_profile(c, pn, p0, 16);
    char load[64] = "";
    FILE *lf = fopen("/proc/loadavg", "r");
    if (lf) { if (!fgets(load, sizeof(load), lf)) load[0] = 0; fclose(lf); load[strcspn(load, " ")] = 0; }
    printf("# %s, %d threads, KV %s, %d steps x %d reps, %s, load %s\n", path, rq_threads(), rq_kv_name(kvt), steps, reps,
           temp > 0 ? "top-p 0.9 sampling" : "greedy", load);
    printf("# %5s %4s %9s %10s %9s %10s\n", "L", "N", "ms/pass", "agg tok/s", "per-seq", "median agg");

    RqTokens text = { 0 }, para = { 0 };
    rq_tokenize(m, "The keeper of the lighthouse rose before dawn, trimmed the wicks, polished the great lens and "
                   "wrote the wind, the swell and the passing ships into the log, as his father had done. ", -1, &para);
    for (int li = 0; li < nL; li++) {
        int L = Ls[li] < 1 ? 1 : Ls[li], cap = L + steps + 1;
        text.n = 0;
        while (text.n < L) rq_tokens_push(&text, para.v, para.n);
        RqSeq *base = rq_seq_new(m, cap, kvt, &e);
        if (!base) die("rq_seq_new", e, NULL);
        if ((e = rq_prefill(c, base, text.v, L, RQ_LOGITS_NONE, NULL, NULL)) != RQ_OK) die("prefill", e, c);
        for (int ni = -dec; ni < nN; ni++) {
            int N = ni < 0 ? 1 : Ns[ni], viadec = ni < 0;
            double gb = (double)N * cap * I.n_layers * 2 * I.n_kv_heads * I.head_dim * (kvt == RQ_KV_F32 ? 4 : 2) / 1e9;
            if (N < 1 || N > nmax || gb > maxgb) {
                printf("  %5d %4d   (skipped: %.1f GB of KV cache)\n", L, N, gb);
                continue;
            }
            for (int i = 0; i < N; i++) {
                if (!sq[i] && !(sq[i] = rq_seq_new(m, cap, kvt, &e))) die("rq_seq_new", e, NULL);
                if (rq_seq_capacity(sq[i]) < cap) {
                    rq_seq_free(sq[i]);
                    if (!(sq[i] = rq_seq_new(m, cap, kvt, &e))) die("rq_seq_new", e, NULL);
                }
                if ((e = rq_seq_copy(sq[i], base, L)) != RQ_OK) die("rq_seq_copy", e, NULL);
                rq_seq_seed(sq[i], 1234 + i);
            }
            double *t = malloc(reps * sizeof(double)), best = 1e30;
            for (int r = 0; r < reps; r++) {
                for (int i = 0; i < N; i++) {
                    rq_seq_truncate(sq[i], L);
                    tok[i] = 1000 + 13 * i;
                }
                if (np) rq_ctx_profile(c, pn, p0, 16);
                double t0 = rq_now();
                for (int s = 0; s < steps; s++) {
                    if (viadec) {
                        float *lg;
                        if ((e = rq_decode(c, sq[0], tok[0], &lg)) != RQ_OK) die("rq_decode", e, c);
                        tok[0] = rq_sample(c, sq[0], lg, &smp);
                        continue;
                    }
                    for (int i = 0; i < N; i++)
                        sg[i] = (RqSegment) { sq[i], &tok[i], 1, RQ_LOGITS_LAST };
                    if ((e = rq_forward_batch(c, sg, N, out, NULL)) != RQ_OK) die("rq_forward_batch", e, c);
                    for (int i = 0; i < N; i++)
                        tok[i] = rq_sample(c, sq[i], out + (size_t)i * V, &smp);
                }
                t[r] = rq_now() - t0;
                if (np) {
                    rq_ctx_profile(c, pn, p1, 16);
                    if (t[r] < best)
                        for (int k = 0; k < np; k++) pbest[k] = p1[k] - p0[k];
                }
                best = t[r] < best ? t[r] : best;
            }
            qsort(t, reps, sizeof(double), cmp_d);
            double med = t[reps / 2];
            printf("  %5d %4d%s %9.2f %10.1f %9.1f %10.1f\n", L, N, viadec ? "d" : " ", best / steps * 1e3,
                   N * steps / best, steps / best, N * steps / med);
            if (np && !viadec && !quiet) {
                printf("        profile ms/pass:");
                for (int k = 0; k < np; k++) printf(" %s %.3f", pn[k], pbest[k] / steps * 1e3);
                printf("\n");
            }
            fflush(stdout);
            free(t);
        }
        rq_seq_free(base);
    }
    for (int i = 0; i < nmax; i++) rq_seq_free(sq[i]);
    rq_tokens_free(&text);
    rq_tokens_free(&para);
    free(out);
    free(sq);
    free(sg);
    free(tok);
    rq_ctx_free(c);
    rq_model_free(m);
    return 0;
usage:
    fprintf(stderr, "usage: bench_batch MODEL [-N 1,2,4,8] [-L 50,500,2000] [-s STEPS] [-r REPS] [-k f16|bf16|f32] "
                    "[-t TEMP] [-m MAXGB] [-d] [-q]\n");
    return 2;
}
