/* Verify-pass benchmark (the library API, src/runq.h; TODO S2): the cost of a speculative verify pass, a
 * segment of r rows of ONE sequence with every row's logits (RQ_LOGITS_ALL), against one decode step of that
 * sequence (rq_decode, logits) and against r decode rows of r different sequences (one rq_forward_batch pass,
 * one row and its logits per sequence), at a given context length. Built by `make bench-verify`
 * (build/bench_verify).
 *
 *   build/bench_verify MODEL [-L 500,2000] [-R 1,2,3,4,5,6,7,8,9] [-r REPS] [-k f16] [-c] [-M MIX] [-A NAME=VALUE ...]
 *
 * Per context length L, one sequence is prefilled with L tokens of text and copied into the others
 * (rq_seq_copy; Qwen3.5: a checkpoint at L, so that rq_seq_truncate goes back to it). Every rep runs each
 * configuration once, in turn (so the shared machine's load hits all of them alike), each pass followed by
 * a truncation back to L (not timed): "dec" (rq_decode of the next token), then per r "seg" (r tokens of the
 * text at positions L..L+r-1, logits of all rows) and "rows" (r sequences, one token each). -c: a checkpoint
 * per row of each segment (rq_seq_checkpoint_rows; Qwen3.5, the rollback of TODO S3). -M MIX: every seg pass
 * also carries MIX decode rows of other sequences (a mixed pass, as the server's verify rows next to other
 * requests' decode rows). Reported per configuration: the median and the best pass in ms and
 * f = median / median(dec). -A "NAME=VALUE ..." (up to 3): one more compute context, created with these
 * environment variables set (e.g. "RUNQ_BD=64,2,8,1 RUNQ_SMB=32,32,32,32,0": verify segments on the prefill path
 * and no prefetch in the small-batch kernels, as before TODO S2);
 * every pass then runs on each context in turn (a paired comparison on the shared machine), one table per
 * context and the ratio of each to the first. A -DRUNQ_PROFILE build also prints the mean time per phase type
 * and pass of each batched configuration (rq_ctx_profile). Threads: OMP_NUM_THREADS (else the default). */
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
    fprintf(stderr, "bench_verify: %s: %s %s\n", what, rq_strerror(e), c ? rq_ctx_error(c) : "");
    exit(1);
}

#define RMAX 32
#define NCTX 4

static void back_to(RqSeq *s, int L, int hybrid) {
    // truncate to L (Qwen3.5: through its checkpoint at L, which stays)
    int e = rq_seq_truncate(s, L);
    if (e != RQ_OK || rq_seq_len(s) != L) die("rq_seq_truncate", e, NULL);
    if (hybrid && (e = rq_seq_checkpoint_at(s, L)) != RQ_OK) die("rq_seq_checkpoint_at", e, NULL);
}

int main(int argc, char **argv) {
    int Rs[RMAX] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, nR = 9, Ls[16] = { 500, 2000 }, nL = 2;
    int reps = 41, kvt = RQ_KV_DEFAULT, ck = 0, mix = 0, nctx = 1;
    const char *path = NULL, *var[NCTX] = { "default" };
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') { path = argv[i]; continue; }
        if (!strcmp(argv[i], "-c")) { ck = 1; continue; }
        if (i + 1 >= argc) goto usage;
        const char *v = argv[++i];
        switch (argv[i - 1][1]) {
            case 'L': nL = parse_list(v, Ls, 16); break;
            case 'R': nR = parse_list(v, Rs, RMAX); break;
            case 'r': reps = atoi(v); break;
            case 'k': kvt = rq_kv_parse(v); break;
            case 'M': mix = atoi(v); break;
            case 'A':
                if (nctx == NCTX || !strchr(v, '=')) goto usage;
                var[nctx++] = v;
                break;
            default: goto usage;
        }
    }
    int rmax = 1;
    for (int i = 0; i < nR; i++) {
        if (Rs[i] < 1 || Rs[i] > RMAX) goto usage;
        rmax = Rs[i] > rmax ? Rs[i] : rmax;
    }
    if (!path || reps < 1 || kvt < 0 || mix < 0 || mix > RMAX) goto usage;

    const char *w = rq_threads_init(0);
    if (w) fprintf(stderr, "%s\n", w);
    char err[512];
    RqModel *m = rq_model_load(path, NULL, err, sizeof(err));
    if (!m) { fprintf(stderr, "bench_verify: %s\n", err); return 1; }
    RqModelInfo I;
    rq_model_info(m, &I);
    int V = I.vocab_size, hybrid = I.linear_layers > 0, nseq = 1 + (rmax > mix ? rmax : mix), e;
    RqCtx *cs[NCTX], *c;
    for (int x = 0; x < nctx; x++) { // (variant x: its environment variables set while the context reads its settings)
        char name[8][128], *old[8] = { NULL }, buf[512];
        int nv = 0;
        snprintf(buf, sizeof(buf), "%s", var[x]);
        for (char *a = x ? strtok(buf, " ") : NULL; a && nv < 8; a = strtok(NULL, " "), nv++) {
            char *eq = strchr(a, '=');
            if (!eq) goto usage;
            snprintf(name[nv], sizeof(name[nv]), "%.*s", (int)(eq - a), a);
            if (getenv(name[nv])) old[nv] = strdup(getenv(name[nv]));
            setenv(name[nv], eq + 1, 1);
        }
        if (!(cs[x] = rq_ctx_new(m, 64, 0, &e))) die("rq_ctx_new", e, NULL);
        for (int i = 0; i < nv; i++) {
            if (old[i]) setenv(name[i], old[i], 1); else unsetenv(name[i]);
            free(old[i]);
        }
    }
    c = cs[0];
    float *out = malloc((size_t)(rmax + mix) * V * sizeof(float));
    RqSeq **sq = calloc(nseq, sizeof(RqSeq *)); // sq[0]: dec and seg; sq[1..]: rows (and the mixed rows)
    RqSegment sg[RMAX + 1];
    int tok[RMAX];
    char load[64] = "";
    FILE *lf = fopen("/proc/loadavg", "r");
    if (lf) { if (!fgets(load, sizeof(load), lf)) load[0] = 0; fclose(lf); load[strcspn(load, " ")] = 0; }
    printf("# %s, %d threads, KV %s, %d reps (median, best), load %s%s\n", path, rq_threads(), rq_kv_name(kvt), reps, load,
           ck && hybrid ? ", a checkpoint per segment row" : "");
    if (mix) printf("# every seg pass: + %d decode rows of other sequences\n", mix);
    for (int x = 1; x < nctx; x++) printf("# context %d: %s\n", x, var[x]);

    RqTokens text = { 0 }, para = { 0 };
    rq_tokenize(m, "The keeper of the lighthouse rose before dawn, trimmed the wicks, polished the great lens and "
                   "wrote the wind, the swell and the passing ships into the log, as his father had done. ", -1, &para);
    int ncfg = 1 + 2 * nR, NC = nctx * ncfg; // (configuration ci on context x: x * ncfg + ci)
    double *t = malloc((size_t)NC * reps * sizeof(double)), *tt = malloc(reps * sizeof(double));
    const char *pn[32];
    double p0[32], p1[32], *pacc = calloc((size_t)NC * 32, sizeof(double));
    int np = rq_ctx_profile(c, pn, p0, 32);
    for (int li = 0; li < nL; li++) {
        int L = Ls[li] < 1 ? 1 : Ls[li], cap = L + rmax + 1;
        text.n = 0;
        while (text.n < L + rmax) rq_tokens_push(&text, para.v, para.n);
        for (int i = 0; i < nseq; i++) {
            rq_seq_free(sq[i]);
            if (!(sq[i] = rq_seq_new(m, cap, kvt, &e))) die("rq_seq_new", e, NULL);
        }
        if ((e = rq_prefill(c, sq[0], text.v, L, RQ_LOGITS_NONE, NULL, NULL)) != RQ_OK) die("prefill", e, c);
        if (hybrid) {
            if ((e = rq_seq_set_checkpoints(sq[0], rmax + 2)) != RQ_OK) die("rq_seq_set_checkpoints", e, NULL);
            back_to(sq[0], L, 1);
        }
        for (int i = 1; i < nseq; i++) {
            if ((e = rq_seq_copy(sq[i], sq[0], L)) != RQ_OK || rq_seq_len(sq[i]) != L) die("rq_seq_copy", e, NULL);
            if (hybrid) back_to(sq[i], L, 1);
        }
        memset(pacc, 0, (size_t)NC * 32 * sizeof(double));
        for (int r = -1; r < reps; r++) // (rep -1: a warm-up)
            for (int ci = 0; ci < ncfg; ci++)
            for (int xi = 0; xi < nctx; xi++) {
                int x = (xi + (r < 0 ? 0 : r)) % nctx, cx = x * ncfg + ci; // (the contexts' order rotated per rep)
                double t0 = 0, t1 = 0;
                c = cs[x];
                if (np) rq_ctx_profile(c, pn, p0, 32);
                if (ci == 0) { // dec
                    float *lg;
                    t0 = rq_now();
                    if ((e = rq_decode(c, sq[0], text.v[L], &lg)) != RQ_OK) die("rq_decode", e, c);
                    t1 = rq_now();
                    back_to(sq[0], L, hybrid);
                } else if (ci <= nR) { // seg: r rows of sq[0] (+ mix decode rows of sq[1..mix])
                    int R = Rs[ci - 1];
                    if (ck && hybrid && (e = rq_seq_checkpoint_rows(sq[0], L, R)) != RQ_OK) die("rq_seq_checkpoint_rows", e, NULL);
                    sg[0] = (RqSegment) { sq[0], text.v + L, R, RQ_LOGITS_ALL };
                    for (int i = 0; i < mix; i++) {
                        tok[i] = text.v[L];
                        sg[1 + i] = (RqSegment) { sq[1 + i], &tok[i], 1, RQ_LOGITS_LAST };
                    }
                    t0 = rq_now();
                    if ((e = rq_forward_batch(c, sg, 1 + mix, out, NULL)) != RQ_OK) die("rq_forward_batch", e, c);
                    t1 = rq_now();
                    back_to(sq[0], L, hybrid);
                    for (int i = 0; i < mix; i++) back_to(sq[1 + i], L, hybrid);
                } else { // rows: R sequences, one row each
                    int R = Rs[ci - 1 - nR];
                    for (int i = 0; i < R; i++) {
                        tok[i] = text.v[L];
                        sg[i] = (RqSegment) { sq[1 + i], &tok[i], 1, RQ_LOGITS_LAST };
                    }
                    t0 = rq_now();
                    if ((e = rq_forward_batch(c, sg, R, out, NULL)) != RQ_OK) die("rq_forward_batch", e, c);
                    t1 = rq_now();
                    for (int i = 0; i < R; i++) back_to(sq[1 + i], L, hybrid);
                }
                if (r >= 0) t[(size_t)cx * reps + r] = t1 - t0;
                if (np && r >= 0) {
                    rq_ctx_profile(c, pn, p1, 32);
                    for (int k = 0; k < np; k++) pacc[(size_t)cx * 32 + k] += p1[k] - p0[k];
                }
            }
        double med[NCTX * (1 + 2 * RMAX)], best[NCTX * (1 + 2 * RMAX)];
        for (int cx = 0; cx < NC; cx++) {
            memcpy(tt, t + (size_t)cx * reps, reps * sizeof(double));
            qsort(tt, reps, sizeof(double), cmp_d);
            med[cx] = tt[reps / 2] * 1e3;
            best[cx] = tt[0] * 1e3;
        }
        for (int x = 0; x < nctx; x++) {
            const double *md = med + x * ncfg, *bs = best + x * ncfg, *m0 = med;
            if (nctx > 1) printf("# context %d (%s)%s\n", x, var[x], x ? ": seg / rows ms, f, and the ratio to context 0" : "");
            printf("# %5s %4s %9s %9s %7s   %9s %9s %7s%s\n", "L", "r", "seg ms", "best", "f(seg)", "rows ms", "best", "f(rows)",
                   x ? "  seg/ctx0 rows/ctx0" : "");
            printf("  %5d %4s %9.3f %9.3f %7.3f", L, "dec", md[0], bs[0], 1.0);
            if (x) printf("   %9s %9s %7s  %8.3f", "", "", "", md[0] / m0[0]);
            printf("\n");
            for (int i = 0; i < nR; i++) {
                printf("  %5d %4d %9.3f %9.3f %7.3f   %9.3f %9.3f %7.3f", L, Rs[i], md[1 + i], bs[1 + i], md[1 + i] / md[0],
                       md[1 + nR + i], bs[1 + nR + i], md[1 + nR + i] / md[0]);
                if (x) printf("  %8.3f %9.3f", md[1 + i] / m0[1 + i], md[1 + nR + i] / m0[1 + nR + i]);
                printf("\n");
            }
            for (int ci = 1; np && ci < ncfg; ci++) {
                printf("        %s %d profile ms/pass:", ci <= nR ? "seg" : "rows", Rs[(ci - 1) % nR]);
                for (int k = 0; k < np; k++)
                    if (pacc[(size_t)(x * ncfg + ci) * 32 + k] > 0) printf(" %s %.3f", pn[k], pacc[(size_t)(x * ncfg + ci) * 32 + k] / reps * 1e3);
                printf("\n");
            }
        }
        fflush(stdout);
    }
    free(pacc);
    for (int i = 0; i < nseq; i++) rq_seq_free(sq[i]);
    rq_tokens_free(&text);
    rq_tokens_free(&para);
    free(t);
    free(tt);
    free(out);
    free(sq);
    for (int x = 0; x < nctx; x++) rq_ctx_free(cs[x]);
    rq_model_free(m);
    return 0;
usage:
    fprintf(stderr, "usage: bench_verify MODEL [-L 500,2000] [-R 1,2,...,9] [-r REPS] [-k f16|bf16|f32] [-c] [-M MIX] "
                    "[-A \"NAME=VALUE ...\"]\n");
    return 2;
}
