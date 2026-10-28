/* Long-context timing (the library API, src/runq.h; TODO G0 (b)): what a turn of an agentic session costs at a context
 * of L positions, without prefilling L tokens first (hours for Qwen3.8-27B at 96k): the sequence is filled with
 * synthetic KV entries and recurrent state (rq_seq_fill_synthetic; the passes' cost does not depend on the values),
 * then per L:
 *   dec      one decode step (rq_decode, the logits): the token rate at L
 *   seg r    a verify segment of r rows of the sequence, every row's logits (rq_forward_batch, RQ_LOGITS_ALL: a
 *            speculative pass with r - 1 guesses); -c: a checkpoint per row (rq_seq_checkpoint_rows), as rq_spec_run
 *   pre P    the prefill of a chunk of P tokens (rq_prefill, the last row's logits, in passes of -b rows): a turn's
 *            new tokens (tool results, the user's message) at L
 * each pass followed by a truncation back to L (Qwen3.5: through its checkpoint at L; not timed). Reps rotate over
 * the configurations (the shared machine's load hits all alike); medians and the best in ms. Built by `make
 * bench-long` (build/bench_long).
 *
 *   build/bench_long MODEL [-L 16384,32768,65536,98304] [-R 2,4,8] [-P 2560] [-r REPS] [-p PREFILL_REPS] [-c]
 *                          [-k f16] [-b 512] [-m GiB] [-D DRAFT [-a P_MODEL] [-C CALIB_L] [-W MS,F2,F3,... -A SCALE]]
 * Defaults: -r 9 (dec, seg), -p 2 (pre; 0: none), -b 512 (the CLI's and runq serve's prefill chunk). -m: the load's
 * memory budget (RqModelOpts.mem; default MemAvailable). Threads: OMP_NUM_THREADS (else the default).
 * Speculation's choice by position (TODO G7, runq serve's auto policy): with -D, the draft model's decode step at each L
 * too (its sequence filled the same way), then per L the best of a plain step, k = r - 1 model guesses (a segment of r
 * rows + k draft steps) and n = r - 1 lookup guesses (a segment of r rows), for each r of -R, under rq_spec_choose's
 * formula with the measured times at L: tokens per pass 1 + p + .. + p^n, p = -a for the model (default 0.6), a 4-token
 * lookup match's prior (rq_lookup_prior); three cases: no lookup match (model or plain), a 4-token match (all three), the
 * lookup alone (no draft). -C L1 (default 8192, 0: none): the same choice from the cost model runq serve calibrates at its
 * start (rq_pass_cost_measure at L1 positions, linear in the position), to compare. -W / -A: an extrapolation (e.g.
 * synth-27b to Qwen3.8-27B): a pass of r rows at L costs MS x F(r) (F(1) = 1, F2 for 2 rows, ...; the last factor's
 * slope past them) + SCALE x (its measured time at L - at the first -L), the model's own short-context costs plus its
 * attention units' (`-W 118,1.10,1.15,1.25,1.34 -A 16`: the 27B at 8 threads, S7). */
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
    fprintf(stderr, "bench_long: %s: %s %s\n", what, rq_strerror(e), c ? rq_ctx_error(c) : "");
    exit(1);
}

static void back_to(RqSeq *s, int L, int hybrid) {
    int e = rq_seq_truncate(s, L);
    if (e != RQ_OK || rq_seq_len(s) != L) die("rq_seq_truncate", e, NULL);
    if (hybrid && (e = rq_seq_checkpoint_at(s, L)) != RQ_OK) die("rq_seq_checkpoint_at", e, NULL);
}

static double median(double *t, int n, double *best) {
    qsort(t, n, sizeof(double), cmp_d);
    *best = t[0];
    return n % 2 ? t[n / 2] : 0.5 * (t[n / 2 - 1] + t[n / 2]);
}

#define RMAX 16
#define LMAX 16

static double ext_cost(const double *W, int nW, double scale, double t_meas, double t_meas0, int r) {
    // -W / -A: MS x F(r) + SCALE x (the measured time at L - at the first L); W[0] = MS (seconds), W[i] = F(i + 1), past
    // the last factor its slope
    double f = 1;
    if (r > 1 && r <= nW) f = W[r - 1];
    else if (r > nW) {
        double last = nW >= 2 ? W[nW - 1] : 1, prev = nW >= 3 ? W[nW - 2] : 1;
        f = last + (last - prev) * (r - nW);
    }
    return W[0] * f + scale * (t_meas - t_meas0);
}

static const char *choice_text(char *b, size_t n, int src, int k, double gain, double tps) {
    if (src == RQ_SPEC_PLAIN) snprintf(b, n, "plain (%.2f tok/s)", tps);
    else snprintf(b, n, "%s %d (x%.2f, %.2f tok/s)", src == RQ_SPEC_MODEL ? "model k" : "lookup", k, gain, tps);
    return b;
}

int main(int argc, char **argv) {
    int Ls[LMAX] = { 16384, 32768, 65536, 98304 }, nL = 4, Rs[RMAX] = { 2, 4, 8 }, nR = 3;
    int P = 2560, reps = 9, preps = 2, ck = 0, kvt = RQ_KV_DEFAULT, chunk = 512;
    double mem = 0, pm = 0.6, W[RMAX + 1], scale = 1;
    int nW = 0, calib = 8192;
    const char *path = NULL, *dpath = NULL;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') { path = argv[i]; continue; }
        if (!strcmp(argv[i], "-c")) { ck = 1; continue; }
        if (i + 1 >= argc) goto usage;
        const char *v = argv[++i];
        switch (argv[i - 1][1]) {
            case 'L': nL = parse_list(v, Ls, LMAX); break;
            case 'R': nR = parse_list(v, Rs, RMAX); break;
            case 'P': P = atoi(v); break;
            case 'r': reps = atoi(v); break;
            case 'p': preps = atoi(v); break;
            case 'k': kvt = rq_kv_parse(v); break;
            case 'b': chunk = atoi(v); break;
            case 'm': mem = atof(v); break;
            case 'D': dpath = v; break;
            case 'a': pm = atof(v); break;
            case 'C': calib = atoi(v); break;
            case 'A': scale = atof(v); break;
            case 'W': {
                const char *q = v;
                nW = 0;
                while (*q && nW <= RMAX) {
                    W[nW++] = atof(q);
                    const char *c = strchr(q, ',');
                    if (!c) break;
                    q = c + 1;
                }
                if (nW < 1) goto usage;
                W[0] *= 1e-3; // (ms)
                break;
            }
            default: goto usage;
        }
    }
    int rmax = 1, lmax = 1;
    for (int i = 0; i < nR; i++) {
        if (Rs[i] < 1 || Rs[i] > RMAX) goto usage;
        rmax = Rs[i] > rmax ? Rs[i] : rmax;
    }
    for (int i = 0; i < nL; i++) {
        if (Ls[i] < 1) goto usage;
        lmax = Ls[i] > lmax ? Ls[i] : lmax;
    }
    if (!path || reps < 1 || preps < 0 || P < 1 || kvt < 0 || chunk < 1 || chunk > RQ_BATCH_MAX) goto usage;

    const char *w = rq_threads_init(0);
    if (w) fprintf(stderr, "%s\n", w);
    char err[512];
    int extra = (preps ? P : 0) > rmax ? (preps ? P : 0) : rmax, cap = lmax + extra + 1, e;
    RqModelOpts o = { 0 };
    o.ctx_len = cap;
    o.kv_bytes = kvt == RQ_KV_F32 ? 4 : 2;
    o.mem = mem;
    double t0 = rq_now();
    RqModel *m = rq_model_load(path, &o, err, sizeof(err));
    if (!m) { fprintf(stderr, "bench_long: %s\n", err); return 1; }
    double tload = rq_now() - t0;
    RqModelInfo I;
    rq_model_info(m, &I);
    int V = I.vocab_size, hybrid = I.linear_layers > 0;
    RqCtx *c = rq_ctx_new(m, chunk, cap, &e);
    if (!c) die("rq_ctx_new", e, NULL);
    RqSeq *s = rq_seq_new(m, cap, kvt, &e);
    if (!s) die("rq_seq_new", e, NULL);
    if (hybrid && (e = rq_seq_set_checkpoints(s, ck ? rmax + 2 : 2)) != RQ_OK) die("rq_seq_set_checkpoints", e, NULL);
    float *out = malloc((size_t)rmax * V * sizeof(float));
    // -D: the draft model (its decode step at each L), -C: both models' calibrated cost models (the server's)
    RqModel *dm = NULL;
    RqCtx *dc = NULL;
    RqSeq *ds = NULL;
    int dhybrid = 0;
    RqPassCost tcost, dcost;
    if (dpath) {
        RqModelOpts od = o;
        if (!(dm = rq_model_load(dpath, &od, err, sizeof(err)))) { fprintf(stderr, "bench_long: -D: %s\n", err); return 1; }
        RqModelInfo DI;
        rq_model_info(dm, &DI);
        dhybrid = DI.linear_layers > 0;
        if (!(dc = rq_ctx_new(dm, chunk, cap, &e))) die("rq_ctx_new (draft)", e, NULL);
        if (!(ds = rq_seq_new(dm, cap, kvt, &e))) die("rq_seq_new (draft)", e, NULL);
        if (calib > 0) {
            int cl = calib < cap - 32 ? calib : cap - 32;
            double tc = rq_now();
            if ((e = rq_pass_cost_measure(c, s, cl, 3, &tcost)) != RQ_OK) die("rq_pass_cost_measure", e, c);
            if ((e = rq_pass_cost_measure(dc, ds, cl, 3, &dcost)) != RQ_OK) die("rq_pass_cost_measure (draft)", e, dc);
            printf("# the cost models (rq_pass_cost_measure at %d positions, %.1f s): target %.2f ms + %.3f / %.3f us x position "
                   "(a segment's first / further rows), f(2/4/8) %.2f / %.2f / %.2f; draft %.2f ms + %.3f us\n", cl, rq_now() - tc,
                   tcost.t_w * 1e3, tcost.att * 1e6, tcost.att_x * 1e6, rq_pass_cost(&tcost, 2, 0, 0) / tcost.t_w,
                   rq_pass_cost(&tcost, 4, 0, 0) / tcost.t_w, rq_pass_cost(&tcost, 8, 0, 0) / tcost.t_w, dcost.t_w * 1e3, dcost.att * 1e6);
        }
    }
    double base[1 + RMAX + 2] = { 0 };
    int have_base = 0;
    RqTokens text = { 0 }, para = { 0 };
    rq_tokenize(m, "The keeper of the lighthouse rose before dawn, trimmed the wicks, polished the great lens and "
                   "wrote the wind, the swell and the passing ships into the log, as his father had done. ", -1, &para);
    while (text.n < extra + 1) rq_tokens_push(&text, para.v, para.n);
    char load[64] = "";
    FILE *lf = fopen("/proc/loadavg", "r");
    if (lf) { if (!fgets(load, sizeof(load), lf)) load[0] = 0; fclose(lf); load[strcspn(load, "\n")] = 0; }
    printf("# %s, %d threads, KV %s, -b %d, capacity %d, load %.1f s; loadavg %s%s\n", path, rq_threads(), rq_kv_name(kvt),
           chunk, cap, tload, load, ck && hybrid ? "; a checkpoint per segment row" : "");
    printf("# %6s %6s %10s %10s %8s %9s\n", "L", "pass", "median ms", "best ms", "f(dec)", "tok/s");
    int ncfg = 1 + nR + (preps ? 1 : 0) + (dm ? 1 : 0), cfg_d = dm ? ncfg - 1 : -1, cfg_pre = preps ? 1 + nR : -1;
    double *t = malloc((size_t)ncfg * reps * sizeof(double));
    for (int li = 0; li < nL; li++) {
        int L = Ls[li];
        double tf = rq_now();
        if ((e = rq_seq_fill_synthetic(s, L, 1 + li)) != RQ_OK) die("rq_seq_fill_synthetic", e, NULL);
        if (hybrid && (e = rq_seq_checkpoint_at(s, L)) != RQ_OK) die("rq_seq_checkpoint_at", e, NULL);
        if (dm) {
            if ((e = rq_seq_fill_synthetic(ds, L, 101 + li)) != RQ_OK) die("rq_seq_fill_synthetic (draft)", e, NULL);
            if (dhybrid && (e = rq_seq_checkpoint_at(ds, L)) != RQ_OK) die("rq_seq_checkpoint_at (draft)", e, NULL);
        }
        tf = rq_now() - tf;
        int nrep[1 + RMAX + 2] = { 0 };
        for (int r = -1; r < reps; r++) // (rep -1: a warm-up of dec and the segments)
            for (int ci = 0; ci < ncfg; ci++) {
                int pre = ci == cfg_pre;
                if (pre && (r < 0 || r >= preps)) continue;
                double a, b;
                if (ci == cfg_d) { // (the draft's decode step at L)
                    float *lg;
                    a = rq_now();
                    if ((e = rq_decode(dc, ds, text.v[0], &lg)) != RQ_OK) die("rq_decode (draft)", e, dc);
                    b = rq_now();
                    back_to(ds, L, dhybrid);
                    if (r >= 0) t[(size_t)ci * reps + nrep[ci]++] = b - a;
                    continue;
                }
                if (ci == 0) {
                    float *lg;
                    a = rq_now();
                    if ((e = rq_decode(c, s, text.v[0], &lg)) != RQ_OK) die("rq_decode", e, c);
                    b = rq_now();
                } else if (!pre) {
                    int R = Rs[ci - 1];
                    if (ck && hybrid && (e = rq_seq_checkpoint_rows(s, L, R)) != RQ_OK) die("rq_seq_checkpoint_rows", e, NULL);
                    RqSegment sg = { s, text.v, R, RQ_LOGITS_ALL };
                    a = rq_now();
                    if ((e = rq_forward_batch(c, &sg, 1, out, NULL)) != RQ_OK) die("rq_forward_batch", e, c);
                    b = rq_now();
                } else {
                    float *lg;
                    a = rq_now();
                    if ((e = rq_prefill(c, s, text.v, P, RQ_LOGITS_LAST, NULL, &lg)) != RQ_OK) die("rq_prefill", e, c);
                    b = rq_now();
                }
                back_to(s, L, hybrid);
                if (r >= 0) t[(size_t)ci * reps + nrep[ci]++] = b - a;
            }
        double md[1 + RMAX + 2], bs[1 + RMAX + 2];
        for (int ci = 0; ci < ncfg; ci++) md[ci] = median(t + (size_t)ci * reps, nrep[ci], &bs[ci]) * 1e3, bs[ci] *= 1e3;
        printf("  %6d %6s %10.2f %10.2f %8.3f %9.2f\n", L, "dec", md[0], bs[0], 1.0, 1e3 / md[0]);
        for (int i = 0; i < nR; i++)
            printf("  %6d %4s%2d %10.2f %10.2f %8.3f %9.2f\n", L, "seg", Rs[i], md[1 + i], bs[1 + i], md[1 + i] / md[0],
                   Rs[i] * 1e3 / md[1 + i]);
        if (preps)
            printf("  %6d %6s %10.1f %10.1f %8.1f %9.1f   (pre: a %d-token chunk, %d reps)\n", L, "pre", md[cfg_pre],
                   bs[cfg_pre], md[cfg_pre] / md[0], P * 1e3 / md[cfg_pre], P, preps);
        if (dm) {
            // the choice by position (G7): the measured times at L (or -W / -A's extrapolation), then the calibrated model's
            printf("  %6d %6s %10.2f %10.2f %8.3f %9.2f   (the draft's decode step)\n", L, "draft", md[cfg_d], bs[cfg_d],
                   md[cfg_d] / md[0], 1e3 / md[cfg_d]);
            if (!have_base) {
                memcpy(base, md, sizeof(base));
                have_base = 1;
            }
            double T1 = nW ? ext_cost(W, nW, scale, md[0] * 1e-3, base[0] * 1e-3, 1) : md[0] * 1e-3, dd = md[cfg_d] * 1e-3;
            double pl = rq_lookup_prior(4);
            const char *what[3] = { "no lookup match", "a 4-token match", "the lookup alone" };
            for (int sc = 0; sc < 3; sc++) {
                int bsrc = RQ_SPEC_PLAIN, bk = 0;
                double bg = 1, bt = 1 / T1;
                for (int i = 0; i < nR; i++) {
                    int n = Rs[i] - 1;
                    if (n < 1) continue;
                    double Tr = nW ? ext_cost(W, nW, scale, md[1 + i] * 1e-3, base[1 + i] * 1e-3, Rs[i]) : md[1 + i] * 1e-3;
                    for (int src = RQ_SPEC_LOOKUP; src <= RQ_SPEC_MODEL; src++) {
                        if ((src == RQ_SPEC_LOOKUP && sc == 0) || (src == RQ_SPEC_MODEL && sc == 2)) continue;
                        double T = Tr + (src == RQ_SPEC_MODEL ? n * dd : 0);
                        double tps = (1 + rq_spec_expect(src == RQ_SPEC_MODEL ? pm : pl, n)) / T;
                        if (tps * T1 > bg) bg = tps * T1, bt = tps, bsrc = src, bk = n;
                    }
                }
                char b1[96], b2[96] = "";
                choice_text(b1, sizeof(b1), bsrc, bk, bg, bt);
                if (calib > 0 && !nW) {
                    RqSpecChoice x = rq_spec_choose(&tcost, sc == 2 ? NULL : &dcost, L, rmax - 1, pm, sc == 0 ? 0 : rmax - 1, pl);
                    choice_text(b2, sizeof(b2), x.src, x.n, x.gain, x.gain / x.t_plain);
                }
                printf("  %6d %6s %s: measured %s%s%s\n", L, "choice", what[sc], b1, b2[0] ? "; the calibrated model: " : "", b2);
            }
        }
        printf("#         (synthetic fill of %d positions: %.2f s)\n", L, tf);
        fflush(stdout);
    }
    free(t);
    free(out);
    rq_tokens_free(&text);
    rq_tokens_free(&para);
    rq_seq_free(s);
    rq_ctx_free(c);
    rq_seq_free(ds);
    if (dc) rq_ctx_free(dc);
    rq_model_free(dm);
    rq_model_free(m);
    return 0;
usage:
    fprintf(stderr, "usage: bench_long MODEL [-L 16384,32768,65536,98304] [-R 2,4,8] [-P 2560] [-r REPS] [-p PREFILL_REPS] "
                    "[-c] [-k f16|bf16|f32] [-b 512] [-m GiB] [-D DRAFT [-a P_MODEL] [-C CALIB_L] [-W MS,F2,... -A SCALE]]\n");
    return 2;
}
