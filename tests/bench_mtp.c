/* MTP drafter costs (the library API, src/runq.h; TODO S9w, docs/qwen38-speculative.md "The MTP drafter (S9)"): what a
 * guess of a model's MTP block (rq_mtp_load: a view of one attention layer + the target's classifier) costs at a
 * context of L positions, on a file with the target's full width (tests/make_synth_wide.py: Qwen3.8-27B's dimensions,
 * a short trunk, its real MTP block and classifier sizes), without the 27B. The sequences are filled with synthetic KV
 * entries (rq_seq_fill_synthetic: the passes' cost does not depend on the values), then per KV type and L:
 *   guess V    one MTP row with its logits (rq_forward_batch, a 1-row segment with h_in: the loop's guesses 2..k) per
 *              view of -V (0: the full vocabulary, else RqMtpOpts.vocab), against rq_pass_cost_default(view) at L (the
 *              cost model's estimate without a calibration: RqModelMem.read)
 *   catch r    a segment of r = a + 1 MTP rows without logits (the catch-up of a round after a accepted guesses) and
 *              the same with the last row's logits (the round's first guess riding in it), first view of -V, a = 0..-A
 *   prompt     a chunk of -P rows of the target without logits (a prompt chunk), the same with every row's hidden state
 *              (RqSegment.hidden, as the lockstep prefill asks), and the MTP's chunk over it (its rows' h_in, no
 *              logits); then extrapolated to a trunk of -X layers (the target's trunk is repeated: the 27B's 64 = 16 x
 *              the 4 layers of 3 linear + 1 attention): the MTP's and the hidden rows' share of a prompt chunk
 * each pass followed by a truncation back to L (the target through its checkpoint at L; not timed). Reps rotate over
 * the configurations; medians and the best in ms. Built by `make bench-mtp` (build/bench_mtp).
 *
 *   build/bench_mtp MODEL [-L 1024,16384,49152,98304] [-V 0,100000,151643] [-k f16,q8] [-r REPS] [-p PROMPT_REPS]
 *                         [-P 512] [-A 6] [-X 64] [-m GiB]
 * Defaults: -r 9, -p 3 (0: no prompt rows), -P 512 (the CLI's and runq serve's prefill chunk), -A 6, -X 64. -m: the
 * loads' memory budget (RqModelOpts.mem / RqMtpOpts.mem; default MemAvailable). Threads: OMP_NUM_THREADS (else the
 * default). */
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
    fprintf(stderr, "bench_mtp: %s: %s %s\n", what, rq_strerror(e), c ? rq_ctx_error(c) : "");
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

#define LMAX 16
#define VMAX 8
#define KMAX 4
#define AMAX 15

int main(int argc, char **argv) {
    int Ls[LMAX] = { 1024, 16384, 49152, 98304 }, nL = 4, Vs[VMAX] = { 0, 100000, 151643 }, nV = 3, kvs[KMAX] = { RQ_KV_F16, RQ_KV_Q8 };
    int nK = 2, reps = 9, preps = 3, P = 512, A = 6, X = 64;
    double mem = 0;
    const char *path = NULL;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') { path = argv[i]; continue; }
        if (i + 1 >= argc) goto usage;
        const char *v = argv[++i];
        switch (argv[i - 1][1]) {
            case 'L': nL = parse_list(v, Ls, LMAX); break;
            case 'V': nV = parse_list(v, Vs, VMAX); break;
            case 'k': {
                nK = 0;
                char b[128];
                snprintf(b, sizeof(b), "%s", v);
                for (char *t = strtok(b, ","); t && nK < KMAX; t = strtok(NULL, ","))
                    if ((kvs[nK++] = rq_kv_parse(t)) < 0) goto usage;
                break;
            }
            case 'r': reps = atoi(v); break;
            case 'p': preps = atoi(v); break;
            case 'P': P = atoi(v); break;
            case 'A': A = atoi(v); break;
            case 'X': X = atoi(v); break;
            case 'm': mem = atof(v); break;
            default: goto usage;
        }
    }
    int lmax = 1;
    for (int i = 0; i < nL; i++) {
        if (Ls[i] < 1) goto usage;
        lmax = Ls[i] > lmax ? Ls[i] : lmax;
    }
    for (int i = 0; i < nV; i++)
        if (Vs[i] < 0) goto usage;
    if (!path || nL < 1 || nV < 1 || nK < 1 || reps < 1 || preps < 0 || P < 1 || P > RQ_BATCH_MAX || A < 0 || A > AMAX || X < 1) goto usage;

    const char *w = rq_threads_init(0);
    if (w) fprintf(stderr, "%s\n", w);
    char err[512];
    int extra = (preps ? P : 0) > A + 1 ? (preps ? P : 0) : A + 1, cap = lmax + extra + 1, e;
    RqModelOpts o = { 0 };
    o.ctx_len = cap;
    o.mem = mem;
    o.page_cache = 1; // (the file's pages: not needed after the load)
    double t0 = rq_now();
    RqModel *m = rq_model_load(path, &o, err, sizeof(err));
    if (!m) { fprintf(stderr, "bench_mtp: %s\n", err); return 1; }
    double tload = rq_now() - t0;
    RqModelInfo I;
    rq_model_info(m, &I);
    int dim = I.dim, hybrid = I.linear_layers > 0;
    RqModel *vw[VMAX];
    RqModelMem vm[VMAX];
    const double M = 1048576.0;
    for (int v = 0; v < nV; v++) {
        RqMtpOpts mo = { Vs[v], mem };
        if (!(vw[v] = rq_mtp_load(m, &mo, err, sizeof(err)))) { fprintf(stderr, "bench_mtp: rq_mtp_load: %s\n", err); return 1; }
        rq_model_mem(vw[v], &vm[v]);
    }
    char load[64] = "";
    FILE *lf = fopen("/proc/loadavg", "r");
    if (lf) { if (!fgets(load, sizeof(load), lf)) load[0] = 0; fclose(lf); load[strcspn(load, "\n")] = 0; }
    printf("# %s: dim %d, %d layers (%d linear), vocab %d; %d threads, capacity %d, load %.1f s; loadavg %s\n", path, dim, I.n_layers,
           I.linear_layers, I.vocab_size, rq_threads(), cap, tload, load);
    for (int v = 0; v < nV; v++)
        printf("# MTP view, vocab %d: weights %.1f MiB, a pass reads %.1f MiB (RqModelMem.read)\n", Vs[v], vm[v].weights / M, vm[v].read / M);
    // the rows: tokens (any valid ids) and the h inputs (any finite values: the cost does not depend on them)
    int *tok = malloc((size_t)extra * sizeof(int));
    float *h = malloc((size_t)extra * dim * sizeof(float)), *hid = malloc((size_t)extra * dim * sizeof(float));
    if (!tok || !h || !hid) die("malloc", RQ_ERR_NOMEM, NULL);
    for (int i = 0; i < extra; i++) tok[i] = (1000 + 7919 * i) % 100000;
    for (size_t i = 0; i < (size_t)extra * dim; i++) h[i] = (float)((int)(i * 2654435761u % 2001) - 1000) * 1e-3f;
    // configurations: per view a guess, the catch-up segments (none / the last row's logits), the prompt's three
    int cfg_catch = nV, ncatch = 2 * (A + 1), cfg_pre = cfg_catch + ncatch, ncfg = cfg_pre + (preps ? 3 : 0);
    int rmax = reps > preps ? reps : preps;
    double *t = malloc((size_t)ncfg * rmax * sizeof(double));
    for (int ki = 0; ki < nK; ki++) {
        int kv = kvs[ki];
        RqCtx *c = preps ? rq_ctx_new(m, P, cap, &e) : NULL;
        if (preps && !c) die("rq_ctx_new", e, NULL);
        RqSeq *s = preps ? rq_seq_new(m, cap, kv, &e) : NULL;
        if (preps && !s) die("rq_seq_new", e, NULL);
        if (s && hybrid && (e = rq_seq_set_checkpoints(s, 2)) != RQ_OK) die("rq_seq_set_checkpoints", e, NULL);
        RqCtx *vc[VMAX];
        RqSeq *vs[VMAX];
        for (int v = 0; v < nV; v++) {
            int nb = v == 0 ? (preps ? P : A + 1) : 8;
            if (!(vc[v] = rq_ctx_new(vw[v], nb > A + 1 ? nb : A + 1, cap, &e))) die("rq_ctx_new (MTP)", e, NULL);
            if (!(vs[v] = rq_seq_new(vw[v], cap, kv, &e))) die("rq_seq_new (MTP)", e, NULL);
        }
        printf("# KV %s\n", rq_kv_name(kv));
        printf("# %6s %-14s %10s %10s %s\n", "L", "pass", "median ms", "best ms", "");
        for (int li = 0; li < nL; li++) {
            int L = Ls[li];
            double tf = rq_now();
            if (s && ((e = rq_seq_fill_synthetic(s, L, 1 + li)) != RQ_OK || (hybrid && (e = rq_seq_checkpoint_at(s, L)) != RQ_OK)))
                die("rq_seq_fill_synthetic", e, NULL);
            for (int v = 0; v < nV; v++)
                if ((e = rq_seq_fill_synthetic(vs[v], L, 101 + li)) != RQ_OK) die("rq_seq_fill_synthetic (MTP)", e, NULL);
            tf = rq_now() - tf;
            int nrep[VMAX + 2 * (AMAX + 1) + 3] = { 0 };
            for (int r = -1; r < rmax; r++) // (rep -1: a warm-up)
                for (int ci = 0; ci < ncfg; ci++) {
                    int pre = ci >= cfg_pre;
                    if ((pre && (r < 0 || r >= preps)) || (!pre && r >= reps)) continue;
                    double a, b;
                    if (ci < cfg_catch) { // a guess: one row with its logits
                        float *lg;
                        RqSegment sg = { vs[ci], tok, 1, RQ_LOGITS_LAST, NULL, h };
                        a = rq_now();
                        if ((e = rq_forward_batch(vc[ci], &sg, 1, NULL, &lg)) != RQ_OK) die("rq_forward_batch (guess)", e, vc[ci]);
                        b = rq_now();
                        back_to(vs[ci], L, 0);
                    } else if (!pre) { // a catch-up segment of a + 1 rows: no logits, or the last row's
                        int k = ci - cfg_catch, rows = k % (A + 1) + 1, last = k > A;
                        float *lg;
                        RqSegment sg = { vs[0], tok, rows, last ? RQ_LOGITS_LAST : RQ_LOGITS_NONE, NULL, h };
                        a = rq_now();
                        if ((e = rq_forward_batch(vc[0], &sg, 1, NULL, &lg)) != RQ_OK) die("rq_forward_batch (catch-up)", e, vc[0]);
                        b = rq_now();
                        back_to(vs[0], L, 0);
                    } else { // the prompt: the target's chunk without / with the hidden rows, the MTP's chunk
                        int k = ci - cfg_pre;
                        RqSegment sg = { k < 2 ? s : vs[0], tok, P, RQ_LOGITS_NONE, k == 1 ? hid : NULL, k == 2 ? h : NULL };
                        a = rq_now();
                        if ((e = rq_forward_batch(k < 2 ? c : vc[0], &sg, 1, NULL, NULL)) != RQ_OK) die("rq_forward_batch (prompt)", e, k < 2 ? c : vc[0]);
                        b = rq_now();
                        if (k < 2) back_to(s, L, hybrid);
                        else back_to(vs[0], L, 0);
                    }
                    if (r >= 0) t[(size_t)ci * rmax + nrep[ci]++] = b - a;
                }
            double md[VMAX + 2 * (AMAX + 1) + 3], bs[VMAX + 2 * (AMAX + 1) + 3];
            for (int ci = 0; ci < ncfg; ci++) md[ci] = median(t + (size_t)ci * rmax, nrep[ci], &bs[ci]) * 1e3, bs[ci] *= 1e3;
            for (int v = 0; v < nV; v++) {
                RqPassCost pc;
                rq_pass_cost_default(vw[v], &pc);
                double est = rq_pass_cost(&pc, 1, L, 0) * 1e3;
                char nm[32];
                snprintf(nm, sizeof(nm), "guess %d", Vs[v]);
                printf("  %6d %-14s %10.2f %10.2f   rq_pass_cost_default %.2f ms (%+.0f%%)\n", L, nm, md[v], bs[v], est,
                       100.0 * (est - md[v]) / md[v]);
            }
            for (int k = 0; k < ncatch; k++) {
                char nm[32];
                snprintf(nm, sizeof(nm), "catch %d%s", k % (A + 1) + 1, k > A ? "+lg" : "");
                printf("  %6d %-14s %10.2f %10.2f\n", L, nm, md[cfg_catch + k], bs[cfg_catch + k]);
            }
            if (preps) {
                double tn = md[cfg_pre], th = md[cfg_pre + 1], tm = md[cfg_pre + 2], sc = (double)X / I.n_layers;
                printf("  %6d %-14s %10.1f %10.1f   (a %d-row chunk, no logits)\n", L, "prompt", tn, bs[cfg_pre], P);
                printf("  %6d %-14s %10.1f %10.1f   (+ every row's hidden state: %+.1f ms)\n", L, "prompt+hidden", th, bs[cfg_pre + 1], th - tn);
                printf("  %6d %-14s %10.1f %10.1f   (the MTP's chunk: %.1f%% of the target's)\n", L, "prompt mtp", tm, bs[cfg_pre + 2],
                       100.0 * tm / tn);
                printf("  %6d %-14s at %d layers (x %.0f): the chunk ~%.0f ms; + hidden %+.2f%%, + the MTP %+.2f%%, both %+.2f%%\n", L,
                       "extrapolated", X, sc, sc * tn, 100.0 * (th - tn) / (sc * tn), 100.0 * tm / (sc * tn), 100.0 * (th - tn + tm) / (sc * tn));
            }
            printf("#         (synthetic fill of %d positions: %.2f s)\n", L, tf);
            fflush(stdout);
        }
        for (int v = 0; v < nV; v++) {
            rq_ctx_free(vc[v]);
            rq_seq_free(vs[v]);
        }
        if (c) rq_ctx_free(c);
        rq_seq_free(s);
    }
    free(t);
    free(tok);
    free(h);
    free(hid);
    for (int v = 0; v < nV; v++) rq_model_free(vw[v]);
    rq_model_free(m);
    return 0;
usage:
    fprintf(stderr, "usage: bench_mtp MODEL [-L 1024,16384,49152,98304] [-V 0,100000,151643] [-k f16,q8] [-r REPS] [-p PROMPT_REPS] "
                    "[-P 512] [-A 6] [-X 64] [-m GiB]\n");
    return 2;
}
