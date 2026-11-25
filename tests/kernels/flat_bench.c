// --act256 (RqModelOpts.act256: one activation scale per 256; docs/v4h.md sections 10-13): the Q2_K / i-quant kernels of
// src/iquants.h and the K-quant (Q3_K .. Q6_K) ones of src/kernels_avx512.h under it, bitwise tests and benchmarks.
//
// Test (no arguments): for every i-quant type (the 27B IQ3_S file's 8, IQ1_S, the non-repacked IQ3_S / IQ3_XXS) the
// AVX-512 rows kernel (iqf_rows: GEMV, verify, batched decode) and the tiled GEMM (iqf_gemm_item) are bit-identical to
// the definition iqf_dot, for 1 .. 77 activation rows, k up to 17408, activations at +-127, the VBMI and the gather
// decodes; for every K-quant type the flat rows (kqf512_rows) and tiled (kqf512_gemm_item) kernels are bit-identical to kqf_dot
// (the same row counts and sizes, plus weights and activations at their extremes), and kqf_dot agrees with today's
// kq_dot up to float rounding.
// Benchmarks (OMP_NUM_THREADS threads): build/flat_bench -b [NB [D [K [REPS]]]]: today's iq_rows / tiled GEMM (scales per
// 32) vs the --act256 ones per type, weighted by the types' shares of ISTA's Qwen3.8-27B IQ3_S file (88% of its weights);
// build/flat_bench -kb [NB [D [K [REPS]]]]: the K-quants, today's GEMV / small-batch / tiled kernels vs the flat ones,
// weighted by the 27B Q3_K_M and Q4_K_M files' mixes, the 0.8B Q4_K_M's and ISTA's K-quant part. Build: make flat-bench.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "../../src/kernels_avx512.h"
#include "../../src/iquants.h"

#ifndef GWENEDEG_IQX
#error "needs the AVX-512 VNNI build"
#endif
int rq_iq_gather = 0;

static uint32_t rs = 12345;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }
static uint16_t rnd_h(int sign) { return (uint16_t)(((rnd() % 7) + 8) << 10 | (rnd() & 0x3ff) | (sign ? (rnd() & 1) << 15 : 0)); }

static void rand_blocks(int type, uint8_t *b, size_t nblk) { // (as iq_gemm_test: finite fp16 scales)
    int bb = iq_block_bytes(type);
    for (size_t i = 0; i < nblk * bb; i++) b[i] = (uint8_t)rnd();
    for (size_t i = 0; i < nblk; i++) {
        uint8_t *p = b + i * bb;
        uint16_t h = rnd_h(1);
        if (type == GGML_Q2_K) {
            memcpy(p + 80, &h, 2);
            h = rnd_h(0);
            memcpy(p + 82, &h, 2);
        } else if (type == GGML_IQ1_M) {
            uint16_t sc[4];
            memcpy(sc, p + 48, 8);
            for (int j = 0; j < 4; j++) sc[j] = (uint16_t)((sc[j] & 0x0fff) | (h >> 4 * j & 15) << 12);
            memcpy(p + 48, sc, 8);
        } else memcpy(p, &h, 2);
    }
}

static uint8_t *make_w(int type, int d, int k) {
    size_t nblk = (size_t)d * (k / QK_K);
    int base = iq_base(type);
    uint8_t *src = malloc(nblk * iq_block_bytes(base)), *kp = aligned_alloc(64, (iq_size(type, d, k) + 63) / 64 * 64);
    rand_blocks(base, src, nblk);
    iq_pack_blocks(type, kp, src, d / KQ_ROWS, k);
    free(src);
    return kp;
}

typedef struct {
    int nb, k;
    int8_t *xq;   // nb x k
    float *xs;    // nb x k / 32 (today's format)
    float *xsf;   // nb x k / 32: --act256's, one scale per 256 written to its 8 groups (forward.c quantize256)
    int32_t *c16; // nb x k / 16
} Act;

static void make_act(Act *a, int nb, int k, int extreme) {
    a->nb = nb; a->k = k;
    a->xq = malloc((size_t)nb * k);
    a->xs = malloc(sizeof(float) * nb * k / 32);
    a->xsf = malloc(sizeof(float) * nb * k / 32);
    a->c16 = malloc(sizeof(int32_t) * ((size_t)nb * k / 16 + 1));
    for (size_t i = 0; i < (size_t)nb * k; i++) a->xq[i] = extreme ? (i / k & 1 ? 127 : -127) : (int8_t)(rnd() % 255 - 127);
    for (size_t i = 0; i < (size_t)nb * k / 32; i++) a->xs[i] = (float)(rnd() % 10000 + 1) * 1e-5f;
    for (size_t i = 0; i < (size_t)nb * k / 32; i += 8) {
        float A = (float)(rnd() % 10000 + 1) * 1e-5f;
        for (int j = 0; j < 8; j++) a->xsf[i + j] = A;
    }
    for (int b = 0; b < nb; b++) kq512_c16(a->c16 + (size_t)b * (k / 16), a->xq + (size_t)b * k, k);
}
static void free_act(Act *a) { free(a->xq); free(a->xs); free(a->xsf); free(a->c16); }

// ---------------------------------------------------------------------------------------------------------------------
// drivers (as the forward pass: rows over 16-row blocks, or (32-row tile, chunk) items with the tile's chunks together)

static void run_rows(int flat, int type, float *y, int d, const uint8_t *kp, const Act *a) {
    int nblk = d / KQ_ROWS, nt = omp_get_max_threads(), ch = nblk / (8 * nt) > 1 ? nblk / (8 * nt) : 1, k = a->k;
    #pragma omp parallel for schedule(dynamic, ch)
    for (int rb = 0; rb < nblk; rb++) {
        (flat ? iqf_rows : iq_rows)(type, y, d, a->xq, k, flat ? a->xsf : a->xs, a->c16, a->nb, kp, k, rb, rb + 1);
    }
}

static void run_gemm(int flat, int type, float *y, int d, const uint8_t *kp, const Act *a, uint8_t *scr, size_t sb) {
    int nt = omp_get_max_threads(), ntile = (d / 16 + 1) / 2, nb = a->nb, k = a->k;
    int nbi = iqx_nb(nb, ntile, nt), nchunk = (nb + nbi - 1) / nbi;
    #pragma omp parallel
    {
        uint8_t *s = scr + (size_t)omp_get_thread_num() * sb;
        const void *last = NULL;
        #pragma omp for schedule(dynamic, 1)
        for (int t = 0; t < ntile; t++)
            for (int c = 0; c < nchunk; c++) {
                (flat ? iqf_gemm_item : iqx_gemm_item)(y, d, kp, type, d, t, c, nbi, a->xq, flat ? a->xsf : a->xs, k, nb, k,
                                                       a->c16, s, &last);
            }
    }
}

// ---------------------------------------------------------------------------------------------------------------------

// the types of ISTA's Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf (% of its weights; IQ3_S / IQ3_XXS as --repack makes them)
static const int types[] = { GGML_IQ3_S_R, GGML_IQ4_XS, GGML_IQ3_XXS_R, GGML_IQ2_S, GGML_IQ2_XS, GGML_Q2_K, GGML_IQ2_XXS,
                             GGML_IQ1_M, GGML_IQ1_S, GGML_IQ3_S, GGML_IQ3_XXS };
static const double share[] = { 31.0, 20.8, 19.8, 9.1, 2.8, 2.7, 1.4, 0.3, 0, 0, 0 };
#define NTYPES ((int)(sizeof(types) / sizeof(types[0])))

static int test(int type, int d, int k, int nb, int extreme) {
    uint8_t *kp = make_w(type, d, k);
    Act a;
    make_act(&a, nb, k, extreme);
    size_t n = (size_t)nb * d, sb = (IQX_SCRATCH(k) + 63) / 64 * 64;
    float *y0 = malloc(4 * n), *y1 = malloc(4 * n);
    uint8_t *scr = aligned_alloc(64, sb * omp_get_max_threads());
    int nsb = k / QK_K, fail = 0;
    for (int rb = 0; rb < d / KQ_ROWS; rb++)
        for (int b = 0; b < nb; b += KQ_NC)
            iqf_dot(type, y0 + (size_t)b * d + rb * KQ_ROWS, d, kp + (size_t)rb * nsb * iq_sb_bytes(type), k,
                    nb - b < KQ_NC ? nb - b : KQ_NC, a.xq + (size_t)b * k, k, a.xsf + (size_t)b * (k / 32), k / 32);
    for (int gth = 0; gth < 2; gth++) {
        rq_iq_gather = gth;
        for (int path = 0; path < 2; path++) {
            memset(y1, 0xff, 4 * n);
            if (path == 0) run_rows(1, type, y1, d, kp, &a);
            else run_gemm(1, type, y1, d, kp, &a, scr, sb);
            if (memcmp(y0, y1, 4 * n)) {
                size_t i = 0;
                while (i < n && !memcmp(y0 + i, y1 + i, 4)) i++;
                printf("FAIL %-9s d %d k %d nb %d%s: flat %s (%s) [%zu] %.9g != %.9g\n", iq_name(type), d, k, nb,
                       extreme ? " extreme" : "", path ? "tiled" : "rows", gth ? "gather" : "VBMI", i, y1[i], y0[i]);
                fail = 1;
            }
        }
    }
    rq_iq_gather = 0;
    if (!fail) printf("ok   %-9s d %4d k %5d nb %2d%s\n", iq_name(type), d, k, nb, extreme ? " (activations +-127)" : "");
    free(kp); free(y0); free(y1); free(scr); free_act(&a);
    return fail;
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void bench(int nb, int d, int k, int reps) {
    int nt = omp_get_max_threads();
    size_t sb = (IQX_SCRATCH(k) + 63) / 64 * 64;
    uint8_t *scr = aligned_alloc(64, sb * nt);
    float *y = malloc(sizeof(float) * nb * d);
    Act a;
    make_act(&a, nb, k, 0);
    double wn = (double)d * k * nb / nt, sw = 0, wbase = 0, wflat = 0, wbase_p = 0, wflat_p = 0;
    printf("%5d x %5d nb %4d %2d thr: ns per weight and row x threads, best of %d (today -> flat; prod = the path\n"
           "                             the forward pass takes today: rows below IQX_MIN = %d rows, else tiled)\n",
           d, k, nb, nt, reps, IQX_MIN);
    printf("  %-9s %6s | %-9s %-9s | %-9s %-9s | %-12s %-12s\n", "type", "share", "rows", "tiled", "rows flat", "tiled flat",
           "best: ratio", "prod: ratio");
    for (int i = 0; i < NTYPES; i++) {
        if (!share[i]) continue;
        int type = types[i];
        uint8_t *kp = make_w(type, d, k);
        double best[4] = { 1e9, 1e9, 1e9, 1e9 }; // rows, tiled, rows flat, tiled flat
        for (int tr = 0; tr < reps; tr++)
            for (int v = 0; v < 4; v++) {
                if ((v & 1) && nb < 4) continue;
                double t0 = now();
                if (v & 1) run_gemm(v >> 1, type, y, d, kp, &a, scr, sb);
                else run_rows(v >> 1, type, y, d, kp, &a);
                double t = now() - t0;
                if (t < best[v]) best[v] = t;
            }
        double b0 = best[0] < best[1] ? best[0] : best[1], b1 = best[2] < best[3] ? best[2] : best[3];
        double p0 = nb < IQX_MIN ? best[0] : best[1], p1 = nb < IQX_MIN ? best[2] : best[3];
        printf("  %-9s %5.1f%% | %.5f   %.5f   | %.5f   %.5f   | x%.2f        x%.2f\n", iq_name(type), share[i],
               1e9 * best[0] / wn, best[1] < 1e9 ? 1e9 * best[1] / wn : 0, 1e9 * best[2] / wn,
               best[3] < 1e9 ? 1e9 * best[3] / wn : 0, b0 / b1, p0 / p1);
        sw += share[i];
        wbase += share[i] * b0; wflat += share[i] * b1;
        wbase_p += share[i] * p0; wflat_p += share[i] * p1;
        fflush(stdout);
        free(kp);
    }
    printf("  weighted by the shares (%.1f%% of the file's weights): best paths x%.3f, today's path choice x%.3f\n",
           sw, wbase / wflat, wbase_p / wflat_p);
    free(scr); free(y); free_act(&a);
}

// =====================================================================================================================
// K-quants under --act256 (docs/v4h.md sections 12-13): the definition kqf_dot (src/kquants.h) and the AVX-512 kernels
// kqf512_rows / kqf512_gemm_item (src/kernels_avx512.h), tested and timed against today's (kq512_gemv_rows,
// smb512_rows, kq512_gemm_item).

// ---------------------------------------------------------------------------------------------------- test, benchmark

static const int ktypes[] = { GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q3_K };
static const char *ktname[] = { "Q4_K", "Q5_K", "Q6_K", "Q3_K" };
#define NKTYPES 4
// % of the matmul weights (2-D tensors but token_embd; gguf header, tensor types: the 27B files at unsloth's commit
// 4a03c64, read without downloading them; ISTA's file: its K-quant part only, its i-quants are section 10's)
static const char *kmix_name[] = { "27B Q3_K_M", "27B Q4_K_M", "0.8B Q4_K_M", "ISTA (K part)" };
static const double kmix[][NKTYPES] = { { 38.3, 1.8, 4.9, 54.7 }, { 72.7, 5.8, 21.2, 0 }, { 60.2, 30.3, 9.3, 0 },
                                        { 10.8, 0, 1.6, 0 } };
#define NKMIX 4

static uint8_t *make_kw(int type, int d, int k, int extreme) {
    // a random (d, k) matrix of the type, packed (kq_pack_blocks); extreme: every value and scale at its largest
    // magnitude (Q4_K sc = m = 63, q = 15; Q5_K q = 31; Q6_K q - 32 = -32, scale -128; Q3_K q = -4, sc - 32 = -32)
    size_t nblk = (size_t)d * (k / QK_K);
    int bb = kq_block_bytes(type);
    uint8_t *src = malloc(nblk * bb), *kp = aligned_alloc(64, (kq_size(type, d, k) + 63) / 64 * 64);
    for (size_t i = 0; i < nblk * bb; i++) src[i] = (uint8_t)rnd();
    for (size_t i = 0; i < nblk; i++) {
        uint8_t *p = src + i * bb;
        uint16_t h = rnd_h(1), h2 = rnd_h(0);
        if (extreme) {
            if (type == GGML_Q4_K || type == GGML_Q5_K) memset(p, 0xff, bb);
            else if (type == GGML_Q6_K) { memset(p, 0, 192); memset(p + 192, 0x80, 16); }
            else memset(p, 0, 108);
        }
        if (type == GGML_Q4_K || type == GGML_Q5_K) { memcpy(p, &h, 2); memcpy(p + 2, &h2, 2); }
        else memcpy(p + (type == GGML_Q6_K ? 208 : 108), &h, 2);
    }
    kq_pack_blocks(type, kp, src, d / KQ_ROWS, k);
    free(src);
    return kp;
}

static void krun_rows(int flat, int type, float *y, int d, const uint8_t *kp, const Act *a, const float *xf) {
    // today: the GEMV for 1 row (matvec_rows), else the small-batch kernel (smb_rows, forward.c's SMB_PF prefetch)
    int nblk = d / KQ_ROWS, nt = omp_get_max_threads(), ch = nblk / (8 * nt) > 1 ? nblk / (8 * nt) : 1, k = a->k, nb = a->nb;
    #pragma omp parallel for schedule(dynamic, ch)
    for (int rb = 0; rb < nblk; rb++) {
        if (flat) kqf512_rows(type, y, d, a->xq, k, a->xsf, a->c16, nb, kp, k, rb, rb + 1, 4096);
        else if (nb == 1) kq512_gemv_rows(type, y, a->xq, a->xs, xf, a->c16, kp, k, rb, rb + 1);
        else smb512_rows(type, y, d, a->xq, k, a->xs, NULL, xf, a->c16, nb, kp, k, 32, rb, rb + 1, 4096);
    }
}

static void krun_gemm(int flat, int type, float *y, int d, const uint8_t *kp, const Act *a, const float *xf, uint8_t *scr,
                      size_t sb) {
    int nt = omp_get_max_threads(), ntile = (d / 16 + 1) / 2, nb = a->nb, k = a->k;
    int nbi = kqx_nb(nb, ntile, nt), nchunk = (nb + nbi - 1) / nbi;
    #pragma omp parallel
    {
        uint8_t *s = scr + (size_t)omp_get_thread_num() * sb;
        const void *last = NULL;
        #pragma omp for schedule(dynamic, 1)
        for (int t = 0; t < ntile; t++)
            for (int c = 0; c < nchunk; c++) {
                if (flat) kqf512_gemm_item(y, d, kp, type, d, t, c, nbi, a->xq, a->xsf, k, nb, k, a->c16, s, &last);
                else kq512_gemm_item(y, d, kp, type, d, t, c, nbi, a->xq, a->xs, k, nb, k, xf, a->c16, s, &last);
            }
    }
}

static float *make_xf(const Act *a, const float *xs) {
    // kq_act_prep of every row (today's Q4_K / Q5_K group sums times the group scales)
    float *xf = malloc(sizeof(float) * a->nb * (a->k / 32));
    for (int b = 0; b < a->nb; b++)
        kq_act_prep(xf + (size_t)b * (a->k / 32), a->xq + (size_t)b * a->k, xs + (size_t)b * (a->k / 32), a->k);
    return xf;
}

static int ktest(int ti, int d, int k, int nb, int extreme, double *maxrel) {
    int type = ktypes[ti];
    uint8_t *kp = make_kw(type, d, k, extreme);
    Act a;
    make_act(&a, nb, k, extreme);
    size_t n = (size_t)nb * d, sb = (KQ512_SCRATCH(k) + 63) / 64 * 64;
    float *y0 = malloc(4 * n), *y1 = malloc(4 * n), *y2 = malloc(4 * n), *xff = make_xf(&a, a.xsf);
    uint8_t *scr = aligned_alloc(64, sb * omp_get_max_threads());
    int nsb = k / QK_K, sbb = kq_sb_bytes(type), fail = 0;
    for (int rb = 0; rb < d / KQ_ROWS; rb++)
        for (int b = 0; b < nb; b += KQ_NC) {
            int nc = nb - b < KQ_NC ? nb - b : KQ_NC;
            const uint8_t *wb = kp + (size_t)rb * nsb * sbb;
            kqf_dot(type, y0 + (size_t)b * d + rb * KQ_ROWS, d, wb, k, nc, a.xq + (size_t)b * k, k, a.xsf + (size_t)b * (k / 32), k / 32);
            // today's definition on the same activations (A in every group): equal up to the float rounding
            kq_dot(type, y2 + (size_t)b * d + rb * KQ_ROWS, d, wb, k, nc, a.xq + (size_t)b * k, k, a.xsf + (size_t)b * (k / 32),
                   k / 32, xff + (size_t)b * (k / 32), k / 32);
        }
    double ym = 0, em = 0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs((double)y2[i] - y0[i]);
        if (fabs(y0[i]) > ym) ym = fabs(y0[i]);
        if (e > em) em = e;
    }
    double rel = ym > 0 ? em / ym : 0;
    if (rel > *maxrel) *maxrel = rel;
    if (rel > 1e-5) {
        printf("FAIL %-5s d %d k %d nb %d%s: flat definition vs today's: max error %.3g of max |y| %.3g\n", ktname[ti], d, k,
               nb, extreme ? " extreme" : "", em, ym);
        fail = 1;
    }
    for (int path = 0; path < 2; path++) {
        memset(y1, 0xff, 4 * n);
        if (path == 0) krun_rows(1, type, y1, d, kp, &a, NULL);
        else krun_gemm(1, type, y1, d, kp, &a, NULL, scr, sb);
        if (memcmp(y0, y1, 4 * n)) {
            size_t i = 0;
            while (i < n && !memcmp(y0 + i, y1 + i, 4)) i++;
            printf("FAIL %-5s d %d k %d nb %d%s: flat %s [%zu] %.9g != %.9g\n", ktname[ti], d, k, nb, extreme ? " extreme" : "",
                   path ? "tiled" : "rows", i, y1[i], y0[i]);
            fail = 1;
        }
    }
    if (!fail)
        printf("ok   %-5s d %4d k %5d nb %2d%s (vs today's definition: %.1e)\n", ktname[ti], d, k, nb,
               extreme ? " (extremes)" : "", rel);
    free(kp); free(y0); free(y1); free(y2); free(xff); free(scr); free_act(&a);
    return fail;
}

static void kbench(int nb, int d, int k, int reps) {
    int nt = omp_get_max_threads();
    size_t sb = (KQ512_SCRATCH(k) + 63) / 64 * 64;
    uint8_t *scr = aligned_alloc(64, sb * nt);
    float *y = malloc(sizeof(float) * nb * d);
    Act a;
    make_act(&a, nb, k, 0);
    float *xf = make_xf(&a, a.xs);
    double wn = (double)d * k * nb / nt, best[NKTYPES][4];
    int smb_max = 32; // (forward.c smb_config: the K-quants' small-batch kernels up to 32 rows, the tiled GEMM above)
    printf("%5d x %5d nb %4d %2d thr: K-quants, ns per weight and row x threads, best of %d (today -> flat; prod = the\n"
           "                             path the forward pass takes: 1 row the GEMV, up to %d the small-batch kernel, else tiled)\n",
           d, k, nb, nt, reps, smb_max);
    printf("  %-5s | %-9s %-9s | %-9s %-9s | %-12s %-12s\n", "type", "rows", "tiled", "rows flat", "tiled flat", "best: ratio",
           "prod: ratio");
    for (int i = 0; i < NKTYPES; i++) {
        int type = ktypes[i];
        uint8_t *kp = make_kw(type, d, k, 0);
        for (int v = 0; v < 4; v++) best[i][v] = 1e9;
        for (int tr = 0; tr < reps; tr++)
            for (int v = 0; v < 4; v++) {
                if ((v & 1) && nb < 4) continue;
                double t0 = now();
                if (v & 1) krun_gemm(v >> 1, type, y, d, kp, &a, xf, scr, sb);
                else krun_rows(v >> 1, type, y, d, kp, &a, xf);
                double t = now() - t0;
                if (t < best[i][v]) best[i][v] = t;
            }
        double *bt = best[i];
        double b0 = bt[0] < bt[1] ? bt[0] : bt[1], b1 = bt[2] < bt[3] ? bt[2] : bt[3];
        double p0 = nb <= smb_max ? bt[0] : bt[1], p1 = nb <= smb_max ? bt[2] : bt[3];
        printf("  %-5s | %.5f   %.5f   | %.5f   %.5f   | x%.2f        x%.2f\n", ktname[i], 1e9 * bt[0] / wn,
               bt[1] < 1e9 ? 1e9 * bt[1] / wn : 0, 1e9 * bt[2] / wn, bt[3] < 1e9 ? 1e9 * bt[3] / wn : 0, b0 / b1, p0 / p1);
        fflush(stdout);
        free(kp);
    }
    for (int m = 0; m < NKMIX; m++) {
        double wb = 0, wf = 0, wbp = 0, wfp = 0, wt = 0, wtl = 0, wfl = 0;
        for (int i = 0; i < NKTYPES; i++) {
            double *bt = best[i], sh = kmix[m][i];
            wb += sh * (bt[0] < bt[1] ? bt[0] : bt[1]);
            wf += sh * (bt[2] < bt[3] ? bt[2] : bt[3]);
            wbp += sh * (nb <= smb_max ? bt[0] : bt[1]);
            wfp += sh * (nb <= smb_max ? bt[2] : bt[3]);
            if (nb >= 4) { wtl += sh * bt[1]; wfl += sh * bt[3]; }
            wt += sh;
        }
        printf("  weighted, %-13s (%5.1f%%): best paths x%.3f, prod x%.3f", kmix_name[m], wt, wb / wf, wbp / wfp);
        if (nb >= 4) printf(", tiled vs tiled x%.3f", wtl / wfl);
        printf("\n");
    }
    free(scr); free(y); free(xf); free_act(&a);
}

int main(int argc, char **argv) {
    if (argc > 1 && (!strcmp(argv[1], "-b") || !strcmp(argv[1], "-kb"))) {
        int nb = argc > 2 ? atoi(argv[2]) : 512, d = argc > 3 ? atoi(argv[3]) : 17408, k = argc > 4 ? atoi(argv[4]) : 5120;
        int reps = argc > 5 ? atoi(argv[5]) : 5;
        if (argv[1][1] == 'k') kbench(nb, d, k, reps);
        else bench(nb, d, k, reps);
        return 0;
    }
    int fails = 0, kfails = 0;
    const int nbs[] = { 1, 2, 3, 7, 8, 9, 13, 32, 41, 77 };
    for (int i = 0; i < NTYPES; i++) {
        for (size_t j = 0; j < sizeof(nbs) / sizeof(nbs[0]); j++) fails += test(types[i], 48, 768, nbs[j], 0);
        fails += test(types[i], 80, 1024, 77, 0);
        fails += test(types[i], 32, 17408, 9, 0);
        fails += test(types[i], 48, 2048, 13, 1);
    }
    printf(fails ? "flat: %d FAILED\n" : "flat: every i-quant type, rows and tiled, bit-identical to iqf_dot\n", fails);
    double maxrel = 0;
    for (int i = 0; i < NKTYPES; i++) {
        for (size_t j = 0; j < sizeof(nbs) / sizeof(nbs[0]); j++) kfails += ktest(i, 48, 768, nbs[j], 0, &maxrel);
        kfails += ktest(i, 80, 1024, 77, 0, &maxrel);
        kfails += ktest(i, 32, 17408, 9, 0, &maxrel);
        kfails += ktest(i, 48, 2048, 13, 1, &maxrel);
        kfails += ktest(i, 16, 2048, 1, 1, &maxrel);
    }
    if (kfails) printf("flat K-quants: %d FAILED\n", kfails);
    else printf("flat K-quants: every type, rows and tiled, bit-identical to kqf_dot (vs today's definition: <= %.1e)\n", maxrel);
    return fails + kfails != 0;
}
