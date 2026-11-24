// --act256 (RqModelOpts.act256: one activation scale per 256; docs/v4h.md section 10): the Q2_K / i-quant kernels of
// src/iquants.h under it, a bitwise test and a micro-benchmark.
//
// Test (no arguments): for every type (the 27B IQ3_S file's 8, IQ1_S, the non-repacked IQ3_S / IQ3_XXS) the AVX-512 rows
// kernel (iqf_rows: GEMV, verify, batched decode) and the tiled GEMM (iqf_gemm_item) are bit-identical to the
// definition iqf_dot, for 1 .. 77 activation rows, k up to 17408, activations at +-127, the VBMI and the gather decodes.
// Benchmark: build/flat_bench -b [NB [D [K [REPS]]]] (OMP_NUM_THREADS threads): today's iq_rows / tiled GEMM (scales
// per 32) vs the --act256 ones per type, and the total weighted by the types' shares of ISTA's Qwen3.8-27B IQ3_S file
// (88% of its weights; its Q4_K 10.3% and Q6_K 1.6% are K-quants, not covered). Build: make flat-bench.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
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

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "-b")) {
        int nb = argc > 2 ? atoi(argv[2]) : 512, d = argc > 3 ? atoi(argv[3]) : 17408, k = argc > 4 ? atoi(argv[4]) : 5120;
        bench(nb, d, k, argc > 5 ? atoi(argv[5]) : 5);
        return 0;
    }
    int fails = 0;
    const int nbs[] = { 1, 2, 3, 7, 8, 9, 13, 32, 41, 77 };
    for (int i = 0; i < NTYPES; i++) {
        for (size_t j = 0; j < sizeof(nbs) / sizeof(nbs[0]); j++) fails += test(types[i], 48, 768, nbs[j], 0);
        fails += test(types[i], 80, 1024, 77, 0);
        fails += test(types[i], 32, 17408, 9, 0);
        fails += test(types[i], 48, 2048, 13, 1);
    }
    printf(fails ? "flat: %d FAILED\n" : "flat: every type, rows and tiled, bit-identical to iqf_dot\n", fails);
    return fails != 0;
}
