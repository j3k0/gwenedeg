// Prefill GEMM of Q2_K and the i-quants on the AVX-512 build (src/iquants.h): a unit test and a micro-benchmark.
// Test (no arguments): on random blocks of every type (and --repack's IQ3_XXS_R / IQ3_S_R), the batched path the forward
// pass takes (iq_gemm: forward.c's gemm_item for these types) gives bit-identical outputs to iq_rows and to the
// generic definition iq_dot, for 1 .. 77 activation rows, odd 16-row block counts and K up to 17408, with the VBMI and
// the gather decodes.
// Benchmark: iq_gemm_test -b [TYPES [NB [D [K]]]] (OMP_NUM_THREADS threads): a 27B-shaped matrix (default D 17408 x
// K 5120: the ffn_gate / ffn_up of Qwen3.8-27B, 38-49 MB per type, i.e. DRAM-resident, not L2 / L3), NB activation rows
// (default 512: the prefill chunk), iq_rows over 16-row blocks (the path before the tiled GEMM) vs the tiled GEMM, best
// of 5 runs; prints ms per call and ns per weight and activation row (wall time x threads). TYPES: comma-separated names
// (IQ3_S,IQ3_XXS,...) or "all".
// Build: make iq-gemm-test
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>
#include "../../src/kernels_avx512.h"
#include "../../src/iquants.h"

#ifndef RUNQ_IQX
#error "needs the AVX-512 VNNI build"
#endif
int rq_iq_gather = 0;

static uint32_t rs = 12345;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }
static uint16_t rnd_h(int sign) { return (uint16_t)(((rnd() % 7) + 8) << 10 | (rnd() & 0x3ff) | (sign ? (rnd() & 1) << 15 : 0)); }

static void rand_blocks(int type, uint8_t *b, size_t nblk) {
    int bb = iq_block_bytes(type);
    for (size_t i = 0; i < nblk * bb; i++) b[i] = (uint8_t)rnd();
    for (size_t i = 0; i < nblk; i++) { // finite fp16 scales: exponent 8..14
        uint8_t *p = b + i * bb;
        uint16_t h = rnd_h(1);
        if (type == GGML_Q2_K) {
            memcpy(p + 80, &h, 2);
            h = rnd_h(0);
            memcpy(p + 82, &h, 2);
        } else if (type == GGML_IQ1_M) { // d in the top nibbles of the 4 scale words at +48
            uint16_t sc[4];
            memcpy(sc, p + 48, 8);
            for (int j = 0; j < 4; j++) sc[j] = (uint16_t)((sc[j] & 0x0fff) | (h >> 4 * j & 15) << 12);
            memcpy(p + 48, sc, 8);
        } else memcpy(p, &h, 2);
    }
}

static void rand_act(int8_t *xq, float *xs, int32_t *c16, int nb, int k) {
    for (size_t i = 0; i < (size_t)nb * k; i++) xq[i] = (int8_t)(rnd() % 255 - 127);
    for (size_t i = 0; i < (size_t)nb * k / 32; i++) xs[i] = (float)(rnd() % 10000 + 1) * 1e-5f;
    for (int b = 0; b < nb; b++) kq512_c16(c16 + (size_t)b * (k / 16), xq + (size_t)b * k, k);
}

static const int types[] = { GGML_Q2_K, GGML_IQ2_XXS, GGML_IQ2_XS, GGML_IQ2_S, GGML_IQ3_XXS, GGML_IQ3_S,
                             GGML_IQ1_S, GGML_IQ1_M, GGML_IQ4_XS, GGML_IQ3_XXS_R, GGML_IQ3_S_R };
#define NTYPES ((int)(sizeof(types) / sizeof(types[0])))

static uint8_t *make_w(int type, int d, int k) {
    // d x k random weights of type (IQ3_*_R: random blocks of the base type, repacked)
    size_t nblk = (size_t)d * (k / QK_K);
    int base = iq_base(type);
    uint8_t *src = malloc(nblk * iq_block_bytes(base)), *kp = aligned_alloc(64, (iq_size(type, d, k) + 63) / 64 * 64);
    rand_blocks(base, src, nblk);
    iq_pack_blocks(type, kp, src, d / KQ_ROWS, k);
    free(src);
    return kp;
}

// the forward pass's batched path for these types (forward.c gemm_plan / gemm_item), one thread's work items in turn
static void run_rows(int type, float *y, int d, const int8_t *xq, int k, const float *xs, const int32_t *c16, int nb,
                     const uint8_t *kp) {
    // before the tiled GEMM: 16-row blocks over all rows (iq_rows), claimed a few at a time
    int nblk = d / KQ_ROWS, nt = omp_get_max_threads();
    int ch = nblk / (8 * nt) > 1 ? nblk / (8 * nt) : 1;
    #pragma omp parallel for schedule(dynamic, ch)
    for (int rb = 0; rb < nblk; rb++)
        iq_rows(type, y, d, xq, k, xs, c16, nb, kp, k, rb, rb + 1);
}

#ifdef IQX_SCRATCH
static void run_gemm(int type, float *y, int d, const int8_t *xq, int k, const float *xs, const int32_t *c16, int nb,
                     const uint8_t *kp, uint8_t *scr, size_t scr_bytes) {
    // the tiled GEMM: (32-row tile, chunk of nbi rows) items, a tile's chunks claimed together, the tile decoded
    // once into the thread's scratch (iqx_gemm_item)
    int nt = omp_get_max_threads(), ntile = (d / 16 + 1) / 2;
    int nbi = iqx_nb(nb, ntile, nt), nchunk = (nb + nbi - 1) / nbi;
    #pragma omp parallel
    {
        uint8_t *s = scr + (size_t)omp_get_thread_num() * scr_bytes;
        const void *last = NULL;
        #pragma omp for schedule(dynamic, 1)
        for (int t = 0; t < ntile; t++)
            for (int c = 0; c < nchunk; c++)
                iqx_gemm_item(y, d, kp, type, d, t, c, nbi, xq, xs, k, nb, k, c16, s, &last);
    }
}
#endif

static int test(int type, int d, int k, int nb) {
    uint8_t *kp = make_w(type, d, k);
    int8_t *xq = malloc((size_t)nb * k);
    float *xs = malloc(sizeof(float) * nb * k / 32), *y[3];
    int32_t *c16 = malloc(sizeof(int32_t) * ((size_t)nb * k / 16 + 1));
    for (int i = 0; i < 3; i++) {
        y[i] = malloc(sizeof(float) * nb * d);
        memset(y[i], 0xff, sizeof(float) * nb * d);
    }
    rand_act(xq, xs, c16, nb, k);
    int nsb = k / QK_K, fail = 0;
    for (int rb = 0; rb < d / KQ_ROWS; rb++) // the definition
        for (int b = 0; b < nb; b += KQ_NC)
            iq_dot(type, y[0] + (size_t)b * d + rb * KQ_ROWS, d, kp + (size_t)rb * nsb * iq_sb_bytes(type), k,
                   nb - b < KQ_NC ? nb - b : KQ_NC, xq + (size_t)b * k, k, xs + (size_t)b * k / 32, k / 32);
    for (int g = 0; g < 2; g++) { // the VBMI and the gather decodes (RUNQ_IQ_GATHER)
        rq_iq_gather = g;
        memset(y[1], 0xff, sizeof(float) * nb * d);
        run_rows(type, y[1], d, xq, k, xs, c16, nb, kp);
        if (memcmp(y[0], y[1], sizeof(float) * nb * d)) {
            printf("FAIL %-9s d %d k %d nb %d: iq_rows (%s) != iq_dot\n", iq_name(type), d, k, nb, g ? "gather" : "VBMI");
            fail = 1;
        }
#ifdef IQX_SCRATCH
        int nt = omp_get_max_threads();
        size_t sb = (IQX_SCRATCH(k) + 63) / 64 * 64;
        uint8_t *scr = aligned_alloc(64, sb * nt);
        memset(y[2], 0xff, sizeof(float) * nb * d);
        run_gemm(type, y[2], d, xq, k, xs, c16, nb, kp, scr, sb);
        free(scr);
        if (memcmp(y[0], y[2], sizeof(float) * nb * d)) {
            printf("FAIL %-9s d %d k %d nb %d: iqx_gemm (%s) != iq_dot\n", iq_name(type), d, k, nb, g ? "gather" : "VBMI");
            fail = 1;
        }
#endif
    }
    rq_iq_gather = 0;
    if (!fail) printf("ok   %-9s d %4d k %5d nb %2d\n", iq_name(type), d, k, nb);
    free(kp); free(xq); free(xs); free(c16);
    for (int i = 0; i < 3; i++) free(y[i]);
    return fail;
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void bench(int type, int d, int k, int nb) {
    uint8_t *kp = make_w(type, d, k);
    int8_t *xq = malloc((size_t)nb * k);
    float *xs = malloc(sizeof(float) * nb * k / 32), *y = malloc(sizeof(float) * nb * d);
    int32_t *c16 = malloc(sizeof(int32_t) * ((size_t)nb * k / 16 + 1));
    rand_act(xq, xs, c16, nb, k);
    int nt = omp_get_max_threads();
#ifdef IQX_SCRATCH
    size_t sb = (IQX_SCRATCH(k) + 63) / 64 * 64;
    uint8_t *scr = aligned_alloc(64, sb * nt);
    const int nl = 2;
#else
    const int nl = 1;
#endif
    double best[2] = { 1e9, 1e9 };
    for (int tr = 0; tr < 5; tr++)
        for (int l = 0; l < nl; l++) {
            double t0 = now();
            if (l == 0) run_rows(type, y, d, xq, k, xs, c16, nb, kp);
#ifdef IQX_SCRATCH
            else run_gemm(type, y, d, xq, k, xs, c16, nb, kp, scr, sb);
#endif
            double t = now() - t0;
            if (t < best[l]) best[l] = t;
        }
#ifdef IQX_SCRATCH
    { // (the two paths' outputs: bit-identical here too)
        float *y2 = malloc(sizeof(float) * nb * d);
        run_rows(type, y, d, xq, k, xs, c16, nb, kp);
        run_gemm(type, y2, d, xq, k, xs, c16, nb, kp, scr, sb);
        if (memcmp(y, y2, sizeof(float) * nb * d)) printf("FAIL %s: the tiled GEMM's output differs\n", iq_name(type));
        free(y2);
    }
#endif
    double w = (double)d * k * nb / nt;
    printf("%-9s %5d x %5d nb %4d %2d thr: iq_rows %7.1f ms %.4f ns", iq_name(type), d, k, nb, nt, 1e3 * best[0], 1e9 * best[0] / w);
    if (nl > 1) printf(" | tiled %7.1f ms %.4f ns (x%.2f)", 1e3 * best[1], 1e9 * best[1] / w, best[0] / best[1]);
    printf("\n");
    fflush(stdout);
#ifdef IQX_SCRATCH
    free(scr);
#endif
    free(kp); free(xq); free(xs); free(y); free(c16);
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "-b")) {
        const char *ts = argc > 2 ? argv[2] : "IQ3_S,IQ3_XXS,IQ4_XS";
        int nb = argc > 3 ? atoi(argv[3]) : 512, d = argc > 4 ? atoi(argv[4]) : 17408, k = argc > 5 ? atoi(argv[5]) : 5120;
        for (int i = 0; i < NTYPES; i++) {
            const char *n = iq_name(types[i]), *p = ts;
            size_t ln = strlen(n);
            int sel = !strcmp(ts, "all");
            while (!sel && (p = strstr(p, n))) {
                if ((p == ts || p[-1] == ',') && (p[ln] == 0 || p[ln] == ',')) sel = 1;
                p += ln;
            }
            if (sel) bench(types[i], d, k, nb);
        }
        return 0;
    }
    int fails = 0;
    const int nbs[] = { 1, 2, 3, 7, 8, 9, 13, 32, 41, 64, 77 };
    for (int i = 0; i < NTYPES; i++) {
        for (size_t j = 0; j < sizeof(nbs) / sizeof(nbs[0]); j++) fails += test(types[i], 48, 768, nbs[j]);
        fails += test(types[i], 80, 1024, 77);
        fails += test(types[i], 32, 17408, 9);
    }
    printf(fails ? "%d FAILED\n" : "all ok\n", fails);
    return fails != 0;
}
