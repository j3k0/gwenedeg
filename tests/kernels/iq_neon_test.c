// Unit test of the NEON kernels of src/iquants.h (Q2_K and the i-quants: iqn_*, via iq_rows) against the generic
// definition iq_dot on random blocks: bit-identical for 1 and several activation rows. With -b [nb [k]]: the single-thread
// time per weight and activation row of each type (1024 x k, default 5120 as docs/iquants.md's table; nb rows, default 1),
// NEON and generic.
// Build: make iq-neon-test
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "../../src/kernels_neon.h"
#include "../../src/iquants.h"

#ifndef RUNQ_IQN
#error "needs the NEON build (AArch64 with the dot product extension)"
#endif

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

static const int types[] = { GGML_Q2_K, GGML_IQ2_XXS, GGML_IQ2_XS, GGML_IQ2_S, GGML_IQ3_XXS, GGML_IQ3_S,
                             GGML_IQ1_S, GGML_IQ1_M, GGML_IQ4_XS };

static void rand_act(int8_t *xq, float *xs, int nb, int k) {
    for (int i = 0; i < nb * k; i++) xq[i] = (int8_t)(rnd() % 255 - 127);
    for (int i = 0; i < nb * k / 32; i++) xs[i] = (float)(rnd() % 10000 + 1) * 1e-5f;
}

static int test(int type, int d, int k, int nb) {
    size_t nblk = (size_t)d * (k / QK_K);
    uint8_t *src = malloc(nblk * iq_block_bytes(type)), *kp = aligned_alloc(64, (iq_size(type, d, k) + 63) / 64 * 64);
    int8_t *xq = malloc((size_t)nb * k);
    float *xs = malloc(sizeof(float) * nb * k / 32), *y = malloc(sizeof(float) * nb * d), *r = malloc(sizeof(float) * nb * d);
    rand_blocks(type, src, nblk);
    iq_pack_blocks(type, kp, src, d / KQ_ROWS, k);
    rand_act(xq, xs, nb, k);
    iq_rows(type, y, d, xq, k, xs, NULL, nb, kp, k, 0, d / KQ_ROWS);
    int nsb = k / QK_K;
    for (int rb = 0; rb < d / KQ_ROWS; rb++)
        for (int b = 0; b < nb; b += KQ_NC)
            iq_dot(type, r + (size_t)b * d + rb * KQ_ROWS, d, kp + (size_t)rb * nsb * iq_sb_bytes(type), k,
                   nb - b < KQ_NC ? nb - b : KQ_NC, xq + (size_t)b * k, k, xs + (size_t)b * k / 32, k / 32);
    int fail = 0;
    for (int i = 0; i < nb * d && !fail; i++)
        if (memcmp(y + i, r + i, 4)) {
            printf("FAIL %-7s d %d k %d nb %d: first diff at row %d col %d: %.9g vs %.9g\n", iq_name(type), d, k, nb,
                   i / d, i % d, y[i], r[i]);
            fail = 1;
        }
    if (!fail) printf("ok   %-7s d %d k %d nb %2d (%d outputs)\n", iq_name(type), d, k, nb, nb * d);
    free(src); free(kp); free(xq); free(xs); free(y); free(r);
    return fail;
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void bench(int type, int nb, int k) {
    int d = 1024;
    size_t nblk = (size_t)d * (k / QK_K);
    uint8_t *src = malloc(nblk * iq_block_bytes(type)), *kp = aligned_alloc(64, (iq_size(type, d, k) + 63) / 64 * 64);
    int8_t *xq = malloc((size_t)nb * k);
    float *xs = malloc(sizeof(float) * nb * k / 32), *y = malloc(sizeof(float) * nb * d);
    rand_blocks(type, src, nblk);
    iq_pack_blocks(type, kp, src, d / KQ_ROWS, k);
    rand_act(xq, xs, nb, k);
    double best = 1e9, bestg = 1e9;
    for (int it = 0; it < 7; it++) {
        double t0 = now();
        iq_rows(type, y, d, xq, k, xs, NULL, nb, kp, k, 0, d / KQ_ROWS);
        double t = now() - t0;
        if (t < best) best = t;
    }
    int nsb = k / QK_K;
    for (int it = 0; it < 2; it++) {
        double t0 = now();
        for (int rb = 0; rb < d / KQ_ROWS; rb++)
            for (int b = 0; b < nb; b += KQ_NC)
                iq_dot(type, y + (size_t)b * d + rb * KQ_ROWS, d, kp + (size_t)rb * nsb * iq_sb_bytes(type), k,
                       nb - b < KQ_NC ? nb - b : KQ_NC, xq + (size_t)b * k, k, xs + (size_t)b * k / 32, k / 32);
        double t = now() - t0;
        if (t < bestg) bestg = t;
    }
    double w = (double)d * k * nb;
    printf("%-7s nb %d: neon %.4f ns/weight/row  generic %.4f  (x%.1f)\n", iq_name(type), nb, 1e9 * best / w, 1e9 * bestg / w,
           bestg / best);
    free(src); free(kp); free(xq); free(xs); free(y);
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "-b")) {
        int nb = argc > 2 ? atoi(argv[2]) : 1, k = argc > 3 ? atoi(argv[3]) : 5120;
        for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); i++) bench(types[i], nb, k);
        return 0;
    }
    int fails = 0;
    const int nbs[] = { 1, 2, 3, 8, 9, 13, 32, 41, 77 }; // (iq_rows: passes of 8 / 32 rows)
    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); i++) {
        fails += test(types[i], 48, 512, 1);
        fails += test(types[i], 32, 1024, 1);
        for (size_t j = 0; j < sizeof(nbs) / sizeof(nbs[0]); j++) fails += test(types[i], 48, 768, nbs[j]);
    }
    printf(fails ? "%d FAILED\n" : "all ok\n", fails);
    return fails != 0;
}
