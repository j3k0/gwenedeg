// Unit test of --repack's layouts (src/iquants.h: IQ3_XXS / IQ3_S as 4-bit codes, GGML_IQ3_XXS_R / GGML_IQ3_S_R) on the
// AVX-512 build: on random blocks, the repacked kernels (iq_rows) give bit-identical outputs to the base types' (VBMI and
// gather decodes) for 1 .. 77 activation rows; the generic kernel iq_dot and the dequantized rows (iq_packed_row) agree
// between the two layouts. With -b [k]: single-thread ns per weight and activation row, base vs repacked (1024 x k,
// default 5120; 1, 4, 8 and 64 rows).
// Build: make iq-repack-test
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "../../src/kernels_avx512.h"
#include "../../src/iquants.h"

#ifndef RUNQ_IQX
#error "needs the AVX-512 VNNI build"
#endif
int rq_iq_gather = 0;

static uint32_t rs = 12345;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }

static void rand_blocks(int type, uint8_t *b, size_t nblk) {
    int bb = iq_block_bytes(type);
    for (size_t i = 0; i < nblk * bb; i++) b[i] = (uint8_t)rnd();
    for (size_t i = 0; i < nblk; i++) { // finite fp16 d: exponent 8..14, either sign
        uint16_t h = (uint16_t)(((rnd() % 7) + 8) << 10 | (rnd() & 0x3ff) | (rnd() & 1) << 15);
        memcpy(b + i * bb, &h, 2);
    }
}

static void rand_act(int8_t *xq, float *xs, int32_t *c16, int nb, int k) {
    for (int i = 0; i < nb * k; i++) xq[i] = (int8_t)(rnd() % 255 - 127);
    for (int i = 0; i < nb * k / 32; i++) xs[i] = (float)(rnd() % 10000 + 1) * 1e-5f;
    for (int b = 0; b < nb; b++) kq512_c16(c16 + (size_t)b * (k / 16), xq + (size_t)b * k, k);
}

typedef struct { uint8_t *kp, *kr; } Pair;

static Pair make(int type, int d, int k) {
    size_t nblk = (size_t)d * (k / QK_K);
    uint8_t *src = malloc(nblk * iq_block_bytes(type));
    Pair p = { aligned_alloc(64, (iq_size(type, d, k) + 63) / 64 * 64),
               aligned_alloc(64, (iq_size(type | IQ_R, d, k) + 63) / 64 * 64) };
    rand_blocks(type, src, nblk);
    iq_pack_blocks(type, p.kp, src, d / KQ_ROWS, k);
    iq_pack_blocks(type | IQ_R, p.kr, src, d / KQ_ROWS, k);
    free(src);
    return p;
}

static int test(int type, int d, int k, int nb) {
    int rt = iq_repack_type(type), fail = 0;
    Pair p = make(type, d, k);
    int8_t *xq = malloc((size_t)nb * k);
    float *xs = malloc(sizeof(float) * nb * k / 32), *y[4];
    int32_t *c16 = malloc(sizeof(int32_t) * ((size_t)nb * k / 16 + 1));
    for (int i = 0; i < 4; i++) y[i] = malloc(sizeof(float) * nb * d);
    rand_act(xq, xs, c16, nb, k);
    for (int g = 0; g < 2; g++) { // the base type's VBMI and gather decodes vs the codes
        rq_iq_gather = g;
        iq_rows(type, y[0], d, xq, k, xs, c16, nb, p.kp, k, 0, d / KQ_ROWS);
        iq_rows(rt, y[1], d, xq, k, xs, c16, nb, p.kr, k, 0, d / KQ_ROWS);
        if (memcmp(y[0], y[1], sizeof(float) * nb * d)) {
            printf("FAIL %-7s d %d k %d nb %d: iq_rows %s vs %s differ\n", iq_name(type), d, k, nb, g ? "gather" : "VBMI", iq_name(rt));
            fail = 1;
        }
    }
    rq_iq_gather = 0;
    int nsb = k / QK_K; // the generic kernel on both layouts (the definition) == the AVX-512 one
    for (int l = 0; l < 2; l++) {
        int t = l ? rt : type;
        const uint8_t *kp = l ? p.kr : p.kp;
        for (int rb = 0; rb < d / KQ_ROWS; rb++)
            for (int b = 0; b < nb; b += KQ_NC)
                iq_dot(t, y[2 + l] + (size_t)b * d + rb * KQ_ROWS, d, kp + (size_t)rb * nsb * iq_sb_bytes(t), k,
                       nb - b < KQ_NC ? nb - b : KQ_NC, xq + (size_t)b * k, k, xs + (size_t)b * k / 32, k / 32);
    }
    if (memcmp(y[2], y[3], sizeof(float) * nb * d) || memcmp(y[1], y[3], sizeof(float) * nb * d)) {
        printf("FAIL %-7s d %d k %d nb %d: iq_dot base / codes / AVX-512 differ\n", iq_name(type), d, k, nb);
        fail = 1;
    }
    float *ra = malloc(sizeof(float) * k), *rr = malloc(sizeof(float) * k);
    for (int r = 0; r < d; r += 7) { // the dequantized rows of both layouts
        iq_packed_row(type, p.kp, r, k, ra);
        iq_packed_row(rt, p.kr, r, k, rr);
        if (memcmp(ra, rr, sizeof(float) * k)) {
            printf("FAIL %-7s d %d k %d: iq_packed_row %d differs\n", iq_name(type), d, k, r);
            fail = 1;
            break;
        }
    }
    if (!fail) printf("ok   %-7s -> %-9s d %d k %d nb %2d\n", iq_name(type), iq_name(rt), d, k, nb);
    free(ra); free(rr); free(p.kp); free(p.kr); free(xq); free(xs); free(c16);
    for (int i = 0; i < 4; i++) free(y[i]);
    return fail;
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void bench(int type, int k) {
    int d = 1024, nbs[] = { 1, 4, 8, 64 }, rt = iq_repack_type(type);
    Pair p = make(type, d, k);
    for (int j = 0; j < 4; j++) {
        int nb = nbs[j];
        int8_t *xq = malloc((size_t)nb * k);
        float *xs = malloc(sizeof(float) * nb * k / 32), *y = malloc(sizeof(float) * nb * d);
        int32_t *c16 = malloc(sizeof(int32_t) * ((size_t)nb * k / 16 + 1));
        rand_act(xq, xs, c16, nb, k);
        double best[2] = { 1e9, 1e9 };
        for (int tr = 0; tr < 5; tr++)
            for (int l = 0; l < 2; l++) {
                int reps = 0;
                double t0 = now(), t;
                do {
                    iq_rows(l ? rt : type, y, d, xq, k, xs, c16, nb, l ? p.kr : p.kp, k, 0, d / KQ_ROWS);
                    reps++;
                } while ((t = now() - t0) < 0.2);
                if (t / reps < best[l]) best[l] = t / reps;
            }
        double w = (double)d * k * nb;
        printf("%-7s nb %2d: %.4f ns/weight/row, %-9s %.4f (x%.2f)\n", iq_name(type), nb, 1e9 * best[0] / w, iq_name(rt),
               1e9 * best[1] / w, best[0] / best[1]);
        free(xq); free(xs); free(y); free(c16);
    }
    free(p.kp); free(p.kr);
}

int main(int argc, char **argv) {
    const int types[] = { GGML_IQ3_XXS, GGML_IQ3_S };
    if (argc > 1 && !strcmp(argv[1], "-b")) {
        int k = argc > 2 ? atoi(argv[2]) : 5120;
        for (int i = 0; i < 2; i++) bench(types[i], k);
        return 0;
    }
    int fails = 0;
    const int nbs[] = { 1, 2, 3, 8, 9, 13, 32, 41, 77 };
    for (int i = 0; i < 2; i++) {
        fails += test(types[i], 32, 1024, 1);
        for (size_t j = 0; j < sizeof(nbs) / sizeof(nbs[0]); j++) fails += test(types[i], 48, 768, nbs[j]);
    }
    printf(fails ? "%d FAILED\n" : "all ok\n", fails);
    return fails != 0;
}
