// Unit test of the NEON K-quant kernels (kernels_neon.h) against the generic ones (kquants.h) on
// random packed blocks: GEMV and GEMM must be bit-identical. Build: make kq-neon-test
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../../src/kquants.h"
#include "../../src/kernels_neon.h"

static uint32_t rs = 12345;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }

static void rand_blocks(int type, uint8_t *b, int nblk) {
    int bb = kq_block_bytes(type);
    for (int i = 0; i < nblk * bb; i++) b[i] = (uint8_t)rnd();
    for (int i = 0; i < nblk; i++) { // finite, small fp16 scales: exponent 8..14
        uint8_t *p = b + (size_t)i * bb;
        uint16_t h = (uint16_t)(((rnd() % 7) + 8) << 10 | (rnd() & 0x3ff) | (rnd() & 1) << 15);
        if (type == GGML_Q4_K) {
            memcpy(p, &h, 2);
            h = (uint16_t)(((rnd() % 7) + 8) << 10 | (rnd() & 0x3ff));
            memcpy(p + 2, &h, 2);
        } else memcpy(p + 208, &h, 2);
    }
}

static int cmp(const char *what, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++)
        if (memcmp(a + i, b + i, 4)) {
            printf("FAIL %s: first diff at %d: %.9g vs %.9g\n", what, i, a[i], b[i]);
            return 1;
        }
    printf("ok   %s (%d outputs)\n", what, n);
    return 0;
}

int main(void) {
    int fails = 0;
    int types[2] = { GGML_Q4_K, GGML_Q6_K };
    int ks[2] = { 1024, 3072 };
    for (int ti = 0; ti < 2; ti++)
        for (int ki = 0; ki < 2; ki++) {
            int type = types[ti], k = ks[ki], d = 80, nb = 13; // 5 blocks: an odd tile at the end
            int nsb = k / QK_K, ng = k / 32;
            uint8_t *src = malloc((size_t)d * nsb * kq_block_bytes(type));
            uint8_t *kp = aligned_alloc(64, kq_size(type, d, k));
            rand_blocks(type, src, d * nsb);
            kq_pack_blocks(type, kp, src, d / KQ_ROWS, k);
            int8_t *xq = malloc((size_t)nb * k);
            float *xs = malloc(sizeof(float) * nb * ng), *xf = malloc(sizeof(float) * nb * ng);
            int32_t *c16 = malloc(sizeof(int32_t) * nb * (k / 16));
            for (int i = 0; i < nb * k; i++) xq[i] = (int8_t)(rnd() % 255 - 127);
            for (int i = 0; i < nb * ng; i++) xs[i] = (float)(rnd() % 10000 + 1) * 1e-5f;
            for (int b = 0; b < nb; b++) {
                kq_act_prep(xf + b * ng, xq + (size_t)b * k, xs + b * ng, k);
                kqx_c16(c16 + b * (k / 16), xq + (size_t)b * k, k);
            }
            float *ref = calloc((size_t)nb * d, 4), *got = calloc((size_t)nb * d, 4);
            for (int rb = 0; rb < d / KQ_ROWS; rb++)
                for (int b = 0; b < nb; b += KQ_NC) {
                    int nc = nb - b < KQ_NC ? nb - b : KQ_NC;
                    kq_dot(type, ref + (size_t)b * d + rb * KQ_ROWS, d, kp + (size_t)rb * nsb * kq_sb_bytes(type), k, nc,
                           xq + (size_t)b * k, k, xs + b * ng, ng, xf + b * ng, ng);
                }
            char name[64];
            // GEMV: one activation row at a time
            for (int b = 0; b < nb; b++)
                kqx_gemv_rows(type, got + (size_t)b * d, xq + (size_t)b * k, xs + b * ng, xf + b * ng, c16 + b * (k / 16), kp, k, 0, d / KQ_ROWS);
            snprintf(name, sizeof name, "gemv %s k=%d", type == GGML_Q4_K ? "Q4_K" : "Q6_K", k);
            fails += cmp(name, got, ref, nb * d);
            // GEMM: all tiles, chunks of 5 rows
            memset(got, 0, sizeof(float) * nb * d);
            uint8_t *scr = aligned_alloc(64, KQX_SCRATCH(k));
            const void *last = NULL;
            int nbi = 5, ntile = (d / 16 + 1) / 2;
            for (int t = 0; t < ntile; t++)
                for (int ch = 0; ch * nbi < nb; ch++)
                    kqx_gemm_item(got, d, kp, type, d, t, ch, nbi, xq, xs, k, nb, k, xf, c16, scr, &last);
            snprintf(name, sizeof name, "gemm %s k=%d", type == GGML_Q4_K ? "Q4_K" : "Q6_K", k);
            fails += cmp(name, got, ref, nb * d);
            free(src); free(kp); free(xq); free(xs); free(xf); free(c16); free(ref); free(got); free(scr);
        }
    printf(fails ? "%d FAILED\n" : "all ok\n", fails);
    return fails != 0;
}
