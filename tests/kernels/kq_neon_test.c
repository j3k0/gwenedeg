// Unit test of the NEON kernels (kernels_neon.h) against the generic ones on random data: K-quants
// (vs kquants.h) and Q8_0 (vs a copy of runq.c's dot_q8), GEMV and GEMM must be bit-identical; the
// attention on the bf16 / f16 KV caches vs the fp32 cache holding the same values (exact conversion,
// the same arithmetic: bit-identical; the fp32 attention vs the generic one: make test).
// Build: make kq-neon-test
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

static float dot_q8_ref(const int8_t *wq, const float *ws, const int8_t *xq, const float *xs, int n, int gs) { RQ_FP_EXACT
    // runq.c's dot_q8, the same expression (GS a parameter here)
    float val = 0;
    for (int j = 0; j <= n - gs; j += gs) {
        int32_t ival = 0;
        for (int k = 0; k < gs; k++)
            ival += xq[j + k] * wq[j + k];
        val += ((float) ival) * ws[j / gs] * xs[j / gs];
    }
    return val;
}

static int test_q8(int gs, int k) {
    int d = 80, nb = 13, ng = k / gs, fails = 0; // 5 blocks: an odd tile at the end
    int8_t *wq = malloc((size_t)d * k), *xq = malloc((size_t)nb * k);
    float *ws = malloc(sizeof(float) * d * ng), *xs = malloc(sizeof(float) * nb * ng);
    for (int i = 0; i < d * k; i++) wq[i] = (int8_t)(rnd() % 255 - 127);
    for (int i = 0; i < nb * k; i++) xq[i] = (int8_t)(rnd() % 255 - 127);
    for (int i = 0; i < d * ng; i++) ws[i] = (float)(rnd() % 10000 + 1) * 1e-6f;
    for (int i = 0; i < nb * ng; i++) xs[i] = (float)(rnd() % 10000 + 1) * 1e-5f;
    uint8_t *pk = aligned_alloc(64, q8x_size(d, k, gs));
    q8x_pack_blocks(pk, wq, ws, d / 16, k, gs);
    float *ref = calloc((size_t)nb * d, 4), *got = calloc((size_t)nb * d, 4);
    for (int b = 0; b < nb; b++)
        for (int i = 0; i < d; i++)
            ref[(size_t)b * d + i] = dot_q8_ref(wq + (size_t)i * k, ws + (size_t)i * ng, xq + (size_t)b * k, xs + b * ng, k, gs);
    char name[64];
    for (int b = 0; b < nb; b++)
        q8x_gemv_rows(got + (size_t)b * d, xq + (size_t)b * k, xs + b * ng, NULL, pk, k, gs, 0, d / 16);
    snprintf(name, sizeof name, "gemv Q8_0 gs=%d k=%d", gs, k);
    fails += cmp(name, got, ref, nb * d);
    memset(got, 0, sizeof(float) * nb * d);
    int nbi = 5, ntile = (d / 16 + 1) / 2;
    for (int t = 0; t < ntile; t++)
        for (int ch = 0; ch * nbi < nb; ch++)
            q8x_gemm_item(got, d, pk, d, t, ch, nbi, xq, xs, k, nb, k, gs, NULL);
    snprintf(name, sizeof name, "gemm Q8_0 gs=%d k=%d", gs, k);
    fails += cmp(name, got, ref, nb * d);
    free(wq); free(xq); free(ws); free(xs); free(pk); free(ref); free(got);
    return fails;
}

static size_t att_tile(int t, int h, int nkv, int hd) {
    // runq.c's kv_tile: blocks of ATT_P positions, in a block each head's ATT_PT tiles contiguous
    return (((size_t)(t / ATT_P) * nkv + h) * ATT_P + (size_t)(t % ATT_P / ATT_T) * ATT_T) * hd;
}

static int test_attn(int kvt, int hd) {
    // head 1 of 2 (so a wrong block stride reads the other head), 3 blocks (the last one partial); random
    // 16-bit values (bf16 / f16 bit patterns, f16 subnormals included) in runq.c's layouts and their
    // exact fp32 values in the fp32 layout
    int nkv = 2, npos = 3 * ATT_P - 37, h = 1, fails = 0;
    size_t bs = (size_t)nkv * ATT_P * hd, n = 3 * bs, pst = ATT_PSTRIDE(hd);
    float *kf = calloc(n, 4), *vf = calloc(n, 4);
    uint16_t *kh = calloc(n, 2), *vh = calloc(n, 2);
    for (int t = 0; t < npos; t++)
        for (int hh = 0; hh < nkv; hh++)
            for (int kv = 0; kv < 2; kv++)
                for (int i = 0; i < hd; i++) {
                    uint16_t b;
                    float f;
                    if (kvt == 1) { // bf16, |x| ~ 2^-7 .. 2^4
                        b = (uint16_t)((rnd() & 1) << 15 | (120 + rnd() % 11) << 7 | (rnd() & 0x7f));
                        uint32_t u = (uint32_t)b << 16;
                        memcpy(&f, &u, 4);
                    } else { // f16, exponent 0 (subnormal) .. 18
                        uint32_t e = rnd() % 20;
                        b = (uint16_t)((rnd() & 1) << 15 | (e > 18 ? 18 : e) << 10 | (rnd() & 0x3ff));
                        f = fp16_to_f32(b);
                    }
                    size_t o = att_tile(t, hh, nkv, hd);
                    int j = t % ATT_T;
                    if (kv == 0) {
                        kf[o + (size_t)i * ATT_T + j] = f;
                        kh[o + (kvt == 1 ? (size_t)(i & ~1) * ATT_T + 2 * j + (i & 1) : (size_t)i * ATT_T + j)] = b;
                    } else {
                        vf[o + (size_t)j * hd + i] = f;
                        vh[o + (size_t)j * hd + (kvt == 1 ? (i & ~31) + 2 * (i & 15) + ((i >> 4) & 1) : i)] = b;
                    }
                }
    enum { NV = 13 };
    float *qv = malloc(sizeof(float) * NV * hd), *S = malloc(sizeof(float) * ATTN_NV * 3 * ATT_P);
    float *o32 = calloc((size_t)NV * 3 * pst, 4), *o16 = calloc((size_t)NV * 3 * pst, 4);
    for (int i = 0; i < NV * hd; i++) qv[i] = (float)((int)(rnd() % 2001) - 1000) * 1e-3f;
    const float *q[NV];
    float *out32[NV], *out16[NV];
    for (int r = 0; r < NV; r++) q[r] = qv + (size_t)r * hd;
    const size_t k0 = att_tile(0, h, nkv, hd);
    float scale = 1.0f / 11.3137085f;
    static const int nvs[] = { 1, 2, 3, 4, 5, 12, 13 };
    char name[96];
    for (unsigned c = 0; c < sizeof nvs / sizeof nvs[0]; c++) {
        int nv = nvs[c], np[NV];
        // one block (part NULL): np <= ATT_P, partial tiles
        for (int r = 0; r < nv; r++) np[r] = ATT_P - (int)((r * 37) % ATT_P);
        for (int r = 0; r < nv; r++) { out32[r] = o32 + (size_t)r * hd; out16[r] = o16 + (size_t)r * hd; }
        memset(o32, 0, sizeof(float) * NV * 3 * pst);
        memset(o16, 0, sizeof(float) * NV * 3 * pst);
        attn_kv(0, nv, q, out32, NULL, np, 0, 1, kf + k0, vf + k0, bs, hd, scale, S);
        attn_kv(kvt, nv, q, out16, NULL, np, 0, 1, kh + k0, vh + k0, bs, hd, scale, S);
        snprintf(name, sizeof name, "attention %s hd=%d nv=%d, one block", kvt == 1 ? "bf16" : "f16", hd, nv);
        fails += cmp(name, o16, o32, nv * hd);
        // block partial results (part), all three blocks and the blocks 1..2 only, causal-like np
        for (int b0 = 0; b0 < 2; b0++) {
            for (int r = 0; r < nv; r++) {
                np[r] = npos - (int)((r * 53) % 300);
                out32[r] = o32 + (size_t)r * 3 * pst;
                out16[r] = o16 + (size_t)r * 3 * pst;
            }
            memset(o32, 0, sizeof(float) * NV * 3 * pst);
            memset(o16, 0, sizeof(float) * NV * 3 * pst);
            attn_kv(0, nv, q, NULL, out32, np, b0, 3, kf + k0, vf + k0, bs, hd, scale, S);
            attn_kv(kvt, nv, q, NULL, out16, np, b0, 3, kh + k0, vh + k0, bs, hd, scale, S);
            snprintf(name, sizeof name, "attention %s hd=%d nv=%d, blocks %d..2", kvt == 1 ? "bf16" : "f16", hd, nv, b0);
            fails += cmp(name, o16, o32, nv * 3 * (int)pst);
        }
    }
    free(kf); free(vf); free(kh); free(vh); free(qv); free(S); free(o32); free(o16);
    return fails;
}

int main(void) {
    int fails = 0;
    fails += test_attn(1, 128) + test_attn(2, 128) + test_attn(1, 64) + test_attn(2, 64);
    fails += test_q8(32, 1024) + test_q8(32, 3072) + test_q8(64, 1024) + test_q8(64, 3072);
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
