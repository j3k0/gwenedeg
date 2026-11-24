// V4H prototype kernels (docs/v4h.md): a scalar definition, AVX-512 VNNI kernels, a bitwise test and a micro-benchmark
// against the existing IQ3_S/r and IQ4_XS kernels (src/iquants.h) on the same shapes, in the same process.
//
// Weights (docs/v4h.md section 3): per 16-row block and 256-column superblock, 2144 bytes: D[16] fp16 | 16 dwords of
// eight 4-bit subscales s (nibble g: group g) | 8 groups x 4 chunks x 64 bytes of nibbles (chunk c, lane r, byte j: low
// nibble column 32 g + 8 c + j, high nibble column 32 g + 8 c + 4 + j), u = q + 8, q in -8..7. w = D s q.
// Activations: "flat" (one fp32 scale A per 256 values, int8 z in -127..127): the quality test of docs/v4h.md
// (GWENEDEG_ACTQ_SIM=flat) measured KL +0.0036 on Qwen3.5-0.8B Q8_0 and no change on the V4H file, so the activation
// subscale a of the proposal is left out here.
// Definition, per output row, activation row and superblock b: I = sum_g s_g sum_{j in g} q_j z_j (int32, exact),
// y = y + (D * A_b) * (float) I, each fp32 operation rounded in that order (kq6_fold).
//
// Kernels (all bit-identical to the definition):
//   v4h_rows  GEMV / small batches from the packed weights: nibbles -> u (and / shift), T_g = sum u z - 8 X_g by
//             vpdpbusd, I += s_g T_g by vpdpwssd (|T_g| <= 32512: int16) or by vpmulld + vpaddd (-m), a fold per 256
//   v4hx_p    tiled prefill, the tile decoded once to signed bytes s q (|s q| <= 120; s u by one vpmullw on byte pairs,
//             then - 8 s), the activations as u8 z + 128 and the bias -128 sum(s q) per row precomputed: per
//             superblock 64 vpdpbusd and one fold, no per-group work at all
//   v4hx_u    tiled prefill, the proposal as written: u codes in scratch, per group T_g and vpdpwssd, a fold per 256
//
// Build: make v4h-bench (build/v4h_bench). Test: build/v4h_bench. Benchmark: build/v4h_bench -b [NB [D [K]]]
// (OMP_NUM_THREADS threads; default 512 rows of 17408 x 5120, Qwen3.8-27B's ffn_gate / ffn_up).
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

#define V4H_SBB 2144
static uint32_t rs = 12345;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }

// ---------------------------------------------------------------------------------------------------------------------
// format: packing and the scalar definition

static void v4h_pack(uint8_t *out, const uint16_t *D, const uint8_t *s, const int8_t *q, int d, int k) {
    // D[row][nsb] fp16 bits, s[row][nsb][8], q[row][k] -> the blocks
    int nsb = k / 256;
    for (int rb = 0; rb < d / 16; rb++)
        for (int b = 0; b < nsb; b++) {
            uint8_t *o = out + ((size_t)rb * nsb + b) * V4H_SBB;
            memset(o, 0, V4H_SBB);
            for (int r = 0; r < 16; r++) {
                int row = 16 * rb + r;
                memcpy(o + 2 * r, &D[(size_t)row * nsb + b], 2);
                uint32_t sv = 0;
                for (int g = 0; g < 8; g++) sv |= (uint32_t)s[((size_t)row * nsb + b) * 8 + g] << 4 * g;
                memcpy(o + 32 + 4 * r, &sv, 4);
                for (int g = 0; g < 8; g++)
                    for (int c = 0; c < 4; c++)
                        for (int j = 0; j < 4; j++) {
                            size_t col = (size_t)256 * b + 32 * g + 8 * c + j;
                            int lo = q[(size_t)row * k + col] + 8, hi = q[(size_t)row * k + col + 4] + 8;
                            o[96 + 256 * g + 64 * c + 4 * r + j] = (uint8_t)(lo | hi << 4);
                        }
            }
        }
}

static float h2f(uint16_t h) { return _cvtsh_ss(h); }

static void v4h_ref(float *y, size_t ldy, const uint8_t *w, int d, int k, const int8_t *z, size_t ldz, const float *A,
                    size_t lda, int nb) {
    // the definition, from the packed blocks
    int nsb = k / 256;
    for (int bb = 0; bb < nb; bb++)
        for (int row = 0; row < d; row++) {
            int rb = row / 16, r = row % 16;
            float f = 0;
            for (int b = 0; b < nsb; b++) {
                const uint8_t *o = w + ((size_t)rb * nsb + b) * V4H_SBB;
                uint16_t dh;
                uint32_t sv;
                memcpy(&dh, o + 2 * r, 2);
                memcpy(&sv, o + 32 + 4 * r, 4);
                int32_t I = 0;
                for (int g = 0; g < 8; g++) {
                    int32_t T = 0;
                    for (int c = 0; c < 4; c++)
                        for (int j = 0; j < 4; j++) {
                            uint8_t p = o[96 + 256 * g + 64 * c + 4 * r + j];
                            const int8_t *x = z + (size_t)bb * ldz + 256 * b + 32 * g + 8 * c + j;
                            T += ((p & 15) - 8) * x[0] + ((p >> 4) - 8) * x[4];
                        }
                    I += (int32_t)(sv >> 4 * g & 15) * T;
                }
                // (volatile: each operation rounded in this order, whatever -Ofast's reassociation would do)
                volatile float da = h2f(dh) * A[bb * lda + b];
                volatile float fi = (float)I;
                volatile float m = da * fi;
                f = f + m;
            }
            y[(size_t)bb * ldy + row] = f;
        }
}

// ---------------------------------------------------------------------------------------------------------------------
// GEMV / small batches from the packed blocks

static inline __m512i bc4u(const uint8_t *p) { int32_t v; memcpy(&v, p, 4); return _mm512_set1_epi32(v); }
static inline __m512i bc4s(const int8_t *p) { int32_t v; memcpy(&v, p, 4); return _mm512_set1_epi32(v); }
static inline __m512 fold(__m512 f, __m512 d, float a, __m512i I) {
    return _mm512_add_round_ps(f, _mm512_mul_round_ps(_mm512_mul_round_ps(d, _mm512_set1_ps(a), Q8P_RND),
                                                      _mm512_cvtepi32_ps(I), Q8P_RND), Q8P_RND);
}

static inline __attribute__((always_inline)) void v4h_blk(const int NC, const int MUL, float *out, size_t ldo,
                                                          const uint8_t *wb, int nsb, int nc, const int8_t *z, size_t ldz,
                                                          const float *A, size_t lda, const int32_t *X, size_t ldX) {
    // 16 rows (one block, wb) x nc <= NC activation rows; X: the activations' sums per 32
    __m512 f[NC];
    for (int c = 0; c < NC; c++) f[c] = _mm512_setzero_ps();
    const __m512i m4 = _mm512_set1_epi8(15), n15 = _mm512_set1_epi32(15);
    for (int b = 0; b < nsb; b++) {
        const uint8_t *o = wb + (size_t)b * V4H_SBB;
        __m512 D = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)o));
        __m512i Sv = _mm512_loadu_si512((const void *)(o + 32)), I[NC];
        for (int c = 0; c < NC; c++) I[c] = _mm512_setzero_si512();
        for (int g = 0; g < 8; g++) {
            __m512i w[8];
            for (int c = 0; c < 4; c++) {
                __m512i q = _mm512_loadu_si512((const void *)(o + 96 + 256 * g + 64 * c));
                w[2 * c] = _mm512_and_si512(q, m4);
                w[2 * c + 1] = _mm512_and_si512(_mm512_srli_epi16(q, 4), m4);
            }
            __m512i S = _mm512_and_si512(_mm512_srli_epi32(Sv, 4 * g), n15);
            for (int c = 0; c < NC; c++) {
                if (c >= nc) break;
                const int8_t *x = z + c * ldz + 256 * b + 32 * g;
                __m512i P = _mm512_set1_epi32(-8 * X[c * ldX + 8 * b + g]);
                for (int t = 0; t < 8; t++) P = _mm512_dpbusd_epi32(P, w[t], bc4s(x + 4 * t));
                I[c] = MUL ? _mm512_add_epi32(I[c], _mm512_mullo_epi32(P, S)) : _mm512_dpwssd_epi32(I[c], P, S);
            }
        }
        for (int c = 0; c < NC && c < nc; c++) f[c] = fold(f[c], D, A[c * lda + b], I[c]);
    }
    for (int c = 0; c < nc && c < NC; c++) _mm512_storeu_ps(out + c * ldo, f[c]);
}

#define V4H_ARGS float *out, size_t ldo, const uint8_t *wb, int nsb, int nc, const int8_t *z, size_t ldz, const float *A, \
                 size_t lda, const int32_t *X, size_t ldX
#define V4H_PASS out, ldo, wb, nsb, nc, z, ldz, A, lda, X, ldX
static void v4h_w1(V4H_ARGS) { v4h_blk(1, 0, V4H_PASS); }
static void v4h_w8(V4H_ARGS) { v4h_blk(8, 0, V4H_PASS); }
static void v4h_m1(V4H_ARGS) { v4h_blk(1, 1, V4H_PASS); }
static void v4h_m8(V4H_ARGS) { v4h_blk(8, 1, V4H_PASS); }

typedef struct { // one activation batch, both layouts
    int nb, k;
    int8_t *z;   // nb x k, -127..127
    uint8_t *zu; // z + 128
    float *A;    // nb x k / 256
    int32_t *X;  // nb x k / 32
    float *xs;   // nb x k / 32: the per-32 format for the i-quant baselines
    int32_t *c16;
} Act;

static void v4h_rows(int mul, float *y, int d, const uint8_t *w, int k, const Act *a, int rb0, int rb1) {
    int nsb = k / 256;
    for (int rb = rb0; rb < rb1; rb++) {
        const uint8_t *wb = w + (size_t)rb * nsb * V4H_SBB;
        if (a->nb == 1) {
            (mul ? v4h_m1 : v4h_w1)(y + 16 * rb, d, wb, nsb, 1, a->z, k, a->A, nsb, a->X, k / 32);
            continue;
        }
        for (int b = 0; b < a->nb; b += 8)
            (mul ? v4h_m8 : v4h_w8)(y + (size_t)b * d + 16 * rb, d, wb, nsb, a->nb - b < 8 ? a->nb - b : 8,
                                    a->z + (size_t)b * k, k, a->A + (size_t)b * nsb, nsb, a->X + (size_t)b * (k / 32), k / 32);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// tiled prefill

#define PREC 4224 // v4hx_p scratch per (16-row block, superblock): 64 x 64 bytes s q, 64 bias (int32), 64 D (fp32)
#define UREC 4672 // v4hx_u: 64 x 64 bytes u, 8 x 64 S (int32), 64 D

static void v4hx_unpack_p(uint8_t *scr, const uint8_t *wt, int rb, int k) {
    int nsb = k / 256;
    const __m512i m4 = _mm512_set1_epi8(15), n15 = _mm512_set1_epi32(15), one = _mm512_set1_epi8(1);
    for (int r = 0; r < rb; r++)
        for (int b = 0; b < nsb; b++) {
            const uint8_t *o = wt + ((size_t)r * nsb + b) * V4H_SBB;
            uint8_t *dst = scr + ((size_t)r * nsb + b) * PREC;
            __m512i Sv = _mm512_loadu_si512((const void *)(o + 32)), acc = _mm512_setzero_si512();
            for (int g = 0; g < 8; g++) {
                __m512i S = _mm512_and_si512(_mm512_srli_epi32(Sv, 4 * g), n15);
                __m512i S16 = _mm512_or_si512(S, _mm512_slli_epi32(S, 16));            // s in each 16-bit lane
                __m512i S8 = _mm512_mullo_epi32(S, _mm512_set1_epi32(0x08080808));    // 8 s in each byte
                for (int c = 0; c < 4; c++) {
                    __m512i q = _mm512_loadu_si512((const void *)(o + 96 + 256 * g + 64 * c));
                    __m512i u0 = _mm512_and_si512(q, m4), u1 = _mm512_and_si512(_mm512_srli_epi16(q, 4), m4);
                    // byte pairs times s in one 16-bit multiply: s u <= 225 < 256, so no carry between the bytes
                    __m512i w0 = _mm512_sub_epi8(_mm512_mullo_epi16(u0, S16), S8), w1 = _mm512_sub_epi8(_mm512_mullo_epi16(u1, S16), S8);
                    _mm512_store_si512((void *)(dst + 64 * (8 * g + 2 * c)), w0);
                    _mm512_store_si512((void *)(dst + 64 * (8 * g + 2 * c + 1)), w1);
                    acc = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(acc, one, w0), one, w1);
                }
            }
            _mm512_store_si512((void *)(dst + 4096), _mm512_mullo_epi32(acc, _mm512_set1_epi32(-128)));
            _mm512_store_ps((float *)(dst + 4160), _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)o)));
        }
}

static inline __attribute__((always_inline)) void v4hx_tile_p(const int RB, const int C, const uint8_t *ws, size_t bstride,
                                                              int nsb, const uint8_t *zu, size_t ldz, const float *A,
                                                              size_t lda, float *out, size_t ldo) {
    __m512 f[2][8];
    for (int r = 0; r < RB; r++)
        for (int c = 0; c < C; c++) f[r][c] = _mm512_setzero_ps();
    for (int b = 0; b < nsb; b++) {
        const uint8_t *wg = ws + (size_t)b * PREC;
        const uint8_t *xb = zu + 256 * b;
        __m512i t[2][8];
        for (int r = 0; r < RB; r++) {
            __m512i bias = _mm512_load_si512((const void *)(wg + r * bstride + 4096));
            for (int c = 0; c < C; c++) t[r][c] = bias;
        }
        #pragma GCC unroll 8
        for (int st = 0; st < 64; st++) {
            __m512i w[2];
            for (int r = 0; r < RB; r++) w[r] = _mm512_load_si512((const void *)(wg + r * bstride + 64 * st));
            for (int c = 0; c < C; c++) {
                __m512i xv = bc4u(xb + c * ldz + 4 * st);
                for (int r = 0; r < RB; r++) t[r][c] = _mm512_dpbusd_epi32(t[r][c], xv, w[r]);
            }
        }
        for (int r = 0; r < RB; r++) {
            __m512 d = _mm512_load_ps((const float *)(wg + r * bstride + 4160));
            for (int c = 0; c < C; c++) f[r][c] = fold(f[r][c], d, A[c * lda + b], t[r][c]);
        }
    }
    for (int r = 0; r < RB; r++)
        for (int c = 0; c < C; c++) _mm512_storeu_ps(out + c * ldo + 16 * r, f[r][c]);
}

static void v4hx_unpack_u(uint8_t *scr, const uint8_t *wt, int rb, int k) {
    int nsb = k / 256;
    const __m512i m4 = _mm512_set1_epi8(15), n15 = _mm512_set1_epi32(15);
    for (int r = 0; r < rb; r++)
        for (int b = 0; b < nsb; b++) {
            const uint8_t *o = wt + ((size_t)r * nsb + b) * V4H_SBB;
            uint8_t *dst = scr + ((size_t)r * nsb + b) * UREC;
            __m512i Sv = _mm512_loadu_si512((const void *)(o + 32));
            for (int g = 0; g < 8; g++) {
                for (int c = 0; c < 4; c++) {
                    __m512i q = _mm512_loadu_si512((const void *)(o + 96 + 256 * g + 64 * c));
                    _mm512_store_si512((void *)(dst + 64 * (8 * g + 2 * c)), _mm512_and_si512(q, m4));
                    _mm512_store_si512((void *)(dst + 64 * (8 * g + 2 * c + 1)), _mm512_and_si512(_mm512_srli_epi16(q, 4), m4));
                }
                _mm512_store_si512((void *)(dst + 4096 + 64 * g), _mm512_and_si512(_mm512_srli_epi32(Sv, 4 * g), n15));
            }
            _mm512_store_ps((float *)(dst + 4608), _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)o)));
        }
}

static inline __attribute__((always_inline)) void v4hx_tile_u(const int RB, const int C, const uint8_t *ws, size_t bstride,
                                                              int nsb, const int8_t *z, size_t ldz, const float *A,
                                                              size_t lda, const int32_t *X, size_t ldX, float *out,
                                                              size_t ldo) {
    __m512 f[2][6];
    for (int r = 0; r < RB; r++)
        for (int c = 0; c < C; c++) f[r][c] = _mm512_setzero_ps();
    for (int b = 0; b < nsb; b++) {
        const uint8_t *wg = ws + (size_t)b * UREC;
        __m512i I[2][6];
        for (int r = 0; r < RB; r++)
            for (int c = 0; c < C; c++) I[r][c] = _mm512_setzero_si512();
        for (int g = 0; g < 8; g++) {
            __m512i t[2][6];
            for (int c = 0; c < C; c++) {
                __m512i cv = _mm512_set1_epi32(-8 * X[c * ldX + 8 * b + g]);
                for (int r = 0; r < RB; r++) t[r][c] = cv;
            }
            #pragma GCC unroll 8
            for (int st = 0; st < 8; st++) {
                __m512i w[2];
                for (int r = 0; r < RB; r++) w[r] = _mm512_load_si512((const void *)(wg + r * bstride + 64 * (8 * g + st)));
                for (int c = 0; c < C; c++) {
                    __m512i xv = bc4s(z + c * ldz + 256 * b + 32 * g + 4 * st);
                    for (int r = 0; r < RB; r++) t[r][c] = _mm512_dpbusd_epi32(t[r][c], w[r], xv);
                }
            }
            for (int r = 0; r < RB; r++) {
                __m512i S = _mm512_load_si512((const void *)(wg + r * bstride + 4096 + 64 * g));
                for (int c = 0; c < C; c++) I[r][c] = _mm512_dpwssd_epi32(I[r][c], t[r][c], S);
            }
        }
        for (int r = 0; r < RB; r++) {
            __m512 d = _mm512_load_ps((const float *)(wg + r * bstride + 4608));
            for (int c = 0; c < C; c++) f[r][c] = fold(f[r][c], d, A[c * lda + b], I[r][c]);
        }
    }
    for (int r = 0; r < RB; r++)
        for (int c = 0; c < C; c++) _mm512_storeu_ps(out + c * ldo + 16 * r, f[r][c]);
}

#ifndef PC
#define PC 5
#endif
#ifndef UC
#define UC 4
#endif
// activation rows per register tile, v4hx_p (2 x 5 int32 + 2 x 5 fp32 accumulators; 1 thread 1024 x 5120 x 512: C 3 / 4 / 5
// / 6 / 7 / 8 0.00349 / 0.00307 / 0.00273 / 0.00357 / 0.00350 / 0.00374 ns, 6 and up spill)
// (UC: v4hx_u's, t, I and f: 3 x 2 x 4)
#define TP_ARGS const uint8_t *ws, size_t bstride, int nsb, const uint8_t *zu, size_t ldz, const float *A, size_t lda, \
                float *out, size_t ldo
#define TU_ARGS const uint8_t *ws, size_t bstride, int nsb, const int8_t *z, size_t ldz, const float *A, size_t lda, \
                const int32_t *X, size_t ldX, float *out, size_t ldo
#define TP(R, C) static void tp_##R##_##C(TP_ARGS) { v4hx_tile_p(R, C, ws, bstride, nsb, zu, ldz, A, lda, out, ldo); }
#define TU(R, C) static void tu_##R##_##C(TU_ARGS) { v4hx_tile_u(R, C, ws, bstride, nsb, z, ldz, A, lda, X, ldX, out, ldo); }
TP(1, 1) TP(1, 2) TP(1, 3) TP(1, 4) TP(1, 5) TP(1, 6) TP(1, 7) TP(1, 8) TP(2, 1) TP(2, 2) TP(2, 3) TP(2, 4) TP(2, 5) TP(2, 6) TP(2, 7) TP(2, 8)
TU(1, 1) TU(1, 2) TU(1, 3) TU(1, 4) TU(1, 5) TU(1, 6) TU(2, 1) TU(2, 2) TU(2, 3) TU(2, 4) TU(2, 5) TU(2, 6)
typedef void (*TpFn)(TP_ARGS);
typedef void (*TuFn)(TU_ARGS);
static const TpFn tp_fns[2][8] = { { tp_1_1, tp_1_2, tp_1_3, tp_1_4, tp_1_5, tp_1_6, tp_1_7, tp_1_8 },
                                   { tp_2_1, tp_2_2, tp_2_3, tp_2_4, tp_2_5, tp_2_6, tp_2_7, tp_2_8 } };
static const TuFn tu_fns[2][6] = { { tu_1_1, tu_1_2, tu_1_3, tu_1_4, tu_1_5, tu_1_6 }, { tu_2_1, tu_2_2, tu_2_3, tu_2_4, tu_2_5, tu_2_6 } };

static void v4hx_gemm(int u, float *y, int d, const uint8_t *w, int k, const Act *a, uint8_t *scr, size_t scr_bytes) {
    // (32-row tile, chunk of nbi activation rows) items, the tile decoded once per thread (as iqx_gemm_item)
    int nt = omp_get_max_threads(), ntile = (d / 16 + 1) / 2, nb = a->nb, nsb = k / 256;
    int nbi = q8p_nb(nb, ntile, nt), nchunk = (nb + nbi - 1) / nbi;
    size_t rec = u ? UREC : PREC, bstride = (size_t)nsb * rec;
    #pragma omp parallel
    {
        uint8_t *s = scr + (size_t)omp_get_thread_num() * scr_bytes;
        #pragma omp for schedule(dynamic, 1)
        for (int t = 0; t < ntile; t++) {
            int rb0 = 2 * t, rb = d / 16 - rb0 < 2 ? d / 16 - rb0 : 2;
            const uint8_t *wt = w + (size_t)rb0 * nsb * V4H_SBB;
            (u ? v4hx_unpack_u : v4hx_unpack_p)(s, wt, rb, k);
            for (int ch = 0; ch < nchunk; ch++) {
                int b1 = (ch + 1) * nbi < nb ? (ch + 1) * nbi : nb, C = u ? UC : PC;
                for (int b = ch * nbi; b < b1; b += C) {
                    int c = b1 - b < C ? b1 - b : C;
                    float *o = y + (size_t)b * d + 16 * rb0;
                    if (u)
                        tu_fns[rb - 1][c - 1](s, bstride, nsb, a->z + (size_t)b * k, k, a->A + (size_t)b * nsb, nsb,
                                              a->X + (size_t)b * (k / 32), k / 32, o, d);
                    else
                        tp_fns[rb - 1][c - 1](s, bstride, nsb, a->zu + (size_t)b * k, k, a->A + (size_t)b * nsb, nsb, o, d);
                }
            }
        }
    }
}

// the control: the existing per-32-scale i-quant scratch (iqx_unpack_tile) with flat activations (one A per 256), the
// integer subscale accumulated per group (I += S T by vpmulld: |T| > int16 for IQ3_S / IQ4_XS) and one fold per 256, i.e.
// what IQ3_S/r or IQ4_XS would gain from the activation format alone, without a new quantization (not bit-identical to
// anything existing: a different definition)
#ifndef FC
#define FC 6 // (1 thread 1024 x 5120 x 512: 3 / 4 / 5 / 6 0.00375 / 0.00336 / 0.00312 / 0.00306 ns, IQ3_S/r)
#endif
static inline __attribute__((always_inline)) void iqf_tile(const int RB, const int C, const uint8_t *ws, size_t bstride, int nsb,
                                                           const int8_t *x, size_t ldx, const float *A, size_t lda,
                                                           const int32_t *c16, size_t ldc, float *out, size_t ldo, int Bi) {
    __m512 f[2][6];
    for (int r = 0; r < RB; r++)
        for (int c = 0; c < C; c++) f[r][c] = _mm512_setzero_ps();
    for (int b = 0; b < nsb; b++) {
        __m512i I[2][6];
        for (int r = 0; r < RB; r++)
            for (int c = 0; c < C; c++) I[r][c] = _mm512_setzero_si512();
        for (int g8 = 0; g8 < 8; g8++) {
            int g = 8 * b + g8;
            const uint8_t *wg = ws + (size_t)g * IQS_REC;
            const int8_t *xg = x + (size_t)g * 32;
            __m512i t[2][6];
            for (int c = 0; c < C; c++) {
                const int32_t *cc = c16 + c * ldc + 2 * g;
                __m512i cv = _mm512_set1_epi32(Bi * (cc[0] >> 5) + Bi * (cc[1] >> 5));
                for (int r = 0; r < RB; r++) t[r][c] = cv;
            }
            #pragma GCC unroll 8
            for (int st = 0; st < 8; st++) {
                __m512i w[2];
                for (int r = 0; r < RB; r++) w[r] = _mm512_load_si512((const void *)(wg + r * bstride + st * 64));
                for (int c = 0; c < C; c++) {
                    __m512i xv = bc4s(xg + c * ldx + st * 4);
                    for (int r = 0; r < RB; r++) t[r][c] = _mm512_dpbusd_epi32(t[r][c], w[r], xv);
                }
            }
            for (int r = 0; r < RB; r++) {
                __m512i sc = _mm512_load_si512((const void *)(wg + r * bstride + 512));
                for (int c = 0; c < C; c++) I[r][c] = _mm512_add_epi32(I[r][c], _mm512_mullo_epi32(sc, t[r][c]));
            }
        }
        for (int r = 0; r < RB; r++) {
            __m512 d = _mm512_load_ps((const float *)(ws + (size_t)8 * b * IQS_REC + r * bstride + 640));
            for (int c = 0; c < C; c++) f[r][c] = fold(f[r][c], d, A[c * lda + b], I[r][c]);
        }
    }
    for (int r = 0; r < RB; r++)
        for (int c = 0; c < C; c++) _mm512_storeu_ps(out + c * ldo + 16 * r, f[r][c]);
}
#define TF_ARGS const uint8_t *ws, size_t bstride, int nsb, const int8_t *x, size_t ldx, const float *A, size_t lda, \
                const int32_t *c16, size_t ldc, float *out, size_t ldo, int Bi
#define TF(R, C) static void tf_##R##_##C(TF_ARGS) { iqf_tile(R, C, ws, bstride, nsb, x, ldx, A, lda, c16, ldc, out, ldo, Bi); }
TF(1, 1) TF(1, 2) TF(1, 3) TF(1, 4) TF(1, 5) TF(1, 6) TF(2, 1) TF(2, 2) TF(2, 3) TF(2, 4) TF(2, 5) TF(2, 6)
typedef void (*TfFn)(TF_ARGS);
static const TfFn tf_fns[2][6] = { { tf_1_1, tf_1_2, tf_1_3, tf_1_4, tf_1_5, tf_1_6 }, { tf_2_1, tf_2_2, tf_2_3, tf_2_4, tf_2_5, tf_2_6 } };

static void iqf_gemm(int type, float *y, int d, const uint8_t *kp, int k, const Act *a, uint8_t *scr, size_t scr_bytes) {
    int nt = omp_get_max_threads(), ntile = (d / 16 + 1) / 2, nb = a->nb, nsb = k / 256, sbb = iq_sb_bytes(type);
    int nbi = q8p_nb(nb, ntile, nt), nchunk = (nb + nbi - 1) / nbi, Bi = iq_bias(type);
    size_t bstride = (size_t)(k / 32) * IQS_REC;
    #pragma omp parallel
    {
        uint8_t *s = scr + (size_t)omp_get_thread_num() * scr_bytes;
        #pragma omp for schedule(dynamic, 1)
        for (int t = 0; t < ntile; t++) {
            int rb0 = 2 * t, rb = d / 16 - rb0 < 2 ? d / 16 - rb0 : 2;
            iqx_unpack_tile(type, s, kp + (size_t)rb0 * nsb * sbb, rb, k);
            for (int ch = 0; ch < nchunk; ch++) {
                int b1 = (ch + 1) * nbi < nb ? (ch + 1) * nbi : nb;
                for (int b = ch * nbi; b < b1; b += FC) {
                    int c = b1 - b < FC ? b1 - b : FC;
                    tf_fns[rb - 1][c - 1](s, bstride, nsb, a->z + (size_t)b * k, k, a->A + (size_t)b * nsb, nsb,
                                          a->c16 + (size_t)b * (k / 16), k / 16, y + (size_t)b * d + 16 * rb0, d, Bi);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// data

static uint8_t *make_v4h(int d, int k, int extreme) {
    int nsb = k / 256;
    uint16_t *D = malloc(2 * (size_t)d * nsb);
    uint8_t *s = malloc((size_t)d * nsb * 8), *w = aligned_alloc(64, (size_t)d / 16 * nsb * V4H_SBB);
    int8_t *q = malloc((size_t)d * k);
    for (size_t i = 0; i < (size_t)d * nsb; i++) D[i] = (uint16_t)(((rnd() % 7) + 8) << 10 | (rnd() & 0x3ff));
    for (size_t i = 0; i < (size_t)d * nsb * 8; i++) s[i] = extreme ? 15 : (uint8_t)(rnd() % 16);
    for (size_t i = 0; i < (size_t)d * k; i++) q[i] = extreme ? -8 : (int8_t)(rnd() % 16 - 8);
    v4h_pack(w, D, s, q, d, k);
    free(D); free(s); free(q);
    return w;
}

static void make_act(Act *a, int nb, int k, int extreme) {
    a->nb = nb; a->k = k;
    a->z = malloc((size_t)nb * k); a->zu = aligned_alloc(64, (size_t)nb * k + 64);
    a->A = malloc(sizeof(float) * nb * k / 256); a->X = malloc(sizeof(int32_t) * nb * k / 32);
    a->xs = malloc(sizeof(float) * nb * k / 32); a->c16 = malloc(sizeof(int32_t) * ((size_t)nb * k / 16 + 1));
    for (size_t i = 0; i < (size_t)nb * k; i++) {
        a->z[i] = extreme ? (i / k & 1 ? 127 : -127) : (int8_t)(rnd() % 255 - 127); // (extremes: T = +-32512)
        a->zu[i] = (uint8_t)(a->z[i] + 128);
    }
    for (size_t i = 0; i < (size_t)nb * k / 256; i++) a->A[i] = (float)(rnd() % 10000 + 1) * 1e-5f;
    for (size_t i = 0; i < (size_t)nb * k / 32; i++) {
        int32_t x = 0;
        for (int j = 0; j < 32; j++) x += a->z[32 * i + j];
        a->X[i] = x;
        a->xs[i] = (float)(rnd() % 10000 + 1) * 1e-5f;
    }
    for (int b = 0; b < nb; b++) kq512_c16(a->c16 + (size_t)b * (k / 16), a->z + (size_t)b * k, k);
}

static void free_act(Act *a) { free(a->z); free(a->zu); free(a->A); free(a->X); free(a->xs); free(a->c16); }

static uint8_t *make_iq(int type, int d, int k) {
    // random blocks of the base type, packed (repacked for the /r types), as iq_gemm_test
    size_t nblk = (size_t)d * (k / QK_K);
    int base = iq_base(type), bb = iq_block_bytes(base);
    uint8_t *src = malloc(nblk * bb), *kp = aligned_alloc(64, (iq_size(type, d, k) + 63) / 64 * 64);
    for (size_t i = 0; i < nblk * bb; i++) src[i] = (uint8_t)rnd();
    for (size_t i = 0; i < nblk; i++) {
        uint16_t h = (uint16_t)(((rnd() % 7) + 8) << 10 | (rnd() & 0x3ff));
        memcpy(src + i * bb, &h, 2);
    }
    iq_pack_blocks(type, kp, src, d / KQ_ROWS, k);
    free(src);
    return kp;
}

static void iq_run_rows(int type, float *y, int d, const uint8_t *kp, int k, const Act *a) {
    int nblk = d / 16, nt = omp_get_max_threads(), ch = nblk / (8 * nt) > 1 ? nblk / (8 * nt) : 1;
    #pragma omp parallel for schedule(dynamic, ch)
    for (int rb = 0; rb < nblk; rb++) iq_rows(type, y, d, a->z, k, a->xs, a->c16, a->nb, kp, k, rb, rb + 1);
}

static void iq_run_gemm(int type, float *y, int d, const uint8_t *kp, int k, const Act *a, uint8_t *scr, size_t sb) {
    int nt = omp_get_max_threads(), ntile = (d / 16 + 1) / 2, nb = a->nb;
    int nbi = iqx_nb(nb, ntile, nt), nchunk = (nb + nbi - 1) / nbi;
    #pragma omp parallel
    {
        uint8_t *s = scr + (size_t)omp_get_thread_num() * sb;
        const void *last = NULL;
        #pragma omp for schedule(dynamic, 1)
        for (int t = 0; t < ntile; t++)
            for (int c = 0; c < nchunk; c++) iqx_gemm_item(y, d, kp, type, d, t, c, nbi, a->z, a->xs, k, nb, k, a->c16, s, &last);
    }
}

static void v4h_run_rows(int mul, float *y, int d, const uint8_t *w, int k, const Act *a) {
    int nblk = d / 16, nt = omp_get_max_threads(), ch = nblk / (8 * nt) > 1 ? nblk / (8 * nt) : 1;
    #pragma omp parallel for schedule(dynamic, ch)
    for (int rb = 0; rb < nblk; rb++) v4h_rows(mul, y, d, w, k, a, rb, rb + 1);
}

// ---------------------------------------------------------------------------------------------------------------------

static int test(int d, int k, int nb, int extreme) {
    uint8_t *w = make_v4h(d, k, extreme);
    Act a;
    make_act(&a, nb, k, extreme);
    size_t n = (size_t)nb * d, sb = (size_t)2 * (k / 256) * UREC;
    float *y0 = malloc(4 * n), *y1 = malloc(4 * n);
    uint8_t *scr = aligned_alloc(64, sb * omp_get_max_threads());
    v4h_ref(y0, d, w, d, k, a.z, k, a.A, k / 256, nb);
    int fail = 0;
    const char *names[4] = { "rows/dpwssd", "rows/mulld", "tiled-p", "tiled-u" };
    for (int v = 0; v < 4; v++) {
        memset(y1, 0xff, 4 * n);
        if (v < 2) v4h_run_rows(v, y1, d, w, k, &a);
        else v4hx_gemm(v == 3, y1, d, w, k, &a, scr, sb);
        if (memcmp(y0, y1, 4 * n)) {
            size_t i = 0;
            while (i < n && !memcmp(y0 + i, y1 + i, 4)) i++;
            printf("FAIL %-11s d %d k %d nb %d%s: [%zu] %.9g != %.9g\n", names[v], d, k, nb, extreme ? " extreme" : "",
                   i, y1[i], y0[i]);
            fail = 1;
        }
    }
    if (!fail) printf("ok   d %4d k %5d nb %3d%s\n", d, k, nb, extreme ? " (extremes: q -8, s 15, z -127 / 127: |T| 32512)" : "");
    free(w); free(y0); free(y1); free(scr); free_act(&a);
    return fail;
}

static int test_flat(int type, int d, int k, int nb) {
    // the control kernel against iq_rows on the same numbers: xs = A of its superblock, so both compute the same sums,
    // folded per 32 (iq_rows) or per 256 (iqf_gemm): equal up to float rounding
    uint8_t *kp = make_iq(type, d, k);
    Act a;
    make_act(&a, nb, k, 0);
    for (size_t i = 0; i < (size_t)nb * k / 32; i++) a.xs[i] = a.A[i / 8];
    size_t n = (size_t)nb * d, sb = (IQX_SCRATCH(k) + 63) / 64 * 64;
    float *y0 = malloc(4 * n), *y1 = malloc(4 * n);
    uint8_t *scr = aligned_alloc(64, sb * omp_get_max_threads());
    iq_run_rows(type, y0, d, kp, k, &a);
    iqf_gemm(type, y1, d, kp, k, &a, scr, sb);
    double em = 0, ym = 0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs((double)y0[i] - y1[i]);
        em = e > em ? e : em;
        ym = fabs(y0[i]) > ym ? fabs(y0[i]) : ym;
    }
    int fail = !(em <= 1e-5 * ym);
    printf("%s  %-8s tiled-flat vs iq_rows d %d k %d nb %d: max |diff| %.3g of max |y| %.3g\n", fail ? "FAIL" : "ok  ",
           iq_name(type), d, k, nb, em, ym);
    free(kp); free(y0); free(y1); free(scr); free_act(&a);
    return fail;
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

enum { K_IQ3SR_ROWS, K_IQ3SR_TILE, K_IQ4XS_ROWS, K_IQ4XS_TILE, K_IQ3SR_FLAT, K_IQ4XS_FLAT, K_V4H_W, K_V4H_M, K_V4H_P, K_V4H_U, NK };
static const char *knames[NK] = { "IQ3_S/r rows", "IQ3_S/r tiled", "IQ4_XS rows", "IQ4_XS tiled", "IQ3_S/r tiled-flat",
                                  "IQ4_XS tiled-flat", "V4H rows dpwssd",
                                  "V4H rows mulld", "V4H tiled-p", "V4H tiled-u" };

static void bench(int nb, int d, int k, int reps) {
    uint8_t *w = make_v4h(d, k, 0), *i3 = make_iq(GGML_IQ3_S_R, d, k), *i4 = make_iq(GGML_IQ4_XS, d, k);
    Act a;
    make_act(&a, nb, k, 0);
    int nt = omp_get_max_threads();
    size_t sbi = (IQX_SCRATCH(k) + 63) / 64 * 64, sbv = (size_t)2 * (k / 256) * UREC, sb = sbi > sbv ? sbi : sbv;
    uint8_t *scr = aligned_alloc(64, sb * nt);
    float *y = malloc(sizeof(float) * nb * d);
    double best[NK];
    for (int i = 0; i < NK; i++) best[i] = 1e9;
    for (int tr = 0; tr < reps; tr++) // (interleaved: the same background load for every kernel)
        for (int kk = 0; kk < NK; kk++) {
            int tiled = kk == K_IQ3SR_TILE || kk == K_IQ4XS_TILE || kk == K_IQ3SR_FLAT || kk == K_IQ4XS_FLAT || kk == K_V4H_P ||
                        kk == K_V4H_U;
            if (tiled && nb < 4) continue;
            double t0 = now();
            switch (kk) {
            case K_IQ3SR_ROWS: iq_run_rows(GGML_IQ3_S_R, y, d, i3, k, &a); break;
            case K_IQ3SR_TILE: iq_run_gemm(GGML_IQ3_S_R, y, d, i3, k, &a, scr, sb); break;
            case K_IQ4XS_ROWS: iq_run_rows(GGML_IQ4_XS, y, d, i4, k, &a); break;
            case K_IQ4XS_TILE: iq_run_gemm(GGML_IQ4_XS, y, d, i4, k, &a, scr, sb); break;
            case K_IQ3SR_FLAT: iqf_gemm(GGML_IQ3_S_R, y, d, i3, k, &a, scr, sb); break;
            case K_IQ4XS_FLAT: iqf_gemm(GGML_IQ4_XS, y, d, i4, k, &a, scr, sb); break;
            case K_V4H_W: v4h_run_rows(0, y, d, w, k, &a); break;
            case K_V4H_M: v4h_run_rows(1, y, d, w, k, &a); break;
            case K_V4H_P: v4hx_gemm(0, y, d, w, k, &a, scr, sb); break;
            case K_V4H_U: v4hx_gemm(1, y, d, w, k, &a, scr, sb); break;
            }
            double t = now() - t0;
            if (t < best[kk]) best[kk] = t;
        }
    double wn = (double)d * k * nb / nt;
    printf("%5d x %5d nb %4d %2d thr (ns per weight and row x threads; best of %d; ratio vs IQ3_S/r's best path)\n", d, k, nb, nt, reps);
    double ref = best[K_IQ3SR_ROWS] < best[K_IQ3SR_TILE] ? best[K_IQ3SR_ROWS] : best[K_IQ3SR_TILE];
    for (int kk = 0; kk < NK; kk++)
        if (best[kk] < 1e9)
            printf("  %-16s %8.3f ms  %.5f ns  x%.2f\n", knames[kk], 1e3 * best[kk], 1e9 * best[kk] / wn, ref / best[kk]);
    fflush(stdout);
    free(w); free(i3); free(i4); free(scr); free(y); free_act(&a);
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "-b")) {
        int nb = argc > 2 ? atoi(argv[2]) : 512, d = argc > 3 ? atoi(argv[3]) : 17408, k = argc > 4 ? atoi(argv[4]) : 5120;
        int reps = argc > 5 ? atoi(argv[5]) : 5;
        bench(nb, d, k, reps);
        return 0;
    }
    int fails = 0;
    const int nbs[] = { 1, 2, 3, 5, 7, 8, 9, 13, 32, 41, 77 };
    for (size_t j = 0; j < sizeof(nbs) / sizeof(nbs[0]); j++) fails += test(48, 768, nbs[j], 0);
    fails += test(80, 1024, 77, 0);
    fails += test(32, 5120, 9, 0);
    fails += test(48, 2048, 13, 1);
    fails += test(48, 2048, 1, 1);
    fails += test_flat(GGML_IQ3_S_R, 80, 1024, 77) + test_flat(GGML_IQ4_XS, 80, 1024, 77) + test_flat(GGML_IQ3_S_R, 32, 5120, 9);
    printf(fails ? "V4H: %d FAILED\n" : "V4H: all bit-identical to the definition\n", fails);
    return fails != 0;
}
