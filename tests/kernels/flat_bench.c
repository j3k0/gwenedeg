// Flat activations for the i-quants (docs/v4h.md section 9, "what this means for the plan" item 1): a bitwise test and a
// micro-benchmark, before any change to forward.c.
//
// Today (src/iquants.h): activations int8 xq with an fp32 scale xs per 32, one fold per group of 32:
//   val = val + (D * xs[g]) * (float) I_g,  I_g = S_0 T_0 + S_1 T_1
// Flat: one fp32 scale A per 256 activations (llama.cpp's Q8_K), the integer sub-block scales accumulated over the
// super-block, one fold per 256:
//   I = sum_g (S_0 T_0 + S_1 T_1)                                  (int32, exact: |I_g| < 2^23 (IQ4_XS 2^24), |I| < 2^27)
//   val = val + (D * A[s]) * (float) I
//   Q2_K: M = sum_g (m_0 X_0 + m_1 X_1), val = val + ((D * A[s]) * (float) I - (Dm * A[s]) * (float) M)
// each fp32 operation rounded in that order. The codes, D, S, c16 and the decode (iq512_group, iqx_unpack_tile) are the
// existing ones; only the activation scale and where the fold happens change.
//
// Kernels: iqf_dot (the definition, generic C), iqf_blk (AVX-512 rows: GEMV, verify / batched decode, < IQX_MIN rows),
// iqf_tile (AVX-512 tiled GEMM on iqx_unpack_tile's scratch; Q2_K, whose scratch codes carry S and have no bias, chains
// vpdpbusd over the whole super-block and takes its mins at the fold). Test (no arguments): every type, rows and tiled
// bit-identical to iqf_dot, both decodes (VBMI / gather). Benchmark: build/flat_bench -b [NB [D [K [REPS]]]]
// (OMP_NUM_THREADS threads): today's iq_rows / tiled GEMM vs the flat ones per type, and the total weighted by the
// types' shares of ISTA's Qwen3.8-27B IQ3_S file (the 88% of its weights in these types; Q4_K 10.3% and Q6_K 1.6% are
// K-quants, not covered here). Build: make flat-bench.
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

static inline uint32_t xpair(const int32_t *cc) { // X_0 | X_1 << 16 from c16 (= -32 X)
    return (uint32_t)(uint16_t)(-(cc[0] >> 5)) | (uint32_t)(-(cc[1] >> 5)) << 16;
}

typedef struct {
    int nb, k;
    int8_t *xq;   // nb x k
    float *xs;    // nb x k / 32 (today's format)
    float *A;     // nb x k / 256 (flat)
    int32_t *c16; // nb x k / 16
    int32_t *xp;  // nb x k / 32: X_0 | X_1 << 16 of each group (Q2_K's mins; the activation prep would store them)
} Act;

static void make_act(Act *a, int nb, int k, int extreme) {
    a->nb = nb; a->k = k;
    a->xq = malloc((size_t)nb * k);
    a->xs = malloc(sizeof(float) * nb * k / 32);
    a->A = malloc(sizeof(float) * nb * k / 256);
    a->c16 = malloc(sizeof(int32_t) * ((size_t)nb * k / 16 + 1));
    for (size_t i = 0; i < (size_t)nb * k; i++) a->xq[i] = extreme ? (i / k & 1 ? 127 : -127) : (int8_t)(rnd() % 255 - 127);
    for (size_t i = 0; i < (size_t)nb * k / 32; i++) a->xs[i] = (float)(rnd() % 10000 + 1) * 1e-5f;
    for (size_t i = 0; i < (size_t)nb * k / 256; i++) a->A[i] = (float)(rnd() % 10000 + 1) * 1e-5f;
    for (int b = 0; b < nb; b++) kq512_c16(a->c16 + (size_t)b * (k / 16), a->xq + (size_t)b * k, k);
    a->xp = malloc(sizeof(int32_t) * nb * k / 32);
    for (size_t i = 0; i < (size_t)nb * k / 32; i++) a->xp[i] = (int32_t)xpair(a->c16 + 2 * i);
}
static void free_act(Act *a) { free(a->xq); free(a->xs); free(a->A); free(a->c16); free(a->xp); }

// ---------------------------------------------------------------------------------------------------------------------
// the definition

KQ_EXACT static void iqf_dot(int type, float *out, size_t ldo, const uint8_t *wb, int k, int nc, const int8_t *xq,
                             size_t ldx, const float *A, size_t lda) { RQ_FP_EXACT
    float val[KQ_NC][KQ_ROWS];
    memset(val, 0, sizeof(val));
    int sbb = iq_sb_bytes(type), q2 = type == GGML_Q2_K;
    for (int s = 0; s < k / QK_K; s++) {
        const uint8_t *o = wb + (size_t)s * sbb;
        float D[KQ_ROWS], Dm[KQ_ROWS];
        int32_t I[KQ_NC][KQ_ROWS], M[KQ_NC][KQ_ROWS];
        memset(I, 0, sizeof(I));
        memset(M, 0, sizeof(M));
        iq_sb_d(type, o, D, Dm);
        for (int g8 = 0; g8 < 8; g8++) {
            int g = 8 * s + g8;
            int8_t v[KQ_ROWS][32];
            int32_t S[KQ_ROWS][2], Mm[KQ_ROWS][2];
            iq_unpack_group(type, o, g8, v, S, Mm);
            for (int c = 0; c < nc; c++) {
                const int8_t *x = xq + c * ldx + 32 * g;
                int32_t X[2] = { 0, 0 };
                for (int i = 0; i < 16; i++) {
                    X[0] += x[i];
                    X[1] += x[16 + i];
                }
                for (int r = 0; r < KQ_ROWS; r++) {
                    for (int h = 0; h < 2; h++) {
                        int32_t t = 0;
                        for (int i = 0; i < 16; i++) t += v[r][16 * h + i] * x[16 * h + i];
                        I[c][r] += S[r][h] * t;
                    }
                    M[c][r] += Mm[r][0] * X[0] + Mm[r][1] * X[1];
                }
            }
        }
        for (int c = 0; c < nc; c++) {
            float as = A[c * lda + s];
            for (int r = 0; r < KQ_ROWS; r++) {
                if (q2) val[c][r] = val[c][r] + ((D[r] * as) * (float)I[c][r] - (Dm[r] * as) * (float)M[c][r]);
                else val[c][r] = val[c][r] + (D[r] * as) * (float)I[c][r];
            }
        }
    }
    for (int c = 0; c < nc; c++) memcpy(out + c * ldo, val[c], sizeof(val[c]));
}

// ---------------------------------------------------------------------------------------------------------------------
// AVX-512 rows (iq512_blk with the fold moved out of the group loop)

static inline __m512 fold1(__m512 f, __m512 d, float a, __m512i I) {
    return _mm512_add_round_ps(f, _mm512_mul_round_ps(_mm512_mul_round_ps(d, _mm512_set1_ps(a), Q8P_RND),
                                                      _mm512_cvtepi32_ps(I), Q8P_RND), Q8P_RND);
}
static inline __m512 fold2(__m512 f, __m512 d, __m512 dm, float a, __m512i I, __m512i M) { // (Q2_K: a - b, then the add)
    __m512 av = _mm512_set1_ps(a);
    __m512 x = _mm512_mul_round_ps(_mm512_mul_round_ps(d, av, Q8P_RND), _mm512_cvtepi32_ps(I), Q8P_RND);
    __m512 y = _mm512_mul_round_ps(_mm512_mul_round_ps(dm, av, Q8P_RND), _mm512_cvtepi32_ps(M), Q8P_RND);
    return _mm512_add_round_ps(f, _mm512_sub_round_ps(x, y, Q8P_RND), Q8P_RND);
}
static inline __attribute__((always_inline)) void iqf_blk(const int type, const int NC, float *out, size_t ldo,
                                                          const uint8_t *wb, int nsb, int nc, const int8_t *xq, size_t ldx,
                                                          const float *A, size_t lda, const int32_t *c16, size_t ldc,
                                                          const int32_t *xp) {
    const int sbb = iq_sb_bytes(type), Bi = iq_bias(type), q2 = type == GGML_Q2_K, per32 = iqx_per32(type);
    const float cm = iq_dmul(type);
    __m512 val[NC];
    for (int c = 0; c < NC; c++) val[c] = _mm512_setzero_ps();
    for (int s = 0; s < nsb; s++) {
        const uint8_t *o = wb + (size_t)s * sbb;
        __m512 D = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)o));
        if (cm != 1.0f) D = _mm512_mul_round_ps(D, _mm512_set1_ps(cm), Q8P_RND);
        __m512 Dm = q2 ? _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(o + 32))) : _mm512_setzero_ps();
        __m512i I[NC], Mp[8];
        for (int c = 0; c < NC; c++) I[c] = _mm512_setzero_si512();
        for (int g8 = 0; g8 < 8; g8++) {
            __m512i w[8], S0, S1, M0 = _mm512_setzero_si512(), M1 = _mm512_setzero_si512();
            iq512_group(type, o, g8, w, &S0, &S1, &M0, &M1);
            if (q2) Mp[g8] = _mm512_or_si512(M0, _mm512_slli_epi32(M1, 16));
            int g = 8 * s + g8;
            for (int c = 0; c < NC; c++) {
                if (c >= nc) break;
                const int8_t *x = xq + c * ldx + 32 * g;
                const int32_t *cc = c16 + c * ldc + 2 * g;
                __m512i p0 = _mm512_set1_epi32(Bi * (cc[0] >> 5)), p1 = _mm512_set1_epi32(Bi * (cc[1] >> 5));
                for (int t = 0; t < 4; t++) {
                    p0 = _mm512_dpbusd_epi32(p0, w[t], iq512_bc4(x + 4 * t));
                    p1 = _mm512_dpbusd_epi32(p1, w[4 + t], iq512_bc4(x + 16 + 4 * t));
                }
                I[c] = per32 ? _mm512_add_epi32(I[c], _mm512_mullo_epi32(S0, _mm512_add_epi32(p0, p1)))
                             : _mm512_add_epi32(I[c], _mm512_add_epi32(_mm512_mullo_epi32(S0, p0), _mm512_mullo_epi32(S1, p1)));
            }
        }
        for (int c = 0; c < NC && c < nc; c++) {
            if (q2) {
                __m512i M = _mm512_setzero_si512();
                for (int g8 = 0; g8 < 8; g8++)
                    M = _mm512_dpwssd_epi32(M, Mp[g8], _mm512_set1_epi32(xp[c * (ldc / 2) + 8 * s + g8]));
                val[c] = fold2(val[c], D, Dm, A[c * lda + s], I[c], M);
            } else val[c] = fold1(val[c], D, A[c * lda + s], I[c]);
        }
    }
    for (int c = 0; c < nc && c < NC; c++) _mm512_storeu_ps(out + c * ldo, val[c]);
}

#ifndef FR_NC
#define FR_NC 8 // activation rows per pass of the flat rows kernel (I and val: 2 accumulators per row)
#endif
#define IQF_ARGS float *out, size_t ldo, const uint8_t *wb, int nsb, int nc, const int8_t *xq, size_t ldx, const float *A, \
                 size_t lda, const int32_t *c16, size_t ldc, const int32_t *xp
#define IQF_INST(NAME, TYPE) \
    static void NAME##_1(IQF_ARGS) { iqf_blk(TYPE, 1, out, ldo, wb, nsb, nc, xq, ldx, A, lda, c16, ldc, xp); } \
    static void NAME##_8(IQF_ARGS) { iqf_blk(TYPE, FR_NC, out, ldo, wb, nsb, nc, xq, ldx, A, lda, c16, ldc, xp); }
IQF_INST(iqf_q2k, GGML_Q2_K)
IQF_INST(iqf_i2xxs, GGML_IQ2_XXS)
IQF_INST(iqf_i2xs, GGML_IQ2_XS)
IQF_INST(iqf_i2s, GGML_IQ2_S)
IQF_INST(iqf_i3xxs, GGML_IQ3_XXS)
IQF_INST(iqf_i3s, GGML_IQ3_S)
IQF_INST(iqf_i1s, GGML_IQ1_S)
IQF_INST(iqf_i1m, GGML_IQ1_M)
IQF_INST(iqf_i4xs, GGML_IQ4_XS)
IQF_INST(iqf_i3xxsr, GGML_IQ3_XXS_R)
IQF_INST(iqf_i3sr, GGML_IQ3_S_R)
typedef void (*IqfFn)(IQF_ARGS);
static void iqf_fns(int type, IqfFn *f1, IqfFn *f8) {
#define C_(T, N) case T: *f1 = N##_1; *f8 = N##_8; break;
    switch (type) {
    C_(GGML_Q2_K, iqf_q2k) C_(GGML_IQ2_XXS, iqf_i2xxs) C_(GGML_IQ2_XS, iqf_i2xs) C_(GGML_IQ2_S, iqf_i2s)
    C_(GGML_IQ3_XXS, iqf_i3xxs) C_(GGML_IQ3_S, iqf_i3s) C_(GGML_IQ1_S, iqf_i1s) C_(GGML_IQ1_M, iqf_i1m)
    C_(GGML_IQ4_XS, iqf_i4xs) C_(GGML_IQ3_XXS_R, iqf_i3xxsr) C_(GGML_IQ3_S_R, iqf_i3sr)
    default: abort();
    }
#undef C_
}

static void iqf_rows(int type, float *out, size_t ldo, const Act *a, const uint8_t *kp, int k, int rb0, int rb1) {
    // iq_rows with flat activations: 16-row blocks rb0..rb1-1, all nb rows, 8 per pass
    int nsb = k / QK_K, nb = a->nb;
    size_t sbb = (size_t)iq_sb_bytes(type);
    IqfFn f1, f8;
    iqf_fns(type, &f1, &f8);
    for (int rb = rb0; rb < rb1; rb++) {
        const uint8_t *wb = kp + (size_t)rb * nsb * sbb;
        if (nb == 1) {
            f1(out + (size_t)rb * KQ_ROWS, ldo, wb, nsb, 1, a->xq, k, a->A, nsb, a->c16, (size_t)k / 16, a->xp);
            continue;
        }
        for (int b = 0; b < nb; b += FR_NC)
            f8(out + (size_t)b * ldo + (size_t)rb * KQ_ROWS, ldo, wb, nsb, nb - b < FR_NC ? nb - b : FR_NC, a->xq + (size_t)b * k, k,
               a->A + (size_t)b * nsb, nsb, a->c16 + (size_t)b * (k / 16), (size_t)k / 16, a->xp + (size_t)b * (k / 32));
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// AVX-512 tiled GEMM (iqx_tile with I accumulated over the super-block, one fold per 256)

#ifndef FC_P32
#define FC_P32 6 // activation rows per register tile: per-32 scale types (t, I, f: 3 x 2 x C)
#endif
#ifndef FC_TWO
#define FC_TWO 5 // two scales per group
#endif
#ifndef FC_Q2
#define FC_Q2 5 // Q2_K (I, its mins M and f: the codes carry S, no bias)
#endif
#ifndef Q2_AT_FOLD
#define Q2M // Q2_K's mins summed in the group loop (8 threads, 512 rows: 1.28x; at the fold: 1.23x; with xpair computed from
            // c16 at the fold instead of the activation prep's X pairs: 0.98x)
#endif

static inline __attribute__((always_inline))
void iqf_tile(const int RB, const int C, const int P32, const int Q2, const int Bi, const uint8_t *ws, size_t bstride, int nsb,
              const int8_t *x, size_t ldx, const float *A, size_t lda, const int32_t *c16, size_t ldc, const int32_t *xp,
              float *out, size_t ldo) {
    // (Q2_K: xp[c * ldc / 2 + g] = X_0 | X_1 << 16 of group g; Q2M: its mins summed in the group loop, else at the fold)
    const int rec = Q2 ? IQS_Q2 : IQS_REC;
    __m512 f[2][8];
    for (int r = 0; r < RB; r++)
        for (int c = 0; c < C; c++) f[r][c] = _mm512_setzero_ps();
    for (int s = 0; s < nsb; s++) {
        __m512i I[2][8], Ms[2][8];
        for (int r = 0; r < RB; r++)
            for (int c = 0; c < C; c++) I[r][c] = Ms[r][c] = _mm512_setzero_si512();
        for (int g8 = 0; g8 < 8; g8++) {
            const int g = 8 * s + g8;
            const uint8_t *wg = ws + (size_t)g * rec;
            const int8_t *xg = x + (size_t)g * 32;
            if (Q2) { // one vpdpbusd chain over the super-block: the scratch codes are v S, B = 0
                #pragma GCC unroll 8
                for (int st = 0; st < 8; st++) {
                    __m512i w[2];
                    for (int r = 0; r < RB; r++) w[r] = _mm512_load_si512((const void *)(wg + r * bstride + st * 64));
                    for (int c = 0; c < C; c++) {
                        __m512i xv = q8p_bcast4(xg + c * ldx + st * 4);
                        for (int r = 0; r < RB; r++) I[r][c] = _mm512_dpbusd_epi32(I[r][c], w[r], xv);
                    }
                }
#ifdef Q2M
                for (int r = 0; r < RB; r++) {
                    __m512i Mp = _mm512_load_si512((const void *)(wg + r * bstride + 704));
                    for (int c = 0; c < C; c++)
                        Ms[r][c] = _mm512_dpwssd_epi32(Ms[r][c], Mp, _mm512_set1_epi32(xp[c * (ldc / 2) + g]));
                }
#endif
            } else if (P32) {
                __m512i t[2][8];
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
                        __m512i xv = q8p_bcast4(xg + c * ldx + st * 4);
                        for (int r = 0; r < RB; r++) t[r][c] = _mm512_dpbusd_epi32(t[r][c], w[r], xv);
                    }
                }
                for (int r = 0; r < RB; r++) {
                    __m512i sc = _mm512_load_si512((const void *)(wg + r * bstride + 512));
                    for (int c = 0; c < C; c++) I[r][c] = _mm512_add_epi32(I[r][c], _mm512_mullo_epi32(sc, t[r][c]));
                }
            } else {
                for (int h = 0; h < 2; h++) {
                    __m512i t[2][8];
                    for (int c = 0; c < C; c++) {
                        __m512i cv = _mm512_set1_epi32(Bi * (c16[c * ldc + 2 * g + h] >> 5));
                        for (int r = 0; r < RB; r++) t[r][c] = cv;
                    }
                    #pragma GCC unroll 4
                    for (int st = 4 * h; st < 4 * h + 4; st++) {
                        __m512i w[2];
                        for (int r = 0; r < RB; r++) w[r] = _mm512_load_si512((const void *)(wg + r * bstride + st * 64));
                        for (int c = 0; c < C; c++) {
                            __m512i xv = q8p_bcast4(xg + c * ldx + st * 4);
                            for (int r = 0; r < RB; r++) t[r][c] = _mm512_dpbusd_epi32(t[r][c], w[r], xv);
                        }
                    }
                    for (int r = 0; r < RB; r++) {
                        __m512i sc = _mm512_load_si512((const void *)(wg + r * bstride + 512 + 64 * h));
                        for (int c = 0; c < C; c++) I[r][c] = _mm512_add_epi32(I[r][c], _mm512_mullo_epi32(sc, t[r][c]));
                    }
                }
            }
        }
        const uint8_t *w0 = ws + (size_t)8 * s * rec; // (D, Dm: the same in every group record of the super-block)
        for (int r = 0; r < RB; r++) {
            __m512 d = _mm512_load_ps((const float *)(w0 + r * bstride + 640));
            if (Q2) {
                __m512 dm = _mm512_load_ps((const float *)(w0 + r * bstride + 768));
                for (int c = 0; c < C; c++) {
                    __m512i M = Ms[r][c];
#ifndef Q2M
                    for (int g8 = 0; g8 < 8; g8++) {
                        __m512i Mp = _mm512_load_si512((const void *)(w0 + (size_t)g8 * rec + r * bstride + 704));
                        M = _mm512_dpwssd_epi32(M, Mp, _mm512_set1_epi32(xp[c * (ldc / 2) + 8 * s + g8]));
                    }
#endif
                    f[r][c] = fold2(f[r][c], d, dm, A[c * lda + s], I[r][c], M);
                }
            } else
                for (int c = 0; c < C; c++) f[r][c] = fold1(f[r][c], d, A[c * lda + s], I[r][c]);
        }
    }
    for (int r = 0; r < RB; r++)
        for (int c = 0; c < C; c++) _mm512_storeu_ps(out + c * ldo + r * KQ_ROWS, f[r][c]);
}

#define IQFT_ARGS const uint8_t *ws, size_t bstride, int nsb, const int8_t *x, size_t ldx, const float *A, size_t lda, \
                  const int32_t *c16, size_t ldc, const int32_t *xp, float *out, size_t ldo, int Bi
#define IQFT_PASS ws, bstride, nsb, x, ldx, A, lda, c16, ldc, xp, out, ldo
#define IQFT_FN(R, CC) \
    static void iqft_p32_##R##_##CC(IQFT_ARGS) { iqf_tile(R, CC, 1, 0, Bi, IQFT_PASS); } \
    static void iqft_two_##R##_##CC(IQFT_ARGS) { iqf_tile(R, CC, 0, 0, Bi, IQFT_PASS); } \
    static void iqft_q2k_##R##_##CC(IQFT_ARGS) { (void)Bi; iqf_tile(R, CC, 0, 1, 0, IQFT_PASS); }
#define IQFT_FNS(R) IQFT_FN(R, 1) IQFT_FN(R, 2) IQFT_FN(R, 3) IQFT_FN(R, 4) IQFT_FN(R, 5) IQFT_FN(R, 6) IQFT_FN(R, 7) IQFT_FN(R, 8)
IQFT_FNS(1)
IQFT_FNS(2)
typedef void (*IqftFn)(IQFT_ARGS);
#define IQFT_ROW(K, R) { iqft_##K##_##R##_1, iqft_##K##_##R##_2, iqft_##K##_##R##_3, iqft_##K##_##R##_4, iqft_##K##_##R##_5, \
                         iqft_##K##_##R##_6, iqft_##K##_##R##_7, iqft_##K##_##R##_8 }
static const IqftFn iqft_fns[3][2][8] = { { IQFT_ROW(p32, 1), IQFT_ROW(p32, 2) }, { IQFT_ROW(two, 1), IQFT_ROW(two, 2) },
                                          { IQFT_ROW(q2k, 1), IQFT_ROW(q2k, 2) } };

static void iqf_gemm_item(float *out, int ldo, const uint8_t *kp, int type, int d, int t, int ch, int nbi, const Act *a,
                          uint8_t *scr, const void **last) {
    // iqx_gemm_item with the flat tiles
    int k = a->k, nb = a->nb, nsb = k / QK_K, sbb = iq_sb_bytes(type), q2 = type == GGML_Q2_K;
    size_t bstride = (size_t)(k / 32) * (q2 ? IQS_Q2 : IQS_REC);
    int nblk = d / KQ_ROWS, rb0 = 2 * t, rb = nblk - rb0 < 2 ? nblk - rb0 : 2;
    const uint8_t *wt = kp + (size_t)rb0 * nsb * sbb;
    if (*last != wt) {
        iqx_unpack_tile(type, scr, wt, rb, k);
        *last = wt;
    }
    int kind = q2 ? 2 : iqx_per32(type) ? 0 : 1, C = kind == 2 ? FC_Q2 : kind ? FC_TWO : FC_P32, Bi = iq_bias(type);
    int b1 = (ch + 1) * nbi < nb ? (ch + 1) * nbi : nb;
    for (int b = ch * nbi; b < b1; b += C) {
        int c = b1 - b < C ? b1 - b : C;
        iqft_fns[kind][rb - 1][c - 1](scr, bstride, nsb, a->xq + (size_t)b * k, k, a->A + (size_t)b * nsb, nsb,
                                      a->c16 + (size_t)b * (k / 16), k / 16, a->xp + (size_t)b * (k / 32), out + (size_t)b * ldo + (size_t)rb0 * KQ_ROWS,
                                      ldo, Bi);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// drivers (as the forward pass: rows over 16-row blocks, or (32-row tile, chunk) items with the tile's chunks together)

static void run_rows(int flat, int type, float *y, int d, const uint8_t *kp, const Act *a) {
    int nblk = d / KQ_ROWS, nt = omp_get_max_threads(), ch = nblk / (8 * nt) > 1 ? nblk / (8 * nt) : 1, k = a->k;
    #pragma omp parallel for schedule(dynamic, ch)
    for (int rb = 0; rb < nblk; rb++) {
        if (flat) iqf_rows(type, y, d, a, kp, k, rb, rb + 1);
        else iq_rows(type, y, d, a->xq, k, a->xs, a->c16, a->nb, kp, k, rb, rb + 1);
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
                if (flat) iqf_gemm_item(y, d, kp, type, d, t, c, nbi, a, s, &last);
                else iqx_gemm_item(y, d, kp, type, d, t, c, nbi, a->xq, a->xs, k, nb, k, a->c16, s, &last);
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
                    nb - b < KQ_NC ? nb - b : KQ_NC, a.xq + (size_t)b * k, k, a.A + (size_t)b * nsb, nsb);
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
