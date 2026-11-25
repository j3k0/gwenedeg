// --act256 (RqModelOpts.act256: one activation scale per 256; docs/v4h.md sections 10-12): the Q2_K / i-quant kernels of
// src/iquants.h under it, the K-quant (Q3_K .. Q6_K) prototype of PLAN-A256 stage A256.1, bitwise tests and benchmarks.
//
// Test (no arguments): for every i-quant type (the 27B IQ3_S file's 8, IQ1_S, the non-repacked IQ3_S / IQ3_XXS) the
// AVX-512 rows kernel (iqf_rows: GEMV, verify, batched decode) and the tiled GEMM (iqf_gemm_item) are bit-identical to
// the definition iqf_dot, for 1 .. 77 activation rows, k up to 17408, activations at +-127, the VBMI and the gather
// decodes; for every K-quant type the flat rows (kqf_rows) and tiled (kqf_gemm_item) kernels are bit-identical to kqf_dot
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
// K-quants under --act256 (PLAN-A256 stage A256.1: a prototype that decides the K-quant work; the kernels move to
// src/kquants.h / src/kernels_avx512.h if it passes). The definition, kqf_dot, per super-block s of 256 values with the
// activation scale A = xs[8 s] (quantize256), each fp32 operation rounded, super-blocks in order:
//   Q4_K / Q5_K (8 sub-blocks j of 32, 6-bit scale sc_j and min m_j; X_j = the sum of the 32 activations of j):
//     I = sum over j of sc_j * (sum of q * x)        int32, exact: |I| <= 8 * 63 * 15 * 127 * 32 = 30.7M (Q5_K: 31: 63.5M)
//     M = sum over j of m_j * X_j                    int32, exact: |M| <= 8 * 63 * 4064 = 2.05M
//     val = val + ((d * A) * (float) I - (dmin * A) * (float) M)
//   Q6_K (16 sub-blocks of 16, int8 scales) / Q3_K (6-bit scales sc - 32), today's per-group integer I_g summed:
//     I = sum over the 16 sub-blocks of sc_j * (sum of (q - 32) * x)   (Q3_K: (sc_j - 32) * (sum of q * x), q in -4..3)
//         int32, exact: Q6_K |I| <= 16 * 128 * 32 * 127 * 16 = 133.2M (< 2^27), Q3_K 16 * 32 * 4 * 127 * 16 = 4.2M
//     val = val + (d * A) * (float) I
// The integer sums are today's (kquants.h), only the fold moves: 1 instead of 8 per super-block. Q4_K / Q5_K's mins
// term, today dmin * m_j * xf[g] in float per group, becomes the int32 M: kq_act_prep's xf is not needed.
// The AVX-512 kernels: rows (kqf_rows: the small-batch kernels of kernels_avx512.h, the fold per super-block; the GEMV
// is its 1-row instance) and tiled (kqf_gemm_item: kq_unpack_tile's KQS scratch for Q6_K / Q3_K; for Q4_K / Q5_K the
// same codes with the integer scales and the mins as int16 pairs, see kqf_unpack_tile).

static void kqf_unpack(int type, const uint8_t *o, int g8, int32_t w[KQ_ROWS][32], int32_t S[KQ_ROWS][2],
                       int32_t Mn[KQ_ROWS]) {
    // group g8 of a repacked super-block o (Q3P .. Q6P): the signed values w (Q4_K / Q5_K: q; Q6_K: q - 32; Q3_K: q),
    // the scale of each half S (Q4_K / Q5_K: sc_j twice), the min Mn (Q4_K / Q5_K, else 0); kquants.h's layouts
    for (int r = 0; r < KQ_ROWS; r++) {
        Mn[r] = 0;
        if (type == GGML_Q4_K || type == GGML_Q5_K) {
            int q5 = type == GGML_Q5_K;
            const uint8_t *qs = o + 320 + (q5 ? 320 : 256) * g8;
            S[r][0] = S[r][1] = o[64 + 16 * g8 + r];
            Mn[r] = o[192 + 16 * g8 + r];
            for (int i = 0; i < 32; i++) {
                int q = qs[64 * (i / 8) + 4 * r + i % 4] >> (4 * (i % 8 / 4)) & 15;
                if (q5) q |= (int)(kq_u64(qs + 256 + 8 * (i / 4)) >> (4 * r + i % 4) & 1) << 4;
                w[r][i] = q;
            }
        } else if (type == GGML_Q6_K) {
            for (int h = 0; h < 2; h++) {
                int j = 2 * g8 + h;
                const uint8_t *ql = o + 320 + 192 * j, *qh = ql + 128;
                S[r][h] = (int8_t)o[64 + 16 * j + r];
                for (int t = 0; t < 4; t++)
                    for (int b = 0; b < 4; b++)
                        w[r][16 * h + 4 * t + b] = ((ql[64 * (t >> 1) + 4 * r + b] >> (4 * (t & 1)) & 15) |
                                                    (qh[4 * r + b] >> (2 * t) & 3) << 4) - 32;
            }
        } else {
            const uint8_t *qg = o + 256 + 192 * g8;
            for (int h = 0; h < 2; h++) {
                int j = 2 * g8 + h;
                S[r][h] = ((o[64 + 16 * g8 + r] >> (4 * h) & 15) | (o[192 + 16 * (j / 4) + r] >> (2 * (j % 4)) & 3) << 4) - 32;
                for (int t = 0; t < 4; t++)
                    for (int b = 0; b < 4; b++)
                        w[r][16 * h + 4 * t + b] = ((qg[64 * h + 4 * r + b] >> (2 * t) & 3) |
                                                    (int)(kq_u64(qg + 128 + 8 * (4 * h + t)) >> (4 * r + b) & 1) << 2) - 4;
            }
        }
    }
}

KQ_EXACT static void kqf_dot(int type, float *out, size_t ldo, const uint8_t *wb, int k, int nc, const int8_t *xq, size_t ldx,
                             const float *xs, size_t ldxs) { RQ_FP_EXACT
    // the definition (above): kq_dot's arguments without xf; xs[c * ldxs + 8 s] is row c's scale of super-block s
    float val[KQ_NC][KQ_ROWS];
    memset(val, 0, sizeof(val));
    int sbb = kq_sb_bytes(type), mins = type == GGML_Q4_K || type == GGML_Q5_K;
    for (int s = 0; s < k / QK_K; s++) {
        const uint8_t *o = wb + (size_t)s * sbb;
        float d[KQ_ROWS], dmin[KQ_ROWS];
        int32_t I[KQ_NC][KQ_ROWS], M[KQ_NC][KQ_ROWS];
        memset(I, 0, sizeof(I));
        memset(M, 0, sizeof(M));
        for (int r = 0; r < KQ_ROWS; r++) {
            if (mins) {
                d[r] = fp16_to_f32(kq_u16(o + 2 * r));
                dmin[r] = fp16_to_f32(kq_u16(o + 32 + 2 * r));
            } else {
                memcpy(&d[r], o + 4 * r, 4);
                dmin[r] = 0.0f;
            }
        }
        for (int g8 = 0; g8 < 8; g8++) {
            int32_t w[KQ_ROWS][32], S[KQ_ROWS][2], Mn[KQ_ROWS];
            kqf_unpack(type, o, g8, w, S, Mn);
            for (int c = 0; c < nc; c++) {
                const int8_t *x = xq + c * ldx + 32 * (8 * s + g8);
                int32_t X = 0;
                for (int i = 0; i < 32; i++)
                    X += x[i];
                for (int r = 0; r < KQ_ROWS; r++) {
                    for (int h = 0; h < 2; h++) {
                        int32_t t = 0;
                        for (int i = 0; i < 16; i++)
                            t += w[r][16 * h + i] * x[16 * h + i];
                        I[c][r] += S[r][h] * t;
                    }
                    M[c][r] += Mn[r] * X;
                }
            }
        }
        for (int c = 0; c < nc; c++) {
            float a = xs[c * ldxs + 8 * s];
            for (int r = 0; r < KQ_ROWS; r++) {
                if (mins) val[c][r] = val[c][r] + ((d[r] * a) * (float)I[c][r] - (dmin[r] * a) * (float)M[c][r]);
                else val[c][r] = val[c][r] + (d[r] * a) * (float)I[c][r];
            }
        }
    }
    for (int c = 0; c < nc; c++)
        memcpy(out + c * ldo, val[c], sizeof(val[c]));
}

// ---------------------------------------------------------------------------------------------------- AVX-512, rows

static inline uint32_t kqf_xpair(const int32_t *cc) {
    // X_g | X_g+1 << 16 of two groups of 32, from their four c16 (-32 x the sums of 16: exact shifts; |X| <= 4064)
    return (uint32_t)(uint16_t)(-((cc[0] + cc[1]) >> 5)) | (uint32_t)(-((cc[2] + cc[3]) >> 5)) << 16;
}

static inline __attribute__((always_inline))
void kqf45_smb_block(const int C, const int Q5, const uint8_t *wb, int nsb, const int8_t *x, size_t ldx, const float *xs,
                     size_t ldxs, const int32_t *c16, size_t ldc, float *out, size_t ldo, int pf) {
    // one Q4P / Q5P block x C activation rows (kq4_smb_block / kq5_smb_block): I += sc_j * (the group's sum) in int32,
    // the mins by vpdpwssd on (m_j | m_j+1 << 16) x (X_j | X_j+1 << 16), one fold per super-block
    const __m512i m4 = _mm512_set1_epi8(0x0f), m16 = _mm512_set1_epi8(0x10);
    const int sbb = Q5 ? Q5P_SB : Q4P_SB;
    __m512 f[8];
    for (int c = 0; c < C; c++)
        f[c] = _mm512_setzero_ps();
    for (int s = 0; s < nsb; s++) {
        const uint8_t *o = wb + (size_t)s * sbb;
        __m512i I[8];
        for (int c = 0; c < C; c++)
            I[c] = _mm512_setzero_si512();
        for (int j = 0; j < 8; j++) {
            int g = 8 * s + j;
            const uint8_t *qs = o + 320 + (Q5 ? 320 : 256) * j, *mp = qs + 256;
            smb_pf(o + j * (sbb / 8), sbb / 8, pf);
            __m512i w[8];
            for (int cc = 0; cc < 4; cc++) {
                __m512i v = _mm512_load_si512((const void *)(qs + 64 * cc));
                w[2 * cc] = _mm512_and_si512(v, m4);
                w[2 * cc + 1] = _mm512_and_si512(_mm512_srli_epi16(v, 4), m4);
                if (Q5) {
                    w[2 * cc] = _mm512_or_si512(w[2 * cc], _mm512_maskz_mov_epi8((__mmask64)kq_u64(mp + 8 * (2 * cc)), m16));
                    w[2 * cc + 1] = _mm512_or_si512(w[2 * cc + 1],
                                                    _mm512_maskz_mov_epi8((__mmask64)kq_u64(mp + 8 * (2 * cc + 1)), m16));
                }
            }
            __m512i sc = _mm512_cvtepu8_epi32(_mm_load_si128((const __m128i *)(o + 64 + 16 * j)));
            for (int c = 0; c < C; c++) {
                const int8_t *xg = x + (size_t)c * ldx + 32 * g;
                __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
                for (int cc = 0; cc < 4; cc++) {
                    a0 = _mm512_dpbusd_epi32(a0, w[2 * cc], q8p_bcast4(xg + 8 * cc));
                    a1 = _mm512_dpbusd_epi32(a1, w[2 * cc + 1], q8p_bcast4(xg + 8 * cc + 4));
                }
                I[c] = _mm512_add_epi32(I[c], _mm512_mullo_epi32(_mm512_add_epi32(a0, a1), sc));
            }
        }
        __m512 d = _mm512_cvtph_ps(_mm256_load_si256((const __m256i *)o));
        __m512 dm = _mm512_cvtph_ps(_mm256_load_si256((const __m256i *)(o + 32)));
        __m512i mp[4];
        for (int p = 0; p < 4; p++)
            mp[p] = _mm512_or_si512(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i *)(o + 192 + 32 * p))),
                                    _mm512_slli_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i *)(o + 208 + 32 * p))), 16));
        for (int c = 0; c < C; c++) {
            __m512i M = _mm512_setzero_si512();
            for (int p = 0; p < 4; p++)
                M = _mm512_dpwssd_epi32(M, mp[p], _mm512_set1_epi32((int)kqf_xpair(c16 + (size_t)c * ldc + 2 * (8 * s + 2 * p))));
            f[c] = iqf_fold2(f[c], d, dm, xs[(size_t)c * ldxs + 8 * s], I[c], M);
        }
    }
    for (int c = 0; c < C; c++)
        _mm512_storeu_ps(out + (size_t)c * ldo, f[c]);
}

static inline __attribute__((always_inline))
void kqf36_smb_block(const int C, const int Q3, const uint8_t *wb, int nsb, const int8_t *x, size_t ldx, const float *xs,
                     size_t ldxs, const int32_t *c16, size_t ldc, float *out, size_t ldo, int pf) {
    // one Q6P / Q3P block x C activation rows (kq6_smb_block / kq3_smb_block): I over the super-block, one fold
    const int sbb = Q3 ? Q3P_SB : Q6P_SB, div = Q3 ? 8 : 1;
    __m512 f[8];
    for (int c = 0; c < C; c++)
        f[c] = _mm512_setzero_ps();
    for (int s = 0; s < nsb; s++) {
        const uint8_t *o = wb + (size_t)s * sbb;
        __m512i I[8], S0 = _mm512_setzero_si512(); // (Q3_K: the sum of the first-half scales, the 8192 bias)
        for (int c = 0; c < C; c++)
            I[c] = _mm512_setzero_si512();
        for (int g8 = 0; g8 < 8; g8++) {
            int g = 8 * s + g8;
            const uint8_t *qg = o + 256 + 192 * g8;
            smb_pf(o + g8 * (sbb / 8), sbb / 8, pf);
            __m512i s3[2];
            if (Q3) {
                kq3_scales(o, g8, &s3[0], &s3[1]);
                S0 = _mm512_add_epi32(S0, s3[0]);
                __m512i sp = _mm512_or_si512(_mm512_and_si512(s3[0], _mm512_set1_epi32(0xffff)), _mm512_slli_epi32(s3[1], 16));
                __m512i q0[4], q1[4];
                kq3_unpack4(qg, qg + 128, q0);
                kq3_unpack4(qg + 64, qg + 160, q1);
                for (int c = 0; c < C; c++) { // (u = t_0 + 65536 t_1: the tile's packing)
                    const int8_t *xg = x + (size_t)c * ldx + 32 * g;
                    __m512i a = _mm512_set1_epi32(c16[(size_t)c * ldc + 2 * g + 1] / 8), b = _mm512_setzero_si512();
                    a = _mm512_dpbusd_epi32(a, q1[0], q8p_bcast4(xg + 16));
                    b = _mm512_dpbusd_epi32(b, q1[1], q8p_bcast4(xg + 20));
                    a = _mm512_dpbusd_epi32(a, q1[2], q8p_bcast4(xg + 24));
                    b = _mm512_dpbusd_epi32(b, q1[3], q8p_bcast4(xg + 28));
                    __m512i u = _mm512_add_epi32(_mm512_slli_epi32(_mm512_add_epi32(a, b), 16),
                                                 _mm512_set1_epi32(c16[(size_t)c * ldc + 2 * g] / 8 + 8192)), v = _mm512_setzero_si512();
                    u = _mm512_dpbusd_epi32(u, q0[0], q8p_bcast4(xg));
                    v = _mm512_dpbusd_epi32(v, q0[1], q8p_bcast4(xg + 4));
                    u = _mm512_dpbusd_epi32(u, q0[2], q8p_bcast4(xg + 8));
                    v = _mm512_dpbusd_epi32(v, q0[3], q8p_bcast4(xg + 12));
                    I[c] = _mm512_dpwssd_epi32(I[c], _mm512_add_epi32(u, v), sp);
                }
                continue;
            }
            for (int h = 0; h < 2; h++) {
                int j = 2 * g8 + h;
                __m512i q[4], sc;
                if (Q3) {
                    kq3_unpack4(qg + 64 * h, qg + 128 + 32 * h, q);
                    sc = s3[h];
                } else {
                    kq6_unpack4(o + 320 + 192 * j, o + 448 + 192 * j, q);
                    sc = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i *)(o + 64 + 16 * j)));
                }
                for (int c = 0; c < C; c++) {
                    const int8_t *xg = x + (size_t)c * ldx + 32 * g + 16 * h;
                    __m512i t0 = _mm512_set1_epi32(c16[(size_t)c * ldc + 2 * g + h] / div), t1 = _mm512_setzero_si512();
                    t0 = _mm512_dpbusd_epi32(t0, q[0], q8p_bcast4(xg));
                    t1 = _mm512_dpbusd_epi32(t1, q[1], q8p_bcast4(xg + 4));
                    t0 = _mm512_dpbusd_epi32(t0, q[2], q8p_bcast4(xg + 8));
                    t1 = _mm512_dpbusd_epi32(t1, q[3], q8p_bcast4(xg + 12));
                    I[c] = _mm512_add_epi32(I[c], _mm512_mullo_epi32(_mm512_add_epi32(t0, t1), sc));
                }
            }
        }
        __m512 d = _mm512_load_ps((const float *)o);
        if (Q3)
            for (int c = 0; c < C; c++)
                I[c] = _mm512_sub_epi32(I[c], _mm512_slli_epi32(S0, 13));
        for (int c = 0; c < C; c++)
            f[c] = iqf_fold(f[c], d, xs[(size_t)c * ldxs + 8 * s], I[c]);
    }
    for (int c = 0; c < C; c++)
        _mm512_storeu_ps(out + (size_t)c * ldo, f[c]);
}

#define KQF_SMB_ARGS const uint8_t *wb, int nsb, const int8_t *x, size_t ldx, const float *xs, size_t ldxs, const int32_t *c16, \
                     size_t ldc, float *out, size_t ldo, int pf
#define KQF_SMB_PASS wb, nsb, x, ldx, xs, ldxs, c16, ldc, out, ldo, pf
#define KQF_SMB_FN(C) \
    static void kqf_smb4_##C(KQF_SMB_ARGS) { kqf45_smb_block(C, 0, KQF_SMB_PASS); } \
    static void kqf_smb5_##C(KQF_SMB_ARGS) { kqf45_smb_block(C, 1, KQF_SMB_PASS); } \
    static void kqf_smb6_##C(KQF_SMB_ARGS) { kqf36_smb_block(C, 0, KQF_SMB_PASS); } \
    static void kqf_smb3_##C(KQF_SMB_ARGS) { kqf36_smb_block(C, 1, KQF_SMB_PASS); }
KQF_SMB_FN(1) KQF_SMB_FN(2) KQF_SMB_FN(3) KQF_SMB_FN(4) KQF_SMB_FN(5) KQF_SMB_FN(6) KQF_SMB_FN(7) KQF_SMB_FN(8)
typedef void (*KqfSmbFn)(KQF_SMB_ARGS);
#define KQF_SMB_ROW(T) { kqf_smb##T##_1, kqf_smb##T##_2, kqf_smb##T##_3, kqf_smb##T##_4, kqf_smb##T##_5, kqf_smb##T##_6, \
                         kqf_smb##T##_7, kqf_smb##T##_8 }
static const KqfSmbFn kqf_smb_fns[4][8] = { KQF_SMB_ROW(4), KQF_SMB_ROW(5), KQF_SMB_ROW(6), KQF_SMB_ROW(3) };
#undef KQF_SMB_ROW
#undef KQF_SMB_FN
#undef KQF_SMB_PASS

static inline int kqf_kind(int type) { return type == GGML_Q4_K ? 0 : type == GGML_Q5_K ? 1 : type == GGML_Q6_K ? 2 : 3; }

static void kqf_rows(int type, float *out, size_t ldo, const int8_t *xq, size_t ldx, const float *xs, const int32_t *c16,
                     int nb, const uint8_t *kp, int k, int rb0, int rb1, int pf) {
    // smb512_rows under --act256 (blocks rb0..rb1-1, activation rows 0..nb-1, 8 per pass; nb = 1: the GEMV)
    int nsb = k / QK_K, n16 = k / 16;
    size_t bstride = (size_t)nsb * kq_sb_bytes(type), ldxs = ldx / 32;
    const KqfSmbFn *fn = kqf_smb_fns[kqf_kind(type)];
    for (int rb = rb0; rb < rb1; rb++)
        for (int b = 0; b < nb; b += 8) {
            int c = nb - b < 8 ? nb - b : 8;
            fn[c - 1](kp + (size_t)rb * bstride, nsb, xq + (size_t)b * ldx, ldx, xs + (size_t)b * ldxs, ldxs,
                      c16 + (size_t)b * n16, n16, out + (size_t)b * ldo + (size_t)rb * KQ_ROWS, ldo, pf);
        }
}

// ---------------------------------------------------------------------------------------------------- AVX-512, tiled

static void kqf_unpack_tile(int type, uint8_t *scr, const uint8_t *wp, int nblk, int k) {
    // kq_unpack_tile under --act256. Q6_K / Q3_K: its KQS_Q6 records as they are (q, sc_j0, sc_j1, d). Q4_K / Q5_K:
    // KQS_Q4 records of 640 bytes per (block, group j): +0 the 8 x 64 codes (kq_unpack_tile's), +512 sc_j (int32),
    // +576 per super-block: j = 0, 2, 4, 6: the mins m_j | m_j+1 << 16 (int16 pairs); j = 1: d, j = 3: dmin (fp32)
    if (type == GGML_Q6_K || type == GGML_Q3_K) {
        kq_unpack_tile(type, scr, wp, nblk, k);
        if (type == GGML_Q3_K) // (+512: the scales sc - 32 of the group's two sub-blocks as one int16 pair; +576 of the
                               // super-block's first record: -8192 x the sum of its 8 first-half scales, kqf36_tile's I start)
            for (size_t sb = 0; sb < (size_t)nblk * (k / QK_K); sb++) {
                __m512i S0 = _mm512_setzero_si512();
                for (int g8 = 0; g8 < 8; g8++) {
                    uint8_t *dst = scr + (8 * sb + g8) * KQS_Q6;
                    __m512i s0 = _mm512_load_si512((const void *)(dst + 512)), s1 = _mm512_load_si512((const void *)(dst + 576));
                    S0 = _mm512_add_epi32(S0, s0);
                    _mm512_store_si512((void *)(dst + 512), _mm512_or_si512(_mm512_and_si512(s0, _mm512_set1_epi32(0xffff)),
                                                                            _mm512_slli_epi32(s1, 16)));
                }
                _mm512_store_si512((void *)(scr + 8 * sb * KQS_Q6 + 576), _mm512_sub_epi32(_mm512_setzero_si512(),
                                                                                          _mm512_slli_epi32(S0, 13)));
            }
        return;
    }
    int nsb = k / QK_K, ng = k / 32, q5 = type == GGML_Q5_K, sbb = q5 ? Q5P_SB : Q4P_SB;
    const __m512i m4 = _mm512_set1_epi8(0x0f), m16 = _mm512_set1_epi8(0x10);
    for (int r = 0; r < nblk; r++)
        for (int s = 0; s < nsb; s++) {
            const uint8_t *o = wp + ((size_t)r * nsb + s) * sbb;
            for (int j = 0; j < 8; j++) {
                uint8_t *dst = scr + ((size_t)r * ng + 8 * s + j) * KQS_Q4;
                const uint8_t *qs = o + 320 + (q5 ? 320 : 256) * j, *mp = qs + 256;
                for (int c = 0; c < 4; c++) {
                    __m512i v = _mm512_load_si512((const void *)(qs + 64 * c));
                    __m512i lo = _mm512_and_si512(v, m4), hi = _mm512_and_si512(_mm512_srli_epi16(v, 4), m4);
                    if (q5) {
                        lo = _mm512_or_si512(lo, _mm512_maskz_mov_epi8((__mmask64)kq_u64(mp + 8 * (2 * c)), m16));
                        hi = _mm512_or_si512(hi, _mm512_maskz_mov_epi8((__mmask64)kq_u64(mp + 8 * (2 * c + 1)), m16));
                    }
                    _mm512_store_si512((void *)(dst + 128 * c), lo);
                    _mm512_store_si512((void *)(dst + 128 * c + 64), hi);
                }
                _mm512_store_si512((void *)(dst + 512), _mm512_cvtepu8_epi32(_mm_load_si128((const __m128i *)(o + 64 + 16 * j))));
                __m512i aux;
                if (!(j & 1))
                    aux = _mm512_or_si512(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i *)(o + 192 + 16 * j))),
                                          _mm512_slli_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i *)(o + 208 + 16 * j))), 16));
                else if (j == 1)
                    aux = _mm512_castps_si512(_mm512_cvtph_ps(_mm256_load_si256((const __m256i *)o)));
                else if (j == 3)
                    aux = _mm512_castps_si512(_mm512_cvtph_ps(_mm256_load_si256((const __m256i *)(o + 32))));
                else
                    aux = _mm512_setzero_si512();
                _mm512_store_si512((void *)(dst + 576), aux);
            }
        }
}

static inline __attribute__((always_inline))
void kqf45_tile(const int RB, const int C, const uint8_t *ws, size_t bstride, int nsb, const int8_t *x, size_t ldx,
                const float *xs, size_t ldxs, const int32_t *c16, size_t ldc, float *out, size_t ldo) {
    // kq45_tile under --act256 on kqf_unpack_tile's records: per group one vpdpbusd chain (8 steps), I += sc * it;
    // per super-block the mins (4 vpdpwssd) and one fold
    __m512 f[KQ_RB][8];
    for (int r = 0; r < RB; r++)
        for (int c = 0; c < C; c++)
            f[r][c] = _mm512_setzero_ps();
    for (int s = 0; s < nsb; s++) {
        __m512i I[KQ_RB][8];
        for (int r = 0; r < RB; r++)
            for (int c = 0; c < C; c++)
                I[r][c] = _mm512_setzero_si512();
        for (int g8 = 0; g8 < 8; g8++) {
            const int g = 8 * s + g8;
            const uint8_t *wg = ws + (size_t)g * KQS_Q4;
            const int8_t *xg = x + (size_t)g * 32;
            __m512i acc[KQ_RB][8];
            for (int r = 0; r < RB; r++)
                for (int c = 0; c < C; c++)
                    acc[r][c] = _mm512_setzero_si512();
            #pragma GCC unroll 8
            for (int st = 0; st < 8; st++) {
                __m512i w[KQ_RB];
                for (int r = 0; r < RB; r++)
                    w[r] = _mm512_load_si512((const void *)(wg + r * bstride + st * 64));
                for (int c = 0; c < C; c++) {
                    __m512i xv = q8p_bcast4(xg + c * ldx + st * 4);
                    for (int r = 0; r < RB; r++)
                        acc[r][c] = _mm512_dpbusd_epi32(acc[r][c], w[r], xv);
                }
            }
            for (int r = 0; r < RB; r++) {
                __m512i sc = _mm512_load_si512((const void *)(wg + r * bstride + 512));
                for (int c = 0; c < C; c++)
                    I[r][c] = _mm512_add_epi32(I[r][c], _mm512_mullo_epi32(acc[r][c], sc));
            }
        }
        int32_t xp[8][4];
        for (int c = 0; c < C; c++)
            for (int p = 0; p < 4; p++)
                xp[c][p] = (int32_t)kqf_xpair(c16 + c * ldc + 2 * (8 * s + 2 * p));
        const uint8_t *w0 = ws + (size_t)8 * s * KQS_Q4 + 576;
        for (int r = 0; r < RB; r++) {
            __m512 d = _mm512_load_ps((const float *)(w0 + r * bstride + KQS_Q4));
            __m512 dm = _mm512_load_ps((const float *)(w0 + r * bstride + 3 * KQS_Q4));
            __m512i mp[4];
            for (int p = 0; p < 4; p++)
                mp[p] = _mm512_load_si512((const void *)(w0 + r * bstride + 2 * p * KQS_Q4));
            for (int c = 0; c < C; c++) {
                __m512i M = _mm512_setzero_si512();
                for (int p = 0; p < 4; p++)
                    M = _mm512_dpwssd_epi32(M, mp[p], _mm512_set1_epi32(xp[c][p]));
                f[r][c] = iqf_fold2(f[r][c], d, dm, xs[c * ldxs + 8 * s], I[r][c], M);
            }
        }
    }
    for (int r = 0; r < RB; r++)
        for (int c = 0; c < C; c++)
            _mm512_storeu_ps(out + c * ldo + r * KQ_ROWS, f[r][c]);
}

static inline __attribute__((always_inline))
void kqf36_tile(const int RB, const int C, const int DIV, const uint8_t *ws, size_t bstride, int nsb, const int8_t *x,
                size_t ldx, const float *xs, size_t ldxs, const int32_t *c16, size_t ldc, float *out, size_t ldo) {
    // kq6_tile under --act256 (the same scratch; Q3_K: DIV 8): I over the super-block, one fold
    __m512 f[KQ_RB][8];
    for (int r = 0; r < RB; r++)
        for (int c = 0; c < C; c++)
            f[r][c] = _mm512_setzero_ps();
    for (int s = 0; s < nsb; s++) {
        __m512i I[KQ_RB][8];
        for (int r = 0; r < RB; r++) {
            __m512i i0 = DIV == 8 ? _mm512_load_si512((const void *)(ws + (size_t)8 * s * KQS_Q6 + r * bstride + 576))
                                  : _mm512_setzero_si512();
            for (int c = 0; c < C; c++)
                I[r][c] = i0;
        }
        for (int g8 = 0; g8 < 8; g8++) {
            const int g = 8 * s + g8;
            const uint8_t *wg = ws + (size_t)g * KQS_Q6;
            const int8_t *xg = x + (size_t)g * 32;
            if (DIV == 8) { // Q3_K: |t_h| <= 4 * 127 * 16 = 8128, so u = (t_0 + 8192) + 65536 t_1 holds t_0 + 8192 (0..16320)
                            // and t_1 as its int16 halves: one vpdpwssd with the scale pair (the 8192 s taken off at the start)
                __m512i u[KQ_RB][8];
                for (int h = 1; h >= 0; h--) {
                    for (int c = 0; c < C; c++) {
                        __m512i cv = _mm512_set1_epi32(c16[c * ldc + 2 * g + h] / 8 + (h ? 0 : 8192));
                        for (int r = 0; r < RB; r++)
                            u[r][c] = h ? cv : _mm512_add_epi32(_mm512_slli_epi32(u[r][c], 16), cv);
                    }
                    #pragma GCC unroll 4
                    for (int st = 4 * h; st < 4 * h + 4; st++) {
                        __m512i w[KQ_RB];
                        for (int r = 0; r < RB; r++)
                            w[r] = _mm512_load_si512((const void *)(wg + r * bstride + st * 64));
                        for (int c = 0; c < C; c++) {
                            __m512i xv = q8p_bcast4(xg + c * ldx + st * 4);
                            for (int r = 0; r < RB; r++)
                                u[r][c] = _mm512_dpbusd_epi32(u[r][c], w[r], xv);
                        }
                    }
                }
                for (int r = 0; r < RB; r++) {
                    __m512i sp = _mm512_load_si512((const void *)(wg + r * bstride + 512));
                    for (int c = 0; c < C; c++)
                        I[r][c] = _mm512_dpwssd_epi32(I[r][c], u[r][c], sp);
                }
                continue;
            }
            for (int h = 0; h < 2; h++) {
                __m512i t[KQ_RB][8];
                for (int c = 0; c < C; c++) {
                    __m512i cv = _mm512_set1_epi32(c16[c * ldc + 2 * g + h] / DIV);
                    for (int r = 0; r < RB; r++)
                        t[r][c] = cv;
                }
                #pragma GCC unroll 4
                for (int st = 4 * h; st < 4 * h + 4; st++) {
                    __m512i w[KQ_RB];
                    for (int r = 0; r < RB; r++)
                        w[r] = _mm512_load_si512((const void *)(wg + r * bstride + st * 64));
                    for (int c = 0; c < C; c++) {
                        __m512i xv = q8p_bcast4(xg + c * ldx + st * 4);
                        for (int r = 0; r < RB; r++)
                            t[r][c] = _mm512_dpbusd_epi32(t[r][c], w[r], xv);
                    }
                }
                for (int r = 0; r < RB; r++) {
                    __m512i sc = _mm512_load_si512((const void *)(wg + r * bstride + 512 + 64 * h));
                    for (int c = 0; c < C; c++)
                        I[r][c] = _mm512_add_epi32(I[r][c], _mm512_mullo_epi32(t[r][c], sc));
                }
            }
        }
        const uint8_t *w0 = ws + (size_t)8 * s * KQS_Q6; // (d: the same in every group record of the super-block)
        for (int r = 0; r < RB; r++) {
            __m512 d = _mm512_load_ps((const float *)(w0 + r * bstride + 640));
            for (int c = 0; c < C; c++)
                f[r][c] = iqf_fold(f[r][c], d, xs[c * ldxs + 8 * s], I[r][c]);
        }
    }
    for (int r = 0; r < RB; r++)
        for (int c = 0; c < C; c++)
            _mm512_storeu_ps(out + c * ldo + r * KQ_ROWS, f[r][c]);
}

#ifndef KQF4_C
#define KQF4_C 4 // activation rows per register tile, Q4_K / Q5_K (f, I, the group sums; 3 / 4 / 5 / 6: 0.99 / 1.06 / 0.99 / 0.99x)
#endif
#ifndef KQF6_C
#define KQF6_C 6 // the same, Q6_K (f, I, t: spills a little; 1 thread, 512 rows: C 3 / 4 / 5 / 6 1.09 / 0.88 / 1.17 / 1.19x)
#endif
#ifndef KQF3_C
#define KQF3_C 5 // the same, Q3_K (f, I, u; 3 / 4 / 5 / 6: 1.09 / 1.22 / 1.26 / 1.26x)
#endif

#define KQFT_ARGS const uint8_t *ws, size_t bstride, int nsb, const int8_t *x, size_t ldx, const float *xs, size_t ldxs, \
                  const int32_t *c16, size_t ldc, float *out, size_t ldo
#define KQFT_PASS ws, bstride, nsb, x, ldx, xs, ldxs, c16, ldc, out, ldo
#define KQFT_FN(R, CC) \
    static void kqft_45_##R##_##CC(KQFT_ARGS) { kqf45_tile(R, CC, KQFT_PASS); } \
    static void kqft_6_##R##_##CC(KQFT_ARGS) { kqf36_tile(R, CC, 1, KQFT_PASS); } \
    static void kqft_3_##R##_##CC(KQFT_ARGS) { kqf36_tile(R, CC, 8, KQFT_PASS); }
#define KQFT_FNS(R) KQFT_FN(R, 1) KQFT_FN(R, 2) KQFT_FN(R, 3) KQFT_FN(R, 4) KQFT_FN(R, 5) KQFT_FN(R, 6) KQFT_FN(R, 7) KQFT_FN(R, 8)
KQFT_FNS(1)
KQFT_FNS(2)
typedef void (*KqftFn)(KQFT_ARGS);
#define KQFT_ROW(K, R) { kqft_##K##_##R##_1, kqft_##K##_##R##_2, kqft_##K##_##R##_3, kqft_##K##_##R##_4, kqft_##K##_##R##_5, \
                         kqft_##K##_##R##_6, kqft_##K##_##R##_7, kqft_##K##_##R##_8 }
static const KqftFn kqft_fns[3][2][8] = { { KQFT_ROW(45, 1), KQFT_ROW(45, 2) }, { KQFT_ROW(6, 1), KQFT_ROW(6, 2) },
                                          { KQFT_ROW(3, 1), KQFT_ROW(3, 2) } };
#undef KQFT_ROW
#undef KQFT_FNS
#undef KQFT_FN
#undef KQFT_PASS

static void kqf_gemm_item(float *out, int ldo, const uint8_t *kp, int type, int d, int t, int ch, int nbi,
                          const int8_t *xq, const float *xs, int ldx, int nb, int k, const int32_t *c16, uint8_t *scr,
                          const void **last) {
    // kq512_gemm_item under --act256 (the same work items and scratch size; no xf)
    int ng = k / 32, n16 = k / 16, sbb = kq_sb_bytes(type), q36 = type == GGML_Q6_K || type == GGML_Q3_K;
    size_t ldxs = (size_t)ldx / 32, bstride = (size_t)ng * (q36 ? KQS_Q6 : KQS_Q4);
    int nblk = d / KQ_ROWS, rb0 = t * KQ_RB, rb = nblk - rb0 < KQ_RB ? nblk - rb0 : KQ_RB;
    const uint8_t *wt = kp + (size_t)rb0 * (k / QK_K) * sbb;
    if (*last != wt) {
        kqf_unpack_tile(type, scr, wt, rb, k);
        *last = wt;
    }
    int kind = type == GGML_Q6_K ? 1 : type == GGML_Q3_K ? 2 : 0, C = kind == 2 ? KQF3_C : kind ? KQF6_C : KQF4_C;
    int b1 = (ch + 1) * nbi < nb ? (ch + 1) * nbi : nb;
    for (int b = ch * nbi; b < b1; b += C) {
        int c = b1 - b < C ? b1 - b : C;
        kqft_fns[kind][rb - 1][c - 1](scr, bstride, ng / 8, xq + (size_t)b * ldx, ldx, xs + (size_t)b * ldxs, ldxs,
                                      c16 + (size_t)b * n16, n16, out + (size_t)b * ldo + (size_t)rb0 * KQ_ROWS, ldo);
    }
}

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
        if (flat) kqf_rows(type, y, d, a->xq, k, a->xsf, a->c16, nb, kp, k, rb, rb + 1, 4096);
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
                if (flat) kqf_gemm_item(y, d, kp, type, d, t, c, nbi, a->xq, a->xsf, k, nb, k, a->c16, s, &last);
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
