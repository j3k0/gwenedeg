/* AVX-512 VNNI Q8_0 kernels for runq.c (GEMM for the batched prefill, GEMV for decode).
 *
 * Enabled when the compiler targets AVX-512 F/BW/VNNI (e.g. -march=native on Zen 4 / Ice Lake+)
 * and RUNQ_GENERIC is not defined; it then defines RUNQ_VNNI. Otherwise this header is empty and
 * runq.c uses its generic C kernels (dot_q8, dot_q8_x4).
 *
 * Weights are repacked once at load ("Q8P" layout) so that the 16 dword lanes of a zmm hold 16
 * different weight rows: no horizontal reductions at all. For a (d, k) Q8_0 matrix (d % 16 == 0,
 * k % 64 == 0, GS == 64), block rb = rows 16*rb .. 16*rb+15, group g = columns 64*g .. 64*g+63:
 *   pk + (rb * k/64 + g) * Q8P_GROUP:
 *     16 steps st of 64 bytes: lane r, byte j = (uint8)(W[16*rb + r][64*g + 4*st + j] + 128)
 *     then 16 floats: the scales ws[16*rb + r][g]
 * 1088 bytes per (block, group), the same size as the original (q, s) data. One block (all k) is
 * contiguous, so a GEMV streams each thread's rows as one sequential read.
 *
 * vpdpbusd multiplies u8 x s8: the weights are stored offset by +128 (u8), the activations stay
 * s8 and are broadcast 4 bytes at a time; the offset is removed exactly with the activation
 * group sums: isum = sum((w+128) * x) - 128 * sum(x) (accumulators start at -128 * sum(x)).
 * The per-group int32 sums are exact, and the float fold is the one of dot_q8 in runq.c, per
 * output and group in order: val = val + (ws * xs) * (float) isum, rounded after every operation
 * (explicit-rounding intrinsics, which gcc can neither contract into an FMA nor reassociate under
 * -Ofast). So the results are bit-identical to the generic kernels.
 */
#ifndef RUNQ_KERNELS_AVX512_H
#define RUNQ_KERNELS_AVX512_H

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__) && !defined(RUNQ_GENERIC)
#define RUNQ_VNNI 1
#include <immintrin.h>

#define Q8P_GS 64
#define Q8P_ROWS 16
#define Q8P_GROUP (Q8P_ROWS * Q8P_GS + Q8P_ROWS * 4) // bytes per (block, group): 1024 + 64

// GEMM register tile: Q8P_RB blocks of 16 weight rows x Q8P_C activation rows; 2 * RB * C zmm
// accumulators (int32 group sums + float outputs) + RB weight registers (<= 32)
#ifndef Q8P_C
#define Q8P_C 6
#endif
#if Q8P_C < 1 || Q8P_C > 8
#error "Q8P_C must be in 1..8"
#endif
#ifndef Q8P_RB
#define Q8P_RB 2 // (1 only to test the odd-block-count tail: every block count of Qwen3-0.6B is even)
#endif
#if Q8P_RB < 1 || Q8P_RB > 2
#error "Q8P_RB must be 1 or 2"
#endif
// activation rows per GEMM work item (the weight tile is reused for these from L1/L2)
#ifndef Q8P_NB
#define Q8P_NB 64
#endif

#define Q8P_RND (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)

static inline size_t q8p_size(int d, int k) { return (size_t)(d / Q8P_ROWS) * (k / Q8P_GS) * Q8P_GROUP; }

static inline int q8p_ok(int d, int k) { return d > 0 && k > 0 && d % Q8P_ROWS == 0 && k % Q8P_GS == 0; }

static void q8p_pack(uint8_t *pk, const int8_t *q, const float *s, int d, int k) {
    // (q, s): row-major Q8_0 (d, k) with k/64 scales per row -> Q8P layout at pk (64-byte aligned)
    int ng = k / Q8P_GS;
    #pragma omp parallel for schedule(static)
    for (int rb = 0; rb < d / Q8P_ROWS; rb++) {
        for (int g = 0; g < ng; g++) {
            uint8_t *o = pk + ((size_t)rb * ng + g) * Q8P_GROUP;
            for (int r = 0; r < Q8P_ROWS; r++) {
                const int8_t *src = q + (size_t)(rb * Q8P_ROWS + r) * k + g * Q8P_GS;
                for (int st = 0; st < Q8P_GS / 4; st++) {
                    uint32_t v;
                    memcpy(&v, src + 4 * st, 4);
                    v ^= 0x80808080u; // int8 w -> uint8 w + 128
                    memcpy(o + st * 64 + r * 4, &v, 4);
                }
                memcpy(o + Q8P_ROWS * Q8P_GS + r * 4, s + (size_t)(rb * Q8P_ROWS + r) * ng + g, 4);
            }
        }
    }
}

static inline int32_t q8p_corr(const int8_t *x) {
    // -128 * (sum of the 64 int8 values at x): the accumulator start removing the +128 weight offset
    __m512i u = _mm512_xor_si512(_mm512_loadu_si512(x), _mm512_set1_epi8((char)0x80)); // x + 128
    __m512i sad = _mm512_sad_epu8(u, _mm512_setzero_si512());
    int32_t sum = (int32_t)_mm512_reduce_add_epi64(sad) - Q8P_GS * 128;
    return -128 * sum;
}

static inline __m512i q8p_bcast4(const int8_t *p) {
    int32_t v;
    memcpy(&v, p, 4);
    return _mm512_set1_epi32(v);
}

static inline __m512 q8p_fold(__m512 acc, __m512 ws, float xs, __m512i isum) {
    // acc + (ws * xs) * (float) isum, each operation rounded (as dot_q8 in runq.c)
    __m512 sc = _mm512_mul_round_ps(ws, _mm512_set1_ps(xs), Q8P_RND);
    __m512 p = _mm512_mul_round_ps(sc, _mm512_cvtepi32_ps(isum), Q8P_RND); // exact: |isum| < 2^24
    return _mm512_add_round_ps(acc, p, Q8P_RND);
}

static inline __attribute__((always_inline))
void q8p_tile(const int RB, const int C, const uint8_t *wp, size_t bstride, int ng,
              const int8_t *x, size_t ldx, const float *xs, size_t ldxs, const int32_t *corr,
              float *out, size_t ldo) {
    // out[c * ldo + 16 * r + lane] for r < RB blocks (at wp, wp + bstride), c < C activation rows
    // (x + c * ldx, scales xs + c * ldxs, corrections corr + c * ng), over all ng groups
    __m512 f[Q8P_RB][Q8P_C];
    for (int r = 0; r < RB; r++)
        for (int c = 0; c < C; c++)
            f[r][c] = _mm512_setzero_ps();
    for (int g = 0; g < ng; g++) {
        const uint8_t *wg = wp + (size_t)g * Q8P_GROUP;
        const int8_t *xg = x + (size_t)g * Q8P_GS;
        __m512i acc[Q8P_RB][Q8P_C];
        for (int c = 0; c < C; c++) {
            __m512i cv = _mm512_set1_epi32(corr[(size_t)c * ng + g]);
            for (int r = 0; r < RB; r++)
                acc[r][c] = cv;
        }
        #pragma GCC unroll 16
        for (int st = 0; st < Q8P_GS / 4; st++) {
            __m512i w[Q8P_RB];
            for (int r = 0; r < RB; r++)
                w[r] = _mm512_load_si512((const void *)(wg + r * bstride + st * 64));
            for (int c = 0; c < C; c++) {
                __m512i xv = q8p_bcast4(xg + c * ldx + st * 4);
                for (int r = 0; r < RB; r++)
                    acc[r][c] = _mm512_dpbusd_epi32(acc[r][c], w[r], xv);
            }
        }
        for (int r = 0; r < RB; r++) {
            __m512 ws = _mm512_load_ps((const float *)(wg + r * bstride + Q8P_ROWS * Q8P_GS));
            for (int c = 0; c < C; c++)
                f[r][c] = q8p_fold(f[r][c], ws, xs[c * ldxs + g], acc[r][c]);
        }
    }
    for (int r = 0; r < RB; r++)
        for (int c = 0; c < C; c++)
            _mm512_storeu_ps(out + c * ldo + r * Q8P_ROWS, f[r][c]);
}

static void q8p_tile_any(int rb, int c, const uint8_t *wp, size_t bstride, int ng, const int8_t *x,
                         size_t ldx, const float *xs, size_t ldxs, const int32_t *corr, float *out, size_t ldo) {
    // dispatch to the instance with compile-time RB (1..Q8P_RB) and C (1..Q8P_C <= 8)
    #define Q8P_CASE(R, CC) case (R) * 16 + (CC): { \
        if ((R) <= Q8P_RB && (CC) <= Q8P_C) \
            q8p_tile((R) <= Q8P_RB ? (R) : 1, (CC) <= Q8P_C ? (CC) : 1, wp, bstride, ng, x, ldx, xs, ldxs, corr, out, ldo); \
        break; }
    switch (rb * 16 + c) {
        Q8P_CASE(1, 1) Q8P_CASE(1, 2) Q8P_CASE(1, 3) Q8P_CASE(1, 4) Q8P_CASE(1, 5) Q8P_CASE(1, 6) Q8P_CASE(1, 7) Q8P_CASE(1, 8)
        Q8P_CASE(2, 1) Q8P_CASE(2, 2) Q8P_CASE(2, 3) Q8P_CASE(2, 4) Q8P_CASE(2, 5) Q8P_CASE(2, 6) Q8P_CASE(2, 7) Q8P_CASE(2, 8)
        default: break;
    }
    #undef Q8P_CASE
}

static int32_t *q8p_corr_buf(size_t n) {
    // scratch for the activation corrections; matmul / matmul_batch are only called from the
    // serial part of the forward pass, so one buffer is enough
    static int32_t *buf = NULL;
    static size_t cap = 0;
    if (n > cap) {
        free(buf);
        cap = n > 4096 ? n : 4096;
        buf = malloc(cap * sizeof(int32_t));
        if (!buf) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    }
    return buf;
}

static void q8p_gemm(float *out, int ldo, const int8_t *xq, const float *xs, int ldx, int nb,
                     const uint8_t *wp, int k, int d) {
    // out[b * ldo + i] = W[i, :] . x[b, :] for b < nb, i < d; W in Q8P layout (k % 64 == 0,
    // d % 16 == 0), x rows at xq + b * ldx with scales xs + b * ldx / 64.
    // Work items: (32-row weight tile, Q8P_NB activation rows), parallel; inside, the weight tile
    // (32 x k bytes + scales, 34-102 KB) is reused from L1/L2 for every Q8P_C activation rows.
    int ng = k / Q8P_GS, nblk = d / Q8P_ROWS;
    size_t bstride = (size_t)ng * Q8P_GROUP, ldxs = (size_t)ldx / Q8P_GS;
    int32_t *corr = q8p_corr_buf((size_t)nb * ng);
    int ntile = (nblk + Q8P_RB - 1) / Q8P_RB, nchunk = (nb + Q8P_NB - 1) / Q8P_NB;
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int b = 0; b < nb; b++)
            for (int g = 0; g < ng; g++)
                corr[(size_t)b * ng + g] = q8p_corr(xq + (size_t)b * ldx + g * Q8P_GS);
        #pragma omp for schedule(static)
        for (int it = 0; it < ntile * nchunk; it++) {
            int t = it / nchunk, ch = it % nchunk;
            int rb0 = t * Q8P_RB, rb = nblk - rb0 < Q8P_RB ? nblk - rb0 : Q8P_RB;
            int b1 = (ch + 1) * Q8P_NB < nb ? (ch + 1) * Q8P_NB : nb;
            const uint8_t *wt = wp + (size_t)rb0 * bstride;
            for (int b = ch * Q8P_NB; b < b1; b += Q8P_C) {
                int c = b1 - b < Q8P_C ? b1 - b : Q8P_C;
                q8p_tile_any(rb, c, wt, bstride, ng, xq + (size_t)b * ldx, ldx, xs + (size_t)b * ldxs, ldxs,
                             corr + (size_t)b * ng, out + (size_t)b * ldo + (size_t)rb0 * Q8P_ROWS, ldo);
            }
        }
    }
}

static void q8p_gemv(float *out, const int8_t *xq, const float *xs, const uint8_t *wp, int k, int d) {
    // out[i] = W[i, :] . x for i < d (decode). Memory-bound: each thread streams a contiguous
    // range of 16-row blocks once; the activation row (k bytes) stays in L1
    int ng = k / Q8P_GS, nblk = d / Q8P_ROWS;
    size_t bstride = (size_t)ng * Q8P_GROUP;
    int32_t *corr = q8p_corr_buf(ng);
    for (int g = 0; g < ng; g++)
        corr[g] = q8p_corr(xq + g * Q8P_GS);
    #pragma omp parallel for schedule(static)
    for (int rb = 0; rb < nblk; rb++) {
        const uint8_t *wb = wp + (size_t)rb * bstride;
        __m512 f = _mm512_setzero_ps();
        for (int g = 0; g < ng; g++) {
            const uint8_t *wg = wb + (size_t)g * Q8P_GROUP;
            const int8_t *xg = xq + g * Q8P_GS;
            // two independent chains (even / odd steps) to halve the vpdpbusd latency chain
            __m512i a0 = _mm512_set1_epi32(corr[g]), a1 = _mm512_setzero_si512();
            #pragma GCC unroll 8
            for (int st = 0; st < Q8P_GS / 4; st += 2) {
                a0 = _mm512_dpbusd_epi32(a0, _mm512_load_si512((const void *)(wg + st * 64)), q8p_bcast4(xg + st * 4));
                a1 = _mm512_dpbusd_epi32(a1, _mm512_load_si512((const void *)(wg + st * 64 + 64)), q8p_bcast4(xg + st * 4 + 4));
            }
            __m512 ws = _mm512_load_ps((const float *)(wg + Q8P_ROWS * Q8P_GS));
            f = q8p_fold(f, ws, xs[g], _mm512_add_epi32(a0, a1));
        }
        _mm512_storeu_ps(out + (size_t)rb * Q8P_ROWS, f);
    }
}

#endif // AVX-512 VNNI
#endif // RUNQ_KERNELS_AVX512_H
