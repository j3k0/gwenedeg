/* AVX-512 VNNI Q8_0 kernels for the forward passes (forward.c: GEMM for the batched prefill, GEMV for decode); further down
 * the same for the K-quants (Q3_K / Q4_K / Q5_K / Q6_K on the Q3P / Q4P / Q5P / Q6P layouts of kquants.h) and the attention.
 *
 * Enabled when the compiler targets AVX-512 F/BW/VNNI (e.g. -march=native on Zen 4 / Ice Lake+)
 * and RUNQ_GENERIC is not defined; it then defines RUNQ_VNNI. Otherwise this header is empty and
 * forward.c uses its generic C kernels (dot_q8, dot_q8_x4).
 *
 * Weights are repacked once at load ("Q8P" layout) so that the 16 dword lanes of a zmm hold 16
 * different weight rows: no horizontal reductions at all. For a (d, k) Q8_0 matrix with group size
 * gs = 64 (.bin) or 32 (GGUF) (d % 16 == 0, k % gs == 0), block rb = rows 16*rb .. 16*rb+15, group
 * g = columns gs*g .. gs*g+gs-1:
 *   pk + (rb * k/gs + g) * Q8P_GROUP(gs):
 *     gs/4 steps st of 64 bytes: lane r, byte j = (uint8)(W[16*rb + r][gs*g + 4*st + j] + 128)
 *     then 16 floats: the scales ws[16*rb + r][g]
 * 1088 (gs 64) or 576 (gs 32) bytes per (block, group), the same size as the (q, s) data with fp32
 * scales. One block (all k) is contiguous, so a GEMV streams each thread's rows as one sequential
 * read. The group size is a parameter of every function; the inner kernels are instantiated for
 * both (compile-time step counts).
 *
 * vpdpbusd multiplies u8 x s8: the weights are stored offset by +128 (u8), the activations stay
 * s8 and are broadcast 4 bytes at a time; the offset is removed exactly with the activation
 * group sums: isum = sum((w+128) * x) - 128 * sum(x) (accumulators start at -128 * sum(x)).
 * The per-group int32 sums are exact, and the float fold is the one of dot_q8 in forward.c, per
 * output and group in order: val = val + (ws * xs) * (float) isum, rounded after every operation
 * (explicit-rounding intrinsics, which gcc can neither contract into an FMA nor reassociate under
 * -Ofast). So the results are bit-identical to the generic kernels.
 */
#ifndef RUNQ_KERNELS_AVX512_H
#define RUNQ_KERNELS_AVX512_H

// Attention constants, shared by the AVX-512 attention below and the generic C one in forward.c
// (attention_vec): the same constants and operations give the same results
#define ATT_T 16                       // positions per K/V cache tile
#ifndef ATT_P
#define ATT_P 256                      // positions per attention block (a multiple of ATT_T; see below)
#endif
#define ATT_PT (ATT_P / ATT_T)         // tiles per block
#define ATT_PSTRIDE(hd) ((hd) + 16)    // floats per block partial result: o[hd], m, l (64-byte aligned)
#define ATT_DC 32                      // dims per fma chain of a score (head_dim % ATT_DC == 0)
#define ATT_EXP_MIN (-86.0f)           // exp argument clamp (2^n stays a normal float)
#define ATT_LOG2E 1.44269504088896341f
#define ATT_LN2_HI 0.693359375f        // Cephes expf: ln 2 = HI + LO, polynomial P0..P5
#define ATT_LN2_LO (-2.12194440e-4f)
#define ATT_P0 1.9875691500e-4f
#define ATT_P1 1.3981999507e-3f
#define ATT_P2 8.3334519073e-3f
#define ATT_P3 4.1665795894e-2f
#define ATT_P4 1.6666665459e-1f
#define ATT_P5 5.0000001201e-1f

static inline float bf16_f32(uint16_t h) {
    // bf16 -> fp32, exact
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static inline uint16_t f32_bf16(float f) {
    // fp32 -> bf16, round to nearest even (the bf16 KV cache stores; the bf16 dot-product attention's roundings of
    // q, the softmax weights and an f16 cache's K / V). Integer arithmetic, the same in every build; equal to
    // vcvtneps2bf16 for every non-denormal input (it flushes denormals, which -Ofast's FTZ / DAZ never produce here,
    // and which the dot products read as zero anyway); NaN stays a (quiet) NaN
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t r = (u + 0x7fffu + ((u >> 16) & 1)) >> 16;
    return (uint16_t)((u & 0x7fffffffu) > 0x7f800000u ? (u >> 16) | 0x40 : r);
}

static inline uint16_t f32_bf16_daz(float f) {
    // vcvtneps2bf16 exactly: f32_bf16 after a denormal input is flushed to a zero of its sign (the bf16 dot-product
    // attention's rounding of q and the softmax weights, the same in every build)
    uint32_t u;
    memcpy(&u, &f, 4);
    return (u & 0x7f800000u) ? f32_bf16(f) : (uint16_t)((u >> 16) & 0x8000u);
}

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__) && !defined(RUNQ_GENERIC)
#define RUNQ_VNNI 1
#include <immintrin.h>

#define Q8P_ROWS 16
#define Q8P_GROUP(gs) (Q8P_ROWS * (gs) + Q8P_ROWS * 4) // bytes per (block, group): 16 x gs + 64

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

static inline size_t q8p_size(int d, int k, int gs) { return (size_t)(d / Q8P_ROWS) * (k / gs) * Q8P_GROUP(gs); }

static inline int q8p_ok(int d, int k, int gs) {
    return (gs == 32 || gs == 64) && d > 0 && k > 0 && d % Q8P_ROWS == 0 && k % gs == 0;
}

static void q8p_pack_blocks(uint8_t *pk, const int8_t *q, const float *s, int nblk, int k, int gs) {
    // nblk blocks of 16 rows of a Q8_0 matrix with k columns: (q, s) = their rows, row-major, with
    // k/gs scales per row -> Q8P layout at pk (64-byte aligned). Serial (the caller splits the work)
    int ng = k / gs;
    for (int rb = 0; rb < nblk; rb++) {
        for (int g = 0; g < ng; g++) {
            uint8_t *o = pk + ((size_t)rb * ng + g) * Q8P_GROUP(gs);
            for (int r = 0; r < Q8P_ROWS; r++) {
                const int8_t *src = q + (size_t)(rb * Q8P_ROWS + r) * k + g * gs;
                for (int st = 0; st < gs / 4; st++) {
                    uint32_t v;
                    memcpy(&v, src + 4 * st, 4);
                    v ^= 0x80808080u; // int8 w -> uint8 w + 128
                    memcpy(o + st * 64 + r * 4, &v, 4);
                }
                memcpy(o + Q8P_ROWS * gs + r * 4, s + (size_t)(rb * Q8P_ROWS + r) * ng + g, 4);
            }
        }
    }
}

static inline int32_t q8p_corr(const int8_t *x, int gs) {
    // -128 * (sum of the gs = 64 or 32 int8 values at x): the accumulator start removing the +128
    // weight offset
    int32_t sum;
    if (gs == 64) {
        __m512i u = _mm512_xor_si512(_mm512_loadu_si512(x), _mm512_set1_epi8((char)0x80)); // x + 128
        __m512i sad = _mm512_sad_epu8(u, _mm512_setzero_si512());
        sum = (int32_t)_mm512_reduce_add_epi64(sad) - 64 * 128;
    } else {
        __m256i u = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)x), _mm256_set1_epi8((char)0x80));
        __m256i sad = _mm256_sad_epu8(u, _mm256_setzero_si256());
        __m128i h = _mm_add_epi64(_mm256_castsi256_si128(sad), _mm256_extracti128_si256(sad, 1));
        sum = (int32_t)(_mm_cvtsi128_si64(h) + _mm_extract_epi64(h, 1)) - 32 * 128;
    }
    return -128 * sum;
}

static inline __m512i q8p_bcast4(const int8_t *p) {
    int32_t v;
    memcpy(&v, p, 4);
    return _mm512_set1_epi32(v);
}

static inline __m512 q8p_fold(__m512 acc, __m512 ws, float xs, __m512i isum) {
    // acc + (ws * xs) * (float) isum, each operation rounded (as dot_q8 in forward.c)
    __m512 sc = _mm512_mul_round_ps(ws, _mm512_set1_ps(xs), Q8P_RND);
    __m512 p = _mm512_mul_round_ps(sc, _mm512_cvtepi32_ps(isum), Q8P_RND); // exact: |isum| < 2^24
    return _mm512_add_round_ps(acc, p, Q8P_RND);
}

static inline __attribute__((always_inline))
void q8p_tile(const int RB, const int C, const int GSZ, const uint8_t *wp, size_t bstride, int ng,
              const int8_t *x, size_t ldx, const float *xs, size_t ldxs, const int32_t *corr,
              float *out, size_t ldo) {
    // out[c * ldo + 16 * r + lane] for r < RB blocks (at wp, wp + bstride), c < C activation rows
    // (x + c * ldx, scales xs + c * ldxs, corrections corr + c * ng), over all ng groups of GSZ
    __m512 f[Q8P_RB][Q8P_C];
    for (int r = 0; r < RB; r++)
        for (int c = 0; c < C; c++)
            f[r][c] = _mm512_setzero_ps();
    for (int g = 0; g < ng; g++) {
        const uint8_t *wg = wp + (size_t)g * Q8P_GROUP(GSZ);
        const int8_t *xg = x + (size_t)g * GSZ;
        __m512i acc[Q8P_RB][Q8P_C];
        for (int c = 0; c < C; c++) {
            __m512i cv = _mm512_set1_epi32(corr[(size_t)c * ng + g]);
            for (int r = 0; r < RB; r++)
                acc[r][c] = cv;
        }
        #pragma GCC unroll 16
        for (int st = 0; st < GSZ / 4; st++) {
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
            __m512 ws = _mm512_load_ps((const float *)(wg + r * bstride + Q8P_ROWS * GSZ));
            for (int c = 0; c < C; c++)
                f[r][c] = q8p_fold(f[r][c], ws, xs[c * ldxs + g], acc[r][c]);
        }
    }
    for (int r = 0; r < RB; r++)
        for (int c = 0; c < C; c++)
            _mm512_storeu_ps(out + c * ldo + r * Q8P_ROWS, f[r][c]);
}

static void q8p_tile_any(int rb, int c, int gs, const uint8_t *wp, size_t bstride, int ng, const int8_t *x,
                         size_t ldx, const float *xs, size_t ldxs, const int32_t *corr, float *out, size_t ldo) {
    // dispatch to the instance with compile-time RB (1..Q8P_RB), C (1..Q8P_C <= 8) and group size
    #define Q8P_CASE(G, R, CC) case ((G) == 32) * 256 + (R) * 16 + (CC): { \
        if ((R) <= Q8P_RB && (CC) <= Q8P_C) \
            q8p_tile((R) <= Q8P_RB ? (R) : 1, (CC) <= Q8P_C ? (CC) : 1, G, wp, bstride, ng, x, ldx, xs, ldxs, corr, out, ldo); \
        break; }
    #define Q8P_CASES(G) \
        Q8P_CASE(G, 1, 1) Q8P_CASE(G, 1, 2) Q8P_CASE(G, 1, 3) Q8P_CASE(G, 1, 4) Q8P_CASE(G, 1, 5) Q8P_CASE(G, 1, 6) \
        Q8P_CASE(G, 1, 7) Q8P_CASE(G, 1, 8) Q8P_CASE(G, 2, 1) Q8P_CASE(G, 2, 2) Q8P_CASE(G, 2, 3) Q8P_CASE(G, 2, 4) \
        Q8P_CASE(G, 2, 5) Q8P_CASE(G, 2, 6) Q8P_CASE(G, 2, 7) Q8P_CASE(G, 2, 8)
    switch ((gs == 32) * 256 + rb * 16 + c) {
        Q8P_CASES(64)
        Q8P_CASES(32)
        default: break;
    }
    #undef Q8P_CASES
    #undef Q8P_CASE
}

static int q8p_nb(int nb, int ntile, int nt) {
    // activation rows per GEMM work item: Q8P_NB, halved (down to 4 * Q8P_C) while there are
    // fewer than ~6 items per thread (e.g. wo / w2 / k / v: 32 tiles x 2 chunks at 128 rows), so
    // 12 or 16 threads get balanced work. (Results do not depend on the partition)
    int nbi = Q8P_NB;
    while (nbi > 4 * Q8P_C && (long)ntile * ((nb + nbi - 1) / nbi) < 6L * nt)
        nbi /= 2;
    return nbi;
}

static void q8p_gemm_item(float *out, int ldo, const uint8_t *wp, int d, int t, int ch, int nbi, const int8_t *xq,
                          const float *xs, int ldx, int nb, int k, int gs, const int32_t *corr) {
    // one GEMM work item: out[b * ldo + i] = W[i, :] . x[b, :] for the rows i of 32-row tile t of W
    // (d x k, Q8P layout, group size gs) and the activation rows b of chunk ch (nbi rows; x at xq +
    // b * ldx, scales xs + b * ldx / gs, corrections corr + b * k / gs: q8p_gemv_corr of each row).
    // Callers run the items of q / k / v (or w1 / w3) in one phase, the chunks of a tile together
    // (the weight tile, 32 x k bytes + scales, 34-102 KB, is reused from L1/L2 for every Q8P_C rows)
    int ng = k / gs;
    size_t bstride = (size_t)ng * Q8P_GROUP(gs), ldxs = (size_t)ldx / gs;
    int nblk = d / Q8P_ROWS, rb0 = t * Q8P_RB;
    int rb = nblk - rb0 < Q8P_RB ? nblk - rb0 : Q8P_RB;
    int b1 = (ch + 1) * nbi < nb ? (ch + 1) * nbi : nb;
    const uint8_t *wt = wp + (size_t)rb0 * bstride;
    for (int b = ch * nbi; b < b1; b += Q8P_C) {
        int c = b1 - b < Q8P_C ? b1 - b : Q8P_C;
        q8p_tile_any(rb, c, gs, wt, bstride, ng, xq + (size_t)b * ldx, ldx, xs + (size_t)b * ldxs, ldxs,
                     corr + (size_t)b * ng, out + (size_t)b * ldo + (size_t)rb0 * Q8P_ROWS, ldo);
    }
}

static inline void q8p_gemv_corr(int32_t *corr, const int8_t *xq, int k, int gs) {
    // the GEMV's activation corrections, one per group (k / gs values)
    for (int g = 0; g < k / gs; g++)
        corr[g] = q8p_corr(xq + g * gs, gs);
}

static inline __attribute__((always_inline))
void q8p_gemv_blocks(const int GSZ, float *out, const int8_t *xq, const float *xs, const int32_t *corr,
                     const uint8_t *wp, int k, int rb0, int rb1) {
    int ng = k / GSZ;
    size_t bstride = (size_t)ng * Q8P_GROUP(GSZ);
    for (int rb = rb0; rb < rb1; rb++) {
        const uint8_t *wb = wp + (size_t)rb * bstride;
        __m512 f = _mm512_setzero_ps();
        for (int g = 0; g < ng; g++) {
            const uint8_t *wg = wb + (size_t)g * Q8P_GROUP(GSZ);
            const int8_t *xg = xq + g * GSZ;
            // two independent chains (even / odd steps) to halve the vpdpbusd latency chain
            __m512i a0 = _mm512_set1_epi32(corr[g]), a1 = _mm512_setzero_si512();
            #pragma GCC unroll 8
            for (int st = 0; st < GSZ / 4; st += 2) {
                a0 = _mm512_dpbusd_epi32(a0, _mm512_load_si512((const void *)(wg + st * 64)), q8p_bcast4(xg + st * 4));
                a1 = _mm512_dpbusd_epi32(a1, _mm512_load_si512((const void *)(wg + st * 64 + 64)), q8p_bcast4(xg + st * 4 + 4));
            }
            __m512 ws = _mm512_load_ps((const float *)(wg + Q8P_ROWS * GSZ));
            f = q8p_fold(f, ws, xs[g], _mm512_add_epi32(a0, a1));
        }
        _mm512_storeu_ps(out + (size_t)rb * Q8P_ROWS, f);
    }
}

static void q8p_gemv_rows(float *out, const int8_t *xq, const float *xs, const int32_t *corr, const uint8_t *wp,
                          int k, int gs, int rb0, int rb1) {
    // out[i] = W[i, :] . x for the rows i of the 16-row blocks rb0..rb1-1 (decode; no OpenMP here:
    // called on disjoint ranges from inside one parallel region). Memory-bound: streams the
    // blocks once; the activation row (k bytes) stays in L1
    if (gs == 64)
        q8p_gemv_blocks(64, out, xq, xs, corr, wp, k, rb0, rb1);
    else
        q8p_gemv_blocks(32, out, xq, xs, corr, wp, k, rb0, rb1);
}

/* ---------------------------------------------------------------------------------------------
 * K-quants (GGUF Q3_K / Q4_K / Q5_K / Q6_K) on the repacked Q3P / Q4P / Q5P / Q6P layouts of kquants.h:
 * GEMV (decode) and GEMM (prefill). The arithmetic is the definition in kquants.h's header comment, as the
 * generic q3p_dot / q4p_dot / q5p_dot / q6p_dot compute it: exact int32 sums (vpdpbusd, q as u8 0..7 /
 * 0..15 / 0..31 / 0..63 x xq as s8),
 * then per (16-row block, activation row, group of 32) the float fold with explicit-rounding
 * intrinsics (no contraction, no reassociation), groups in order:
 *   Q4_K / Q5_K: val = val + ((ds * xs) * (float) isum - dm * xf),  ds = d * sc, dm = dmin * m (exact)
 *   Q6_K: val = val + (d * xs) * (float) (sc_j0 * t_j0 + sc_j1 * t_j1),
 *         t_j = sum of q * xq over the 16 values of sub-block j - 32 * (sum of their 16 xq)
 *   Q3_K: the same with q' (0..7), - 4 * (sum of the 16 xq) and the scales sc - 32
 * The 16 lanes of a zmm are the 16 rows of a block (d, dmin, sc, m per lane; xs, xf broadcast).
 * Activation side, per row: xf[g] (kq_act_prep, kquants.h) and c16[i] = -32 * (sum of the 16 xq
 * at 16 i) (kq512_c16): Q6_K's accumulators start there (Q3_K's at c16 / 8, exact), so the -32 / -4
 * costs nothing per weight.
 *
 * GEMV: one 16-row block at a time, streamed (decode is memory-bound: the 3- to 6-bit data is read
 * once, packed, and unpacked in registers: and 0x0f / shift right 4 for the nibbles, Q5_K's high
 * bit from the masks (maskz_mov_epi8, or), Q6_K's high
 * 2 bits shifted into bits 4-5 of each byte and or'ed in, Q3_K's 2-bit crumbs shifted and masked, + 4
 * where its mask bit is set (mask_add_epi8)).
 * GEMM: work items (2-block = 32-row weight tile, chunk of activation rows), as q8p_gemm_item. Each
 * thread unpacks its weight tile once into a per-thread scratch ("KQS": per block and group of 32
 * columns 8 steps of 64 bytes of u8 q, lane r = row r, 4 consecutive columns per step, the Q8P
 * step layout; Q4_K / Q5_K: + ds, dm as fp32; Q3_K / Q6_K: + sc_j0, sc_j1 as int32, d as fp32) and reuses it
 * for all its activation rows (consecutive items of the same tile: unpacked once); the register
 * tile then reads u8 weights as the Q8 GEMM does, with no unpacking in the inner loop (Q3_K: Q6_K's tile).
 */
#include "kquants.h"

#define KQS_Q4 640 // scratch bytes per (block, group): 8 x 64 q, 64 ds, 64 dm (Q5_K: the same, u8 0..31)
#define KQS_Q6 704 // 8 x 64 q, 64 sc_j0 (int32), 64 sc_j1, 64 d (Q3_K: the same, q' 0..7, sc - 32)
// GEMM register tiles: KQ_RB blocks x KQ4_C / KQ6_C activation rows. Q4: 2 x C int32 + 2 x C
// float accumulators; Q6: 3 x 2 x C (the two sub-block sums t_j and I are live together)
#ifndef KQ4_C
#define KQ4_C 6
#endif
#ifndef KQ6_C
#define KQ6_C 4
#endif
#if KQ4_C < 1 || KQ4_C > 8 || KQ6_C < 1 || KQ6_C > 8
#error "KQ4_C / KQ6_C must be in 1..8"
#endif
#define KQ_RB 2

static inline void kq512_c16(int32_t *c16, const int8_t *xq, int n) {
    // c16[i] = -32 * (sum of the 16 int8 at xq + 16 i), i < n / 16 (sad on x + 128, minus 16 * 128)
    const __m128i off = _mm_set1_epi8((char)0x80);
    for (int i = 0; i < n / 16; i++) {
        __m128i s = _mm_sad_epu8(_mm_xor_si128(_mm_loadu_si128((const __m128i *)(xq + 16 * i)), off), _mm_setzero_si128());
        c16[i] = -32 * (_mm_cvtsi128_si32(s) + _mm_extract_epi16(s, 4) - 16 * 128);
    }
}

static inline __m512 kq512_u8ps(const uint8_t *p) { return _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i *)p))); }

static inline __m512 kq4_fold(__m512 f, __m512 ds, __m512 dm, float xs, float xf, __m512i isum) {
    // f + ((ds * xs) * (float) isum - dm * xf), each operation rounded (q4p_dot)
    __m512 a = _mm512_mul_round_ps(_mm512_mul_round_ps(ds, _mm512_set1_ps(xs), Q8P_RND), _mm512_cvtepi32_ps(isum), Q8P_RND);
    __m512 b = _mm512_mul_round_ps(dm, _mm512_set1_ps(xf), Q8P_RND);
    return _mm512_add_round_ps(f, _mm512_sub_round_ps(a, b, Q8P_RND), Q8P_RND);
}

static inline __m512 kq6_fold(__m512 f, __m512 d, float xs, __m512i I) {
    // f + (d * xs) * (float) I, each operation rounded (q6p_dot)
    return _mm512_add_round_ps(f, _mm512_mul_round_ps(_mm512_mul_round_ps(d, _mm512_set1_ps(xs), Q8P_RND),
                                                      _mm512_cvtepi32_ps(I), Q8P_RND), Q8P_RND);
}

static inline void kq6_unpack4(const uint8_t *ql, const uint8_t *qh, __m512i q[4]) {
    // the 4 steps (columns 4 t .. 4 t + 3) of a Q6P sub-block as u8 0..63: low nibbles from the 2 ql
    // chunks, high 2 bits (qh bits 2 t, 2 t + 1) into bits 4-5 (16-bit shifts; the mask keeps bytes apart)
    const __m512i m4 = _mm512_set1_epi8(0x0f);
    __m512i v0 = _mm512_load_si512((const void *)ql), v1 = _mm512_load_si512((const void *)(ql + 64));
    __m512i h = _mm512_load_si512((const void *)qh);
    // (ternary logic 0xd8 = c ? b : a, bitwise: low nibble from the ql value, bits 4-5 from qh)
    q[0] = _mm512_ternarylogic_epi32(_mm512_slli_epi16(h, 4), v0, m4, 0xd8);
    q[1] = _mm512_ternarylogic_epi32(_mm512_slli_epi16(h, 2), _mm512_srli_epi16(v0, 4), m4, 0xd8);
    q[2] = _mm512_ternarylogic_epi32(h, v1, m4, 0xd8);
    q[3] = _mm512_ternarylogic_epi32(_mm512_srli_epi16(h, 2), _mm512_srli_epi16(v1, 4), m4, 0xd8);
    for (int t = 0; t < 4; t++)
        q[t] = _mm512_and_si512(q[t], _mm512_set1_epi8(0x3f));
}

static inline void kq3_unpack4(const uint8_t *ql, const uint8_t *mp, __m512i q[4]) {
    // the 4 steps (columns 4 t .. 4 t + 3) of a Q3P sub-block as u8 q' (0..7): the low 2 bits from the chunk
    // ql (bits 2 t, 2 t + 1: 16-bit shifts, the and keeps bytes apart), + 4 where mask t (at mp + 8 t) is set
    const __m512i m3 = _mm512_set1_epi8(3), four = _mm512_set1_epi8(4);
    __m512i v = _mm512_load_si512((const void *)ql);
    __m512i l[4] = { _mm512_and_si512(v, m3), _mm512_and_si512(_mm512_srli_epi16(v, 2), m3),
                     _mm512_and_si512(_mm512_srli_epi16(v, 4), m3), _mm512_and_si512(_mm512_srli_epi16(v, 6), m3) };
    for (int t = 0; t < 4; t++)
        q[t] = _mm512_mask_add_epi8(l[t], (__mmask64)kq_u64(mp + 8 * t), l[t], four);
}

static inline void kq3_scales(const uint8_t *o, int g8, __m512i *s0, __m512i *s1) {
    // the scales sc - 32 of sub-blocks 2 g8, 2 g8 + 1 of the 16 rows as int32 (Q3P: the low nibbles at
    // 64 + 16 g8, the high 2 bits at 192 + 16 (g8 / 2), bits 4 (g8 % 2) ..)
    const __m512i m15 = _mm512_set1_epi32(15), m3 = _mm512_set1_epi32(3), c32 = _mm512_set1_epi32(32);
    __m512i L = _mm512_cvtepu8_epi32(_mm_load_si128((const __m128i *)(o + 64 + 16 * g8)));
    __m512i H = _mm512_srl_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i *)(o + 192 + 16 * (g8 / 2)))),
                                 _mm_cvtsi32_si128(4 * (g8 & 1)));
    *s0 = _mm512_sub_epi32(_mm512_or_si512(_mm512_and_si512(L, m15), _mm512_slli_epi32(_mm512_and_si512(H, m3), 4)), c32);
    *s1 = _mm512_sub_epi32(_mm512_or_si512(_mm512_srli_epi32(L, 4), _mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(H, 2), m3), 4)),
                           c32);
}

static void kq3_gemv_rows(float *out, const int8_t *xq, const float *xs, const int32_t *c16, const uint8_t *wp, int k,
                          int rb0, int rb1) {
    // out[i] = W[i, :] . x for the rows of the 16-row Q3P blocks rb0..rb1-1 (decode, no OpenMP here; c16:
    // kq512_c16 of xq, / 8 = Q3_K's accumulator start)
    int nsb = k / QK_K;
    for (int rb = rb0; rb < rb1; rb++) {
        const uint8_t *wb = wp + (size_t)rb * nsb * Q3P_SB;
        __m512 f = _mm512_setzero_ps();
        for (int s = 0; s < nsb; s++) {
            const uint8_t *o = wb + (size_t)s * Q3P_SB;
            __m512 d = _mm512_load_ps((const float *)o);
            #pragma GCC unroll 2
            for (int g8 = 0; g8 < 8; g8++) {
                int g = 8 * s + g8;
                const uint8_t *qg = o + 256 + 192 * g8;
                __m512i sc[2], I = _mm512_setzero_si512();
                kq3_scales(o, g8, &sc[0], &sc[1]);
                for (int h = 0; h < 2; h++) {
                    const int8_t *x = xq + 32 * g + 16 * h;
                    __m512i q[4];
                    kq3_unpack4(qg + 64 * h, qg + 128 + 32 * h, q);
                    __m512i t0 = _mm512_set1_epi32(c16[2 * g + h] / 8), t1 = _mm512_setzero_si512();
                    t0 = _mm512_dpbusd_epi32(t0, q[0], q8p_bcast4(x));
                    t1 = _mm512_dpbusd_epi32(t1, q[1], q8p_bcast4(x + 4));
                    t0 = _mm512_dpbusd_epi32(t0, q[2], q8p_bcast4(x + 8));
                    t1 = _mm512_dpbusd_epi32(t1, q[3], q8p_bcast4(x + 12));
                    I = _mm512_add_epi32(I, _mm512_mullo_epi32(_mm512_add_epi32(t0, t1), sc[h]));
                }
                f = kq6_fold(f, d, xs[g], I);
            }
        }
        _mm512_storeu_ps(out + (size_t)rb * KQ_ROWS, f);
    }
}

static void kq4_gemv_rows(float *out, const int8_t *xq, const float *xs, const float *xf, const uint8_t *wp, int k,
                          int rb0, int rb1) {
    // out[i] = W[i, :] . x for the rows of the 16-row Q4P blocks rb0..rb1-1 (decode, no OpenMP here)
    int nsb = k / QK_K;
    const __m512i m4 = _mm512_set1_epi8(0x0f);
    for (int rb = rb0; rb < rb1; rb++) {
        const uint8_t *wb = wp + (size_t)rb * nsb * Q4P_SB;
        __m512 f = _mm512_setzero_ps();
        for (int s = 0; s < nsb; s++) {
            const uint8_t *o = wb + (size_t)s * Q4P_SB;
            __m512 d = _mm512_cvtph_ps(_mm256_load_si256((const __m256i *)o));
            __m512 dmin = _mm512_cvtph_ps(_mm256_load_si256((const __m256i *)(o + 32)));
            #pragma GCC unroll 2
            for (int j = 0; j < 8; j++) {
                int g = 8 * s + j;
                const uint8_t *qs = o + 320 + 256 * j;
                const int8_t *x = xq + 32 * g;
                __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
                #pragma GCC unroll 4
                for (int c = 0; c < 4; c++) {
                    __m512i v = _mm512_load_si512((const void *)(qs + 64 * c));
                    a0 = _mm512_dpbusd_epi32(a0, _mm512_and_si512(v, m4), q8p_bcast4(x + 8 * c));
                    a1 = _mm512_dpbusd_epi32(a1, _mm512_and_si512(_mm512_srli_epi16(v, 4), m4), q8p_bcast4(x + 8 * c + 4));
                }
                __m512 ds = _mm512_mul_round_ps(d, kq512_u8ps(o + 64 + 16 * j), Q8P_RND);
                __m512 dm = _mm512_mul_round_ps(dmin, kq512_u8ps(o + 192 + 16 * j), Q8P_RND);
                f = kq4_fold(f, ds, dm, xs[g], xf[g], _mm512_add_epi32(a0, a1));
            }
        }
        _mm512_storeu_ps(out + (size_t)rb * KQ_ROWS, f);
    }
}

static void kq5_gemv_rows(float *out, const int8_t *xq, const float *xs, const float *xf, const uint8_t *wp, int k,
                          int rb0, int rb1) {
    // out[i] = W[i, :] . x for the rows of the 16-row Q5P blocks rb0..rb1-1 (decode, no OpenMP here)
    int nsb = k / QK_K;
    const __m512i m4 = _mm512_set1_epi8(0x0f), m16 = _mm512_set1_epi8(0x10);
    for (int rb = rb0; rb < rb1; rb++) {
        const uint8_t *wb = wp + (size_t)rb * nsb * Q5P_SB;
        __m512 f = _mm512_setzero_ps();
        for (int s = 0; s < nsb; s++) {
            const uint8_t *o = wb + (size_t)s * Q5P_SB;
            __m512 d = _mm512_cvtph_ps(_mm256_load_si256((const __m256i *)o));
            __m512 dmin = _mm512_cvtph_ps(_mm256_load_si256((const __m256i *)(o + 32)));
            #pragma GCC unroll 2
            for (int j = 0; j < 8; j++) {
                int g = 8 * s + j;
                const uint8_t *qs = o + 320 + 320 * j, *mp = qs + 256;
                const int8_t *x = xq + 32 * g;
                __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
                #pragma GCC unroll 4
                for (int c = 0; c < 4; c++) {
                    __m512i v = _mm512_load_si512((const void *)(qs + 64 * c));
                    __m512i h0 = _mm512_maskz_mov_epi8((__mmask64)kq_u64(mp + 8 * (2 * c)), m16);
                    __m512i h1 = _mm512_maskz_mov_epi8((__mmask64)kq_u64(mp + 8 * (2 * c + 1)), m16);
                    a0 = _mm512_dpbusd_epi32(a0, _mm512_or_si512(_mm512_and_si512(v, m4), h0), q8p_bcast4(x + 8 * c));
                    a1 = _mm512_dpbusd_epi32(a1, _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(v, 4), m4), h1),
                                             q8p_bcast4(x + 8 * c + 4));
                }
                __m512 ds = _mm512_mul_round_ps(d, kq512_u8ps(o + 64 + 16 * j), Q8P_RND);
                __m512 dm = _mm512_mul_round_ps(dmin, kq512_u8ps(o + 192 + 16 * j), Q8P_RND);
                f = kq4_fold(f, ds, dm, xs[g], xf[g], _mm512_add_epi32(a0, a1));
            }
        }
        _mm512_storeu_ps(out + (size_t)rb * KQ_ROWS, f);
    }
}

static void kq6_gemv_rows(float *out, const int8_t *xq, const float *xs, const int32_t *c16, const uint8_t *wp, int k,
                          int rb0, int rb1) {
    // the same for Q6P blocks (c16: kq512_c16 of xq)
    int nsb = k / QK_K;
    for (int rb = rb0; rb < rb1; rb++) {
        const uint8_t *wb = wp + (size_t)rb * nsb * Q6P_SB;
        __m512 f = _mm512_setzero_ps();
        for (int s = 0; s < nsb; s++) {
            const uint8_t *o = wb + (size_t)s * Q6P_SB;
            __m512 d = _mm512_load_ps((const float *)o);
            #pragma GCC unroll 2
            for (int g8 = 0; g8 < 8; g8++) {
                int g = 8 * s + g8;
                __m512i I = _mm512_setzero_si512();
                for (int h = 0; h < 2; h++) {
                    int j = 2 * g8 + h;
                    const int8_t *x = xq + 32 * g + 16 * h;
                    __m512i q[4];
                    kq6_unpack4(o + 320 + 192 * j, o + 448 + 192 * j, q);
                    __m512i t0 = _mm512_set1_epi32(c16[2 * g + h]), t1 = _mm512_setzero_si512();
                    t0 = _mm512_dpbusd_epi32(t0, q[0], q8p_bcast4(x));
                    t1 = _mm512_dpbusd_epi32(t1, q[1], q8p_bcast4(x + 4));
                    t0 = _mm512_dpbusd_epi32(t0, q[2], q8p_bcast4(x + 8));
                    t1 = _mm512_dpbusd_epi32(t1, q[3], q8p_bcast4(x + 12));
                    __m512i sc = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i *)(o + 64 + 16 * j)));
                    I = _mm512_add_epi32(I, _mm512_mullo_epi32(_mm512_add_epi32(t0, t1), sc));
                }
                f = kq6_fold(f, d, xs[g], I);
            }
        }
        _mm512_storeu_ps(out + (size_t)rb * KQ_ROWS, f);
    }
}

static void kq512_gemv_rows(int type, float *out, const int8_t *xq, const float *xs, const float *xf, const int32_t *c16,
                            const uint8_t *wp, int k, int rb0, int rb1) {
    if (type == GGML_Q4_K)
        kq4_gemv_rows(out, xq, xs, xf, wp, k, rb0, rb1);
    else if (type == GGML_Q5_K)
        kq5_gemv_rows(out, xq, xs, xf, wp, k, rb0, rb1);
    else if (type == GGML_Q3_K)
        kq3_gemv_rows(out, xq, xs, c16, wp, k, rb0, rb1);
    else
        kq6_gemv_rows(out, xq, xs, c16, wp, k, rb0, rb1);
}

static void kq_unpack_tile(int type, uint8_t *scr, const uint8_t *wp, int nblk, int k) {
    // nblk (<= KQ_RB) consecutive Q3P / Q4P / Q5P / Q6P blocks at wp -> the KQS scratch layout (block r at
    // scr + r * (k / 32) * KQS_Q4 / KQS_Q6)
    int nsb = k / QK_K, ng = k / 32;
    const __m512i m4 = _mm512_set1_epi8(0x0f), m16 = _mm512_set1_epi8(0x10);
    for (int r = 0; r < nblk; r++)
        for (int s = 0; s < nsb; s++) {
            if (type == GGML_Q4_K) {
                const uint8_t *o = wp + ((size_t)r * nsb + s) * Q4P_SB;
                __m512 d = _mm512_cvtph_ps(_mm256_load_si256((const __m256i *)o));
                __m512 dmin = _mm512_cvtph_ps(_mm256_load_si256((const __m256i *)(o + 32)));
                for (int j = 0; j < 8; j++) {
                    uint8_t *dst = scr + ((size_t)r * ng + 8 * s + j) * KQS_Q4;
                    for (int c = 0; c < 4; c++) {
                        __m512i v = _mm512_load_si512((const void *)(o + 320 + 256 * j + 64 * c));
                        _mm512_store_si512((void *)(dst + 128 * c), _mm512_and_si512(v, m4));
                        _mm512_store_si512((void *)(dst + 128 * c + 64), _mm512_and_si512(_mm512_srli_epi16(v, 4), m4));
                    }
                    _mm512_store_ps((float *)(dst + 512), _mm512_mul_round_ps(d, kq512_u8ps(o + 64 + 16 * j), Q8P_RND));
                    _mm512_store_ps((float *)(dst + 576), _mm512_mul_round_ps(dmin, kq512_u8ps(o + 192 + 16 * j), Q8P_RND));
                }
            } else if (type == GGML_Q5_K) {
                const uint8_t *o = wp + ((size_t)r * nsb + s) * Q5P_SB;
                __m512 d = _mm512_cvtph_ps(_mm256_load_si256((const __m256i *)o));
                __m512 dmin = _mm512_cvtph_ps(_mm256_load_si256((const __m256i *)(o + 32)));
                for (int j = 0; j < 8; j++) {
                    uint8_t *dst = scr + ((size_t)r * ng + 8 * s + j) * KQS_Q4;
                    const uint8_t *qs = o + 320 + 320 * j, *mp = qs + 256;
                    for (int c = 0; c < 4; c++) { // the 5-bit values (u8 0..31): nibbles | mask bit << 4
                        __m512i v = _mm512_load_si512((const void *)(qs + 64 * c));
                        __m512i h0 = _mm512_maskz_mov_epi8((__mmask64)kq_u64(mp + 8 * (2 * c)), m16);
                        __m512i h1 = _mm512_maskz_mov_epi8((__mmask64)kq_u64(mp + 8 * (2 * c + 1)), m16);
                        _mm512_store_si512((void *)(dst + 128 * c), _mm512_or_si512(_mm512_and_si512(v, m4), h0));
                        _mm512_store_si512((void *)(dst + 128 * c + 64),
                                           _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(v, 4), m4), h1));
                    }
                    _mm512_store_ps((float *)(dst + 512), _mm512_mul_round_ps(d, kq512_u8ps(o + 64 + 16 * j), Q8P_RND));
                    _mm512_store_ps((float *)(dst + 576), _mm512_mul_round_ps(dmin, kq512_u8ps(o + 192 + 16 * j), Q8P_RND));
                }
            } else if (type == GGML_Q3_K) {
                const uint8_t *o = wp + ((size_t)r * nsb + s) * Q3P_SB;
                __m512 d = _mm512_load_ps((const float *)o);
                for (int g8 = 0; g8 < 8; g8++) {
                    uint8_t *dst = scr + ((size_t)r * ng + 8 * s + g8) * KQS_Q6;
                    const uint8_t *qg = o + 256 + 192 * g8;
                    __m512i sc[2];
                    kq3_scales(o, g8, &sc[0], &sc[1]);
                    for (int h = 0; h < 2; h++) {
                        __m512i q[4];
                        kq3_unpack4(qg + 64 * h, qg + 128 + 32 * h, q);
                        for (int t = 0; t < 4; t++)
                            _mm512_store_si512((void *)(dst + 256 * h + 64 * t), q[t]);
                        _mm512_store_si512((void *)(dst + 512 + 64 * h), sc[h]);
                    }
                    _mm512_store_ps((float *)(dst + 640), d);
                }
            } else {
                const uint8_t *o = wp + ((size_t)r * nsb + s) * Q6P_SB;
                __m512 d = _mm512_load_ps((const float *)o);
                for (int g8 = 0; g8 < 8; g8++) {
                    uint8_t *dst = scr + ((size_t)r * ng + 8 * s + g8) * KQS_Q6;
                    for (int h = 0; h < 2; h++) {
                        int j = 2 * g8 + h;
                        __m512i q[4];
                        kq6_unpack4(o + 320 + 192 * j, o + 448 + 192 * j, q);
                        for (int t = 0; t < 4; t++)
                            _mm512_store_si512((void *)(dst + 256 * h + 64 * t), q[t]);
                        _mm512_store_si512((void *)(dst + 512 + 64 * h),
                                           _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i *)(o + 64 + 16 * j))));
                    }
                    _mm512_store_ps((float *)(dst + 640), d);
                }
            }
        }
}

static inline __attribute__((always_inline))
void kq45_tile(const int RB, const int C, const uint8_t *ws, size_t bstride, int ng, const int8_t *x, size_t ldx,
              const float *xs, size_t ldxs, const float *xf, size_t ldxf, float *out, size_t ldo) {
    // out[c * ldo + 16 * r + lane] for r < RB blocks of the Q4 / Q5 scratch tile ws (block stride
    // bstride: the same KQS layout, Q5's q are 0..31), c < C activation rows (x + c * ldx, scales
    // xs + c * ldxs, group sums xf + c * ldxf)
    __m512 f[KQ_RB][8];
    for (int r = 0; r < RB; r++)
        for (int c = 0; c < C; c++)
            f[r][c] = _mm512_setzero_ps();
    for (int g = 0; g < ng; g++) {
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
            __m512 ds = _mm512_load_ps((const float *)(wg + r * bstride + 512));
            __m512 dm = _mm512_load_ps((const float *)(wg + r * bstride + 576));
            for (int c = 0; c < C; c++)
                f[r][c] = kq4_fold(f[r][c], ds, dm, xs[c * ldxs + g], xf[c * ldxf + g], acc[r][c]);
        }
    }
    for (int r = 0; r < RB; r++)
        for (int c = 0; c < C; c++)
            _mm512_storeu_ps(out + c * ldo + r * KQ_ROWS, f[r][c]);
}

static inline __attribute__((always_inline))
void kq6_tile(const int RB, const int C, const int DIV, const uint8_t *ws, size_t bstride, int ng, const int8_t *x,
              size_t ldx, const float *xs, size_t ldxs, const int32_t *c16, size_t ldc, float *out, size_t ldo) {
    // the same for the Q6 scratch tile (c16 + c * ldc: row c's -32 * sums of 16), and the Q3 one (DIV 8: the
    // accumulators start at c16 / 8 = -4 * sums of 16; Q6_K: DIV 1)
    __m512 f[KQ_RB][8];
    for (int r = 0; r < RB; r++)
        for (int c = 0; c < C; c++)
            f[r][c] = _mm512_setzero_ps();
    for (int g = 0; g < ng; g++) {
        const uint8_t *wg = ws + (size_t)g * KQS_Q6;
        const int8_t *xg = x + (size_t)g * 32;
        __m512i I[KQ_RB][8];
        for (int r = 0; r < RB; r++)
            for (int c = 0; c < C; c++)
                I[r][c] = _mm512_setzero_si512(); // (0 + x: folded by the compiler)
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
        for (int r = 0; r < RB; r++) {
            __m512 d = _mm512_load_ps((const float *)(wg + r * bstride + 640));
            for (int c = 0; c < C; c++)
                f[r][c] = kq6_fold(f[r][c], d, xs[c * ldxs + g], I[r][c]);
        }
    }
    for (int r = 0; r < RB; r++)
        for (int c = 0; c < C; c++)
            _mm512_storeu_ps(out + c * ldo + r * KQ_ROWS, f[r][c]);
}

static void kq_tile_any(int type, int rb, int c, const uint8_t *ws, size_t bstride, int ng, const int8_t *x, size_t ldx,
                        const float *xs, size_t ldxs, const float *xf, size_t ldxf, const int32_t *c16, size_t ldc,
                        float *out, size_t ldo) {
    // dispatch to the instance with compile-time RB (1..2) and C (1..KQ4_C / KQ6_C; Q3_K: Q6_K's tile)
    #define KQ_CASE(R, CC) \
        case (R) * 16 + (CC): \
            if (type == GGML_Q6_K) { if ((CC) <= KQ6_C) kq6_tile(R, (CC) <= KQ6_C ? (CC) : 1, 1, ws, bstride, ng, x, ldx, xs, ldxs, c16, ldc, out, ldo); } \
            else if (type == GGML_Q3_K) { if ((CC) <= KQ6_C) kq6_tile(R, (CC) <= KQ6_C ? (CC) : 1, 8, ws, bstride, ng, x, ldx, xs, ldxs, c16, ldc, out, ldo); } \
            else if ((CC) <= KQ4_C) kq45_tile(R, (CC) <= KQ4_C ? (CC) : 1, ws, bstride, ng, x, ldx, xs, ldxs, xf, ldxf, out, ldo); \
            break;
    switch (rb * 16 + c) {
        KQ_CASE(1, 1) KQ_CASE(1, 2) KQ_CASE(1, 3) KQ_CASE(1, 4) KQ_CASE(1, 5) KQ_CASE(1, 6) KQ_CASE(1, 7) KQ_CASE(1, 8)
        KQ_CASE(2, 1) KQ_CASE(2, 2) KQ_CASE(2, 3) KQ_CASE(2, 4) KQ_CASE(2, 5) KQ_CASE(2, 6) KQ_CASE(2, 7) KQ_CASE(2, 8)
        default: break;
    }
    #undef KQ_CASE
}

#define KQ512_SCRATCH(k) ((size_t)KQ_RB * ((k) / 32) * KQS_Q6) // per-thread GEMM scratch bytes (the larger type)

static void kq512_gemm_item(float *out, int ldo, const uint8_t *kp, int type, int d, int t, int ch, int nbi,
                            const int8_t *xq, const float *xs, int ldx, int nb, int k, const float *xf,
                            const int32_t *c16, uint8_t *scr, const void **last) {
    // one K-quant GEMM work item, as q8p_gemm_item: 32-row tile t of W (Q3P .. Q6P), activation rows
    // of chunk ch (xf + b * k / 32: kq_act_prep, c16 + b * k / 16: kq512_c16 of each row). The
    // thread's scratch scr (KQ512_SCRATCH(k) bytes) holds the tile unpacked (KQS layout); *last =
    // the tile it holds (unpacked again only for another tile: callers claim all chunks of a tile at once)
    int ng = k / 32, n16 = k / 16, sbb = kq_sb_bytes(type);
    size_t ldxs = (size_t)ldx / 32;
    int nblk = d / KQ_ROWS, rb0 = t * KQ_RB;
    int rb = nblk - rb0 < KQ_RB ? nblk - rb0 : KQ_RB;
    int q36 = type == GGML_Q6_K || type == GGML_Q3_K; // (the Q6 scratch layout and register tile)
    size_t bstride = (size_t)ng * (q36 ? KQS_Q6 : KQS_Q4);
    const uint8_t *wt = kp + (size_t)rb0 * (k / QK_K) * sbb;
    if (*last != wt) {
        kq_unpack_tile(type, scr, wt, rb, k);
        *last = wt;
    }
    int b1 = (ch + 1) * nbi < nb ? (ch + 1) * nbi : nb, C = q36 ? KQ6_C : KQ4_C;
    for (int b = ch * nbi; b < b1; b += C) {
        int c = b1 - b < C ? b1 - b : C;
        kq_tile_any(type, rb, c, scr, bstride, ng, xq + (size_t)b * ldx, ldx, xs + (size_t)b * ldxs, ldxs,
                    xf + (size_t)b * ng, ng, c16 + (size_t)b * n16, n16, out + (size_t)b * ldo + (size_t)rb0 * KQ_ROWS, ldo);
    }
}

/* ---------------------------------------------------------------------------------------------
 * Small-batch matmul kernels (Step 10a: a few rows, e.g. decode rows of several sequences in one
 * rq_forward_batch pass): out[b * ldo + i] = W[i, :] . x[b, :] for the rows i of 16-row blocks
 * rb0..rb1-1 and all nb activation rows. Like the GEMV, each thread streams its blocks once, the
 * 4 / 6-bit data unpacked in registers (no scratch tile: the GEMM's per-tile unpacking costs as much
 * as the arithmetic at a few rows); each unpacked group is used for up to SMB_C activation rows
 * (more rows: the block again from L1 for the next SMB_C). The K-quant kernels prefetch their weights pf bytes
 * ahead (forward.c SMB_PF: at 2-6 rows the hardware prefetchers alone leave the loads waiting on memory while
 * the arithmetic runs; a prefetch changes no result). The integer sums are exact and the folds
 * the same functions in the same order per (block, row, group) as the GEMV / GEMM / generic kernels,
 * so every row's result is bit-identical to them, whatever rows it is batched with.
 * Activation side per row b: xq + b * ldx, xs + b * ldx / gs; Q8P corr + b * k / gs; K-quants xf +
 * b * k / 32, c16 + b * k / 16 (the strides of the GEMM's act_prep_row in forward.c).
 */
#ifndef SMB_C
#define SMB_C 8 // activation rows per pass over an unpacked group (register tile 1 block x SMB_C rows)
#endif
#if SMB_C < 1 || SMB_C > 8
#error "SMB_C must be in 1..8"
#endif

static inline __attribute__((always_inline)) void smb_pf(const uint8_t *p, int bytes, int pf) {
    // prefetch the lines of [p + pf, p + pf + bytes) (pf 0: none; past the matrix: harmless, a prefetch never faults)
    if (pf)
        for (int i = 0; i < bytes; i += 64)
            _mm_prefetch((const char *)p + pf + i, _MM_HINT_T0);
}

static inline __attribute__((always_inline))
void q8p_smb_block(const int C, const int GSZ, const uint8_t *wb, int ng, const int8_t *x, size_t ldx, const float *xs,
                   size_t ldxs, const int32_t *corr, size_t ldc, float *out, size_t ldo) {
    // one Q8P block (wb: its ng groups) x C activation rows; two chains per row (even / odd steps, as the
    // GEMV: vpdpbusd's latency)
    __m512 f[8];
    for (int c = 0; c < C; c++)
        f[c] = _mm512_setzero_ps();
    for (int g = 0; g < ng; g++) {
        const uint8_t *wg = wb + (size_t)g * Q8P_GROUP(GSZ);
        __m512i a[8], b[8];
        for (int c = 0; c < C; c++) {
            a[c] = _mm512_set1_epi32(corr[(size_t)c * ldc + g]);
            b[c] = _mm512_setzero_si512();
        }
        #pragma GCC unroll 16
        for (int st = 0; st < GSZ / 4; st++) {
            __m512i w = _mm512_load_si512((const void *)(wg + st * 64));
            for (int c = 0; c < C; c++) {
                __m512i xv = q8p_bcast4(x + (size_t)c * ldx + (size_t)g * GSZ + st * 4);
                if (st & 1)
                    b[c] = _mm512_dpbusd_epi32(b[c], w, xv);
                else
                    a[c] = _mm512_dpbusd_epi32(a[c], w, xv);
            }
        }
        __m512 ws = _mm512_load_ps((const float *)(wg + Q8P_ROWS * GSZ));
        for (int c = 0; c < C; c++)
            f[c] = q8p_fold(f[c], ws, xs[(size_t)c * ldxs + g], _mm512_add_epi32(a[c], b[c]));
    }
    for (int c = 0; c < C; c++)
        _mm512_storeu_ps(out + (size_t)c * ldo, f[c]);
}

static inline __attribute__((always_inline))
void kq4_smb_block(const int C, const uint8_t *wb, int nsb, const int8_t *x, size_t ldx, const float *xs, size_t ldxs,
                   const float *xf, size_t ldxf, float *out, size_t ldo, int pf) {
    // one Q4P block (wb: its nsb super-blocks) x C activation rows: per group the 8 steps unpacked once
    const __m512i m4 = _mm512_set1_epi8(0x0f);
    __m512 f[8];
    for (int c = 0; c < C; c++)
        f[c] = _mm512_setzero_ps();
    for (int s = 0; s < nsb; s++) {
        const uint8_t *o = wb + (size_t)s * Q4P_SB;
        __m512 d = _mm512_cvtph_ps(_mm256_load_si256((const __m256i *)o));
        __m512 dmin = _mm512_cvtph_ps(_mm256_load_si256((const __m256i *)(o + 32)));
        for (int j = 0; j < 8; j++) {
            int g = 8 * s + j;
            const uint8_t *qs = o + 320 + 256 * j;
            smb_pf(o + j * (Q4P_SB / 8), Q4P_SB / 8, pf);
            __m512i w[8];
            for (int cc = 0; cc < 4; cc++) {
                __m512i v = _mm512_load_si512((const void *)(qs + 64 * cc));
                w[2 * cc] = _mm512_and_si512(v, m4);
                w[2 * cc + 1] = _mm512_and_si512(_mm512_srli_epi16(v, 4), m4);
            }
            __m512 ds = _mm512_mul_round_ps(d, kq512_u8ps(o + 64 + 16 * j), Q8P_RND);
            __m512 dm = _mm512_mul_round_ps(dmin, kq512_u8ps(o + 192 + 16 * j), Q8P_RND);
            for (int c = 0; c < C; c++) {
                const int8_t *xg = x + (size_t)c * ldx + 32 * g;
                __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
                for (int cc = 0; cc < 4; cc++) {
                    a0 = _mm512_dpbusd_epi32(a0, w[2 * cc], q8p_bcast4(xg + 8 * cc));
                    a1 = _mm512_dpbusd_epi32(a1, w[2 * cc + 1], q8p_bcast4(xg + 8 * cc + 4));
                }
                f[c] = kq4_fold(f[c], ds, dm, xs[(size_t)c * ldxs + g], xf[(size_t)c * ldxf + g], _mm512_add_epi32(a0, a1));
            }
        }
    }
    for (int c = 0; c < C; c++)
        _mm512_storeu_ps(out + (size_t)c * ldo, f[c]);
}

static inline __attribute__((always_inline))
void kq5_smb_block(const int C, const uint8_t *wb, int nsb, const int8_t *x, size_t ldx, const float *xs, size_t ldxs,
                   const float *xf, size_t ldxf, float *out, size_t ldo, int pf) {
    // one Q5P block (wb: its nsb super-blocks) x C activation rows, as kq4_smb_block with the masks
    const __m512i m4 = _mm512_set1_epi8(0x0f), m16 = _mm512_set1_epi8(0x10);
    __m512 f[8];
    for (int c = 0; c < C; c++)
        f[c] = _mm512_setzero_ps();
    for (int s = 0; s < nsb; s++) {
        const uint8_t *o = wb + (size_t)s * Q5P_SB;
        __m512 d = _mm512_cvtph_ps(_mm256_load_si256((const __m256i *)o));
        __m512 dmin = _mm512_cvtph_ps(_mm256_load_si256((const __m256i *)(o + 32)));
        for (int j = 0; j < 8; j++) {
            int g = 8 * s + j;
            const uint8_t *qs = o + 320 + 320 * j, *mp = qs + 256;
            smb_pf(o + j * (Q5P_SB / 8), Q5P_SB / 8, pf);
            __m512i w[8];
            for (int cc = 0; cc < 4; cc++) {
                __m512i v = _mm512_load_si512((const void *)(qs + 64 * cc));
                w[2 * cc] = _mm512_or_si512(_mm512_and_si512(v, m4),
                                            _mm512_maskz_mov_epi8((__mmask64)kq_u64(mp + 8 * (2 * cc)), m16));
                w[2 * cc + 1] = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(v, 4), m4),
                                                _mm512_maskz_mov_epi8((__mmask64)kq_u64(mp + 8 * (2 * cc + 1)), m16));
            }
            __m512 ds = _mm512_mul_round_ps(d, kq512_u8ps(o + 64 + 16 * j), Q8P_RND);
            __m512 dm = _mm512_mul_round_ps(dmin, kq512_u8ps(o + 192 + 16 * j), Q8P_RND);
            for (int c = 0; c < C; c++) {
                const int8_t *xg = x + (size_t)c * ldx + 32 * g;
                __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
                for (int cc = 0; cc < 4; cc++) {
                    a0 = _mm512_dpbusd_epi32(a0, w[2 * cc], q8p_bcast4(xg + 8 * cc));
                    a1 = _mm512_dpbusd_epi32(a1, w[2 * cc + 1], q8p_bcast4(xg + 8 * cc + 4));
                }
                f[c] = kq4_fold(f[c], ds, dm, xs[(size_t)c * ldxs + g], xf[(size_t)c * ldxf + g], _mm512_add_epi32(a0, a1));
            }
        }
    }
    for (int c = 0; c < C; c++)
        _mm512_storeu_ps(out + (size_t)c * ldo, f[c]);
}

static inline __attribute__((always_inline))
void kq6_smb_block(const int C, const uint8_t *wb, int nsb, const int8_t *x, size_t ldx, const float *xs, size_t ldxs,
                   const int32_t *c16, size_t ldc, float *out, size_t ldo, int pf) {
    // one Q6P block x C activation rows: per sub-block the 4 steps unpacked once
    __m512 f[8];
    for (int c = 0; c < C; c++)
        f[c] = _mm512_setzero_ps();
    for (int s = 0; s < nsb; s++) {
        const uint8_t *o = wb + (size_t)s * Q6P_SB;
        __m512 d = _mm512_load_ps((const float *)o);
        for (int g8 = 0; g8 < 8; g8++) {
            int g = 8 * s + g8;
            smb_pf(o + g8 * (Q6P_SB / 8), Q6P_SB / 8, pf);
            __m512i I[8];
            for (int c = 0; c < C; c++)
                I[c] = _mm512_setzero_si512();
            for (int h = 0; h < 2; h++) {
                int j = 2 * g8 + h;
                __m512i q[4];
                kq6_unpack4(o + 320 + 192 * j, o + 448 + 192 * j, q);
                __m512i sc = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i *)(o + 64 + 16 * j)));
                for (int c = 0; c < C; c++) {
                    const int8_t *xg = x + (size_t)c * ldx + 32 * g + 16 * h;
                    __m512i t0 = _mm512_set1_epi32(c16[(size_t)c * ldc + 2 * g + h]), t1 = _mm512_setzero_si512();
                    t0 = _mm512_dpbusd_epi32(t0, q[0], q8p_bcast4(xg));
                    t1 = _mm512_dpbusd_epi32(t1, q[1], q8p_bcast4(xg + 4));
                    t0 = _mm512_dpbusd_epi32(t0, q[2], q8p_bcast4(xg + 8));
                    t1 = _mm512_dpbusd_epi32(t1, q[3], q8p_bcast4(xg + 12));
                    I[c] = _mm512_add_epi32(I[c], _mm512_mullo_epi32(_mm512_add_epi32(t0, t1), sc));
                }
            }
            for (int c = 0; c < C; c++)
                f[c] = kq6_fold(f[c], d, xs[(size_t)c * ldxs + g], I[c]);
        }
    }
    for (int c = 0; c < C; c++)
        _mm512_storeu_ps(out + (size_t)c * ldo, f[c]);
}

static inline __attribute__((always_inline))
void kq3_smb_block(const int C, const uint8_t *wb, int nsb, const int8_t *x, size_t ldx, const float *xs, size_t ldxs,
                   const int32_t *c16, size_t ldc, float *out, size_t ldo, int pf) {
    // one Q3P block x C activation rows, as kq6_smb_block (accumulators from c16 / 8, the scales sc - 32)
    __m512 f[8];
    for (int c = 0; c < C; c++)
        f[c] = _mm512_setzero_ps();
    for (int s = 0; s < nsb; s++) {
        const uint8_t *o = wb + (size_t)s * Q3P_SB;
        __m512 d = _mm512_load_ps((const float *)o);
        for (int g8 = 0; g8 < 8; g8++) {
            int g = 8 * s + g8;
            const uint8_t *qg = o + 256 + 192 * g8;
            smb_pf(o + g8 * (Q3P_SB / 8), Q3P_SB / 8, pf);
            __m512i I[8], sc[2];
            for (int c = 0; c < C; c++)
                I[c] = _mm512_setzero_si512();
            kq3_scales(o, g8, &sc[0], &sc[1]);
            for (int h = 0; h < 2; h++) {
                __m512i q[4];
                kq3_unpack4(qg + 64 * h, qg + 128 + 32 * h, q);
                for (int c = 0; c < C; c++) {
                    const int8_t *xg = x + (size_t)c * ldx + 32 * g + 16 * h;
                    __m512i t0 = _mm512_set1_epi32(c16[(size_t)c * ldc + 2 * g + h] / 8), t1 = _mm512_setzero_si512();
                    t0 = _mm512_dpbusd_epi32(t0, q[0], q8p_bcast4(xg));
                    t1 = _mm512_dpbusd_epi32(t1, q[1], q8p_bcast4(xg + 4));
                    t0 = _mm512_dpbusd_epi32(t0, q[2], q8p_bcast4(xg + 8));
                    t1 = _mm512_dpbusd_epi32(t1, q[3], q8p_bcast4(xg + 12));
                    I[c] = _mm512_add_epi32(I[c], _mm512_mullo_epi32(_mm512_add_epi32(t0, t1), sc[h]));
                }
            }
            for (int c = 0; c < C; c++)
                f[c] = kq6_fold(f[c], d, xs[(size_t)c * ldxs + g], I[c]);
        }
    }
    for (int c = 0; c < C; c++)
        _mm512_storeu_ps(out + (size_t)c * ldo, f[c]);
}

// instances per row count C = 1..SMB_C (and Q8P group size)
#define SMB_INST(NAME, BODY) \
    static void NAME##_1 BODY(1) static void NAME##_2 BODY(2) static void NAME##_3 BODY(3) static void NAME##_4 BODY(4) \
    static void NAME##_5 BODY(5) static void NAME##_6 BODY(6) static void NAME##_7 BODY(7) static void NAME##_8 BODY(8)
#define SMB_Q8_64(C) (const uint8_t *wb, int ng, const int8_t *x, size_t ldx, const float *xs, size_t ldxs, const int32_t *corr, \
    size_t ldc, float *out, size_t ldo) { q8p_smb_block(C, 64, wb, ng, x, ldx, xs, ldxs, corr, ldc, out, ldo); }
#define SMB_Q8_32(C) (const uint8_t *wb, int ng, const int8_t *x, size_t ldx, const float *xs, size_t ldxs, const int32_t *corr, \
    size_t ldc, float *out, size_t ldo) { q8p_smb_block(C, 32, wb, ng, x, ldx, xs, ldxs, corr, ldc, out, ldo); }
#define SMB_KQ4(C) (const uint8_t *wb, int nsb, const int8_t *x, size_t ldx, const float *xs, size_t ldxs, const float *xf, \
    size_t ldxf, float *out, size_t ldo, int pf) { kq4_smb_block(C, wb, nsb, x, ldx, xs, ldxs, xf, ldxf, out, ldo, pf); }
#define SMB_KQ5(C) (const uint8_t *wb, int nsb, const int8_t *x, size_t ldx, const float *xs, size_t ldxs, const float *xf, \
    size_t ldxf, float *out, size_t ldo, int pf) { kq5_smb_block(C, wb, nsb, x, ldx, xs, ldxs, xf, ldxf, out, ldo, pf); }
#define SMB_KQ6(C) (const uint8_t *wb, int nsb, const int8_t *x, size_t ldx, const float *xs, size_t ldxs, const int32_t *c16, \
    size_t ldc, float *out, size_t ldo, int pf) { kq6_smb_block(C, wb, nsb, x, ldx, xs, ldxs, c16, ldc, out, ldo, pf); }
#define SMB_KQ3(C) (const uint8_t *wb, int nsb, const int8_t *x, size_t ldx, const float *xs, size_t ldxs, const int32_t *c16, \
    size_t ldc, float *out, size_t ldo, int pf) { kq3_smb_block(C, wb, nsb, x, ldx, xs, ldxs, c16, ldc, out, ldo, pf); }
SMB_INST(q8p_smb64, SMB_Q8_64)
SMB_INST(q8p_smb32, SMB_Q8_32)
SMB_INST(kq4_smb, SMB_KQ4)
SMB_INST(kq5_smb, SMB_KQ5)
SMB_INST(kq6_smb, SMB_KQ6)
SMB_INST(kq3_smb, SMB_KQ3)
#undef SMB_INST
#undef SMB_Q8_64
#undef SMB_Q8_32
#undef SMB_KQ4
#undef SMB_KQ5
#undef SMB_KQ6
#undef SMB_KQ3

typedef void (*SmbQ8Fn)(const uint8_t *, int, const int8_t *, size_t, const float *, size_t, const int32_t *, size_t, float *, size_t);
typedef void (*SmbKq4Fn)(const uint8_t *, int, const int8_t *, size_t, const float *, size_t, const float *, size_t, float *, size_t, int);
typedef void (*SmbKq6Fn)(const uint8_t *, int, const int8_t *, size_t, const float *, size_t, const int32_t *, size_t, float *, size_t, int);
static const SmbQ8Fn smb_q8_64[8] = { q8p_smb64_1, q8p_smb64_2, q8p_smb64_3, q8p_smb64_4, q8p_smb64_5, q8p_smb64_6, q8p_smb64_7, q8p_smb64_8 };
static const SmbQ8Fn smb_q8_32[8] = { q8p_smb32_1, q8p_smb32_2, q8p_smb32_3, q8p_smb32_4, q8p_smb32_5, q8p_smb32_6, q8p_smb32_7, q8p_smb32_8 };
static const SmbKq4Fn smb_kq4[8] = { kq4_smb_1, kq4_smb_2, kq4_smb_3, kq4_smb_4, kq4_smb_5, kq4_smb_6, kq4_smb_7, kq4_smb_8 };
static const SmbKq4Fn smb_kq5[8] = { kq5_smb_1, kq5_smb_2, kq5_smb_3, kq5_smb_4, kq5_smb_5, kq5_smb_6, kq5_smb_7, kq5_smb_8 };
static const SmbKq6Fn smb_kq6[8] = { kq6_smb_1, kq6_smb_2, kq6_smb_3, kq6_smb_4, kq6_smb_5, kq6_smb_6, kq6_smb_7, kq6_smb_8 };
static const SmbKq6Fn smb_kq3[8] = { kq3_smb_1, kq3_smb_2, kq3_smb_3, kq3_smb_4, kq3_smb_5, kq3_smb_6, kq3_smb_7, kq3_smb_8 };

static void smb512_rows(int type, float *out, size_t ldo, const int8_t *xq, size_t ldx, const float *xs, const int32_t *corr,
                        const float *xf, const int32_t *c16, int nb, const uint8_t *wp, int k, int gs, int rb0, int rb1, int pf) {
    // the small-batch kernel over blocks rb0..rb1-1 of W (Q3P .. Q6P by type; any other type: Q8P
    // with group size gs) and activation rows 0..nb-1, SMB_C rows per pass over a block; pf: the K-quant
    // kernels' prefetch distance (bytes, 0: none)
    size_t ldxs = (size_t)ldx / gs;
    if (type != GGML_Q3_K && type != GGML_Q4_K && type != GGML_Q5_K && type != GGML_Q6_K) {
        int ng = k / gs;
        size_t bstride = (size_t)ng * Q8P_GROUP(gs);
        const SmbQ8Fn *fn = gs == 64 ? smb_q8_64 : smb_q8_32;
        for (int rb = rb0; rb < rb1; rb++)
            for (int b = 0; b < nb; b += SMB_C) {
                int c = nb - b < SMB_C ? nb - b : SMB_C;
                fn[c - 1](wp + (size_t)rb * bstride, ng, xq + (size_t)b * ldx, ldx, xs + (size_t)b * ldxs, ldxs,
                          corr + (size_t)b * ng, ng, out + (size_t)b * ldo + (size_t)rb * Q8P_ROWS, ldo);
            }
        return;
    }
    int nsb = k / QK_K, ng = k / 32, n16 = k / 16;
    size_t bstride = (size_t)nsb * kq_sb_bytes(type);
    for (int rb = rb0; rb < rb1; rb++)
        for (int b = 0; b < nb; b += SMB_C) {
            int c = nb - b < SMB_C ? nb - b : SMB_C;
            if (type == GGML_Q4_K || type == GGML_Q5_K)
                (type == GGML_Q4_K ? smb_kq4 : smb_kq5)[c - 1](wp + (size_t)rb * bstride, nsb, xq + (size_t)b * ldx, ldx,
                                                              xs + (size_t)b * ldxs, ldxs, xf + (size_t)b * ng, ng,
                                                              out + (size_t)b * ldo + (size_t)rb * KQ_ROWS, ldo, pf);
            else
                (type == GGML_Q6_K ? smb_kq6 : smb_kq3)[c - 1](wp + (size_t)rb * bstride, nsb, xq + (size_t)b * ldx, ldx,
                                                              xs + (size_t)b * ldxs, ldxs, c16 + (size_t)b * n16, n16,
                                                              out + (size_t)b * ldo + (size_t)rb * KQ_ROWS, ldo, pf);
        }
}

// the small-batch interface of forward.c (AVX-512 only: other builds keep the GEMM, the same results)
#define RUNQ_SMB 1
#define smb_rows smb512_rows

// the K-quant SIMD kernel interface of forward.c (kernels_neon.h implements the same on AArch64)
#define RUNQ_KQX 1
#define kqx_c16 kq512_c16
#define kqx_gemv_rows kq512_gemv_rows
#define kqx_gemm_item kq512_gemm_item
#define KQX_SCRATCH KQ512_SCRATCH
#define kqx_nb q8p_nb
// the Q8_0 SIMD kernel interface (kernels_neon.h: the same on AArch64, Q8N layout)
#define RUNQ_Q8X 1
#define Q8X_GROUP Q8P_GROUP
#define q8x_size q8p_size
#define q8x_ok q8p_ok
#define q8x_pack_blocks q8p_pack_blocks
#define q8x_gemv_corr q8p_gemv_corr
#define q8x_gemv_rows q8p_gemv_rows
#define q8x_gemm_item q8p_gemm_item

#endif // AVX-512 VNNI

/* ---------------------------------------------------------------------------------------------
 * fp32 attention (AVX-512F), for decode (rq_decode) and prefill (rq_forward_batch) alike, through
 * attention_kv() in forward.c; RUNQ_GENERIC or a CPU without AVX-512F uses the generic C version
 * there (attention_vec), which computes the same operations in the same order, so all give
 * bit-identical results.
 *
 * KV cache layout (per layer): blocks of ATT_P = 256 positions (the attention blocks below), in a
 * block the kv heads one after the other, each its 16 tiles of ATT_T = 16 positions: tile tb of head
 * h at (((tb / 16) * n_kv_heads + h) * 256 + (tb % 16) * 16) * head_dim elements, i.e. each (head,
 * block) is one contiguous stream (64-128 KB) and a block's positions of all heads are contiguous
 * (pre-faulted per range, transparent huge pages possible). Inside a tile, K is transposed (dim i,
 * position j at i * 16 + j), so one zmm holds dim i of 16 positions, and V is row-major (j *
 * head_dim + i). (Step 7b: the tiles of the heads used to alternate per tile; with 4-8 KB tiles every
 * tile was a new page 32-64 KB further on and the hardware prefetchers restarted per tile: decode
 * after 3,000 tokens, 8 threads, 16-bit cache ~100 -> ~130 tok/s, prefill +3-5%. One contiguous range
 * per head instead (2 MB per head and layer at -c 4096) was as fast but made the pre-fault ~2x as
 * slow: 8 ranges, no huge pages)
 * bf16 cache (-k bf16, Step 7b): the same tiles with 2-byte elements, stored as fp32 -> bf16 with
 * round-to-nearest-even (f32_bf16 in forward.c), and inside a tile the dims paired so that one 64-byte
 * load gives two zmm of fp32 by a shift / a mask (bf16 -> fp32 is exact: the bits << 16):
 *   K: dims 2p, 2p + 1 of position j at 32 p + 2 j and 32 p + 2 j + 1 (dword lane j: low / high half)
 *   V: position j, dims 32 c + l and 32 c + 16 + l (l < 16) at j * head_dim + 32 c + 2 l and + 1
 * The attention arithmetic below is unchanged (fp32 on the converted values): only the stored K / V
 * differ from the fp32 cache (relative rounding <= 2^-9).
 *
 * Arithmetic, per query vector q (n positions), the definition of the attention. The positions
 * are split into blocks of ATT_P = 256 (a constant: it never depends on the thread count or the
 * chunk size), block b = positions b * 256 .. min(n, (b + 1) * 256) - 1, nb = ceil(n / 256) blocks:
 *   s[t] = (sum over chunks c of ATT_DC = 32 dims, in order, of (fma chain over the dims i of c
 *          of q[i] * k[t][i], from 0)) * scale. (One 128-long chain: ~10% larger deviation of
 *          the logits from a double-precision attention; chunks of 32: as close as the old
 *          vectorized dot product with 16 partial sums)
 *   per block b: m_b = max of its s[t]; e[t] = exp_att(s[t] - m_b) (exp_att: Cephes-style expf,
 *        see att512_exp); l_b: 16 partial sums (lane j: e[t] for t = j mod 16, t in order), then a
 *        fixed tree (lanes j + 8, then + 4, + 2, + 1); o_b[i] = fma chain over the block's t of
 *        e[t] * v[t][i], from 0
 *   nb = 1: out[i] = o_0[i] * inv, inv = 1 / l_0 (one division). (Up to 256 positions this is
 *        exactly the Step 3b definition: same results as before for every prompt that short)
 *   nb > 1: M = max of the m_b; c_b = exp_att(m_b - M); L = fma chain over b of c_b * l_b, from 0;
 *        out[i] = (fma chain over b of c_b * o_b[i], from 0) * (1 / L)   (forward.c: att_combine)
 * The blocks let decode split a long context over threads (flash-decoding: one work item per kv
 * head and block, the last one to finish combines), with the same results as the prefill, which
 * runs all blocks of a vector in one work item.
 * Every operation is rounded (explicit-rounding intrinsics, opaque to -Ofast's contraction and
 * reassociation; the C version is compiled with fp-contract=off, no-associative-math). The
 * results of a query vector do not depend on the other vectors processed with it, so decode
 * (2 vectors: the query heads of one kv head) and prefill (2 x ATT_ROWS vectors) agree.
 *
 * Kernel: groups of up to ATT512_NV = 12 query vectors of one kv head, over a range of blocks.
 * Scores: register tile of NV vectors x NT tiles (NV * NT <= 24 accumulators), the K tile loaded
 * once per dim for all NV vectors (q broadcast; no horizontal sums); scores stored per tile as
 * [NV][16]. Softmax per vector and block. V: register tile NV vectors x DW zmm of dims, the score
 * of each vector broadcast per position (each pass of DW * 16 dims runs over the call's blocks, the
 * accumulators restarting at each block); a vector's positions beyond its own n (causal rows of a
 * block) are skipped, so the fma chain is exactly that of the decode path.
 * att512_kv splits a call's vectors into groups of at most ATT512_G = 6 (near-equal sizes): NV 6
 * runs NT = 4 / DW = 4 (6 broadcasts per 24 FMAs instead of 12 at NV 12), ~108 vs ~99 GFLOPS on one
 * Zen 4 core with the K / V in L2 (peak 111: one zmm FMA per cycle); and it sweeps a long range
 * ATT512_BC = 2 blocks at a time, all the groups over those blocks before the next ones, so a
 * block's K / V come from memory once per call instead of once per group (TODO G4: a 2,560-token
 * chunk at 98k positions on the 27B-shaped model, 8 threads, read ~3 x 100 MB per item before).
 * Neither changes a bit: the vectors are independent and each block's partial result is computed
 * whole by one call.
 */
#if defined(__AVX512F__) && !defined(RUNQ_GENERIC)
#define RUNQ_ATT512 1
#include <immintrin.h>
#include <float.h>

#define ATT512_NV 12    // max query vectors per group
#define ATT512_HDMAX 256
#ifndef ATT512_BC
#define ATT512_BC 2     // blocks per step of a long range's sweep (att512_kv)
#endif
#ifndef ATT512_G
#define ATT512_G 6      // query vectors per group, at most (att512_kv: ceil(nv / 6) groups)
#endif
#define ATT512_RND (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)

static inline __m512 att512_exp(__m512 x) {
    // exp(x) for x <= 0 (clamped at ATT_EXP_MIN): x = n ln2 + r, |r| <= ln2 / 2, Cephes polynomial
    // exp(r) = 1 + r + r^2 P(r), times 2^n by an exponent add. Max rel. error ~2 ulp
    x = _mm512_max_ps(x, _mm512_set1_ps(ATT_EXP_MIN));
    __m512 n = _mm512_roundscale_ps(_mm512_mul_round_ps(x, _mm512_set1_ps(ATT_LOG2E), ATT512_RND), ATT512_RND);
    __m512 r = _mm512_fmadd_ps(n, _mm512_set1_ps(-ATT_LN2_HI), x);
    r = _mm512_fmadd_ps(n, _mm512_set1_ps(-ATT_LN2_LO), r);
    __m512 z = _mm512_mul_round_ps(r, r, ATT512_RND);
    __m512 y = _mm512_set1_ps(ATT_P0);
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(ATT_P1));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(ATT_P2));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(ATT_P3));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(ATT_P4));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(ATT_P5));
    y = _mm512_fmadd_ps(y, z, r);
    y = _mm512_add_round_ps(y, _mm512_set1_ps(1.0f), ATT512_RND);
    __m512i e = _mm512_slli_epi32(_mm512_cvt_roundps_epi32(n, ATT512_RND), 23);
    return _mm512_castsi512_ps(_mm512_add_epi32(_mm512_castps_si512(y), e));
}

static inline float att512_sum16(__m512 v) {
    // fixed reduction tree: lane j += lane j + 8, then j + 4, j + 2, j + 1 (as attention_vec)
    v = _mm512_add_round_ps(v, _mm512_shuffle_f32x4(v, v, 0x4e), ATT512_RND);
    v = _mm512_add_round_ps(v, _mm512_shuffle_f32x4(v, v, 0xb1), ATT512_RND);
    v = _mm512_add_round_ps(v, _mm512_permute_ps(v, 0x4e), ATT512_RND);
    v = _mm512_add_round_ps(v, _mm512_permute_ps(v, 0xb1), ATT512_RND);
    return _mm512_cvtss_f32(v);
}

static inline __m512 att512_bf_lo(__m512i v) { return _mm512_castsi512_ps(_mm512_slli_epi32(v, 16)); }
static inline __m512 att512_ld16(const int KT, const void *p, size_t i) {
    // 16 cache elements from index i as fp32: KT 0 fp32, 2 f16 (vcvtph2ps, exact)
    if (KT == 2)
        return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)((const uint16_t *)p + i)));
    return _mm512_loadu_ps((const float *)p + i);
}
static inline __m512 att512_bf_hi(__m512i v) {
    return _mm512_castsi512_ps(_mm512_and_si512(v, _mm512_set1_epi32((int)0xffff0000u)));
}

static inline __attribute__((always_inline))
void att512_qk(const int NV, const int NT, const int KT, const float *qt, const void *kt, size_t ts, int hd, __m512 scale,
               float *s) {
    // scores of NV query vectors (transposed: qt[i * NV + r]) for NT tiles (tile u at kt + u * ts
    // elements) -> s[(u * NV + r) * 16 + j]: per ATT_DC dims an fma chain, the chunks added in order
    // (the running sum kept in s), times scale. KT (cache type) 1: bf16 tiles, dims i, i + 1 of 16
    // positions in one 64-byte load (low / high halves of the dwords); 2: f16; converted exactly, the
    // same fma chains
    for (int i0 = 0; i0 < hd; i0 += ATT_DC) {
        __m512 acc[ATT512_NV][4];
        for (int r = 0; r < NV; r++)
            for (int u = 0; u < NT; u++)
                acc[r][u] = _mm512_setzero_ps();
        if (KT == 1) { // (the pair loaded per dim, as the fp32 loop: converting both halves at once spilled)
            const uint16_t *kb = (const uint16_t *)kt;
            for (int i = i0; i < i0 + ATT_DC; i++) {
                __m512 k[4];
                for (int u = 0; u < NT; u++) {
                    __m512i v = _mm512_loadu_si512((const void *)(kb + u * ts + (size_t)(i & ~1) * ATT_T));
                    k[u] = i & 1 ? att512_bf_hi(v) : att512_bf_lo(v);
                }
                for (int r = 0; r < NV; r++) {
                    __m512 qb = _mm512_set1_ps(qt[i * NV + r]);
                    for (int u = 0; u < NT; u++)
                        acc[r][u] = _mm512_fmadd_ps(k[u], qb, acc[r][u]);
                }
            }
        } else {
            for (int i = i0; i < i0 + ATT_DC; i++) {
                __m512 k[4];
                for (int u = 0; u < NT; u++)
                    k[u] = att512_ld16(KT, kt, u * ts + (size_t)i * ATT_T);
                for (int r = 0; r < NV; r++) {
                    __m512 qb = _mm512_set1_ps(qt[i * NV + r]);
                    for (int u = 0; u < NT; u++)
                        acc[r][u] = _mm512_fmadd_ps(k[u], qb, acc[r][u]);
                }
            }
        }
        for (int u = 0; u < NT; u++)
            for (int r = 0; r < NV; r++) {
                float *o = s + (u * NV + r) * ATT_T;
                __m512 a = acc[r][u];
                if (i0 > 0)
                    a = _mm512_add_round_ps(_mm512_loadu_ps(o), a, ATT512_RND);
                if (i0 + ATT_DC == hd)
                    a = _mm512_mul_round_ps(a, scale, ATT512_RND);
                _mm512_storeu_ps(o, a);
            }
    }
}

#define ATT512_PV_STEP(COND) do { \
        const float *p_ = s + (size_t)(t >> 4) * NV * ATT_T + (t & 15); \
        size_t vo_ = (size_t)(t / ATT_P) * bs + (size_t)((t % ATT_P) >> 4) * ts + (size_t)(t & 15) * hd + d0; \
        __m512 vv[8]; \
        if (KT == 1) /* bf16: dims d0 + 32 c .. + 15 / + 16 .. + 31 in the low / high halves of a 64-byte load */ \
            for (int u = 0; u < DW; u += 2) { \
                __m512i w_ = _mm512_loadu_si512((const void *)((const uint16_t *)vt + vo_ + u * 16)); \
                vv[u] = att512_bf_lo(w_); \
                vv[u + 1] = att512_bf_hi(w_); \
            } \
        else \
            for (int u = 0; u < DW; u++) \
                vv[u] = att512_ld16(KT, vt, vo_ + u * 16); \
        for (int r = 0; r < NV; r++) \
            if (COND) { \
                __m512 pb = _mm512_set1_ps(p_[r * ATT_T]); \
                for (int u = 0; u < DW; u++) \
                    acc[r][u] = _mm512_fmadd_ps(vv[u], pb, acc[r][u]); \
            } \
    } while (0)

static inline __attribute__((always_inline))
void att512_pv(const int NV, const int DW, const int KT, const float *s, int nbk, const int *np, const void *vt, size_t ts,
               size_t bs, int hd, const float *inv, float *const *out, int ob, size_t pst) {
    // inv != NULL (nbk = 1): out[r][i] = (sum over t < np[r] of e[r][t] * v[t][i]) * inv[r], e in s
    // as [tile][NV][16]. inv NULL: per block j < nbk of ATT_P positions the unnormalized sum over its
    // positions t < np[r], into out[r] + (ob + j) * pst (blocks vector r reaches). DW zmm (16 * DW
    // dims) per pass; the passes run over all the blocks (long V streams). KT: cache type (bf16: DW even)
    for (int d0 = 0; d0 < hd; d0 += DW * 16) {
        for (int j = 0; j < nbk; j++) {
            int tb = j * ATT_P, nmin = tb + ATT_P, nmax = tb;
            for (int r = 0; r < NV; r++) {
                int e = np[r] < tb + ATT_P ? np[r] : tb + ATT_P;
                nmin = e < nmin ? e : nmin;
                nmax = e > nmax ? e : nmax;
            }
            if (inv) { nmin = np[0]; nmax = np[0]; for (int r = 1; r < NV; r++) { nmin = np[r] < nmin ? np[r] : nmin; nmax = np[r] > nmax ? np[r] : nmax; } }
            __m512 acc[ATT512_NV][8];
            for (int r = 0; r < NV; r++)
                for (int u = 0; u < DW; u++)
                    acc[r][u] = _mm512_setzero_ps();
            int t = tb;
            for (; t < nmin; t++)
                ATT512_PV_STEP(1);
            for (; t < nmax; t++) // causal tail: only the vectors that see position t
                ATT512_PV_STEP(t < np[r]);
            for (int r = 0; r < NV; r++) {
                if (!inv) { // a block's partial result: unnormalized
                    if (np[r] > tb)
                        for (int u = 0; u < DW; u++)
                            _mm512_storeu_ps(out[r] + (size_t)(ob + j) * pst + d0 + u * 16, acc[r][u]);
                    continue;
                }
                __m512 iv = _mm512_set1_ps(inv[r]);
                for (int u = 0; u < DW; u++)
                    _mm512_storeu_ps(out[r] + d0 + u * 16, _mm512_mul_round_ps(acc[r][u], iv, ATT512_RND));
            }
        }
    }
}
#undef ATT512_PV_STEP

static inline __attribute__((always_inline))
void att512_group_t(const int KT, int nv, const float *const *q, float *const *out, float *const *part, const int *np,
                    int b0, int b1, const void *kt0, const void *vt0, size_t bs, int hd, float scale, float *s) {
    // nv <= ATT512_NV query vectors of one kv head, vector r over positions 0..np[r]-1 (kt / vt: tile
    // 0 of the head; ts: elements (KT 0: fp32, 1: bf16, 2: f16) from one tile of the head to the next). part
    // NULL: the attention into out[r] (all np <= ATT_P; b0 = 0). Else the
    // partial results of blocks b0..b1-1 (ATT_P positions each): block b of vector r at part[r] + b *
    // ATT_PSTRIDE(hd) (o[hd] unnormalized, m, l), for the blocks the vector reaches. s: scratch of
    // nv * (positions in the blocks) floats. Scores, softmax per block, V: each over all the blocks
    size_t pst = ATT_PSTRIDE(hd), ts = (size_t)ATT_T * hd, o0 = (size_t)b0 * bs, esz = KT ? 2 : 4;
    int npr[ATT512_NV], nmax = 0; // positions relative to block b0, at most to the end of block b1 - 1
    for (int r = 0; r < nv; r++) {
        int n = np[r] - b0 * ATT_P, e = (b1 - b0) * ATT_P;
        npr[r] = n < 0 ? 0 : n < e ? n : e;
        nmax = npr[r] > nmax ? npr[r] : nmax;
    }
    const char *kt = (const char *)kt0 + o0 * esz, *vt = (const char *)vt0 + o0 * esz;
    int ntl = (nmax + ATT_T - 1) / ATT_T, nbk = part ? (nmax + ATT_P - 1) / ATT_P : 1;
    float qt[ATT512_HDMAX * ATT512_NV] __attribute__((aligned(64)));
    for (int i = 0; i < hd; i++)
        for (int r = 0; r < nv; r++)
            qt[i * nv + r] = q[r][i];
    __m512 sc = _mm512_set1_ps(scale);

    // scores
    #define ATT512_QK(V, T) case (V) * 8 + (T): att512_qk(V, T, KT, qt, kt + ((size_t)(tb / ATT_PT) * bs + (size_t)(tb % ATT_PT) * ts) * esz, ts, hd, sc, s + (size_t)tb * nv * ATT_T); break;
    #define ATT512_QK_ANY(nt) switch (nv * 8 + (nt)) { \
        ATT512_QK(1, 1) ATT512_QK(2, 1) ATT512_QK(3, 1) ATT512_QK(4, 1) ATT512_QK(5, 1) ATT512_QK(6, 1) \
        ATT512_QK(7, 1) ATT512_QK(8, 1) ATT512_QK(9, 1) ATT512_QK(10, 1) ATT512_QK(11, 1) ATT512_QK(12, 1) \
        ATT512_QK(1, 4) ATT512_QK(2, 4) ATT512_QK(3, 4) ATT512_QK(4, 4) ATT512_QK(5, 4) ATT512_QK(6, 4) \
        ATT512_QK(7, 2) ATT512_QK(8, 2) ATT512_QK(9, 2) ATT512_QK(10, 2) ATT512_QK(11, 2) ATT512_QK(12, 2) \
        default: break; }
    int ntw = nv <= 6 ? 4 : 2, tb = 0;
    for (; tb + ntw <= ntl; tb += ntw)
        ATT512_QK_ANY(ntw)
    for (; tb < ntl; tb++)
        ATT512_QK_ANY(1)
    #undef ATT512_QK_ANY
    #undef ATT512_QK

    // softmax per vector and block (scores of vector r, tile tb at s + (tb * nv + r) * 16)
    float inv[ATT512_NV];
    size_t st = (size_t)nv * ATT_T;
    for (int r = 0; r < nv; r++)
        for (int j = 0; j < nbk; j++) {
            int n = npr[r] - j * ATT_P;
            if (n <= 0)
                break;
            n = n < ATT_P || !part ? n : ATT_P;
            int nt = (n + ATT_T - 1) / ATT_T;
            float *sr = s + r * ATT_T + (size_t)j * (ATT_P / ATT_T) * st;
            __mmask16 last = (__mmask16)(0xffffu >> (nt * ATT_T - n));
            __m512 mx = _mm512_set1_ps(-FLT_MAX);
            for (int tb = 0; tb < nt - 1; tb++)
                mx = _mm512_max_ps(mx, _mm512_loadu_ps(sr + tb * st));
            mx = _mm512_mask_max_ps(mx, last, mx, _mm512_loadu_ps(sr + (nt - 1) * st));
            __m512 m = _mm512_set1_ps(_mm512_reduce_max_ps(mx)), sum = _mm512_setzero_ps();
            for (int tb = 0; tb < nt; tb++) {
                __m512 e = att512_exp(_mm512_sub_round_ps(_mm512_loadu_ps(sr + tb * st), m, ATT512_RND));
                if (tb == nt - 1)
                    e = _mm512_maskz_mov_ps(last, e);
                _mm512_storeu_ps(sr + tb * st, e);
                sum = _mm512_add_round_ps(sum, e, ATT512_RND);
            }
            float tot = att512_sum16(sum);
            if (part) {
                float *ml = part[r] + (size_t)(b0 + j) * pst + hd;
                ml[0] = _mm_cvtss_f32(_mm512_castps512_ps128(m));
                ml[1] = tot;
            } else {
                inv[r] = _mm_cvtss_f32(_mm_div_round_ss(_mm_set_ss(1.0f), _mm_set_ss(tot), ATT512_RND));
            }
        }

    // weighted sum of V
    #define ATT512_PV(V, D) case V: att512_pv(V, D, KT, s, nbk, npr, vt, ts, bs, hd, part ? NULL : inv, part ? part : out, b0, pst); break;
    switch (nv) {
        ATT512_PV(1, 8) ATT512_PV(2, 8) ATT512_PV(3, 4) ATT512_PV(4, 4) ATT512_PV(5, 4) ATT512_PV(6, 4)
        ATT512_PV(7, 2) ATT512_PV(8, 2) ATT512_PV(9, 2) ATT512_PV(10, 2) ATT512_PV(11, 2) ATT512_PV(12, 2)
        default: break;
    }
    #undef ATT512_PV
}

static void att512_group_f32(int nv, const float *const *q, float *const *out, float *const *part, const int *np, int b0,
                             int b1, const void *kt, const void *vt, size_t bs, int hd, float scale, float *s) {
    att512_group_t(0, nv, q, out, part, np, b0, b1, kt, vt, bs, hd, scale, s);
}

static void att512_group_bf16(int nv, const float *const *q, float *const *out, float *const *part, const int *np, int b0,
                              int b1, const void *kt, const void *vt, size_t bs, int hd, float scale, float *s) {
    att512_group_t(1, nv, q, out, part, np, b0, b1, kt, vt, bs, hd, scale, s);
}

static void att512_group_f16(int nv, const float *const *q, float *const *out, float *const *part, const int *np, int b0,
                             int b1, const void *kt, const void *vt, size_t bs, int hd, float scale, float *s) {
    att512_group_t(2, nv, q, out, part, np, b0, b1, kt, vt, bs, hd, scale, s);
}

static inline int att512_ok(int hd) { return hd % 128 == 0 && hd <= ATT512_HDMAX; }

static void att512_kv(int kvt, int nv, const float *const *q, float *const *out, float *const *part, const int *np, int b0,
                      int b1, const void *kt, const void *vt, size_t bs, int hd, float scale, float *s) {
    // nv query vectors of one kv head (any nv), in ceil(nv / ATT512_G) groups of near-equal size (att512_group_t);
    // kvt: cache type. A range of more than ATT512_BC blocks (prefill against a long cache) is swept ATT512_BC
    // blocks at a time, every group over those blocks before the next ones: their K / V (256 KB per block and kv
    // head in f16 / bf16) come from memory once and stay in L2 for the other groups. (The blocks' results do not
    // depend on the range a call covers.) s: att_scores_floats (internal.h), at least ATT512_NV * ATT512_BC * ATT_P
    if (part && b1 - b0 > ATT512_BC) {
        for (int c0 = b0; c0 < b1; c0 += ATT512_BC)
            att512_kv(kvt, nv, q, out, part, np, c0, c0 + ATT512_BC < b1 ? c0 + ATT512_BC : b1, kt, vt, bs, hd, scale, s);
        return;
    }
    int ng = (nv + ATT512_G - 1) / ATT512_G;
    for (int i = 0, g = 0; i < ng; i++) {
        int gn = (nv - g) / (ng - i);
        (kvt == 2 ? att512_group_f16 : kvt ? att512_group_bf16 : att512_group_f32)(gn, q + g, out + g, part ? part + g : NULL, np + g, b0, b1,
                                                                                   kt, vt, bs, hd, scale, s);
        g += gn;
    }
}

/* ---------------------------------------------------------------------------------------------
 * bf16 dot-product attention (the sequence's KV type RQ_KV_BF16_DOT / RQ_KV_F16_DOT, runq --attn bf16; TODO G4 (b)):
 * the two products of the attention as vdpbf16ps does them, on q, K, the softmax weights and V rounded to bf16, so
 * twice the dot-product rate of the fp32 FMAs (Zen 4: one vdpbf16ps per cycle per core, 64 FLOP). Opt-in: it is less
 * accurate than the fp32 attention (README, docs/qwen38-speculative.md "G4"). A sequence's every path (prefill,
 * decode, verify, batched decode) uses its type, so the invariants (batched == sequential, threads, builds) hold.
 * Definition, per query vector and kv head, as the fp32 attention above except:
 *   bf16(x) = f32_bf16_daz(x) = vcvtneps2bf16: round to nearest even, a denormal input flushed to a zero of its sign;
 *   q' = bf16(q[i]); K', V': a bf16-dot cache stores f32_bf16(k), f32_bf16(v) (as -k bf16: a bf16 denormal is read as
 *        zero by the products below); an f16-dot cache stores f16 (as -k f16) and is read as bf16(the f16 value as
 *        fp32), i.e. rounded again (11 -> 8 significant bits; never a denormal);
 *   dot(c, a, b) over a pair (a0, a1), (b0, b1): vdpbf16ps's lane, two FMAs, the odd element first:
 *        c = fma(a1, b1, c); c = fma(a0, b0, c), each rounded to nearest even, with DAZ on all inputs and FTZ on the
 *        results (flush when tiny after rounding, the sign kept: x86's MXCSR FTZ / DAZ rules; checked against the
 *        instruction on this CPU by build/bf16dp_test, tests/bf16dp_test.c);
 *   s[t] = (sum over chunks of ATT_DC = 32 dims, in order, of (the dot chain over the chunk's 16 dim pairs (2p,
 *        2p + 1), in order, from 0, of (q'[2p], q'[2p + 1]) x (k'[t][2p], k'[t][2p + 1]))) * scale;
 *   the softmax per block as before (m_b, e[t] = exp_att(s[t] - m_b) >= ~2^-125.1: no denormal), then w[t] = bf16(e[t])
 *        and l_b = the 16-lane
 *        sum (+ tree) of the w[t] (the rounded weights, so that they are what the o_b chain multiplies);
 *   o_b[i] = the dot chain over the block's position pairs (t, t + 1), t = its first position + 2 j, in order, from 0,
 *        of (w[t], w[t + 1]) x (v'[t][i], v'[t + 1][i]); a position t + 1 the vector does not reach (its last
 *        position t, an odd count in the block) enters as w = +0, v = +0 (whatever the cache holds there);
 *   the combine of the blocks (att_combine) and the normalization unchanged (fp32).
 * The whole call runs with FTZ / DAZ (forward.c's attention_part sets MXCSR for it on x86), so the fp32 parts round
 * the same in every build and the C twin of the products (forward.c, attention_vec_dot: the two fmaf per pair) equals
 * the instruction. Without AVX512_BF16 (an AVX-512 CPU before Zen 4 / Cooper Lake) att512_dp emulates the lane by
 * the same two FMAs: the same results, at the fp32 rate.
 * Layouts: K as the bf16 cache (dims 2p, 2p + 1 of position j at 32 p + 2 j, + 1: one 64-byte load = 16 positions x
 * a dim pair, the q pair broadcast); V of a bf16-dot cache in position pairs (positions 2m, 2m + 1 of a tile, dim i at
 * 2m * head_dim + 2 i, + 1: one 64-byte load = 16 dims x a position pair, the weight pair broadcast). An f16-dot
 * cache keeps the f16 layout; each call converts its range (at most ATT512_BC blocks) into those two layouts in the
 * thread's scratch (conv), once for all its groups. The softmax weights are stored as bf16 over their fp32 scores
 * (the first 32 bytes of each [tile][vector] slot of s).
 */
#ifdef __AVX512BF16__
#define ATT512_DP_HW 1 // (bf16dp_test: the instruction is used)
static inline __m512 att512_dp(__m512 c, __m512i a, __m512i b) { return _mm512_dpbf16_ps(c, (__m512bh)a, (__m512bh)b); }
static inline __m256i att512_cvt_bf16(__m512 x) { return (__m256i)_mm512_cvtneps_pbh(x); } // = f32_bf16_daz
#else
#define ATT512_DP_HW 0
static inline __m512 att512_dp(__m512 c, __m512i a, __m512i b) {
    // vdpbf16ps's lane without AVX512_BF16: the same two FMAs, the odd elements first (FTZ / DAZ: MXCSR, set by the caller)
    c = _mm512_fmadd_ps(att512_bf_hi(a), att512_bf_hi(b), c);
    return _mm512_fmadd_ps(att512_bf_lo(a), att512_bf_lo(b), c);
}
static inline __m256i att512_cvt_bf16(__m512 x) {
    // f32_bf16_daz of 16 values (vcvtneps2bf16)
    __m512i u = _mm512_castps_si512(x);
    __m512i r = _mm512_srli_epi32(_mm512_add_epi32(_mm512_add_epi32(u, _mm512_set1_epi32(0x7fff)),
                                                   _mm512_and_si512(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(1))), 16);
    __mmask16 nan = _mm512_cmpgt_epu32_mask(_mm512_and_si512(u, _mm512_set1_epi32(0x7fffffff)), _mm512_set1_epi32(0x7f800000));
    __mmask16 den = _mm512_testn_epi32_mask(u, _mm512_set1_epi32(0x7f800000));
    r = _mm512_mask_or_epi32(r, nan, _mm512_srli_epi32(u, 16), _mm512_set1_epi32(0x40));
    r = _mm512_mask_and_epi32(r, den, _mm512_srli_epi32(u, 16), _mm512_set1_epi32(0x8000));
    return _mm512_cvtepi32_epi16(r);
}
#endif

static inline __m512 att512_bf16_ps(__m256i w) { return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(w), 16)); }

static inline __m512i att512_f16_pairs(const uint16_t *a, const uint16_t *b) {
    // 16 f16 values at a and 16 at b -> 16 dwords bf16(a[l]) | bf16(b[l]) << 16 (fp32 exactly, then f32_bf16)
    __m512 fa = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)a));
    __m512 fb = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)b));
#ifdef __AVX512BF16__
    __m512i w = (__m512i)_mm512_cvtne2ps_pbh(fb, fa); // (fa's 16 below fb's; an f16 value is never an fp32 denormal)
#else
    __m512i w = _mm512_inserti64x4(_mm512_castsi256_si512(att512_cvt_bf16(fa)), att512_cvt_bf16(fb), 1);
#endif
    const __m512i il = _mm512_set_epi16(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8,
                                        23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);
    return _mm512_permutexvar_epi16(il, w); // words a0 b0 a1 b1 ...
}

static void att512_dot_conv(uint16_t *kd, uint16_t *vd, const uint16_t *kt, const uint16_t *vt, int ntl, size_t bs, int hd) {
    // the first ntl tiles of a range of an f16 cache (one kv head; bs: elements from a block to its next) as bf16 tiles
    // in the dot layouts, blocks ATT_P * hd elements apart in kd / vd
    size_t ts = (size_t)ATT_T * hd;
    for (int tb = 0; tb < ntl; tb++) {
        size_t o = (size_t)(tb / ATT_PT) * bs + (size_t)(tb % ATT_PT) * ts, od = (size_t)tb * ts;
        const uint16_t *k = kt + o, *v = vt + o;
        for (int p = 0; p < hd / 2; p++) // K: dims 2p, 2p + 1 (rows of 16 positions) -> per position a pair
            _mm512_storeu_si512((void *)(kd + od + (size_t)p * 2 * ATT_T), att512_f16_pairs(k + (size_t)2 * p * ATT_T, k + (size_t)(2 * p + 1) * ATT_T));
        for (int m = 0; m < ATT_T / 2; m++) // V: positions 2m, 2m + 1 (rows of hd dims) -> per dim a pair
            for (int d = 0; d < hd; d += 16)
                _mm512_storeu_si512((void *)(vd + od + (size_t)2 * m * hd + 2 * d),
                                    att512_f16_pairs(v + (size_t)2 * m * hd + d, v + (size_t)(2 * m + 1) * hd + d));
    }
}

typedef struct { const char *p[2 * ATT512_BC]; size_t n[2 * ATT512_BC]; int nr, k; } Att512Pf; // lines to prefetch

static inline const char *att512_pf_take(Att512Pf *f, int *n) {
    // up to *n lines of f's current region (*n: the lines taken, 0 if none left)
    while (f && f->k < f->nr && !f->n[f->k])
        f->k++;
    if (!f || f->k >= f->nr) {
        *n = 0;
        return NULL;
    }
    size_t m = f->n[f->k] < (size_t)*n ? f->n[f->k] : (size_t)*n;
    const char *p = f->p[f->k];
    f->p[f->k] += m * 64;
    f->n[f->k] -= m;
    *n = (int)m;
    return p;
}

static inline __attribute__((always_inline))
void att512_qk_dot(const int NV, const int NT, const uint32_t *qp, const uint16_t *kt, size_t ts, int hd, __m512 scale, float *s,
                   const char *pf, int npf) {
    // att512_qk on bf16 pairs: qp[r * hd / 2 + p] = vector r's dims 2p, 2p + 1 (q'), tile u at kt + u * ts; per ATT_DC dims
    // a dot chain over its 16 pairs, the chunks added in order, times scale -> s[(u * NV + r) * 16 + j]. Meanwhile npf
    // (<= hd) cache lines from pf are prefetched into L2, two per dim pair (the next range of a sweep: att512_kv_dot)
    for (int i0 = 0; i0 < hd; i0 += ATT_DC) {
        __m512 acc[ATT512_NV][4];
        for (int r = 0; r < NV; r++)
            for (int u = 0; u < NT; u++)
                acc[r][u] = _mm512_setzero_ps();
        for (int p = i0 / 2; p < (i0 + ATT_DC) / 2; p++) {
            if (2 * p < npf) {
                _mm_prefetch(pf + (size_t)128 * p, _MM_HINT_T1);
                _mm_prefetch(pf + (size_t)128 * p + 64, _MM_HINT_T1);
            }
            __m512i k[4];
            for (int u = 0; u < NT; u++)
                k[u] = _mm512_loadu_si512((const void *)(kt + u * ts + (size_t)p * 2 * ATT_T));
            for (int r = 0; r < NV; r++) {
                __m512i qb = _mm512_set1_epi32((int)qp[(size_t)r * (hd / 2) + p]);
                for (int u = 0; u < NT; u++)
                    acc[r][u] = att512_dp(acc[r][u], k[u], qb);
            }
        }
        for (int u = 0; u < NT; u++)
            for (int r = 0; r < NV; r++) {
                float *o = s + (u * NV + r) * ATT_T;
                __m512 a = acc[r][u];
                if (i0 > 0)
                    a = _mm512_add_round_ps(_mm512_loadu_ps(o), a, ATT512_RND);
                if (i0 + ATT_DC == hd)
                    a = _mm512_mul_round_ps(a, scale, ATT512_RND);
                _mm512_storeu_ps(o, a);
            }
    }
}

#define ATT512_PVD_STEP(VSEL, COND) do { \
        const uint32_t *p_ = (const uint32_t *)(s + (size_t)(t >> 4) * NV * ATT_T) + ((t & 15) >> 1); \
        size_t vo_ = (size_t)(t / ATT_P) * bs + (size_t)((t % ATT_P) >> 4) * ts + (size_t)(t & 14) * hd + 2 * (size_t)d0; \
        __m512i vv[8]; \
        for (int u = 0; u < DW; u++) \
            vv[u] = _mm512_loadu_si512((const void *)(vt + vo_ + u * 32)); \
        for (int r = 0; r < NV; r++) \
            if (COND) { \
                __m512i pb = _mm512_set1_epi32((int)p_[r * ATT_T]); \
                for (int u = 0; u < DW; u++) \
                    acc[r][u] = att512_dp(acc[r][u], VSEL, pb); \
            } \
    } while (0)

static inline __attribute__((always_inline))
void att512_pv_dot(const int NV, const int DW, const float *s, int nbk, const int *np, const uint16_t *vt, size_t ts, size_t bs,
                   int hd, const float *inv, float *const *out, int ob, size_t pst) {
    // att512_pv on position pairs: the bf16 weights in s (the pair of positions 2m, 2m + 1 of tile tb for vector r: dword m
    // of s + (tb * NV + r) * 16), V in the pair layout; per block the chain over its pairs, a vector's last pair with only
    // its even position (np odd) as (w, +0) x (v, +0): the weight is 0 there (the softmax's mask), the V half cleared
    const __m512i lo16 = _mm512_set1_epi32(0xffff);
    for (int d0 = 0; d0 < hd; d0 += DW * 16) {
        for (int j = 0; j < nbk; j++) {
            int tb = j * ATT_P, nmin = tb + ATT_P, nmax = tb;
            for (int r = 0; r < NV; r++) {
                int e = np[r] < tb + ATT_P ? np[r] : tb + ATT_P;
                nmin = e < nmin ? e : nmin;
                nmax = e > nmax ? e : nmax;
            }
            if (inv) { nmin = np[0]; nmax = np[0]; for (int r = 1; r < NV; r++) { nmin = np[r] < nmin ? np[r] : nmin; nmax = np[r] > nmax ? np[r] : nmax; } }
            __m512 acc[ATT512_NV][8];
            for (int r = 0; r < NV; r++)
                for (int u = 0; u < DW; u++)
                    acc[r][u] = _mm512_setzero_ps();
            int t = tb;
            for (; t + 2 <= nmin; t += 2)
                ATT512_PVD_STEP(vv[u], 1);
            for (; t < nmax; t += 2) { // causal tail: the vectors that see position t, t + 1 or only t
                ATT512_PVD_STEP(vv[u], np[r] > t + 1);
                ATT512_PVD_STEP(_mm512_and_si512(vv[u], lo16), np[r] == t + 1);
            }
            for (int r = 0; r < NV; r++) {
                if (!inv) { // a block's partial result: unnormalized
                    if (np[r] > tb)
                        for (int u = 0; u < DW; u++)
                            _mm512_storeu_ps(out[r] + (size_t)(ob + j) * pst + d0 + u * 16, acc[r][u]);
                    continue;
                }
                __m512 iv = _mm512_set1_ps(inv[r]);
                for (int u = 0; u < DW; u++)
                    _mm512_storeu_ps(out[r] + d0 + u * 16, _mm512_mul_round_ps(acc[r][u], iv, ATT512_RND));
            }
        }
    }
}
#undef ATT512_PVD_STEP

static void att512_group_dot(int nv, const float *const *q, float *const *out, float *const *part, const int *np, int b0, int b1,
                             const uint16_t *kt, const uint16_t *vt, size_t bs, int hd, float scale, float *s, Att512Pf *pf) {
    // att512_group_t for the bf16 dot products: kt / vt hold block b0 of the kv head (the dot layouts), blocks bs
    // elements apart; the rest as att512_group_t. pf (or NULL): lines to prefetch meanwhile (in the scores' loop)
    size_t pst = ATT_PSTRIDE(hd), ts = (size_t)ATT_T * hd;
    int npr[ATT512_NV], nmax = 0;
    for (int r = 0; r < nv; r++) {
        int n = np[r] - b0 * ATT_P, e = (b1 - b0) * ATT_P;
        npr[r] = n < 0 ? 0 : n < e ? n : e;
        nmax = npr[r] > nmax ? npr[r] : nmax;
    }
    int ntl = (nmax + ATT_T - 1) / ATT_T, nbk = part ? (nmax + ATT_P - 1) / ATT_P : 1;
    uint32_t qp[ATT512_HDMAX / 2 * ATT512_NV] __attribute__((aligned(64))); // vector r's dim pairs at r * hd / 2
    for (int r = 0; r < nv; r++)
        for (int i = 0; i < hd; i += 16) // (bf16 of 16 dims at a time)
            _mm256_storeu_si256((__m256i *)(qp + (size_t)r * (hd / 2) + i / 2), att512_cvt_bf16(_mm512_loadu_ps(q[r] + i)));
    __m512 sc = _mm512_set1_ps(scale);

    #define ATT512_QKD(V, T) case (V) * 8 + (T): att512_qk_dot(V, T, qp, kt + (size_t)(tb / ATT_PT) * bs + (size_t)(tb % ATT_PT) * ts, ts, hd, sc, s + (size_t)tb * nv * ATT_T, pfp, npf); break;
    #define ATT512_QKD_ANY(nt) npf = hd; pfp = att512_pf_take(pf, &npf); switch (nv * 8 + (nt)) { \
        ATT512_QKD(1, 1) ATT512_QKD(2, 1) ATT512_QKD(3, 1) ATT512_QKD(4, 1) ATT512_QKD(5, 1) ATT512_QKD(6, 1) \
        ATT512_QKD(7, 1) ATT512_QKD(8, 1) ATT512_QKD(9, 1) ATT512_QKD(10, 1) ATT512_QKD(11, 1) ATT512_QKD(12, 1) \
        ATT512_QKD(1, 4) ATT512_QKD(2, 4) ATT512_QKD(3, 4) ATT512_QKD(4, 4) ATT512_QKD(5, 4) ATT512_QKD(6, 4) \
        ATT512_QKD(7, 2) ATT512_QKD(8, 2) ATT512_QKD(9, 2) ATT512_QKD(10, 2) ATT512_QKD(11, 2) ATT512_QKD(12, 2) \
        default: break; }
    int ntw = nv <= 6 ? 4 : 2, tb = 0, npf;
    const char *pfp;
    for (; tb + ntw <= ntl; tb += ntw) {
        ATT512_QKD_ANY(ntw)
    }
    for (; tb < ntl; tb++) {
        ATT512_QKD_ANY(1)
    }
    #undef ATT512_QKD_ANY
    #undef ATT512_QKD

    // softmax per vector and block; the weights rounded to bf16 over their scores, l_b their sum
    float inv[ATT512_NV];
    size_t st = (size_t)nv * ATT_T;
    for (int r = 0; r < nv; r++)
        for (int j = 0; j < nbk; j++) {
            int n = npr[r] - j * ATT_P;
            if (n <= 0)
                break;
            n = n < ATT_P || !part ? n : ATT_P;
            int nt = (n + ATT_T - 1) / ATT_T;
            float *sr = s + r * ATT_T + (size_t)j * (ATT_P / ATT_T) * st;
            __mmask16 last = (__mmask16)(0xffffu >> (nt * ATT_T - n));
            __m512 mx = _mm512_set1_ps(-FLT_MAX);
            for (int tb = 0; tb < nt - 1; tb++)
                mx = _mm512_max_ps(mx, _mm512_loadu_ps(sr + tb * st));
            mx = _mm512_mask_max_ps(mx, last, mx, _mm512_loadu_ps(sr + (nt - 1) * st));
            __m512 m = _mm512_set1_ps(_mm512_reduce_max_ps(mx)), sum = _mm512_setzero_ps();
            for (int tb = 0; tb < nt; tb++) {
                __m512 e = att512_exp(_mm512_sub_round_ps(_mm512_loadu_ps(sr + tb * st), m, ATT512_RND));
                if (tb == nt - 1)
                    e = _mm512_maskz_mov_ps(last, e);
                __m256i w = att512_cvt_bf16(e);
                _mm256_storeu_si256((__m256i *)(sr + tb * st), w);
                sum = _mm512_add_round_ps(sum, att512_bf16_ps(w), ATT512_RND);
            }
            float tot = att512_sum16(sum);
            if (part) {
                float *ml = part[r] + (size_t)(b0 + j) * pst + hd;
                ml[0] = _mm_cvtss_f32(_mm512_castps512_ps128(m));
                ml[1] = tot;
            } else {
                inv[r] = _mm_cvtss_f32(_mm_div_round_ss(_mm_set_ss(1.0f), _mm_set_ss(tot), ATT512_RND));
            }
        }

    #define ATT512_PVD(V, D) case V: att512_pv_dot(V, D, s, nbk, npr, vt, ts, bs, hd, part ? NULL : inv, part ? part : out, b0, pst); break;
    switch (nv) {
        ATT512_PVD(1, 8) ATT512_PVD(2, 8) ATT512_PVD(3, 4) ATT512_PVD(4, 4) ATT512_PVD(5, 4) ATT512_PVD(6, 4)
        ATT512_PVD(7, 2) ATT512_PVD(8, 2) ATT512_PVD(9, 2) ATT512_PVD(10, 2) ATT512_PVD(11, 2) ATT512_PVD(12, 2)
        default: break;
    }
    #undef ATT512_PVD
}

static void att512_kv_dot1(int kvt, int nv, const float *const *q, float *const *out, float *const *part, const int *np, int b0,
                           int b1, const void *kt, const void *vt, size_t bs, int hd, float scale, float *s, uint16_t *conv,
                           Att512Pf *pf) {
    // one range of att512_kv_dot (at most ATT512_BC blocks if part)
    const uint16_t *kb = (const uint16_t *)kt + (size_t)b0 * bs, *vb = (const uint16_t *)vt + (size_t)b0 * bs;
    size_t bsx = bs;
    if (kvt == 2) {
        int nmax = 0;
        for (int r = 0; r < nv; r++) {
            int n = np[r] - b0 * ATT_P, e = (b1 - b0) * ATT_P;
            n = n < 0 ? 0 : n < e ? n : e;
            nmax = n > nmax ? n : nmax;
        }
        if (nmax == 0)
            return;
        att512_dot_conv(conv, conv + (size_t)2 * ATT_P * hd, kb, vb, (nmax + ATT_T - 1) / ATT_T, bs, hd);
        kb = conv;
        vb = conv + (size_t)2 * ATT_P * hd;
        bsx = (size_t)ATT_P * hd;
    }
    int ng = (nv + ATT512_G - 1) / ATT512_G;
    for (int i = 0, g = 0; i < ng; i++) {
        int gn = (nv - g) / (ng - i);
        att512_group_dot(gn, q + g, out + g, part ? part + g : NULL, np + g, b0, b1, kb, vb, bsx, hd, scale, s, pf);
        g += gn;
    }
}

static void att512_kv_dot(int kvt, int nv, const float *const *q, float *const *out, float *const *part, const int *np, int b0,
                          int b1, const void *kt, const void *vt, size_t bs, int hd, float scale, float *s, uint16_t *conv) {
    // att512_kv for the bf16 dot products (kvt: the storage, KV_BF16 in the dot layouts or KV_F16, converted per range
    // into conv: 4 * ATT_P * hd elements, the thread's); the same groups and sweep. While a range is computed, the
    // next one's K / V (the positions some vector reaches) are prefetched into L2: at twice the fp32 rate the first
    // group of a range waited on memory (synth-27b, 2,560-token chunk at 98k, 8 threads: ~5% of the attention)
    if (!part || b1 - b0 <= ATT512_BC) {
        att512_kv_dot1(kvt, nv, q, out, part, np, b0, b1, kt, vt, bs, hd, scale, s, conv, NULL);
        return;
    }
    int nmax = 0;
    for (int r = 0; r < nv; r++)
        nmax = np[r] > nmax ? np[r] : nmax;
    for (int c0 = b0; c0 < b1; c0 += ATT512_BC) {
        int c1 = c0 + ATT512_BC < b1 ? c0 + ATT512_BC : b1;
        Att512Pf pf = { .nr = 0, .k = 0 };
        for (int b = c1; b < c1 + ATT512_BC && b < b1 && b * ATT_P < nmax; b++) {
            int n = nmax - b * ATT_P < ATT_P ? nmax - b * ATT_P : ATT_P;
            size_t lines = ((size_t)n * hd * 2 + 63) / 64; // (2-byte elements: positions < n of the block)
            pf.p[pf.nr] = (const char *)kt + (size_t)b * bs * 2;
            pf.n[pf.nr++] = lines;
            pf.p[pf.nr] = (const char *)vt + (size_t)b * bs * 2;
            pf.n[pf.nr++] = lines;
        }
        att512_kv_dot1(kvt, nv, q, out, part, np, c0, c1, kt, vt, bs, hd, scale, s, conv, &pf);
    }
}

#endif // AVX-512F attention
#endif // RUNQ_KERNELS_AVX512_H
