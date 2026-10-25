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

// Attention constants, shared by the AVX-512 attention below and the generic C one in runq.c
// (attention_vec): the same constants and operations give the same results
#define ATT_T 16                       // positions per K/V cache tile
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

static int q8p_nb(int nb, int ntile) {
    // activation rows per GEMM work item: Q8P_NB, halved (down to 4 * Q8P_C) while there are
    // fewer than ~6 items per thread (e.g. wo / w2 / k / v: 32 tiles x 2 chunks at 128 rows), so
    // 12 or 16 threads get balanced work. (Results do not depend on the partition)
    int nt = 1, nbi = Q8P_NB;
#ifdef _OPENMP
    nt = omp_get_max_threads();
#endif
    while (nbi > 4 * Q8P_C && (long)ntile * ((nb + nbi - 1) / nbi) < 6L * nt)
        nbi /= 2;
    return nbi;
}

typedef struct { float *out; int ldo; const uint8_t *wp; int d; } Q8PJob;

static void q8p_gemm_n(int nj, const Q8PJob *job, const int8_t *xq, const float *xs, int ldx, int nb, int k) {
    // for each job j: job[j].out[b * ldo + i] = W_j[i, :] . x[b, :] for b < nb, i < d_j; all W_j
    // in Q8P layout (k % 64 == 0, d_j % 16 == 0) on the same input rows x (at xq + b * ldx, scales
    // xs + b * ldx / 64), in one parallel loop (e.g. q, k, v: one barrier instead of three).
    // Work items: (32-row weight tile of some W_j, nbi activation rows), static schedule
    // (contiguous items: a thread does all the chunks of a tile in a row; dynamic scheduling
    // measured the same at 8-14 threads and does not prevent the 16-thread collapse); inside,
    // the weight tile (32 x k bytes + scales, 34-102 KB) is reused from L1/L2 for every Q8P_C rows
    int ng = k / Q8P_GS;
    size_t bstride = (size_t)ng * Q8P_GROUP, ldxs = (size_t)ldx / Q8P_GS;
    int32_t *corr = q8p_corr_buf((size_t)nb * ng);
    int t0[9], ntile = 0;
    if (nj > 8) { fprintf(stderr, "q8p_gemm_n: too many jobs\n"); exit(EXIT_FAILURE); }
    for (int j = 0; j < nj; j++) {
        t0[j] = ntile;
        ntile += (job[j].d / Q8P_ROWS + Q8P_RB - 1) / Q8P_RB;
    }
    t0[nj] = ntile;
    int nbi = q8p_nb(nb, ntile), nchunk = (nb + nbi - 1) / nbi;
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int b = 0; b < nb; b++)
            for (int g = 0; g < ng; g++)
                corr[(size_t)b * ng + g] = q8p_corr(xq + (size_t)b * ldx + g * Q8P_GS);
        #pragma omp for schedule(static)
        for (int it = 0; it < ntile * nchunk; it++) {
            int t = it / nchunk, ch = it % nchunk, j = 0;
            while (t >= t0[j + 1]) j++;
            int nblk = job[j].d / Q8P_ROWS, rb0 = (t - t0[j]) * Q8P_RB;
            int rb = nblk - rb0 < Q8P_RB ? nblk - rb0 : Q8P_RB;
            int b1 = (ch + 1) * nbi < nb ? (ch + 1) * nbi : nb;
            const uint8_t *wt = job[j].wp + (size_t)rb0 * bstride;
            for (int b = ch * nbi; b < b1; b += Q8P_C) {
                int c = b1 - b < Q8P_C ? b1 - b : Q8P_C;
                q8p_tile_any(rb, c, wt, bstride, ng, xq + (size_t)b * ldx, ldx, xs + (size_t)b * ldxs, ldxs,
                             corr + (size_t)b * ng, job[j].out + (size_t)b * job[j].ldo + (size_t)rb0 * Q8P_ROWS, job[j].ldo);
            }
        }
    }
}

static inline void q8p_gemv_corr(int32_t *corr, const int8_t *xq, int k) {
    // the GEMV's activation corrections, one per 64-group (k / 64 values)
    for (int g = 0; g < k / Q8P_GS; g++)
        corr[g] = q8p_corr(xq + g * Q8P_GS);
}

static void q8p_gemv_rows(float *out, const int8_t *xq, const float *xs, const int32_t *corr, const uint8_t *wp,
                          int k, int rb0, int rb1) {
    // out[i] = W[i, :] . x for the rows i of the 16-row blocks rb0..rb1-1 (decode; no OpenMP here:
    // called on disjoint ranges from inside one parallel region). Memory-bound: streams the
    // blocks once; the activation row (k bytes) stays in L1
    int ng = k / Q8P_GS;
    size_t bstride = (size_t)ng * Q8P_GROUP;
    for (int rb = rb0; rb < rb1; rb++) {
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

/* ---------------------------------------------------------------------------------------------
 * fp32 attention (AVX-512F), for decode (forward) and prefill (forward_batch) alike, through
 * attention_kv() in runq.c; RUNQ_GENERIC or a CPU without AVX-512F uses the generic C version
 * there (attention_vec), which computes the same operations in the same order, so all give
 * bit-identical results.
 *
 * KV cache layout (per layer): tiles of ATT_T = 16 positions; tile tb of kv head h at
 * ((tb * n_kv_heads + h) * 16 * head_dim) floats; inside, K is transposed (dim i, position j at
 * i * 16 + j), so one zmm holds dim i of 16 positions, and V is row-major (j * head_dim + i).
 *
 * Arithmetic, per query vector q (n positions), the definition of the attention:
 *   s[t] = (sum over chunks c of ATT_DC = 32 dims, in order, of (fma chain over the dims i of c
 *          of q[i] * k[t][i], from 0)) * scale. (One 128-long chain: ~10% larger deviation of
 *          the logits from a double-precision attention; chunks of 32: as close as the old
 *          vectorized dot product with 16 partial sums)
 *   m = max of s[0..n); e[t] = exp_att(s[t] - m) (exp_att: Cephes-style expf, see att512_exp)
 *   sum: 16 partial sums (lane j: e[t] for t = j mod 16, t in order), then a fixed tree
 *        (lanes j + 8, then + 4, + 2, + 1); inv = 1 / sum (one division)
 *   out[i] = (fma chain over t = 0..n-1 of e[t] * v[t][i], from 0) * inv
 * Every operation is rounded (explicit-rounding intrinsics, opaque to -Ofast's contraction and
 * reassociation; the C version is compiled with fp-contract=off, no-associative-math). The
 * results of a query vector do not depend on the other vectors processed with it, so decode
 * (2 vectors: the query heads of one kv head) and prefill (2 x ATT_ROWS vectors) agree.
 *
 * Kernel: groups of up to ATT512_NV = 12 query vectors of one kv head. Scores: register tile of
 * NV vectors x NT tiles (NV * NT <= 24 accumulators), the K tile loaded once per dim for all NV
 * vectors (q broadcast; no horizontal sums); scores stored per tile as [NV][16]. Softmax per
 * vector. V: register tile NV vectors x DW zmm of dims, the score of each vector broadcast per
 * position, V streamed once per group; a vector's positions beyond its own n (causal rows of a
 * block) are skipped, so the fma chain is exactly that of the decode path.
 */
#if defined(__AVX512F__) && !defined(RUNQ_GENERIC)
#define RUNQ_ATT512 1
#include <immintrin.h>
#include <float.h>

#define ATT512_NV 12    // max query vectors per group
#define ATT512_HDMAX 256
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

static inline __attribute__((always_inline))
void att512_qk(const int NV, const int NT, const float *qt, const float *kt, size_t ts, int hd, __m512 scale, float *s) {
    // scores of NV query vectors (transposed: qt[i * NV + r]) for NT tiles (tile u at kt + u * ts)
    // -> s[(u * NV + r) * 16 + j]: per ATT_DC dims an fma chain, the chunks added in order (the
    // running sum kept in s), times scale
    for (int i0 = 0; i0 < hd; i0 += ATT_DC) {
        __m512 acc[ATT512_NV][4];
        for (int r = 0; r < NV; r++)
            for (int u = 0; u < NT; u++)
                acc[r][u] = _mm512_setzero_ps();
        for (int i = i0; i < i0 + ATT_DC; i++) {
            __m512 k[4];
            for (int u = 0; u < NT; u++)
                k[u] = _mm512_loadu_ps(kt + u * ts + (size_t)i * ATT_T);
            for (int r = 0; r < NV; r++) {
                __m512 qb = _mm512_set1_ps(qt[i * NV + r]);
                for (int u = 0; u < NT; u++)
                    acc[r][u] = _mm512_fmadd_ps(k[u], qb, acc[r][u]);
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
        const float *v_ = vt + (size_t)(t >> 4) * ts + (size_t)(t & 15) * hd + d0; \
        __m512 vv[8]; \
        for (int u = 0; u < DW; u++) \
            vv[u] = _mm512_loadu_ps(v_ + u * 16); \
        for (int r = 0; r < NV; r++) \
            if (COND) { \
                __m512 pb = _mm512_set1_ps(p_[r * ATT_T]); \
                for (int u = 0; u < DW; u++) \
                    acc[r][u] = _mm512_fmadd_ps(vv[u], pb, acc[r][u]); \
            } \
    } while (0)

static inline __attribute__((always_inline))
void att512_pv(const int NV, const int DW, const float *s, int nmin, int nmax, const int *np, const float *vt,
               size_t ts, int hd, const float *inv, float *const *out) {
    // out[r][i] = (sum over t < np[r] of e[r][t] * v[t][i]) * inv[r], e in s as [tile][NV][16];
    // DW zmm (16 * DW dims) per pass
    for (int d0 = 0; d0 < hd; d0 += DW * 16) {
        __m512 acc[ATT512_NV][8];
        for (int r = 0; r < NV; r++)
            for (int u = 0; u < DW; u++)
                acc[r][u] = _mm512_setzero_ps();
        int t = 0;
        for (; t < nmin; t++)
            ATT512_PV_STEP(1);
        for (; t < nmax; t++) // causal tail: only the vectors that see position t
            ATT512_PV_STEP(t < np[r]);
        for (int r = 0; r < NV; r++) {
            __m512 iv = _mm512_set1_ps(inv[r]);
            for (int u = 0; u < DW; u++)
                _mm512_storeu_ps(out[r] + d0 + u * 16, _mm512_mul_round_ps(acc[r][u], iv, ATT512_RND));
        }
    }
}
#undef ATT512_PV_STEP

static void att512_group(int nv, const float *const *q, float *const *out, const int *np, const float *kt,
                         const float *vt, size_t ts, int hd, float scale, float *s) {
    // nv <= ATT512_NV query vectors of one kv head; s: scratch of nv * (max np rounded to 16) floats
    int nmin = np[0], nmax = np[0];
    for (int r = 1; r < nv; r++) {
        nmin = np[r] < nmin ? np[r] : nmin;
        nmax = np[r] > nmax ? np[r] : nmax;
    }
    int ntl = (nmax + ATT_T - 1) / ATT_T;
    float qt[ATT512_HDMAX * ATT512_NV] __attribute__((aligned(64)));
    for (int i = 0; i < hd; i++)
        for (int r = 0; r < nv; r++)
            qt[i * nv + r] = q[r][i];
    __m512 sc = _mm512_set1_ps(scale);

    // scores
    #define ATT512_QK(V, T) case (V) * 8 + (T): att512_qk(V, T, qt, kt + (size_t)tb * ts, ts, hd, sc, s + (size_t)tb * nv * ATT_T); break;
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

    // softmax per vector (scores of vector r, tile tb at s + (tb * nv + r) * 16)
    float inv[ATT512_NV];
    for (int r = 0; r < nv; r++) {
        int n = np[r], nt = (n + ATT_T - 1) / ATT_T;
        float *sr = s + r * ATT_T;
        size_t st = (size_t)nv * ATT_T;
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
        inv[r] = _mm_cvtss_f32(_mm_div_round_ss(_mm_set_ss(1.0f), _mm_set_ss(att512_sum16(sum)), ATT512_RND));
    }

    // weighted sum of V
    #define ATT512_PV(V, D) case V: att512_pv(V, D, s, nmin, nmax, np, vt, ts, hd, inv, out); break;
    switch (nv) {
        ATT512_PV(1, 8) ATT512_PV(2, 8) ATT512_PV(3, 4) ATT512_PV(4, 4) ATT512_PV(5, 4) ATT512_PV(6, 4)
        ATT512_PV(7, 2) ATT512_PV(8, 2) ATT512_PV(9, 2) ATT512_PV(10, 2) ATT512_PV(11, 2) ATT512_PV(12, 2)
        default: break;
    }
    #undef ATT512_PV
}

static inline int att512_ok(int hd) { return hd % 128 == 0 && hd <= ATT512_HDMAX; }

static void att512_kv(int nv, const float *const *q, float *const *out, const int *np, const float *kt,
                      const float *vt, size_t ts, int hd, float scale, float *s) {
    // nv query vectors of one kv head (any nv), in groups of ATT512_NV; s: ATT512_NV * kv_len floats
    for (int g = 0; g < nv; g += ATT512_NV) {
        int gn = nv - g < ATT512_NV ? nv - g : ATT512_NV;
        att512_group(gn, q + g, out + g, np + g, kt, vt, ts, hd, scale, s);
    }
}

#endif // AVX-512F attention
#endif // RUNQ_KERNELS_AVX512_H
