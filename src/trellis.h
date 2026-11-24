/* NariQ trellis-coded weights (nari-labs' Qwen3.8-27B-NariQ GGUFs; format: nari-labs/nari-llama.cpp nariq/FORMAT.md,
 * version 1; docs/nariq.md). GGUF types TRL1 .. TRL6 = 44 .. 49 (fork-only ids: valid only with trellis.version = 1).
 *
 * A coded linear with d outputs and k inputs (both multiples of 16) computes
 *     y = scatter(post * T_rows(V (s16 * T_cols(pre * gather_pin(x)))))
 * T_cols / T_rows: normalized block Sylvester Hadamards (block cblk / rblk, a power of two; 0: none; symmetric and
 * involutory), s16 one scale per 16 inputs (shared by every row), pre / post per-input / per-output scales, pin an input
 * gather (x_used[j] = x[pin[j]]), the output scatter scat = the inverse of the file's pout gather (out[scat[i]] = y[i]).
 * V is the integer grid: v = f(state) in [-126, 126] with state the 16-bit window of a tile's circular bit stream
 * ending at bit (p + 1) bits (stream bit q = bit 31 - q % 32 of word q / 32), f(state) = the sum of the four 6-bit
 * fields of state * 0x83DCD12D (bits 2, 10, 18, 26) - 126 (exllamav3's "MUL1" INT8 grid). Tiles are 16 x 16 values
 * (8 bits words each), output-tile-major (tile row J, then input tile i); ring position p = 8 a + j of a tile is
 * output row (a >> 2) + 8 ((j >> 2) & 1), input column 2 (a & 3) + (j & 1) + 8 ((j >> 1) & 1).
 *
 * Gwenedeg's arithmetic (the definition every build computes, bit for bit):
 *   activation (per row): u_j = x[pin_j] * pre_j (j < k), T_cols on each cblk block (trl_fwht), u_j = u_j * s16[j / 16],
 *     then quantize (forward.c: int8 q_j, scale s_g per group of 32, the GGUF activation groups) and the group sums
 *   GEMV: y_n = sum over groups g in order of ((float) I_ng) * s_g, from 0, one rounding per multiply and per add,
 *     I_ng = sum_{j in g} v_nj q_j (exact int32)
 *   output: T_rows on each rblk block of y, then out[scat_n] = y_n * post_n
 * The AVX-512 kernel computes I_ng with vpdpbusd on the u8 fields of state * mult (lane = output row, the activation
 * byte broadcast to the lane's 4 bytes: the lane sums (v + 126) q) minus 126 * sum(q_g); the generic one the same
 * integers from trl_value; then the same float operations.
 */
#ifndef GWENEDEG_TRELLIS_H
#define GWENEDEG_TRELLIS_H

#include <stdint.h>
#include <string.h>
#include <math.h>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

enum { GGML_TRL1 = 44, GGML_TRL6 = 49 };
static inline int trl_type(int t) { return t >= GGML_TRL1 && t <= GGML_TRL6; }
static inline int trl_bits(int t) { return t - GGML_TRL1 + 1; }

#define TRL_MULT 0x83DCD12Du
#define TRL_PAD 128 // readable bytes after a matrix's codes (the AVX-512 kernel loads 32 / 48 words per tile)

typedef struct TrlMat {
    int bits, d, k;          // d outputs, k inputs (multiples of 16)
    int rblk, cblk;          // output / input side Hadamard blocks (0: none)
    uint8_t *codes;          // the tiles (output-tile-major), TRL_PAD readable bytes after (prepare_weights' copy)
    const uint8_t *file;     // the tiles in the file mapping (the embedding is decoded from there)
    const float *s16;        // k / 16 scales (the file's fp16, widened: exact)
    const float *pre;        // k
    const float *post;       // d
    const int32_t *pin;      // k, or NULL
    const int32_t *scat;     // d, or NULL
    // the side tensors in the file mapping (validated shapes and types; copied and checked by model.c's trl_load_sides)
    const void *f_s16, *f_pre, *f_post, *f_pin, *f_pout;
} TrlMat;

void trl_tab_init(void); // the AVX-512 kernel's tables (forward.c; idempotent; at model load; a no-op in other builds)

static inline size_t trl_code_bytes(int bits, int d, int k) { return (size_t)d * k / 8 * bits; }
static inline size_t trl_block_bytes(const TrlMat *t) { return (size_t)t->k * 2 * t->bits; } // one 16-row tile row

// ring position of (row r, column c) within a tile
static inline int trl_ring(int r, int c) { return 8 * (4 * (r % 8) + (c % 8) / 2) + 4 * (r / 8) + 2 * (c / 8) + c % 2; }

static inline int trl_value(const uint32_t *tile, int bits, int p) {
    // the grid value at ring position p of a tile (the format's definition)
    const int total = 256 * bits, nw = 8 * bits;
    const int start = ((p + 1) * bits - 16 + total) % total;
    const int w = start >> 5, o = start & 31;
    const uint64_t pair = ((uint64_t)tile[w] << 32) | tile[(w + 1) % nw];
    const uint32_t state = (uint32_t)(pair >> (48 - o)) & 0xFFFFu;
    const uint32_t x = state * TRL_MULT;
    return (int)(((x >> 2) & 63) + ((x >> 10) & 63) + ((x >> 18) & 63) + ((x >> 26) & 63)) - 126;
}

// the normalized Sylvester Hadamard of each block of `block` values of u[0 .. n): butterflies h = 1, 2, 4 .. (a + b,
// a - b), then one multiply by (float) block^-1/2 (KQ_EXACT: no contraction, no reassociation; vectorized lane-wise)
KQ_EXACT static void trl_fwht(float *u, int n, int block) { RQ_FP_EXACT
    const float c = (float)pow((double)block, -0.5);
    for (int b0 = 0; b0 < n; b0 += block) {
        float *v = u + b0;
        for (int h = 1; h < block; h *= 2)
            for (int i = 0; i < block; i += 2 * h)
                for (int j = i; j < i + h; j++) {
                    float a = v[j], b = v[j + h];
                    v[j] = a + b;
                    v[j + h] = a - b;
                }
        for (int i = 0; i < block; i++)
            v[i] = v[i] * c;
    }
}

// The input side of one activation row on inputs [c0, c1) (whole cblk blocks, multiples of 32): u (fp32 scratch)
KQ_EXACT static void trl_act_side(const TrlMat *t, const float *x, float *u, int c0, int c1) { RQ_FP_EXACT
    for (int j = c0; j < c1; j++)
        u[j] = (t->pin ? x[t->pin[j]] : x[j]) * t->pre[j];
    if (t->cblk)
        trl_fwht(u + c0, c1 - c0, t->cblk);
    for (int j = c0; j < c1; j++)
        u[j] = u[j] * t->s16[j / 16];
}

// The output side of one row on outputs [r0, r1) (whole rblk blocks): T_rows on y in place, then out[scat] = y * post
KQ_EXACT static void trl_out_side(const TrlMat *t, float *y, float *out, int r0, int r1) { RQ_FP_EXACT
    if (t->rblk)
        trl_fwht(y + r0, r1 - r0, t->rblk);
    if (t->scat)
        for (int i = r0; i < r1; i++)
            out[t->scat[i]] = y[i] * t->post[i];
    else
        for (int i = r0; i < r1; i++)
            out[i] = y[i] * t->post[i];
}

// A row of a coded embedding (rows unrotated, no gathers; checked at load): the decoded row times s16, T_cols
// (its own inverse), times pre_j * post_token
KQ_EXACT static void trl_embed_row(const TrlMat *t, const uint8_t *codes, int token, float *x) { RQ_FP_EXACT
    const int kt = t->k / 16, nw = 8 * t->bits, J = token / 16, r = token % 16;
    for (int i = 0; i < kt; i++) {
        const uint32_t *tile = (const uint32_t *)(codes + ((size_t)J * kt + i) * nw * 4);
        for (int c = 0; c < 16; c++)
            x[16 * i + c] = (float)trl_value(tile, t->bits, trl_ring(r, c)) * t->s16[i];
    }
    if (t->cblk)
        trl_fwht(x, t->k, t->cblk);
    float pt = t->post[token];
    for (int j = 0; j < t->k; j++)
        x[j] = x[j] * (t->pre[j] * pt);
}

// ------------------------------------------------------------------------------------------------ GEMV kernels

// Rows of 16-row blocks blk0 .. blk1-1 for nb activation rows: y[b * ldy + n] = the GEMV above. q: int8 rows (stride
// ldq), s: their group scales (stride ldq / 32), sum: their group sums (stride ldq / 32; the AVX-512 kernel's bias)

KQ_EXACT static void trl_rows_generic(const TrlMat *t, float *y, size_t ldy, const int8_t *q, const float *s, size_t ldq, int nb,
                                      int blk0, int blk1) { RQ_FP_EXACT
    const int bits = t->bits, kt = t->k / 16, nw = 8 * bits, ng = t->k / 32;
    int8_t v[2][16][16]; // the group's two tiles: [tile][row][column]
    for (int J = blk0; J < blk1; J++) {
        for (int b = 0; b < nb; b++)
            for (int r = 0; r < 16; r++)
                y[(size_t)b * ldy + 16 * J + r] = 0;
        for (int g = 0; g < ng; g++) {
            for (int h = 0; h < 2; h++) {
                const uint32_t *tile = (const uint32_t *)(t->codes + ((size_t)J * kt + 2 * g + h) * nw * 4);
                for (int r = 0; r < 16; r++)
                    for (int c = 0; c < 16; c++)
                        v[h][r][c] = (int8_t)trl_value(tile, bits, trl_ring(r, c));
            }
            for (int b = 0; b < nb; b++) {
                const int8_t *qg = q + (size_t)b * ldq + 32 * g;
                float sg = s[(size_t)b * (ldq / 32) + g];
                float *yb = y + (size_t)b * ldy + 16 * J;
                for (int r = 0; r < 16; r++) {
                    int32_t I = 0;
                    for (int c = 0; c < 32; c++)
                        I += v[c / 16][r][c % 16] * qg[c];
                    yb[r] = yb[r] + (float)I * sg;
                }
            }
        }
    }
}

#if defined(GWENEDEG_VNNI)
#define GWENEDEG_TRLX 1
// Per bit width and tile column c: for each lane (output row) the words of its window's high and low halves (indices into
// the tile's words: vpermt2d over words 0..31, words 32..47 by a masked vpermd: hmask / lmask), the funnel shift count
typedef struct {
    int32_t h[16][16], l[16][16], cnt[16][16];
    uint16_t hm[16], lm[16];
    // quads: columns 2g, 2g + 1, 2g + 8, 2g + 9 of a row are its ring positions p .. p + 3, whose windows lie in the
    // 32 bits from s = (p + 1) bits - 16 (bits <= 5; 6: the first three): per quad and lane the words of that window
    // (qh: from bit s, ql the next word) and s % 32 (the vpshldvd count)
    int32_t qh[4][16], ql[4][16], qo[4][16];
    uint16_t qhm[4], qlm[4];
} TrlTab;
extern TrlTab trl_tab[6];

#if defined(__AVX512VBMI2__)
#define TRL_FUNNEL(L, H, cnt) _mm512_shrdv_epi32((L), (H), (cnt))
#else
#define TRL_FUNNEL(L, H, cnt) _mm512_or_si512(_mm512_srlv_epi32((L), (cnt)), \
                                              _mm512_sllv_epi32((H), _mm512_sub_epi32(_mm512_set1_epi32(32), (cnt))))
#endif

// the u8 fields of column c of a tile (lane = output row r): ((state * mult) >> 2) & 0x3F3F3F3F, state the window
#define TRL_FIELDS(tb, c, T0, T1, T2, big, out) do { \
        __m512i H_ = _mm512_permutex2var_epi32(T0, _mm512_loadu_si512((tb)->h[c]), T1); \
        __m512i L_ = _mm512_permutex2var_epi32(T0, _mm512_loadu_si512((tb)->l[c]), T1); \
        if (big) { \
            H_ = _mm512_mask_permutexvar_epi32(H_, (tb)->hm[c], _mm512_loadu_si512((tb)->h[c]), T2); \
            L_ = _mm512_mask_permutexvar_epi32(L_, (tb)->lm[c], _mm512_loadu_si512((tb)->l[c]), T2); \
        } \
        __m512i st_ = _mm512_and_si512(TRL_FUNNEL(L_, H_, _mm512_loadu_si512((tb)->cnt[c])), _mm512_set1_epi32(0xFFFF)); \
        out = _mm512_and_si512(_mm512_srli_epi32(_mm512_mullo_epi32(st_, _mm512_set1_epi32((int)TRL_MULT)), 2), \
                               _mm512_set1_epi32(0x3F3F3F3F)); \
    } while (0)

// the u8 fields of quad g of a tile (columns 2g, 2g + 1, 2g + 8, 2g + 9: f0 .. f3): one 32-bit window W (two permutes,
// vpshldvd), each column's state W >> (16 - j bits) & 0xFFFF (bits 6: column 2g + 9 by TRL_FIELDS)
#define TRL_FLD(st) _mm512_and_si512(_mm512_srli_epi32(_mm512_mullo_epi32((st), _mm512_set1_epi32((int)TRL_MULT)), 2), \
                                     _mm512_set1_epi32(0x3F3F3F3F))
#define TRL_QUAD(tb, g, T0, T1, T2, big, bits, f0, f1, f2, f3) do { \
        __m512i H_ = _mm512_permutex2var_epi32(T0, _mm512_loadu_si512((tb)->qh[g]), T1); \
        __m512i L_ = _mm512_permutex2var_epi32(T0, _mm512_loadu_si512((tb)->ql[g]), T1); \
        if (big) { \
            H_ = _mm512_mask_permutexvar_epi32(H_, (tb)->qhm[g], _mm512_loadu_si512((tb)->qh[g]), T2); \
            L_ = _mm512_mask_permutexvar_epi32(L_, (tb)->qlm[g], _mm512_loadu_si512((tb)->ql[g]), T2); \
        } \
        const __m512i W_ = _mm512_shldv_epi32(H_, L_, _mm512_loadu_si512((tb)->qo[g])), m16_ = _mm512_set1_epi32(0xFFFF); \
        f0 = TRL_FLD(_mm512_srli_epi32(W_, 16)); \
        f1 = TRL_FLD(_mm512_and_si512(_mm512_srli_epi32(W_, 16 - (bits)), m16_)); \
        f2 = TRL_FLD(_mm512_and_si512(_mm512_srli_epi32(W_, 16 - 2 * (bits)), m16_)); \
        if (3 * (bits) <= 16) \
            f3 = TRL_FLD(_mm512_and_si512(_mm512_srli_epi32(W_, 16 - 3 * (bits)), m16_)); \
        else \
            TRL_FIELDS(tb, 2 * (g) + 9, T0, T1, T2, big, f3); \
    } while (0)

#if defined(__GNUC__) || defined(__clang__)
#define TRL_INLINE __attribute__((always_inline)) inline
#else
#define TRL_INLINE inline
#endif

static TRL_INLINE void trl512_gemv1_body(const TrlMat *t, float *y, const int32_t *q4, const float *s, const int32_t *sum,
                                         int blk0, int blk1, const int bits) { RQ_FP_EXACT
    // one activation row (q4: each int8 value replicated in the 4 bytes of a dword: vpdpbusd's embedded broadcast; a
    // byte broadcast would compete with the permutes for the shuffle unit): four interleaved integer accumulators.
    // bits is a constant at each call (trl512_gemv1): the column loop unrolls with the tables' offsets known (~25%
    // faster than one function for every width)
    const int kt = t->k / 16, nw = 8 * bits, big = bits > 4;
    const TrlTab *tb = &trl_tab[bits - 1];
    for (int J = blk0; J < blk1; J++) {
        const uint32_t *row = (const uint32_t *)(t->codes + (size_t)J * kt * nw * 4);
        __m512 acc = _mm512_setzero_ps();
        for (int i = 0; i < kt; i += 2) {
            __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
            for (int h = 0; h < 2; h++) {
                const uint32_t *tile = row + (size_t)(i + h) * nw;
                const __m512i T0 = _mm512_loadu_si512(tile), T1 = _mm512_loadu_si512(tile + 16);
                const __m512i T2 = big ? _mm512_loadu_si512(tile + 32) : T0;
                const int32_t *xa = q4 + 16 * (i + h);
                for (int g = 0; g < 4; g++) {
                    __m512i f0, f1, f2, f3;
                    TRL_QUAD(tb, g, T0, T1, T2, big, bits, f0, f1, f2, f3);
                    a0 = _mm512_dpbusd_epi32(a0, f0, _mm512_set1_epi32(xa[2 * g]));
                    a1 = _mm512_dpbusd_epi32(a1, f1, _mm512_set1_epi32(xa[2 * g + 1]));
                    a2 = _mm512_dpbusd_epi32(a2, f2, _mm512_set1_epi32(xa[2 * g + 8]));
                    a3 = _mm512_dpbusd_epi32(a3, f3, _mm512_set1_epi32(xa[2 * g + 9]));
                }
            }
            const __m512i I = _mm512_sub_epi32(_mm512_add_epi32(_mm512_add_epi32(a0, a1), _mm512_add_epi32(a2, a3)),
                                               _mm512_set1_epi32(126 * sum[i / 2]));
            acc = _mm512_add_ps(acc, _mm512_mul_ps(_mm512_cvtepi32_ps(I), _mm512_set1_ps(s[i / 2])));
        }
        _mm512_storeu_ps(y + 16 * J, acc);
    }
}

#define TRL_GEMV1(b) \
    KQ_EXACT static void trl512_gemv1_##b(const TrlMat *t, float *y, const int32_t *q4, const float *s, const int32_t *sum, \
                                          int blk0, int blk1) { RQ_FP_EXACT trl512_gemv1_body(t, y, q4, s, sum, blk0, blk1, b); }
TRL_GEMV1(1) TRL_GEMV1(2) TRL_GEMV1(3) TRL_GEMV1(4) TRL_GEMV1(5) TRL_GEMV1(6)

static void trl512_gemv1(const TrlMat *t, float *y, const int32_t *q4, const float *s, const int32_t *sum, int blk0, int blk1) {
    switch (t->bits) {
        case 1: trl512_gemv1_1(t, y, q4, s, sum, blk0, blk1); break;
        case 2: trl512_gemv1_2(t, y, q4, s, sum, blk0, blk1); break;
        case 3: trl512_gemv1_3(t, y, q4, s, sum, blk0, blk1); break;
        case 4: trl512_gemv1_4(t, y, q4, s, sum, blk0, blk1); break;
        case 5: trl512_gemv1_5(t, y, q4, s, sum, blk0, blk1); break;
        default: trl512_gemv1_6(t, y, q4, s, sum, blk0, blk1); break;
    }
}

// Several activation rows (spec verify segments, prefill): each 16-row block's codes decoded once into the thread's
// scratch W as u8 v + 126 (lane = output row, byte j = column 4 q + j of the dword's 4 columns: the quads' fields summed
// by vpdpbusd with ones, then packed), then an int8 GEMM over the rows (64 products per vpdpbusd, each activation dword
// broadcast from the int8 row). The integers and the float operations are the GEMV's (1.5x the field kernel at 8 rows,
// 4x at 256)
static TRL_INLINE void trl512_unpack_body(const TrlMat *t, int J, __m512i *W, const int bits) {
    const int kt = t->k / 16, nw = 8 * bits, big = bits > 4;
    const TrlTab *tb = &trl_tab[bits - 1];
    const __m512i ones = _mm512_set1_epi8(1), z = _mm512_setzero_si512();
    const uint32_t *row = (const uint32_t *)(t->codes + (size_t)J * kt * nw * 4);
    for (int i = 0; i < kt; i++) {
        const uint32_t *tile = row + (size_t)i * nw;
        const __m512i T0 = _mm512_loadu_si512(tile), T1 = _mm512_loadu_si512(tile + 16);
        const __m512i T2 = big ? _mm512_loadu_si512(tile + 32) : T0;
        __m512i sv[4][4];
        for (int g = 0; g < 4; g++) {
            __m512i f0, f1, f2, f3;
            TRL_QUAD(tb, g, T0, T1, T2, big, bits, f0, f1, f2, f3);
            sv[g][0] = _mm512_dpbusd_epi32(z, f0, ones);
            sv[g][1] = _mm512_dpbusd_epi32(z, f1, ones);
            sv[g][2] = _mm512_dpbusd_epi32(z, f2, ones);
            sv[g][3] = _mm512_dpbusd_epi32(z, f3, ones);
        }
        // quad g: columns 2g, 2g + 1 (j 0, 1) and 2g + 8, 2g + 9 (j 2, 3) -> W[4 i + 2 h + a]: columns 8 h + 4 a .. + 3
        for (int h = 0; h < 2; h++)
            for (int a = 0; a < 2; a++) {
                __m512i w = _mm512_or_si512(sv[2 * a][2 * h], _mm512_slli_epi32(sv[2 * a][2 * h + 1], 8));
                w = _mm512_or_si512(w, _mm512_slli_epi32(sv[2 * a + 1][2 * h], 16));
                w = _mm512_or_si512(w, _mm512_slli_epi32(sv[2 * a + 1][2 * h + 1], 24));
                _mm512_store_si512(W + 4 * i + 2 * h + a, w);
            }
    }
}

#define TRL_GR 8 // activation rows per register block of the GEMM

static TRL_INLINE void trl512_gemm_body(const TrlMat *t, float *y, size_t ldy, const int8_t *q, const float *s, const int32_t *sum,
                                        size_t ldq, int nb, int blk0, int blk1, __m512i *W, const int bits) { RQ_FP_EXACT
    const int ng = t->k / 32;
    const size_t lds = ldq / 32;
    for (int J = blk0; J < blk1; J++) {
        trl512_unpack_body(t, J, W, bits);
        for (int b0 = 0; b0 < nb; b0 += TRL_GR) {
            const int n = nb - b0 < TRL_GR ? nb - b0 : TRL_GR;
            __m512 acc[TRL_GR];
            for (int r = 0; r < TRL_GR; r++)
                acc[r] = _mm512_setzero_ps();
            for (int g = 0; g < ng; g++) {
                __m512i ia[TRL_GR];
                for (int r = 0; r < TRL_GR; r++)
                    ia[r] = _mm512_setzero_si512();
                for (int qd = 0; qd < 8; qd++) {
                    const __m512i w = _mm512_load_si512(W + 8 * g + qd);
                    for (int r = 0; r < n; r++) {
                        int32_t xv;
                        memcpy(&xv, q + (size_t)(b0 + r) * ldq + 32 * g + 4 * qd, 4);
                        ia[r] = _mm512_dpbusd_epi32(ia[r], w, _mm512_set1_epi32(xv));
                    }
                }
                for (int r = 0; r < n; r++) {
                    size_t o = (size_t)(b0 + r) * lds + g;
                    const __m512i I = _mm512_sub_epi32(ia[r], _mm512_set1_epi32(126 * sum[o]));
                    acc[r] = _mm512_add_ps(acc[r], _mm512_mul_ps(_mm512_cvtepi32_ps(I), _mm512_set1_ps(s[o])));
                }
            }
            for (int r = 0; r < n; r++)
                _mm512_storeu_ps(y + (size_t)(b0 + r) * ldy + 16 * J, acc[r]);
        }
    }
}

#define TRL_GEMM(b) \
    KQ_EXACT static void trl512_gemm_##b(const TrlMat *t, float *y, size_t ldy, const int8_t *q, const float *s, const int32_t *sum, \
                                         size_t ldq, int nb, int blk0, int blk1, __m512i *W) { RQ_FP_EXACT \
        trl512_gemm_body(t, y, ldy, q, s, sum, ldq, nb, blk0, blk1, W, b); }
TRL_GEMM(1) TRL_GEMM(2) TRL_GEMM(3) TRL_GEMM(4) TRL_GEMM(5) TRL_GEMM(6)

static void trl512_gemm(const TrlMat *t, float *y, size_t ldy, const int8_t *q, const float *s, const int32_t *sum, size_t ldq,
                        int nb, int blk0, int blk1, void *W) {
    switch (t->bits) {
        case 1: trl512_gemm_1(t, y, ldy, q, s, sum, ldq, nb, blk0, blk1, W); break;
        case 2: trl512_gemm_2(t, y, ldy, q, s, sum, ldq, nb, blk0, blk1, W); break;
        case 3: trl512_gemm_3(t, y, ldy, q, s, sum, ldq, nb, blk0, blk1, W); break;
        case 4: trl512_gemm_4(t, y, ldy, q, s, sum, ldq, nb, blk0, blk1, W); break;
        case 5: trl512_gemm_5(t, y, ldy, q, s, sum, ldq, nb, blk0, blk1, W); break;
        default: trl512_gemm_6(t, y, ldy, q, s, sum, ldq, nb, blk0, blk1, W); break;
    }
}
#endif // GWENEDEG_VNNI

static inline void trl_rows(const TrlMat *t, float *y, size_t ldy, const int8_t *q, const int32_t *q4, const float *s,
                            const int32_t *sum, size_t ldq, int nb, int blk0, int blk1, void *W) {
    // the GEMV / GEMM of 16-row blocks blk0 .. blk1-1 (the build's kernel; every build: the same bits). q: the int8
    // activations, q4 the same replicated (AVX-512, one row), s / sum their group scales and sums; W: 16 k bytes of
    // 64-byte aligned scratch (AVX-512, several rows)
#ifdef GWENEDEG_TRLX
    if (nb == 1)
        trl512_gemv1(t, y, q4, s, sum, blk0, blk1);
    else
        trl512_gemm(t, y, ldy, q, s, sum, ldq, nb, blk0, blk1, W);
#else
    (void)sum; (void)q4; (void)W;
    trl_rows_generic(t, y, ldy, q, s, ldq, nb, blk0, blk1);
#endif
}

#endif // GWENEDEG_TRELLIS_H
