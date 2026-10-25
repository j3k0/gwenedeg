/* K-quants (GGUF Q4_K and Q6_K, as in llama.cpp's Q4_K_M files) for runq.c: the block formats,
 * their dequantization, the repacked layouts (Q4P, Q6P) the matmul kernels read, and the generic C
 * kernels. The arithmetic of a matmul output is defined here (below); the generic kernels follow
 * it, and AVX-512 kernels must follow it too (Step 6b-2), so that every build gives bit-identical
 * logits.
 *
 * Block formats (ggml-common.h, ggml-quants.c; super-blocks of QK_K = 256 values of one row):
 *   Q4_K, 144 bytes: fp16 d, fp16 dmin, scales[12], qs[128]. 8 sub-blocks j of 32 values, a 6-bit
 *     scale sc_j and min m_j each, packed in scales[] (get_scale_min_k4: j < 4: sc = s[j] & 63,
 *     m = s[j + 4] & 63; else sc = (s[j + 4] & 15) | (s[j - 4] >> 6) << 4, m = (s[j + 4] >> 4) |
 *     (s[j] >> 6) << 4). Value i = 64 c + 32 h + l (c < 4, h < 2, l < 32): sub-block j = 2 c + h,
 *     q = (qs[32 c + l] >> 4 h) & 15 (per 64 values: the low nibbles hold the first 32, the high
 *     nibbles the next 32). w = (d * sc_j) * q - dmin * m_j
 *   Q6_K, 210 bytes: ql[128], qh[64], int8 scales[16], fp16 d. 16 sub-blocks j of 16 values.
 *     Value i = 128 n + 32 p + l (n < 2, p < 4, l < 32): j = i / 16,
 *     q = ((ql[64 n + 32 (p & 1) + l] >> 4 (p >> 1)) & 15) | ((qh[32 n + l] >> 2 p) & 3) << 4,
 *     0..63. w = (d * scales[j]) * (q - 32)
 * Dequantization (kq_dequant_row: embedding rows, the -m dequant test hook) uses exactly these
 * fp32 expressions, i.e. ggml's dequantize_row_q4_K / _q6_K (and gguf-py's numpy version): d * sc,
 * dmin * m and (d * sc) * q are exact in fp32 (11 + 6 + 4 bits; 11 + 8 + 6), so each value is one
 * rounding of an exact expression, whatever FP contraction or reassociation the compiler applies.
 *
 * Repacked layouts, for (d, k) matrices with d % 16 == 0, k % 256 == 0: block rb = rows 16 rb ..
 * 16 rb + 15, super-block s = columns 256 s .. 256 s + 255, at kp + (rb * k / 256 + s) * SB (one
 * block contiguous over all k, like Q8P). Within, "lane r, byte b" = byte 4 r + b of a 64-byte
 * chunk: a zmm's dword lane r holds 4 consecutive values of row 16 rb + r, the u8 operand of
 * vpdpbusd once unpacked. The 4- and 6-bit data stays packed (decode is bandwidth-bound).
 *   Q4P, SB = 2368 bytes (37 x 64): 148 bytes per row and super-block (file: 144, +2.8%)
 *     +0    uint16 d[16]      fp16 d of row r (as in the file)
 *     +32   uint16 dmin[16]   fp16 dmin of row r
 *     +64   uint8 sc[8][16]   sc_j of row r at 16 j + r (6-bit, unpacked from scales[])
 *     +192  uint8 m[8][16]    m_j of row r
 *     +320  qs: per sub-block j 4 chunks c of 64 bytes (at 320 + 256 j + 64 c): lane r, byte b:
 *           low nibble = q of column 32 j + 8 c + b, high nibble = column 32 j + 8 c + 4 + b
 *           (one 64-byte load = 2 vpdpbusd steps of 4 columns: and 15, and shift right 4)
 *   Q6P, SB = 3392 bytes (53 x 64): 212 bytes per row and super-block (file: 210, +1.0%)
 *     +0    float d[16]       d of row r as fp32 (exact; fp16 + 32 bytes of padding is no smaller)
 *     +64   int8 sc[16][16]   scales[j] of row r at 16 j + r
 *     +320  ql: per sub-block j 2 chunks c of 64 bytes (at 320 + 128 j + 64 c): lane r, byte b:
 *           low nibble = low 4 bits of q of column 16 j + 8 c + b, high nibble = column + 4
 *     +2368 qh: per sub-block j 64 bytes (at 2368 + 64 j): lane r, byte b, bits 2 t, 2 t + 1 =
 *           high 2 bits of q of column 16 j + 4 t + b (t < 4: the 4 steps of the sub-block)
 *           (q of step t = low nibble | ((qh << 4 >> 2 t) & 0x30), shifts on 16-bit lanes)
 *
 * Activations: int8 xq, quantized in groups of 32 (GS = 32 for GGUF, as for Q8_0) with fp32 scales
 * xs, the same quantized row for every matrix it feeds (q / k / v may mix Q4_K and Q6_K). Q6_K's
 * sub-blocks of 16 use the same groups of 32: two weight scales per activation group, combined in
 * int32 (exact; groups of 16 would need a second quantization of the same row for a fused q / k / v
 * with mixed types, and twice the float folds). Per activation row and group g, kq_act_prep:
 *   xf[g] = xs[g] * (float) (sum of the 32 xq of group g)       (one rounding)
 *
 * Arithmetic of output row r (W row r . x), the definition. Every operation is an fp32 operation
 * rounded to nearest, no FMA contraction, no reassociation; groups g = 0 .. k/32 - 1 in order,
 * val = 0 at the start; super-block s = g / 8 of the row, d, dmin, sc, m of that super-block:
 *   Q4_K, sub-block j = g % 8 (= the activation group):
 *     isum = sum over the 32 values of q * xq                   (int32, exact: |isum| <= 60,960)
 *     ds = d * (float) sc_j, dm = dmin * (float) m_j            (exact)
 *     a = (ds * xs[g]) * (float) isum                           (float of isum exact)
 *     b = dm * xf[g]
 *     val = val + (a - b)
 *   Q6_K, sub-blocks j0 = 2 (g % 8), j0 + 1 (16 values each):
 *     t_j = sum over the 16 values of q * xq - 32 * (sum of their 16 xq)    (int32, exact, = the
 *           sum of (q - 32) * xq: |t_j| <= 32 * 127 * 16 = 65,024)
 *     I = sc_j0 * t_j0 + sc_j0+1 * t_j0+1                       (int32, exact, |I| < 2^24)
 *     val = val + (d * xs[g]) * (float) I                       (d: fp32 of the fp16, exact)
 * The integer parts may be computed in any order (vpdpbusd: q as u8 x xq as s8; Q4_K's mins and
 * Q6_K's -32 go through the activation sums); the float part per row only depends on per-row values
 * and per-group activation values, so 16 rows vectorize across the 16 lanes of a zmm (d, dmin,
 * ds, dm per lane; xs, xf broadcast) with no horizontal operation. 6 float ops per 32 values for
 * Q4_K (ds * xs, cvt, mul, dm * xf, sub, add; ds, dm once per 16-row block and group), 4 for Q6_K.
 */
#ifndef RUNQ_KQUANTS_H
#define RUNQ_KQUANTS_H

#include <stdint.h>
#include <string.h>

#define QK_K 256
#define Q4_K_BLOCK 144
#define Q6_K_BLOCK 210
#define KQ_ROWS 16
#define Q4P_SB 2368
#define Q6P_SB 3392
#define KQ_NC 4          // activation rows per generic kernel call (weights unpacked once for these)

enum { GGML_Q4_K = 12, GGML_Q6_K = 14 }; // ggml type numbers

static float fp16_to_f32(uint16_t h) {
    // IEEE half -> float, exact (every half value is a float)
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, e = h >> 10 & 0x1f, m = h & 0x3ff, bits;
    if (e == 0x1f) bits = sign | 0x7f800000u | m << 13; // inf / NaN
    else if (e) bits = sign | (e + 112) << 23 | m << 13; // normal: exponent bias 15 -> 127
    else if (m) { // subnormal: m * 2^-24, exact as a float
        float f = (float)m * 5.9604644775390625e-8f;
        memcpy(&bits, &f, 4);
        bits |= sign;
    } else bits = sign; // +-0
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static inline uint16_t kq_u16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }

static inline int kq_block_bytes(int type) { return type == GGML_Q4_K ? Q4_K_BLOCK : Q6_K_BLOCK; }
static inline int kq_sb_bytes(int type) { return type == GGML_Q4_K ? Q4P_SB : Q6P_SB; }
static inline size_t kq_size(int type, int d, int k) { return (size_t)(d / KQ_ROWS) * (k / QK_K) * kq_sb_bytes(type); }

static inline void q4k_scale_min(const uint8_t *s, int j, int *sc, int *m) {
    // ggml's get_scale_min_k4: 6-bit scale and min of sub-block j from the 12 packed bytes
    if (j < 4) {
        *sc = s[j] & 63;
        *m = s[j + 4] & 63;
    } else {
        *sc = (s[j + 4] & 15) | (s[j - 4] >> 6) << 4;
        *m = (s[j + 4] >> 4) | (s[j] >> 6) << 4;
    }
}

static inline int q4k_q(const uint8_t *blk, int i) {
    // 4-bit quant of value i (0..255) of a Q4_K block
    return (blk[16 + 32 * (i / 64) + i % 32] >> (4 * (i / 32 % 2))) & 15;
}

static inline int q6k_q(const uint8_t *blk, int i) {
    // 6-bit quant (0..63, before the -32) of value i of a Q6_K block
    int n = i / 128, p = i / 32 % 4, l = i % 32;
    int lo = (blk[64 * n + 32 * (p & 1) + l] >> (4 * (p >> 1))) & 15;
    int hi = (blk[128 + 32 * n + l] >> (2 * p)) & 3;
    return lo | hi << 4;
}

static int kq_block_ok(int type, const uint8_t *blk) {
    // the fp16 scales are finite (Q4_K: d and dmin; Q6_K: d); 0 for a corrupt block
    if (type == GGML_Q4_K)
        return (kq_u16(blk) & 0x7c00) != 0x7c00 && (kq_u16(blk + 2) & 0x7c00) != 0x7c00;
    return (kq_u16(blk + 208) & 0x7c00) != 0x7c00;
}

static void kq_dequant_row(int type, const uint8_t *blk, int n, float *y) {
    // n values (n % 256 == 0) from their blocks, ggml's expressions (see the header comment)
    for (int s = 0; s < n / QK_K; s++, y += QK_K) {
        if (type == GGML_Q4_K) {
            const uint8_t *b = blk + (size_t)s * Q4_K_BLOCK;
            float d = fp16_to_f32(kq_u16(b)), dmin = fp16_to_f32(kq_u16(b + 2));
            for (int j = 0; j < 8; j++) {
                int sc, m;
                q4k_scale_min(b + 4, j, &sc, &m);
                float d1 = d * sc, m1 = dmin * m;
                for (int l = 0; l < 32; l++)
                    y[32 * j + l] = d1 * q4k_q(b, 32 * j + l) - m1;
            }
        } else {
            const uint8_t *b = blk + (size_t)s * Q6_K_BLOCK;
            float d = fp16_to_f32(kq_u16(b + 208));
            for (int i = 0; i < QK_K; i++)
                y[i] = d * (int8_t)b[192 + i / 16] * (q6k_q(b, i) - 32);
        }
    }
}

static void kq_pack_blocks(int type, uint8_t *kp, const uint8_t *src, int nblk, int k) {
    // nblk blocks of 16 rows (k values each, row-major GGUF blocks at src) -> the Q4P / Q6P layout
    // at kp (block 0 of the run). Serial (the caller splits the work). Every output byte is written
    // once, built from the file's bytes (the header comment's index formulas)
    int nsb = k / QK_K, bb = kq_block_bytes(type), sbb = kq_sb_bytes(type);
    for (int rb = 0; rb < nblk; rb++)
        for (int s = 0; s < nsb; s++) {
            uint8_t *o = kp + ((size_t)rb * nsb + s) * sbb;
            for (int r = 0; r < KQ_ROWS; r++) {
                const uint8_t *b = src + ((size_t)(rb * KQ_ROWS + r) * nsb + s) * bb;
                if (type == GGML_Q4_K) {
                    memcpy(o + 2 * r, b, 2);
                    memcpy(o + 32 + 2 * r, b + 2, 2);
                    for (int j = 0; j < 8; j++) {
                        int sc, m;
                        q4k_scale_min(b + 4, j, &sc, &m);
                        o[64 + 16 * j + r] = (uint8_t)sc;
                        o[192 + 16 * j + r] = (uint8_t)m;
                        // sub-block j = nibble j % 2 of qs[32 (j / 2) + 0..31]
                        const uint8_t *qs = b + 16 + 32 * (j / 2);
                        int sh = 4 * (j % 2);
                        for (int c = 0; c < 4; c++)
                            for (int e = 0; e < 4; e++)
                                o[320 + 256 * j + 64 * c + 4 * r + e] =
                                    (uint8_t)((qs[8 * c + e] >> sh & 15) | (qs[8 * c + 4 + e] >> sh & 15) << 4);
                    }
                } else {
                    float d = fp16_to_f32(kq_u16(b + 208));
                    memcpy(o + 4 * r, &d, 4);
                    uint8_t q[QK_K]; // the 6-bit values (ggml's loop order: n, l, p)
                    for (int n = 0; n < 2; n++)
                        for (int l = 0; l < 32; l++) {
                            uint8_t ql0 = b[64 * n + l], ql1 = b[64 * n + 32 + l], qh = b[128 + 32 * n + l];
                            q[128 * n + l] = (uint8_t)((ql0 & 15) | (qh & 3) << 4);
                            q[128 * n + 32 + l] = (uint8_t)((ql1 & 15) | (qh >> 2 & 3) << 4);
                            q[128 * n + 64 + l] = (uint8_t)((ql0 >> 4) | (qh >> 4 & 3) << 4);
                            q[128 * n + 96 + l] = (uint8_t)((ql1 >> 4) | (qh >> 6 & 3) << 4);
                        }
                    for (int j = 0; j < 16; j++) {
                        o[64 + 16 * j + r] = b[192 + j];
                        const uint8_t *qj = q + 16 * j;
                        for (int c = 0; c < 2; c++)
                            for (int e = 0; e < 4; e++)
                                o[320 + 128 * j + 64 * c + 4 * r + e] = (uint8_t)((qj[8 * c + e] & 15) | (qj[8 * c + 4 + e] & 15) << 4);
                        for (int e = 0; e < 4; e++)
                            o[2368 + 64 * j + 4 * r + e] = (uint8_t)((qj[e] >> 4) | (qj[4 + e] >> 4) << 2 |
                                                                     (qj[8 + e] >> 4) << 4 | (qj[12 + e] >> 4) << 6);
                    }
                }
            }
        }
}

static void kq_packed_row(int type, const uint8_t *kp, int row, int k, float *y) {
    // row `row` of a repacked matrix, dequantized with the same expressions as kq_dequant_row
    // (test hook: checks the repacking against the file's blocks)
    int nsb = k / QK_K, sbb = kq_sb_bytes(type), r = row % KQ_ROWS;
    for (int s = 0; s < nsb; s++, y += QK_K) {
        const uint8_t *o = kp + ((size_t)(row / KQ_ROWS) * nsb + s) * sbb;
        if (type == GGML_Q4_K) {
            float d = fp16_to_f32(kq_u16(o + 2 * r)), dmin = fp16_to_f32(kq_u16(o + 32 + 2 * r));
            for (int j = 0; j < 8; j++) {
                float d1 = d * o[64 + 16 * j + r], m1 = dmin * o[192 + 16 * j + r];
                for (int w = 0; w < 32; w++) {
                    int q = (o[320 + 256 * j + 64 * (w / 8) + 4 * r + w % 4] >> (4 * (w % 8 / 4))) & 15;
                    y[32 * j + w] = d1 * q - m1;
                }
            }
        } else {
            float d;
            memcpy(&d, o + 4 * r, 4);
            for (int i = 0; i < QK_K; i++) {
                int j = i / 16, w = i % 16;
                int lo = (o[320 + 128 * j + 64 * (w / 8) + 4 * r + w % 4] >> (4 * (w % 8 / 4))) & 15;
                int hi = (o[2368 + 64 * j + 4 * r + w % 4] >> (2 * (w / 4))) & 3;
                y[i] = d * (int8_t)o[64 + 16 * j + r] * ((lo | hi << 4) - 32);
            }
        }
    }
}

static void kq_act_prep(float *xf, const int8_t *xq, const float *xs, int n) {
    // xf[g] = xs[g] * (float) (sum of the 32 xq of group g), g < n / 32 (see the definition)
    for (int g = 0; g < n / 32; g++) {
        int32_t sum = 0;
        for (int i = 0; i < 32; i++)
            sum += xq[32 * g + i];
        xf[g] = xs[g] * (float)sum;
    }
}

// The generic kernels: exactly the operations of the definition (gcc: no contraction, no
// reassociation in these functions, whatever the global flags)
#if defined(__GNUC__) && !defined(__clang__)
    #define KQ_EXACT __attribute__((optimize("fp-contract=off", "no-associative-math", "no-reciprocal-math")))
#else
    #define KQ_EXACT
#endif

KQ_EXACT static void q4p_dot(float *out, size_t ldo, const uint8_t *wb, int k, int nc, const int8_t *xq, size_t ldx,
                             const float *xs, size_t ldxs, const float *xf, size_t ldxf) {
    // out[c * ldo + r] = row r of the 16-row Q4P block wb . activation row c (c < nc <= KQ_NC; row c
    // at xq + c * ldx, its group scales at xs + c * ldxs, its group sums at xf + c * ldxf)
    float val[KQ_NC][KQ_ROWS];
    memset(val, 0, sizeof(val));
    for (int s = 0; s < k / QK_K; s++) {
        const uint8_t *o = wb + (size_t)s * Q4P_SB;
        float d[KQ_ROWS], dmin[KQ_ROWS];
        for (int r = 0; r < KQ_ROWS; r++) {
            d[r] = fp16_to_f32(kq_u16(o + 2 * r));
            dmin[r] = fp16_to_f32(kq_u16(o + 32 + 2 * r));
        }
        for (int j = 0; j < 8; j++) {
            int g = 8 * s + j;
            const uint8_t *qs = o + 320 + 256 * j;
            uint8_t w[KQ_ROWS][32]; // the sub-block unpacked, row-major
            for (int c = 0; c < 4; c++)
                for (int r = 0; r < KQ_ROWS; r++)
                    for (int b = 0; b < 4; b++) {
                        uint8_t v = qs[64 * c + 4 * r + b];
                        w[r][8 * c + b] = v & 15;
                        w[r][8 * c + 4 + b] = v >> 4;
                    }
            float ds[KQ_ROWS], dm[KQ_ROWS];
            for (int r = 0; r < KQ_ROWS; r++) {
                ds[r] = d[r] * (float)o[64 + 16 * j + r];
                dm[r] = dmin[r] * (float)o[192 + 16 * j + r];
            }
            for (int c = 0; c < nc; c++) {
                const int8_t *x = xq + c * ldx + 32 * g;
                float xsg = xs[c * ldxs + g], xfg = xf[c * ldxf + g];
                for (int r = 0; r < KQ_ROWS; r++) {
                    int32_t isum = 0;
                    for (int i = 0; i < 32; i++)
                        isum += w[r][i] * x[i];
                    float a = (ds[r] * xsg) * (float)isum;
                    float b = dm[r] * xfg;
                    val[c][r] = val[c][r] + (a - b);
                }
            }
        }
    }
    for (int c = 0; c < nc; c++)
        memcpy(out + c * ldo, val[c], sizeof(val[c]));
}

KQ_EXACT static void q6p_dot(float *out, size_t ldo, const uint8_t *wb, int k, int nc, const int8_t *xq, size_t ldx,
                             const float *xs, size_t ldxs) {
    // the same for a Q6P block (no xf: the -32 is applied in int32)
    float val[KQ_NC][KQ_ROWS];
    memset(val, 0, sizeof(val));
    for (int s = 0; s < k / QK_K; s++) {
        const uint8_t *o = wb + (size_t)s * Q6P_SB;
        float d[KQ_ROWS];
        memcpy(d, o, sizeof(d));
        for (int g8 = 0; g8 < 8; g8++) {
            int g = 8 * s + g8;
            uint8_t w[2][KQ_ROWS][16]; // the two sub-blocks unpacked, row-major
            int32_t sc[2][KQ_ROWS];
            for (int h = 0; h < 2; h++) {
                int j = 2 * g8 + h;
                const uint8_t *ql = o + 320 + 128 * j, *qh = o + 2368 + 64 * j;
                for (int r = 0; r < KQ_ROWS; r++) {
                    sc[h][r] = (int8_t)o[64 + 16 * j + r];
                    for (int t = 0; t < 4; t++)
                        for (int b = 0; b < 4; b++) {
                            int lo = (ql[64 * (t >> 1) + 4 * r + b] >> (4 * (t & 1))) & 15;
                            int hi = (qh[4 * r + b] >> (2 * t)) & 3;
                            w[h][r][4 * t + b] = (uint8_t)(lo | hi << 4);
                        }
                }
            }
            for (int c = 0; c < nc; c++) {
                const int8_t *x = xq + c * ldx + 32 * g;
                float xsg = xs[c * ldxs + g];
                int32_t xsum[2] = { 0, 0 };
                for (int i = 0; i < 16; i++) {
                    xsum[0] += x[i];
                    xsum[1] += x[16 + i];
                }
                for (int r = 0; r < KQ_ROWS; r++) {
                    int32_t I = 0;
                    for (int h = 0; h < 2; h++) {
                        int32_t t = 0;
                        for (int i = 0; i < 16; i++)
                            t += w[h][r][i] * x[16 * h + i];
                        I += sc[h][r] * (t - 32 * xsum[h]);
                    }
                    val[c][r] = val[c][r] + (d[r] * xsg) * (float)I;
                }
            }
        }
    }
    for (int c = 0; c < nc; c++)
        memcpy(out + c * ldo, val[c], sizeof(val[c]));
}

static void kq_dot(int type, float *out, size_t ldo, const uint8_t *wb, int k, int nc, const int8_t *xq, size_t ldx,
                   const float *xs, size_t ldxs, const float *xf, size_t ldxf) {
    // one 16-row block of a K-quant matrix times nc <= KQ_NC activation rows
    if (type == GGML_Q4_K)
        q4p_dot(out, ldo, wb, k, nc, xq, ldx, xs, ldxs, xf, ldxf);
    else
        q6p_dot(out, ldo, wb, k, nc, xq, ldx, xs, ldxs);
}

#endif // RUNQ_KQUANTS_H
