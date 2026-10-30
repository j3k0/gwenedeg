/* Low-bit GGUF quants: Q2_K and ggml's i-quants IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS (as in
 * llama.cpp's IQ2_M / IQ3_M / "IQ2-mix" files, e.g. Underdog Saluki 27B 1.0): the block formats, their dequantization,
 * the repacked layouts the matmul kernels read, the arithmetic of a matmul output (the definition below), the generic
 * C kernel and the AVX-512 VNNI one (any build: the generic one; RUNQ_VNNI: iq512_*; NEON: the generic one). Like
 * kquants.h for Q3_K .. Q6_K, and on the same activations (int8 xq in groups of 32 with fp32 scales xs), so a matrix
 * of any of these types can sit next to K-quant and Q8_0 ones (a fused q / k / v or gate / up phase mixes them).
 *
 * Block formats (ggml-common.h, ggml-quants.c dequantize_row_*; super-blocks of QK_K = 256 values of one row; the
 * codebooks are in iq_grids.h; ksigns(s) = s | (popcount(s) & 1) << 7 completes 7 stored sign bits of 8 values):
 *   Q2_K, 84 bytes: scales[16], qs[64], fp16 d, fp16 dmin. Value i: q = (qs[32 (i / 128) + i % 32] >> 2 (i / 32 % 4))
 *     & 3, sub-block j = i / 16 with sc = scales[j] & 15, m = scales[j] >> 4. w = (d * sc) * q - dmin * m
 *   IQ2_XXS, 66 bytes: fp16 d, then per group ib (32 values) two u32 a0, a1: byte l of a0 = grid index of values 8 l ..
 *     8 l + 7 (iq2xxs_grid), signs ksigns(a1 >> 7 l & 127), ls = a1 >> 28. w = (d * (0.5 + ls) * 0.25) * grid * sign
 *   IQ2_XS, 74 bytes: fp16 d, u16 qs[32] (qs[4 ib + l]: 9-bit index into iq2xs_grid, signs ksigns(qs >> 9)),
 *     scales[8] (scales[ib]: low nibble the first 16 values, high nibble the second). w = (d * (0.5 + sc) * 0.25) * grid * sign
 *   IQ2_S, 82 bytes: fp16 d, qs[32] (low 8 bits of the index of chunk 4 ib + l), signs[32] (raw: bit j = value j of the
 *     chunk), qh[8] (bits 2 l, 2 l + 1 of qh[ib]: index bits 8, 9; iq2s_grid), scales[8] (as IQ2_XS)
 *   IQ3_XXS, 98 bytes: fp16 d, qs[64] (qs[8 ib + t]: index into iq3xxs_grid of values 4 t .. 4 t + 3), then per group a
 *     u32 a: signs ksigns(a >> 7 l & 127) of chunk l (8 values), ls = a >> 28. w = (d * (0.5 + ls) * 0.5) * grid * sign
 *   IQ3_S, 110 bytes: fp16 d, qs[64] (qs[8 ib + t]: low 8 bits of step t's index), qh[8] (bit t of qh[ib]: bit 8),
 *     signs[32] (raw, chunk 4 ib + l), scales[4] (nibble ib % 2 of scales[ib / 2]; iq3s_grid). w = (d * (1 + 2 sc)) * grid * sign
 *   IQ1_S, 50 bytes: fp16 d, qs[32] (low 8 bits of chunk 4 ib + l's index into iq1s_grid: int8 -1 / 0 / 1), u16 qh[8]:
 *     qh[ib] bits 3 l .. 3 l + 2 = index bits 8..10 of chunk l, bits 12..14 the scale s, bit 15 the sign of delta.
 *     w = (d * (2 s + 1)) * (grid + delta), delta = +-0.125
 *   IQ1_M, 56 bytes: qs[32] (as IQ1_S), qh[16] (nibble l % 2 of qh[2 ib + l / 2]: index bits 8..10 of chunk l, bit 3 its
 *     delta sign), u16 sc[4]: the 3-bit scales of the 16 sub-blocks of 16 (bits 6 (ib % 2) and + 3 of sc[ib / 2]), the
 *     fp16 d in their top nibbles (sc[0] >> 12 | (sc[1] >> 8 & 0xf0) | (sc[2] >> 4 & 0xf00) | (sc[3] & 0xf000))
 *   IQ4_XS, 136 bytes: fp16 d, u16 scales_h, scales_l[4], qs[128]: ls = (nibble ib % 2 of scales_l[ib / 2]) | (bits
 *     2 ib, 2 ib + 1 of scales_h) << 4; qs[16 ib + j] low nibble = value j, high nibble = value 16 + j (kvalues_iq4nl).
 *     w = (d * (ls - 32)) * kvalues_iq4nl[q]
 * Dequantization (iq_dequant_row: embedding rows, the -m dequant hook) uses ggml's expressions. Every one of them is
 * exact in fp32 except Q2_K's final subtraction (d has 11 significant bits, the scale factor <= 5, the value <= 7:
 * IQ4_XS 11 + 5 + 7 = 23 bits; delta = 1 / 8), so the value does not depend on how the product is grouped:
 * w = (D * S) * v below gives the same float, sign of zero included.
 *
 * Integer form: every type is w = D * S * v with D per row and super-block (fp32, exact), S an integer per sub-block of
 * 16 values and v a small integer per value:
 *   type      D           S (per 16; = per 32 where marked)   v                           bias B (AVX-512)
 *   Q2_K      d           sc (and the min m: dmin)            q 0..3                      0
 *   IQ2_XXS   d / 8       2 ls + 1 (32)                       +-{8, 25, 43}               43
 *   IQ2_XS    d / 8       2 sc + 1                            +-{8, 25, 43}               43
 *   IQ2_S     d / 8       2 sc + 1                            +-{8, 25, 43}               43
 *   IQ3_XXS   d / 4       2 ls + 1 (32)                       +-{4 .. 62}                 62
 *   IQ3_S     d           2 sc + 1 (32)                       +-{1 .. 15}                 15
 *   IQ1_S     d / 8       2 s + 1 (32)                        8 grid + (delta < 0 ? -1 : 1)   9
 *   IQ1_M     d / 8       2 s + 1                             8 grid +- 1 (delta per 8)   9
 *   IQ4_XS    d           ls - 32 (32)                        kvalues_iq4nl -127..113     128
 *
 * Arithmetic of output row r (W row r . x), the definition. Every operation is an fp32 operation rounded to nearest,
 * no FMA contraction, no reassociation; groups g = 0 .. k/32 - 1 in order, val = 0 at the start; super-block s = g / 8
 * of the row, sub-blocks h = 0, 1 (the 2 x 16 values of group g):
 *   T_h = sum over the 16 values of v * xq                    (int32, exact)
 *   I = S_0 * T_0 + S_1 * T_1                                (int32, exact: |I| <= 31 * 62 * 127 * 32 < 2^23,
 *                                                              IQ4_XS 32 * 127 * 127 * 32 < 2^24)
 *   val = val + (D * xs[g]) * (float) I
 *   Q2_K: also M = m_0 * X_0 + m_1 * X_1 (X_h: the sum of the 16 xq of sub-block h), and
 *   val = val + ((d * xs[g]) * (float) I - (dmin * xs[g]) * (float) M)
 * The integer parts may be computed in any order. AVX-512 (vpdpbusd: u8 x s8) takes the codes as u8 v + B and removes
 * the bias with the activation sums of 16 (c16 of kquants.h, -32 X): T_h = sum (v + B) xq + B (c16 >> 5).
 *
 * Repacked layouts, for (d, k) matrices with d % 16 == 0, k % 256 == 0: as kquants.h, block rb = rows 16 rb .. 16 rb +
 * 15, super-block s at kp + (rb * k / 256 + s) * SB (one block contiguous over all k). The file's fields stay as they
 * are, transposed over the block's 16 rows so that a field of the 16 rows is one load ("lane r" = row r): per group g
 * the grid indices of a chunk as 16 bytes (zero-extended to the 16 dword lanes, then gathered), the sign bits as one
 * 64-bit mask per step of 4 columns t (bit 4 r + b = value 4 t + b of row r is negative: a masked negation), the scales
 * as nibbles in one dword per row. Sizes per row and super-block (file: ...): SB / 16.
 *   common  +0  uint16 d[16]   the fp16 d of row r (IQ1_M: the assembled one; Q2_K: then uint16 dmin[16] at +32)
 *   Q2P     SB 1344 (84 = file). +64 uint8 scm[16][16] at 16 j + r: scales[j] of row r. +320 + 128 g + 64 h: lane r,
 *           byte b, bits 2 t, 2 t + 1 = q of column 32 g + 16 h + 4 t + b (t < 4)
 *   I2XXS   SB 1120 (70; file 66). +32 uint32 sc[16]: nibble g = ls of group g. +96 + 128 g: idx[4][16] (byte 16 l + r:
 *           the grid index of chunk l), +64 sgn[8] (uint64, step t of the group)
 *   I2XS,   SB 1312 (82; IQ2_XS file 74, IQ2_S 82). +32 uint32 sc[2][16]: at 64 p + 4 r nibble 2 (g % 4) + h = the scale
 *   I2S     of sub-block h of group g = 4 p + g % 4. +160 + 144 g: idx[4][16] (low 8 bits), +64 hi[16] (bits 2 l,
 *           2 l + 1: index bits 8, 9 of chunk l; IQ2_XS: bit 2 l only), +80 sgn[8]
 *   I3XXS   SB 1632 (102; file 98). +32 uint32 sc[16] (nibble g). +96 + 192 g: idx[8][16] (step t), +128 sgn[8]
 *   I3S     SB 1760 (110 = file). +32 uint32 sc[16] (nibble g). +96 + 208 g: idx[8][16] (low 8 bits), +128 hi[16] (bit t:
 *           index bit 8 of step t: the file's qh[g]), +144 sgn[8]
 *   I1S     SB 800 (50 = file). +32 + 96 g: idx[4][16] (low 8 bits), +64 uint16 qh[16] (the file's qh[g] of row r)
 *   I1M     SB 928 (58; file 56). +32 uint32 sc[2][16] (as I2XS, 3-bit scales). +160 + 96 g: idx[4][16], +64 qh[2][16]
 *           (16 p + r: the file's qh[2 g + p] of row r)
 *   I4XS    SB 2240 (140; file 136). +32 int8 sc[8][16] (16 g + r: ls - 32). +192 + 256 g: 4 chunks c of 64 bytes: lane
 *           r, byte b: low nibble = q of column 32 g + 8 c + b, high nibble = of column + 4 (Q4P's nibble order)
 */
#ifndef RUNQ_IQUANTS_H
#define RUNQ_IQUANTS_H

#include <stdint.h>
#include <string.h>

#include "iq_grids.h"
#include "kquants.h"

enum { GGML_Q2_K = 10, GGML_IQ2_XXS = 16, GGML_IQ2_XS = 17, GGML_IQ3_XXS = 18, GGML_IQ1_S = 19, GGML_IQ3_S = 21,
       GGML_IQ2_S = 22, GGML_IQ4_XS = 23, GGML_IQ1_M = 29 }; // ggml type numbers

static inline int iq_type(int type) {
    // 1 for the types of this header
    return type == GGML_Q2_K || type == GGML_IQ2_XXS || type == GGML_IQ2_XS || type == GGML_IQ3_XXS || type == GGML_IQ1_S ||
           type == GGML_IQ3_S || type == GGML_IQ2_S || type == GGML_IQ4_XS || type == GGML_IQ1_M;
}

static inline const char *iq_name(int type) {
    switch (type) {
    case GGML_Q2_K: return "Q2_K";
    case GGML_IQ2_XXS: return "IQ2_XXS";
    case GGML_IQ2_XS: return "IQ2_XS";
    case GGML_IQ3_XXS: return "IQ3_XXS";
    case GGML_IQ1_S: return "IQ1_S";
    case GGML_IQ3_S: return "IQ3_S";
    case GGML_IQ2_S: return "IQ2_S";
    case GGML_IQ4_XS: return "IQ4_XS";
    case GGML_IQ1_M: return "IQ1_M";
    }
    return "?";
}

static inline int iq_block_bytes(int type) {
    switch (type) {
    case GGML_Q2_K: return 84;
    case GGML_IQ2_XXS: return 66;
    case GGML_IQ2_XS: return 74;
    case GGML_IQ3_XXS: return 98;
    case GGML_IQ1_S: return 50;
    case GGML_IQ3_S: return 110;
    case GGML_IQ2_S: return 82;
    case GGML_IQ4_XS: return 136;
    case GGML_IQ1_M: return 56;
    }
    return 0;
}

static inline int iq_sb_bytes(int type) {
    switch (type) {
    case GGML_Q2_K: return 1344;
    case GGML_IQ2_XXS: return 1120;
    case GGML_IQ2_XS: case GGML_IQ2_S: return 1312;
    case GGML_IQ3_XXS: return 1632;
    case GGML_IQ1_S: return 800;
    case GGML_IQ3_S: return 1760;
    case GGML_IQ4_XS: return 2240;
    case GGML_IQ1_M: return 928;
    }
    return 0;
}

static inline size_t iq_size(int type, int d, int k) { return (size_t)(d / KQ_ROWS) * (k / QK_K) * iq_sb_bytes(type); }

// any GGUF matmul type with a repacked layout (kp): Q3_K .. Q6_K (kquants.h) or the types here
static inline int lq_block_bytes(int type) { return iq_type(type) ? iq_block_bytes(type) : kq_block_bytes(type); }
static inline int lq_sb_bytes(int type) { return iq_type(type) ? iq_sb_bytes(type) : kq_sb_bytes(type); }
static inline size_t lq_size(int type, int d, int k) { return iq_type(type) ? iq_size(type, d, k) : kq_size(type, d, k); }

static inline float iq_dmul(int type) {
    // D = fp32 d * this (exact): the constant factors of the scales (see the table above)
    return type == GGML_IQ3_XXS ? 0.25f : type == GGML_IQ2_XXS || type == GGML_IQ2_XS || type == GGML_IQ2_S ||
           type == GGML_IQ1_S || type == GGML_IQ1_M ? 0.125f : 1.0f;
}

static inline int iq_bias(int type) {
    // the AVX-512 kernels' code bias B (v + B in 0..255)
    switch (type) {
    case GGML_IQ2_XXS: case GGML_IQ2_XS: case GGML_IQ2_S: return 43;
    case GGML_IQ3_XXS: return 62;
    case GGML_IQ3_S: return 15;
    case GGML_IQ1_S: case GGML_IQ1_M: return 9;
    case GGML_IQ4_XS: return 128;
    }
    return 0;
}

static inline uint8_t iq_ksigns(unsigned s) { return (uint8_t)(s | (__builtin_popcount(s) & 1) << 7); }
static inline uint32_t iq_u32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }

static inline uint16_t iq1m_d(const uint8_t *blk) {
    // the fp16 d of an IQ1_M block (the top nibbles of its 4 scale words at +48)
    uint16_t sc[4];
    memcpy(sc, blk + 48, 8);
    return (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
}

static int iq_block_ok(int type, const uint8_t *blk) {
    // the fp16 scales are finite; 0 for a corrupt block
    if (type == GGML_Q2_K) return (kq_u16(blk + 80) & 0x7c00) != 0x7c00 && (kq_u16(blk + 82) & 0x7c00) != 0x7c00;
    if (type == GGML_IQ1_M) return (iq1m_d(blk) & 0x7c00) != 0x7c00;
    return (kq_u16(blk) & 0x7c00) != 0x7c00;
}

// ---------------------------------------------------------------------------- dequantization (ggml's expressions)

static void iq_dequant_row(int type, const uint8_t *blk, int n, float *y) {
    // n values (n % 256 == 0) from their blocks: ggml's dequantize_row_* (see the header comment)
    int bb = iq_block_bytes(type);
    for (int s = 0; s < n / QK_K; s++, y += QK_K) {
        const uint8_t *b = blk + (size_t)s * bb;
        switch (type) {
        case GGML_Q2_K: {
            float d = fp16_to_f32(kq_u16(b + 80)), dmin = fp16_to_f32(kq_u16(b + 82));
            for (int i = 0; i < QK_K; i++) {
                int sc = b[i / 16];
                float dl = d * (sc & 15), ml = dmin * (sc >> 4);
                y[i] = dl * ((b[16 + 32 * (i / 128) + i % 32] >> (2 * (i / 32 % 4))) & 3) - ml;
            }
            break;
        }
        case GGML_IQ2_XXS: {
            float d = fp16_to_f32(kq_u16(b));
            for (int ib = 0; ib < 8; ib++) {
                uint32_t a0 = iq_u32(b + 2 + 8 * ib), a1 = iq_u32(b + 6 + 8 * ib);
                float db = d * (0.5f + (a1 >> 28)) * 0.25f;
                for (int l = 0; l < 4; l++) {
                    uint64_t gr = iq2xxs_grid[a0 >> 8 * l & 255];
                    uint8_t sg = iq_ksigns(a1 >> 7 * l & 127);
                    for (int j = 0; j < 8; j++)
                        y[32 * ib + 8 * l + j] = db * (float)(gr >> 8 * j & 255) * (sg >> j & 1 ? -1.f : 1.f);
                }
            }
            break;
        }
        case GGML_IQ2_XS: case GGML_IQ2_S: {
            int xs = type == GGML_IQ2_XS;
            float d = fp16_to_f32(kq_u16(b));
            const uint8_t *sc = b + (xs ? 66 : 74);
            for (int ib = 0; ib < 8; ib++)
                for (int l = 0; l < 4; l++) {
                    float dl = d * (0.5f + (sc[ib] >> 4 * (l / 2) & 15)) * 0.25f;
                    uint64_t gr;
                    uint8_t sg;
                    if (xs) {
                        int q = kq_u16(b + 2 + 2 * (4 * ib + l));
                        gr = iq2xs_grid[q & 511];
                        sg = iq_ksigns(q >> 9);
                    } else {
                        gr = iq2s_grid[b[2 + 4 * ib + l] | (b[66 + ib] << (8 - 2 * l) & 0x300)];
                        sg = b[34 + 4 * ib + l];
                    }
                    for (int j = 0; j < 8; j++)
                        y[32 * ib + 8 * l + j] = dl * (float)(gr >> 8 * j & 255) * (sg >> j & 1 ? -1.f : 1.f);
                }
            break;
        }
        case GGML_IQ3_XXS: {
            float d = fp16_to_f32(kq_u16(b));
            for (int ib = 0; ib < 8; ib++) {
                uint32_t a = iq_u32(b + 66 + 4 * ib);
                float db = d * (0.5f + (a >> 28)) * 0.5f;
                for (int l = 0; l < 4; l++) {
                    uint8_t sg = iq_ksigns(a >> 7 * l & 127);
                    uint32_t g1 = iq3xxs_grid[b[2 + 8 * ib + 2 * l]], g2 = iq3xxs_grid[b[2 + 8 * ib + 2 * l + 1]];
                    for (int j = 0; j < 4; j++) {
                        y[32 * ib + 8 * l + j] = db * (float)(g1 >> 8 * j & 255) * (sg >> j & 1 ? -1.f : 1.f);
                        y[32 * ib + 8 * l + 4 + j] = db * (float)(g2 >> 8 * j & 255) * (sg >> (4 + j) & 1 ? -1.f : 1.f);
                    }
                }
            }
            break;
        }
        case GGML_IQ3_S: {
            float d = fp16_to_f32(kq_u16(b));
            for (int ib = 0; ib < 8; ib++) {
                float db = d * (1 + 2 * (b[106 + ib / 2] >> 4 * (ib % 2) & 15));
                int qh = b[66 + ib];
                for (int l = 0; l < 4; l++) {
                    uint32_t g1 = iq3s_grid[b[2 + 8 * ib + 2 * l] | (qh << (8 - 2 * l) & 256)];
                    uint32_t g2 = iq3s_grid[b[2 + 8 * ib + 2 * l + 1] | (qh << (7 - 2 * l) & 256)];
                    uint8_t sg = b[74 + 4 * ib + l];
                    for (int j = 0; j < 4; j++) {
                        y[32 * ib + 8 * l + j] = db * (float)(g1 >> 8 * j & 255) * (sg >> j & 1 ? -1.f : 1.f);
                        y[32 * ib + 8 * l + 4 + j] = db * (float)(g2 >> 8 * j & 255) * (sg >> (4 + j) & 1 ? -1.f : 1.f);
                    }
                }
            }
            break;
        }
        case GGML_IQ1_S: {
            float d = fp16_to_f32(kq_u16(b));
            for (int ib = 0; ib < 8; ib++) {
                int qh = kq_u16(b + 34 + 2 * ib);
                float dl = d * (2 * (qh >> 12 & 7) + 1), delta = qh & 0x8000 ? -0.125f : 0.125f;
                for (int l = 0; l < 4; l++) {
                    uint64_t gr = iq1s_grid[b[2 + 4 * ib + l] | (qh >> 3 * l & 7) << 8];
                    for (int j = 0; j < 8; j++)
                        y[32 * ib + 8 * l + j] = dl * ((float)(int8_t)(gr >> 8 * j) + delta);
                }
            }
            break;
        }
        case GGML_IQ1_M: {
            float d = fp16_to_f32(iq1m_d(b));
            for (int ib = 0; ib < 8; ib++) {
                int sc = kq_u16(b + 48 + 2 * (ib / 2));
                for (int l = 0; l < 4; l++) {
                    float dl = d * (2 * (sc >> (6 * (ib % 2) + 3 * (l / 2)) & 7) + 1);
                    int q = b[32 + 2 * ib + l / 2] >> 4 * (l % 2);
                    float delta = q & 8 ? -0.125f : 0.125f;
                    uint64_t gr = iq1s_grid[b[4 * ib + l] | (q & 7) << 8];
                    for (int j = 0; j < 8; j++)
                        y[32 * ib + 8 * l + j] = dl * ((float)(int8_t)(gr >> 8 * j) + delta);
                }
            }
            break;
        }
        case GGML_IQ4_XS: {
            float d = fp16_to_f32(kq_u16(b));
            int sh = kq_u16(b + 2);
            for (int ib = 0; ib < 8; ib++) {
                int ls = (b[4 + ib / 2] >> 4 * (ib % 2) & 15) | (sh >> 2 * ib & 3) << 4;
                float dl = d * (ls - 32);
                for (int j = 0; j < 16; j++) {
                    y[32 * ib + j] = dl * kvalues_iq4nl[b[8 + 16 * ib + j] & 15];
                    y[32 * ib + 16 + j] = dl * kvalues_iq4nl[b[8 + 16 * ib + j] >> 4];
                }
            }
            break;
        }
        }
    }
}

// ---------------------------------------------------------------------------- repacking

static inline void iq_or64(uint8_t *p, uint64_t v) {
    uint64_t u;
    memcpy(&u, p, 8);
    u |= v;
    memcpy(p, &u, 8);
}
static inline void iq_or32(uint8_t *p, uint32_t v) {
    uint32_t u;
    memcpy(&u, p, 4);
    u |= v;
    memcpy(p, &u, 4);
}
static inline void iq_signs8(uint8_t *sgn, int r, int l, unsigned s8) {
    // the 8 sign bits s8 of chunk l (values 8 l .. 8 l + 7 of a group) of row r into the step masks sgn[8]
    iq_or64(sgn + 8 * (2 * l), (uint64_t)(s8 & 15) << 4 * r);
    iq_or64(sgn + 8 * (2 * l + 1), (uint64_t)(s8 >> 4 & 15) << 4 * r);
}
static inline void iq_signs4(uint8_t *sgn, int r, int t, unsigned s4) {
    // the 4 sign bits of step t of row r
    iq_or64(sgn + 8 * t, (uint64_t)(s4 & 15) << 4 * r);
}

static void iq_pack_blocks(int type, uint8_t *kp, const uint8_t *src, int nblk, int k) {
    // nblk blocks of 16 rows (k values each, row-major GGUF blocks at src) -> the repacked layout at kp (block 0 of the
    // run); serial (the caller splits the work). Every output byte is built from the file's bytes (the header comment)
    int nsb = k / QK_K, bb = iq_block_bytes(type), sbb = iq_sb_bytes(type);
    for (int rb = 0; rb < nblk; rb++)
        for (int s = 0; s < nsb; s++) {
            uint8_t *o = kp + ((size_t)rb * nsb + s) * sbb;
            memset(o, 0, sbb);
            for (int r = 0; r < KQ_ROWS; r++) {
                const uint8_t *b = src + ((size_t)(rb * KQ_ROWS + r) * nsb + s) * bb;
                switch (type) {
                case GGML_Q2_K:
                    memcpy(o + 2 * r, b + 80, 2);
                    memcpy(o + 32 + 2 * r, b + 82, 2);
                    for (int j = 0; j < 16; j++)
                        o[64 + 16 * j + r] = b[j];
                    for (int i = 0; i < QK_K; i++) {
                        int q = (b[16 + 32 * (i / 128) + i % 32] >> (2 * (i / 32 % 4))) & 3;
                        int g = i / 32, h = i / 16 % 2, t = i % 16 / 4, c = i % 4;
                        o[320 + 128 * g + 64 * h + 4 * r + c] |= (uint8_t)(q << 2 * t);
                    }
                    break;
                case GGML_IQ2_XXS:
                    memcpy(o + 2 * r, b, 2);
                    for (int g = 0; g < 8; g++) {
                        uint32_t a0 = iq_u32(b + 2 + 8 * g), a1 = iq_u32(b + 6 + 8 * g);
                        uint8_t *gp = o + 96 + 128 * g;
                        iq_or32(o + 32 + 4 * r, (a1 >> 28) << 4 * g);
                        for (int l = 0; l < 4; l++) {
                            gp[16 * l + r] = (uint8_t)(a0 >> 8 * l);
                            iq_signs8(gp + 64, r, l, iq_ksigns(a1 >> 7 * l & 127));
                        }
                    }
                    break;
                case GGML_IQ2_XS: case GGML_IQ2_S: {
                    int xs = type == GGML_IQ2_XS;
                    memcpy(o + 2 * r, b, 2);
                    const uint8_t *sc = b + (xs ? 66 : 74);
                    for (int g = 0; g < 8; g++) {
                        uint8_t *gp = o + 160 + 144 * g;
                        iq_or32(o + 32 + 64 * (g / 4) + 4 * r, (uint32_t)sc[g] << 8 * (g % 4));
                        for (int l = 0; l < 4; l++) {
                            if (xs) {
                                int q = kq_u16(b + 2 + 2 * (4 * g + l));
                                gp[16 * l + r] = (uint8_t)q;
                                gp[64 + r] |= (uint8_t)((q >> 8 & 1) << 2 * l);
                                iq_signs8(gp + 80, r, l, iq_ksigns(q >> 9));
                            } else {
                                gp[16 * l + r] = b[2 + 4 * g + l];
                                iq_signs8(gp + 80, r, l, b[34 + 4 * g + l]);
                            }
                        }
                        if (!xs) gp[64 + r] = b[66 + g];
                    }
                    break;
                }
                case GGML_IQ3_XXS:
                    memcpy(o + 2 * r, b, 2);
                    for (int g = 0; g < 8; g++) {
                        uint32_t a = iq_u32(b + 66 + 4 * g);
                        uint8_t *gp = o + 96 + 192 * g;
                        iq_or32(o + 32 + 4 * r, (a >> 28) << 4 * g);
                        for (int t = 0; t < 8; t++)
                            gp[16 * t + r] = b[2 + 8 * g + t];
                        for (int l = 0; l < 4; l++)
                            iq_signs8(gp + 128, r, l, iq_ksigns(a >> 7 * l & 127));
                    }
                    break;
                case GGML_IQ3_S:
                    memcpy(o + 2 * r, b, 2);
                    for (int g = 0; g < 8; g++) {
                        uint8_t *gp = o + 96 + 208 * g;
                        iq_or32(o + 32 + 4 * r, (uint32_t)(b[106 + g / 2] >> 4 * (g % 2) & 15) << 4 * g);
                        for (int t = 0; t < 8; t++)
                            gp[16 * t + r] = b[2 + 8 * g + t];
                        gp[128 + r] = b[66 + g];
                        for (int l = 0; l < 4; l++)
                            iq_signs8(gp + 144, r, l, b[74 + 4 * g + l]);
                    }
                    break;
                case GGML_IQ1_S:
                    memcpy(o + 2 * r, b, 2);
                    for (int g = 0; g < 8; g++) {
                        uint8_t *gp = o + 32 + 96 * g;
                        for (int l = 0; l < 4; l++)
                            gp[16 * l + r] = b[2 + 4 * g + l];
                        memcpy(gp + 64 + 2 * r, b + 34 + 2 * g, 2);
                    }
                    break;
                case GGML_IQ1_M: {
                    uint16_t d = iq1m_d(b);
                    memcpy(o + 2 * r, &d, 2);
                    for (int g = 0; g < 8; g++) {
                        uint8_t *gp = o + 160 + 96 * g;
                        int sc = kq_u16(b + 48 + 2 * (g / 2)) >> 6 * (g % 2);
                        iq_or32(o + 32 + 64 * (g / 4) + 4 * r, (uint32_t)((sc & 7) | (sc >> 3 & 7) << 4) << 8 * (g % 4));
                        for (int l = 0; l < 4; l++)
                            gp[16 * l + r] = b[4 * g + l];
                        gp[64 + r] = b[32 + 2 * g];
                        gp[80 + r] = b[32 + 2 * g + 1];
                    }
                    break;
                }
                case GGML_IQ4_XS: {
                    memcpy(o + 2 * r, b, 2);
                    int sh = kq_u16(b + 2);
                    for (int g = 0; g < 8; g++) {
                        int ls = (b[4 + g / 2] >> 4 * (g % 2) & 15) | (sh >> 2 * g & 3) << 4;
                        o[32 + 16 * g + r] = (uint8_t)(int8_t)(ls - 32);
                        const uint8_t *qs = b + 8 + 16 * g;
                        uint8_t q[32];
                        for (int j = 0; j < 16; j++) {
                            q[j] = qs[j] & 15;
                            q[16 + j] = qs[j] >> 4;
                        }
                        for (int c = 0; c < 4; c++)
                            for (int e = 0; e < 4; e++)
                                o[192 + 256 * g + 64 * c + 4 * r + e] = (uint8_t)(q[8 * c + e] | q[8 * c + 4 + e] << 4);
                    }
                    break;
                }
                }
            }
        }
}

// ---------------------------------------------------------------------------- the repacked layouts, scalar

static inline void iq_sb_d(int type, const uint8_t *o, float D[KQ_ROWS], float Dm[KQ_ROWS]) {
    // D (and Q2_K's dmin) of the 16 rows of a repacked super-block
    float cm = iq_dmul(type);
    for (int r = 0; r < KQ_ROWS; r++) {
        D[r] = fp16_to_f32(kq_u16(o + 2 * r)) * cm;
        Dm[r] = type == GGML_Q2_K ? fp16_to_f32(kq_u16(o + 32 + 2 * r)) : 0.0f;
    }
}

static inline int iq_sgn(const uint8_t *sgn, int r, int i) {
    // the sign bit of value i (< 32) of row r from a group's step masks
    return (int)(kq_u64(sgn + 8 * (i / 4)) >> (4 * r + i % 4) & 1);
}

static void iq_unpack_group(int type, const uint8_t *o, int g, int8_t v[KQ_ROWS][32], int32_t S[KQ_ROWS][2],
                            int32_t M[KQ_ROWS][2]) {
    // group g of a repacked super-block: the codes v, the sub-block scales S (and Q2_K's mins M) of the 16 rows
    for (int r = 0; r < KQ_ROWS; r++) {
        M[r][0] = M[r][1] = 0;
        switch (type) {
        case GGML_Q2_K: {
            const uint8_t *gp = o + 320 + 128 * g;
            for (int h = 0; h < 2; h++) {
                int scm = o[64 + 16 * (2 * g + h) + r];
                S[r][h] = scm & 15;
                M[r][h] = scm >> 4;
                for (int i = 0; i < 16; i++)
                    v[r][16 * h + i] = (int8_t)(gp[64 * h + 4 * r + i % 4] >> 2 * (i / 4) & 3);
            }
            break;
        }
        case GGML_IQ2_XXS: {
            const uint8_t *gp = o + 96 + 128 * g;
            S[r][0] = S[r][1] = 2 * (int)(iq_u32(o + 32 + 4 * r) >> 4 * g & 15) + 1;
            for (int i = 0; i < 32; i++) {
                int m = (int)(iq2xxs_grid[gp[16 * (i / 8) + r]] >> 8 * (i % 8) & 255);
                v[r][i] = (int8_t)(iq_sgn(gp + 64, r, i) ? -m : m);
            }
            break;
        }
        case GGML_IQ2_XS: case GGML_IQ2_S: {
            const uint8_t *gp = o + 160 + 144 * g;
            uint32_t sc = iq_u32(o + 32 + 64 * (g / 4) + 4 * r) >> 8 * (g % 4);
            S[r][0] = 2 * (int)(sc & 15) + 1;
            S[r][1] = 2 * (int)(sc >> 4 & 15) + 1;
            for (int i = 0; i < 32; i++) {
                int l = i / 8, idx = gp[16 * l + r] | (gp[64 + r] >> 2 * l & 3) << 8;
                int m = (int)((type == GGML_IQ2_XS ? iq2xs_grid[idx] : iq2s_grid[idx]) >> 8 * (i % 8) & 255);
                v[r][i] = (int8_t)(iq_sgn(gp + 80, r, i) ? -m : m);
            }
            break;
        }
        case GGML_IQ3_XXS: {
            const uint8_t *gp = o + 96 + 192 * g;
            S[r][0] = S[r][1] = 2 * (int)(iq_u32(o + 32 + 4 * r) >> 4 * g & 15) + 1;
            for (int i = 0; i < 32; i++) {
                int m = (int)(iq3xxs_grid[gp[16 * (i / 4) + r]] >> 8 * (i % 4) & 255);
                v[r][i] = (int8_t)(iq_sgn(gp + 128, r, i) ? -m : m);
            }
            break;
        }
        case GGML_IQ3_S: {
            const uint8_t *gp = o + 96 + 208 * g;
            S[r][0] = S[r][1] = 2 * (int)(iq_u32(o + 32 + 4 * r) >> 4 * g & 15) + 1;
            for (int i = 0; i < 32; i++) {
                int t = i / 4, idx = gp[16 * t + r] | (gp[128 + r] >> t & 1) << 8;
                int m = (int)(iq3s_grid[idx] >> 8 * (i % 4) & 255);
                v[r][i] = (int8_t)(iq_sgn(gp + 144, r, i) ? -m : m);
            }
            break;
        }
        case GGML_IQ1_S: {
            const uint8_t *gp = o + 32 + 96 * g;
            int qh = kq_u16(gp + 64 + 2 * r), dl = qh & 0x8000 ? -1 : 1;
            S[r][0] = S[r][1] = 2 * (qh >> 12 & 7) + 1;
            for (int i = 0; i < 32; i++) {
                int l = i / 8;
                int8_t gv = (int8_t)(iq1s_grid[gp[16 * l + r] | (qh >> 3 * l & 7) << 8] >> 8 * (i % 8));
                v[r][i] = (int8_t)(8 * gv + dl);
            }
            break;
        }
        case GGML_IQ1_M: {
            const uint8_t *gp = o + 160 + 96 * g;
            uint32_t sc = iq_u32(o + 32 + 64 * (g / 4) + 4 * r) >> 8 * (g % 4);
            S[r][0] = 2 * (int)(sc & 7) + 1;
            S[r][1] = 2 * (int)(sc >> 4 & 7) + 1;
            for (int i = 0; i < 32; i++) {
                int l = i / 8, q = gp[64 + 16 * (l / 2) + r] >> 4 * (l % 2);
                int8_t gv = (int8_t)(iq1s_grid[gp[16 * l + r] | (q & 7) << 8] >> 8 * (i % 8));
                v[r][i] = (int8_t)(8 * gv + (q & 8 ? -1 : 1));
            }
            break;
        }
        case GGML_IQ4_XS: {
            const uint8_t *gp = o + 192 + 256 * g;
            S[r][0] = S[r][1] = (int8_t)o[32 + 16 * g + r];
            for (int i = 0; i < 32; i++)
                v[r][i] = kvalues_iq4nl[gp[64 * (i / 8) + 4 * r + i % 4] >> 4 * (i % 8 / 4) & 15];
            break;
        }
        }
    }
}

static void iq_packed_row(int type, const uint8_t *kp, int row, int k, float *y) {
    // row `row` of a repacked matrix, dequantized as (D * S) * v (Q2_K: ggml's (d * sc) * q - dmin * m): the same floats
    // as iq_dequant_row (test hook: checks the repacking against the file's blocks)
    int nsb = k / QK_K, sbb = iq_sb_bytes(type), r = row % KQ_ROWS;
    for (int s = 0; s < nsb; s++, y += QK_K) {
        const uint8_t *o = kp + ((size_t)(row / KQ_ROWS) * nsb + s) * sbb;
        float D[KQ_ROWS], Dm[KQ_ROWS];
        iq_sb_d(type, o, D, Dm);
        for (int g = 0; g < 8; g++) {
            int8_t v[KQ_ROWS][32];
            int32_t S[KQ_ROWS][2], M[KQ_ROWS][2];
            iq_unpack_group(type, o, g, v, S, M);
            for (int i = 0; i < 32; i++) {
                int h = i / 16;
                if (type == GGML_Q2_K)
                    y[32 * g + i] = D[r] * S[r][h] * v[r][i] - Dm[r] * M[r][h];
                else
                    y[32 * g + i] = (D[r] * S[r][h]) * v[r][i];
            }
        }
    }
}

// ---------------------------------------------------------------------------- the generic kernel (the definition)

KQ_EXACT static void iq_dot(int type, float *out, size_t ldo, const uint8_t *wb, int k, int nc, const int8_t *xq, size_t ldx,
                            const float *xs, size_t ldxs) { RQ_FP_EXACT
    // out[c * ldo + r] = row r of the 16-row block wb . activation row c (c < nc <= KQ_NC; row c at xq + c * ldx, its
    // group scales at xs + c * ldxs): exactly the operations of the definition
    float val[KQ_NC][KQ_ROWS];
    memset(val, 0, sizeof(val));
    int sbb = iq_sb_bytes(type), q2 = type == GGML_Q2_K;
    for (int s = 0; s < k / QK_K; s++) {
        const uint8_t *o = wb + (size_t)s * sbb;
        float D[KQ_ROWS], Dm[KQ_ROWS];
        iq_sb_d(type, o, D, Dm);
        for (int g8 = 0; g8 < 8; g8++) {
            int g = 8 * s + g8;
            int8_t v[KQ_ROWS][32];
            int32_t S[KQ_ROWS][2], M[KQ_ROWS][2];
            iq_unpack_group(type, o, g8, v, S, M);
            for (int c = 0; c < nc; c++) {
                const int8_t *x = xq + c * ldx + 32 * g;
                float xsg = xs[c * ldxs + g];
                int32_t X[2] = { 0, 0 };
                for (int i = 0; i < 16; i++) {
                    X[0] += x[i];
                    X[1] += x[16 + i];
                }
                for (int r = 0; r < KQ_ROWS; r++) {
                    int32_t I = 0;
                    for (int h = 0; h < 2; h++) {
                        int32_t t = 0;
                        for (int i = 0; i < 16; i++)
                            t += v[r][16 * h + i] * x[16 * h + i];
                        I += S[r][h] * t;
                    }
                    if (q2) {
                        int32_t Mi = M[r][0] * X[0] + M[r][1] * X[1];
                        val[c][r] = val[c][r] + ((D[r] * xsg) * (float)I - (Dm[r] * xsg) * (float)Mi);
                    } else
                        val[c][r] = val[c][r] + (D[r] * xsg) * (float)I;
                }
            }
        }
    }
    for (int c = 0; c < nc; c++)
        memcpy(out + c * ldo, val[c], sizeof(val[c]));
}

// ---------------------------------------------------------------------------- AVX-512 VNNI

#ifdef RUNQ_VNNI
#define RUNQ_IQX 1

static inline __m512i iq512_u8(const uint8_t *p) { return _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)p)); }
static inline __m512i iq512_bc4(const int8_t *p) { int32_t v; memcpy(&v, p, 4); return _mm512_set1_epi32(v); }
static inline __m512i iq512_s21(__m512i n) { return _mm512_or_si512(_mm512_slli_epi32(n, 1), _mm512_set1_epi32(1)); }

static inline __attribute__((always_inline)) void iq512_g64(const uint64_t *grid, __m512i idx, __m512i *m0, __m512i *m1) {
    // the 8-value grid entries of the 16 lanes' indices -> step 0 (values 0..3, the low dwords) and step 1
    const __m512i ev = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    const __m512i od = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
    __m512i a = _mm512_i32gather_epi64(_mm512_castsi512_si256(idx), (const void *)grid, 8);
    __m512i b = _mm512_i32gather_epi64(_mm512_extracti64x4_epi64(idx, 1), (const void *)grid, 8);
    *m0 = _mm512_permutex2var_epi32(a, ev, b);
    *m1 = _mm512_permutex2var_epi32(a, od, b);
}

static inline __m512i iq512_sgn(__m512i m, const uint8_t *mp, __m512i B) {
    // B + m, or B - m where the step's sign mask is set
    uint64_t k;
    memcpy(&k, mp, 8);
    return _mm512_mask_sub_epi8(_mm512_add_epi8(B, m), _cvtu64_mask64(k), B, m);
}

// AVX512-VBMI decode (Zen 4, Ice Lake and later; RUNQ_IQ_GATHER: the gathers below instead): the grid indices are
// looked up in the compact code tables of iq_grids.h (*_code: 2 or 3 bits per value) held in registers, 32 at a time
// (vpermi2w over 64 entries, a blend tree on the higher index bits; IQ1's 2048 entries: one 16-lane gather of codes per
// chunk), then each step's codes are cut out of the lane's code word (vpmultishiftqb) and mapped to v + B (vpermb on a
// 64-byte table that only depends on the code bits), the negative values by a second, masked vpermb on the table of
// B - |v| (IQ1: of delta -1). The same integers as the gathers' (and iq_unpack_group's) codes. ~3x faster than the
// gathers on Zen 4 (2.5 gathered elements / ns per core against ~15 table lookups / ns)
#if defined(__AVX512VBMI__) && !defined(RUNQ_IQ_GATHER)
#define RUNQ_IQ_VBMI 1

static inline __m512i iq512_rep4(int a, int b, int c, int d) {
    // a 64-byte vpermb table whose entry i only depends on i % 4
    return _mm512_set1_epi32((int)((uint32_t)a | (uint32_t)b << 8 | (uint32_t)c << 16 | (uint32_t)d << 24));
}
static inline __m512i iq512_rep8(int a, int b, int c, int d, int e, int f, int g, int h) {
    // the same for i % 8
    return _mm512_set1_epi64((long long)((uint64_t)a | (uint64_t)b << 8 | (uint64_t)c << 16 | (uint64_t)d << 24 |
                                         (uint64_t)e << 32 | (uint64_t)f << 40 | (uint64_t)g << 48 | (uint64_t)h << 56));
}

static inline __attribute__((always_inline)) __m512i iq512_lutw(const uint16_t *tab, const int n, __m512i idx) {
    // tab[idx] for the 32 word lanes of idx (< n, n = 256, 512 or 1024)
    __m512i r[16];
    for (int p = 0; p < n / 64; p++)
        r[p] = _mm512_permutex2var_epi16(_mm512_loadu_si512((const void *)(tab + 64 * p)), idx,
                                         _mm512_loadu_si512((const void *)(tab + 64 * p + 32)));
    for (int bit = 6, m = n / 64; m > 1; bit++, m /= 2) {
        __mmask32 k = _mm512_test_epi16_mask(idx, _mm512_set1_epi16((short)(1 << bit)));
        for (int q = 0; q < m / 2; q++)
            r[q] = _mm512_mask_blend_epi16(k, r[2 * q], r[2 * q + 1]);
    }
    return r[0];
}

// vpmultishiftqb controls: byte b of an even lane at bit offset o_b of the qword, of an odd lane at 32 + o_b: 2-bit codes
// of step 2 l + p of chunk l: o_b = 8 p + 2 b (0x2624222006040200, 0x2e2c2a280e0c0a08); 3-bit: o_b = 3 b (0x2926232009060300)

static inline __m512i iq512_cw(__m512i c, int half) {
    // word lanes 16 half .. 16 half + 15 (the 16 rows of one chunk or step) as dword lanes
    return _mm512_cvtepu16_epi32(half ? _mm512_extracti64x4_epi64(c, 1) : _mm512_castsi512_si256(c));
}

static inline __m512i iq512_vb(__m512i ms, __m512i lp, __m512i lm, uint64_t k) {
    // lp[code], or lm[code] where k is set (the step's sign / delta bits)
    return _mm512_mask_permutexvar_epi8(_mm512_permutexvar_epi8(ms, lp), _cvtu64_mask64(k), ms, lm);
}

static inline __m512i iq512_hiw(const uint8_t *p, int s0, int s1, int bits) {
    // 32 word lanes: (p[r] >> s0) & bits for lanes r < 16, (p[r] >> s1) & bits for lanes 16 + r, shifted to bit 8
    __m256i h = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)p));
    __m512i h2 = _mm512_inserti64x4(_mm512_castsi256_si512(h), h, 1);
    __m512i sh = _mm512_mask_blend_epi16(0xFFFF0000u, _mm512_set1_epi16((short)s0), _mm512_set1_epi16((short)s1));
    return _mm512_slli_epi16(_mm512_and_si512(_mm512_srlv_epi16(h2, sh), _mm512_set1_epi16((short)bits)), 8);
}

static inline uint64_t iq_k64(__mmask16 k) {
    // a lane mask as a byte mask (the 4 bytes of each set lane)
    return _pdep_u64((uint64_t)_cvtmask16_u32(k), 0x1111111111111111ull) * 15;
}

static inline __attribute__((always_inline)) int iq512_group_vbmi(const int type, const uint8_t *o, int g, __m512i w[8],
                                                                  __m512i *S0, __m512i *S1) {
    // iq512_group's codes for the IQ2 / IQ3 / IQ1 types (1), else 0 (Q2_K, IQ4_XS: no tables)
    const __m512i n15 = _mm512_set1_epi32(15);
    switch (type) {
    case GGML_IQ2_XXS: case GGML_IQ2_XS: case GGML_IQ2_S: {
        int xxs = type == GGML_IQ2_XXS;
        const uint8_t *gp = o + (xxs ? 96 + 128 * g : 160 + 144 * g), *sgn = gp + (xxs ? 64 : 80);
        const uint16_t *tab = xxs ? iq2xxs_code : type == GGML_IQ2_XS ? iq2xs_code : iq2s_code;
        const int n = xxs ? 256 : type == GGML_IQ2_XS ? 512 : 1024;
        if (xxs)
            *S0 = *S1 = iq512_s21(_mm512_and_si512(_mm512_srli_epi32(_mm512_loadu_si512((const void *)(o + 32)), 4 * g), n15));
        else {
            __m512i sc = _mm512_srli_epi32(_mm512_loadu_si512((const void *)(o + 32 + 64 * (g / 4))), 8 * (g % 4));
            *S0 = iq512_s21(_mm512_and_si512(sc, n15));
            *S1 = iq512_s21(_mm512_and_si512(_mm512_srli_epi32(sc, 4), n15));
        }
        __m512i ib = _mm512_loadu_si512((const void *)gp);
        __m512i i01 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(ib)), i23 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(ib, 1));
        if (!xxs) {
            i01 = _mm512_or_si512(i01, iq512_hiw(gp + 64, 0, 2, 3));
            i23 = _mm512_or_si512(i23, iq512_hiw(gp + 64, 4, 6, 3));
        }
        __m512i c[2] = { iq512_lutw(tab, n, i01), iq512_lutw(tab, n, i23) };
        const __m512i lp = iq512_rep4(51, 68, 86, 43), lm = iq512_rep4(35, 18, 0, 43);
        const __m512i ms0 = _mm512_set1_epi64(0x2624222006040200ll), ms1 = _mm512_set1_epi64(0x2e2c2a280e0c0a08ll);
        for (int l = 0; l < 4; l++) {
            __m512i cw = iq512_cw(c[l / 2], l % 2);
            w[2 * l] = iq512_vb(_mm512_multishift_epi64_epi8(ms0, cw), lp, lm, kq_u64(sgn + 16 * l));
            w[2 * l + 1] = iq512_vb(_mm512_multishift_epi64_epi8(ms1, cw), lp, lm, kq_u64(sgn + 16 * l + 8));
        }
        return 1;
    }
    case GGML_IQ3_XXS: case GGML_IQ3_S: {
        int xxs = type == GGML_IQ3_XXS;
        const uint8_t *gp = o + 96 + (xxs ? 192 : 208) * g, *sgn = gp + (xxs ? 128 : 144);
        *S0 = *S1 = iq512_s21(_mm512_and_si512(_mm512_srli_epi32(_mm512_loadu_si512((const void *)(o + 32)), 4 * g), n15));
        const __m512i lp = xxs ? iq512_rep8(66, 74, 82, 90, 98, 106, 114, 124) : iq512_rep8(16, 18, 20, 22, 24, 26, 28, 30);
        const __m512i lm = xxs ? iq512_rep8(58, 50, 42, 34, 26, 18, 10, 0) : iq512_rep8(14, 12, 10, 8, 6, 4, 2, 0);
        const __m512i ms = _mm512_set1_epi64(0x2926232009060300ll);
        for (int h = 0; h < 2; h++) {
            __m512i ib = _mm512_loadu_si512((const void *)(gp + 64 * h));
            __m512i i0 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(ib)), i1 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(ib, 1));
            if (!xxs) {
                i0 = _mm512_or_si512(i0, iq512_hiw(gp + 128, 4 * h, 4 * h + 1, 1));
                i1 = _mm512_or_si512(i1, iq512_hiw(gp + 128, 4 * h + 2, 4 * h + 3, 1));
            }
            __m512i c[2] = { iq512_lutw(xxs ? iq3xxs_code : iq3s_code, xxs ? 256 : 512, i0),
                             iq512_lutw(xxs ? iq3xxs_code : iq3s_code, xxs ? 256 : 512, i1) };
            for (int t = 0; t < 4; t++)
                w[4 * h + t] = iq512_vb(_mm512_multishift_epi64_epi8(ms, iq512_cw(c[t / 2], t % 2)), lp, lm, kq_u64(sgn + 8 * (4 * h + t)));
        }
        return 1;
    }
    case GGML_IQ1_S: case GGML_IQ1_M: {
        const __m512i lp = iq512_rep4(2, 10, 18, 9), lm = iq512_rep4(0, 8, 16, 9), m7 = _mm512_set1_epi32(7);
        const __m512i ms0 = _mm512_set1_epi64(0x2624222006040200ll), ms1 = _mm512_set1_epi64(0x2e2c2a280e0c0a08ll);
        if (type == GGML_IQ1_S) {
            const uint8_t *gp = o + 32 + 96 * g;
            __m512i qh = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(gp + 64)));
            *S0 = *S1 = iq512_s21(_mm512_and_si512(_mm512_srli_epi32(qh, 12), m7));
            uint64_t kd = iq_k64(_mm512_test_epi32_mask(qh, _mm512_set1_epi32(0x8000)));
            for (int l = 0; l < 4; l++) {
                __m512i idx = _mm512_or_si512(iq512_u8(gp + 16 * l), _mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(qh, 3 * l), m7), 8));
                __m512i cw = _mm512_i32gather_epi32(idx, (const void *)iq1s_code, 2); // (the high 16 bits: never read)
                w[2 * l] = iq512_vb(_mm512_multishift_epi64_epi8(ms0, cw), lp, lm, kd);
                w[2 * l + 1] = iq512_vb(_mm512_multishift_epi64_epi8(ms1, cw), lp, lm, kd);
            }
        } else {
            const uint8_t *gp = o + 160 + 96 * g;
            __m512i sc = _mm512_srli_epi32(_mm512_loadu_si512((const void *)(o + 32 + 64 * (g / 4))), 8 * (g % 4));
            *S0 = iq512_s21(_mm512_and_si512(sc, m7));
            *S1 = iq512_s21(_mm512_and_si512(_mm512_srli_epi32(sc, 4), m7));
            for (int l = 0; l < 4; l++) {
                __m512i q = _mm512_srli_epi32(iq512_u8(gp + 64 + 16 * (l / 2)), 4 * (l % 2));
                __m512i idx = _mm512_or_si512(iq512_u8(gp + 16 * l), _mm512_slli_epi32(_mm512_and_si512(q, m7), 8));
                uint64_t kd = iq_k64(_mm512_test_epi32_mask(q, _mm512_set1_epi32(8)));
                __m512i cw = _mm512_i32gather_epi32(idx, (const void *)iq1s_code, 2);
                w[2 * l] = iq512_vb(_mm512_multishift_epi64_epi8(ms0, cw), lp, lm, kd);
                w[2 * l + 1] = iq512_vb(_mm512_multishift_epi64_epi8(ms1, cw), lp, lm, kd);
            }
        }
        return 1;
    }
    }
    return 0;
}
#endif // AVX512-VBMI

static inline __attribute__((always_inline)) void iq512_group(const int type, const uint8_t *o, int g, __m512i w[8],
                                                              __m512i *S0, __m512i *S1, __m512i *M0, __m512i *M1) {
    // group g of a repacked super-block: the 8 steps of biased codes v + B (u8), the sub-block scales (Q2_K: the mins)
#ifdef RUNQ_IQ_VBMI
    if (iq512_group_vbmi(type, o, g, w, S0, S1)) return;
#endif
    const __m512i B = _mm512_set1_epi8((char)iq_bias(type)), n15 = _mm512_set1_epi32(15);
    switch (type) {
    case GGML_Q2_K: {
        const uint8_t *gp = o + 320 + 128 * g;
        __m512i s0 = iq512_u8(o + 64 + 32 * g), s1 = iq512_u8(o + 80 + 32 * g);
        *S0 = _mm512_and_si512(s0, n15);
        *S1 = _mm512_and_si512(s1, n15);
        *M0 = _mm512_srli_epi32(s0, 4);
        *M1 = _mm512_srli_epi32(s1, 4);
        const __m512i m3 = _mm512_set1_epi8(3);
        for (int h = 0; h < 2; h++) {
            __m512i q = _mm512_loadu_si512((const void *)(gp + 64 * h));
            w[4 * h] = _mm512_and_si512(q, m3);
            w[4 * h + 1] = _mm512_and_si512(_mm512_srli_epi16(q, 2), m3);
            w[4 * h + 2] = _mm512_and_si512(_mm512_srli_epi16(q, 4), m3);
            w[4 * h + 3] = _mm512_and_si512(_mm512_srli_epi16(q, 6), m3);
        }
        break;
    }
    case GGML_IQ2_XXS: {
        const uint8_t *gp = o + 96 + 128 * g;
        *S0 = *S1 = iq512_s21(_mm512_and_si512(_mm512_srli_epi32(_mm512_loadu_si512((const void *)(o + 32)), 4 * g), n15));
        for (int l = 0; l < 4; l++) {
            __m512i m0, m1;
            iq512_g64(iq2xxs_grid, iq512_u8(gp + 16 * l), &m0, &m1);
            w[2 * l] = iq512_sgn(m0, gp + 64 + 16 * l, B);
            w[2 * l + 1] = iq512_sgn(m1, gp + 72 + 16 * l, B);
        }
        break;
    }
    case GGML_IQ2_XS: case GGML_IQ2_S: {
        const uint8_t *gp = o + 160 + 144 * g;
        const uint64_t *grid = type == GGML_IQ2_XS ? iq2xs_grid : iq2s_grid;
        __m512i sc = _mm512_srli_epi32(_mm512_loadu_si512((const void *)(o + 32 + 64 * (g / 4))), 8 * (g % 4));
        *S0 = iq512_s21(_mm512_and_si512(sc, n15));
        *S1 = iq512_s21(_mm512_and_si512(_mm512_srli_epi32(sc, 4), n15));
        __m512i hi = iq512_u8(gp + 64);
        for (int l = 0; l < 4; l++) {
            __m512i idx = _mm512_or_si512(iq512_u8(gp + 16 * l),
                                          _mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(hi, 2 * l), _mm512_set1_epi32(3)), 8));
            __m512i m0, m1;
            iq512_g64(grid, idx, &m0, &m1);
            w[2 * l] = iq512_sgn(m0, gp + 80 + 16 * l, B);
            w[2 * l + 1] = iq512_sgn(m1, gp + 88 + 16 * l, B);
        }
        break;
    }
    case GGML_IQ3_XXS: {
        const uint8_t *gp = o + 96 + 192 * g;
        *S0 = *S1 = iq512_s21(_mm512_and_si512(_mm512_srli_epi32(_mm512_loadu_si512((const void *)(o + 32)), 4 * g), n15));
        for (int t = 0; t < 8; t++)
            w[t] = iq512_sgn(_mm512_i32gather_epi32(iq512_u8(gp + 16 * t), (const void *)iq3xxs_grid, 4), gp + 128 + 8 * t, B);
        break;
    }
    case GGML_IQ3_S: {
        const uint8_t *gp = o + 96 + 208 * g;
        *S0 = *S1 = iq512_s21(_mm512_and_si512(_mm512_srli_epi32(_mm512_loadu_si512((const void *)(o + 32)), 4 * g), n15));
        __m512i hi = iq512_u8(gp + 128);
        for (int t = 0; t < 8; t++) {
            __m512i idx = _mm512_or_si512(iq512_u8(gp + 16 * t),
                                          _mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(hi, t), _mm512_set1_epi32(1)), 8));
            w[t] = iq512_sgn(_mm512_i32gather_epi32(idx, (const void *)iq3s_grid, 4), gp + 144 + 8 * t, B);
        }
        break;
    }
    case GGML_IQ1_S: {
        const uint8_t *gp = o + 32 + 96 * g;
        __m512i qh = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(gp + 64)));
        *S0 = *S1 = iq512_s21(_mm512_and_si512(_mm512_srli_epi32(qh, 12), _mm512_set1_epi32(7)));
        __mmask16 kd = _mm512_test_epi32_mask(qh, _mm512_set1_epi32(0x8000));
        __m512i dv = _mm512_mask_blend_epi32(kd, _mm512_set1_epi32(0x01010101), _mm512_set1_epi32(-1));
        for (int l = 0; l < 4; l++) {
            __m512i idx = _mm512_or_si512(iq512_u8(gp + 16 * l),
                                          _mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(qh, 3 * l), _mm512_set1_epi32(7)), 8));
            __m512i m0, m1;
            iq512_g64(iq1s_grid9, idx, &m0, &m1);
            w[2 * l] = _mm512_add_epi8(m0, dv);
            w[2 * l + 1] = _mm512_add_epi8(m1, dv);
        }
        break;
    }
    case GGML_IQ1_M: {
        const uint8_t *gp = o + 160 + 96 * g;
        __m512i sc = _mm512_srli_epi32(_mm512_loadu_si512((const void *)(o + 32 + 64 * (g / 4))), 8 * (g % 4));
        *S0 = iq512_s21(_mm512_and_si512(sc, _mm512_set1_epi32(7)));
        *S1 = iq512_s21(_mm512_and_si512(_mm512_srli_epi32(sc, 4), _mm512_set1_epi32(7)));
        for (int l = 0; l < 4; l++) {
            __m512i q = _mm512_srli_epi32(iq512_u8(gp + 64 + 16 * (l / 2)), 4 * (l % 2));
            __m512i idx = _mm512_or_si512(iq512_u8(gp + 16 * l), _mm512_slli_epi32(_mm512_and_si512(q, _mm512_set1_epi32(7)), 8));
            __mmask16 kd = _mm512_test_epi32_mask(q, _mm512_set1_epi32(8));
            __m512i dv = _mm512_mask_blend_epi32(kd, _mm512_set1_epi32(0x01010101), _mm512_set1_epi32(-1));
            __m512i m0, m1;
            iq512_g64(iq1s_grid9, idx, &m0, &m1);
            w[2 * l] = _mm512_add_epi8(m0, dv);
            w[2 * l + 1] = _mm512_add_epi8(m1, dv);
        }
        break;
    }
    case GGML_IQ4_XS: {
        const uint8_t *gp = o + 192 + 256 * g;
        *S0 = *S1 = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(o + 32 + 16 * g)));
        const __m512i lut = _mm512_set_epi8(
            (char)241, (char)217, (char)197, (char)181, (char)166, (char)153, (char)141, (char)129, 118, 106, 93, 79, 63, 45, 24, 1,
            (char)241, (char)217, (char)197, (char)181, (char)166, (char)153, (char)141, (char)129, 118, 106, 93, 79, 63, 45, 24, 1,
            (char)241, (char)217, (char)197, (char)181, (char)166, (char)153, (char)141, (char)129, 118, 106, 93, 79, 63, 45, 24, 1,
            (char)241, (char)217, (char)197, (char)181, (char)166, (char)153, (char)141, (char)129, 118, 106, 93, 79, 63, 45, 24, 1);
        const __m512i m4 = _mm512_set1_epi8(15);
        for (int c = 0; c < 4; c++) {
            __m512i q = _mm512_loadu_si512((const void *)(gp + 64 * c));
            w[2 * c] = _mm512_shuffle_epi8(lut, _mm512_and_si512(q, m4));
            w[2 * c + 1] = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(q, 4), m4));
        }
        break;
    }
    }
}

static inline __attribute__((always_inline)) void iq512_blk(const int type, const int NC, float *out, size_t ldo,
                                                            const uint8_t *wb, int nsb, int nc, const int8_t *xq, size_t ldx,
                                                            const float *xs, size_t ldxs, const int32_t *c16, size_t ldc) {
    // iq_dot for nc <= NC activation rows, on the group's decoded codes (each operation rounded: kq6_fold's intrinsics)
    const int sbb = iq_sb_bytes(type), Bi = iq_bias(type), q2 = type == GGML_Q2_K;
    const int per32 = type == GGML_IQ2_XXS || type == GGML_IQ3_XXS || type == GGML_IQ3_S || type == GGML_IQ1_S ||
                      type == GGML_IQ4_XS;
    const float cm = iq_dmul(type);
    __m512 val[NC];
    for (int c = 0; c < NC; c++)
        val[c] = _mm512_setzero_ps();
    for (int s = 0; s < nsb; s++) {
        const uint8_t *o = wb + (size_t)s * sbb;
        __m512 D = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)o));
        if (cm != 1.0f) D = _mm512_mul_round_ps(D, _mm512_set1_ps(cm), Q8P_RND); // (exact: a power of 2)
        __m512 Dm = q2 ? _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(o + 32))) : _mm512_setzero_ps();
        for (int g8 = 0; g8 < 8; g8++) {
            __m512i w[8], S0, S1, M0 = _mm512_setzero_si512(), M1 = _mm512_setzero_si512();
            iq512_group(type, o, g8, w, &S0, &S1, &M0, &M1);
            int g = 8 * s + g8;
            for (int c = 0; c < NC; c++) {
                if (c >= nc) break;
                const int8_t *x = xq + c * ldx + 32 * g;
                const int32_t *cc = c16 + c * ldc + 2 * g;
                int n0 = cc[0] >> 5, n1 = cc[1] >> 5; // -X_0, -X_1 (c16 = -32 X, exact)
                __m512i p0 = _mm512_set1_epi32(Bi * n0), p1 = _mm512_set1_epi32(Bi * n1);
                for (int t = 0; t < 4; t++) {
                    p0 = _mm512_dpbusd_epi32(p0, w[t], iq512_bc4(x + 4 * t));
                    p1 = _mm512_dpbusd_epi32(p1, w[4 + t], iq512_bc4(x + 16 + 4 * t));
                }
                __m512i I = per32 ? _mm512_mullo_epi32(S0, _mm512_add_epi32(p0, p1))
                                  : _mm512_add_epi32(_mm512_mullo_epi32(S0, p0), _mm512_mullo_epi32(S1, p1));
                __m512 xsv = _mm512_set1_ps(xs[c * ldxs + g]);
                __m512 a = _mm512_mul_round_ps(_mm512_mul_round_ps(D, xsv, Q8P_RND), _mm512_cvtepi32_ps(I), Q8P_RND);
                if (q2) {
                    __m512i Mi = _mm512_add_epi32(_mm512_mullo_epi32(M0, _mm512_set1_epi32(-n0)),
                                                  _mm512_mullo_epi32(M1, _mm512_set1_epi32(-n1)));
                    __m512 b = _mm512_mul_round_ps(_mm512_mul_round_ps(Dm, xsv, Q8P_RND), _mm512_cvtepi32_ps(Mi), Q8P_RND);
                    a = _mm512_sub_round_ps(a, b, Q8P_RND);
                }
                val[c] = _mm512_add_round_ps(val[c], a, Q8P_RND);
            }
        }
    }
    for (int c = 0; c < nc && c < NC; c++)
        _mm512_storeu_ps(out + c * ldo, val[c]);
}

#define IQ512_ARGS float *out, size_t ldo, const uint8_t *wb, int nsb, int nc, const int8_t *xq, size_t ldx, const float *xs, \
                   size_t ldxs, const int32_t *c16, size_t ldc
#define IQ512_INST(NAME, TYPE) \
    static void NAME##_1(IQ512_ARGS) { iq512_blk(TYPE, 1, out, ldo, wb, nsb, nc, xq, ldx, xs, ldxs, c16, ldc); } \
    static void NAME##_8(IQ512_ARGS) { iq512_blk(TYPE, 8, out, ldo, wb, nsb, nc, xq, ldx, xs, ldxs, c16, ldc); }
IQ512_INST(iq512_q2k, GGML_Q2_K)
IQ512_INST(iq512_i2xxs, GGML_IQ2_XXS)
IQ512_INST(iq512_i2xs, GGML_IQ2_XS)
IQ512_INST(iq512_i2s, GGML_IQ2_S)
IQ512_INST(iq512_i3xxs, GGML_IQ3_XXS)
IQ512_INST(iq512_i3s, GGML_IQ3_S)
IQ512_INST(iq512_i1s, GGML_IQ1_S)
IQ512_INST(iq512_i1m, GGML_IQ1_M)
IQ512_INST(iq512_i4xs, GGML_IQ4_XS)
typedef void (*Iq512Fn)(IQ512_ARGS);

static inline void iq512_fns(int type, Iq512Fn *f1, Iq512Fn *f8) {
    switch (type) {
    case GGML_Q2_K: *f1 = iq512_q2k_1; *f8 = iq512_q2k_8; break;
    case GGML_IQ2_XXS: *f1 = iq512_i2xxs_1; *f8 = iq512_i2xxs_8; break;
    case GGML_IQ2_XS: *f1 = iq512_i2xs_1; *f8 = iq512_i2xs_8; break;
    case GGML_IQ2_S: *f1 = iq512_i2s_1; *f8 = iq512_i2s_8; break;
    case GGML_IQ3_XXS: *f1 = iq512_i3xxs_1; *f8 = iq512_i3xxs_8; break;
    case GGML_IQ3_S: *f1 = iq512_i3s_1; *f8 = iq512_i3s_8; break;
    case GGML_IQ1_S: *f1 = iq512_i1s_1; *f8 = iq512_i1s_8; break;
    case GGML_IQ1_M: *f1 = iq512_i1m_1; *f8 = iq512_i1m_8; break;
    default: *f1 = iq512_i4xs_1; *f8 = iq512_i4xs_8; break;
    }
}
#endif // RUNQ_VNNI

static void iq_rows(int type, float *out, size_t ldo, const int8_t *xq, size_t ldx, const float *xs, const int32_t *c16,
                    int nb, const uint8_t *kp, int k, int rb0, int rb1) {
    // out[b * ldo + i] = W row i . x row b for the rows i of the 16-row blocks rb0..rb1-1, b < nb (x row b at xq + b * ldx,
    // its scales at xs + b * ldx / GS = 32, its sums of 16 at c16 + b * k / 16; c16 is only read by the AVX-512 kernels)
    int nsb = k / QK_K;
    size_t sbb = (size_t)iq_sb_bytes(type), ldxs = ldx / 32;
#ifdef RUNQ_IQX
    Iq512Fn f1, f8;
    iq512_fns(type, &f1, &f8);
    for (int rb = rb0; rb < rb1; rb++) {
        const uint8_t *wb = kp + (size_t)rb * nsb * sbb;
        if (nb == 1) {
            f1(out + (size_t)rb * KQ_ROWS, ldo, wb, nsb, 1, xq, ldx, xs, ldxs, c16, (size_t)k / 16);
            continue;
        }
        for (int b = 0; b < nb; b += 8)
            f8(out + (size_t)b * ldo + (size_t)rb * KQ_ROWS, ldo, wb, nsb, nb - b < 8 ? nb - b : 8, xq + (size_t)b * ldx, ldx,
               xs + (size_t)b * ldxs, ldxs, c16 + (size_t)b * (k / 16), (size_t)k / 16);
    }
#else
    (void)c16;
    for (int rb = rb0; rb < rb1; rb++) {
        const uint8_t *wb = kp + (size_t)rb * nsb * sbb;
        for (int b = 0; b < nb; b += KQ_NC)
            iq_dot(type, out + (size_t)b * ldo + (size_t)rb * KQ_ROWS, ldo, wb, k, nb - b < KQ_NC ? nb - b : KQ_NC,
                   xq + (size_t)b * ldx, ldx, xs + (size_t)b * ldxs, ldxs);
    }
#endif
}

#endif // RUNQ_IQUANTS_H
