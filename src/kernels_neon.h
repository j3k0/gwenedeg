/* NEON (AArch64 + dot product: Apple M1 and later, Graviton 2+, ...) kernels for the GGUF K-quants
 * (Q4_K / Q6_K on the repacked Q4P / Q6P layouts of kquants.h): GEMV (decode) and GEMM (prefill).
 * Experimental (Step 8: Mac port). Further down: Q8_0 (GEMV, GEMM) and the attention (fp32 arithmetic
 * on the f32 / bf16 / f16 KV cache).
 *
 * The arithmetic is the definition in kquants.h's header comment, as the generic q4p_dot / q6p_dot
 * (and the AVX-512 kernels) compute it: exact int32 sums, then per (row, activation row, group of 32)
 * the float fold, every operation rounded, groups in order, so the results are bit-identical to the
 * generic build (make test: the q4_kernels group). The float code runs under RQ_FP_EXACT (no contraction of
 * the vmulq / vaddq intrinsics into FMAs, no reassociation, whatever -ffast-math says).
 *
 * Mapping of the layouts to NEON: a Q4P / Q6P 64-byte chunk holds 4 consecutive columns of each of the
 * 16 rows of a block (lane r = bytes 4 r .. 4 r + 3), i.e. 4 q registers of 4 rows ("quads"); sdot by
 * element (vdotq_laneq_s32) multiplies such a register by 4 activation bytes broadcast from a register,
 * the vpdpbusd pattern. The weights are 0..15 / 0..63 after unpacking, valid as s8: no +128 offset
 * (sdot is s8 x s8). Rows stay in lanes, so the fold has no horizontal operation.
 *
 * GEMV: one 16-row block at a time, the packed data unpacked in registers (Q4: and / shift by 4;
 * Q6: the 2 high bits shift-inserted, vsli). GEMM: as the AVX-512 one, per work item (32-row tile,
 * chunk of activation rows) the thread unpacks the tile once into its scratch (the KQS layout of
 * kernels_avx512.h: per block and group 8 steps of 64 bytes of q, step st = columns 4 st .. 4 st + 3
 * of the group, then Q4: ds, dm as fp32; Q6: sc_j0, sc_j1 as int32, d as fp32), then register tiles
 * of KQN_RQ quads x KQN_C activation rows read it with no unpacking in the inner loop.
 */
#ifndef RUNQ_KERNELS_NEON_H
#define RUNQ_KERNELS_NEON_H

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD) && !defined(RUNQ_GENERIC) && !defined(RUNQ_KQX)
#define RUNQ_NEON 1
#define RUNQ_KQX 1 // K-quant SIMD kernels: kqx_* (forward.c)
// clang's arm_neon.h defines vmulq_f32, vaddq_f32, ... as inline functions with plain C operators,
// which take the FP options of the place the header is included at, not of their callers: under
// -ffast-math they would be fused into fmla / fmls and reassociated after inlining, even inside
// RQ_FP_EXACT bodies. Included in a precise region, each one is one rounded operation
#ifdef __ARM_NEON_H
#error "kernels_neon.h must be the first to include arm_neon.h (see the comment)"
#endif
#if defined(__clang__)
#pragma float_control(precise, on, push)
#endif
#include <arm_neon.h>
#if defined(__clang__)
#pragma float_control(pop)
#endif
#include "kquants.h"

#define KQS_Q4 640 // scratch bytes per (block, group): 8 x 64 q, 64 ds, 64 dm
#define KQS_Q6 704 // 8 x 64 q, 64 sc_j0 (int32), 64 sc_j1, 64 d
// GEMM register tiles: KQN*_RQ quads (4 rows each; 1, 2 or 4) x KQN*_C activation rows (1..KQN_CMAX)
#ifndef KQN4_RQ
#define KQN4_RQ 2
#endif
#ifndef KQN4_C
#define KQN4_C 4
#endif
#ifndef KQN6_RQ
#define KQN6_RQ 2
#endif
#ifndef KQN6_C
#define KQN6_C 3
#endif
#define KQN_CMAX 6
#if KQN4_C < 1 || KQN4_C > KQN_CMAX || KQN6_C < 1 || KQN6_C > KQN_CMAX
#error "KQN4_C / KQN6_C must be in 1..KQN_CMAX"
#endif
#if (KQN4_RQ != 1 && KQN4_RQ != 2 && KQN4_RQ != 4) || (KQN6_RQ != 1 && KQN6_RQ != 2 && KQN6_RQ != 4)
#error "KQN4_RQ / KQN6_RQ must be 1, 2 or 4"
#endif
#ifndef KQN_NB
#define KQN_NB 64 // activation rows per GEMM work item
#endif

#define KQX_SCRATCH(k) ((size_t)2 * ((k) / 32) * KQS_Q6) // per-thread GEMM scratch bytes (2 blocks, the larger type)

static inline int kqx_nb(int nb, int ntile, int nt) {
    // activation rows per GEMM work item (as q8p_nb): KQN_NB, halved (down to 16) while there are
    // fewer than ~6 items per thread. (Results do not depend on the partition)
    int nbi = KQN_NB;
    while (nbi > 16 && (long)ntile * ((nb + nbi - 1) / nbi) < 6L * nt)
        nbi /= 2;
    return nbi;
}

static inline void kqx_c16(int32_t *c16, const int8_t *xq, int n) {
    // c16[i] = -32 * (sum of the 16 int8 at xq + 16 i), i < n / 16
    for (int i = 0; i < n / 16; i++)
        c16[i] = -32 * (int32_t)vaddlvq_s8(vld1q_s8(xq + 16 * i));
}

static inline void kqn_u8f(const uint8_t *p, float32x4_t f[4]) {
    // 16 u8 -> 4 x 4 fp32 (exact)
    uint8x16_t v = vld1q_u8(p);
    uint16x8_t lo = vmovl_u8(vget_low_u8(v)), hi = vmovl_high_u8(v);
    f[0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo)));
    f[1] = vcvtq_f32_u32(vmovl_high_u16(lo));
    f[2] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi)));
    f[3] = vcvtq_f32_u32(vmovl_high_u16(hi));
}

static inline void kqn_s8i(const uint8_t *p, int32x4_t s[4]) {
    // 16 s8 -> 4 x 4 int32
    int8x16_t v = vld1q_s8((const int8_t *)p);
    int16x8_t lo = vmovl_s8(vget_low_s8(v)), hi = vmovl_high_s8(v);
    s[0] = vmovl_s16(vget_low_s16(lo));
    s[1] = vmovl_high_s16(lo);
    s[2] = vmovl_s16(vget_low_s16(hi));
    s[3] = vmovl_high_s16(hi);
}

static inline void kqn_h2f(const uint8_t *p, float32x4_t f[4]) {
    // 16 fp16 -> 4 x 4 fp32 (exact, as fp16_to_f32)
    float16x8_t a = vreinterpretq_f16_u16(vld1q_u16((const uint16_t *)p));
    float16x8_t b = vreinterpretq_f16_u16(vld1q_u16((const uint16_t *)(p + 16)));
    f[0] = vcvt_f32_f16(vget_low_f16(a));
    f[1] = vcvt_high_f32_f16(a);
    f[2] = vcvt_f32_f16(vget_low_f16(b));
    f[3] = vcvt_high_f32_f16(b);
}

static inline void kqn6_unpack(uint8x16_t v0, uint8x16_t v1, uint8x16_t h, int8x16_t q[4]) {
    // one quad of a Q6P sub-block: the 4 steps t (columns 4 t .. 4 t + 3) as 0..63. Low nibbles:
    // t = 0, 1 from the ql chunk v0 (low, high nibble), t = 2, 3 from v1; high 2 bits: qh bits 2 t,
    // 2 t + 1, shift-inserted into bits 4-5
    const uint8x16_t m6 = vdupq_n_u8(0x3f);
    q[0] = vreinterpretq_s8_u8(vandq_u8(vsliq_n_u8(v0, h, 4), m6));
    q[1] = vreinterpretq_s8_u8(vandq_u8(vsliq_n_u8(vshrq_n_u8(v0, 4), vshrq_n_u8(h, 2), 4), m6));
    q[2] = vreinterpretq_s8_u8(vandq_u8(vsliq_n_u8(v1, vshrq_n_u8(h, 4), 4), m6));
    q[3] = vreinterpretq_s8_u8(vsliq_n_u8(vshrq_n_u8(v1, 4), vshrq_n_u8(h, 6), 4));
}

// ------------------------------------------------------------------------------------------- GEMV

static void kqn4_gemv_rows(float *out, const int8_t *xq, const float *xs, const float *xf, const uint8_t *wp, int k,
                           int rb0, int rb1) { RQ_FP_EXACT
    // out[i] = W[i, :] . x for the rows of the 16-row Q4P blocks rb0..rb1-1 (decode)
    int nsb = k / QK_K;
    const uint8x16_t m4 = vdupq_n_u8(0x0f);
    for (int rb = rb0; rb < rb1; rb++) {
        const uint8_t *wb = wp + (size_t)rb * nsb * Q4P_SB;
        float32x4_t f[4];
        for (int qd = 0; qd < 4; qd++)
            f[qd] = vdupq_n_f32(0.0f);
        for (int s = 0; s < nsb; s++) {
            const uint8_t *o = wb + (size_t)s * Q4P_SB;
            float32x4_t d[4], dmin[4];
            kqn_h2f(o, d);
            kqn_h2f(o + 32, dmin);
            for (int j = 0; j < 8; j++) {
                int g = 8 * s + j;
                const uint8_t *qs = o + 320 + 256 * j;
                int8x16_t x0 = vld1q_s8(xq + 32 * g), x1 = vld1q_s8(xq + 32 * g + 16);
                int32x4_t a[4], b[4];
                for (int qd = 0; qd < 4; qd++) {
                    const uint8_t *p = qs + 16 * qd;
                    uint8x16_t v0 = vld1q_u8(p), v1 = vld1q_u8(p + 64), v2 = vld1q_u8(p + 128), v3 = vld1q_u8(p + 192);
                    int32x4_t A = vdupq_n_s32(0), B = vdupq_n_s32(0);
                    A = vdotq_laneq_s32(A, vreinterpretq_s8_u8(vandq_u8(v0, m4)), x0, 0);
                    B = vdotq_laneq_s32(B, vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)), x0, 1);
                    A = vdotq_laneq_s32(A, vreinterpretq_s8_u8(vandq_u8(v1, m4)), x0, 2);
                    B = vdotq_laneq_s32(B, vreinterpretq_s8_u8(vshrq_n_u8(v1, 4)), x0, 3);
                    A = vdotq_laneq_s32(A, vreinterpretq_s8_u8(vandq_u8(v2, m4)), x1, 0);
                    B = vdotq_laneq_s32(B, vreinterpretq_s8_u8(vshrq_n_u8(v2, 4)), x1, 1);
                    A = vdotq_laneq_s32(A, vreinterpretq_s8_u8(vandq_u8(v3, m4)), x1, 2);
                    B = vdotq_laneq_s32(B, vreinterpretq_s8_u8(vshrq_n_u8(v3, 4)), x1, 3);
                    a[qd] = A;
                    b[qd] = B;
                }
                float32x4_t sc[4], mn[4];
                kqn_u8f(o + 64 + 16 * j, sc);
                kqn_u8f(o + 192 + 16 * j, mn);
                for (int qd = 0; qd < 4; qd++) {
                    float32x4_t ds = vmulq_f32(d[qd], sc[qd]), dm = vmulq_f32(dmin[qd], mn[qd]);
                    float32x4_t A = vmulq_f32(vmulq_n_f32(ds, xs[g]), vcvtq_f32_s32(vaddq_s32(a[qd], b[qd])));
                    float32x4_t B = vmulq_n_f32(dm, xf[g]);
                    f[qd] = vaddq_f32(f[qd], vsubq_f32(A, B));
                }
            }
        }
        for (int qd = 0; qd < 4; qd++)
            vst1q_f32(out + (size_t)rb * KQ_ROWS + 4 * qd, f[qd]);
    }
}

static void kqn6_gemv_rows(float *out, const int8_t *xq, const float *xs, const int32_t *c16, const uint8_t *wp, int k,
                           int rb0, int rb1) { RQ_FP_EXACT
    // the same for Q6P blocks (c16: kqx_c16 of xq)
    int nsb = k / QK_K;
    for (int rb = rb0; rb < rb1; rb++) {
        const uint8_t *wb = wp + (size_t)rb * nsb * Q6P_SB;
        float32x4_t f[4];
        for (int qd = 0; qd < 4; qd++)
            f[qd] = vdupq_n_f32(0.0f);
        for (int s = 0; s < nsb; s++) {
            const uint8_t *o = wb + (size_t)s * Q6P_SB;
            float32x4_t d[4];
            for (int qd = 0; qd < 4; qd++)
                d[qd] = vld1q_f32((const float *)o + 4 * qd);
            for (int g8 = 0; g8 < 8; g8++) {
                int g = 8 * s + g8;
                int32x4_t I[4];
                for (int h = 0; h < 2; h++) {
                    int j = 2 * g8 + h;
                    const uint8_t *ql = o + 320 + 192 * j;
                    int8x16_t xr = vld1q_s8(xq + 32 * g + 16 * h);
                    int32x4_t cv = vdupq_n_s32(c16[2 * g + h]), sc[4];
                    kqn_s8i(o + 64 + 16 * j, sc);
                    for (int qd = 0; qd < 4; qd++) {
                        int8x16_t q[4];
                        kqn6_unpack(vld1q_u8(ql + 16 * qd), vld1q_u8(ql + 64 + 16 * qd), vld1q_u8(ql + 128 + 16 * qd), q);
                        int32x4_t t0 = cv, t1 = vdupq_n_s32(0);
                        t0 = vdotq_laneq_s32(t0, q[0], xr, 0);
                        t1 = vdotq_laneq_s32(t1, q[1], xr, 1);
                        t0 = vdotq_laneq_s32(t0, q[2], xr, 2);
                        t1 = vdotq_laneq_s32(t1, q[3], xr, 3);
                        int32x4_t t = vaddq_s32(t0, t1);
                        I[qd] = h ? vmlaq_s32(I[qd], t, sc[qd]) : vmulq_s32(t, sc[qd]);
                    }
                }
                for (int qd = 0; qd < 4; qd++)
                    f[qd] = vaddq_f32(f[qd], vmulq_f32(vmulq_n_f32(d[qd], xs[g]), vcvtq_f32_s32(I[qd])));
            }
        }
        for (int qd = 0; qd < 4; qd++)
            vst1q_f32(out + (size_t)rb * KQ_ROWS + 4 * qd, f[qd]);
    }
}

static void kqx_gemv_rows(int type, float *out, const int8_t *xq, const float *xs, const float *xf, const int32_t *c16,
                          const uint8_t *wp, int k, int rb0, int rb1) {
    if (type == GGML_Q4_K)
        kqn4_gemv_rows(out, xq, xs, xf, wp, k, rb0, rb1);
    else
        kqn6_gemv_rows(out, xq, xs, c16, wp, k, rb0, rb1);
}

// ------------------------------------------------------------------------------------------- GEMM

static void kqn_unpack_tile(int type, uint8_t *scr, const uint8_t *wp, int nblk, int k) { RQ_FP_EXACT
    // nblk (<= 2) consecutive Q4P / Q6P blocks at wp -> the KQS scratch layout (block r at
    // scr + r * (k / 32) * KQS_Q4 / KQS_Q6)
    int nsb = k / QK_K, ng = k / 32;
    const uint8x16_t m4 = vdupq_n_u8(0x0f);
    for (int r = 0; r < nblk; r++)
        for (int s = 0; s < nsb; s++) {
            if (type == GGML_Q4_K) {
                const uint8_t *o = wp + ((size_t)r * nsb + s) * Q4P_SB;
                float32x4_t d[4], dmin[4];
                kqn_h2f(o, d);
                kqn_h2f(o + 32, dmin);
                for (int j = 0; j < 8; j++) {
                    uint8_t *dst = scr + ((size_t)r * ng + 8 * s + j) * KQS_Q4;
                    for (int c = 0; c < 4; c++)
                        for (int qd = 0; qd < 4; qd++) {
                            uint8x16_t v = vld1q_u8(o + 320 + 256 * j + 64 * c + 16 * qd);
                            vst1q_u8(dst + 128 * c + 16 * qd, vandq_u8(v, m4));
                            vst1q_u8(dst + 128 * c + 64 + 16 * qd, vshrq_n_u8(v, 4));
                        }
                    float32x4_t sc[4], mn[4];
                    kqn_u8f(o + 64 + 16 * j, sc);
                    kqn_u8f(o + 192 + 16 * j, mn);
                    for (int qd = 0; qd < 4; qd++) {
                        vst1q_f32((float *)(dst + 512) + 4 * qd, vmulq_f32(d[qd], sc[qd]));
                        vst1q_f32((float *)(dst + 576) + 4 * qd, vmulq_f32(dmin[qd], mn[qd]));
                    }
                }
            } else {
                const uint8_t *o = wp + ((size_t)r * nsb + s) * Q6P_SB;
                for (int g8 = 0; g8 < 8; g8++) {
                    uint8_t *dst = scr + ((size_t)r * ng + 8 * s + g8) * KQS_Q6;
                    for (int h = 0; h < 2; h++) {
                        int j = 2 * g8 + h;
                        const uint8_t *ql = o + 320 + 192 * j;
                        for (int qd = 0; qd < 4; qd++) {
                            int8x16_t q[4];
                            kqn6_unpack(vld1q_u8(ql + 16 * qd), vld1q_u8(ql + 64 + 16 * qd), vld1q_u8(ql + 128 + 16 * qd), q);
                            for (int t = 0; t < 4; t++)
                                vst1q_s8((int8_t *)(dst + 256 * h + 64 * t + 16 * qd), q[t]);
                        }
                        int32x4_t sc[4];
                        kqn_s8i(o + 64 + 16 * j, sc);
                        for (int qd = 0; qd < 4; qd++)
                            vst1q_s32((int32_t *)(dst + 512 + 64 * h) + 4 * qd, sc[qd]);
                    }
                    memcpy(dst + 640, o, 64); // d[16] (fp32)
                }
            }
        }
}

// one step st (0..7: columns 4 st .. 4 st + 3 of the group, the activation bytes = 32-bit element st
// of the group's 32 bytes: register XR = xa (st < 4) or xb, lane L = st % 4) of a register tile
#define KQN_STEP(ACC, st, XR, L) do { \
        int8x16_t w_[4]; \
        for (int q = 0; q < RQ; q++) \
            w_[q] = vld1q_s8((const int8_t *)(wg + (st) * 64 + 16 * q)); \
        for (int c = 0; c < C; c++) \
            for (int q = 0; q < RQ; q++) \
                ACC[q][c] = vdotq_laneq_s32(ACC[q][c], w_[q], XR[c], L); \
    } while (0)

static inline __attribute__((always_inline))
void kqn4_tile(const int RQ, const int C, const uint8_t *ws, int ng, const int8_t *x, size_t ldx, const float *xs,
               size_t ldxs, const float *xf, size_t ldxf, float *out, size_t ldo) { RQ_FP_EXACT
    // out[c * ldo + 4 q + lane] for RQ quads of a Q4 scratch block (ws: the block's group 0, plus
    // 16 * first quad), c < C activation rows (x + c * ldx, scales xs + c * ldxs, sums xf + c * ldxf)
    float32x4_t f[4][KQN_CMAX];
    for (int q = 0; q < RQ; q++)
        for (int c = 0; c < C; c++)
            f[q][c] = vdupq_n_f32(0.0f);
    for (int g = 0; g < ng; g++) {
        const uint8_t *wg = ws + (size_t)g * KQS_Q4;
        int8x16_t xa[KQN_CMAX], xb[KQN_CMAX];
        for (int c = 0; c < C; c++) {
            xa[c] = vld1q_s8(x + c * ldx + 32 * g);
            xb[c] = vld1q_s8(x + c * ldx + 32 * g + 16);
        }
        int32x4_t acc[4][KQN_CMAX];
        for (int q = 0; q < RQ; q++)
            for (int c = 0; c < C; c++)
                acc[q][c] = vdupq_n_s32(0);
        KQN_STEP(acc, 0, xa, 0); KQN_STEP(acc, 1, xa, 1); KQN_STEP(acc, 2, xa, 2); KQN_STEP(acc, 3, xa, 3);
        KQN_STEP(acc, 4, xb, 0); KQN_STEP(acc, 5, xb, 1); KQN_STEP(acc, 6, xb, 2); KQN_STEP(acc, 7, xb, 3);
        for (int q = 0; q < RQ; q++) {
            float32x4_t ds = vld1q_f32((const float *)(wg + 512) + 4 * q), dm = vld1q_f32((const float *)(wg + 576) + 4 * q);
            for (int c = 0; c < C; c++) {
                float32x4_t A = vmulq_f32(vmulq_n_f32(ds, xs[c * ldxs + g]), vcvtq_f32_s32(acc[q][c]));
                float32x4_t B = vmulq_n_f32(dm, xf[c * ldxf + g]);
                f[q][c] = vaddq_f32(f[q][c], vsubq_f32(A, B));
            }
        }
    }
    for (int q = 0; q < RQ; q++)
        for (int c = 0; c < C; c++)
            vst1q_f32(out + c * ldo + 4 * q, f[q][c]);
}

static inline __attribute__((always_inline))
void kqn6_tile(const int RQ, const int C, const uint8_t *ws, int ng, const int8_t *x, size_t ldx, const float *xs,
               size_t ldxs, const int32_t *c16, size_t ldc, float *out, size_t ldo) { RQ_FP_EXACT
    // the same for a Q6 scratch block (c16 + c * ldc: row c's -32 * sums of 16)
    float32x4_t f[4][KQN_CMAX];
    for (int q = 0; q < RQ; q++)
        for (int c = 0; c < C; c++)
            f[q][c] = vdupq_n_f32(0.0f);
    for (int g = 0; g < ng; g++) {
        const uint8_t *wg = ws + (size_t)g * KQS_Q6;
        int8x16_t xa[KQN_CMAX], xb[KQN_CMAX];
        int32x4_t ta[4][KQN_CMAX], tb[4][KQN_CMAX];
        for (int c = 0; c < C; c++) {
            xa[c] = vld1q_s8(x + c * ldx + 32 * g);
            xb[c] = vld1q_s8(x + c * ldx + 32 * g + 16);
            int32x4_t c0 = vdupq_n_s32(c16[c * ldc + 2 * g]), c1 = vdupq_n_s32(c16[c * ldc + 2 * g + 1]);
            for (int q = 0; q < RQ; q++) {
                ta[q][c] = c0;
                tb[q][c] = c1;
            }
        }
        KQN_STEP(ta, 0, xa, 0); KQN_STEP(ta, 1, xa, 1); KQN_STEP(ta, 2, xa, 2); KQN_STEP(ta, 3, xa, 3);
        KQN_STEP(tb, 4, xb, 0); KQN_STEP(tb, 5, xb, 1); KQN_STEP(tb, 6, xb, 2); KQN_STEP(tb, 7, xb, 3);
        for (int q = 0; q < RQ; q++) {
            int32x4_t s0 = vld1q_s32((const int32_t *)(wg + 512) + 4 * q), s1 = vld1q_s32((const int32_t *)(wg + 576) + 4 * q);
            float32x4_t d = vld1q_f32((const float *)(wg + 640) + 4 * q);
            for (int c = 0; c < C; c++) {
                int32x4_t I = vmlaq_s32(vmulq_s32(ta[q][c], s0), tb[q][c], s1);
                f[q][c] = vaddq_f32(f[q][c], vmulq_f32(vmulq_n_f32(d, xs[c * ldxs + g]), vcvtq_f32_s32(I)));
            }
        }
    }
    for (int q = 0; q < RQ; q++)
        for (int c = 0; c < C; c++)
            vst1q_f32(out + c * ldo + 4 * q, f[q][c]);
}
#undef KQN_STEP

static void kqn_tile_any(int type, int c, const uint8_t *ws, int ng, const int8_t *x, size_t ldx, const float *xs,
                         size_t ldxs, const float *xf, size_t ldxf, const int32_t *c16, size_t ldc, float *out, size_t ldo) {
    // dispatch to the instance with compile-time C (1..KQN4_C / KQN6_C); RQ = KQN4_RQ / KQN6_RQ quads
    #define KQN_CASE(CC) \
        case CC: \
            if (type == GGML_Q4_K) { if ((CC) <= KQN4_C) kqn4_tile(KQN4_RQ, (CC) <= KQN4_C ? (CC) : 1, ws, ng, x, ldx, xs, ldxs, xf, ldxf, out, ldo); } \
            else if ((CC) <= KQN6_C) kqn6_tile(KQN6_RQ, (CC) <= KQN6_C ? (CC) : 1, ws, ng, x, ldx, xs, ldxs, c16, ldc, out, ldo); \
            break;
    switch (c) {
        KQN_CASE(1) KQN_CASE(2) KQN_CASE(3) KQN_CASE(4) KQN_CASE(5) KQN_CASE(6)
        default: break;
    }
    #undef KQN_CASE
}

static void kqx_gemm_item(float *out, int ldo, const uint8_t *kp, int type, int d, int t, int ch, int nbi,
                          const int8_t *xq, const float *xs, int ldx, int nb, int k, const float *xf,
                          const int32_t *c16, uint8_t *scr, const void **last) {
    // one K-quant GEMM work item (as kq512_gemm_item): 32-row tile t of W (Q4P / Q6P), the activation
    // rows of chunk ch (xf + b * k / 32: kq_act_prep, c16 + b * k / 16: kqx_c16 of each row). The
    // thread's scratch scr (KQX_SCRATCH(k) bytes) holds the tile unpacked; *last = the tile it holds
    int ng = k / 32, n16 = k / 16, sbb = kq_sb_bytes(type);
    size_t ldxs = (size_t)ldx / 32;
    int nblk = d / KQ_ROWS, rb0 = t * 2;
    int rb = nblk - rb0 < 2 ? nblk - rb0 : 2;
    size_t bstride = (size_t)ng * (type == GGML_Q4_K ? KQS_Q4 : KQS_Q6);
    const uint8_t *wt = kp + (size_t)rb0 * (k / QK_K) * sbb;
    if (*last != wt) {
        kqn_unpack_tile(type, scr, wt, rb, k);
        *last = wt;
    }
    int b1 = (ch + 1) * nbi < nb ? (ch + 1) * nbi : nb;
    int C = type == GGML_Q4_K ? KQN4_C : KQN6_C, RQ = type == GGML_Q4_K ? KQN4_RQ : KQN6_RQ;
    for (int b = ch * nbi; b < b1; b += C) {
        int c = b1 - b < C ? b1 - b : C;
        for (int Q = 0; Q < 4 * rb; Q += RQ) // quads of the tile (RQ divides 4: within one block)
            kqn_tile_any(type, c, scr + (size_t)(Q / 4) * bstride + 16 * (Q % 4), ng, xq + (size_t)b * ldx, ldx,
                         xs + (size_t)b * ldxs, ldxs, xf + (size_t)b * ng, ng, c16 + (size_t)b * n16, n16,
                         out + (size_t)b * ldo + (size_t)rb0 * KQ_ROWS + 4 * Q, ldo);
    }
}

/* ---------------------------------------------------------------------------------------------
 * Q8_0 (the .bin's group-64 Q8 and GGUF Q8_0, group 32): GEMV (decode) and GEMM (prefill) on weights
 * repacked at load. Layout "Q8N" = kernels_avx512.h's Q8P without the +128 offset (sdot is s8 x s8,
 * so the weights stay signed and there is no activation correction): 16-row blocks, per (block,
 * group) gs / 4 steps of 64 bytes (lane r = bytes 4 r .. 4 r + 3 = 4 consecutive columns of row r of
 * the block), then the 16 rows' fp32 scales. A quad (4 rows) of a step = 16 bytes at 16 q.
 *
 * Arithmetic: dot_q8 / dot_q8_x4 in forward.c as written, per output and group in order:
 *   val = val + ((float) isum * ws) * xs     (each operation rounded; isum exact in int32)
 * so the results are bit-identical to the generic build on the same compiler (make test: the kernels group).
 * (gcc -Ofast on x86 reassociates that expression to (ws * xs) * isum, which the AVX-512 Q8P kernels
 * follow: arm64 and x86 Q8_0 logits differ in the last bits, as they already did with the generic
 * kernels.)
 */
#define RUNQ_Q8X 1 // Q8_0 SIMD kernels: q8x_* (forward.c)
#define Q8N_ROWS 16
#define Q8X_GROUP(gs) (Q8N_ROWS * (gs) + Q8N_ROWS * 4) // bytes per (block, group): 16 x gs + 64
// GEMM register tile: Q8N_RQ quads (1, 2 or 4) x Q8N_C activation rows (1..Q8N_CMAX)
#ifndef Q8N_RQ
#define Q8N_RQ 4
#endif
#ifndef Q8N_C
#define Q8N_C 3
#endif
#define Q8N_CMAX 6
#if Q8N_C < 1 || Q8N_C > Q8N_CMAX || (Q8N_RQ != 1 && Q8N_RQ != 2 && Q8N_RQ != 4)
#error "Q8N_C must be in 1..Q8N_CMAX, Q8N_RQ 1, 2 or 4"
#endif

static inline size_t q8x_size(int d, int k, int gs) { return (size_t)(d / Q8N_ROWS) * (k / gs) * Q8X_GROUP(gs); }

static inline int q8x_ok(int d, int k, int gs) {
    return (gs == 32 || gs == 64) && d > 0 && k > 0 && d % Q8N_ROWS == 0 && k % gs == 0;
}

static void q8x_pack_blocks(uint8_t *pk, const int8_t *q, const float *s, int nblk, int k, int gs) {
    // nblk blocks of 16 rows of a Q8_0 matrix with k columns: (q, s) = their rows, row-major, k / gs
    // scales per row -> Q8N layout at pk. Serial (the caller splits the work)
    int ng = k / gs;
    for (int rb = 0; rb < nblk; rb++)
        for (int g = 0; g < ng; g++) {
            uint8_t *o = pk + ((size_t)rb * ng + g) * Q8X_GROUP(gs);
            for (int r = 0; r < Q8N_ROWS; r++) {
                const int8_t *src = q + (size_t)(rb * Q8N_ROWS + r) * k + g * gs;
                for (int st = 0; st < gs / 4; st++)
                    memcpy(o + st * 64 + r * 4, src + 4 * st, 4);
                memcpy(o + Q8N_ROWS * gs + r * 4, s + (size_t)(rb * Q8N_ROWS + r) * ng + g, 4);
            }
        }
}

static inline void q8x_gemv_corr(int32_t *corr, const int8_t *xq, int k, int gs) {
    (void)corr; (void)xq; (void)k; (void)gs; // (no weight offset: nothing to correct)
}

static inline __attribute__((always_inline))
void q8n_gemv_blocks(const int GSZ, float *out, const int8_t *xq, const float *xs, const uint8_t *wp, int k,
                     int rb0, int rb1) { RQ_FP_EXACT
    int ng = k / GSZ;
    size_t bstride = (size_t)ng * Q8X_GROUP(GSZ);
    for (int rb = rb0; rb < rb1; rb++) {
        const uint8_t *wb = wp + (size_t)rb * bstride;
        float32x4_t f[4];
        for (int qd = 0; qd < 4; qd++)
            f[qd] = vdupq_n_f32(0.0f);
        for (int g = 0; g < ng; g++) {
            const uint8_t *wg = wb + (size_t)g * Q8X_GROUP(GSZ);
            const int8_t *xg = xq + g * GSZ;
            int32x4_t a[4], b[4]; // two chains per quad (even / odd steps)
            for (int qd = 0; qd < 4; qd++)
                a[qd] = b[qd] = vdupq_n_s32(0);
            for (int ch = 0; ch < GSZ / 16; ch++) { // 16 activation bytes = 4 steps
                int8x16_t xv = vld1q_s8(xg + 16 * ch);
                const uint8_t *w = wg + ch * 256;
                for (int qd = 0; qd < 4; qd++) {
                    a[qd] = vdotq_laneq_s32(a[qd], vld1q_s8((const int8_t *)(w + 16 * qd)), xv, 0);
                    b[qd] = vdotq_laneq_s32(b[qd], vld1q_s8((const int8_t *)(w + 64 + 16 * qd)), xv, 1);
                    a[qd] = vdotq_laneq_s32(a[qd], vld1q_s8((const int8_t *)(w + 128 + 16 * qd)), xv, 2);
                    b[qd] = vdotq_laneq_s32(b[qd], vld1q_s8((const int8_t *)(w + 192 + 16 * qd)), xv, 3);
                }
            }
            const float *ws = (const float *)(wg + Q8N_ROWS * GSZ);
            float xsg = xs[g];
            for (int qd = 0; qd < 4; qd++) {
                float32x4_t p = vmulq_f32(vcvtq_f32_s32(vaddq_s32(a[qd], b[qd])), vld1q_f32(ws + 4 * qd));
                f[qd] = vaddq_f32(f[qd], vmulq_n_f32(p, xsg));
            }
        }
        for (int qd = 0; qd < 4; qd++)
            vst1q_f32(out + (size_t)rb * Q8N_ROWS + 4 * qd, f[qd]);
    }
}

static void q8x_gemv_rows(float *out, const int8_t *xq, const float *xs, const int32_t *corr, const uint8_t *wp,
                          int k, int gs, int rb0, int rb1) {
    // out[i] = W[i, :] . x for the rows i of the 16-row blocks rb0..rb1-1 (decode)
    (void)corr;
    if (gs == 64)
        q8n_gemv_blocks(64, out, xq, xs, wp, k, rb0, rb1);
    else
        q8n_gemv_blocks(32, out, xq, xs, wp, k, rb0, rb1);
}

static inline __attribute__((always_inline))
void q8n_tile(const int RQ, const int C, const int GSZ, const uint8_t *wq, int ng, const int8_t *x, size_t ldx,
              const float *xs, size_t ldxs, float *out, size_t ldo) { RQ_FP_EXACT
    // out[c * ldo + 4 q + lane] for RQ quads starting at wq (a block's group 0 + 16 * first quad: the
    // scales of quad q are then at the same 16-byte offset in the scale area), c < C activation rows
    // (x + c * ldx, scales xs + c * ldxs), over all ng groups of GSZ
    float32x4_t f[4][Q8N_CMAX];
    for (int q = 0; q < RQ; q++)
        for (int c = 0; c < C; c++)
            f[q][c] = vdupq_n_f32(0.0f);
    for (int g = 0; g < ng; g++) {
        const uint8_t *wg = wq + (size_t)g * Q8X_GROUP(GSZ);
        const int8_t *xg = x + (size_t)g * GSZ;
        int32x4_t acc[4][Q8N_CMAX];
        for (int q = 0; q < RQ; q++)
            for (int c = 0; c < C; c++)
                acc[q][c] = vdupq_n_s32(0);
        for (int ch = 0; ch < GSZ / 16; ch++) {
            int8x16_t xv[Q8N_CMAX];
            for (int c = 0; c < C; c++)
                xv[c] = vld1q_s8(xg + c * ldx + 16 * ch);
            const uint8_t *w = wg + ch * 256;
            #define Q8N_STEP(L) do { \
                    int8x16_t w_[4]; \
                    for (int q = 0; q < RQ; q++) \
                        w_[q] = vld1q_s8((const int8_t *)(w + (L) * 64 + 16 * q)); \
                    for (int c = 0; c < C; c++) \
                        for (int q = 0; q < RQ; q++) \
                            acc[q][c] = vdotq_laneq_s32(acc[q][c], w_[q], xv[c], L); \
                } while (0)
            Q8N_STEP(0); Q8N_STEP(1); Q8N_STEP(2); Q8N_STEP(3);
            #undef Q8N_STEP
        }
        const float *ws = (const float *)(wg + Q8N_ROWS * GSZ);
        for (int q = 0; q < RQ; q++) {
            float32x4_t wsq = vld1q_f32(ws + 4 * q);
            for (int c = 0; c < C; c++) {
                float32x4_t p = vmulq_f32(vcvtq_f32_s32(acc[q][c]), wsq);
                f[q][c] = vaddq_f32(f[q][c], vmulq_n_f32(p, xs[c * ldxs + g]));
            }
        }
    }
    for (int q = 0; q < RQ; q++)
        for (int c = 0; c < C; c++)
            vst1q_f32(out + c * ldo + 4 * q, f[q][c]);
}

static void q8n_tile_any(int gs, int c, const uint8_t *wq, int ng, const int8_t *x, size_t ldx, const float *xs,
                         size_t ldxs, float *out, size_t ldo) {
    // dispatch to the instance with compile-time group size and C (1..Q8N_C); RQ = Q8N_RQ quads
    #define Q8N_CASE(G, CC) case (G) * 16 + (CC): \
        if ((CC) <= Q8N_C) q8n_tile(Q8N_RQ, (CC) <= Q8N_C ? (CC) : 1, G, wq, ng, x, ldx, xs, ldxs, out, ldo); break;
    switch (gs * 16 + c) {
        Q8N_CASE(64, 1) Q8N_CASE(64, 2) Q8N_CASE(64, 3) Q8N_CASE(64, 4) Q8N_CASE(64, 5) Q8N_CASE(64, 6)
        Q8N_CASE(32, 1) Q8N_CASE(32, 2) Q8N_CASE(32, 3) Q8N_CASE(32, 4) Q8N_CASE(32, 5) Q8N_CASE(32, 6)
        default: break;
    }
    #undef Q8N_CASE
}

static void q8x_gemm_item(float *out, int ldo, const uint8_t *wp, int d, int t, int ch, int nbi, const int8_t *xq,
                          const float *xs, int ldx, int nb, int k, int gs, const int32_t *corr) {
    // one GEMM work item (as q8p_gemm_item): out[b * ldo + i] = W[i, :] . x[b, :] for the rows i of
    // 32-row tile t of W (d x k, Q8N, group size gs) and the activation rows b of chunk ch (nbi rows;
    // x at xq + b * ldx, scales xs + b * ldx / gs). The weight tile (32 x k bytes + scales) is reused
    // from L1 / L2 for every C activation rows
    (void)corr;
    int ng = k / gs;
    size_t bstride = (size_t)ng * Q8X_GROUP(gs), ldxs = (size_t)ldx / gs;
    int nblk = d / Q8N_ROWS, rb0 = t * 2;
    int rb = nblk - rb0 < 2 ? nblk - rb0 : 2;
    int b1 = (ch + 1) * nbi < nb ? (ch + 1) * nbi : nb;
    const uint8_t *wt = wp + (size_t)rb0 * bstride;
    for (int b = ch * nbi; b < b1; b += Q8N_C) {
        int c = b1 - b < Q8N_C ? b1 - b : Q8N_C;
        for (int Q = 0; Q < 4 * rb; Q += Q8N_RQ) // quads of the tile (Q8N_RQ divides 4: within one block)
            q8n_tile_any(gs, c, wt + (size_t)(Q / 4) * bstride + 16 * (Q % 4), ng, xq + (size_t)b * ldx, ldx,
                         xs + (size_t)b * ldxs, ldxs, out + (size_t)b * ldo + (size_t)rb0 * Q8N_ROWS + 4 * Q, ldo);
    }
}

/* ---------------------------------------------------------------------------------------------
 * fp32 attention (NEON), the definition in kernels_avx512.h (scores in fma chains of ATT_DC dims, the
 * chunks added in order, times scale; per block of ATT_P positions max, exp_att, 16 lane sums in
 * tile order then the fixed tree j + 8, + 4, + 2, + 1; P.V as one fma chain per output from 0), as
 * att512_group computes it with 4-lane vectors: a 16-position tile = 4 registers. Bit-identical to
 * attention_vec (forward.c) and att512_kv. Cache layout (kernels_avx512.h): per kv head and block of
 * ATT_P positions ATT_PT contiguous tiles (kt / vt: tile 0 of block 0, block b at + b * bs elements):
 * K transposed in a tile (dim i, position j at i * 16 + j), V rows contiguous in a block. All three
 * cache types (-k f32 / bf16 / f16): the 16-bit elements are converted to fp32 in registers (exact:
 * bf16 by a shift / a mask of the dword holding a dim pair, f16 by fcvtl), then the same arithmetic,
 * so each type gives the generic build's results bit for bit.
 * Scores: RV (<= 4) query vectors x 1 or 2 tiles per register tile, q transposed per group of 4
 * vectors (fmla by element); P.V: RV vectors x DW registers of dims per pass over the positions.
 */
#include <float.h>
#include "kernels_avx512.h" // the attention constants (ATT_T, ATT_P, ATT_DC, exp_att's; no kernels on AArch64)
#define ATTN_NV 12    // max query vectors per group
#define ATTN_HDMAX 256

static inline float32x4_t attn_exp(float32x4_t x) { RQ_FP_EXACT
    // exp_att on 4 lanes (x <= 0), operation for operation
    x = vmaxq_f32(x, vdupq_n_f32(ATT_EXP_MIN));
    float32x4_t n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(ATT_LOG2E)));
    float32x4_t r = vfmaq_f32(x, n, vdupq_n_f32(-ATT_LN2_HI));
    r = vfmaq_f32(r, n, vdupq_n_f32(-ATT_LN2_LO));
    float32x4_t z = vmulq_f32(r, r);
    float32x4_t y = vdupq_n_f32(ATT_P0);
    y = vfmaq_f32(vdupq_n_f32(ATT_P1), y, r);
    y = vfmaq_f32(vdupq_n_f32(ATT_P2), y, r);
    y = vfmaq_f32(vdupq_n_f32(ATT_P3), y, r);
    y = vfmaq_f32(vdupq_n_f32(ATT_P4), y, r);
    y = vfmaq_f32(vdupq_n_f32(ATT_P5), y, r);
    y = vfmaq_f32(r, y, z);
    y = vaddq_f32(y, vdupq_n_f32(1.0f));
    int32x4_t e = vshlq_n_s32(vcvtq_s32_f32(n), 23);
    return vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(y), e));
}

// acc[r][u] += k[u] * lane r of qv, r < RV (lane: a constant)
#define ATTN_FMA_LANES(ACC, U, K, QV) do { \
        ACC[0][U] = vfmaq_laneq_f32(ACC[0][U], K, QV, 0); \
        if (RV > 1) ACC[1][U] = vfmaq_laneq_f32(ACC[1][U], K, QV, 1); \
        if (RV > 2) ACC[2][U] = vfmaq_laneq_f32(ACC[2][U], K, QV, 2); \
        if (RV > 3) ACC[3][U] = vfmaq_laneq_f32(ACC[3][U], K, QV, 3); \
    } while (0)

static inline float32x4_t attn_bf_lo(uint16x8_t w) {
    // the bf16 values in the low halves of 4 dwords, as fp32 (exact: the bits << 16)
    return vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(w), 16));
}

static inline float32x4_t attn_bf_hi(uint16x8_t w) {
    // the bf16 values in the high halves
    return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_u16(w), vdupq_n_u32(0xffff0000u)));
}

static inline __attribute__((always_inline))
void attn_qk(const int RV, const int NT, const int KT, const float *q4, const void *kt, size_t ts, int hd, float scale,
             float *s, size_t st) { RQ_FP_EXACT
    // scores of RV vectors (q4: their q transposed, dim i at q4 + 4 i, lane r) for NT tiles (tile u at
    // kt + u * ts elements) -> s + u * st + r * 16 + j. KT (cache type) 0: fp32; 1: bf16, dims i, i + 1
    // (i even) of 4 positions in one 16-byte load (the low / high halves of its dwords); 2: f16, 8
    // positions per load (fcvtl / fcvtl2). Converted exactly; per accumulator the dims in order, the same
    // fma chains as for fp32
    const float *kf = (const float *)kt;
    const uint16_t *kh = (const uint16_t *)kt;
    for (int i0 = 0; i0 < hd; i0 += ATT_DC) {
        float32x4_t acc[4][8];
        for (int r = 0; r < RV; r++)
            for (int u = 0; u < 4 * NT; u++)
                acc[r][u] = vdupq_n_f32(0.0f);
        if (KT == 1) {
            for (int i = i0; i < i0 + ATT_DC; i += 2) {
                float32x4_t q0 = vld1q_f32(q4 + 4 * i), q1 = vld1q_f32(q4 + 4 * i + 4);
                for (int u = 0; u < 4 * NT; u++) {
                    uint16x8_t w = vld1q_u16(kh + (u >> 2) * ts + (size_t)i * ATT_T + 8 * (u & 3));
                    float32x4_t k0 = attn_bf_lo(w), k1 = attn_bf_hi(w);
                    ATTN_FMA_LANES(acc, u, k0, q0);
                    ATTN_FMA_LANES(acc, u, k1, q1);
                }
            }
        } else if (KT == 2) {
            for (int i = i0; i < i0 + ATT_DC; i++) {
                float32x4_t qv = vld1q_f32(q4 + 4 * i);
                for (int u = 0; u < 4 * NT; u += 2) {
                    float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(kh + (u >> 2) * ts + (size_t)i * ATT_T + 4 * (u & 3)));
                    float32x4_t k0 = vcvt_f32_f16(vget_low_f16(h)), k1 = vcvt_high_f32_f16(h);
                    ATTN_FMA_LANES(acc, u, k0, qv);
                    ATTN_FMA_LANES(acc, u + 1, k1, qv);
                }
            }
        } else {
            for (int i = i0; i < i0 + ATT_DC; i++) {
                float32x4_t qv = vld1q_f32(q4 + 4 * i);
                for (int u = 0; u < 4 * NT; u++) {
                    float32x4_t k = vld1q_f32(kf + (u >> 2) * ts + (size_t)i * ATT_T + 4 * (u & 3));
                    ATTN_FMA_LANES(acc, u, k, qv);
                }
            }
        }
        for (int r = 0; r < RV; r++)
            for (int u = 0; u < 4 * NT; u++) {
                float *o = s + (u >> 2) * st + r * ATT_T + 4 * (u & 3);
                float32x4_t a = acc[r][u];
                if (i0 > 0)
                    a = vaddq_f32(vld1q_f32(o), a);
                if (i0 + ATT_DC == hd)
                    a = vmulq_n_f32(a, scale);
                vst1q_f32(o, a);
            }
    }
}
#undef ATTN_FMA_LANES

static inline int attn_vdim(const int KT, int d0, int u) {
    // the first of the 4 dims held by V register u of the pass at d0. KT 1 (bf16): registers 2 k, 2 k + 1
    // = the low / high halves of dwords w .. w + 3 of the row, w = d0 / 2 + 4 k: dims 32 (w / 16) + w % 16
    // and + 16 (the layout: dims 32 c + l, 32 c + 16 + l share dword 16 c + l)
    if (KT != 1)
        return d0 + 4 * u;
    int w = d0 / 2 + 4 * (u >> 1);
    return 32 * (w >> 4) + (w & 15) + 16 * (u & 1);
}

// one position t of attn_pv: acc[r][u] += e[r][t] * (V row t, register u of the pass), for the vectors r
// with COND (16-bit: 2 registers per 16-byte load, converted exactly)
#define ATTN_PV_STEP(COND) do { \
        const float *p_ = s + (size_t)(t >> 4) * st + (t & 15); \
        size_t o_ = vo + (size_t)(t - tb) * hd + d0; \
        if (KT == 0) { \
            for (int u = 0; u < DW; u++) { \
                float32x4_t vv = vld1q_f32((const float *)vt + o_ + 4 * u); \
                for (int r = 0; r < RV; r++) \
                    if (COND) \
                        acc[r][u] = vfmaq_n_f32(acc[r][u], vv, p_[r * ATT_T]); \
            } \
        } else { \
            for (int k = 0; k < DW / 2; k++) { \
                uint16x8_t w_ = vld1q_u16((const uint16_t *)vt + o_ + 8 * k); \
                float32x4_t v0 = KT == 1 ? attn_bf_lo(w_) : vcvt_f32_f16(vget_low_f16(vreinterpretq_f16_u16(w_))); \
                float32x4_t v1 = KT == 1 ? attn_bf_hi(w_) : vcvt_high_f32_f16(vreinterpretq_f16_u16(w_)); \
                for (int r = 0; r < RV; r++) \
                    if (COND) { \
                        acc[r][2 * k] = vfmaq_n_f32(acc[r][2 * k], v0, p_[r * ATT_T]); \
                        acc[r][2 * k + 1] = vfmaq_n_f32(acc[r][2 * k + 1], v1, p_[r * ATT_T]); \
                    } \
            } \
        } \
    } while (0)

static inline __attribute__((always_inline))
void attn_pv(const int RV, const int DW, const int KT, const float *s, size_t st, int nbk, const int *np, const void *vt,
             size_t bs, int hd, const float *inv, float *const *out, int ob, size_t pst) { RQ_FP_EXACT
    // as att512_pv for RV vectors (scores s + r * 16, tile stride st): inv != NULL (nbk = 1): out[r][i]
    // = (sum over t < np[r] of e[r][t] * v[t][i]) * inv[r]; inv NULL: per block j < nbk the unnormalized
    // sums into out[r] + (ob + j) * pst. V: block j at vt + j * bs elements, its rows contiguous. DW
    // registers (4 * DW dims, attn_vdim) per pass over the positions; KT: cache type (16-bit: DW even)
    for (int d0 = 0; d0 < hd; d0 += 4 * DW) {
        for (int j = 0; j < nbk; j++) {
            int tb = j * ATT_P, nmin, nmax;
            if (inv) {
                nmin = nmax = np[0];
                for (int r = 1; r < RV; r++) { nmin = np[r] < nmin ? np[r] : nmin; nmax = np[r] > nmax ? np[r] : nmax; }
            } else {
                nmin = tb + ATT_P; nmax = tb;
                for (int r = 0; r < RV; r++) {
                    int e = np[r] < tb + ATT_P ? np[r] : tb + ATT_P;
                    nmin = e < nmin ? e : nmin;
                    nmax = e > nmax ? e : nmax;
                }
            }
            float32x4_t acc[4][16];
            for (int r = 0; r < RV; r++)
                for (int u = 0; u < DW; u++)
                    acc[r][u] = vdupq_n_f32(0.0f);
            size_t vo = (size_t)j * bs; // block j: row t at vo + (t - tb) * hd elements
            int t = tb;
            for (; t < nmin; t++)
                ATTN_PV_STEP(1);
            for (; t < nmax; t++) // causal tail: only the vectors that see position t
                ATTN_PV_STEP(t < np[r]);
            for (int r = 0; r < RV; r++) {
                if (!inv) {
                    if (np[r] > tb)
                        for (int u = 0; u < DW; u++)
                            vst1q_f32(out[r] + (size_t)(ob + j) * pst + attn_vdim(KT, d0, u), acc[r][u]);
                    continue;
                }
                for (int u = 0; u < DW; u++)
                    vst1q_f32(out[r] + attn_vdim(KT, d0, u), vmulq_n_f32(acc[r][u], inv[r]));
            }
        }
    }
}
#undef ATTN_PV_STEP

static inline __attribute__((always_inline))
void attn_group_t(const int KT, int nv, const float *const *q, float *const *out, float *const *part, const int *np,
                  int b0, int b1, const void *kt0, const void *vt0, size_t bs, int hd, float scale, float *s) { RQ_FP_EXACT
    // as att512_group_t: nv <= ATTN_NV query vectors of one kv head, vector r over positions 0..np[r]-1.
    // part NULL: the attention into out[r] (all np <= ATT_P; b0 = 0); else the partial results of blocks
    // b0..b1-1 at part[r] + b * ATT_PSTRIDE(hd) (o, m, l). s: nv * (positions in the blocks) floats.
    // KT: cache type (0 fp32, 1 bf16, 2 f16; kt / vt, bs in its elements). Tile tb of the range: block
    // tb / ATT_PT at + bs, tile tb % ATT_PT in it at + ts (a 2-tile register tile starts at an even tb:
    // never across a block)
    size_t pst = ATT_PSTRIDE(hd), ts = (size_t)ATT_T * hd, esz = KT ? 2 : 4;
    int npr[ATTN_NV], nmax = 0;
    for (int r = 0; r < nv; r++) {
        int n = np[r] - b0 * ATT_P, e = (b1 - b0) * ATT_P;
        npr[r] = n < 0 ? 0 : n < e ? n : e;
        nmax = npr[r] > nmax ? npr[r] : nmax;
    }
    const char *kt = (const char *)kt0 + (size_t)b0 * bs * esz;
    const void *vt = (const char *)vt0 + (size_t)b0 * bs * esz;
    int ntl = (nmax + ATT_T - 1) / ATT_T, nbk = part ? (nmax + ATT_P - 1) / ATT_P : 1;
    size_t st = (size_t)nv * ATT_T; // floats per tile of scores (s + tile * st + r * 16 + j)

    // scores, per group of 4 vectors (q transposed: dim i of vector r0 + l at q4[4 i + l])
    float q4[ATTN_HDMAX * 4] __attribute__((aligned(16)));
    for (int r0 = 0; r0 < nv; r0 += 4) {
        int rv = nv - r0 < 4 ? nv - r0 : 4;
        for (int i = 0; i < hd; i++)
            for (int l = 0; l < 4; l++)
                q4[4 * i + l] = l < rv ? q[r0 + l][i] : 0.0f;
        float *sr = s + r0 * ATT_T;
        int tb = 0;
        #define ATTN_KTILE(tb) (kt + ((size_t)((tb) / ATT_PT) * bs + (size_t)((tb) % ATT_PT) * ts) * esz)
        switch (rv) {
            case 1: for (; tb + 2 <= ntl; tb += 2) attn_qk(1, 2, KT, q4, ATTN_KTILE(tb), ts, hd, scale, sr + tb * st, st);
                    for (; tb < ntl; tb++) attn_qk(1, 1, KT, q4, ATTN_KTILE(tb), ts, hd, scale, sr + tb * st, st); break;
            case 2: for (; tb + 2 <= ntl; tb += 2) attn_qk(2, 2, KT, q4, ATTN_KTILE(tb), ts, hd, scale, sr + tb * st, st);
                    for (; tb < ntl; tb++) attn_qk(2, 1, KT, q4, ATTN_KTILE(tb), ts, hd, scale, sr + tb * st, st); break;
            case 3: for (; tb < ntl; tb++) attn_qk(3, 1, KT, q4, ATTN_KTILE(tb), ts, hd, scale, sr + tb * st, st); break;
            default: for (; tb < ntl; tb++) attn_qk(4, 1, KT, q4, ATTN_KTILE(tb), ts, hd, scale, sr + tb * st, st); break;
        }
        #undef ATTN_KTILE
    }

    // softmax per vector and block
    float inv[ATTN_NV];
    for (int r = 0; r < nv; r++)
        for (int j = 0; j < nbk; j++) {
            int n = npr[r] - j * ATT_P;
            if (n <= 0)
                break;
            n = n < ATT_P || !part ? n : ATT_P;
            int nt = (n + ATT_T - 1) / ATT_T, nl = n - (nt - 1) * ATT_T; // valid positions in the last tile
            float *sr = s + r * ATT_T + (size_t)j * (ATT_P / ATT_T) * st;
            float32x4_t mx = vdupq_n_f32(-FLT_MAX);
            for (int tb = 0; tb < nt; tb++)
                for (int u = 0; u < 4; u++) {
                    if (tb == nt - 1 && 4 * u >= nl) break;
                    float32x4_t v = vld1q_f32(sr + tb * st + 4 * u);
                    if (tb == nt - 1 && 4 * u + 4 > nl) { // partial register: lanes >= nl - 4 u out
                        uint32x4_t lane = { 0, 1, 2, 3 };
                        v = vbslq_f32(vcltq_u32(lane, vdupq_n_u32((uint32_t)(nl - 4 * u))), v, vdupq_n_f32(-FLT_MAX));
                    }
                    mx = vmaxq_f32(mx, v);
                }
            float m = vmaxvq_f32(mx);
            float32x4_t mv = vdupq_n_f32(m), sum[4];
            for (int u = 0; u < 4; u++)
                sum[u] = vdupq_n_f32(0.0f);
            for (int tb = 0; tb < nt; tb++)
                for (int u = 0; u < 4; u++) {
                    float *p = sr + tb * st + 4 * u;
                    float32x4_t e = attn_exp(vsubq_f32(vld1q_f32(p), mv));
                    if (tb == nt - 1) {
                        uint32x4_t lane = { 0, 1, 2, 3 };
                        int lim = nl - 4 * u;
                        e = vbslq_f32(vcltq_s32(vreinterpretq_s32_u32(lane), vdupq_n_s32(lim)), e, vdupq_n_f32(0.0f));
                    }
                    vst1q_f32(p, e);
                    sum[u] = vaddq_f32(sum[u], e);
                }
            // tree: lane j += lane j + 8, then + 4, + 2, + 1
            sum[0] = vaddq_f32(sum[0], sum[2]);
            sum[1] = vaddq_f32(sum[1], sum[3]);
            sum[0] = vaddq_f32(sum[0], sum[1]);
            float32x4_t h = vaddq_f32(sum[0], vextq_f32(sum[0], sum[0], 2));
            float tot = vgetq_lane_f32(h, 0) + vgetq_lane_f32(h, 1);
            if (part) {
                float *ml = part[r] + (size_t)(b0 + j) * pst + hd;
                ml[0] = m;
                ml[1] = tot;
            } else {
                inv[r] = 1.0f / tot;
            }
        }

    // weighted sum of V, per group of up to 4 vectors
    for (int r0 = 0; r0 < nv; r0 += 4) {
        int rv = nv - r0 < 4 ? nv - r0 : 4;
        const float *sr = s + r0 * ATT_T;
        const float *iv = part ? NULL : inv + r0;
        float *const *o = part ? part + r0 : out + r0;
        switch (rv) {
            case 1: attn_pv(1, 16, KT, sr, st, nbk, npr + r0, vt, bs, hd, iv, o, b0, pst); break;
            case 2: attn_pv(2, 8, KT, sr, st, nbk, npr + r0, vt, bs, hd, iv, o, b0, pst); break;
            case 3: attn_pv(3, 4, KT, sr, st, nbk, npr + r0, vt, bs, hd, iv, o, b0, pst); break;
            default: attn_pv(4, 4, KT, sr, st, nbk, npr + r0, vt, bs, hd, iv, o, b0, pst); break;
        }
    }
}

static void attn_group_f32(int nv, const float *const *q, float *const *out, float *const *part, const int *np, int b0,
                           int b1, const void *kt, const void *vt, size_t bs, int hd, float scale, float *s) {
    attn_group_t(0, nv, q, out, part, np, b0, b1, kt, vt, bs, hd, scale, s);
}

static void attn_group_bf16(int nv, const float *const *q, float *const *out, float *const *part, const int *np, int b0,
                            int b1, const void *kt, const void *vt, size_t bs, int hd, float scale, float *s) {
    attn_group_t(1, nv, q, out, part, np, b0, b1, kt, vt, bs, hd, scale, s);
}

static void attn_group_f16(int nv, const float *const *q, float *const *out, float *const *part, const int *np, int b0,
                           int b1, const void *kt, const void *vt, size_t bs, int hd, float scale, float *s) {
    attn_group_t(2, nv, q, out, part, np, b0, b1, kt, vt, bs, hd, scale, s);
}

static inline int attn_ok(int hd) { return hd % 64 == 0 && hd <= ATTN_HDMAX; }

static void attn_kv(int kvt, int nv, const float *const *q, float *const *out, float *const *part, const int *np, int b0,
                    int b1, const void *kt, const void *vt, size_t bs, int hd, float scale, float *s) {
    // nv query vectors of one kv head (any nv), in groups of ATTN_NV (attn_group_t); kvt: cache type, 0
    // fp32, 1 bf16, 2 f16 (KV_F32, KV_BF16, KV_F16: RQ_KV_* of runq.h); bs: elements from a head's block to its next
    for (int g = 0; g < nv; g += ATTN_NV) {
        int gn = nv - g < ATTN_NV ? nv - g : ATTN_NV;
        (kvt == 2 ? attn_group_f16 : kvt ? attn_group_bf16 : attn_group_f32)(gn, q + g, out + g, part ? part + g : NULL,
                                                                          np + g, b0, b1, kt, vt, bs, hd, scale, s);
    }
}
#define RUNQ_ATTNEON 1

#endif // NEON
#endif // RUNQ_KERNELS_NEON_H
