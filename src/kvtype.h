/* KV cache types (runq.h's RQ_KV_*): element formats of the K and V caches, tile sizes. Shared by internal.h and the
 * attention kernels (kernels_avx512.h, also included alone by tests/bf16dp_test.c). */
#ifndef RUNQ_KVTYPE_H
#define RUNQ_KVTYPE_H
#include <stddef.h>
#include "runq.h"

// KV cache element type: fp32, bf16 or f16 (16-bit: half the bytes; layouts in kernels_avx512.h), q8 / q4 / q4k4
// (8-bit; 8-bit K and 4-bit V; 4-bit K and V: per-32-dim scales, rotated: forward.c, "8-bit / 4-bit KV cache"); KV_DOT:
// the flag of the bf16 dot-product attention in a sequence's attention type (RqSeq.att: RQ_KV_BF16_DOT = KV_BF16 |
// KV_DOT, ...). The element formats of one half (kv_fmt): KV_F32, KV_BF16, KV_F16, KV_Q8, KV_Q4
enum { KV_F32 = RQ_KV_F32, KV_BF16 = RQ_KV_BF16, KV_F16 = RQ_KV_F16, KV_Q8 = RQ_KV_Q8, KV_DOT = 4, KV_Q4 = RQ_KV_Q4,
       KV_Q4K4 = RQ_KV_Q4K4 };
static inline int kv_quant(int t) { t &= ~KV_DOT; return t == KV_Q8 || t == KV_Q4 || t == KV_Q4K4; } // (a type or a format)
static inline int kv_type_ok(int t) {
    int e = t & ~KV_DOT;
    return t >= 0 && (e == KV_F32 || e == KV_BF16 || e == KV_F16 || kv_quant(e)) && t != (KV_F32 | KV_DOT);
}
// the type for a model of head_dim hd: 16-bit, a multiple of 32 (the K layout's dim pairs, the attention's chunks);
// q8 / q4 a power of two >= 32 (the rotation, the 32-dim scale groups)
static inline int kv_type_fits(int t, int hd) {
    return kv_type_ok(t) && (t == KV_F32 || (hd % 32 == 0 && (!kv_quant(t) || (hd & (hd - 1)) == 0)));
}
// the element format of a type's K (kv 0) or V (kv 1) cache: the type's, except q4's K: q8 (4-bit K costs Qwen3 dense
// models ~600x the KL of 8-bit; docs/kv-quant.md), and q4k4's: q4 for both
static inline int kv_fmt(int t, int kv) {
    int e = t & ~KV_DOT;
    return e == KV_Q4K4 ? KV_Q4 : e == KV_Q4 && !kv ? KV_Q8 : e;
}
// bytes of one cache tile (ATT_T = 16 positions x hd of one kv head) of K (kv 0) or V (kv 1) of type t: fp32 64 hd,
// 16-bit 32 hd; q8: 16 hd of int8 + 16 x hd / 32 bf16 scales = 17 hd; q4: 8 hd + hd = 9 hd (layouts: forward.c). Every
// cache offset the code computes is a whole number of tiles (layers, blocks, kv_tile): bytes = elements / (16 hd) x
// tile bytes
static inline size_t kv_tile_bytes(int t, int hd, int kv) {
    int e = kv_fmt(t, kv);
    return (size_t)hd * (e == KV_F32 ? 64 : e == KV_Q8 ? 17 : e == KV_Q4 ? 9 : 32);
}


#endif // RUNQ_KVTYPE_H
