/* Inference for Qwen-3 Transformer model in pure C, int8 quantized forward pass. */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif
#ifdef _OPENMP
    #include <omp.h>
#endif
// AVX-512 VNNI kernels (defines RUNQ_VNNI when built for AVX-512 VNNI without -DRUNQ_GENERIC)
#include "kernels_avx512.h"
// K-quants (GGUF Q4_K / Q6_K): block formats, repacked layouts, arithmetic, generic kernels
#include "kquants.h"
// phase synchronization and the parallel region of a decode step / prefill chunk
#include "threads.h"

// ----------------------------------------------------------------------------
// Globals
int GS = 0; // group size global for quantization of the weights
int KQ = 0; // the model has K-quant weights (GGUF Q4_K / Q6_K): the activation sums xf are needed

// Maximum input prompt buffer size
#define PROMPT_BUFFER_SIZE 32768

// Batched prefill (forward_batch): prompt tokens are processed in chunks of prefill_chunk
// rows (-b, at most B_MAX); 0 or 1 = one token at a time through forward(). Bigger chunks are
// faster (fewer barriers and weight passes per token): 8 threads, 1,500 / 3,000 tokens:
// -b 64 1,237 / 911 tok/s, 128 1,370 / 1,018, 256 1,467 / 1,093, 512 1,547 / 1,112, 1024 within
// noise of 512. The chunk buffers take ~58 KB per row (~30 MB at 512)
#define B_MAX 512
#define PREFILL_CHUNK_DEFAULT 512
int prefill_chunk = PREFILL_CHUNK_DEFAULT;

double now_s(void);
#ifdef RUNQ_PROFILE
// build with -DRUNQ_PROFILE: wall time per forward_batch phase, printed at exit
enum { PF_NORM, PF_QKV, PF_ROPE, PF_ATT, PF_WO, PF_FFN13, PF_SWIGLU, PF_FFN2, PF_CLS, PF_N };
static const char *pf_name[PF_N] = { "norm+quant+resid", "gemm qkv", "qk-norm+rope+kv", "attention+quant",
                                      "gemm wo", "gemm w1+w3", "swiglu+quant", "gemm w2", "classifier" };
static double pf_time[PF_N], pf_t;
#define PROF_START() (pf_t = now_s())
#define PROF(i) do { double t_ = now_s(); pf_time[i] += t_ - pf_t; pf_t = t_; } while (0)
#else
#define PROF_START() ((void)0)
#define PROF(i) ((void)0)
#endif

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int magic_number; // checkpoint magic number
    int version; // file format version
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
    int head_dim; // head dimension
    int shared_classifier; // 1 if wcls == p_tokens
    int group_size; // quantization group size (export.py uses 64; GGUF Q8_0: 32)
} Config;

typedef struct {
    int8_t *q;    // quantized values
    float *s; // scaling factors
    uint8_t *p;   // weights only: the same matrix repacked for the VNNI kernels (Q8P layout,
                  // see kernels_avx512.h), or NULL (generic kernels on q / s)
    const uint8_t *b; // GGUF weights: the blocks in the file mapping (Q8_0: per 32 values an fp16
                  // scale + 32 int8, Q8_0_BLOCK bytes; Q4_K / Q6_K: kquants.h), converted at load
                  // (prepare_weights); or NULL
    int type;     // weights: ggml type, GGML_Q8_0 (also the .bin's group-64 Q8), GGML_Q4_K, GGML_Q6_K
    uint8_t *kp;  // Q4_K / Q6_K weights: repacked (Q4P / Q6P layout, kquants.h), else NULL
} QuantizedTensor;

typedef struct {
    // token embedding table
    QuantizedTensor *q_tokens; // (vocab_size, dim); rows dequantized at lookup (embed_row)
    // weights for rmsnorms
    float *rms_att_weight; // (layer, dim) rmsnorm weights
    float *rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    QuantizedTensor *wq; // (layer, dim, n_heads * head_size)
    QuantizedTensor *wk; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wv; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wo; // (layer, n_heads * head_size, dim)
    // QK-RMSNorm for Qwen3
    float *q_norm_weights;
    float *k_norm_weights;
    // weights for ffn
    QuantizedTensor *w1; // (layer, hidden_dim, dim)
    QuantizedTensor *w2; // (layer, dim, hidden_dim)
    QuantizedTensor *w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float *rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    QuantizedTensor *wcls;
    // one allocation holding the matmul weights converted at load (prepare_weights): VNNI build:
    // repacked (QuantizedTensor.p); generic kernels on a GGUF file: q / s arrays; else NULL
    uint8_t *packed;
    size_t packed_size;
    float *norms; // GGUF: the norm weights, copied in the .bin layout (NULL: .bin, in the mapping)
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    QuantizedTensor xq; // quantized x (dim,)
    QuantizedTensor hq; // quantized hb (hidden_dim,)
    QuantizedTensor aq; // quantized attention output (all_heads_dim,), the input of wo
    float *q; // query (dim,)
    float *k; // key (kv_dim,), copied into the cache tiles after QK-norm + RoPE
    float *v; // value (kv_dim,), the same
    float *att; // buffer for scores/attention values (n_heads, kv_len)
    float *logits; // output logits
    // kv cache: per layer kv_len / ATT_T tiles of ATT_T positions x n_kv_heads, layout in
    // kernels_avx512.h (attention); kv_len = seq_len rounded up to a multiple of ATT_T
    float *key_cache;   // (layer, kv_len, kv_dim)
    float *value_cache; // (layer, kv_len, kv_dim)
    int kv_len;
    size_t kv_size;     // floats per cache
    float att_scale;    // 1 / sqrt(head_dim)
    int32_t *xq_corr;   // VNNI GEMV activation corrections of xq / hq (q8p_gemv_corr), per GS-group
    int32_t *hq_corr;
    float *xq_xf;       // K-quant kernels: activation group sums of xq / hq (kq_act_prep), or NULL
    float *hq_xf;
    int32_t *xq_c16;    // AVX-512 K-quant GEMV: -32 * sums of 16 of xq / hq (kq512_c16), or NULL
    int32_t *hq_c16;
    int32_t *aq_corr;   // the same for aq
    float *aq_xf;
    int32_t *aq_c16;
    int ff_rows;        // hidden rows per forward() ffn work item (w1 + w3 + swiglu + quantize)
    // batched prefill buffers (forward_batch), nb_max rows each; not allocated if nb_max <= 1
    int nb_max;
    int xs;         // row stride of xb_b / xq_b: max(dim, all_heads_dim), a multiple of GS
    float *x_b;     // (nb_max, dim) residual stream
    float *xb_b;    // (nb_max, xs)
    float *q_b;     // (nb_max, all_heads_dim)
    float *k_b;     // (nb_max, kv_dim), copied into the cache tiles after QK-norm + RoPE
    float *v_b;     // (nb_max, kv_dim), the same
    float *hb_b;    // (nb_max, hidden_dim)
    float *hb2_b;   // (nb_max, hidden_dim)
    QuantizedTensor xq_b; // (nb_max, xs)
    QuantizedTensor hq_b; // (nb_max, hidden_dim)
    float *att_b;   // (att_threads, att_rows * kv_mul * att_nb * ATT_P): attention scores scratch per OpenMP thread
    int att_rows;   // query rows per prefill attention work item
    int att_threads;
    float *rope_b;  // (nb_max, head_dim): RoPE cos / sin of each row's position (rope_row)
    float *rope;    // (head_dim,): the same for forward()'s position
    RqSync sync;    // phase counters of forward()'s parallel job (threads.h)
    int att_nb;     // attention blocks of ATT_P positions in kv_len
    float *att_part; // forward(): per (kv head, query head, block) partial results (ATT_PSTRIDE floats)
    float *att_pb;  // forward_batch: the same per thread, att_rows * kv_mul vectors
    int pf_nrb;     // forward_batch: attention row blocks per chunk (at most)
    int32_t *bq_corr, *bq_c16, *bh_corr, *bh_c16; // forward_batch GEMM activation prep of xq_b / hq_b rows
    float *bq_xf, *bh_xf;                         //   (Gemm, act_prep_row)
    uint8_t *kq_scr; size_t kq_scr_bytes;         // per-thread K-quant GEMM tile scratch (AVX-512)
    int dec_chunk, cls_chunk; // 16-row blocks per claim in forward()'s GEMVs / classifier
} RunState;

// ----------------------------------------------------------------------------
// GGUF files (llama.cpp's format, versions 2 and 3), read directly: hyperparameters and tokenizer
// from the metadata, Q8_0 / Q4_K / Q6_K / F32 tensors from the mapped file. Layout (little-endian):
//   "GGUF", u32 version, u64 n_tensors, u64 n_kv;
//   n_kv x (string key, u32 value type, value); string = u64 length + bytes (no NUL); array =
//     u32 element type, u64 count, the elements (no nested arrays here);
//   n_tensors x (string name, u32 n_dims, u64 ne[n_dims], u32 ggml type, u64 offset);
//   padding to general.alignment (default 32), then the tensor data, tensor i at data_off + offset.
// ne[0] is the fastest dimension: a row-major (d, k) matrix has ne = { k, d }.
// The file is untrusted input: every count, length and offset is checked against the file size
// (gguf_take, overflow-safe) before use, counts before any allocation, and every tensor's data
// range when its header is read; errors exit with a message naming the file.

#define GGUF_MAGIC 0x46554747u // "GGUF"
enum { GV_U8, GV_I8, GV_U16, GV_I16, GV_U32, GV_I32, GV_F32, GV_BOOL, GV_STR, GV_ARR, GV_U64, GV_I64, GV_F64, GV_N };
static const uint8_t gv_size[GV_N] = { 1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8 }; // scalar value bytes
enum { GGML_F32 = 0, GGML_F16 = 1, GGML_Q8_0 = 8 }; // tensor types (ggml numbering) used here
#define Q8_0_BLOCK 34 // ggml block_q8_0: fp16 scale d, int8 qs[32]; value = qs[i] * d

typedef struct { const char *key; uint64_t key_len, off, n; uint32_t type, arr_type; } GgufKV; // off: value / first element
typedef struct { const char *name; uint64_t name_len, ne[4], off, size; uint32_t type, n_dims; } GgufTensor; // off: in file
typedef struct {
    const uint8_t *map; uint64_t size; const char *path; // the whole file, mapped read-only
    uint32_t version;
    uint64_t alignment, data_off;
    GgufKV *kv; uint64_t n_kv;
    GgufTensor *t; uint64_t n_t;
} Gguf;

static void gguf_fail(const Gguf *g, const char *fmt, const char *arg) {
    fprintf(stderr, "GGUF %s: ", g->path);
    fprintf(stderr, fmt, arg ? arg : "");
    fprintf(stderr, "\n");
    exit(EXIT_FAILURE);
}

static const uint8_t *gguf_take(const Gguf *g, uint64_t *pos, uint64_t n, const char *what) {
    // n bytes at *pos (advanced past them), all inside the file, else a clean error
    if (*pos > g->size || n > g->size - *pos) gguf_fail(g, "truncated or corrupt (%s runs past the end of the file)", what);
    const uint8_t *p = g->map + *pos;
    *pos += n;
    return p;
}

static uint32_t gguf_u32(const Gguf *g, uint64_t *pos, const char *what) { uint32_t v; memcpy(&v, gguf_take(g, pos, 4, what), 4); return v; }
static uint64_t gguf_u64(const Gguf *g, uint64_t *pos, const char *what) { uint64_t v; memcpy(&v, gguf_take(g, pos, 8, what), 8); return v; }

static uint64_t gguf_str(const Gguf *g, uint64_t *pos, const char **s, const char *what) {
    uint64_t n = gguf_u64(g, pos, what);
    *s = (const char *)gguf_take(g, pos, n, what);
    return n;
}

static int gguf_type_bytes(uint32_t type, uint64_t n, uint64_t ne0, uint64_t *bytes) {
    // data size of n elements of a ggml type (rows of ne0, blocks never straddle a row); 0 if the
    // type is unknown here (such a tensor cannot be used) or the shape does not fit its blocks.
    // Block sizes / bytes from ggml.c (type_traits); the k-quants for Step 6b's Q4_K_M
    static const struct { uint32_t type, blk, bytes; } tt[] = {
        { 0, 1, 4 }, { 1, 1, 2 }, { 30, 1, 2 }, { 2, 32, 18 }, { 3, 32, 20 }, { 6, 32, 22 }, { 7, 32, 24 },
        { 8, 32, 34 }, { 10, 256, 84 }, { 11, 256, 110 }, { 12, 256, 144 }, { 13, 256, 176 }, { 14, 256, 210 } };
    for (size_t i = 0; i < sizeof(tt) / sizeof(tt[0]); i++)
        if (tt[i].type == type) {
            if (ne0 % tt[i].blk) return 0;
            *bytes = n / tt[i].blk * tt[i].bytes; // n <= 2^56: no overflow
            return 1;
        }
    return 0;
}

static void gguf_parse(Gguf *g) {
    // header, metadata and tensor table of the mapped file g->map (g->size bytes)
    uint64_t pos = 0;
    char what[96];
    if (gguf_u32(g, &pos, "header") != GGUF_MAGIC) gguf_fail(g, "not a GGUF file (bad magic)%s", NULL);
    g->version = gguf_u32(g, &pos, "header");
    if (g->version != 2 && g->version != 3) {
        snprintf(what, sizeof(what), "%u", g->version);
        gguf_fail(g, "unsupported GGUF version %s (need 2 or 3)", what);
    }
    g->n_t = gguf_u64(g, &pos, "header");
    g->n_kv = gguf_u64(g, &pos, "header");
    // every metadata entry takes >= 13 bytes and every tensor entry >= 32: counts the rest of the
    // file cannot hold are corrupt (and must not size an allocation); models have hundreds
    if (g->n_kv > (g->size - pos) / 13 || g->n_t > (g->size - pos) / 32 || g->n_kv > 65536 || g->n_t > 65536)
        gguf_fail(g, "corrupt header (tensor / metadata counts too large)%s", NULL);
    g->kv = calloc(g->n_kv + 1, sizeof(GgufKV));
    g->t = calloc(g->n_t + 1, sizeof(GgufTensor));
    if (!g->kv || !g->t) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }

    g->alignment = 32;
    for (uint64_t i = 0; i < g->n_kv; i++) {
        GgufKV *kv = &g->kv[i];
        snprintf(what, sizeof(what), "metadata entry %llu", (unsigned long long)i);
        kv->key_len = gguf_str(g, &pos, &kv->key, what);
        kv->type = gguf_u32(g, &pos, what);
        kv->n = 1;
        uint32_t et = kv->type;
        if (kv->type == GV_ARR) {
            et = kv->arr_type = gguf_u32(g, &pos, what);
            kv->n = gguf_u64(g, &pos, what);
        }
        if (et >= GV_N || et == GV_ARR) gguf_fail(g, "corrupt metadata (bad value type in %s)", what);
        kv->off = pos;
        if (et == GV_STR) {
            if (kv->n > (g->size - pos) / 8) gguf_fail(g, "truncated or corrupt (%s runs past the end of the file)", what);
            for (uint64_t j = 0; j < kv->n; j++) {
                const char *s;
                gguf_str(g, &pos, &s, what);
            }
        } else {
            if (kv->n > (g->size - pos) / gv_size[et]) gguf_fail(g, "truncated or corrupt (%s runs past the end of the file)", what);
            gguf_take(g, &pos, kv->n * gv_size[et], what);
        }
        if (kv->key_len == 17 && !memcmp(kv->key, "general.alignment", 17)) {
            uint32_t a = 0;
            if (kv->type == GV_U32) memcpy(&a, g->map + kv->off, 4);
            if (a == 0 || a > (1u << 16) || (a & (a - 1))) gguf_fail(g, "bad general.alignment (need a power of two <= 65536, u32)%s", NULL);
            g->alignment = a;
        }
    }

    for (uint64_t i = 0; i < g->n_t; i++) {
        GgufTensor *t = &g->t[i];
        snprintf(what, sizeof(what), "tensor entry %llu", (unsigned long long)i);
        t->name_len = gguf_str(g, &pos, &t->name, what);
        t->n_dims = gguf_u32(g, &pos, what);
        if (t->n_dims < 1 || t->n_dims > 4) gguf_fail(g, "corrupt tensor table (bad dimension count in %s)", what);
        uint64_t n = 1;
        for (uint32_t d = 0; d < 4; d++) {
            t->ne[d] = d < t->n_dims ? gguf_u64(g, &pos, what) : 1;
            if (t->ne[d] == 0 || t->ne[d] > (1ull << 56) / n) gguf_fail(g, "corrupt tensor table (bad shape in %s)", what);
            n *= t->ne[d];
        }
        t->type = gguf_u32(g, &pos, what);
        t->off = gguf_u64(g, &pos, what);
        if (!gguf_type_bytes(t->type, n, t->ne[0], &t->size))
            t->size = UINT64_MAX; // unknown type or shape: unusable (checked if used, gguf_weight)
    }
    // the data section starts at the next multiple of the alignment
    g->data_off = pos + (g->alignment - pos % g->alignment) % g->alignment;
    if (g->data_off > g->size) gguf_fail(g, "truncated (no tensor data)%s", NULL);
    for (uint64_t i = 0; i < g->n_t; i++) {
        GgufTensor *t = &g->t[i];
        if (t->off % g->alignment || t->off > g->size - g->data_off ||
            (t->size != UINT64_MAX && t->size > g->size - g->data_off - t->off)) {
            snprintf(what, sizeof(what), "%.*s", (int)(t->name_len < 64 ? t->name_len : 64), t->name);
            gguf_fail(g, "truncated or corrupt (data of tensor '%s' misaligned or past the end of the file)", what);
        }
        t->off += g->data_off;
        for (uint64_t j = 0; j < i; j++)
            if (g->t[j].name_len == t->name_len && !memcmp(g->t[j].name, t->name, t->name_len)) {
                snprintf(what, sizeof(what), "%.*s", (int)(t->name_len < 64 ? t->name_len : 64), t->name);
                gguf_fail(g, "corrupt tensor table (duplicate tensor '%s')", what);
            }
    }
}

static const GgufKV *gguf_find(const Gguf *g, const char *key) {
    size_t n = strlen(key);
    for (uint64_t i = 0; i < g->n_kv; i++)
        if (g->kv[i].key_len == n && !memcmp(g->kv[i].key, key, n))
            return &g->kv[i];
    return NULL;
}

static int gguf_get_int(const Gguf *g, const char *key, int64_t *v) {
    // integer scalar metadata (any integer type), 0 if absent; a value of another type fails
    const GgufKV *kv = gguf_find(g, key);
    if (!kv) return 0;
    const uint8_t *p = g->map + kv->off;
    switch (kv->type) {
        case GV_U8: *v = *p; break;
        case GV_I8: *v = (int8_t)*p; break;
        case GV_U16: { uint16_t x; memcpy(&x, p, 2); *v = x; break; }
        case GV_I16: { int16_t x; memcpy(&x, p, 2); *v = x; break; }
        case GV_U32: { uint32_t x; memcpy(&x, p, 4); *v = x; break; }
        case GV_I32: { int32_t x; memcpy(&x, p, 4); *v = x; break; }
        case GV_U64: { uint64_t x; memcpy(&x, p, 8); *v = x > INT64_MAX ? INT64_MAX : (int64_t)x; break; }
        case GV_I64: memcpy(v, p, 8); break;
        default: gguf_fail(g, "metadata %s is not an integer", key);
    }
    return 1;
}

static int gguf_get_float(const Gguf *g, const char *key, float *v) {
    // f32 (or f64, rounded) scalar metadata, 0 if absent
    const GgufKV *kv = gguf_find(g, key);
    if (!kv) return 0;
    if (kv->type == GV_F32) memcpy(v, g->map + kv->off, 4);
    else if (kv->type == GV_F64) { double d; memcpy(&d, g->map + kv->off, 8); *v = (float)d; }
    else gguf_fail(g, "metadata %s is not a float", key);
    return 1;
}

static const char *gguf_get_str(const Gguf *g, const char *key, uint64_t *len) {
    // string metadata (not NUL-terminated: *len bytes), NULL if absent
    const GgufKV *kv = gguf_find(g, key);
    if (!kv) return NULL;
    if (kv->type != GV_STR) gguf_fail(g, "metadata %s is not a string", key);
    uint64_t pos = kv->off;
    const char *s;
    *len = gguf_str(g, &pos, &s, key);
    return s;
}

static const GgufKV *gguf_get_array(const Gguf *g, const char *key, uint32_t type) {
    // array metadata with elements of this type, NULL if absent
    const GgufKV *kv = gguf_find(g, key);
    if (kv && (kv->type != GV_ARR || kv->arr_type != type)) gguf_fail(g, "metadata %s: unexpected type", key);
    return kv;
}

static const GgufTensor *gguf_tensor(const Gguf *g, const char *name) {
    size_t n = strlen(name);
    for (uint64_t i = 0; i < g->n_t; i++)
        if (g->t[i].name_len == n && !memcmp(g->t[i].name, name, n))
            return &g->t[i];
    return NULL;
}

static const uint8_t *gguf_weight(const Gguf *g, const char *name, uint32_t type, uint64_t ne0, uint64_t ne1) {
    // the data of a required tensor of this type and shape ({ ne0 } if ne1 == 0, else { ne0, ne1 })
    const GgufTensor *t = gguf_tensor(g, name);
    if (!t) gguf_fail(g, "tensor %s not found", name);
    int dims = ne1 ? 2 : 1;
    if (t->type != type) {
        char msg[160];
        snprintf(msg, sizeof(msg), "tensor %%s has ggml type %u, need %u (%s)", t->type, type,
                 type == GGML_Q8_0 ? "Q8_0" : type == GGML_F32 ? "F32" : "?");
        gguf_fail(g, msg, name);
    }
    if (t->n_dims != (uint32_t)dims || t->ne[0] != ne0 || (ne1 && t->ne[1] != ne1)) {
        char msg[200];
        snprintf(msg, sizeof(msg), "tensor %%s has shape [%llu, %llu], need [%llu, %llu]", (unsigned long long)t->ne[0],
                 (unsigned long long)(t->n_dims > 1 ? t->ne[1] : 0), (unsigned long long)ne0, (unsigned long long)ne1);
        gguf_fail(g, msg, name);
    }
    if (t->size == UINT64_MAX) gguf_fail(g, "tensor %s: rows are not a whole number of blocks", name);
    return g->map + t->off;
}

static void gguf_config(const Gguf *g, Config *c) {
    // the hyperparameters from the metadata (llama.cpp's qwen3 keys), checked against what this
    // engine implements: Qwen3 with RoPE theta 1e6 and RMSNorm eps 1e-6 (constants in rope_row /
    // rmsnorm); Q8_0 / Q4_K / Q6_K weights (gguf_matrices)
    uint64_t n;
    const char *arch = gguf_get_str(g, "general.architecture", &n);
    if (!arch || n != 5 || memcmp(arch, "qwen3", 5)) {
        char a[40];
        snprintf(a, sizeof(a), "%.*s", arch ? (int)(n < 32 ? n : 32) : 6, arch ? arch : "(none)");
        gguf_fail(g, "architecture '%s' is not supported (need qwen3)", a);
    }
    const char *keys[] = { "qwen3.block_count", "qwen3.embedding_length", "qwen3.feed_forward_length",
                           "qwen3.attention.head_count", "qwen3.attention.head_count_kv", "qwen3.context_length",
                           "qwen3.attention.key_length", "qwen3.attention.value_length" };
    int64_t v[8];
    for (int i = 0; i < 8; i++) {
        if (!gguf_get_int(g, keys[i], &v[i])) {
            if (i < 6) gguf_fail(g, "metadata %s missing", keys[i]);
            v[i] = i == 6 ? v[1] / (v[3] ? v[3] : 1) : v[6]; // llama.cpp's defaults: dim / heads
        }
        if (v[i] < 1 || v[i] > (1 << 20)) gguf_fail(g, "metadata %s out of range", keys[i]);
    }
    float theta = 0, eps = 0;
    if (!gguf_get_float(g, "qwen3.rope.freq_base", &theta) || theta != 1e6f)
        gguf_fail(g, "qwen3.rope.freq_base must be 1e6 (the engine's RoPE)%s", NULL);
    if (!gguf_get_float(g, "qwen3.attention.layer_norm_rms_epsilon", &eps) || eps != 1e-6f)
        gguf_fail(g, "qwen3.attention.layer_norm_rms_epsilon must be 1e-6 (the engine's RMSNorm)%s", NULL);
    const char *rs = gguf_get_str(g, "qwen3.rope.scaling.type", &n);
    if (rs && !(n == 4 && !memcmp(rs, "none", 4))) gguf_fail(g, "RoPE scaling is not supported%s", NULL);
    if (v[6] != v[7]) gguf_fail(g, "attention key_length != value_length is not supported%s", NULL);
    if (v[3] % v[4]) gguf_fail(g, "head_count is not a multiple of head_count_kv%s", NULL);
    const GgufTensor *emb = gguf_tensor(g, "token_embd.weight");
    if (!emb || emb->n_dims != 2 || emb->ne[0] != (uint64_t)v[1] || emb->ne[1] > (1 << 24))
        gguf_fail(g, "tensor token_embd.weight missing or not [embedding_length, vocab]%s", NULL);
    memset(c, 0, sizeof(*c));
    c->magic_number = GGUF_MAGIC;
    c->version = (int)g->version;
    c->n_layers = (int)v[0];
    c->dim = (int)v[1];
    c->hidden_dim = (int)v[2];
    c->n_heads = (int)v[3];
    c->n_kv_heads = (int)v[4];
    c->seq_len = (int)v[5];
    c->head_dim = (int)v[6];
    c->vocab_size = (int)emb->ne[1];
    c->shared_classifier = gguf_tensor(g, "output.weight") == NULL; // absent: tied to token_embd
    c->group_size = 32; // activation groups: Q8_0 blocks, Q4_K sub-blocks, 2 Q6_K sub-blocks
}

static void gguf_open(Gguf *g, const char *path) {
    // map and parse a GGUF file (-m tokenize; read_checkpoint parses its own mapping)
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Couldn't open %s\n", path); exit(EXIT_FAILURE); }
#if defined _WIN32
    _fseeki64(f, 0, SEEK_END);
    int64_t size = _ftelli64(f);
#else
    fseek(f, 0, SEEK_END);
    int64_t size = ftell(f);
#endif
    if (size < 24) { fprintf(stderr, "GGUF %s: file too small\n", path); exit(EXIT_FAILURE); }
    void *map = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fileno(f), 0);
    fclose(f);
    if (map == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    memset(g, 0, sizeof(*g));
    g->map = map;
    g->size = (uint64_t)size;
    g->path = path;
    gguf_parse(g);
}

static int file_is_gguf(const char *path) {
    // the checkpoint format by magic: GGUF, else a qwen3.c .bin
    FILE *f = fopen(path, "rb");
    uint32_t magic = 0;
    if (f && fread(&magic, 4, 1, f) != 1) magic = 0;
    if (f) fclose(f);
    return magic == GGUF_MAGIC;
}

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    float *data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
    Gguf gguf; // GGUF checkpoint: the parsed file (gguf.map == NULL: .bin)
} Transformer;

#if defined _WIN32
    #include <malloc.h> // _aligned_malloc
#endif
static void *alloc_big(size_t n) {
    // large zeroed 64-byte aligned buffer (repacked weights, KV cache): page-aligned anonymous
    // memory on Linux, touched lazily. No MADV_HUGEPAGE: no measurable decode gain here, and with
    // THP defrag=madvise the faults can stall on compaction (seen for the KV cache: 2-3x slower
    // prefill)
#if defined _WIN32
    void *p = _aligned_malloc(n, 64);
    if (p) memset(p, 0, n);
    return p;
#else
    void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? NULL : p;
#endif
}

static void free_big(void *p, size_t n) {
#if defined _WIN32
    (void)n;
    _aligned_free(p);
#else
    if (p) munmap(p, n);
#endif
}

#ifndef ATT_ROWS
#define ATT_ROWS 6 // query rows per prefill attention work item (x kv_mul query heads)
#endif
#define ATT_NV_MAX 64 // max query vectors per attention_kv call

void malloc_run_state(RunState* s, Config *p, int nb_max) {
    // we calloc instead of malloc to keep valgrind happy
    int all_heads_dim = p->n_heads * p->head_dim;
    int kv_dim = p->n_kv_heads * p->head_dim;

    int kv_mul = p->n_heads / p->n_kv_heads;

    memset(s, 0, sizeof(*s));
    s->kv_len = (p->seq_len + ATT_T - 1) / ATT_T * ATT_T;
    s->att_nb = (s->kv_len + ATT_P - 1) / ATT_P;
    s->kv_size = (size_t)p->n_layers * s->kv_len * kv_dim;
    s->att_scale = 1.0f / sqrtf(p->head_dim);
    s->att_rows = ATT_NV_MAX / kv_mul < ATT_ROWS ? ATT_NV_MAX / kv_mul : ATT_ROWS;
    s->ff_rows = GS > 64 ? GS : 64;
    if (kv_mul < 1 || s->att_rows < 1 || p->head_dim % 16 || (kv_mul * p->head_dim) % GS ||
        p->hidden_dim % s->ff_rows || s->ff_rows % GS || p->dim % 16) {
        fprintf(stderr, "unsupported model shape (heads %d / %d, head_dim %d, dim %d, hidden %d, GS %d)\n",
                p->n_heads, p->n_kv_heads, p->head_dim, p->dim, p->hidden_dim, GS);
        exit(EXIT_FAILURE);
    }
    s->nb_max = nb_max > B_MAX ? B_MAX : nb_max;
    if (s->nb_max > 1) {
        size_t nb = s->nb_max;
        s->xs = p->dim > all_heads_dim ? p->dim : all_heads_dim;
        s->x_b = calloc(nb * p->dim, sizeof(float));
        s->xb_b = calloc(nb * s->xs, sizeof(float));
        s->q_b = calloc(nb * all_heads_dim, sizeof(float));
        s->k_b = calloc(nb * kv_dim, sizeof(float));
        s->v_b = calloc(nb * kv_dim, sizeof(float));
        s->hb_b = calloc(nb * p->hidden_dim, sizeof(float));
        s->hb2_b = calloc(nb * p->hidden_dim, sizeof(float));
        s->xq_b = (QuantizedTensor) { .q = calloc(nb * s->xs, sizeof(int8_t)), .s = calloc(nb * s->xs / GS, sizeof(float)) };
        s->hq_b = (QuantizedTensor) { .q = calloc(nb * p->hidden_dim, sizeof(int8_t)), .s = calloc(nb * p->hidden_dim / GS, sizeof(float)) };
        s->att_threads = rq_nthreads();
        s->pf_nrb = (s->nb_max + s->att_rows - 1) / s->att_rows;
        s->bq_corr = calloc(nb * (s->xs / GS) + 1, sizeof(int32_t));
        s->bh_corr = calloc(nb * (p->hidden_dim / GS) + 1, sizeof(int32_t));
        if (!s->bq_corr || !s->bh_corr) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
        if (KQ) {
            s->bq_xf = calloc(nb * (s->xs / GS) + 1, sizeof(float));
            s->bh_xf = calloc(nb * (p->hidden_dim / GS) + 1, sizeof(float));
            s->bq_c16 = calloc(nb * (s->xs / 16) + 1, sizeof(int32_t));
            s->bh_c16 = calloc(nb * (p->hidden_dim / 16) + 1, sizeof(int32_t));
            if (!s->bq_xf || !s->bh_xf || !s->bq_c16 || !s->bh_c16) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
#ifdef RUNQ_VNNI
            int kmax = p->hidden_dim > s->xs ? p->hidden_dim : s->xs;
            s->kq_scr_bytes = (KQ512_SCRATCH(kmax) + 63) / 64 * 64;
            s->kq_scr = aligned_alloc(64, s->kq_scr_bytes * s->att_threads);
            if (!s->kq_scr) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
#endif
        }
        s->att_b = calloc((size_t)s->att_threads * s->att_rows * kv_mul * s->att_nb * ATT_P, sizeof(float));
        s->att_pb = calloc((size_t)s->att_threads * s->att_rows * kv_mul * s->att_nb * ATT_PSTRIDE(p->head_dim), sizeof(float));
        s->rope_b = calloc(nb * p->head_dim, sizeof(float));
        if (!s->rope_b || !s->k_b || !s->v_b || !s->x_b || !s->xb_b || !s->q_b || !s->hb_b || !s->hb2_b || !s->xq_b.q || !s->xq_b.s ||
            !s->hq_b.q || !s->hq_b.s || !s->att_b || !s->att_pb) {
            fprintf(stderr, "malloc failed!\n");
            exit(EXIT_FAILURE);
        }
    }

    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(all_heads_dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->xq = (QuantizedTensor) { .q = calloc(all_heads_dim, sizeof(int8_t)), .s = calloc(all_heads_dim / GS, sizeof(float)) };
    s->hq = (QuantizedTensor) { .q = calloc(p->hidden_dim, sizeof(int8_t)), .s = calloc(p->hidden_dim / GS, sizeof(float)) };
    s->q = calloc(all_heads_dim, sizeof(float));
    s->xq_corr = calloc((all_heads_dim > p->dim ? all_heads_dim : p->dim) / GS + 1, sizeof(int32_t));
    s->hq_corr = calloc(p->hidden_dim / GS + 1, sizeof(int32_t));
    s->aq = (QuantizedTensor) { .q = calloc(all_heads_dim, sizeof(int8_t)), .s = calloc(all_heads_dim / GS, sizeof(float)) };
    s->aq_corr = calloc(all_heads_dim / GS + 1, sizeof(int32_t));
    if (!s->aq.q || !s->aq.s || !s->aq_corr) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    if (KQ) {
        s->aq_xf = calloc(all_heads_dim / GS + 1, sizeof(float));
        s->aq_c16 = calloc(all_heads_dim / 16 + 1, sizeof(int32_t));
        if (!s->aq_xf || !s->aq_c16) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
        s->xq_xf = calloc((all_heads_dim > p->dim ? all_heads_dim : p->dim) / GS + 1, sizeof(float));
        s->hq_xf = calloc(p->hidden_dim / GS + 1, sizeof(float));
        s->xq_c16 = calloc((all_heads_dim > p->dim ? all_heads_dim : p->dim) / 16 + 1, sizeof(int32_t));
        s->hq_c16 = calloc(p->hidden_dim / 16 + 1, sizeof(int32_t));
        if (!s->xq_xf || !s->hq_xf || !s->xq_c16 || !s->hq_c16) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    }
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc((size_t)p->n_heads * s->att_nb * ATT_P, sizeof(float));
    s->att_part = calloc((size_t)p->n_heads * s->att_nb * ATT_PSTRIDE(p->head_dim), sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->rope = calloc(p->head_dim, sizeof(float));
    // phase / counter capacity: decode 4 / layer, 2 counters per (layer, kv head); prefill 9 / layer,
    // a counter per (layer, attention row block)
    int ncnt = 2 * p->n_kv_heads > s->pf_nrb ? 2 * p->n_kv_heads : s->pf_nrb;
    rq_sync_init(&s->sync, 10 * p->n_layers + 8, p->n_layers * ncnt + 64);
    s->key_cache = alloc_big(s->kv_size * sizeof(float));
    s->value_cache = alloc_big(s->kv_size * sizeof(float));

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->hb || !s->hb2 || !s->q || !s->xq_corr || !s->hq_corr || !s->k || !s->v || !s->att || !s->att_part || !s->logits || !s->rope || !s->key_cache || !s->value_cache) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->hb);
    free(s->hb2);
    free(s->xq.q);
    free(s->xq.s);
    free(s->hq.q);
    free(s->hq.s);
    free(s->q);
    free(s->xq_corr);
    free(s->hq_corr);
    free(s->xq_xf);
    free(s->hq_xf);
    free(s->xq_c16);
    free(s->hq_c16);
    free(s->aq.q);
    free(s->aq.s);
    free(s->aq_corr);
    free(s->aq_xf);
    free(s->aq_c16);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free_big(s->key_cache, s->kv_size * sizeof(float));
    free_big(s->value_cache, s->kv_size * sizeof(float));
    free(s->x_b);
    free(s->xb_b);
    free(s->q_b);
    free(s->k_b);
    free(s->v_b);
    free(s->hb_b);
    free(s->hb2_b);
    free(s->xq_b.q);
    free(s->xq_b.s);
    free(s->hq_b.q);
    free(s->hq_b.s);
    free(s->att_b);
    free(s->att_pb);
    free(s->bq_corr);
    free(s->bq_xf);
    free(s->bq_c16);
    free(s->bh_corr);
    free(s->bh_xf);
    free(s->bh_c16);
    free(s->kq_scr);
    free(s->att_part);
    free(s->rope_b);
    free(s->rope);
    rq_sync_free(&s->sync);
}

// ----------------------------------------------------------------------------
// Quantization functions

void quantize(QuantizedTensor *qx, float *x, int n) {
    for (int group = 0; group < n / GS; group++) {
        // find the max absolute value in the current group
        float wmax = 0;
        for (int i = 0; i < GS; i++) {
            float val = fabs(x[group * GS + i]);
            if (val > wmax)
                wmax = val;
        }

        // calculate and write the scaling factor
        float scale = wmax / 127.0f;
        qx->s[group] = scale;
        if (!(wmax > 0)) { // all-zero group: q = 0 (was 0/0 = NaN cast to int8, UB)
            memset(qx->q + group * GS, 0, GS);
            continue;
        }
        // (keep `/ scale` as is: under -Ofast it becomes a multiply by the reciprocal, and any
        // rewrite of this loop changes the rounding of a few values, i.e. the outputs)

        // calculate and write the quantized values
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale; // scale
            int8_t quantized = (int8_t) round(quant_value); // round and clamp
            qx->q[group * GS + i] = quantized;
        }
    }
}

// (fp16_to_f32: kquants.h)

static int q8_from_blocks(int8_t *q, float *s, const uint8_t *blk, size_t nblk) {
    // nblk GGUF Q8_0 blocks -> 32 * nblk int8 values + nblk fp32 scales (exact); 0 if a scale is
    // inf / NaN (a corrupt file)
    int ok = 1;
    for (size_t i = 0; i < nblk; i++, blk += Q8_0_BLOCK) {
        uint16_t h;
        memcpy(&h, blk, 2);
        ok &= (h & 0x7c00) != 0x7c00;
        s[i] = fp16_to_f32(h);
        memcpy(q + 32 * i, blk + 2, 32);
    }
    return ok;
}

static void embed_row(float *x, const QuantizedTensor *t, int token, int dim) {
    // row `token` of the Q8_0 embedding table t, dequantized with upstream's dequantize()
    // expression, x[i] = q[i] * s[i / GS] (exact int8 -> float, one rounding): the same floats
    // as the whole dequantized table upstream kept in memory. GGUF: from its blocks, the same
    // expression with the fp16 scale as a float (dim % 32 == 0, checked at load)
    size_t e0 = (size_t)token * dim;
    if (t->kp) { // GGUF Q4_K / Q6_K: ggml's dequantization of the row's blocks (dim % 256 == 0)
        kq_dequant_row(t->type, t->b + (size_t)token * (dim / QK_K) * kq_block_bytes(t->type), dim, x);
        return;
    }
    if (t->b) {
        const uint8_t *blk = t->b + e0 / 32 * Q8_0_BLOCK;
        for (int g = 0; g < dim / 32; g++, blk += Q8_0_BLOCK) {
            uint16_t h;
            memcpy(&h, blk, 2);
            float s = fp16_to_f32(h);
            for (int i = 0; i < 32; i++)
                x[g * 32 + i] = (int8_t)blk[2 + i] * s;
        }
        return;
    }
    if (dim % GS == 0) {
        const int8_t *q = t->q + e0;
        const float *s = t->s + e0 / GS;
        for (int g = 0; g < dim / GS; g++)
            for (int i = g * GS; i < (g + 1) * GS; i++)
                x[i] = q[i] * s[g];
    } else {
        for (int i = 0; i < dim; i++)
            x[i] = t->q[e0 + i] * t->s[(e0 + i) / GS];
    }
}

/* initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr */
QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) {
    QuantizedTensor *res = malloc(n * sizeof(QuantizedTensor));

    for (int i = 0; i < n; i++) {
        // map quantized int8 values
        res[i].q = (int8_t*)*ptr;
        *ptr = (int8_t*)*ptr + size_each;
        // map scale factors
        res[i].s = (float*)*ptr;
        *ptr = (float*)*ptr + size_each / GS;
        res[i].p = NULL;
        res[i].b = NULL;
        res[i].type = GGML_Q8_0;
        res[i].kp = NULL;
    }
    return res;
}

void memory_map_weights(TransformerWeights *w, Config *p, void *ptr) {
    // first are the parameters that are kept in fp32 (the rmsnorm (1D) weights)
    float *fptr = (float*) ptr; // cast our pointer to float*

    w->rms_att_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_ffn_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_final_weight = fptr;
    fptr += p->dim;
    w->q_norm_weights = fptr;
    fptr += p->n_layers * p->head_dim;
    w->k_norm_weights = fptr;
    fptr += p->n_layers * p->head_dim;

    // now read all the quantized weights
    ptr = (void *)fptr; // now cast the pointer back to void*
    w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);
    // (the token embeddings are dequantized per row at lookup, embed_row; upstream dequantized
    // the whole table here: 622 MB of floats, ~0.3 s of page faults at every start)

    w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * p->head_dim));
    w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * p->head_dim));
    w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * p->head_dim));
    w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * p->head_dim) * p->dim);

    w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
    w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);

    w->wcls = p->shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);
}

static QuantizedTensor *gguf_matrices(const Gguf *g, int n, const char *fmt, int d, int k) {
    // n matrices (d, k) (fmt: name with the layer number, or a plain name for n = 1), each Q8_0,
    // Q4_K or Q6_K (dispatch per tensor: Q4_K_M mixes them within a role): their blocks in the
    // mapping; q / s / p / kp are set by prepare_weights
    QuantizedTensor *res = calloc(n, sizeof(QuantizedTensor));
    if (!res) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    for (int l = 0; l < n; l++) {
        char name[96];
        snprintf(name, sizeof(name), fmt, l);
        const GgufTensor *t = gguf_tensor(g, name);
        uint32_t type = t ? t->type : GGML_Q8_0; // (missing: gguf_weight reports it)
        if (type != GGML_Q8_0 && type != GGML_Q4_K && type != GGML_Q6_K) {
            char msg[160];
            snprintf(msg, sizeof(msg), "tensor %%s has ggml type %u, need Q8_0 (8), Q4_K (12) or Q6_K (14)", type);
            gguf_fail(g, msg, name);
        }
        res[l].b = gguf_weight(g, name, type, k, d);
        res[l].type = (int)type;
        if (type != GGML_Q8_0) {
            // the repacked layouts: 16-row blocks, 256-column super-blocks (activation groups of 32)
            if (d % KQ_ROWS || k % QK_K || GS != 32) gguf_fail(g, "tensor %s: K-quant shape not supported (rows % 16, columns % 256)", name);
            KQ = 1;
        }
    }
    return res;
}

void gguf_map_weights(TransformerWeights *w, Config *p, const Gguf *g) {
    // tensor names of llama.cpp's convert_hf_to_gguf.py for qwen3 (no Q/K row permutation: that
    // is done for the llama architecture only; checked by the logits test against HF). ffn_gate /
    // ffn_up / ffn_down are w1 / w3 / w2. The norm weights (F32) are copied into one buffer in the
    // .bin layout, so that the forward pass indexes them as before
    int L = p->n_layers, dim = p->dim, hd = p->head_dim, qd = p->n_heads * hd, kvd = p->n_kv_heads * hd;
    size_t nn = (size_t)2 * L * dim + dim + (size_t)2 * L * hd;
    float *f = w->norms = malloc(nn * sizeof(float));
    if (!f) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    struct { float **dst; const char *fmt; int n, len; } nm[] = {
        { &w->rms_att_weight, "blk.%d.attn_norm.weight", L, dim }, { &w->rms_ffn_weight, "blk.%d.ffn_norm.weight", L, dim },
        { &w->rms_final_weight, "output_norm.weight", 1, dim }, { &w->q_norm_weights, "blk.%d.attn_q_norm.weight", L, hd },
        { &w->k_norm_weights, "blk.%d.attn_k_norm.weight", L, hd } };
    for (size_t i = 0; i < sizeof(nm) / sizeof(nm[0]); i++) {
        *nm[i].dst = f;
        for (int l = 0; l < nm[i].n; l++, f += nm[i].len) {
            char name[96];
            snprintf(name, sizeof(name), nm[i].fmt, l);
            memcpy(f, gguf_weight(g, name, GGML_F32, nm[i].len, 0), nm[i].len * sizeof(float));
        }
    }
    w->q_tokens = gguf_matrices(g, 1, "token_embd.weight", p->vocab_size, dim);
    w->wq = gguf_matrices(g, L, "blk.%d.attn_q.weight", qd, dim);
    w->wk = gguf_matrices(g, L, "blk.%d.attn_k.weight", kvd, dim);
    w->wv = gguf_matrices(g, L, "blk.%d.attn_v.weight", kvd, dim);
    w->wo = gguf_matrices(g, L, "blk.%d.attn_output.weight", dim, qd);
    w->w1 = gguf_matrices(g, L, "blk.%d.ffn_gate.weight", p->hidden_dim, dim);
    w->w2 = gguf_matrices(g, L, "blk.%d.ffn_down.weight", dim, p->hidden_dim);
    w->w3 = gguf_matrices(g, L, "blk.%d.ffn_up.weight", p->hidden_dim, dim);
    w->wcls = p->shared_classifier ? w->q_tokens : gguf_matrices(g, 1, "output.weight", p->vocab_size, dim);
}

static void *runq_data = NULL; // the checkpoint mapping (read_checkpoint), for read_at

static void read_at(int fd, void *dst, size_t n, const void *mapped) {
    // n bytes of the checkpoint at the file offset of `mapped` (a pointer into its mapping) into
    // dst with pread: the file pages are copied from the page cache without being mapped into the
    // process (no page faults, not part of our resident size, nothing to drop afterwards). Falls
    // back to copying from the mapping
    size_t done = 0;
#ifndef _WIN32
    off_t off = (off_t)((const char *)mapped - (const char *)runq_data);
    while (done < n && fd >= 0) {
        ssize_t r = pread(fd, (char *)dst + done, n - done, off + (off_t)done);
        if (r <= 0) break;
        done += (size_t)r;
    }
#else
    (void)fd;
#endif
    if (done < n)
        memcpy((char *)dst + done, (const char *)mapped + done, n - done);
}

typedef struct { QuantizedTensor *t; int d, k, bpi; size_t off; long it0; } PrepItem; // a matrix of prepare_weights

typedef struct {
    const PrepItem *job; long nit;
    uint8_t *packed; size_t total, pg_chunk; long npg;
    size_t rows_max; int kmax, gguf, pack, fd;
    long next_pg, next_it; int bad; // shared: claim counters, any corrupt scale
} PrepJob;

static void prepare_worker(void *arg, int tid, int nt) {
    // prepare_weights' parallel part: pre-fault the destination (2 MB per item), then convert the
    // items (runs of 16-row blocks of one matrix), both claimed from shared counters
    (void)tid; (void)nt;
    PrepJob *P = arg;
    const PrepItem *job = P->job;
    size_t rows_max = P->rows_max;
    int kmax = P->kmax, gguf = P->gguf, pack = P->pack, fd = P->fd, bad = 0;
#if !defined _WIN32 && defined MADV_POPULATE_WRITE
    for (long c; (c = __atomic_fetch_add(&P->next_pg, 1, __ATOMIC_RELAXED)) < P->npg;) {
        size_t a = (size_t)c * P->pg_chunk, e = a + P->pg_chunk < P->total ? a + P->pg_chunk : P->total;
        madvise(P->packed + a, e - a, MADV_POPULATE_WRITE); // (errors ignored: plain faults then)
    }
#endif
    // scratch: source rows (VNNI: q / s before packing; GGUF: the blocks, Q8_0 the largest)
    int8_t *qb = malloc(rows_max * kmax);
    float *sb = malloc(rows_max * (kmax / GS) * sizeof(float));
    uint8_t *bb = gguf ? malloc(rows_max * (kmax / 32) * Q8_0_BLOCK) : NULL;
    if (!qb || !sb || (gguf && !bb)) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    int jj = 0;
    for (long i0; (i0 = __atomic_fetch_add(&P->next_it, 4, __ATOMIC_RELAXED)) < P->nit;)
    for (long it = i0; it < i0 + 4 && it < P->nit; it++) {
        while (it >= job[jj + 1].it0) jj++; // (items in order per claim)
        while (it < job[jj].it0) jj--;
        QuantizedTensor *t = job[jj].t;
        int k = job[jj].k, ng = k / GS, nblk = (job[jj].d + 15) / 16; // (VNNI: d % 16 == 0)
        int rb0 = (int)(it - job[jj].it0) * job[jj].bpi;
        int nb = nblk - rb0 < job[jj].bpi ? nblk - rb0 : job[jj].bpi;
        size_t r0 = (size_t)rb0 * 16, nr = (size_t)nb * 16;
        if (r0 + nr > (size_t)job[jj].d) nr = job[jj].d - r0;
        if (t->kp) { // Q4_K / Q6_K: blocks of nr rows (k / 256 each) -> Q4P / Q6P
            size_t rowb = (size_t)(k / QK_K) * kq_block_bytes(t->type);
            read_at(fd, bb, nr * rowb, t->b + r0 * rowb);
            for (size_t i = 0; i < nr * (k / QK_K); i++)
                bad |= !kq_block_ok(t->type, bb + i * kq_block_bytes(t->type));
            kq_pack_blocks(t->type, t->kp + (size_t)rb0 * (k / QK_K) * kq_sb_bytes(t->type), bb, nb, k);
            continue;
        }
        int8_t *q = pack ? qb : t->q + r0 * k;
        float *s = pack ? sb : t->s + r0 * ng;
        if (t->b) { // GGUF (GS == 32: one block per group)
            read_at(fd, bb, nr * ng * Q8_0_BLOCK, t->b + r0 * ng * Q8_0_BLOCK);
            bad |= !q8_from_blocks(q, s, bb, nr * ng);
        } else {
            read_at(fd, q, nr * k, t->q + r0 * k);
            read_at(fd, s, nr * ng * sizeof(float), t->s + r0 * ng);
        }
#ifdef RUNQ_VNNI
        if (pack)
            q8p_pack_blocks(t->p + (size_t)rb0 * ng * Q8P_GROUP(GS), qb, sb, nb, k, GS);
#endif
    }
    free(qb);
    free(sb);
    free(bb);
    if (bad)
        __atomic_fetch_or(&P->bad, 1, __ATOMIC_RELAXED);
}

void prepare_weights(TransformerWeights *w, Config *p, int fd) {
    // Converts every matmul weight (7 per layer + the classifier) once at load, into one allocation,
    // in one parallel loop over (matrix, block of rows) items:
    //   VNNI build: repacked into the Q8P layout of kernels_avx512.h (group size GS: 64 for .bin,
    //     32 for GGUF; the same size as the Q8_0 data with fp32 scales)
    //   generic kernels on a GGUF file: its Q8_0 blocks (fp16 scale + 32 int8) into the q / s
    //     arrays the .bin format has (fp16 -> fp32 is exact)
    //   generic kernels on a .bin file: nothing to do, q / s are the mapping
    //   GGUF Q4_K / Q6_K tensors (any build): repacked into the Q4P / Q6P layouts of kquants.h
    // The source rows are read with pread (read_at), not through the mapping: only the norms and
    // the token embedding rows looked up later (the classifier is shared with them) are ever
    // mapped. The destination is pre-faulted in parallel (MADV_POPULATE_WRITE, 2 MB per call).
    // (On the thread pool of threads.h, like the forward passes: no second set of threads)
    // Was: faults on both sides + one MADV_DONTNEED of the mapped source per tensor, ~90 ms; now
    // ~25 ms (12 threads)
    int L = p->n_layers, dim = p->dim, hid = p->hidden_dim;
    int qd = p->n_heads * p->head_dim, kvd = p->n_kv_heads * p->head_dim;
    struct { QuantizedTensor *t; int n, d, k; } m[] = {
        { w->wq, L, qd, dim }, { w->wk, L, kvd, dim }, { w->wv, L, kvd, dim }, { w->wo, L, dim, qd },
        { w->w1, L, hid, dim }, { w->w2, L, dim, hid }, { w->w3, L, hid, dim }, { w->wcls, 1, p->vocab_size, dim } };
    int nm = sizeof(m) / sizeof(m[0]);
    int gguf = w->wq[0].b != NULL, pack = 0;
#ifdef RUNQ_VNNI
    pack = 1;
    for (int i = 0; i < nm && pack; i++)
        if (!q8p_ok(m[i].d, m[i].k, GS)) {
            fprintf(stderr, "warning: weights not packable for the VNNI kernels (GS %d, %d x %d), using the generic ones\n", GS, m[i].d, m[i].k);
            pack = 0;
        }
#endif
    if (!pack && !gguf)
        return;
    int nt = 0, kmax = 0;
    for (int i = 0; i < nm; i++) {
        nt += m[i].n;
        kmax = m[i].k > kmax ? m[i].k : kmax;
    }

    // the matrices in order, their place in the packed buffer, and the work items: runs of
    // bpi(k) 16-row blocks, ~64 KB of source rows each
    PrepItem *job = malloc((nt + 1) * sizeof(*job));
    if (!job) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    size_t off = 0;
    long nit = 0;
    int j = 0;
    for (int i = 0; i < nm; i++)
        for (int l = 0; l < m[i].n; l++, j++) {
            // items of 16-row blocks (the VNNI blocks; the generic conversion uses the same split)
            int bpi = 65536 / (16 * m[i].k);
            QuantizedTensor *t = m[i].t + l;
            size_t nq = (size_t)m[i].d * m[i].k, sz = nq + (size_t)m[i].d * (m[i].k / GS) * sizeof(float);
            job[j].t = t;
            job[j].d = m[i].d;
            job[j].k = m[i].k;
            job[j].bpi = bpi > 0 ? bpi : 1;
            job[j].off = off;
            job[j].it0 = nit;
            if (t->type != GGML_Q8_0) // Q4_K / Q6_K (GGUF; d % 16 == 0, k % 256 == 0 checked)
                sz = kq_size(t->type, m[i].d, m[i].k);
#ifdef RUNQ_VNNI
            else if (pack)
                sz = q8p_size(m[i].d, m[i].k, GS);
#endif
            off += (sz + 63) / 64 * 64;
            nit += ((m[i].d + 15) / 16 + job[j].bpi - 1) / job[j].bpi;
        }
    job[nt].it0 = nit;
    size_t total = off;
    w->packed = alloc_big(total);
    if (!w->packed) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    w->packed_size = total;
    for (j = 0; j < nt; j++) {
        QuantizedTensor *t = job[j].t;
        size_t nq = (size_t)job[j].d * job[j].k;
        if (t->type != GGML_Q8_0)
            t->kp = w->packed + job[j].off;
#ifdef RUNQ_VNNI
        else if (pack)
            t->p = w->packed + job[j].off;
#endif
        else { // generic kernels on GGUF: q / s arrays
            t->q = (int8_t *)(w->packed + job[j].off);
            t->s = (float *)(w->packed + job[j].off + nq); // (nq is a multiple of 16 * GS)
        }
    }
    size_t rows_max = 65536 / kmax > 16 ? 65536 / kmax : 16;
    rows_max = rows_max / 16 * 16 + 16;
    size_t pg_chunk = (size_t)2 << 20;
    PrepJob pj = { job, nit, w->packed, total, pg_chunk, (long)((total + pg_chunk - 1) / pg_chunk), rows_max, kmax,
                   gguf, pack, fd, 0, 0, 0 };
    rq_parallel(prepare_worker, &pj);
    int bad = pj.bad;
    free(job);
    if (bad) { fprintf(stderr, "Corrupt checkpoint: a weight scale (fp16 d / dmin) is inf or NaN\n"); exit(EXIT_FAILURE); }
}

void read_checkpoint(char *checkpoint, Config *config, TransformerWeights* weights, float** data, ssize_t* file_size, int ctx_length,
                     Gguf *gguf) {
    // a qwen3.c .bin checkpoint or a GGUF file (Q8_0, Q4_K_M), by magic; gguf->map stays NULL for a .bin
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open checkpoint %s\n", checkpoint); exit(EXIT_FAILURE); }

    #if defined _WIN32
        _fseeki64(file, 0, SEEK_END); // move file pointer to end of file
        *file_size = _ftelli64(file); // get the file size, in bytes
    #else
        fseek(file, 0, SEEK_END); // move file pointer to end of file
        *file_size = ftell(file); // get the file size, in bytes
    #endif
    if (*file_size < 256) { fprintf(stderr, "File %s is too small for a checkpoint\n", checkpoint); exit(EXIT_FAILURE); }

    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, fileno(file), 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    runq_data = *data;
    memset(gguf, 0, sizeof(*gguf));
    memset(weights, 0, sizeof(*weights));

    uint32_t magic;
    memcpy(&magic, *data, 4);
    if (magic == GGUF_MAGIC) {
        gguf->map = (const uint8_t *)*data;
        gguf->size = (uint64_t)*file_size;
        gguf->path = checkpoint;
        gguf_parse(gguf);
        gguf_config(gguf, config);
        GS = config->group_size;
        gguf_map_weights(weights, config, gguf);
    } else {
        // checkpoint format is 256-byte header, and then the model weights
        memcpy(config, *data, sizeof(Config));
        if (config->magic_number != 0x616a6331) { fprintf(stderr, "File %s is not a qwen3.c checkpoint or a GGUF file\n", checkpoint); exit(EXIT_FAILURE); }
        if (config->version != 1) { fprintf(stderr, "Checkpoint %s is version %d, need version 1\n", checkpoint, config->version); exit(EXIT_FAILURE); }

        GS = config->group_size; // set as global, as it will be used in many places

        void *weights_ptr = ((char *)*data) + 256; // skip the header (256 bytes)
        memory_map_weights(weights, config, weights_ptr);
    }
    if (ctx_length != 0 && ctx_length <= config->seq_len)
        config->seq_len = ctx_length;
    prepare_weights(weights, config, fileno(file));
    fclose(file);
}

void build_transformer(Transformer *t, char *checkpoint_path, int ctx_length, int nb_max) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->data, &t->file_size, ctx_length, &t->gguf);
    // allocate the RunState buffers (batched prefill buffers for nb_max rows if nb_max > 1)
    malloc_run_state(&t->state, &t->config, nb_max);
}

void free_transformer(Transformer *t) {
    // free QuantizedTensors
    free(t->weights.q_tokens);
    free(t->weights.wq);
    free(t->weights.wk);
    free(t->weights.wv);
    free(t->weights.wo);
    free(t->weights.w1);
    free(t->weights.w2);
    free(t->weights.w3);
    if(t->weights.wcls != t->weights.q_tokens) free(t->weights.wcls);
    free_big(t->weights.packed, t->weights.packed_size);
    free(t->weights.norms);
    free(t->gguf.kv);
    free(t->gguf.t);
    // close the memory mapping
    if (t->data != MAP_FAILED) munmap(t->data, t->file_size);
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float *o, float *x, float *weight, int size) {
    // calculate sum of squares
    float ss = 0;
    for (int j = 0; j < size; j++)
        ss += x[j] * x[j];

    ss = 1.0f / sqrtf((ss / size) + 1e-6f);

    // normalize and scale
    for (int j = 0; j < size; j++)
        o[j] = weight[j] * (ss * x[j]);
}

void softmax(float *x, int size) {
    // find max value (for numerical stability)
    float max_val = 0;
    for (int i = 0; i < size; i++)
        if (x[i] > max_val)
            max_val = x[i];

    // exp and sum
    float sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    // normalize
    for (int i = 0; i < size; i++)
        x[i] /= sum;
}

// Q8_0 dot product of one weight row with one quantized activation row (n values, n/GS groups):
// exact int32 sum per group of GS, then accumulated in float, group by group in order.
// Shared by matmul (GEMV) and matmul_batch (GEMM) so both give bit-identical results.
static inline float dot_q8(const int8_t *wq, const float *ws, const int8_t *xq, const float *xs, int n) {
    float val = 0;
    for (int j = 0; j <= n - GS; j += GS) {
        int32_t ival = 0;
        for (int k = 0; k < GS; k++)
            ival += xq[j + k] * wq[j + k];

        val += ((float) ival) * ws[j / GS] * xs[j / GS];
    }
    return val;
}

// Decode GEMV, in pieces: forward() runs all of a token in one parallel region, with the rows
// of each matrix split into work items. matvec_prep: the VNNI kernel's activation corrections for
// x[i0 .. i0+n) (one per GS-group; nothing to do for the generic kernel), and with K-quant weights
// (xf != NULL) the activation group sums xf of kquants.h, plus for the AVX-512 K-quant GEMV the
// -32 * sums of 16 (c16, Q6_K); matvec_rows: rows i0..i1-1
static inline void matvec_prep(int32_t *corr, float *xf, int32_t *c16, QuantizedTensor *x, int i0, int n) {
#ifdef RUNQ_VNNI
    q8p_gemv_corr(corr + i0 / GS, x->q + i0, n, GS);
    if (c16)
        kq512_c16(c16 + i0 / 16, x->q + i0, n);
#else
    (void)corr; (void)c16;
#endif
    if (xf)
        kq_act_prep(xf + i0 / GS, x->q + i0, x->s + i0 / GS, n);
}

static inline void matvec_rows(float *xout, QuantizedTensor *x, const int32_t *corr, const float *xf, const int32_t *c16,
                               QuantizedTensor *w, int n, int i0, int i1) {
    // xout[i] = W (d,n) row i . x (n,) for i0 <= i < i1 (multiples of 16, as the VNNI blocks)
    if (w->kp) { // Q4_K / Q6_K, 16-row blocks (kquants.h; AVX-512: kernels_avx512.h)
#ifdef RUNQ_VNNI
        kq512_gemv_rows(w->type, xout, x->q, x->s, xf, c16, w->kp, n, i0 / KQ_ROWS, i1 / KQ_ROWS);
#else
        (void)c16;
        for (int i = i0; i < i1; i += KQ_ROWS)
            kq_dot(w->type, xout + i, 0, w->kp + (size_t)(i / KQ_ROWS) * (n / QK_K) * kq_sb_bytes(w->type), n, 1,
                   x->q, 0, x->s, 0, xf, 0);
#endif
        return;
    }
#ifdef RUNQ_VNNI
    if (w->p) { q8p_gemv_rows(xout, x->q, x->s, corr, w->p, n, GS, i0 / 16, i1 / 16); return; }
#else
    (void)corr;
#endif
    for (int i = i0; i < i1; i++) {
        size_t in = (size_t)i * n;
        xout[i] = dot_q8(w->q + in, w->s + in / GS, x->q, x->s, n);
    }
}

// The scale chain below must round exactly like dot_q8 as gcc -Ofast compiles it there,
// val += (ws * xs) * (float) ival as two multiplies and an add. In the 4-row kernel gcc would
// contract it into an FMA (one rounding less), and the batched logits would then differ from the
// sequential ones (enough to flip a greedy token). The batched-vs-sequential test checks this
#if defined(__GNUC__) && !defined(__clang__)
    #define NO_FP_CONTRACT __attribute__((optimize("fp-contract=off")))
#else
    #define NO_FP_CONTRACT
#endif

// rows of W per work item in matmul_batch: a block of MB_ROWS rows (MB_ROWS * k bytes,
// 4-12 KB here) stays in L1 while all nb activation rows stream past it
#define MB_ROWS 4

NO_FP_CONTRACT static inline void dot_q8_x4(float *o, const int8_t *w0, const float *s0, int k, const int8_t *xq, const float *xs) {
    // o[r] = dot_q8(row r of the 4 consecutive W rows at w0, x) for r < 4: each activation value
    // is loaded once for 4 rows (~35% faster than 4 dot_q8 calls). Per output, the same
    // arithmetic in the same order as dot_q8 (see NO_FP_CONTRACT)
    const int8_t *w1 = w0 + k, *w2 = w1 + k, *w3 = w2 + k;
    const float *s1 = s0 + k / GS, *s2 = s1 + k / GS, *s3 = s2 + k / GS;
    float v0 = 0, v1 = 0, v2 = 0, v3 = 0;
    for (int j = 0; j <= k - GS; j += GS) {
        int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (int kk = 0; kk < GS; kk++) {
            int32_t xv = xq[j + kk];
            a0 += xv * w0[j + kk]; a1 += xv * w1[j + kk]; a2 += xv * w2[j + kk]; a3 += xv * w3[j + kk];
        }
        v0 += ((float) a0) * s0[j / GS] * xs[j / GS];
        v1 += ((float) a1) * s1[j / GS] * xs[j / GS];
        v2 += ((float) a2) * s2[j / GS] * xs[j / GS];
        v3 += ((float) a3) * s3[j / GS] * xs[j / GS];
    }
    o[0] = v0; o[1] = v1; o[2] = v2; o[3] = v3;
}

typedef struct { float *out; int ldo; QuantizedTensor *w; int d; } MatJob; // one GEMM of a Gemm phase

// One batched-matmul phase of forward_batch: up to 3 GEMMs on the same nb quantized activation rows
// (q / k / v, w1 / w3), out[b * ldo + i] = W[i, :] . x[b, :] (W d x k; x row b at x->q + b * ldx,
// its k / GS scales at x->s + b * ldx / GS), as work items of one phase:
//   AVX-512 build: (32-row tile, chunk of nbi activation rows), Q8P (q8p_gemm_item) or Q4P / Q6P
//     (kq512_gemm_item); the items of a tile are claimed together (chunk = nchunk), so the weight
//     tile is read (K-quants: unpacked) once per claim and reused from L1/L2
//   generic: blocks of MB_ROWS rows (Q8_0, dot_q8_x4) or of 16 rows (K-quants, kq_dot) over all nb rows
// Each output is computed exactly as in the decode GEMV (matvec_rows), so the batched path matches the
// sequential one bit for bit. Activation prep (row stride k / GS, k / GS, k / 16): corr (AVX-512 Q8P),
// xf (K-quants), c16 (AVX-512 Q6_K), computed per row when the row is quantized (act_prep_row)
typedef struct {
    int nj; MatJob job[3];
    int it0[4];          // items of job j: it0[j] .. it0[j + 1] - 1
    int tiled[3];        // job j in tiles x chunks (else blocks over all rows)
    int nbi, nchunk;     // tiled: activation rows per chunk, chunks
    int chunk, unit;     // items per claim; ranges of whole units (tiled: a tile's chunks)
    QuantizedTensor *x; int ldx, nb, k;
    const int32_t *corr; const float *xf; const int32_t *c16;
} Gemm;

static void gemm_plan(Gemm *g, int nj, const MatJob *job, QuantizedTensor *x, int ldx, int nb, int k, const int32_t *corr,
                      const float *xf, const int32_t *c16, int nt) {
    // (computed by every thread of the job: the same plan everywhere; results never depend on it)
    g->nj = nj; g->x = x; g->ldx = ldx; g->nb = nb; g->k = k; g->corr = corr; g->xf = xf; g->c16 = c16;
    int ntile = 0, all_tiled = 1;
    for (int j = 0; j < nj; j++) {
        g->job[j] = job[j];
        g->tiled[j] = 0;
#ifdef RUNQ_VNNI
        g->tiled[j] = job[j].w->kp || job[j].w->p;
#endif
        all_tiled &= g->tiled[j];
        if (g->tiled[j])
            ntile += (job[j].d / 16 + 1) / 2;
    }
    g->nbi = nb;
    g->nchunk = 1;
#ifdef RUNQ_VNNI
    if (ntile) {
        g->nbi = q8p_nb(nb, ntile, nt);
        g->nchunk = (nb + g->nbi - 1) / g->nbi;
    }
#endif
    int n = 0;
    for (int j = 0; j < nj; j++) {
        g->it0[j] = n;
        n += g->tiled[j] ? (job[j].d / 16 + 1) / 2 * g->nchunk : job[j].w->kp ? job[j].d / KQ_ROWS : (job[j].d + MB_ROWS - 1) / MB_ROWS;
    }
    g->it0[nj] = n;
    // each thread's range in whole tiles, taken one tile (its chunks) at a time; generic: a few blocks
    g->unit = all_tiled ? g->nchunk : 1;
    g->chunk = all_tiled ? g->nchunk : (n / (8 * nt) > 1 ? n / (8 * nt) : 1);
}

NO_FP_CONTRACT static void gemm_item(const Gemm *g, int it, uint8_t *scr, const void **last) {
    // work item it of the phase (scr / last: this thread's K-quant scratch, kq512_gemm_item)
    int j = 0;
    while (it >= g->it0[j + 1]) j++;
    const MatJob *jb = &g->job[j];
    QuantizedTensor *w = jb->w, *x = g->x;
    int i = it - g->it0[j], k = g->k, ldx = g->ldx, nb = g->nb;
#ifdef RUNQ_VNNI
    if (g->tiled[j]) {
        int t = i / g->nchunk, ch = i % g->nchunk;
        if (w->kp)
            kq512_gemm_item(jb->out, jb->ldo, w->kp, w->type, jb->d, t, ch, g->nbi, x->q, x->s, ldx, nb, k, g->xf, g->c16, scr, last);
        else
            q8p_gemm_item(jb->out, jb->ldo, w->p, jb->d, t, ch, g->nbi, x->q, x->s, ldx, nb, k, GS, g->corr);
        return;
    }
#endif
    (void)scr; (void)last;
    int ng = k / GS;
    if (w->kp) { // K-quants, generic: one 16-row block over all rows, KQ_NC rows per kq_dot call
        const uint8_t *wb = w->kp + (size_t)i * (k / QK_K) * kq_sb_bytes(w->type);
        for (int b = 0; b < nb; b += KQ_NC) {
            int nc = nb - b < KQ_NC ? nb - b : KQ_NC;
            kq_dot(w->type, jb->out + (size_t)b * jb->ldo + i * KQ_ROWS, jb->ldo, wb, k, nc, x->q + (size_t)b * ldx, ldx,
                   x->s + (size_t)b * ldx / GS, ldx / GS, g->xf + (size_t)b * ng, ng);
        }
        return;
    }
    // Q8_0, generic: MB_ROWS weight rows (in L1) over all activation rows
    int d = jb->d, i0 = i * MB_ROWS, i1 = i0 + MB_ROWS < d ? i0 + MB_ROWS : d;
    for (int b = 0; b < nb; b++) {
        const int8_t *xq = x->q + (size_t)b * ldx;
        const float *xsc = x->s + (size_t)b * ldx / GS;
        float *o = jb->out + (size_t)b * jb->ldo;
        if (i1 - i0 == 4) {
            dot_q8_x4(o + i0, w->q + (size_t)i0 * k, w->s + (size_t)i0 * k / GS, k, xq, xsc);
            continue;
        }
        for (int r = i0; r < i1; r++) {
            size_t in = (size_t)r * k;
            o[r] = dot_q8(w->q + in, w->s + in / GS, xq, xsc, k);
        }
    }
}

static void rope_row(float *cs, int head_dim, int pos) {
    // RoPE rotation of position pos: cs[j] = cos, cs[j + head_dim/2] = sin of pos * freq_j, computed
    // once per position (was: in qk_norm_rope for every layer and head, ~6% of a 451-token
    // prefill), with the original expressions, so the values are the same
    for (int j = 0; j < head_dim/2; j++) {
        float freq = powf(1e6, -(float)j / (head_dim/2));
        float cos_freq = cosf(pos * freq), sin_freq = sinf(pos * freq);
        cs[j] = cos_freq;
        cs[j + head_dim/2] = sin_freq;
    }
}

static void qk_norm_rope(float *v, int n_heads, const float *norm_w, int head_dim, const float *cs) {
    // Qwen3 QK-norm (RMSNorm per head) then NeoX-style RoPE (pairs j, j + head_dim/2), in place;
    // cs: cos / sin of this row's position (rope_row)
    for (int h = 0; h < n_heads; h++) {
        float *q = v + h * head_dim;

        rmsnorm(q, q, (float *)norm_w, head_dim);
        for (int j = 0; j < head_dim/2; j++) {
            float cos_freq = cs[j], sin_freq = cs[j + head_dim/2];

            float x = q[j]; // real part
            float y = q[j + head_dim/2]; // imag part

            q[j] = x * cos_freq - y * sin_freq; // new real
            q[j + head_dim/2] = x * sin_freq + y * cos_freq; // new imag
        }
    }
}

// Attention. The arithmetic is defined in kernels_avx512.h (with the KV cache tile layout):
// AVX-512 builds use att512_kv there, the others attention_vec below, the same operations in the
// same order (bit-identical results; tested by make test's kernels/* with -DRUNQ_GENERIC).
// gcc: no FP contraction or reassociation in this C version (it would round differently)
#if defined(__GNUC__) && !defined(__clang__)
    #define ATT_EXACT __attribute__((optimize("fp-contract=off", "no-associative-math", "no-reciprocal-math")))
#else
    #define ATT_EXACT
#endif
#ifdef FP_FAST_FMAF
    #define ATT_FMA(a, b, c) fmaf(a, b, c)
#else
    #define ATT_FMA(a, b, c) ((a) * (b) + (c)) // no fast FMA: rounds differently from AVX-512 builds
#endif

ATT_EXACT static inline float exp_att(float x) {
    // the expf of the attention softmax (x <= 0), operation for operation as att512_exp
    x = x > ATT_EXP_MIN ? x : ATT_EXP_MIN;
    float n = rintf(x * ATT_LOG2E);
    float r = ATT_FMA(n, -ATT_LN2_HI, x);
    r = ATT_FMA(n, -ATT_LN2_LO, r);
    float z = r * r;
    float y = ATT_P0;
    y = ATT_FMA(y, r, ATT_P1);
    y = ATT_FMA(y, r, ATT_P2);
    y = ATT_FMA(y, r, ATT_P3);
    y = ATT_FMA(y, r, ATT_P4);
    y = ATT_FMA(y, r, ATT_P5);
    y = ATT_FMA(y, z, r);
    y = y + 1.0f;
    uint32_t bits;
    memcpy(&bits, &y, 4);
    bits += (uint32_t)(int32_t)n << 23;
    memcpy(&y, &bits, 4);
    return y;
}

ATT_EXACT static void attention_vec(const float *q, float *out, float *ml, int n, const float *kt, const float *vt,
                                    size_t ts, int hd, float scale, float *S) {
    // one query vector over positions 0..n-1 of one kv head: kt / vt = its tile 0 in the layer's
    // K / V cache, ts = floats from one tile of this head to the next; S: n (rounded up to 16) floats.
    // ml != NULL: one block's partial result (out unnormalized, ml[0] = max score, ml[1] = sum)
    int nt = (n + ATT_T - 1) / ATT_T;
    for (int tb = 0; tb < nt; tb++) {
        const float *k = kt + (size_t)tb * ts;
        float a[ATT_T], c[ATT_T];
        for (int j = 0; j < ATT_T; j++)
            a[j] = 0.0f; // (not used: chunk 0 is assigned)
        for (int i0 = 0; i0 < hd; i0 += ATT_DC) {
            for (int j = 0; j < ATT_T; j++)
                c[j] = 0.0f;
            for (int i = i0; i < i0 + ATT_DC && i < hd; i++)
                for (int j = 0; j < ATT_T; j++)
                    c[j] = ATT_FMA(q[i], k[i * ATT_T + j], c[j]);
            for (int j = 0; j < ATT_T; j++)
                a[j] = i0 > 0 ? a[j] + c[j] : c[j];
        }
        for (int j = 0; j < ATT_T; j++)
            S[tb * ATT_T + j] = a[j] * scale;
    }
    float m = S[0];
    for (int t = 1; t < n; t++)
        if (S[t] > m)
            m = S[t];
    float sum[ATT_T];
    for (int j = 0; j < ATT_T; j++)
        sum[j] = 0.0f;
    for (int tb = 0; tb < nt; tb++)
        for (int j = 0; j < ATT_T; j++) {
            int t = tb * ATT_T + j;
            float e = t < n ? exp_att(S[t] - m) : 0.0f;
            S[t] = e;
            sum[j] = sum[j] + e;
        }
    for (int w = ATT_T / 2; w >= 1; w /= 2) // the reduction tree of att512_sum16
        for (int j = 0; j < w; j++)
            sum[j] = sum[j] + sum[j + w];
    float inv = 1.0f / sum[0];
    if (ml) {
        ml[0] = m;
        ml[1] = sum[0];
    }
    for (int i = 0; i < hd; i++)
        out[i] = 0.0f;
    for (int t = 0; t < n; t++) {
        const float *v = vt + (size_t)(t / ATT_T) * ts + (size_t)(t % ATT_T) * hd;
        float e = S[t];
        for (int i = 0; i < hd; i++)
            out[i] = ATT_FMA(e, v[i], out[i]);
    }
    if (!ml)
        for (int i = 0; i < hd; i++)
            out[i] = out[i] * inv;
}

ATT_EXACT static void att_combine(float *out, int nb, const float *part, size_t stride, int hd) {
    // the attention of one query vector from its nb >= 1 block partial results (part + b * stride:
    // o_b[hd], m_b, l_b, block order), see the definition in kernels_avx512.h
    if (nb == 1) {
        float inv = 1.0f / part[hd + 1];
        for (int i = 0; i < hd; i++)
            out[i] = part[i] * inv;
        return;
    }
    float M = part[hd];
    for (int b = 1; b < nb; b++)
        if (part[b * stride + hd] > M)
            M = part[b * stride + hd];
    float L = 0.0f;
    for (int i = 0; i < hd; i++)
        out[i] = 0.0f;
    for (int b = 0; b < nb; b++) {
        const float *o = part + b * stride;
        float c = exp_att(o[hd] - M);
        L = ATT_FMA(c, o[hd + 1], L);
        for (int i = 0; i < hd; i++)
            out[i] = ATT_FMA(c, o[i], out[i]);
    }
    float inv = 1.0f / L;
    for (int i = 0; i < hd; i++)
        out[i] = out[i] * inv;
}

static void attention_part(int nv, const float *const *q, float *const *out, float *const *part, const int *np, int b0,
                           int b1, const float *kt, const float *vt, size_t ts, int hd, float scale, float *S) {
    // nv query vectors q[r] sharing one kv head (kt / vt: its tile 0), vector r over positions
    // 0..np[r]-1. part NULL: the whole attention into out[r] (all np <= ATT_P). Else the partial
    // results of the blocks b0..b1-1 of ATT_P positions that vector r reaches, block b at part[r] +
    // b * ATT_PSTRIDE(hd): o (unnormalized), m (max score), l (sum of exponentials).
    // S: scratch of nv * (b1 - b0) * ATT_P floats
#ifdef RUNQ_ATT512
    if (att512_ok(hd)) { att512_kv(nv, q, out, part, np, b0, b1, kt, vt, ts, hd, scale, S); return; }
#endif
    for (int r = 0; r < nv; r++) {
        if (!part) {
            attention_vec(q[r], out[r], NULL, np[r], kt, vt, ts, hd, scale, S);
            continue;
        }
        for (int b = b0; b < b1 && b * ATT_P < np[r]; b++) {
            float *o = part[r] + (size_t)b * ATT_PSTRIDE(hd);
            int n = np[r] - b * ATT_P < ATT_P ? np[r] - b * ATT_P : ATT_P;
            size_t t0 = (size_t)(b * (ATT_P / ATT_T)) * ts;
            attention_vec(q[r], o, o + hd, n, kt + t0, vt + t0, ts, hd, scale, S);
        }
    }
}

static void attention_kv(int nv, const float *const *q, float *const *out, const int *np, const float *kt,
                         const float *vt, size_t ts, int hd, float scale, float *S, float *part) {
    // nv query vectors q[r] sharing one kv head, vector r over positions 0..np[r]-1, into out[r]:
    // per block of ATT_P positions the partial results, then their combination (one block: the
    // attention directly). S: scratch of nv * (blocks of max np) * ATT_P floats; part: nv * (blocks of
    // max np) * ATT_PSTRIDE(hd) floats. Each vector's result does not depend on the others
    int nmax = 0;
    for (int r = 0; r < nv; r++)
        nmax = np[r] > nmax ? np[r] : nmax;
    if (nmax <= ATT_P) {
        attention_part(nv, q, out, NULL, np, 0, 1, kt, vt, ts, hd, scale, S);
        return;
    }
    int nbm = (nmax + ATT_P - 1) / ATT_P;
    size_t st = ATT_PSTRIDE(hd);
    float *pv[ATT_NV_MAX];
    for (int r = 0; r < nv; r++)
        pv[r] = part + (size_t)r * nbm * st;
    attention_part(nv, q, NULL, pv, np, 0, nbm, kt, vt, ts, hd, scale, S);
    for (int r = 0; r < nv; r++)
        att_combine(out[r], (np[r] + ATT_P - 1) / ATT_P, pv[r], st, hd);
}

static inline size_t kv_tile(int t, int h, int n_kv, int hd) {
    // offset (floats) of the cache tile holding position t of kv head h, inside a layer
    return ((size_t)(t / ATT_T) * n_kv + h) * ATT_T * hd;
}

static void kv_store(float *kl, float *vl, int t, int h, int n_kv, int hd, const float *k, const float *v) {
    // head h of the key / value rows k, v of position t into the layer's cache tiles kl, vl
    // (k or v may be NULL: only the other one)
    size_t o = kv_tile(t, h, n_kv, hd);
    int j = t % ATT_T;
    if (k)
        for (int i = 0; i < hd; i++)
            kl[o + (size_t)i * ATT_T + j] = k[i];
    if (v)
        memcpy(vl + o + (size_t)j * hd, v, hd * sizeof(float));
}

static void swiglu(float *hb, const float *hb2, int n) {
    // silu(hb) * hb2, silu(x) = x * sigmoid(x). The exponent is clamped: under -Ofast
    // (finite-math) expf overflowing to inf for x < -88 can give NaN; for x < -80 silu is ~0 anyway
    for (int i = 0; i < n; i++)
        hb[i] *= hb2[i] * (1.0f / (1.0f + expf(fminf(-hb[i], 80.0f))));
}

static void kv_populate_one(RunState *s, Config *p, int pos0, int n, int it) {
    // Pre-fault the KV cache tiles of positions pos0..pos0+n-1 of layer it / 2, K (it even) or V
    // (anonymous memory, untouched pages are not mapped yet): one madvise per range instead of one
    // page fault per 4 KB, the 2 * n_layers items in parallel. Otherwise these faults happen inside
    // the K/V GEMMs, serialized in the kernel: ~55 ms of a 451-token prefill (8 threads), vs ~7 ms
    // here. Linux >= 5.14; errors ignored
#if !defined _WIN32 && defined MADV_POPULATE_WRITE
    int kv_dim = p->n_kv_heads * p->head_dim, l = it / 2, kv = it % 2;
    uintptr_t pg = (uintptr_t)sysconf(_SC_PAGESIZE);
    // positions pos0.. of all heads: the tiles pos0 / ATT_T .. (pos0 + n - 1) / ATT_T
    float *c = (kv ? s->value_cache : s->key_cache) + l * (uint64_t)s->kv_len * kv_dim;
    uintptr_t a = (uintptr_t)(c + (size_t)(pos0 / ATT_T) * ATT_T * kv_dim) & ~(pg - 1);
    uintptr_t e = (uintptr_t)(c + (size_t)((pos0 + n - 1) / ATT_T + 1) * ATT_T * kv_dim);
    madvise((void *)a, e - a, MADV_POPULATE_WRITE);
#else
    (void)s; (void)p; (void)pos0; (void)n; (void)it;
#endif
}

static inline QuantizedTensor qsub(QuantizedTensor *t, int i0) {
    // the quantized values from index i0 on (a multiple of GS)
    return (QuantizedTensor) { .q = t->q + i0, .s = t->s + i0 / GS, .p = NULL };
}

// Phases of a decode job (threads.h): PH_BEGIN(n, chunk) ... [PH_EPILOGUE ...] PH_END runs phase k_
// with n items, claimed in ranges [lo, hi) of up to chunk items; the epilogue (if any) runs once, on
// the thread finishing the last item, before the phase is published; every thread then waits for it
#define PH_BEGIN(n_, chunk_) PH_BEGIN_U(n_, chunk_, 1)
#define PH_BEGIN_DYN(n_, chunk_) PH_BEGIN_U(n_, chunk_, 0) // one shared counter, items in order
#define PH_BEGIN_U(n_, chunk_, unit_) { RqIt it_ = rq_it(k_, (n_), (chunk_), (unit_), tid); int lo, hi; \
    while (rq_next(S, &it_, &lo, &hi)) {
#define PH_EPILOGUE if (rq_finish(S, &it_, hi - lo)) {
#define PH_END rq_publish(S, k_ + 1); } } rq_wait(S, ++k_); }
#define PH_END_NOEPI PH_EPILOGUE PH_END

typedef struct { Transformer *tr; int token, pos, compute_logits; } DecodeJob;

// phases of a decode job: per layer 4 (qkv + attention, wo, w1 + w3, w2), the classifier, the KV
// tile pre-fault every ATT_T positions; dependency counters: per (layer, kv head) its qkv items
// done, and its attention blocks done
static int decode_phases(Config *p) { return 4 * p->n_layers + 2; }
static int decode_counters(Config *p) { return 2 * p->n_layers * p->n_kv_heads; }

static void decode_norm(RunState *s, Config *p, TransformerWeights *w, int l, int resid) {
    // (x += xb,) rmsnorm with layer l's attention weights (l < n_layers) or the final ones (l ==
    // n_layers: into x itself, as before), quantize into xq, the GEMV activation prep
    int dim = p->dim;
    if (resid)
        for (int i = 0; i < dim; i++)
            s->x[i] += s->xb[i];
    if (l < p->n_layers) {
        rmsnorm(s->xb, s->x, w->rms_att_weight + l * dim, dim);
        quantize(&s->xq, s->xb, dim);
    } else {
        rmsnorm(s->x, s->x, w->rms_final_weight, dim);
        quantize(&s->xq, s->x, dim);
    }
    matvec_prep(s->xq_corr, s->xq_xf, s->xq_c16, &s->xq, 0, dim);
}

static void decode_worker(void *arg, int tid, int nt) {
    (void)nt;
    DecodeJob *job = arg;
    Transformer *transformer = job->tr;
    Config *p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    RqSync *S = &s->sync;
    int pos = job->pos, k_ = 0;
    int dim = p->dim, hd = p->head_dim, hidden = p->hidden_dim, nh = p->n_heads, nkv = p->n_kv_heads;
    int kv_dim = nkv * hd;
    int kv_mul = nh / nkv; // integer multiplier of the kv sharing in multiquery
    int all_heads_dim = nh * hd, ff = s->ff_rows;
    int ng = kv_mul + 2; // qkv items per kv head: its kv_mul query heads, its key head, its value head
    int nb = (pos + 1 + ATT_P - 1) / ATT_P; // attention blocks (kernels_avx512.h): items per kv head
    size_t ts = (size_t)nkv * ATT_T * hd, pst = ATT_PSTRIDE(hd);

    if (pos % ATT_T == 0) { // first position of a KV cache tile: pre-fault the tile (all layers)
        PH_BEGIN(2 * p->n_layers, 1)
        for (int it = lo; it < hi; it++)
            kv_populate_one(s, p, pos, 1, it);
        PH_END_NOEPI
    }

    // forward all the layers (layer 0's attention rmsnorm + quantize: done by forward())
    for (int l = 0; l < p->n_layers; l++) {
        uint64_t loff = l * (uint64_t)s->kv_len * kv_dim; // kv cache layer offset for convenience
        float *kl = s->key_cache + loff, *vl = s->value_cache + loff;
        int *dep = S->cnt + l * nkv; (void)dep;

        // qkv matmuls for this position, then the attention, in one phase. Items 0 .. nkv * ng - 1:
        // per kv head g its query heads (QK-norm + RoPE), its key head (QK-norm + RoPE, cache store)
        // and value head (cache store), counted in dependency counter g; then per kv head nb items,
        // one per block of ATT_P positions (its kv_mul query heads together, after the ng items of
        // its group): with one block the attention itself, else the block's partial results, which
        // the last of the nb items to finish combines (att_combine: the same definition as the
        // prefill's attention_kv); then the quantization of the outputs for wo (into aq: xq may
        // still be read)
        PH_BEGIN_DYN(nkv * ng + nkv * nb, 1)
        for (int it = lo; it < hi; it++) {
            if (it < nkv * ng) {
                int g = it / ng, r = it % ng;
                if (r < kv_mul) {
                    int h = g * kv_mul + r;
                    matvec_rows(s->q, &s->xq, s->xq_corr, s->xq_xf, s->xq_c16, w->wq + l, dim, h * hd, (h + 1) * hd);
                    qk_norm_rope(s->q + h * hd, 1, w->q_norm_weights + l * hd, hd, s->rope);
                } else if (r == kv_mul) {
                    matvec_rows(s->k, &s->xq, s->xq_corr, s->xq_xf, s->xq_c16, w->wk + l, dim, g * hd, (g + 1) * hd);
                    qk_norm_rope(s->k + g * hd, 1, w->k_norm_weights + l * hd, hd, s->rope);
                    kv_store(kl, NULL, pos, g, nkv, hd, s->k + g * hd, NULL);
                } else {
                    matvec_rows(s->v, &s->xq, s->xq_corr, s->xq_xf, s->xq_c16, w->wv + l, dim, g * hd, (g + 1) * hd);
                    kv_store(NULL, vl, pos, g, nkv, hd, NULL, s->v + g * hd);
                }
                rq_count(S, l * nkv + g, 1);
                continue;
            }
            int g = (it - nkv * ng) / nb, b = (it - nkv * ng) % nb;
            rq_wait_count(S, l * nkv + g, ng);
            const float *qv[ATT_NV_MAX];
            float *ov[ATT_NV_MAX], *pv[ATT_NV_MAX];
            int npv[ATT_NV_MAX];
            float *part = s->att_part + (size_t)g * kv_mul * s->att_nb * pst; // vector r, block b: + (r * att_nb + b) * pst
            for (int r = 0; r < kv_mul; r++) {
                qv[r] = s->q + (g * kv_mul + r) * hd;
                ov[r] = s->xb + (g * kv_mul + r) * hd;
                pv[r] = part + (size_t)r * s->att_nb * pst;
                npv[r] = pos + 1;
            }
            size_t o = kv_tile(0, g, nkv, hd);
            attention_part(kv_mul, qv, ov, nb == 1 ? NULL : pv, npv, b, b + 1, kl + o, vl + o, ts, hd, s->att_scale,
                           s->att + ((size_t)g * s->att_nb + b) * kv_mul * ATT_P);
            if (nb > 1) { // the last block of this kv head to finish combines them
                if (rq_count(S, (p->n_layers + l) * nkv + g, 1) < nb)
                    continue;
                for (int r = 0; r < kv_mul; r++)
                    att_combine(ov[r], nb, pv[r], pst, hd);
            }
            int i0 = g * kv_mul * hd;
            QuantizedTensor ag = qsub(&s->aq, i0);
            quantize(&ag, s->xb + i0, kv_mul * hd);
            matvec_prep(s->aq_corr, s->aq_xf, s->aq_c16, &s->aq, i0, kv_mul * hd);
        }
        PH_END_NOEPI

        // final matmul to get the output of the attention (16-row blocks); epilogue: residual
        // connection back into s->x, ffn rmsnorm, quantize
        PH_BEGIN(dim / 16, s->dec_chunk)
            matvec_rows(s->xb, &s->aq, s->aq_corr, s->aq_xf, s->aq_c16, w->wo + l, all_heads_dim, lo * 16, hi * 16);
        PH_EPILOGUE
            for (int i = 0; i < dim; i++)
                s->x[i] += s->xb[i];
            rmsnorm(s->xb, s->x, w->rms_ffn_weight + l * dim, dim);
            quantize(&s->xq, s->xb, dim);
            matvec_prep(s->xq_corr, s->xq_xf, s->xq_c16, &s->xq, 0, dim);
        PH_END

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x)): per item,
        // ff rows of self.w1(x) and self.w3(x), the SwiGLU non-linearity and their quantization
        PH_BEGIN(hidden / ff, 1)
        for (int i0 = lo * ff; i0 < hi * ff; i0 += ff) {
            matvec_rows(s->hb, &s->xq, s->xq_corr, s->xq_xf, s->xq_c16, w->w1 + l, dim, i0, i0 + ff);
            matvec_rows(s->hb2, &s->xq, s->xq_corr, s->xq_xf, s->xq_c16, w->w3 + l, dim, i0, i0 + ff);
            swiglu(s->hb + i0, s->hb2 + i0, ff);
            QuantizedTensor hg = qsub(&s->hq, i0);
            quantize(&hg, s->hb + i0, ff);
            matvec_prep(s->hq_corr, s->hq_xf, s->hq_c16, &s->hq, i0, ff);
        }
        PH_END_NOEPI

        // final matmul to get the output of the ffn; epilogue: its residual, then the next layer's
        // attention rmsnorm + quantize (or the final rmsnorm for the classifier; none without logits)
        PH_BEGIN(dim / 16, s->dec_chunk)
            matvec_rows(s->xb, &s->hq, s->hq_corr, s->hq_xf, s->hq_c16, w->w2 + l, hidden, lo * 16, hi * 16);
        PH_EPILOGUE
            if (l + 1 < p->n_layers || job->compute_logits)
                decode_norm(s, p, w, l + 1, 1);
        PH_END
    }

    if (job->compute_logits) { // classifier into logits
        int nblk = (p->vocab_size + 15) / 16;
        PH_BEGIN(nblk, s->cls_chunk)
            matvec_rows(s->logits, &s->xq, s->xq_corr, s->xq_xf, s->xq_c16, w->wcls, dim, lo * 16,
                        hi * 16 < p->vocab_size ? hi * 16 : p->vocab_size);
        PH_END_NOEPI
    }
}

float *forward(Transformer *transformer, int token, int pos, int compute_logits) {
    // compute_logits = 0 skips the final rmsnorm + classifier (prompt tokens whose
    // logits are discarded) and returns NULL; the KV cache is filled either way.
    // One parallel job for the whole token (decode_worker), in phases (threads.h), 4 per layer:
    //   [q, k, v, one item per head, with its QK-norm + RoPE and the K/V cache store; attention per
    //   kv head (once the 4 items of its heads are done: dependency counters) + quantize] ->
    //   [wo; epilogue: residual + rmsnorm + quantize] -> [w1 + w3 + SwiGLU + quantize per ff_rows
    //   hidden rows] -> [w2; epilogue: residual + the next layer's rmsnorm + quantize], then the
    //   classifier: 4 * 28 + 1 = 113 synchronization points per token (was 7 * 28 + 2 = 198 OpenMP
    //   barriers). Every output is computed whole by one thread, by the same functions as before
    //   (same results)
    Config *p = &transformer->config;
    RunState* s = &transformer->state;

    // the token embedding into s->x
    embed_row(s->x, transformer->weights.q_tokens, token, p->dim);
    rope_row(s->rope, p->head_dim, pos);
    decode_norm(s, p, &transformer->weights, 0, 0); // layer 0's attention rmsnorm + quantize

    int nt = rq_nthreads();
    rq_begin(&s->sync, decode_phases(p), decode_counters(p), nt, rq_spin_dec);
    // GEMV ranges of 16-row blocks: ~4 per thread (dim / 16 = 64 blocks), the classifier ~8
    s->dec_chunk = p->dim / 16 / (4 * nt) > 1 ? p->dim / 16 / (4 * nt) : 1;
    s->cls_chunk = (p->vocab_size + 15) / 16 / (8 * nt) > 1 ? (p->vocab_size + 15) / 16 / (8 * nt) : 1;
    DecodeJob job = { transformer, token, pos, compute_logits };
    rq_parallel(decode_worker, &job);
    return compute_logits ? s->logits : NULL;
}

static inline QuantizedTensor qrow(QuantizedTensor *t, int b, int ld) {
    // row b of a batch of quantized rows with row stride ld (a multiple of GS)
    return (QuantizedTensor) { .q = t->q + (size_t)b * ld, .s = t->s + (size_t)b * ld / GS, .p = NULL };
}

enum { LOGITS_NONE = 0, LOGITS_LAST = 1, LOGITS_ALL = 2 };

static void act_prep_row(RunState *s, QuantizedTensor *xq, int k, int32_t *corr, float *xf, int32_t *c16, int b) {
    // the GEMM activation prep of quantized row b (k values; the Gemm strides): matvec_prep
    matvec_prep(corr + (size_t)b * (k / GS), xf ? xf + (size_t)b * (k / GS) : NULL, c16 ? c16 + (size_t)b * (k / 16) : NULL,
                xq, 0, k);
    (void)s;
}

typedef struct { Transformer *tr; int *tokens; int n, pos0, logits_mode; float *logits_out; } PrefillJob;

static int prefill_phases(Config *p) { return 9 * p->n_layers + 4; }

#ifdef RUNQ_PROFILE
#define PPROF(i) do { if (tid == 0) PROF(i); } while (0)
#else
#define PPROF(i) ((void)0)
#endif

static void prefill_worker(void *arg, int tid, int nt) {
    // forward_batch's job: per layer 9 phases (the first layer 8): [residual + rmsnorm + quantize per
    // row] -> [q / k / v GEMM] -> [QK-norm + RoPE + K/V cache store per 16-position tile] ->
    // [attention per (block of att_rows rows, kv head), longest first; the last head of a row block
    // to finish quantizes its rows for wo] -> [wo GEMM] -> [residual + rmsnorm + quantize per row]
    // -> [w1 / w3 GEMM] -> [SwiGLU + quantize per row] -> [w2 GEMM]; the first phase also embeds the
    // tokens and pre-faults the chunk's KV cache tiles. Each phase's items as before (same results)
    PrefillJob *job = arg;
    Transformer *transformer = job->tr;
    Config *p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    RqSync *S = &s->sync;
    int n = job->n, pos0 = job->pos0, k_ = 0;
    int dim = p->dim, hd = p->head_dim, hidden = p->hidden_dim, xs = s->xs, nkv = p->n_kv_heads;
    int kv_dim = nkv * hd;
    int kv_mul = p->n_heads / nkv;
    int all_heads_dim = p->n_heads * hd;
    int rch = n / (4 * nt) > 1 ? n / (4 * nt) : 1; // rows per claim in the per-row phases
    int ar = s->att_rows, nrb = (n + ar - 1) / ar;
    int tb0 = pos0 / ATT_T, ntb = (pos0 + n - 1) / ATT_T - tb0 + 1;
    size_t ts = (size_t)nkv * ATT_T * hd;
    uint8_t *scr = s->kq_scr ? s->kq_scr + (size_t)tid * s->kq_scr_bytes : NULL;
    float *att = s->att_b + (size_t)tid * ar * kv_mul * s->att_nb * ATT_P;
    float *part = s->att_pb + (size_t)tid * ar * kv_mul * s->att_nb * ATT_PSTRIDE(hd);
    Gemm G;

    // embeddings, RoPE cos / sin per row, layer 0's attention rmsnorm + quantize; KV tile pre-faults
    PH_BEGIN(n + 2 * p->n_layers, 1)
    for (int b = lo; b < hi; b++) {
        if (b >= n) {
            kv_populate_one(s, p, pos0, n, b - n);
            continue;
        }
        float *x = s->x_b + (size_t)b * dim, *xb = s->xb_b + (size_t)b * xs;
        QuantizedTensor xq = qrow(&s->xq_b, b, xs);
        embed_row(x, w->q_tokens, job->tokens[b], dim);
        rope_row(s->rope_b + (size_t)b * hd, hd, pos0 + b);
        rmsnorm(xb, x, w->rms_att_weight, dim);
        quantize(&xq, xb, dim);
        act_prep_row(s, &xq, dim, s->bq_corr, s->bq_xf, s->bq_c16, b);
    }
    PH_END_NOEPI
    PPROF(PF_NORM);

    for (int l = 0; l < p->n_layers; l++) {
        uint64_t loff = l * (uint64_t)s->kv_len * kv_dim;
        float *kl = s->key_cache + loff, *vl = s->value_cache + loff;

        if (l > 0) { // residual of the previous layer's ffn, attention rmsnorm + quantize, per row
            PH_BEGIN(n, rch)
            for (int b = lo; b < hi; b++) {
                float *x = s->x_b + (size_t)b * dim, *xb = s->xb_b + (size_t)b * xs;
                QuantizedTensor xq = qrow(&s->xq_b, b, xs);
                for (int i = 0; i < dim; i++)
                    x[i] += xb[i];
                rmsnorm(xb, x, w->rms_att_weight + l * dim, dim);
                quantize(&xq, xb, dim);
                act_prep_row(s, &xq, dim, s->bq_corr, s->bq_xf, s->bq_c16, b);
            }
            PH_END_NOEPI
            PPROF(PF_NORM);
        }

        // qkv GEMMs (one phase)
        MatJob qkv[3] = { { s->q_b, all_heads_dim, w->wq + l, all_heads_dim }, { s->k_b, kv_dim, w->wk + l, kv_dim },
                          { s->v_b, kv_dim, w->wv + l, kv_dim } };
        gemm_plan(&G, 3, qkv, &s->xq_b, xs, n, dim, s->bq_corr, s->bq_xf, s->bq_c16, nt);
        {
            const void *last = NULL;
            PH_BEGIN_U(G.it0[G.nj], G.chunk, G.unit)
            for (int it = lo; it < hi; it++)
                gemm_item(&G, it, scr, &last);
            PH_END_NOEPI
        }
        PPROF(PF_QKV);

        // QK-norm + RoPE, each row at its own position, and the K / V rows into the cache: one
        // 16-position cache tile per item (its rows, all heads)
        PH_BEGIN(ntb, 1)
        for (int it = lo; it < hi; it++) {
            int tb = tb0 + it;
            int t1 = (tb + 1) * ATT_T < pos0 + n ? (tb + 1) * ATT_T : pos0 + n;
            for (int t = tb * ATT_T > pos0 ? tb * ATT_T : pos0; t < t1; t++) {
                int b = t - pos0;
                const float *cs = s->rope_b + (size_t)b * hd;
                qk_norm_rope(s->q_b + (size_t)b * all_heads_dim, p->n_heads, w->q_norm_weights + l * hd, hd, cs);
                qk_norm_rope(s->k_b + (size_t)b * kv_dim, nkv, w->k_norm_weights + l * hd, hd, cs);
                for (int h = 0; h < nkv; h++)
                    kv_store(kl, vl, t, h, nkv, hd, s->k_b + (size_t)b * kv_dim + h * hd, s->v_b + (size_t)b * kv_dim + h * hd);
            }
        }
        PH_END_NOEPI
        PPROF(PF_ROPE);

        // causal attention: row b (position pos0 + b) attends to positions 0..pos0+b, whose K/V
        // are all in the cache now. Work item = (block of att_rows rows, kv head): its rows x
        // kv_mul query heads in one attention_kv call (K/V read once for all of them), last
        // (longest) row blocks first, for the load balance. The last of a row block's nkv items
        // to finish quantizes its rows for wo (into xq_b: the qkv GEMM is done with it)
        PH_BEGIN_DYN(nrb * nkv, 1)
        for (int it = lo; it < hi; it++) {
            int rb = nrb - 1 - it / nkv, g = it % nkv;
            int b0 = rb * ar, b1 = b0 + ar < n ? b0 + ar : n, nv = 0;
            const float *qv[ATT_NV_MAX];
            float *ov[ATT_NV_MAX];
            int npv[ATT_NV_MAX];
            for (int b = b0; b < b1; b++)
                for (int r = 0; r < kv_mul; r++, nv++) {
                    int h = g * kv_mul + r;
                    qv[nv] = s->q_b + (size_t)b * all_heads_dim + h * hd;
                    ov[nv] = s->xb_b + (size_t)b * xs + h * hd;
                    npv[nv] = pos0 + b + 1;
                }
            size_t o = kv_tile(0, g, nkv, hd);
            attention_kv(nv, qv, ov, npv, kl + o, vl + o, ts, hd, s->att_scale, att, part);
            if (rq_count(S, l * s->pf_nrb + rb, 1) < nkv)
                continue;
            for (int b = b0; b < b1; b++) {
                QuantizedTensor xq = qrow(&s->xq_b, b, xs);
                quantize(&xq, s->xb_b + (size_t)b * xs, all_heads_dim);
                act_prep_row(s, &xq, all_heads_dim, s->bq_corr, s->bq_xf, s->bq_c16, b);
            }
        }
        PH_END_NOEPI
        PPROF(PF_ATT);

        MatJob wo = { s->xb_b, xs, w->wo + l, dim };
        gemm_plan(&G, 1, &wo, &s->xq_b, xs, n, all_heads_dim, s->bq_corr, s->bq_xf, s->bq_c16, nt);
        {
            const void *last = NULL;
            PH_BEGIN_U(G.it0[G.nj], G.chunk, G.unit)
            for (int it = lo; it < hi; it++)
                gemm_item(&G, it, scr, &last);
            PH_END_NOEPI
        }
        PPROF(PF_WO);

        // residual, ffn rmsnorm, quantize
        PH_BEGIN(n, rch)
        for (int b = lo; b < hi; b++) {
            float *x = s->x_b + (size_t)b * dim, *xb = s->xb_b + (size_t)b * xs;
            QuantizedTensor xq = qrow(&s->xq_b, b, xs);
            for (int i = 0; i < dim; i++)
                x[i] += xb[i];
            rmsnorm(xb, x, w->rms_ffn_weight + l * dim, dim);
            quantize(&xq, xb, dim);
            act_prep_row(s, &xq, dim, s->bq_corr, s->bq_xf, s->bq_c16, b);
        }
        PH_END_NOEPI
        PPROF(PF_NORM);

        MatJob w13[2] = { { s->hb_b, hidden, w->w1 + l, hidden }, { s->hb2_b, hidden, w->w3 + l, hidden } };
        gemm_plan(&G, 2, w13, &s->xq_b, xs, n, dim, s->bq_corr, s->bq_xf, s->bq_c16, nt);
        {
            const void *last = NULL;
            PH_BEGIN_U(G.it0[G.nj], G.chunk, G.unit)
            for (int it = lo; it < hi; it++)
                gemm_item(&G, it, scr, &last);
            PH_END_NOEPI
        }
        PPROF(PF_FFN13);

        PH_BEGIN(n, rch)
        for (int b = lo; b < hi; b++) {
            QuantizedTensor hq = qrow(&s->hq_b, b, hidden);
            swiglu(s->hb_b + (size_t)b * hidden, s->hb2_b + (size_t)b * hidden, hidden);
            quantize(&hq, s->hb_b + (size_t)b * hidden, hidden);
            act_prep_row(s, &hq, hidden, s->bh_corr, s->bh_xf, s->bh_c16, b);
        }
        PH_END_NOEPI
        PPROF(PF_SWIGLU);

        MatJob w2 = { s->xb_b, xs, w->w2 + l, dim };
        gemm_plan(&G, 1, &w2, &s->hq_b, hidden, n, hidden, s->bh_corr, s->bh_xf, s->bh_c16, nt);
        {
            const void *last = NULL;
            PH_BEGIN_U(G.it0[G.nj], G.chunk, G.unit)
            for (int it = lo; it < hi; it++)
                gemm_item(&G, it, scr, &last);
            PH_END_NOEPI
        }
        PPROF(PF_FFN2);
        // (its residual add is done in the next layer's first phase, or below)
    }

    if (job->logits_mode == LOGITS_LAST) {
        // residual of the last row, final rmsnorm + classifier for it, exactly as in forward()
        PH_BEGIN(1, 1)
            float *x = s->x_b + (size_t)(n - 1) * dim, *xb = s->xb_b + (size_t)(n - 1) * xs;
            for (int i = 0; i < dim; i++)
                x[i] += xb[i];
            rmsnorm(s->x, x, w->rms_final_weight, dim);
            quantize(&s->xq, s->x, dim);
            matvec_prep(s->xq_corr, s->xq_xf, s->xq_c16, &s->xq, 0, dim);
        PH_END_NOEPI
        int nblk = (p->vocab_size + 15) / 16, cc = nblk / (8 * nt) > 1 ? nblk / (8 * nt) : 1;
        PH_BEGIN(nblk, cc)
            matvec_rows(s->logits, &s->xq, s->xq_corr, s->xq_xf, s->xq_c16, w->wcls, dim, lo * 16,
                        hi * 16 < p->vocab_size ? hi * 16 : p->vocab_size);
        PH_END_NOEPI
        PPROF(PF_CLS);
    } else if (job->logits_mode == LOGITS_ALL) {
        PH_BEGIN(n, rch)
        for (int b = lo; b < hi; b++) {
            float *x = s->x_b + (size_t)b * dim, *xb = s->xb_b + (size_t)b * xs;
            QuantizedTensor xq = qrow(&s->xq_b, b, xs);
            for (int i = 0; i < dim; i++)
                x[i] += xb[i];
            rmsnorm(xb, x, w->rms_final_weight, dim);
            quantize(&xq, xb, dim);
            act_prep_row(s, &xq, dim, s->bq_corr, s->bq_xf, s->bq_c16, b);
        }
        PH_END_NOEPI
        MatJob cls = { job->logits_out, p->vocab_size, w->wcls, p->vocab_size };
        gemm_plan(&G, 1, &cls, &s->xq_b, xs, n, dim, s->bq_corr, s->bq_xf, s->bq_c16, nt);
        {
            const void *last = NULL;
            PH_BEGIN_U(G.it0[G.nj], G.chunk, G.unit)
            for (int it = lo; it < hi; it++)
                gemm_item(&G, it, scr, &last);
            PH_END_NOEPI
        }
        PPROF(PF_CLS);
    }
}

float *forward_batch(Transformer *transformer, int *tokens, int n, int pos0, int logits_mode, float *logits_out) {
    // Batched prefill: tokens[0..n-1] at positions pos0..pos0+n-1 (1 <= n <= state.nb_max),
    // the same computation as n calls of forward() but with every weight matrix applied to all
    // n rows at once (GEMM work items), and bit-identical results. Fills the KV cache.
    //   LOGITS_NONE: returns NULL
    //   LOGITS_LAST: logits of the last row in state.logits (returned)
    //   LOGITS_ALL:  logits of all rows into logits_out (n, vocab_size) (returned)
    // Note: in LOGITS_LAST mode state.x / state.xq are clobbered (as by forward()).
    // One parallel job per chunk (prefill_worker), in phases (threads.h)
    Config *p = &transformer->config;
    RunState* s = &transformer->state;
    if (n < 1 || n > s->nb_max || pos0 < 0 || pos0 + n > p->seq_len) {
        fprintf(stderr, "forward_batch: bad chunk n=%d pos0=%d (nb_max %d, seq_len %d)\n", n, pos0, s->nb_max, p->seq_len);
        exit(EXIT_FAILURE);
    }
    PROF_START();
    rq_begin(&s->sync, prefill_phases(p), p->n_layers * s->pf_nrb, rq_nthreads(), rq_spin_pf);
    PrefillJob job = { transformer, tokens, n, pos0, logits_mode, logits_out };
    rq_parallel(prefill_worker, &job);
    return logits_mode == LOGITS_LAST ? s->logits : logits_mode == LOGITS_ALL ? logits_out : NULL;
}

float *prefill(Transformer *transformer, int *tokens, int n, int pos0) {
    // feed tokens[0..n-1] at positions pos0.. in chunks of state.nb_max (> 1) rows;
    // returns the logits of the last token (state.logits)
    int nb = transformer->state.nb_max;
    float *logits = NULL;
    for (int i = 0; i < n; i += nb) {
        int m = n - i < nb ? n - i : nb;
        logits = forward_batch(transformer, tokens + i, m, pos0 + i, i + m == n ? LOGITS_LAST : LOGITS_NONE, NULL);
    }
    return logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens
//
// cwayn rewrite, matching HF tokenizers on Qwen2/Qwen3's tokenizer.json:
//   1. added tokens (<|im_start|>, <think>, ...) are split out first, leftmost-longest match
//   2. normalizer (NFC), on the text between added tokens (nfc_check / nfc_normalize, tables in
//      unicode_nfc.h: Unicode 15.1 data with the overrides that match HF tokenizers' older data)
//   3. pre-tokenizer: Qwen2's split regex, hand-written in pretok_next(), with the Unicode
//      classes \p{L}, \p{N}, \s from unicode_tables.h (tools/gen_unicode_tables.py)
//   4. byte-level BPE inside each chunk, starting from the single-byte tokens
// Merge order: the .tokenizer file (third_party/qwen3.c/export.py) has one score per token,
// -log(rank + 1) of the merge producing it, and no merge pairs. bpe_chunk() merges any adjacent
// pair whose concatenation is a vocab token, best score first, leftmost on ties (tiktoken
// style). Verified against HF's BPE with tokenizer.json's model.merges: identical on all
// 151,643 vocab strings and 300k random token concatenations (no pre-tokenizer), since Qwen's
// merge list is the tiktoken-consistent one: one merge per token, in id order, each the split
// tiktoken-style BPE itself produces from the lower-ranked tokens.

#include "unicode_tables.h"
#include "unicode_nfc.h"

// tokens[] capacity encode() needs for a text of n bytes: one token per byte of the NFC-normalized
// text, which is at most 3x as long (e.g. U+1D160: 4 bytes -> 3 code points, 12 bytes)
#define ENCODE_MAX_TOKENS(n) (3 * (size_t)(n) + 3)

typedef struct {
    char **vocab;           // token bytes, NUL-terminated (the token for byte 0x00 reads as "")
    int *vocab_len;         // byte length of each token
    float *merge_scores;
    int vocab_size;
    unsigned int max_token_length;
    unsigned int bos_token_id;
    unsigned int eos_token_id;
    char prompt_template[1024];
    char system_prompt_template[1024];
    char *vocab_data;       // one buffer holding all the vocab strings
    int32_t *hash_ids;      // open-addressing hash map: token bytes -> id (-1 = empty slot);
    uint32_t *hash_tags;    // regular tokens only (no added tokens), so BPE never produces those
    uint32_t hash_mask;
    int byte_token[256];    // token id of each single byte
    int *added;             // added tokens (special or not), longest first
    int n_added;
    unsigned char added_first[256]; // 1 if some added token starts with this byte
} Tokenizer;

void load_prompt_template(char *checkpoint_path, char *out_template, int with_system_prompt, int enable_thinking) {
    char prompt_path[1024];

    strcpy(prompt_path, checkpoint_path);
    if (with_system_prompt)
        strcat(prompt_path, enable_thinking ? ".template.with-system-and-thinking" : ".template.with-system");
    else
        strcat(prompt_path, enable_thinking ? ".template.with-thinking" : ".template");

    memset(out_template, 0, 1024);
    FILE *file = fopen(prompt_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't load prompt template %s\n", prompt_path); exit(EXIT_FAILURE); }
    fread(out_template, 1024, 1, file);
    fclose(file);
}

static inline uint64_t bytes_hash(const unsigned char *s, int n) {
    uint64_t h = 1469598103934665603ull; // FNV-1a, then a murmur3 finalizer for the low bits
    for (int i = 0; i < n; i++) { h ^= s[i]; h *= 1099511628211ull; }
    h ^= h >> 33; h *= 0xff51afd7ed558ccdull; h ^= h >> 33;
    return h;
}

void tokenizer_index(Tokenizer *t);

int vocab_lookup(Tokenizer *t, const unsigned char *s, int n) {
    // id of the regular token with exactly these n bytes, or -1
    uint64_t h = bytes_hash(s, n);
    uint32_t tag = (uint32_t)(h >> 32);
    for (uint32_t i = (uint32_t)h & t->hash_mask;; i = (i + 1) & t->hash_mask) {
        int id = t->hash_ids[i];
        if (id < 0) return -1;
        if (t->hash_tags[i] == tag && t->vocab_len[id] == n && !memcmp(t->vocab[id], s, n)) return id;
    }
}

void build_tokenizer(Tokenizer *t, char *checkpoint_path, int vocab_size, int enable_thinking) {
    char tokenizer_path[1024];

    strcpy(tokenizer_path, checkpoint_path);
    strcat(tokenizer_path, ".tokenizer");

    // read the whole file: header (max_token_length, bos, eos), then per token
    // float score, int32 length, bytes; tokens missing at the end (padding ids) are empty
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't load tokenizer model %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    fseek(file, 0, SEEK_END);
    long fsize = ftell(file);
    fseek(file, 0, SEEK_SET);
    unsigned char *buf = malloc(fsize > 0 ? fsize : 1);
    if (fsize < 12 || fread(buf, 1, fsize, file) != (size_t)fsize) {
        fprintf(stderr, "Couldn't read tokenizer model %s\n", tokenizer_path); exit(EXIT_FAILURE);
    }
    fclose(file);
    memcpy(&t->max_token_length, buf, 4);
    memcpy(&t->bos_token_id, buf + 4, 4);
    memcpy(&t->eos_token_id, buf + 8, 4);

    t->vocab_size = vocab_size;
    t->vocab = malloc(vocab_size * sizeof(char *));
    t->vocab_len = malloc(vocab_size * sizeof(int));
    t->merge_scores = malloc(vocab_size * sizeof(float));
    t->vocab_data = malloc(fsize + vocab_size); // string bytes + one NUL per token
    if (!t->vocab || !t->vocab_len || !t->merge_scores || !t->vocab_data) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    long off = 12;
    char *dst = t->vocab_data;
    unsigned int max_len = 0;
    for (int i = 0; i < vocab_size; i++) {
        int32_t len = 0;
        float score = -1e6f;
        if (off + 8 <= fsize) {
            memcpy(&score, buf + off, 4);
            memcpy(&len, buf + off + 4, 4);
            if (len < 0 || off + 8 + len > fsize) { fprintf(stderr, "Corrupt tokenizer model %s\n", tokenizer_path); exit(EXIT_FAILURE); }
            memcpy(dst, buf + off + 8, len);
            off += 8 + len;
        }
        t->vocab[i] = dst;
        t->vocab_len[i] = len;
        t->merge_scores[i] = score;
        dst[len] = 0;
        dst += len + 1;
        if ((unsigned int)len > max_len) max_len = len;
    }
    free(buf);
    t->max_token_length = max_len;
    tokenizer_index(t);

    load_prompt_template(checkpoint_path, t->prompt_template, 0, enable_thinking);
    load_prompt_template(checkpoint_path, t->system_prompt_template, 1, enable_thinking);
}

void tokenizer_index(Tokenizer *t) {
    // classify: single bytes, merge results (score > -1e6) and added tokens (the rest, len > 1;
    // export.py gives every token that no merge produces the score -1e6)
    int vocab_size = t->vocab_size;
    for (int b = 0; b < 256; b++) t->byte_token[b] = -1;
    uint32_t cap = 1;
    while (cap < 2u * (uint32_t)vocab_size) cap <<= 1;
    t->hash_mask = cap - 1;
    t->hash_ids = malloc(cap * sizeof(int32_t));
    t->hash_tags = malloc(cap * sizeof(uint32_t));
    t->added = malloc(vocab_size * sizeof(int));
    if (!t->hash_ids || !t->hash_tags || !t->added) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    memset(t->hash_ids, 0xff, cap * sizeof(int32_t));
    t->n_added = 0;
    memset(t->added_first, 0, sizeof(t->added_first));
    for (int i = 0; i < vocab_size; i++) {
        int len = t->vocab_len[i];
        const unsigned char *s = (const unsigned char *)t->vocab[i];
        if (len == 0) continue;
        if (len > 1 && t->merge_scores[i] <= -1e6f) {
            t->added[t->n_added++] = i;
            t->added_first[s[0]] = 1;
            continue;
        }
        if (len == 1 && t->byte_token[s[0]] < 0) t->byte_token[s[0]] = i;
        uint64_t h = bytes_hash(s, len);
        uint32_t j = (uint32_t)h & t->hash_mask;
        int dup = 0;
        for (; t->hash_ids[j] >= 0; j = (j + 1) & t->hash_mask) {
            int o = t->hash_ids[j];
            if (t->vocab_len[o] == len && !memcmp(t->vocab[o], s, len)) { dup = 1; break; } // keep the first
        }
        if (dup) continue;
        t->hash_ids[j] = i;
        t->hash_tags[j] = (uint32_t)(h >> 32);
    }
    for (int b = 0; b < 256; b++) {
        if (t->byte_token[b] < 0) { fprintf(stderr, "Tokenizer has no token for byte 0x%02x (not byte-level BPE?)\n", b); exit(EXIT_FAILURE); }
    }
    for (int i = 1; i < t->n_added; i++) { // longest first (insertion sort, ~26 entries), stable
        int id = t->added[i], k = i;
        for (; k > 0 && t->vocab_len[t->added[k - 1]] < t->vocab_len[id]; k--) t->added[k] = t->added[k - 1];
        t->added[k] = id;
    }
}

// Qwen3's chat template (tokenizer_config.json) rendered for one user message "%s", with and without
// a system message "%s" and enable_thinking: the strings export.py writes into the .template* files
// (byte-identical; make test compares the chat token ids of a .bin and a GGUF run). Used for GGUF
// files, whose tokenizer.chat_template is Jinja (not interpreted here)
#define QWEN3_TMPL_SYS "<|im_start|>system\n%s<|im_end|>\n"
#define QWEN3_TMPL_USER "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"
#define QWEN3_TMPL_NOTHINK "<think>\n\n</think>\n\n"

static int gpt2_byte(uint32_t cp) {
    // GPT-2's bytes_to_unicode, inverted: the printable bytes ('!'..'~', 0xa1..0xac, 0xae..0xff)
    // stand for themselves, the other 68 (0x00..0x20, 0x7f..0xa0, 0xad) for U+0100.. in byte order;
    // -1 if cp is neither
    if ((cp >= 0x21 && cp <= 0x7e) || (cp >= 0xa1 && cp <= 0xac) || (cp >= 0xae && cp <= 0xff)) return (int)cp;
    if (cp < 0x100 || cp >= 0x100 + 68) return -1;
    int n = (int)cp - 0x100;
    return n <= 0x20 ? n : n < 33 + 34 ? 0x7f + (n - 33) : 0xad;
}

static inline uint32_t utf8_cp(const unsigned char *p, const unsigned char *end, int *len);

static int gpt2_decode(const char *s, uint64_t n, char *out) {
    // a token string in GPT-2's byte-level form -> its raw bytes (at most n); -1 if it is not one
    const unsigned char *p = (const unsigned char *)s, *end = p + n;
    int len = 0;
    while (p < end) {
        if (*p >= 0x21 && *p <= 0x7e) { out[len++] = (char)*p++; continue; } // printable ASCII: itself
        int cl, b = gpt2_byte(utf8_cp(p, end, &cl)); // (invalid UTF-8: U+FFFD, not a byte)
        if (b < 0) return -1;
        out[len++] = (char)b;
        p += cl;
    }
    return len;
}

typedef struct { Tokenizer *t; const Gguf *g; const uint64_t *mo; int *mid; long nmg; unsigned int max_len; long next; } MergeJob;

static void merge_worker(void *arg, int tid, int nt) {
    // build_tokenizer_gguf: the tokens produced by the merges (mid[r]: id, -1 none, -2 not GPT-2
    // form), in claims of 4,096 merges
    (void)tid; (void)nt;
    MergeJob *M = arg;
    char *buf = malloc(2 * (size_t)M->max_len + 2);
    if (!buf) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    for (long r0; (r0 = __atomic_fetch_add(&M->next, 4096, __ATOMIC_RELAXED)) < M->nmg;)
        for (long r = r0; r < r0 + 4096 && r < M->nmg; r++) {
            uint64_t sl, p = M->mo[r];
            memcpy(&sl, M->g->map + p, 8); // (in the file: checked by the walk above)
            const char *s = (const char *)M->g->map + p + 8, *sp;
            M->mid[r] = -1;
            if (sl > 2 * (uint64_t)M->max_len + 1 || !(sp = memchr(s, ' ', sl))) continue;
            int la = gpt2_decode(s, sp - s, buf), lb = la < 0 ? -1 : gpt2_decode(sp + 1, sl - (sp + 1 - s), buf + la);
            M->mid[r] = la < 0 || lb < 0 ? -2 : vocab_lookup(M->t, (const unsigned char *)buf, la + lb);
        }
    free(buf);
}

void build_tokenizer_gguf(Tokenizer *t, const Gguf *g, int vocab_size, int enable_thinking) {
    // The tokenizer from GGUF metadata (llama.cpp's "gpt2" model with the "qwen2" pre-tokenizer,
    // which is what encode() implements):
    //   tokenizer.ggml.tokens: token strings, in GPT-2's byte-to-unicode form for normal tokens
    //     (decoded to raw bytes, as the .tokenizer file stores them), as is for added ones
    //   tokenizer.ggml.token_type: 1 normal, 3 control and 4 user-defined = added tokens (HF's
    //     added_tokens, matched in the raw text), 5 unused (llama.cpp's [PAD...] fillers up to the
    //     embedding size: empty here, as the ids missing from the .tokenizer file)
    //   tokenizer.ggml.merges: "left right" in rank order. Score of the token a merge of rank r
    //     produces: -r (the .tokenizer file has -log(r + 1): the same order, so the same BPE);
    //     the first merge counts if several give one token; a normal token of several bytes that
    //     no merge produces gets -999999 (after all merges; Qwen3 has none). Checked: the tokens
    //     decode to exactly the .tokenizer file's bytes and the merges equal tokenizer.json's
    //   eos = tokenizer.ggml.eos_token_id; bos (generation also stops there) =
    //     tokenizer.ggml.bos_token_id, else the token <|endoftext|> (Qwen's, as config.json), else eos
    uint64_t n;
    const char *model = gguf_get_str(g, "tokenizer.ggml.model", &n);
    if (!model || n != 4 || memcmp(model, "gpt2", 4)) gguf_fail(g, "tokenizer.ggml.model must be gpt2 (byte-level BPE)%s", NULL);
    const char *pre = gguf_get_str(g, "tokenizer.ggml.pre", &n);
    if (!pre || n != 5 || memcmp(pre, "qwen2", 5)) gguf_fail(g, "tokenizer.ggml.pre must be qwen2 (the pre-tokenizer runq implements)%s", NULL);
    const GgufKV *tk = gguf_get_array(g, "tokenizer.ggml.tokens", GV_STR);
    const GgufKV *ty = gguf_get_array(g, "tokenizer.ggml.token_type", GV_I32);
    const GgufKV *mg = gguf_get_array(g, "tokenizer.ggml.merges", GV_STR);
    if (!tk || !mg) gguf_fail(g, "tokenizer.ggml.tokens / merges missing%s", NULL);
    if (tk->n > (uint64_t)vocab_size) gguf_fail(g, "more tokens than embedding rows%s", NULL);
    if (ty && ty->n != tk->n) gguf_fail(g, "tokenizer.ggml.token_type has the wrong length%s", NULL);

    t->vocab_size = vocab_size;
    t->vocab = malloc(vocab_size * sizeof(char *));
    t->vocab_len = malloc(vocab_size * sizeof(int));
    t->merge_scores = malloc(vocab_size * sizeof(float));
    uint64_t pos = tk->off, bytes = 0;
    for (uint64_t i = 0; i < tk->n; i++) { // (bounds checked by gguf_parse, again here)
        const char *s;
        bytes += gguf_str(g, &pos, &s, "tokenizer.ggml.tokens");
    }
    t->vocab_data = malloc(bytes + vocab_size);
    if (!t->vocab || !t->vocab_len || !t->merge_scores || !t->vocab_data) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    char *dst = t->vocab_data;
    unsigned int max_len = 0;
    pos = tk->off;
    for (int i = 0; i < vocab_size; i++) {
        const char *s = "";
        uint64_t sl = 0;
        int32_t type = 1, len = 0;
        if ((uint64_t)i < tk->n) {
            sl = gguf_str(g, &pos, &s, "tokenizer.ggml.tokens");
            if (sl > 65536) gguf_fail(g, "a token string is too long%s", NULL);
            if (ty) memcpy(&type, g->map + ty->off + 4 * (uint64_t)i, 4);
        } else {
            type = 5;
        }
        if (type == 5) {
            len = 0;
        } else if (type == 3 || type == 4) { // added: raw text
            len = (int)sl;
            memcpy(dst, s, sl);
        } else if ((len = gpt2_decode(s, sl, dst)) < 0) {
            char id[24];
            snprintf(id, sizeof(id), "%d", i);
            gguf_fail(g, "token %s is not in GPT-2 byte-level form", id);
        }
        t->vocab[i] = dst;
        t->vocab_len[i] = len;
        t->merge_scores[i] = (type == 3 || type == 4) && len > 1 ? -1e6f : -999999.0f;
        dst[len] = 0;
        dst += len + 1;
        if ((unsigned int)len > max_len) max_len = len;
    }
    t->max_token_length = max_len;
    tokenizer_index(t);

    // merge ranks -> scores of the tokens they produce: the merge strings located (serial walk),
    // decoded and looked up in parallel (151k hash lookups, ~12 ms on one thread), then the scores
    // set in rank order. A GPT-2 form character is 1-2 bytes, so a merge string longer than
    // 2 * max_len + 1 cannot give a token (and the decoded bytes fit buf)
    long nmg = (long)mg->n;
    uint64_t *mo = malloc((nmg + 1) * sizeof(uint64_t));
    int *mid = malloc((nmg + 1) * sizeof(int));
    if (!mo || !mid) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    pos = mg->off;
    for (long r = 0; r < nmg; r++) {
        const char *s;
        mo[r] = pos;
        gguf_str(g, &pos, &s, "tokenizer.ggml.merges");
    }
    MergeJob mj = { t, g, mo, mid, nmg, max_len, 0 };
    rq_parallel(merge_worker, &mj);
    for (long r = 0; r < nmg; r++) {
        if (mid[r] == -2) gguf_fail(g, "a merge is not in GPT-2 byte-level form%s", NULL);
        if (mid[r] >= 0 && t->merge_scores[mid[r]] == -999999.0f)
            t->merge_scores[mid[r]] = -(float)r;
    }
    free(mo);
    free(mid);

    int64_t v;
    if (!gguf_get_int(g, "tokenizer.ggml.eos_token_id", &v) || v < 0 || v >= vocab_size)
        gguf_fail(g, "tokenizer.ggml.eos_token_id missing or out of range%s", NULL);
    t->eos_token_id = (unsigned int)v;
    t->bos_token_id = t->eos_token_id;
    if (gguf_get_int(g, "tokenizer.ggml.bos_token_id", &v)) {
        if (v < 0 || v >= vocab_size) gguf_fail(g, "tokenizer.ggml.bos_token_id out of range%s", NULL);
        t->bos_token_id = (unsigned int)v;
    } else {
        for (int k = 0; k < t->n_added; k++)
            if (t->vocab_len[t->added[k]] == 13 && !memcmp(t->vocab[t->added[k]], "<|endoftext|>", 13))
                t->bos_token_id = t->added[k];
    }

    strcpy(t->prompt_template, enable_thinking ? QWEN3_TMPL_USER : QWEN3_TMPL_USER QWEN3_TMPL_NOTHINK);
    strcpy(t->system_prompt_template, enable_thinking ? QWEN3_TMPL_SYS QWEN3_TMPL_USER : QWEN3_TMPL_SYS QWEN3_TMPL_USER QWEN3_TMPL_NOTHINK);
}

void free_tokenizer(Tokenizer *t) {
    free(t->vocab);
    free(t->vocab_len);
    free(t->vocab_data);
    free(t->merge_scores);
    free(t->hash_ids);
    free(t->hash_tags);
    free(t->added);
}

char *decode(Tokenizer *t, int token) {
    // raw token bytes; a code point split over several tokens comes out whole once all are printed
    return t->vocab[token];
}

// --- pre-tokenizer: Qwen2's Split regex (tokenizer.json, behavior Isolated)
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+

typedef struct { uint32_t cp; int len; int cls; } UChar;

static inline int ucls(uint32_t cp) {
    if (cp < 0x80) {
        if ((cp | 0x20) >= 'a' && (cp | 0x20) <= 'z') return UCLS_L;
        if (cp >= '0' && cp <= '9') return UCLS_N;
        if (cp == ' ' || (cp >= '\t' && cp <= '\r')) return UCLS_WS;
        return UCLS_OTHER;
    }
    int lo = 0, hi = UCLS_NRANGES - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (cp < ucls_lo[mid]) hi = mid - 1;
        else if (cp > (ucls_hi_cls[mid] & 0xffffff)) lo = mid + 1;
        else return ucls_hi_cls[mid] >> 24;
    }
    return UCLS_OTHER;
}

static inline uint32_t utf8_cp(const unsigned char *p, const unsigned char *end, int *len) {
    // decode one UTF-8 code point; an invalid byte gives U+FFFD with *len = 1
    unsigned char b = p[0];
    *len = 1;
    if (b < 0x80) return b;
    int n = b >= 0xf0 ? (b <= 0xf4 ? 4 : 0) : b >= 0xe0 ? 3 : b >= 0xc2 ? 2 : 0;
    if (n == 0 || end - p < n) return 0xfffd;
    uint32_t cp = b & (0x7f >> n);
    for (int i = 1; i < n; i++) {
        if ((p[i] & 0xc0) != 0x80) return 0xfffd;
        cp = (cp << 6) | (p[i] & 0x3f);
    }
    if ((n == 3 && (cp < 0x800 || (cp >= 0xd800 && cp <= 0xdfff))) || (n == 4 && (cp < 0x10000 || cp > 0x10ffff)))
        return 0xfffd;
    *len = n;
    return cp;
}

static inline UChar uchar_at(const unsigned char *p, const unsigned char *end) {
    // decode one UTF-8 code point; an invalid byte counts as one "other" character (U+FFFD is one)
    UChar c;
    c.cp = utf8_cp(p, end, &c.len);
    c.cls = ucls(c.cp);
    return c;
}

static inline const unsigned char *skip_class(const unsigned char *p, const unsigned char *end, int cls) {
    while (p < end) {
        UChar c = uchar_at(p, end);
        if (c.cls != cls) break;
        p += c.len;
    }
    return p;
}

static inline int ascii_lower(int c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

static int contraction_len(const unsigned char *p, const unsigned char *end) {
    // bytes matched by (?i:s|t|re|ve|m|ll|d) after the apostrophe, 0 if none. Case folding is
    // Unicode's: U+017F LATIN SMALL LETTER LONG S also matches 's' (checked against HF)
    if (p >= end) return 0;
    int c0 = ascii_lower(p[0]), c1 = p + 1 < end ? ascii_lower(p[1]) : 0;
    if (c0 == 's' || c0 == 't' || c0 == 'm' || c0 == 'd') return 1;
    if (p[0] == 0xc5 && c1 == 0xbf) return 2;
    if (((c0 == 'r' || c0 == 'v') && c1 == 'e') || (c0 == 'l' && c1 == 'l')) return 2;
    return 0;
}

static const unsigned char *pretok_next(const unsigned char *p, const unsigned char *end) {
    // end of the regex match starting at p (p < end); every character matches some alternative
    UChar c = uchar_at(p, end);
    const unsigned char *q = p + c.len;
    int crlf = c.cp == '\r' || c.cp == '\n';
    // (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (c.cp == '\'') {
        int n = contraction_len(q, end);
        if (n) return q + n;
    }
    // [^\r\n\p{L}\p{N}]?\p{L}+
    if (c.cls == UCLS_L) return skip_class(q, end, UCLS_L);
    if (c.cls != UCLS_N && !crlf && q < end && uchar_at(q, end).cls == UCLS_L) return skip_class(q, end, UCLS_L);
    // \p{N}: one digit at a time
    if (c.cls == UCLS_N) return q;
    // ' ?[^\s\p{L}\p{N}]+[\r\n]*' (only ASCII space as the optional prefix)
    const unsigned char *r = c.cp == ' ' ? q : p;
    if (r < end && uchar_at(r, end).cls == UCLS_OTHER) {
        r = skip_class(r, end, UCLS_OTHER);
        while (r < end && (*r == '\r' || *r == '\n')) r++;
        return r;
    }
    // c is whitespace. \s*[\r\n]+ : the whitespace run up to its last CR/LF, if any;
    // else \s+(?!\S) : the run minus its last character if a non-space follows (that last
    // space then starts the next chunk, e.g. " word"); else \s+ : the run (1 char, or at the end)
    const unsigned char *last_nl = NULL, *last_start = p;
    r = p;
    while (r < end) {
        UChar w = uchar_at(r, end);
        if (w.cls != UCLS_WS) break;
        if (w.cp == '\r' || w.cp == '\n') last_nl = r + w.len;
        last_start = r;
        r += w.len;
    }
    if (last_nl) return last_nl;
    if (r == end || last_start == p) return r;
    return last_start;
}

// --- normalizer: NFC (tokenizer.json). All added tokens are "normalized": false, so HF matches them
// on the raw text and normalizes each piece between them on its own: so does encode_segment().
// Invalid UTF-8 (no HF equivalent: a Python str cannot hold it) passes through unchanged, each
// invalid byte acting as U+FFFD would (a starter that composes with nothing): no reordering or
// composition across it, and the valid text around it is normalized as usual.

enum { NFC_YES = 0, NFC_MAYBE = 1, NFC_NO = 2 };
enum { HG_S = 0xAC00, HG_L = 0x1100, HG_V = 0x1161, HG_T = 0x11A7, HG_LN = 19, HG_VN = 21, HG_TN = 28, HG_SN = 11172 };
#define NFC_RAW 0x110000u // + byte: an invalid UTF-8 byte in the code point buffer (in no table)

static inline int nfc_find(uint32_t cp, const uint32_t *blk) {
    // index of cp in nfc_cp[], or -1; blk = nfc_any_blk, or nfc_qc_blk when only ccc / qc matter
    if (cp < NFC_MIN_CP || cp > 0x10ffff || !(blk[cp >> 11] >> (cp >> 6 & 31) & 1)) return -1;
    int lo = 0, hi = NFC_N - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        uint32_t m = nfc_cp[mid] & 0x1fffff;
        if (cp < m) hi = mid - 1;
        else if (cp > m) lo = mid + 1;
        else return mid;
    }
    return -1;
}

static inline int nfc_info(int k) { // ccc | 256 if NFC_QC = Maybe (can combine with what precedes it)
    return k < 0 ? 0 : (int)(nfc_cp[k] >> 23) | ((nfc_cp[k] >> 21 & 3) == NFC_MAYBE) << 8;
}

static const unsigned char *nfc_check(const unsigned char *p, const unsigned char *end) {
    // quick check: end if the text is NFC (every code point NFC_QC = Yes, combining marks in
    // canonical order), else where normalizing must start: the last code point before the first
    // failure with ccc 0 and NFC_QC = Yes (nothing before it interacts with what follows)
    const unsigned char *safe = p;
    int last = 0;
    while (p < end) {
        if (*p < 0x80) { // ASCII: ccc 0, Yes
            while (++p < end && *p < 0x80) {}
            safe = p - 1;
            last = 0;
            continue;
        }
        int n, ccc = 0;
        int k = nfc_find(utf8_cp(p, end, &n), nfc_qc_blk);
        if (k >= 0) {
            ccc = nfc_cp[k] >> 23;
            if ((nfc_cp[k] >> 21 & 3) != NFC_YES || (ccc && ccc < last)) return safe;
        }
        if (!ccc) safe = p;
        last = ccc;
        p += n;
    }
    return end;
}

static uint32_t nfc_compose(uint32_t a, uint32_t b) {
    // primary composite of the pair, or 0 (Hangul algorithmically; NFC_RAW values are in no key)
    if (a - HG_L < HG_LN && b - HG_V < HG_VN) return HG_S + ((a - HG_L) * HG_VN + (b - HG_V)) * HG_TN;
    if (a - HG_S < HG_SN && (a - HG_S) % HG_TN == 0 && b - HG_T - 1 < HG_TN - 1) return a + (b - HG_T);
    uint64_t key = (uint64_t)a << 21 | b;
    int lo = 0, hi = NFC_NCOMP - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        uint64_t m = nfc_comp[mid] >> 21;
        if (key < m) hi = mid - 1;
        else if (key > m) lo = mid + 1;
        else return (uint32_t)(nfc_comp[mid] & 0x1fffff);
    }
    return 0;
}

static void nfc_sort_marks(uint32_t *cp, uint16_t *inf, int n, uint32_t *tcp, uint16_t *tinf) {
    // stable sort of a run of combining marks by ccc: insertion sort, counting sort for long runs
    if (n < 32) {
        for (int i = 1; i < n; i++) {
            uint32_t c = cp[i];
            uint16_t f = inf[i];
            int j = i;
            for (; j > 0 && (inf[j - 1] & 0xff) > (f & 0xff); j--) { cp[j] = cp[j - 1]; inf[j] = inf[j - 1]; }
            cp[j] = c;
            inf[j] = f;
        }
        return;
    }
    int cnt[257] = {0};
    for (int i = 0; i < n; i++) cnt[(inf[i] & 0xff) + 1]++;
    for (int i = 1; i < 257; i++) cnt[i] += cnt[i - 1];
    for (int i = 0; i < n; i++) { int j = cnt[inf[i] & 0xff]++; tcp[j] = cp[i]; tinf[j] = inf[i]; }
    memcpy(cp, tcp, n * sizeof(uint32_t));
    memcpy(inf, tinf, n * sizeof(uint16_t));
}

static unsigned char *nfc_normalize(const unsigned char *p, const unsigned char *q, const unsigned char *end, int *out_len) {
    // NFC of [p, end) as a new buffer: [p, q) copied (nfc_check), [q, end) decomposed, its marks
    // put in canonical order and composed. Output <= 3x the bytes, <= 2 code points per byte
    // (both asserted by the table generator)
    int nb = (int)(end - q), cap = 2 * nb + 1;
    uint32_t *cp = malloc(2 * (size_t)cap * sizeof(uint32_t)); // code points, then sort scratch
    uint16_t *inf = malloc(2 * (size_t)cap * sizeof(uint16_t)); // nfc_info of each
    unsigned char *out = malloc((q - p) + 3 * (size_t)nb + 1);
    if (!cp || !inf || !out) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    int n = 0;
    for (const unsigned char *r = q; r < end;) { // decompose
        int len;
        uint32_t c = utf8_cp(r, end, &len);
        if (c == 0xfffd && len == 1) c = NFC_RAW + *r;
        r += len;
        if (c - HG_S < HG_SN) {
            uint32_t s = c - HG_S;
            cp[n] = HG_L + s / (HG_VN * HG_TN); inf[n++] = 0;
            cp[n] = HG_V + s % (HG_VN * HG_TN) / HG_TN; inf[n++] = 256;
            if (s % HG_TN) { cp[n] = HG_T + s % HG_TN; inf[n++] = 256; }
            continue;
        }
        int k = nfc_find(c, nfc_any_blk);
        if (k >= 0 && nfc_dec[k]) {
            const uint32_t *d = nfc_pool + (nfc_dec[k] >> 3);
            for (int i = 0; i < (nfc_dec[k] & 7); i++) { cp[n] = d[i]; inf[n++] = nfc_info(nfc_find(d[i], nfc_qc_blk)); }
        } else {
            cp[n] = c; inf[n++] = nfc_info(k);
        }
    }
    for (int i = 0; i < n;) { // canonical order: each run of combining marks sorted by ccc
        if (!(inf[i] & 0xff)) { i++; continue; }
        int j = i;
        while (j < n && (inf[j] & 0xff)) j++;
        if (j - i > 1) nfc_sort_marks(cp + i, inf + i, j - i, cp + cap, inf + cap);
        i = j;
    }
    int w = 0, starter = -1, last = 0; // compose: last = ccc of the last code point kept
    for (int i = 0; i < n; i++) {
        int cc = inf[i] & 0xff;
        // not blocked: adjacent to the starter, or every mark in between has a lower ccc
        if (starter >= 0 && (inf[i] & 256) && (w == starter + 1 || last < cc)) {
            uint32_t x = nfc_compose(cp[starter], cp[i]);
            if (x) { cp[starter] = x; continue; }
        }
        if (!cc) starter = w;
        last = cc;
        cp[w++] = cp[i];
    }
    unsigned char *o = out;
    memcpy(o, p, q - p);
    o += q - p;
    for (int i = 0; i < w; i++) {
        uint32_t c = cp[i];
        if (c >= NFC_RAW) *o++ = (unsigned char)(c - NFC_RAW);
        else if (c < 0x80) *o++ = c;
        else if (c < 0x800) { *o++ = 0xc0 | c >> 6; *o++ = 0x80 | (c & 0x3f); }
        else if (c < 0x10000) { *o++ = 0xe0 | c >> 12; *o++ = 0x80 | (c >> 6 & 0x3f); *o++ = 0x80 | (c & 0x3f); }
        else { *o++ = 0xf0 | c >> 18; *o++ = 0x80 | (c >> 12 & 0x3f); *o++ = 0x80 | (c >> 6 & 0x3f); *o++ = 0x80 | (c & 0x3f); }
    }
    free(cp);
    free(inf);
    *out_len = (int)(o - out);
    return out;
}

// --- byte-level BPE inside one pre-tokenizer chunk

typedef struct { float score; int pos, len, id; } BpeCand; // merge candidate: symbol at pos + next one

typedef struct {
    int *len, *tok, *prev; // per byte offset: symbol length (0 = not a symbol start), token id, previous symbol
    BpeCand *heap;
    int cap;
} BpeScratch;

static void bpe_reserve(BpeScratch *s, int n) {
    // scratch for chunks of up to n bytes
    if (n < 1) n = 1;
    if (s->len && n <= s->cap) return;
    free(s->len);
    free(s->heap);
    s->cap = n;
    s->len = malloc(3 * (size_t)n * sizeof(int));
    s->tok = s->len + n;
    s->prev = s->tok + n;
    s->heap = malloc(3 * (size_t)n * sizeof(BpeCand)); // n-1 initial pairs + 2 per merge
    if (!s->len || !s->heap) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
}

static inline int cand_before(const BpeCand *a, const BpeCand *b) {
    return a->score > b->score || (a->score == b->score && a->pos < b->pos);
}

static void heap_push(BpeCand *h, int *n, BpeCand c) {
    int i = (*n)++;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (!cand_before(&c, &h[p])) break;
        h[i] = h[p];
        i = p;
    }
    h[i] = c;
}

static BpeCand heap_pop(BpeCand *h, int *n) {
    BpeCand top = h[0], last = h[--(*n)];
    int i = 0;
    for (;;) {
        int l = 2 * i + 1, r = l + 1;
        if (l >= *n) break;
        int m = (r < *n && cand_before(&h[r], &h[l])) ? r : l;
        if (!cand_before(&h[m], &last)) break;
        h[i] = h[m];
        i = m;
    }
    if (*n > 0) h[i] = last;
    return top;
}

static inline void bpe_push(Tokenizer *t, const unsigned char *chunk, BpeScratch *s, int *nh, int a, int b) {
    // symbols a and b are adjacent, so their concatenation is chunk[a .. a + len)
    int len = s->len[a] + s->len[b];
    if ((unsigned int)len > t->max_token_length) return;
    int id = vocab_lookup(t, chunk + a, len);
    if (id < 0) return;
    BpeCand c = { t->merge_scores[id], a, len, id };
    heap_push(s->heap, nh, c);
}

static void bpe_chunk(Tokenizer *t, const unsigned char *chunk, int n, BpeScratch *s, int *tokens, int *n_tokens) {
    if (n == 1) { tokens[(*n_tokens)++] = t->byte_token[chunk[0]]; return; }
    for (int i = 0; i < n; i++) {
        s->len[i] = 1;
        s->tok[i] = t->byte_token[chunk[i]];
        s->prev[i] = i - 1;
    }
    int nh = 0;
    for (int i = 0; i + 1 < n; i++) bpe_push(t, chunk, s, &nh, i, i + 1);
    while (nh > 0) {
        BpeCand c = heap_pop(s->heap, &nh);
        int i = c.pos;
        if (s->len[i] == 0) continue; // symbol merged into its left neighbour since
        int j = i + s->len[i];
        if (j >= n || s->len[i] + s->len[j] != c.len) continue; // stale: the pair changed
        s->tok[i] = c.id;
        s->len[i] = c.len;
        s->len[j] = 0;
        int k = i + c.len;
        if (k < n) s->prev[k] = i;
        if (i > 0) bpe_push(t, chunk, s, &nh, s->prev[i], i);
        if (k < n) bpe_push(t, chunk, s, &nh, i, k);
    }
    for (int i = 0; i < n; i += s->len[i]) tokens[(*n_tokens)++] = s->tok[i];
}

static void encode_segment(Tokenizer *t, const unsigned char *p, const unsigned char *end, BpeScratch *s, int *tokens, int *n_tokens) {
    // text between added tokens: NFC, pre-tokenize, then BPE each chunk
    unsigned char *norm = NULL;
    const unsigned char *q = nfc_check(p, end);
    if (q < end) {
        int n;
        norm = nfc_normalize(p, q, end, &n);
        p = norm;
        end = norm + n;
        bpe_reserve(s, n);
    }
    while (p < end) {
        q = pretok_next(p, end);
        bpe_chunk(t, p, (int)(q - p), s, tokens, n_tokens);
        p = q;
    }
    free(norm);
}

static int match_added(Tokenizer *t, const unsigned char *p, const unsigned char *end, int *len) {
    // longest added token starting at p (t->added is sorted longest first), or -1
    for (int k = 0; k < t->n_added; k++) {
        int id = t->added[k], n = t->vocab_len[id];
        if (n <= end - p && !memcmp(p, t->vocab[id], n)) { *len = n; return id; }
    }
    return -1;
}

void encode(Tokenizer *t, char *text, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array of
    // ENCODE_MAX_TOKENS(strlen(text)) ints
    const unsigned char *s = (const unsigned char *)text, *end = s + strlen(text);
    BpeScratch sc = {0};
    bpe_reserve(&sc, (int)(end - s));

    *n_tokens = 0;
    const unsigned char *seg = s;
    for (const unsigned char *p = s; p < end;) {
        int len, id = t->added_first[*p] ? match_added(t, p, end, &len) : -1;
        if (id < 0) { p++; continue; }
        encode_segment(t, seg, p, &sc, tokens, n_tokens);
        tokens[(*n_tokens)++] = id;
        p += len;
        seg = p;
    }
    encode_segment(t, seg, end, &sc, tokens, n_tokens);

    free(sc.len);
    free(sc.heap);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex *probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float *probabilities, int n) {
    // return the index that has the highest probability: the first one holding the maximum (as the
    // original scalar loop with a strict >). Two passes that vectorize, the maximum (exact in any
    // order) and the search for its first occurrence: ~125 -> ~20 us per token on 151,936 logits
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++)
        max_p = probabilities[i] > max_p ? probabilities[i] : max_p;
    int i = 0;
#if defined(__AVX512F__) && !defined(RUNQ_GENERIC)
    __m512 m = _mm512_set1_ps(max_p);
    for (; i + 16 <= n; i += 16) {
        __mmask16 k = _mm512_cmp_ps_mask(_mm512_loadu_ps(probabilities + i), m, _CMP_EQ_OQ);
        if (k)
            return i + __builtin_ctz(k);
    }
#endif
    for (; i < n; i++)
        if (probabilities[i] == max_p)
            return i;
    return 0;
}

int sample_mult(float *probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf)
            return i;
    }
    return n - 1; // in case of rounding errors
}

static inline int topp_before(const ProbIndex *a, const ProbIndex *b) {
    // nucleus order: higher probability first, lower index first on ties. (The old code sorted the
    // candidates, collected in index order, with qsort on the probability only: glibc's qsort is a
    // stable merge sort at these sizes (checked on glibc 2.41), so ties came out by index too)
    return a->prob > b->prob || (a->prob == b->prob && a->index < b->index);
}

static void topp_sift_down(ProbIndex *h, int n, int i) {
    // max-heap (topp_before order) of h[0..n): restore the heap property below i
    ProbIndex x = h[i];
    for (;;) {
        int l = 2 * i + 1;
        if (l >= n) break;
        int m = l + 1 < n && topp_before(&h[l + 1], &h[l]) ? l + 1 : l;
        if (!topp_before(&h[m], &x)) break;
        h[i] = h[m];
        i = m;
    }
    h[i] = x;
}

static int topp_candidates(const float *p, int n, float cutoff, ProbIndex *out) {
    // the (prob, index) of every p[i] >= cutoff, in index order; returns their count. They are few
    // (median ~40 of 151,936 at -t 0.8 -p 0.9), so the scan looks for blocks holding any
    int n0 = 0, i = 0;
#if defined(__AVX512F__) && !defined(RUNQ_GENERIC)
    __m512 c = _mm512_set1_ps(cutoff);
    for (; i + 16 <= n; i += 16) // one compare per 16 values, then the set bits
        for (__mmask16 m = _mm512_cmp_ps_mask(_mm512_loadu_ps(p + i), c, _CMP_GE_OQ); m; m &= m - 1) {
            int j = i + __builtin_ctz(m);
            out[n0].index = j;
            out[n0].prob = p[j];
            n0++;
        }
#else
    for (; i + 64 <= n; i += 64) { // blocks of 64: their max in a loop gcc vectorizes, then a scan
        float m = p[i];
        for (int j = 1; j < 64; j++)
            m = p[i + j] > m ? p[i + j] : m;
        if (!(m >= cutoff))
            continue;
        for (int j = i; j < i + 64; j++)
            if (p[j] >= cutoff) {
                out[n0].index = j;
                out[n0].prob = p[j];
                n0++;
            }
    }
#endif
    for (; i < n; i++)
        if (p[i] >= cutoff) {
            out[n0].index = i;
            out[n0].prob = p[i];
            n0++;
        }
    return n0;
}

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()
    // Values smaller than (1 - topp) / (n - 1) cannot be part of the result, so only the others
    // are candidates. They are not sorted (was: qsort, ~5 ms at -t 2 with ~44k candidates): a
    // max-heap of them is popped only until the nucleus is complete (heapify O(n0), then
    // O(log n0) per nucleus token). The same order (topp_before), the same sums in the same order,
    // so the same token for the same coin as the old full sort.
    const float cutoff = (1.0f - topp) / (n - 1);
    int n0 = topp_candidates(probabilities, n, cutoff, probindex);
    for (int i = n0 / 2 - 1; i >= 0; i--)
        topp_sift_down(probindex, n0, i);

    // pop the candidates in order (the i-th one lands at probindex[n0 - 1 - i]) until the
    // cumulative probability exceeds topp
    float cumulative_prob = 0;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        int h = n0 - i; // heap size
        ProbIndex top = probindex[0];
        probindex[0] = probindex[h - 1];
        probindex[h - 1] = top;
        topp_sift_down(probindex, h - 1, 0);
        cumulative_prob += top.prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[n0 - 1 - i].prob;
        if (r < cdf)
            return probindex[n0 - 1 - i].index;
    }
    return probindex[n0 - 1 - last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler *sampler, float *logits) {
    // sample the token given the logits and some hyperparameters
    if (sampler->temperature == 0) {
        // greedy argmax sampling: take the token with the highest probability
        return sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q = 0; q < sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            return sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            return sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
}

// ----------------------------------------------------------------------------
// timing / step limit (cwayn patch)

int max_new_tokens = 0; // -n: stop after this many generated tokens, 0 = unlimited

double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void report_timing(int n_prompt, double t0, double t_prefill, int n_gen, double t_end) {
    double pf = t_prefill - t0, dc = t_end - t_prefill;
    fprintf(stderr, "\n[timing] prompt: %d tok in %.3f s (%.1f tok/s) | generated: %d tok in %.3f s (%.1f tok/s)\n",
            n_prompt, pf, n_prompt / (pf > 0 ? pf : 1e-9), n_gen, dc, n_gen / (dc > 0 ? dc : 1e-9));
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int*)malloc(ENCODE_MAX_TOKENS(strlen(prompt)) * sizeof(int));
    encode(tokenizer, prompt, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "Please provide a prompt using -i <string> on the command line.\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    int next = 0;    // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    int n_gen = 0;
    double t0 = now_s(), t_prefill = t0;
    float *pre_logits = NULL; // logits of the last prompt token, from the batched prefill

    if (transformer->state.nb_max > 1 && num_prompt_tokens <= transformer->config.seq_len) {
        // batched prefill of the whole prompt, then resume the loop at its last token
        pre_logits = prefill(transformer, prompt_tokens, num_prompt_tokens, 0);
        for (int i = 0; i < num_prompt_tokens - 1; i++)
            printf("%s", decode(tokenizer, prompt_tokens[i]));
        fflush(stdout);
        pos = num_prompt_tokens - 1;
        token = prompt_tokens[pos];
    }

    while (pos < transformer->config.seq_len) {
        // forward the transformer to get logits for the next token
        // (only needed from the last prompt token on)
        float *logits = pre_logits ? pre_logits : forward(transformer, token, pos, pos >= num_prompt_tokens - 1);
        pre_logits = NULL;

        // advance the state state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            if (pos == num_prompt_tokens - 1) t_prefill = now_s();
            else n_gen++;
            next = sample(sampler, logits);
        }
        pos++;
        if (max_new_tokens > 0 && n_gen >= max_new_tokens) break;

        // print the token as string, decode it with the Tokenizer object
        printf("%s", decode(tokenizer, token));
        fflush(stdout);
        token = next;

        // data-dependent terminating condition: the BOS token delimits sequences
        if (pos >= num_prompt_tokens && (next == tokenizer->bos_token_id || next == tokenizer->eos_token_id))
            break;
    }
    printf("\n");
    report_timing(num_prompt_tokens, t0, t_prefill, n_gen, now_s());
    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// chat loop (cwayn: multi-turn, the conversation stays in the KV cache)
//
// The conversation is kept as token pieces, rendered with Qwen3's chat template (ChatML,
// models/Qwen3-0.6B/tokenizer_config.json; the exporter's .template* files hold its renderings):
//   [system]  "<|im_start|>system\n" SYS "<|im_end|>\n"                       (-y; = the start of
//             .template.with-system, i.e. that file minus .template)
//   turn i    P_i = "<|im_start|>user\n" U_i "<|im_end|>\n<|im_start|>assistant\n" [+ G]
//             (= .template / .template.with-thinking; G = "<think>\n\n</think>\n\n" with -r 0)
//             then the reply R_i as generated (token ids, without the final <|im_end|>)
//             then E = "<|im_end|>\n" (template: '<|im_end|>' + '\n' after every message)
// A new turn k is the canonical sequence  [system] + sum over kept turns i < k (P_i + H_i + E) + P_k,
// where H_i is the history form of R_i: -r 0: R_i as generated (and the empty think block G stays in
// P_i); -r 1: R_i after its last </think>, leading newlines stripped, as the template does for
// earlier assistant turns (it drops their reasoning). Of that sequence, the longest prefix that is
// already in the cache is kept and the rest is prefilled from there: normally just
// E + P_k (the sampled <|im_end|> that ended the reply is the first token of the next prefill, so
// it is in the cache before the next turn), with -r 1 from the start of the previous answer
// (its reasoning is dropped and the answer re-prefilled). Text pieces are encoded separately; they
// are separated by added tokens (<|im_start|>, <|im_end|>), so this is the same as encoding the
// whole rendered string.
// Context full: if the canonical sequence plus a reserve for the reply (-n, else seq_len / 4, at most
// seq_len / 2) does not fit in -c, the oldest turns are dropped until it is at most half of the
// rest (seq_len - reserve) / 2, keeping the system prompt (cached prefix) and the newest turns, and
// the rest is re-prefilled. A reply reaching the end of the context is cut there.
// RNG: one draw per prefilled token but the last, then one per sampled token (seeded output does
// not depend on -b; for a single turn the same draws as before batched prefill).

typedef struct { int *v; int n, cap; } IntVec;

static void iv_reserve(IntVec *a, size_t n) {
    if (n <= (size_t)a->cap) return;
    size_t cap = a->cap ? (size_t)a->cap : 256;
    while (cap < n) cap *= 2;
    a->v = realloc(a->v, cap * sizeof(int));
    if (!a->v || cap > 0x7fffffff) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    a->cap = (int)cap;
}

static void iv_append(IntVec *a, const int *x, int n) {
    if (n <= 0) return; // (x may be NULL then)
    iv_reserve(a, (size_t)a->n + n);
    memcpy(a->v + a->n, x, n * sizeof(int));
    a->n += n;
}

static void iv_push(IntVec *a, int x) { iv_append(a, &x, 1); }

static void encode_append(Tokenizer *t, IntVec *v, const char *text) {
    // append the tokens of text (raw, no template)
    iv_reserve(v, (size_t)v->n + ENCODE_MAX_TOKENS(strlen(text)));
    int n = 0;
    encode(t, (char *)text, v->v + v->n, &n);
    v->n += n;
}

static char *render1(const char *tmpl, const char *arg) {
    // tmpl with its one %s replaced by arg (malloc'd)
    int n = snprintf(NULL, 0, tmpl, arg);
    char *s = malloc((size_t)n + 1);
    if (!s) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    snprintf(s, (size_t)n + 1, tmpl, arg);
    return s;
}

static char *read_line(FILE *f) {
    // one line of any length without its "\n" (or "\r\n"), malloc'd; NULL at EOF with nothing read
    size_t cap = 256, n = 0;
    char *s = malloc(cap);
    if (!s) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    int c;
    while ((c = fgetc(f)) != EOF && c != '\n') {
        if (n + 1 >= cap && !(s = realloc(s, cap *= 2))) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
        s[n++] = (char)c;
    }
    if (c == EOF && n == 0) { free(s); return NULL; }
    if (n > 0 && s[n - 1] == '\r') n--;
    s[n] = 0;
    return s;
}

static float *feed_tokens(Transformer *tr, const int *tokens, int n, int pos0) {
    // tokens[0..n-1] at positions pos0.. (n >= 1, pos0 + n <= seq_len): batched prefill, or one
    // token at a time with -b 0/1; returns the logits of the last one
    if (tr->state.nb_max > 1)
        return prefill(tr, (int *)tokens, n, pos0);
    float *logits = NULL;
    for (int i = 0; i < n; i++)
        logits = forward(tr, tokens[i], pos0 + i, i == n - 1);
    return logits;
}

static void history_reply(Tokenizer *t, const IntVec *r, int think_end, IntVec *h) {
    // -r 1: the reply as the template renders an earlier assistant turn: the text after the last
    // </think> (if any), leading "\n" stripped (content.split('</think>')[-1].lstrip('\n'))
    h->n = 0;
    int i = 0;
    for (int j = r->n - 1; j >= 0 && think_end >= 0; j--)
        if (r->v[j] == think_end) { i = j + 1; break; }
    if (i == 0) { iv_append(h, r->v, r->n); return; } // no reasoning block: as generated
    for (; i < r->n; i++) {
        const char *s = t->vocab[r->v[i]];
        int m = (int)strspn(s, "\n");
        if (m == t->vocab_len[r->v[i]]) continue; // only newlines
        if (m > 0) encode_append(t, h, s + m), i++; // newlines + text in one token: re-encode the text
        break;
    }
    iv_append(h, r->v + i, r->n - i);
}

// test hook: RUNQ_CHAT_DUMP=<prefix> writes <prefix>.ids (the token ids in the KV cache at the end,
// space-separated) and <prefix>.logits (int32 n, int32 vocab, int32 pos[n], float32 logits[n][vocab]:
// every logits vector chat sampled from, pos = the position of the token that produced it; those of
// positions overwritten later, by a re-prefill, are dropped). tests/check.py compares them with
// -m logits on the decoded conversation
static struct { const char *prefix; int n, cap, vocab; int *pos; float *logits; } cdump;

static void cdump_record(int pos, const float *logits) {
    if (!cdump.prefix) return;
    if (cdump.n == cdump.cap) {
        cdump.cap = cdump.cap ? 2 * cdump.cap : 64;
        cdump.pos = realloc(cdump.pos, cdump.cap * sizeof(int));
        cdump.logits = realloc(cdump.logits, (size_t)cdump.cap * cdump.vocab * sizeof(float));
        if (!cdump.pos || !cdump.logits) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    }
    cdump.pos[cdump.n] = pos;
    memcpy(cdump.logits + (size_t)cdump.n * cdump.vocab, logits, cdump.vocab * sizeof(float));
    cdump.n++;
}

static void cdump_truncate(int pos) {
    // positions >= pos are about to be overwritten
    while (cdump.n > 0 && cdump.pos[cdump.n - 1] >= pos)
        cdump.n--;
}

static void cdump_write(const IntVec *kv) {
    if (!cdump.prefix) return;
    size_t len = strlen(cdump.prefix) + 8;
    char *path = malloc(len);
    snprintf(path, len, "%s.ids", cdump.prefix);
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Couldn't open %s for writing\n", path); exit(EXIT_FAILURE); }
    for (int i = 0; i < kv->n; i++)
        fprintf(f, i ? " %d" : "%d", kv->v[i]);
    fprintf(f, "\n");
    fclose(f);
    snprintf(path, len, "%s.logits", cdump.prefix);
    f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Couldn't open %s for writing\n", path); exit(EXIT_FAILURE); }
    int32_t header[2] = { cdump.n, cdump.vocab };
    int ok = fwrite(header, sizeof(int32_t), 2, f) == 2 && fwrite(cdump.pos, sizeof(int32_t), cdump.n, f) == (size_t)cdump.n
          && fwrite(cdump.logits, sizeof(float), (size_t)cdump.n * cdump.vocab, f) == (size_t)cdump.n * cdump.vocab;
    if (fclose(f) != 0 || !ok) { fprintf(stderr, "Error writing %s\n", path); exit(EXIT_FAILURE); }
    free(path);
}

typedef struct {
    IntVec prompt; // P_i: user message + assistant header (+ empty think block with -r 0)
    IntVec reply;  // R_i: the generated tokens, without the <|im_end|> that ended them
    IntVec hist;   // H_i: the reply as part of the history
} ChatTurn;

static void render_conversation(IntVec *out, const IntVec *sys, const ChatTurn *turns, int j0, int k, const IntVec *close) {
    // [system] + turns j0..k-1 (prompt, history reply, <|im_end|>\n) + the prompt of turn k
    out->n = 0;
    iv_append(out, sys->v, sys->n);
    for (int i = j0; i < k; i++) {
        iv_append(out, turns[i].prompt.v, turns[i].prompt.n);
        iv_append(out, turns[i].hist.v, turns[i].hist.n);
        iv_append(out, close->v, close->n);
    }
    iv_append(out, turns[k].prompt.v, turns[k].prompt.n);
}

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *cli_user_prompt, char *system_prompt,
          int enable_thinking) {
    int seq_len = transformer->config.seq_len;
    IntVec sys = { 0 }, close = { 0 }, kv = { 0 }, want = { 0 };
    ChatTurn *turns = NULL;
    int n_turns = 0, j0 = 0; // turns[j0..] are in the conversation, older ones were dropped (context full)

    // the system block template: .template.with-system = system block + .template
    char sys_tmpl[1024] = "<|im_start|>system\n%s<|im_end|>\n";
    size_t ls = strlen(tokenizer->system_prompt_template), lp = strlen(tokenizer->prompt_template);
    if (ls > lp && !strcmp(tokenizer->system_prompt_template + ls - lp, tokenizer->prompt_template)) {
        memcpy(sys_tmpl, tokenizer->system_prompt_template, ls - lp);
        sys_tmpl[ls - lp] = 0;
    }
    if (system_prompt) {
        char *s = render1(sys_tmpl, system_prompt);
        encode_append(tokenizer, &sys, s);
        free(s);
    }
    encode_append(tokenizer, &close, "<|im_end|>\n");
    IntVec te = { 0 };
    encode_append(tokenizer, &te, "</think>");
    int think_end = te.n == 1 ? te.v[0] : -1;
    free(te.v);
    int reserve = max_new_tokens > 0 ? max_new_tokens : seq_len / 4;
    if (reserve > seq_len / 2) reserve = seq_len / 2;
    if (reserve < 1) reserve = 1;
    cdump.prefix = getenv("RUNQ_CHAT_DUMP");
    cdump.vocab = transformer->config.vocab_size;
#ifndef _WIN32
    int echo = !isatty(fileno(stdin)); // piped stdin: echo each user line, for a readable transcript
#else
    int echo = 0;
#endif

    for (int asked = 0;; asked++) {
        // the user message: -i (one turn only), else one line of stdin per turn (empty line or EOF ends)
        char *user;
        if (cli_user_prompt) {
            if (asked) break;
            user = render1("%s", cli_user_prompt);
        } else {
            printf("\n> ");
            fflush(stdout);
            user = read_line(stdin);
            if (!user || !user[0]) { free(user); break; }
            if (echo) printf("%s\n", user);
            fflush(stdout);
        }
        turns = realloc(turns, (size_t)(n_turns + 1) * sizeof(ChatTurn));
        if (!turns) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
        int k = n_turns;
        ChatTurn *tn = &turns[k];
        memset(tn, 0, sizeof(*tn));
        char *r = render1(tokenizer->prompt_template, user);
        encode_append(tokenizer, &tn->prompt, r);
        free(r);
        free(user);

        // the canonical conversation; context full: drop the oldest turns (see above)
        int j = j0;
        render_conversation(&want, &sys, turns, j, k, &close);
        if (want.n + reserve > seq_len)
            while (j < k && want.n > (seq_len - reserve) / 2)
                render_conversation(&want, &sys, turns, ++j, k, &close);
        if (want.n > seq_len) {
            fprintf(stderr, "(message too long: %d tokens with the template%s, context %d (-c); skipped)\n", want.n,
                    sys.n ? " and the system prompt" : "", seq_len);
            free(tn->prompt.v);
            continue;
        }
        n_turns++;
        if (j > j0)
            fprintf(stderr, "[context] full (-c %d): dropped the %d oldest turn(s), kept %s%d earlier turn(s)\n", seq_len,
                    j - j0, sys.n ? "the system prompt and " : "", k - j);
        j0 = j;

        // prefill what is not in the cache yet (the cache holds kv.v[0 .. kv.n), positions 0 .. kv.n-1)
        int L = 0;
        while (L < kv.n && L < want.n && kv.v[L] == want.v[L]) L++;
        cdump_truncate(L);
        kv.n = L;
        int n_new = want.n - L;
        double t0 = now_s();
        float *logits = feed_tokens(transformer, want.v + L, n_new, L);
        iv_append(&kv, want.v + L, n_new);
        if (sampler->temperature != 0)
            for (int i = 0; i < n_new - 1; i++)
                random_f32(&sampler->rng_state);
        double t_prefill = now_s();

        // the reply: sample, print, feed back; ends at <|im_end|> (sampled, not fed yet: the next
        // turn's prefill starts with it), at -n tokens, or at the end of the context
        int n_gen = 0;
        while (1) {
            cdump_record(kv.n - 1, logits);
            int next = sample(sampler, logits);
            if (next == (int)tokenizer->eos_token_id || next == (int)tokenizer->bos_token_id)
                break;
            printf("%s", decode(tokenizer, next));
            fflush(stdout);
            iv_push(&tn->reply, next);
            if (kv.n >= seq_len) {
                fprintf(stderr, "\n(context window full (-c %d): reply cut)", seq_len);
                break;
            }
            int last = max_new_tokens > 0 && n_gen + 1 >= max_new_tokens;
            logits = forward(transformer, next, kv.n, !last);
            iv_push(&kv, next);
            n_gen++;
            if (last)
                break;
        }
        printf("\n");
        report_timing(n_new, t0, t_prefill, n_gen, now_s());
        if (enable_thinking)
            history_reply(tokenizer, &tn->reply, think_end, &tn->hist);
        else
            iv_append(&tn->hist, tn->reply.v, tn->reply.n);
    }
    cdump_write(&kv);
    for (int i = 0; i < n_turns; i++) {
        free(turns[i].prompt.v);
        free(turns[i].reply.v);
        free(turns[i].hist.v);
    }
    free(turns);
    free(sys.v);
    free(close.v);
    free(kv.v);
    free(want.v);
    free(cdump.pos);
    free(cdump.logits);
}

// ----------------------------------------------------------------------------
// test hooks (cwayn patch): -m tokenize, -m logits (see tests/)

void tokenize(Tokenizer *tokenizer, char *text, double t_load) {
    // encode the raw text (no chat template), print the token ids space-separated on stdout;
    // timings on stderr (encode only, and the tokenizer load before it)
    int n_tokens = 0;
    int *tokens = (int *)malloc(ENCODE_MAX_TOKENS(strlen(text)) * sizeof(int));
    double t0 = now_s();
    encode(tokenizer, text, tokens, &n_tokens);
    double t_enc = now_s() - t0;
    for (int i = 0; i < n_tokens; i++)
        printf(i ? " %d" : "%d", tokens[i]);
    printf("\n");
    fprintf(stderr, "[tokenize] %zu bytes -> %d tokens, encode %.3f ms, tokenizer load %.1f ms\n",
            strlen(text), n_tokens, t_enc * 1e3, t_load * 1e3);
    free(tokens);
}

void forward_sequential(Transformer *transformer, int *tokens, int n_tokens, float *out) {
    // reference prefill: one token at a time at positions 0..n_tokens-1,
    // row pos of out (n_tokens, vocab_size) = logits after feeding tokens[pos]
    int vocab_size = transformer->config.vocab_size;
    for (int pos = 0; pos < n_tokens; pos++) {
        float *logits = forward(transformer, tokens[pos], pos, 1);
        memcpy(out + (size_t)pos * vocab_size, logits, vocab_size * sizeof(float));
    }
}

void forward_chunked(Transformer *transformer, int *tokens, int n_tokens, float *out) {
    // the same with the batched path: chunks of state.nb_max rows, logits of every row
    int vocab_size = transformer->config.vocab_size, nb = transformer->state.nb_max;
    for (int i = 0; i < n_tokens; i += nb) {
        int m = n_tokens - i < nb ? n_tokens - i : nb;
        forward_batch(transformer, tokens + i, m, i, LOGITS_ALL, out + (size_t)i * vocab_size);
    }
}

void dump_logits(Transformer *transformer, Tokenizer *tokenizer, char *text, char *out_path) {
    // file format: int32 n_tokens, int32 vocab_size, int32 ids[n_tokens], float32 logits[n_tokens][vocab_size]
    // -b > 1: batched path (forward_chunked), else one token at a time (forward_sequential)
    int n_tokens = 0, vocab_size = transformer->config.vocab_size;
    int *tokens = (int *)malloc(ENCODE_MAX_TOKENS(strlen(text)) * sizeof(int));
    encode(tokenizer, text, tokens, &n_tokens);
    if (n_tokens < 1 || n_tokens > transformer->config.seq_len) {
        fprintf(stderr, "dump_logits: need 1..%d tokens, got %d\n", transformer->config.seq_len, n_tokens);
        exit(EXIT_FAILURE);
    }

    float *out = (float *)malloc((size_t)n_tokens * vocab_size * sizeof(float));
    if (!out) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    if (transformer->state.nb_max > 1)
        forward_chunked(transformer, tokens, n_tokens, out);
    else
        forward_sequential(transformer, tokens, n_tokens, out);

    FILE *file = fopen(out_path, "wb");
    if (!file) { fprintf(stderr, "Couldn't open %s for writing\n", out_path); exit(EXIT_FAILURE); }
    int32_t header[2] = { n_tokens, vocab_size };
    int ok = fwrite(header, sizeof(int32_t), 2, file) == 2
          && fwrite(tokens, sizeof(int32_t), n_tokens, file) == (size_t)n_tokens
          && fwrite(out, sizeof(float), (size_t)n_tokens * vocab_size, file) == (size_t)n_tokens * vocab_size;
    if (fclose(file) != 0 || !ok) { fprintf(stderr, "Error writing %s\n", out_path); exit(EXIT_FAILURE); }
    free(out);
    free(tokens);
}

void dump_dequant(Transformer *tr, const char *spec, const char *out_path) {
    // test hook, GGUF only: -m dequant -i NAME[:ROW0[:NROWS]] -o FILE writes rows ROW0.. (default 0,
    // 1 row) of the matrix NAME (llama.cpp tensor name) dequantized from the file's blocks (embed_row:
    // ggml's expressions) and, for Q4_K / Q6_K, also from the repacked weights (kq_packed_row, the
    // same expressions: checks the repacking). FILE: int32 nrows, ncols, npacked (0 or 1), float32
    // [nrows][ncols] from the blocks, then [nrows][ncols] from the repacked layout if npacked
    Config *p = &tr->config;
    TransformerWeights *w = &tr->weights;
    if (!tr->gguf.map) { fprintf(stderr, "-m dequant: GGUF files only\n"); exit(EXIT_FAILURE); }
    int L = p->n_layers, dim = p->dim, hid = p->hidden_dim, qd = p->n_heads * p->head_dim, kvd = p->n_kv_heads * p->head_dim;
    struct { const char *fmt; QuantizedTensor *t; int n, d, k; } m[] = {
        { "token_embd.weight", w->q_tokens, 1, p->vocab_size, dim }, { "output.weight", w->wcls, 1, p->vocab_size, dim },
        { "blk.%d.attn_q.weight", w->wq, L, qd, dim }, { "blk.%d.attn_k.weight", w->wk, L, kvd, dim },
        { "blk.%d.attn_v.weight", w->wv, L, kvd, dim }, { "blk.%d.attn_output.weight", w->wo, L, dim, qd },
        { "blk.%d.ffn_gate.weight", w->w1, L, hid, dim }, { "blk.%d.ffn_down.weight", w->w2, L, dim, hid },
        { "blk.%d.ffn_up.weight", w->w3, L, hid, dim } };
    size_t nlen = strcspn(spec, ":");
    long r0 = 0, nr = 1;
    if (spec[nlen]) {
        char *e;
        r0 = strtol(spec + nlen + 1, &e, 10);
        if (*e == ':') nr = strtol(e + 1, &e, 10);
        if (*e) { fprintf(stderr, "-m dequant: bad spec %s\n", spec); exit(EXIT_FAILURE); }
    }
    QuantizedTensor *t = NULL;
    int d = 0, k = 0;
    for (size_t i = 0; i < sizeof(m) / sizeof(m[0]) && !t; i++) {
        if (i == 1 && p->shared_classifier) continue; // (no output.weight: tied)
        for (int l = 0; l < m[i].n && !t; l++) {
            char name[96];
            snprintf(name, sizeof(name), m[i].fmt, l);
            if (strlen(name) == nlen && !memcmp(name, spec, nlen)) { t = m[i].t + l; d = m[i].d; k = m[i].k; }
        }
    }
    if (!t || !t->b) { fprintf(stderr, "-m dequant: no matrix %.*s\n", (int)nlen, spec); exit(EXIT_FAILURE); }
    if (r0 < 0 || nr < 1 || r0 > d - nr) { fprintf(stderr, "-m dequant: rows %ld..%ld out of range (%d rows)\n", r0, r0 + nr - 1, d); exit(EXIT_FAILURE); }
    float *out = malloc((size_t)2 * nr * k * sizeof(float));
    if (!out) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    int32_t hdr[3] = { (int32_t)nr, k, t->kp != NULL };
    for (long r = 0; r < nr; r++) {
        embed_row(out + (size_t)r * k, t, (int)(r0 + r), k);
        if (t->kp)
            kq_packed_row(t->type, t->kp, (int)(r0 + r), k, out + (size_t)(nr + r) * k);
    }
    FILE *f = fopen(out_path, "wb");
    if (!f) { fprintf(stderr, "Couldn't open %s for writing\n", out_path); exit(EXIT_FAILURE); }
    size_t nv = (size_t)(1 + hdr[2]) * nr * k;
    int ok = fwrite(hdr, sizeof(int32_t), 3, f) == 3 && fwrite(out, sizeof(float), nv, f) == nv;
    if (fclose(f) != 0 || !ok) { fprintf(stderr, "Error writing %s\n", out_path); exit(EXIT_FAILURE); }
    free(out);
}

// ----------------------------------------------------------------------------
// CLI

void error_usage() {
    fprintf(stderr, "Usage:   runq <checkpoint> [options]\n");
    fprintf(stderr, "         checkpoint: a qwen3.c .bin (with its .tokenizer / .template* files) or a GGUF file (Q8_0, Q4_K_M)\n");
    fprintf(stderr, "Example: runq Qwen3-0.6B-Q8_0.gguf -c 4096 -r 1\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1], default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -c <int>    context window size, 0 (default) = max_seq_len; chat: holds the whole conversation,\n");
    fprintf(stderr, "              when full the oldest turns are dropped (the system prompt is kept)\n");
    fprintf(stderr, "  -m <string> mode: generate|chat|tokenize|logits|dequant, default: chat\n");
    fprintf(stderr, "              tokenize: print the token ids of -i (raw text, no template)\n");
    fprintf(stderr, "              logits: write the logits at every position of -i to -o (binary)\n");
    fprintf(stderr, "  -i <string> input prompt (chat: one turn, then exit; without -i, one turn per stdin line,\n");
    fprintf(stderr, "              the conversation is kept until an empty line or EOF)\n");
    fprintf(stderr, "              dequant (GGUF, test hook): rows of the matrix -i NAME[:ROW0[:NROWS]] to -o (binary)\n");
    fprintf(stderr, "  -o <string> output file for -m logits / dequant\n");
    fprintf(stderr, "  -y <string> system prompt in chat mode, default is none\n");
    fprintf(stderr, "  -r <int>    reasoning mode, 0 (default) = no thinking, 1 = thinking\n");
    fprintf(stderr, "  -n <int>    max new tokens per response, 0 (default) = unlimited\n");
    fprintf(stderr, "  -b <int>    prefill chunk size (batched prompt processing), default %d, max %d;\n", PREFILL_CHUNK_DEFAULT, B_MAX);
    fprintf(stderr, "              0 or 1 = one token at a time\n");
    fprintf(stderr, "Threads: OMP_NUM_THREADS, default min(12, nproc - nproc/4) (all CPUs collapse on a busy machine)\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    float temperature = 1.0f;   // 0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "chat";        // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode
    int enable_thinking = 0;    // 1 enables thinking
    int ctx_length = 0;         // context length
    char *out_path = NULL;      // output file for -m logits

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'c') { ctx_length = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else if (argv[i][1] == 'r') { enable_thinking = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { max_new_tokens = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'o') { out_path = argv[i + 1]; }
        else if (argv[i][1] == 'b') { prefill_chunk = atoi(argv[i + 1]); }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0) temperature = 0;
    if (topp < 0 || 1.0 < topp) topp = 0.9;
    if ((strcmp(mode, "tokenize") == 0 || strcmp(mode, "logits") == 0) && !prompt) { error_usage(); }
    if ((strcmp(mode, "logits") == 0 || strcmp(mode, "dequant") == 0) && (!out_path || !prompt)) { error_usage(); }
    if (prefill_chunk < 0) prefill_chunk = 0;
    if (prefill_chunk > B_MAX) prefill_chunk = B_MAX;

    if (strcmp(mode, "tokenize") == 0) {
        // only the header is needed (vocab_size): skip mapping and dequantizing the weights
        // (GGUF: the metadata, the tensor data stays unread)
        Config config;
        Tokenizer tokenizer;
        rq_set_threads(1); // (one light process: the tests run many in parallel)
        double t0 = now_s();
        if (file_is_gguf(checkpoint_path)) {
            Gguf g;
            gguf_open(&g, checkpoint_path);
            gguf_config(&g, &config);
            build_tokenizer_gguf(&tokenizer, &g, config.vocab_size, enable_thinking);
            munmap((void *)g.map, g.size);
            free(g.kv);
            free(g.t);
        } else {
            FILE *file = fopen(checkpoint_path, "rb");
            if (!file || fread(&config, sizeof(Config), 1, file) != 1 || config.magic_number != 0x616a6331) {
                fprintf(stderr, "Couldn't read qwen3.c checkpoint header from %s\n", checkpoint_path);
                exit(EXIT_FAILURE);
            }
            fclose(file);
            build_tokenizer(&tokenizer, checkpoint_path, config.vocab_size, enable_thinking);
        }
        tokenize(&tokenizer, prompt, now_s() - t0);
        free_tokenizer(&tokenizer);
        return 0;
    }

    // threads: OMP_NUM_THREADS, else the default: not all CPUs. With OpenMP's spin barriers all of
    // them collapsed on a shared machine (any other process preempting one thread stalled the
    // others at every barrier, ~200 per token: 16 threads ~20 tok/s decode on the 16-vCPU dev VM)
    if (rq_env_threads()) {
        rq_set_threads(rq_env_threads());
    } else {
        int np = rq_ncpu();
        int nt = np > 4 ? np - np / 4 : np;
        rq_set_threads(nt < 12 ? nt : 12);
    }
    rq_config();

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path, ctx_length, prefill_chunk);
    if (strcmp(mode, "dequant") == 0) {
        dump_dequant(&transformer, prompt, out_path);
        free_transformer(&transformer);
        return 0;
    }

    // build the Tokenizer via the tokenizer .bin file (GGUF: from its metadata)
    Tokenizer tokenizer;
    if (transformer.gguf.map)
        build_tokenizer_gguf(&tokenizer, &transformer.gguf, transformer.config.vocab_size, enable_thinking);
    else
        build_tokenizer(&tokenizer, checkpoint_path, transformer.config.vocab_size, enable_thinking);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    if (!prompt)
        printf("hidden_size=%d, intermediate_size=%d, num_hidden_layers=%d, num_attention_heads=%d, num_kv_heads=%d, head_dim=%d, ctx_length=%d, vocab_size=%d, shared_classifier=%d, quantization_block_size=%d\n", transformer.config.dim, transformer.config.hidden_dim, transformer.config.n_layers, transformer.config.n_heads, transformer.config.n_kv_heads, transformer.config.head_dim, transformer.config.seq_len, transformer.config.vocab_size, transformer.config.shared_classifier, transformer.config.group_size);

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, enable_thinking);
    } else if (strcmp(mode, "logits") == 0) {
        dump_logits(&transformer, &tokenizer, prompt, out_path);
    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
#ifdef RUNQ_PROFILE
    double pf_total = 0;
    for (int i = 0; i < PF_N; i++) pf_total += pf_time[i];
    fprintf(stderr, "[profile] forward_batch total %.3f s\n", pf_total);
    for (int i = 0; i < PF_N; i++)
        fprintf(stderr, "[profile]   %-18s %8.3f s  %5.1f%%\n", pf_name[i], pf_time[i], 100 * pf_time[i] / (pf_total > 0 ? pf_total : 1));
#endif
    return 0;
}
