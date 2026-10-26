/* Internal definitions shared by the runq modules (not a public API: see runq.h). */
#ifndef RUNQ_INTERNAL_H
#define RUNQ_INTERNAL_H

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
// NEON K-quant kernels (AArch64 with the dot product extension; defines RUNQ_NEON, RUNQ_KQX)
#include "kernels_neon.h"
// K-quants (GGUF Q4_K / Q6_K): block formats, repacked layouts, arithmetic, generic kernels
#include "kquants.h"
// phase synchronization and the parallel region of a decode step / prefill chunk
#include "threads.h"

// ----------------------------------------------------------------------------
// Globals
extern int GS; // group size global for quantization of the weights
extern int KQ; // the model has K-quant weights (GGUF Q4_K / Q6_K): the activation sums xf are needed

// Batched prefill (forward_batch): prompt tokens are processed in chunks of prefill_chunk
// rows (-b, at most B_MAX); 0 or 1 = one token at a time through forward(). Bigger chunks are
// faster (fewer barriers and weight passes per token): 8 threads, 1,500 / 3,000 tokens:
// -b 64 1,237 / 911 tok/s, 128 1,370 / 1,018, 256 1,467 / 1,093, 512 1,547 / 1,112, 1024 within
// noise of 512. The chunk buffers take ~58 KB per row (~30 MB at 512)
#define B_MAX 512
#define PREFILL_CHUNK_DEFAULT 512
extern int prefill_chunk;

// KV cache element type (-k): fp32, bf16 or f16 (16-bit: half the bytes; layouts in kernels_avx512.h)
enum { KV_F32 = 0, KV_BF16 = 1, KV_F16 = 2 };
extern const char *kv_type_name[];
#define KV_TYPE_DEFAULT KV_F16
extern int kv_type;
extern int max_new_tokens;

double now_s(void);
#ifdef RUNQ_PROFILE
// build with -DRUNQ_PROFILE: wall time per forward_batch phase, printed at exit
enum { PF_NORM, PF_QKV, PF_ROPE, PF_ATT, PF_WO, PF_FFN13, PF_SWIGLU, PF_FFN2, PF_CLS, PF_N };
extern const char *pf_name[PF_N];
extern double pf_time[PF_N], pf_t;
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
    uint8_t *p;   // weights only: the same matrix repacked for the SIMD Q8_0 kernels (Q8P layout,
                  // kernels_avx512.h; Q8N on AArch64, kernels_neon.h), or NULL (generic kernels on q / s)
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
    void *key_cache;    // (layer, kv_len, kv_dim) elements: float, or bf16 / f16 (uint16_t) by kvt
    void *value_cache;  // (layer, kv_len, kv_dim)
    int kv_len;
    int kvt;            // cache element type (-k): KV_F32, KV_BF16, KV_F16
    size_t kv_esz;      // bytes per cache element: 4 or 2
    size_t kv_size;     // elements per cache
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

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    float *data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
    Gguf gguf; // GGUF checkpoint: the parsed file (gguf.map == NULL: .bin)
} Transformer;

enum { LOGITS_NONE = 0, LOGITS_LAST = 1, LOGITS_ALL = 2 };

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


// gguf.c
void gguf_fail(const Gguf *g, const char *fmt, const char *arg);
uint64_t gguf_str(const Gguf *g, uint64_t *pos, const char **s, const char *what);
void gguf_parse(Gguf *g);
int gguf_get_int(const Gguf *g, const char *key, int64_t *v);
const char *gguf_get_str(const Gguf *g, const char *key, uint64_t *len);
const GgufKV *gguf_get_array(const Gguf *g, const char *key, uint32_t type);
const GgufTensor *gguf_tensor(const Gguf *g, const char *name);
const uint8_t *gguf_weight(const Gguf *g, const char *name, uint32_t type, uint64_t ne0, uint64_t ne1);
void gguf_config(const Gguf *g, Config *c);
void gguf_open(Gguf *g, const char *path);
int file_is_gguf(const char *path);
// model.c
void *alloc_big(size_t n);
void free_big(void *p, size_t n);
void build_transformer(Transformer *t, char *checkpoint_path, int ctx_length, int nb_max);
void free_transformer(Transformer *t);
// forward.c
void malloc_run_state(RunState* s, Config *p, int nb_max);
void free_run_state(RunState* s);
void embed_row(float *x, const QuantizedTensor *t, int token, int dim);
float *forward(Transformer *transformer, int token, int pos, int compute_logits);
float *forward_batch(Transformer *transformer, int *tokens, int n, int pos0, int logits_mode, float *logits_out);
float *prefill(Transformer *transformer, int *tokens, int n, int pos0);
// tokenizer.c
void build_tokenizer(Tokenizer *t, char *checkpoint_path, int vocab_size, int enable_thinking);
void build_tokenizer_gguf(Tokenizer *t, const Gguf *g, int vocab_size, int enable_thinking);
void free_tokenizer(Tokenizer *t);
char *decode(Tokenizer *t, int token);
void encode(Tokenizer *t, char *text, int *tokens, int *n_tokens);
// sampler.c
void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed);
void free_sampler(Sampler *sampler);
float random_f32(unsigned long long *state);
int sample(Sampler *sampler, float *logits);

#endif // RUNQ_INTERNAL_H
