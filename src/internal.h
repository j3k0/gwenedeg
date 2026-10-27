/* Internal definitions shared by the runq modules (the public API is runq.h):
 *   model.c      RqModel: loading (.bin, GGUF), weight conversion / repacking, chat templates
 *   gguf.c       the GGUF reader (untrusted input: every count, length and offset checked)
 *   forward.c    RqSeq, RqCtx and every numeric part of the forward passes (decode, batched
 *                prefill, attention and matmul dispatch, quantization, norms); one translation unit
 *                so that the -Ofast code generation of the exact arithmetic stays as measured
 *   tokenizer.c  the BPE tokenizer (HF-compatible: added tokens, NFC, Qwen2 / Qwen3.5 pre-tokenizer), detok
 *   sampler.c    greedy / top-p sampling
 *   chat.c       chat template rendering, the generation loop (rq_generate)
 *   threads.c    the thread pool (threads.h: phase synchronization)
 *   cli.c        the runq command line (not part of the library)
 */
#ifndef RUNQ_INTERNAL_H
#define RUNQ_INTERNAL_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <setjmp.h>
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
#include "runq.h"
// AVX-512 VNNI kernels (defines RUNQ_VNNI when built for AVX-512 VNNI without -DRUNQ_GENERIC)
#include "kernels_avx512.h"
// NEON K-quant kernels (AArch64 with the dot product extension; defines RUNQ_NEON, RUNQ_KQX)
#include "kernels_neon.h"
// K-quants (GGUF Q4_K / Q5_K / Q6_K): block formats, repacked layouts, arithmetic, generic kernels
#include "kquants.h"
// phase synchronization and the parallel region of a decode step / prefill chunk
#include "threads.h"

// Activation quantization group size of the loaded model (64 .bin, 32 GGUF): a process-wide
// setting of the kernels, read in their loops (a parameter would change the -Ofast code generation
// of the exact functions); set by rq_model_load, which rejects a second model with another one
extern int GS;

// Batched prefill: rows per chunk (-b, at most B_MAX); 0 or 1 = one token at a time through the decode
// path. Bigger chunks are faster (fewer barriers and weight passes per token): 8 threads, 1,500 /
// 3,000 tokens: -b 64 1,237 / 911 tok/s, 128 1,370 / 1,018, 256 1,467 / 1,093, 512 1,547 / 1,112, 1024
// within noise of 512. The chunk buffers take ~58 KB per row (~30 MB at 512)
#define B_MAX RQ_BATCH_MAX

// Qwen3.5: recurrent-state checkpoints per sequence (rq_seq_checkpoint_at)
#define RQ_SEQ_CK 2

// KV cache element type: fp32, bf16 or f16 (16-bit: half the bytes; layouts in kernels_avx512.h)
enum { KV_F32 = RQ_KV_F32, KV_BF16 = RQ_KV_BF16, KV_F16 = RQ_KV_F16 };

// ---------------------------------------------------------------------------------------- the model

typedef struct {
    // the header of a qwen3.c .bin file (its first 48 bytes; GGUF: filled from the metadata)
    int magic_number; // checkpoint magic number
    int version; // file format version
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size
    int seq_len; // max sequence length
    int head_dim; // head dimension
    int shared_classifier; // 1 if wcls == q_tokens
    int group_size; // quantization group size (export.py uses 64; GGUF Q8_0: 32)
} Config;
#define BIN_MAGIC 0x616a6331

// Qwen3.5 (GGUF qwen35, docs/qwen35.md): the shapes of the hybrid layers (all 0 for Qwen3). Layer l is a
// gated full-attention layer if (l + 1) % interval == 0, else a linear-attention (Gated DeltaNet) layer
typedef struct {
    int on;        // a hybrid model: linear-attention layers, gated attention ([q | gate] per head in attn_q)
    int interval;  // full_attention_interval
    int conv;      // causal conv kernel (ssm.conv_kernel: 4); the state keeps the last conv - 1 inputs
    int dk, nk;    // key head dim (ssm.state_size: 128), key heads (ssm.group_count: 16)
    int dv, nv;    // value head dim (ssm.inner_size / nv: 128), value heads (ssm.time_step_rank: 16; == nk)
    int conv_dim;  // 2 nk dk + nv dv: the qkv projection rows = the conv channels [q | k | v] (6144)
    int vdim;      // nv dv: the output gate z and the input of the output projection (2048)
    int n_lin;     // linear-attention layers (18 of 24)
} Hybrid;

typedef struct {
    int8_t *q;    // quantized values
    float *s; // scaling factors
    uint8_t *p;   // weights only: the same matrix repacked for the SIMD Q8_0 kernels (Q8P layout,
                  // kernels_avx512.h; Q8N on AArch64, kernels_neon.h), or NULL (generic kernels on q / s)
    const uint8_t *b; // GGUF weights: the blocks in the file mapping (Q8_0: per 32 values an fp16
                  // scale + 32 int8, Q8_0_BLOCK bytes; Q4_K / Q5_K / Q6_K: kquants.h), converted at load
                  // (prepare_weights); or NULL
    int type;     // weights: ggml type, GGML_Q8_0 (also the .bin's group-64 Q8), GGML_Q4_K / Q5_K / Q6_K
    uint8_t *kp;  // K-quant weights: repacked (Q4P / Q5P / Q6P layout, kquants.h), else NULL
} QuantizedTensor;

typedef struct {
    // (attention weights are per attention layer, RqModel.kv_layer: every layer for Qwen3, 6 of 24 for Qwen3.5)
    // token embedding table
    QuantizedTensor *q_tokens; // (vocab_size, dim); rows dequantized at lookup (embed_row)
    // weights for rmsnorms
    float *rms_att_weight; // (layer, dim) rmsnorm weights
    float *rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    QuantizedTensor *wq; // (layer, dim, n_heads * head_size); Qwen3.5: (layer, dim, n_heads * 2 head_size), [q | gate] per head
    QuantizedTensor *wk; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wv; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wo; // (layer, n_heads * head_size, dim)
    // QK-RMSNorm for Qwen3
    float *q_norm_weights; // (layer, head_dim)
    float *k_norm_weights;
    // weights for ffn
    QuantizedTensor *w1; // (layer, hidden_dim, dim)
    QuantizedTensor *w2; // (layer, dim, hidden_dim)
    QuantizedTensor *w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float *rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    QuantizedTensor *wcls;
    // Qwen3.5's linear-attention layers (per linear layer, RqModel.lin_layer; NULL for Qwen3); Hybrid shapes
    QuantizedTensor *wqkv; // (conv_dim, dim): the q / k / v projection, the conv's input
    QuantizedTensor *wz;   // (vdim, dim): the output gate z
    QuantizedTensor *wa;   // (nv, dim): alpha, the decay's input
    QuantizedTensor *wb;   // (nv, dim): beta
    QuantizedTensor *wso;  // (dim, vdim): the output projection
    float *ssm_conv;       // the conv kernel (the GGUF's (conv_dim, conv) transposed) per channel group of a head's q / k / v (conv, n) rows
    float *ssm_a;          // (nv): -exp(A_log)
    float *ssm_dt;         // (nv): the dt bias
    float *ssm_norm;       // (dv): the gated RMSNorm's weight (shared by the heads)
    // one allocation holding the matmul weights converted at load (prepare_weights): VNNI build:
    // repacked (QuantizedTensor.p); generic kernels on a GGUF file: q / s arrays; else NULL
    uint8_t *packed;
    size_t packed_size;
    float *norms; // GGUF: the norm weights, copied in the .bin layout (NULL: .bin, in the mapping)
} TransformerWeights;

// Load errors: the loader's functions report with load_fail / gguf_fail, which format the message
// and longjmp back to rq_model_load (it frees what was allocated and returns NULL with the message)
#if defined(__GNUC__) || defined(__clang__)
    #define RQ_NORETURN __attribute__((noreturn))
#else
    #define RQ_NORETURN
#endif
typedef struct { jmp_buf jb; char *msg; size_t len; } LoadErr;
RQ_NORETURN void load_fail(LoadErr *e, const char *fmt, ...);

// GGUF files (llama.cpp's format, versions 2 and 3), read directly: hyperparameters and tokenizer
// from the metadata, Q8_0 / Q4_K / Q5_K / Q6_K / F32 tensors from the mapped file (layout: gguf.c)
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
    LoadErr *err; // where errors go (gguf_fail)
} Gguf;

typedef struct {
    char **vocab;           // token bytes, NUL-terminated (the token for byte 0x00 reads as "")
    int *vocab_len;         // byte length of each token
    float *merge_scores;
    int vocab_size;
    unsigned int max_token_length;
    unsigned int bos_token_id;
    unsigned int eos_token_id;
    char *vocab_data;       // one buffer holding all the vocab strings
    int32_t *hash_ids;      // open-addressing hash map: token bytes -> id (-1 = empty slot);
    uint32_t *hash_tags;    // regular tokens only (no added tokens), so BPE never produces those
    uint32_t hash_mask;
    int byte_token[256];    // token id of each single byte
    int *added;             // added tokens (special or not), longest first
    int n_added;
    unsigned char added_first[256]; // 1 if some added token starts with this byte
    int marks35;            // pre-tokenizer: Qwen3.5's regex (tokenizer.ggml.pre qwen35: \p{M} joins letters), else Qwen2's
} Tokenizer;

#define TMPL_MAX 1024 // bytes of a chat template (the .bin's .template* files are read up to this)

struct RqModel {
    Config config;              // seq_len: the ctx_len cap applied
    TransformerWeights weights; // (has_weights)
    Tokenizer tok;              // (has_tok)
    Gguf gguf;                  // gguf.map != NULL: a GGUF file (the mapping below)
    void *data;                 // the file mapping, or NULL
    size_t file_size;
    int has_weights, has_tok, is_gguf;
    int gs_counted;             // counted in model.c's gs_models (one group size per process)
    char warning[256];          // a load-time warning for the caller to show ("" if none)
    int kq;                     // K-quant weights (Q4_K / Q5_K / Q6_K): the activation sums xf are needed
    // derived shape parameters of the forward passes (checked at load)
    int kv_dim, all_heads_dim, kv_mul;
    int xs;                     // row stride of the batched xb / xq rows: max(dim, all_heads_dim)
    int ff_rows;                // hidden rows per decode ffn work item (w1 + w3 + swiglu + quantize)
    int att_rows;               // query rows per prefill attention work item
    float att_scale;            // 1 / sqrt(head_dim)
    float rope_theta;           // RoPE base (GGUF qwen3.rope.freq_base: Qwen3 1e6, the 2507 models 5e6; .bin 1e6)
    float rope_log_theta;       // (float)ln(rope_theta), the form rope_row uses
    int rope_dim;               // rotated dims per head (Qwen3: head_dim; Qwen3.5: 64 of 256, the rest passes through)
    Hybrid hy;                  // Qwen3.5's hybrid layers (hy.on), else all 0
    int n_attn;                 // attention layers: the KV cache's layers (Qwen3: n_layers)
    int *kv_layer;              // [n_layers]: the layer's attention layer index (KV cache, attention weights), or -1
    int *lin_layer;             // [n_layers]: the layer's linear-attention layer index (state, weights), or -1
    int qh;                     // rows of wq per head: head_dim, Qwen3.5 2 x head_dim ([q | gate])
    size_t ssm_layer;           // floats of one linear layer's state in a sequence (conv inputs + S per head)
    int ctx_max;                // the model's context length (metadata) before the ctx_len cap
    int chat;                   // chat template variant (RQ_CHAT_*): from the GGUF's tokenizer.chat_template
    // chat templates per thinking mode (0 off, 1 on): the user turn + assistant header with one %s
    // (.template / .template.with-thinking), the system block with one %s (the start of
    // .template.with-system[-and-thinking]); tmpl_missing: the .bin side file that could not be read
    char user_tmpl[2][TMPL_MAX], sys_tmpl[2][TMPL_MAX];
    char tmpl_missing[2][TMPL_MAX + 64];
};

// gguf.c
RQ_NORETURN void gguf_fail(const Gguf *g, const char *fmt, const char *arg);
uint64_t gguf_str(const Gguf *g, uint64_t *pos, const char **s, const char *what);
void gguf_parse(Gguf *g);
int gguf_get_int(const Gguf *g, const char *key, int64_t *v);
const char *gguf_get_str(const Gguf *g, const char *key, uint64_t *len);
const GgufKV *gguf_get_array(const Gguf *g, const char *key, uint32_t type);
const GgufTensor *gguf_tensor(const Gguf *g, const char *name);
const uint8_t *gguf_weight(const Gguf *g, const char *name, uint32_t type, uint64_t ne0, uint64_t ne1);
void gguf_config(const Gguf *g, Config *c, float *rope_theta, int *rope_dim, Hybrid *hy);
int gguf_chat_variant(const Gguf *g); // RQ_CHAT_* from tokenizer.chat_template (hybrid if absent / unknown)
// model.c
const char *chat_gen_suffix(const RqModel *m, int thinking); // after the assistant header (the variant's)
void *alloc_big(size_t n);
void free_big(void *p, size_t n);
// tokenizer.c (errors: load_fail / gguf_fail)
void build_tokenizer(Tokenizer *t, const char *checkpoint_path, int vocab_size, LoadErr *e);
void build_tokenizer_gguf(Tokenizer *t, const Gguf *g, int vocab_size);
void free_tokenizer(Tokenizer *t);
// tokens[] capacity encode() needs for a text of n bytes: one token per byte of the NFC-normalized
// text, which is at most 3x as long (e.g. U+1D160: 4 bytes -> 3 code points, 12 bytes)
#define ENCODE_MAX_TOKENS(n) (3 * (size_t)(n) + 3)
int encode(const Tokenizer *t, const char *text, size_t len, int *tokens, int *n_tokens); // RQ_OK / RQ_ERR_NOMEM
// forward.c
void embed_row(float *x, const QuantizedTensor *t, int token, int dim);
void *ctx_sampler_scratch(RqCtx *c); // vocab_size * 8 bytes
void *ctx_sampler_scratch_threads(RqCtx *c, int nt); // nt x vocab_size * 8 bytes (rq_sample_batch), NULL: no memory
int ctx_vocab(const RqCtx *c);
// sampler.c
float random_f32(unsigned long long *state);
int forward_model_init(RqModel *m); // the derived shape parameters; 0 if the shape is not supported

// ------------------------------------------------------------------------------------- a sequence

struct RqSeq {
    const RqModel *m;
    int cap;            // positions
    int kv_len;         // cap rounded up to whole attention blocks (ATT_P positions: the cache layout)
    int att_nb;         // attention blocks of ATT_P positions in kv_len
    int kvt;            // cache element type: KV_F32, KV_BF16, KV_F16
    size_t kv_esz;      // bytes per cache element: 4 or 2
    size_t kv_size;     // elements per cache
    // kv cache: per layer kv_len / ATT_T tiles of ATT_T positions x n_kv_heads, layout in
    // kernels_avx512.h (attention)
    void *key_cache;    // (layer, kv_len, kv_dim) elements: float, or bf16 / f16 (uint16_t) by kvt
    void *value_cache;  // (layer, kv_len, kv_dim)
    int len;            // positions in the cache
    int *tokens;        // [cap]: the token at each position
    // Qwen3.5: the recurrent state of the linear-attention layers after position len - 1 (it cannot be rewound),
    // per linear layer RqModel.ssm_layer floats: the last conv - 1 inputs of the conv (per channel group (conv - 1, n) rows, as ssm_conv), then
    // per value head S (dk, dv); and a few checkpoints of it (position + copy), for truncate / copy to an
    // earlier length (forward.c, rq_seq_checkpoint_at)
    float *ssm;
    size_t ssm_bytes;
    struct { int pos; float *st; } ck[RQ_SEQ_CK];
    int ck_at;          // a checkpoint to take when the sequence reaches this length (0: none)
    int ck_slot;        // the checkpoint slot being written by the pass in progress (-1: none)
    unsigned long long rng; // sampling (xorshift)
};

#endif // RUNQ_INTERNAL_H
