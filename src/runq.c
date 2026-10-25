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

// ----------------------------------------------------------------------------
// Globals
int GS = 0; // group size global for quantization of the weights

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
    int group_size; // quantization group size (export.py uses 64)
} Config;

typedef struct {
    int8_t *q;    // quantized values
    float *s; // scaling factors
    uint8_t *p;   // weights only: the same matrix repacked for the VNNI kernels (Q8P layout,
                  // see kernels_avx512.h), or NULL (generic kernels on q / s)
} QuantizedTensor;

typedef struct {
    // token embedding table
    QuantizedTensor *q_tokens; // (vocab_size, dim)
    float *token_embedding_table; // same, but dequantized
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
    // VNNI build: one allocation holding all repacked matmul weights (QuantizedTensor.p)
    uint8_t *packed;
    size_t packed_size;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    QuantizedTensor xq; // quantized x (dim,)
    QuantizedTensor hq; // quantized hb (hidden_dim,)
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
    int32_t *xq_corr;   // VNNI GEMV activation corrections of xq / hq (q8p_gemv_corr), per 64-group
    int32_t *hq_corr;
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
    float *att_b;   // (att_threads, att_rows * kv_mul * kv_len): attention scratch per OpenMP thread
    int att_rows;   // query rows per prefill attention work item
    int att_threads;
    float *rope_b;  // (nb_max, head_dim): RoPE cos / sin of each row's position (rope_row)
    float *rope;    // (head_dim,): the same for forward()'s position
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    float *data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
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
#ifdef _OPENMP
        s->att_threads = omp_get_max_threads();
#else
        s->att_threads = 1;
#endif
        s->att_b = calloc((size_t)s->att_threads * s->att_rows * kv_mul * s->kv_len, sizeof(float));
        s->rope_b = calloc(nb * p->head_dim, sizeof(float));
        if (!s->rope_b || !s->k_b || !s->v_b || !s->x_b || !s->xb_b || !s->q_b || !s->hb_b || !s->hb2_b || !s->xq_b.q || !s->xq_b.s ||
            !s->hq_b.q || !s->hq_b.s || !s->att_b) {
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
    s->xq_corr = calloc((all_heads_dim > p->dim ? all_heads_dim : p->dim) / 64 + 1, sizeof(int32_t));
    s->hq_corr = calloc(p->hidden_dim / 64 + 1, sizeof(int32_t));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc((size_t)p->n_heads * s->kv_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->rope = calloc(p->head_dim, sizeof(float));
    s->key_cache = alloc_big(s->kv_size * sizeof(float));
    s->value_cache = alloc_big(s->kv_size * sizeof(float));

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->hb || !s->hb2 || !s->q || !s->xq_corr || !s->hq_corr || !s->k || !s->v || !s->att || !s->logits || !s->rope || !s->key_cache || !s->value_cache) {
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
    free(s->rope_b);
    free(s->rope);
}

// ----------------------------------------------------------------------------
// Quantization functions

void dequantize(QuantizedTensor *qx, float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = qx->q[i] * qx->s[i / GS];
}

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
    // dequantize token embedding table
    w->token_embedding_table = malloc(p->vocab_size * p->dim * sizeof(float));
    dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);

    w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * p->head_dim));
    w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * p->head_dim));
    w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * p->head_dim));
    w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * p->head_dim) * p->dim);

    w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
    w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);

    w->wcls = p->shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);
}

#ifdef RUNQ_VNNI

static void drop_mapped(const void *begin, const void *end) {
    // release the pages of [begin, end) of the read-only checkpoint mapping (MAP_PRIVATE, never
    // written, so a page touched again later is simply re-read from the page cache)
#if !defined _WIN32 && defined MADV_DONTNEED
    uintptr_t pg = (uintptr_t)sysconf(_SC_PAGESIZE);
    uintptr_t a = (uintptr_t)begin & ~(pg - 1), b = ((uintptr_t)end + pg - 1) & ~(pg - 1);
    madvise((void *)a, b - a, MADV_DONTNEED);
#else
    (void)begin; (void)end;
#endif
}

void pack_weights(TransformerWeights *w, Config *p) {
    // VNNI build: repack every matmul weight (7 per layer + the classifier) into the Q8P layout
    // of kernels_avx512.h, in one allocation (the same size as the mapped Q8_0 data). The mapped
    // pages of each tensor are dropped right after packing it, so the resident size stays about
    // the same as without repacking (the token embedding is already dequantized at this point)
    int L = p->n_layers, dim = p->dim, hid = p->hidden_dim;
    int qd = p->n_heads * p->head_dim, kvd = p->n_kv_heads * p->head_dim;
    struct { QuantizedTensor *t; int n, d, k; } m[] = {
        { w->wq, L, qd, dim }, { w->wk, L, kvd, dim }, { w->wv, L, kvd, dim }, { w->wo, L, dim, qd },
        { w->w1, L, hid, dim }, { w->w2, L, dim, hid }, { w->w3, L, hid, dim }, { w->wcls, 1, p->vocab_size, dim } };
    int nm = sizeof(m) / sizeof(m[0]);
    size_t total = 0;
    for (int i = 0; i < nm; i++) {
        if (GS != Q8P_GS || !q8p_ok(m[i].d, m[i].k)) {
            fprintf(stderr, "warning: weights not packable for the VNNI kernels (GS %d, %d x %d), using the generic ones\n", GS, m[i].d, m[i].k);
            return;
        }
        total += m[i].n * q8p_size(m[i].d, m[i].k);
    }
    w->packed = alloc_big(total);
    if (!w->packed) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    w->packed_size = total;
    size_t off = 0;
    for (int i = 0; i < nm; i++)
        for (int l = 0; l < m[i].n; l++) {
            QuantizedTensor *t = m[i].t + l;
            t->p = w->packed + off;
            q8p_pack(t->p, t->q, t->s, m[i].d, m[i].k);
            off += q8p_size(m[i].d, m[i].k);
            drop_mapped(t->q, t->s + (size_t)m[i].d * m[i].k / GS);
        }
}
#endif

void read_checkpoint(char *checkpoint, Config *config, TransformerWeights* weights, float** data, ssize_t* file_size, int ctx_length) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open checkpoint %s\n", checkpoint); exit(EXIT_FAILURE); }

    #if defined _WIN32
        _fseeki64(file, 0, SEEK_END); // move file pointer to end of file
        *file_size = _ftelli64(file); // get the file size, in bytes
    #else
        fseek(file, 0, SEEK_END); // move file pointer to end of file
        *file_size = ftell(file); // get the file size, in bytes
    #endif

    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, fileno(file), 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    fclose(file);

    // checkpoint format is 256-byte header, and then the model weights

    memcpy(config, *data, sizeof(Config));
    if (config->magic_number != 0x616a6331) { fprintf(stderr, "File %s is not a qwen3.c checkpoint\n", checkpoint); exit(EXIT_FAILURE); }
    if (config->version != 1) { fprintf(stderr, "Checkpoint %s is version %d, need version 1\n", checkpoint, config->version); exit(EXIT_FAILURE); }

    if (ctx_length != 0 && ctx_length <= config->seq_len)
        config->seq_len = ctx_length;

    GS = config->group_size; // set as global, as it will be used in many places

    void *weights_ptr = ((char *)*data) + 256; // skip the header (256 bytes)
    memory_map_weights(weights, config, weights_ptr);
#ifdef RUNQ_VNNI
    pack_weights(weights, config);
#endif
}

void build_transformer(Transformer *t, char *checkpoint_path, int ctx_length, int nb_max) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->data, &t->file_size, ctx_length);
    // allocate the RunState buffers (batched prefill buffers for nb_max rows if nb_max > 1)
    malloc_run_state(&t->state, &t->config, nb_max);
}

void free_transformer(Transformer *t) {
    // free QuantizedTensors
    free(t->weights.q_tokens);
    free(t->weights.token_embedding_table);
    free(t->weights.wq);
    free(t->weights.wk);
    free(t->weights.wv);
    free(t->weights.wo);
    free(t->weights.w1);
    free(t->weights.w2);
    free(t->weights.w3);
    if(t->weights.wcls != t->weights.q_tokens) free(t->weights.wcls);
#ifdef RUNQ_VNNI
    free_big(t->weights.packed, t->weights.packed_size);
#endif
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
// x[i0 .. i0+n) (one per 64-group; nothing to do for the generic kernel); matvec_rows: rows i0..i1-1
static inline void matvec_prep(int32_t *corr, QuantizedTensor *x, int i0, int n) {
#ifdef RUNQ_VNNI
    q8p_gemv_corr(corr + i0 / 64, x->q + i0, n);
#else
    (void)corr; (void)x; (void)i0; (void)n;
#endif
}

static inline void matvec_rows(float *xout, QuantizedTensor *x, const int32_t *corr, QuantizedTensor *w, int n, int i0, int i1) {
    // xout[i] = W (d,n) row i . x (n,) for i0 <= i < i1 (multiples of 16, as the VNNI blocks)
#ifdef RUNQ_VNNI
    if (w->p) { q8p_gemv_rows(xout, x->q, x->s, corr, w->p, n, i0 / 16, i1 / 16); return; }
#else
    (void)corr;
#endif
    for (int i = i0; i < i1; i++) {
        size_t in = (size_t)i * n;
        xout[i] = dot_q8(w->q + in, w->s + in / GS, x->q, x->s, n);
    }
}

void matmul(float *xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,), its own parallel region (forward_batch's last-row classifier)
    if (n % GS != 0) { fprintf(stderr, "matmul: n=%d is not a multiple of GS=%d\n", n, GS); exit(EXIT_FAILURE); }
    int32_t *corr = NULL;
#ifdef RUNQ_VNNI
    corr = q8p_corr_buf(n / 64 + 1);
#endif
    matvec_prep(corr, x, 0, n);
    #pragma omp parallel for schedule(static)
    for (int i0 = 0; i0 < d; i0 += 16)
        matvec_rows(xout, x, corr, w, n, i0, i0 + 16 < d ? i0 + 16 : d);
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

typedef struct { float *out; int ldo; QuantizedTensor *w; int d; } MatJob; // one GEMM of matmul_batch_n

NO_FP_CONTRACT
void matmul_batch_n(int nj, const MatJob *job, QuantizedTensor *x, int ldx, int nb, int k) {
    // GEMMs on the same input rows, for each job: out[b * ldo + i] = W[i, :] . x[b, :]  for b < nb, i < d
    //   W (d, k) Q8_0, row-major (as in matmul); x: nb quantized rows of k values, row b at
    //   x->q + b * ldx with its k/GS scales at x->s + b * ldx / GS (ldx a multiple of GS).
    // All jobs in one parallel loop (q / k / v, w1 / w3: one barrier, better balance).
    // Each output is computed exactly as dot_q8() in matmul(), so the batched path matches the
    // sequential one bit for bit. Work items: blocks of MB_ROWS output rows; each weight row is
    // loaded from memory once per call and reused for all nb activation rows from L1.
    // The VNNI build uses q8p_gemm_n (kernels_avx512.h) on the repacked weights instead, with the
    // same per-output arithmetic.
    if (k % GS != 0 || ldx % GS != 0) { fprintf(stderr, "matmul_batch: k=%d / ldx=%d not multiples of GS=%d\n", k, ldx, GS); exit(EXIT_FAILURE); }
    if (nj > 8) { fprintf(stderr, "matmul_batch_n: too many jobs\n"); exit(EXIT_FAILURE); }
#ifdef RUNQ_VNNI
    int packed = 1;
    Q8PJob qj[8];
    for (int j = 0; j < nj; j++) {
        packed &= job[j].w->p != NULL;
        qj[j] = (Q8PJob) { job[j].out, job[j].ldo, job[j].w->p, job[j].d };
    }
    if (packed) { q8p_gemm_n(nj, qj, x->q, x->s, ldx, nb, k); return; }
#endif
    int b0[9], nblk = 0;
    for (int j = 0; j < nj; j++) {
        b0[j] = nblk;
        nblk += (job[j].d + MB_ROWS - 1) / MB_ROWS;
    }
    b0[nj] = nblk;
    #pragma omp parallel for schedule(static)
    for (int blk = 0; blk < nblk; blk++) {
        int j = 0;
        while (blk >= b0[j + 1]) j++;
        QuantizedTensor *w = job[j].w;
        int d = job[j].d, i0 = (blk - b0[j]) * MB_ROWS, i1 = i0 + MB_ROWS < d ? i0 + MB_ROWS : d;
        for (int b = 0; b < nb; b++) {
            const int8_t *xq = x->q + (size_t)b * ldx;
            const float *xsc = x->s + (size_t)b * ldx / GS;
            float *o = job[j].out + (size_t)b * job[j].ldo;
            if (i1 - i0 == 4) {
                dot_q8_x4(o + i0, w->q + (size_t)i0 * k, w->s + (size_t)i0 * k / GS, k, xq, xsc);
                continue;
            }
            for (int i = i0; i < i1; i++) {
                size_t in = (size_t)i * k;
                o[i] = dot_q8(w->q + in, w->s + in / GS, xq, xsc, k);
            }
        }
    }
}

void matmul_batch(float *out, int ldo, QuantizedTensor *x, int ldx, int nb, QuantizedTensor *w, int k, int d) {
    MatJob j = { out, ldo, w, d };
    matmul_batch_n(1, &j, x, ldx, nb, k);
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

ATT_EXACT static void attention_vec(const float *q, float *out, int n, const float *kt, const float *vt,
                                    size_t ts, int hd, float scale, float *S) {
    // one query vector over positions 0..n-1 of one kv head: kt / vt = its tile 0 in the layer's
    // K / V cache, ts = floats from one tile of this head to the next; S: n (rounded up to 16) floats
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
    for (int i = 0; i < hd; i++)
        out[i] = 0.0f;
    for (int t = 0; t < n; t++) {
        const float *v = vt + (size_t)(t / ATT_T) * ts + (size_t)(t % ATT_T) * hd;
        float e = S[t];
        for (int i = 0; i < hd; i++)
            out[i] = ATT_FMA(e, v[i], out[i]);
    }
    for (int i = 0; i < hd; i++)
        out[i] = out[i] * inv;
}

static void attention_kv(int nv, const float *const *q, float *const *out, const int *np, const float *kt,
                         const float *vt, size_t ts, int hd, float scale, float *S) {
    // nv query vectors q[r] sharing one kv head, vector r over positions 0..np[r]-1, into out[r];
    // S: scratch of nv * kv_len floats. Each vector's result does not depend on the others
#ifdef RUNQ_ATT512
    if (att512_ok(hd)) { att512_kv(nv, q, out, np, kt, vt, ts, hd, scale, S); return; }
#endif
    for (int r = 0; r < nv; r++)
        attention_vec(q[r], out[r], np[r], kt, vt, ts, hd, scale, S);
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

static void kv_populate(RunState *s, Config *p, int pos0, int n) {
    // Pre-fault the KV cache tiles of positions pos0..pos0+n-1 of every layer (anonymous memory,
    // untouched pages are not mapped yet), in parallel and one madvise per range instead of one page fault
    // per 4 KB. Otherwise these faults happen inside the K/V GEMMs, serialized in the kernel:
    // ~55 ms of a 451-token prefill (8 threads), vs ~7 ms here. Linux >= 5.14; errors ignored
#if !defined _WIN32 && defined MADV_POPULATE_WRITE
    int kv_dim = p->n_kv_heads * p->head_dim;
    uintptr_t pg = (uintptr_t)sysconf(_SC_PAGESIZE);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int l = 0; l < p->n_layers; l++)
        for (int kv = 0; kv < 2; kv++) {
            // positions pos0.. of all heads: the tiles pos0 / ATT_T .. (pos0 + n - 1) / ATT_T
            float *c = (kv ? s->value_cache : s->key_cache) + l * (uint64_t)s->kv_len * kv_dim;
            uintptr_t a = (uintptr_t)(c + (size_t)(pos0 / ATT_T) * ATT_T * kv_dim) & ~(pg - 1);
            uintptr_t e = (uintptr_t)(c + (size_t)((pos0 + n - 1) / ATT_T + 1) * ATT_T * kv_dim);
            madvise((void *)a, e - a, MADV_POPULATE_WRITE);
        }
#else
    (void)s; (void)p; (void)pos0; (void)n;
#endif
}

static inline QuantizedTensor qsub(QuantizedTensor *t, int i0) {
    // the quantized values from index i0 on (a multiple of GS)
    return (QuantizedTensor) { .q = t->q + i0, .s = t->s + i0 / GS, .p = NULL };
}

float *forward(Transformer *transformer, int token, int pos, int compute_logits) {
    // compute_logits = 0 skips the final rmsnorm + classifier (prompt tokens whose
    // logits are discarded) and returns NULL; the KV cache is filled either way.
    // One parallel region for the whole token (was: one per matmul and for the attention, ~230
    // fork/joins per token): per layer 7 barriers, the work split into items:
    //   [single] residual + rmsnorm + quantize -> [for] q, k, v, one item per head, with its
    //   QK-norm + RoPE and the K/V cache store -> [for] attention per kv head + quantize ->
    //   [for] wo -> [single] residual + rmsnorm + quantize -> [for] w1 + w3 + SwiGLU + quantize per
    //   ff_rows hidden rows -> [for] w2
    // Every output is computed whole by one thread, by the same functions as before (same results)
    Config *p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    int dim = p->dim, hd = p->head_dim, hidden = p->hidden_dim, nh = p->n_heads, nkv = p->n_kv_heads;
    int kv_dim = nkv * hd;
    int kv_mul = nh / nkv; // integer multiplier of the kv sharing in multiquery
    int all_heads_dim = nh * hd, ff = s->ff_rows;
    size_t ts = (size_t)nkv * ATT_T * hd;

    // copy the token embedding into s->x
    memcpy(s->x, w->token_embedding_table + token * dim, dim * sizeof(float));
    rope_row(s->rope, hd, pos);
    if (pos % ATT_T == 0)
        kv_populate(s, p, pos, 1); // first position of a KV cache tile: pre-fault the tile (all layers)

    #pragma omp parallel
    {
    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {
        uint64_t loff = l * (uint64_t)s->kv_len * kv_dim; // kv cache layer offset for convenience
        float *kl = s->key_cache + loff, *vl = s->value_cache + loff;

        // (residual connection of the previous layer's ffn,) attention rmsnorm, quantize
        #pragma omp single
        {
            if (l > 0)
                for (int i = 0; i < dim; i++)
                    s->x[i] += s->xb[i];
            rmsnorm(s->xb, s->x, w->rms_att_weight + l * dim, dim);
            quantize(&s->xq, s->xb, dim);
            matvec_prep(s->xq_corr, &s->xq, 0, dim);
        }

        // qkv matmuls for this position, one head per item: Q-RMSNorm + rotate each query head,
        // the same for each key head, then save key, value at this time step (pos) to our kv cache
        #pragma omp for schedule(static)
        for (int it = 0; it < nh + 2 * nkv; it++) {
            if (it < nh) {
                matvec_rows(s->q, &s->xq, s->xq_corr, w->wq + l, dim, it * hd, (it + 1) * hd);
                qk_norm_rope(s->q + it * hd, 1, w->q_norm_weights + l * hd, hd, s->rope);
            } else if (it < nh + nkv) {
                int h = it - nh;
                matvec_rows(s->k, &s->xq, s->xq_corr, w->wk + l, dim, h * hd, (h + 1) * hd);
                qk_norm_rope(s->k + h * hd, 1, w->k_norm_weights + l * hd, hd, s->rope);
                kv_store(kl, NULL, pos, h, nkv, hd, s->k + h * hd, NULL);
            } else {
                int h = it - nh - nkv;
                matvec_rows(s->v, &s->xq, s->xq_corr, w->wv + l, dim, h * hd, (h + 1) * hd);
                kv_store(NULL, vl, pos, h, nkv, hd, NULL, s->v + h * hd);
            }
        }

        // multihead attention: per kv head, its kv_mul query heads together; then quantize
        // their outputs (kv_mul * hd values, whole quantization groups) for wo
        #pragma omp for schedule(static)
        for (int g = 0; g < nkv; g++) {
            const float *qv[ATT_NV_MAX];
            float *ov[ATT_NV_MAX];
            int npv[ATT_NV_MAX];
            for (int r = 0; r < kv_mul; r++) {
                qv[r] = s->q + (g * kv_mul + r) * hd;
                ov[r] = s->xb + (g * kv_mul + r) * hd;
                npv[r] = pos + 1;
            }
            size_t o = kv_tile(0, g, nkv, hd);
            attention_kv(kv_mul, qv, ov, npv, kl + o, vl + o, ts, hd, s->att_scale, s->att + (size_t)g * kv_mul * s->kv_len);
            int i0 = g * kv_mul * hd;
            QuantizedTensor xg = qsub(&s->xq, i0);
            quantize(&xg, s->xb + i0, kv_mul * hd);
            matvec_prep(s->xq_corr, &s->xq, i0, kv_mul * hd);
        }

        // final matmul to get the output of the attention
        #pragma omp for schedule(static)
        for (int i0 = 0; i0 < dim; i0 += 16)
            matvec_rows(s->xb, &s->xq, s->xq_corr, w->wo + l, all_heads_dim, i0, i0 + 16);

        // residual connection back into s->x, ffn rmsnorm, quantize
        #pragma omp single
        {
            for (int i = 0; i < dim; i++)
                s->x[i] += s->xb[i];
            rmsnorm(s->xb, s->x, w->rms_ffn_weight + l * dim, dim);
            quantize(&s->xq, s->xb, dim);
            matvec_prep(s->xq_corr, &s->xq, 0, dim);
        }

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x)): per item,
        // ff rows of self.w1(x) and self.w3(x), the SwiGLU non-linearity and their quantization
        #pragma omp for schedule(static)
        for (int i0 = 0; i0 < hidden; i0 += ff) {
            matvec_rows(s->hb, &s->xq, s->xq_corr, w->w1 + l, dim, i0, i0 + ff);
            matvec_rows(s->hb2, &s->xq, s->xq_corr, w->w3 + l, dim, i0, i0 + ff);
            swiglu(s->hb + i0, s->hb2 + i0, ff);
            QuantizedTensor hg = qsub(&s->hq, i0);
            quantize(&hg, s->hb + i0, ff);
            matvec_prep(s->hq_corr, &s->hq, i0, ff);
        }

        // final matmul to get the output of the ffn (its residual: next layer, or below)
        #pragma omp for schedule(static)
        for (int i0 = 0; i0 < dim; i0 += 16)
            matvec_rows(s->xb, &s->hq, s->hq_corr, w->w2 + l, hidden, i0, i0 + 16);
    }

    if (compute_logits) {
        // residual connection, final rmsnorm, classifier into logits
        #pragma omp single
        {
            for (int i = 0; i < dim; i++)
                s->x[i] += s->xb[i];
            rmsnorm(s->x, s->x, w->rms_final_weight, dim);
            quantize(&s->xq, s->x, dim);
            matvec_prep(s->xq_corr, &s->xq, 0, dim);
        }
        #pragma omp for schedule(static)
        for (int i0 = 0; i0 < p->vocab_size; i0 += 16)
            matvec_rows(s->logits, &s->xq, s->xq_corr, w->wcls, dim, i0, i0 + 16 < p->vocab_size ? i0 + 16 : p->vocab_size);
    }
    } // omp parallel
    return compute_logits ? s->logits : NULL;
}

static inline QuantizedTensor qrow(QuantizedTensor *t, int b, int ld) {
    // row b of a batch of quantized rows with row stride ld (a multiple of GS)
    return (QuantizedTensor) { .q = t->q + (size_t)b * ld, .s = t->s + (size_t)b * ld / GS, .p = NULL };
}

enum { LOGITS_NONE = 0, LOGITS_LAST = 1, LOGITS_ALL = 2 };

float *forward_batch(Transformer *transformer, int *tokens, int n, int pos0, int logits_mode, float *logits_out) {
    // Batched prefill: tokens[0..n-1] at positions pos0..pos0+n-1 (1 <= n <= state.nb_max),
    // the same computation as n calls of forward() but with every weight matrix applied to all
    // n rows at once (matmul_batch), and bit-identical results. Fills the KV cache.
    //   LOGITS_NONE: returns NULL
    //   LOGITS_LAST: logits of the last row in state.logits (returned)
    //   LOGITS_ALL:  logits of all rows into logits_out (n, vocab_size) (returned)
    // Note: in LOGITS_LAST mode state.x / state.xq are clobbered (as by forward()).
    Config *p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    int dim = p->dim, hd = p->head_dim, hidden = p->hidden_dim, xs = s->xs;
    int kv_dim = p->n_kv_heads * hd;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int all_heads_dim = p->n_heads * hd;
    if (n < 1 || n > s->nb_max || pos0 < 0 || pos0 + n > p->seq_len) {
        fprintf(stderr, "forward_batch: bad chunk n=%d pos0=%d (nb_max %d, seq_len %d)\n", n, pos0, s->nb_max, p->seq_len);
        exit(EXIT_FAILURE);
    }
    PROF_START();

    #pragma omp parallel for
    for (int b = 0; b < n; b++) {
        memcpy(s->x_b + (size_t)b * dim, w->token_embedding_table + (size_t)tokens[b] * dim, dim * sizeof(float));
        rope_row(s->rope_b + (size_t)b * hd, hd, pos0 + b);
    }
    kv_populate(s, p, pos0, n);

    for (int l = 0; l < p->n_layers; l++) {
        uint64_t loff = l * (uint64_t)s->kv_len * kv_dim;
        float *kl = s->key_cache + loff, *vl = s->value_cache + loff;

        // (residual of the previous layer's ffn,) attention rmsnorm + quantize, per row
        #pragma omp parallel for
        for (int b = 0; b < n; b++) {
            float *x = s->x_b + (size_t)b * dim, *xb = s->xb_b + (size_t)b * xs;
            QuantizedTensor xq = qrow(&s->xq_b, b, xs);
            if (l > 0)
                for (int i = 0; i < dim; i++)
                    x[i] += xb[i];
            rmsnorm(xb, x, w->rms_att_weight + l * dim, dim);
            quantize(&xq, xb, dim);
        }
        PROF(PF_NORM);

        // qkv GEMMs (one parallel loop)
        MatJob qkv[3] = { { s->q_b, all_heads_dim, w->wq + l, all_heads_dim }, { s->k_b, kv_dim, w->wk + l, kv_dim },
                          { s->v_b, kv_dim, w->wv + l, kv_dim } };
        matmul_batch_n(3, qkv, &s->xq_b, xs, n, dim);
        PROF(PF_QKV);

        // QK-norm + RoPE, each row at its own position; then the chunk's K / V rows into the
        // cache, one (tile, kv head) per work item
        int tb0 = pos0 / ATT_T, ntb = (pos0 + n - 1) / ATT_T - tb0 + 1;
        #pragma omp parallel
        {
            #pragma omp for
            for (int b = 0; b < n; b++) {
                const float *cs = s->rope_b + (size_t)b * hd;
                qk_norm_rope(s->q_b + (size_t)b * all_heads_dim, p->n_heads, w->q_norm_weights + l * hd, hd, cs);
                qk_norm_rope(s->k_b + (size_t)b * kv_dim, p->n_kv_heads, w->k_norm_weights + l * hd, hd, cs);
            }
            #pragma omp for
            for (int it = 0; it < ntb * p->n_kv_heads; it++) {
                int tb = tb0 + it / p->n_kv_heads, h = it % p->n_kv_heads;
                int t1 = (tb + 1) * ATT_T < pos0 + n ? (tb + 1) * ATT_T : pos0 + n;
                for (int t = tb * ATT_T > pos0 ? tb * ATT_T : pos0; t < t1; t++)
                    kv_store(kl, vl, t, h, p->n_kv_heads, hd, s->k_b + (size_t)(t - pos0) * kv_dim + h * hd,
                             s->v_b + (size_t)(t - pos0) * kv_dim + h * hd);
            }
        }
        PROF(PF_ROPE);

        // causal attention: row b (position pos0 + b) attends to positions 0..pos0+b, whose K/V
        // are all in the cache now. Work item = (block of att_rows rows, kv head): its rows x
        // kv_mul query heads in one attention_kv call (K/V read once for all of them). Dynamic
        // schedule, last (longest) row blocks first, for the load balance
        int ar = s->att_rows, nrb = (n + ar - 1) / ar;
        size_t ts = (size_t)p->n_kv_heads * ATT_T * hd;
        #pragma omp parallel
        {
        #pragma omp for schedule(dynamic, 1)
        for (int it = 0; it < nrb * p->n_kv_heads; it++) {
            int rb = nrb - 1 - it / p->n_kv_heads, g = it % p->n_kv_heads;
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
#ifdef _OPENMP
            float *att = s->att_b + (size_t)omp_get_thread_num() * ar * kv_mul * s->kv_len;
#else
            float *att = s->att_b;
#endif
            size_t o = kv_tile(0, g, p->n_kv_heads, hd);
            attention_kv(nv, qv, ov, npv, kl + o, vl + o, ts, hd, s->att_scale, att);
        }
        // (then quantize the attention output rows for wo, in the same parallel region)
        #pragma omp for
        for (int b = 0; b < n; b++) {
            QuantizedTensor xq = qrow(&s->xq_b, b, xs);
            quantize(&xq, s->xb_b + (size_t)b * xs, all_heads_dim);
        }
        } // omp parallel
        PROF(PF_ATT);
        matmul_batch(s->xb_b, xs, &s->xq_b, xs, n, w->wo + l, all_heads_dim, dim);
        PROF(PF_WO);

        // residual, ffn rmsnorm, quantize
        #pragma omp parallel for
        for (int b = 0; b < n; b++) {
            float *x = s->x_b + (size_t)b * dim, *xb = s->xb_b + (size_t)b * xs;
            QuantizedTensor xq = qrow(&s->xq_b, b, xs);
            for (int i = 0; i < dim; i++)
                x[i] += xb[i];
            rmsnorm(xb, x, w->rms_ffn_weight + l * dim, dim);
            quantize(&xq, xb, dim);
        }
        PROF(PF_NORM);

        MatJob w13[2] = { { s->hb_b, hidden, w->w1 + l, hidden }, { s->hb2_b, hidden, w->w3 + l, hidden } };
        matmul_batch_n(2, w13, &s->xq_b, xs, n, dim);
        PROF(PF_FFN13);

        #pragma omp parallel for
        for (int b = 0; b < n; b++) {
            QuantizedTensor hq = qrow(&s->hq_b, b, hidden);
            swiglu(s->hb_b + (size_t)b * hidden, s->hb2_b + (size_t)b * hidden, hidden);
            quantize(&hq, s->hb_b + (size_t)b * hidden, hidden);
        }
        PROF(PF_SWIGLU);

        matmul_batch(s->xb_b, xs, &s->hq_b, hidden, n, w->w2 + l, hidden, dim);
        PROF(PF_FFN2);
        // (its residual add is done in the next layer's first loop, or below)
    }
    if (logits_mode == LOGITS_NONE)
        return NULL; // the residual stream x_b is not needed any more

    // residual connection of the last layer
    #pragma omp parallel for
    for (int b = 0; b < n; b++) {
        float *x = s->x_b + (size_t)b * dim, *xb = s->xb_b + (size_t)b * xs;
        for (int i = 0; i < dim; i++)
            x[i] += xb[i];
    }

    if (logits_mode == LOGITS_LAST) {
        // final rmsnorm + classifier for the last row only, exactly as in forward()
        rmsnorm(s->x, s->x_b + (size_t)(n - 1) * dim, w->rms_final_weight, dim);
        quantize(&s->xq, s->x, dim);
        matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
        PROF(PF_CLS);
        return s->logits;
    }
    if (logits_mode == LOGITS_ALL) {
        #pragma omp parallel for
        for (int b = 0; b < n; b++) {
            QuantizedTensor xq = qrow(&s->xq_b, b, xs);
            rmsnorm(s->xb_b + (size_t)b * xs, s->x_b + (size_t)b * dim, w->rms_final_weight, dim);
            quantize(&xq, s->xb_b + (size_t)b * xs, dim);
        }
        matmul_batch(logits_out, p->vocab_size, &s->xq_b, xs, n, w->wcls, dim, p->vocab_size);
        PROF(PF_CLS);
        return logits_out;
    }
    return NULL;
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
//   2. normalizer (NFC): NOT implemented, the input is assumed to be NFC already
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

    // classify: single bytes, merge results (score > -1e6) and added tokens (the rest, len > 1;
    // export.py gives every token that no merge produces the score -1e6)
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

    load_prompt_template(checkpoint_path, t->prompt_template, 0, enable_thinking);
    load_prompt_template(checkpoint_path, t->system_prompt_template, 1, enable_thinking);
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

static inline UChar uchar_at(const unsigned char *p, const unsigned char *end) {
    // decode one UTF-8 code point; an invalid byte counts as one "other" character
    UChar c = { p[0], 1, UCLS_OTHER };
    unsigned char b = p[0];
    if (b < 0x80) { c.cls = ucls(b); return c; }
    int n = b >= 0xf0 ? (b <= 0xf4 ? 4 : 0) : b >= 0xe0 ? 3 : b >= 0xc2 ? 2 : 0;
    if (n == 0 || end - p < n) { c.cp = 0xfffd; return c; }
    uint32_t cp = b & (0x7f >> n);
    for (int i = 1; i < n; i++) {
        if ((p[i] & 0xc0) != 0x80) { c.cp = 0xfffd; return c; }
        cp = (cp << 6) | (p[i] & 0x3f);
    }
    if ((n == 3 && (cp < 0x800 || (cp >= 0xd800 && cp <= 0xdfff))) || (n == 4 && (cp < 0x10000 || cp > 0x10ffff))) {
        c.cp = 0xfffd; return c;
    }
    c.cp = cp; c.len = n; c.cls = ucls(cp);
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

// --- byte-level BPE inside one pre-tokenizer chunk

typedef struct { float score; int pos, len, id; } BpeCand; // merge candidate: symbol at pos + next one

typedef struct {
    int *len, *tok, *prev; // per byte offset: symbol length (0 = not a symbol start), token id, previous symbol
    BpeCand *heap;
    int cap;
} BpeScratch;

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
    // text between added tokens: pre-tokenize, then BPE each chunk
    while (p < end) {
        const unsigned char *q = pretok_next(p, end);
        bpe_chunk(t, p, (int)(q - p), s, tokens, n_tokens);
        p = q;
    }
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
    // encode the string text (input) into an upper-bound preallocated tokens[] array:
    // at most one token per byte, so strlen(text) ints are enough
    const unsigned char *s = (const unsigned char *)text, *end = s + strlen(text);
    int n = (int)(end - s);
    BpeScratch sc;
    sc.cap = n > 0 ? n : 1;
    sc.len = malloc(3 * (size_t)sc.cap * sizeof(int));
    sc.tok = sc.len + sc.cap;
    sc.prev = sc.tok + sc.cap;
    sc.heap = malloc(3 * (size_t)sc.cap * sizeof(BpeCand)); // n-1 initial pairs + 2 per merge
    if (!sc.len || !sc.heap) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }

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
    int *prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
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
    // append the tokens of text (raw, no template); encode needs at most one token per byte
    iv_reserve(v, (size_t)v->n + strlen(text) + 3);
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
    int *tokens = (int *)malloc((strlen(text) + 3) * sizeof(int));
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
    int *tokens = (int *)malloc((strlen(text) + 3) * sizeof(int));
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

// ----------------------------------------------------------------------------
// CLI

void error_usage() {
    fprintf(stderr, "Usage:   runq <checkpoint> [options]\n");
    fprintf(stderr, "Example: runq Qwen3-4B.bin -r 1\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1], default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -c <int>    context window size, 0 (default) = max_seq_len; chat: holds the whole conversation,\n");
    fprintf(stderr, "              when full the oldest turns are dropped (the system prompt is kept)\n");
    fprintf(stderr, "  -m <string> mode: generate|chat|tokenize|logits, default: chat\n");
    fprintf(stderr, "              tokenize: print the token ids of -i (raw text, no template)\n");
    fprintf(stderr, "              logits: write the logits at every position of -i to -o (binary)\n");
    fprintf(stderr, "  -i <string> input prompt (chat: one turn, then exit; without -i, one turn per stdin line,\n");
    fprintf(stderr, "              the conversation is kept until an empty line or EOF)\n");
    fprintf(stderr, "  -o <string> output file for -m logits\n");
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
    if (strcmp(mode, "logits") == 0 && !out_path) { error_usage(); }
    if (prefill_chunk < 0) prefill_chunk = 0;
    if (prefill_chunk > B_MAX) prefill_chunk = B_MAX;

    if (strcmp(mode, "tokenize") == 0) {
        // only the header is needed (vocab_size): skip mapping and dequantizing the weights
        Config config;
        FILE *file = fopen(checkpoint_path, "rb");
        if (!file || fread(&config, sizeof(Config), 1, file) != 1 || config.magic_number != 0x616a6331) {
            fprintf(stderr, "Couldn't read qwen3.c checkpoint header from %s\n", checkpoint_path);
            exit(EXIT_FAILURE);
        }
        fclose(file);
        Tokenizer tokenizer;
        double t0 = now_s();
        build_tokenizer(&tokenizer, checkpoint_path, config.vocab_size, enable_thinking);
        tokenize(&tokenizer, prompt, now_s() - t0);
        free_tokenizer(&tokenizer);
        return 0;
    }

#ifdef _OPENMP
    // default thread count if OMP_NUM_THREADS is not set: not all CPUs. libgomp's default (all of
    // them) collapses on a shared machine: with every CPU running a spinning thread, any other
    // process preempting one of them stalls the others at every barrier (~200 per token). On the
    // 16-vCPU dev VM: 16 threads ~20 tok/s decode, 12 the best prefill with decode as at 8
    if (!getenv("OMP_NUM_THREADS")) {
        int np = omp_get_num_procs();
        int nt = np > 4 ? np - np / 4 : np;
        omp_set_num_threads(nt < 12 ? nt : 12);
    }
#endif

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path, ctx_length, prefill_chunk);

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
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
