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

// ----------------------------------------------------------------------------
// Globals
int GS = 0; // group size global for quantization of the weights

// Maximum input prompt buffer size
#define PROMPT_BUFFER_SIZE 32768

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
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float *key_cache;   // (layer, seq_len, dim)
    float *value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    float *data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState* s, Config *p) {
    // we calloc instead of malloc to keep valgrind happy
    int all_heads_dim = p->n_heads * p->head_dim;
    int kv_dim = p->n_kv_heads * p->head_dim;

    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(all_heads_dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->xq = (QuantizedTensor) { .q = calloc(all_heads_dim, sizeof(int8_t)), .s = calloc(all_heads_dim / GS, sizeof(float)) };
    s->hq = (QuantizedTensor) { .q = calloc(p->hidden_dim, sizeof(int8_t)), .s = calloc(p->hidden_dim / GS, sizeof(float)) };
    s->q = calloc(all_heads_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc(p->n_layers * (uint64_t)p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * (uint64_t)p->seq_len * kv_dim, sizeof(float));

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->hb || !s->hb2 || !s->q || !s->att || !s->logits || !s->key_cache || !s->value_cache) {
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
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
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
}

void build_transformer(Transformer *t, char *checkpoint_path, int ctx_length) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->data, &t->file_size, ctx_length);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
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

void matmul(float *xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized

    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        float val = 0;
        int in = i * n;

        // do the matmul in groups of GS
        for (int j = 0; j <= n - GS; j += GS) {
            int32_t ival = 0;
            for (int k = 0; k < GS; k++)
                ival += x->q[j + k] * w->q[in + j + k];

            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
        }

        xout[i] = val;
    }
}

float *forward(Transformer *transformer, int token, int pos, int compute_logits) {
    // compute_logits = 0 skips the final rmsnorm + classifier (prompt tokens whose
    // logits are discarded) and returns NULL; the KV cache is filled either way
    Config *p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    int kv_dim = p->n_kv_heads * p->head_dim;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int all_heads_dim = p->n_heads * p->head_dim;

    // copy the token embedding into s->x
    memcpy(s->x, w->token_embedding_table + token * p->dim, p->dim * sizeof(float));

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {
        // save key,value at this time step (pos) to our kv cache
        uint64_t loff = l * (uint64_t)p->seq_len * kv_dim; // kv cache layer offset for convenience

        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // attention rmsnorm
        rmsnorm(s->xb, s->x, w->rms_att_weight + l * p->dim, p->dim);

        // qkv matmuls for this position
        quantize(&s->xq, s->xb, p->dim);
        matmul(s->q, &s->xq, w->wq + l, p->dim, all_heads_dim);
        matmul(s->k, &s->xq, w->wk + l, p->dim, kv_dim);
        matmul(s->v, &s->xq, w->wv + l, p->dim, kv_dim);

        /* ------------ Q-RMSNorm + rotate each query head ------------- */
        for (int h = 0; h < p->n_heads; h++) {
            float *q = s->q + h * p->head_dim;

            rmsnorm(q, q, w->q_norm_weights + l * p->head_dim, p->head_dim);
            for (int j = 0; j < p->head_dim/2; j++) {
                float freq = powf(1e6, -(float)j / (p->head_dim/2));
                float cos_freq = cosf(pos * freq), sin_freq = sinf(pos * freq);

                float x = q[j]; // real part
                float y = q[j + p->head_dim/2]; // imag part

                q[j] = x * cos_freq - y * sin_freq; // new real
                q[j + p->head_dim/2] = x * sin_freq + y * cos_freq; // new imag
            }
        }

        /* ------------ K-RMSNorm + rotate each key head ------------ */
        for (int h = 0; h < p->n_kv_heads; h++) {
            float *k = s->k + h * p->head_dim;

            rmsnorm(k, k, w->k_norm_weights + l * p->head_dim, p->head_dim);
            for (int j = 0; j < p->head_dim/2; j++) {
                float freq = powf(1e6, -(float)j / (p->head_dim/2));
                float cos_freq = cosf(pos * freq), sin_freq = sinf(pos * freq);

                float x = k[j];
                float y = k[j + p->head_dim/2];

                k[j] = x * cos_freq - y * sin_freq;
                k[j + p->head_dim/2] = x * sin_freq + y * cos_freq;
            }
        }

        // multihead attention. iterate over all heads
        #pragma omp parallel for
        for (int h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float *q = s->q + h * p->head_dim;
            // attention scores for this head
            float *att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * p->head_dim;
                // calculate the attention score as the dot product of q and k
                float score = 0;
                for (int i = 0; i < p->head_dim; i++)
                    score += q[i] * k[i];

                // save the score to the attention buffer
                att[t] = score / sqrtf(p->head_dim);
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float *xb = s->xb + h * p->head_dim;
            memset(xb, 0, p->head_dim * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * p->head_dim;
                // get the attention weight for this timestep, then accumulate the weighted value into xb
                for (int i = 0; i < p->head_dim; i++)
                    xb[i] += att[t] * v[i];
            }
        }

        // final matmul to get the output of the attention
        quantize(&s->xq, s->xb, all_heads_dim);
        matmul(s->xb, &s->xq, w->wo + l, all_heads_dim, p->dim);

        // residual connection back into s->x
        for (int i = 0; i < p->dim; i++)
            s->x[i] += s->xb[i];

        // ffn rmsnorm
        rmsnorm(s->xb, s->x, w->rms_ffn_weight + l * p->dim, p->dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        quantize(&s->xq, s->xb, p->dim);
        matmul(s->hb, &s->xq, w->w1 + l, p->dim, p->hidden_dim);
        matmul(s->hb2, &s->xq, w->w3 + l, p->dim, p->hidden_dim);

        // SwiGLU non-linearity
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid. The exponent is clamped: under
        // -Ofast, expf(-x) = inf for x < -88 gives NaN (vrcp14ps + Newton step: 0 * inf), and
        // the NaN broke quantize()'s vectorized max, i.e. the scale of its whole group
        for (int i = 0; i < p->hidden_dim; i++)
            s->hb[i] *= s->hb2[i] * (1.0f / (1.0f + expf(fminf(-s->hb[i], 80.0f))));

        // final matmul to get the output of the ffn
        quantize(&s->hq, s->hb, p->hidden_dim);
        matmul(s->xb, &s->hq, w->w2 + l, p->hidden_dim, p->dim);

        // residual connection
        for (int i = 0; i < p->dim; i++)
            s->x[i] += s->xb[i];
    }

    if (!compute_logits)
        return NULL;

    // final rmsnorm
    rmsnorm(s->x, s->x, w->rms_final_weight, p->dim);

    // classifier into logits
    quantize(&s->xq, s->x, p->dim);
    matmul(s->logits, &s->xq, w->wcls, p->dim, p->vocab_size);
    return s->logits;
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
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
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

int compare(const void *a, const void *b) {
    ProbIndex *a_ = (ProbIndex *) a;
    ProbIndex *b_ = (ProbIndex *) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf)
            return probindex[i].index;
    }
    return probindex[last_idx].index; // in case of rounding errors
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
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    int n_gen = 0;
    double t0 = now_s(), t_prefill = t0;

    while (pos < transformer->config.seq_len) {
        // forward the transformer to get logits for the next token
        // (only needed from the last prompt token on)
        float *logits = forward(transformer, token, pos, pos >= num_prompt_tokens - 1);

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

void read_stdin(const char *guide, char *buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n')
            buffer[len - 1] = 0; // strip newline
    }
}

// ----------------------------------------------------------------------------
// chat loop

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *cli_user_prompt, char *system_prompt) {
    // buffers for reading the system prompt and user prompt from stdin
    char user_prompt[PROMPT_BUFFER_SIZE];
    char rendered_prompt[PROMPT_BUFFER_SIZE];
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc(PROMPT_BUFFER_SIZE * sizeof(int));

    // start the main loop
    int user_turn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int pos = 0;     // position in the sequence
    int n_gen = 0;
    double t0 = 0, t_prefill = 0;

    while (1) {
        // if context window is exceeded, clear it
        if (pos >= transformer->config.seq_len) {
            printf("\n(context window full, clearing)\n");
            user_turn = 1;
            pos = 0;
        }

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the user prompt
            if (cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                if (pos > 0)
                    break;
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                read_stdin("\n> ", user_prompt, sizeof(user_prompt));
                // terminate if user enters a blank prompt
                if (!user_prompt[0])
                    break;
            }

            // render user/system prompts into the Qwen3 prompt template schema
            if (pos == 0 && system_prompt)
                sprintf(rendered_prompt, tokenizer->system_prompt_template, system_prompt, user_prompt);
            else
                sprintf(rendered_prompt, tokenizer->prompt_template, user_prompt);

            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, prompt_tokens, &num_prompt_tokens);
            pos = 0; // reset the user index
            user_turn = 0;
            n_gen = 0;
            t0 = t_prefill = now_s();
        }

        // determine the token to pass into the transformer next
        if (pos < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[pos];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }

        // forward the transformer to get logits for the next token; `next` is only used
        // from the last prompt token on, so earlier prompt tokens skip the classifier
        float *logits = forward(transformer, token, pos, pos >= num_prompt_tokens - 1);
        pos++;
        if (logits)
            next = sample(sampler, logits);
        else if (sampler->temperature != 0)
            random_f32(&sampler->rng_state); // sample() used to run here too: keep the seeded RNG sequence
        if (pos == num_prompt_tokens) t_prefill = now_s();
        else if (pos > num_prompt_tokens) n_gen++;

        // assistant is responding
        if (pos >= num_prompt_tokens) {
            int limit_hit = max_new_tokens > 0 && n_gen >= max_new_tokens;
            if (limit_hit || token == tokenizer->bos_token_id || token == tokenizer->eos_token_id) {
                // EOS token (or -n limit) ends the assistant turn
                printf("\n");
                report_timing(num_prompt_tokens, t0, t_prefill, n_gen, now_s());
                user_turn = 1;
            } else if (next != tokenizer->bos_token_id && next != tokenizer->eos_token_id) {
                printf("%s", decode(tokenizer, next));
                fflush(stdout);
            }
        }
    }
    free(prompt_tokens);
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

void dump_logits(Transformer *transformer, Tokenizer *tokenizer, char *text, char *out_path) {
    // file format: int32 n_tokens, int32 vocab_size, int32 ids[n_tokens], float32 logits[n_tokens][vocab_size]
    int n_tokens = 0, vocab_size = transformer->config.vocab_size;
    int *tokens = (int *)malloc((strlen(text) + 3) * sizeof(int));
    encode(tokenizer, text, tokens, &n_tokens);
    if (n_tokens < 1 || n_tokens > transformer->config.seq_len) {
        fprintf(stderr, "dump_logits: need 1..%d tokens, got %d\n", transformer->config.seq_len, n_tokens);
        exit(EXIT_FAILURE);
    }

    float *out = (float *)malloc((size_t)n_tokens * vocab_size * sizeof(float));
    if (!out) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
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
    fprintf(stderr, "  -c <int>    context window size, 0 (default) = max_seq_len\n");
    fprintf(stderr, "  -m <string> mode: generate|chat|tokenize|logits, default: chat\n");
    fprintf(stderr, "              tokenize: print the token ids of -i (raw text, no template)\n");
    fprintf(stderr, "              logits: write the logits at every position of -i to -o (binary)\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -o <string> output file for -m logits\n");
    fprintf(stderr, "  -y <string> system prompt in chat mode, default is none\n");
    fprintf(stderr, "  -r <int>    reasoning mode, 0 (default) = no thinking, 1 = thinking\n");
    fprintf(stderr, "  -n <int>    max new tokens per response, 0 (default) = unlimited\n");
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
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0) temperature = 0;
    if (topp < 0 || 1.0 < topp) topp = 0.9;
    if ((strcmp(mode, "tokenize") == 0 || strcmp(mode, "logits") == 0) && !prompt) { error_usage(); }
    if (strcmp(mode, "logits") == 0 && !out_path) { error_usage(); }

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

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path, ctx_length);

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
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt);
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
    return 0;
}
