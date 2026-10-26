/* GGUF reader (llama.cpp's format, versions 2 and 3). Layout (little-endian):
 *   "GGUF", u32 version, u64 n_tensors, u64 n_kv;
 *   n_kv x (string key, u32 value type, value); string = u64 length + bytes (no NUL); array =
 *     u32 element type, u64 count, the elements (no nested arrays here);
 *   n_tensors x (string name, u32 n_dims, u64 ne[n_dims], u32 ggml type, u64 offset);
 *   padding to general.alignment (default 32), then the tensor data, tensor i at data_off + offset.
 * ne[0] is the fastest dimension: a row-major (d, k) matrix has ne = { k, d }.
 * The file is untrusted input: every count, length and offset is checked against the file size
 * (gguf_take, overflow-safe) before use, counts before any allocation, and every tensor's data
 * range when its header is read; errors (gguf_fail) name the file and end the load (LoadErr).
 */
#include "internal.h"

static const uint8_t gv_size[GV_N] = { 1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8 }; // scalar value bytes

void gguf_fail(const Gguf *g, const char *fmt, const char *arg) {
    // "GGUF <path>: " + fmt with its one %s = arg, then back to rq_model_load
    char m[512];
    snprintf(m, sizeof(m), fmt, arg ? arg : "");
    load_fail(g->err, "GGUF %s: %s", g->path, m);
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

uint64_t gguf_str(const Gguf *g, uint64_t *pos, const char **s, const char *what) {
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

void gguf_parse(Gguf *g) {
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
    if (!g->kv || !g->t) load_fail(g->err, "malloc failed!");

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

int gguf_get_int(const Gguf *g, const char *key, int64_t *v) {
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

const char *gguf_get_str(const Gguf *g, const char *key, uint64_t *len) {
    // string metadata (not NUL-terminated: *len bytes), NULL if absent
    const GgufKV *kv = gguf_find(g, key);
    if (!kv) return NULL;
    if (kv->type != GV_STR) gguf_fail(g, "metadata %s is not a string", key);
    uint64_t pos = kv->off;
    const char *s;
    *len = gguf_str(g, &pos, &s, key);
    return s;
}

const GgufKV *gguf_get_array(const Gguf *g, const char *key, uint32_t type) {
    // array metadata with elements of this type, NULL if absent
    const GgufKV *kv = gguf_find(g, key);
    if (kv && (kv->type != GV_ARR || kv->arr_type != type)) gguf_fail(g, "metadata %s: unexpected type", key);
    return kv;
}

const GgufTensor *gguf_tensor(const Gguf *g, const char *name) {
    size_t n = strlen(name);
    for (uint64_t i = 0; i < g->n_t; i++)
        if (g->t[i].name_len == n && !memcmp(g->t[i].name, name, n))
            return &g->t[i];
    return NULL;
}

const uint8_t *gguf_weight(const Gguf *g, const char *name, uint32_t type, uint64_t ne0, uint64_t ne1) {
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

void gguf_config(const Gguf *g, Config *c, float *rope_theta) {
    // the hyperparameters from the metadata (llama.cpp's qwen3 keys), checked against what this
    // engine implements: Qwen3 (any RoPE theta: 1e6, the 2507 models 5e6; no RoPE scaling) with RMSNorm
    // eps 1e-6 (the constant in rmsnorm); Q8_0 / Q4_K / Q6_K weights (gguf_matrices)
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
    if (!gguf_get_float(g, "qwen3.rope.freq_base", &theta) || !(theta >= 1e3f && theta <= 1e9f))
        gguf_fail(g, "qwen3.rope.freq_base missing or out of range (1e3 .. 1e9)%s", NULL);
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
    *rope_theta = theta;
}

static int has_text(const char *s, uint64_t n, const char *pat) {
    size_t k = strlen(pat);
    for (uint64_t i = 0; k <= n && i <= n - k; i++)
        if (!memcmp(s + i, pat, k)) return 1;
    return 0;
}

int gguf_chat_variant(const Gguf *g) {
    // The Qwen3 chat template variant, from the Jinja source in tokenizer.chat_template (not interpreted:
    // the renderings are built in, model.c). The Qwen3 / Qwen3-2507 templates (HF's and Unsloth's
    // rewrites in their GGUFs) differ in the generation prompt:
    //   hybrid (Qwen3-0.6B .. 14B): enable_thinking; '<|im_start|>assistant\n' + '<think>\n\n</think>\n\n'
    //     when thinking is off
    //   thinking only (Qwen3-*-Thinking-2507): '<|im_start|>assistant\n<think>\n', no enable_thinking
    //   instruct (Qwen3-*-Instruct-2507): '<|im_start|>assistant\n', no think block at all
    // Absent, or not a ChatML template: hybrid (the built-in templates, as before)
    uint64_t n = 0;
    const char *t = gguf_get_str(g, "tokenizer.chat_template", &n);
    if (!t || !has_text(t, n, "<|im_start|>") || has_text(t, n, "enable_thinking")) return RQ_CHAT_HYBRID;
    if (has_text(t, n, "<|im_start|>assistant\\n<think>\\n'") || has_text(t, n, "<|im_start|>assistant\n<think>\n'") ||
        has_text(t, n, "<|im_start|>assistant\\n<think>\\n\"") || has_text(t, n, "<|im_start|>assistant\n<think>\n\""))
        return RQ_CHAT_THINKING;
    return RQ_CHAT_INSTRUCT;
}
