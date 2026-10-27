/* The model (RqModel): loading a qwen3.c .bin or a GGUF file, the weights converted / repacked for the
 * kernels once (prepare_weights), the tokenizer, the chat templates. Read-only once loaded.
 * Errors anywhere in the load (corrupt or unsupported files, allocation failures) go through
 * load_fail / gguf_fail, which longjmp back to rq_model_load: it frees what was allocated so far
 * (every allocation is reachable from the model) and returns NULL with the message. */
#include "internal.h"

int GS = 0;
static int gs_models = 0; // loaded models with weights (all of the same group size GS)

void load_fail(LoadErr *e, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(e->msg, e->len, fmt, ap);
    va_end(ap);
    longjmp(e->jb, 1);
}

#if defined _WIN32
    #include <malloc.h> // _aligned_malloc
#endif
void *alloc_big(size_t n) {
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

void free_big(void *p, size_t n) {
#if defined _WIN32
    (void)n;
    _aligned_free(p);
#else
    if (p) munmap(p, n);
#endif
}

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

// ---------------------------------------------------------------------------- .bin weights

static void bin_tensors(QuantizedTensor **dst, char **ptr, int n, size_t size_each, LoadErr *e) {
    // n quantized tensors of size_each values from *ptr on (the .bin layout: int8 values, then one
    // float scale per GS values), into *dst (malloc'd)
    QuantizedTensor *res = *dst = calloc(n, sizeof(QuantizedTensor));
    if (!res) load_fail(e, "malloc failed!");
    for (int i = 0; i < n; i++) {
        res[i].q = (int8_t *)*ptr;
        *ptr += size_each;
        res[i].s = (float *)*ptr;
        *ptr += size_each / GS * sizeof(float);
        res[i].type = GGML_Q8_0;
    }
}

static void bin_map_weights(RqModel *m, const char *path, LoadErr *e) {
    // the weights of a .bin in its mapping: after the 256-byte header the norm weights (fp32), then
    // the quantized matrices (the token embeddings first; their rows are dequantized at lookup,
    // embed_row: upstream dequantized the whole table here, 622 MB of floats, ~0.3 s of page faults)
    Config *p = &m->config;
    TransformerWeights *w = &m->weights;
    size_t L = p->n_layers, dim = p->dim, hd = p->head_dim, hid = p->hidden_dim, V = p->vocab_size;
    size_t qd = (size_t)p->n_heads * hd, kvd = (size_t)p->n_kv_heads * hd;
    size_t nq = V * dim + L * (dim * qd + 2 * dim * kvd + qd * dim + 3 * dim * hid) + (p->shared_classifier ? 0 : dim * V);
    size_t need = 256 + (2 * L * dim + dim + 2 * L * hd) * sizeof(float) + nq + nq / GS * sizeof(float);
    if (need > m->file_size) load_fail(e, "File %s is truncated (%zu bytes, need %zu)", path, m->file_size, need);
    float *f = (float *)((char *)m->data + 256);
    w->rms_att_weight = f;
    f += L * dim;
    w->rms_ffn_weight = f;
    f += L * dim;
    w->rms_final_weight = f;
    f += dim;
    w->q_norm_weights = f;
    f += L * hd;
    w->k_norm_weights = f;
    f += L * hd;
    char *ptr = (char *)f;
    bin_tensors(&w->q_tokens, &ptr, 1, V * dim, e);
    bin_tensors(&w->wq, &ptr, (int)L, dim * qd, e);
    bin_tensors(&w->wk, &ptr, (int)L, dim * kvd, e);
    bin_tensors(&w->wv, &ptr, (int)L, dim * kvd, e);
    bin_tensors(&w->wo, &ptr, (int)L, qd * dim, e);
    bin_tensors(&w->w1, &ptr, (int)L, dim * hid, e);
    bin_tensors(&w->w2, &ptr, (int)L, hid * dim, e);
    bin_tensors(&w->w3, &ptr, (int)L, dim * hid, e);
    if (p->shared_classifier)
        w->wcls = w->q_tokens;
    else
        bin_tensors(&w->wcls, &ptr, 1, dim * V, e);
}

// ---------------------------------------------------------------------------- GGUF weights

static void gguf_matrix(const Gguf *g, QuantizedTensor *res, const char *name, int d, int k, int f32) {
    // one matrix (d, k): Q8_0, Q3_K, Q4_K, Q5_K or Q6_K (dispatch per tensor: Q4_K_M mixes them within a role), or F32 if
    // f32 (only where forward.c's fp32 path takes it: ssm_alpha / ssm_beta), its data in the mapping (q / s / p /
    // kp / f are set by prepare_weights)
    const GgufTensor *t = gguf_tensor(g, name);
    uint32_t type = t ? t->type : GGML_Q8_0; // (missing: gguf_weight reports it)
    if (type != GGML_Q8_0 && type != GGML_Q3_K && type != GGML_Q4_K && type != GGML_Q5_K && type != GGML_Q6_K &&
        !(f32 && type == GGML_F32)) {
        char msg[160];
        snprintf(msg, sizeof(msg), "tensor %%s has ggml type %u, need Q8_0 (8), Q3_K (11), Q4_K (12), Q5_K (13)%s or Q6_K (14)", type,
                 f32 ? ", F32 (0)" : "");
        gguf_fail(g, msg, name);
    }
    res->b = gguf_weight(g, name, type, k, d);
    res->type = (int)type;
    // the repacked layouts: 16-row blocks, 256-column super-blocks (activation groups of 32)
    if (type != GGML_Q8_0 && type != GGML_F32 && (d % KQ_ROWS || k % QK_K || GS != 32))
        gguf_fail(g, "tensor %s: K-quant shape not supported (rows % 16, columns % 256)", name);
}

static void gguf_matrices(RqModel *m, QuantizedTensor **dst, int L, const int *sel, const char *fmt, int d, int k, int f32) {
    // the matrices (d, k) of layers 0..L-1 (fmt: name with the layer number, or a plain name for L = 1), those
    // with sel[l] >= 0 into (*dst)[sel[l]] (sel NULL: all, into (*dst)[l]): gguf_matrix (f32: F32 accepted too)
    const Gguf *g = &m->gguf;
    int n = 0;
    for (int l = 0; l < L; l++)
        n += !sel || sel[l] >= 0;
    QuantizedTensor *res = *dst = calloc(n > 0 ? n : 1, sizeof(QuantizedTensor));
    if (!res) load_fail(g->err, "malloc failed!");
    for (int l0 = 0; l0 < L; l0++) {
        if (sel && sel[l0] < 0) continue;
        int l = sel ? sel[l0] : l0;
        char name[96];
        snprintf(name, sizeof(name), fmt, l0);
        gguf_matrix(g, &res[l], name, d, k, f32);
        if (res[l].type != GGML_Q8_0 && res[l].type != GGML_F32)
            m->kq = 1;
    }
}

static void gguf_map_mtp(RqModel *m) {
    // the MTP block (MtpBlock): qwen35 with nextn_predict_layers 1, blk.<n_layers> after the main layers. Validated
    // from the tensor directory (every tensor present with its shape and a type the kernels take; norms F32); not
    // read. Other MTP layouts (several blocks, another architecture) are refused: the layer pattern of a
    // future one is unknown
    const Config *p = &m->config;
    const Gguf *g = &m->gguf;
    MtpBlock *x = &m->mtp;
    int dim = p->dim, hd = p->head_dim, hid = p->hidden_dim, kvd = p->n_kv_heads * hd, b = p->n_layers;
    if (!m->hy.on || m->hy.nextn != 1) {
        char a[64];
        snprintf(a, sizeof(a), "%s.nextn_predict_layers = %d", m->hy.on ? "qwen35" : "qwen3", m->hy.nextn);
        gguf_fail(g, "MTP (multi-token prediction) blocks: %s is not supported (qwen35 with one block only)", a);
    }
    struct { QuantizedTensor *t; const char *role; int d, k; } mt[] = {
        { &x->wq, "attn_q", p->n_heads * m->qh, dim }, { &x->wk, "attn_k", kvd, dim }, { &x->wv, "attn_v", kvd, dim },
        { &x->wo, "attn_output", dim, p->n_heads * hd }, { &x->w1, "ffn_gate", hid, dim }, { &x->w2, "ffn_down", dim, hid },
        { &x->w3, "ffn_up", hid, dim }, { &x->eh_proj, "nextn.eh_proj", dim, 2 * dim } };
    struct { const float **v; const char *role; int n; } nv[] = {
        { &x->attn_norm, "attn_norm", dim }, { &x->ffn_norm, "post_attention_norm", dim }, { &x->q_norm, "attn_q_norm", hd },
        { &x->k_norm, "attn_k_norm", hd }, { &x->enorm, "nextn.enorm", dim }, { &x->hnorm, "nextn.hnorm", dim },
        { &x->head_norm, "nextn.shared_head_norm", dim } };
    char name[96];
    for (size_t i = 0; i < sizeof(mt) / sizeof(mt[0]); i++) {
        snprintf(name, sizeof(name), "blk.%d.%s.weight", b, mt[i].role);
        gguf_matrix(g, mt[i].t, name, mt[i].d, mt[i].k, 0);
    }
    for (size_t i = 0; i < sizeof(nv) / sizeof(nv[0]); i++) {
        snprintf(name, sizeof(name), "blk.%d.%s.weight", b, nv[i].role);
        *nv[i].v = (const float *)gguf_weight(g, name, GGML_F32, nv[i].n, 0);
    }
    x->layer = b;
}

static void model_layers(RqModel *m, LoadErr *e) {
    // the layer maps: per layer its attention layer index (KV cache, attention weights) and its linear-attention
    // layer index (Qwen3.5: layer l is attention iff (l + 1) % interval == 0)
    int L = m->config.n_layers;
    m->kv_layer = malloc(L * sizeof(int));
    m->lin_layer = malloc(L * sizeof(int));
    if (!m->kv_layer || !m->lin_layer) load_fail(e, "malloc failed!");
    m->n_attn = 0;
    for (int l = 0, nl = 0; l < L; l++) {
        int att = !m->hy.on || (l + 1) % m->hy.interval == 0;
        m->kv_layer[l] = att ? m->n_attn++ : -1;
        m->lin_layer[l] = att ? -1 : nl++;
    }
}

static size_t norm_floats(const RqModel *m) {
    // the F32 tensors of gguf_copy_norms: floats
    const Config *p = &m->config;
    const Hybrid *hy = &m->hy;
    return (size_t)2 * p->n_layers * p->dim + p->dim + (size_t)2 * m->n_attn * p->head_dim +
           (size_t)hy->n_lin * (hy->conv * hy->conv_dim + 2 * hy->nv + hy->dv);
}

static void gguf_copy_norms(RqModel *m) {
    // The F32 tensors (norms; qwen35's conv kernel, transposed to tap-major, and ssm_a / ssm_dt / ssm_norm) are
    // copied into one buffer, attention ones per attention layer, the others per linear layer (after the memory
    // check: the first tensor data read)
    Config *p = &m->config;
    TransformerWeights *w = &m->weights;
    const Gguf *g = &m->gguf;
    const Hybrid *hy = &m->hy;
    int L = p->n_layers, dim = p->dim, hd = p->head_dim, NL = hy->n_lin;
    float *f = w->norms = malloc(norm_floats(m) * sizeof(float));
    if (!f) load_fail(g->err, "malloc failed!");
    // (n: layers, 1 for a plain name; sel: the layers with sel[l] >= 0, NULL all)
    struct { float **dst; const char *fmt; int n; const int *sel; int len; } nm[] = {
        { &w->rms_att_weight, "blk.%d.attn_norm.weight", L, NULL, dim },
        { &w->rms_ffn_weight, hy->on ? "blk.%d.post_attention_norm.weight" : "blk.%d.ffn_norm.weight", L, NULL, dim },
        { &w->rms_final_weight, "output_norm.weight", 1, NULL, dim },
        { &w->q_norm_weights, "blk.%d.attn_q_norm.weight", L, m->kv_layer, hd },
        { &w->k_norm_weights, "blk.%d.attn_k_norm.weight", L, m->kv_layer, hd },
        { &w->ssm_a, "blk.%d.ssm_a", NL ? L : 0, m->lin_layer, hy->nv },
        { &w->ssm_dt, "blk.%d.ssm_dt.bias", NL ? L : 0, m->lin_layer, hy->nv },
        { &w->ssm_norm, "blk.%d.ssm_norm.weight", NL ? L : 0, m->lin_layer, hy->dv } };
    for (size_t i = 0; i < sizeof(nm) / sizeof(nm[0]); i++) {
        if (!nm[i].n) continue;
        *nm[i].dst = f;
        for (int l = 0; l < nm[i].n; l++) {
            if (nm[i].sel && nm[i].sel[l] < 0) continue;
            char name[96];
            snprintf(name, sizeof(name), nm[i].fmt, l);
            memcpy(f, gguf_weight(g, name, GGML_F32, nm[i].len, 0), nm[i].len * sizeof(float));
            f += nm[i].len;
        }
    }
    if (NL) { // the conv kernels: (conv_dim, conv) in the file, per channel group (conv, n) here
        w->ssm_conv = f;
        for (int l = 0; l < L; l++) {
            if (m->lin_layer[l] < 0) continue;
            char name[96];
            snprintf(name, sizeof(name), "blk.%d.ssm_conv1d.weight", l);
            const float *src = (const float *)gguf_weight(g, name, GGML_F32, hy->conv, hy->conv_dim);
            // (per channel group, forward.c dn_prep: q heads, k heads, v heads; group g = channels c0 .. c0 + n - 1 at conv * c0)
            for (int g = 0; g < 2 * hy->nk + hy->nv; g++) {
                int n = g < 2 * hy->nk ? hy->dk : hy->dv;
                int c0 = g < 2 * hy->nk ? g * hy->dk : 2 * hy->nk * hy->dk + (g - 2 * hy->nk) * hy->dv;
                for (int c = 0; c < n; c++)
                    for (int j = 0; j < hy->conv; j++)
                        memcpy(f + (size_t)hy->conv * c0 + (size_t)j * n + c, src + (size_t)(c0 + c) * hy->conv + j, sizeof(float));
            }
            f += (size_t)hy->conv * hy->conv_dim;
        }
    }
}

static void gguf_map_weights(RqModel *m) {
    // the matrices (from the tensor directory: no tensor data read). Tensor names of llama.cpp's
    // convert_hf_to_gguf.py for qwen3 / qwen35 (no Q/K row permutation: that is done for the llama architecture
    // only; checked by the logits test against HF). ffn_gate / ffn_up / ffn_down are w1 / w3 / w2 (qwen35: the FFN
    // norm is post_attention_norm; gguf_copy_norms)
    Config *p = &m->config;
    TransformerWeights *w = &m->weights;
    const Hybrid *hy = &m->hy;
    int L = p->n_layers, dim = p->dim, hd = p->head_dim, qd = p->n_heads * hd, kvd = p->n_kv_heads * hd, NL = hy->n_lin;
    gguf_matrices(m, &w->q_tokens, 1, NULL, "token_embd.weight", p->vocab_size, dim, 0);
    gguf_matrices(m, &w->wq, L, m->kv_layer, "blk.%d.attn_q.weight", p->n_heads * m->qh, dim, 0);
    gguf_matrices(m, &w->wk, L, m->kv_layer, "blk.%d.attn_k.weight", kvd, dim, 0);
    gguf_matrices(m, &w->wv, L, m->kv_layer, "blk.%d.attn_v.weight", kvd, dim, 0);
    gguf_matrices(m, &w->wo, L, m->kv_layer, "blk.%d.attn_output.weight", dim, qd, 0);
    gguf_matrices(m, &w->w1, L, NULL, "blk.%d.ffn_gate.weight", p->hidden_dim, dim, 0);
    gguf_matrices(m, &w->w2, L, NULL, "blk.%d.ffn_down.weight", dim, p->hidden_dim, 0);
    gguf_matrices(m, &w->w3, L, NULL, "blk.%d.ffn_up.weight", p->hidden_dim, dim, 0);
    if (NL) { // (alpha / beta: F32 in Qwen3.8-27B's file, [48, 5120]; forward.c's fp32 path, f32_rows)
        gguf_matrices(m, &w->wqkv, L, m->lin_layer, "blk.%d.attn_qkv.weight", hy->conv_dim, dim, 0);
        gguf_matrices(m, &w->wz, L, m->lin_layer, "blk.%d.attn_gate.weight", hy->vdim, dim, 0);
        gguf_matrices(m, &w->wa, L, m->lin_layer, "blk.%d.ssm_alpha.weight", hy->nv, dim, 1);
        gguf_matrices(m, &w->wb, L, m->lin_layer, "blk.%d.ssm_beta.weight", hy->nv, dim, 1);
        gguf_matrices(m, &w->wso, L, m->lin_layer, "blk.%d.ssm_out.weight", dim, hy->vdim, 0);
    }
    if (p->shared_classifier)
        w->wcls = w->q_tokens;
    else
        gguf_matrices(m, &w->wcls, 1, NULL, "output.weight", p->vocab_size, dim, 0);
    if (hy->nextn)
        gguf_map_mtp(m);
}

// ---------------------------------------------------------------------------- weight conversion

static void read_at(int fd, const void *base, void *dst, size_t n, const void *mapped) {
    // n bytes of the checkpoint at the file offset of `mapped` (a pointer into its mapping at base)
    // into dst with pread: the file pages are copied from the page cache without being mapped into
    // the process (no page faults, not part of our resident size, nothing to drop afterwards). Falls
    // back to copying from the mapping
    size_t done = 0;
#ifndef _WIN32
    off_t off = (off_t)((const char *)mapped - (const char *)base);
    while (done < n && fd >= 0) {
        ssize_t r = pread(fd, (char *)dst + done, n - done, off + (off_t)done);
        if (r <= 0) break;
        done += (size_t)r;
    }
#else
    (void)fd; (void)base;
#endif
    if (done < n)
        memcpy((char *)dst + done, (const char *)mapped + done, n - done);
}

static void drop_cached(int fd, const void *base, const void *mapped, size_t n) {
    // drops the page cache of n bytes of the file at the offset of `mapped` (the whole file if n == 0), after
    // read_at copied them: clean, unmapped pages go now instead of competing with the repacked copy for the RAM
    // (partial pages at the ends stay: the final whole-file call in load_model takes them). Per matrix, not per
    // work item: each call costs (an LRU drain on every CPU; per 64 KB item: +1.1 s on the 8B's 6 s cold load)
#if !defined _WIN32 && defined POSIX_FADV_DONTNEED
    if (fd >= 0) posix_fadvise(fd, (off_t)((const char *)mapped - (const char *)base), (off_t)n, POSIX_FADV_DONTNEED);
#else
    (void)fd; (void)base; (void)mapped; (void)n;
#endif
}

typedef struct { QuantizedTensor *t; int n, d, k; } PrepMat; // a role of prepare_weights: n matrices (d, k)

static int prep_mats(const RqModel *m, PrepMat *mt) {
    // the matmul weights prepare_weights converts (7 per layer, the classifier, Qwen3.5's linear layers): mt[13]
    const Config *p = &m->config;
    const TransformerWeights *w = &m->weights;
    const Hybrid *hy = &m->hy;
    int L = p->n_layers, dim = p->dim, hid = p->hidden_dim, NA = m->n_attn, NL = hy->n_lin;
    int qd = p->n_heads * p->head_dim, kvd = p->n_kv_heads * p->head_dim;
    PrepMat t[] = {
        { w->wq, NA, p->n_heads * m->qh, dim }, { w->wk, NA, kvd, dim }, { w->wv, NA, kvd, dim }, { w->wo, NA, dim, qd },
        { w->w1, L, hid, dim }, { w->w2, L, dim, hid }, { w->w3, L, hid, dim }, { w->wcls, 1, p->vocab_size, dim },
        { w->wqkv, NL, hy->conv_dim, dim }, { w->wz, NL, hy->vdim, dim }, { w->wa, NL, hy->nv, dim },
        { w->wb, NL, hy->nv, dim }, { w->wso, NL, dim, hy->vdim } };
    memcpy(mt, t, sizeof(t));
    return sizeof(t) / sizeof(t[0]);
}

static int prep_pack(const PrepMat *mt, int nm, char *warning, size_t wlen) {
    // 1 if the Q8_0 matrices are repacked for the SIMD kernels (else a warning, if not NULL)
    int pack = 0;
    (void)mt; (void)nm; (void)warning; (void)wlen;
#ifdef RUNQ_Q8X
    pack = 1;
    for (int i = 0; i < nm && pack; i++) {
        int q8 = 0; // (roles without a Q8_0 matrix do not count: e.g. F32 alpha / beta of 12 rows)
        for (int l = 0; l < mt[i].n && !q8; l++)
            q8 = mt[i].t[l].type == GGML_Q8_0;
        if (q8 && !q8x_ok(mt[i].d, mt[i].k, GS)) {
            if (warning)
                snprintf(warning, wlen, "weights not packable for the SIMD Q8_0 kernels (GS %d, %d x %d), using the generic ones", GS, mt[i].d, mt[i].k);
            pack = 0;
        }
    }
#endif
    return pack;
}

static size_t prep_bytes(const QuantizedTensor *t, int d, int k, int pack) {
    // a matrix's place in the packed buffer (64-byte multiples)
    size_t sz = (size_t)d * k + (size_t)d * (k / GS) * sizeof(float); // (generic kernels on GGUF Q8_0: q / s arrays)
    if (t->type == GGML_F32) // (the rows as in the file)
        sz = (size_t)d * k * sizeof(float);
    else if (t->type != GGML_Q8_0) // Q3_K .. Q6_K (GGUF; d % 16 == 0, k % 256 == 0 checked)
        sz = kq_size(t->type, d, k);
#ifdef RUNQ_Q8X
    else if (pack)
        sz = q8x_size(d, k, GS);
#endif
    (void)pack;
    return (sz + 63) / 64 * 64;
}

typedef struct { QuantizedTensor *t; int d, k, bpi; size_t off; long it0; } PrepItem; // a matrix of prepare_weights

typedef struct {
    const PrepItem *job; long nit;
    uint8_t *packed; size_t total, pg_chunk; long npg;
    size_t rows_max; int kmax, gguf, pack, fd;
    long *left;                   // drop: per matrix its items not done yet (the last one drops its page cache), else NULL
    const void *base;             // the file mapping (read_at)
    long next_pg, next_it; int bad; // shared: claim counters; bad: 1 a corrupt scale, 2 no memory, 4 an F32 inf / NaN
} PrepJob;

static void prep_item_done(PrepJob *P, int j) {
    // drop: the page cache of matrix j's rows in the file once its last item is done
    if (!P->left || __atomic_sub_fetch(&P->left[j], 1, __ATOMIC_ACQ_REL) != 0) return;
    const PrepItem *it = &P->job[j];
    const QuantizedTensor *t = it->t;
    if (t->b) {
        drop_cached(P->fd, P->base, t->b, t->type == GGML_F32 ? (size_t)it->d * it->k * sizeof(float)
                                          : t->type != GGML_Q8_0 ? (size_t)it->d * (it->k / QK_K) * kq_block_bytes(t->type)
                                                                 : (size_t)it->d * (it->k / 32) * Q8_0_BLOCK);
    } else {
        drop_cached(P->fd, P->base, t->q, (size_t)it->d * it->k);
        drop_cached(P->fd, P->base, t->s, (size_t)it->d * (it->k / GS) * sizeof(float));
    }
}

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
    if (!qb || !sb || (gguf && !bb)) {
        free(qb);
        free(sb);
        free(bb);
        __atomic_fetch_or(&P->bad, 2, __ATOMIC_RELAXED);
        return; // (the other threads do the work; the load fails anyway)
    }
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
        if (t->f) { // F32: the rows as they are; a value inf / NaN is a corrupt file (as a Q8_0 scale)
            float *dst = t->f + r0 * k;
            read_at(fd, P->base, dst, nr * k * sizeof(float), t->b + r0 * k * sizeof(float));
            for (size_t i = 0; i < nr * k; i++) {
                uint32_t u;
                memcpy(&u, dst + i, 4);
                bad |= ((u & 0x7f800000u) == 0x7f800000u) << 2;
            }
            prep_item_done(P, jj);
            continue;
        }
        if (t->kp) { // Q3_K .. Q6_K: blocks of nr rows (k / 256 each) -> Q3P / Q4P / Q5P / Q6P
            size_t rowb = (size_t)(k / QK_K) * kq_block_bytes(t->type);
            read_at(fd, P->base, bb, nr * rowb, t->b + r0 * rowb);
            for (size_t i = 0; i < nr * (k / QK_K); i++)
                bad |= !kq_block_ok(t->type, bb + i * kq_block_bytes(t->type));
            kq_pack_blocks(t->type, t->kp + (size_t)rb0 * (k / QK_K) * kq_sb_bytes(t->type), bb, nb, k);
            prep_item_done(P, jj);
            continue;
        }
        int8_t *q = pack ? qb : t->q + r0 * k;
        float *s = pack ? sb : t->s + r0 * ng;
        if (t->b) { // GGUF (GS == 32: one block per group)
            read_at(fd, P->base, bb, nr * ng * Q8_0_BLOCK, t->b + r0 * ng * Q8_0_BLOCK);
            bad |= !q8_from_blocks(q, s, bb, nr * ng);
        } else {
            read_at(fd, P->base, q, nr * k, t->q + r0 * k);
            read_at(fd, P->base, s, nr * ng * sizeof(float), t->s + r0 * ng);
        }
#ifdef RUNQ_Q8X
        if (pack)
            q8x_pack_blocks(t->p + (size_t)rb0 * ng * Q8X_GROUP(GS), qb, sb, nb, k, GS);
#endif
        prep_item_done(P, jj);
    }
    free(qb);
    free(sb);
    free(bb);
    if (bad)
        __atomic_fetch_or(&P->bad, bad, __ATOMIC_RELAXED);
}

static int prepare_weights(RqModel *m, int fd, int drop, LoadErr *e, char *warning, size_t wlen) {
    // Converts every matmul weight (7 per layer + the classifier) once at load, into one allocation,
    // in one parallel loop over (matrix, block of rows) items:
    //   VNNI build: repacked into the Q8P layout of kernels_avx512.h (group size GS: 64 for .bin,
    //     32 for GGUF; the same size as the Q8_0 data with fp32 scales)
    //   generic kernels on a GGUF file: its Q8_0 blocks (fp16 scale + 32 int8) into the q / s
    //     arrays the .bin format has (fp16 -> fp32 is exact)
    //   generic kernels on a .bin file: nothing to do, q / s are the mapping
    //   GGUF Q3_K / Q4_K / Q5_K / Q6_K tensors (any build): repacked into the Q3P / Q4P / Q5P / Q6P layouts of kquants.h
    // The source rows are read with pread (read_at), not through the mapping: only the norms and
    // the token embedding rows looked up later (the classifier is shared with them) are ever
    // mapped. The destination is pre-faulted in parallel (MADV_POPULATE_WRITE, 2 MB per call).
    // (On the thread pool of threads.h, like the forward passes: no second set of threads)
    // Was: faults on both sides + one MADV_DONTNEED of the mapped source per tensor, ~90 ms; now
    // ~25 ms (12 threads). drop: the page cache of each matrix is dropped once it is converted (drop_cached).
    // Returns 1 if the weights were converted (0: a .bin on the generic kernels, which run from the mapping)
    TransformerWeights *w = &m->weights;
    PrepMat mt[16];
    int nm = prep_mats(m, mt);
    int gguf = w->wq[0].b != NULL, pack = prep_pack(mt, nm, warning, wlen);
    if (!pack && !gguf)
        return 0;
    int nt = 0, kmax = 0;
    for (int i = 0; i < nm; i++) {
        nt += mt[i].n;
        if (mt[i].n > 0)
            kmax = mt[i].k > kmax ? mt[i].k : kmax;
    }

    // the matrices in order, their place in the packed buffer, and the work items: runs of
    // bpi(k) 16-row blocks, ~64 KB of source rows each
    PrepItem *job = malloc((nt + 1) * sizeof(*job));
    if (!job) load_fail(e, "malloc failed!");
    size_t off = 0;
    long nit = 0;
    int j = 0;
    for (int i = 0; i < nm; i++)
        for (int l = 0; l < mt[i].n; l++, j++) {
            // items of 16-row blocks (the VNNI blocks; the generic conversion uses the same split)
            int bpi = 65536 / (16 * mt[i].k);
            QuantizedTensor *t = mt[i].t + l;
            job[j].t = t;
            job[j].d = mt[i].d;
            job[j].k = mt[i].k;
            job[j].bpi = bpi > 0 ? bpi : 1;
            job[j].off = off;
            job[j].it0 = nit;
            off += prep_bytes(t, mt[i].d, mt[i].k, pack);
            nit += ((mt[i].d + 15) / 16 + job[j].bpi - 1) / job[j].bpi;
        }
    job[nt].it0 = nit;
    size_t total = off;
    w->packed = alloc_big(total);
    if (!w->packed) {
        free(job);
        load_fail(e, "malloc failed!");
    }
    w->packed_size = total;
    for (j = 0; j < nt; j++) {
        QuantizedTensor *t = job[j].t;
        size_t nq = (size_t)job[j].d * job[j].k;
        if (t->type == GGML_F32)
            t->f = (float *)(w->packed + job[j].off);
        else if (t->type != GGML_Q8_0)
            t->kp = w->packed + job[j].off;
#ifdef RUNQ_Q8X
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
    long *left = drop ? malloc(nt * sizeof(long)) : NULL;
    for (j = 0; left && j < nt; j++)
        left[j] = job[j + 1].it0 - job[j].it0;
    PrepJob pj = { job, nit, w->packed, total, pg_chunk, (long)((total + pg_chunk - 1) / pg_chunk), rows_max, kmax,
                   gguf, pack, fd, left, m->data, 0, 0, 0 };
    rq_parallel(prepare_worker, &pj);
    int bad = pj.bad;
    free(job);
    free(left);
    if (bad & 2) load_fail(e, "malloc failed!");
    if (bad & 1) load_fail(e, "Corrupt checkpoint: a weight scale (fp16 d / dmin) is inf or NaN");
    if (bad) load_fail(e, "Corrupt checkpoint: an F32 weight is inf or NaN");
    return 1;
}

// ---------------------------------------------------------------------------- memory estimates

static int seq_checkpoints(const RqModel *m, int n) {
    // the recurrent-state checkpoints a sequence keeps at most (rq_seq_checkpoint_at; n > 0: rq_seq_set_checkpoints(n),
    // e.g. the server's --checkpoints): read only here
    return m->hy.on ? (n > 0 ? n : RQ_SEQ_CK) : 0;
}

static size_t pages(size_t n) { return (n + 4095) / 4096 * 4096; }

static void seq_mem(const RqModel *m, int cap, size_t esz, int ck, RqSeqMem *o) {
    // rq_seq_new's buffers (KV cache of whole attention blocks, tokens, state) and its checkpoints (ck_slot)
    size_t kv_len = ((size_t)cap + ATT_P - 1) / ATT_P * ATT_P;
    o->kv = 2 * pages((size_t)m->n_attn * kv_len * m->kv_dim * esz) + (size_t)cap * sizeof(int);
    o->state = m->hy.on ? pages((size_t)m->hy.n_lin * m->ssm_layer * sizeof(float)) : 0;
    o->checkpoints = seq_checkpoints(m, ck);
    o->total = o->kv + (1 + (size_t)o->checkpoints) * o->state;
}

static size_t ctx_mem(const RqModel *m, int nb, int max_ctx, int nt) {
    // rq_ctx_new's buffers for nb rows and max_ctx positions on nt threads (the large ones, all of them: what
    // the passes touch at most)
    const Config *p = &m->config;
    const Hybrid *hy = &m->hy;
    size_t xs = m->xs, hid = p->hidden_dim, hd = p->head_dim, V = p->vocab_size;
    size_t lw = hy->on ? (size_t)hy->conv_dim + hy->vdim + 2 * hy->nv : 0, qd = (size_t)p->n_heads * m->qh;
    size_t ng = xs / GS + hid / GS;
    size_t row = 4 * (p->dim + xs + qd + lw + 2 * (size_t)m->kv_dim + 2 * hid + hd) + xs + hid + 8 * ng; // floats, int8, scales
    if (m->kq) row += 4 * ng + 4 * (xs / 16 + hid / 16);
    size_t att_nb = ((size_t)max_ctx + ATT_P - 1) / ATT_P, blk = (size_t)(ATT_P + ATT_PSTRIDE(hd)) * sizeof(float);
    size_t n = row + 12 * V + (size_t)nt * 2 * ATT_P * hd * sizeof(float) + (size_t)p->n_heads * att_nb * blk;
    if (nb > 1) {
        n += (size_t)nb * row + (size_t)nt * m->att_rows * m->kv_mul * att_nb * blk;
#ifdef RUNQ_KQX
        size_t kmax = hid > xs ? hid : xs;
        if (m->kq) n += (size_t)nt * ((KQX_SCRATCH(kmax) + 63) / 64 * 64);
#endif
    }
    return n;
}

static void model_mem(RqModel *m, size_t esz, int ck) {
    // the estimates (rq_model_mem) from the shapes and the tensor directory (no tensor data read)
    RqModelMem *o = &m->mem;
    PrepMat mt[16];
    int nm = prep_mats(m, mt), pack = prep_pack(mt, nm, NULL, 0);
    size_t wb = 0;
    if (!pack && !m->is_gguf) // (the generic kernels on a .bin: the weights are the file's mapping)
        wb = m->file_size;
    else
        for (int i = 0; i < nm; i++)
            for (int l = 0; l < mt[i].n; l++)
                wb += prep_bytes(mt[i].t + l, mt[i].d, mt[i].k, pack);
    o->weights = wb + norm_floats(m) * sizeof(float);
    // the tokenizer (vocab, lengths, scores, added; its hash table; the strings: within the metadata's size), the
    // file's metadata (GGUF: mapped and read by the tokenizer; .bin: the .tokenizer file, about), a compute
    // context, the process (code, thread stacks, heap: measured ~8 MiB)
    size_t V = m->config.vocab_size, cap = 1;
    while (cap < 2 * V) cap <<= 1;
    size_t meta = m->is_gguf ? m->gguf.data_off : 16 * V;
    o->runtime = V * (sizeof(char *) + 3 * sizeof(int)) + cap * 8 + 2 * meta +
                 ctx_mem(m, RQ_BATCH_MAX, m->config.seq_len, rq_threads()) + ((size_t)8 << 20);
    seq_mem(m, m->config.seq_len, esz, ck, &o->seq);
    o->total = o->weights + o->runtime + o->seq.total;
}

double rq_mem_available(void) {
#ifdef __linux__
    FILE *f = fopen("/proc/meminfo", "r");
    char line[256];
    long long kb = -1;
    while (f && fgets(line, sizeof(line), f))
        if (sscanf(line, "MemAvailable: %lld kB", &kb) == 1) break;
    if (f) fclose(f);
    return kb >= 0 ? kb * 1024.0 : -1;
#else
    return -1;
#endif
}

#define GIB(x) ((double)(x) / 1073741824.0)

static void mem_check(RqModel *m, const RqModelOpts *o, const char *path, LoadErr *e) {
    // the estimate against the budget (RqModelOpts.mem), before any weight is read: a refusal if it does not fit
    RqModelMem *mm = &m->mem;
    model_mem(m, o->kv_bytes == 4 ? 4 : 2, o->checkpoints);
    double avail = o->mem < 0 ? -1 : o->mem > 0 ? o->mem * 1073741824.0 : rq_mem_available();
    if (avail < 0) return;
    mm->budget = avail;
    mm->budget_src = o->mem > 0 ? 2 : 1;
    char what[96];
    snprintf(what, sizeof(what), o->mem > 0 ? "the budget of %.2f GiB (--mem)" : "MemAvailable (%.2f GiB)", GIB(avail));
    char seq[128] = "";
    if (mm->seq.state)
        snprintf(seq, sizeof(seq), " + recurrent state %.0f MiB x (1 + %d checkpoints)", mm->seq.state / 1048576.0, mm->seq.checkpoints);
    if ((double)mm->total > avail)
        load_fail(e, "%s: needs ~%.2f GiB (weights %.2f, runtime %.2f, a sequence of %d positions: KV cache %.2f%s), more "
                  "than %s: not loaded (a smaller -c lowers it; --mem 0 loads anyway)", path, GIB(mm->total), GIB(mm->weights),
                  GIB(mm->runtime), m->config.seq_len, GIB(mm->seq.kv), seq, what);
}

static void mem_warn(RqModel *m) {
    // a warning past 3/4 of the budget (after prepare_weights, which may set one too)
    const RqModelMem *mm = &m->mem;
    if (mm->budget > 0 && (double)mm->total > 0.75 * mm->budget) {
        size_t l = strlen(m->warning);
        snprintf(m->warning + l, sizeof(m->warning) - l, "%smemory: needs ~%.2f GiB, %.0f%% of %s %.2f GiB", l ? "; " : "",
                 GIB(mm->total), 100.0 * mm->total / mm->budget, mm->budget_src == 2 ? "--mem" : "MemAvailable", GIB(mm->budget));
    }
}

int rq_model_mem(const RqModel *m, RqModelMem *mem) {
    memset(mem, 0, sizeof(*mem));
    if (!m->has_weights) return RQ_ERR_STATE;
    *mem = m->mem;
    return RQ_OK;
}

int rq_seq_mem(const RqModel *m, int capacity, int kv_type, RqSeqMem *mem) {
    memset(mem, 0, sizeof(*mem));
    if (!m->has_weights) return RQ_ERR_STATE;
    if (capacity <= 0) capacity = m->config.seq_len;
    if (capacity > m->config.seq_len || kv_type < KV_F32 || kv_type > KV_F16) return RQ_ERR_ARG;
    seq_mem(m, capacity, kv_type == KV_F32 ? 4 : 2, 0, mem);
    return RQ_OK;
}

static int drop_page_cache(const RqModel *m, const RqModelOpts *o) {
    // RqModelOpts.page_cache: 0 = if the file is >= 1/4 of the RAM (a big model: its cached file would take as
    // much RAM again as the repacked copy), 1 = always, -1 = never
    if (o->page_cache) return o->page_cache > 0;
#ifndef _WIN32
    long pg = sysconf(_SC_PHYS_PAGES), psz = sysconf(_SC_PAGESIZE);
    return pg > 0 && psz > 0 && (double)m->file_size >= 0.25 * (double)pg * psz;
#else
    (void)m;
    return 0;
#endif
}

// ---------------------------------------------------------------------------- chat templates

// Qwen3's chat template (tokenizer_config.json) rendered for one user message "%s", with and without
// a system message "%s" and enable_thinking: the strings export.py writes into the .template* files
// (byte-identical; make test compares the chat token ids of a .bin and a GGUF run). Used for GGUF
// files, whose tokenizer.chat_template is Jinja (not interpreted here)
#define QWEN3_TMPL_SYS "<|im_start|>system\n%s<|im_end|>\n"
#define QWEN3_TMPL_USER "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"
#define QWEN3_TMPL_NOTHINK "<think>\n\n</think>\n\n"
#define QWEN3_TMPL_THINK "<think>\n" // Thinking-2507's generation prompt opens the reasoning block

const char *rq_chat_name(int v) {
    return v == RQ_CHAT_INSTRUCT ? "instruct-2507" : v == RQ_CHAT_THINKING ? "thinking-2507" : v == RQ_CHAT_QWEN35 ? "qwen3.5"
         : v == RQ_CHAT_QWEN38 ? "qwen3.8" : "qwen3";
}

const char *chat_gen_suffix(const RqModel *m, int thinking) {
    // what the generation prompt adds after "<|im_start|>assistant\n" (chat.c, load_templates)
    if (m->chat == RQ_CHAT_INSTRUCT) return "";
    if (m->chat == RQ_CHAT_THINKING || m->chat == RQ_CHAT_QWEN35 || m->chat == RQ_CHAT_QWEN38)
        return thinking ? QWEN3_TMPL_THINK : QWEN3_TMPL_NOTHINK;
    return thinking ? "" : QWEN3_TMPL_NOTHINK;
}

static int read_template(const char *path, const char *suffix, char *out, char *missing) {
    // <path><suffix> (at most TMPL_MAX - 1 bytes); 0 and its name in missing if it cannot be read
    char p[TMPL_MAX + 64];
    snprintf(p, sizeof(p), "%s%s", path, suffix);
    memset(out, 0, TMPL_MAX);
    FILE *f = fopen(p, "rb");
    if (!f) {
        snprintf(missing, TMPL_MAX + 64, "%s", p);
        return 0;
    }
    size_t n = fread(out, 1, TMPL_MAX - 1, f);
    (void)n;
    fclose(f);
    return 1;
}

static void load_templates(RqModel *m, const char *path) {
    // per thinking mode: the user turn template and the system block (the system template is the
    // system block + the user turn: its start, if it ends with the user template)
    for (int t = 0; t < 2; t++) {
        m->tmpl_missing[t][0] = 0;
        if (m->is_gguf) { // (the variant's generation prompt: chat_gen_suffix)
            snprintf(m->user_tmpl[t], TMPL_MAX, "%s%s", QWEN3_TMPL_USER, chat_gen_suffix(m, t));
            strcpy(m->sys_tmpl[t], QWEN3_TMPL_SYS);
            continue;
        }
        char sys[TMPL_MAX];
        if (!read_template(path, t ? ".template.with-thinking" : ".template", m->user_tmpl[t], m->tmpl_missing[t]))
            continue;
        if (!read_template(path, t ? ".template.with-system-and-thinking" : ".template.with-system", sys, m->tmpl_missing[t]))
            continue;
        strcpy(m->sys_tmpl[t], QWEN3_TMPL_SYS);
        size_t ls = strlen(sys), lp = strlen(m->user_tmpl[t]);
        if (ls > lp && !strcmp(sys + ls - lp, m->user_tmpl[t])) {
            memcpy(m->sys_tmpl[t], sys, ls - lp);
            m->sys_tmpl[t][ls - lp] = 0;
        }
    }
}

int rq_chat_check(const RqModel *m, int thinking, char *err, size_t errlen) {
    thinking = thinking != 0;
    if (!m->has_tok) {
        snprintf(err, errlen, "model loaded without a tokenizer");
        return RQ_ERR_STATE;
    }
    if (m->tmpl_missing[thinking][0]) {
        snprintf(err, errlen, "%s", m->tmpl_missing[thinking]);
        return RQ_ERR_LOAD;
    }
    return RQ_OK;
}

// ---------------------------------------------------------------------------- loading

static int file_is_gguf(const char *path) {
    // the checkpoint format by magic: GGUF, else a qwen3.c .bin
    FILE *f = fopen(path, "rb");
    uint32_t magic = 0;
    if (f && fread(&magic, 4, 1, f) != 1) magic = 0;
    if (f) fclose(f);
    return magic == GGUF_MAGIC;
}

static int64_t file_size_of(FILE *f) {
#if defined _WIN32
    _fseeki64(f, 0, SEEK_END);
    return _ftelli64(f);
#else
    fseek(f, 0, SEEK_END);
    return ftell(f);
#endif
}

static void unmap_file(RqModel *m) {
    if (m->data) munmap(m->data, m->file_size);
    m->data = NULL;
    m->gguf.map = NULL;
}

typedef struct { LoadErr e; FILE *file; } Loader; // (the open file is closed after a failure too)

static void load_tokenizer_only(RqModel *m, const char *path, Loader *ld) {
    // -m tokenize: the header / metadata and the tokenizer, nothing else mapped or read
    LoadErr *e = &ld->e;
    if (file_is_gguf(path)) {
        FILE *f = ld->file = fopen(path, "rb");
        if (!f) load_fail(e, "Couldn't open %s", path);
        int64_t size = file_size_of(f);
        if (size < 24) load_fail(e, "GGUF %s: file too small", path);
        void *map = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fileno(f), 0);
        fclose(f);
        ld->file = NULL;
        if (map == MAP_FAILED) load_fail(e, "mmap failed!");
        m->data = map;
        m->file_size = (size_t)size;
        m->is_gguf = 1;
        m->gguf.map = map;
        m->gguf.size = (uint64_t)size;
        m->gguf.path = path;
        m->gguf.err = e;
        gguf_parse(&m->gguf);
        gguf_config(&m->gguf, &m->config, &m->rope_theta, &m->rope_dim, &m->hy);
        m->chat = gguf_chat_variant(&m->gguf);
        m->samp_set = gguf_sampling(&m->gguf, &m->samp_temp, &m->samp_topp, &m->samp_topk);
        build_tokenizer_gguf(&m->tok, &m->gguf, m->config.vocab_size);
        m->has_tok = 1;
        unmap_file(m); // (the tokenizer copied what it needs)
        free(m->gguf.kv);
        free(m->gguf.t);
        m->gguf.kv = NULL;
        m->gguf.t = NULL;
    } else {
        FILE *f = ld->file = fopen(path, "rb");
        if (!f || fread(&m->config, sizeof(Config), 1, f) != 1 || m->config.magic_number != BIN_MAGIC)
            load_fail(e, "Couldn't read qwen3.c checkpoint header from %s", path);
        fclose(f);
        ld->file = NULL;
        if (m->config.vocab_size < 256 || m->config.vocab_size > (1 << 24))
            load_fail(e, "Couldn't read qwen3.c checkpoint header from %s", path);
        build_tokenizer(&m->tok, path, m->config.vocab_size, e);
        m->has_tok = 1;
    }
}

static void load_model(RqModel *m, const char *path, const RqModelOpts *o, Loader *ld, char *warning, size_t wlen) {
    LoadErr *e = &ld->e;
    FILE *file = ld->file = fopen(path, "rb");
    if (!file) load_fail(e, "Couldn't open checkpoint %s", path);
    int64_t size = file_size_of(file);
    if (size < 256) load_fail(e, "File %s is too small for a checkpoint", path);
    void *map = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fileno(file), 0);
    if (map == MAP_FAILED) load_fail(e, "mmap failed!");
    m->data = map;
    m->file_size = (size_t)size;
    Config *c = &m->config;
    uint32_t magic;
    memcpy(&magic, map, 4);
    if (magic == GGUF_MAGIC) {
        m->is_gguf = 1;
        m->gguf.map = map;
        m->gguf.size = (uint64_t)size;
        m->gguf.path = path;
        m->gguf.err = e;
        gguf_parse(&m->gguf);
        gguf_config(&m->gguf, c, &m->rope_theta, &m->rope_dim, &m->hy);
        m->chat = gguf_chat_variant(&m->gguf);
        m->samp_set = gguf_sampling(&m->gguf, &m->samp_temp, &m->samp_topp, &m->samp_topk);
        // (an MTP block: not in c->n_layers, gguf_config; validated by gguf_map_weights, gguf_map_mtp)
    } else {
        // checkpoint format is 256-byte header, and then the model weights
        memcpy(c, map, sizeof(Config));
        if (c->magic_number != BIN_MAGIC) load_fail(e, "File %s is not a qwen3.c checkpoint or a GGUF file", path);
        if (c->version != 1) load_fail(e, "Checkpoint %s is version %d, need version 1", path, c->version);
        if (c->dim < 1 || c->hidden_dim < 1 || c->n_layers < 1 || c->n_heads < 1 || c->n_kv_heads < 1 || c->vocab_size < 256 ||
            c->seq_len < 1 || c->head_dim < 1 || c->group_size < 1 || c->dim > (1 << 20) || c->hidden_dim > (1 << 20) ||
            c->n_layers > 4096 || c->n_heads > 4096 || c->head_dim > 4096 || c->vocab_size > (1 << 24) || c->group_size > 4096)
            load_fail(e, "Checkpoint %s has an invalid header", path);
        m->rope_dim = c->head_dim;
    }
    model_layers(m, e);
    if (gs_models > 0 && GS != c->group_size)
        load_fail(e, "%s: group size %d, but a model with group size %d is loaded (one group size per process)", path,
                  c->group_size, GS);
    GS = c->group_size; // (process-wide: the kernels read it)
    if (o->ctx_len < 0) load_fail(e, "bad context length %d", o->ctx_len);
    m->ctx_max = c->seq_len;
    if (o->ctx_len != 0 && o->ctx_len <= c->seq_len)
        c->seq_len = o->ctx_len;
    else if (o->ctx_len == 0 && c->seq_len > RQ_CTX_DEFAULT_MAX) // (Qwen3: 40,960 = the cap; 2507: 262,144)
        c->seq_len = RQ_CTX_DEFAULT_MAX;
    if (!forward_model_init(m))
        load_fail(e, "unsupported model shape (heads %d / %d, head_dim %d, dim %d, hidden %d, GS %d%s)", c->n_heads,
                  c->n_kv_heads, c->head_dim, c->dim, c->hidden_dim, GS, m->hy.on ? ", linear attention" : "");
    if (m->is_gguf)
        gguf_map_weights(m);
    else
        bin_map_weights(m, path, e);
    mem_check(m, o, path, e); // (nothing of the tensor data read yet)
    if (m->is_gguf)
        gguf_copy_norms(m);
    int drop = drop_page_cache(m, o);
    if (prepare_weights(m, fileno(file), drop, e, warning, wlen) && drop)
        drop_cached(fileno(file), map, map, 0); // (the rest: partial pages, readahead; mapped pages stay)
    mem_warn(m);
    m->has_weights = 1;
    fclose(file);
    ld->file = NULL;
    if (!o->no_tokenizer) {
        if (m->is_gguf)
            build_tokenizer_gguf(&m->tok, &m->gguf, c->vocab_size);
        else
            build_tokenizer(&m->tok, path, c->vocab_size, e);
        m->has_tok = 1;
    }
}

static int load_guarded(RqModel *m0, Loader *ld0, const char *path0, const RqModelOpts *o0) {
    // the load with its error exit: 0 after a load_fail / gguf_fail (only volatile locals are live
    // across the setjmp)
    RqModel *volatile m = m0;
    Loader *volatile ld = ld0;
    const char *volatile path = path0;
    const RqModelOpts *volatile o = o0;
    if (setjmp(ld->e.jb))
        return 0;
    if (o->tokenizer_only) {
        load_tokenizer_only(m, path, ld);
    } else {
        load_model(m, path, o, ld, m->warning, sizeof(m->warning));
        gs_models++;
        m->gs_counted = 1;
    }
    if (m->has_tok)
        load_templates(m, path);
    return 1;
}

RqModel *rq_model_load(const char *path, const RqModelOpts *opts, char *err, size_t errlen) {
    static const RqModelOpts defaults = { 0, 0, 0, 0, 0, 0 };
    char dummy[8];
    if (!err || errlen == 0) {
        err = dummy;
        errlen = sizeof(dummy);
    }
    err[0] = 0;
    RqModel *m = calloc(1, sizeof(RqModel));
    Loader *ld = calloc(1, sizeof(Loader));
    if (!m || !ld) {
        free(m);
        free(ld);
        snprintf(err, errlen, "malloc failed!");
        return NULL;
    }
    m->rope_theta = 1e6f; // (.bin files: Qwen3's; GGUF: gguf_config)
    m->mtp.layer = -1;
    ld->e.msg = err;
    ld->e.len = errlen;
    int ok = load_guarded(m, ld, path, opts ? opts : &defaults);
    if (ld->file) fclose(ld->file); // (open only after a failure)
    free(ld);
    if (!ok) {
        rq_model_free(m); // (frees what exists: everything is reachable from m, NULL otherwise)
        return NULL;
    }
    m->gguf.err = NULL;
    m->rope_log_theta = (float)log((double)m->rope_theta);
    if (!m->ctx_max) m->ctx_max = m->config.seq_len; // (tokenizer only)
    return m;
}

void rq_model_free(RqModel *m) {
    if (!m) return;
    TransformerWeights *w = &m->weights;
    if (m->gs_counted) gs_models--;
    {
        free(w->q_tokens);
        free(w->wq);
        free(w->wk);
        free(w->wv);
        free(w->wo);
        free(w->w1);
        free(w->w2);
        free(w->w3);
        free(w->wqkv);
        free(w->wz);
        free(w->wa);
        free(w->wb);
        free(w->wso);
        if (w->wcls != w->q_tokens) free(w->wcls);
        free_big(w->packed, w->packed_size);
        free(w->norms);
    }
    free(m->gguf.kv);
    free(m->gguf.t);
    free(m->kv_layer);
    free(m->lin_layer);
    free_tokenizer(&m->tok);
    unmap_file(m);
    free(m);
}

void rq_model_info(const RqModel *m, RqModelInfo *info) {
    const Config *c = &m->config;
    info->dim = c->dim;
    info->hidden_dim = c->hidden_dim;
    info->n_layers = c->n_layers;
    info->n_heads = c->n_heads;
    info->n_kv_heads = c->n_kv_heads;
    info->head_dim = c->head_dim;
    info->vocab_size = c->vocab_size;
    info->seq_len = c->seq_len;
    info->group_size = c->group_size;
    info->shared_classifier = c->shared_classifier;
    info->gguf = m->is_gguf;
    info->bos = m->has_tok ? (int)m->tok.bos_token_id : -1;
    info->eos = m->has_tok ? (int)m->tok.eos_token_id : -1;
    info->warning = m->warning[0] ? m->warning : NULL;
    info->ctx_max = m->ctx_max;
    info->chat = m->chat;
    info->rope_theta = m->rope_theta;
    info->linear_layers = m->hy.n_lin;
}

// ---------------------------------------------------------------------------- test hook

int rq_dequant_rows(const RqModel *m, const char *name, size_t nlen, long r0, long nr, float **rows, float **packed,
                    int *ncols, int *npacked, char *err, size_t errlen) {
    // rows of the matrix `name` dequantized from the file's blocks (embed_row: ggml's expressions) and,
    // for the K-quants, also from the repacked weights (kq_packed_row, the same expressions: checks the
    // repacking)
    const Config *p = &m->config;
    const TransformerWeights *w = &m->weights;
    *rows = NULL;
    if (packed) *packed = NULL;
    if (!m->has_weights) { snprintf(err, errlen, "model loaded without weights"); return RQ_ERR_STATE; }
    if (!m->is_gguf) { snprintf(err, errlen, "GGUF files only"); return RQ_ERR_STATE; }
    const Hybrid *hy = &m->hy;
    int L = p->n_layers, dim = p->dim, hid = p->hidden_dim, qd = p->n_heads * p->head_dim, kvd = p->n_kv_heads * p->head_dim;
    // (n: layers, 1 for a plain name; sel: the layer's index into t, NULL l itself; -1 none)
    struct { const char *fmt; QuantizedTensor *t; int n; const int *sel; int d, k; } mt[] = {
        { "token_embd.weight", w->q_tokens, 1, NULL, p->vocab_size, dim }, { "output.weight", w->wcls, 1, NULL, p->vocab_size, dim },
        { "blk.%d.attn_q.weight", w->wq, L, m->kv_layer, p->n_heads * m->qh, dim }, { "blk.%d.attn_k.weight", w->wk, L, m->kv_layer, kvd, dim },
        { "blk.%d.attn_v.weight", w->wv, L, m->kv_layer, kvd, dim }, { "blk.%d.attn_output.weight", w->wo, L, m->kv_layer, dim, qd },
        { "blk.%d.ffn_gate.weight", w->w1, L, NULL, hid, dim }, { "blk.%d.ffn_down.weight", w->w2, L, NULL, dim, hid },
        { "blk.%d.ffn_up.weight", w->w3, L, NULL, hid, dim },
        { "blk.%d.attn_qkv.weight", w->wqkv, hy->n_lin ? L : 0, m->lin_layer, hy->conv_dim, dim },
        { "blk.%d.attn_gate.weight", w->wz, hy->n_lin ? L : 0, m->lin_layer, hy->vdim, dim },
        { "blk.%d.ssm_alpha.weight", w->wa, hy->n_lin ? L : 0, m->lin_layer, hy->nv, dim },
        { "blk.%d.ssm_beta.weight", w->wb, hy->n_lin ? L : 0, m->lin_layer, hy->nv, dim },
        { "blk.%d.ssm_out.weight", w->wso, hy->n_lin ? L : 0, m->lin_layer, dim, hy->vdim } };
    const QuantizedTensor *t = NULL;
    int d = 0, k = 0;
    for (size_t i = 0; i < sizeof(mt) / sizeof(mt[0]) && !t; i++) {
        if (i == 1 && p->shared_classifier) continue; // (no output.weight: tied)
        for (int l = 0; l < mt[i].n && !t; l++) {
            int li = mt[i].sel ? mt[i].sel[l] : l;
            if (li < 0) continue;
            char nm[96];
            snprintf(nm, sizeof(nm), mt[i].fmt, l);
            if (strlen(nm) == nlen && !memcmp(nm, name, nlen)) { t = mt[i].t + li; d = mt[i].d; k = mt[i].k; }
        }
    }
    if (!t || !t->b) { snprintf(err, errlen, "no matrix %.*s", (int)nlen, name); return RQ_ERR_ARG; }
    if (r0 < 0 || nr < 1 || r0 > d - nr) { snprintf(err, errlen, "rows %ld..%ld out of range (%d rows)", r0, r0 + nr - 1, d); return RQ_ERR_ARG; }
    float *a = malloc((size_t)nr * k * sizeof(float)), *b = t->kp && packed ? malloc((size_t)nr * k * sizeof(float)) : NULL;
    if (!a || (t->kp && packed && !b)) {
        free(a);
        free(b);
        snprintf(err, errlen, "malloc failed!");
        return RQ_ERR_NOMEM;
    }
    for (long r = 0; r < nr; r++) {
        embed_row(a + (size_t)r * k, t, (int)(r0 + r), k);
        if (b)
            kq_packed_row(t->type, t->kp, (int)(r0 + r), k, b + (size_t)r * k);
    }
    *rows = a;
    if (packed) *packed = b;
    *ncols = k;
    *npacked = b != NULL;
    return RQ_OK;
}
