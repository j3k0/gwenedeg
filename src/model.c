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

static void gguf_matrices(RqModel *m, QuantizedTensor **dst, int L, const int *sel, const char *fmt, int d, int k) {
    // the matrices (d, k) of layers 0..L-1 (fmt: name with the layer number, or a plain name for L = 1), those
    // with sel[l] >= 0 into (*dst)[sel[l]] (sel NULL: all, into (*dst)[l]), each Q8_0, Q4_K or Q6_K (dispatch
    // per tensor: Q4_K_M mixes them within a role): their blocks in the mapping; q / s / p / kp are set by
    // prepare_weights
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
        const GgufTensor *t = gguf_tensor(g, name);
        uint32_t type = t ? t->type : GGML_Q8_0; // (missing: gguf_weight reports it)
        if (type != GGML_Q8_0 && type != GGML_Q4_K && type != GGML_Q5_K && type != GGML_Q6_K) {
            char msg[160];
            snprintf(msg, sizeof(msg), "tensor %%s has ggml type %u, need Q8_0 (8), Q4_K (12), Q5_K (13) or Q6_K (14)", type);
            gguf_fail(g, msg, name);
        }
        res[l].b = gguf_weight(g, name, type, k, d);
        res[l].type = (int)type;
        if (type != GGML_Q8_0) {
            // the repacked layouts: 16-row blocks, 256-column super-blocks (activation groups of 32)
            if (d % KQ_ROWS || k % QK_K || GS != 32) gguf_fail(g, "tensor %s: K-quant shape not supported (rows % 16, columns % 256)", name);
            m->kq = 1;
        }
    }
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

static void gguf_map_weights(RqModel *m) {
    // tensor names of llama.cpp's convert_hf_to_gguf.py for qwen3 / qwen35 (no Q/K row permutation: that
    // is done for the llama architecture only; checked by the logits test against HF). ffn_gate /
    // ffn_up / ffn_down are w1 / w3 / w2 (qwen35: the FFN norm is post_attention_norm). The F32 tensors
    // (norms; qwen35's conv kernel, transposed to tap-major, and ssm_a / ssm_dt / ssm_norm) are copied into one
    // buffer, attention ones per attention layer, the others per linear layer
    Config *p = &m->config;
    TransformerWeights *w = &m->weights;
    const Gguf *g = &m->gguf;
    const Hybrid *hy = &m->hy;
    int L = p->n_layers, dim = p->dim, hd = p->head_dim, qd = p->n_heads * hd, kvd = p->n_kv_heads * hd, NA = m->n_attn;
    int NL = hy->n_lin;
    size_t nn = (size_t)2 * L * dim + dim + (size_t)2 * NA * hd + (size_t)NL * (hy->conv * hy->conv_dim + 2 * hy->nv + hy->dv);
    float *f = w->norms = malloc(nn * sizeof(float));
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
    if (NL) { // the conv kernels: (conv_dim, conv) in the file, (conv, conv_dim) here
        w->ssm_conv = f;
        for (int l = 0; l < L; l++) {
            if (m->lin_layer[l] < 0) continue;
            char name[96];
            snprintf(name, sizeof(name), "blk.%d.ssm_conv1d.weight", l);
            const float *src = (const float *)gguf_weight(g, name, GGML_F32, hy->conv, hy->conv_dim);
            for (int c = 0; c < hy->conv_dim; c++)
                for (int j = 0; j < hy->conv; j++)
                    memcpy(f + (size_t)j * hy->conv_dim + c, src + (size_t)c * hy->conv + j, sizeof(float));
            f += (size_t)hy->conv * hy->conv_dim;
        }
    }
    gguf_matrices(m, &w->q_tokens, 1, NULL, "token_embd.weight", p->vocab_size, dim);
    gguf_matrices(m, &w->wq, L, m->kv_layer, "blk.%d.attn_q.weight", p->n_heads * m->qh, dim);
    gguf_matrices(m, &w->wk, L, m->kv_layer, "blk.%d.attn_k.weight", kvd, dim);
    gguf_matrices(m, &w->wv, L, m->kv_layer, "blk.%d.attn_v.weight", kvd, dim);
    gguf_matrices(m, &w->wo, L, m->kv_layer, "blk.%d.attn_output.weight", dim, qd);
    gguf_matrices(m, &w->w1, L, NULL, "blk.%d.ffn_gate.weight", p->hidden_dim, dim);
    gguf_matrices(m, &w->w2, L, NULL, "blk.%d.ffn_down.weight", dim, p->hidden_dim);
    gguf_matrices(m, &w->w3, L, NULL, "blk.%d.ffn_up.weight", p->hidden_dim, dim);
    if (NL) {
        gguf_matrices(m, &w->wqkv, L, m->lin_layer, "blk.%d.attn_qkv.weight", hy->conv_dim, dim);
        gguf_matrices(m, &w->wz, L, m->lin_layer, "blk.%d.attn_gate.weight", hy->vdim, dim);
        gguf_matrices(m, &w->wa, L, m->lin_layer, "blk.%d.ssm_alpha.weight", hy->nv, dim);
        gguf_matrices(m, &w->wb, L, m->lin_layer, "blk.%d.ssm_beta.weight", hy->nv, dim);
        gguf_matrices(m, &w->wso, L, m->lin_layer, "blk.%d.ssm_out.weight", dim, hy->vdim);
    }
    if (p->shared_classifier)
        w->wcls = w->q_tokens;
    else
        gguf_matrices(m, &w->wcls, 1, NULL, "output.weight", p->vocab_size, dim);
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

typedef struct { QuantizedTensor *t; int d, k, bpi; size_t off; long it0; } PrepItem; // a matrix of prepare_weights

typedef struct {
    const PrepItem *job; long nit;
    uint8_t *packed; size_t total, pg_chunk; long npg;
    size_t rows_max; int kmax, gguf, pack, fd;
    const void *base;             // the file mapping (read_at)
    long next_pg, next_it; int bad; // shared: claim counters; bad: 1 a corrupt scale, 2 no memory
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
        if (t->kp) { // Q4_K / Q5_K / Q6_K: blocks of nr rows (k / 256 each) -> Q4P / Q5P / Q6P
            size_t rowb = (size_t)(k / QK_K) * kq_block_bytes(t->type);
            read_at(fd, P->base, bb, nr * rowb, t->b + r0 * rowb);
            for (size_t i = 0; i < nr * (k / QK_K); i++)
                bad |= !kq_block_ok(t->type, bb + i * kq_block_bytes(t->type));
            kq_pack_blocks(t->type, t->kp + (size_t)rb0 * (k / QK_K) * kq_sb_bytes(t->type), bb, nb, k);
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
    }
    free(qb);
    free(sb);
    free(bb);
    if (bad)
        __atomic_fetch_or(&P->bad, 1, __ATOMIC_RELAXED);
}

static void prepare_weights(RqModel *m, int fd, LoadErr *e, char *warning, size_t wlen) {
    // Converts every matmul weight (7 per layer + the classifier) once at load, into one allocation,
    // in one parallel loop over (matrix, block of rows) items:
    //   VNNI build: repacked into the Q8P layout of kernels_avx512.h (group size GS: 64 for .bin,
    //     32 for GGUF; the same size as the Q8_0 data with fp32 scales)
    //   generic kernels on a GGUF file: its Q8_0 blocks (fp16 scale + 32 int8) into the q / s
    //     arrays the .bin format has (fp16 -> fp32 is exact)
    //   generic kernels on a .bin file: nothing to do, q / s are the mapping
    //   GGUF Q4_K / Q5_K / Q6_K tensors (any build): repacked into the Q4P / Q5P / Q6P layouts of kquants.h
    // The source rows are read with pread (read_at), not through the mapping: only the norms and
    // the token embedding rows looked up later (the classifier is shared with them) are ever
    // mapped. The destination is pre-faulted in parallel (MADV_POPULATE_WRITE, 2 MB per call).
    // (On the thread pool of threads.h, like the forward passes: no second set of threads)
    // Was: faults on both sides + one MADV_DONTNEED of the mapped source per tensor, ~90 ms; now
    // ~25 ms (12 threads)
    Config *p = &m->config;
    TransformerWeights *w = &m->weights;
    const Hybrid *hy = &m->hy;
    int L = p->n_layers, dim = p->dim, hid = p->hidden_dim, NA = m->n_attn, NL = hy->n_lin;
    int qd = p->n_heads * p->head_dim, kvd = p->n_kv_heads * p->head_dim;
    struct { QuantizedTensor *t; int n, d, k; } mt[] = {
        { w->wq, NA, p->n_heads * m->qh, dim }, { w->wk, NA, kvd, dim }, { w->wv, NA, kvd, dim }, { w->wo, NA, dim, qd },
        { w->w1, L, hid, dim }, { w->w2, L, dim, hid }, { w->w3, L, hid, dim }, { w->wcls, 1, p->vocab_size, dim },
        { w->wqkv, NL, hy->conv_dim, dim }, { w->wz, NL, hy->vdim, dim }, { w->wa, NL, hy->nv, dim },
        { w->wb, NL, hy->nv, dim }, { w->wso, NL, dim, hy->vdim } };
    int nm = sizeof(mt) / sizeof(mt[0]);
    int gguf = w->wq[0].b != NULL, pack = 0;
    (void)warning; (void)wlen;
#ifdef RUNQ_Q8X
    pack = 1;
    for (int i = 0; i < nm && pack; i++)
        if (mt[i].n > 0 && !q8x_ok(mt[i].d, mt[i].k, GS)) {
            snprintf(warning, wlen, "weights not packable for the SIMD Q8_0 kernels (GS %d, %d x %d), using the generic ones", GS, mt[i].d, mt[i].k);
            pack = 0;
        }
#endif
    if (!pack && !gguf)
        return;
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
            size_t nq = (size_t)mt[i].d * mt[i].k, sz = nq + (size_t)mt[i].d * (mt[i].k / GS) * sizeof(float);
            job[j].t = t;
            job[j].d = mt[i].d;
            job[j].k = mt[i].k;
            job[j].bpi = bpi > 0 ? bpi : 1;
            job[j].off = off;
            job[j].it0 = nit;
            if (t->type != GGML_Q8_0) // Q4_K / Q5_K / Q6_K (GGUF; d % 16 == 0, k % 256 == 0 checked)
                sz = kq_size(t->type, mt[i].d, mt[i].k);
#ifdef RUNQ_Q8X
            else if (pack)
                sz = q8x_size(mt[i].d, mt[i].k, GS);
#endif
            off += (sz + 63) / 64 * 64;
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
        if (t->type != GGML_Q8_0)
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
    PrepJob pj = { job, nit, w->packed, total, pg_chunk, (long)((total + pg_chunk - 1) / pg_chunk), rows_max, kmax,
                   gguf, pack, fd, m->data, 0, 0, 0 };
    rq_parallel(prepare_worker, &pj);
    int bad = pj.bad;
    free(job);
    if (bad & 2) load_fail(e, "malloc failed!");
    if (bad) load_fail(e, "Corrupt checkpoint: a weight scale (fp16 d / dmin) is inf or NaN");
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
                                                                                                                 : "qwen3";
}

const char *chat_gen_suffix(const RqModel *m, int thinking) {
    // what the generation prompt adds after "<|im_start|>assistant\n" (chat.c, load_templates)
    if (m->chat == RQ_CHAT_INSTRUCT) return "";
    if (m->chat == RQ_CHAT_THINKING || m->chat == RQ_CHAT_QWEN35) return thinking ? QWEN3_TMPL_THINK : QWEN3_TMPL_NOTHINK;
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
    prepare_weights(m, fileno(file), e, warning, wlen);
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
    static const RqModelOpts defaults = { 0, 0, 0 };
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
    // for Q4_K / Q6_K, also from the repacked weights (kq_packed_row, the same expressions: checks the
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
