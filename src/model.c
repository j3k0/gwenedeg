/* Model loading. */
#include "internal.h"

int GS = 0;
int KQ = 0;

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
#ifdef RUNQ_Q8X
    pack = 1;
    for (int i = 0; i < nm && pack; i++)
        if (!q8x_ok(m[i].d, m[i].k, GS)) {
            fprintf(stderr, "warning: weights not packable for the SIMD Q8_0 kernels (GS %d, %d x %d), using the generic ones\n", GS, m[i].d, m[i].k);
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
#ifdef RUNQ_Q8X
            else if (pack)
                sz = q8x_size(m[i].d, m[i].k, GS);
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

