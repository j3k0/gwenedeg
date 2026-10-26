/* The forward passes: sequences (RqSeq: KV cache, position, tokens), compute contexts (RqCtx:
 * activation and scratch buffers, synchronization, profiling) and every numeric part of the decode
 * step and the batched prefill (quantization, norms, matmul and attention dispatch, KV cache).
 * All of it is one translation unit: the exact arithmetic (dot_q8, the attention, the quantizer)
 * depends on gcc -Ofast's code generation in this context (inlining; TODO.md Steps 3-7). */
#include "internal.h"

#ifndef ATT_ROWS
#define ATT_ROWS 6 // query rows per prefill attention work item (x kv_mul query heads)
#endif
#define ATT_NV_MAX 64 // max query vectors per attention_kv call

#ifdef RUNQ_PROFILE
// build with -DRUNQ_PROFILE: wall time per batched-prefill phase type (rq_ctx_profile)
enum { PF_NORM, PF_QKV, PF_ROPE, PF_ATT, PF_WO, PF_FFN13, PF_SWIGLU, PF_FFN2, PF_CLS, PF_BDQKV, PF_BDFFN13, PF_N };
static const char *pf_name[PF_N] = { "norm+quant+resid", "gemm qkv", "qk-norm+rope+kv", "attention+quant",
                                     "gemm wo", "gemm w1+w3", "swiglu+quant", "gemm w2", "classifier",
                                     "rows qkv+att", "rows w1+w3+swiglu" };
#define PROF_START(c) ((c)->pf_t = rq_now())
#define PROF(c, i) do { double t_ = rq_now(); (c)->pf_time[i] += t_ - (c)->pf_t; (c)->pf_t = t_; } while (0)
#else
#define PROF_START(c) ((void)0)
#define PROF(c, i) ((void)0)
#endif

static void smb_config(RqCtx *c); // (RUNQ_SMB, RUNQ_BD: below)

typedef struct { RqSeq *seq; int b0, n, pos0; } PfSeg; // rows b0..b0+n-1 of a batched pass: seq's positions pos0..

struct RqCtx {
    RqModel *m;         // (read-only)
    int nt;             // threads: the per-thread scratch below
    // decode (one row) and the logits of the last call
    float *x;           // activation at current time stamp (dim,)
    float *xb;          // same, but inside a residual branch (dim,)
    float *hb;          // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2;         // buffer for hidden dimension in the ffn (hidden_dim,)
    QuantizedTensor xq; // quantized x (dim,)
    QuantizedTensor hq; // quantized hb (hidden_dim,)
    QuantizedTensor aq; // quantized attention output (all_heads_dim,), the input of wo
    float *q;           // query (dim,)
    float *k;           // key (kv_dim,), copied into the cache tiles after QK-norm + RoPE
    float *v;           // value (kv_dim,), the same
    float *logits;      // output logits (vocab_size,)
    int32_t *xq_corr;   // VNNI GEMV activation corrections of xq / hq (q8p_gemv_corr), per GS-group
    int32_t *hq_corr;
    float *xq_xf;       // K-quant kernels: activation group sums of xq / hq (kq_act_prep), or NULL
    float *hq_xf;
    int32_t *xq_c16;    // AVX-512 K-quant GEMV: -32 * sums of 16 of xq / hq (kq512_c16), or NULL
    int32_t *hq_c16;
    int32_t *aq_corr;   // the same for aq
    float *aq_xf;
    int32_t *aq_c16;
    float *rope;        // (head_dim,): RoPE cos / sin of the decoded position (rope_row)
    RqSync sync;        // phase counters of the parallel jobs (threads.h)
    // attention scratch, sized for att_nb blocks of ATT_P positions (grown for a longer sequence)
    int att_nb;
    float *att;         // decode: per (kv head, block) the scores of its kv_mul query heads (n_heads, att_nb * ATT_P)
    float *att_part;    // decode: per (kv head, query head, block) partial results (ATT_PSTRIDE floats)
    float *att_b;       // prefill: per thread att_rows * kv_mul * att_nb * ATT_P scores
    float *att_pb;      // prefill: per thread the partial results of att_rows * kv_mul vectors
    float *att_conv;    // per thread 2 * ATT_P * head_dim floats (the generic attention's fp32 block copy)
    // batched passes (rq_forward_batch), nb_max rows each; not allocated if nb_max <= 1
    int nb_max;
    float *x_b;         // (nb_max, dim) residual stream
    float *xb_b;        // (nb_max, xs)
    float *q_b;         // (nb_max, all_heads_dim)
    float *k_b;         // (nb_max, kv_dim), copied into the cache tiles after QK-norm + RoPE
    float *v_b;         // (nb_max, kv_dim), the same
    float *hb_b;        // (nb_max, hidden_dim)
    float *hb2_b;       // (nb_max, hidden_dim)
    QuantizedTensor xq_b; // (nb_max, xs)
    QuantizedTensor hq_b; // (nb_max, hidden_dim)
    float *rope_b;      // (nb_max, head_dim): RoPE cos / sin of each row's position (rope_row)
    int pf_nrb;         // attention row blocks of one segment of nb_max rows
    int32_t *bq_corr, *bq_c16, *bh_corr, *bh_c16; // GEMM activation prep of xq_b / hq_b rows (Gemm, act_prep_row)
    float *bq_xf, *bh_xf;
    uint8_t *kq_scr; size_t kq_scr_bytes;         // per-thread K-quant GEMM tile scratch (AVX-512, NEON)
    // the plan of a batched pass: per row its token and position; the segments; the QK-norm + RoPE +
    // K/V store items (a cache tile of one segment); the attention row blocks (rows of one segment,
    // longest first); the rows whose logits are computed
    int *row_tok, *row_pos;
    PfSeg *seg;
    int *tile_seg, *tile_tb;
    int *rb_seg, *rb_b0, *rb_b1;
    int *rb_it0, *rb_sp; // attention items of row blocks 0..rb-1; split row block: its index (att_sp), else -1
    int *pop_seg;       // segments starting a new KV cache tile (pre-faulted in the first phase)
    int *sel_row;
    // batched decode rows (rq_forward_batch with one row per segment, batch_decode_worker): at most bd_max
    // rows; the norms as an epilogue up to bd_epi rows (else a phase); a row's attention split over its
    // blocks of ATT_P positions when the pass has fewer than bd_split * threads (row, kv head) items; the
    // small-batch kernels up to smb_max rows per weight type (gemm_plan). Tuning: RUNQ_SMB, RUNQ_BD
    int smb_max[3], bd_max, bd_epi, bd_split;
    float *att_sp; size_t att_sp_n; // split attention: per (row, query head, block) partial results
    int *row_it0;       // batch_decode_worker: attention items of rows 0..b-1 (per kv head)
    void *probindex;    // sampler scratch (sampler.c: vocab_size candidates)
    void *probindex_t;  // the same per thread (rq_sample_batch), allocated at its first use
    int probindex_nt;
    char err[256];      // the last error's details
#ifdef RUNQ_PROFILE
    double pf_time[PF_N], pf_t;
#endif
};

static int ctx_fail(RqCtx *c, int code, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(c->err, sizeof(c->err), fmt, ap);
    va_end(ap);
    return code;
}

int forward_model_init(RqModel *m) {
    // the shape parameters the forward passes derive from the config; 0 if not supported
    Config *p = &m->config;
    m->all_heads_dim = p->n_heads * p->head_dim;
    m->kv_dim = p->n_kv_heads * p->head_dim;
    m->kv_mul = p->n_heads / p->n_kv_heads;
    m->xs = p->dim > m->all_heads_dim ? p->dim : m->all_heads_dim;
    m->att_scale = 1.0f / sqrtf(p->head_dim);
    m->att_rows = m->kv_mul < 1 ? 0 : ATT_NV_MAX / m->kv_mul < ATT_ROWS ? ATT_NV_MAX / m->kv_mul : ATT_ROWS;
    m->ff_rows = GS > 64 ? GS : 64;
    return !(m->kv_mul < 1 || m->att_rows < 1 || p->head_dim % 16 || (m->kv_mul * p->head_dim) % GS ||
             p->hidden_dim % m->ff_rows || m->ff_rows % GS || p->dim % 16);
}

// ---------------------------------------------------------------------------------------- sequences

RqSeq *rq_seq_new(const RqModel *m, int capacity, int kv_type, int *err) {
    int e = RQ_OK;
    RqSeq *s = NULL;
    const Config *p = &m->config;
    if (!m->has_weights) { e = RQ_ERR_STATE; goto fail; }
    if (capacity <= 0) capacity = p->seq_len;
    if (capacity > p->seq_len || kv_type < KV_F32 || kv_type > KV_F16 || (kv_type != KV_F32 && p->head_dim % 32)) {
        e = RQ_ERR_ARG;
        goto fail;
    }
    s = calloc(1, sizeof(RqSeq));
    if (!s) { e = RQ_ERR_NOMEM; goto fail; }
    s->m = m;
    s->cap = capacity;
    s->kv_len = (capacity + ATT_P - 1) / ATT_P * ATT_P; // (whole blocks: the cache layout)
    s->att_nb = s->kv_len / ATT_P;
    s->kvt = kv_type;
    s->kv_esz = kv_type == KV_F32 ? 4 : 2;
    s->kv_size = (size_t)p->n_layers * s->kv_len * m->kv_dim;
    s->key_cache = alloc_big(s->kv_size * s->kv_esz);
    s->value_cache = alloc_big(s->kv_size * s->kv_esz);
    s->tokens = malloc((size_t)capacity * sizeof(int));
    if (!s->key_cache || !s->value_cache || !s->tokens) { e = RQ_ERR_NOMEM; goto fail; }
    if (err) *err = RQ_OK;
    return s;
fail:
    rq_seq_free(s);
    if (err) *err = e;
    return NULL;
}

void rq_seq_free(RqSeq *s) {
    if (!s) return;
    free_big(s->key_cache, s->kv_size * s->kv_esz);
    free_big(s->value_cache, s->kv_size * s->kv_esz);
    free(s->tokens);
    free(s);
}

void rq_seq_reset(RqSeq *s) { s->len = 0; }

int rq_seq_truncate(RqSeq *s, int n) {
    // (the cache entries past n are overwritten by the next tokens: positions only ever grow from len)
    if (n < 0 || n > s->len) return RQ_ERR_ARG;
    s->len = n;
    return RQ_OK;
}

int rq_seq_len(const RqSeq *s) { return s->len; }
int rq_seq_capacity(const RqSeq *s) { return s->cap; }
const int *rq_seq_tokens(const RqSeq *s) { return s->tokens; }
void rq_seq_seed(RqSeq *s, unsigned long long seed) { s->rng = seed; }
unsigned long long *rq_seq_rng(RqSeq *s) { return &s->rng; }

int rq_seq_common_prefix(const RqSeq *s, const int *tokens, int n) {
    int L = 0;
    while (L < s->len && L < n && s->tokens[L] == tokens[L]) L++;
    return L;
}

static inline size_t kv_tile(int t, int h, int n_kv, int hd); // (the cache layout, below)
static void kv_populate_one(const RqSeq *s, const Config *p, int pos0, int n, int it);

typedef struct { RqSeq *dst; const RqSeq *src; int n, next; } CopyJob;

static void copy_worker(void *arg, int tid, int nt) {
    // rq_seq_copy's items (layer, K / V) claimed from a shared counter: the destination's tiles pre-faulted
    // (a first copy into a fresh sequence was ~85% page faults: 14.5 vs 2.1 ms for 253 positions), then the
    // whole blocks of ATT_P positions below n as one range and the last block's tiles up to position n - 1
    // per kv head (the layout: kv_tile)
    (void)tid; (void)nt;
    CopyJob *j = arg;
    const Config *p = &j->src->m->config;
    int nkv = p->n_kv_heads, hd = p->head_dim, kv_dim = j->src->m->kv_dim, full = j->n / ATT_P, rem = j->n % ATT_P;
    size_t esz = j->src->kv_esz, ntl = (size_t)(rem + ATT_T - 1) / ATT_T;
    for (int it; (it = __atomic_fetch_add(&j->next, 1, __ATOMIC_RELAXED)) < 2 * p->n_layers;) {
        int l = it / 2, kv = it % 2;
        kv_populate_one(j->dst, p, 0, j->n, it);
        const char *s = (const char *)(kv ? j->src->value_cache : j->src->key_cache) + (size_t)l * j->src->kv_len * kv_dim * esz;
        char *d = (char *)(kv ? j->dst->value_cache : j->dst->key_cache) + (size_t)l * j->dst->kv_len * kv_dim * esz;
        memcpy(d, s, (size_t)full * ATT_P * kv_dim * esz);
        for (int h = 0; h < nkv && rem; h++) {
            size_t o = kv_tile(full * ATT_P, h, nkv, hd) * esz;
            memcpy(d + o, s + o, ntl * ATT_T * hd * esz);
        }
    }
}

int rq_seq_copy(RqSeq *dst, const RqSeq *src, int n) {
    // (on the thread pool: one parallel job between forward passes, as rq_sample_batch)
    if (!dst || !src || dst == src || dst->m != src->m || dst->kvt != src->kvt || n < 0 || n > src->len || n > dst->cap)
        return RQ_ERR_ARG;
    if (n > 0) {
        CopyJob j = { dst, src, n, 0 };
        rq_parallel(copy_worker, &j);
    }
    memcpy(dst->tokens, src->tokens, (size_t)n * sizeof(int));
    dst->len = n;
    return RQ_OK;
}

const char *rq_kv_name(int t) {
    static const char *name[] = { "f32", "bf16", "f16" };
    return t >= KV_F32 && t <= KV_F16 ? name[t] : "?";
}

int rq_kv_parse(const char *s) {
    for (int t = KV_F32; t <= KV_F16; t++)
        if (!strcmp(s, rq_kv_name(t))) return t;
    return RQ_ERR_ARG;
}

// --------------------------------------------------------------------------------- compute contexts

static int ctx_alloc_att(RqCtx *c, int att_nb) {
    // the attention scratch for att_nb blocks (decode, and prefill if batched); RQ_OK or RQ_ERR_NOMEM
    const RqModel *m = c->m;
    const Config *p = &m->config;
    size_t pst = ATT_PSTRIDE(p->head_dim), vr = (size_t)m->att_rows * m->kv_mul;
    float *att = calloc((size_t)p->n_heads * att_nb * ATT_P, sizeof(float));
    float *part = calloc((size_t)p->n_heads * att_nb * pst, sizeof(float));
    float *att_b = NULL, *att_pb = NULL;
    if (c->nb_max > 1) {
        att_b = calloc((size_t)c->nt * vr * att_nb * ATT_P, sizeof(float));
        att_pb = calloc((size_t)c->nt * vr * att_nb * pst, sizeof(float));
    }
    if (!att || !part || (c->nb_max > 1 && (!att_b || !att_pb))) {
        free(att);
        free(part);
        free(att_b);
        free(att_pb);
        return RQ_ERR_NOMEM;
    }
    free(c->att);
    free(c->att_part);
    free(c->att_b);
    free(c->att_pb);
    c->att = att;
    c->att_part = part;
    c->att_b = att_b;
    c->att_pb = att_pb;
    c->att_nb = att_nb;
    return RQ_OK;
}

static int ctx_check_seq(RqCtx *c, const RqSeq *s) {
    // s can run on c (the same model; its attention scratch grown if s is longer than any before)
    if (s->m != c->m) return ctx_fail(c, RQ_ERR_ARG, "sequence of another model");
    if (s->att_nb > c->att_nb && ctx_alloc_att(c, s->att_nb) != RQ_OK)
        return ctx_fail(c, RQ_ERR_NOMEM, "malloc failed!");
    return RQ_OK;
}

RqCtx *rq_ctx_new(const RqModel *mc, int nb_max, int max_ctx, int *err) {
    RqModel *m = (RqModel *)mc; // (the forward passes only read it)
    const Config *p = &m->config;
    RqCtx *c = NULL;
    int e = RQ_ERR_NOMEM;
    if (!m->has_weights) { e = RQ_ERR_STATE; goto fail; }
    c = calloc(1, sizeof(RqCtx));
    if (!c) goto fail;
    int all_heads_dim = m->all_heads_dim, kv_dim = m->kv_dim, hd = p->head_dim;
    int ad = all_heads_dim > p->dim ? all_heads_dim : p->dim;
    c->m = m;
    c->nt = rq_nthreads();
    c->nb_max = nb_max < 0 ? 0 : nb_max > B_MAX ? B_MAX : nb_max;
    smb_config(c);
    if (max_ctx <= 0 || max_ctx > p->seq_len) max_ctx = p->seq_len;
    if (c->nb_max > 1) {
        size_t nb = c->nb_max;
        int xs = m->xs;
        c->x_b = calloc(nb * p->dim, sizeof(float));
        c->xb_b = calloc(nb * xs, sizeof(float));
        c->q_b = calloc(nb * all_heads_dim, sizeof(float));
        c->k_b = calloc(nb * kv_dim, sizeof(float));
        c->v_b = calloc(nb * kv_dim, sizeof(float));
        c->hb_b = calloc(nb * p->hidden_dim, sizeof(float));
        c->hb2_b = calloc(nb * p->hidden_dim, sizeof(float));
        c->xq_b = (QuantizedTensor) { .q = calloc(nb * xs, sizeof(int8_t)), .s = calloc(nb * xs / GS, sizeof(float)) };
        c->hq_b = (QuantizedTensor) { .q = calloc(nb * p->hidden_dim, sizeof(int8_t)), .s = calloc(nb * p->hidden_dim / GS, sizeof(float)) };
        c->pf_nrb = (c->nb_max + m->att_rows - 1) / m->att_rows;
        c->bq_corr = calloc(nb * (xs / GS) + 1, sizeof(int32_t));
        c->bh_corr = calloc(nb * (p->hidden_dim / GS) + 1, sizeof(int32_t));
        if (!c->bq_corr || !c->bh_corr) goto fail;
        if (m->kq) {
            c->bq_xf = calloc(nb * (xs / GS) + 1, sizeof(float));
            c->bh_xf = calloc(nb * (p->hidden_dim / GS) + 1, sizeof(float));
            c->bq_c16 = calloc(nb * (xs / 16) + 1, sizeof(int32_t));
            c->bh_c16 = calloc(nb * (p->hidden_dim / 16) + 1, sizeof(int32_t));
            if (!c->bq_xf || !c->bh_xf || !c->bq_c16 || !c->bh_c16) goto fail;
#ifdef RUNQ_KQX
            int kmax = p->hidden_dim > xs ? p->hidden_dim : xs;
            c->kq_scr_bytes = (KQX_SCRATCH(kmax) + 63) / 64 * 64;
            c->kq_scr = aligned_alloc(64, c->kq_scr_bytes * c->nt);
            if (!c->kq_scr) goto fail;
#endif
        }
        c->rope_b = calloc(nb * hd, sizeof(float));
        c->row_tok = malloc(nb * sizeof(int));
        c->row_pos = malloc(nb * sizeof(int));
        c->seg = malloc(nb * sizeof(PfSeg));
        c->tile_seg = malloc(2 * nb * sizeof(int)); // (tiles of the segments: <= n / ATT_T + nseg)
        c->tile_tb = malloc(2 * nb * sizeof(int));
        c->rb_seg = malloc((c->pf_nrb + nb) * sizeof(int)); // (row blocks: <= n / att_rows + nseg)
        c->rb_b0 = malloc((c->pf_nrb + nb) * sizeof(int));
        c->rb_b1 = malloc((c->pf_nrb + nb) * sizeof(int));
        c->rb_it0 = malloc((c->pf_nrb + nb + 1) * sizeof(int));
        c->rb_sp = malloc((c->pf_nrb + nb) * sizeof(int));
        c->pop_seg = malloc(nb * sizeof(int));
        c->sel_row = malloc(nb * sizeof(int));
        c->row_it0 = malloc((nb + 1) * sizeof(int));
        if (!c->rope_b || !c->k_b || !c->v_b || !c->x_b || !c->xb_b || !c->q_b || !c->hb_b || !c->hb2_b || !c->xq_b.q ||
            !c->xq_b.s || !c->hq_b.q || !c->hq_b.s || !c->row_tok || !c->row_pos || !c->seg || !c->tile_seg ||
            !c->tile_tb || !c->rb_seg || !c->rb_b0 || !c->rb_b1 || !c->sel_row || !c->row_it0 ||
            !c->rb_it0 || !c->rb_sp || !c->pop_seg)
            goto fail;
    }

    c->x = calloc(p->dim, sizeof(float));
    c->xb = calloc(ad, sizeof(float)); // (dim and all_heads_dim values: max of both, ad; Qwen3: all_heads_dim >= dim)
    c->hb = calloc(p->hidden_dim, sizeof(float));
    c->hb2 = calloc(p->hidden_dim, sizeof(float));
    c->xq = (QuantizedTensor) { .q = calloc(ad, sizeof(int8_t)), .s = calloc(ad / GS, sizeof(float)) };
    c->hq = (QuantizedTensor) { .q = calloc(p->hidden_dim, sizeof(int8_t)), .s = calloc(p->hidden_dim / GS, sizeof(float)) };
    c->q = calloc(all_heads_dim, sizeof(float));
    c->xq_corr = calloc(ad / GS + 1, sizeof(int32_t));
    c->hq_corr = calloc(p->hidden_dim / GS + 1, sizeof(int32_t));
    c->aq = (QuantizedTensor) { .q = calloc(all_heads_dim, sizeof(int8_t)), .s = calloc(all_heads_dim / GS, sizeof(float)) };
    c->aq_corr = calloc(all_heads_dim / GS + 1, sizeof(int32_t));
    if (!c->aq.q || !c->aq.s || !c->aq_corr || !c->xq.q || !c->xq.s || !c->hq.q || !c->hq.s) goto fail;
    if (m->kq) {
        c->aq_xf = calloc(all_heads_dim / GS + 1, sizeof(float));
        c->aq_c16 = calloc(all_heads_dim / 16 + 1, sizeof(int32_t));
        c->xq_xf = calloc(ad / GS + 1, sizeof(float));
        c->hq_xf = calloc(p->hidden_dim / GS + 1, sizeof(float));
        c->xq_c16 = calloc(ad / 16 + 1, sizeof(int32_t));
        c->hq_c16 = calloc(p->hidden_dim / 16 + 1, sizeof(int32_t));
        if (!c->aq_xf || !c->aq_c16 || !c->xq_xf || !c->hq_xf || !c->xq_c16 || !c->hq_c16) goto fail;
    }
    c->k = calloc(kv_dim, sizeof(float));
    c->v = calloc(kv_dim, sizeof(float));
    c->logits = calloc(p->vocab_size, sizeof(float));
    c->rope = calloc(hd, sizeof(float));
    c->att_conv = malloc((size_t)c->nt * 2 * ATT_P * hd * sizeof(float)); // (touched only by the generic attention)
    c->probindex = malloc((size_t)p->vocab_size * 8);
    if (!c->x || !c->xb || !c->hb || !c->hb2 || !c->q || !c->xq_corr || !c->hq_corr || !c->k || !c->v || !c->logits ||
        !c->rope || !c->att_conv || !c->probindex)
        goto fail;
    if (ctx_alloc_att(c, (max_ctx + ATT_P - 1) / ATT_P) != RQ_OK) goto fail;
    // phase / counter capacity: decode 4 / layer, 2 counters per (layer, kv head); prefill 9 / layer,
    // a counter per (layer, attention row block)
    // (batch_decode_worker: per (layer, kv head), (layer, row), (layer, row, kv head))
    int ncnt = 2 * p->n_kv_heads > c->pf_nrb + c->nb_max ? 2 * p->n_kv_heads : c->pf_nrb + c->nb_max;
    // (prefill_worker with split attention: per (layer, row block), (layer, split row block, kv head))
    if (ncnt < 2 * p->n_kv_heads + c->pf_nrb + c->nb_max * (1 + p->n_kv_heads))
        ncnt = 2 * p->n_kv_heads + c->pf_nrb + c->nb_max * (1 + p->n_kv_heads);
    if (rq_sync_init(&c->sync, 10 * p->n_layers + 8, p->n_layers * ncnt + 64) != 0) goto fail;
    if (err) *err = RQ_OK;
    return c;
fail:
    rq_ctx_free(c);
    if (err) *err = e;
    return NULL;
}

void rq_ctx_free(RqCtx *c) {
    if (!c) return;
    void *bufs[] = { c->x, c->xb, c->hb, c->hb2, c->xq.q, c->xq.s, c->hq.q, c->hq.s, c->q, c->xq_corr, c->hq_corr,
                     c->xq_xf, c->hq_xf, c->xq_c16, c->hq_c16, c->aq.q, c->aq.s, c->aq_corr, c->aq_xf, c->aq_c16, c->k,
                     c->v, c->logits, c->rope, c->att, c->att_part, c->att_b, c->att_pb, c->att_conv, c->x_b, c->xb_b,
                     c->q_b, c->k_b, c->v_b, c->hb_b, c->hb2_b, c->xq_b.q, c->xq_b.s, c->hq_b.q, c->hq_b.s, c->rope_b,
                     c->bq_corr, c->bq_xf, c->bq_c16, c->bh_corr, c->bh_xf, c->bh_c16, c->kq_scr, c->row_tok,
                     c->row_pos, c->seg, c->tile_seg, c->tile_tb, c->rb_seg, c->rb_b0, c->rb_b1, c->sel_row, c->row_it0, c->att_sp, c->rb_it0, c->rb_sp, c->pop_seg, c->probindex, c->probindex_t };
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++)
        free(bufs[i]);
    rq_sync_free(&c->sync);
    free(c);
}

int rq_ctx_max_batch(const RqCtx *c) { return c->nb_max; }
const char *rq_ctx_error(const RqCtx *c) { return c->err; }
void *ctx_sampler_scratch(RqCtx *c) { return c->probindex; }

void *ctx_sampler_scratch_threads(RqCtx *c, int nt) {
    if (nt > c->probindex_nt) {
        void *p = malloc((size_t)nt * c->m->config.vocab_size * 8);
        if (!p) return NULL;
        free(c->probindex_t);
        c->probindex_t = p;
        c->probindex_nt = nt;
    }
    return c->probindex_t;
}
int ctx_vocab(const RqCtx *c) { return c->m->config.vocab_size; }

int rq_ctx_profile(const RqCtx *c, const char **names, double *seconds, int max) {
#ifdef RUNQ_PROFILE
    int n = max < PF_N ? max : PF_N;
    for (int i = 0; i < n; i++) {
        names[i] = pf_name[i];
        seconds[i] = c->pf_time[i];
    }
    return n;
#else
    (void)c; (void)names; (void)seconds; (void)max;
    return 0;
#endif
}

// ----------------------------------------------------------------------------
// Quantization functions

void quantize(QuantizedTensor *qx, float *x, int n) {
    // per group of GS values: scale = max |x| / 127, q = x / scale rounded half away from zero.
    // Written as the operations gcc -Ofast made of upstream's expressions (`wmax / 127.0f`,
    // `(int8_t) round(x / scale)`; read off its asm): scale = wmax * (float)(1 / 127), inv = 127 /
    // wmax (one division), q = (int) trunc(x * inv + copysign(0.49999997, x * inv)), so the results
    // are the same bits; with local pointers (the int8 stores no longer alias GS or qx) the loop now
    // vectorizes: 1,024 values ~0.6 -> ~0.1 us (it was scalar, ~8 instructions per value)
    const int gs = GS;
    for (int group = 0; group < n / gs; group++) {
        const float *restrict xg = x + (size_t)group * gs;
        int8_t *restrict qg = qx->q + (size_t)group * gs;
        // find the max absolute value in the current group
        float wmax = 0;
        for (int i = 0; i < gs; i++) {
            float val = fabsf(xg[i]);
            if (val > wmax)
                wmax = val;
        }

        // calculate and write the scaling factor
        qx->s[group] = wmax * (1.0f / 127.0f);
        if (!(wmax > 0)) { // all-zero group: q = 0 (was 0/0 = NaN cast to int8, UB)
            memset(qg, 0, gs);
            continue;
        }
        float inv = 127.0f / wmax;

        // calculate and write the quantized values
        for (int i = 0; i < gs; i++) {
            float v = xg[i] * inv;
            qg[i] = (int8_t)(int32_t)(v + copysignf(0.49999997f, v));
        }
    }
}

// (fp16_to_f32: kquants.h)
void embed_row(float *x, const QuantizedTensor *t, int token, int dim) {
    // row `token` of the Q8_0 embedding table t, dequantized with upstream's dequantize()
    // expression, x[i] = q[i] * s[i / GS] (exact int8 -> float, one rounding): the same floats
    // as the whole dequantized table upstream kept in memory. GGUF: from its blocks, the same
    // expression with the fp16 scale as a float (dim % 32 == 0, checked at load)
    size_t e0 = (size_t)token * dim;
    if (t->b && t->type != GGML_Q8_0) {
        // GGUF Q4_K / Q6_K: ggml's dequantization of the row's blocks (dim % 256 == 0). By the tensor's
        // type, not by t->kp: an untied embedding table (Qwen3-8B: Q4_K, output.weight Q6_K) is never
        // repacked (only the matmul weights are), so it has no kp and used to be read as Q8_0 blocks (NaN)
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

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float *o, float *x, float *weight, int size) {
    // calculate sum of squares
    float ss = 0;
    for (int j = 0; j < size; j++)
        ss += x[j] * x[j];

    // (an explicit reciprocal: with ss / size, -Ofast (reciprocal math) multiplied by a hoisted 1 / size
    // where rmsnorm is inlined into a per-row loop but divided in the single decode call, which rounds
    // differently unless size is a power of two: Qwen3-4B's dim 2560 broke the bit-identity of batched
    // vs sequential vs generic. For power-of-two sizes both are exact: the same bits as before)
    ss = 1.0f / sqrtf((ss * (1.0f / size)) + 1e-6f);

    // normalize and scale
    for (int j = 0; j < size; j++)
        o[j] = weight[j] * (ss * x[j]);
}

// Q8_0 dot product of one weight row with one quantized activation row (n values, n/GS groups):
// exact int32 sum per group of GS, then accumulated in float, group by group in order.
// Shared by the GEMV (matvec_rows) and the generic GEMM (gemm_item) so both give bit-identical results.
static inline float dot_q8(const int8_t *wq, const float *ws, const int8_t *xq, const float *xs, int n) { RQ_FP_EXACT
    float val = 0;
    for (int j = 0; j <= n - GS; j += GS) {
        int32_t ival = 0;
        for (int k = 0; k < GS; k++)
            ival += xq[j + k] * wq[j + k];

        val += ((float) ival) * ws[j / GS] * xs[j / GS];
    }
    return val;
}

// Decode GEMV, in pieces: decode_step runs all of a token in one parallel region, with the rows
// of each matrix split into work items. matvec_prep: the VNNI kernel's activation corrections for
// x[i0 .. i0+n) (one per GS-group; nothing to do for the generic kernel), and with K-quant weights
// (xf != NULL) the activation group sums xf of kquants.h, plus for the AVX-512 K-quant GEMV the
// -32 * sums of 16 (c16, Q6_K); matvec_rows: rows i0..i1-1
static inline void matvec_prep(int32_t *corr, float *xf, int32_t *c16, QuantizedTensor *x, int i0, int n) {
#ifdef RUNQ_Q8X
    q8x_gemv_corr(corr + i0 / GS, x->q + i0, n, GS);
#else
    (void)corr;
#endif
#ifdef RUNQ_KQX
    if (c16)
        kqx_c16(c16 + i0 / 16, x->q + i0, n);
#else
    (void)c16;
#endif
    if (xf)
        kq_act_prep(xf + i0 / GS, x->q + i0, x->s + i0 / GS, n);
}

static inline void matvec_rows(float *xout, QuantizedTensor *x, const int32_t *corr, const float *xf, const int32_t *c16,
                               QuantizedTensor *w, int n, int i0, int i1) {
    // xout[i] = W (d,n) row i . x (n,) for i0 <= i < i1 (multiples of 16, as the VNNI blocks)
    if (w->kp) { // Q4_K / Q6_K, 16-row blocks (kquants.h; AVX-512: kernels_avx512.h)
#ifdef RUNQ_KQX
        kqx_gemv_rows(w->type, xout, x->q, x->s, xf, c16, w->kp, n, i0 / KQ_ROWS, i1 / KQ_ROWS);
#else
        (void)c16;
        for (int i = i0; i < i1; i += KQ_ROWS)
            kq_dot(w->type, xout + i, 0, w->kp + (size_t)(i / KQ_ROWS) * (n / QK_K) * kq_sb_bytes(w->type), n, 1,
                   x->q, 0, x->s, 0, xf, 0);
#endif
        return;
    }
#ifdef RUNQ_Q8X
    if (w->p) { q8x_gemv_rows(xout, x->q, x->s, corr, w->p, n, GS, i0 / 16, i1 / 16); return; }
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

NO_FP_CONTRACT static inline void dot_q8_x4(float *o, const int8_t *w0, const float *s0, int k, const int8_t *xq, const float *xs) { RQ_FP_EXACT
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

// One batched-matmul phase of a batched pass: up to 3 GEMMs on the same nb quantized activation rows
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
    int smb[3];          // job j by the small-batch kernel: one item per 16-row block, all rows (smb_rows)
    int nbi, nchunk;     // tiled: activation rows per chunk, chunks
    int chunk, unit;     // items per claim; ranges of whole units (tiled: a tile's chunks)
    QuantizedTensor *x; int ldx, nb, k;
    const int32_t *corr; const float *xf; const int32_t *c16;
} Gemm;

// Small-batch kernels (AVX-512, kernels_avx512.h smb_rows): in prefill_worker's GEMM phases for a matmul
// over at most smb_max rows of its weight type (else the GEMM's tiles), and always in batch_decode_worker.
// Crossover (one thread, TODO.md Step 10a): Q4_K / Q6_K faster than the GEMM up to ~24 rows (1.6-2.3x at
// 1-4 rows: no tile unpacking), equal at 32, ~4% slower at 48-64; Q8_0 equal (the GEMM tile streams the
// packed weights too). RUNQ_SMB=Q8,Q4,Q6 overrides them (tuning; 0 = never). Results never depend on it
enum { SMB_Q8, SMB_Q4, SMB_Q6 };
#ifndef BD_MAX
#define BD_MAX 64 // decode rows per batched pass in batch_decode_worker (if the small-batch kernels take them)
#endif

static void env_ints(const char *name, int *v, int n) {
    // comma-separated integers from the environment variable name into v[0..n-1] (missing: unchanged)
    const char *e = getenv(name);
    for (int i = 0; e && *e && i < n; i++) {
        if (*e != ',') v[i] = atoi(e);
        e = strchr(e, ',');
        if (e) e++;
    }
}

static void smb_config(RqCtx *c) {
    // the context's small-batch / batched-decode settings: defaults, RUNQ_SMB=Q8,Q4,Q6 (rows), RUNQ_BD=max,epi,split
    c->smb_max[SMB_Q8] = 32;
    c->smb_max[SMB_Q4] = 32;
    c->smb_max[SMB_Q6] = 32;
    env_ints("RUNQ_SMB", c->smb_max, 3);
    int bd[3] = { BD_MAX, 2, 8 };
    env_ints("RUNQ_BD", bd, 3);
#ifdef RUNQ_NEON
    bd[0] = 0; // (NEON: its GEMM for batched rows, as before; no small-batch kernels there yet)
#endif
    c->bd_max = bd[0] < c->nb_max ? bd[0] : c->nb_max;
    c->bd_epi = bd[1];
    c->bd_split = bd[2];
}

static inline int smb_use(const RqCtx *c, const QuantizedTensor *w, int nb) {
    // the small-batch kernel for nb rows of w
#ifdef RUNQ_SMB
    if (w->kp) return nb <= c->smb_max[w->type == GGML_Q4_K ? SMB_Q4 : SMB_Q6];
    return w->p && nb <= c->smb_max[SMB_Q8];
#else
    (void)c; (void)w; (void)nb;
    return 0;
#endif
}

static void gemm_plan(const RqCtx *c, Gemm *g, int nj, const MatJob *job, QuantizedTensor *x, int ldx, int nb, int k,
                      const int32_t *corr, const float *xf, const int32_t *c16, int nt) {
    // (computed by every thread of the job: the same plan everywhere; results never depend on it)
    g->nj = nj; g->x = x; g->ldx = ldx; g->nb = nb; g->k = k; g->corr = corr; g->xf = xf; g->c16 = c16;
    int ntile = 0, all_tiled = 1;
    for (int j = 0; j < nj; j++) {
        g->job[j] = job[j];
        g->tiled[j] = 0;
#ifdef RUNQ_KQX
        g->tiled[j] = job[j].w->kp != NULL;
#endif
#ifdef RUNQ_Q8X
        g->tiled[j] |= job[j].w->p != NULL;
#endif
        g->smb[j] = smb_use(c, job[j].w, nb);
        if (g->smb[j])
            g->tiled[j] = 0;
        all_tiled &= g->tiled[j];
        if (g->tiled[j])
            ntile += (job[j].d / 16 + 1) / 2;
    }
    g->nbi = nb;
    g->nchunk = 1;
#if defined RUNQ_Q8X || defined RUNQ_KQX
    if (ntile) {
        g->nbi = kqx_nb(nb, ntile, nt);
        g->nchunk = (nb + g->nbi - 1) / g->nbi;
    }
#endif
    int n = 0;
    for (int j = 0; j < nj; j++) {
        g->it0[j] = n;
        n += g->tiled[j] ? (job[j].d / 16 + 1) / 2 * g->nchunk : job[j].w->kp || g->smb[j] ? job[j].d / 16
                                                                 : (job[j].d + MB_ROWS - 1) / MB_ROWS;
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
#ifdef RUNQ_SMB
    if (g->smb[j]) { // 16-row block i, all rows
        smb_rows(w->kp ? w->type : 0, jb->out, jb->ldo, x->q, ldx, x->s, g->corr, g->xf, g->c16, nb, w->kp ? w->kp : w->p, k, GS,
                 i, i + 1);
        return;
    }
#endif
#if defined RUNQ_Q8X || defined RUNQ_KQX
    if (g->tiled[j]) {
        int t = i / g->nchunk, ch = i % g->nchunk;
#ifdef RUNQ_KQX
        if (w->kp) {
            kqx_gemm_item(jb->out, jb->ldo, w->kp, w->type, jb->d, t, ch, g->nbi, x->q, x->s, ldx, nb, k, g->xf, g->c16, scr, last);
            return;
        }
#endif
#ifdef RUNQ_Q8X
        q8x_gemm_item(jb->out, jb->ldo, w->p, jb->d, t, ch, g->nbi, x->q, x->s, ldx, nb, k, GS, g->corr);
#endif
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

static void rope_row(float *cs, int head_dim, int pos, float log_theta) {
    // RoPE rotation of position pos: cs[j] = cos, cs[j + head_dim/2] = sin of pos * freq_j, computed
    // once per position (was: in qk_norm_rope for every layer and head, ~6% of a 451-token
    // prefill), with the original expressions, so the values are the same. freq_j = theta^(-j / (hd/2))
    // as exp(ln(theta) * x): the form gcc -Ofast made of the constant powf(1e6, x) before (vectorized
    // expf), so theta 1e6 (log_theta = (float)ln(1e6)) gives the same bits; with a variable base, powf
    // would be the vector powf (other bits). theta: the model's (RqModel.rope_log_theta; Qwen3 1e6,
    // the 2507 models 5e6)
    for (int j = 0; j < head_dim/2; j++) {
        float freq = expf(log_theta * (-(float)j / (head_dim/2)));
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

ATT_EXACT static inline float exp_att(float x) { RQ_FP_EXACT
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

static inline float bf16_f32(uint16_t h) {
    // bf16 -> fp32, exact
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static inline uint16_t f32_f16(float f) {
    // fp32 -> IEEE half, round to nearest even (the f16 KV cache stores; the same operations in every
    // build, branch-free so that the stores vectorize): equal to vcvtps2ph for every non-denormal input
    // (-Ofast's FTZ / DAZ: none here), except that magnitudes that would round to inf saturate at 65504
    // (|K| <= ~500, |V| <= ~100 measured for Qwen3-0.6B); f16 subnormals (|x| < 2^-14) rounded exactly
    // (x * 2^24 is exact, rintf rounds to even); NaN stays a quiet NaN. Checked exhaustively over all
    // 2^32 inputs against vcvtps2ph, and fp16_to_f32 (kquants.h, the loads) against vcvtph2ps
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000u, a = u & 0x7fffffffu, n = a - 0x38000000u;
    uint32_t h = (n + 0xfffu + ((n >> 13) & 1)) >> 13; // normal: exponent bias 127 -> 15, 13 bits off
    uint32_t as = a < 0x38800000u ? a : 0;               // below 2^-14: m * 2^-24, m = rint(|x| 2^24)
    float fs;
    memcpy(&fs, &as, 4);
    uint32_t sub = (uint32_t)rintf(fs * 0x1p24f);
    h = a < 0x38800000u ? sub : h;
    h = a >= 0x477ff000u ? 0x7bffu : h;                  // >= 65520: would round to inf
    h = a > 0x7f800000u ? (0x7e00u | ((a >> 13) & 0x3ffu)) : h;
    return (uint16_t)(sign | h);
}

static inline uint16_t f32_bf16(float f) {
    // fp32 -> bf16, round to nearest even (the bf16 KV cache stores). Integer arithmetic, the same in
    // every build; equal to vcvtneps2bf16 for every non-denormal input (it flushes denormals, which
    // -Ofast's FTZ / DAZ never produce here); NaN stays a (quiet) NaN
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t r = (u + 0x7fffu + ((u >> 16) & 1)) >> 16;
    return (uint16_t)((u & 0x7fffffffu) > 0x7f800000u ? (u >> 16) | 0x40 : r);
}

ATT_EXACT static void attention_vec(const float *q, float *out, float *ml, int n, const float *kt, const float *vt,
                                    size_t ts, int hd, float scale, float *S) { RQ_FP_EXACT
    // one query vector over positions 0..n-1 (n <= ATT_P: one block) of one kv head: kt / vt = its fp32
    // tile 0 of the block (layout in kernels_avx512.h), ts = floats from one tile to the next (ATT_T *
    // head_dim: a block's tiles are contiguous); S: n (rounded up to 16) floats.
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

ATT_EXACT static void att_combine(float *out, int nb, const float *part, size_t stride, int hd) { RQ_FP_EXACT
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

static void kv_block_f32(int kvt, float *kf, float *vf, const uint16_t *kh, const uint16_t *vh, int ntl, int hd) {
    // the first ntl tiles of one (kv head, block) of a 16-bit cache as fp32 tiles (the fp32 layout, exact
    // conversion; layouts in kernels_avx512.h): the generic attention's input
    size_t ne = (size_t)ntl * ATT_T * hd;
    if (kvt == KV_F16) {
        for (size_t e = 0; e < ne; e++) {
            kf[e] = fp16_to_f32(kh[e]);
            vf[e] = fp16_to_f32(vh[e]);
        }
        return;
    }
    for (size_t t0 = 0; t0 < ne; t0 += (size_t)ATT_T * hd)     // K: dims 2p, 2p + 1 of position j at
        for (int i = 0; i < hd; i++)                            // 32 p + 2 j (+ 1)
            for (int j = 0; j < ATT_T; j++)
                kf[t0 + (size_t)i * ATT_T + j] = bf16_f32(kh[t0 + (size_t)(i & ~1) * ATT_T + 2 * j + (i & 1)]);
    for (size_t r0 = 0; r0 < ne; r0 += hd)                      // V rows: dims 32 c + l, 32 c + 16 + l at
        for (int i = 0; i < hd; i++)                            // 32 c + 2 l (+ 1)
            vf[r0 + i] = bf16_f32(vh[r0 + (i & ~31) + 2 * (i & 15) + ((i >> 4) & 1)]);
}

static void attention_part(int kvt, int nv, const float *const *q, float *const *out, float *const *part, const int *np,
                           int b0, int b1, const void *kt, const void *vt, size_t bs, int hd, float scale, float *S,
                           float *conv) {
    // nv query vectors q[r] sharing one kv head (kt / vt: its tile 0; kvt: cache type), vector r over
    // positions 0..np[r]-1. part NULL: the whole attention into out[r] (all np <= ATT_P). Else the
    // partial results of the blocks b0..b1-1 of ATT_P positions that vector r reaches, block b at
    // part[r] + b * ATT_PSTRIDE(hd): o (unnormalized), m (max score), l (sum of exponentials).
    // S: scratch of nv * (b1 - b0) * ATT_P floats; conv: 2 * ATT_P * hd floats (the generic C version's
    // fp32 copy of a 16-bit cache block; this thread's)
#ifdef RUNQ_ATT512
    if (att512_ok(hd)) { att512_kv(kvt, nv, q, out, part, np, b0, b1, kt, vt, bs, hd, scale, S); return; }
#endif
#ifdef RUNQ_ATTNEON
    if (attn_ok(hd)) { attn_kv(kvt, nv, q, out, part, np, b0, b1, kt, vt, bs, hd, scale, S); return; }
#endif
    // generic C: per block of ATT_P positions (a 16-bit cache converted to fp32 once, for all the
    // vectors), attention_vec per vector
    size_t esz = kvt == KV_F32 ? 4 : 2, ts = (size_t)ATT_T * hd;
    if (kvt == KV_F32)
        conv = NULL;
    for (int b = b0; b < b1; b++) {
        int nmax = 0; // positions of block b any vector needs
        for (int r = 0; r < nv; r++) {
            int n = np[r] - b * ATT_P < ATT_P ? np[r] - b * ATT_P : ATT_P;
            nmax = n > nmax ? n : nmax;
        }
        if (nmax <= 0)
            break;
        const char *kb = (const char *)kt + (size_t)b * bs * esz, *vb = (const char *)vt + (size_t)b * bs * esz;
        const float *kf = (const float *)kb, *vf = (const float *)vb;
        if (conv) {
            kv_block_f32(kvt, conv, conv + (size_t)ATT_P * hd, (const uint16_t *)kb, (const uint16_t *)vb,
                         (nmax + ATT_T - 1) / ATT_T, hd);
            kf = conv;
            vf = conv + (size_t)ATT_P * hd;
        }
        for (int r = 0; r < nv; r++) {
            int n = np[r] - b * ATT_P < ATT_P ? np[r] - b * ATT_P : ATT_P;
            if (n <= 0)
                continue;
            if (!part)
                attention_vec(q[r], out[r], NULL, n, kf, vf, ts, hd, scale, S);
            else
                attention_vec(q[r], part[r] + (size_t)b * ATT_PSTRIDE(hd), part[r] + (size_t)b * ATT_PSTRIDE(hd) + hd, n,
                              kf, vf, ts, hd, scale, S);
        }
    }
}

static void attention_kv(int kvt, int nv, const float *const *q, float *const *out, const int *np, const void *kt,
                         const void *vt, size_t bs, int hd, float scale, float *S, float *part, float *conv) {
    // nv query vectors q[r] sharing one kv head, vector r over positions 0..np[r]-1, into out[r]:
    // per block of ATT_P positions the partial results, then their combination (one block: the
    // attention directly). S: scratch of nv * (blocks of max np) * ATT_P floats; part: nv * (blocks of
    // max np) * ATT_PSTRIDE(hd) floats. Each vector's result does not depend on the others
    int nmax = 0;
    for (int r = 0; r < nv; r++)
        nmax = np[r] > nmax ? np[r] : nmax;
    if (nmax <= ATT_P) {
        attention_part(kvt, nv, q, out, NULL, np, 0, 1, kt, vt, bs, hd, scale, S, conv);
        return;
    }
    int nbm = (nmax + ATT_P - 1) / ATT_P;
    size_t st = ATT_PSTRIDE(hd);
    float *pv[ATT_NV_MAX];
    for (int r = 0; r < nv; r++)
        pv[r] = part + (size_t)r * nbm * st;
    attention_part(kvt, nv, q, NULL, pv, np, 0, nbm, kt, vt, bs, hd, scale, S, conv);
    for (int r = 0; r < nv; r++)
        att_combine(out[r], (np[r] + ATT_P - 1) / ATT_P, pv[r], st, hd);
}

static inline size_t kv_tile(int t, int h, int n_kv, int hd) {
    // offset (elements) of the cache tile holding position t of kv head h, inside a layer: blocks of
    // ATT_P positions, in a block the heads one after the other, each its ATT_P / ATT_T tiles (one
    // contiguous stream per head and block; see kernels_avx512.h)
    return (((size_t)(t / ATT_P) * n_kv + h) * ATT_P + (size_t)(t % ATT_P / ATT_T) * ATT_T) * hd;
}

static void kv_store(int kvt, void *kl0, void *vl0, int t, int h, int n_kv, int hd, const float *k, const float *v) {
    // head h of the key / value rows k, v of position t into the layer's cache tiles kl, vl
    // (k or v may be NULL: only the other one); kvt: cache type (bf16 / f16: rounded to nearest even,
    // layouts in kernels_avx512.h)
    size_t o = kv_tile(t, h, n_kv, hd);
    int j = t % ATT_T;
    if (kvt == KV_F16) {
        uint16_t *kl = kl0, *vl = vl0;
        if (k)
            for (int i = 0; i < hd; i++)
                kl[o + (size_t)i * ATT_T + j] = f32_f16(k[i]);
        if (v)
            for (int i = 0; i < hd; i++)
                vl[o + (size_t)j * hd + i] = f32_f16(v[i]);
        return;
    }
    if (kvt == KV_BF16) {
        uint16_t *kl = kl0, *vl = vl0;
        if (k)
            for (int i = 0; i < hd; i++)
                kl[o + (size_t)(i & ~1) * ATT_T + 2 * j + (i & 1)] = f32_bf16(k[i]);
        if (v)
            for (int i = 0; i < hd; i++)
                vl[o + (size_t)j * hd + (i & ~31) + 2 * (i & 15) + ((i >> 4) & 1)] = f32_bf16(v[i]);
        return;
    }
    float *kl = kl0, *vl = vl0;
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

static void kv_populate_one(const RqSeq *s, const Config *p, int pos0, int n, int it) {
    // Pre-fault the KV cache tiles of positions pos0..pos0+n-1 of layer it / 2, K (it even) or V
    // (anonymous memory, untouched pages are not mapped yet): one madvise per kv head's range instead
    // of one page fault per 4 KB, the 2 * n_layers items in parallel. Otherwise these faults happen inside
    // the K/V GEMMs, serialized in the kernel: ~55 ms of a 451-token prefill (8 threads), vs ~7 ms
    // here. Linux >= 5.14; errors ignored
#if !defined _WIN32 && defined MADV_POPULATE_WRITE
    int kv_dim = p->n_kv_heads * p->head_dim, l = it / 2, kv = it % 2;
    uintptr_t pg = (uintptr_t)sysconf(_SC_PAGESIZE);
    // positions pos0.. of every head: per block of ATT_P positions and head the tiles pos0 / ATT_T ..
    // (pos0 + n - 1) / ATT_T it holds (one range; adjacent ranges merged: whole blocks are contiguous)
    char *c = (char *)(kv ? s->value_cache : s->key_cache) + l * (uint64_t)s->kv_len * kv_dim * s->kv_esz;
    size_t esz = s->kv_esz;
    uintptr_t ra = 0, re = 0;
    for (int b = pos0 / ATT_P; b <= (pos0 + n - 1) / ATT_P; b++)
        for (int h = 0; h < p->n_kv_heads; h++) {
            int t0 = pos0 > b * ATT_P ? pos0 : b * ATT_P, t1 = pos0 + n < (b + 1) * ATT_P ? pos0 + n : (b + 1) * ATT_P;
            uintptr_t a = (uintptr_t)(c + kv_tile(t0, h, p->n_kv_heads, p->head_dim) * esz) & ~(pg - 1);
            uintptr_t e = (uintptr_t)(c + (kv_tile(t1 - 1, h, p->n_kv_heads, p->head_dim) + (size_t)ATT_T * p->head_dim) * esz);
            if (a > re) { // (a gap: populate the range so far)
                if (re > ra)
                    madvise((void *)ra, re - ra, MADV_POPULATE_WRITE);
                ra = a;
            }
            re = e;
        }
    madvise((void *)ra, re - ra, MADV_POPULATE_WRITE);
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
// (Items that wait on a dependency counter inside a phase need the shared counter: every item they
// depend on is claimed before them, by a running thread. With per-thread ranges and stealing, all
// threads can end up waiting in stolen items whose dependencies sit unclaimed in their own ranges:
// an intermediate version deadlocked that way at 4 threads.) The GEMV / GEMM phases use per-thread
// ranges: each thread streams a contiguous part of the weights (decode, 16-vCPU VM, best / median of
// 6: 4 threads 142 / 106 tok/s vs 125 / 111 with shared-counter claims of 1-4 blocks, 8 threads 212 /
// 189 vs 173 / 161, 12 threads 217 / 183 vs 199 / 176; prefill: claims of single tiles were ~15%
// slower than ranges)
#define PH_BEGIN_U(n_, chunk_, unit_) { RqIt it_ = rq_it(k_, (n_), (chunk_), (unit_), tid); int lo, hi; \
    while (rq_next(S, &it_, &lo, &hi)) {
#define PH_EPILOGUE if (rq_finish(S, &it_, hi - lo)) {
#define PH_END rq_publish(S, k_ + 1); } } rq_wait(S, ++k_); }
#define PH_END_NOEPI PH_EPILOGUE PH_END

// ---------------------------------------------------------------------------------------- decode

typedef struct { RqCtx *c; RqSeq *s; int pos; float *logits; } DecodeJob;

// phases of a decode job: per layer 4 (qkv + attention, wo, w1 + w3, w2), the classifier, the KV
// tile pre-fault every ATT_T positions; dependency counters: per (layer, kv head) its qkv items
// done, and its attention blocks done
static int decode_phases(const Config *p) { return 4 * p->n_layers + 2; }
static int decode_counters(const Config *p) { return 2 * p->n_layers * p->n_kv_heads; }

static void decode_norm(RqCtx *c, const Config *p, const TransformerWeights *w, int l, int resid) {
    // (x += xb,) rmsnorm with layer l's attention weights (l < n_layers) or the final ones (l ==
    // n_layers: into x itself, as before), quantize into xq, the GEMV activation prep
    int dim = p->dim;
    if (resid)
        for (int i = 0; i < dim; i++)
            c->x[i] += c->xb[i];
    if (l < p->n_layers) {
        rmsnorm(c->xb, c->x, w->rms_att_weight + l * dim, dim);
        quantize(&c->xq, c->xb, dim);
    } else {
        rmsnorm(c->x, c->x, w->rms_final_weight, dim);
        quantize(&c->xq, c->x, dim);
    }
    matvec_prep(c->xq_corr, c->xq_xf, c->xq_c16, &c->xq, 0, dim);
}

static void decode_worker(void *arg, int tid, int nt) {
    (void)nt;
    DecodeJob *job = arg;
    RqCtx *c = job->c;
    RqSeq *s = job->s;
    const RqModel *m = c->m;
    const Config *p = &m->config;
    TransformerWeights *w = (TransformerWeights *)&m->weights;
    RqSync *S = &c->sync;
    int pos = job->pos, k_ = 0;
    int dim = p->dim, hd = p->head_dim, hidden = p->hidden_dim, nh = p->n_heads, nkv = p->n_kv_heads;
    int kv_dim = nkv * hd;
    int kv_mul = nh / nkv; // integer multiplier of the kv sharing in multiquery
    int all_heads_dim = nh * hd, ff = m->ff_rows;
    int ng = kv_mul + 2; // qkv items per kv head: its kv_mul query heads, its key head, its value head
    int nb = (pos + 1 + ATT_P - 1) / ATT_P; // attention blocks (kernels_avx512.h): items per kv head
    size_t bs = (size_t)nkv * ATT_P * hd, pst = ATT_PSTRIDE(hd); // (cache: elements from a head's block to its next)
    float *conv = c->att_conv + (size_t)tid * 2 * ATT_P * hd;

    if (pos % ATT_T == 0) { // first position of a KV cache tile: pre-fault the tile (all layers)
        PH_BEGIN_DYN(2 * p->n_layers, 1)
        for (int it = lo; it < hi; it++)
            kv_populate_one(s, p, pos, 1, it);
        PH_END_NOEPI
    }

    // forward all the layers (layer 0's attention rmsnorm + quantize: done by decode_step)
    for (int l = 0; l < p->n_layers; l++) {
        uint64_t loff = l * (uint64_t)s->kv_len * kv_dim; // kv cache layer offset for convenience
        char *kl = (char *)s->key_cache + loff * s->kv_esz, *vl = (char *)s->value_cache + loff * s->kv_esz;

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
                    matvec_rows(c->q, &c->xq, c->xq_corr, c->xq_xf, c->xq_c16, w->wq + l, dim, h * hd, (h + 1) * hd);
                    qk_norm_rope(c->q + h * hd, 1, w->q_norm_weights + l * hd, hd, c->rope);
                } else if (r == kv_mul) {
                    matvec_rows(c->k, &c->xq, c->xq_corr, c->xq_xf, c->xq_c16, w->wk + l, dim, g * hd, (g + 1) * hd);
                    qk_norm_rope(c->k + g * hd, 1, w->k_norm_weights + l * hd, hd, c->rope);
                    kv_store(s->kvt, kl, NULL, pos, g, nkv, hd, c->k + g * hd, NULL);
                } else {
                    matvec_rows(c->v, &c->xq, c->xq_corr, c->xq_xf, c->xq_c16, w->wv + l, dim, g * hd, (g + 1) * hd);
                    kv_store(s->kvt, NULL, vl, pos, g, nkv, hd, NULL, c->v + g * hd);
                }
                rq_count(S, l * nkv + g, 1);
                continue;
            }
            int g = (it - nkv * ng) / nb, b = (it - nkv * ng) % nb;
            rq_wait_count(S, l * nkv + g, ng);
            const float *qv[ATT_NV_MAX];
            float *ov[ATT_NV_MAX], *pv[ATT_NV_MAX];
            int npv[ATT_NV_MAX];
            float *part = c->att_part + (size_t)g * kv_mul * c->att_nb * pst; // vector r, block b: + (r * att_nb + b) * pst
            for (int r = 0; r < kv_mul; r++) {
                qv[r] = c->q + (g * kv_mul + r) * hd;
                ov[r] = c->xb + (g * kv_mul + r) * hd;
                pv[r] = part + (size_t)r * c->att_nb * pst;
                npv[r] = pos + 1;
            }
            size_t o = kv_tile(0, g, nkv, hd);
            attention_part(s->kvt, kv_mul, qv, ov, nb == 1 ? NULL : pv, npv, b, b + 1, kl + o * s->kv_esz, vl + o * s->kv_esz, bs,
                           hd, m->att_scale, c->att + ((size_t)g * c->att_nb + b) * kv_mul * ATT_P, conv);
            if (nb > 1) { // the last block of this kv head to finish combines them
                if (rq_count(S, (p->n_layers + l) * nkv + g, 1) < nb)
                    continue;
                for (int r = 0; r < kv_mul; r++)
                    att_combine(ov[r], nb, pv[r], pst, hd);
            }
            int i0 = g * kv_mul * hd;
            QuantizedTensor ag = qsub(&c->aq, i0);
            quantize(&ag, c->xb + i0, kv_mul * hd);
            matvec_prep(c->aq_corr, c->aq_xf, c->aq_c16, &c->aq, i0, kv_mul * hd);
        }
        PH_END_NOEPI

        // final matmul to get the output of the attention (16-row blocks); epilogue: residual
        // connection back into x, ffn rmsnorm, quantize
        PH_BEGIN(dim / 16, 1)
            matvec_rows(c->xb, &c->aq, c->aq_corr, c->aq_xf, c->aq_c16, w->wo + l, all_heads_dim, lo * 16, hi * 16);
        PH_EPILOGUE
            for (int i = 0; i < dim; i++)
                c->x[i] += c->xb[i];
            rmsnorm(c->xb, c->x, w->rms_ffn_weight + l * dim, dim);
            quantize(&c->xq, c->xb, dim);
            matvec_prep(c->xq_corr, c->xq_xf, c->xq_c16, &c->xq, 0, dim);
        PH_END

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x)): per item,
        // ff rows of self.w1(x) and self.w3(x), the SwiGLU non-linearity and their quantization
        PH_BEGIN(hidden / ff, 1)
        for (int i0 = lo * ff; i0 < hi * ff; i0 += ff) {
            matvec_rows(c->hb, &c->xq, c->xq_corr, c->xq_xf, c->xq_c16, w->w1 + l, dim, i0, i0 + ff);
            matvec_rows(c->hb2, &c->xq, c->xq_corr, c->xq_xf, c->xq_c16, w->w3 + l, dim, i0, i0 + ff);
            swiglu(c->hb + i0, c->hb2 + i0, ff);
            QuantizedTensor hg = qsub(&c->hq, i0);
            quantize(&hg, c->hb + i0, ff);
            matvec_prep(c->hq_corr, c->hq_xf, c->hq_c16, &c->hq, i0, ff);
        }
        PH_END_NOEPI

        // final matmul to get the output of the ffn; epilogue: its residual, then the next layer's
        // attention rmsnorm + quantize (or the final rmsnorm for the classifier; none without logits)
        PH_BEGIN(dim / 16, 1)
            matvec_rows(c->xb, &c->hq, c->hq_corr, c->hq_xf, c->hq_c16, w->w2 + l, hidden, lo * 16, hi * 16);
        PH_EPILOGUE
            if (l + 1 < p->n_layers || job->logits)
                decode_norm(c, p, w, l + 1, 1);
        PH_END
    }

    if (job->logits) { // classifier into logits
        int nblk = (p->vocab_size + 15) / 16;
        PH_BEGIN(nblk, 8)
            matvec_rows(job->logits, &c->xq, c->xq_corr, c->xq_xf, c->xq_c16, w->wcls, dim, lo * 16,
                        hi * 16 < p->vocab_size ? hi * 16 : p->vocab_size);
        PH_END_NOEPI
    }
}

static int decode_step(RqCtx *c, RqSeq *s, int token, float *logits) {
    // token at position s->len (checked by the caller); logits NULL skips the final rmsnorm +
    // classifier (prompt tokens whose logits are discarded); the KV cache is filled either way.
    // One parallel job for the whole token (decode_worker), in phases (threads.h), 4 per layer:
    //   [q, k, v, one item per head, with its QK-norm + RoPE and the K/V cache store; attention per
    //   kv head (once the 4 items of its heads are done: dependency counters) + quantize] ->
    //   [wo; epilogue: residual + rmsnorm + quantize] -> [w1 + w3 + SwiGLU + quantize per ff_rows
    //   hidden rows] -> [w2; epilogue: residual + the next layer's rmsnorm + quantize], then the
    //   classifier: 4 * 28 + 1 = 113 synchronization points per token (was 7 * 28 + 2 = 198 OpenMP
    //   barriers). Every output is computed whole by one thread, by the same functions as before
    //   (same results)
    const Config *p = &c->m->config;
    int pos = s->len;
    // the token embedding into x
    embed_row(c->x, c->m->weights.q_tokens, token, p->dim);
    rope_row(c->rope, p->head_dim, pos, c->m->rope_log_theta);
    decode_norm(c, p, &c->m->weights, 0, 0); // layer 0's attention rmsnorm + quantize

    if (rq_begin(&c->sync, decode_phases(p), decode_counters(p), rq_nthreads(), rq_spin_dec) != 0)
        return ctx_fail(c, RQ_ERR_NOMEM, "malloc failed!");
    DecodeJob job = { c, s, pos, logits };
    rq_parallel(decode_worker, &job);
    s->tokens[pos] = token;
    s->len = pos + 1;
    return RQ_OK;
}

int rq_decode(RqCtx *c, RqSeq *s, int token, float **logits) {
    c->err[0] = 0;
    int e = ctx_check_seq(c, s);
    if (e != RQ_OK) return e;
    if (token < 0 || token >= c->m->config.vocab_size) return ctx_fail(c, RQ_ERR_ARG, "token %d out of range", token);
    if (s->len >= s->cap) return ctx_fail(c, RQ_ERR_CONTEXT, "sequence full (%d positions)", s->cap);
    e = decode_step(c, s, token, logits ? c->logits : NULL);
    if (logits) *logits = e == RQ_OK ? c->logits : NULL;
    return e;
}

// ---------------------------------------------------------------------------------- batched passes

static inline QuantizedTensor qrow(QuantizedTensor *t, int b, int ld) {
    // row b of a batch of quantized rows with row stride ld (a multiple of GS)
    return (QuantizedTensor) { .q = t->q + (size_t)b * ld, .s = t->s + (size_t)b * ld / GS, .p = NULL };
}

static void act_prep_row(QuantizedTensor *xq, int k, int32_t *corr, float *xf, int32_t *c16, int b) {
    // the GEMM activation prep of quantized row b (k values; the Gemm strides): matvec_prep
    matvec_prep(corr + (size_t)b * (k / GS), xf ? xf + (size_t)b * (k / GS) : NULL, c16 ? c16 + (size_t)b * (k / 16) : NULL,
                xq, 0, k);
}

// A batched pass: n rows in segments (rows of one sequence at consecutive positions), planned by
// rq_forward_batch into the context's row / tile / row-block / logits-row arrays
typedef struct {
    RqCtx *c;
    int n, nseg, ntile, nrb, nsel;
    int gemv;    // nsel == 1, logits by the decode GEMV (a last-row selection) into out
    float *out;  // logits: nsel rows of vocab_size
    int npop;    // segments starting a new KV cache tile (pop_seg)
    int nsp;     // row blocks (single rows) whose attention is split over its blocks (rb_sp, att_sp)
    int nbmax;   // blocks of ATT_P positions of the longest split row (att_sp layout)
} PrefillJob;

static int prefill_phases(const Config *p) { return 9 * p->n_layers + 4; }

#ifdef RUNQ_PROFILE
#define PPROF(i) do { if (tid == 0) PROF(c, i); } while (0)
#else
#define PPROF(i) ((void)0)
#endif

static void prefill_worker(void *arg, int tid, int nt) {
    // rq_forward_batch's job: per layer 9 phases (the first layer 8): [residual + rmsnorm + quantize per
    // row] -> [q / k / v GEMM] -> [QK-norm + RoPE + K/V cache store per 16-position tile] ->
    // [attention per (block of att_rows rows, kv head), longest first; the last head of a row block
    // to finish quantizes its rows for wo] -> [wo GEMM] -> [residual + rmsnorm + quantize per row]
    // -> [w1 / w3 GEMM] -> [SwiGLU + quantize per row] -> [w2 GEMM]; the first phase also embeds the
    // tokens and pre-faults the chunk's KV cache tiles. Each row is computed by the same functions
    // whatever the other rows are: a sequence's results do not depend on what it is batched with
    PrefillJob *job = arg;
    RqCtx *c = job->c;
    const RqModel *m = c->m;
    const Config *p = &m->config;
    TransformerWeights *w = (TransformerWeights *)&m->weights;
    RqSync *S = &c->sync;
    int n = job->n, k_ = 0;
    int dim = p->dim, hd = p->head_dim, hidden = p->hidden_dim, xs = m->xs, nkv = p->n_kv_heads;
    int kv_dim = nkv * hd;
    int kv_mul = p->n_heads / nkv;
    int all_heads_dim = p->n_heads * hd;
    int rch = n / (4 * nt) > 1 ? n / (4 * nt) : 1; // rows per claim in the per-row phases
    const PfSeg *seg = c->seg;
    size_t bs = (size_t)nkv * ATT_P * hd;
    uint8_t *scr = c->kq_scr ? c->kq_scr + (size_t)tid * c->kq_scr_bytes : NULL;
    float *att = c->att_b + (size_t)tid * m->att_rows * kv_mul * c->att_nb * ATT_P;
    float *part = c->att_pb + (size_t)tid * m->att_rows * kv_mul * c->att_nb * ATT_PSTRIDE(hd);
    float *conv = c->att_conv + (size_t)tid * 2 * ATT_P * hd;
    Gemm G;

    // embeddings, RoPE cos / sin per row, layer 0's attention rmsnorm + quantize; KV tile pre-faults
    PH_BEGIN(n + 2 * p->n_layers * job->npop, 1)
    for (int b = lo; b < hi; b++) {
        if (b >= n) {
            const PfSeg *sg = &seg[c->pop_seg[(b - n) / (2 * p->n_layers)]];
            kv_populate_one(sg->seq, p, sg->pos0, sg->n, (b - n) % (2 * p->n_layers));
            continue;
        }
        float *x = c->x_b + (size_t)b * dim, *xb = c->xb_b + (size_t)b * xs;
        QuantizedTensor xq = qrow(&c->xq_b, b, xs);
        embed_row(x, w->q_tokens, c->row_tok[b], dim);
        rope_row(c->rope_b + (size_t)b * hd, hd, c->row_pos[b], m->rope_log_theta);
        rmsnorm(xb, x, w->rms_att_weight, dim);
        quantize(&xq, xb, dim);
        act_prep_row(&xq, dim, c->bq_corr, c->bq_xf, c->bq_c16, b);
    }
    PH_END_NOEPI
    PPROF(PF_NORM);

    for (int l = 0; l < p->n_layers; l++) {
        uint64_t loff = l * (uint64_t)kv_dim; // (times the sequence's kv_len)

        if (l > 0) { // residual of the previous layer's ffn, attention rmsnorm + quantize, per row
            PH_BEGIN(n, rch)
            for (int b = lo; b < hi; b++) {
                float *x = c->x_b + (size_t)b * dim, *xb = c->xb_b + (size_t)b * xs;
                QuantizedTensor xq = qrow(&c->xq_b, b, xs);
                for (int i = 0; i < dim; i++)
                    x[i] += xb[i];
                rmsnorm(xb, x, w->rms_att_weight + l * dim, dim);
                quantize(&xq, xb, dim);
                act_prep_row(&xq, dim, c->bq_corr, c->bq_xf, c->bq_c16, b);
            }
            PH_END_NOEPI
            PPROF(PF_NORM);
        }

        // qkv GEMMs (one phase)
        MatJob qkv[3] = { { c->q_b, all_heads_dim, w->wq + l, all_heads_dim }, { c->k_b, kv_dim, w->wk + l, kv_dim },
                          { c->v_b, kv_dim, w->wv + l, kv_dim } };
        gemm_plan(c, &G, 3, qkv, &c->xq_b, xs, n, dim, c->bq_corr, c->bq_xf, c->bq_c16, nt);
        {
            const void *last = NULL;
            PH_BEGIN_U(G.it0[G.nj], G.chunk, G.unit)
            for (int it = lo; it < hi; it++)
                gemm_item(&G, it, scr, &last);
            PH_END_NOEPI
        }
        PPROF(PF_QKV);

        // QK-norm + RoPE, each row at its own position, and the K / V rows into the cache: one item per
        // (16-position cache tile of one segment, kv head): its rows' query heads of that kv head, its key
        // head (+ store) and value head (store). (Per tile only: a short chunk, e.g. 16 rows next to decode
        // rows, ran on one thread, ~1.5 ms of a 16-row pass at 12 threads)
        PH_BEGIN(job->ntile * nkv, 1)
        for (int it = lo; it < hi; it++) {
            const PfSeg *sg = &seg[c->tile_seg[it / nkv]];
            RqSeq *sq = sg->seq;
            char *kl = (char *)sq->key_cache + loff * sq->kv_len * sq->kv_esz;
            char *vl = (char *)sq->value_cache + loff * sq->kv_len * sq->kv_esz;
            int tb = c->tile_tb[it / nkv], pos0 = sg->pos0, g = it % nkv;
            int t1 = (tb + 1) * ATT_T < pos0 + sg->n ? (tb + 1) * ATT_T : pos0 + sg->n;
            for (int t = tb * ATT_T > pos0 ? tb * ATT_T : pos0; t < t1; t++) {
                int b = sg->b0 + t - pos0;
                const float *cs = c->rope_b + (size_t)b * hd;
                float *kr = c->k_b + (size_t)b * kv_dim + g * hd;
                qk_norm_rope(c->q_b + (size_t)b * all_heads_dim + g * kv_mul * hd, kv_mul, w->q_norm_weights + l * hd, hd, cs);
                qk_norm_rope(kr, 1, w->k_norm_weights + l * hd, hd, cs);
                kv_store(sq->kvt, kl, vl, t, g, nkv, hd, kr, c->v_b + (size_t)b * kv_dim + g * hd);
            }
        }
        PH_END_NOEPI
        PPROF(PF_ROPE);

        // causal attention: row b (position pos of its sequence) attends to positions 0..pos of that
        // sequence, whose K/V are all in its cache now. Work item = (row block: up to att_rows rows of
        // one segment, kv head): its rows x kv_mul query heads in one attention_kv call (K/V read once
        // for all of them); a split row block (a single row, few items in the pass) has one item per
        // (kv head, block of ATT_P positions), the last block of a kv head to finish combines (as in
        // decode). Longest items first, for the load balance. The last of a row block's kv heads to
        // finish quantizes its rows for wo (into xq_b: the qkv GEMM is done with it)
        int ncl = job->nrb + job->nsp * nkv; // dependency counters per layer: row blocks, (split row block, kv head)
        PH_BEGIN_DYN(c->rb_it0[job->nrb], 1)
        for (int it = lo; it < hi; it++) {
            int rb = 0, r1 = job->nrb; // the row block holding item it: rb_it0[rb] <= it < rb_it0[rb + 1]
            while (r1 - rb > 1) {
                int mid = (rb + r1) / 2;
                if (c->rb_it0[mid] <= it) rb = mid; else r1 = mid;
            }
            int i = it - c->rb_it0[rb], sp = c->rb_sp[rb];
            RqSeq *sq = seg[c->rb_seg[rb]].seq;
            char *kl = (char *)sq->key_cache + loff * sq->kv_len * sq->kv_esz;
            char *vl = (char *)sq->value_cache + loff * sq->kv_len * sq->kv_esz;
            int b0 = c->rb_b0[rb], b1 = c->rb_b1[rb], nv = 0, nbk = (c->row_pos[b0] + 1 + ATT_P - 1) / ATT_P;
            int g = sp >= 0 ? i / nbk : i;
            const float *qv[ATT_NV_MAX];
            float *ov[ATT_NV_MAX], *pv[ATT_NV_MAX];
            int npv[ATT_NV_MAX];
            for (int b = b0; b < b1; b++)
                for (int r = 0; r < kv_mul; r++, nv++) {
                    int h = g * kv_mul + r;
                    qv[nv] = c->q_b + (size_t)b * all_heads_dim + h * hd;
                    ov[nv] = c->xb_b + (size_t)b * xs + h * hd;
                    npv[nv] = c->row_pos[b] + 1;
                    if (sp >= 0)
                        pv[nv] = c->att_sp + ((size_t)sp * p->n_heads + h) * job->nbmax * ATT_PSTRIDE(hd);
                }
            size_t o = kv_tile(0, g, nkv, hd);
            if (sp >= 0) { // block j of the single row b0
                int j = i % nbk;
                attention_part(sq->kvt, nv, qv, ov, pv, npv, j, j + 1, kl + o * sq->kv_esz, vl + o * sq->kv_esz, bs, hd, m->att_scale,
                               att, conv);
                if (rq_count(S, l * ncl + job->nrb + sp * nkv + g, 1) < nbk)
                    continue;
                for (int r = 0; r < nv; r++)
                    att_combine(ov[r], nbk, pv[r], ATT_PSTRIDE(hd), hd);
            } else {
                attention_kv(sq->kvt, nv, qv, ov, npv, kl + o * sq->kv_esz, vl + o * sq->kv_esz, bs, hd, m->att_scale, att, part,
                             conv);
            }
            if (rq_count(S, l * ncl + rb, 1) < nkv)
                continue;
            for (int b = b0; b < b1; b++) {
                QuantizedTensor xq = qrow(&c->xq_b, b, xs);
                quantize(&xq, c->xb_b + (size_t)b * xs, all_heads_dim);
                act_prep_row(&xq, all_heads_dim, c->bq_corr, c->bq_xf, c->bq_c16, b);
            }
        }
        PH_END_NOEPI
        PPROF(PF_ATT);

        MatJob wo = { c->xb_b, xs, w->wo + l, dim };
        gemm_plan(c, &G, 1, &wo, &c->xq_b, xs, n, all_heads_dim, c->bq_corr, c->bq_xf, c->bq_c16, nt);
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
            float *x = c->x_b + (size_t)b * dim, *xb = c->xb_b + (size_t)b * xs;
            QuantizedTensor xq = qrow(&c->xq_b, b, xs);
            for (int i = 0; i < dim; i++)
                x[i] += xb[i];
            rmsnorm(xb, x, w->rms_ffn_weight + l * dim, dim);
            quantize(&xq, xb, dim);
            act_prep_row(&xq, dim, c->bq_corr, c->bq_xf, c->bq_c16, b);
        }
        PH_END_NOEPI
        PPROF(PF_NORM);

        MatJob w13[2] = { { c->hb_b, hidden, w->w1 + l, hidden }, { c->hb2_b, hidden, w->w3 + l, hidden } };
        gemm_plan(c, &G, 2, w13, &c->xq_b, xs, n, dim, c->bq_corr, c->bq_xf, c->bq_c16, nt);
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
            QuantizedTensor hq = qrow(&c->hq_b, b, hidden);
            swiglu(c->hb_b + (size_t)b * hidden, c->hb2_b + (size_t)b * hidden, hidden);
            quantize(&hq, c->hb_b + (size_t)b * hidden, hidden);
            act_prep_row(&hq, hidden, c->bh_corr, c->bh_xf, c->bh_c16, b);
        }
        PH_END_NOEPI
        PPROF(PF_SWIGLU);

        MatJob w2 = { c->xb_b, xs, w->w2 + l, dim };
        gemm_plan(c, &G, 1, &w2, &c->hq_b, hidden, n, hidden, c->bh_corr, c->bh_xf, c->bh_c16, nt);
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

    if (job->gemv) {
        // one row's logits: its residual, final rmsnorm + classifier by the decode path's GEMV
        int r = c->sel_row[0];
        PH_BEGIN(1, 1)
            float *x = c->x_b + (size_t)r * dim, *xb = c->xb_b + (size_t)r * xs;
            for (int i = 0; i < dim; i++)
                x[i] += xb[i];
            rmsnorm(c->x, x, w->rms_final_weight, dim);
            quantize(&c->xq, c->x, dim);
            matvec_prep(c->xq_corr, c->xq_xf, c->xq_c16, &c->xq, 0, dim);
        PH_END_NOEPI
        int nblk = (p->vocab_size + 15) / 16, cc = nblk / (8 * nt) > 1 ? nblk / (8 * nt) : 1;
        PH_BEGIN(nblk, cc)
            matvec_rows(job->out, &c->xq, c->xq_corr, c->xq_xf, c->xq_c16, w->wcls, dim, lo * 16,
                        hi * 16 < p->vocab_size ? hi * 16 : p->vocab_size);
        PH_END_NOEPI
        PPROF(PF_CLS);
    } else if (job->nsel > 0) {
        // the selected rows (sel_row[j] -> logits row j): residual, final rmsnorm, quantize into
        // xq_b row j (the w2 GEMM read hq_b: xq_b is free), then the classifier GEMM over them
        int nsel = job->nsel, sch = nsel / (4 * nt) > 1 ? nsel / (4 * nt) : 1;
        PH_BEGIN(nsel, sch)
        for (int j = lo; j < hi; j++) {
            int b = c->sel_row[j];
            float *x = c->x_b + (size_t)b * dim, *xb = c->xb_b + (size_t)b * xs;
            QuantizedTensor xq = qrow(&c->xq_b, j, xs);
            for (int i = 0; i < dim; i++)
                x[i] += xb[i];
            rmsnorm(xb, x, w->rms_final_weight, dim);
            quantize(&xq, xb, dim);
            act_prep_row(&xq, dim, c->bq_corr, c->bq_xf, c->bq_c16, j);
        }
        PH_END_NOEPI
        MatJob cls = { job->out, p->vocab_size, w->wcls, p->vocab_size };
        gemm_plan(c, &G, 1, &cls, &c->xq_b, xs, nsel, dim, c->bq_corr, c->bq_xf, c->bq_c16, nt);
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

static int prefill_plan(RqCtx *c, int nseg, int nrb) {
    // the attention items of prefill_worker: per row block nkv items, or nkv x its blocks of ATT_P positions
    // for a split row block (a single row reaching past one block, when the pass has fewer than bd_split x
    // threads row-block items, e.g. a few decode rows next to a short chunk); the row blocks ordered by
    // the cost of their items, largest first (stable: one segment keeps its longest-first order).
    // Returns the number of split row blocks
    int nkv = c->m->config.n_kv_heads, nsp = 0, split = nrb * nkv < c->bd_split * rq_nthreads();
    (void)nseg;
    for (int rb = 0; rb < nrb; rb++) { // (rb_it0 holds the sort key meanwhile)
        int last = c->row_pos[c->rb_b1[rb] - 1] + 1, single = c->rb_b1[rb] - c->rb_b0[rb] == 1;
        c->rb_sp[rb] = split && single && last > ATT_P ? 0 : -1;
        int key = c->rb_sp[rb] >= 0 ? ATT_P : (c->rb_b1[rb] - c->rb_b0[rb]) * last, k = rb;
        int sg = c->rb_seg[rb], b0 = c->rb_b0[rb], b1 = c->rb_b1[rb], spf = c->rb_sp[rb];
        for (; k > 0 && c->rb_it0[k - 1] < key; k--) {
            c->rb_it0[k] = c->rb_it0[k - 1];
            c->rb_seg[k] = c->rb_seg[k - 1];
            c->rb_b0[k] = c->rb_b0[k - 1];
            c->rb_b1[k] = c->rb_b1[k - 1];
            c->rb_sp[k] = c->rb_sp[k - 1];
        }
        c->rb_it0[k] = key;
        c->rb_seg[k] = sg;
        c->rb_b0[k] = b0;
        c->rb_b1[k] = b1;
        c->rb_sp[k] = spf;
    }
    int it = 0;
    for (int rb = 0; rb < nrb; rb++) {
        c->rb_it0[rb] = it;
        if (c->rb_sp[rb] >= 0) {
            c->rb_sp[rb] = nsp++;
            it += nkv * ((c->row_pos[c->rb_b0[rb]] + ATT_P) / ATT_P);
        } else {
            it += nkv;
        }
    }
    c->rb_it0[nrb] = it;
    return nsp;
}

// ------------------------------------------------------------------------------ batched decode rows

static void rows_mm(QuantizedTensor *w, float *out, int ldo, QuantizedTensor *x, int ldx, const int32_t *corr, const float *xf,
                    const int32_t *c16, int nb, int k, int i0, int i1) {
    // out[b * ldo + i] = W row i . x row b for i0 <= i < i1 (multiples of 16), b < nb (x row b at x->q +
    // b * ldx; the activation prep of act_prep_row): the small-batch kernel, else the decode GEMV per row.
    // Either way each output is the GEMV's (bit-identical)
#ifdef RUNQ_SMB
    if (w->kp || w->p) {
        smb_rows(w->kp ? w->type : 0, out, ldo, x->q, ldx, x->s, corr, xf, c16, nb, w->kp ? w->kp : w->p, k, GS, i0 / 16, i1 / 16);
        return;
    }
#endif
    for (int b = 0; b < nb; b++) {
        QuantizedTensor xr = qrow(x, b, ldx);
        matvec_rows(out + (size_t)b * ldo, &xr, corr + (size_t)b * (k / GS), xf ? xf + (size_t)b * (k / GS) : NULL,
                    c16 ? c16 + (size_t)b * (k / 16) : NULL, w, k, i0, i1);
    }
}

typedef struct {
    RqCtx *c;
    int n;       // rows = segments, one row each (row b = segment b, at position row_pos[b] of seg[b].seq)
    int nsel;    // rows with logits (sel_row), into out
    float *out;
    int epi;     // the norms as phase epilogues (else phases of their own)
    int split;   // attention split over blocks of ATT_P positions (row_it0: items per row and kv head)
    int nbmax;   // blocks of the longest row (att_sp layout)
} BatchDecJob;

static int bd_phases(const Config *p, int epi) { return (epi ? 4 : 6) * p->n_layers + 4; }
// per layer: per kv head its q / k / v items done, its QK-norm + RoPE + stores done; per row its kv heads'
// attention done; per (row, kv head) its blocks done (split attention)
static int bd_counters(int nkv, int n) { return 2 * nkv + n * (1 + nkv); }
#ifndef BD_QKV_ROWS
#define BD_QKV_ROWS 64 // weight rows per q / k / v item (a multiple of 16 dividing head_dim; else whole heads)
#endif

static void bd_norm_row(RqCtx *c, const Config *p, const TransformerWeights *w, int b, int l, int resid, int dst) {
    // row b: (x += xb,) rmsnorm with layer l's attention weights (l < n_layers; l = -1 - l2: layer l2's ffn
    // weights; l == n_layers: the final ones), quantized into xq_b row dst + its activation prep (as
    // prefill_worker's per-row phases)
    int dim = p->dim, xs = c->m->xs;
    float *x = c->x_b + (size_t)b * dim, *xb = c->xb_b + (size_t)b * xs;
    QuantizedTensor xq = qrow(&c->xq_b, dst, xs);
    if (resid)
        for (int i = 0; i < dim; i++)
            x[i] += xb[i];
    const float *wt = l < 0 ? w->rms_ffn_weight + (size_t)(-1 - l) * dim : l < p->n_layers ? w->rms_att_weight + (size_t)l * dim
                                                                                           : w->rms_final_weight;
    rmsnorm(xb, x, (float *)wt, dim);
    quantize(&xq, xb, dim);
    act_prep_row(&xq, dim, c->bq_corr, c->bq_xf, c->bq_c16, dst);
}

static void batch_decode_worker(void *arg, int tid, int nt) {
    // rq_forward_batch over N decode rows of N sequences: decode_worker's structure for N rows at once,
    // per layer 4 phases (6 with the norms as phases): [q, k, v per head over all rows (rows_mm) + QK-norm +
    // RoPE + each row's K / V into its sequence's cache; attention per (row, kv head) or (row, kv head,
    // block), once the 4 items of its kv head are done (dependency counters); the last kv head of a row
    // quantizes its output] [wo; residual + rmsnorm + quantize per row] [w1 + w3 + SwiGLU + quantize per
    // ff_rows hidden rows, all rows] [w2; residual + the next rmsnorm + quantize per row], then the
    // classifier over the rows with logits. Every output is computed whole by the same functions as in
    // decode_worker / prefill_worker: each row's results are those of its sequence decoded alone
    (void)nt;
    BatchDecJob *job = arg;
    RqCtx *c = job->c;
    const RqModel *m = c->m;
    const Config *p = &m->config;
    TransformerWeights *w = (TransformerWeights *)&m->weights;
    RqSync *S = &c->sync;
    const PfSeg *seg = c->seg;
    int n = job->n, k_ = 0;
    int dim = p->dim, hd = p->head_dim, hidden = p->hidden_dim, xs = m->xs, nkv = p->n_kv_heads, nh = p->n_heads;
    int kv_dim = nkv * hd, kv_mul = nh / nkv, ahd = nh * hd, ff = m->ff_rows;
    int qr = hd % BD_QKV_ROWS ? hd : BD_QKV_ROWS, ng = (kv_mul + 2) * hd / qr; // q / k / v items per kv head, qr rows each
    int ncl = bd_counters(nkv, n); // dependency counters per layer (bd_counters)
    int nat = job->split ? c->row_it0[n] : n; // attention items per kv head
    size_t bs = (size_t)nkv * ATT_P * hd, pst = ATT_PSTRIDE(hd);
    float *att = c->att_b + (size_t)tid * m->att_rows * kv_mul * c->att_nb * ATT_P;
    float *part = c->att_pb + (size_t)tid * m->att_rows * kv_mul * c->att_nb * pst;
    float *conv = c->att_conv + (size_t)tid * 2 * ATT_P * hd;

    // embeddings, RoPE cos / sin per row, layer 0's attention rmsnorm + quantize; the KV tile pre-fault of
    // the rows starting a new tile (row_tok of the others: -1)
    int npop = 0;
    for (int b = 0; b < n; b++)
        npop += c->row_pos[b] % ATT_T == 0;
    PH_BEGIN(n + 2 * p->n_layers * npop, 1)
    for (int it = lo; it < hi; it++) {
        if (it >= n) {
            int q = (it - n) / (2 * p->n_layers), b = 0;
            for (;; b++)
                if (c->row_pos[b] % ATT_T == 0 && q-- == 0) break;
            kv_populate_one(seg[b].seq, p, c->row_pos[b], 1, (it - n) % (2 * p->n_layers));
            continue;
        }
        embed_row(c->x_b + (size_t)it * dim, w->q_tokens, c->row_tok[it], dim);
        rope_row(c->rope_b + (size_t)it * hd, hd, c->row_pos[it], m->rope_log_theta);
        bd_norm_row(c, p, w, it, 0, 0, it);
    }
    PH_END_NOEPI
    PPROF(PF_NORM);

    for (int l = 0; l < p->n_layers; l++) {
        uint64_t loff = l * (uint64_t)kv_dim; // (times the sequence's kv_len)
        int cb = l * ncl;                     // this layer's dependency counters

        // q / k / v over all rows in items of qr weight rows (the kv head's query heads, its key head, its
        // value head: ng items per kv head); the last of a kv head's items to finish runs its QK-norm +
        // RoPE and the K / V stores of every row, then the attention items of that kv head may start
        PH_BEGIN_DYN(nkv * ng + nkv * nat, 1)
        for (int it = lo; it < hi; it++) {
            // (the attention items of each kv head interleaved after the next head's q / k / v items instead:
            // ~5-10% slower at 8 decode rows, 8 and 14 threads)
            if (it < nkv * ng) {
                int g = it / ng, r = (it % ng) * qr, nq = kv_mul * hd; // r: row offset in [q heads | k | v] of head g
                if (r < nq)
                    rows_mm(w->wq + l, c->q_b, ahd, &c->xq_b, xs, c->bq_corr, c->bq_xf, c->bq_c16, n, dim, g * nq + r, g * nq + r + qr);
                else if (r < nq + hd)
                    rows_mm(w->wk + l, c->k_b, kv_dim, &c->xq_b, xs, c->bq_corr, c->bq_xf, c->bq_c16, n, dim, g * hd + r - nq,
                            g * hd + r - nq + qr);
                else
                    rows_mm(w->wv + l, c->v_b, kv_dim, &c->xq_b, xs, c->bq_corr, c->bq_xf, c->bq_c16, n, dim, g * hd + r - nq - hd,
                            g * hd + r - nq - hd + qr);
                if (rq_count(S, cb + g, 1) < ng)
                    continue;
                for (int b = 0; b < n; b++) {
                    RqSeq *sq = seg[b].seq;
                    const float *cs = c->rope_b + (size_t)b * hd;
                    float *kr = c->k_b + (size_t)b * kv_dim + g * hd;
                    qk_norm_rope(c->q_b + (size_t)b * ahd + g * nq, kv_mul, w->q_norm_weights + l * hd, hd, cs);
                    qk_norm_rope(kr, 1, w->k_norm_weights + l * hd, hd, cs);
                    kv_store(sq->kvt, (char *)sq->key_cache + loff * sq->kv_len * sq->kv_esz,
                             (char *)sq->value_cache + loff * sq->kv_len * sq->kv_esz, c->row_pos[b], g, nkv, hd, kr,
                             c->v_b + (size_t)b * kv_dim + g * hd);
                }
                rq_count(S, cb + nkv + g, 1);
                continue;
            }
            // attention item: kv head g, row b (rows longest first), block j of the row if split
            int a = it - nkv * ng, g = a / nat, i = a % nat, b = i, j = 0;
            if (job->split) {
                int lo_ = 0, hi_ = n; // the row whose items hold i: row_it0[b] <= i < row_it0[b + 1]
                while (hi_ - lo_ > 1) {
                    int mid = (lo_ + hi_) / 2;
                    if (c->row_it0[mid] <= i) lo_ = mid; else hi_ = mid;
                }
                b = lo_;
                j = i - c->row_it0[b];
            }
            rq_wait_count(S, cb + nkv + g, 1);
            RqSeq *sq = seg[b].seq;
            const char *kl = (const char *)sq->key_cache + loff * sq->kv_len * sq->kv_esz;
            const char *vl = (const char *)sq->value_cache + loff * sq->kv_len * sq->kv_esz;
            size_t o = kv_tile(0, g, nkv, hd) * sq->kv_esz;
            int np = c->row_pos[b] + 1, nbk = (np + ATT_P - 1) / ATT_P;
            const float *qv[ATT_NV_MAX];
            float *ov[ATT_NV_MAX], *pv[ATT_NV_MAX];
            int npv[ATT_NV_MAX];
            for (int r = 0; r < kv_mul; r++) {
                qv[r] = c->q_b + (size_t)b * ahd + (g * kv_mul + r) * hd;
                ov[r] = c->xb_b + (size_t)b * xs + (g * kv_mul + r) * hd;
                pv[r] = c->att_sp + ((size_t)b * nh + g * kv_mul + r) * job->nbmax * pst;
                npv[r] = np;
            }
            if (job->split && nbk > 1) { // block j's partial results; the last block of (row, kv head) combines
                attention_part(sq->kvt, kv_mul, qv, ov, pv, npv, j, j + 1, kl + o, vl + o, bs, hd, m->att_scale, att, conv);
                if (rq_count(S, cb + 2 * nkv + n + b * nkv + g, 1) < nbk)
                    continue;
                for (int r = 0; r < kv_mul; r++)
                    att_combine(ov[r], nbk, pv[r], pst, hd);
            } else {
                attention_kv(sq->kvt, kv_mul, qv, ov, npv, kl + o, vl + o, bs, hd, m->att_scale, att, part, conv);
            }
            if (rq_count(S, cb + 2 * nkv + b, 1) < nkv)
                continue;
            // the row's last kv head: its attention output quantized for wo (xq_b: every qkv item is done)
            QuantizedTensor xq = qrow(&c->xq_b, b, xs);
            quantize(&xq, c->xb_b + (size_t)b * xs, ahd);
            act_prep_row(&xq, ahd, c->bq_corr, c->bq_xf, c->bq_c16, b);
        }
        PH_END_NOEPI
        PPROF(PF_BDQKV);

        // wo (16-row blocks, all rows); residual + ffn rmsnorm + quantize per row
        if (job->epi) {
            PH_BEGIN(dim / 16, 1)
                rows_mm(w->wo + l, c->xb_b, xs, &c->xq_b, xs, c->bq_corr, c->bq_xf, c->bq_c16, n, ahd, lo * 16, hi * 16);
            PH_EPILOGUE
                for (int b = 0; b < n; b++)
                    bd_norm_row(c, p, w, b, -1 - l, 1, b);
            PH_END
            PPROF(PF_WO);
        } else {
            PH_BEGIN(dim / 16, 1)
                rows_mm(w->wo + l, c->xb_b, xs, &c->xq_b, xs, c->bq_corr, c->bq_xf, c->bq_c16, n, ahd, lo * 16, hi * 16);
            PH_END_NOEPI
            PPROF(PF_WO);
            PH_BEGIN(n, 1)
            for (int b = lo; b < hi; b++)
                bd_norm_row(c, p, w, b, -1 - l, 1, b);
            PH_END_NOEPI
            PPROF(PF_NORM);
        }

        // w1 + w3 + SwiGLU + quantize per ff hidden rows, all rows
        PH_BEGIN(hidden / ff, 1)
        for (int i0 = lo * ff; i0 < hi * ff; i0 += ff) {
            rows_mm(w->w1 + l, c->hb_b, hidden, &c->xq_b, xs, c->bq_corr, c->bq_xf, c->bq_c16, n, dim, i0, i0 + ff);
            rows_mm(w->w3 + l, c->hb2_b, hidden, &c->xq_b, xs, c->bq_corr, c->bq_xf, c->bq_c16, n, dim, i0, i0 + ff);
            for (int b = 0; b < n; b++) {
                float *hb = c->hb_b + (size_t)b * hidden;
                swiglu(hb + i0, c->hb2_b + (size_t)b * hidden + i0, ff);
                QuantizedTensor hr = qrow(&c->hq_b, b, hidden), hg = qsub(&hr, i0);
                quantize(&hg, hb + i0, ff);
                matvec_prep(c->bh_corr + (size_t)b * (hidden / GS), c->bh_xf ? c->bh_xf + (size_t)b * (hidden / GS) : NULL,
                            c->bh_c16 ? c->bh_c16 + (size_t)b * (hidden / 16) : NULL, &hr, i0, ff);
            }
        }
        PH_END_NOEPI
        PPROF(PF_BDFFN13);

        // w2; its residual + the next layer's attention rmsnorm (the last layer: the final rmsnorm of the
        // rows with logits, into xq_b rows 0..nsel-1) + quantize
        int last = l + 1 == p->n_layers, nr = last ? job->nsel : n;
        if (job->epi) {
            PH_BEGIN(dim / 16, 1)
                rows_mm(w->w2 + l, c->xb_b, xs, &c->hq_b, hidden, c->bh_corr, c->bh_xf, c->bh_c16, n, hidden, lo * 16, hi * 16);
            PH_EPILOGUE
                for (int j = 0; j < nr; j++)
                    bd_norm_row(c, p, w, last ? c->sel_row[j] : j, l + 1, 1, j);
            PH_END
            PPROF(PF_FFN2);
        } else {
            PH_BEGIN(dim / 16, 1)
                rows_mm(w->w2 + l, c->xb_b, xs, &c->hq_b, hidden, c->bh_corr, c->bh_xf, c->bh_c16, n, hidden, lo * 16, hi * 16);
            PH_END_NOEPI
            PPROF(PF_FFN2);
            if (nr > 0) {
                PH_BEGIN(nr, 1)
                for (int j = lo; j < hi; j++)
                    bd_norm_row(c, p, w, last ? c->sel_row[j] : j, l + 1, 1, j);
                PH_END_NOEPI
                PPROF(PF_NORM);
            }
        }
    }

    if (job->nsel > 0) { // classifier over the rows with logits (xq_b rows 0..nsel-1)
        // (a vocabulary that is not a multiple of 16: the generic kernels' last block is partial, as in decode)
        int nblk = (p->vocab_size + 15) / 16, cc = nblk / (8 * nt) > 1 ? nblk / (8 * nt) : 1;
        PH_BEGIN(nblk, cc)
            rows_mm(w->wcls, job->out, p->vocab_size, &c->xq_b, xs, c->bq_corr, c->bq_xf, c->bq_c16, job->nsel, dim, lo * 16,
                    hi * 16 < p->vocab_size ? hi * 16 : p->vocab_size);
        PH_END_NOEPI
        PPROF(PF_CLS);
    }
}

static int batch_decode(RqCtx *c, int n, int nsel, float *out) {
    // rq_forward_batch's plan (segments of one row each) -> batch_decode_worker; the rows' attention split
    // over blocks if the (row, kv head) items alone are few for the threads
    const Config *p = &c->m->config;
    int nkv = p->n_kv_heads, nbmax = 1, nt = rq_nthreads();
    for (int b = 0; b < n; b++) {
        int nbk = (c->row_pos[b] + 1 + ATT_P - 1) / ATT_P;
        nbmax = nbk > nbmax ? nbk : nbmax;
    }
    BatchDecJob job = { c, n, nsel, out, n <= c->bd_epi, nbmax > 1 && n * nkv < c->bd_split * nt, nbmax };
    if (job.split) {
        size_t need = (size_t)n * p->n_heads * nbmax * ATT_PSTRIDE(p->head_dim);
        if (need > c->att_sp_n) {
            float *sp = malloc(need * sizeof(float));
            if (!sp) return ctx_fail(c, RQ_ERR_NOMEM, "malloc failed!");
            free(c->att_sp);
            c->att_sp = sp;
            c->att_sp_n = need;
        }
        c->row_it0[0] = 0;
        for (int b = 0; b < n; b++)
            c->row_it0[b + 1] = c->row_it0[b] + (c->row_pos[b] + 1 + ATT_P - 1) / ATT_P;
    }
    PROF_START(c);
    if (rq_begin(&c->sync, bd_phases(p, job.epi), p->n_layers * bd_counters(nkv, n), nt, rq_spin_dec) != 0)
        return ctx_fail(c, RQ_ERR_NOMEM, "malloc failed!");
    rq_parallel(batch_decode_worker, &job);
    return RQ_OK;
}

int rq_forward_batch(RqCtx *c, const RqSegment *sg, int nseg, float *out, float **logits) {
    // One batched pass over the rows of nseg segments: the same computation per row as rq_decode on
    // its sequence, with every weight matrix applied to all rows at once (GEMM / small-batch work
    // items), and bit-identical results. Fills the KV caches. One parallel job in phases (threads.h):
    // batch_decode_worker if every segment is one row (decode rows, up to bd_max), else prefill_worker
    const RqModel *m = c->m;
    const Config *p = &m->config;
    int ar = m->att_rows, n = 0, ntile = 0, nrb = 0, nsel = 0, any_all = 0;
    c->err[0] = 0;
    if (logits) *logits = NULL;
    if (c->nb_max <= 1) return ctx_fail(c, RQ_ERR_STATE, "context without batch buffers (max_batch %d)", c->nb_max);
    if (nseg < 1 || nseg > c->nb_max) return ctx_fail(c, RQ_ERR_ARG, "bad segment count %d", nseg);
    for (int i = 0; i < nseg; i++) {
        RqSeq *s = sg[i].seq;
        int e = s ? ctx_check_seq(c, s) : ctx_fail(c, RQ_ERR_ARG, "segment %d: no sequence", i);
        if (e != RQ_OK) return e;
        if (sg[i].n < 1 || n + sg[i].n > c->nb_max || sg[i].logits < RQ_LOGITS_NONE || sg[i].logits > RQ_LOGITS_ALL)
            return ctx_fail(c, RQ_ERR_ARG, "rq_forward_batch: bad segment %d (n %d, %d rows so far, max_batch %d)", i,
                            sg[i].n, n, c->nb_max);
        if (s->len + sg[i].n > s->cap)
            return ctx_fail(c, RQ_ERR_CONTEXT, "rq_forward_batch: %d tokens at position %d past the capacity %d", sg[i].n,
                            s->len, s->cap);
        for (int j = 0; j < i; j++)
            if (sg[j].seq == s) return ctx_fail(c, RQ_ERR_ARG, "sequence in segments %d and %d", j, i);
        for (int j = 0; j < sg[i].n; j++)
            if (sg[i].tokens[j] < 0 || sg[i].tokens[j] >= p->vocab_size)
                return ctx_fail(c, RQ_ERR_ARG, "token %d out of range", sg[i].tokens[j]);
        n += sg[i].n;
        nsel += sg[i].logits == RQ_LOGITS_ALL ? sg[i].n : sg[i].logits == RQ_LOGITS_LAST;
        any_all |= sg[i].logits == RQ_LOGITS_ALL;
    }
    int gemv = nsel == 1 && !any_all && !out;
    if (nsel > 0 && !out && !gemv) return ctx_fail(c, RQ_ERR_ARG, "logits of %d rows need an output buffer", nsel);
    if (gemv) out = c->logits;

    // the plan: rows, segments, tiles, row blocks, logits rows
    n = 0;
    for (int i = 0; i < nseg; i++) {
        PfSeg *s = &c->seg[i];
        s->seq = sg[i].seq;
        s->b0 = n;
        s->n = sg[i].n;
        s->pos0 = s->seq->len;
        for (int j = 0; j < s->n; j++) {
            c->row_tok[n + j] = sg[i].tokens[j];
            c->row_pos[n + j] = s->pos0 + j;
        }
        for (int tb = s->pos0 / ATT_T; tb <= (s->pos0 + s->n - 1) / ATT_T; tb++, ntile++) {
            c->tile_seg[ntile] = i;
            c->tile_tb[ntile] = tb;
        }
        for (int rb = (s->n + ar - 1) / ar - 1; rb >= 0; rb--, nrb++) { // (one segment: longest first)
            c->rb_seg[nrb] = i;
            c->rb_b0[nrb] = n + rb * ar;
            c->rb_b1[nrb] = n + rb * ar + ar < n + s->n ? n + rb * ar + ar : n + s->n;
        }
        n += s->n;
    }
    for (int i = 0, j = 0; i < nseg; i++) {
        if (sg[i].logits == RQ_LOGITS_ALL)
            for (int r = 0; r < sg[i].n; r++) c->sel_row[j++] = c->seg[i].b0 + r;
        else if (sg[i].logits == RQ_LOGITS_LAST)
            c->sel_row[j++] = c->seg[i].b0 + sg[i].n - 1;
    }

    if (n == nseg && n <= c->bd_max) { // decode rows only (one per segment): batch_decode_worker
        int e = batch_decode(c, n, nsel, out);
        if (e != RQ_OK) return e;
    } else {
        int e = prefill_plan(c, nseg, nrb);
        if (e < 0) return e;
        PrefillJob job = { c, n, nseg, ntile, nrb, nsel, gemv, out, 0, e, 1 };
        for (int i = 0; i < nseg; i++) { // segments starting a new cache tile; the longest split row
            const PfSeg *s = &c->seg[i];
            if (s->pos0 % ATT_T == 0 || (s->pos0 + s->n - 1) / ATT_T != s->pos0 / ATT_T)
                c->pop_seg[job.npop++] = i;
        }
        for (int rb = 0; rb < nrb; rb++)
            if (c->rb_sp[rb] >= 0 && (c->row_pos[c->rb_b0[rb]] + ATT_P) / ATT_P > job.nbmax)
                job.nbmax = (c->row_pos[c->rb_b0[rb]] + ATT_P) / ATT_P;
        if (job.nsp > 0) {
            size_t need = (size_t)job.nsp * p->n_heads * job.nbmax * ATT_PSTRIDE(p->head_dim);
            if (need > c->att_sp_n) {
                float *sp = malloc(need * sizeof(float));
                if (!sp) return ctx_fail(c, RQ_ERR_NOMEM, "malloc failed!");
                free(c->att_sp);
                c->att_sp = sp;
                c->att_sp_n = need;
            }
        }
        PROF_START(c);
        if (rq_begin(&c->sync, prefill_phases(p), p->n_layers * (nrb + job.nsp * p->n_kv_heads), rq_nthreads(), rq_spin_pf) != 0)
            return ctx_fail(c, RQ_ERR_NOMEM, "malloc failed!");
        rq_parallel(prefill_worker, &job);
    }
    for (int i = 0; i < nseg; i++) {
        RqSeq *s = c->seg[i].seq;
        memcpy(s->tokens + s->len, sg[i].tokens, sg[i].n * sizeof(int));
        s->len += sg[i].n;
    }
    if (logits) *logits = nsel ? out : NULL;
    return RQ_OK;
}

int rq_prefill(RqCtx *c, RqSeq *s, const int *tokens, int n, int mode, float *out, float **logits) {
    // tokens[0..n-1] at positions len.. in chunks of nb_max (> 1) rows, or one token at a time
    // through the decode path (nb_max <= 1); the same results either way
    const Config *p = &c->m->config;
    int V = p->vocab_size, e = ctx_check_seq(c, s);
    c->err[0] = 0;
    if (logits) *logits = NULL;
    if (e != RQ_OK) return e;
    if (n < 1 || mode < RQ_LOGITS_NONE || mode > RQ_LOGITS_ALL || (mode == RQ_LOGITS_ALL && !out))
        return ctx_fail(c, RQ_ERR_ARG, "bad prefill arguments (n %d, mode %d)", n, mode);
    if (s->len + n > s->cap) return ctx_fail(c, RQ_ERR_CONTEXT, "%d tokens at position %d: past the capacity %d", n, s->len, s->cap);
    for (int i = 0; i < n; i++)
        if (tokens[i] < 0 || tokens[i] >= V) return ctx_fail(c, RQ_ERR_ARG, "token %d out of range", tokens[i]);
    float *lg = NULL;
    if (c->nb_max <= 1) {
        for (int i = 0; i < n && e == RQ_OK; i++) {
            int want = mode == RQ_LOGITS_ALL || (mode == RQ_LOGITS_LAST && i == n - 1);
            e = decode_step(c, s, tokens[i], want ? c->logits : NULL);
            if (mode == RQ_LOGITS_ALL && e == RQ_OK)
                memcpy(out + (size_t)i * V, c->logits, V * sizeof(float));
        }
        lg = mode == RQ_LOGITS_ALL ? out : mode == RQ_LOGITS_LAST ? c->logits : NULL;
    } else {
        int nb = c->nb_max;
        for (int i = 0; i < n && e == RQ_OK; i += nb) {
            int k = n - i < nb ? n - i : nb;
            RqSegment sg = { s, tokens + i, k, mode == RQ_LOGITS_ALL ? RQ_LOGITS_ALL : mode == RQ_LOGITS_LAST && i + k == n ? RQ_LOGITS_LAST : RQ_LOGITS_NONE };
            e = rq_forward_batch(c, &sg, 1, mode == RQ_LOGITS_ALL ? out + (size_t)i * V : NULL, &lg);
        }
        lg = mode == RQ_LOGITS_ALL ? out : lg;
    }
    if (logits && e == RQ_OK) *logits = lg;
    return e;
}
