/* The forward passes: sequences (RqSeq: KV cache, position, tokens), compute contexts (RqCtx:
 * activation and scratch buffers, synchronization, profiling) and every numeric part of the decode
 * step and the batched prefill (quantization, norms, matmul and attention dispatch, KV cache).
 * All of it is one translation unit: the exact arithmetic (dot_q8, the attention, the quantizer)
 * depends on gcc -Ofast's code generation in this context (inlining; Steps 3-7 of the development log). */
#include "internal.h"

#ifndef ATT_ROWS
#define ATT_ROWS 6 // query rows per prefill attention work item (x kv_mul query heads)
#endif
#define ATT_NV_MAX 64 // max query vectors per attention_kv call

#ifdef RUNQ_PROFILE
// build with -DRUNQ_PROFILE: wall time per batched-prefill phase type (rq_ctx_profile)
enum { PF_NORM, PF_QKV, PF_ROPE, PF_ATT, PF_WO, PF_FFN13, PF_SWIGLU, PF_FFN2, PF_CLS, PF_BDQKV, PF_BDFFN13, PF_DNPREP, PF_DNREC, PF_DNOUT, PF_N };
static const char *pf_name[PF_N] = { "norm+quant+resid", "gemm qkv", "qk-norm+rope+kv", "attention+quant",
                                     "gemm wo", "gemm w1+w3", "swiglu+quant", "gemm w2", "classifier",
                                     "rows qkv+att", "rows w1+w3+swiglu", "dn prep (conv, l2norm)", "dn recurrence", "dn out+quant" };
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
    float *lin;         // Qwen3.5 linear-attention layers: the projections of the row (lin_width): qkv, z, alpha, beta
    RqSync sync;        // phase counters of the parallel jobs (threads.h)
    // attention scratch, sized for att_nb blocks of ATT_P positions (grown for a longer sequence)
    int att_nb;
    float *att;         // decode: per (kv head, block) the scores of its kv_mul query heads (n_heads, att_nb * ATT_P)
    float *att_part;    // decode: per (kv head, query head, block) partial results (ATT_PSTRIDE floats)
    float *att_b;       // prefill: per thread att_scores_floats(att_rows * kv_mul, att_nb) scores
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
    float *lin_b;       // (nb_max, lin_width): the linear-attention projections per row (as lin)
    int pf_nrb;         // attention row blocks of one segment of nb_max rows
    int32_t *bq_corr, *bq_c16, *bh_corr, *bh_c16; // GEMM activation prep of xq_b / hq_b rows (Gemm, act_prep_row)
    float *bq_xf, *bh_xf;
    uint8_t *kq_scr; size_t kq_scr_bytes;         // per-thread K-quant GEMM tile scratch (AVX-512, NEON)
    // the plan of a batched pass: per row its token and position; the segments; the QK-norm + RoPE +
    // K/V store items (a cache tile of one segment); the attention row blocks (rows of one segment,
    // longest first); the rows whose logits are computed
    int *row_tok, *row_pos, *row_seg; // (row_seg: the row's segment)
    PfSeg *seg;
    int *tile_seg, *tile_tb;
    int *rb_seg, *rb_b0, *rb_b1;
    int *rb_it0, *rb_sp; // attention items of row blocks 0..rb-1; split row block: its index (att_sp), else -1
    int *pop_seg;       // segments starting a new KV cache tile (pre-faulted in the first phase)
    int *sel_row;
    // batched decode rows (rq_forward_batch with segments of at most bd_seg rows, e.g. one row per segment or a
    // speculative verify segment: batch_decode_worker): at most bd_max rows; the norms as an epilogue up to bd_epi
    // rows (else a phase); a row block's attention split over its blocks of ATT_P positions when the pass has fewer
    // than bd_split * threads (row block, kv head) items; the small-batch kernels up to smb_max rows per weight
    // type (gemm_plan). Tuning: RUNQ_SMB, RUNQ_BD
    int smb_max[5], smb_pf, bd_max, bd_epi, bd_split, bd_seg;
    float *att_sp; size_t att_sp_n; // split attention: per (row, query head, block) partial results
    int *row_it0;       // batch_decode_worker: attention items of row blocks 0..u-1 (per kv head)
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

// Qwen3.5's linear-attention layers: the projections of one row, [qkv (conv_dim) | z (vdim) | alpha (nv) | beta (nv)]
static inline int lin_width(const Hybrid *hy) { return hy->conv_dim + hy->vdim + 2 * hy->nv; }
// the floats of one linear layer's conv inputs in a sequence's state (then the S of each head)
// dependency counters of the linear attention (decode_worker: after the attention layers' 2 per (layer, kv head)):
// per layer 1 + 2 nv (alpha / beta done, per head its prep done, per head its blocks done)
static inline int dn_counter0(const RqModel *m, int l) { return 2 * m->config.n_layers * m->config.n_kv_heads + l * (1 + 2 * m->hy.nv); }
static inline int dn_counters(const RqModel *m) { return m->hy.on ? m->config.n_layers * (1 + 2 * m->hy.nv) : 0; }
static inline size_t ssm_conv_floats(const Hybrid *hy) { return ((size_t)(hy->conv - 1) * hy->conv_dim + 15) / 16 * 16; }

int forward_model_init(RqModel *m) {
    // the shape parameters the forward passes derive from the config; 0 if not supported
    Config *p = &m->config;
    const Hybrid *hy = &m->hy;
    m->all_heads_dim = p->n_heads * p->head_dim;
    m->kv_dim = p->n_kv_heads * p->head_dim;
    m->kv_mul = p->n_heads / p->n_kv_heads;
    m->qh = (hy->on ? 2 : 1) * p->head_dim;
    m->xs = p->dim > m->all_heads_dim ? p->dim : m->all_heads_dim;
    m->xs = hy->vdim > m->xs ? hy->vdim : m->xs; // (the linear layers' output: the input of their projection)
    m->att_scale = 1.0f / sqrtf(p->head_dim);
    m->att_rows = m->kv_mul < 1 ? 0 : ATT_NV_MAX / m->kv_mul < ATT_ROWS ? ATT_NV_MAX / m->kv_mul : ATT_ROWS;
    m->ff_rows = GS > 64 ? GS : 64;
    m->ssm_layer = hy->on ? ssm_conv_floats(hy) + (size_t)hy->nv * hy->dk * hy->dv : 0;
    return !(m->kv_mul < 1 || m->att_rows < 1 || p->head_dim % 16 || (m->kv_mul * p->head_dim) % GS ||
             p->hidden_dim % m->ff_rows || m->ff_rows % GS || p->dim % 16 || m->rope_dim % 2 || m->rope_dim > p->head_dim ||
             m->n_attn < 1 || (hy->on && (hy->dv % GS || hy->dk % 16)));
}

// ---------------------------------------------------------------------------------------- sequences

RqSeq *rq_seq_new(const RqModel *m, int capacity, int kv_type, int *err) {
    int e = RQ_OK;
    RqSeq *s = NULL;
    const Config *p = &m->config;
    if (!m->has_weights) { e = RQ_ERR_STATE; goto fail; }
    if (capacity <= 0) capacity = p->seq_len;
    if (capacity > p->seq_len || !kv_type_ok(kv_type) || (kv_type != KV_F32 && p->head_dim % 32)) {
        e = RQ_ERR_ARG;
        goto fail;
    }
    s = calloc(1, sizeof(RqSeq));
    if (!s) { e = RQ_ERR_NOMEM; goto fail; }
    s->m = m;
    s->cap = capacity;
    s->kv_len = (capacity + ATT_P - 1) / ATT_P * ATT_P; // (whole blocks: the cache layout)
    s->att_nb = s->kv_len / ATT_P;
    s->kvt = kv_type & ~KV_DOT;
    s->att = kv_type;
    s->kv_esz = kv_type == KV_F32 ? 4 : 2;
    s->kv_size = (size_t)m->n_attn * s->kv_len * m->kv_dim; // (the attention layers only)
    s->key_cache = alloc_big(s->kv_size * s->kv_esz);
    s->value_cache = alloc_big(s->kv_size * s->kv_esz);
    s->tokens = malloc((size_t)capacity * sizeof(int));
    s->nck = RQ_SEQ_CK;
    for (int i = 0; i < RQ_SEQ_CK_MAX; i++)
        s->ck[i].pos = -1;
    if (m->hy.on) {
        s->ssm_bytes = (size_t)m->hy.n_lin * m->ssm_layer * sizeof(float);
        s->ssm = alloc_big(s->ssm_bytes);
        if (!s->ssm) { e = RQ_ERR_NOMEM; goto fail; }
    }
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
    free_big(s->ssm, s->ssm_bytes);
    for (int i = 0; i < RQ_SEQ_CK_MAX; i++)
        free_big(s->ck[i].st, s->ssm_bytes);
    free(s->tokens);
    free(s);
}

// Qwen3.5: the recurrent state of the linear-attention layers is the state after the last position: it cannot
// be rewound. A sequence keeps a set of nck checkpoints of it (position + copy; rq_seq_set_checkpoints, default
// RQ_SEQ_CK = 2, buffers allocated at first use) and a sorted list of pending positions: rq_seq_checkpoint_at /
// _rows ask for checkpoints at positions >= the length, taken by the forward pass that reaches them (ck_plan
// assigns their slots, CkPass; each (layer, head / channel group) work item copies its part after each of those
// rows, dn_prep_part / dn_rec_run), or at once if the sequence is there. A new checkpoint takes a free slot, else
// the one at the lowest position (the pass's earlier ones if all slots are its own: the highest stay).
// Truncating / copying to an earlier length goes back to the longest checkpoint at or below it (else to 0): the
// KV entries up to there are kept, the rest is re-prefilled by the caller (rq_seq_len tells how much was kept).
// The state of a position-0 token is zero (dn_prep, dn_rec clear it), so the state needs no reset
static void ck_unplan(RqSeq *s) {
    // the slots of a pass that ck_plan prepared but that failed before running (rq_forward_batch's errors after
    // ck_plan): free again (their old checkpoints were given up already)
    for (int i = 0; i < s->ckw.n; i++) {
        s->ck[s->ckw.slot[i]].pos = -1;
        s->ck[s->ckw.slot[i]].wr = 0;
    }
    s->ckw.n = 0;
}

static void ck_drop(RqSeq *s, int above) {
    // forget the checkpoints past position `above`
    for (int i = 0; i < s->nck; i++)
        if (s->ck[i].pos > above) s->ck[i].pos = -1;
}

static int ck_best(const RqSeq *s, int n) {
    // the checkpoint with the largest position <= n, or -1
    int b = -1;
    for (int i = 0; i < s->nck; i++)
        if (s->ck[i].pos >= 0 && !s->ck[i].wr && s->ck[i].pos <= n && (b < 0 || s->ck[i].pos > s->ck[b].pos)) b = i;
    return b;
}

static int ck_slot(RqSeq *s) {
    // a checkpoint slot to (over)write: a free one (one with a buffer first), else the one at the lowest position
    // (a slot the pass being planned writes: its lowest, removed from the plan); its buffer allocated (-1: no memory,
    // no checkpoint then)
    int b = -1;
    for (int i = 0; i < s->nck; i++)
        if (s->ck[i].pos < 0 && (b < 0 || (s->ck[i].st && !s->ck[b].st))) b = i;
    if (b < 0) {
        b = 0;
        for (int i = 1; i < s->nck; i++)
            if (s->ck[i].pos < s->ck[b].pos) b = i;
    }
    if (!s->ck[b].st && !(s->ck[b].st = alloc_big(s->ssm_bytes))) return -1;
    if (s->ck[b].wr) { // (the plan's lowest: its first entry)
        CkPass *w = &s->ckw;
        int j = 0;
        while (w->slot[j] != b) j++;
        for (w->n--; j < w->n; j++) {
            w->at[j] = w->at[j + 1];
            w->row[j] = w->row[j + 1];
            w->slot[j] = w->slot[j + 1];
            w->st[j] = w->st[j + 1];
        }
    }
    s->ck[b].pos = -1;
    s->ck[b].wr = 0;
    s->ck[b].rows = 0;
    return b;
}

static void pend_add(RqSeq *s, int pos, int rows) {
    // a pending checkpoint at length pos (kept sorted; already there: rows only if both are; the list full: the
    // lowest goes, as its slot would)
    int i = 0;
    while (i < s->npend && s->pend[i].pos < pos) i++;
    if (i < s->npend && s->pend[i].pos == pos) {
        s->pend[i].rows &= rows;
        return;
    }
    if (s->npend == RQ_SEQ_CK_MAX) {
        if (i == 0) return;
        memmove(s->pend, s->pend + 1, (size_t)--s->npend * sizeof(s->pend[0]));
        i--;
    }
    memmove(s->pend + i + 1, s->pend + i, (size_t)(s->npend - i) * sizeof(s->pend[0]));
    s->pend[i].pos = pos;
    s->pend[i].rows = rows;
    s->npend++;
}

static void ck_forget_pending(RqSeq *s) {
    // (rq_seq_truncate / copy / reset: the pending checkpoints were for the tokens that were to follow)
    ck_unplan(s);
    s->npend = 0;
}

void rq_seq_reset(RqSeq *s) {
    s->len = 0;
    ck_forget_pending(s);
    ck_drop(s, -1);
}

int rq_seq_checkpoint_at(RqSeq *s, int pos) {
    if (!s->m->hy.on) return RQ_OK; // (nothing to keep: every length can be truncated to)
    if (pos < s->len && pos >= 0) { // (one held there, e.g. a verify row's: kept from now on)
        for (int i = 0; i < s->nck; i++)
            if (s->ck[i].pos == pos && !s->ck[i].wr) {
                s->ck[i].rows = 0;
                return RQ_OK;
            }
        return RQ_ERR_ARG;
    }
    if (pos < s->len || pos > s->cap) return RQ_ERR_ARG;
    ck_unplan(s);
    if (pos == s->len) {
        int b = ck_best(s, pos);
        if (b < 0 || s->ck[b].pos != pos) {
            if ((b = ck_slot(s)) < 0) return RQ_ERR_NOMEM;
            memcpy(s->ck[b].st, s->ssm, s->ssm_bytes);
            s->ck[b].pos = pos;
        }
        s->ck[b].rows = 0;
        return RQ_OK;
    }
    pend_add(s, pos, 0);
    return RQ_OK;
}

int rq_seq_checkpoint_rows(RqSeq *s, int pos, int n) {
    if (!s->m->hy.on) return RQ_OK;
    if (n < 1 || n > RQ_SEQ_CK_MAX || pos < s->len || pos > s->cap - n) return RQ_ERR_ARG;
    ck_unplan(s);
    for (int i = 0; i < s->nck; i++) // the previous call's: free (their buffers kept)
        if (s->ck[i].rows) {
            s->ck[i].pos = -1;
            s->ck[i].rows = 0;
        }
    int k = 0;
    for (int i = 0; i < s->npend; i++)
        if (!s->pend[i].rows) s->pend[k++] = s->pend[i];
    s->npend = k;
    for (int i = 1; i <= n; i++)
        pend_add(s, pos + i, 1);
    return RQ_OK;
}

void rq_seq_checkpoint_rows_drop(RqSeq *s) {
    if (!s->m->hy.on) return;
    ck_unplan(s);
    for (int i = 0; i < s->nck; i++)
        if (s->ck[i].rows) {
            s->ck[i].pos = -1;
            s->ck[i].rows = 0;
        }
    int k = 0;
    for (int i = 0; i < s->npend; i++)
        if (!s->pend[i].rows) s->pend[k++] = s->pend[i];
    s->npend = k;
}

int rq_seq_set_checkpoints(RqSeq *s, int n) {
    if (n < 1 || n > RQ_SEQ_CK_MAX) return RQ_ERR_ARG;
    ck_unplan(s);
    while (s->nck > n) { // a free slot goes first, else the lowest position; the last slot moves into its place
        int b = 0;
        for (int i = 1; i < s->nck; i++)
            if (s->ck[i].pos < s->ck[b].pos) b = i;
        free_big(s->ck[b].st, s->ssm_bytes);
        s->ck[b] = s->ck[--s->nck];
        s->ck[s->nck].pos = -1;
        s->ck[s->nck].rows = 0;
        s->ck[s->nck].st = NULL;
    }
    s->nck = n;
    return RQ_OK;
}

int rq_seq_checkpoints(const RqSeq *s, int *pos, int max) {
    // the lengths of the checkpoints held (taken, not pending), ascending; their count (only the first max written)
    int k = 0, last = -1;
    for (;;) {
        int b = -1;
        for (int i = 0; i < s->nck; i++)
            if (s->ck[i].pos > last && !s->ck[i].wr && (b < 0 || s->ck[i].pos < s->ck[b].pos)) b = i;
        if (b < 0) return k;
        last = s->ck[b].pos;
        if (k < max && pos) pos[k] = last;
        k++;
    }
}

int rq_seq_checkpoint_drop(RqSeq *s, int pos) {
    // forget the checkpoint at length pos (its buffer stays, for the next one)
    for (int i = 0; i < s->nck; i++)
        if (s->ck[i].pos == pos && pos >= 0 && !s->ck[i].wr) {
            s->ck[i].pos = -1;
            s->ck[i].rows = 0;
            return RQ_OK;
        }
    return RQ_ERR_ARG;
}

int rq_seq_keep(const RqSeq *s, int n) {
    if (n < 0 || n > s->len) return RQ_ERR_ARG;
    if (!s->m->hy.on || n == s->len) return n;
    int b = ck_best(s, n);
    return b >= 0 ? s->ck[b].pos : 0;
}

int rq_seq_truncate(RqSeq *s, int n) {
    // (the cache entries past n are overwritten by the next tokens: positions only ever grow from len). A
    // hybrid model: back to the longest checkpoint <= n, else to 0, unless n == len
    if (n < 0 || n > s->len) return RQ_ERR_ARG;
    if (s->m->hy.on) {
        ck_forget_pending(s);
        if (n < s->len) {
            int b = ck_best(s, n);
            n = b >= 0 ? s->ck[b].pos : 0;
            if (b >= 0) memcpy(s->ssm, s->ck[b].st, s->ssm_bytes);
            ck_drop(s, n);
        }
    }
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
    // rq_seq_copy's items (attention layer, K / V) claimed from a shared counter: the destination's tiles pre-faulted
    // (a first copy into a fresh sequence was ~85% page faults: 14.5 vs 2.1 ms for 253 positions), then the
    // whole blocks of ATT_P positions below n as one range and the last block's tiles up to position n - 1
    // per kv head (the layout: kv_tile)
    (void)tid; (void)nt;
    CopyJob *j = arg;
    const Config *p = &j->src->m->config;
    int nkv = p->n_kv_heads, hd = p->head_dim, kv_dim = j->src->m->kv_dim, full = j->n / ATT_P, rem = j->n % ATT_P;
    size_t esz = j->src->kv_esz, ntl = (size_t)(rem + ATT_T - 1) / ATT_T;
    for (int it; (it = __atomic_fetch_add(&j->next, 1, __ATOMIC_RELAXED)) < 2 * j->src->m->n_attn;) {
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
    // (on the thread pool: one parallel job between forward passes, as rq_sample_batch). A hybrid model: n ==
    // src's length (its current state), else src's longest checkpoint <= n, else 0
    if (!dst || !src || dst == src || dst->m != src->m || dst->att != src->att || n < 0 || n > src->len || n > dst->cap)
        return RQ_ERR_ARG;
    if (src->m->hy.on) {
        int b = n == src->len ? -1 : ck_best(src, n);
        n = n == src->len ? n : b >= 0 ? src->ck[b].pos : 0;
        if (n > 0) memcpy(dst->ssm, b >= 0 ? src->ck[b].st : src->ssm, src->ssm_bytes);
        ck_forget_pending(dst);
        ck_drop(dst, -1);
    }
    if (n > 0) {
        CopyJob j = { dst, src, n, 0 };
        rq_parallel(copy_worker, &j);
    }
    memcpy(dst->tokens, src->tokens, (size_t)n * sizeof(int));
    dst->len = n;
    return RQ_OK;
}

const char *rq_kv_name(int t) {
    static const char *name[] = { "f32", "bf16", "f16", "?", "?", "bf16-dot", "f16-dot" };
    return kv_type_ok(t) ? name[t] : "?";
}

int rq_kv_parse(const char *s) {
    for (int t = KV_F32; t <= (KV_F16 | KV_DOT); t++)
        if (kv_type_ok(t) && !strcmp(s, rq_kv_name(t))) return t;
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
        att_b = calloc((size_t)c->nt * att_scores_floats(vr, att_nb), sizeof(float));
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
    int ad = m->xs; // (dim, all_heads_dim and Qwen3.5's vdim values: the max of them)
    int qd = p->n_heads * m->qh; // (wq's rows: Qwen3.5 [q | gate] per head)
    int mix = all_heads_dim > m->hy.vdim ? all_heads_dim : m->hy.vdim; // (the input of wo, Qwen3.5's ssm_out)
    int lw = m->hy.on ? lin_width(&m->hy) : 0;
    c->m = m;
    c->nt = rq_threads_fix(); // (every later job has at most this many threads: the per-thread scratch below)
    c->nb_max = nb_max < 0 ? 0 : nb_max > B_MAX ? B_MAX : nb_max;
    smb_config(c);
    if (max_ctx <= 0 || max_ctx > p->seq_len) max_ctx = p->seq_len;
    if (c->nb_max > 1) {
        size_t nb = c->nb_max;
        int xs = m->xs;
        c->x_b = calloc(nb * p->dim, sizeof(float));
        c->xb_b = calloc(nb * xs, sizeof(float));
        c->q_b = calloc(nb * qd, sizeof(float));
        if (lw && !(c->lin_b = calloc(nb * lw, sizeof(float)))) goto fail;
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
        c->row_seg = malloc(nb * sizeof(int));
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
            !c->xq_b.s || !c->hq_b.q || !c->hq_b.s || !c->row_tok || !c->row_pos || !c->row_seg || !c->seg || !c->tile_seg ||
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
    c->q = calloc(qd, sizeof(float));
    c->xq_corr = calloc(ad / GS + 1, sizeof(int32_t));
    c->hq_corr = calloc(p->hidden_dim / GS + 1, sizeof(int32_t));
    c->aq = (QuantizedTensor) { .q = calloc(mix, sizeof(int8_t)), .s = calloc(mix / GS, sizeof(float)) };
    c->aq_corr = calloc(mix / GS + 1, sizeof(int32_t));
    if (!c->aq.q || !c->aq.s || !c->aq_corr || !c->xq.q || !c->xq.s || !c->hq.q || !c->hq.s) goto fail;
    if (lw && !(c->lin = calloc(lw, sizeof(float)))) goto fail;
    if (m->kq) {
        c->aq_xf = calloc(mix / GS + 1, sizeof(float));
        c->aq_c16 = calloc(mix / 16 + 1, sizeof(int32_t));
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
    if (m->hy.on && ncnt < 1 + 2 * m->hy.nv) ncnt = 1 + 2 * m->hy.nv; // (the linear layers of batch_decode_worker)
    int dcnt = 2 * p->n_layers * p->n_kv_heads + dn_counters(m); // (decode_worker)
    if (rq_sync_init(&c->sync, 10 * p->n_layers + 8, dcnt > p->n_layers * ncnt + 64 ? dcnt : p->n_layers * ncnt + 64) != 0) goto fail;
    if (err) *err = RQ_OK;
    return c;
fail:
    rq_ctx_free(c);
    if (err) *err = e;
    return NULL;
}

void rq_ctx_free(RqCtx *c) {
    if (!c) return;
    void *bufs[] = { c->lin, c->lin_b, c->x, c->xb, c->hb, c->hb2, c->xq.q, c->xq.s, c->hq.q, c->hq.s, c->q, c->xq_corr, c->hq_corr,
                     c->xq_xf, c->hq_xf, c->xq_c16, c->hq_c16, c->aq.q, c->aq.s, c->aq_corr, c->aq_xf, c->aq_c16, c->k,
                     c->v, c->logits, c->rope, c->att, c->att_part, c->att_b, c->att_pb, c->att_conv, c->x_b, c->xb_b,
                     c->q_b, c->k_b, c->v_b, c->hb_b, c->hb2_b, c->xq_b.q, c->xq_b.s, c->hq_b.q, c->hq_b.s, c->rope_b,
                     c->bq_corr, c->bq_xf, c->bq_c16, c->bh_corr, c->bh_xf, c->bh_c16, c->kq_scr, c->row_tok,
                     c->row_pos, c->row_seg, c->seg, c->tile_seg, c->tile_tb, c->rb_seg, c->rb_b0, c->rb_b1, c->sel_row, c->row_it0, c->att_sp, c->rb_it0, c->rb_sp, c->pop_seg, c->probindex, c->probindex_t };
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
    if (t->b && t->type == GGML_F32) { // (F32 alpha / beta: rq_dequant_rows)
        memcpy(x, t->b + e0 * sizeof(float), (size_t)dim * sizeof(float));
        return;
    }
    if (t->b && iq_type(t->type)) { // Q2_K, i-quants (Saluki: an IQ1_M token_embd): ggml's dequantization (iquants.h)
        iq_dequant_row(t->type, t->b + (size_t)token * (dim / QK_K) * iq_block_bytes(t->type), dim, x);
        return;
    }
    if (t->b && t->type != GGML_Q8_0) {
        // GGUF K-quants (Q3_K .. Q6_K): ggml's dequantization of the row's blocks (dim % 256 == 0). By the tensor's
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

// F32 weights (only ssm_alpha / ssm_beta, as in Qwen3.8-27B's file: [48, 5120]; model.c gguf_matrix): out[i] = W row i
// . x for i0 <= i < i1, on the fp32 activations x (the rmsnorm'd row that the int8 one is quantized from: an F32
// matrix stays fp32 on both sides, as in llama.cpp and tests/ref_qwen35.py). The order of the operations, the
// definition for every path (decode, batched prefill, batched decode rows) and build (AVX-512, generic, NEON):
// 16 partial sums s_j (j = column mod 16), each over its columns in increasing order, s_j = s_j + w * x (the
// product rounded, then the sum: no FMA), from s_j = 0; then the fixed tree s_j = s_j + s_(j+h) for h = 8, 4, 2, 1
// (j < h); out = s_0. k % 16 == 0 (dim: forward_model_init). KQ_EXACT (gcc) / RQ_FP_EXACT (clang): no contraction,
// no reassociation; the 16 lanes may still be one vector register (each lane's operations stay in their order).
// Tiny: 2 x 48 rows per linear layer (~1 MB each; decode splits them into items of AB_ROWS rows)
KQ_EXACT static void f32_rows(float *out, const float *w, const float *x, int k, int i0, int i1) { RQ_FP_EXACT
    for (int i = i0; i < i1; i++) {
        const float *wr = w + (size_t)i * k;
        float s[16];
        for (int j = 0; j < 16; j++)
            s[j] = 0;
        for (int c = 0; c < k; c += 16)
            for (int j = 0; j < 16; j++)
                s[j] = s[j] + wr[c + j] * x[c + j];
        for (int h = 8; h >= 1; h /= 2)
            for (int j = 0; j < h; j++)
                s[j] = s[j] + s[j + h];
        out[i] = s[0];
    }
}

// decode's (and the batched decode rows') alpha / beta items of a linear layer: one for both if quantized (item 0,
// as before), else per matrix, F32: blocks of AB_ROWS rows (2 x 3 items of ~320 KB for the 27B's)
#define AB_ROWS 16
static inline int ab_items_of(const QuantizedTensor *w, int nv) { return w->f ? (nv + AB_ROWS - 1) / AB_ROWS : 1; }
static inline int ab_items(const QuantizedTensor *wa, const QuantizedTensor *wb, int nv) {
    return wa->f || wb->f ? ab_items_of(wa, nv) + ab_items_of(wb, nv) : 1;
}

static inline const QuantizedTensor *ab_item(const QuantizedTensor *wa, const QuantizedTensor *wb, int nv, int it, int *r0,
                                             int *r1, int *beta) {
    // item it (< ab_items, which is > 1): its matrix (wa or wb: *beta), rows r0..r1-1
    int na = ab_items_of(wa, nv), j = it < na ? it : it - na;
    const QuantizedTensor *t = it < na ? wa : wb;
    *beta = it >= na;
    *r0 = t->f ? j * AB_ROWS : 0;
    *r1 = t->f && *r0 + AB_ROWS < nv ? *r0 + AB_ROWS : nv;
    return t;
}

// Decode GEMV, in pieces: decode_step runs all of a token in one parallel region, with the rows
// of each matrix split into work items. matvec_prep: the VNNI kernel's activation corrections for
// x[i0 .. i0+n) (one per GS-group; nothing to do for the generic kernel), and with K-quant weights
// (xf != NULL) the activation group sums xf of kquants.h, plus for the AVX-512 K-quant GEMV the
// -32 * sums of 16 (c16, Q6_K; Q3_K: c16 / 8); matvec_rows: rows i0..i1-1
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
    if (w->kp && iq_type(w->type)) { // Q2_K, i-quants (iquants.h: AVX-512 or generic; c16 read by the AVX-512 kernels)
        iq_rows(w->type, xout, 0, x->q, 0, x->s, c16, 1, w->kp, n, i0 / KQ_ROWS, i1 / KQ_ROWS);
        return;
    }
    if (w->kp) { // K-quants, 16-row blocks (kquants.h; AVX-512: kernels_avx512.h)
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

// One batched-matmul phase of a batched pass: up to 4 GEMMs on the same nb quantized activation rows
// (q / k / v, w1 / w3; Qwen3.5's qkv / z / alpha / beta), out[b * ldo + i] = W[i, :] . x[b, :] (W d x k; x row b at x->q + b * ldx,
// its k / GS scales at x->s + b * ldx / GS), as work items of one phase:
//   AVX-512 build: (32-row tile, chunk of nbi activation rows), Q8P (q8p_gemm_item) or Q3P .. Q6P
//     (kq512_gemm_item); the items of a tile are claimed together (chunk = nchunk), so the weight
//     tile is read (K-quants: unpacked) once per claim and reused from L1/L2
//   generic: blocks of MB_ROWS rows (Q8_0, dot_q8_x4) or of 16 rows (K-quants, kq_dot) over all nb rows
// Each output is computed exactly as in the decode GEMV (matvec_rows), so the batched path matches the
// sequential one bit for bit. Activation prep (row stride k / GS, k / GS, k / 16): corr (AVX-512 Q8P),
// xf (K-quants), c16 (AVX-512 / NEON Q3_K, Q6_K), computed per row when the row is quantized (act_prep_row)
typedef struct {
    int nj; MatJob job[4];
    int it0[5];          // items of job j: it0[j] .. it0[j + 1] - 1
    int tiled[4];        // job j in tiles x chunks (else blocks over all rows)
    int smb[4];          // job j by the small-batch kernel: one item per 16-row block, all rows (smb_rows)
    int pf;              // small-batch kernels: their prefetch distance (smb_pf)
    int nbi, nchunk;     // tiled: activation rows per chunk, chunks
    int chunk, unit;     // items per claim; ranges of whole units (tiled: a tile's chunks)
    QuantizedTensor *x; int ldx, nb, k;
    const int32_t *corr; const float *xf; const int32_t *c16;
    const float *xa; int ldxa; // the fp32 activation rows (row b at xa + b * ldxa) for F32 weights (f32_rows), else NULL
} Gemm;

// Small-batch kernels (AVX-512, kernels_avx512.h smb_rows): in prefill_worker's GEMM phases for a matmul
// over at most smb_max rows of its weight type (else the GEMM's tiles), and always in batch_decode_worker.
// Crossover (one thread, measured in Step 10a): Q4_K / Q6_K faster than the GEMM up to ~24 rows (1.6-2.3x at
// 1-4 rows: no tile unpacking), equal at 32, ~4% slower at 48-64; Q8_0 equal (the GEMM tile streams the
// packed weights too). RUNQ_SMB=Q8,Q4,Q6,Q5,pf,Q3 overrides them (tuning; 0 = never). Results never depend on it.
// The K-quant kernels prefetch their weights SMB_PF bytes ahead (TODO S2, 8 threads, Qwen3-8B Q4_K_M, 500
// positions: a verify segment of 2 / 4 / 6 rows -10% / -12% / -5%, 1 and 9 rows unchanged; 1,024 / 2,048 / 8,192
// bytes: about the same; Q8_0: none of it, within noise or up to +5% at 1-2 rows, so its kernel does not prefetch)
enum { SMB_Q8, SMB_Q4, SMB_Q6, SMB_Q5, SMB_Q3 };
#ifndef BD_MAX
#define BD_MAX 64 // decode rows per batched pass in batch_decode_worker (if the small-batch kernels take them)
#endif
#ifndef SMB_PF
#define SMB_PF 4096 // the K-quant small-batch kernels' software prefetch distance in bytes (0: none)
#endif
#ifndef BD_SEG
#define BD_SEG 16 // rows of one sequence per segment in batch_decode_worker, at most (more: prefill_worker; TODO S2:
                  // 2-16 rows mostly 0-7% faster there on Qwen3, 2-24% on Qwen3.5; 24-32 within noise / up to 3% slower)
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
    // the context's small-batch / batched-decode settings: defaults, RUNQ_SMB=Q8,Q4,Q6,Q5,pf,Q3 (rows per weight
    // type; the kernels' prefetch distance in bytes; Q3_K's rows are the 6th value: pf was there first), RUNQ_BD=max,epi,split,seg
    int smb[6] = { 32, 32, 32, 32, SMB_PF, 32 };
    env_ints("RUNQ_SMB", smb, 6);
    memcpy(c->smb_max, smb, 4 * sizeof(int));
    c->smb_max[SMB_Q3] = smb[5];
    c->smb_pf = smb[4];
    int bd[4] = { BD_MAX, 2, 8, BD_SEG };
    env_ints("RUNQ_BD", bd, 4);
#ifdef RUNQ_NEON
    bd[0] = 0; // (NEON: its GEMM for batched rows, as before; no small-batch kernels there yet)
#endif
    c->bd_max = bd[0] < c->nb_max ? bd[0] : c->nb_max;
    c->bd_epi = bd[1];
    c->bd_split = bd[2];
    c->bd_seg = bd[3];
}

static inline int smb_slot(int type) {
    // smb_max index of a K-quant type (the small-batch kernels have one per type)
    return type == GGML_Q4_K ? SMB_Q4 : type == GGML_Q5_K ? SMB_Q5 : type == GGML_Q3_K ? SMB_Q3 : SMB_Q6;
}

static inline int smb_use(const RqCtx *c, const QuantizedTensor *w, int nb) {
    // the small-batch kernel for nb rows of w
#ifdef RUNQ_SMB
    if (w->kp) return !iq_type(w->type) && nb <= c->smb_max[smb_slot(w->type)]; // (Q2_K, i-quants: iq_rows)
    return w->p && nb <= c->smb_max[SMB_Q8];
#else
    (void)c; (void)w; (void)nb;
    return 0;
#endif
}

static void gemm_plan(const RqCtx *c, Gemm *g, int nj, const MatJob *job, QuantizedTensor *x, int ldx, int nb, int k,
                      const int32_t *corr, const float *xf, const int32_t *c16, int nt) {
    // (computed by every thread of the job: the same plan everywhere; results never depend on it)
    g->nj = nj; g->x = x; g->ldx = ldx; g->nb = nb; g->k = k; g->corr = corr; g->xf = xf; g->c16 = c16; g->pf = c->smb_pf;
    g->xa = NULL; g->ldxa = 0; // (F32 weights: the caller sets them)
    int ntile = 0, all_tiled = 1;
    for (int j = 0; j < nj; j++) {
        g->job[j] = job[j];
        g->tiled[j] = 0;
#ifdef RUNQ_KQX
        g->tiled[j] = job[j].w->kp != NULL && !iq_type(job[j].w->type); // (Q2_K, i-quants: blocks over all rows, iq_rows)
#endif
#ifdef RUNQ_Q8X
        g->tiled[j] |= job[j].w->p != NULL;
#endif
        g->smb[j] = smb_use(c, job[j].w, nb);
        if (g->smb[j])
            g->tiled[j] = 0;
        all_tiled &= g->tiled[j] || job[j].w->f; // (F32 alpha / beta, last in their phase: blocks of MB_ROWS rows, any range)
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
    if (w->f) { // F32: MB_ROWS rows over all activation rows, on the fp32 rows
        int i0 = i * MB_ROWS, i1 = i0 + MB_ROWS < jb->d ? i0 + MB_ROWS : jb->d;
        for (int b = 0; b < nb; b++)
            f32_rows(jb->out + (size_t)b * jb->ldo, w->f, g->xa + (size_t)b * g->ldxa, k, i0, i1);
        return;
    }
#ifdef RUNQ_SMB
    if (g->smb[j]) { // 16-row block i, all rows
        smb_rows(w->kp ? w->type : 0, jb->out, jb->ldo, x->q, ldx, x->s, g->corr, g->xf, g->c16, nb, w->kp ? w->kp : w->p, k, GS,
                 i, i + 1, g->pf);
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
    if (w->kp && iq_type(w->type)) { // Q2_K, i-quants: one 16-row block over all rows (iquants.h, AVX-512 or generic)
        iq_rows(w->type, jb->out, jb->ldo, x->q, ldx, x->s, g->c16, nb, w->kp, k, i, i + 1);
        return;
    }
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
    // (head_dim: the rotated dims, RqModel.rope_dim: Qwen3 all, Qwen3.5 64 of 256: HF's partial rotary factor,
    // freq_j = theta^(-2j / rope_dim))
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

static void qk_norm_rope(float *v, int n_heads, int stride, const float *norm_w, int head_dim, const float *cs, int rd) {
    // Qwen3 QK-norm (RMSNorm per head) then NeoX-style RoPE (pairs j, j + rd/2) on the first rd dims of each
    // head (Qwen3: rd = head_dim; Qwen3.5: 64, the rest passes through), in place; heads at v + h * stride
    // (Qwen3.5's q: [q | gate] per head); cs: cos / sin of this row's position (rope_row)
    for (int h = 0; h < n_heads; h++) {
        float *q = v + h * stride;

        rmsnorm(q, q, (float *)norm_w, head_dim);
        for (int j = 0; j < rd/2; j++) {
            float cos_freq = cs[j], sin_freq = cs[j + rd/2];

            float x = q[j]; // real part
            float y = q[j + rd/2]; // imag part

            q[j] = x * cos_freq - y * sin_freq; // new real
            q[j + rd/2] = x * sin_freq + y * cos_freq; // new imag
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

// (bf16_f32, f32_bf16: kernels_avx512.h)

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

// ------------------------------------------------ the bf16 dot-product attention (RQ_KV_BF16_DOT / RQ_KV_F16_DOT)
// Definition in kernels_avx512.h (att512_kv_dot); here its C twin for the generic build (and NEON: no kernel of its
// own), operation for operation: vdpbf16ps's lane is two fmaf, the odd element first, which with FTZ / DAZ set (the
// whole call runs so: att_ftz_on) equal the instruction bit for bit (build/bf16dp_test, tests/bf16dp_test.c)
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
static inline unsigned att_ftz_on(void) {
    // MXCSR's FTZ and DAZ for a bf16 dot-product attention call (the instruction's rule; runq's -Ofast startup sets
    // them already); returns the MXCSR to restore
    unsigned c = _mm_getcsr();
    if ((c & 0x8040) != 0x8040)
        _mm_setcsr(c | 0x8040);
    return c;
}
static inline void att_ftz_off(unsigned c) {
    if ((c & 0x8040) != 0x8040)
        _mm_setcsr(c);
}
#else
// (other platforms: their flush mode, -Ofast's (AArch64: FPCR.FZ, which detects tininess before rounding: the twin's
// results there can differ from an x86 CPU's in the products' underflow cases))
static inline unsigned att_ftz_on(void) { return 0; }
static inline void att_ftz_off(unsigned c) { (void)c; }
#endif

static volatile float att_zero = 0.0f; // (+0 that the compiler cannot fold: fma(+0, +0, c) is c + (+0), not c)

ATT_EXACT static inline float att_dot2(float c, float a1, float b1, float a0, float b0) { RQ_FP_EXACT
    // a dword lane of vdpbf16ps: c + a1 b1 (the odd element first), then + a0 b0, two FMAs each rounded
    c = ATT_FMA(a1, b1, c);
    return ATT_FMA(a0, b0, c);
}

static inline float bf16_round(float x) { return bf16_f32(f32_bf16_daz(x)); } // (the definition's bf16(x), as fp32)

ATT_EXACT static void attention_vec_dot(const float *qb, float *out, float *ml, int n, const float *kt, const float *vt,
                                        size_t ts, int hd, float scale, float *S) { RQ_FP_EXACT
    // attention_vec with the bf16 dot products: qb = q rounded to bf16, kt / vt = the block's K' / V' (bf16 values) in
    // the fp32 tile layout; S: n (rounded up to 16) floats, then the weights w (bf16 values)
    int nt = (n + ATT_T - 1) / ATT_T;
    for (int tb = 0; tb < nt; tb++) {
        const float *k = kt + (size_t)tb * ts;
        float a[ATT_T], c[ATT_T];
        for (int j = 0; j < ATT_T; j++)
            a[j] = 0.0f; // (not used: chunk 0 is assigned)
        for (int i0 = 0; i0 < hd; i0 += ATT_DC) {
            for (int j = 0; j < ATT_T; j++)
                c[j] = 0.0f;
            for (int i = i0; i < i0 + ATT_DC; i += 2) // dim pairs (i, i + 1), the odd one first
                for (int j = 0; j < ATT_T; j++)
                    c[j] = att_dot2(c[j], qb[i + 1], k[(i + 1) * ATT_T + j], qb[i], k[i * ATT_T + j]);
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
            float w = bf16_round(t < n ? exp_att(S[t] - m) : 0.0f);
            S[t] = w;
            sum[j] = sum[j] + w;
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
    for (int t = 0; t < n; t += 2) { // position pairs (t, t + 1), the odd one first; t + 1 beyond n: (+0, +0)
        const float *v0 = vt + (size_t)(t / ATT_T) * ts + (size_t)(t % ATT_T) * hd, *v1 = v0 + hd;
        float w0 = S[t];
        if (t + 1 < n) {
            float w1 = S[t + 1];
            for (int i = 0; i < hd; i++)
                out[i] = att_dot2(out[i], w1, v1[i], w0, v0[i]);
        } else {
            float z = att_zero;
            for (int i = 0; i < hd; i++)
                out[i] = att_dot2(out[i], z, z, w0, v0[i]);
        }
    }
    if (!ml)
        for (int i = 0; i < hd; i++)
            out[i] = out[i] * inv;
}

static void kv_block_dot(int kvt, float *kf, float *vf, const uint16_t *kh, const uint16_t *vh, int ntl, int hd) {
    // the first ntl tiles of one (kv head, block) of a bf16-dot / f16-dot cache as K' / V' (bf16 values) in the fp32
    // tile layout (kv_block_f32 for the dot products: an f16 element rounded to bf16, the bf16-dot V in position pairs)
    size_t ne = (size_t)ntl * ATT_T * hd;
    if (kvt == KV_F16) {
        for (size_t e = 0; e < ne; e++) { // (an f16 value is never an fp32 denormal: f32_bf16 = f32_bf16_daz)
            kf[e] = bf16_f32(f32_bf16(fp16_to_f32(kh[e])));
            vf[e] = bf16_f32(f32_bf16(fp16_to_f32(vh[e])));
        }
        return;
    }
    for (size_t t0 = 0; t0 < ne; t0 += (size_t)ATT_T * hd)     // K: as the bf16 cache
        for (int i = 0; i < hd; i++)
            for (int j = 0; j < ATT_T; j++)
                kf[t0 + (size_t)i * ATT_T + j] = bf16_f32(kh[t0 + (size_t)(i & ~1) * ATT_T + 2 * j + (i & 1)]);
    for (size_t t = 0; t < (size_t)ntl * ATT_T; t++)           // V: dim i of positions 2m, 2m + 1 at 2m hd + 2 i (+ 1)
        for (int i = 0; i < hd; i++)
            vf[t * hd + i] = bf16_f32(vh[(t & ~(size_t)1) * hd + 2 * (size_t)i + (t & 1)]);
}

static void attention_part_dot(int kvt, int nv, const float *const *q, float *const *out, float *const *part, const int *np,
                               int b0, int b1, const void *kt, const void *vt, size_t bs, int hd, float scale, float *S,
                               float *conv) {
    // attention_part's generic C version for the bf16 dot products (kvt: the storage, KV_BF16 or KV_F16)
    size_t ts = (size_t)ATT_T * hd;
    float qb[hd];
    for (int b = b0; b < b1; b++) {
        int nmax = 0;
        for (int r = 0; r < nv; r++) {
            int n = np[r] - b * ATT_P < ATT_P ? np[r] - b * ATT_P : ATT_P;
            nmax = n > nmax ? n : nmax;
        }
        if (nmax <= 0)
            break;
        kv_block_dot(kvt, conv, conv + (size_t)ATT_P * hd, (const uint16_t *)kt + (size_t)b * bs, (const uint16_t *)vt + (size_t)b * bs,
                     (nmax + ATT_T - 1) / ATT_T, hd);
        for (int r = 0; r < nv; r++) {
            int n = np[r] - b * ATT_P < ATT_P ? np[r] - b * ATT_P : ATT_P;
            if (n <= 0)
                continue;
            for (int i = 0; i < hd; i++)
                qb[i] = bf16_round(q[r][i]);
            if (!part)
                attention_vec_dot(qb, out[r], NULL, n, conv, conv + (size_t)ATT_P * hd, ts, hd, scale, S);
            else
                attention_vec_dot(qb, part[r] + (size_t)b * ATT_PSTRIDE(hd), part[r] + (size_t)b * ATT_PSTRIDE(hd) + hd, n, conv,
                                  conv + (size_t)ATT_P * hd, ts, hd, scale, S);
        }
    }
}

static void attention_part(int att, int nv, const float *const *q, float *const *out, float *const *part, const int *np,
                           int b0, int b1, const void *kt, const void *vt, size_t bs, int hd, float scale, float *S,
                           float *conv) {
    // nv query vectors q[r] sharing one kv head (kt / vt: its tile 0; att: the sequence's attention type, RqSeq.att),
    // vector r over positions 0..np[r]-1. part NULL: the whole attention into out[r] (all np <= ATT_P). Else the
    // partial results of the blocks b0..b1-1 of ATT_P positions that vector r reaches, block b at
    // part[r] + b * ATT_PSTRIDE(hd): o (unnormalized), m (max score), l (sum of exponentials).
    // S: scratch of att_scores_floats(nv, b1 - b0) floats (internal.h); conv: 2 * ATT_P * hd floats (the
    // generic C version's fp32 copy of a 16-bit cache block, the AVX-512 f16-dot conversion; this thread's)
    int kvt = att & ~KV_DOT;
    if (att & KV_DOT) {
        unsigned csr = att_ftz_on();
#ifdef RUNQ_ATT512
        if (att512_ok(hd))
            att512_kv_dot(kvt, nv, q, out, part, np, b0, b1, kt, vt, bs, hd, scale, S, (uint16_t *)conv);
        else
#endif
            attention_part_dot(kvt, nv, q, out, part, np, b0, b1, kt, vt, bs, hd, scale, S, conv);
        att_ftz_off(csr);
        return;
    }
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

static void attention_kv(int att, int nv, const float *const *q, float *const *out, const int *np, const void *kt,
                         const void *vt, size_t bs, int hd, float scale, float *S, float *part, float *conv) {
    // nv query vectors q[r] sharing one kv head, vector r over positions 0..np[r]-1, into out[r]:
    // per block of ATT_P positions the partial results, then their combination (one block: the
    // attention directly). S: scratch of att_scores_floats(nv, blocks of max np) floats; part: nv * (blocks
    // of max np) * ATT_PSTRIDE(hd) floats. Each vector's result does not depend on the others
    int nmax = 0;
    for (int r = 0; r < nv; r++)
        nmax = np[r] > nmax ? np[r] : nmax;
    if (nmax <= ATT_P) {
        attention_part(att, nv, q, out, NULL, np, 0, 1, kt, vt, bs, hd, scale, S, conv);
        return;
    }
    int nbm = (nmax + ATT_P - 1) / ATT_P;
    size_t st = ATT_PSTRIDE(hd);
    float *pv[ATT_NV_MAX];
    for (int r = 0; r < nv; r++)
        pv[r] = part + (size_t)r * nbm * st;
    attention_part(att, nv, q, NULL, pv, np, 0, nbm, kt, vt, bs, hd, scale, S, conv);
    for (int r = 0; r < nv; r++)
        att_combine(out[r], (np[r] + ATT_P - 1) / ATT_P, pv[r], st, hd);
}

static inline size_t kv_tile(int t, int h, int n_kv, int hd) {
    // offset (elements) of the cache tile holding position t of kv head h, inside a layer: blocks of
    // ATT_P positions, in a block the heads one after the other, each its ATT_P / ATT_T tiles (one
    // contiguous stream per head and block; see kernels_avx512.h)
    return (((size_t)(t / ATT_P) * n_kv + h) * ATT_P + (size_t)(t % ATT_P / ATT_T) * ATT_T) * hd;
}

static void kv_store(int att, void *kl0, void *vl0, int t, int h, int n_kv, int hd, const float *k, const float *v) {
    // head h of the key / value rows k, v of position t into the layer's cache tiles kl, vl
    // (k or v may be NULL: only the other one); att: the sequence's attention type (RqSeq.att: the cache type,
    // bf16 / f16 rounded to nearest even; a bf16-dot cache's V in position pairs; layouts in kernels_avx512.h)
    size_t o = kv_tile(t, h, n_kv, hd);
    int j = t % ATT_T, kvt = att & ~KV_DOT;
    if (att == (KV_BF16 | KV_DOT)) {
        uint16_t *kl = kl0, *vl = vl0;
        if (k)
            for (int i = 0; i < hd; i++)
                kl[o + (size_t)(i & ~1) * ATT_T + 2 * j + (i & 1)] = f32_bf16(k[i]);
        if (v)
            for (int i = 0; i < hd; i++)
                vl[o + (size_t)(j & ~1) * hd + 2 * i + (j & 1)] = f32_bf16(v[i]);
        return;
    }
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

// ---------------------------------------------------------------- Qwen3.5: linear attention (Gated DeltaNet)
// One token of one value head h of linear-attention layer li (docs/qwen35.md), with key head kh = h % nk (nv is a
// multiple of nk: llama.cpp's converter writes the value heads in a tiled order, value head h = i nk + kh is the
// i-th one of key head kh; with nv == nk, kh = h):
//   q, k, v = silu(causal depthwise conv over the last `conv` inputs of the channels: q, k (dk each) of kh, v (dv) of h)
//   q, k = l2norm (x / sqrt(sum x^2 + 1e-6)); q *= 1 / sqrt(dk)
//   beta = sigmoid(b), g = ssm_a * softplus(a + ssm_dt)
//   S (dk x dv, k-major) *= exp(g); delta = beta (v - S^T k); S += k delta^T; o = S^T q
//   o = rmsnorm(o, ssm_norm) * silu(z)
// The same function for every token in every path (decode, batched prefill, batched decode rows), so a sequence's
// results do not depend on the path or the batch. Prefill runs this exact recurrence token by token too, not the
// chunked (WY) form of HF / llama.cpp, which rounds differently. Arithmetic: elementwise along the channels / the
// v index (vectorized along them; ATT_EXACT: no reassociation or contraction, ATT_FMA as the attention), the sums
// over k in k order per v lane, the l2 / rms sums in 16 lanes and a fixed tree (dn_dot16), exp by dn_exp (the
// attention's polynomial), the per-head scalars by one non-inlined scalar function (dn_gates): the same bits in
// every build and at any vector width (given the same FMA availability, as the attention).
// DN_EXACT is ATT_EXACT without unsafe math at all: with it, gcc replaces a vectorized float division by a
// reciprocal approximation + a Newton step (vrcp14ps), whatever no-reciprocal-math says, so a sigmoid's 1 / (1 + e)
// rounded differently depending on whether its loop was vectorized, which depends on inlining (it differed between
// the AVX-512 and the generic build). The attention's code divides only in scalar code
#if defined(__GNUC__) && !defined(__clang__)
    #define DN_EXACT __attribute__((optimize("fp-contract=off", "no-unsafe-math-optimizations", "no-associative-math", \
                                             "no-reciprocal-math")))
    #define DN_INLINE __attribute__((always_inline)) inline
#else
    #define DN_EXACT
    #define DN_INLINE inline
#endif

DN_EXACT static DN_INLINE float dn_exp(float x) { RQ_FP_EXACT
    // exp(x) for x <= 0: exp_att, operation for operation (a copy with DN_EXACT's attributes, so that it inlines)
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

DN_EXACT static DN_INLINE float dn_sigmoid(float x) { RQ_FP_EXACT
    // 1 / (1 + exp(-x)) with e = exp(-|x|) <= 1 (no overflow)
    float e = dn_exp(-fabsf(x));
    float r = 1.0f / (1.0f + e);
    return x >= 0.0f ? r : e * r;
}

DN_EXACT static DN_INLINE float dn_dot16(const float *x, const float *y, int n) { RQ_FP_EXACT
    // sum of x[i] y[i], n % 16 == 0: 16 lanes (lane j: i = j mod 16, in order), then the tree of att512_sum16
    float a[16];
    for (int j = 0; j < 16; j++)
        a[j] = 0.0f;
    for (int i = 0; i < n; i += 16)
        for (int j = 0; j < 16; j++)
            a[j] = ATT_FMA(x[i + j], y[i + j], a[j]);
    for (int w = 8; w >= 1; w /= 2)
        for (int j = 0; j < w; j++)
            a[j] = a[j] + a[j + w];
    return a[0];
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
DN_EXACT static void dn_gates(float a, float b, float A, float dt, float *eg, float *beta) { RQ_FP_EXACT
    // a token's per-head scalars: exp(g), g = A * softplus(a + dt) (torch's softplus: x itself past 20; A < 0),
    // and beta = sigmoid(b). Scalar libm, one call per head (never vectorized: the same bits everywhere)
    float x = a + dt;
    float sp = x > 20.0f ? x : log1pf(expf(x));
    *eg = expf(A * sp);
    *beta = dn_sigmoid(b);
}

DN_EXACT static void dn_conv(float *out, const float *x, float *st, const float *w, int n, int K) { RQ_FP_EXACT
    // n channels: out = silu(sum_j w_j x_{t-K+1+j}) with x_t = x (this token's projections) and the last K - 1
    // inputs in st (K - 1 rows of n, oldest first), which then shifts x in; w: the K taps (rows of n). (The state and
    // the taps of a channel group are contiguous: with the rows of the whole layer, 6144 floats apart, the 8 or so
    // streams of an index i all fell on one set of the L1 cache)
    for (int i = 0; i < n; i++)
        out[i] = w[i] * st[i];
    for (int j = 1; j < K - 1; j++)
        for (int i = 0; i < n; i++)
            out[i] = ATT_FMA(w[(size_t)j * n + i], st[(size_t)j * n + i], out[i]);
    for (int i = 0; i < n; i++)
        out[i] = ATT_FMA(w[(size_t)(K - 1) * n + i], x[i], out[i]);
    for (int j = 0; j < K - 2; j++)
        memcpy(st + (size_t)j * n, st + (size_t)(j + 1) * n, n * sizeof(float));
    memcpy(st + (size_t)(K - 2) * n, x, n * sizeof(float));
    for (int i = 0; i < n; i++)
        out[i] = out[i] * dn_sigmoid(out[i]);
}

// The recurrence is split in three steps (docs/qwen35.md), the same functions in every path, so a sequence's
// results do not depend on the path or the batch:
//   dn_prep (per row and channel group, rows in order: the conv state): conv + SiLU, l2norm of q / k, the gates; the
//     results go back into the row (q, k, v in place of the conv inputs, exp(g) and beta in place of alpha / beta).
//     The groups: q and k of each key head, v (and the gates) of each value head: a key head's q / k are shared by
//     its nv / nk value heads, so they are computed (and their conv state advanced) once, by the item of value
//     head kh (dn_prep) or as groups of their own (prefill)
//   dn_rec (per row, head and block of DN_BW v columns, rows in order: S): the delta rule on a 128 x DN_BW block
//     of S, which stays in L1; each v lane is independent of the others, so the blocks give the bits of the whole
//   dn_out (per row and head, once all the blocks of the row are done): the gated RMSNorm over the head's dv outputs
// The conv state and the conv taps are stored per channel group (a head's q, k or v channels: group g at
// (conv - 1) * c0 resp. conv * c0, c0 = its first channel, as rows of its channels); S block-major: head h's dk x dv floats are dv / DN_BW blocks of (dk, DN_BW), each contiguous.
#if defined(__GNUC__) && !defined(__clang__)
    #define DN_UNROLL _Pragma("GCC unroll 32")
#else
    #define DN_UNROLL _Pragma("clang loop unroll(full)")
#endif
#define DN_BW 32 // v columns per block (gguf.c checks dv % 32 == 0)

static inline int dn_nblk(const Hybrid *hy) { return hy->dv / DN_BW; }

static inline void dn_chan(const Hybrid *hy, int h, int t, size_t *c0, int *cn) {
    // the channel group t (0 q, 1 k: of key head h; 2 v: of value head h): first channel, channels
    *c0 = t == 0 ? (size_t)h * hy->dk : t == 1 ? (size_t)hy->nk * hy->dk + (size_t)h * hy->dk : (size_t)2 * hy->nk * hy->dk + (size_t)h * hy->dv;
    *cn = t < 2 ? hy->dk : hy->dv;
}

static inline float *dn_sblock(const RqModel *m, float *st, int h, int jb) {
    // block jb of head h's S in the layer's state st
    return st + ssm_conv_floats(&m->hy) + (size_t)h * m->hy.dk * m->hy.dv + (size_t)jb * m->hy.dk * DN_BW;
}

#define DN_MAX 512 // head dims (Hybrid dk, dv), checked at load

static inline void dn_pf(const float *p, int n) {
    // prefetch n floats from p (the next row's inputs of a head: 512 B pieces of rows 33 KB apart, last written by
    // other cores' GEMM items: without it each line is a miss that waits for the last one)
    for (int i = 0; i < n; i += 16)
        __builtin_prefetch(p + i);
}

DN_EXACT static void dn_prep_part(const RqModel *m, int li, int h, int t, float *st, int first, float *ck, float *row,
                                  const float *next) { RQ_FP_EXACT
    // step 1 for one token, channel group t of linear layer li (0: q, 1: k of key head h; 2: v and the gates of value
    // head h; all independent): st = the sequence's state of the layer, first: the token at position 0 (the conv state
    // before it is zero: cleared here), row = the token's projections (lin_width: qkv | z | alpha | beta), updated in
    // place; ck != NULL: the checkpoint's state of the layer, which gets the group's conv state after this token;
    // next: the following row of the batch (prefetched), or NULL
    const Hybrid *hy = &m->hy;
    const TransformerWeights *w = &m->weights;
    int K = hy->conv, cd = hy->conv_dim, dk = hy->dk, nv = hy->nv, cn;
    size_t c0;
    dn_chan(hy, h, t, &c0, &cn);
    float buf[DN_MAX];
    float *sg = st + (size_t)(K - 1) * c0;
    if (next) {
        dn_pf(next + c0, cn);
        if (t == 2) {
            dn_pf(next + cd + hy->vdim + h, 1);
            dn_pf(next + cd + hy->vdim + nv + h, 1);
        }
    }
    if (first)
        memset(sg, 0, (size_t)(K - 1) * cn * sizeof(float));
    dn_conv(buf, row + c0, sg, w->ssm_conv + (size_t)li * K * cd + (size_t)K * c0, cn, K);
    if (t < 2) { // l2norm; q is also scaled by 1 / sqrt(dk)
        float inv = 1.0f / sqrtf(dn_dot16(buf, buf, dk) + 1e-6f);
        if (t == 0) {
            float sc = 1.0f / sqrtf((float)dk);
            for (int i = 0; i < dk; i++)
                buf[i] = buf[i] * inv * sc;
        } else {
            for (int i = 0; i < dk; i++)
                buf[i] = buf[i] * inv;
        }
    } else {
        float *ab = row + cd + hy->vdim + h; // alpha, beta -> exp(g), beta
        float eg, beta;
        dn_gates(ab[0], ab[nv], w->ssm_a[(size_t)li * nv + h], w->ssm_dt[(size_t)li * nv + h], &eg, &beta);
        ab[0] = eg;
        ab[nv] = beta;
    }
    memcpy(row + c0, buf, cn * sizeof(float));
    if (ck)
        memcpy(ck + (size_t)(K - 1) * c0, sg, (size_t)(K - 1) * cn * sizeof(float));
}

DN_EXACT static void dn_prep(const RqModel *m, int li, int h, float *st, int first, float *ck, float *row) { RQ_FP_EXACT
    // the groups of value head h's item (decode, batched decode rows): its v and gates, and the q / k of key head h
    // if h < nk (the value heads h' with h' % nk == h wait for it)
    for (int t = h < m->hy.nk ? 0 : 2; t < 3; t++)
        dn_prep_part(m, li, h, t, st, first, ck, row, NULL);
}

static inline int dn_prep_items(const Hybrid *hy) { return 3 * hy->nk + hy->nv - hy->nk; }

static inline void dn_prep_item(const Hybrid *hy, int i, int *h, int *t) {
    // prefill's dn_prep item i of a segment (0 <= i < dn_prep_items): (key head, 0 q / 1 k) or (value head, 2 v): the
    // q, k, v of heads 0 .. nk - 1 in turn, then the v of the other value heads
    if (i < 3 * hy->nk) {
        *h = i / 3;
        *t = i % 3;
    } else {
        *h = hy->nk + i - 3 * hy->nk;
        *t = 2;
    }
}

// The two passes of dn_rec over a block of S (dk, DN_BW), per v lane, k in order. p1: u = (S * eg)^T k. p2: S = S * eg +
// k delta^T and o = S^T q. p21: p2 of one row fused with p1 of the next row of the same sequence (the next row's p1
// reads each S row right after this row's p2 wrote it: the same floats, S * eg' of the stored value), which gives
// the FMA chains (u', o: dk dependent FMAs per lane each) twice the independent work. The decay S * eg is recomputed
// in p2 instead of being stored by p1 (the same product of the same floats). The j loops are unrolled so that the
// accumulators live in registers (a chain through memory would wait for a store forward at every step)
DN_EXACT static DN_INLINE void dn_p1(const float *S, const float *kv, float eg, float *u, int dk) { RQ_FP_EXACT
    DN_UNROLL
    for (int j = 0; j < DN_BW; j++)
        u[j] = 0.0f;
    for (int k = 0; k < dk; k++) {
        const float *restrict Sk = S + (size_t)k * DN_BW, kk = kv[k];
        DN_UNROLL
        for (int j = 0; j < DN_BW; j++)
            u[j] = ATT_FMA(Sk[j] * eg, kk, u[j]);
    }
}

DN_EXACT static DN_INLINE void dn_p2(float *S, const float *qv, const float *kv, float eg, const float *u, float *oa,
                                     int dk) { RQ_FP_EXACT
    DN_UNROLL
    for (int j = 0; j < DN_BW; j++)
        oa[j] = 0.0f;
    for (int k = 0; k < dk; k++) {
        float *restrict Sk = S + (size_t)k * DN_BW, kk = kv[k], qq = qv[k];
        DN_UNROLL
        for (int j = 0; j < DN_BW; j++) {
            float sv = ATT_FMA(kk, u[j], Sk[j] * eg);
            Sk[j] = sv;
            oa[j] = ATT_FMA(sv, qq, oa[j]);
        }
    }
}

DN_EXACT static DN_INLINE void dn_p21(float *S, const float *qv, const float *kv, float eg, const float *u, float *oa,
                                      const float *kn, float egn, float *un, int dk) { RQ_FP_EXACT
    DN_UNROLL
    for (int j = 0; j < DN_BW; j++) {
        oa[j] = 0.0f;
        un[j] = 0.0f;
    }
    for (int k = 0; k < dk; k++) {
        float *restrict Sk = S + (size_t)k * DN_BW, kk = kv[k], qq = qv[k], kk2 = kn[k];
        DN_UNROLL
        for (int j = 0; j < DN_BW; j++) {
            float sv = ATT_FMA(kk, u[j], Sk[j] * eg);
            Sk[j] = sv;
            oa[j] = ATT_FMA(sv, qq, oa[j]);
            un[j] = ATT_FMA(sv * egn, kk2, un[j]);
        }
    }
}

DN_EXACT static void dn_rec_run(const RqModel *m, int h, int jb, float *st, int first, const float *row, size_t rs, float *o,
                                size_t os, int n, const CkPass *ck, size_t ckoff) { RQ_FP_EXACT
    // step 2 for n consecutive tokens of one sequence, value head h, v columns jb * DN_BW ..: row = the first
    // token's projections as dn_prep left them (the next ones rs floats on), o = its head's dv outputs (os floats
    // to the next; only this block's columns are written: the raw S^T q); first: the first token is at position 0
    // (S cleared); ck: the checkpoints of the run's sequence that this pass takes (ck_plan: after its rows ck->row[i],
    // ascending, all in this run), whose block of layer state (ck->st[i] + ckoff) gets S after that token
    const Hybrid *hy = &m->hy;
    int cd = hy->conv_dim, dk = hy->dk, dv = hy->dv, nv = hy->nv;
    size_t oq = (size_t)(h % hy->nk) * dk, ok = (size_t)hy->nk * dk + oq; // (the key head's q / k)
    size_t ov = (size_t)2 * hy->nk * dk + (size_t)h * dv + (size_t)jb * DN_BW, oe = (size_t)cd + hy->vdim + h;
    float *S = dn_sblock(m, st, h, jb);
    float u[DN_BW], un[DN_BW], oa[DN_BW];
    int ci = 0, cn = ck->n;
    if (first)
        memset(S, 0, (size_t)dk * DN_BW * sizeof(float));
    dn_p1(S, row + ok, row[oe], u, dk);
    for (int t = 0; t < n; t++) {
        const float *r = row + (size_t)t * rs;
        if (t + 2 < n) { // (the inputs of the row after next: pieces of rows 33 KB apart that other cores wrote)
            const float *r2 = r + 2 * rs;
            dn_pf(r2 + oq, dk);
            dn_pf(r2 + ok, dk);
            dn_pf(r2 + ov, DN_BW);
            dn_pf(r2 + oe, 1);
            dn_pf(r2 + oe + nv, 1);
        }
        float beta = r[oe + nv];
        DN_UNROLL
        for (int j = 0; j < DN_BW; j++)
            u[j] = (r[ov + j] - u[j]) * beta;
        if (t + 1 < n) {
            dn_p21(S, r + oq, r + ok, r[oe], u, oa, r + rs + ok, r[rs + oe], un, dk);
            DN_UNROLL
            for (int j = 0; j < DN_BW; j++)
                u[j] = un[j];
        } else {
            dn_p2(S, r + oq, r + ok, r[oe], u, oa, dk);
        }
        float *ob = o + (size_t)t * os + (size_t)jb * DN_BW;
        DN_UNROLL
        for (int j = 0; j < DN_BW; j++)
            ob[j] = oa[j];
        if (ci < cn && t == ck->row[ci]) {
            memcpy(dn_sblock(m, ck->st[ci] + ckoff, h, jb), S, (size_t)dk * DN_BW * sizeof(float));
            ci++;
        }
    }
}

DN_EXACT static void dn_rec(const RqModel *m, int h, int jb, float *st, int first, const CkPass *ck, size_t ckoff,
                            const float *row, float *o) { RQ_FP_EXACT
    // dn_rec_run for one token (the pass's only row of its sequence)
    dn_rec_run(m, h, jb, st, first, row, 0, o, 0, 1, ck, ckoff);
}

DN_EXACT static void dn_out(const RqModel *m, int li, int h, const float *row, float *o) { RQ_FP_EXACT
    // step 3: the gated RMSNorm (the form of rmsnorm: weight * (1 / rms * x)), times silu(z), in place on the head's
    // dv outputs o; row: the token's projections (z is untouched by the steps before)
    const Hybrid *hy = &m->hy;
    int dv = hy->dv;
    float ss = 1.0f / sqrtf(dn_dot16(o, o, dv) * (1.0f / dv) + 1e-6f);
    const float *nw = m->weights.ssm_norm + (size_t)li * dv, *z = row + hy->conv_dim + (size_t)h * dv;
    for (int j = 0; j < dv; j++)
        o[j] = nw[j] * (ss * o[j]) * (z[j] * dn_sigmoid(z[j]));
}

DN_EXACT static void attn_gate(float *o, const float *gate, int n) { RQ_FP_EXACT
    // Qwen3.5's gated attention: the attention output times sigmoid(gate), elementwise
    for (int i = 0; i < n; i++)
        o[i] = o[i] * dn_sigmoid(gate[i]);
}

static inline float *dn_ck(const RqSeq *s, int li, int pos) {
    // the checkpoint state of layer li to write after the token at pos, or NULL (ck_plan)
    for (int i = 0; i < s->ckw.n; i++)
        if (s->ckw.at[i] == pos + 1) return s->ckw.st[i] + (size_t)li * s->m->ssm_layer;
    return NULL;
}

static void ck_plan(RqSeq *s, int pos0, int n) {
    // before a pass adding positions pos0..pos0+n-1 to s: a slot for each pending checkpoint the pass reaches (s->ckw;
    // no memory: that checkpoint is dropped)
    if (s->ckw.n) ck_unplan(s);
    for (int i = 0; i < s->npend && s->pend[i].pos <= pos0 + n;) {
        int at = s->pend[i].pos, b = ck_slot(s);
        if (b < 0) {
            memmove(s->pend + i, s->pend + i + 1, (size_t)(--s->npend - i) * sizeof(s->pend[0]));
            continue;
        }
        CkPass *w = &s->ckw;
        s->ck[b].pos = at;
        s->ck[b].wr = 1;
        s->ck[b].rows = s->pend[i].rows;
        w->at[w->n] = at;
        w->row[w->n] = at - 1 - pos0;
        w->slot[w->n] = b;
        w->st[w->n++] = s->ck[b].st;
        i++;
    }
}

static void ck_done(RqSeq *s) {
    // after the pass (s->len updated): the checkpoints it took are usable, the pending ones it reached gone
    for (int i = 0; i < s->ckw.n; i++)
        s->ck[s->ckw.slot[i]].wr = 0;
    s->ckw.n = 0;
    int k = 0;
    while (k < s->npend && s->pend[k].pos <= s->len) k++;
    if (k) memmove(s->pend, s->pend + k, (size_t)(s->npend -= k) * sizeof(s->pend[0]));
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

// rq_seq_fill_synthetic (a test hook, runq.h): a long context without its prefill, for timing (TODO G0 (b))
typedef struct { RqSeq *s; int n, next; uint64_t seed; } FillJob;

static inline float fill_val(uint64_t *x, float a) {
    // xorshift64*: a value in [-a, a)
    *x ^= *x >> 12; *x ^= *x << 25; *x ^= *x >> 27;
    return ((float)((*x * 0x2545F4914F6CDD1DULL) >> 40) * 0x1p-24f * 2.0f - 1.0f) * a;
}

static void fill_worker(void *arg, int tid, int nt) {
    // items: (attention layer, K / V) of the cache, its whole blocks up to position n - 1 pre-faulted (as a prefill
    // does: kv_populate_one), the first block written with pseudo-random values in the cache's type and copied to the
    // others; item 2 n_attn: the recurrent state
    (void)tid; (void)nt;
    FillJob *j = arg;
    RqSeq *s = j->s;
    const RqModel *m = s->m;
    const Config *p = &m->config;
    size_t be = (size_t)ATT_P * m->kv_dim, nb = (size_t)(j->n + ATT_P - 1) / ATT_P; // (elements per block, blocks)
    for (int it; (it = __atomic_fetch_add(&j->next, 1, __ATOMIC_RELAXED)) <= 2 * m->n_attn;) {
        uint64_t x = j->seed * 0x9E3779B97F4A7C15ULL + (uint64_t)it * 0xD1B54A32D192ED03ULL + 1;
        if (it == 2 * m->n_attn) { // the state: conv inputs and S, small values
            float *st = s->ssm;
            for (size_t i = 0; i < s->ssm_bytes / sizeof(float); i++)
                st[i] = fill_val(&x, 0.05f);
            continue;
        }
        int l = it / 2, kv = it % 2;
        kv_populate_one(s, p, 0, j->n, it);
        char *c = (char *)(kv ? s->value_cache : s->key_cache) + (size_t)l * s->kv_len * m->kv_dim * s->kv_esz;
        float a = kv ? 1.0f : 4.0f; // (|K| after the k norm: a few units, |V| ~1)
        for (size_t e = 0; e < be; e++) {
            float v = fill_val(&x, a);
            if (s->kvt == KV_F32) ((float *)c)[e] = v;
            else ((uint16_t *)c)[e] = s->kvt == KV_F16 ? f32_f16(v) : f32_bf16(v);
        }
        for (size_t b = 1; b < nb; b++)
            memcpy(c + b * be * s->kv_esz, c, be * s->kv_esz);
    }
}

int rq_seq_fill_synthetic(RqSeq *s, int n, unsigned long long seed) {
    if (!s || n < 1 || n > s->cap) return RQ_ERR_ARG;
    if (s->m->hy.on) {
        ck_forget_pending(s);
        ck_drop(s, -1);
    }
    FillJob j = { s, n, 0, seed };
    rq_parallel(fill_worker, &j);
    int V = s->m->config.vocab_size;
    for (int i = 0; i < n; i++) // (tokens: common words' ids would do; any valid id)
        s->tokens[i] = (int)((seed + (unsigned long long)i * 7919u) % (unsigned long long)V);
    s->len = n;
    return RQ_OK;
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
static int decode_counters(const RqModel *m) { return 2 * m->config.n_layers * m->config.n_kv_heads + dn_counters(m); }

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

    const Hybrid *hy = &m->hy;
    int qh = m->qh, rd = m->rope_dim;

    if (pos % ATT_T == 0) { // first position of a KV cache tile: pre-fault the tile (all attention layers)
        PH_BEGIN_DYN(2 * m->n_attn, 1)
        for (int it = lo; it < hi; it++)
            kv_populate_one(s, p, pos, 1, it);
        PH_END_NOEPI
    }

    // forward all the layers (layer 0's attention rmsnorm + quantize: done by decode_step)
    for (int l = 0; l < p->n_layers; l++) {
        int al = m->kv_layer[l], li = m->lin_layer[l];
        if (li >= 0) {
            // Qwen3.5's linear attention, in one phase (dependency counters dc .. dc + 2 nv): item 0 the alpha / beta
            // rows (counter dc); then per value head its v / z rows (and if h < nk key head h's q / k rows) and dn_prep
            // (after item 0; counter dc + 1 + h); then per (head, block of DN_BW v columns) dn_rec (after the prep of
            // the head and of its key head's item h % nk); the last block of a head to finish runs dn_out and quantizes
            // the head's outputs for ssm_out (into aq; head slices are whole groups)
            // (F32 alpha / beta: items 0 .. nab - 1, ab_items, on the fp32 row xb; counter dc counts them)
            float *st = s->ssm + (size_t)li * m->ssm_layer;
            int nblk = dn_nblk(hy), dc = dn_counter0(m, l), nab = ab_items(w->wa + li, w->wb + li, hy->nv);
            PH_BEGIN_DYN(nab + hy->nv + hy->nv * nblk, 1)
            for (int it = lo; it < hi; it++) {
                if (it < nab) {
                    float *ab = c->lin + hy->conv_dim + hy->vdim;
                    if (nab == 1) {
                        matvec_rows(ab, &c->xq, c->xq_corr, c->xq_xf, c->xq_c16, w->wa + li, dim, 0, hy->nv);
                        matvec_rows(ab + hy->nv, &c->xq, c->xq_corr, c->xq_xf, c->xq_c16, w->wb + li, dim, 0, hy->nv);
                    } else {
                        int r0, r1, beta;
                        QuantizedTensor *t = (QuantizedTensor *)ab_item(w->wa + li, w->wb + li, hy->nv, it, &r0, &r1, &beta);
                        if (t->f)
                            f32_rows(ab + beta * hy->nv, t->f, c->xb, dim, r0, r1);
                        else
                            matvec_rows(ab + beta * hy->nv, &c->xq, c->xq_corr, c->xq_xf, c->xq_c16, t, dim, r0, r1);
                    }
                    rq_count(S, dc, 1);
                    continue;
                }
                int iv = it - (nab - 1); // (the items below numbered as with one alpha / beta item)
                if (iv <= hy->nv) {
                    int h = iv - 1, i0 = h * hy->dv;
                    int r0[3] = { h * hy->dk, hy->nk * hy->dk + h * hy->dk, 2 * hy->nk * hy->dk + h * hy->dv }, rn[3] = { hy->dk, hy->dk, hy->dv };
                    for (int t = h < hy->nk ? 0 : 2; t < 3; t++)
                        matvec_rows(c->lin, &c->xq, c->xq_corr, c->xq_xf, c->xq_c16, w->wqkv + li, dim, r0[t], r0[t] + rn[t]);
                    matvec_rows(c->lin + hy->conv_dim, &c->xq, c->xq_corr, c->xq_xf, c->xq_c16, w->wz + li, dim, i0, i0 + hy->dv);
                    rq_wait_count(S, dc, nab);
                    dn_prep(m, li, h, st, pos == 0, dn_ck(s, li, pos), c->lin);
                    rq_count(S, dc + 1 + h, 1);
                    continue;
                }
                int h = (iv - 1 - hy->nv) / nblk, jb = (iv - 1 - hy->nv) % nblk, i0 = h * hy->dv;
                rq_wait_count(S, dc + 1 + h, 1);
                if (h >= hy->nk) // (its key head's q / k)
                    rq_wait_count(S, dc + 1 + h % hy->nk, 1);
                dn_rec(m, h, jb, st, pos == 0, &s->ckw, (size_t)li * m->ssm_layer, c->lin, c->xb + i0);
                if (rq_count(S, dc + 1 + hy->nv + h, 1) < nblk)
                    continue;
                dn_out(m, li, h, c->lin, c->xb + i0);
                QuantizedTensor ag = qsub(&c->aq, i0);
                quantize(&ag, c->xb + i0, hy->dv);
                matvec_prep(c->aq_corr, c->aq_xf, c->aq_c16, &c->aq, i0, hy->dv);
            }
            PH_END_NOEPI
        } else {
        uint64_t loff = al * (uint64_t)s->kv_len * kv_dim; // kv cache layer offset for convenience
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
                if (r < kv_mul) { // (Qwen3.5: the head's rows are [q | gate])
                    int h = g * kv_mul + r;
                    matvec_rows(c->q, &c->xq, c->xq_corr, c->xq_xf, c->xq_c16, w->wq + al, dim, h * qh, (h + 1) * qh);
                    qk_norm_rope(c->q + h * qh, 1, qh, w->q_norm_weights + al * hd, hd, c->rope, rd);
                } else if (r == kv_mul) {
                    matvec_rows(c->k, &c->xq, c->xq_corr, c->xq_xf, c->xq_c16, w->wk + al, dim, g * hd, (g + 1) * hd);
                    qk_norm_rope(c->k + g * hd, 1, hd, w->k_norm_weights + al * hd, hd, c->rope, rd);
                    kv_store(s->att, kl, NULL, pos, g, nkv, hd, c->k + g * hd, NULL);
                } else {
                    matvec_rows(c->v, &c->xq, c->xq_corr, c->xq_xf, c->xq_c16, w->wv + al, dim, g * hd, (g + 1) * hd);
                    kv_store(s->att, NULL, vl, pos, g, nkv, hd, NULL, c->v + g * hd);
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
                qv[r] = c->q + (g * kv_mul + r) * qh;
                ov[r] = c->xb + (g * kv_mul + r) * hd;
                pv[r] = part + (size_t)r * c->att_nb * pst;
                npv[r] = pos + 1;
            }
            size_t o = kv_tile(0, g, nkv, hd);
            attention_part(s->att, kv_mul, qv, ov, nb == 1 ? NULL : pv, npv, b, b + 1, kl + o * s->kv_esz, vl + o * s->kv_esz, bs,
                           hd, m->att_scale, c->att + ((size_t)g * c->att_nb + b) * kv_mul * ATT_P, conv);
            if (nb > 1) { // the last block of this kv head to finish combines them
                if (rq_count(S, (p->n_layers + l) * nkv + g, 1) < nb)
                    continue;
                for (int r = 0; r < kv_mul; r++)
                    att_combine(ov[r], nb, pv[r], pst, hd);
            }
            if (hy->on)
                for (int r = 0; r < kv_mul; r++)
                    attn_gate(ov[r], qv[r] + hd, hd);
            int i0 = g * kv_mul * hd;
            QuantizedTensor ag = qsub(&c->aq, i0);
            quantize(&ag, c->xb + i0, kv_mul * hd);
            matvec_prep(c->aq_corr, c->aq_xf, c->aq_c16, &c->aq, i0, kv_mul * hd);
        }
        PH_END_NOEPI
        }

        // final matmul to get the output of the attention (16-row blocks; Qwen3.5's linear layers: ssm_out);
        // epilogue: residual connection back into x, ffn rmsnorm, quantize
        PH_BEGIN(dim / 16, 1)
            if (li >= 0)
                matvec_rows(c->xb, &c->aq, c->aq_corr, c->aq_xf, c->aq_c16, w->wso + li, hy->vdim, lo * 16, hi * 16);
            else
                matvec_rows(c->xb, &c->aq, c->aq_corr, c->aq_xf, c->aq_c16, w->wo + al, all_heads_dim, lo * 16, hi * 16);
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
    rope_row(c->rope, c->m->rope_dim, pos, c->m->rope_log_theta);
    decode_norm(c, p, &c->m->weights, 0, 0); // layer 0's attention rmsnorm + quantize

    if (rq_begin(&c->sync, decode_phases(p), decode_counters(c->m), rq_nthreads(), rq_spin_dec) != 0)
        return ctx_fail(c, RQ_ERR_NOMEM, "malloc failed!");
    DecodeJob job = { c, s, pos, logits };
    ck_plan(s, pos, 1);
    rq_parallel(decode_worker, &job);
    s->tokens[pos] = token;
    s->len = pos + 1;
    ck_done(s);
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

static int prefill_phases(const Config *p) { return 10 * p->n_layers + 4; } // (a linear layer: 10 with its 3 recurrence phases)

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
    const Hybrid *hy = &m->hy;
    int qh = m->qh, qd = p->n_heads * qh, rd = m->rope_dim, lw = hy->on ? lin_width(hy) : 0;
    int rch = n / (4 * nt) > 1 ? n / (4 * nt) : 1; // rows per claim in the per-row phases
    const PfSeg *seg = c->seg;
    size_t bs = (size_t)nkv * ATT_P * hd;
    uint8_t *scr = c->kq_scr ? c->kq_scr + (size_t)tid * c->kq_scr_bytes : NULL;
    float *att = c->att_b + (size_t)tid * att_scores_floats((size_t)m->att_rows * kv_mul, c->att_nb);
    float *part = c->att_pb + (size_t)tid * m->att_rows * kv_mul * c->att_nb * ATT_PSTRIDE(hd);
    float *conv = c->att_conv + (size_t)tid * 2 * ATT_P * hd;
    Gemm G;

    // embeddings, RoPE cos / sin per row, layer 0's attention rmsnorm + quantize; KV tile pre-faults
    PH_BEGIN(n + 2 * m->n_attn * job->npop, 1)
    for (int b = lo; b < hi; b++) {
        if (b >= n) {
            const PfSeg *sg = &seg[c->pop_seg[(b - n) / (2 * m->n_attn)]];
            kv_populate_one(sg->seq, p, sg->pos0, sg->n, (b - n) % (2 * m->n_attn));
            continue;
        }
        float *x = c->x_b + (size_t)b * dim, *xb = c->xb_b + (size_t)b * xs;
        QuantizedTensor xq = qrow(&c->xq_b, b, xs);
        embed_row(x, w->q_tokens, c->row_tok[b], dim);
        rope_row(c->rope_b + (size_t)b * hd, rd, c->row_pos[b], m->rope_log_theta);
        rmsnorm(xb, x, w->rms_att_weight, dim);
        quantize(&xq, xb, dim);
        act_prep_row(&xq, dim, c->bq_corr, c->bq_xf, c->bq_c16, b);
    }
    PH_END_NOEPI
    PPROF(PF_NORM);

    for (int l = 0; l < p->n_layers; l++) {
        int al = m->kv_layer[l], li = m->lin_layer[l];
        uint64_t loff = al * (uint64_t)kv_dim; // (times the sequence's kv_len)

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

        if (li >= 0) {
            // Qwen3.5's linear attention: the qkv / z / alpha / beta GEMMs (one phase, into lin_b rows), then dn_prep per
            // (segment, channel group: q / k of a key head, v of a value head, dn_prep_item) over the segment's rows in
            // order (the conv state), dn_rec per (segment, value head, block
            // of DN_BW v columns) over the rows in order (the block of S, 16 KB, stays in this thread's L1 for the whole
            // chunk), and per row dn_out + the quantization of the row's outputs for ssm_out
            float *lb = c->lin_b;
            MatJob lj[4] = { { c->lin_b, lw, w->wqkv + li, hy->conv_dim }, { c->lin_b + hy->conv_dim, lw, w->wz + li, hy->vdim },
                             { c->lin_b + hy->conv_dim + hy->vdim, lw, w->wa + li, hy->nv },
                             { c->lin_b + hy->conv_dim + hy->vdim + hy->nv, lw, w->wb + li, hy->nv } };
            gemm_plan(c, &G, 4, lj, &c->xq_b, xs, n, dim, c->bq_corr, c->bq_xf, c->bq_c16, nt);
            G.xa = c->xb_b; // (F32 alpha / beta: the rmsnorm'd rows)
            G.ldxa = xs;
            {
                const void *last = NULL;
                PH_BEGIN_U(G.it0[G.nj], G.chunk, G.unit)
                for (int it = lo; it < hi; it++)
                    gemm_item(&G, it, scr, &last);
                PH_END_NOEPI
            }
            PPROF(PF_QKV);
            int npi = dn_prep_items(hy);
            PH_BEGIN_DYN(job->nseg * npi, 1)
            for (int it = lo; it < hi; it++) {
                const PfSeg *sg = &seg[it / npi];
                RqSeq *sq = sg->seq;
                int h, t;
                dn_prep_item(hy, it % npi, &h, &t);
                float *st = sq->ssm + (size_t)li * m->ssm_layer;
                for (int b = sg->b0; b < sg->b0 + sg->n; b++) {
                    int pos = c->row_pos[b];
                    dn_prep_part(m, li, h, t, st, pos == 0, dn_ck(sq, li, pos), lb + (size_t)b * lw,
                                 b + 1 < sg->b0 + sg->n ? lb + (size_t)(b + 1) * lw : NULL);
                }
            }
            PH_END_NOEPI
                        PPROF(PF_DNPREP);
            int nblk = dn_nblk(hy);
            PH_BEGIN_DYN(job->nseg * hy->nv * nblk, 1)
            for (int it = lo; it < hi; it++) {
                const PfSeg *sg = &seg[it / (hy->nv * nblk)];
                RqSeq *sq = sg->seq;
                int h = it / nblk % hy->nv, jb = it % nblk, i0 = h * hy->dv;
                float *st = sq->ssm + (size_t)li * m->ssm_layer;
                dn_rec_run(m, h, jb, st, sg->pos0 == 0, lb + (size_t)sg->b0 * lw, lw, c->xb_b + (size_t)sg->b0 * xs + i0, xs, sg->n,
                           &sq->ckw, (size_t)li * m->ssm_layer);
            }
            PH_END_NOEPI
            PPROF(PF_DNREC);
            PH_BEGIN(n, rch)
            for (int b = lo; b < hi; b++) {
                for (int h = 0; h < hy->nv; h++)
                    dn_out(m, li, h, lb + (size_t)b * lw, c->xb_b + (size_t)b * xs + h * hy->dv);
                QuantizedTensor xr = qrow(&c->xq_b, b, xs);
                quantize(&xr, c->xb_b + (size_t)b * xs, hy->vdim);
                act_prep_row(&xr, hy->vdim, c->bq_corr, c->bq_xf, c->bq_c16, b);
            }
            PH_END_NOEPI
            PPROF(PF_DNOUT);
            MatJob so = { c->xb_b, xs, w->wso + li, dim };
            gemm_plan(c, &G, 1, &so, &c->xq_b, xs, n, hy->vdim, c->bq_corr, c->bq_xf, c->bq_c16, nt);
            {
                const void *last = NULL;
                PH_BEGIN_U(G.it0[G.nj], G.chunk, G.unit)
                for (int it = lo; it < hi; it++)
                    gemm_item(&G, it, scr, &last);
                PH_END_NOEPI
            }
            PPROF(PF_WO);
        } else {
        // qkv GEMMs (one phase)
        MatJob qkv[3] = { { c->q_b, qd, w->wq + al, qd }, { c->k_b, kv_dim, w->wk + al, kv_dim },
                          { c->v_b, kv_dim, w->wv + al, kv_dim } };
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
                qk_norm_rope(c->q_b + (size_t)b * qd + g * kv_mul * qh, kv_mul, qh, w->q_norm_weights + al * hd, hd, cs, rd);
                qk_norm_rope(kr, 1, hd, w->k_norm_weights + al * hd, hd, cs, rd);
                kv_store(sq->att, kl, vl, t, g, nkv, hd, kr, c->v_b + (size_t)b * kv_dim + g * hd);
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
                    qv[nv] = c->q_b + (size_t)b * qd + h * qh;
                    ov[nv] = c->xb_b + (size_t)b * xs + h * hd;
                    npv[nv] = c->row_pos[b] + 1;
                    if (sp >= 0)
                        pv[nv] = c->att_sp + ((size_t)sp * p->n_heads + h) * job->nbmax * ATT_PSTRIDE(hd);
                }
            size_t o = kv_tile(0, g, nkv, hd);
            if (sp >= 0) { // block j of the single row b0
                int j = i % nbk;
                attention_part(sq->att, nv, qv, ov, pv, npv, j, j + 1, kl + o * sq->kv_esz, vl + o * sq->kv_esz, bs, hd, m->att_scale,
                               att, conv);
                if (rq_count(S, l * ncl + job->nrb + sp * nkv + g, 1) < nbk)
                    continue;
                for (int r = 0; r < nv; r++)
                    att_combine(ov[r], nbk, pv[r], ATT_PSTRIDE(hd), hd);
            } else {
                attention_kv(sq->att, nv, qv, ov, npv, kl + o * sq->kv_esz, vl + o * sq->kv_esz, bs, hd, m->att_scale, att, part,
                             conv);
            }
            if (rq_count(S, l * ncl + rb, 1) < nkv)
                continue;
            for (int b = b0; b < b1; b++) {
                QuantizedTensor xq = qrow(&c->xq_b, b, xs);
                if (hy->on) // (the gate: after every kv head of the row is done)
                    for (int h = 0; h < p->n_heads; h++)
                        attn_gate(c->xb_b + (size_t)b * xs + h * hd, c->q_b + (size_t)b * qd + h * qh + hd, hd);
                quantize(&xq, c->xb_b + (size_t)b * xs, all_heads_dim);
                act_prep_row(&xq, all_heads_dim, c->bq_corr, c->bq_xf, c->bq_c16, b);
            }
        }
        PH_END_NOEPI
        PPROF(PF_ATT);

        MatJob wo = { c->xb_b, xs, w->wo + al, dim };
        gemm_plan(c, &G, 1, &wo, &c->xq_b, xs, n, all_heads_dim, c->bq_corr, c->bq_xf, c->bq_c16, nt);
        {
            const void *last = NULL;
            PH_BEGIN_U(G.it0[G.nj], G.chunk, G.unit)
            for (int it = lo; it < hi; it++)
                gemm_item(&G, it, scr, &last);
            PH_END_NOEPI
        }
        PPROF(PF_WO);
        }

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

static void rows_mm(const RqCtx *c, QuantizedTensor *w, float *out, int ldo, QuantizedTensor *x, int ldx, const int32_t *corr,
                    const float *xf, const int32_t *c16, int nb, int k, int i0, int i1) {
    // out[b * ldo + i] = W row i . x row b for i0 <= i < i1 (multiples of 16), b < nb (x row b at x->q +
    // b * ldx; the activation prep of act_prep_row): the small-batch kernel (c: its prefetch distance), else the
    // decode GEMV per row. Either way each output is the GEMV's (bit-identical). Q2_K, i-quants: iq_rows
    if (w->kp && iq_type(w->type)) {
        iq_rows(w->type, out, ldo, x->q, ldx, x->s, c16, nb, w->kp, k, i0 / 16, i1 / 16);
        return;
    }
#ifdef RUNQ_SMB
    if (w->kp || w->p) {
        smb_rows(w->kp ? w->type : 0, out, ldo, x->q, ldx, x->s, corr, xf, c16, nb, w->kp ? w->kp : w->p, k, GS, i0 / 16, i1 / 16,
                 c->smb_pf);
        return;
    }
#else
    (void)c;
#endif
    for (int b = 0; b < nb; b++) {
        QuantizedTensor xr = qrow(x, b, ldx);
        matvec_rows(out + (size_t)b * ldo, &xr, corr + (size_t)b * (k / GS), xf ? xf + (size_t)b * (k / GS) : NULL,
                    c16 ? c16 + (size_t)b * (k / 16) : NULL, w, k, i0, i1);
    }
}

typedef struct {
    RqCtx *c;
    int n;       // rows (row b at position row_pos[b] of seg[row_seg[b]].seq)
    int nseg;    // segments (consecutive rows of one sequence each, at most bd_seg)
    int nrb;     // attention row blocks (rb_seg / rb_b0 / rb_b1: rows of one segment, at most att_rows)
    int nsel;    // rows with logits (sel_row), into out
    float *out;
    int epi;     // the norms as phase epilogues (else phases of their own)
    int split;   // attention split over blocks of ATT_P positions (row_it0: items of row blocks 0..rb-1 per kv head)
    int nbmax;   // blocks of the longest row (att_sp layout)
} BatchDecJob;

static int bd_phases(const RqModel *m, int epi) { return ((epi ? 4 : 6) + m->hy.on) * m->config.n_layers + 4; } // (+ Qwen3.5's quantization)
// per layer: per kv head its q / k / v items done, its QK-norm + RoPE + stores done; per row block its kv heads'
// attention done; per (row block, kv head) its blocks done (split attention); (row blocks <= rows n)
static int bd_counters(int nkv, int n, int nv) { // (nv: Qwen3.5's value heads: a linear layer needs 1 + 2 nv)
    int a = 2 * nkv + n * (1 + nkv), b = nv ? 1 + 2 * nv : 0;
    return a > b ? a : b;
}
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
    // rq_forward_batch over N rows in short segments (decode rows of N sequences, a speculative verify segment
    // of a few rows of one sequence, or both): decode_worker's structure for N rows at once, per layer 4
    // phases (6 with the norms as phases): [q, k, v per head over all rows (rows_mm) + QK-norm + RoPE + each
    // row's K / V into its sequence's cache; attention per (row block, kv head) or (row block, kv head, block),
    // once the 4 items of its kv head are done (dependency counters): a row block is up to att_rows rows of
    // one segment, as prefill_worker's, each row over its positions 0..pos (causal: the K / V of the
    // segment's earlier rows are stored by then); the last kv head of a row block quantizes its rows' output]
    // [wo; residual + rmsnorm + quantize per row] [w1 + w3 + SwiGLU + quantize per ff_rows hidden rows, all
    // rows] [w2; residual + the next rmsnorm + quantize per row], then the classifier over the rows with
    // logits. Qwen3.5's linear layers: dn_prep per value head over the rows in order, dn_rec per (value head,
    // block) over each segment's rows in order (dn_rec_run, as prefill_worker). Every output is computed
    // whole by the same functions as in decode_worker / prefill_worker: each row's results are those of its
    // sequence decoded alone
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
    const Hybrid *hy = &m->hy;
    int qh = m->qh, qd = nh * qh, rd = m->rope_dim, lw = hy->on ? lin_width(hy) : 0;
    // q / k / v items per kv head, qr rows each (Qwen3.5: a query head's rows are [q | gate], 2 hd)
    int qr = hd % BD_QKV_ROWS ? hd : BD_QKV_ROWS, ng = (kv_mul * qh + 2 * hd) / qr;
    int ncl = bd_counters(nkv, n, hy->nv); // dependency counters per layer (bd_counters)
    int nrb = job->nrb, nat = job->split ? c->row_it0[nrb] : nrb; // attention items per kv head
    size_t bs = (size_t)nkv * ATT_P * hd, pst = ATT_PSTRIDE(hd);
    float *att = c->att_b + (size_t)tid * att_scores_floats((size_t)m->att_rows * kv_mul, c->att_nb);
    float *part = c->att_pb + (size_t)tid * m->att_rows * kv_mul * c->att_nb * pst;
    float *conv = c->att_conv + (size_t)tid * 2 * ATT_P * hd;

    // embeddings, RoPE cos / sin per row, layer 0's attention rmsnorm + quantize; the KV tile pre-fault of
    // the rows starting a new tile (row_tok of the others: -1)
    int npop = 0, NA = m->n_attn;
    for (int b = 0; b < n; b++)
        npop += c->row_pos[b] % ATT_T == 0;
    PH_BEGIN(n + 2 * NA * npop, 1)
    for (int it = lo; it < hi; it++) {
        if (it >= n) {
            int q = (it - n) / (2 * NA), b = 0;
            for (;; b++)
                if (c->row_pos[b] % ATT_T == 0 && q-- == 0) break;
            kv_populate_one(seg[c->row_seg[b]].seq, p, c->row_pos[b], 1, (it - n) % (2 * NA));
            continue;
        }
        embed_row(c->x_b + (size_t)it * dim, w->q_tokens, c->row_tok[it], dim);
        rope_row(c->rope_b + (size_t)it * hd, rd, c->row_pos[it], m->rope_log_theta);
        bd_norm_row(c, p, w, it, 0, 0, it);
    }
    PH_END_NOEPI
    PPROF(PF_NORM);

    for (int l = 0; l < p->n_layers; l++) {
        int al = m->kv_layer[l], li = m->lin_layer[l];
        uint64_t loff = al * (uint64_t)kv_dim; // (times the sequence's kv_len)
        int cb = l * ncl;                     // this layer's dependency counters

        if (li >= 0) {
            // Qwen3.5's linear attention over all rows, as decode_worker's: item 0 alpha / beta (counter cb), then per
            // value head its v / z rows (h < nk: and key head h's q / k rows) and, per row, dn_prep (counter cb + 1 + h),
            // then per (head, block) dn_rec (after the preps of h and h % nk) over the rows (each on its sequence), the
            // last block of a head to finish runs dn_out on every row
            // (the rows are quantized by the next phase)
            int nblk = dn_nblk(hy);
            int nab = ab_items(w->wa + li, w->wb + li, hy->nv); // (F32 alpha / beta: items 0 .. nab - 1 on the fp32 rows)
            PH_BEGIN_DYN(nab + hy->nv + hy->nv * nblk, 1)
            for (int it = lo; it < hi; it++) {
                float *lb = c->lin_b;
                if (it < nab) {
                    float *ab = lb + hy->conv_dim + hy->vdim;
                    if (nab == 1) {
                        rows_mm(c, w->wa + li, ab, lw, &c->xq_b, xs, c->bq_corr, c->bq_xf, c->bq_c16, n, dim, 0, hy->nv);
                        rows_mm(c, w->wb + li, ab + hy->nv, lw, &c->xq_b, xs, c->bq_corr, c->bq_xf, c->bq_c16, n, dim, 0, hy->nv);
                    } else {
                        int r0, r1, beta;
                        QuantizedTensor *t = (QuantizedTensor *)ab_item(w->wa + li, w->wb + li, hy->nv, it, &r0, &r1, &beta);
                        for (int b = 0; b < n && t->f; b++)
                            f32_rows(ab + (size_t)b * lw + beta * hy->nv, t->f, c->xb_b + (size_t)b * xs, dim, r0, r1);
                        if (!t->f)
                            rows_mm(c, t, ab + beta * hy->nv, lw, &c->xq_b, xs, c->bq_corr, c->bq_xf, c->bq_c16, n, dim, r0, r1);
                    }
                    rq_count(S, cb, 1);
                    continue;
                }
                int iv = it - (nab - 1); // (the items below numbered as with one alpha / beta item)
                if (iv <= hy->nv) {
                    int h = iv - 1, i0 = h * hy->dv;
                    int r0[3] = { h * hy->dk, hy->nk * hy->dk + h * hy->dk, 2 * hy->nk * hy->dk + h * hy->dv }, rn[3] = { hy->dk, hy->dk, hy->dv };
                    for (int t = h < hy->nk ? 0 : 2; t < 3; t++)
                        rows_mm(c, w->wqkv + li, lb, lw, &c->xq_b, xs, c->bq_corr, c->bq_xf, c->bq_c16, n, dim, r0[t], r0[t] + rn[t]);
                    rows_mm(c, w->wz + li, lb + hy->conv_dim, lw, &c->xq_b, xs, c->bq_corr, c->bq_xf, c->bq_c16, n, dim, i0, i0 + hy->dv);
                    rq_wait_count(S, cb, nab);
                    for (int b = 0; b < n; b++) { // (a segment's rows in order: the conv state)
                        RqSeq *sq = seg[c->row_seg[b]].seq;
                        int pos = c->row_pos[b];
                        dn_prep(m, li, h, sq->ssm + (size_t)li * m->ssm_layer, pos == 0, dn_ck(sq, li, pos), lb + (size_t)b * lw);
                    }
                    rq_count(S, cb + 1 + h, 1);
                    continue;
                }
                int h = (iv - 1 - hy->nv) / nblk, jb = (iv - 1 - hy->nv) % nblk, i0 = h * hy->dv;
                rq_wait_count(S, cb + 1 + h, 1);
                if (h >= hy->nk) // (its key head's q / k)
                    rq_wait_count(S, cb + 1 + h % hy->nk, 1);
                for (int i = 0; i < job->nseg; i++) { // (a segment's rows in order: the recurrence)
                    const PfSeg *sg = &seg[i];
                    RqSeq *sq = sg->seq;
                    dn_rec_run(m, h, jb, sq->ssm + (size_t)li * m->ssm_layer, sg->pos0 == 0, lb + (size_t)sg->b0 * lw, lw,
                               c->xb_b + (size_t)sg->b0 * xs + i0, xs, sg->n, &sq->ckw, (size_t)li * m->ssm_layer);
                }
                if (rq_count(S, cb + 1 + hy->nv + h, 1) < nblk)
                    continue;
                for (int b = 0; b < n; b++)
                    dn_out(m, li, h, lb + (size_t)b * lw, c->xb_b + (size_t)b * xs + i0);
            }
            PH_END_NOEPI
            PPROF(PF_BDQKV);
            // (the outputs are quantized per row by the next phase's items: xq_b is read by the items above)
        } else {
        // q / k / v over all rows in items of qr weight rows (the kv head's query heads, its key head, its
        // value head: ng items per kv head); the last of a kv head's items to finish runs its QK-norm +
        // RoPE and the K / V stores of every row, then the attention items of that kv head may start
        PH_BEGIN_DYN(nkv * ng + nkv * nat, 1)
        for (int it = lo; it < hi; it++) {
            // (the attention items of each kv head interleaved after the next head's q / k / v items instead:
            // ~5-10% slower at 8 decode rows, 8 and 14 threads)
            if (it < nkv * ng) {
                int g = it / ng, r = (it % ng) * qr, nq = kv_mul * qh; // r: row offset in [q heads | k | v] of head g
                if (r < nq)
                    rows_mm(c, w->wq + al, c->q_b, qd, &c->xq_b, xs, c->bq_corr, c->bq_xf, c->bq_c16, n, dim, g * nq + r, g * nq + r + qr);
                else if (r < nq + hd)
                    rows_mm(c, w->wk + al, c->k_b, kv_dim, &c->xq_b, xs, c->bq_corr, c->bq_xf, c->bq_c16, n, dim, g * hd + r - nq,
                            g * hd + r - nq + qr);
                else
                    rows_mm(c, w->wv + al, c->v_b, kv_dim, &c->xq_b, xs, c->bq_corr, c->bq_xf, c->bq_c16, n, dim, g * hd + r - nq - hd,
                            g * hd + r - nq - hd + qr);
                if (rq_count(S, cb + g, 1) < ng)
                    continue;
                for (int b = 0; b < n; b++) {
                    RqSeq *sq = seg[c->row_seg[b]].seq;
                    const float *cs = c->rope_b + (size_t)b * hd;
                    float *kr = c->k_b + (size_t)b * kv_dim + g * hd;
                    qk_norm_rope(c->q_b + (size_t)b * qd + g * nq, kv_mul, qh, w->q_norm_weights + al * hd, hd, cs, rd);
                    qk_norm_rope(kr, 1, hd, w->k_norm_weights + al * hd, hd, cs, rd);
                    kv_store(sq->att, (char *)sq->key_cache + loff * sq->kv_len * sq->kv_esz,
                             (char *)sq->value_cache + loff * sq->kv_len * sq->kv_esz, c->row_pos[b], g, nkv, hd, kr,
                             c->v_b + (size_t)b * kv_dim + g * hd);
                }
                rq_count(S, cb + nkv + g, 1);
                continue;
            }
            // attention item: kv head g, row block u (rows b0..b1-1 of one segment, each row's kv_mul query heads
            // in one call: the K / V read once for them), block j of its longest row if split
            int a = it - nkv * ng, g = a / nat, i = a % nat, u = i, j = 0;
            if (job->split) {
                int lo_ = 0, hi_ = nrb; // the row block whose items hold i: row_it0[u] <= i < row_it0[u + 1]
                while (hi_ - lo_ > 1) {
                    int mid = (lo_ + hi_) / 2;
                    if (c->row_it0[mid] <= i) lo_ = mid; else hi_ = mid;
                }
                u = lo_;
                j = i - c->row_it0[u];
            }
            rq_wait_count(S, cb + nkv + g, 1);
            RqSeq *sq = seg[c->rb_seg[u]].seq;
            const char *kl = (const char *)sq->key_cache + loff * sq->kv_len * sq->kv_esz;
            const char *vl = (const char *)sq->value_cache + loff * sq->kv_len * sq->kv_esz;
            size_t o = kv_tile(0, g, nkv, hd) * sq->kv_esz;
            int b0 = c->rb_b0[u], b1 = c->rb_b1[u], nv = 0, nbk = (c->row_pos[b1 - 1] + 1 + ATT_P - 1) / ATT_P;
            const float *qv[ATT_NV_MAX];
            float *ov[ATT_NV_MAX], *pv[ATT_NV_MAX];
            int npv[ATT_NV_MAX];
            for (int b = b0; b < b1; b++)
                for (int r = 0; r < kv_mul; r++, nv++) {
                    qv[nv] = c->q_b + (size_t)b * qd + (g * kv_mul + r) * qh;
                    ov[nv] = c->xb_b + (size_t)b * xs + (g * kv_mul + r) * hd;
                    pv[nv] = c->att_sp + ((size_t)b * nh + g * kv_mul + r) * job->nbmax * pst;
                    npv[nv] = c->row_pos[b] + 1;
                }
            if (job->split && nbk > 1) { // block j's partial results; the last block of (row block, kv head) combines
                attention_part(sq->att, nv, qv, ov, pv, npv, j, j + 1, kl + o, vl + o, bs, hd, m->att_scale, att, conv);
                if (rq_count(S, cb + 2 * nkv + n + u * nkv + g, 1) < nbk)
                    continue;
                for (int r = 0; r < nv; r++) // (each row over its own blocks)
                    att_combine(ov[r], (npv[r] + ATT_P - 1) / ATT_P, pv[r], pst, hd);
            } else {
                attention_kv(sq->att, nv, qv, ov, npv, kl + o, vl + o, bs, hd, m->att_scale, att, part, conv);
            }
            if (rq_count(S, cb + 2 * nkv + u, 1) < nkv)
                continue;
            // the row block's last kv head: its rows' attention output quantized for wo (xq_b: every qkv item is done)
            for (int b = b0; b < b1; b++) {
                QuantizedTensor xq = qrow(&c->xq_b, b, xs);
                if (hy->on)
                    for (int h = 0; h < nh; h++)
                        attn_gate(c->xb_b + (size_t)b * xs + h * hd, c->q_b + (size_t)b * qd + h * qh + hd, hd);
                quantize(&xq, c->xb_b + (size_t)b * xs, ahd);
                act_prep_row(&xq, ahd, c->bq_corr, c->bq_xf, c->bq_c16, b);
            }
        }
        PH_END_NOEPI
        PPROF(PF_BDQKV);
        }

        // wo (16-row blocks, all rows; Qwen3.5's linear layers: ssm_out, after the quantization of the rows' outputs);
        // residual + ffn rmsnorm + quantize per row
        QuantizedTensor *wout = li >= 0 ? w->wso + li : w->wo + al;
        int kin = li >= 0 ? hy->vdim : ahd;
        if (li >= 0) {
            PH_BEGIN(n, 1)
            for (int b = lo; b < hi; b++) {
                QuantizedTensor xq = qrow(&c->xq_b, b, xs);
                quantize(&xq, c->xb_b + (size_t)b * xs, kin);
                act_prep_row(&xq, kin, c->bq_corr, c->bq_xf, c->bq_c16, b);
            }
            PH_END_NOEPI
        }
        if (job->epi) {
            PH_BEGIN(dim / 16, 1)
                rows_mm(c, wout, c->xb_b, xs, &c->xq_b, xs, c->bq_corr, c->bq_xf, c->bq_c16, n, kin, lo * 16, hi * 16);
            PH_EPILOGUE
                for (int b = 0; b < n; b++)
                    bd_norm_row(c, p, w, b, -1 - l, 1, b);
            PH_END
            PPROF(PF_WO);
        } else {
            PH_BEGIN(dim / 16, 1)
                rows_mm(c, wout, c->xb_b, xs, &c->xq_b, xs, c->bq_corr, c->bq_xf, c->bq_c16, n, kin, lo * 16, hi * 16);
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
            rows_mm(c, w->w1 + l, c->hb_b, hidden, &c->xq_b, xs, c->bq_corr, c->bq_xf, c->bq_c16, n, dim, i0, i0 + ff);
            rows_mm(c, w->w3 + l, c->hb2_b, hidden, &c->xq_b, xs, c->bq_corr, c->bq_xf, c->bq_c16, n, dim, i0, i0 + ff);
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
                rows_mm(c, w->w2 + l, c->xb_b, xs, &c->hq_b, hidden, c->bh_corr, c->bh_xf, c->bh_c16, n, hidden, lo * 16, hi * 16);
            PH_EPILOGUE
                for (int j = 0; j < nr; j++)
                    bd_norm_row(c, p, w, last ? c->sel_row[j] : j, l + 1, 1, j);
            PH_END
            PPROF(PF_FFN2);
        } else {
            PH_BEGIN(dim / 16, 1)
                rows_mm(c, w->w2 + l, c->xb_b, xs, &c->hq_b, hidden, c->bh_corr, c->bh_xf, c->bh_c16, n, hidden, lo * 16, hi * 16);
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
            rows_mm(c, w->wcls, job->out, p->vocab_size, &c->xq_b, xs, c->bq_corr, c->bq_xf, c->bq_c16, job->nsel, dim, lo * 16,
                    hi * 16 < p->vocab_size ? hi * 16 : p->vocab_size);
        PH_END_NOEPI
        PPROF(PF_CLS);
    }
}

static int batch_decode(RqCtx *c, int n, int nseg, int nrb, int nsel, float *out) {
    // rq_forward_batch's plan (segments of at most bd_seg rows, row blocks) -> batch_decode_worker; the row
    // blocks' attention split over blocks of ATT_P positions if the (row block, kv head) items alone are few
    // for the threads
    const Config *p = &c->m->config;
    int nkv = p->n_kv_heads, nbmax = 1, nt = rq_nthreads();
    for (int b = 0; b < n; b++) {
        int nbk = (c->row_pos[b] + 1 + ATT_P - 1) / ATT_P;
        nbmax = nbk > nbmax ? nbk : nbmax;
    }
    BatchDecJob job = { c, n, nseg, nrb, nsel, out, n <= c->bd_epi, nbmax > 1 && nrb * nkv < c->bd_split * nt, nbmax };
    if (job.split) {
        size_t need = (size_t)n * p->n_heads * nbmax * ATT_PSTRIDE(p->head_dim);
        if (need > c->att_sp_n) {
            float *sp = malloc(need * sizeof(float));
            if (!sp) return ctx_fail(c, RQ_ERR_NOMEM, "malloc failed!");
            free(c->att_sp);
            c->att_sp = sp;
            c->att_sp_n = need;
        }
        c->row_it0[0] = 0; // (per row block: the blocks of its last row)
        for (int u = 0; u < nrb; u++)
            c->row_it0[u + 1] = c->row_it0[u] + (c->row_pos[c->rb_b1[u] - 1] + 1 + ATT_P - 1) / ATT_P;
    }
    PROF_START(c);
    if (rq_begin(&c->sync, bd_phases(c->m, job.epi), p->n_layers * bd_counters(nkv, n, c->m->hy.on ? c->m->hy.nv : 0), nt, rq_spin_dec) != 0)
        return ctx_fail(c, RQ_ERR_NOMEM, "malloc failed!");
    rq_parallel(batch_decode_worker, &job);
    return RQ_OK;
}

int rq_forward_batch(RqCtx *c, const RqSegment *sg, int nseg, float *out, float **logits) {
    // One batched pass over the rows of nseg segments: the same computation per row as rq_decode on
    // its sequence, with every weight matrix applied to all rows at once (GEMM / small-batch work
    // items), and bit-identical results. Fills the KV caches. One parallel job in phases (threads.h):
    // batch_decode_worker if every segment is short (decode rows, a verify segment: at most bd_seg rows, up to
    // bd_max rows in all), else prefill_worker
    const RqModel *m = c->m;
    const Config *p = &m->config;
    int ar = m->att_rows, n = 0, ntile = 0, nrb = 0, nsel = 0, any_all = 0, segmax = 0;
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
        segmax = sg[i].n > segmax ? sg[i].n : segmax;
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
            c->row_seg[n + j] = i;
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

    for (int i = 0; i < nseg; i++) // (Qwen3.5: a checkpoint this pass reaches)
        ck_plan(c->seg[i].seq, c->seg[i].pos0, c->seg[i].n);
    if (n <= c->bd_max && segmax <= c->bd_seg) { // short segments only: batch_decode_worker
        int e = batch_decode(c, n, nseg, nrb, nsel, out);
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
        ck_done(s);
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
