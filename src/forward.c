/* Forward passes. */
#include "internal.h"

int prefill_chunk = PREFILL_CHUNK_DEFAULT;
const char *kv_type_name[] = { "f32", "bf16", "f16" };
int kv_type = KV_TYPE_DEFAULT;
#ifdef RUNQ_PROFILE
const char *pf_name[PF_N] = { "norm+quant+resid", "gemm qkv", "qk-norm+rope+kv", "attention+quant",
                              "gemm wo", "gemm w1+w3", "swiglu+quant", "gemm w2", "classifier" };
double pf_time[PF_N], pf_t;
#endif

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
    s->kv_len = (p->seq_len + ATT_P - 1) / ATT_P * ATT_P; // (whole blocks: the cache layout)
    s->att_nb = (s->kv_len + ATT_P - 1) / ATT_P;
    s->kv_size = (size_t)p->n_layers * s->kv_len * kv_dim;
    s->kvt = kv_type;
    s->kv_esz = s->kvt == KV_F32 ? 4 : 2;
    s->att_scale = 1.0f / sqrtf(p->head_dim);
    s->att_rows = ATT_NV_MAX / kv_mul < ATT_ROWS ? ATT_NV_MAX / kv_mul : ATT_ROWS;
    s->ff_rows = GS > 64 ? GS : 64;
    if (kv_mul < 1 || s->att_rows < 1 || p->head_dim % 16 || (kv_mul * p->head_dim) % GS ||
        p->hidden_dim % s->ff_rows || s->ff_rows % GS || p->dim % 16 || (s->kvt != KV_F32 && p->head_dim % 32)) {
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
        s->att_threads = rq_nthreads();
        s->pf_nrb = (s->nb_max + s->att_rows - 1) / s->att_rows;
        s->bq_corr = calloc(nb * (s->xs / GS) + 1, sizeof(int32_t));
        s->bh_corr = calloc(nb * (p->hidden_dim / GS) + 1, sizeof(int32_t));
        if (!s->bq_corr || !s->bh_corr) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
        if (KQ) {
            s->bq_xf = calloc(nb * (s->xs / GS) + 1, sizeof(float));
            s->bh_xf = calloc(nb * (p->hidden_dim / GS) + 1, sizeof(float));
            s->bq_c16 = calloc(nb * (s->xs / 16) + 1, sizeof(int32_t));
            s->bh_c16 = calloc(nb * (p->hidden_dim / 16) + 1, sizeof(int32_t));
            if (!s->bq_xf || !s->bh_xf || !s->bq_c16 || !s->bh_c16) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
#ifdef RUNQ_KQX
            int kmax = p->hidden_dim > s->xs ? p->hidden_dim : s->xs;
            s->kq_scr_bytes = (KQX_SCRATCH(kmax) + 63) / 64 * 64;
            s->kq_scr = aligned_alloc(64, s->kq_scr_bytes * s->att_threads);
            if (!s->kq_scr) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
#endif
        }
        s->att_b = calloc((size_t)s->att_threads * s->att_rows * kv_mul * s->att_nb * ATT_P, sizeof(float));
        s->att_pb = calloc((size_t)s->att_threads * s->att_rows * kv_mul * s->att_nb * ATT_PSTRIDE(p->head_dim), sizeof(float));
        s->rope_b = calloc(nb * p->head_dim, sizeof(float));
        if (!s->rope_b || !s->k_b || !s->v_b || !s->x_b || !s->xb_b || !s->q_b || !s->hb_b || !s->hb2_b || !s->xq_b.q || !s->xq_b.s ||
            !s->hq_b.q || !s->hq_b.s || !s->att_b || !s->att_pb) {
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
    s->xq_corr = calloc((all_heads_dim > p->dim ? all_heads_dim : p->dim) / GS + 1, sizeof(int32_t));
    s->hq_corr = calloc(p->hidden_dim / GS + 1, sizeof(int32_t));
    s->aq = (QuantizedTensor) { .q = calloc(all_heads_dim, sizeof(int8_t)), .s = calloc(all_heads_dim / GS, sizeof(float)) };
    s->aq_corr = calloc(all_heads_dim / GS + 1, sizeof(int32_t));
    if (!s->aq.q || !s->aq.s || !s->aq_corr) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    if (KQ) {
        s->aq_xf = calloc(all_heads_dim / GS + 1, sizeof(float));
        s->aq_c16 = calloc(all_heads_dim / 16 + 1, sizeof(int32_t));
        if (!s->aq_xf || !s->aq_c16) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
        s->xq_xf = calloc((all_heads_dim > p->dim ? all_heads_dim : p->dim) / GS + 1, sizeof(float));
        s->hq_xf = calloc(p->hidden_dim / GS + 1, sizeof(float));
        s->xq_c16 = calloc((all_heads_dim > p->dim ? all_heads_dim : p->dim) / 16 + 1, sizeof(int32_t));
        s->hq_c16 = calloc(p->hidden_dim / 16 + 1, sizeof(int32_t));
        if (!s->xq_xf || !s->hq_xf || !s->xq_c16 || !s->hq_c16) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    }
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc((size_t)p->n_heads * s->att_nb * ATT_P, sizeof(float));
    s->att_part = calloc((size_t)p->n_heads * s->att_nb * ATT_PSTRIDE(p->head_dim), sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->rope = calloc(p->head_dim, sizeof(float));
    // phase / counter capacity: decode 4 / layer, 2 counters per (layer, kv head); prefill 9 / layer,
    // a counter per (layer, attention row block)
    int ncnt = 2 * p->n_kv_heads > s->pf_nrb ? 2 * p->n_kv_heads : s->pf_nrb;
    rq_sync_init(&s->sync, 10 * p->n_layers + 8, p->n_layers * ncnt + 64);
    s->key_cache = alloc_big(s->kv_size * s->kv_esz);
    s->value_cache = alloc_big(s->kv_size * s->kv_esz);

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->hb || !s->hb2 || !s->q || !s->xq_corr || !s->hq_corr || !s->k || !s->v || !s->att || !s->att_part || !s->logits || !s->rope || !s->key_cache || !s->value_cache) {
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
    free(s->xq_xf);
    free(s->hq_xf);
    free(s->xq_c16);
    free(s->hq_c16);
    free(s->aq.q);
    free(s->aq.s);
    free(s->aq_corr);
    free(s->aq_xf);
    free(s->aq_c16);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free_big(s->key_cache, s->kv_size * s->kv_esz);
    free_big(s->value_cache, s->kv_size * s->kv_esz);
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
    free(s->att_pb);
    free(s->bq_corr);
    free(s->bq_xf);
    free(s->bq_c16);
    free(s->bh_corr);
    free(s->bh_xf);
    free(s->bh_c16);
    free(s->kq_scr);
    free(s->att_part);
    free(s->rope_b);
    free(s->rope);
    rq_sync_free(&s->sync);
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
    if (t->kp) { // GGUF Q4_K / Q6_K: ggml's dequantization of the row's blocks (dim % 256 == 0)
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

    ss = 1.0f / sqrtf((ss / size) + 1e-6f);

    // normalize and scale
    for (int j = 0; j < size; j++)
        o[j] = weight[j] * (ss * x[j]);
}

// Q8_0 dot product of one weight row with one quantized activation row (n values, n/GS groups):
// exact int32 sum per group of GS, then accumulated in float, group by group in order.
// Shared by matmul (GEMV) and matmul_batch (GEMM) so both give bit-identical results.
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

// Decode GEMV, in pieces: forward() runs all of a token in one parallel region, with the rows
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

// One batched-matmul phase of forward_batch: up to 3 GEMMs on the same nb quantized activation rows
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
    int nbi, nchunk;     // tiled: activation rows per chunk, chunks
    int chunk, unit;     // items per claim; ranges of whole units (tiled: a tile's chunks)
    QuantizedTensor *x; int ldx, nb, k;
    const int32_t *corr; const float *xf; const int32_t *c16;
} Gemm;

static void gemm_plan(Gemm *g, int nj, const MatJob *job, QuantizedTensor *x, int ldx, int nb, int k, const int32_t *corr,
                      const float *xf, const int32_t *c16, int nt) {
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
        n += g->tiled[j] ? (job[j].d / 16 + 1) / 2 * g->nchunk : job[j].w->kp ? job[j].d / KQ_ROWS : (job[j].d + MB_ROWS - 1) / MB_ROWS;
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
                           int b0, int b1, const void *kt, const void *vt, size_t bs, int hd, float scale, float *S) {
    // nv query vectors q[r] sharing one kv head (kt / vt: its tile 0; kvt: cache type), vector r over
    // positions 0..np[r]-1. part NULL: the whole attention into out[r] (all np <= ATT_P). Else the
    // partial results of the blocks b0..b1-1 of ATT_P positions that vector r reaches, block b at
    // part[r] + b * ATT_PSTRIDE(hd): o (unnormalized), m (max score), l (sum of exponentials).
    // S: scratch of nv * (b1 - b0) * ATT_P floats
#ifdef RUNQ_ATT512
    if (att512_ok(hd)) { att512_kv(kvt, nv, q, out, part, np, b0, b1, kt, vt, bs, hd, scale, S); return; }
#endif
#ifdef RUNQ_ATTNEON
    if (attn_ok(hd)) { attn_kv(kvt, nv, q, out, part, np, b0, b1, kt, vt, bs, hd, scale, S); return; }
#endif
    // generic C: per block of ATT_P positions (a 16-bit cache converted to fp32 once, for all the
    // vectors), attention_vec per vector
    size_t esz = kvt == KV_F32 ? 4 : 2, ts = (size_t)ATT_T * hd;
    float *conv = kvt == KV_F32 ? NULL : malloc((size_t)2 * ATT_P * hd * sizeof(float));
    if (kvt != KV_F32 && !conv) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
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
    free(conv);
}

static void attention_kv(int kvt, int nv, const float *const *q, float *const *out, const int *np, const void *kt,
                         const void *vt, size_t bs, int hd, float scale, float *S, float *part) {
    // nv query vectors q[r] sharing one kv head, vector r over positions 0..np[r]-1, into out[r]:
    // per block of ATT_P positions the partial results, then their combination (one block: the
    // attention directly). S: scratch of nv * (blocks of max np) * ATT_P floats; part: nv * (blocks of
    // max np) * ATT_PSTRIDE(hd) floats. Each vector's result does not depend on the others
    int nmax = 0;
    for (int r = 0; r < nv; r++)
        nmax = np[r] > nmax ? np[r] : nmax;
    if (nmax <= ATT_P) {
        attention_part(kvt, nv, q, out, NULL, np, 0, 1, kt, vt, bs, hd, scale, S);
        return;
    }
    int nbm = (nmax + ATT_P - 1) / ATT_P;
    size_t st = ATT_PSTRIDE(hd);
    float *pv[ATT_NV_MAX];
    for (int r = 0; r < nv; r++)
        pv[r] = part + (size_t)r * nbm * st;
    attention_part(kvt, nv, q, NULL, pv, np, 0, nbm, kt, vt, bs, hd, scale, S);
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

static void kv_populate_one(RunState *s, Config *p, int pos0, int n, int it) {
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

typedef struct { Transformer *tr; int token, pos, compute_logits; } DecodeJob;

// phases of a decode job: per layer 4 (qkv + attention, wo, w1 + w3, w2), the classifier, the KV
// tile pre-fault every ATT_T positions; dependency counters: per (layer, kv head) its qkv items
// done, and its attention blocks done
static int decode_phases(Config *p) { return 4 * p->n_layers + 2; }
static int decode_counters(Config *p) { return 2 * p->n_layers * p->n_kv_heads; }

static void decode_norm(RunState *s, Config *p, TransformerWeights *w, int l, int resid) {
    // (x += xb,) rmsnorm with layer l's attention weights (l < n_layers) or the final ones (l ==
    // n_layers: into x itself, as before), quantize into xq, the GEMV activation prep
    int dim = p->dim;
    if (resid)
        for (int i = 0; i < dim; i++)
            s->x[i] += s->xb[i];
    if (l < p->n_layers) {
        rmsnorm(s->xb, s->x, w->rms_att_weight + l * dim, dim);
        quantize(&s->xq, s->xb, dim);
    } else {
        rmsnorm(s->x, s->x, w->rms_final_weight, dim);
        quantize(&s->xq, s->x, dim);
    }
    matvec_prep(s->xq_corr, s->xq_xf, s->xq_c16, &s->xq, 0, dim);
}

static void decode_worker(void *arg, int tid, int nt) {
    (void)nt;
    DecodeJob *job = arg;
    Transformer *transformer = job->tr;
    Config *p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    RqSync *S = &s->sync;
    int pos = job->pos, k_ = 0;
    int dim = p->dim, hd = p->head_dim, hidden = p->hidden_dim, nh = p->n_heads, nkv = p->n_kv_heads;
    int kv_dim = nkv * hd;
    int kv_mul = nh / nkv; // integer multiplier of the kv sharing in multiquery
    int all_heads_dim = nh * hd, ff = s->ff_rows;
    int ng = kv_mul + 2; // qkv items per kv head: its kv_mul query heads, its key head, its value head
    int nb = (pos + 1 + ATT_P - 1) / ATT_P; // attention blocks (kernels_avx512.h): items per kv head
    size_t bs = (size_t)nkv * ATT_P * hd, pst = ATT_PSTRIDE(hd); // (cache: elements from a head's block to its next)

    if (pos % ATT_T == 0) { // first position of a KV cache tile: pre-fault the tile (all layers)
        PH_BEGIN_DYN(2 * p->n_layers, 1)
        for (int it = lo; it < hi; it++)
            kv_populate_one(s, p, pos, 1, it);
        PH_END_NOEPI
    }

    // forward all the layers (layer 0's attention rmsnorm + quantize: done by forward())
    for (int l = 0; l < p->n_layers; l++) {
        uint64_t loff = l * (uint64_t)s->kv_len * kv_dim; // kv cache layer offset for convenience
        char *kl = (char *)s->key_cache + loff * s->kv_esz, *vl = (char *)s->value_cache + loff * s->kv_esz;
        int *dep = S->cnt + l * nkv; (void)dep;

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
                    matvec_rows(s->q, &s->xq, s->xq_corr, s->xq_xf, s->xq_c16, w->wq + l, dim, h * hd, (h + 1) * hd);
                    qk_norm_rope(s->q + h * hd, 1, w->q_norm_weights + l * hd, hd, s->rope);
                } else if (r == kv_mul) {
                    matvec_rows(s->k, &s->xq, s->xq_corr, s->xq_xf, s->xq_c16, w->wk + l, dim, g * hd, (g + 1) * hd);
                    qk_norm_rope(s->k + g * hd, 1, w->k_norm_weights + l * hd, hd, s->rope);
                    kv_store(s->kvt, kl, NULL, pos, g, nkv, hd, s->k + g * hd, NULL);
                } else {
                    matvec_rows(s->v, &s->xq, s->xq_corr, s->xq_xf, s->xq_c16, w->wv + l, dim, g * hd, (g + 1) * hd);
                    kv_store(s->kvt, NULL, vl, pos, g, nkv, hd, NULL, s->v + g * hd);
                }
                rq_count(S, l * nkv + g, 1);
                continue;
            }
            int g = (it - nkv * ng) / nb, b = (it - nkv * ng) % nb;
            rq_wait_count(S, l * nkv + g, ng);
            const float *qv[ATT_NV_MAX];
            float *ov[ATT_NV_MAX], *pv[ATT_NV_MAX];
            int npv[ATT_NV_MAX];
            float *part = s->att_part + (size_t)g * kv_mul * s->att_nb * pst; // vector r, block b: + (r * att_nb + b) * pst
            for (int r = 0; r < kv_mul; r++) {
                qv[r] = s->q + (g * kv_mul + r) * hd;
                ov[r] = s->xb + (g * kv_mul + r) * hd;
                pv[r] = part + (size_t)r * s->att_nb * pst;
                npv[r] = pos + 1;
            }
            size_t o = kv_tile(0, g, nkv, hd);
            attention_part(s->kvt, kv_mul, qv, ov, nb == 1 ? NULL : pv, npv, b, b + 1, kl + o * s->kv_esz, vl + o * s->kv_esz, bs, hd, s->att_scale,
                           s->att + ((size_t)g * s->att_nb + b) * kv_mul * ATT_P);
            if (nb > 1) { // the last block of this kv head to finish combines them
                if (rq_count(S, (p->n_layers + l) * nkv + g, 1) < nb)
                    continue;
                for (int r = 0; r < kv_mul; r++)
                    att_combine(ov[r], nb, pv[r], pst, hd);
            }
            int i0 = g * kv_mul * hd;
            QuantizedTensor ag = qsub(&s->aq, i0);
            quantize(&ag, s->xb + i0, kv_mul * hd);
            matvec_prep(s->aq_corr, s->aq_xf, s->aq_c16, &s->aq, i0, kv_mul * hd);
        }
        PH_END_NOEPI

        // final matmul to get the output of the attention (16-row blocks); epilogue: residual
        // connection back into s->x, ffn rmsnorm, quantize
        PH_BEGIN(dim / 16, 1)
            matvec_rows(s->xb, &s->aq, s->aq_corr, s->aq_xf, s->aq_c16, w->wo + l, all_heads_dim, lo * 16, hi * 16);
        PH_EPILOGUE
            for (int i = 0; i < dim; i++)
                s->x[i] += s->xb[i];
            rmsnorm(s->xb, s->x, w->rms_ffn_weight + l * dim, dim);
            quantize(&s->xq, s->xb, dim);
            matvec_prep(s->xq_corr, s->xq_xf, s->xq_c16, &s->xq, 0, dim);
        PH_END

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x)): per item,
        // ff rows of self.w1(x) and self.w3(x), the SwiGLU non-linearity and their quantization
        PH_BEGIN(hidden / ff, 1)
        for (int i0 = lo * ff; i0 < hi * ff; i0 += ff) {
            matvec_rows(s->hb, &s->xq, s->xq_corr, s->xq_xf, s->xq_c16, w->w1 + l, dim, i0, i0 + ff);
            matvec_rows(s->hb2, &s->xq, s->xq_corr, s->xq_xf, s->xq_c16, w->w3 + l, dim, i0, i0 + ff);
            swiglu(s->hb + i0, s->hb2 + i0, ff);
            QuantizedTensor hg = qsub(&s->hq, i0);
            quantize(&hg, s->hb + i0, ff);
            matvec_prep(s->hq_corr, s->hq_xf, s->hq_c16, &s->hq, i0, ff);
        }
        PH_END_NOEPI

        // final matmul to get the output of the ffn; epilogue: its residual, then the next layer's
        // attention rmsnorm + quantize (or the final rmsnorm for the classifier; none without logits)
        PH_BEGIN(dim / 16, 1)
            matvec_rows(s->xb, &s->hq, s->hq_corr, s->hq_xf, s->hq_c16, w->w2 + l, hidden, lo * 16, hi * 16);
        PH_EPILOGUE
            if (l + 1 < p->n_layers || job->compute_logits)
                decode_norm(s, p, w, l + 1, 1);
        PH_END
    }

    if (job->compute_logits) { // classifier into logits
        int nblk = (p->vocab_size + 15) / 16;
        PH_BEGIN(nblk, 8)
            matvec_rows(s->logits, &s->xq, s->xq_corr, s->xq_xf, s->xq_c16, w->wcls, dim, lo * 16,
                        hi * 16 < p->vocab_size ? hi * 16 : p->vocab_size);
        PH_END_NOEPI
    }
}

float *forward(Transformer *transformer, int token, int pos, int compute_logits) {
    // compute_logits = 0 skips the final rmsnorm + classifier (prompt tokens whose
    // logits are discarded) and returns NULL; the KV cache is filled either way.
    // One parallel job for the whole token (decode_worker), in phases (threads.h), 4 per layer:
    //   [q, k, v, one item per head, with its QK-norm + RoPE and the K/V cache store; attention per
    //   kv head (once the 4 items of its heads are done: dependency counters) + quantize] ->
    //   [wo; epilogue: residual + rmsnorm + quantize] -> [w1 + w3 + SwiGLU + quantize per ff_rows
    //   hidden rows] -> [w2; epilogue: residual + the next layer's rmsnorm + quantize], then the
    //   classifier: 4 * 28 + 1 = 113 synchronization points per token (was 7 * 28 + 2 = 198 OpenMP
    //   barriers). Every output is computed whole by one thread, by the same functions as before
    //   (same results)
    Config *p = &transformer->config;
    RunState* s = &transformer->state;

    // the token embedding into s->x
    embed_row(s->x, transformer->weights.q_tokens, token, p->dim);
    rope_row(s->rope, p->head_dim, pos);
    decode_norm(s, p, &transformer->weights, 0, 0); // layer 0's attention rmsnorm + quantize

    int nt = rq_nthreads();
    rq_begin(&s->sync, decode_phases(p), decode_counters(p), nt, rq_spin_dec);
    DecodeJob job = { transformer, token, pos, compute_logits };
    rq_parallel(decode_worker, &job);
    return compute_logits ? s->logits : NULL;
}

static inline QuantizedTensor qrow(QuantizedTensor *t, int b, int ld) {
    // row b of a batch of quantized rows with row stride ld (a multiple of GS)
    return (QuantizedTensor) { .q = t->q + (size_t)b * ld, .s = t->s + (size_t)b * ld / GS, .p = NULL };
}


static void act_prep_row(RunState *s, QuantizedTensor *xq, int k, int32_t *corr, float *xf, int32_t *c16, int b) {
    // the GEMM activation prep of quantized row b (k values; the Gemm strides): matvec_prep
    matvec_prep(corr + (size_t)b * (k / GS), xf ? xf + (size_t)b * (k / GS) : NULL, c16 ? c16 + (size_t)b * (k / 16) : NULL,
                xq, 0, k);
    (void)s;
}

typedef struct { Transformer *tr; int *tokens; int n, pos0, logits_mode; float *logits_out; } PrefillJob;

static int prefill_phases(Config *p) { return 9 * p->n_layers + 4; }

#ifdef RUNQ_PROFILE
#define PPROF(i) do { if (tid == 0) PROF(i); } while (0)
#else
#define PPROF(i) ((void)0)
#endif

static void prefill_worker(void *arg, int tid, int nt) {
    // forward_batch's job: per layer 9 phases (the first layer 8): [residual + rmsnorm + quantize per
    // row] -> [q / k / v GEMM] -> [QK-norm + RoPE + K/V cache store per 16-position tile] ->
    // [attention per (block of att_rows rows, kv head), longest first; the last head of a row block
    // to finish quantizes its rows for wo] -> [wo GEMM] -> [residual + rmsnorm + quantize per row]
    // -> [w1 / w3 GEMM] -> [SwiGLU + quantize per row] -> [w2 GEMM]; the first phase also embeds the
    // tokens and pre-faults the chunk's KV cache tiles. Each phase's items as before (same results)
    PrefillJob *job = arg;
    Transformer *transformer = job->tr;
    Config *p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    RqSync *S = &s->sync;
    int n = job->n, pos0 = job->pos0, k_ = 0;
    int dim = p->dim, hd = p->head_dim, hidden = p->hidden_dim, xs = s->xs, nkv = p->n_kv_heads;
    int kv_dim = nkv * hd;
    int kv_mul = p->n_heads / nkv;
    int all_heads_dim = p->n_heads * hd;
    int rch = n / (4 * nt) > 1 ? n / (4 * nt) : 1; // rows per claim in the per-row phases
    int ar = s->att_rows, nrb = (n + ar - 1) / ar;
    int tb0 = pos0 / ATT_T, ntb = (pos0 + n - 1) / ATT_T - tb0 + 1;
    size_t bs = (size_t)nkv * ATT_P * hd;
    uint8_t *scr = s->kq_scr ? s->kq_scr + (size_t)tid * s->kq_scr_bytes : NULL;
    float *att = s->att_b + (size_t)tid * ar * kv_mul * s->att_nb * ATT_P;
    float *part = s->att_pb + (size_t)tid * ar * kv_mul * s->att_nb * ATT_PSTRIDE(hd);
    Gemm G;

    // embeddings, RoPE cos / sin per row, layer 0's attention rmsnorm + quantize; KV tile pre-faults
    PH_BEGIN(n + 2 * p->n_layers, 1)
    for (int b = lo; b < hi; b++) {
        if (b >= n) {
            kv_populate_one(s, p, pos0, n, b - n);
            continue;
        }
        float *x = s->x_b + (size_t)b * dim, *xb = s->xb_b + (size_t)b * xs;
        QuantizedTensor xq = qrow(&s->xq_b, b, xs);
        embed_row(x, w->q_tokens, job->tokens[b], dim);
        rope_row(s->rope_b + (size_t)b * hd, hd, pos0 + b);
        rmsnorm(xb, x, w->rms_att_weight, dim);
        quantize(&xq, xb, dim);
        act_prep_row(s, &xq, dim, s->bq_corr, s->bq_xf, s->bq_c16, b);
    }
    PH_END_NOEPI
    PPROF(PF_NORM);

    for (int l = 0; l < p->n_layers; l++) {
        uint64_t loff = l * (uint64_t)s->kv_len * kv_dim;
        char *kl = (char *)s->key_cache + loff * s->kv_esz, *vl = (char *)s->value_cache + loff * s->kv_esz;

        if (l > 0) { // residual of the previous layer's ffn, attention rmsnorm + quantize, per row
            PH_BEGIN(n, rch)
            for (int b = lo; b < hi; b++) {
                float *x = s->x_b + (size_t)b * dim, *xb = s->xb_b + (size_t)b * xs;
                QuantizedTensor xq = qrow(&s->xq_b, b, xs);
                for (int i = 0; i < dim; i++)
                    x[i] += xb[i];
                rmsnorm(xb, x, w->rms_att_weight + l * dim, dim);
                quantize(&xq, xb, dim);
                act_prep_row(s, &xq, dim, s->bq_corr, s->bq_xf, s->bq_c16, b);
            }
            PH_END_NOEPI
            PPROF(PF_NORM);
        }

        // qkv GEMMs (one phase)
        MatJob qkv[3] = { { s->q_b, all_heads_dim, w->wq + l, all_heads_dim }, { s->k_b, kv_dim, w->wk + l, kv_dim },
                          { s->v_b, kv_dim, w->wv + l, kv_dim } };
        gemm_plan(&G, 3, qkv, &s->xq_b, xs, n, dim, s->bq_corr, s->bq_xf, s->bq_c16, nt);
        {
            const void *last = NULL;
            PH_BEGIN_U(G.it0[G.nj], G.chunk, G.unit)
            for (int it = lo; it < hi; it++)
                gemm_item(&G, it, scr, &last);
            PH_END_NOEPI
        }
        PPROF(PF_QKV);

        // QK-norm + RoPE, each row at its own position, and the K / V rows into the cache: one
        // 16-position cache tile per item (its rows, all heads)
        PH_BEGIN(ntb, 1)
        for (int it = lo; it < hi; it++) {
            int tb = tb0 + it;
            int t1 = (tb + 1) * ATT_T < pos0 + n ? (tb + 1) * ATT_T : pos0 + n;
            for (int t = tb * ATT_T > pos0 ? tb * ATT_T : pos0; t < t1; t++) {
                int b = t - pos0;
                const float *cs = s->rope_b + (size_t)b * hd;
                qk_norm_rope(s->q_b + (size_t)b * all_heads_dim, p->n_heads, w->q_norm_weights + l * hd, hd, cs);
                qk_norm_rope(s->k_b + (size_t)b * kv_dim, nkv, w->k_norm_weights + l * hd, hd, cs);
                for (int h = 0; h < nkv; h++)
                    kv_store(s->kvt, kl, vl, t, h, nkv, hd, s->k_b + (size_t)b * kv_dim + h * hd, s->v_b + (size_t)b * kv_dim + h * hd);
            }
        }
        PH_END_NOEPI
        PPROF(PF_ROPE);

        // causal attention: row b (position pos0 + b) attends to positions 0..pos0+b, whose K/V
        // are all in the cache now. Work item = (block of att_rows rows, kv head): its rows x
        // kv_mul query heads in one attention_kv call (K/V read once for all of them), last
        // (longest) row blocks first, for the load balance. The last of a row block's nkv items
        // to finish quantizes its rows for wo (into xq_b: the qkv GEMM is done with it)
        PH_BEGIN_DYN(nrb * nkv, 1)
        for (int it = lo; it < hi; it++) {
            int rb = nrb - 1 - it / nkv, g = it % nkv;
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
            size_t o = kv_tile(0, g, nkv, hd);
            attention_kv(s->kvt, nv, qv, ov, npv, kl + o * s->kv_esz, vl + o * s->kv_esz, bs, hd, s->att_scale, att, part);
            if (rq_count(S, l * s->pf_nrb + rb, 1) < nkv)
                continue;
            for (int b = b0; b < b1; b++) {
                QuantizedTensor xq = qrow(&s->xq_b, b, xs);
                quantize(&xq, s->xb_b + (size_t)b * xs, all_heads_dim);
                act_prep_row(s, &xq, all_heads_dim, s->bq_corr, s->bq_xf, s->bq_c16, b);
            }
        }
        PH_END_NOEPI
        PPROF(PF_ATT);

        MatJob wo = { s->xb_b, xs, w->wo + l, dim };
        gemm_plan(&G, 1, &wo, &s->xq_b, xs, n, all_heads_dim, s->bq_corr, s->bq_xf, s->bq_c16, nt);
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
            float *x = s->x_b + (size_t)b * dim, *xb = s->xb_b + (size_t)b * xs;
            QuantizedTensor xq = qrow(&s->xq_b, b, xs);
            for (int i = 0; i < dim; i++)
                x[i] += xb[i];
            rmsnorm(xb, x, w->rms_ffn_weight + l * dim, dim);
            quantize(&xq, xb, dim);
            act_prep_row(s, &xq, dim, s->bq_corr, s->bq_xf, s->bq_c16, b);
        }
        PH_END_NOEPI
        PPROF(PF_NORM);

        MatJob w13[2] = { { s->hb_b, hidden, w->w1 + l, hidden }, { s->hb2_b, hidden, w->w3 + l, hidden } };
        gemm_plan(&G, 2, w13, &s->xq_b, xs, n, dim, s->bq_corr, s->bq_xf, s->bq_c16, nt);
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
            QuantizedTensor hq = qrow(&s->hq_b, b, hidden);
            swiglu(s->hb_b + (size_t)b * hidden, s->hb2_b + (size_t)b * hidden, hidden);
            quantize(&hq, s->hb_b + (size_t)b * hidden, hidden);
            act_prep_row(s, &hq, hidden, s->bh_corr, s->bh_xf, s->bh_c16, b);
        }
        PH_END_NOEPI
        PPROF(PF_SWIGLU);

        MatJob w2 = { s->xb_b, xs, w->w2 + l, dim };
        gemm_plan(&G, 1, &w2, &s->hq_b, hidden, n, hidden, s->bh_corr, s->bh_xf, s->bh_c16, nt);
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

    if (job->logits_mode == LOGITS_LAST) {
        // residual of the last row, final rmsnorm + classifier for it, exactly as in forward()
        PH_BEGIN(1, 1)
            float *x = s->x_b + (size_t)(n - 1) * dim, *xb = s->xb_b + (size_t)(n - 1) * xs;
            for (int i = 0; i < dim; i++)
                x[i] += xb[i];
            rmsnorm(s->x, x, w->rms_final_weight, dim);
            quantize(&s->xq, s->x, dim);
            matvec_prep(s->xq_corr, s->xq_xf, s->xq_c16, &s->xq, 0, dim);
        PH_END_NOEPI
        int nblk = (p->vocab_size + 15) / 16, cc = nblk / (8 * nt) > 1 ? nblk / (8 * nt) : 1;
        PH_BEGIN(nblk, cc)
            matvec_rows(s->logits, &s->xq, s->xq_corr, s->xq_xf, s->xq_c16, w->wcls, dim, lo * 16,
                        hi * 16 < p->vocab_size ? hi * 16 : p->vocab_size);
        PH_END_NOEPI
        PPROF(PF_CLS);
    } else if (job->logits_mode == LOGITS_ALL) {
        PH_BEGIN(n, rch)
        for (int b = lo; b < hi; b++) {
            float *x = s->x_b + (size_t)b * dim, *xb = s->xb_b + (size_t)b * xs;
            QuantizedTensor xq = qrow(&s->xq_b, b, xs);
            for (int i = 0; i < dim; i++)
                x[i] += xb[i];
            rmsnorm(xb, x, w->rms_final_weight, dim);
            quantize(&xq, xb, dim);
            act_prep_row(s, &xq, dim, s->bq_corr, s->bq_xf, s->bq_c16, b);
        }
        PH_END_NOEPI
        MatJob cls = { job->logits_out, p->vocab_size, w->wcls, p->vocab_size };
        gemm_plan(&G, 1, &cls, &s->xq_b, xs, n, dim, s->bq_corr, s->bq_xf, s->bq_c16, nt);
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

float *forward_batch(Transformer *transformer, int *tokens, int n, int pos0, int logits_mode, float *logits_out) {
    // Batched prefill: tokens[0..n-1] at positions pos0..pos0+n-1 (1 <= n <= state.nb_max),
    // the same computation as n calls of forward() but with every weight matrix applied to all
    // n rows at once (GEMM work items), and bit-identical results. Fills the KV cache.
    //   LOGITS_NONE: returns NULL
    //   LOGITS_LAST: logits of the last row in state.logits (returned)
    //   LOGITS_ALL:  logits of all rows into logits_out (n, vocab_size) (returned)
    // Note: in LOGITS_LAST mode state.x / state.xq are clobbered (as by forward()).
    // One parallel job per chunk (prefill_worker), in phases (threads.h)
    Config *p = &transformer->config;
    RunState* s = &transformer->state;
    if (n < 1 || n > s->nb_max || pos0 < 0 || pos0 + n > p->seq_len) {
        fprintf(stderr, "forward_batch: bad chunk n=%d pos0=%d (nb_max %d, seq_len %d)\n", n, pos0, s->nb_max, p->seq_len);
        exit(EXIT_FAILURE);
    }
    PROF_START();
    rq_begin(&s->sync, prefill_phases(p), p->n_layers * s->pf_nrb, rq_nthreads(), rq_spin_pf);
    PrefillJob job = { transformer, tokens, n, pos0, logits_mode, logits_out };
    rq_parallel(prefill_worker, &job);
    return logits_mode == LOGITS_LAST ? s->logits : logits_mode == LOGITS_ALL ? logits_out : NULL;
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

