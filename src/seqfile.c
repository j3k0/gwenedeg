/* Sequences on disk (runq.h: rq_seq_save, rq_seq_file_info, rq_seq_file_keep, rq_seq_load): a KV cache tier for
 * runq serve --kv-disk (engine.c). Its own translation unit: _GNU_SOURCE (O_DIRECT) stays out of forward.c, whose
 * code generation the exact arithmetic depends on. */
#if defined __linux__ && !defined _GNU_SOURCE
#define _GNU_SOURCE // (O_DIRECT, posix_fallocate, posix_fadvise)
#endif
#include "internal.h"

// One file per sequence, every section at a multiple of
// SF_ALIGN bytes, so that the KV blocks go straight from / to the cache with O_DIRECT (the cache is page-aligned
// anonymous memory, alloc_big; a block of ATT_P positions is ATT_P / ATT_T x n_kv_heads tiles (kv_tile_bytes: ATT_P
// x kv_dim x esz bytes for the 16 / 32-bit types), a multiple of 4 KiB for the shapes here; else buffered I/O):
//   the header (SF_ALIGN bytes: SeqFileHdr), the tokens (len ints)
//   the KV cache: per attention layer, K then V, its first nblk blocks of ATT_P positions as in memory (kv_tile's
//   layout; the last block whole: its positions past len hold whatever the cache held, never read before written)
//   Qwen3.5: the recurrent state after position len - 1, then the checkpoints held (their lengths in the header)
// The state sections are ssm_stride = ssm_bytes rounded up to SF_ALIGN, transferred from / into the state buffers
// themselves (alloc_big: whole pages, so the rounded length stays inside their mapping). The I/O runs on the thread
// pool, one item per (attention layer, K / V) and per state section (a deep queue for an SSD: ~1.4 GB/s written,
// up to ~7 GB/s read on the dev box). Not on Windows (RQ_ERR_STATE).
#define SF_ALIGN 4096
#define SF_VERSION 1
static const char sf_magic[8] = { 'R', 'Q', 'S', 'E', 'Q', '\r', '\n', 0x1a };

typedef struct {
    char magic[8];
    uint32_t version, endian;      // SF_VERSION, 0x01020304 as written
    uint64_t model_fp;             // RqModel.fp
    int32_t n_attn, kv_dim, n_kv_heads, head_dim, att_p, att_t, att, kv_esz, n_lin, len, nblk, nck;
    int32_t ck[RQ_SEQ_CK_MAX];     // the checkpoints' lengths, ascending (nck of them)
    uint64_t ssm_bytes, ssm_stride, blk_bytes; // a state section's bytes / on disk, a K block's bytes (V's: sf_blk_bytes)
    uint64_t off_kv, off_ssm, file_bytes, rng; // (the tokens at SF_ALIGN; the checkpoints after the state)
    uint64_t hash;                 // FNV-1a of this header (hash 0) and the tokens
} SeqFileHdr;

static inline uint64_t sf_round(uint64_t n) { return (n + SF_ALIGN - 1) / SF_ALIGN * SF_ALIGN; }
// the header's kv_esz: V's bytes per element, rounded down (4, 2; q8 1, q4 0); a K (kv 0) or V (kv 1) block's bytes (all
// its kv heads). On disk per attention layer its K blocks, then its V blocks (equal sizes but for q4: the 16 / 32-bit
// types' files as before)
static inline int32_t sf_esz(int att, int hd) { return (int32_t)(kv_tile_bytes(att, hd, 1) / ((size_t)ATT_T * hd)); }
static inline uint64_t sf_blk_bytes(int att, int nkv, int hd, int kv) { return (uint64_t)(ATT_P / ATT_T) * nkv * kv_tile_bytes(att, hd, kv); }

#if !defined _WIN32
#include <errno.h>
#include <sys/stat.h>

static int sf_io(int fd, int wr, void *buf, size_t n, uint64_t off) {
    // the whole of n bytes at off; 0, or -1 (a read past the end included). O_DIRECT refused for this transfer
    // (EINVAL: the file system's alignment, or none): the descriptor goes on buffered
    char *p = buf;
    while (n) {
        size_t k = n < ((size_t)1 << 28) ? n : (size_t)1 << 28;
        ssize_t r = wr ? pwrite(fd, p, k, (off_t)off) : pread(fd, p, k, (off_t)off);
        if (r < 0 && errno == EINTR) continue;
#ifdef O_DIRECT
        if (r < 0 && errno == EINVAL) {
            int fl = fcntl(fd, F_GETFL);
            if (fl >= 0 && (fl & O_DIRECT) && fcntl(fd, F_SETFL, fl & ~O_DIRECT) == 0) continue;
        }
#endif
        if (r <= 0) return -1;
        p += r;
        off += (uint64_t)r;
        n -= (size_t)r;
    }
    return 0;
}

static int sf_open(const char *path, int wr) {
    // (O_DIRECT where the file system takes it; macOS: F_NOCACHE)
    int fl = wr ? O_WRONLY | O_CREAT | O_TRUNC : O_RDONLY, fd = -1;
#ifdef O_CLOEXEC
    fl |= O_CLOEXEC;
#endif
#ifdef O_DIRECT
    fd = open(path, fl | O_DIRECT, 0644);
#endif
    if (fd < 0) fd = open(path, fl, 0644);
#ifdef F_NOCACHE
    if (fd >= 0) fcntl(fd, F_NOCACHE, 1);
#endif
    return fd;
}

typedef struct {
    int fd, wr, nblk, nst, next, err; // nblk: KV blocks to transfer per item; nst: state sections
    RqSeq *s;
    const SeqFileHdr *h;
    float *st[1 + RQ_SEQ_CK_MAX];
    uint64_t st_off[1 + RQ_SEQ_CK_MAX];
} SfJob;

static void sf_worker(void *arg, int tid, int nt) {
    // items: (attention layer, K / V) = it < 2 n_attn (a load pre-faults the destination's blocks first, as
    // rq_seq_copy does), then the state sections
    (void)tid; (void)nt;
    SfJob *j = arg;
    RqSeq *s = j->s;
    const RqModel *m = s->m;
    int nkv = 2 * m->n_attn;
    for (int it; (it = __atomic_fetch_add(&j->next, 1, __ATOMIC_RELAXED)) < nkv + j->nst;) {
        if (__atomic_load_n(&j->err, __ATOMIC_RELAXED)) continue;
        void *buf;
        size_t n;
        uint64_t off;
        if (it < nkv) {
            int l = it / 2, kv = it % 2;
            if (!j->wr) seq_kv_populate(s, j->nblk * ATT_P, it);
            buf = (char *)(kv ? s->value_cache : s->key_cache) + seq_kvb(s, (size_t)l * s->kv_len * m->kv_dim, kv);
            uint64_t bk = j->h->blk_bytes, bv = sf_blk_bytes(j->h->att, j->h->n_kv_heads, j->h->head_dim, 1);
            n = (size_t)j->nblk * (kv ? bv : bk);
            off = j->h->off_kv + (uint64_t)j->h->nblk * ((uint64_t)l * (bk + bv) + (kv ? bk : 0));
        } else {
            buf = j->st[it - nkv];
            n = j->h->ssm_stride;
            off = j->st_off[it - nkv];
        }
        if (sf_io(j->fd, j->wr, buf, n, off)) __atomic_store_n(&j->err, 1, __ATOMIC_RELAXED);
    }
}

static uint64_t sf_hash(const SeqFileHdr *h, const int *tokens) {
    SeqFileHdr c = *h;
    c.hash = 0;
    return fnv1a64(fnv1a64(1469598103934665603ull, &c, sizeof(c)), tokens, (size_t)h->len * sizeof(int));
}

static void sf_header(const RqSeq *s, SeqFileHdr *h) {
    // the header of s as it would be saved (the checkpoints held: taken, not being written)
    const RqModel *m = s->m;
    memset(h, 0, sizeof(*h));
    memcpy(h->magic, sf_magic, 8);
    h->version = SF_VERSION;
    h->endian = 0x01020304;
    h->model_fp = m->fp;
    h->n_attn = m->n_attn;
    h->kv_dim = m->kv_dim;
    h->n_kv_heads = m->config.n_kv_heads;
    h->head_dim = m->config.head_dim;
    h->att_p = ATT_P;
    h->att_t = ATT_T;
    h->att = s->att;
    h->kv_esz = sf_esz(s->att, m->config.head_dim);
    h->n_lin = m->hy.on ? m->hy.n_lin : 0;
    h->len = s->len;
    h->nblk = (s->len + ATT_P - 1) / ATT_P;
    h->nck = m->hy.on ? rq_seq_checkpoints(s, h->ck, RQ_SEQ_CK_MAX) : 0;
    h->ssm_bytes = s->ssm_bytes;
    h->ssm_stride = sf_round(s->ssm_bytes);
    h->blk_bytes = sf_blk_bytes(s->att, m->config.n_kv_heads, m->config.head_dim, 0);
    h->off_kv = SF_ALIGN + sf_round((uint64_t)s->len * sizeof(int));
    h->off_ssm = h->off_kv + (uint64_t)m->n_attn * h->nblk * (h->blk_bytes + sf_blk_bytes(s->att, m->config.n_kv_heads, m->config.head_dim, 1));
    h->file_bytes = h->off_ssm + (m->hy.on ? (uint64_t)(1 + h->nck) * h->ssm_stride : 0);
    h->rng = s->rng;
}

static int sf_check(const RqModel *m, const SeqFileHdr *h, uint64_t size) {
    // the header against the model (and itself): RQ_OK or RQ_ERR_MISMATCH. (Its length is bounded by the model's
    // context, not the loaded ctx_len cap: a prefix of a longer save still loads into a shorter sequence)
    SeqFileHdr w;
    if (memcmp(h->magic, sf_magic, 8) || h->version != SF_VERSION || h->endian != 0x01020304 || h->model_fp != m->fp ||
        h->n_attn != m->n_attn || h->kv_dim != m->kv_dim || h->n_kv_heads != m->config.n_kv_heads ||
        h->head_dim != m->config.head_dim || h->att_p != ATT_P || h->att_t != ATT_T || !kv_type_fits(h->att, h->head_dim) ||
        h->kv_esz != sf_esz(h->att, h->head_dim) || h->n_lin != (m->hy.on ? m->hy.n_lin : 0) ||
        h->len < 1 || h->len > (m->ctx_max > m->config.seq_len ? m->ctx_max : m->config.seq_len) || h->nblk != (h->len + ATT_P - 1) / ATT_P ||
        h->nck < 0 || h->nck > RQ_SEQ_CK_MAX || (!m->hy.on && h->nck) ||
        h->ssm_bytes != (m->hy.on ? (uint64_t)m->hy.n_lin * m->ssm_layer * sizeof(float) : 0))
        return RQ_ERR_MISMATCH;
    for (int i = 0; i < h->nck; i++)
        if (h->ck[i] < 0 || h->ck[i] > h->len || (i && h->ck[i] <= h->ck[i - 1])) return RQ_ERR_MISMATCH;
    // the derived fields: as sf_header computes them for this shape
    memset(&w, 0, sizeof(w));
    w.ssm_stride = sf_round(h->ssm_bytes);
    w.blk_bytes = sf_blk_bytes(h->att, h->n_kv_heads, h->head_dim, 0);
    w.off_kv = SF_ALIGN + sf_round((uint64_t)h->len * sizeof(int));
    w.off_ssm = w.off_kv + (uint64_t)h->n_attn * h->nblk * (w.blk_bytes + sf_blk_bytes(h->att, h->n_kv_heads, h->head_dim, 1));
    w.file_bytes = w.off_ssm + (m->hy.on ? (uint64_t)(1 + h->nck) * w.ssm_stride : 0);
    if (h->ssm_stride != w.ssm_stride || h->blk_bytes != w.blk_bytes || h->off_kv != w.off_kv ||
        h->off_ssm != w.off_ssm || h->file_bytes != w.file_bytes || size != h->file_bytes)
        return RQ_ERR_MISMATCH;
    return RQ_OK;
}

static int sf_read_head(const RqModel *m, const char *path, int *fd, SeqFileHdr *h, char **buf) {
    // open path, read and check its header and tokens: *buf = an alloc_big of h->off_kv bytes (the header, then the
    // tokens at SF_ALIGN), *fd open; RQ_OK, else nothing left open or allocated
    struct stat st;
    size_t bn = SF_ALIGN;
    int e = RQ_ERR_IO;
    char *b = NULL;
    *buf = NULL;
    if ((*fd = sf_open(path, 0)) < 0) return RQ_ERR_IO;
    if (!(b = alloc_big(bn))) { e = RQ_ERR_NOMEM; goto fail; }
    if (fstat(*fd, &st) || (uint64_t)st.st_size < SF_ALIGN || sf_io(*fd, 0, b, SF_ALIGN, 0)) goto fail;
    memcpy(h, b, sizeof(*h));
    if ((e = sf_check(m, h, (uint64_t)st.st_size)) != RQ_OK) goto fail;
    free_big(b, bn);
    bn = h->off_kv;
    if (!(b = alloc_big(bn))) { e = RQ_ERR_NOMEM; goto fail; }
    e = RQ_ERR_IO;
    if (sf_io(*fd, 0, b, bn, 0)) goto fail;
    e = RQ_ERR_MISMATCH;
    if (sf_hash(h, (const int *)(b + SF_ALIGN)) != h->hash) goto fail;
    *buf = b;
    return RQ_OK;
fail:
    free_big(b, bn);
    close(*fd);
    *fd = -1;
    return e;
}

int rq_seq_save(const RqSeq *s, const char *path) {
    if (!s || !path || s->len < 1) return RQ_ERR_ARG;
    SeqFileHdr h;
    sf_header(s, &h);
    size_t pl = strlen(path);
    char *tmp = malloc(pl + 5);
    char *b = alloc_big(h.off_kv);
    int e = RQ_ERR_NOMEM, fd = -1;
    if (!tmp || !b) goto done;
    memcpy(tmp, path, pl);
    memcpy(tmp + pl, ".tmp", 5);
    e = RQ_ERR_IO;
    if ((fd = sf_open(tmp, 1)) < 0) goto done;
#if defined __linux__
    // the blocks up front: a full disk fails here, before anything is written
    if (posix_fallocate(fd, 0, (off_t)h.file_bytes)) goto fail;
#else
    if (ftruncate(fd, (off_t)h.file_bytes)) goto fail;
#endif
    memcpy(b + SF_ALIGN, s->tokens, (size_t)s->len * sizeof(int));
    h.hash = sf_hash(&h, s->tokens);
    memcpy(b, &h, sizeof(h));
    SfJob j = { fd, 1, h.nblk, 0, 0, 0, (RqSeq *)s, &h, { 0 }, { 0 } };
    if (s->m->hy.on) { // the state, then the checkpoints by their lengths
        j.st[0] = s->ssm;
        j.st_off[0] = h.off_ssm;
        for (int i = 0; i < h.nck; i++) {
            int k = 0;
            while (s->ck[k].pos != h.ck[i] || s->ck[k].wr) k++;
            j.st[1 + i] = s->ck[k].st;
            j.st_off[1 + i] = h.off_ssm + (uint64_t)(1 + i) * h.ssm_stride;
        }
        j.nst = 1 + h.nck;
    }
    if (sf_io(fd, 1, b, h.off_kv, 0)) goto fail;
    rq_parallel(sf_worker, &j);
    if (j.err || fdatasync(fd)) goto fail;
#if defined POSIX_FADV_DONTNEED
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); // (a buffered write: its pages out of the page cache)
#endif
    if (close(fd)) { fd = -1; goto fail; }
    fd = -1;
    if (rename(tmp, path)) goto fail;
    e = RQ_OK;
    goto done;
fail:
    if (fd >= 0) close(fd);
    fd = -1;
    unlink(tmp);
done:
    if (fd >= 0) close(fd);
    free(tmp);
    free_big(b, h.off_kv);
    return e;
}

long long rq_seq_save_bytes(const RqSeq *s) {
    SeqFileHdr h;
    if (!s || s->len < 1) return 0;
    sf_header(s, &h);
    return (long long)h.file_bytes;
}

int rq_seq_file_info(const RqModel *m, const char *path, RqSeqFile *info, int **tokens) {
    if (!m || !path || !info) return RQ_ERR_ARG;
    SeqFileHdr h;
    char *b;
    int fd, e = sf_read_head(m, path, &fd, &h, &b);
    if (e != RQ_OK) return e;
    close(fd);
    memset(info, 0, sizeof(*info));
    info->len = h.len;
    info->kv_type = h.att;
    info->linear = h.n_lin > 0;
    info->n_ck = h.nck;
    memcpy(info->ck, h.ck, (size_t)h.nck * sizeof(int));
    info->bytes = (long long)h.file_bytes;
    if (tokens) {
        if (!(*tokens = malloc((size_t)h.len * sizeof(int)))) e = RQ_ERR_NOMEM;
        else memcpy(*tokens, b + SF_ALIGN, (size_t)h.len * sizeof(int));
    }
    free_big(b, h.off_kv);
    return e;
}

int rq_seq_load(RqSeq *dst, const char *path, int n) {
    if (!dst || !path || n < 0 || n > dst->cap) return RQ_ERR_ARG;
    const RqModel *m = dst->m;
    SeqFileHdr h;
    char *b;
    int fd, e = sf_read_head(m, path, &fd, &h, &b);
    if (e != RQ_OK) return e;
    RqSeqFile f = { h.len, h.att, h.n_lin > 0, h.nck, { 0 }, (long long)h.file_bytes };
    memcpy(f.ck, h.ck, (size_t)h.nck * sizeof(int));
    int L = rq_seq_file_keep(&f, n);
    if (h.att != dst->att) e = RQ_ERR_MISMATCH;
    else if (n > h.len) e = RQ_ERR_ARG;
    if (e != RQ_OK) goto done;
    rq_seq_reset(dst); // (Qwen3.5: no checkpoints, none pending)
    SfJob j = { fd, 0, (L + ATT_P - 1) / ATT_P, 0, 0, 0, dst, &h, { 0 }, { 0 } };
    if (m->hy.on && L > 0) {
        // the state at L: the saved one (L the saved length) or the checkpoint there; the saved checkpoints at or
        // below L into dst's slots (the highest ones if fewer slots)
        int k = 0, top = 0;
        while (top < h.nck && h.ck[top] <= L) top++;
        j.st[0] = dst->ssm;
        j.st_off[0] = h.off_ssm + (L == h.len ? 0 : (uint64_t)top * h.ssm_stride); // (h.ck[top - 1] == L then)
        for (int i = top - dst->nck > 0 ? top - dst->nck : 0; i < top; i++, k++) {
            if (!dst->ck[k].st && !(dst->ck[k].st = alloc_big(dst->ssm_bytes))) { e = RQ_ERR_NOMEM; goto done; }
            j.st[1 + k] = dst->ck[k].st;
            j.st_off[1 + k] = h.off_ssm + (uint64_t)(1 + i) * h.ssm_stride;
        }
        j.nst = 1 + k;
        rq_parallel(sf_worker, &j);
        if (j.err) { e = RQ_ERR_IO; goto done; }
        for (int i = top - k, q = 0; q < k; i++, q++) {
            dst->ck[q].pos = h.ck[i];
            dst->ck[q].rows = dst->ck[q].wr = 0;
        }
    } else if (L > 0) {
        rq_parallel(sf_worker, &j);
        if (j.err) { e = RQ_ERR_IO; goto done; }
    }
    memcpy(dst->tokens, b + SF_ALIGN, (size_t)L * sizeof(int));
    dst->len = L;
    dst->rng = h.rng;
done:
    if (e == RQ_ERR_IO) rq_seq_reset(dst);
    close(fd);
    free_big(b, h.off_kv);
    return e;
}

#else // _WIN32: not supported

int rq_seq_save(const RqSeq *s, const char *path) { (void)s; (void)path; return RQ_ERR_STATE; }
long long rq_seq_save_bytes(const RqSeq *s) { (void)s; return 0; }
int rq_seq_file_info(const RqModel *m, const char *path, RqSeqFile *info, int **tokens) {
    (void)m; (void)path; (void)info; (void)tokens;
    return RQ_ERR_STATE;
}
int rq_seq_load(RqSeq *dst, const char *path, int n) { (void)dst; (void)path; (void)n; return RQ_ERR_STATE; }

#endif

int rq_seq_file_keep(const RqSeqFile *f, int n) {
    n = n < 0 ? 0 : n > f->len ? f->len : n;
    if (!f->linear || n == f->len) return n;
    int k = 0;
    for (int i = 0; i < f->n_ck; i++)
        if (f->ck[i] <= n) k = f->ck[i];
    return k;
}
