/* Long-context driver (the library API, src/runq.h; TODO G2): the long-context check (tests/check_long.py) runs runq
 * through it on texts of 30-100k tokens, which the CLI's hooks cannot take (-i is one argument: 128 KiB at most;
 * -m logits writes every position's logits: 95 GB at 96k for a 248k vocabulary). Built by `make long-ctx`
 * (build/long_ctx, and build/long_ctx-generic on the generic kernels).
 *
 *   build/long_ctx MODEL -T TEXT -o IDS                tokenize a text file (raw text: added tokens recognized, no
 *                                                      template) into int32 ids; stderr: the count
 *   build/long_ctx MODEL -I IDS [-n N] [-P p1,p2,... | -A] [-o OUT] [-V] [-g G] [-c CTX] [-b CHUNK] [-k KV]
 *       -I IDS     the token ids (int32), the first N of them (-n; default all)
 *       -P ...     positions (ascending, < N): the logits after the token at each (row p of -m logits) into OUT:
 *                  int32 npos, int32 vocab, int32 pos[npos], float32 logits[npos][vocab]. The sequence is
 *                  prefilled up to each position in turn (RQ_LOGITS_LAST at the position, in chunks of -b rows:
 *                  the same bits as any chunking)
 *       -V         at each -P position p >= 64 also the same row by two other paths, which must give the same bits:
 *                  a verify segment (rq_forward_batch, the 4 rows p-3..p, every row's logits; the decode-rows worker)
 *                  and a decode step (rq_decode of token p); the sequence goes back to p - 63 between them (Qwen3.5:
 *                  a checkpoint there), so the reference row is the last of a 64-row prefill chunk. stdout: one
 *                  line per position, "verify p SAME|DIFFER segment=<0/1> decode=<0/1>"
 *       -g G       after the N tokens (and the positions), generate up to G tokens greedily (stop at an end token):
 *                  stdout "gen <ids...>" then "text <the text, escapes as JSON would not need: raw>" lines
 *       -A         every row's logits of the N tokens into OUT, streamed per chunk of -b rows (rq_prefill RQ_LOGITS_ALL:
 *                  the bits of runq -m logits, in its file format: int32 N, vocab, ids[N], float32 logits[N][vocab]; the
 *                  memory of one chunk, not of N rows: scripts/kl_kv.py)
 *       -c CTX     sequence capacity (default N + G + 1); -k f16|bf16|f32|bf16-dot|f16-dot (default f16); -b chunk rows
 *                  (default 512)
 *   stderr: timings ([long_ctx] lines). Threads: OMP_NUM_THREADS (else the default). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/runq.h"

static void die(const char *what, int e, RqCtx *c) {
    fprintf(stderr, "long_ctx: %s: %s %s\n", what, rq_strerror(e), c ? rq_ctx_error(c) : "");
    exit(1);
}

static void *read_file(const char *path, long *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "long_ctx: cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    *n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = malloc(*n + 1);
    if (!b || fread(b, 1, *n, f) != (size_t)*n) { fprintf(stderr, "long_ctx: cannot read %s\n", path); exit(1); }
    b[*n] = 0;
    fclose(f);
    return b;
}

static int parse_list(const char *s, int **v) {
    int n = 0, cap = 16;
    *v = malloc(cap * sizeof(int));
    while (*s) {
        if (n == cap) *v = realloc(*v, (cap *= 2) * sizeof(int));
        (*v)[n++] = atoi(s);
        const char *c = strchr(s, ',');
        if (!c) break;
        s = c + 1;
    }
    return n;
}

static int same(const float *a, const float *b, int n) { return !memcmp(a, b, (size_t)n * sizeof(float)); }

static void prefill_to(RqCtx *c, RqSeq *s, const int *ids, int n, float **lg) {
    // the sequence's next tokens up to length n; lg: the last row's logits (NULL: none)
    int len = rq_seq_len(s), e;
    if (n <= len) return;
    if ((e = rq_prefill(c, s, ids + len, n - len, lg ? RQ_LOGITS_LAST : RQ_LOGITS_NONE, NULL, lg)) != RQ_OK)
        die("rq_prefill", e, c);
}

int main(int argc, char **argv) {
    const char *path = NULL, *text = NULL, *idsf = NULL, *outf = NULL, *plist = NULL;
    int N = -1, G = 0, cap = 0, chunk = 512, kvt = RQ_KV_DEFAULT, vfy = 0, all = 0;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') { path = argv[i]; continue; }
        if (!strcmp(argv[i], "-V")) { vfy = 1; continue; }
        if (!strcmp(argv[i], "-A")) { all = 1; continue; }
        if (i + 1 >= argc) goto usage;
        const char *v = argv[++i];
        switch (argv[i - 1][1]) {
            case 'T': text = v; break;
            case 'I': idsf = v; break;
            case 'o': outf = v; break;
            case 'P': plist = v; break;
            case 'n': N = atoi(v); break;
            case 'g': G = atoi(v); break;
            case 'c': cap = atoi(v); break;
            case 'b': chunk = atoi(v); break;
            case 'k': kvt = rq_kv_parse(v); break;
            default: goto usage;
        }
    }
    if (!path || (!text && !idsf) || kvt < 0 || chunk < 1 || chunk > RQ_BATCH_MAX || G < 0) goto usage;
    char err[512];
    if (text) { // tokenize only
        if (!outf) goto usage;
        RqModelOpts o = { 0 };
        o.tokenizer_only = 1;
        RqModel *m = rq_model_load(path, &o, err, sizeof(err));
        if (!m) { fprintf(stderr, "long_ctx: %s\n", err); return 1; }
        long nb;
        char *t = read_file(text, &nb);
        RqTokens ids = { 0 };
        double t0 = rq_now();
        int e = rq_tokenize(m, t, nb, &ids);
        if (e != RQ_OK) die("rq_tokenize", e, NULL);
        FILE *f = fopen(outf, "wb");
        if (!f || fwrite(ids.v, sizeof(int), ids.n, f) != (size_t)ids.n || fclose(f)) { fprintf(stderr, "long_ctx: write %s\n", outf); return 1; }
        fprintf(stderr, "[long_ctx] %ld bytes -> %d tokens (%.2f s)\n", nb, ids.n, rq_now() - t0);
        rq_tokens_free(&ids);
        free(t);
        rq_model_free(m);
        return 0;
    }
    long nb;
    int *ids = read_file(idsf, &nb), *pos = NULL, np = plist ? parse_list(plist, &pos) : 0;
    int nids = (int)(nb / sizeof(int));
    if (N < 0 || N > nids) N = nids;
    if (N < 1 || ((np || all) && !outf) || (np && all)) goto usage;
    for (int i = 0; i < np; i++)
        if (pos[i] < 0 || pos[i] >= N || (i && pos[i] <= pos[i - 1])) { fprintf(stderr, "long_ctx: bad -P (ascending, < %d)\n", N); return 2; }
    if (!cap) cap = N + G + 1;

    const char *w = rq_threads_init(0);
    if (w) fprintf(stderr, "%s\n", w);
    RqModelOpts o = { 0 };
    o.ctx_len = cap;
    o.kv_bytes = kvt == RQ_KV_F32 ? 4 : 2;
    double t0 = rq_now();
    RqModel *m = rq_model_load(path, &o, err, sizeof(err));
    if (!m) { fprintf(stderr, "long_ctx: %s\n", err); return 1; }
    RqModelInfo I;
    rq_model_info(m, &I);
    int V = I.vocab_size, hybrid = I.linear_layers > 0, e;
    RqCtx *c = rq_ctx_new(m, chunk, cap, &e);
    if (!c) die("rq_ctx_new", e, NULL);
    RqSeq *s = rq_seq_new(m, cap, kvt, &e);
    if (!s) die("rq_seq_new", e, NULL);
    if (hybrid && (e = rq_seq_set_checkpoints(s, 8)) != RQ_OK) die("rq_seq_set_checkpoints", e, NULL);
    fprintf(stderr, "[long_ctx] %s: %d tokens, capacity %d, KV %s, -b %d, %d threads, load %.1f s\n", path, N, cap,
            rq_kv_name(kvt), chunk, rq_threads(), rq_now() - t0);

    if (all) { // every row's logits, a chunk at a time
        FILE *f = fopen(outf, "wb");
        int32_t hdr[2] = { N, V };
        float *buf = malloc((size_t)chunk * V * sizeof(float));
        if (!f || !buf || fwrite(hdr, 4, 2, f) != 2 || fwrite(ids, 4, N, f) != (size_t)N) { fprintf(stderr, "long_ctx: write %s\n", outf); return 1; }
        double ta = rq_now();
        for (int p = 0; p < N; p += chunk) {
            int k = N - p < chunk ? N - p : chunk;
            if ((e = rq_prefill(c, s, ids + p, k, RQ_LOGITS_ALL, buf, NULL)) != RQ_OK) die("rq_prefill", e, c);
            if (fwrite(buf, sizeof(float), (size_t)k * V, f) != (size_t)k * V) { fprintf(stderr, "long_ctx: write %s\n", outf); return 1; }
        }
        if (fclose(f)) { fprintf(stderr, "long_ctx: write %s\n", outf); return 1; }
        fprintf(stderr, "[long_ctx] every row's logits of %d tokens: %.1f s\n", N, rq_now() - ta);
        free(buf);
    }
    float *out = np ? malloc((size_t)np * V * sizeof(float)) : NULL, *alt = malloc((size_t)4 * V * sizeof(float));
    double tp = rq_now();
    for (int i = 0; i < np; i++) {
        int p = pos[i];
        float *lg;
        if (vfy && p >= 64 && rq_seq_len(s) <= p - 63) {
            // the reference row: the last of a 64-row chunk p-63..p; then the segment and the decode step
            prefill_to(c, s, ids, p - 63, NULL);
            if (hybrid && (e = rq_seq_checkpoint_at(s, p - 63)) != RQ_OK) die("rq_seq_checkpoint_at", e, NULL);
            if ((e = rq_prefill(c, s, ids + p - 63, 64, RQ_LOGITS_LAST, NULL, &lg)) != RQ_OK) die("rq_prefill", e, c);
            memcpy(out + (size_t)i * V, lg, (size_t)V * sizeof(float));
            if ((e = rq_seq_truncate(s, p - 63)) != RQ_OK || rq_seq_len(s) != p - 63) die("rq_seq_truncate", e, NULL);
            if (hybrid && (e = rq_seq_checkpoint_at(s, p - 63)) != RQ_OK) die("rq_seq_checkpoint_at", e, NULL);
            prefill_to(c, s, ids, p - 3, NULL);
            RqSegment sg = { s, ids + p - 3, 4, RQ_LOGITS_ALL };
            if ((e = rq_forward_batch(c, &sg, 1, alt, NULL)) != RQ_OK) die("rq_forward_batch", e, c);
            int seg_ok = same(alt + (size_t)3 * V, out + (size_t)i * V, V);
            if ((e = rq_seq_truncate(s, p - 63)) != RQ_OK || rq_seq_len(s) != p - 63) die("rq_seq_truncate", e, NULL);
            prefill_to(c, s, ids, p, NULL);
            if ((e = rq_decode(c, s, ids[p], &lg)) != RQ_OK) die("rq_decode", e, c);
            int dec_ok = same(lg, out + (size_t)i * V, V);
            printf("verify %d %s segment=%d decode=%d\n", p, seg_ok && dec_ok ? "SAME" : "DIFFER", seg_ok, dec_ok);
        } else {
            prefill_to(c, s, ids, p + 1, &lg);
            memcpy(out + (size_t)i * V, lg, (size_t)V * sizeof(float));
        }
        fprintf(stderr, "[long_ctx] position %d: %.1f s\n", p, rq_now() - tp);
    }
    if (np) {
        FILE *f = fopen(outf, "wb");
        int32_t hdr[2] = { np, V };
        if (!f || fwrite(hdr, 4, 2, f) != 2 || fwrite(pos, 4, np, f) != (size_t)np ||
            fwrite(out, sizeof(float), (size_t)np * V, f) != (size_t)np * V || fclose(f)) {
            fprintf(stderr, "long_ctx: write %s\n", outf);
            return 1;
        }
    }
    if (G > 0) {
        float *lg = NULL;
        if (rq_seq_len(s) < N) prefill_to(c, s, ids, N, &lg);
        else lg = out + (size_t)(np - 1) * V; // (the last position was N - 1)
        double tg = rq_now();
        fprintf(stderr, "[long_ctx] prefilled %d tokens: %.1f s\n", N, tg - tp);
        RqSampling greedy = { 0 };
        RqTokens gen = { 0 };
        for (int k = 0; k < G; k++) {
            int t = rq_sample(c, s, lg, &greedy);
            if (t < 0) die("rq_sample", t, c);
            if (rq_is_end_token(m, t)) break;
            rq_tokens_push(&gen, &t, 1);
            if (k + 1 < G && rq_seq_len(s) < cap && (e = rq_decode(c, s, t, &lg)) != RQ_OK) die("rq_decode", e, c);
        }
        printf("gen");
        for (int k = 0; k < gen.n; k++) printf(" %d", gen.v[k]);
        printf("\ntext ");
        for (int k = 0; k < gen.n; k++) {
            int len;
            const char *b = rq_token_bytes(m, gen.v[k], &len);
            fwrite(b, 1, len, stdout);
        }
        printf("\n");
        fprintf(stderr, "[long_ctx] generated %d tokens: %.2f s (%.2f tok/s)\n", gen.n, rq_now() - tg,
                gen.n / (rq_now() - tg + 1e-9));
        rq_tokens_free(&gen);
    }
    free(out);
    free(alt);
    free(ids);
    free(pos);
    rq_seq_free(s);
    rq_ctx_free(c);
    rq_model_free(m);
    return 0;
usage:
    fprintf(stderr, "usage: long_ctx MODEL -T TEXT -o IDS | long_ctx MODEL -I IDS [-n N] [-P p1,p2,... | -A] [-o OUT] [-V] [-g G] "
                    "[-c CTX] [-b CHUNK] [-k f16|bf16|f32|bf16-dot|f16-dot]\n");
    return 2;
}
