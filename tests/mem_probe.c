/* mem_probe: the load-time memory estimate (rq_model_mem) against the measured memory, and the model file's page
 * cache after the load (TODO P4; tests/README.md, Memory). Linux (/proc/self/status, mincore).
 *   build/mem_probe MODEL CTX [--cold] [--page-cache auto|drop|keep] [-k f16|bf16|f32]
 * Loads MODEL with a context of CTX positions (no memory check), then prints one line: the load time, the
 * estimate's parts, the peak RSS of the load (VmHWM) against weights + runtime, the file's bytes in the page cache
 * after the load (mincore of a fresh mapping), and the peak RSS after filling a sequence of CTX positions (a prefill
 * in chunks of 512, the last 8 decoded; for Qwen3.5 with its checkpoints taken) against the whole estimate.
 * --cold: drops the file's page cache first (posix_fadvise DONTNEED: clean pages nobody maps). Threads:
 * OMP_NUM_THREADS. `make mem-probe` builds it */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "../src/runq.h"

static double status_mib(const char *key) {
    FILE *f = fopen("/proc/self/status", "r");
    char l[256];
    long v = -1;
    size_t n = strlen(key);
    while (f && fgets(l, sizeof l, f))
        if (!strncmp(l, key, n) && l[n] == ':') v = atol(l + n + 1);
    if (f) fclose(f);
    return v / 1024.0;
}

static double cached_mib(const char *path) {
    // the file's bytes in the page cache (mincore of a mapping that touches nothing)
    int fd = open(path, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) || st.st_size == 0) return -1;
    void *p = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return -1;
    long pg = sysconf(_SC_PAGESIZE);
    size_t np = (st.st_size + pg - 1) / pg, in = 0;
    unsigned char *v = malloc(np);
    if (v && !mincore(p, st.st_size, v))
        for (size_t i = 0; i < np; i++) in += v[i] & 1;
    free(v);
    munmap(p, st.st_size);
    return (double)in * pg / 1048576.0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: mem_probe MODEL CTX [--cold] [--page-cache auto|drop|keep] [-k f16|bf16|f32]\n");
        return 2;
    }
    const char *path = argv[1];
    int ctx = atoi(argv[2]), cold = 0, page_cache = 0, kv = RQ_KV_DEFAULT;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--cold")) cold = 1;
        else if (!strcmp(argv[i], "--page-cache") && i + 1 < argc) {
            const char *v = argv[++i];
            page_cache = !strcmp(v, "drop") ? 1 : !strcmp(v, "keep") ? -1 : 0;
        } else if (!strcmp(argv[i], "-k") && i + 1 < argc) {
            if ((kv = rq_kv_parse(argv[++i])) < 0) return 2;
        } else return 2;
    }
    if (cold) {
        int fd = open(path, O_RDONLY);
        if (fd >= 0) {
            posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            close(fd);
        }
    }
    rq_threads_init(0);
    double c0 = cached_mib(path), t0 = rq_now();
    char err[512];
    RqModelOpts o = { ctx, 0, 0, -1, kv == RQ_KV_F32 ? 4 : 2, page_cache };
    RqModel *m = rq_model_load(path, &o, err, sizeof err);
    if (!m) { fprintf(stderr, "%s\n", err); return 1; }
    double tl = rq_now() - t0, h0 = status_mib("VmHWM"), c1 = cached_mib(path);
    RqModelMem mm;
    rq_model_mem(m, &mm);
    int e;
    RqSeq *s = rq_seq_new(m, 0, kv, &e);
    RqCtx *c = rq_ctx_new(m, RQ_BATCH_MAX, 0, &e);
    if (!s || !c) { fprintf(stderr, "out of memory\n"); return 1; }
    int cap = rq_seq_capacity(s), n = cap - 8;
    int *tok = malloc(cap * sizeof(int));
    for (int i = 0; i < cap; i++) tok[i] = 1000 + (i * 7919) % 20000;
    for (int k = 1; k <= mm.seq.checkpoints; k++) rq_seq_checkpoint_at(s, n * k / (mm.seq.checkpoints + 1));
    if (rq_prefill(c, s, tok, n, RQ_LOGITS_LAST, NULL, NULL) != RQ_OK) { fprintf(stderr, "prefill failed\n"); return 1; }
    for (int i = n; i < cap; i++) rq_decode(c, s, tok[i], NULL);
    double h1 = status_mib("VmHWM");
    const double M = 1048576.0;
    printf("%s, ctx %d: load %.2f s | estimate MiB: weights %.1f, runtime %.1f, sequence %.1f (KV %.1f, state %.1f x (1 + %d)), "
           "total %.1f | load peak RSS %.1f MiB (weights + runtime %+.1f%%) | file cached %.0f -> %.0f MiB | peak RSS with the "
           "context full %.1f MiB (estimate %+.1f%%)\n",
           path, cap, tl, mm.weights / M, mm.runtime / M, mm.seq.total / M, mm.seq.kv / M, mm.seq.state / M, mm.seq.checkpoints,
           mm.total / M, h0, 100.0 * ((mm.weights + mm.runtime) / M - h0) / h0, c0, c1, h1, 100.0 * (mm.total / M - h1) / h1);
    free(tok);
    rq_ctx_free(c);
    rq_seq_free(s);
    rq_model_free(m);
    return 0;
}
