/* Threading for runq.c: phase synchronization that tolerates a preempted thread, and the parallel
 * region the phases run in (OpenMP, or a small persistent thread pool).
 *
 * Why not OpenMP's barriers: decode runs ~100-200 synchronization points per token (~25 us apart).
 * libgomp's barriers spin, and every thread must arrive. With as many threads as vCPUs on a shared
 * machine, any other process preempting one thread (for a scheduler slice, ~ms) stalls all the others
 * at the next barrier, while they keep spinning on CPUs the preempted thread could use: decode
 * collapsed from ~180 to ~20 tok/s at 16 threads on the 16-vCPU dev VM. OMP_WAIT_POLICY=passive
 * (futex at once) costs a sleep / wake per barrier instead (~40 tok/s).
 *
 * Phases. A parallel job (one decoded token, one prefill chunk) is a sequence of phases 0, 1, ...;
 * phase k has n_k >= 1 work items. Threads claim items with an atomic counter (rq_next; chunks of
 * items) and count the finished ones (rq_finish); the thread finishing the last item of phase k may
 * run an epilogue (serial work, e.g. the residual + rmsnorm + quantize after a matmul) and then
 * publishes "k + 1 phases complete" (rq_publish); a thread enters phase k + 1 once that is visible
 * (rq_wait). So a phase ends when its *work* is done, not when every thread has arrived: a thread
 * preempted between phases holds nothing up (it finds the phases done when it runs again, and
 * catches up), only one preempted inside an item does, until it runs again. Waiting threads spin
 * (with pause) for RQ_SPIN rounds, then sleep on a futex (Linux; elsewhere they yield), so a stall
 * frees their CPUs for the preempted thread and the other processes; the publisher wakes sleepers
 * only if there are any (no syscall in the common case).
 * Results do not depend on which thread runs an item (each output is computed whole by one thread,
 * as with the static schedules before), so any thread count and any interleaving give the same bits.
 * Static mode (RUNQ_SYNC=static, for comparison): thread t takes items [t n / nt, (t + 1) n / nt) of
 * every phase, i.e. a barrier per phase with the same wait.
 *
 * Parallel region (rq_parallel): a persistent pthread pool (default, except on Windows), which waits
 * at the end only for the threads that actually joined the job (a worker that has not started when
 * the work is done skips it), with the same spin-then-futex wait for the next job. With
 * -DRUNQ_NO_POOL (and on Windows) `#pragma omp parallel` instead: libgomp's fork / join per job is
 * itself a barrier every thread must reach, and at 16 threads on the 16-vCPU VM decode got ~80 tok/s
 * that way vs ~180 with the pool (~15-20 before, with OpenMP barriers). The pool also gives a
 * ThreadSanitizer build without OpenMP (libgomp is not instrumented).
 */
#ifndef RUNQ_THREADS_H
#define RUNQ_THREADS_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(__x86_64__) || defined(__i386__)
    #include <immintrin.h>
    #define rq_relax() _mm_pause()
#else
    #define rq_relax() ((void)0)
#endif
#if defined(__linux__)
    #include <linux/futex.h>
    #include <sys/syscall.h>
    #include <unistd.h>
    #include <limits.h>
    #define RQ_FUTEX 1
#elif defined(_WIN32)
    #include <windows.h>
#else
    #include <sched.h>
#endif
#ifdef _OPENMP
    #include <omp.h>
#endif
#if !defined(_WIN32) && !defined(RUNQ_NO_POOL) && !defined(RUNQ_POOL)
    #define RUNQ_POOL 1
#endif
#ifdef RUNQ_POOL
    #include <pthread.h>
#endif

#define RQ_LOAD(p) __atomic_load_n((p), __ATOMIC_ACQUIRE)
#define RQ_ADD(p, v) __atomic_fetch_add((p), (v), __ATOMIC_ACQ_REL)

#ifndef RQ_SPIN
#define RQ_SPIN 4000 // pause rounds before sleeping (Zen 4: ~10 ns per round + the load)
#endif
static int rq_spin = RQ_SPIN;
static int rq_static = 0; // RUNQ_SYNC=static: fixed item ranges per thread (a plain barrier per phase)

static void rq_sleep(int *word, int v) {
    // until *word (probably) differs from v
#ifdef RQ_FUTEX
    syscall(SYS_futex, word, FUTEX_WAIT_PRIVATE, v, NULL, NULL, 0);
#elif defined(_WIN32)
    (void)word; (void)v;
    SwitchToThread();
#else
    (void)word; (void)v;
    sched_yield();
#endif
}

static void rq_wake(int *word) {
#ifdef RQ_FUTEX
    syscall(SYS_futex, word, FUTEX_WAKE_PRIVATE, INT_MAX, NULL, NULL, 0);
#else
    (void)word;
#endif
}

static void rq_wait_ge(int *word, int target, int *sleepers) {
    // until *word >= target (a counter that only grows while anyone waits on it): spin, then sleep.
    // Sleepers register first and re-check (seq_cst both sides, see rq_store_wake), so a wake-up is
    // never lost
    for (int i = 0; i < rq_spin; i++) {
        if (RQ_LOAD(word) >= target)
            return;
        rq_relax();
    }
    __atomic_fetch_add(sleepers, 1, __ATOMIC_SEQ_CST);
    int v;
    while ((v = __atomic_load_n(word, __ATOMIC_SEQ_CST)) < target)
        rq_sleep(word, v);
    __atomic_fetch_sub(sleepers, 1, __ATOMIC_RELAXED);
}

static void rq_store_wake(int *word, int v, int *sleepers) {
    __atomic_store_n(word, v, __ATOMIC_SEQ_CST);
    if (__atomic_load_n(sleepers, __ATOMIC_SEQ_CST))
        rq_wake(word);
}

// ----------------------------------------------------------------------------------- phases

typedef struct { int claim, done; char pad[56]; } RqPhase; // one cache line per phase

typedef struct {
    RqPhase *ph;   // per phase: items claimed / finished (reset by rq_begin)
    int nph;       // capacity
    int nt;        // threads of the job
    int *ready;    // phases complete (futex word; own cache line)
    int *sleepers; // threads sleeping on ready
    void *mem;
} RqSync;

static void rq_sync_init(RqSync *s, int nph) {
    s->mem = aligned_alloc(64, (size_t)(nph + 2) * 64);
    if (!s->mem) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    memset(s->mem, 0, (size_t)(nph + 2) * 64);
    s->ph = (RqPhase *)((char *)s->mem + 128);
    s->ready = (int *)s->mem;
    s->sleepers = (int *)((char *)s->mem + 64);
    s->nph = nph;
    s->nt = 1;
}

static void rq_sync_free(RqSync *s) { free(s->mem); s->mem = NULL; }

static void rq_begin(RqSync *s, int nph, int nt) {
    // before a job (serial): reset the counters of phases 0..nph-1
    if (nph > s->nph) { fprintf(stderr, "rq_begin: too many phases\n"); exit(EXIT_FAILURE); }
    memset(s->ph, 0, (size_t)nph * sizeof(RqPhase));
    __atomic_store_n(s->ready, 0, __ATOMIC_RELAXED);
    s->nt = nt;
}

// per-thread iteration state of one phase
typedef struct { int k, n, chunk, tid, done; } RqIt;

static inline RqIt rq_it(int k, int n, int chunk, int tid) { RqIt it = { k, n, chunk > 0 ? chunk : 1, tid, 0 }; return it; }

static inline int rq_next(RqSync *s, RqIt *it, int *lo, int *hi) {
    // the next range of items [lo, hi) of phase it->k for this thread; 0 when there are no more
    if (rq_static) {
        if (it->done) return 0;
        it->done = 1;
        *lo = (int)((long)it->n * it->tid / s->nt);
        *hi = (int)((long)it->n * (it->tid + 1) / s->nt);
        return *hi > *lo;
    }
    if (RQ_LOAD(&s->ph[it->k].claim) >= it->n) return 0; // (no RMW on a finished phase)
    int a = RQ_ADD(&s->ph[it->k].claim, it->chunk);
    if (a >= it->n) return 0;
    *lo = a;
    *hi = a + it->chunk < it->n ? a + it->chunk : it->n;
    return 1;
}

static inline int rq_finish(RqSync *s, const RqIt *it, int cnt) {
    // cnt items of the phase finished; 1 if they were its last ones (the caller runs the epilogue,
    // if any, then rq_publish(k + 1))
    return RQ_ADD(&s->ph[it->k].done, cnt) + cnt == it->n;
}

static inline void rq_publish(RqSync *s, int nk) { rq_store_wake(s->ready, nk, s->sleepers); }

static inline void rq_wait(RqSync *s, int nk) {
    // until phases 0..nk-1 are complete
    if (RQ_LOAD(s->ready) >= nk) return;
    rq_wait_ge(s->ready, nk, s->sleepers);
}

static inline void rq_wait_count(RqSync *s, int *cnt, int target) {
    // until a dependency counter (inside a phase) reaches target
    if (RQ_LOAD(cnt) >= target) return;
    rq_wait_ge(cnt, target, s->sleepers);
}

// ----------------------------------------------------------------------------------- parallel region

typedef void (*RqFn)(void *ctx, int tid, int nt);

static int rq_threads(void) {
    // threads per job
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    const char *e = getenv("OMP_NUM_THREADS");
    int n = e ? atoi(e) : 1;
    return n > 0 ? (n < 256 ? n : 256) : 1;
#endif
}

#ifdef RUNQ_POOL
// Persistent pool: nt - 1 workers (tid 1..nt-1) + the caller (tid 0). Job protocol: the caller sets
// fn / ctx, bumps gen (futex word), then opens state (bit 0 = closed, the rest 2 x workers inside);
// a worker that sees a new gen enters only while the job is open. After its own share the caller
// closes the job and waits until no worker is inside: workers that have not started yet skip the
// job (with dynamic phases all the work is done by then; static mode is not supported here)
static struct {
    int gen, sleepers, state, state_sleepers, nt, entered;
    RqFn fn;
    void *ctx;
    pthread_t *th;
} rq_pool;

static void *rq_pool_main(void *arg) {
    int tid = (int)(intptr_t)arg, my_gen = 0;
    for (;;) {
        rq_wait_ge(&rq_pool.gen, my_gen + 1, &rq_pool.sleepers);
        int s = __atomic_load_n(&rq_pool.state, __ATOMIC_ACQUIRE);
        int in = 0;
        while (!(s & 1))
            if (__atomic_compare_exchange_n(&rq_pool.state, &s, s + 2, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) { in = 1; break; }
        my_gen = __atomic_load_n(&rq_pool.gen, __ATOMIC_ACQUIRE); // (the gen of the job entered, or a later one)
        if (!in)
            continue;
        rq_pool.fn(rq_pool.ctx, tid, rq_pool.nt);
        if (__atomic_sub_fetch(&rq_pool.state, 2, __ATOMIC_ACQ_REL) == 1 && __atomic_load_n(&rq_pool.state_sleepers, __ATOMIC_SEQ_CST))
            rq_wake(&rq_pool.state);
    }
    return NULL;
}

static void rq_parallel(RqFn fn, void *ctx) {
    int nt = rq_threads();
    if (!rq_pool.th) {
        rq_pool.nt = nt;
        rq_pool.state = 1;
        rq_pool.th = calloc(nt, sizeof(pthread_t));
        for (int i = 1; i < nt; i++)
            if (pthread_create(&rq_pool.th[i], NULL, rq_pool_main, (void *)(intptr_t)i)) { fprintf(stderr, "pthread_create failed\n"); exit(EXIT_FAILURE); }
    }
    nt = rq_pool.nt;
    rq_pool.fn = fn;
    rq_pool.ctx = ctx;
    __atomic_add_fetch(&rq_pool.gen, 1, __ATOMIC_SEQ_CST);
    __atomic_store_n(&rq_pool.state, 0, __ATOMIC_SEQ_CST);
    if (__atomic_load_n(&rq_pool.sleepers, __ATOMIC_SEQ_CST))
        rq_wake(&rq_pool.gen);
    fn(ctx, 0, nt);
    int s = __atomic_or_fetch(&rq_pool.state, 1, __ATOMIC_SEQ_CST);
    if (s != 1) {
        for (int i = 0; i < rq_spin && s != 1; i++) {
            rq_relax();
            s = __atomic_load_n(&rq_pool.state, __ATOMIC_ACQUIRE);
        }
        if (s != 1) {
            __atomic_fetch_add(&rq_pool.state_sleepers, 1, __ATOMIC_SEQ_CST);
            while ((s = __atomic_load_n(&rq_pool.state, __ATOMIC_SEQ_CST)) != 1)
                rq_sleep(&rq_pool.state, s);
            __atomic_fetch_sub(&rq_pool.state_sleepers, 1, __ATOMIC_RELAXED);
        }
    }
}

static inline int rq_nthreads(void) { return rq_pool.th ? rq_pool.nt : rq_threads(); }
#else
static void rq_parallel(RqFn fn, void *ctx) {
#ifdef _OPENMP
    #pragma omp parallel
    fn(ctx, omp_get_thread_num(), omp_get_num_threads());
#else
    fn(ctx, 0, 1);
#endif
}

static inline int rq_nthreads(void) { return rq_threads(); }
#endif

static void rq_config(void) {
    // tuning knobs from the environment (RUNQ_SPIN, RUNQ_SYNC=static)
    const char *e = getenv("RUNQ_SPIN");
    if (e) rq_spin = atoi(e);
    e = getenv("RUNQ_SYNC");
    if (e && !strcmp(e, "static")) rq_static = 1;
#ifdef RUNQ_POOL
    if (rq_static) { fprintf(stderr, "RUNQ_SYNC=static: not with the pool (workers may skip a job), ignored\n"); rq_static = 0; }
#endif
}

#endif // RUNQ_THREADS_H
