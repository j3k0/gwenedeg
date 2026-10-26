/* Threading for the forward passes (forward.c): phase synchronization that tolerates a preempted thread, and the parallel
 * region the phases run in (a small persistent thread pool, or OpenMP). Measurements: TODO.md Step 7a.
 *
 * Why not OpenMP's barriers: decode runs ~100-200 synchronization points per token (~25 us apart).
 * libgomp's barriers spin, and every thread must arrive. With as many threads as vCPUs on a shared
 * machine, any other process preempting one thread (for a scheduler slice, ~ms) stalls all the others
 * at the next barrier, while they keep spinning on CPUs the preempted thread could use: decode
 * collapsed from ~180 to ~20 tok/s at 16 threads on the 16-vCPU dev VM. OMP_WAIT_POLICY=passive
 * (futex at once) costs a sleep / wake per barrier instead (~40 tok/s).
 *
 * Phases. A parallel job (one decoded token, one prefill chunk) is a sequence of phases 0, 1, ...;
 * phase k has n_k >= 1 work items. Each thread works through its own contiguous range of them and
 * then steals single items from the back of the others' ranges, or (phases whose items wait on
 * other items, and unequal items) all threads claim in order from one shared counter (rq_next), and
 * count the finished ones (rq_finish); the thread finishing the last item of phase k may run an
 * epilogue (serial work, e.g. the residual + rmsnorm + quantize after a matmul) and then
 * publishes "k + 1 phases complete" (rq_publish); a thread enters phase k + 1 once that is visible
 * (rq_wait). So a phase ends when its *work* is done, not when every thread has arrived: a thread
 * preempted between phases holds nothing up (it finds the phases done when it runs again, and
 * catches up), only one preempted inside an item does, until it runs again. Waiting threads spin
 * (with pause) for a while (RQ_SPIN_DEC / RQ_SPIN_PF rounds), then sleep on a futex (Linux; macOS
 * 14.4+: os_sync_wait_on_address; elsewhere they yield), so a stall frees their CPUs for the preempted thread and the other
 * processes; the publisher wakes sleepers only if there are any (no syscall in the common case).
 * Results do not depend on which thread runs an item (each output is computed whole by one thread,
 * as with the static schedules before), so any thread count and any interleaving give the same bits.
 * Static mode (RUNQ_SYNC=static, for comparison, OpenMP-region builds only): no stealing and no
 * shared counter, thread t does exactly its range [t n / nt, (t + 1) n / nt) of every phase, i.e. a
 * barrier per phase with the same wait.
 *
 * Parallel region (rq_parallel): a persistent pthread pool (default, except on Windows), which waits
 * at the end only for the threads that actually joined the job (a worker that has not started when
 * the work is done skips it), with the same spin-then-futex wait for the next job. With
 * -DRUNQ_NO_POOL (and on Windows) `#pragma omp parallel` instead: libgomp's fork / join per job is
 * itself a barrier every thread must reach, and at 16 threads on the 16-vCPU VM decode got ~70 tok/s
 * that way vs ~150-170 with the pool (~2-20 before, with OpenMP barriers). The pool also gives a
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
#elif defined(__aarch64__)
    #define rq_relax() __asm__ __volatile__("isb" ::: "memory") // (yield is a nop on Apple cores)
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
#elif defined(__APPLE__) && __has_include(<os/os_sync_wait_on_address.h>) && \
      defined(__MAC_OS_X_VERSION_MIN_REQUIRED) && __MAC_OS_X_VERSION_MIN_REQUIRED >= 140400
    #include <os/os_sync_wait_on_address.h> // (deployment target macOS 14.4+: the futex equivalent)
    #include <unistd.h>
    #define RQ_OSSYNC 1
#else
    #include <sched.h>
    #include <unistd.h>
#endif
#if defined(__APPLE__) && defined(__aarch64__)
    #include <sys/types.h>
    #include <sys/sysctl.h> // (rq_ncpu_perf)
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

// pause rounds before sleeping (Zen 4: ~28 ns per round, so 4,000 ~ 110 us), in decode phases (~10-60
// us apart) and prefill phases (~1-2 ms); 1,000 / 4,000 / 16,000 decode within noise at 4-16 threads
// (16-vCPU VM, best and median of 6), 4,000 prefill ~5% faster than 1,000 at 8-12 threads; and
// between jobs (RQ_SPIN_JOB)
#ifndef RQ_SPIN_DEC
#define RQ_SPIN_DEC 4000
#endif
#ifndef RQ_SPIN_PF
#define RQ_SPIN_PF 4000
#endif
#ifndef RQ_SPIN_JOB
#define RQ_SPIN_JOB 20000 // ~0.55 ms: covers the serial part between two decode steps (sampling at -t > 0)
#endif
extern int rq_spin_dec, rq_spin_pf; // (threads.c)
extern int rq_static; // RUNQ_SYNC=static: fixed item ranges per thread (a plain barrier per phase)

static void rq_sleep(int *word, int v) {
    // until *word (probably) differs from v
#ifdef RQ_FUTEX
    syscall(SYS_futex, word, FUTEX_WAIT_PRIVATE, v, NULL, NULL, 0);
#elif defined(RQ_OSSYNC)
    os_sync_wait_on_address(word, (uint64_t)(uint32_t)v, sizeof(int), OS_SYNC_WAIT_ON_ADDRESS_NONE);
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
#elif defined(RQ_OSSYNC)
    os_sync_wake_by_address_all(word, sizeof(int), OS_SYNC_WAKE_BY_ADDRESS_NONE);
#else
    (void)word;
#endif
}

static void rq_wait_ge(int *word, int target, int *sleepers, int spin) {
    // until *word >= target (a counter that only grows while anyone waits on it): spin, then sleep.
    // Sleepers register first and re-check (seq_cst both sides, see rq_store_wake), so a wake-up is
    // never lost
    for (int i = 0; i < spin; i++) {
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

typedef struct { int done, claim; char pad[56]; } RqPhase; // one cache line per phase

typedef struct {
    RqPhase *ph;   // per phase: items finished (reset by rq_begin)
    int nph;       // capacity
    int nt;        // threads of the job
    int *ready;    // phases complete (futex word; own cache line)
    int *sleepers; // threads sleeping on ready (or on a dependency counter)
    int *cnt;      // dependency counters inside phases (reset by rq_begin)
    int ncnt;
    int spin;      // pause rounds before a waiting thread sleeps (per job)
    uint64_t *own; // per (thread, phase): the thread's remaining item range, [tid * nph + k]
    int own_nt;    // threads own[] is sized for
    uint32_t epoch; // job number (16 bits): own[] words of older jobs are stale = the initial range
    void *mem;
} RqSync;

static int rq_sync_init(RqSync *s, int nph, int ncnt) {
    // capacity for jobs of up to nph phases and ncnt dependency counters; 0, or -1 (no memory)
    size_t bytes = (size_t)(nph + 2) * 64 + ((size_t)ncnt * sizeof(int) + 63) / 64 * 64;
    memset(s, 0, sizeof(*s));
    s->mem = aligned_alloc(64, bytes);
    if (!s->mem) return -1;
    memset(s->mem, 0, bytes);
    s->ph = (RqPhase *)((char *)s->mem + 128);
    s->ready = (int *)s->mem;
    s->sleepers = (int *)((char *)s->mem + 64);
    s->cnt = (int *)((char *)s->mem + (size_t)(nph + 2) * 64);
    s->nph = nph;
    s->ncnt = ncnt;
    s->nt = 1;
    return 0;
}

static void rq_sync_free(RqSync *s) { free(s->mem); free(s->own); s->mem = NULL; s->own = NULL; }

static int rq_begin(RqSync *s, int nph, int ncnt, int nt, int spin) {
    // before a job (serial): reset the counters of phases 0..nph-1 and dependency counters 0..ncnt-1;
    // a new epoch invalidates every thread's ranges (all zero again after 65,535 jobs). 0, or -1 (more
    // phases / counters than rq_sync_init's capacity: a bug) or -2 (no memory)
    if (nph > s->nph || ncnt > s->ncnt) return -1;
    memset(s->ph, 0, (size_t)nph * sizeof(RqPhase));
    memset(s->cnt, 0, (size_t)ncnt * sizeof(int));
    __atomic_store_n(s->ready, 0, __ATOMIC_RELAXED);
    if (nt > s->own_nt) {
        free(s->own);
        s->own = aligned_alloc(64, ((size_t)nt * s->nph * 8 + 63) / 64 * 64);
        if (!s->own) { s->own_nt = 0; return -2; }
        s->own_nt = nt;
        s->epoch = 0xffff;
    }
    s->epoch = (s->epoch + 1) & 0xffff;
    if (s->epoch == 0) {
        memset(s->own, 0, (size_t)s->own_nt * s->nph * 8);
        s->epoch = 1;
    }
    s->nt = nt;
    s->spin = spin;
    return 0;
}

// Items of a phase: thread t owns the range [t n / nt, (t + 1) n / nt) (in whole units of `unit`
// items, e.g. the chunks of a GEMM weight tile) and takes it from the front, `chunk` items at a time
// (contiguous, as a static schedule: long memory streams for the GEMV / GEMM weights); a thread with
// nothing left steals single items from the back of the others' ranges (a preempted thread's work
// gets done, and the end of a phase is balanced item by item). A range is one 64-bit word
// (epoch 16 | lo 24 | hi 24), updated by CAS; a word of an older epoch stands for the initial range
// Or (unit 0, rq_it_dyn) one shared counter: items claimed in order, `chunk` at a time (e.g. the
// prefill attention, longest items first: the best balance for items of unequal cost)
typedef struct { int k, n, unit, chunk, tid, state, victim; } RqIt;

static inline RqIt rq_it(int k, int n, int chunk, int unit, int tid) {
    // phase k, n items, the owner's chunks of `chunk` items, ranges of whole units (thieves take
    // single items from the back: the finest balance at the end of a phase); unit 0: shared counter
    if (unit < 0) unit = 1;
    if (chunk < 1) chunk = 1;
    RqIt it = { k, n, unit, chunk, tid, 0, 1 };
    return it;
}

static inline uint64_t rq_range(const RqSync *s, const RqIt *it, int t, uint64_t w, int *lo, int *hi) {
    // decode the range word w of thread t (the initial range if w is of another epoch)
    if ((uint32_t)(w >> 48) != s->epoch) {
        int nu = (it->n + it->unit - 1) / it->unit;
        *lo = (int)((long)nu * t / s->nt) * it->unit;
        *hi = (int)((long)nu * (t + 1) / s->nt) * it->unit;
        if (*hi > it->n) *hi = it->n;
    } else {
        *lo = (int)(w >> 24 & 0xffffff);
        *hi = (int)(w & 0xffffff);
    }
    return w;
}

static inline uint64_t rq_word(const RqSync *s, int lo, int hi) { return (uint64_t)s->epoch << 48 | (uint64_t)lo << 24 | (uint64_t)hi; }

static inline int rq_take(RqSync *s, RqIt *it, int t, int front, int *lo, int *hi) {
    // take a chunk from the front (owner) or the back (thief) of thread t's range
    uint64_t *p = &s->own[(size_t)t * s->nph + it->k];
    uint64_t w = __atomic_load_n(p, __ATOMIC_ACQUIRE);
    for (;;) {
        int a, b;
        rq_range(s, it, t, w, &a, &b);
        if (a >= b) return 0;
        int na = a, nb = b;
        if (front) {
            na = a + it->chunk < b ? a + it->chunk : b;
            *lo = a; *hi = na;
        } else { // one item from the back
            nb = b - 1;
            *lo = nb; *hi = b;
        }
        if (__atomic_compare_exchange_n(p, &w, rq_word(s, na, nb), 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
            return 1;
    }
}

static inline int rq_next(RqSync *s, RqIt *it, int *lo, int *hi) {
    // the next range of items [lo, hi) of phase it->k for this thread; 0 when there are no more
    if (it->unit == 0 && !rq_static) { // shared counter
        if (RQ_LOAD(&s->ph[it->k].claim) >= it->n) return 0; // (no RMW on a finished phase)
        int a = RQ_ADD(&s->ph[it->k].claim, it->chunk);
        if (a >= it->n) return 0;
        *lo = a;
        *hi = a + it->chunk < it->n ? a + it->chunk : it->n;
        return 1;
    }
    if (it->unit == 0) it->unit = 1;
    if (it->state == 0) {
        if (rq_take(s, it, it->tid, 1, lo, hi)) return 1;
        it->state = rq_static ? 2 : 1;
    }
    for (; it->state == 1 && it->victim < s->nt; it->victim++)
        if (rq_take(s, it, (it->tid + it->victim) % s->nt, 0, lo, hi)) return 1;
    it->state = 2;
    return 0;
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
    rq_wait_ge(s->ready, nk, s->sleepers, s->spin);
}

static inline int rq_count(RqSync *s, int i, int v) {
    // dependency counter i += v (after the work it stands for); returns the new value. Wakes the
    // sleepers, if any (they may wait for another counter or phase: they re-check and sleep again)
    int r = __atomic_add_fetch(&s->cnt[i], v, __ATOMIC_SEQ_CST);
    if (__atomic_load_n(s->sleepers, __ATOMIC_SEQ_CST))
        rq_wake(&s->cnt[i]);
    return r;
}

static inline void rq_wait_count(RqSync *s, int i, int target) {
    // until dependency counter i (inside a phase) reaches target
    if (RQ_LOAD(&s->cnt[i]) >= target) return;
    rq_wait_ge(&s->cnt[i], target, s->sleepers, s->spin);
}

// ----------------------------------------------------------------------------------- parallel region
// (threads.c: one pool per process)

typedef void (*RqFn)(void *ctx, int tid, int nt);

void rq_set_threads(int n);   // threads per job (before the first job: the pool's size)
int rq_env_threads(void);     // OMP_NUM_THREADS (its first value), 0 if unset or invalid
int rq_ncpu(void);            // online CPUs
int rq_ncpu_perf(void);       // Apple Silicon: the performance cores, else 0
void rq_parallel(RqFn fn, void *ctx); // run fn on every thread of the pool (tid 0 = the caller)
int rq_nthreads(void);        // threads of a job
const char *rq_config(void);  // tuning knobs from the environment (RUNQ_SPIN, RUNQ_SYNC); a warning or NULL

#endif // RUNQ_THREADS_H
