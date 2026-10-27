/* The thread pool of threads.h (one per process) and the threading configuration. */
#include "threads.h"
#include "runq.h"

int rq_spin_dec = RQ_SPIN_DEC, rq_spin_pf = RQ_SPIN_PF;
int rq_static = 0;

// ----------------------------------------------------------------------------------- parallel region

static int rq_nt = 0; // threads per job (rq_set_threads); 0: OMP_NUM_THREADS, else 1
static int rq_fixed = 0; // rq_threads_fix was called: the count no longer changes

void rq_set_threads(int n) {
    if (rq_fixed) return; // (a compute context's per-thread scratch is sized for it: rq_threads_fix)
    rq_nt = n < 1 ? 1 : n > 256 ? 256 : n;
#ifdef _OPENMP
    omp_set_num_threads(rq_nt); // (the OpenMP parallel region of -DRUNQ_NO_POOL / Windows)
#endif
}

int rq_env_threads(void) {
    // OMP_NUM_THREADS (its first value), 0 if unset or invalid
    const char *e = getenv("OMP_NUM_THREADS");
    int n = e ? atoi(e) : 0;
    return n > 0 ? (n < 256 ? n : 256) : 0;
}

int rq_ncpu(void) {
    // online CPUs
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#endif
}

int rq_ncpu_perf(void) {
    // Apple Silicon with more than one core type: the performance cores (hw.perflevel0, the fastest
    // level), at most the online CPUs; else (and on any other system) 0
#if defined(__APPLE__) && defined(__aarch64__)
    int levels = 0, n = 0;
    size_t len = sizeof levels;
    if (sysctlbyname("hw.nperflevels", &levels, &len, NULL, 0) || levels < 2)
        return 0;
    len = sizeof n;
    if (sysctlbyname("hw.perflevel0.logicalcpu", &n, &len, NULL, 0) || n < 1)
        return 0;
    return n < rq_ncpu() ? n : rq_ncpu();
#else
    return 0;
#endif
}

static int rq_threads_cfg(void) {
    // threads per job
    if (rq_nt) return rq_nt;
    int n = rq_env_threads();
    return n ? n : 1;
}

#ifdef RUNQ_POOL
// Persistent pool: nt - 1 workers (tid 1..nt-1) + the caller (tid 0). Job protocol: the caller sets
// fn / ctx, opens state (bit 0 = closed, the rest 2 x workers inside), then bumps gen (futex word);
// a worker that sees a new gen enters only while the job is open. After its own share the caller
// closes the job and waits until no worker is inside: workers that have not started yet skip the
// job (with dynamic phases all the work is done by then; static mode is not supported here). A
// worker delayed between its gen check and entering may enter the next job while the caller has
// opened it but not bumped gen yet, and then enter that job a second time: harmless (it runs fn
// after its first run returned, with the same tid: claims find their items taken, scratch per tid)
static struct {
    int gen, sleepers, state, state_sleepers, nt, started;
    RqFn fn;
    void *ctx;
    pthread_t *th;
} rq_pool;

static void *rq_pool_main(void *arg) {
    int tid = (int)(intptr_t)arg, my_gen = 0;
    for (;;) {
        rq_wait_ge(&rq_pool.gen, my_gen + 1, &rq_pool.sleepers, RQ_SPIN_JOB);
        int s = __atomic_load_n(&rq_pool.state, __ATOMIC_ACQUIRE);
        int in = 0;
        while (!(s & 1))
            if (__atomic_compare_exchange_n(&rq_pool.state, &s, s + 2, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) { in = 1; break; }
        my_gen = __atomic_load_n(&rq_pool.gen, __ATOMIC_ACQUIRE); // (the gen of the job entered, or one less)
        if (!in)
            continue;
        rq_pool.fn(rq_pool.ctx, tid, rq_pool.nt);
        if (__atomic_sub_fetch(&rq_pool.state, 2, __ATOMIC_ACQ_REL) == 1 && __atomic_load_n(&rq_pool.state_sleepers, __ATOMIC_SEQ_CST))
            rq_wake(&rq_pool.state);
    }
    return NULL;
}

static void rq_pool_start(void) {
    // the workers (at the first job); if the system refuses some, the pool has the ones it got
    int nt = rq_threads_cfg();
    rq_pool.state = 1;
    rq_pool.th = calloc(nt, sizeof(pthread_t));
    rq_pool.nt = 1;
    for (int i = 1; i < nt && rq_pool.th; i++) {
        if (pthread_create(&rq_pool.th[i], NULL, rq_pool_main, (void *)(intptr_t)i))
            break;
        rq_pool.nt = i + 1;
    }
    rq_pool.started = 1;
}

void rq_parallel(RqFn fn, void *ctx) {
    if (!rq_pool.started)
        rq_pool_start();
    int nt = rq_pool.nt;
    if (nt == 1) {
        fn(ctx, 0, 1);
        return;
    }
    rq_pool.fn = fn;
    rq_pool.ctx = ctx;
    // open the job, then announce it: in the other order, a worker spinning on gen saw the new gen
    // while the job was still closed and skipped it (most of them, every other job: decode ran with
    // a fraction of the threads every second token, ~20-40% slower at 4-12 threads)
    __atomic_store_n(&rq_pool.state, 0, __ATOMIC_SEQ_CST);
    __atomic_add_fetch(&rq_pool.gen, 1, __ATOMIC_SEQ_CST);
    if (__atomic_load_n(&rq_pool.sleepers, __ATOMIC_SEQ_CST))
        rq_wake(&rq_pool.gen);
    fn(ctx, 0, nt);
    int s = __atomic_or_fetch(&rq_pool.state, 1, __ATOMIC_SEQ_CST);
    if (s != 1) {
        for (int i = 0; i < RQ_SPIN_JOB && s != 1; i++) {
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

int rq_nthreads(void) { return rq_pool.started ? rq_pool.nt : rq_threads_cfg(); }
#else
void rq_parallel(RqFn fn, void *ctx) {
#ifdef _OPENMP
    #pragma omp parallel
    fn(ctx, omp_get_thread_num(), omp_get_num_threads());
#else
    fn(ctx, 0, 1);
#endif
}

int rq_nthreads(void) { return rq_threads_cfg(); }
#endif

const char *rq_config(void) {
    // tuning knobs from the environment (RUNQ_SPIN=decode[,prefill], RUNQ_SYNC=static); a warning
    // for the caller to show, or NULL
    const char *e = getenv("RUNQ_SPIN");
    if (e) {
        rq_spin_dec = rq_spin_pf = atoi(e);
        if (strchr(e, ','))
            rq_spin_pf = atoi(strchr(e, ',') + 1);
    }
    e = getenv("RUNQ_SYNC");
    if (e && !strcmp(e, "static")) rq_static = 1;
#ifdef RUNQ_POOL
    if (rq_static) {
        rq_static = 0;
        return "RUNQ_SYNC=static: not with the pool (workers may skip a job), ignored";
    }
#endif
    return NULL;
}


const char *rq_threads_init(int n) {
    // threads: n, else OMP_NUM_THREADS, else the default nproc - nproc/8 (14 of 16 vCPUs). Measured on
    // the shared 16-vCPU dev VM (~3 CPUs busy with other work; Step 7a): decode 12 ~ 14 > 16 threads
    // (memory-bound: 12 already saturate the bandwidth), prefill 14 > 16 > 12 (~2,580 / ~2,290-2,500 /
    // ~2,420 tok/s at 439 tokens). All 16 work too (the phases tolerate a preempted thread; with
    // OpenMP's barriers they collapsed to ~2-20 tok/s), just not faster with other processes running.
    // Apple Silicon with efficiency cores: the performance cores only (rq_ncpu_perf). M1 Pro, 8 P + 2 E
    // (BENCHMARKS.md "Apple M1 Pro"): 9 or 10 threads instead of 8 decode 5-13% slower, prefill within 3%
    if (n > 0) {
        rq_set_threads(n);
    } else if (rq_env_threads()) {
        rq_set_threads(rq_env_threads());
    } else {
        int np = rq_ncpu(), npc = rq_ncpu_perf();
        rq_set_threads(npc ? npc : np - np / 8);
    }
    return rq_config();
}

int rq_threads(void) { return rq_nthreads(); }

int rq_threads_fix(void) {
    // the threads of every later job, fixed from now on (rq_ctx_new sizes a context's per-thread scratch by it;
    // the pool's size cannot change once started anyway). Was: a context made before the pool started (a .bin
    // model on the generic build loads without a parallel job) or in an OpenMP-region build, then
    // rq_threads_init with more threads (e.g. before loading a second model): jobs with tids past its scratch
    if (!rq_fixed) {
        rq_set_threads(rq_threads_cfg()); // (OpenMP: its thread count = ours, also when rq_threads_init was not called)
        rq_fixed = 1;
    }
    return rq_nthreads();
}
