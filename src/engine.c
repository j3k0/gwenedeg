/* runq serve's engine (serve.h): one thread runs every Job on one compute context, with continuous
 * batching: each engine step is one rq_forward_batch pass over (a) the next token(s) of every decoding
 * job (one row each; a constrained job's forced tokens ride along with its sampled one) and (b) prompt
 * chunks of jobs being prefilled, within a token budget per pass (--chunk while other jobs decode, -b
 * when none does); then rq_sample_batch with each job's sampler settings and token mask. Jobs join at
 * the next step (FIFO admission, at most --parallel running) and leave when done, cancelled or failed;
 * the connection threads only queue Jobs and read their output, so /health and queueing never wait on
 * a pass.
 *
 * Bit-identity: a sequence's logits do not depend on the other rows of a pass, on the chunking or the
 * thread count (runq.h), and each job has its own sequence (slot) with its own RNG, so a request gives
 * the same output whatever it is batched with (tests/server_test.py, group batch).
 *
 * KV memory: a pool of sequences (slots, --slots, bounded by --kv-mem, each counted with its KV cache and, for
 * Qwen3.5, its recurrent state and checkpoints: rq_seq_mem) of -c positions each (address space, touched as
 * used). A running job owns one; idle ones keep their tokens as prefix caches. A new
 * job takes the idle slot sharing the longest token prefix with its prompt if that is a real reuse
 * (>= 32 tokens or half the prompt), truncates it to that prefix and prefills only the rest; if a busy
 * slot (another running job) shares a longer prefix, it is copied (rq_seq_copy: the same bits) into the
 * least recently used idle slot instead; if a job being prefilled will share a longer prefix than
 * anything cached (e.g. LiteLLM's rubric for a burst of classifications), the new job waits until that
 * part is prefilled and then copies it. Otherwise the least recently used idle slot is taken (LRU
 * eviction). The whole prompt cached: its last token is prefilled again, for the logits. Slot affinity: an idle
 * slot that would lose more tokens than it keeps (it shares only a prefix, e.g. the system prompt with a new
 * session or with a compaction's next request) is not truncated: the prefix is copied into the least recently
 * used other idle slot, so that the conversation it holds finds it whole at its next request.
 *
 * Disk tier (--kv-disk, below "the disk tier"): a slot whose tokens a job would discard is saved to a file first, and
 * a prompt that continues a saved sequence further than any slot gets it loaded into a slot first (admit plans the
 * I/O, the engine thread does it between passes without mu); the files outlive the process.
 *
 * Qwen3.5 / 3.8 (hybrid: the recurrent state cannot be rewound, a slot keeps only checkpointed lengths): each job
 * asks for checkpoints where its conversation's next requests leave the cache (ck_plan_job, --checkpoints per slot,
 * default 4): the prompt's assistant header (a reply sent back differently from what was generated: an empty
 * reasoning re-tokenized, a stop string), the end of the system block (the system prompt and the tools: shared by
 * every request of the agent, kept through a compaction), Qwen3.5's turn start (its template drops the turn's
 * reasoning at the next user query) and the reply's end (ck_reply_end: a retry or a branch after the reply). The
 * slot's other checkpoints are dropped first (rq_seq_checkpoint_drop), so the lowest-first replacement of new ones
 * never takes these. An agent's next turn then prefills only its new tokens (scripts/replay_sessions.py).
 *
 * Generation, per job (as Step 9's one-job loop did it): the RNG is seeded per request and draws once
 * per prompt token but the last, as the CLI's chat does for a first turn (rq_generate), however much of
 * the prompt was cached: a seeded request gives the same text as `runq -m chat -s SEED` on the same
 * single-turn conversation. Then sample / feed until an end token (not fed), max tokens, a stop string,
 * the end of the context, a cancel (client gone), the deadline or the shutdown's grace period. With a
 * json_schema constraint (constrain.c) the forced tokens are appended without a forward pass of their
 * own: the initial ones go into the prompt's prefill, the ones after a decision are fed with the next
 * pass (a segment of several rows), and the last run never if the output is then complete.
 */
#ifndef _WIN32
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include "serve.h"

enum { R_PREFILL = 1, R_DECODE = 2 };

typedef struct Run Run;

typedef struct {
    RqSeq *seq;
    RqSeq *dseq;          // speculation: the draft model's sequence (follows seq's tokens; NULL: no draft model)
    double used;          // last use (LRU)
    Run *run;             // the running job on it, or NULL: idle (a prefix cache)
    unsigned char *allow; // constrained decoding: the token mask (V bytes, all 0 between steps)
    int *list;            // the allowed tokens set in allow
    float *lcopy;         // logprobs: the raw logits of the step (sampling modifies them)
    unsigned gen, dk_gen; // the disk tier: the contents' generation (+1 per job, per load), the last one saved (or
                          // given up on, or loaded): gen != dk_gen means the disk has not seen these tokens
} Slot;

// the disk tier: a saved sequence, <id>.rqseq (+ <id>.draft.rqseq: the draft's, dlen positions, 0: none)
typedef struct {
    int id, dlen;
    int *tok;             // its tokens (f.len)
    RqSeqFile f;
    long long bytes;      // both files
    double used;          // last save / load / hit (rq_now's clock; the mtime's at the start)
} DiskEnt;


struct Engine {
    RqModel *m;
    RqCtx *c;
    Slot *slot;
    int nslot, parallel, maxb;
    EngineOpts o;
    RqSeqMem sm;          // the memory of one slot (rq_seq_mem: KV cache of ctx positions, recurrent state, checkpoints)
    const char *mem_src;  // where the pool's budget came from (engine_describe)
    ConVocab *vocab;
    int V, eos, think_start, think_end, chat; // chat: the template variant (RQ_CHAT_*)
    int hybrid;                               // Qwen3.5's linear layers: a copy keeps only a checkpointed length
    int ckn;                                  // their checkpoints per slot (--checkpoints)
    int t_im_start, t_system, t_user, t_assistant, t_nl, t_toolresp; // the chat tokens ck_plan_job looks for
    Run **run;            // running jobs, in admission order
    int nrun;
    // per pass
    RqSegment *seg;
    Run **segrun;
    float *logits;        // parallel x V
    RqSeq **sseq;
    RqSampling *scfg;
    int *stok;
    Run **srun;
    // speculation (S8 / G7): the draft model and its context, the cost models, the guesses per pass at most (all jobs),
    // the sampling rounds' rows, the draft passes' segments, the acceptance seen (server-wide: the model's guesses,
    // the lookup's per match length; checked / accepted, decayed)
    RqModel *dm;
    RqCtx *dc;
    int dhybrid;
    RqSeqMem dsm;         // the memory of a slot's draft sequence
    double spec_res;      // bytes kept out of the pool for the verify rows' transient checkpoints
    RqPassCost tcost, dcost; // as the planner uses them: the calibrated ones (tcal, dcal) x the measured / predicted ratio
    RqPassCost tcal, dcal;
    double tlr, dlr;      // log of that ratio, a moving average over the passes (target, draft)
    int gmax, dmaxb;
    float *slg, *dlogits;
    RqSegment *dseg;
    Run **dsegrun;
    struct Cand *cand;
    double pm[2], pl[RQ_LOOKUP_LMAX + 1][2];
    unsigned long long sp_passes, sp_rows, sp_guess[3], sp_acc[3], d_passes, d_rows;
    double t_draft, t_cal;
    int since_model;      // passes since the policy last chose the model for a job (its drafts are kept in sync while recent)
    // the MTP drafter (--draft mtp: dm is the model's MTP view; below "the MTP drafter"): the hidden state's width, the
    // target's pass's hidden rows (H: maxb x dim, positions as the pass's segments) and the view's passes' h inputs (Hin),
    // auto's prior per guess (rq_spec_prior), the counters of the -v summary (A: the guesses' passes, B: the follow passes;
    // the steps' time), RUNQ_TEST_MTP_CHECK (the at-rest check after each step) and RUNQ_TEST_MTP_DUMP (a directory: the
    // finished jobs' sequences; tests only)
    int mtp, dim;
    float *H, *Hin;
    double dprior;
    unsigned long long mA_passes, mA_rows, mB_passes, mB_rows, m_approx, m_gaps, steps;
    double t_mtpA, t_mtpB, t_step;
    int mtp_chk;
    const char *mtp_dump;
    // the queue (shared with the connection threads, under mu)
    pthread_t th;
    int started;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    Job **q;
    int qn;
    int running;          // nrun as published for /health
    int stop;
    double grace_until;
    unsigned long long passes, rows;
    // the disk tier (EngineOpts.kv_disk): the directory, its lock, the saved sequences, their bytes and the budget; the
    // I/O admission planned (under mu: admit), done by the engine thread without mu (dk_run)
    char *dk_dir;
    int dk_lock, ndk, dk_cap, dk_next;
    DiskEnt *dk;
    long long dk_bytes;
    double dk_max;
    struct { int spill, slot, load, L, Lv; } plan; // (slots / entry id, -1: none; Lv: the saved tokens' common prefix with the prompt)
    unsigned long long dk_saves, dk_loads;
    double dk_save_s, dk_load_s;
    int *rv;              // --admit bound: the idle slots a reclaim may take (nslot)
    double job_max;       // the lease of a -c job that speculates with the defaults (the startup's check)
    // step 3's live budget: MemAvailable as last read (-1: none, then the static budget only), when; the test hook's file
    // (RUNQ_TEST_MEMAVAILABLE: bytes, read at each use); the last warning; /health's snapshot (under mu)
    double ma, ma_t, ma_warn, ma_res; // (ma_res: the slots' resident memory at that reading: mem_avail)
    const char *ma_file;
    int kcap;             // --admit bound, Qwen3.5 / 3.8: a request's guesses at most (engine_submit; the most one -c job holds)
    Job *mwj;             // the job refused for memory in this step's admissions (NULL: none; only compared)
    EngineMem pub;
    // --admit chunk (step 4): the jobs parked (mem_renewals; no admission while > 0), the jobs started (Run.id), the
    // last warning of a renewal over the budget
    int npark, njob;
    double ow_warn;
    // the swap (--admit chunk, --kv-disk DIR: dk_dir/swap): the jobs to swap out in this step and the one to swap in (into
    // slot swin_si), planned under mu (mem_renewals), done by sw_run without it; the jobs swapped out, their files' bytes
    // (outside --kv-disk-max), the swaps, RUNQ_TEST_SWAP_DELAY (s: a sleep in each swap-out; tests only)
    char *sw_dir;
    Run **swo, *swin;
    int nswo, swin_si, nswap;
    double sw_bytes, sw_delay;
    unsigned long long swaps;
    const char *sw_fail;  // RUNQ_TEST_SWAP_FAIL=FILE (tests only): "draft-save" / "draft-load" in it make the draft's swap I/O fail
    // --admit-yield (mem_renewals): e->run's capacity (parallel + queue: yielded jobs hold no seat), the jobs yielded (no
    // seat; nrun - nyld seats taken), the earliest yield of those still waiting (0: none; the queue's younger jobs wait
    // behind it), a yielded job first for a seat but not back yet (admissions wait)
    int rcap, nyld, yblock;
    double yt;
};

// -------------------------------------------------------------------------------------------- jobs

typedef struct { pthread_mutex_t mu; pthread_cond_t cv; } JobSync;

Job *job_new(void) {
    Job *j = calloc(1, sizeof(Job));
    JobSync *s = calloc(1, sizeof(JobSync));
    if (!j || !s) {
        free(j);
        free(s);
        return NULL;
    }
    pthread_mutex_init(&s->mu, NULL);
#ifdef __APPLE__
    pthread_cond_init(&s->cv, NULL);  // waits use pthread_cond_timedwait_relative_np
#else
    pthread_condattr_t ca;
    pthread_condattr_init(&ca);
    pthread_condattr_setclock(&ca, CLOCK_MONOTONIC);
    pthread_cond_init(&s->cv, &ca);
    pthread_condattr_destroy(&ca);
#endif
    j->mu = s;
    j->cv = s;
    j->refs = 1;
    j->top_logprobs = -1;
    j->sp_mtp_approx = -1; // (no MTP view: no "mtp_approx"; start_run sets 0 on a --draft mtp server)
    return j;
}

void job_lock(Job *j) { pthread_mutex_lock(&((JobSync *)j->mu)->mu); }
void job_unlock(Job *j) { pthread_mutex_unlock(&((JobSync *)j->mu)->mu); }

int job_wait(Job *j, double seconds) {
    // one wait of at most `seconds` for news: a tick, not a deadline (serve.c's wait_job waits 0.1 s at a
    // time and re-checks the job's state, so spurious wakeups are harmless and nothing accumulates; the
    // request's deadline is checked by the engine against rq_now()). Monotonic on both systems, so a
    // wall-clock change cannot stretch a tick: Linux waits until an absolute CLOCK_MONOTONIC time (the
    // condition's clock, job_new); macOS has no pthread_condattr_setclock (its conditions time out on
    // the wall clock), so it waits a relative time, which the kernel measures on the monotonic mach clock
    JobSync *s = j->mu;
    if (seconds < 0) seconds = 0;
#ifdef __APPLE__
    long long rel = (long long)(seconds * 1e9);
    struct timespec rt = { (time_t)(rel / 1000000000LL), (long)(rel % 1000000000LL) };
    return pthread_cond_timedwait_relative_np(&s->cv, &s->mu, &rt) == 0;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    long long ns = ts.tv_nsec + (long long)(seconds * 1e9);
    ts.tv_sec += ns / 1000000000LL;
    ts.tv_nsec = ns % 1000000000LL;
    return pthread_cond_timedwait(&s->cv, &s->mu, &ts) == 0;
#endif
}

static void job_notify(Job *j) { pthread_cond_broadcast(&((JobSync *)j->mu)->cv); }

void job_release(Job *j) {
    if (!j) return;
    job_lock(j);
    int left = --j->refs;
    job_unlock(j);
    if (left > 0) return;
    JobSync *s = j->mu;
    pthread_mutex_destroy(&s->mu);
    pthread_cond_destroy(&s->cv);
    free(s);
    rq_tokens_free(&j->prompt);
    rq_tokens_free(&j->force);
    for (int i = 0; i < j->nstop; i++) free(j->stop[i]);
    free(j->stop);
    jb_free(&j->content);
    jb_free(&j->reasoning);
    jb_free(&j->logprobs);
    strlist_free(&j->calls);
    free(j);
}

static void job_finish(Job *j, int finish, int status, const char *code, const char *msg) {
    job_lock(j);
    j->finish = finish;
    if (status) {
        j->status = status;
        snprintf(j->errcode, sizeof(j->errcode), "%s", code ? code : "");
        snprintf(j->errmsg, sizeof(j->errmsg), "%s", msg ? msg : "");
    }
    j->state = JOB_DONE;
    j->t_end = rq_now();
    job_notify(j);
    job_unlock(j);
}

// ------------------------------------------------------------------------------------------ output

typedef struct {
    Engine *e;
    Job *j;
    RqDetok dt;
    int mode;           // 0: before the first token (thinking: <think> expected), 1: reasoning, 2: content
    JBuf text;          // content so far (engine side; published up to `sent`)
    size_t sent;
    int maxstop;
    int held_nl;        // reasoning: trailing newlines not published yet
    int lead;           // content after </think>: leading newlines still to strip
    int stopped;        // a stop string matched
    int tstop;          // tool calls: the reply is complete (tscan_push)
    ToolScan ts;        // tool calls (j->tools): the content scanned for <tool_call> blocks before it goes on to `text`
} Out;

static void pub_reasoning(Out *o, const char *s, size_t n) {
    // (strip leading newlines; hold trailing ones until more text follows or the block ends)
    JBuf *r = &o->j->reasoning;
    for (size_t i = 0; i < n; i++) {
        if (s[i] == '\n') {
            if (r->n > 0) o->held_nl++;
            continue;
        }
        for (; o->held_nl > 0; o->held_nl--) jb_raw(r, "\n", 1);
        jb_raw(r, s + i, 1);
    }
}

static void pub_content(Out *o, int final) {
    // publish the content that cannot be the start of a stop string; stop at a matched one
    size_t len = o->text.n, upto = len;
    const char *t = o->text.p;
    if (o->j->nstop > 0 && !o->stopped) {
        size_t best = (size_t)-1;
        for (int i = 0; i < o->j->nstop; i++) {
            size_t sl = strlen(o->j->stop[i]);
            for (size_t p = o->sent; p + sl <= len && p < best; p++)
                if (!memcmp(t + p, o->j->stop[i], sl)) {
                    best = p;
                    break;
                }
        }
        if (best != (size_t)-1) {
            o->stopped = 1;
            upto = best;
            o->text.n = best;
        } else if (!final) {
            upto = len > o->sent + (size_t)(o->maxstop - 1) ? len - (o->maxstop - 1) : o->sent;
            while (upto > o->sent && upto < len && ((unsigned char)t[upto] & 0xc0) == 0x80) upto--; // (whole characters)
        }
    }
    if (upto > o->sent) {
        jb_raw(&o->j->content, t + o->sent, upto - o->sent);
        o->sent = upto;
    }
}

static void out_bytes(Out *o, const char *b, int n) {
    if (n <= 0) return;
    if (o->mode == 1) {
        pub_reasoning(o, b, n);
        return;
    }
    if (o->lead) { // (after </think>: the template's "\n\n" before the answer)
        while (n > 0 && *b == '\n') { b++; n--; }
        if (n == 0) return;
        o->lead = 0;
    }
    if (o->j->tools) { // (the content around the calls goes on; the calls into the job)
        if (!o->tstop) o->tstop = tscan_push(&o->ts, b, (size_t)n, 0, &o->text, &o->j->calls);
    } else
        jb_raw(&o->text, b, n);
    pub_content(o, o->ts.ncalls > 0); // (after a call no more content comes: all of it out, before the call's delta)
}

static void out_token(Out *o, int tok) {
    // a generated token's text, under the job's lock (the connection thread reads content / reasoning)
    Engine *e = o->e;
    char buf[256 + 16];
    if (o->mode == 0) {
        o->mode = 2;
        if (o->j->thinking && tok == e->think_start) {
            o->mode = 1;
            return;
        }
    }
    if (o->mode == 1 && tok == e->think_end) {
        int n = rq_detok_flush(&o->dt, buf, sizeof(buf));
        if (n > 0) pub_reasoning(o, buf, n);
        o->held_nl = 0;
        o->mode = 2;
        o->lead = 1;
        return;
    }
    int len;
    rq_token_bytes(e->m, tok, &len);
    char *big = len + 8 > (int)sizeof(buf) ? malloc(len + 8) : NULL;
    int n = rq_detok_push(e->m, &o->dt, tok, big ? big : buf, big ? (size_t)len + 8 : sizeof(buf));
    out_bytes(o, big ? big : buf, n);
    free(big);
}

static void out_end(Out *o) {
    char buf[16];
    int n = rq_detok_flush(&o->dt, buf, sizeof(buf));
    out_bytes(o, buf, n);
    if (o->j->tools && !o->tstop) o->tstop = tscan_push(&o->ts, NULL, 0, 1, &o->text, &o->j->calls);
    if (!o->stopped) pub_content(o, 1);
}

static void logprob_entry(JBuf *b, const RqModel *m, int tok, float lp) {
    int len;
    const char *s = rq_token_bytes(m, tok, &len);
    jb_cstr(b, "{\"token\": ");
    jb_str(b, s, len);
    jb_printf(b, ", \"logprob\": %.9g, \"bytes\": [", lp);
    for (int i = 0; i < len; i++) jb_printf(b, i ? ", %d" : "%d", (unsigned char)s[i]);
    jb_cstr(b, "]");
}

static void logprobs_add(Engine *e, Job *j, int tok, const float *l, const unsigned char *allow, int forced) {
    // the sampled (or forced) token's log probability in the model's distribution (raw logits, no
    // temperature; with a constraint renormalized over the allowed tokens; a forced token: 0), and
    // the top_logprobs alternatives
    JBuf *b = &j->logprobs;
    int k = j->top_logprobs;
    if (b->n) jb_cstr(b, ", ");
    if (forced) {
        logprob_entry(b, e->m, tok, 0.0f);
        jb_cstr(b, ", \"top_logprobs\": [");
        if (k > 0) {
            logprob_entry(b, e->m, tok, 0.0f);
            jb_cstr(b, "}");
        }
        jb_cstr(b, "]}");
        return;
    }
    float mx = -3e38f;
    for (int i = 0; i < e->V; i++)
        if ((!allow || allow[i]) && l[i] > mx) mx = l[i];
    double sum = 0;
    for (int i = 0; i < e->V; i++)
        if (!allow || allow[i]) sum += exp((double)l[i] - mx);
    float lse = (float)(mx + log(sum));
    int top[20];
    float tv[20];
    int nt = 0;
    for (int i = 0; i < e->V && k > 0; i++) {
        if (allow && !allow[i]) continue;
        if (nt == k && l[i] <= tv[nt - 1]) continue;
        int p = nt < k ? nt++ : k - 1;
        while (p > 0 && tv[p - 1] < l[i]) {
            tv[p] = tv[p - 1];
            top[p] = top[p - 1];
            p--;
        }
        tv[p] = l[i];
        top[p] = i;
    }
    logprob_entry(b, e->m, tok, l[tok] - lse);
    jb_cstr(b, ", \"top_logprobs\": [");
    for (int i = 0; i < nt; i++) {
        if (i) jb_cstr(b, ", ");
        logprob_entry(b, e->m, top[i], tv[i] - lse);
        jb_cstr(b, "}");
    }
    jb_cstr(b, "]}");
}

// ----------------------------------------------------------------------------------------- a run

// a running job: its slot, what it feeds next, its constraint state and its output
struct Run {
    Job *j;
    int si;               // slot
    int phase;            // R_PREFILL: feeding the prompt (+ initial forced tokens); R_DECODE: generating
    RqTokens feed;        // to feed: the prompt's uncached part (+ forced) / the last sampled token (+ forced)
    int fed;              // feed[0..fed) fed so far
    int nforced0;         // initial forced tokens (the end of the prefill's feed)
    int seg;              // rows in this pass (0: none)
    int row;              // its logits row in this pass, or -1
    int nallow;           // allowed tokens set in the slot's mask
    int forcing;          // force_reply: 0 not yet fed, 1 fed whole, 2 fed but cut (max tokens / context)
    int thought;          // the reply's </think> was sampled
    int ckw[4], nckw;     // Qwen3.5: the checkpoints the job wants kept (ck_plan_job; + the reply's </think>, ck_think)
    int done;             // finished: the slot is free, reaped after the step
    ConState cs;
    Out o;
    // speculation (the job's spec != SPEC_OFF): its settings, the lookup index, this pass's source / guesses, the verify
    // segment in flight (its first position, guesses, the row being sampled, the last row's guess accepted), a </think>
    // checkpoint waiting for the verify's truncation, the draft's catch-up tokens, the acceptance seen in this job
    int spec, sk, sl, slmin;
    float stau;
    RqLookup *lk;
    int src, want, nd, mlen, guess[SPEC_GUESS_MAX + 1];
    int vP, vn, vrow, vacc, think_pos;
    int dwant, dlag, dseg, dn, ddone;
    RqTokens dbuf;
    double pm[2], pl[RQ_LOOKUP_LMAX + 1][2];
    int nck0, dnck0;      // Qwen3.5: the checkpoint slots before the job (the speculative rows' are transient)
    // --draft mtp ("the MTP drafter"): a guess's own hidden state h' (2 x dim: the passes' input and output, mcur the
    // input's); this pass's rows in the target's hidden rows (hrow: its first in Engine.H, hp0 its position, hn of them, hk
    // those still in the target after the sampling); the follow passes run so far (mfol) and the rows a failed one left
    // behind (mgap); the follow pass's end (mend), its current segment's rows that took zeros (map), a failure logged (mlog);
    // the rows that took zeros after the job's end was published (mlate: B of its last step, its Job no longer written)
    float *mh;
    int mcur, hrow, hp0, hn, hk, mfol, mgap, mend, map, mlog, mlate;
    // --admit bound ("memory admission"): the positions it can ever write (P), its lease's end (E; P while the chunk is
    // unlimited, step 4 renews it), its guesses per pass at most, the lease's KV and state bytes (target + draft)
    int P, E, kk;
    double lkv, lst;
    // --admit chunk: its number (admission order, the log), parked (a renewal denied: step gives it no rows), since when
    int id, park;
    double t_park;
    // --admit chunk, the swap ("swap-out"): 1 to be swapped out (planned: sw_run), 2 swapped out (si -1: its sequences in
    // DIR/swap/<id>.rqseq, + .draft); since when, their lengths (0: no file), the files' bytes, the RNG, what the slot held,
    // the constraint's mask (the tokens set in the slot's allow: nallow of them)
    int swap, sw_len, sw_dlen;
    double t_swap, sw_res;
    long long sw_bytes;
    unsigned long long rng;
    int *alist;
    // --admit-yield: its seat yielded (it waits for one: no rows; in its slot, or swapped out), since when; n_gen when it
    // started or last resumed (its chunks since)
    int yld, g0;
    double t_yld;
};

static int slot_scratch(Engine *e, Slot *sl, int logprobs) {
    // the slot's mask / list (constrained jobs) and logits copy (logprobs), allocated at first use
    if (!sl->allow && !(sl->allow = calloc(e->V, 1))) return RQ_ERR_NOMEM;
    if (!sl->list && !(sl->list = malloc(e->V * sizeof(int)))) return RQ_ERR_NOMEM;
    if (logprobs && !sl->lcopy && !(sl->lcopy = malloc(e->V * sizeof(float)))) return RQ_ERR_NOMEM;
    return RQ_OK;
}

static void ck_reply_end(Engine *e, Run *r) {
    // Qwen3.5: a checkpoint at the end of a finished reply (the state now: the next request of the conversation starts
    // here, and a request that branches off after it, a retry or another slot's copy, lands here instead of the header),
    // in a free checkpoint slot, else in place of the lowest one the job does not want (ck_plan_job); none if all are
    RqSeq *s = e->slot[r->si].seq;
    int len = rq_seq_len(s), held[RQ_CHECKPOINTS_MAX], nh = rq_seq_checkpoints(s, held, RQ_CHECKPOINTS_MAX);
    if (len < 1) return;
    for (int k = 0; k < nh; k++)
        if (held[k] == len) return;
    if (nh >= e->ckn) {
        int drop = -1;
        for (int k = 0; k < nh && drop < 0; k++) {
            int w = 0;
            for (int i = 0; i < r->nckw; i++) w |= r->ckw[i] == held[k];
            if (!w) drop = held[k];
        }
        if (drop < 0) return;
        rq_seq_checkpoint_drop(s, drop);
    }
    rq_seq_checkpoint_at(s, len);
}

static int verify_done(Engine *e, Run *r, int ended);
static void spec_release(Engine *e, Run *r);

static void sw_path(const Engine *e, int id, int draft, char *buf, size_t n) {
    snprintf(buf, n, "%s/%d%s.rqseq", e->sw_dir, id, draft ? ".draft" : "");
}

static void sw_drop(Engine *e, Run *r) {
    // a swapped-out job's files deleted (swapped in, or the job ended out of memory)
    char p[4096];
    for (int d = 0; d < 2; d++) {
        sw_path(e, r->id, d, p, sizeof(p));
        unlink(p);
    }
    e->sw_bytes -= (double)r->sw_bytes;
    r->sw_bytes = 0;
    r->sw_len = r->sw_dlen = 0;
}

static void end_run(Engine *e, Run *r, int finish, int status, int err, const char *emsg) {
    // the job is over (finish reason, an HTTP status for interruptions, or an RQ_ERR_*): flush its text,
    // publish the result, free its slot (a prefix cache from now on). Swapped out (--admit chunk: no slot, r->si -1): its
    // files deleted
    Job *j = r->j;
    Slot *sl = r->si >= 0 ? &e->slot[r->si] : NULL;
    double now = rq_now();
    if (r->park) { // (--admit chunk: cancelled, timed out or stopped while parked)
        j->paused += now - r->t_park;
        r->park = 0;
    }
    if (r->yld) { // (--admit-yield: while waiting for a seat)
        j->paused += now - r->t_yld;
        r->yld = 0;
    }
    if (!sl) { // (cancelled, timed out or stopped while swapped out: always at a clean point, no verify in flight)
        sw_drop(e, r);
        j->swapped += now - r->t_swap;
        r->swap = 0;
        free(r->alist);
        r->alist = NULL;
        r->nallow = 0;
    } else {
        if (r->vn > 0 && verify_done(e, r, 1) != RQ_OK) { // (the cache can no longer be trusted)
            rq_seq_reset(sl->seq);
            if (!status && err == RQ_OK) err = RQ_ERR_STATE, emsg = "speculative decoding: the rollback failed";
        }
        spec_release(e, r);
        if (e->hybrid && !status && err == RQ_OK) ck_reply_end(e, r);
        for (int i = 0; i < r->nallow; i++) sl->allow[sl->list[i]] = 0;
        r->nallow = 0;
    }
    job_lock(j);
    out_end(&r->o);
    job_unlock(j);
    if (status) {
        job_finish(j, FINISH_NONE, status, status == 499 ? "client_closed" : status == 504 ? "timeout" : "shutting_down",
                   status == 499 ? "the client closed the connection" : status == 504 ? "the request timed out" : "the server is shutting down");
    } else if (err != RQ_OK) {
        int ctxe = err == RQ_ERR_CONTEXT;
        job_finish(j, FINISH_NONE, ctxe ? 400 : 500, ctxe ? "context_length_exceeded" : "internal_error", emsg && emsg[0] ? emsg : rq_strerror(err));
    } else {
        job_finish(j, finish, 0, NULL, NULL);
    }
    r->done = 1;
    if (sl) {
        sl->run = NULL;
        sl->used = now;
    }
}

static int prep(Engine *e, Run *r) {
    // after the prefill or a sampled token, for a constrained job: the forced tokens (fed with its next
    // row), then the mask of its next decision. 0 if the job is over
    Job *j = r->j;
    if (!j->con) return 1;
    Slot *sl = &e->slot[r->si];
    RqTokens forced = { 0 };
    int err = con_forced(&r->cs, e->m, &forced);
    if (err != RQ_OK) {
        rq_tokens_free(&forced);
        end_run(e, r, 0, 0, err, NULL);
        return 0;
    }
    job_lock(j);
    for (int i = 0; i < forced.n && j->n_gen < j->max_new && err == RQ_OK; i++) {
        out_token(&r->o, forced.v[i]);
        if (j->top_logprobs >= 0) logprobs_add(e, j, forced.v[i], NULL, NULL, 1);
        j->n_gen++;
        err = rq_tokens_push(&r->feed, &forced.v[i], 1);
    }
    job_notify(j);
    job_unlock(j);
    rq_tokens_free(&forced);
    if (err != RQ_OK) { end_run(e, r, 0, 0, err, NULL); return 0; }
    if (r->cs.done) { end_run(e, r, FINISH_STOP, 0, RQ_OK, NULL); return 0; }
    if (j->n_gen >= j->max_new) { end_run(e, r, FINISH_LENGTH, 0, RQ_OK, NULL); return 0; }
    if ((err = slot_scratch(e, sl, j->top_logprobs >= 0)) != RQ_OK) { end_run(e, r, 0, 0, err, NULL); return 0; }
    r->nallow = con_mask(&r->cs, e->vocab, sl->allow, sl->list, e->V);
    if (r->nallow == 0) {
        end_run(e, r, 0, 0, RQ_ERR_STATE, "constrained decoding: no token continues the output");
        return 0;
    }
    return 1;
}

static void ck_think(Engine *e, Run *r, int pos) {
    // Qwen3.5: a checkpoint after the reply's </think> (length pos, the token not fed yet): a reply whose content or calls
    // come back differently (a token sampled where the text's tokenization has another) rolls back there, not to the
    // header (the reasoning is most of a reply). In the slot ck_plan_job left for the reply; kept by ck_reply_end.
    // (Speculation: pos may be at or below the length, a verify row's checkpoint: kept as this one)
    if (!e->hybrid || r->nckw >= 4 || pos > rq_seq_capacity(e->slot[r->si].seq)) return;
    if (rq_seq_checkpoint_at(e->slot[r->si].seq, pos) == RQ_OK) r->ckw[r->nckw++] = pos;
}

// ---------------------------------------------------------------------------------------------- speculation
// (S8 / G7; the header comment, Speculation.) A decoding job's pass is [t, g1..gn] (t its sampled token, the guesses
// from the prompt lookup or the draft model) with every row's logits; its rows are sampled in order, in rounds over
// the jobs (rq_sample_batch: one row per job and round, so each sequence's RNG draws in plain decoding's order), each
// row's token through take_token as a plain step's (the output, the stops, the limits); while the token is the next
// row's guess, that row is valid; the first other one is the next t, and the sequence is cut after the row it came
// from (verify_done): the tokens, the RNG draws and the cache are plain decoding's

static int run_len(const Engine *e, const Run *r) {
    // the sequence's length as plain decoding has it at this token: in a verify, the positions up to the row sampled
    return r->vn > 0 ? r->vP + r->vrow + 1 : rq_seq_len(e->slot[r->si].seq);
}

static double acc_p(const double job[2], const double all[2], double prior) {
    // the chance that a guess is accepted ([0] checked, [1] accepted): the job's own count shrunk to the server's, that
    // one shrunk to the prior (8 guesses' worth each)
    double g = (all[1] + 8 * prior) / (all[0] + 8);
    return (job[1] + 8 * g) / (job[0] + 8);
}

static void acc_add(double c[2], int checked, int accepted, double cap) {
    c[0] += checked;
    c[1] += accepted;
    if (c[0] > cap) { // (decayed: the recent guesses count most)
        c[0] /= 2;
        c[1] /= 2;
    }
}

static int verify_done(Engine *e, Run *r, int ended) {
    // the verify segment sampled up to row vrow (its token is the next t, or the job ended at it): the counters, the
    // sequence cut after that row (Qwen3.5: on the row's checkpoint; were it missing, the positions are fed again), and a
    // </think> checkpoint now that the length is plain decoding's
    RqSeq *s = e->slot[r->si].seq;
    Job *j = r->j;
    int acc = r->vrow, checked = acc + (!ended && r->vrow < r->vn), keep = r->vP + r->vrow + 1, src = r->src, err = RQ_OK;
    int *c = src == RQ_SPEC_LOOKUP ? j->sp_lookup : j->sp_model;
    c[0]++;
    c[1] += r->vn;
    c[2] += acc;
    e->sp_guess[src] += r->vn;
    e->sp_acc[src] += acc;
    if (src == RQ_SPEC_LOOKUP) {
        acc_add(r->pl[r->mlen], checked, acc, 1e9);
        acc_add(e->pl[r->mlen], checked, acc, 4096);
    } else {
        acc_add(r->pm, checked, acc, 1e9);
        acc_add(e->pm, checked, acc, 4096);
    }
    r->vn = 0;
    if (keep < rq_seq_len(s)) {
        err = rq_seq_truncate(s, keep);
        int L = rq_seq_len(s);
        if (err == RQ_OK && L < keep) { // (the tokens are still in the sequence's array past its length)
            int n = keep - L, *t = malloc(n * sizeof(int));
            j->sp_refed += n;
            if (!t) err = RQ_ERR_NOMEM;
            else {
                memcpy(t, rq_seq_tokens(s) + L, n * sizeof(int));
                err = rq_prefill(e->c, s, t, n, RQ_LOGITS_NONE, NULL, NULL);
                free(t);
            }
        }
    }
    if (r->think_pos > 0) {
        int p = r->think_pos;
        r->think_pos = 0;
        if (err == RQ_OK) ck_think(e, r, p);
    }
    return err;
}

static void draft_ck_pending(Engine *e, Run *r) {
    // Qwen3.5 draft: the job's wanted checkpoints at or past the draft's length, pending (a truncation forgot them)
    RqSeq *d = e->slot[r->si].dseq;
    for (int i = 0; i < r->nckw; i++)
        if (r->ckw[i] >= rq_seq_len(d) && r->ckw[i] <= rq_seq_capacity(d)) rq_seq_checkpoint_at(d, r->ckw[i]);
}

static void draft_ck_plan(Engine *e, Run *r) {
    // Qwen3.5 draft: the target's planned checkpoints (ck_plan_job) on the draft too, so that the conversation's next
    // request rolls the draft back where it rolls the target back, not to 0; its other ones dropped beyond the budget
    RqSeq *d = e->slot[r->si].dseq;
    int held[RQ_CHECKPOINTS_MAX], nh = rq_seq_checkpoints(d, held, RQ_CHECKPOINTS_MAX), room = e->ckn - r->nckw - 1;
    for (int k = nh - 1; k >= 0; k--) {
        int w = 0;
        for (int i = 0; i < r->nckw; i++) w |= r->ckw[i] == held[k];
        if (w) continue;
        if (room > 0) room--;
        else rq_seq_checkpoint_drop(d, held[k]);
    }
    draft_ck_pending(e, r);
}

static int spec_start(Engine *e, Run *r, int src, int L) {
    // the job's speculation: its settings (none for a constrained or forced reply), the lookup index, Qwen3.5's transient
    // checkpoint slots for the verify rows (the slot's --checkpoints stay as they are), the draft's sequence on the
    // prompt's prefix (the copy source's draft, or its own common prefix with the prompt; an MTP view: mtp_start, for every
    // job)
    Job *j = r->j;
    Slot *sl = &e->slot[r->si];
    r->spec = j->con || j->force.n > 0 ? SPEC_OFF : j->spec;
    r->sk = sl->dseq ? j->spec_k : 0;
    r->sl = j->spec_lookup;
    r->slmin = j->spec_lmin;
    r->stau = j->spec_tau;
    if (r->sk <= 0 && r->sl <= 0) r->spec = SPEC_OFF;
    if (r->spec == SPEC_OFF) return RQ_OK;
    int kk = r->sk > r->sl ? r->sk : r->sl, n = j->prompt.n;
    if (r->sl > 0 && !(r->lk = rq_lookup_new(rq_seq_capacity(sl->seq)))) return RQ_ERR_NOMEM;
    if (e->hybrid) {
        r->nck0 = e->ckn;
        rq_seq_set_checkpoints(sl->seq, e->ckn + kk + 1 < RQ_CHECKPOINTS_MAX ? e->ckn + kk + 1 : RQ_CHECKPOINTS_MAX);
    }
    if (r->sk > 0 && !e->mtp) {
        RqSeq *d = sl->dseq;
        if (src >= 0 && e->slot[src].dseq) {
            int c = rq_seq_common_prefix(e->slot[src].dseq, j->prompt.v, L < n ? L : n);
            if (rq_seq_copy(d, e->slot[src].dseq, c) != RQ_OK) rq_seq_reset(d);
        } else
            rq_seq_truncate(d, rq_seq_keep(d, rq_seq_common_prefix(d, j->prompt.v, n)));
        if (e->dhybrid) {
            r->dnck0 = e->ckn;
            rq_seq_set_checkpoints(d, e->ckn + kk + 1 < RQ_CHECKPOINTS_MAX ? e->ckn + kk + 1 : RQ_CHECKPOINTS_MAX);
            draft_ck_plan(e, r);
        }
    }
    return RQ_OK;
}

static void spec_release(Engine *e, Run *r) {
    // the job is over: the verify rows' checkpoints dropped and the slots' count back (the target's as plain decoding
    // leaves them: the next request lands where it would); the draft's reply-end checkpoint if it holds the whole reply
    Slot *sl = &e->slot[r->si];
    if (r->nck0 > 0) {
        rq_seq_checkpoint_rows_drop(sl->seq);
        rq_seq_set_checkpoints(sl->seq, r->nck0);
        r->nck0 = 0;
    }
    if (r->dnck0 > 0) {
        RqSeq *d = sl->dseq, *s = sl->seq;
        rq_seq_checkpoint_rows_drop(d);
        int len = rq_seq_len(d);
        if (len > 0 && len == rq_seq_len(s) && rq_seq_common_prefix(d, rq_seq_tokens(s), len) == len) {
            int held[RQ_CHECKPOINTS_MAX], nh = rq_seq_checkpoints(d, held, RQ_CHECKPOINTS_MAX), have = 0, drop = -1;
            for (int k = 0; k < nh; k++) have |= held[k] == len;
            for (int k = 0; k < nh && drop < 0 && !have && nh >= e->ckn; k++) {
                int w = 0;
                for (int i = 0; i < r->nckw; i++) w |= r->ckw[i] == held[k];
                if (!w) drop = held[k];
            }
            if (drop >= 0) rq_seq_checkpoint_drop(d, drop);
            if (!have && (nh < e->ckn || drop >= 0)) rq_seq_checkpoint_at(d, len);
        }
        rq_seq_set_checkpoints(d, r->dnck0);
        r->dnck0 = 0;
    }
}

static void mtp_start(Engine *e, Run *r, int src) {
    // --draft mtp ("the MTP drafter"; start_run, every job, before spec_start): the view on the prompt's prefix. Copied from
    // the source slot's view to its common prefix with the prompt (not capped at the target's copy, which a checkpoint may
    // shorten: the kept state comes along if it is below), else its own common prefix with the prompt, in place (also a
    // view a failed load left: dk_load returns before it). Ahead of the target (a hybrid target's restart lower down), its
    // rows wait for the prefill (their h was true); level with it, its first row takes the kept state (an agent's next
    // turn). At most the prompt: within the lease (job_lease's E >= the prompt)
    RqSeq *v = e->slot[r->si].dseq;
    const int *P = r->j->prompt.v;
    int n = r->j->prompt.n;
    if (src >= 0 && e->slot[src].dseq) {
        RqSeq *sv = e->slot[src].dseq;
        if (rq_seq_copy(v, sv, rq_seq_common_prefix(sv, P, n)) != RQ_OK) rq_seq_reset(v);
    } else
        rq_seq_truncate(v, rq_seq_common_prefix(v, P, n));
    r->j->sp_mtp_approx = 0;
}

static void force_feed(Engine *e, Run *r) {
    // force_reply, after the prefill: its tokens are the reply (output as if sampled, fed as one decode segment, in passes
    // of several rows), then the reply ends at the segment's logits row (step) as if the end token had been sampled
    Job *j = r->j;
    RqSeq *s = e->slot[r->si].seq;
    int k = j->force.n, room = rq_seq_capacity(s) - rq_seq_len(s);
    if (k > j->max_new) k = j->max_new;
    if (k > room) k = room;
    job_lock(j);
    for (int i = 0; i < k; i++) out_token(&r->o, j->force.v[i]);
    j->n_gen += k;
    job_notify(j);
    job_unlock(j);
    for (int i = 0; i < k && j->thinking; i++)
        if (j->force.v[i] == e->think_end) {
            ck_think(e, r, rq_seq_len(s) + i + 1);
            break;
        }
    r->forcing = k < j->force.n ? 2 : 1;
    r->feed.n = r->fed = 0;
    int err = rq_tokens_push(&r->feed, j->force.v, k);
    if (err != RQ_OK) end_run(e, r, 0, 0, err, NULL);
    else if (k == 0) end_run(e, r, r->forcing == 2 ? FINISH_LENGTH : FINISH_STOP, 0, RQ_OK, NULL);
}

static void take_token(Engine *e, Run *r, int tok) {
    // a sampled token: output, the end conditions, else it is fed with the job's next row
    Job *j = r->j;
    Slot *sl = &e->slot[r->si];
    if (tok < 0) { end_run(e, r, 0, 0, tok, "sampling failed"); return; }
    job_lock(j);
    int end = j->con ? tok == e->eos : rq_is_end_token(e->m, tok);
    if (!end) {
        if (j->top_logprobs >= 0) logprobs_add(e, j, tok, sl->lcopy, j->con ? sl->allow : NULL, 0);
        out_token(&r->o, tok);
        j->n_gen++;
    }
    job_notify(j);
    job_unlock(j);
    for (int i = 0; i < r->nallow; i++) sl->allow[sl->list[i]] = 0;
    r->nallow = 0;
    if (end) { end_run(e, r, FINISH_STOP, 0, RQ_OK, NULL); return; }
    if (j->con) {
        int len;
        const char *b = rq_token_bytes(e->m, tok, &len);
        if (!con_advance(&r->cs, b, len)) { end_run(e, r, 0, 0, RQ_ERR_STATE, "constrained decoding: a disallowed token"); return; }
        if (r->cs.done) { end_run(e, r, FINISH_STOP, 0, RQ_OK, NULL); return; }
    }
    if (r->o.stopped || r->o.tstop) { end_run(e, r, FINISH_STOP, 0, RQ_OK, NULL); return; }
    if (j->n_gen >= j->max_new || run_len(e, r) >= rq_seq_capacity(sl->seq)) { end_run(e, r, FINISH_LENGTH, 0, RQ_OK, NULL); return; }
    if (tok == e->think_end && j->thinking && !r->thought) { // (the reasoning's end: fed with the next pass)
        r->thought = 1;
        if (r->vn > 0) r->think_pos = run_len(e, r) + 1; // (after the verify's truncation)
        else ck_think(e, r, rq_seq_len(sl->seq) + 1);
    }
    r->vacc = 0;
    if (r->vn > 0) { // (a verify row: the next row's guess accepted, else the sequence cut here)
        if (r->vrow < r->vn && tok == r->feed.v[r->vrow + 1]) {
            r->vacc = 1;
            return;
        }
        int err = verify_done(e, r, 0);
        if (err != RQ_OK) {
            rq_seq_reset(sl->seq);
            end_run(e, r, 0, 0, err, "speculative decoding: the rollback failed");
            return;
        }
    }
    r->feed.n = 0;
    r->fed = 0;
    int err = rq_tokens_push(&r->feed, &tok, 1);
    if (err != RQ_OK) { end_run(e, r, 0, 0, err, NULL); return; }
    prep(e, r);
}

// ------------------------------------------------------------------------------------- admission

static int lcp(const int *a, int na, const int *b, int nb) {
    int k = 0;
    while (k < na && k < nb && a[k] == b[k]) k++;
    return k;
}

static int real_reuse(int L, int n) { return L >= 32 || (L > 0 && 2 * L >= n); }

// ---------------------------------------------------------------------------------------- the disk tier
// --kv-disk DIR: sequences saved to files (rq_seq_save: their KV blocks, tokens, recurrent state and checkpoints; the
// same bits back) in DIR/<model fingerprint>-<KV type>/, one directory per server (a lock file). A slot is saved when a
// job would discard >= --kv-disk-min of its tokens (an LRU eviction, a truncation, a copy over it) and at shutdown,
// once per change of its contents (Slot.gen); a save whose tokens extend an older one's (the same conversation, further
// on) replaces it: one file per conversation. A new job whose prompt shares a longer usable prefix with a saved sequence than with any slot (by a
// real margin: >= --prefix-copy tokens and 1/8 of what is read) gets it loaded into the least recently used idle
// slot first (only the blocks up to that prefix are read), then is admitted as for a cached prefix. Saves and loads
// run between passes, without mu (admit plans them: dk_run), so queueing and /health never wait on the disk; the
// running jobs do (a 6 GiB sequence: ~2.5 s to save, ~1 s to load at ~2.5 / ~6 GB/s). The files survive a
// restart (indexed at the start; their mtime orders them). Budget: --kv-disk-max GiB (default half the free space
// at the start, our files included), and 1 GiB of the file system kept free; the least recently used go first.
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>

#define DK_RESERVE (1024.0 * 1048576.0) // bytes of the file system kept free

static void start_run(Engine *e, Job *j, int si, int src, int L);

static void dk_path(const Engine *e, int id, int draft, char *buf, size_t n) {
    snprintf(buf, n, "%s/%d%s.rqseq", e->dk_dir, id, draft ? ".draft" : "");
}

static double dk_free(const Engine *e) {
    struct statvfs v;
    return statvfs(e->dk_dir, &v) ? -1 : (double)v.f_bavail * (double)v.f_frsize;
}

static void dk_touch(const Engine *e, DiskEnt *d) {
    // (the LRU order across restarts: the mtime)
    char p[4096];
    d->used = rq_now();
    dk_path(e, d->id, 0, p, sizeof(p));
    utimensat(AT_FDCWD, p, NULL, 0);
}

static void dk_remove(Engine *e, int k) {
    char p[4096];
    DiskEnt *d = &e->dk[k];
    dk_path(e, d->id, 0, p, sizeof(p));
    unlink(p);
    dk_path(e, d->id, 1, p, sizeof(p));
    unlink(p);
    e->dk_bytes -= d->bytes;
    free(d->tok);
    e->dk[k] = e->dk[--e->ndk];
}

static int dk_find(const Engine *e, int id) {
    for (int k = 0; k < e->ndk; k++)
        if (e->dk[k].id == id) return k;
    return -1;
}

static int dk_fit(Engine *e, double need, int keep_id) {
    // room for need more bytes (the budget, the file system's reserve): the least recently used entries removed (not
    // keep_id) until it fits; 0 if it cannot
    // (--admit chunk: the budget kept free beyond the reserve for the swap files, sw_open)
    double sw = e->sw_dir && e->o.kv_mem * 1073741824.0 > e->sw_bytes ? e->o.kv_mem * 1073741824.0 - e->sw_bytes : 0;
    for (;;) {
        double fr = dk_free(e);
        if (e->dk_bytes + need <= e->dk_max && (fr < 0 || fr - need >= DK_RESERVE + sw)) return 1;
        int v = -1;
        for (int k = 0; k < e->ndk; k++)
            if (e->dk[k].id != keep_id && (v < 0 || e->dk[k].used < e->dk[v].used)) v = k;
        if (v < 0) return 0;
        if (e->o.verbose) fprintf(stderr, "[engine] disk: entry %d removed (%d tokens, %.1f MB): room\n", e->dk[v].id, e->dk[v].f.len, e->dk[v].bytes / 1e6);
        dk_remove(e, v);
    }
}

static int dk_add(Engine *e, int id, double used) {
    // index <id>.rqseq (and its draft's file); its position in e->dk, or -1 (the files removed if they do not check out;
    // kept, not indexed, after a read error)
    char p[4096];
    DiskEnt d = { id, 0, NULL, { 0 }, 0, used };
    dk_path(e, id, 0, p, sizeof(p));
    int rc = rq_seq_file_info(e->m, p, &d.f, &d.tok);
    if (rc != RQ_OK && rc != RQ_ERR_MISMATCH) {
        fprintf(stderr, "runq serve: --kv-disk: %s: %s, skipped\n", p, rq_strerror(rc));
        free(d.tok);
        if (id >= e->dk_next) e->dk_next = id + 1; // (its id stays taken)
        return -1;
    }
    if (rc != RQ_OK || d.f.kv_type != e->o.kv_type) {
        fprintf(stderr, "runq serve: --kv-disk: %s: %s, removed\n", p, rc != RQ_OK ? rq_strerror(rc) : "another KV type");
        free(d.tok);
        unlink(p);
        dk_path(e, id, 1, p, sizeof(p));
        unlink(p);
        return -1;
    }
    d.bytes = d.f.bytes;
    dk_path(e, id, 1, p, sizeof(p));
    struct stat st;
    if (!stat(p, &st)) { // the draft's: used if it is the draft model's and follows these tokens; kept (counted) without one;
        // else removed, said (e.g. another draft model's files when a server switches to --draft mtp: the targets stay)
        RqSeqFile df;
        int *dt = NULL, drc = e->dm ? rq_seq_file_info(e->dm, p, &df, &dt) : RQ_OK;
        const char *why = !e->dm ? NULL : drc != RQ_OK ? rq_strerror(drc) : df.kv_type != e->o.kv_type ? "another KV type"
                          : df.len > d.f.len || memcmp(dt, d.tok, (size_t)df.len * sizeof(int)) ? "other tokens" : NULL;
        if (e->dm && !why)
            d.dlen = df.len;
        else if (e->dm) {
            fprintf(stderr, "runq serve: --kv-disk: %s: not the draft's (%s), removed\n", p, why);
            unlink(p);
            st.st_size = 0;
        }
        free(dt);
        d.bytes += st.st_size;
    }
    if (e->ndk == e->dk_cap) {
        int nc = e->dk_cap ? 2 * e->dk_cap : 16;
        DiskEnt *n = realloc(e->dk, (size_t)nc * sizeof(DiskEnt));
        if (!n) {
            free(d.tok);
            return -1;
        }
        e->dk = n;
        e->dk_cap = nc;
    }
    e->dk[e->ndk] = d;
    e->dk_bytes += d.bytes;
    if (id >= e->dk_next) e->dk_next = id + 1;
    return e->ndk++;
}

static int sw_open(Engine *e, char *err, size_t errlen) {
    // --admit chunk: the swap's directory dk_dir/swap/ (under the disk tier's lock: one server per model and KV type),
    // emptied (a swapped-out job does not outlive the server: its client is gone); the free space must hold the budget (at
    // worst every job but one is swapped out; the disk tier's files count as free: dk_fit gives the swap that room) + the
    // disk tier's reserve (DK_RESERVE: dk_fit keeps it free beyond the swap's room)
    size_t n = strlen(e->dk_dir) + 8;
    if (!(e->sw_dir = malloc(n))) { snprintf(err, errlen, "out of memory"); return -1; }
    snprintf(e->sw_dir, n, "%s/swap", e->dk_dir);
    if (mkdir(e->sw_dir, 0755) && errno != EEXIST) {
        snprintf(err, errlen, "--admit chunk: cannot create %s: %s", e->sw_dir, strerror(errno));
        return -1;
    }
    DIR *dir = opendir(e->sw_dir);
    if (!dir) {
        snprintf(err, errlen, "--admit chunk: cannot read %s: %s", e->sw_dir, strerror(errno));
        return -1;
    }
    char p[4096];
    struct dirent *de;
    int nf = 0;
    while ((de = readdir(dir)))
        if (strcmp(de->d_name, ".") && strcmp(de->d_name, "..")) {
            snprintf(p, sizeof(p), "%s/%s", e->sw_dir, de->d_name);
            nf += !unlink(p);
        }
    closedir(dir);
    if (nf) fprintf(stderr, "runq serve: --admit chunk: %d leftover swap file(s) removed from %s\n", nf, e->sw_dir);
    double fr = dk_free(e), B = e->o.kv_mem * 1073741824.0;
    if (fr >= 0 && fr + (double)e->dk_bytes < B + DK_RESERVE) {
        snprintf(err, errlen, "--admit chunk: the swap needs the memory budget's %.2f GiB + %.0f GiB kept free under %s, it has %.2f GiB "
                 "(with the disk tier's files); free space or lower --kv-mem", B / 1073741824.0, DK_RESERVE / 1073741824.0, e->sw_dir,
                 (fr + (double)e->dk_bytes) / 1073741824.0);
        return -1;
    }
    return 0;
}

static int dk_open(Engine *e, char *err, size_t errlen) {
    // the directory (created), its lock, the index of the files there (leftover .tmp files and others' removed)
    RqModelInfo mi;
    rq_model_info(e->m, &mi);
    size_t n = strlen(e->o.kv_disk) + 64;
    if (!(e->dk_dir = malloc(n))) { snprintf(err, errlen, "out of memory"); return -1; }
    snprintf(e->dk_dir, n, "%s/%016llx-%s", e->o.kv_disk, mi.fingerprint, rq_kv_name(e->o.kv_type));
    if ((mkdir(e->o.kv_disk, 0755) && errno != EEXIST) || (mkdir(e->dk_dir, 0755) && errno != EEXIST)) {
        snprintf(err, errlen, "--kv-disk: cannot create %s: %s", e->dk_dir, strerror(errno));
        return -1;
    }
    char p[4096];
    snprintf(p, sizeof(p), "%s/lock", e->dk_dir);
    if ((e->dk_lock = open(p, O_RDWR | O_CREAT | O_CLOEXEC, 0644)) < 0 || flock(e->dk_lock, LOCK_EX | LOCK_NB)) {
        snprintf(err, errlen, "--kv-disk: %s: %s", p, e->dk_lock < 0 ? strerror(errno) : "in use by another runq serve");
        return -1;
    }
    DIR *dir = opendir(e->dk_dir);
    if (!dir) {
        snprintf(err, errlen, "--kv-disk: cannot read %s: %s", e->dk_dir, strerror(errno));
        return -1;
    }
    double now = rq_now(), wall = (double)time(NULL);
    struct dirent *de;
    while ((de = readdir(dir))) {
        const char *s = de->d_name;
        size_t l = strlen(s);
        char *end;
        long id = strtol(s, &end, 10);
        snprintf(p, sizeof(p), "%s/%s", e->dk_dir, s);
        if (l > 4 && !strcmp(s + l - 4, ".tmp")) unlink(p); // (a save cut short)
        else if (end != s && id >= 0 && id < (1L << 30) && !strcmp(end, ".rqseq")) {
            struct stat st;
            double used = !stat(p, &st) ? now - (wall - (double)st.st_mtime) : now - wall;
            dk_add(e, (int)id, used);
        } else if (end != s && !strcmp(end, ".draft.rqseq")) {
            char q[4096];
            snprintf(q, sizeof(q), "%s/%ld.rqseq", e->dk_dir, id);
            if (access(q, F_OK)) unlink(p); // (an orphan)
        }
    }
    closedir(dir);
    if (e->o.admit == ADMIT_CHUNK && sw_open(e, err, errlen)) return -1;
    double fr = dk_free(e);
    e->dk_max = e->o.kv_disk_max > 0 ? e->o.kv_disk_max * 1073741824.0 : fr >= 0 ? 0.5 * (fr + (double)e->dk_bytes) : 16.0 * 1073741824.0;
    dk_fit(e, 0, -1);
    return 0;
}

static void dk_close(Engine *e) {
    for (int k = 0; k < e->ndk; k++)
        free(e->dk[k].tok);
    free(e->dk);
    e->dk = NULL;
    e->ndk = 0;
    if (e->dk_dir && e->dk_lock >= 0) close(e->dk_lock); // (the lock with it)
    free(e->dk_dir);
    e->dk_dir = NULL;
    free(e->sw_dir);
    e->sw_dir = NULL;
}

static int dk_wants(const Engine *e, const Slot *sl, int keep) {
    // a job would keep the first keep positions of the slot: are the others worth saving first?
    int len = rq_seq_len(sl->seq);
    return e->dk_dir && sl->gen != sl->dk_gen && len >= e->o.kv_disk_min && len - keep >= e->o.kv_disk_min;
}

static int dk_best(const Engine *e, const int *P, int n, int *best, int *lv) {
    // the saved sequence with the longest usable prefix of the prompt (as start_run would keep it: up to n - 1;
    // Qwen3.5: a checkpointed length); its length, *best = its index (-1: none), *lv = its tokens' common prefix with
    // the prompt (the MTP view's usable prefix: no state, not capped at the target's)
    int L = 0;
    *best = -1;
    *lv = 0;
    for (int k = 0; k < e->ndk; k++) {
        const DiskEnt *d = &e->dk[k];
        int c = lcp(d->tok, d->f.len, P, n), u = rq_seq_file_keep(&d->f, c < n - 1 ? c : n - 1);
        if (u > L) {
            L = u;
            *best = k;
            *lv = c;
        }
    }
    return L;
}

static void dk_spill(Engine *e, int si, int keep_id) {
    // save slot si (and its draft's sequence), unless a saved sequence holds these tokens already; entries it holds
    // the tokens of are replaced (not keep_id: a load to follow)
    Slot *sl = &e->slot[si];
    RqSeq *s = sl->seq;
    const int *t = rq_seq_tokens(s);
    int len = rq_seq_len(s);
    char p[4096];
    sl->dk_gen = sl->gen; // (tried once per generation, whatever happens)
    if (len < 1) return;
    for (int k = 0; k < e->ndk; k++) {
        DiskEnt *d = &e->dk[k];
        if (rq_seq_file_keep(&d->f, len) == len && lcp(d->tok, d->f.len, t, len) == len) {
            dk_touch(e, d);
            return;
        }
    }
    int dl = 0;
    if (sl->dseq && (dl = rq_seq_len(sl->dseq)) > 0 && rq_seq_common_prefix(sl->dseq, t, len) != dl) dl = 0;
    double need = (double)rq_seq_save_bytes(s) + (dl ? (double)rq_seq_save_bytes(sl->dseq) : 0);
    if (!dk_fit(e, need, keep_id)) {
        if (e->o.verbose) fprintf(stderr, "[engine] disk: slot %d (%d tokens, %.1f MB) not saved: no room\n", si, len, need / 1e6);
        return;
    }
    int id = e->dk_next++;
    double t0 = rq_now();
    dk_path(e, id, 0, p, sizeof(p));
    int rc = rq_seq_save(s, p);
    if (rc != RQ_OK) {
        fprintf(stderr, "[engine] disk: slot %d not saved to %s: %s\n", si, p, rq_strerror(rc));
        return;
    }
    if (dl) {
        dk_path(e, id, 1, p, sizeof(p));
        if (rq_seq_save(sl->dseq, p) != RQ_OK) dl = 0;
    }
    double dt = rq_now() - t0;
    int k = dk_add(e, id, rq_now());
    if (k < 0) return;
    e->dk_saves++;
    e->dk_save_s += dt;
    DiskEnt nd = e->dk[k];
    if (e->o.verbose)
        fprintf(stderr, "[engine] disk: slot %d saved as entry %d: %d tokens%s, %.1f MB, %.3f s (%.2f GB/s); %d entries, %.2f of %.2f GiB\n", si, id,
                len, dl ? " + the draft's" : "", nd.bytes / 1e6, dt, nd.bytes / 1e9 / (dt > 0 ? dt : 1e-9), e->ndk,
                e->dk_bytes / 1073741824.0, e->dk_max / 1073741824.0);
    // the older saves of this conversation (their tokens a prefix of these): one file per conversation. (Qwen3.5: the new
    // file may not restore every length an older one did, e.g. the end of the previous reply, whose checkpoint the slot
    // gave up for the new turn's; the slot made that choice in memory already, and the disk keeps to it)
    for (int j = e->ndk - 1; j >= 0; j--) {
        DiskEnt *d = &e->dk[j];
        if (d->id != id && d->id != keep_id && d->f.len <= len && lcp(d->tok, d->f.len, t, len) == d->f.len) {
            if (e->o.verbose) fprintf(stderr, "[engine] disk: entry %d (%d tokens) replaced by entry %d\n", d->id, d->f.len, id);
            dk_remove(e, j);
        }
    }
}

static void dk_load(Engine *e, int si, int id, int L, int Lv) {
    // slot si := the saved sequence id's first L positions (and its draft's, as far as they go: a draft model's to the
    // target's length; the MTP view's to Lv, the prompt's common prefix, ahead of a target cut back to a checkpoint: its
    // rows' h were the target's, mtp_start keeps them; the whole file: its kept state too, the next row's h)
    int k = dk_find(e, id);
    if (k < 0) return;
    Slot *sl = &e->slot[si];
    DiskEnt *d = &e->dk[k];
    char p[4096];
    double t0 = rq_now();
    dk_path(e, id, 0, p, sizeof(p));
    int rc = rq_seq_load(sl->seq, p, L);
    if (rc != RQ_OK) {
        fprintf(stderr, "[engine] disk: entry %d not loaded: %s (removed)\n", id, rq_strerror(rc));
        dk_remove(e, k);
        if (rc == RQ_ERR_IO) sl->gen++, sl->dk_gen = sl->gen; // (the slot is empty then)
        return;
    }
    sl->gen++;
    sl->dk_gen = sl->gen;
    sl->used = rq_now();
    if (sl->dseq) {
        int top = e->mtp ? Lv : rq_seq_len(sl->seq), dn = d->dlen < top ? d->dlen : top;
        dk_path(e, id, 1, p, sizeof(p));
        if (dn < 1 || rq_seq_load(sl->dseq, p, dn) != RQ_OK) rq_seq_reset(sl->dseq);
    }
    double dt = rq_now() - t0;
    e->dk_loads++;
    e->dk_load_s += dt;
    dk_touch(e, d);
    if (e->o.verbose)
        fprintf(stderr, "[engine] disk: entry %d -> slot %d: %d of %d tokens%s%.0d, %.3f s\n", id, si, rq_seq_len(sl->seq), d->f.len,
                sl->dseq && rq_seq_len(sl->dseq) ? " + the draft's " : "", sl->dseq ? rq_seq_len(sl->dseq) : 0, dt);
}

static void dk_run(Engine *e) {
    // (the engine thread, without mu) the I/O admit planned: a spill, then a load into that slot
    if (e->plan.spill >= 0) dk_spill(e, e->plan.spill, e->plan.load);
    if (e->plan.load >= 0) dk_load(e, e->plan.slot, e->plan.load, e->plan.L, e->plan.Lv);
    e->plan.spill = e->plan.load = e->plan.slot = -1;
}

static int dk_go(Engine *e, Job *j, int si, int src, int L) {
    // admit's start_run(e, j, si, src, L), unless the slot's tokens it would discard should be saved first: 2 then
    // (planned; the job is admitted again after dk_run)
    Slot *sl = &e->slot[si];
    int keep = src >= 0 ? 0 : rq_seq_keep(sl->seq, L < j->prompt.n ? L : j->prompt.n - 1);
    if (keep < 0) keep = 0;
    if (dk_wants(e, sl, keep)) {
        e->plan.spill = si;
        return 2;
    }
    start_run(e, j, si, src, L);
    return 1;
}

// --------------------------------------------------------------------------------------- memory admission
// --admit bound (docs/memory-admission.md, steps 1-2): a job is admitted on its lease, the memory it can ever use: the
// KV of positions 0..E-1 (E = P = prompt + max tokens: serve.c clamps max_new to the context, spec_cands the guesses
// to max_new, so nothing is written past P) and its state copies (Qwen3.5: the live one, --checkpoints, and kk + 1
// transient ones for the verify rows if it speculates), target + draft. A slot's res: what its sequences hold
// (rq_seq_resident: the KV blocks up to the high-water mark, the state buffers; an upper bound but for THP). committed =
// the sum over the slots of max(res, lease) (KV and state each) if running, else res; admissions keep it <= --kv-mem,
// and between them an idle slot never grows, a running one never past its lease (start_run releases it past E).
// committed is recomputed from the slots at each admission (a few slots, ~1 us each) rather than kept up to date: every
// path that changes a sequence (start_run's truncate / copy / release, dk_load, the verify rows' transient checkpoints,
// a reclaim, end_run / reap) counts as it is, without bookkeeping of its own to drift. A job that does not fit takes the
// room of idle slots, least recently used first (their caches lost; --kv-disk: saved first), else waits (strict FIFO);
// with nothing running it never waits. Step 3, the live budget: the static check does not see the other processes, so a
// job is admitted only if its growth also fits in what the box has left: growth = lease - res(si) (what start_run's
// release gives back returns to MemAvailable: netted), live = MemAvailable - margin - promised, promised = committed -
// sum of res (what the running jobs may still touch: not out of MemAvailable yet); reclaimed idle slots count for both.
// MemAvailable is read at most every 100 ms (mem_avail). The requests' guesses are capped at what one -c job can hold
// (engine_submit, Qwen3.5 / 3.8: their state copies), so that one -c job always fits. Outputs never depend on any of it: only when a
// job runs, and on which slot
typedef struct { double kv, st; } Mem;

static double mem_read(const char *file) {
    // MemAvailable in bytes (-1: unknown, e.g. not Linux), or the test hook's (RUNQ_TEST_MEMAVAILABLE=FILE: bytes)
    if (!file) return rq_mem_available();
    FILE *f = fopen(file, "r");
    double v;
    if (!f || fscanf(f, "%lf", &v) != 1) v = -1;
    if (f) fclose(f);
    return v;
}

static double mem_avail(Engine *e, double res) {
    // (the engine thread) mem_read at most every 100 ms; the test hook's file at each call. res: the slots' resident memory
    // now; between readings what they touched (or gave back) since is taken out of (put back into) the last reading
    if (e->ma_file) return e->ma = mem_read(e->ma_file);
    double now = rq_now();
    if (e->ma_t == 0 || now - e->ma_t >= 0.1) {
        e->ma = mem_read(NULL);
        e->ma_t = now;
        e->ma_res = res;
    }
    return e->ma < 0 ? -1 : e->ma + e->ma_res - res;
}

static int job_kk(const Engine *e, const Job *j, int *dk) {
    // spec_start's rule: the job's guesses per pass at most (0: it does not speculate); *dk: the draft model's among them
    int sk = e->dm ? j->spec_k : 0, sl = j->spec_lookup;
    *dk = 0;
    if (j->con || j->force.n > 0 || j->spec == SPEC_OFF || (sk <= 0 && sl <= 0)) return 0;
    *dk = sk > 0;
    return sk > sl ? sk : sl;
}

static double kv_at(const Slot *sl, int n) {
    // the KV bytes of positions 0..n-1, target + draft (n 0: the token arrays)
    return (double)rq_seq_kv_bytes(sl->seq, n) + (sl->dseq ? (double)rq_seq_kv_bytes(sl->dseq, n) : 0);
}

static Mem job_lease(const Engine *e, const Job *j, int *P, int *E, int *kk) {
    // the job's first lease (to E: P while the chunk is unlimited, --admit chunk: min(P, prompt + C)) and P, kk
    const Slot *sl = &e->slot[0];
    long long p = (long long)j->prompt.n + (j->max_new > 0 ? j->max_new : 0), cap = rq_seq_capacity(sl->seq);
    int dk, k = job_kk(e, j, &dk), ck = e->ckn + (k > 0 ? k + 1 : 0), dck = e->ckn + (dk ? k + 1 : 0);
    if (p > cap) p = cap;
    long long x = e->o.admit == ADMIT_CHUNK && (long long)j->prompt.n + e->o.admit_chunk < p ? (long long)j->prompt.n + e->o.admit_chunk : p;
    if (ck > RQ_CHECKPOINTS_MAX) ck = RQ_CHECKPOINTS_MAX; // (spec_start's rq_seq_set_checkpoints)
    if (dck > RQ_CHECKPOINTS_MAX) dck = RQ_CHECKPOINTS_MAX;
    RqSeqMem t, d = { 0 };
    rq_seq_resident(sl->seq, &t);
    if (sl->dseq) rq_seq_resident(sl->dseq, &d);
    *P = (int)p;
    *E = (int)x;
    *kk = k;
    return (Mem){ kv_at(sl, (int)x), (1.0 + ck) * (double)t.state + (1.0 + dck) * (double)d.state };
}

static Mem slot_res(const Slot *sl) {
    RqSeqMem a, b = { 0 };
    rq_seq_resident(sl->seq, &a);
    if (sl->dseq) rq_seq_resident(sl->dseq, &b);
    return (Mem){ (double)a.kv + (double)b.kv, (double)(a.total - a.kv) + (double)(b.total - b.kv) };
}

static double slot_mem(const Slot *sl) {
    // the slot's part of committed
    Mem m = slot_res(sl);
    if (sl->run) {
        if (sl->run->lkv > m.kv) m.kv = sl->run->lkv;
        if (sl->run->lst > m.st) m.st = sl->run->lst;
    }
    return m.kv + m.st;
}

static double slot_floor(const Slot *sl) {
    // what an emptied and released slot still holds (its token arrays)
    return kv_at(sl, 0);
}

static void slot_reclaim(Engine *e, int si) {
    // an idle slot's memory given back: its cache lost (on disk if dk_spill saved it)
    Slot *sl = &e->slot[si];
    int len = rq_seq_len(sl->seq), disk = 0;
    double g = slot_mem(sl) - slot_floor(sl);
    for (int k = 0; k < e->ndk && len > 0 && !disk; k++)
        disk = rq_seq_file_keep(&e->dk[k].f, len) == len && lcp(e->dk[k].tok, e->dk[k].f.len, rq_seq_tokens(sl->seq), len) == len;
    rq_seq_reset(sl->seq);
    rq_seq_release(sl->seq, 0);
    if (sl->dseq) {
        rq_seq_reset(sl->dseq);
        rq_seq_release(sl->dseq, 0);
    }
    if (e->o.verbose)
        fprintf(stderr, "[engine] slot %d reclaimed: %d tokens, %.2f GiB%s\n", si, len, g / 1073741824.0,
                disk ? " (saved to disk)" : e->dk_dir ? " (not on disk)" : "");
}

static int mem_reclaim(Engine *e, int nv, double need, double grow, double B, double lv, double *acc) {
    // idle slots e->rv[0..nv) reclaimed, least recently used first, until need - *acc <= B and grow - *acc <= lv (*acc: what
    // they gave back); 2 if one is to be saved first (e->plan.spill: dk_run, then the caller again), else 1
    for (int a = 1; a < nv; a++)
        for (int b = a; b > 0 && e->slot[e->rv[b]].used < e->slot[e->rv[b - 1]].used; b--) {
            int x = e->rv[b];
            e->rv[b] = e->rv[b - 1];
            e->rv[b - 1] = x;
        }
    for (int k = 0; k < nv && !(need - *acc <= B && grow - *acc <= lv); k++) {
        Slot *sl = &e->slot[e->rv[k]];
        if (dk_wants(e, sl, 0)) {
            e->plan.spill = e->rv[k];
            return 2;
        }
        *acc += slot_mem(sl) - slot_floor(sl);
        slot_reclaim(e, e->rv[k]);
    }
    return 1;
}

static int mem_admit(Engine *e, Job *j, int si, int src, int dry) {
    // (admit) room for job j on idle slot si (copying from slot src, or -1): 1 it fits (idle slots reclaimed if need be),
    // 2 a reclaim's save planned first (e->plan.spill: dk_run, then the job again), -1 it waits. Computed first, done only
    // if it makes the job fit: a job that waits anyway destroys no cache. dry: only whether it fits with every idle slot
    // reclaimed (1 / -1), nothing done. --admit chunk: none while a job is parked or swapped out, and one chunk per running
    // job (this one included) stays free after it (rc), so that their renewals do not park at once
    if (e->o.admit == ADMIT_WORST) return 1;
    int P, E, kk, live = 0, nv = 0, np = e->o.admit == ADMIT_CHUNK && (e->npark > 0 || e->nswap > 0);
    Mem l = job_lease(e, j, &P, &E, &kk);
    double G = 1073741824.0, B = e->o.kv_mem * G, com = 0, res = 0, gain = 0, own = 0, M = e->o.admit_margin * G, rc = 0;
    for (int i = 0; i < e->nslot; i++) {
        Slot *sl = &e->slot[i];
        Mem r = slot_res(sl);
        double m = slot_mem(sl), f = slot_floor(sl);
        com += m;
        res += r.kv + r.st;
        if (i == si) own = m; // (start_run releases it past E: its old contents do not count)
        if (sl->run) live++;
        else if (i != si && i != src && m > f) {
            gain += m - f;
            e->rv[nv++] = i;
        }
    }
    if (e->o.admit == ADMIT_CHUNK) rc = (live + 1) * (kv_at(&e->slot[0], e->o.admit_chunk) - slot_floor(&e->slot[0]));
    // the static check (need <= B) and the live one (grow <= lv), each with g bytes of idle slots reclaimed
    double A = mem_avail(e, res), need = com - own + l.kv + l.st, grow = l.kv + l.st - own, lv = A >= 0 ? A - M - (com - res) : 1e300, acc = 0;
#define MEM_FITS(g) (!np && need + rc - (g) <= B && grow + rc - (g) <= lv)
    if (MEM_FITS(0)) return 1;
    if (!MEM_FITS(gain) && (live > 0 || dry || np)) { // (np: also with every job swapped out, none in a slot)
        if (!dry && !j->mem_waited && e->o.verbose) {
            char lt[160] = "", ct[96] = "";
            if (A >= 0)
                snprintf(lt, sizeof(lt), "; live %.2f: MemAvailable %.2f - margin %.2f - promised %.2f, it grows %.2f", (lv + gain) / G, A / G,
                         M / G, (com - res) / G, grow / G);
            if (e->o.admit == ADMIT_CHUNK) snprintf(ct, sizeof(ct), "; a chunk each kept free %.2f, %d parked, %d swapped out", rc / G, e->npark, e->nswap);
            fprintf(stderr, "[engine] job waits for memory: needs %.2f GiB (%d + %d positions%s), %.2f free with the idle slots reclaimed "
                    "(budget %.2f, committed %.2f, %d running%s%s)\n", (l.kv + l.st) / G, j->prompt.n, E - j->prompt.n, kk ? ", speculating" : "",
                    (B - com + own + gain) / G, B / G, com / G, live, ct, lt);
        }
        if (!dry) {
            j->mem_waited = 1;
            if (j->t_mw == 0) j->t_mw = rq_now();
            e->mwj = j;
        }
        return -1;
    }
    if (dry) return 1;
    if (mem_reclaim(e, nv, need + rc, grow + rc, B, lv, &acc) == 2) return 2;
    if (need - acc > B) // (nothing running: the startup made sure one -c job fits, guesses capped to fit: engine_submit)
        fprintf(stderr, "[engine] warning: a job admitted over the memory budget (nothing running): %.2f of %.2f GiB\n", (need - acc) / G, B / G);
    else if (grow - acc > lv) // (step 3: the box is short of memory, but one job must run)
        fprintf(stderr, "[engine] warning: a job admitted beyond the live budget (nothing running): it grows %.2f GiB, MemAvailable %.2f - "
                "margin %.2f\n", (grow - acc) / G, A / G, M / G);
#undef MEM_FITS
    return 1;
}

static int run_reach(const Engine *e, const Run *r) {
    // --admit chunk: the length the job's next passes could reach without a renewal check between them: the rest of its
    // feed (a prompt's prefill whole, its forced tokens, a forced reply: never parked half-way) + its guesses per pass at
    // most if this pass may speculate (step's rule: one token to feed, no forced reply; spec_cands caps them by the context
    // and max tokens, so never past P)
    int h = rq_seq_len(e->slot[r->si].seq) + r->feed.n - r->fed;
    if (r->phase == R_DECODE && r->spec != SPEC_OFF && !r->forcing && r->feed.n - r->fed == 1) h += r->sk > r->sl ? r->sk : r->sl;
    return h < r->P ? h : r->P;
}

static int run_held(const Engine *e, const Run *r) {
    // (step) no rows for the job in this pass: swapped out, parked, its seat yielded, or its next pass could write past its
    // lease (a renewal first)
    return e->o.admit == ADMIT_CHUNK && (r->swap || r->park || r->yld || run_reach(e, r) > r->E);
}

static int mem_swap_out(Engine *e, const Run *old, double *acc, int yonly) {
    // (under mu: mem_renew, mem_swap_in, mem_resume, engine_main) plan the swap-out of the youngest job in a slot but old,
    // parked or yielded ones first, then running ones (yonly: yielded ones only); between steps every job is at a clean
    // point (no verify segment in flight, r->vn 0); *acc += its slot's part of committed (its lease leaves it: sw_out gives
    // the slot's memory back, also if the save fails). 1, 0 if there is none left
    Run *v = NULL;
    for (int pass = 0; pass < 2 - yonly && !v; pass++)
        for (int i = e->nrun - 1; i >= 0 && !v; i--) {
            Run *r = e->run[i];
            if (r != old && !r->done && !r->swap && r->si >= 0 && (pass || r->yld || (r->park && !yonly))) v = r;
        }
    if (!v) return 0;
    Slot *sl = &e->slot[v->si];
    *acc += slot_mem(sl) - slot_floor(sl);
    v->swap = 1;
    e->swo[e->nswo++] = v;
    return 1;
}

static int mem_renew(Engine *e, Run *r, int h, int oldest, int behind) {
    // (mem_renewals) --admit chunk: job r's lease extended to cover h positions, by whole chunks: E' = min(P, E + C x n). 1
    // granted, 0 denied (the job parks), 2 a reclaim's save planned first (dk_run; no rows for it in this step, renewed
    // at the next one). No growth (E' in its last KV block): granted. Else denied if behind a parked job (FIFO), else the
    // static check (committed + its growth <= budget) and the live one (growth <= MemAvailable - margin - promised); the
    // oldest job is never denied: the idle slots reclaimed (LRU), then the younger jobs swapped out (mem_swap_out, the
    // youngest first) until it fits; with nothing left to move, granted with a warning (the live budget below one job: one
    // -c job fits the static one, the startup's check). A younger one takes no idle slot's cache: it parks until the memory
    // is there. A slot whose job is planned to be swapped out in this step (the oldest's renewal: r->swap 1) counts as
    // emptied: sw_run gives its memory back before the pass (also if the save fails), so a job between the oldest and the
    // swapped-out ones may grow into it
    Slot *sl = &e->slot[r->si];
    int C = e->o.admit_chunk, nv = 0;
    long long x = r->E + (long long)(h - r->E + C - 1) / C * C;
    int E2 = x < r->P ? (int)x : r->P;
    double G = 1073741824.0, B = e->o.kv_mem * G, M = e->o.admit_margin * G, com = 0, res = 0, gain = 0, kv2 = kv_at(sl, E2);
    for (int i = 0; i < e->nslot; i++) {
        Slot *t = &e->slot[i];
        Mem m = slot_res(t);
        double c = slot_mem(t), f = slot_floor(t);
        if (t->run && t->run->swap == 1) c = f, m = (Mem){ f, 0 };
        com += c;
        res += m.kv + m.st;
        if (!t->run && c > f) {
            gain += c - f;
            e->rv[nv++] = i;
        }
    }
    Mem rs = slot_res(sl);
    double d = (rs.kv > kv2 ? rs.kv : kv2) - (rs.kv > r->lkv ? rs.kv : r->lkv); // (the growth of its part of committed)
    double A = mem_avail(e, res), lv = A >= 0 ? A - M - (com - res) : 1e300, acc = 0;
#define MEM_FITS(g) (d <= 0 || (com + d - (g) <= B && d - (g) <= lv)) // (no growth: within its last KV block, always granted)
    if (!oldest && d > 0 && behind) {
        if (!r->park && e->o.verbose)
            fprintf(stderr, "[engine] job %d parked at %d tokens (%d generated): behind a parked job\n", r->id, rq_seq_len(sl->seq), r->j->n_gen);
        return 0;
    }
    if (!MEM_FITS(0) && !oldest) {
        if (!r->park && e->o.verbose)
            fprintf(stderr, "[engine] job %d parked at %d tokens (%d generated): its lease to %d positions does not fit (+%.3f GiB; budget %.2f, "
                    "committed %.2f, idle %.2f, live %s%.2f)\n", r->id, rq_seq_len(sl->seq), r->j->n_gen, E2, d / G, B / G, com / G, gain / G,
                    A >= 0 ? "" : "none ", A >= 0 ? lv / G : 0.0);
        return 0;
    }
    if (!MEM_FITS(0)) {
        if (mem_reclaim(e, nv, com + d, d, B, lv, &acc) == 2) return 2;
        while (!MEM_FITS(acc) && mem_swap_out(e, r, &acc, 0)) {}
        if (!MEM_FITS(acc) && (e->ow_warn == 0 || rq_now() - e->ow_warn >= 60)) { // (one job must run: the oldest)
            e->ow_warn = rq_now();
            fprintf(stderr, "[engine] warning: the oldest job's lease renewed %s (nothing left to swap out): job %d grows %.3f GiB; "
                    "committed %.2f of %.2f, live %.2f\n", com + d - acc > B ? "over the memory budget" : "beyond the live budget", r->id,
                    d / G, (com + d - acc) / G, B / G, A >= 0 ? (lv + acc) / G : 0.0);
        }
    }
#undef MEM_FITS
    r->E = E2;
    r->lkv = kv2;
    r->j->lease = r->lkv + r->lst;
    r->j->renewals++;
    return 1;
}

static int mem_swap_in(Engine *e, Run *r, int oldest);

static void mem_yield(Engine *e, Run *r, double now, const char *why) {
    // --admit-yield: job r gives its seat up at its lease's end; it keeps its slot and memory until another job needs them
    // (mem_resume, engine_main: swapped out), gets no rows, and is resumed when a seat is its turn (mem_resume)
    if (r->park) {
        r->j->paused += now - r->t_park;
        r->park = 0;
    }
    r->yld = 1;
    r->t_yld = now;
    r->j->yields++;
    if (e->o.verbose)
        fprintf(stderr, "[engine] job %d yielded its seat at %d tokens (%d generated, %d since it resumed): %s\n", r->id,
                rq_seq_len(e->slot[r->si].seq), r->j->n_gen, r->j->n_gen - r->g0, why);
}

static int mem_resume(Engine *e, int nseat, int blocked, int io, double now) {
    // (mem_renewals) --admit-yield: the free seats (nseat taken) to the yielded jobs, FIFO by the time they began to wait,
    // against the queue's head (its submission; not while admissions are held: blocked, a job parked or swapped out for
    // memory). One in its slot resumes at once (its renewal at the next step); a swapped-out one is swapped in (mem_swap_in,
    // one per step, no other I/O planned): the oldest job of all, or any when no other job holding a seat is in a slot,
    // goes in whatever it takes (younger jobs swapped out for it); else, if it does not fit (or no slot is idle), the
    // youngest yielded job in a slot is swapped out for it and it tries again at the next step. Sets e->nyld, e->yt and
    // e->yblock (the first for a seat could not take it: admissions wait). 1 if I/O was planned
    int pl = 0;
    for (;;) {
        Run *w = NULL;
        int first = 1, wold = 0, seated = 0;
        for (int i = 0; i < e->nrun; i++) {
            Run *r = e->run[i];
            if (r->done) continue;
            if (!r->yld) seated += r->si >= 0;
            else if (r->swap != 1 && (!w || r->t_yld < w->t_yld)) w = r, wold = first;
            first = 0;
        }
        if (!w || nseat >= e->parallel || (e->qn > 0 && !blocked && e->q[0]->t_submit < w->t_yld)) break;
        if (w->si >= 0) {
            w->yld = 0;
            w->j->paused += now - w->t_yld;
            w->g0 = w->j->n_gen;
            nseat++;
            if (e->o.verbose)
                fprintf(stderr, "[engine] job %d resumed at %d tokens after %.2f s without a seat\n", w->id, rq_seq_len(e->slot[w->si].seq),
                        now - w->t_yld);
            continue;
        }
        if (io || pl || e->nswo > 0 || e->swin) {
            e->yblock = 1;
            break;
        }
        int g = mem_swap_in(e, w, wold || !seated);
        if (g == 1) { // (sw_in at this step's sw_run; held until then: r->swap)
            w->yld = 0;
            w->j->paused += now - w->t_yld;
            if (e->o.verbose) fprintf(stderr, "[engine] job %d resumed after %.2f s without a seat (swapped in)\n", w->id, now - w->t_yld);
        } else {
            double acc = 0;
            e->yblock = 1;
            if (g == 2 || mem_swap_out(e, w, &acc, 1)) pl = 1;
        }
        break;
    }
    e->nyld = 0;
    e->yt = 0;
    for (int i = 0; i < e->nrun; i++) {
        Run *r = e->run[i];
        if (r->done || !r->yld) continue;
        e->nyld++;
        if (e->yt == 0 || r->t_yld < e->yt) e->yt = r->t_yld;
    }
    return pl;
}

static int mem_renewals(Engine *e) {
    // (engine_main, under mu, before the admissions: a renewal or a resume comes before a new job) --admit chunk (step 4):
    // each running job whose next pass could write past its lease's end renews it, the oldest first (e->run: the admission
    // order). A denied one parks: step gives it no rows, it keeps its slot and memory, and it asks again at every step;
    // once one is denied, the younger ones whose renewal needs memory park too (FIFO). Every job is at a clean point here: a
    // verify segment is sampled whole in its step (r->vn is 0 again at its end), and a prefill or a forced reply is
    // covered whole by run_reach, so a job is held only before a pass, never inside one; and a job that runs has E >=
    // run_reach, which bounds what its pass writes: the decode rows, its guesses (the draft's sequence and the lookup
    // follow the target's tokens: their rows are below that too). A swapped-out job counts as a parked one for the younger
    // ones (FIFO); then the oldest swapped-out job is swapped in if it fits (mem_swap_in: one per step, none while I/O is
    // planned). --admit-yield N: at its lease's end, with every seat taken and someone waiting for one (a yielded job, or
    // the queue while admissions are not held), a job that has generated N chunks since it started or resumed yields its
    // seat (mem_yield), and so does one denied its renewal while a yielded job waits (parked, it would keep the seat that
    // job needs); then the free seats go to the yielded jobs whose turn it is (mem_resume). A yielded job asks for no
    // renewal and holds back no younger one (it waits for a seat, not for memory); the oldest job keeps its precedence for
    // memory (never parked; its renewals and its swap-in move the others out), yielded or not. 1 if I/O was planned (a
    // reclaim's save, swaps: sw_run first; no admission in this step)
    int io = 0, denied = 0, first = 1, out_old = 0, out_behind = 0, yw = e->o.admit_yield > 0, nseat = 0, nyw = 0, blocked = 0;
    Run *out = NULL;
    e->npark = e->nswap = e->nswo = 0;
    e->swin = NULL;
    e->yblock = 0;
    if (e->o.admit != ADMIT_CHUNK) return 0;
    double now = rq_now();
    long long yg = (long long)e->o.admit_yield * e->o.admit_chunk;
    for (int i = 0; i < e->nrun; i++) { // (--admit-yield: the seats taken, the jobs waiting for one, admissions held)
        Run *r = e->run[i];
        if (r->done) continue;
        if (r->yld) nyw++;
        else nseat++, blocked |= r->park || r->swap;
    }
    for (int i = 0; i < e->nrun; i++) {
        Run *r = e->run[i];
        if (r->done) continue;
        int oldest = first, g = 1;
        first = 0;
        if (r->yld) continue; // (waiting for a seat: mem_resume)
        if (r->swap) { // (swapped out, or planned to be in this step by the oldest's renewal)
            e->nswap++;
            if (r->swap == 2 && !out) out = r, out_old = oldest, out_behind = denied; // (behind an older parked job: FIFO, it waits)
            denied = 1;
            continue;
        }
        int h = run_reach(e, r);
        Mem rs = slot_res(&e->slot[r->si]);
        if (rs.kv > r->lkv + 1) // (a pass wrote past E: a bug; the tests look for it)
            fprintf(stderr, "[engine] bug: job %d's slot holds %.0f bytes of KV, its lease %.0f (to %d positions, at %d)\n", r->id, rs.kv,
                    r->lkv, r->E, rq_seq_len(e->slot[r->si].seq));
        if (h > r->E) {
            int full = yw && nseat >= e->parallel, wait = nyw > 0 || (e->qn > 0 && !blocked);
            if (full && wait && r->j->n_gen - r->g0 >= yg) g = 3; // (--admit-yield: its turn is over)
            else if (io) g = 2; // (one save at a time: renewed at the next step)
            else g = mem_renew(e, r, h, oldest, denied);
            if (g == 0 && nyw > 0) g = 3; // (not only when full: parked with a seat free, it could hold the memory the first yielded
                                          // job needs while that one waits for it: nothing runs)
        }
        if (g == 3) {
            mem_yield(e, r, now, nyw > 0 ? "a job without a seat waits" : "a request waits in the queue");
            nseat--;
            nyw++;
        } else if (g == 2) io = 1, e->npark += r->park; // (a parked one stays so until it is renewed)
        else if (g == 0) {
            denied = 1;
            if (!r->park) r->park = 1, r->t_park = now;
            e->npark++;
        } else if (g == 1 && r->park) {
            r->park = 0;
            r->j->paused += now - r->t_park;
            r->g0 = r->j->n_gen;
            if (e->o.verbose)
                fprintf(stderr, "[engine] job %d resumed at %d tokens after %.2f s (its lease to %d positions)\n", r->id,
                        rq_seq_len(e->slot[r->si].seq), now - r->t_park, r->E);
        }
    }
    if (out && !io && e->nswo == 0 && (out_old || !out_behind) && mem_swap_in(e, out, out_old) == 2) io = 1;
    if (yw && mem_resume(e, nseat, e->npark > 0 || e->nswap > 0, io, now)) io = 1;
    return io || e->nswo > 0 || e->swin != NULL;
}

static int spec_ckn(const Engine *e, const Run *r) {
    // spec_start's checkpoint slots of a speculating job (--checkpoints + its guesses + 1)
    int kk = r->sk > r->sl ? r->sk : r->sl, n = e->ckn + kk + 1;
    return n < RQ_CHECKPOINTS_MAX ? n : RQ_CHECKPOINTS_MAX;
}

static void slot_empty(Slot *sl) {
    // an idle slot emptied and its memory given back (no cache: nothing for the disk tier)
    rq_seq_reset(sl->seq);
    rq_seq_release(sl->seq, 0);
    if (sl->dseq) {
        rq_seq_reset(sl->dseq);
        rq_seq_release(sl->dseq, 0);
    }
    sl->gen++;
    sl->dk_gen = sl->gen;
}

// ------------------------------------------------------------------------------------------ the swap
// --admit chunk (docs/memory-admission.md, step 4): when the oldest job's renewal does not fit with the idle slots
// reclaimed, the youngest other jobs are swapped out (parked ones first) until it does: their sequences saved to
// DIR/<model>-<KV type>/swap/<job>.rqseq (+ .draft.rqseq), their slots emptied, the Runs kept detached (r->si -1: no rows,
// the deadline, the cancel and the shutdown still checked). The oldest swapped-out job comes back first, when its lease +
// one chunk fits (or, the oldest of all, by swapping out the younger ones in turn), into an idle slot, and continues from
// its pending token: the same bits (KV, tokens, recurrent state, checkpoints, the RNG), so only when it computes changes.
// Planned under mu (mem_renewals), done by the engine thread between passes without it (sw_run, as dk_run): the HTTP
// threads (SSE keep-alives, /health) never wait on the disk. A swap file that cannot be written or read ends that job
// (500; its files deleted, its slot emptied all the same); no other job is touched. The files are outside --kv-disk-max;
// the disk tier keeps the budget free for them (dk_fit), and DIR/.../swap/ is emptied at the start (sw_open)

static int mem_swap_in(Engine *e, Run *r, int oldest) {
    // (mem_renewals, mem_resume, under mu) room for swapped-out job r: its lease (>= what its slot held; committed counts it
    // whole once it is back) + one chunk (its next renewal) + the oldest job's next renewal (its next chunk's growth, if it
    // is in a slot: so that the oldest's next KV block does not swap r out again at once; a ping-pong guard), the last two
    // not for the oldest job (oldest: it goes in whatever it takes), as an admission's checks, on the idle slot holding the
    // least (emptied for it; saved first if the disk tier wants it). 1 planned (e->swin: sw_run), 0 it stays out, 2 a save
    // planned first (dk_run; again at the next step). The other idle slots are reclaimed (LRU) if that makes it fit; the
    // oldest job also swaps out the younger ones until it fits, and is swapped in with a warning if nothing is left (the live
    // budget below one job: one -c job fits the static budget, the startup's check)
    int si = -1, nv = 0;
    for (int i = 0; i < e->nslot; i++) {
        Slot *t = &e->slot[i];
        if (t->run) continue;
        double m = slot_mem(t) - slot_floor(t), b = si < 0 ? 0 : slot_mem(&e->slot[si]) - slot_floor(&e->slot[si]);
        if (si < 0 || m < b || (m == b && t->used < e->slot[si].used)) si = i;
    }
    if (si < 0) { // (--admit-yield: the slots held by yielded jobs, one swapped out to free one; else cannot be: a swapped-out
        double acc = 0; // job's seat is a slot nobody runs on)
        return mem_swap_out(e, r, &acc, !oldest) ? 2 : 0;
    }
    Slot *sl = &e->slot[si];
    double G = 1073741824.0, B = e->o.kv_mem * G, M = e->o.admit_margin * G, com = 0, res = 0, gain = 0, own = 0, acc = 0;
    for (int i = 0; i < e->nslot; i++) {
        Slot *t = &e->slot[i];
        Mem m = slot_res(t);
        double c = slot_mem(t), f = slot_floor(t);
        com += c;
        res += m.kv + m.st;
        if (i == si) own = c;
        else if (!t->run && c > f) {
            gain += c - f;
            e->rv[nv++] = i;
        }
    }
    double rc = oldest ? 0 : kv_at(sl, e->o.admit_chunk) - slot_floor(sl), l = r->lkv + r->lst;
    for (int i = 0; i < e->nrun && !oldest; i++) { // (the oldest job, if it is in a slot: its next chunk's growth)
        Run *o = e->run[i];
        if (o->done) continue;
        if (o != r && o->si >= 0 && !o->swap) {
            Slot *os = &e->slot[o->si];
            Mem m = slot_res(os);
            long long x = (long long)o->E + e->o.admit_chunk;
            double k2 = kv_at(os, x < o->P ? (int)x : o->P), a = m.kv > k2 ? m.kv : k2, b = m.kv > o->lkv ? m.kv : o->lkv;
            if (a > b) rc += a - b;
        }
        break;
    }
    double need = com - own + l + rc, grow = l + rc - own, A = mem_avail(e, res), lv = A >= 0 ? A - M - (com - res) : 1e300;
#define MEM_FITS(g) (need - (g) <= B && grow - (g) <= lv)
    if (!oldest && !MEM_FITS(gain)) return 0;
    if (dk_wants(e, sl, 0)) {
        e->plan.spill = si;
        return 2;
    }
    if (!MEM_FITS(0)) {
        if (mem_reclaim(e, nv, need, grow, B, lv, &acc) == 2) return 2;
        while (oldest && !MEM_FITS(acc) && mem_swap_out(e, r, &acc, 0)) {}
        if (!MEM_FITS(acc) && (e->ow_warn == 0 || rq_now() - e->ow_warn >= 60)) {
            e->ow_warn = rq_now();
            fprintf(stderr, "[engine] warning: the oldest job (or the only one with a seat in a slot) swapped in %s (nothing left to swap out): job %d, its lease %.3f GiB; "
                    "committed %.2f of %.2f, live %.2f\n", need - acc > B ? "over the memory budget" : "beyond the live budget", r->id,
                    l / G, (need - acc) / G, B / G, A >= 0 ? (lv + acc) / G : 0.0);
        }
    }
#undef MEM_FITS
    if (own > slot_floor(sl)) slot_reclaim(e, si);
    e->swin = r;
    e->swin_si = si;
    return 1;
}

static int sw_failing(const Engine *e, const char *what) {
    // RUNQ_TEST_SWAP_FAIL=FILE (tests only): whether the file names this I/O ("draft-save", "draft-load": it then fails), read
    // at each use
    if (!e->sw_fail) return 0;
    FILE *f = fopen(e->sw_fail, "r");
    char b[256] = "";
    if (f) {
        b[fread(b, 1, sizeof(b) - 1, f)] = 0;
        fclose(f);
    }
    return strstr(b, what) != NULL;
}

static void sw_out(Engine *e, Run *r) {
    // (sw_run) job r out of memory: the verify rows' transient checkpoints dropped (as spec_release; the regular ones go
    // into the file), its sequences saved (the draft's only if it uses it; an MTP view always), its slot emptied and given
    // back, the Run kept (si -1) with its RNG (also in the file; kept for a job not prefilled yet: no file) and its
    // constraint's mask. A draft that cannot be saved is reset at the swap-in (its guesses only change the speed; an MTP
    // view's rows then catch up as a gap, zeros counted); a target that cannot be
    // saved (the disk full, an I/O error) ends the job with a 500, its slot emptied all the same (the renewal that asked
    // for the swap counted on that memory)
    Slot *sl = &e->slot[r->si];
    Job *j = r->j;
    double t0 = rq_now();
    char p[4096];
    int rc = RQ_OK;
    if (r->park) {
        j->paused += t0 - r->t_park;
        r->park = 0;
    }
    if (r->nck0 > 0) rq_seq_checkpoint_rows_drop(sl->seq);
    if (r->dnck0 > 0) rq_seq_checkpoint_rows_drop(sl->dseq);
    Mem m = slot_res(sl);
    r->sw_res = m.kv + m.st;
    r->rng = *rq_seq_rng(sl->seq);
    r->sw_len = rq_seq_len(sl->seq);
    r->sw_dlen = sl->dseq && (r->sk > 0 || e->mtp) ? rq_seq_len(sl->dseq) : 0; // (an MTP view: every job's, its kept state too)
    r->sw_bytes = 0;
    if (r->nallow > 0) {
        if (!(r->alist = malloc((size_t)r->nallow * sizeof(int)))) rc = RQ_ERR_NOMEM;
        else memcpy(r->alist, sl->list, (size_t)r->nallow * sizeof(int));
    }
    if (e->sw_delay > 0) usleep((useconds_t)(e->sw_delay * 1e6));
    if (rc == RQ_OK && r->sw_len > 0) {
        sw_path(e, r->id, 0, p, sizeof(p));
        if ((rc = rq_seq_save(sl->seq, p)) == RQ_OK) r->sw_bytes += rq_seq_save_bytes(sl->seq);
    }
    if (rc == RQ_OK && r->sw_dlen > 0) {
        sw_path(e, r->id, 1, p, sizeof(p));
        int dr = sw_failing(e, "draft-save") ? RQ_ERR_IO : rq_seq_save(sl->dseq, p);
        if (dr == RQ_OK) r->sw_bytes += rq_seq_save_bytes(sl->dseq);
        else {
            fprintf(stderr, "[engine] job %d: the draft's sequence could not be swapped out (%s): reset at the swap-in (only the speed "
                    "changes)\n", r->id, rq_strerror(dr));
            r->sw_dlen = 0;
        }
    }
    e->sw_bytes += (double)r->sw_bytes;
    if (rc != RQ_OK) {
        fprintf(stderr, "[engine] job %d could not be swapped out to %s: %s (the job fails)\n", r->id, e->sw_dir, rq_strerror(rc));
        free(r->alist);
        r->alist = NULL;
        sw_drop(e, r);
        end_run(e, r, 0, 0, rc, "the request could not be swapped out to disk (--kv-disk)");
        slot_empty(sl);
        return;
    }
    for (int i = 0; i < r->nallow; i++) sl->allow[sl->list[i]] = 0;
    slot_empty(sl);
    if (r->nck0 > 0) rq_seq_set_checkpoints(sl->seq, r->nck0);
    if (r->dnck0 > 0) rq_seq_set_checkpoints(sl->dseq, r->dnck0);
    sl->run = NULL;
    if (e->o.verbose) {
        char pf[64] = "";
        if (r->phase == R_PREFILL) snprintf(pf, sizeof(pf), "; prefilling: %d of %d prompt tokens", r->sw_len, j->prompt.n);
        fprintf(stderr, "[engine] job %d swapped out: %.2f GB, %.2f s (slot %d: %d tokens%s, %.3f GiB given back%s)\n", r->id, r->sw_bytes / 1e9,
                rq_now() - t0, r->si, r->sw_len, r->sw_dlen ? " + the draft's" : "", (r->sw_res - slot_floor(sl)) / 1073741824.0, pf);
    }
    r->si = -1;
    r->swap = 2;
    r->t_swap = t0;
    j->swaps++;
    e->swaps++;
}

static void sw_in(Engine *e, Run *r, int si) {
    // (sw_run) swapped-out job r back into idle slot si (emptied by mem_swap_in): its files loaded (the checkpoint slots it
    // had: the saved checkpoints all come back) and deleted, its pending checkpoints asked for again (rq_seq_load forgets
    // them), its RNG and mask back; it continues from its pending token. A target that cannot be read ends the job with a
    // 500 (its slot emptied); a draft that cannot: the draft is reset
    Slot *sl = &e->slot[si];
    Job *j = r->j;
    double t0 = rq_now();
    char p[4096];
    int rc = RQ_OK;
    rq_seq_reset(sl->seq);
    if (r->nck0 > 0) rq_seq_set_checkpoints(sl->seq, spec_ckn(e, r));
    if (r->sw_len > 0) {
        sw_path(e, r->id, 0, p, sizeof(p));
        rc = rq_seq_load(sl->seq, p, r->sw_len);
        if (rc == RQ_OK && rq_seq_len(sl->seq) != r->sw_len) rc = RQ_ERR_STATE;
    }
    if (sl->dseq) {
        rq_seq_reset(sl->dseq);
        if (r->dnck0 > 0) rq_seq_set_checkpoints(sl->dseq, spec_ckn(e, r));
        sw_path(e, r->id, 1, p, sizeof(p));
        int dr = rc == RQ_OK && r->sw_dlen > 0 ? sw_failing(e, "draft-load") ? RQ_ERR_IO : rq_seq_load(sl->dseq, p, r->sw_dlen) : RQ_OK;
        if (dr != RQ_OK) {
            fprintf(stderr, "[engine] job %d: the draft's sequence could not be swapped back in (%s): reset (only the speed changes)\n", r->id,
                    rq_strerror(dr));
            rq_seq_reset(sl->dseq);
        }
    }
    *rq_seq_rng(sl->seq) = r->rng;
    int len = r->sw_len, dl = sl->dseq ? rq_seq_len(sl->dseq) : 0;
    double now = rq_now();
    sw_drop(e, r);
    r->si = si;
    r->swap = 0;
    r->g0 = j->n_gen;
    sl->run = r;
    sl->used = now;
    sl->gen++;
    job_lock(j);
    j->slot = si;
    j->swapped += now - r->t_swap;
    job_unlock(j);
    if (rc == RQ_OK && e->hybrid)
        for (int i = 0; i < r->nckw; i++)
            if (r->ckw[i] > len && r->ckw[i] <= rq_seq_capacity(sl->seq)) rq_seq_checkpoint_at(sl->seq, r->ckw[i]);
    if (rc == RQ_OK && e->dhybrid && sl->dseq && r->sk > 0) draft_ck_pending(e, r);
    if (rc == RQ_OK && e->mtp) { // (a view reset: the gap's rows take zeros, counted; ahead or level: as it was)
        int Lv = rq_seq_len(sl->dseq), end = len < rq_seq_capacity(sl->dseq) ? len : rq_seq_capacity(sl->dseq);
        r->mgap = Lv < end ? end - Lv : 0;
    }
    if (rc == RQ_OK && r->nallow > 0) {
        if ((rc = slot_scratch(e, sl, j->top_logprobs >= 0)) == RQ_OK)
            for (int i = 0; i < r->nallow; i++) sl->allow[sl->list[i] = r->alist[i]] = 1;
    }
    free(r->alist);
    r->alist = NULL;
    if (rc != RQ_OK) {
        fprintf(stderr, "[engine] job %d could not be swapped back in: %s (the job fails)\n", r->id, rq_strerror(rc));
        r->nallow = 0; // (nothing set in the slot's mask: the load or slot_scratch failed first)
        end_run(e, r, 0, 0, rc, "the request could not be swapped back in from disk (--kv-disk)");
        slot_empty(sl);
        return;
    }
    if (e->o.verbose)
        fprintf(stderr, "[engine] job %d swapped in -> slot %d: %.2f s (%d tokens%s; out %.2f s)\n", r->id, si, now - t0, len,
                dl > 0 ? " + the draft's" : "", now - r->t_swap);
}

static void sw_run(Engine *e) {
    // (the engine thread, without mu, between passes) the I/O mem_renewals / admit planned: the swap-outs (their memory is
    // the room of the renewal or the swap-in), the disk tier's save / load, then the swap-in
    for (int k = 0; k < e->nswo; k++) sw_out(e, e->swo[k]);
    e->nswo = 0;
    dk_run(e);
    if (e->swin) sw_in(e, e->swin, e->swin_si);
    e->swin = NULL;
}

static void mem_publish(Engine *e) {
    // (under mu, the engine thread) /health's memory; the warning when the running jobs' promise no longer fits (step 3:
    // another process took memory; bound never takes it back, chunk denies renewals), at most once a minute
    if (e->o.admit == ADMIT_WORST) return;
    double com = 0, res = 0, M = e->o.admit_margin * 1073741824.0, now = rq_now();
    int np = 0, ns = 0, ny = 0;
    for (int i = 0; i < e->nslot; i++) {
        Mem r = slot_res(&e->slot[i]);
        com += slot_mem(&e->slot[i]);
        res += r.kv + r.st;
    }
    for (int i = 0; i < e->nrun; i++) {
        np += !e->run[i]->done && e->run[i]->park && !e->run[i]->swap; // (a parked one planned to be swapped out: swapped)
        ns += !e->run[i]->done && e->run[i]->swap;
        ny += !e->run[i]->done && e->run[i]->yld;
    }
    double A = mem_avail(e, res);
    e->pub = (EngineMem){ e->o.kv_mem * 1073741824.0, A - M - (com - res), com, res, A >= 0, e->mwj != NULL, e->o.admit == ADMIT_CHUNK, np, ns,
                          e->o.admit_yield > 0, ny };
    if (A >= 0 && e->nrun > 0 && A - M < com - res && (e->ma_warn == 0 || now - e->ma_warn >= 60)) {
        e->ma_warn = now;
        fprintf(stderr, "[engine] warning: MemAvailable %.2f GiB - margin %.2f is below the %.2f GiB the running jobs may still grow "
                "into (another process took memory): new jobs wait\n", A / 1073741824.0, M / 1073741824.0, (com - res) / 1073741824.0);
    }
}

static int go(Engine *e, Job *j, int si, int src, int L) {
    // admit's start: the memory first (--admit bound), then dk_go
    int a = mem_admit(e, j, si, src, 0);
    return a == 1 ? dk_go(e, j, si, src, L) : a;
}

static int prompt_sys_end(const Engine *e, const int *P, int n) {
    // the end of the prompt's leading system block ("<|im_start|>system...<|im_end|>\n": the system prompt and the tools),
    // the position of the next <|im_start|>; -1 if none
    if (n < 3 || P[0] != e->t_im_start || P[1] != e->t_system) return -1;
    for (int i = 2; i < n; i++)
        if (P[i] == e->t_im_start) return i;
    return -1;
}

static int prompt_turn_start(const Engine *e, const int *P, int n) {
    // the first assistant header after the last user query (a user turn that is not "\n<tool_response>..."): Qwen3.5's
    // template drops the reasoning of the turn's replies once the next user query comes, so that request leaves the
    // cache there; -1 if none
    int u = -1;
    for (int i = n - 2; i >= 0 && u < 0; i--)
        if (P[i] == e->t_im_start && P[i + 1] == e->t_user && !(i + 3 < n && P[i + 2] == e->t_nl && P[i + 3] == e->t_toolresp)) u = i;
    for (int i = u + 2; u >= 0 && i + 1 < n; i++)
        if (P[i] == e->t_im_start && P[i + 1] == e->t_assistant) return i;
    return -1;
}

static void ck_plan_job(Engine *e, Run *r, int L) {
    // Qwen3.5: the job's checkpoints, in priority order: the prompt's assistant header (a reply sent back differently from
    // what was generated rolls back there), the end of the system block (every request with this system prompt and these
    // tools shares it: a compaction's next request, a new session), the turn's first reply (Qwen3.5: where the next user
    // query's request leaves the cache); at most --checkpoints of them, one slot left for the reply's end (ck_reply_end).
    // Those at or past L are taken by the passes (or now, at L); of those below L only the held ones remain; the slot's
    // other checkpoints are dropped, the highest kept within the budget, so that the lowest-first replacement of new
    // checkpoints (forward.c) never takes the wanted ones
    RqSeq *s = e->slot[r->si].seq;
    const int *P = r->j->prompt.v;
    int n = r->j->prompt.n, B = e->ckn, held[RQ_CHECKPOINTS_MAX], nh = rq_seq_checkpoints(s, held, RQ_CHECKPOINTS_MAX);
    int cand[3] = { rq_chat_checkpoint_pos(e->m, P, n), prompt_sys_end(e, P, n), e->chat == RQ_CHAT_QWEN35 ? prompt_turn_start(e, P, n) : -1 };
    r->nckw = 0;
    for (int k = 0; k < 3 && r->nckw < B; k++) {
        int p = cand[k], skip = p < 1 || p >= n, h = 0; // (position 0: the empty state)
        for (int i = 0; i < r->nckw && !skip; i++) skip = r->ckw[i] == p;
        for (int i = 0; i < nh; i++) h |= held[i] == p;
        if (!skip && (p >= L || h)) r->ckw[r->nckw++] = p;
    }
    int room = B - r->nckw - 1;
    for (int k = nh - 1; k >= 0; k--) {
        int w = 0;
        for (int i = 0; i < r->nckw; i++) w |= r->ckw[i] == held[k];
        if (w) continue;
        if (room > 0) room--;
        else rq_seq_checkpoint_drop(s, held[k]);
    }
    for (int i = 0; i < r->nckw; i++)
        if (r->ckw[i] >= L) rq_seq_checkpoint_at(s, r->ckw[i]);
}

static void start_run(Engine *e, Job *j, int si, int src, int L) {
    // the job takes slot si with its first L prompt positions (cached in place, or copied from slot src)
    RqModel *m = e->m;
    Slot *sl = &e->slot[si];
    RqSeq *s = sl->seq;
    int n = j->prompt.n, err = RQ_OK;
    double now = rq_now();
    job_lock(j);
    j->state = JOB_RUNNING;
    j->t_start = now;
    j->slot = si;
    if (j->t_mw > 0) j->mem_wait += now - j->t_mw;
    j->t_mw = 0;
    job_unlock(j);
    sl->used = now;
    sl->gen++; // (the disk tier: new tokens from here on)
    Run *r = calloc(1, sizeof(Run));
    if (!r) {
        job_finish(j, FINISH_NONE, 500, "internal_error", "out of memory");
        job_release(j);
        return;
    }
    if (n < 1 || n >= rq_seq_capacity(s)) {
        job_finish(j, FINISH_NONE, 400, "context_length_exceeded", "the prompt does not fit in the context");
        job_release(j);
        free(r);
        return;
    }
    r->j = j;
    r->si = si;
    r->row = -1;
    r->o.e = e;
    r->o.j = j;
    if (j->tools) tscan_init(&r->o.ts, j->tools, j->single_call, j->call_seed);
    if (j->thinking && (e->chat == RQ_CHAT_THINKING || e->chat == RQ_CHAT_QWEN35 || e->chat == RQ_CHAT_QWEN38))
        r->o.mode = 1; // (Thinking-2507, Qwen3.5 / 3.8: the prompt opened the <think> block, the reply starts in the reasoning)
    for (int i = 0; i < j->nstop; i++)
        if ((int)strlen(j->stop[i]) > r->o.maxstop) r->o.maxstop = (int)strlen(j->stop[i]);
    sl->run = r;
    e->run[e->nrun++] = r;
    if (L >= n) L = n - 1; // (the last prompt token is prefilled again: its logits)
    const RqSeq *from = src >= 0 ? e->slot[src].seq : s; // (the stats: what the prompt met there)
    j->n_slot = rq_seq_len(from);
    j->n_common = rq_seq_common_prefix(from, j->prompt.v, n);
    if (j->n_common >= n) j->n_common = n - 1;
    j->copied = src >= 0;
    if (src < 0 || rq_seq_copy(s, e->slot[src].seq, L) != RQ_OK) {
        if (src >= 0) L = 0;
        rq_seq_truncate(s, L);
    }
    // (Qwen3.5: copy / truncate keep a checkpointed length <= L; the checkpoints this conversation's next requests
    // need: ck_plan_job)
    L = rq_seq_len(s);
    if (e->hybrid) ck_plan_job(e, r, L);
    r->id = ++e->njob;
    if (e->mtp) { // (every job: the view follows it whatever it asks for)
        if (!(r->mh = malloc(2 * (size_t)e->dim * sizeof(float)))) {
            end_run(e, r, 0, 0, RQ_ERR_NOMEM, NULL);
            return;
        }
        mtp_start(e, r, src);
    }
    if ((err = spec_start(e, r, src, L)) != RQ_OK) {
        end_run(e, r, 0, 0, err, NULL);
        return;
    }
    if (e->o.admit != ADMIT_WORST) { // the lease; the slot holds no more than it from here on (the draft: not past E either)
        Mem l = job_lease(e, j, &r->P, &r->E, &r->kk);
        j->chunked = e->o.admit == ADMIT_CHUNK;
        r->lkv = l.kv;
        r->lst = l.st;
        j->lease = l.kv + l.st;
        rq_seq_release(s, r->E);
        if (sl->dseq) {
            if (rq_seq_len(sl->dseq) > r->E) // (a job without the draft: spec_start left its sequence as it was)
                rq_seq_truncate(sl->dseq, rq_seq_keep(sl->dseq, rq_seq_common_prefix(sl->dseq, j->prompt.v, n)));
            rq_seq_release(sl->dseq, r->E);
        }
    }
    if (e->o.verbose) {
        char cp[48] = "";
        if (src >= 0) snprintf(cp, sizeof(cp), " (copied from slot %d)", src);
        fprintf(stderr, "[engine] job -> slot %d: %d of %d prompt tokens cached%s, %d running", si, L, n, cp, e->nrun);
        if (e->o.admit == ADMIT_CHUNK) fprintf(stderr, "; job %d, its lease to %d of %d positions", r->id, r->E, r->P);
        if (e->hybrid) {
            int held[RQ_CHECKPOINTS_MAX], nh = rq_seq_checkpoints(s, held, RQ_CHECKPOINTS_MAX);
            fprintf(stderr, "; checkpoints");
            for (int k = 0; k < nh; k++) fprintf(stderr, " %d", held[k]);
            fprintf(stderr, " + at");
            for (int k = 0; k < r->nckw; k++)
                if (r->ckw[k] > L) fprintf(stderr, " %d", r->ckw[k]);
        }
        fprintf(stderr, "\n");
        if (j->n_common < j->n_slot && j->n_common < n - 1) { // (where the prompt left the cached tokens: a few tokens of each side)
            const int *ct = rq_seq_tokens(from);
            fprintf(stderr, "[engine]   diverged at %d of %d cached tokens:", j->n_common, j->n_slot);
            for (int side = 0; side < 2; side++) {
                fprintf(stderr, side ? " | prompt" : " cache");
                for (int k = j->n_common - 2 < 0 ? 0 : j->n_common - 2; k < j->n_common + 4 && k < (side ? n : j->n_slot); k++) {
                    int len;
                    const char *b = rq_token_bytes(m, side ? j->prompt.v[k] : ct[k], &len);
                    fprintf(stderr, " %d:", side ? j->prompt.v[k] : ct[k]);
                    for (int c = 0; c < len && c < 24; c++) {
                        if (b[c] == '\n') fputs("\\n", stderr);
                        else fputc(b[c], stderr);
                    }
                }
            }
            fprintf(stderr, "\n");
        }
    }
    job_lock(j);
    j->n_prompt = n;
    j->n_cached = L;
    job_unlock(j);
    rq_seq_seed(s, j->seed);
    if (j->temperature != 0) { // (rq_generate's draws for a prefill of the whole prompt: xorshift steps)
        unsigned long long *g = rq_seq_rng(s);
        for (int i = 0; i < n - 1; i++) {
            *g ^= *g >> 12;
            *g ^= *g << 25;
            *g ^= *g >> 27;
        }
    }
    // a constrained reply starts with forced tokens (e.g. `{"tier": "`): they go into the prefill
    RqTokens forced = { 0 };
    if (j->con) {
        con_start(&r->cs, j->con);
        if ((err = con_forced(&r->cs, m, &forced)) != RQ_OK) goto out;
        if (r->cs.done) { // (one valid output: no forward pass at all)
            job_lock(j);
            for (int i = 0; i < forced.n && j->n_gen < j->max_new; i++) {
                out_token(&r->o, forced.v[i]);
                if (j->top_logprobs >= 0) logprobs_add(e, j, forced.v[i], NULL, NULL, 1);
                j->n_gen++;
            }
            job_unlock(j);
            end_run(e, r, j->n_gen < forced.n ? FINISH_LENGTH : FINISH_STOP, 0, RQ_OK, NULL);
            rq_tokens_free(&forced);
            return;
        }
        if (forced.n >= j->max_new || n + forced.n > rq_seq_capacity(s))
            forced.n = 0, con_start(&r->cs, j->con); // (rare: no room; the decode steps handle them)
    }
    if ((err = rq_tokens_push(&r->feed, j->prompt.v + L, n - L)) != RQ_OK || (err = rq_tokens_push(&r->feed, forced.v, forced.n)) != RQ_OK)
        goto out;
    r->nforced0 = forced.n;
    r->phase = R_PREFILL;
out:
    rq_tokens_free(&forced);
    if (err != RQ_OK) end_run(e, r, 0, 0, err, NULL);
}

static int admit(Engine *e, Job *j) {
    // (under e->mu) start the job on a slot: 1, or -1 if no slot is idle (--admit bound: or no room for it: go), 0 if it
    // should wait for a prefix that a job being prefilled will share with it, 2 if the disk tier has I/O to do first
    // (e->plan: dk_run, then the job is admitted again)
    const int *P = j->prompt.v;
    int n = j->prompt.n, ib = -1, li = -1, lru = -1, bb = -1, lb = 0;
    for (int i = 0; i < e->nslot; i++) {
        Slot *s = &e->slot[i];
        int L = j->no_cache ? 0 : rq_seq_keep(s->seq, rq_seq_common_prefix(s->seq, P, n)); // (Qwen3.5: a checkpoint)
        if (!s->run) {
            if (L > li || (L == li && s->used > e->slot[ib].used)) { ib = i; li = L; }
            if (lru < 0 || s->used < e->slot[lru].used) lru = i;
        } else if (L > lb) {
            bb = i;
            lb = L;
        }
    }
    if (lru < 0) return -1;
    if (j->no_cache) // (the least recently used slot: keeps the useful caches)
        return go(e, j, lru, -1, 0);
    if (e->ndk && !j->dk_tried) { // a saved sequence shares a longer prefix (by a real margin: it is read whole up to there)
        int k, lv, Ld = dk_best(e, P, n, &k, &lv), have = li > lb ? li : lb, gain = Ld - have;
        if (k >= 0 && real_reuse(Ld, n) && gain >= (e->o.copy_min > 0 ? e->o.copy_min : 32) && 8 * gain >= Ld) {
            int ma = mem_admit(e, j, lru, -1, 0); // (the load fills lru: its room first)
            if (ma != 1) return ma;
            j->dk_tried = 1;
            e->plan.spill = dk_wants(e, &e->slot[lru], 0) ? lru : -1;
            e->plan.slot = lru;
            e->plan.load = e->dk[k].id;
            e->plan.L = Ld;
            e->plan.Lv = lv;
            return 2;
        }
    }
    if (e->o.copy_min > 0) {
        int have = li > lb ? li : lb, pend = 0;
        for (int k = 0; k < e->nrun; k++) {
            Run *r = e->run[k];
            if (r->done || r->phase != R_PREFILL || r->si < 0) continue;
            int sh = lcp(r->j->prompt.v, r->j->prompt.n, P, n);
            if (sh > rq_seq_len(e->slot[r->si].seq) && sh > pend) pend = sh;
        }
        // (Qwen3.5: no waiting for a prefix being prefilled: its state is gone once the prefill moves past it)
        if (!e->hybrid && pend >= have + e->o.copy_min && real_reuse(pend, n)) return 0;
        if (bb >= 0 && lb >= li + e->o.copy_min && real_reuse(lb, n)) return go(e, j, lru, bb, lb);
    }
    if (real_reuse(li, n)) {
        // slot affinity: a slot holding a longer conversation than it would keep (it shares only a prefix, e.g. the system
        // prompt with a new session or a compaction's next request) stays for that conversation's next request; the
        // prefix is copied into the least recently used other idle slot (the same bits, rq_seq_copy)
        int lost = rq_seq_len(e->slot[ib].seq) - li, other = -1;
        for (int i = 0; i < e->nslot; i++)
            if (i != ib && !e->slot[i].run && (other < 0 || e->slot[i].used < e->slot[other].used)) other = i;
        // (--admit bound: unless the copy cannot fit beside its source; then the slot itself)
        if (e->o.copy_min > 0 && other >= 0 && lost >= e->o.copy_min && lost > li && mem_admit(e, j, other, ib, 1) > 0)
            return go(e, j, other, ib, li);
        return go(e, j, ib, -1, li);
    }
    return go(e, j, lru, -1, rq_seq_keep(e->slot[lru].seq, rq_seq_common_prefix(e->slot[lru].seq, P, n)));
}

// ------------------------------------------------------------------------------------------- a step

// the speculation of one pass: a candidate job (decoding, speculating, one token to feed) and its choice
typedef struct Cand {
    Run *r;
    int L;          // its position (the pass's first row)
    int lim;        // guesses at most (its settings, max tokens, the context)
    int nl;         // lookup guesses available (its match), and their chance of acceptance each
    double plk;
    int mk;         // model guesses possible this pass (0: none: no draft, not wanted, or too far behind)
    double pmd;
    int lag;        // the draft's tokens to catch up before t
    int fixed;      // SPEC_FIXED: the choice is the job's settings
    int src, n;     // the choice
} Cand;

static double plan_cost(const Engine *e, const Cand *c, int nc, int rbase, double sbase, int d0, double ds0) {
    // the pass's time in the cost model: the target's rows (the decode rows and prompt chunks: rbase, sbase; plus each
    // choice's guesses), the draft's passes (the first one: the catch-up rows d0 of other jobs, each model job's lag + t;
    // pass p: the jobs guessing at least p tokens, one row each)
    // (positions: each segment's first row's, sbase for the decode rows, and its further rows', the guesses)
    int R = rbase, kmax = 0;
    double S2 = 0, T;
    for (int i = 0; i < nc; i++) {
        if (c[i].src == RQ_SPEC_PLAIN) continue;
        R += c[i].n;
        S2 += c[i].n * (double)c[i].L + c[i].n * (c[i].n + 1) / 2.0;
        if (c[i].src == RQ_SPEC_MODEL && c[i].n > kmax) kmax = c[i].n;
    }
    T = rq_pass_cost(&e->tcost, R, sbase, S2);
    for (int p = 1; p <= kmax || (p == 1 && d0 > 0); p++) {
        int rows = p == 1 ? d0 : 0;
        double s1 = p == 1 ? ds0 : 0, s2 = 0;
        for (int i = 0; i < nc; i++) {
            if (c[i].src != RQ_SPEC_MODEL || c[i].n < p) continue;
            if (p == 1) { // (its catch-up rows and t: one segment)
                rows += c[i].lag + 1;
                s1 += c[i].L - c[i].lag;
                s2 += c[i].lag * (double)c[i].L - c[i].lag * (c[i].lag - 1) / 2.0;
            } else {
                rows++;
                s1 += c[i].L + p - 1;
            }
        }
        T += rq_pass_cost(&e->dcost, rows, s1, s2);
    }
    return T;
}

static double cand_yield(const Cand *c) {
    return c->src == RQ_SPEC_LOOKUP ? rq_spec_expect(c->plk, c->n) : c->src == RQ_SPEC_MODEL ? rq_spec_expect(c->pmd, c->n) : 0;
}

static void spec_plan(Engine *e, Cand *c, int nc, int ndec, double sdec, int rp, int d0, double ds0) {
    // auto (G7): the choices that maximize the expected tokens per second of the pass's decoding jobs. A plain pass gives
    // ndec tokens in T_dec (the cost model's decode rows alone); a choice adds its expected accepted guesses for its
    // extra time (its rows in the target's pass, the draft's passes). Greedy: the best (job, source, guess count) while it
    // raises the rate, so a choice is made only if its expected gain is > 1 (never slower than plain, as far as the model
    // knows) and the batch's own efficiency is the bar: the fuller the batch, the less speculation pays
    int rbase = ndec + rp, guesses = 0;
    double tdec = rq_pass_cost(&e->tcost, ndec, sdec, 0), y = ndec;
    for (int i = 0; i < nc; i++) // (the fixed jobs' choices as they are, within the pass's guesses at most)
        if (c[i].src != RQ_SPEC_PLAIN) {
            int room = e->gmax - guesses < e->maxb - 1 - ndec - guesses ? e->gmax - guesses : e->maxb - 1 - ndec - guesses;
            if (c[i].n > room) c[i].n = room;
            if (c[i].n <= 0) {
                c[i].src = RQ_SPEC_PLAIN;
                c[i].n = 0;
                continue;
            }
            y += cand_yield(&c[i]);
            guesses += c[i].n;
        }
    // t0: the pass without any guess (the extra time of a plan is plan_cost - t0)
    int src0[nc];
    for (int i = 0; i < nc; i++) {
        src0[i] = c[i].src;
        c[i].src = RQ_SPEC_PLAIN;
    }
    double t0 = plan_cost(e, c, nc, rbase, sdec, d0, ds0);
    for (int i = 0; i < nc; i++) c[i].src = src0[i];
    double cur = y / (tdec + plan_cost(e, c, nc, rbase, sdec, d0, ds0) - t0);
    for (;;) {
        int bi = -1, bs = 0, bn = 0;
        double best = cur * (1 + 1e-9);
        for (int i = 0; i < nc; i++) {
            if (c[i].fixed || c[i].src != RQ_SPEC_PLAIN) continue;
            for (int src = RQ_SPEC_LOOKUP; src <= RQ_SPEC_MODEL; src++) {
                int nmax = src == RQ_SPEC_LOOKUP ? c[i].nl : c[i].mk;
                for (int n = 1; n <= nmax; n++) {
                    if (guesses + n > e->gmax || ndec + guesses + n > e->maxb - 1) break;
                    c[i].src = src;
                    c[i].n = n;
                    double rate = (y + cand_yield(&c[i])) / (tdec + plan_cost(e, c, nc, rbase, sdec, d0, ds0) - t0);
                    c[i].src = RQ_SPEC_PLAIN;
                    c[i].n = 0;
                    if (rate > best) best = rate, bi = i, bs = src, bn = n;
                }
            }
        }
        if (bi < 0) break;
        c[bi].src = bs;
        c[bi].n = bn;
        y += cand_yield(&c[bi]);
        guesses += bn;
        cur = best;
    }
}

static int draft_wanted(const Engine *e, const Run *r, int L, int lag) {
    // the draft model may guess for this job at position L, so its sequence follows the target's (lag positions behind):
    // fixed, if the job has a model k; auto, if a model pass beats plain decoding for the job alone there (the cost model,
    // its acceptance) and the time it saves over the reply's remaining tokens is worth catching up (twice the cost of
    // the lag's rows: e.g. a long prompt for a short reply is not). An MTP view: whenever the job may guess (it follows every
    // job anyway, B: no lag to pay for; spec_plan chooses)
    if (!e->dm || r->spec == SPEC_OFF || r->sk <= 0) return 0;
    if (r->spec == SPEC_FIXED || e->mtp) return 1;
    RqSpecChoice x = rq_spec_choose(&e->tcost, &e->dcost, L, r->sk, acc_p(r->pm, e->pm, e->dprior), 0, 0);
    if (x.src != RQ_SPEC_MODEL) return 0;
    if (lag <= 2 * SPEC_GUESS_MAX) return 1;
    double left = r->j->max_new - r->j->n_gen, save = left * x.t_plain * (1 - 1 / x.gain);
    return save > 2 * rq_pass_cost(&e->dcost, lag, L - lag, (lag - 1) * (L - lag / 2.0));
}

static void spec_cands(Engine *e, Cand *c, int nc, int dbud) {
    // each candidate's options: the lookup's match (guesses and their chance), the model (if its draft can catch up and
    // guess in this pass); a fixed job's choice (as rq_spec_run: the lookup when it matches, else the model)
    for (int i = 0; i < nc; i++) {
        Run *r = c[i].r;
        Job *j = r->j;
        Slot *sl = &e->slot[r->si];
        RqSeq *s = sl->seq;
        const int *toks = rq_seq_tokens(s);
        int P = rq_seq_len(s), t = r->feed.v[r->fed], lim = SPEC_GUESS_MAX;
        if (j->max_new - j->n_gen - 1 < lim) lim = j->max_new - j->n_gen - 1;
        if (rq_seq_capacity(s) - 1 - P < lim) lim = rq_seq_capacity(s) - 1 - P;
        if (e->o.admit == ADMIT_CHUNK && r->E - 1 - P < lim) lim = r->E - 1 - P; // (within the lease: run_reach kept kk free)
        c[i].L = P;
        c[i].lim = lim;
        c[i].nl = c[i].mk = c[i].lag = c[i].n = 0;
        c[i].src = RQ_SPEC_PLAIN;
        c[i].fixed = r->spec == SPEC_FIXED;
        int dl = sl->dseq && r->sk > 0 ? rq_seq_keep(sl->dseq, rq_seq_common_prefix(sl->dseq, toks, P)) : 0;
        r->dwant = draft_wanted(e, r, P, P - dl);
        r->dlag = P - dl;
        if (lim <= 0) continue;
        if (r->sl > 0) {
            int ml = 0, lmin = r->slmin > 0 ? r->slmin : r->spec == SPEC_FIXED ? RQ_LOOKUP_MIN_DEFAULT : 2;
            if (rq_lookup_add(r->lk, toks, P) != RQ_OK) { // (a sequence cut below the indexed length)
                rq_lookup_reset(r->lk);
                rq_lookup_add(r->lk, toks, P);
            }
            c[i].nl = rq_lookup_draft(r->lk, toks, t, r->sl < lim ? r->sl : lim, lmin, r->guess + 1, &ml);
            r->mlen = ml > RQ_LOOKUP_LMAX ? RQ_LOOKUP_LMAX : ml;
            c[i].plk = acc_p(r->pl[r->mlen], e->pl[r->mlen], rq_lookup_prior(r->mlen));
        }
        if (r->dwant) {
            RqSeq *d = sl->dseq;
            int mk = r->sk < lim ? r->sk : lim;
            if (mk > rq_seq_capacity(d) - P) mk = rq_seq_capacity(d) - P;
            if (dl >= 0 && P - dl + 1 <= dbud && mk > 0) {
                c[i].mk = mk;
                c[i].lag = P - dl;
                c[i].pmd = acc_p(r->pm, e->pm, e->dprior);
            }
        }
        if (c[i].fixed) {
            if (c[i].nl > 0) c[i].src = RQ_SPEC_LOOKUP, c[i].n = c[i].nl;
            else if (c[i].mk > 0) c[i].src = RQ_SPEC_MODEL, c[i].n = c[i].mk;
        }
    }
}

static double seg_cost(const RqPassCost *pc, const RqSegment *sg, int n, int rows) {
    // the cost model's time for a pass over these segments (before it: their sequences' lengths are the positions)
    double s1 = 0, s2 = 0;
    for (int i = 0; i < n; i++) {
        double p = rq_seq_len(sg[i].seq);
        s1 += p;
        s2 += (sg[i].n - 1) * p + (sg[i].n - 1) * sg[i].n / 2.0;
    }
    return rq_pass_cost(pc, rows, s1, s2);
}

static void cost_learn(RqPassCost *eff, const RqPassCost *cal, double *lr, double pred, double took) {
    // the measured time against the calibrated model's (pred, from cal): a moving average of the log ratio (the last
    // ~20 passes), clamped to 0.5-4x; eff = cal scaled by it (the load, the thread count, what the calibration missed)
    if (!(pred > 0) || !(took > 0)) return;
    double x = log(took / pred);
    *lr = 0.95 * *lr + 0.05 * x;
    if (*lr < -0.69) *lr = -0.69;
    if (*lr > 1.39) *lr = 1.39;
    double r = exp(*lr);
    *eff = *cal;
    eff->t_w *= r;
    eff->att *= r;
    eff->att_x *= r;
}

static int draft_pass(Engine *e, int nds, int drows, int nsel, float **lg) {
    // one pass of the draft model over e->dseg (the decode path for a single row); on failure the segments' draft
    // sequences are emptied (a draft only ever guesses: the jobs go on without it)
    double pred = seg_cost(&e->dcal, e->dseg, nds, drows), t0 = rq_now();
    int err = nds == 1 && drows == 1 && nsel == 1 ? rq_decode(e->dc, e->dseg[0].seq, e->dseg[0].tokens[0], lg)
                                                  : rq_forward_batch(e->dc, e->dseg, nds, nsel > 1 ? e->dlogits : NULL, lg);
    double took = rq_now() - t0;
    e->t_draft += took;
    if (err == RQ_OK && nsel == nds && drows == nds) cost_learn(&e->dcost, &e->dcal, &e->dlr, pred, took); // (guess rows only)
    e->d_passes++;
    e->d_rows += drows;
    for (int i = 0; i < nds; i++) e->dsegrun[i]->j->sp_draft_rows += e->dseg[i].n;
    if (err != RQ_OK) {
        if (e->o.verbose) fprintf(stderr, "[engine] draft pass failed: %s (the jobs go on without it)\n",
                                  rq_ctx_error(e->dc)[0] ? rq_ctx_error(e->dc) : rq_strerror(err));
        for (int i = 0; i < nds; i++) {
            Run *r = e->dsegrun[i];
            rq_seq_reset(e->dseg[i].seq);
            if (r->src == RQ_SPEC_MODEL) r->src = RQ_SPEC_PLAIN, r->want = r->nd = 0, r->ddone = 1;
        }
    }
    return err;
}

static void spec_draft(Engine *e, int dbud, int ndec) {
    // the draft model's passes of this step: the first one with each model job's catch-up rows + t (the last row's
    // logits: its first guess), then the catch-up rows of the jobs whose draft follows the target (decode jobs that
    // want it, prompts being prefilled: this pass's chunk too), within dbud rows; then one pass per further guess with
    // one row per job still guessing (its previous guess). Qwen3.5 drafts: a checkpoint after t and each guess row but
    // the last (the next round's common prefix lands on it)
    int V = e->V, nds = 0, drows = 0, nsel = 0;
    for (int i = 0; i < e->nrun; i++) {
        Run *r = e->run[i];
        if (r->done || r->src != RQ_SPEC_MODEL) continue;
        Slot *sl = &e->slot[r->si];
        RqSeq *d = sl->dseq, *s = sl->seq;
        int P = rq_seq_len(s), cp = rq_seq_common_prefix(d, rq_seq_tokens(s), P);
        if (cp < rq_seq_len(d)) {
            rq_seq_truncate(d, cp);
            draft_ck_pending(e, r);
        }
        int dl = rq_seq_len(d), m = P - dl + 1;
        r->dbuf.n = 0;
        if (drows + m > e->dmaxb || rq_tokens_push(&r->dbuf, rq_seq_tokens(s) + dl, P - dl) != RQ_OK ||
            rq_tokens_push(&r->dbuf, &r->feed.v[r->fed], 1) != RQ_OK || (e->dhybrid && rq_seq_checkpoint_rows(d, P, r->want) != RQ_OK)) {
            r->src = RQ_SPEC_PLAIN;
            r->want = 0;
            continue;
        }
        r->dn = r->ddone = 0;
        e->dseg[nds] = (RqSegment) { d, r->dbuf.v, m, RQ_LOGITS_LAST };
        e->dsegrun[nds++] = r;
        drows += m;
        nsel++;
    }
    // the catch-up rows: in a pass that runs anyway (a model job's), else only a lag of half the budget or more (a prompt;
    // a lookup or plain spell) while the model is in use (chosen in the last 32 passes, or a job alone): a pass of the
    // draft for a token or two per step would cost more than its guesses save, and so do drafts that a full batch
    // does not use
    int room = dbud - drows, standalone = nds == 0, idle = e->since_model > 32 && ndec > 1;
    for (int i = 0; i < e->nrun && room > 0; i++) {
        Run *r = e->run[i];
        if (r->done || r->src == RQ_SPEC_MODEL || r->spec == SPEC_OFF || r->sk <= 0 || !e->dm || run_held(e, r)) continue;
        Slot *sl = &e->slot[r->si];
        RqSeq *d = sl->dseq, *s = sl->seq;
        const int *ref;
        int nref, nfull;
        if (r->phase == R_PREFILL) {
            ref = r->j->prompt.v;
            nfull = r->j->prompt.n;
            nref = rq_seq_len(s) + r->seg; // (the prompt up to this pass's chunk)
            if ((standalone && idle && ndec > 0) || !draft_wanted(e, r, nfull, nfull - rq_seq_keep(d, rq_seq_common_prefix(d, ref, nfull))))
                continue;
        } else {
            if (!r->dwant || (standalone && (r->dlag < dbud / 2 || idle))) continue;
            ref = rq_seq_tokens(s);
            nfull = nref = rq_seq_len(s);
        }
        int cp = rq_seq_common_prefix(d, ref, nfull);
        if (cp < rq_seq_len(d)) {
            rq_seq_truncate(d, cp);
            draft_ck_pending(e, r);
        }
        int dl = rq_seq_len(d), m = nref - dl;
        if (m > room) m = room;
        if (m > rq_seq_capacity(d) - dl) m = rq_seq_capacity(d) - dl;
        if (m <= 0) continue;
        e->dseg[nds] = (RqSegment) { d, ref + dl, m, RQ_LOGITS_NONE };
        e->dsegrun[nds++] = r;
        drows += m;
        room -= m;
    }
    if (nds == 0) return;
    float *lg = NULL;
    if (draft_pass(e, nds, drows, nsel, &lg) != RQ_OK) return;
    // the guesses: each model job's row, then a pass per further guess
    for (;;) {
        int k = 0, ng = 0;
        for (int i = 0; i < nds; i++) {
            Run *r = e->dsegrun[i];
            if (e->dseg[i].logits != RQ_LOGITS_LAST) continue;
            const float *row = lg + (size_t)k++ * V;
            if (r->ddone) continue;
            float p1;
            int g = rq_spec_pick(row, V, NULL, r->stau, &p1);
            if (g < 0 || (r->stau > 0 && p1 < r->stau)) {
                r->ddone = 1;
                continue;
            }
            r->guess[++r->dn] = g;
            if (r->dn >= r->want) r->ddone = 1;
            else ng++;
        }
        if (ng == 0) break;
        nds = drows = 0;
        for (int i = 0; i < e->nrun; i++) {
            Run *r = e->run[i];
            if (r->done || r->src != RQ_SPEC_MODEL || r->ddone) continue;
            e->dseg[nds] = (RqSegment) { e->slot[r->si].dseq, &r->guess[r->dn], 1, RQ_LOGITS_LAST };
            e->dsegrun[nds++] = r;
            drows++;
        }
        if (draft_pass(e, nds, drows, nds, &lg) != RQ_OK) return;
    }
    for (int i = 0; i < e->nrun; i++) {
        Run *r = e->run[i];
        if (!r->done && r->src == RQ_SPEC_MODEL) r->nd = r->dn;
    }
}

// ---------------------------------------------------------------------------------------------- the MTP drafter
// --draft mtp (TODO S9c; docs/server.md, "Speculative decoding"): the draft is the model's own MTP block (rq_mtp_load: a
// view of the target), one view sequence per slot (Slot.dseq) on its own context (dc). The view's row q takes the
// target's final hidden state at q - 1, which exists only in the pass that computed it, so every job is followed
// (prompts, forced tokens, replies; constrained and "speculative": false ones too) by B, the follow pass after each step's
// sampling: the view's rows of the target's new positions with the hidden rows of the target's pass that just ran
// (Engine.H: the target's passes ask for them only here), batched over the pass's jobs; the target's hidden state at its
// last position is kept with the view (rq_mtp_keep: the next row's h; a v2 seqfile carries it to disk and through the
// swap). At rest (between steps) a job's view is as long as its target (to its capacity), every row final, that state
// kept; it may be ahead only while the prompt is prefilled, as a prefix of the prompt (a copy, a load, a hybrid target's
// restart lower down: those rows' h was true). A, the guesses, runs before the target's pass for the jobs whose plan takes
// the model: row P (the token t to feed, h the kept state) with its logits (the first guess) and its own hidden state h',
// then one row per further guess (d_{j-1}, h'), provisional: B's cut to the common prefix (rq_seq_common_prefix counts
// final rows only) drops them, and whatever the verify cut. A row whose h is unknown (a gap after a failed view pass or a
// view reset; a restart exactly at a divergence: the cut drops the kept state) takes zeros, counted (Job.sp_mtp_approx,
// timings.speculative.mtp_approx): only the guesses get worse. The outputs stay plain decoding's: the view only proposes

static int mtp_pass(Engine *e, int nds, int rows, float **lg, double *t) {
    // one pass of the view over e->dseg (the decode path for a single row); lg: the segments' last rows' logits (A), NULL:
    // none (B). A failed pass appended nothing (every check comes first): the views stay as they were. The cost model learns
    // from passes of one row per segment (the guesses' shape, as draft_pass)
    RqSegment *g = e->dseg;
    double pred = lg && rows == nds ? seg_cost(&e->dcal, g, nds, rows) : 0, t0 = rq_now();
    int err = nds == 1 && rows == 1 ? rq_decode_hidden(e->dc, g->seq, g->tokens[0], lg, g->hidden, g->h_in)
                                    : rq_forward_batch(e->dc, g, nds, lg && nds > 1 ? e->dlogits : NULL, lg);
    double took = rq_now() - t0;
    *t += took;
    if (err == RQ_OK && pred > 0) cost_learn(&e->dcost, &e->dcal, &e->dlr, pred, took);
    if (err == RQ_OK) {
        for (int i = 0; i < nds; i++) // (a job that ended in this step: published, the connection thread may read its Job)
            if (!e->dsegrun[i]->done) e->dsegrun[i]->j->sp_draft_rows += g[i].n;
    } else if (e->o.verbose)
        fprintf(stderr, "[engine] the MTP view's pass failed (%d rows): %s (only the speed changes)\n", rows,
                rq_ctx_error(e->dc)[0] ? rq_ctx_error(e->dc) : rq_strerror(err));
    return err;
}

static void mtp_draft(Engine *e, int dbud) {
    // A (step, before the target's pass; spec_draft's part): the guesses of the jobs whose plan takes the model. Pass 1, one
    // segment per job: the view's rows from its length to P - 1 (a gap's, within dbud rows; h the kept state or zeros, as
    // H belongs to the coming pass) then t at P (h_{P-1}: the kept state at rest), all final; its logits the first guess,
    // its hidden state h' (into Run.mh). Then one pass per further guess, one row per job still guessing: (its last guess,
    // h'), its own h' next, provisional. A failed pass: its jobs stop guessing in this step (the view is not reset: nothing
    // was appended, B cuts the rest)
    int V = e->V, D = e->dim, nds = 0, rows = 0;
    float *lg = NULL;
    for (int i = 0; i < e->nrun; i++) {
        Run *r = e->run[i];
        if (r->done || r->src != RQ_SPEC_MODEL) continue;
        Slot *sl = &e->slot[r->si];
        RqSeq *v = sl->dseq, *s = sl->seq;
        int P = rq_seq_len(s), ML = rq_seq_common_prefix(v, rq_seq_tokens(s), P), m = P - ML + 1;
        if (ML < rq_seq_len(v)) rq_seq_truncate(v, ML); // (at rest a prefix of the target's tokens already)
        r->dbuf.n = 0;
        if (rows + m > dbud || rows + m > e->dmaxb || rq_tokens_push(&r->dbuf, rq_seq_tokens(s) + ML, P - ML) != RQ_OK ||
            rq_tokens_push(&r->dbuf, &r->feed.v[r->fed], 1) != RQ_OK) {
            r->src = RQ_SPEC_PLAIN;
            r->want = 0;
            continue;
        }
        r->map = rq_mtp_hin(v, NULL, 0, 0, m, e->Hin + (size_t)rows * D);
        r->dn = r->ddone = 0;
        e->dseg[nds] = (RqSegment) { v, r->dbuf.v, m, RQ_LOGITS_LAST, e->H + (size_t)rows * D, e->Hin + (size_t)rows * D };
        e->dsegrun[nds++] = r;
        rows += m;
    }
    if (nds == 0) return;
    int err = mtp_pass(e, nds, rows, &lg, &e->t_mtpA);
    e->mA_passes++;
    e->mA_rows += rows;
    for (int i = 0, off = 0; i < nds; off += e->dseg[i++].n) {
        Run *r = e->dsegrun[i];
        RqSeq *v = e->dseg[i].seq;
        if (err != RQ_OK) {
            r->src = RQ_SPEC_PLAIN;
            r->want = r->nd = 0;
            continue;
        }
        rq_mtp_final(v, rq_seq_len(v) - e->dseg[i].n);
        r->j->sp_mtp_approx += r->map;
        e->m_approx += r->map;
        memcpy(r->mh, e->H + (size_t)(off + e->dseg[i].n - 1) * D, (size_t)D * sizeof(float));
        r->mcur = 0;
    }
    if (err != RQ_OK) return;
    for (;;) { // the guesses: each job's row, then a pass per further guess
        int ng = 0;
        for (int i = 0; i < nds; i++) {
            Run *r = e->dsegrun[i];
            const float *row = lg + (size_t)i * V;
            float p1;
            int g = rq_spec_pick(row, V, NULL, r->stau, &p1);
            if (g < 0 || (r->stau > 0 && p1 < r->stau)) {
                r->ddone = 1;
                continue;
            }
            r->guess[++r->dn] = g;
            if (r->dn >= r->want) r->ddone = 1;
            else ng++;
        }
        if (ng == 0) break;
        nds = 0;
        for (int i = 0; i < e->nrun; i++) {
            Run *r = e->run[i];
            if (r->done || r->src != RQ_SPEC_MODEL || r->ddone) continue;
            e->dseg[nds] = (RqSegment) { e->slot[r->si].dseq, &r->guess[r->dn], 1, RQ_LOGITS_LAST, r->mh + (size_t)(1 - r->mcur) * D,
                                         r->mh + (size_t)r->mcur * D };
            e->dsegrun[nds++] = r;
        }
        err = mtp_pass(e, nds, nds, &lg, &e->t_mtpA);
        e->mA_passes++;
        e->mA_rows += nds;
        for (int i = 0; i < nds; i++) {
            Run *r = e->dsegrun[i];
            if (err == RQ_OK) r->mcur ^= 1;
            else r->src = RQ_SPEC_PLAIN, r->want = r->nd = 0, r->ddone = 1;
        }
        if (err != RQ_OK) break;
    }
    for (int i = 0; i < e->nrun; i++) {
        Run *r = e->run[i];
        if (!r->done && r->src == RQ_SPEC_MODEL) r->nd = r->dn;
    }
}

static void mtp_follow(Engine *e, int nseg) {
    // B (step, after the sampling): every job of this pass followed, those that ended in it too (their Run lives until reap,
    // their slot is not reused before). Per job: the view cut to its common prefix with the target (with the prompt while
    // the view's final rows are ahead of a target being prefilled: A's provisional rows do not count, they are past the
    // target's length after every verify that does not take them all), then its rows up to the target's length (its capacity), the tokens
    // the target's; h from this pass's hidden rows (hk: those still in the target, 0 if its pass failed or the rollback
    // reset it), else the kept state, else zeros (counted). A gap (the rows before this pass's: a failed view pass, a view
    // reset) runs whole if it fits in one pass (dmaxb rows), else dmaxb of its rows per step, this pass's rows then waiting
    // as gap rows too. In passes of at most dmaxb rows (a job's rows in order, several passes if need be), no logits. Then
    // the target's state at its last position kept: from this pass, else as kept if it is that one, else none. A failed
    // pass appended nothing: its jobs keep the gap (mgap), the next B carries it; logged once per job
    int D = e->dim, any = 0;
    for (int i = 0; i < nseg; i++) {
        Run *r = e->segrun[i];
        Slot *sl = &e->slot[r->si];
        RqSeq *v = sl->dseq, *s = sl->seq;
        int Lt = rq_seq_len(s), Lv = rq_seq_len(v), cap = rq_seq_capacity(v), end = Lt < cap ? Lt : cap;
        int Lf = rq_seq_common_prefix(v, rq_seq_tokens(v), Lv), n = r->j->prompt.n; // (Lf: its final rows)
        int ML = Lt >= Lf || Lt >= n ? rq_seq_common_prefix(v, rq_seq_tokens(s), Lt) : rq_seq_common_prefix(v, r->j->prompt.v, n);
        if (ML < Lv) rq_seq_truncate(v, ML);
        int hk = Lt - r->hp0 < r->hn ? Lt - r->hp0 : r->hn, gap = (r->hp0 < end ? r->hp0 : end) - ML;
        r->hk = hk > 0 ? hk : 0;
        r->mend = ML >= end ? ML : gap > e->dmaxb ? ML + e->dmaxb : end;
        any |= r->mend > ML;
    }
    while (any) {
        int nds = 0, rows = 0;
        any = 0;
        for (int i = 0; i < nseg && rows < e->dmaxb; i++) {
            Run *r = e->segrun[i];
            Slot *sl = &e->slot[r->si];
            RqSeq *v = sl->dseq;
            int q = rq_seq_len(v), k = r->mend - q;
            if (k <= 0) continue;
            if (k > e->dmaxb - rows) k = e->dmaxb - rows;
            float *hin = e->Hin + (size_t)rows * D;
            r->map = rq_mtp_hin(v, e->H + (size_t)r->hrow * D, r->hp0, r->hk, k, hin);
            e->dseg[nds] = (RqSegment) { v, rq_seq_tokens(sl->seq) + q, k, RQ_LOGITS_NONE, NULL, hin };
            e->dsegrun[nds++] = r;
            rows += k;
        }
        if (nds == 0) break;
        int err = mtp_pass(e, nds, rows, NULL, &e->t_mtpB);
        e->mB_passes++;
        e->mB_rows += rows;
        for (int i = 0; i < nds; i++) {
            Run *r = e->dsegrun[i];
            RqSeq *v = e->dseg[i].seq;
            if (err == RQ_OK) {
                rq_mtp_final(v, rq_seq_len(v) - e->dseg[i].n);
                if (r->done) r->mlate += r->map; // (published: job_finish ran in this step)
                else r->j->sp_mtp_approx += r->map;
                e->m_approx += r->map;
                any |= rq_seq_len(v) < r->mend;
                continue;
            }
            if (!r->mlog)
                fprintf(stderr, "[engine] job %d: the MTP view's follow pass failed (%s): its rows from %d wait (zeros for their h; only "
                        "the speed changes)\n", r->id, rq_strerror(err), rq_seq_len(v));
            r->mlog = 1;
            r->mend = rq_seq_len(v);
        }
    }
    for (int i = 0; i < nseg; i++) {
        Run *r = e->segrun[i];
        Slot *sl = &e->slot[r->si];
        RqSeq *v = sl->dseq;
        int Lt = rq_seq_len(sl->seq), Lv = rq_seq_len(v), cap = rq_seq_capacity(v), end = Lt < cap ? Lt : cap;
        if (Lt > 0 && r->hp0 <= Lt - 1 && Lt - 1 < r->hp0 + r->hk) rq_mtp_keep(v, Lt - 1, e->H + (size_t)(r->hrow + Lt - 1 - r->hp0) * D);
        else if (rq_mtp_kept(v, NULL) != Lt - 1) rq_mtp_keep(v, 0, NULL);
        r->mfol++;
        r->mgap = Lv < end ? end - Lv : 0;
        e->m_gaps += r->mgap > 0;
    }
}

static void mtp_check(Engine *e) {
    // RUNQ_TEST_MTP_CHECK=1 (tests only; engine_main, after each step): every slot's view at rest, by the public calls.
    // I1 every row final; a running job's view ahead only while its prompt is prefilled, as a prefix of the prompt; else I2
    // the target's tokens, I3 as long as the target (to its capacity) and I4 the target's state at its last position kept
    // (not at the capacity), unless no B has run yet or a failed one left a gap; an idle slot's: the target's tokens as far
    // as both go
    for (int i = 0; i < e->nslot; i++) {
        Slot *sl = &e->slot[i];
        RqSeq *v = sl->dseq, *s = sl->seq;
        Run *r = sl->run;
        if (!v) continue;
        int Lv = rq_seq_len(v), Lt = rq_seq_len(s), cap = rq_seq_capacity(v), hp = rq_mtp_kept(v, NULL), m = Lt < Lv ? Lt : Lv;
        const char *bad = NULL;
        if (rq_seq_common_prefix(v, rq_seq_tokens(v), Lv) != Lv) bad = "I1 (a row not final)";
        else if (!r) {
            if (rq_seq_common_prefix(v, rq_seq_tokens(s), m) != m) bad = "idle: not the target's tokens";
        } else if (Lv > Lt) {
            int n = r->j->prompt.n;
            if (Lt >= n || Lv > n || rq_seq_common_prefix(v, r->j->prompt.v, n) != Lv) bad = "ahead, not a prompt prefix being prefilled";
        } else if (rq_seq_common_prefix(v, rq_seq_tokens(s), Lt) != Lv) bad = "I2 (not the target's tokens)";
        else if (r->mfol > 0 && r->mgap == 0 && Lv != (Lt < cap ? Lt : cap)) bad = "I3 (the length)";
        else if (r->mfol > 0 && r->mgap == 0 && Lv < cap && hp != Lt - 1) bad = "I4 (the kept state)";
        if (bad) fprintf(stderr, "[engine] bug: slot %d's MTP view: %s (Lv %d, Lt %d, hpos %d)\n", i, bad, Lv, Lt, hp);
    }
}

static void mtp_dump(Engine *e, Run *r) {
    // RUNQ_TEST_MTP_DUMP=DIR (tests only; reap): a job that finished (no error status) with "mtp_approx" 0 in its timings
    // ("speculative" not off; and none after it was published: mlate) after a follow pass: its target's and view's sequences as DIR/<job>.rqseq and .draft.rqseq
    // (the view's in version 2 with its kept state), for api_test --mtp-seqcheck
    if (r->si < 0 || r->mfol == 0 || r->j->status || r->j->spec == SPEC_OFF || r->j->sp_mtp_approx != 0 || r->mlate) return;
    Slot *sl = &e->slot[r->si];
    char p[4096];
    snprintf(p, sizeof(p), "%s/%d.rqseq", e->mtp_dump, r->id);
    int rc = rq_seq_save(sl->seq, p);
    snprintf(p, sizeof(p), "%s/%d.draft.rqseq", e->mtp_dump, r->id);
    if (rc == RQ_OK) rc = rq_seq_save(sl->dseq, p);
    if (rc != RQ_OK) fprintf(stderr, "[engine] RUNQ_TEST_MTP_DUMP: job %d not saved: %s\n", r->id, rq_strerror(rc));
}

static void step(Engine *e, int stopping) {
    // one batched pass over every running job, then the sampling of the rows with logits
    int V = e->V, nseg = 0, rows = 0, ndec = 0, nc = 0;
    double now = rq_now(), sdec = 0;
    for (int i = 0; i < e->nrun; i++) { // (client gone, deadline, shutdown past the grace period)
        Run *r = e->run[i];
        r->seg = 0;
        r->row = -1;
        r->src = RQ_SPEC_PLAIN;
        r->want = r->nd = 0;
        if (r->done) continue;
        Job *j = r->j;
        int st = __atomic_load_n(&j->cancel, __ATOMIC_ACQUIRE) ? 499 : j->deadline > 0 && now > j->deadline ? 504 : stopping ? 503 : 0;
        if (st) end_run(e, r, 0, st, RQ_OK, NULL);
    }
    int npre = 0, pre_left = 0;
    Cand *cand = e->cand;
    for (int i = 0; i < e->nrun; i++) { // decode rows first: every decoding job's next token(s)
        Run *r = e->run[i];
        if (r->done || run_held(e, r)) continue; // (--admit chunk: parked, or a renewal first)
        if (r->phase == R_PREFILL) {
            npre++;
            pre_left += r->feed.n - r->fed;
            continue;
        }
        RqSeq *s = e->slot[r->si].seq;
        int left = r->feed.n - r->fed, k = left < e->maxb - rows ? left : e->maxb - rows;
        if (rq_seq_len(s) + left > rq_seq_capacity(s)) {
            end_run(e, r, FINISH_LENGTH, 0, RQ_OK, NULL);
            continue;
        }
        if (k <= 0) continue;
        r->seg = k;
        rows += k;
        ndec++;
        sdec += rq_seq_len(s);
        if (r->spec != SPEC_OFF && left == 1 && !r->forcing) cand[nc++] = (Cand) { r };
    }
    // speculation: the jobs' choices (G7), their planned guesses
    int budget = ndec ? e->o.chunk : e->o.batch;
    if (nc > 0) {
        int rp = pre_left < budget ? pre_left : budget;
        spec_cands(e, cand, nc, budget);
        spec_plan(e, cand, nc, ndec, sdec, rp, 0, 0);
        for (int i = 0; i < nc; i++) {
            Run *r = cand[i].r;
            r->src = cand[i].n > 0 ? cand[i].src : RQ_SPEC_PLAIN;
            r->want = cand[i].n;
            rows += r->want;
        }
    }
    // then prompt tokens within the budget (--chunk while jobs decode, else -b): first up to an equal share per
    // prompt (FIFO; a short prompt does not wait for a long one's whole prefill), then the rest FIFO
    if (budget > e->maxb - rows) budget = e->maxb - rows;
    int share = npre ? budget / npre : 0;
    if (share < 8) share = 8;
    for (int round = 0; round < 2 && budget > 0; round++)
        for (int i = 0; i < e->nrun && budget > 0; i++) {
            Run *r = e->run[i];
            if (r->done || r->phase != R_PREFILL || run_held(e, r)) continue;
            int left = r->feed.n - r->fed - r->seg, k = left < budget ? left : budget;
            if (round == 0 && k > share) k = share;
            r->seg += k;
            rows += k;
            budget -= k;
        }
    // the draft model's passes (its guesses; its sequences following the targets'), then the guesses into the segments
    if (e->dm) {
        int used = 0;
        for (int i = 0; i < e->nrun; i++) used |= !e->run[i]->done && e->run[i]->src == RQ_SPEC_MODEL;
        e->since_model = used ? 0 : e->since_model + 1;
        if (e->mtp) mtp_draft(e, ndec ? e->o.chunk : e->o.batch); // (A; the view follows in B, after the sampling)
        else spec_draft(e, ndec ? e->o.chunk : e->o.batch, ndec);
    }
    int nspec = 0;
    for (int i = 0; i < e->nrun; i++) {
        Run *r = e->run[i];
        if (r->done || r->phase != R_DECODE || r->src == RQ_SPEC_PLAIN) continue;
        RqSeq *s = e->slot[r->si].seq;
        if (r->src == RQ_SPEC_LOOKUP) r->nd = r->want;
        rows -= r->want - r->nd;
        if (r->nd <= 0) {
            r->src = RQ_SPEC_PLAIN;
            continue;
        }
        r->feed.n = r->fed + 1;
        if (rq_tokens_push(&r->feed, r->guess + 1, r->nd) != RQ_OK || rq_seq_checkpoint_rows(s, rq_seq_len(s), r->nd) != RQ_OK) {
            r->feed.n = r->fed + 1;
            rows -= r->nd;
            r->src = RQ_SPEC_PLAIN;
            r->nd = 0;
            continue;
        }
        r->seg = 1 + r->nd;
        r->vP = rq_seq_len(s);
        r->vn = r->nd;
        r->vrow = r->vacc = 0;
        nspec++;
    }
    for (int i = 0; i < e->nrun; i++) { // (the segments in admission order)
        Run *r = e->run[i];
        if (r->done || r->seg == 0) continue;
        e->seg[nseg] = (RqSegment) { e->slot[r->si].seq, r->feed.v + r->fed, r->seg,
                                     r->vn > 0 ? RQ_LOGITS_ALL : r->fed + r->seg == r->feed.n ? RQ_LOGITS_LAST : RQ_LOGITS_NONE };
        e->segrun[nseg++] = r;
    }
    if (nseg == 0) return;
    if (e->mtp) // (the MTP drafter: every row's hidden state, for B; a server without it does not pay for them)
        for (int i = 0, off = 0; i < nseg; off += e->seg[i++].n) {
            Run *r = e->segrun[i];
            e->seg[i].hidden = e->H + (size_t)off * e->dim;
            r->hrow = off;
            r->hp0 = rq_seq_len(e->seg[i].seq);
            r->hn = e->seg[i].n;
        }

    int nsel = 0, err;
    for (int i = 0; i < nseg; i++) nsel += e->seg[i].logits == RQ_LOGITS_ALL ? e->seg[i].n : e->seg[i].logits == RQ_LOGITS_LAST;
    float *lg = NULL;
    int dec_like = e->o.spec_used; // (the passes the cost model describes: decode rows and verify segments, every row's logits)
    for (int i = 0; i < nseg && dec_like; i++)
        dec_like = e->seg[i].logits == RQ_LOGITS_ALL || (e->seg[i].logits == RQ_LOGITS_LAST && e->seg[i].n == 1);
    double pred = dec_like ? seg_cost(&e->tcal, e->seg, nseg, rows) : 0, t0 = rq_now();
    if (nseg == 1 && rows == 1 && nsel == 1) // (one decode row: the decode path)
        err = e->mtp ? rq_decode_hidden(e->c, e->seg[0].seq, e->seg[0].tokens[0], &lg, e->H, NULL)
                     : rq_decode(e->c, e->seg[0].seq, e->seg[0].tokens[0], &lg);
    else
        err = rq_forward_batch(e->c, e->seg, nseg, nsel > 1 ? e->logits : NULL, &lg);
    if (err == RQ_OK && pred > 0) cost_learn(&e->tcost, &e->tcal, &e->tlr, pred, rq_now() - t0); // (decode-like passes)
    if (err == RQ_OK) {
        for (int i = 0, sel = 0; i < nseg; i++)
            if (e->seg[i].logits != RQ_LOGITS_NONE) {
                e->segrun[i]->row = sel;
                sel += e->seg[i].logits == RQ_LOGITS_ALL ? e->seg[i].n : 1;
            }
    } else { // each segment alone (nothing was computed: the passes check everything first): only a failing job fails
        char msg[300];
        for (int i = 0, sel = 0; i < nseg; i++) {
            Run *r = e->segrun[i];
            int want = e->seg[i].logits == RQ_LOGITS_ALL ? e->seg[i].n : e->seg[i].logits == RQ_LOGITS_LAST;
            int ei = nseg == 1 ? err : rq_forward_batch(e->c, &e->seg[i], 1, want ? e->logits + (size_t)sel * V : NULL, NULL);
            if (ei != RQ_OK) {
                snprintf(msg, sizeof(msg), "%s", rq_ctx_error(e->c)[0] ? rq_ctx_error(e->c) : rq_strerror(ei));
                if (r->vn > 0) r->vn = 0, r->feed.n = r->fed + 1; // (nothing fed: no rollback)
                r->hn = 0;
                end_run(e, r, 0, 0, ei, msg);
                continue;
            }
            if (want) {
                r->row = sel;
                sel += want;
            }
        }
        lg = e->logits;
    }
    e->passes++;
    e->rows += rows;
    if (nspec) {
        e->sp_passes++;
        e->sp_rows += rows;
    }

    // the rows with logits: prompts just prefilled (their first token), decode rows, verify segments (in rounds: row
    // vrow of each job still verifying, so that each job's RNG draws in its rows' order)
    now = rq_now();
    int ns = 0;
    for (int i = 0; i < nseg; i++) {
        Run *r = e->segrun[i];
        if (r->done) continue;
        r->fed += r->seg;
        if (r->row < 0) continue; // (more to feed)
        Job *j = r->j;
        Slot *sl = &e->slot[r->si];
        if (r->phase == R_PREFILL) {
            r->phase = R_DECODE;
            job_lock(j);
            j->t_first = now;
            for (int k = r->feed.n - r->nforced0; k < r->feed.n; k++) {
                out_token(&r->o, r->feed.v[k]);
                if (j->top_logprobs >= 0) logprobs_add(e, j, r->feed.v[k], NULL, NULL, 1);
                j->n_gen++;
            }
            job_notify(j);
            job_unlock(j);
            r->feed.n = r->fed = 0;
            if (!prep(e, r) || r->feed.n > 0) continue; // (forced tokens right after the prompt: fed first)
            if (j->force.n > 0) { // (force_reply: its tokens next)
                force_feed(e, r);
                continue;
            }
        } else
            j->sp_passes++;
        if (r->forcing) { // (force_reply fed: the reply ends here)
            end_run(e, r, r->forcing == 2 ? FINISH_LENGTH : FINISH_STOP, 0, RQ_OK, NULL);
            continue;
        }
        if (j->top_logprobs >= 0 && slot_scratch(e, sl, 1) != RQ_OK) {
            end_run(e, r, 0, 0, RQ_ERR_NOMEM, NULL);
            continue;
        }
        r->vrow = 0;
        e->srun[ns++] = r;
    }
    while (ns > 0) {
        float *b = nspec ? e->slg : lg; // (without verify rows: compacted in place, row k <= its own)
        for (int k = 0; k < ns; k++) {
            Run *r = e->srun[k];
            Job *j = r->j;
            Slot *sl = &e->slot[r->si];
            const float *row = lg + (size_t)(r->row + r->vrow) * V;
            if (row != b + (size_t)k * V) memcpy(b + (size_t)k * V, row, V * sizeof(float));
            if (j->top_logprobs >= 0) memcpy(sl->lcopy, b + (size_t)k * V, V * sizeof(float));
            e->sseq[k] = sl->seq;
            e->scfg[k] = (RqSampling) { j->temperature, j->top_p, j->con ? sl->allow : NULL, j->top_k };
        }
        err = rq_sample_batch(e->c, e->sseq, b, ns, e->scfg, ns, e->stok);
        int nn = 0;
        for (int k = 0; k < ns; k++) {
            Run *r = e->srun[k];
            take_token(e, r, err == RQ_OK ? e->stok[k] : err);
            if (!r->done && r->vn > 0 && r->vacc) {
                r->vrow++;
                e->srun[nn++] = r;
            }
        }
        ns = nn;
    }
    if (e->mtp) mtp_follow(e, nseg); // (B: the views of this pass's jobs to their targets' lengths)
}

static void reap(Engine *e) {
    // drop the finished runs (their slots are already free); keep the admission order
    int k = 0;
    for (int i = 0; i < e->nrun; i++) {
        Run *r = e->run[i];
        if (!r->done) {
            e->run[k++] = r;
            continue;
        }
        if (e->mtp_dump) mtp_dump(e, r);
        free(r->mh);
        rq_tokens_free(&r->feed);
        rq_tokens_free(&r->dbuf);
        rq_lookup_free(r->lk);
        free(r->alist);
        jb_free(&r->o.text);
        tscan_free(&r->o.ts);
        job_release(r->j);
        free(r);
    }
    e->nrun = k;
}

// ------------------------------------------------------------------------------------------ engine

static void *engine_main(void *arg) {
    Engine *e = arg;
    pthread_mutex_lock(&e->mu);
    for (;;) {
        while (e->nrun == 0 && e->qn == 0 && !e->stop) pthread_cond_wait(&e->cv, &e->mu);
        if (e->nrun == 0 && e->qn == 0) break;
        double now = rq_now();
        int k = 0;
        for (int i = 0; i < e->qn; i++) { // (queued jobs whose client left or whose time is up)
            Job *j = e->q[i];
            int st = __atomic_load_n(&j->cancel, __ATOMIC_ACQUIRE) ? 499 : j->deadline > 0 && now > j->deadline ? 504 : 0;
            if (!st) {
                e->q[k++] = j;
                continue;
            }
            job_finish(j, FINISH_NONE, st, st == 499 ? "client_closed" : "timeout",
                       st == 499 ? "the client closed the connection" : "the request timed out");
            job_release(j);
        }
        e->qn = k;
        int io = mem_renewals(e); // (--admit chunk: the running jobs' leases first; a save planned: no admission in this step)
        e->mwj = NULL;
        if (io && e->qn > 0 && (e->npark > 0 || e->nswap > 0)) { // (the head waits for memory: a job parked or swapped out)
            if (e->q[0]->t_mw == 0) e->q[0]->t_mw = rq_now();
            e->mwj = e->q[0];
        }
        // FIFO (a job waiting for a shared prefix keeps its place); --admit-yield: the seats are nrun - nyld (a yielded job holds
        // none), and the queue's jobs that came after a yielded job's yield wait behind it (mem_resume gives it the seat)
        int refused = 0;
        for (int i = 0; i < e->qn && e->nrun - e->nyld < e->parallel && e->nrun < e->rcap && !io && !e->yblock;) {
            if (e->yt > 0 && e->q[i]->t_submit > e->yt) break;
            int a = admit(e, e->q[i]);
            if (a < 0) {
                refused = 1;
                break;
            }
            if (a == 2) { // (the disk tier's I/O first, then this job again: it keeps its place)
                io = 1;
                break;
            }
            if (a == 0) {
                i++;
                continue;
            }
            memmove(&e->q[i], &e->q[i + 1], (size_t)(e->qn - i - 1) * sizeof(Job *));
            e->qn--;
        }
        if (refused && !io && e->nyld > 0 && e->npark == 0 && e->nswap == 0) { // (--admit-yield: no room or no idle slot for the
            double acc = 0;                                                        // job whose turn it is: a yielded job's swapped out)
            io = mem_swap_out(e, NULL, &acc, 1);
        }
        for (int i = 0; i < e->qn; i++) // (mem_wait_ms: a wait for memory ends when the job is refused for something else)
            if (e->q[i]->t_mw > 0 && e->q[i] != e->mwj) {
                e->q[i]->mem_wait += now - e->q[i]->t_mw;
                e->q[i]->t_mw = 0;
            }
        int stopping = e->stop && now > e->grace_until;
        e->running = e->nrun;
        mem_publish(e);
        pthread_mutex_unlock(&e->mu);
        if (io) sw_run(e);
        if (e->nrun > 0) {
            double ts = rq_now();
            step(e, stopping);
            e->t_step += rq_now() - ts;
            e->steps++;
        }
        reap(e);
        if (e->mtp_chk) mtp_check(e);
        pthread_mutex_lock(&e->mu);
        e->running = e->nrun;
        mem_publish(e);
    }
    pthread_mutex_unlock(&e->mu);
    for (int i = 0; e->dk_dir && i < e->nslot; i++) // the disk tier: what the disk has not seen yet
        if (e->slot[i].gen != e->slot[i].dk_gen && rq_seq_len(e->slot[i].seq) >= e->o.kv_disk_min) dk_spill(e, i, -1);
    if (e->dk_dir && (e->dk_saves || e->dk_loads))
        fprintf(stderr, "[engine] disk: %llu saves (%.2f s), %llu loads (%.2f s); %d entries, %.2f GiB in %s\n", e->dk_saves, e->dk_save_s,
                e->dk_loads, e->dk_load_s, e->ndk, e->dk_bytes / 1073741824.0, e->dk_dir);
    if (e->o.verbose) {
        fprintf(stderr, "[engine] %llu passes, %llu rows\n", e->passes, e->rows);
        if (e->sp_passes || e->d_passes)
            fprintf(stderr, "[engine] speculation: %llu passes with guesses (%llu rows); model %llu / %llu guesses accepted, lookup "
                    "%llu / %llu; draft %llu passes, %llu rows, %.3f s; measured / calibrated costs x%.2f (target), x%.2f (draft)\n",
                    e->sp_passes, e->sp_rows, e->sp_acc[RQ_SPEC_MODEL], e->sp_guess[RQ_SPEC_MODEL], e->sp_acc[RQ_SPEC_LOOKUP],
                    e->sp_guess[RQ_SPEC_LOOKUP], e->d_passes, e->d_rows, e->t_draft, exp(e->tlr), exp(e->dlr));
        if (e->mtp) // (A: the guesses' passes, B: the follow passes; approx: rows with zeros for h; gaps: follow passes that left one)
            fprintf(stderr, "[engine] mtp: A %llu passes %llu rows %.3f s; B %llu passes %llu rows %.3f s; steps %llu %.3f s; approx %llu; "
                    "gaps %llu\n", e->mA_passes, e->mA_rows, e->t_mtpA, e->mB_passes, e->mB_rows, e->t_mtpB, e->steps, e->t_step,
                    e->m_approx, e->m_gaps);
    }
    return NULL;
}

void engine_free(Engine *e) {
    if (!e) return;
    for (int i = 0; e->slot && i < e->nslot; i++) {
        rq_seq_free(e->slot[i].seq);
        rq_seq_free(e->slot[i].dseq);
        free(e->slot[i].allow);
        free(e->slot[i].list);
        free(e->slot[i].lcopy);
    }
    free(e->slot);
    free(e->rv);
    free(e->swo);
    if (e->c) rq_ctx_free(e->c);
    con_vocab_free(e->vocab);
    free(e->run);
    free(e->seg);
    free(e->segrun);
    free(e->logits);
    free(e->sseq);
    free(e->scfg);
    free(e->stok);
    free(e->srun);
    free(e->slg);
    free(e->dlogits);
    free(e->dseg);
    free(e->dsegrun);
    free(e->H);
    free(e->Hin);
    free(e->cand);
    if (e->dc) rq_ctx_free(e->dc);
    dk_close(e);
    free(e->q);
    pthread_mutex_destroy(&e->mu);
    pthread_cond_destroy(&e->cv);
    free(e);
}

static int single_token(const RqModel *m, const char *text) {
    RqTokens t = { 0 };
    int id = rq_tokenize(m, text, -1, &t) == RQ_OK && t.n == 1 ? t.v[0] : -1;
    rq_tokens_free(&t);
    return id;
}

Engine *engine_start(RqModel *m, const EngineOpts *o, char *err, size_t errlen) {
    Engine *e = calloc(1, sizeof(Engine));
    if (!e) { snprintf(err, errlen, "out of memory"); return NULL; }
    pthread_mutex_init(&e->mu, NULL);
    pthread_cond_init(&e->cv, NULL);
    e->m = m;
    e->o = *o;
    e->dk_lock = -1;
    e->plan.spill = e->plan.slot = e->plan.load = -1;
    if (e->o.kv_disk_min < 1) e->o.kv_disk_min = 1;
    if (e->o.queue < 1) e->o.queue = 1;
    if (e->o.batch < 1) e->o.batch = 1;
    if (e->o.batch > RQ_BATCH_MAX) e->o.batch = RQ_BATCH_MAX;
    if (e->o.chunk < 1) e->o.chunk = 64;
    if (e->o.chunk > e->o.batch) e->o.chunk = e->o.batch;
    if (o->admit == ADMIT_CHUNK && !(o->kv_disk && *o->kv_disk)) { // (the swap: the oldest job's guarantee)
        snprintf(err, errlen, "--admit chunk needs --kv-disk DIR: a request is swapped out to DIR when the oldest one's lease needs its memory");
        engine_free(e);
        return NULL;
    }
    if (o->admit_yield > 0 && o->admit != ADMIT_CHUNK) {
        snprintf(err, errlen, "--admit-yield needs --admit chunk (a request yields its seat at a lease's end)");
        engine_free(e);
        return NULL;
    }
    RqModelInfo info;
    rq_model_info(m, &info);
    e->V = info.vocab_size;
    e->eos = info.eos;
    e->chat = info.chat;
    e->hybrid = info.linear_layers > 0;
    e->ckn = o->checkpoints > 0 ? o->checkpoints : 4;
    if (e->ckn > RQ_CHECKPOINTS_MAX) e->ckn = RQ_CHECKPOINTS_MAX;
    e->t_im_start = single_token(m, "<|im_start|>");
    e->t_system = single_token(m, "system");
    e->t_user = single_token(m, "user");
    e->t_assistant = single_token(m, "assistant");
    e->t_nl = single_token(m, "\n");
    e->t_toolresp = single_token(m, "<tool_response>");
    e->think_start = single_token(m, "<think>");
    e->think_end = single_token(m, "</think>");
    // the pool: --slots sequences of -c positions (default --parallel + 4), within --kv-mem (default: half the RAM, at
    // most MemAvailable now, after the load), each counted with its KV cache, recurrent state and checkpoints
    int rc = rq_seq_mem(m, o->ctx, o->kv_type, &e->sm);
    if (rc != RQ_OK) {
        snprintf(err, errlen, "cannot create a sequence of %d positions: %s", o->ctx, rq_strerror(rc));
        engine_free(e);
        return NULL;
    }
    if (e->sm.state) { // (the slots keep --checkpoints copies of the state: rq_seq_set_checkpoints below)
        e->sm.checkpoints = e->ckn;
        e->sm.total = e->sm.kv + (1 + (size_t)e->ckn) * e->sm.state;
    }
    // speculation: a draft sequence per slot (its KV cache, state and checkpoints), and per running job the transient
    // checkpoints of the verify rows (Qwen3.5: spec_kmax + 1 states of each model)
    e->dm = o->draft;
    e->mtp = e->dm && o->draft_mtp > 0; // (the MTP view: KV only, one attention layer; no state)
    e->dim = info.dim;
    e->dprior = rq_spec_prior(e->dm);
    if (e->dm) {
        RqModelInfo di;
        rq_model_info(e->dm, &di);
        e->dhybrid = di.linear_layers > 0;
        if ((rc = rq_seq_mem(e->dm, o->ctx, o->kv_type, &e->dsm)) != RQ_OK) {
            snprintf(err, errlen, "the draft model: cannot create a sequence of %d positions: %s", o->ctx, rq_strerror(rc));
            engine_free(e);
            return NULL;
        }
        if (e->dsm.state) {
            e->dsm.checkpoints = e->ckn;
            e->dsm.total = e->dsm.kv + (1 + (size_t)e->ckn) * e->dsm.state;
        }
    }
    double per_slot = (double)e->sm.total + (double)e->dsm.total, mem = o->kv_mem > 0 ? o->kv_mem * 1073741824.0 : 0;
    e->mem_src = "--kv-mem";
    if (mem <= 0) {
        long pages = sysconf(_SC_PHYS_PAGES), psz = sysconf(_SC_PAGESIZE);
        double avail = rq_mem_available();
        mem = pages > 0 && psz > 0 ? 0.5 * (double)pages * psz : 8.0 * 1073741824.0;
        e->mem_src = "half the RAM";
        if (avail > 0 && avail < mem) {
            mem = avail;
            e->mem_src = "MemAvailable";
        }
    }
    e->parallel = o->parallel > 0 ? o->parallel : 16;
    e->nslot = o->slots > 0 ? o->slots : e->parallel + 4;
    if (o->admit != ADMIT_WORST) { // (the slots are not lowered: the leases are within the budget, and one -c job fits)
        int ck = e->ckn + (o->spec_used ? o->spec_kmax + 1 : 0);
        if (ck > RQ_CHECKPOINTS_MAX) ck = RQ_CHECKPOINTS_MAX;
        e->job_max = per_slot + (double)(ck - e->ckn) * ((double)e->sm.state + (double)e->dsm.state);
        // a request's guesses at most (Qwen3.5 / 3.8: their state copies): the most one -c job can hold within the budget
        // (>= spec_kmax: the check below), so a request with "speculative" and no -G / --lookup still speculates
        double st = (double)e->sm.state + (double)e->dsm.state;
        for (e->kcap = SPEC_GUESS_MAX; st > 0 && e->kcap > 0; e->kcap--) {
            int c = e->ckn + e->kcap + 1;
            if (c > RQ_CHECKPOINTS_MAX) c = RQ_CHECKPOINTS_MAX;
            if (per_slot + (double)(c - e->ckn) * st <= mem) break;
        }
        if (e->job_max > mem) {
            snprintf(err, errlen, "--admit %s: a job of -c %d positions%s needs %.2f GiB (KV cache and recurrent state), more than the "
                     "pool's %.2f GiB (%s)", o->admit == ADMIT_CHUNK ? "chunk" : "bound", o->ctx, o->spec_used ? ", speculating" : "", e->job_max / 1073741824.0, mem / 1073741824.0, e->mem_src);
            engine_free(e);
            return NULL;
        }
    } else {
        if (o->spec_used) e->spec_res = (double)e->parallel * (o->spec_kmax + 1) * ((double)e->sm.state + (double)e->dsm.state);
        if (e->nslot * per_slot > mem - e->spec_res) e->nslot = (int)((mem - e->spec_res) / per_slot);
    }
    if (e->nslot < 1) {
        snprintf(err, errlen, "-c %d needs %.2f GiB per sequence (KV cache and recurrent state), more than the pool's %.2f GiB (%s)", o->ctx,
                 per_slot / 1073741824.0, mem / 1073741824.0, e->mem_src);
        engine_free(e);
        return NULL;
    }
    if (e->parallel > e->nslot) e->parallel = e->nslot;
    e->o.kv_mem = mem / 1073741824.0;
    // (the guesses of a pass: at most SPEC_GUESS_MAX per job, gmax in all; the context's rows and logits rows for them)
    e->gmax = e->parallel * SPEC_GUESS_MAX < 64 ? e->parallel * SPEC_GUESS_MAX : 64;
    e->maxb = e->o.batch > e->o.chunk + e->parallel + e->gmax ? e->o.batch : e->o.chunk + e->parallel + e->gmax;
    if (e->maxb > RQ_BATCH_MAX) e->maxb = RQ_BATCH_MAX;
    if (e->maxb < 2) e->maxb = 2;
    e->c = rq_ctx_new(m, e->maxb, o->ctx, &rc);
    e->slg = malloc((size_t)e->parallel * e->V * sizeof(float));
    e->cand = calloc(e->parallel, sizeof(Cand));
    if (e->dm) {
        e->dmaxb = e->maxb;
        e->dc = rq_ctx_new(e->dm, e->dmaxb, o->ctx, &rc);
        e->dseg = calloc(e->parallel, sizeof(RqSegment));
        e->dsegrun = calloc(e->parallel, sizeof(Run *));
        e->dlogits = malloc((size_t)e->parallel * e->V * sizeof(float));
        if (e->mtp) { // (the target's pass's hidden rows, the view's passes' h inputs: maxb rows each)
            e->H = malloc((size_t)e->maxb * e->dim * sizeof(float));
            e->Hin = malloc((size_t)e->maxb * e->dim * sizeof(float));
        }
        if (!e->dc || !e->dseg || !e->dsegrun || !e->dlogits || (e->mtp && (!e->H || !e->Hin))) {
            snprintf(err, errlen, "out of memory (the draft model's compute context)");
            engine_free(e);
            return NULL;
        }
    }
    e->slot = calloc(e->nslot, sizeof(Slot));
    e->rv = calloc(e->nslot, sizeof(int));
    // (--admit-yield: the runs are the seats + the yielded jobs, and queued + started jobs are at most queue + parallel:
    // engine_submit)
    e->rcap = o->admit_yield > 0 && o->admit == ADMIT_CHUNK ? e->parallel + e->o.queue : e->parallel;
    e->swo = calloc(e->rcap, sizeof(Run *));
    e->q = calloc(e->o.queue + e->parallel, sizeof(Job *));
    e->run = calloc(e->rcap, sizeof(Run *));
    e->seg = calloc(e->parallel, sizeof(RqSegment));
    e->segrun = calloc(e->parallel, sizeof(Run *));
    e->logits = malloc((size_t)(e->parallel + e->gmax) * e->V * sizeof(float));
    e->sseq = calloc(e->parallel, sizeof(RqSeq *));
    e->scfg = calloc(e->parallel, sizeof(RqSampling));
    e->stok = calloc(e->parallel, sizeof(int));
    e->srun = calloc(e->parallel, sizeof(Run *));
    e->vocab = con_vocab_new(m);
    if (!e->c || !e->slot || !e->q || !e->run || !e->seg || !e->segrun || !e->logits || !e->sseq || !e->scfg || !e->stok || !e->srun || !e->vocab ||
        !e->slg || !e->cand || !e->rv || !e->swo) {
        snprintf(err, errlen, "out of memory (compute context)");
        engine_free(e);
        return NULL;
    }
    for (int i = 0; i < e->nslot; i++) {
        e->slot[i].seq = rq_seq_new(m, o->ctx, o->kv_type, &rc);
        if (e->slot[i].seq && e->hybrid) rq_seq_set_checkpoints(e->slot[i].seq, e->ckn);
        if (!e->slot[i].seq) {
            snprintf(err, errlen, "cannot create a sequence of %d positions: %s", o->ctx, rq_strerror(rc));
            engine_free(e);
            return NULL;
        }
        if (e->dm) {
            e->slot[i].dseq = rq_seq_new(e->dm, o->ctx, o->kv_type, &rc);
            if (e->slot[i].dseq && e->dhybrid) rq_seq_set_checkpoints(e->slot[i].dseq, e->ckn);
            if (!e->slot[i].dseq) {
                snprintf(err, errlen, "the draft model: cannot create a sequence of %d positions: %s", o->ctx, rq_strerror(rc));
                engine_free(e);
                return NULL;
            }
        }
    }
    // the cost model (G7): calibrated on slot 0's sequences (empty; emptied again), else estimated
    rq_pass_cost_default(m, &e->tcost);
    if (e->dm) rq_pass_cost_default(e->dm, &e->dcost);
    if (o->spec_used && o->calib > 0) {
        double t0 = rq_now();
        int L = o->calib < o->ctx - 32 ? o->calib : o->ctx - 32;
        if ((rc = rq_pass_cost_measure(e->c, e->slot[0].seq, L, 5, &e->tcost)) != RQ_OK ||
            (e->dm && (rc = rq_pass_cost_measure(e->dc, e->slot[0].dseq, L, 5, &e->dcost)) != RQ_OK))
            fprintf(stderr, "runq serve: the speculation cost calibration failed (%s): estimates instead\n", rq_strerror(rc));
        if (e->hybrid) rq_seq_set_checkpoints(e->slot[0].seq, e->ckn);
        if (e->dhybrid) rq_seq_set_checkpoints(e->slot[0].dseq, e->ckn);
        // (the calibration's positions given back, in both modes: up to --spec-calibrate of them stayed resident)
        rq_seq_reset(e->slot[0].seq);
        rq_seq_release(e->slot[0].seq, 0);
        if (e->slot[0].dseq) {
            rq_seq_reset(e->slot[0].dseq);
            rq_seq_release(e->slot[0].dseq, 0);
        }
        e->t_cal = rq_now() - t0;
    }
    e->tcal = e->tcost;
    e->dcal = e->dcost;
    if (o->kv_disk && *o->kv_disk && dk_open(e, err, errlen)) {
        engine_free(e);
        return NULL;
    }
    const char *mf = getenv("RUNQ_TEST_MEMAVAILABLE"); // (step 3's test hook: MemAvailable from this file)
    e->ma_file = mf && *mf ? mf : NULL;
    const char *sd = getenv("RUNQ_TEST_SWAP_DELAY"); // (tests: s of sleep in each swap-out, so that one lasts)
    e->sw_delay = sd && *sd ? atof(sd) : 0;
    const char *sf = getenv("RUNQ_TEST_SWAP_FAIL"); // (tests: a file naming the swap I/O to fail, draft-save / draft-load)
    e->sw_fail = sf && *sf ? sf : NULL;
    const char *mc = getenv("RUNQ_TEST_MTP_CHECK"), *md = getenv("RUNQ_TEST_MTP_DUMP"); // (tests: the MTP view's at-rest check, dumps)
    e->mtp_chk = e->mtp && mc && !strcmp(mc, "1");
    e->mtp_dump = e->mtp && md && *md ? md : NULL;
    if (e->o.admit_margin < 0) e->o.admit_margin = 0;
    mem_publish(e);
    if (pthread_create(&e->th, NULL, engine_main, e)) {
        snprintf(err, errlen, "cannot start the engine thread");
        engine_free(e);
        return NULL;
    }
    e->started = 1;
    return e;
}

void engine_describe(Engine *e, char *buf, size_t n) {
    const double G = 1073741824.0;
    char st[96] = "";
    if (e->sm.state)
        snprintf(st, sizeof(st), " + recurrent state %.1f MiB x (1 + %d checkpoints)", e->sm.state / 1048576.0, e->sm.checkpoints);
    char dr[160] = "";
    if (e->mtp) // (no state: KV only, the block's one attention layer)
        snprintf(dr, sizeof(dr), "; the MTP view's sequence %.1f MiB (blk.%d, one attention layer)", e->dsm.total / 1048576.0, e->o.draft_mtp);
    else if (e->dm)
        snprintf(dr, sizeof(dr), "; the draft's sequence %.1f MiB", e->dsm.total / 1048576.0);
    if (e->spec_res > 0) {
        size_t l = strlen(dr);
        snprintf(dr + l, sizeof(dr) - l, "; %.0f MiB kept for the verify rows' checkpoints", e->spec_res / 1048576.0);
    }
    snprintf(buf, n, "%d running at most, %d slot(s) of %d positions (per slot %.1f MiB: KV %.0f KB per position%s%s; %.2f GiB at most, "
             "touched as used; pool budget %.2f GiB, %s), prefill chunks %d / %d tokens per pass (others decoding / not)",
             e->parallel, e->nslot, e->o.ctx, (e->sm.total + e->dsm.total) / 1048576.0, (double)e->sm.kv / e->o.ctx / 1000.0, st, dr,
             ((double)e->sm.total + e->dsm.total) * e->nslot / G, e->o.kv_mem, e->mem_src, e->o.chunk, e->o.batch);
    if (e->dk_dir) {
        size_t l = strlen(buf);
        snprintf(buf + l, n - l, "; disk tier %s: %d saved sequence(s), %.2f of %.2f GiB, slots saved from %d tokens", e->dk_dir, e->ndk,
                 e->dk_bytes / G, e->dk_max / G, e->o.kv_disk_min);
    }
}

static void cost_text(char *b, size_t n, const RqPassCost *c) {
    snprintf(b, n, "%.2f ms + %.3f / %.3f us x position (a segment's first / further rows; %s), f(2/4/8/16) %.2f / %.2f / %.2f / %.2f",
             c->t_w * 1e3, c->att * 1e6, c->att_x * 1e6, c->calibrated ? "measured" : "estimated", rq_pass_cost(c, 2, 0, 0) / c->t_w,
             rq_pass_cost(c, 4, 0, 0) / c->t_w, rq_pass_cost(c, 8, 0, 0) / c->t_w, rq_pass_cost(c, 16, 0, 0) / c->t_w);
}

void engine_describe_spec(Engine *e, char *buf, size_t n) {
    // the cost model, and the choice for one sequence alone by position (the priors: the model's guesses accepted at 0.6, an
    // MTP view's at 0.8: rq_spec_prior; a lookup match of 4 tokens)
    char t[200], d[200] = "";
    cost_text(t, sizeof(t), &e->tcost);
    if (e->dm) cost_text(d, sizeof(d), &e->dcost);
    int l = snprintf(buf, n, "costs: target %s%s%s (calibration %.1f s); one sequence alone", t, e->dm ? "; draft " : "", d, e->t_cal);
    const int Ls[] = { 512, 4096, 16384, 32768, 65536, 98304 };
    for (int i = 0; i < 6 && Ls[i] <= e->o.ctx && l < (int)n; i++) {
        RqSpecChoice c = rq_spec_choose(&e->tcost, e->dm ? &e->dcost : NULL, Ls[i], 3, e->dprior, 3, rq_lookup_prior(4));
        if (c.src == RQ_SPEC_PLAIN) l += snprintf(buf + l, n - l, "%s at %d: plain", i ? "," : "", Ls[i]);
        else l += snprintf(buf + l, n - l, "%s at %d: %s %d (x%.2f)", i ? "," : "", Ls[i],
                           c.src == RQ_SPEC_MODEL ? e->mtp ? "mtp k" : "model k" : "lookup",
                           c.n, c.gain);
    }
}

int engine_describe_admit(Engine *e, char *buf, size_t n) {
    // --admit bound / chunk: the budget, a lease's terms (per position, a state copy: target + draft), the -c job that must fit
    if (e->o.admit == ADMIT_WORST) return 0;
    const double G = 1073741824.0;
    double kv = ((double)e->sm.kv + (double)e->dsm.kv) / e->o.ctx, st = (double)e->sm.state + (double)e->dsm.state;
    int l = e->o.admit == ADMIT_CHUNK
                ? snprintf(buf, n, "admission chunk: leases (prompt + %d tokens, renewed by %d as the reply grows, at most prompt + max tokens: KV "
                           "%.1f KB per position", e->o.admit_chunk, e->o.admit_chunk, kv / 1000.0)
                : snprintf(buf, n, "admission bound: leases (prompt + max tokens: KV %.1f KB per position", kv / 1000.0);
    if (st > 0) l += snprintf(buf + l, n - l, ", state %.1f MiB x (1 + %d checkpoints + the verify rows' guesses + 1)", st / 1048576.0, e->ckn);
    l += snprintf(buf + l, n - l, ") within the budget %.2f GiB (%s); idle slots reclaimed (LRU) when needed, else the request waits; the "
                  "largest job (-c %d%s) %.2f GiB", e->o.kv_mem, e->mem_src, e->o.ctx, e->o.spec_used ? ", speculating" : "", e->job_max / G);
    double A = mem_read(e->ma_file); // (not mem_avail: the engine thread's)
    if (A >= 0)
        l += snprintf(buf + l, n - l, "; live budget: MemAvailable (%.2f GiB%s) - margin %.2f GiB - the running jobs' growth", A / G,
                      e->ma_file ? ", RUNQ_TEST_MEMAVAILABLE" : "", e->o.admit_margin);
    else l += snprintf(buf + l, n - l, "; no live budget (no MemAvailable)");
    if (st > 0) l += snprintf(buf + l, n - l, "; a request's guesses at most %d", e->kcap);
    if (e->o.admit == ADMIT_CHUNK) {
        double fr = e->sw_dir ? dk_free(e) : -1;
        l += snprintf(buf + l, n - l, "; a chunk per running request kept free at admission; a renewal denied: the request is parked, and "
                      "swapped out to %s (%.2f GiB free) if the oldest one's renewal needs its memory", e->sw_dir ? e->sw_dir : "?",
                      fr >= 0 ? fr / G : 0.0);
        if (e->o.admit_yield > 0 && l < (int)n)
            snprintf(buf + l, n - l, "; a request yields its seat after %d chunks (%d tokens) when others wait for one", e->o.admit_yield,
                     e->o.admit_yield * e->o.admit_chunk);
    }
    return 1;
}

int engine_submit(Engine *e, Job *j) {
    pthread_mutex_lock(&e->mu);
    int r = 0;
    if (e->stop) r = -2;
    else if (e->qn >= e->o.queue + (e->parallel - e->running)) r = -1; // (jobs that start at the next step do not wait)
    else {
        if (e->o.admit != ADMIT_WORST && e->sm.state + e->dsm.state > 0) {
            // the request's guesses ("speculative": k, lookup) at most the most whose state copies one -c job can hold
            // (engine_start): one -c job always fits. Speculation changes only the speed, never the output
            int c = e->kcap;
            if (j->spec_k > c) j->spec_k = c;
            if (j->spec_lookup > c) j->spec_lookup = c;
        }
        j->refs++; // (no other thread knows the job yet)
        j->t_submit = rq_now();
        j->state = JOB_QUEUED;
        e->q[e->qn++] = j;
        pthread_cond_signal(&e->cv);
    }
    pthread_mutex_unlock(&e->mu);
    return r;
}

int engine_mem(Engine *e, EngineMem *out) {
    if (e->o.admit == ADMIT_WORST) return 0;
    pthread_mutex_lock(&e->mu);
    *out = e->pub;
    pthread_mutex_unlock(&e->mu);
    return 1;
}

int engine_queued(Engine *e, int *running) {
    pthread_mutex_lock(&e->mu);
    int n = e->qn;
    if (running) *running = e->running;
    pthread_mutex_unlock(&e->mu);
    return n;
}

const ConVocab *engine_vocab(Engine *e) { return e->vocab; }

void engine_stop(Engine *e, double grace) {
    if (!e) return;
    pthread_mutex_lock(&e->mu);
    e->stop = 1;
    e->grace_until = rq_now() + grace;
    for (int i = 0; i < e->qn; i++) { // queued jobs: rejected
        job_finish(e->q[i], FINISH_NONE, 503, "shutting_down", "the server is shutting down");
        job_release(e->q[i]);
    }
    e->qn = 0;
    pthread_cond_broadcast(&e->cv);
    pthread_mutex_unlock(&e->mu);
    if (e->started) pthread_join(e->th, NULL);
    e->started = 0;
}
#endif
