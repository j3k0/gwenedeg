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
 * eviction). The whole prompt cached: its last token is prefilled again, for the logits.
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
    double used;          // last use (LRU)
    Run *run;             // the running job on it, or NULL: idle (a prefix cache)
    unsigned char *allow; // constrained decoding: the token mask (V bytes, all 0 between steps)
    int *list;            // the allowed tokens set in allow
    float *lcopy;         // logprobs: the raw logits of the step (sampling modifies them)
} Slot;


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
    int done;             // finished: the slot is free, reaped after the step
    ConState cs;
    Out o;
};

static int slot_scratch(Engine *e, Slot *sl, int logprobs) {
    // the slot's mask / list (constrained jobs) and logits copy (logprobs), allocated at first use
    if (!sl->allow && !(sl->allow = calloc(e->V, 1))) return RQ_ERR_NOMEM;
    if (!sl->list && !(sl->list = malloc(e->V * sizeof(int)))) return RQ_ERR_NOMEM;
    if (logprobs && !sl->lcopy && !(sl->lcopy = malloc(e->V * sizeof(float)))) return RQ_ERR_NOMEM;
    return RQ_OK;
}

static void end_run(Engine *e, Run *r, int finish, int status, int err, const char *emsg) {
    // the job is over (finish reason, an HTTP status for interruptions, or an RQ_ERR_*): flush its text,
    // publish the result, free its slot (a prefix cache from now on)
    Job *j = r->j;
    Slot *sl = &e->slot[r->si];
    for (int i = 0; i < r->nallow; i++) sl->allow[sl->list[i]] = 0;
    r->nallow = 0;
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
    sl->run = NULL;
    sl->used = rq_now();
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
    if (j->n_gen >= j->max_new || rq_seq_len(sl->seq) >= rq_seq_capacity(sl->seq)) { end_run(e, r, FINISH_LENGTH, 0, RQ_OK, NULL); return; }
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
    job_unlock(j);
    sl->used = now;
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
    if (src < 0 || rq_seq_copy(s, e->slot[src].seq, L) != RQ_OK) {
        if (src >= 0) L = 0;
        rq_seq_truncate(s, L);
    }
    // (Qwen3.5: copy / truncate keep a checkpointed length <= L; the next request of this conversation diverges
    // after its assistant header, so the state there is kept, rq_chat_checkpoint_pos)
    L = rq_seq_len(s);
    int ckp = rq_chat_checkpoint_pos(m, j->prompt.v, n);
    if (ckp >= L) rq_seq_checkpoint_at(s, ckp);
    if (e->o.verbose) {
        char from[48] = "";
        if (src >= 0) snprintf(from, sizeof(from), " (copied from slot %d)", src);
        fprintf(stderr, "[engine] job -> slot %d: %d of %d prompt tokens cached%s, %d running\n", si, L, n, from, e->nrun);
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
    // (under e->mu) start the job on a slot: 1, or -1 if no slot is idle, 0 if it should wait for a
    // prefix that a job being prefilled will share with it
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
    if (j->no_cache) { // (the least recently used slot: keeps the useful caches)
        start_run(e, j, lru, -1, 0);
        return 1;
    }
    if (e->o.copy_min > 0) {
        int have = li > lb ? li : lb, pend = 0;
        for (int k = 0; k < e->nrun; k++) {
            Run *r = e->run[k];
            if (r->done || r->phase != R_PREFILL) continue;
            int sh = lcp(r->j->prompt.v, r->j->prompt.n, P, n);
            if (sh > rq_seq_len(e->slot[r->si].seq) && sh > pend) pend = sh;
        }
        // (Qwen3.5: no waiting for a prefix being prefilled: its state is gone once the prefill moves past it)
        if (!e->hybrid && pend >= have + e->o.copy_min && real_reuse(pend, n)) return 0;
        if (bb >= 0 && lb >= li + e->o.copy_min && real_reuse(lb, n)) {
            start_run(e, j, lru, bb, lb);
            return 1;
        }
    }
    if (real_reuse(li, n)) start_run(e, j, ib, -1, li);
    else start_run(e, j, lru, -1, rq_seq_keep(e->slot[lru].seq, rq_seq_common_prefix(e->slot[lru].seq, P, n)));
    return 1;
}

// ------------------------------------------------------------------------------------------- a step

static void step(Engine *e, int stopping) {
    // one batched pass over every running job, then the sampling of the rows with logits
    int V = e->V, nseg = 0, rows = 0, ndec = 0;
    double now = rq_now();
    for (int i = 0; i < e->nrun; i++) { // (client gone, deadline, shutdown past the grace period)
        Run *r = e->run[i];
        r->seg = 0;
        r->row = -1;
        if (r->done) continue;
        Job *j = r->j;
        int st = __atomic_load_n(&j->cancel, __ATOMIC_ACQUIRE) ? 499 : j->deadline > 0 && now > j->deadline ? 504 : stopping ? 503 : 0;
        if (st) end_run(e, r, 0, st, RQ_OK, NULL);
    }
    int npre = 0;
    for (int i = 0; i < e->nrun; i++) { // decode rows first: every decoding job's next token(s)
        Run *r = e->run[i];
        if (r->done) continue;
        if (r->phase == R_PREFILL) {
            npre++;
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
    }
    // then prompt tokens within the budget (--chunk while jobs decode, else -b): first up to an equal share per
    // prompt (FIFO; a short prompt does not wait for a long one's whole prefill), then the rest FIFO
    int budget = ndec ? e->o.chunk : e->o.batch;
    if (budget > e->maxb - rows) budget = e->maxb - rows;
    int share = npre ? budget / npre : 0;
    if (share < 8) share = 8;
    for (int round = 0; round < 2 && budget > 0; round++)
        for (int i = 0; i < e->nrun && budget > 0; i++) {
            Run *r = e->run[i];
            if (r->done || r->phase != R_PREFILL) continue;
            int left = r->feed.n - r->fed - r->seg, k = left < budget ? left : budget;
            if (round == 0 && k > share) k = share;
            r->seg += k;
            rows += k;
            budget -= k;
        }
    for (int i = 0; i < e->nrun; i++) { // (the segments in admission order)
        Run *r = e->run[i];
        if (r->done || r->seg == 0) continue;
        e->seg[nseg] = (RqSegment) { e->slot[r->si].seq, r->feed.v + r->fed, r->seg,
                                     r->fed + r->seg == r->feed.n ? RQ_LOGITS_LAST : RQ_LOGITS_NONE };
        e->segrun[nseg++] = r;
    }
    if (nseg == 0) return;

    int nsel = 0, err;
    for (int i = 0; i < nseg; i++) nsel += e->seg[i].logits == RQ_LOGITS_LAST;
    float *lg = NULL;
    if (nseg == 1 && rows == 1 && nsel == 1) // (one decode row: the decode path)
        err = rq_decode(e->c, e->seg[0].seq, e->seg[0].tokens[0], &lg);
    else
        err = rq_forward_batch(e->c, e->seg, nseg, nsel > 1 ? e->logits : NULL, &lg);
    if (err == RQ_OK) {
        for (int i = 0, sel = 0; i < nseg; i++)
            if (e->seg[i].logits == RQ_LOGITS_LAST) e->segrun[i]->row = sel++;
    } else { // each segment alone (nothing was computed: the passes check everything first): only a failing job fails
        char msg[300];
        for (int i = 0, sel = 0; i < nseg; i++) {
            Run *r = e->segrun[i];
            int want = e->seg[i].logits == RQ_LOGITS_LAST;
            int ei = nseg == 1 ? err : rq_forward_batch(e->c, &e->seg[i], 1, want ? e->logits + (size_t)sel * V : NULL, NULL);
            if (ei != RQ_OK) {
                snprintf(msg, sizeof(msg), "%s", rq_ctx_error(e->c)[0] ? rq_ctx_error(e->c) : rq_strerror(ei));
                end_run(e, r, 0, 0, ei, msg);
                continue;
            }
            if (want) r->row = sel++;
        }
        lg = e->logits;
    }
    e->passes++;
    e->rows += rows;

    // the rows with logits: prompts just prefilled (their first token), decode rows
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
        }
        if (j->top_logprobs >= 0 && slot_scratch(e, sl, 1) != RQ_OK) {
            end_run(e, r, 0, 0, RQ_ERR_NOMEM, NULL);
            continue;
        }
        if (r->row != ns) memcpy(lg + (size_t)ns * V, lg + (size_t)r->row * V, V * sizeof(float));
        if (j->top_logprobs >= 0) memcpy(sl->lcopy, lg + (size_t)ns * V, V * sizeof(float));
        e->sseq[ns] = sl->seq;
        e->scfg[ns] = (RqSampling) { j->temperature, j->top_p, j->con ? sl->allow : NULL, j->top_k };
        e->srun[ns++] = r;
    }
    if (ns > 0) {
        err = rq_sample_batch(e->c, e->sseq, lg, ns, e->scfg, ns, e->stok);
        for (int k = 0; k < ns; k++) take_token(e, e->srun[k], err == RQ_OK ? e->stok[k] : err);
    }
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
        rq_tokens_free(&r->feed);
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
        for (int i = 0; i < e->qn && e->nrun < e->parallel;) { // FIFO (a job waiting for a shared prefix keeps its place)
            int a = admit(e, e->q[i]);
            if (a < 0) break;
            if (a == 0) {
                i++;
                continue;
            }
            memmove(&e->q[i], &e->q[i + 1], (size_t)(e->qn - i - 1) * sizeof(Job *));
            e->qn--;
        }
        int stopping = e->stop && now > e->grace_until;
        e->running = e->nrun;
        pthread_mutex_unlock(&e->mu);
        if (e->nrun > 0) step(e, stopping);
        reap(e);
        pthread_mutex_lock(&e->mu);
        e->running = e->nrun;
    }
    pthread_mutex_unlock(&e->mu);
    if (e->o.verbose) fprintf(stderr, "[engine] %llu passes, %llu rows\n", e->passes, e->rows);
    return NULL;
}

void engine_free(Engine *e) {
    if (!e) return;
    for (int i = 0; e->slot && i < e->nslot; i++) {
        rq_seq_free(e->slot[i].seq);
        free(e->slot[i].allow);
        free(e->slot[i].list);
        free(e->slot[i].lcopy);
    }
    free(e->slot);
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
    if (e->o.queue < 1) e->o.queue = 1;
    if (e->o.batch < 1) e->o.batch = 1;
    if (e->o.batch > RQ_BATCH_MAX) e->o.batch = RQ_BATCH_MAX;
    if (e->o.chunk < 1) e->o.chunk = 64;
    if (e->o.chunk > e->o.batch) e->o.chunk = e->o.batch;
    RqModelInfo info;
    rq_model_info(m, &info);
    e->V = info.vocab_size;
    e->eos = info.eos;
    e->chat = info.chat;
    e->hybrid = info.linear_layers > 0;
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
    double per_slot = (double)e->sm.total, mem = o->kv_mem > 0 ? o->kv_mem * 1073741824.0 : 0;
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
    if (e->nslot * per_slot > mem) e->nslot = (int)(mem / per_slot);
    if (e->nslot < 1) {
        snprintf(err, errlen, "-c %d needs %.2f GiB per sequence (KV cache and recurrent state), more than the pool's %.2f GiB (%s)", o->ctx,
                 per_slot / 1073741824.0, mem / 1073741824.0, e->mem_src);
        engine_free(e);
        return NULL;
    }
    if (e->parallel > e->nslot) e->parallel = e->nslot;
    e->o.kv_mem = mem / 1073741824.0;
    e->maxb = e->o.batch > e->o.chunk + e->parallel ? e->o.batch : e->o.chunk + e->parallel;
    if (e->maxb > RQ_BATCH_MAX) e->maxb = RQ_BATCH_MAX;
    if (e->maxb < 2) e->maxb = 2;
    e->c = rq_ctx_new(m, e->maxb, o->ctx, &rc);
    e->slot = calloc(e->nslot, sizeof(Slot));
    e->q = calloc(e->o.queue + e->parallel, sizeof(Job *));
    e->run = calloc(e->parallel, sizeof(Run *));
    e->seg = calloc(e->parallel, sizeof(RqSegment));
    e->segrun = calloc(e->parallel, sizeof(Run *));
    e->logits = malloc((size_t)e->parallel * e->V * sizeof(float));
    e->sseq = calloc(e->parallel, sizeof(RqSeq *));
    e->scfg = calloc(e->parallel, sizeof(RqSampling));
    e->stok = calloc(e->parallel, sizeof(int));
    e->srun = calloc(e->parallel, sizeof(Run *));
    e->vocab = con_vocab_new(m);
    if (!e->c || !e->slot || !e->q || !e->run || !e->seg || !e->segrun || !e->logits || !e->sseq || !e->scfg || !e->stok || !e->srun || !e->vocab) {
        snprintf(err, errlen, "out of memory (compute context)");
        engine_free(e);
        return NULL;
    }
    for (int i = 0; i < e->nslot; i++) {
        e->slot[i].seq = rq_seq_new(m, o->ctx, o->kv_type, &rc);
        if (!e->slot[i].seq) {
            snprintf(err, errlen, "cannot create a sequence of %d positions: %s", o->ctx, rq_strerror(rc));
            engine_free(e);
            return NULL;
        }
    }
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
    snprintf(buf, n, "%d running at most, %d slot(s) of %d positions (per slot %.1f MiB: KV %.0f KB per position%s; %.2f GiB at most, "
             "touched as used; pool budget %.2f GiB, %s), prefill chunks %d / %d tokens per pass (others decoding / not)",
             e->parallel, e->nslot, e->o.ctx, e->sm.total / 1048576.0, (double)e->sm.kv / e->o.ctx / 1000.0, st,
             (double)e->sm.total * e->nslot / G, e->o.kv_mem, e->mem_src, e->o.chunk, e->o.batch);
}

int engine_submit(Engine *e, Job *j) {
    pthread_mutex_lock(&e->mu);
    int r = 0;
    if (e->stop) r = -2;
    else if (e->qn >= e->o.queue + (e->parallel - e->running)) r = -1; // (jobs that start at the next step do not wait)
    else {
        j->refs++; // (no other thread knows the job yet)
        j->t_submit = rq_now();
        j->state = JOB_QUEUED;
        e->q[e->qn++] = j;
        pthread_cond_signal(&e->cv);
    }
    pthread_mutex_unlock(&e->mu);
    return r;
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
