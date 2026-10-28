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
    RqPassCost tcost, dcost;
    int gmax, dmaxb;
    float *slg, *dlogits;
    RqSegment *dseg;
    Run **dsegrun;
    struct Cand *cand;
    double pm[2], pl[RQ_LOOKUP_LMAX + 1][2];
    unsigned long long sp_passes, sp_rows, sp_guess[3], sp_acc[3], d_passes, d_rows;
    double t_draft, t_cal;
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

static void end_run(Engine *e, Run *r, int finish, int status, int err, const char *emsg) {
    // the job is over (finish reason, an HTTP status for interruptions, or an RQ_ERR_*): flush its text,
    // publish the result, free its slot (a prefix cache from now on)
    Job *j = r->j;
    Slot *sl = &e->slot[r->si];
    if (r->vn > 0 && verify_done(e, r, 1) != RQ_OK) { // (the cache can no longer be trusted)
        rq_seq_reset(sl->seq);
        if (!status && err == RQ_OK) err = RQ_ERR_STATE, emsg = "speculative decoding: the rollback failed";
    }
    spec_release(e, r);
    if (e->hybrid && !status && err == RQ_OK) ck_reply_end(e, r);
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
    // prompt's prefix (the copy source's draft, or its own common prefix with the prompt)
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
    if (r->sk > 0) {
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
    if ((err = spec_start(e, r, src, L)) != RQ_OK) {
        end_run(e, r, 0, 0, err, NULL);
        return;
    }
    if (e->o.verbose) {
        char cp[48] = "";
        if (src >= 0) snprintf(cp, sizeof(cp), " (copied from slot %d)", src);
        fprintf(stderr, "[engine] job -> slot %d: %d of %d prompt tokens cached%s, %d running", si, L, n, cp, e->nrun);
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
    if (real_reuse(li, n)) {
        // slot affinity: a slot holding a longer conversation than it would keep (it shares only a prefix, e.g. the system
        // prompt with a new session or a compaction's next request) stays for that conversation's next request; the
        // prefix is copied into the least recently used other idle slot (the same bits, rq_seq_copy)
        int lost = rq_seq_len(e->slot[ib].seq) - li, other = -1;
        for (int i = 0; i < e->nslot; i++)
            if (i != ib && !e->slot[i].run && (other < 0 || e->slot[i].used < e->slot[other].used)) other = i;
        if (e->o.copy_min > 0 && other >= 0 && lost >= e->o.copy_min && lost > li) start_run(e, j, other, ib, li);
        else start_run(e, j, ib, -1, li);
    } else start_run(e, j, lru, -1, rq_seq_keep(e->slot[lru].seq, rq_seq_common_prefix(e->slot[lru].seq, P, n)));
    return 1;
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
    int R = rbase, kmax = 0;
    double S = sbase, T;
    for (int i = 0; i < nc; i++) {
        if (c[i].src == RQ_SPEC_PLAIN) continue;
        R += c[i].n;
        S += c[i].n * (double)c[i].L + c[i].n * (c[i].n + 1) / 2.0;
        if (c[i].src == RQ_SPEC_MODEL && c[i].n > kmax) kmax = c[i].n;
    }
    T = rq_pass_cost(&e->tcost, R, S);
    for (int p = 1; p <= kmax || (p == 1 && d0 > 0); p++) {
        int rows = p == 1 ? d0 : 0;
        double sp = p == 1 ? ds0 : 0;
        for (int i = 0; i < nc; i++) {
            if (c[i].src != RQ_SPEC_MODEL || c[i].n < p) continue;
            if (p == 1) {
                rows += c[i].lag + 1;
                sp += (c[i].lag + 1.0) * c[i].L - c[i].lag * (c[i].lag + 1) / 2.0;
            } else {
                rows++;
                sp += c[i].L + p - 1;
            }
        }
        T += rq_pass_cost(&e->dcost, rows, sp);
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
    double tdec = rq_pass_cost(&e->tcost, ndec, sdec), y = ndec;
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

static int draft_wanted(const Engine *e, const Run *r, int L) {
    // the draft model may guess for this job at position L (so its sequence follows the target's): fixed, if the job has
    // a model k; auto, if a model pass beats plain decoding for the job alone there (the cost model, its acceptance)
    if (!e->dm || r->spec == SPEC_OFF || r->sk <= 0) return 0;
    if (r->spec == SPEC_FIXED) return 1;
    return rq_spec_choose(&e->tcost, &e->dcost, L, r->sk, acc_p(r->pm, e->pm, 0.6), 0, 0).src == RQ_SPEC_MODEL;
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
        c[i].L = P;
        c[i].lim = lim;
        c[i].nl = c[i].mk = c[i].lag = c[i].n = 0;
        c[i].src = RQ_SPEC_PLAIN;
        c[i].fixed = r->spec == SPEC_FIXED;
        r->dwant = draft_wanted(e, r, P);
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
            int dl = rq_seq_keep(d, rq_seq_common_prefix(d, toks, P)), mk = r->sk < lim ? r->sk : lim;
            if (mk > rq_seq_capacity(d) - P) mk = rq_seq_capacity(d) - P;
            if (dl >= 0 && P - dl + 1 <= dbud && mk > 0) {
                c[i].mk = mk;
                c[i].lag = P - dl;
                c[i].pmd = acc_p(r->pm, e->pm, 0.6);
            }
        }
        if (c[i].fixed) {
            if (c[i].nl > 0) c[i].src = RQ_SPEC_LOOKUP, c[i].n = c[i].nl;
            else if (c[i].mk > 0) c[i].src = RQ_SPEC_MODEL, c[i].n = c[i].mk;
        }
    }
}

static int draft_pass(Engine *e, int nds, int drows, int nsel, float **lg) {
    // one pass of the draft model over e->dseg (the decode path for a single row); on failure the segments' draft
    // sequences are emptied (a draft only ever guesses: the jobs go on without it)
    double t0 = rq_now();
    int err = nds == 1 && drows == 1 && nsel == 1 ? rq_decode(e->dc, e->dseg[0].seq, e->dseg[0].tokens[0], lg)
                                                  : rq_forward_batch(e->dc, e->dseg, nds, nsel > 1 ? e->dlogits : NULL, lg);
    e->t_draft += rq_now() - t0;
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

static void spec_draft(Engine *e, int dbud) {
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
    int room = dbud - drows;
    for (int i = 0; i < e->nrun && room > 0; i++) { // (the catch-up rows)
        Run *r = e->run[i];
        if (r->done || r->src == RQ_SPEC_MODEL || r->spec == SPEC_OFF || r->sk <= 0 || !e->dm) continue;
        Slot *sl = &e->slot[r->si];
        RqSeq *d = sl->dseq, *s = sl->seq;
        const int *ref;
        int nref, nfull;
        if (r->phase == R_PREFILL) {
            if (!draft_wanted(e, r, r->j->prompt.n)) continue;
            ref = r->j->prompt.v;
            nfull = r->j->prompt.n;
            nref = rq_seq_len(s) + r->seg; // (the prompt up to this pass's chunk)
        } else {
            if (!r->dwant) continue;
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
        if (r->done) continue;
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
            if (r->done || r->phase != R_PREFILL) continue;
            int left = r->feed.n - r->fed - r->seg, k = left < budget ? left : budget;
            if (round == 0 && k > share) k = share;
            r->seg += k;
            rows += k;
            budget -= k;
        }
    // the draft model's passes (its guesses; its sequences following the targets'), then the guesses into the segments
    if (e->dm) spec_draft(e, ndec ? e->o.chunk : e->o.batch);
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

    int nsel = 0, err;
    for (int i = 0; i < nseg; i++) nsel += e->seg[i].logits == RQ_LOGITS_ALL ? e->seg[i].n : e->seg[i].logits == RQ_LOGITS_LAST;
    float *lg = NULL;
    if (nseg == 1 && rows == 1 && nsel == 1) // (one decode row: the decode path)
        err = rq_decode(e->c, e->seg[0].seq, e->seg[0].tokens[0], &lg);
    else
        err = rq_forward_batch(e->c, e->seg, nseg, nsel > 1 ? e->logits : NULL, &lg);
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
        rq_tokens_free(&r->dbuf);
        rq_lookup_free(r->lk);
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
    if (e->o.verbose) {
        fprintf(stderr, "[engine] %llu passes, %llu rows\n", e->passes, e->rows);
        if (e->sp_passes || e->d_passes)
            fprintf(stderr, "[engine] speculation: %llu passes with guesses (%llu rows); model %llu / %llu guesses accepted, lookup "
                    "%llu / %llu; draft %llu passes, %llu rows, %.3f s\n", e->sp_passes, e->sp_rows, e->sp_acc[RQ_SPEC_MODEL],
                    e->sp_guess[RQ_SPEC_MODEL], e->sp_acc[RQ_SPEC_LOOKUP], e->sp_guess[RQ_SPEC_LOOKUP], e->d_passes, e->d_rows,
                    e->t_draft);
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
    free(e->cand);
    if (e->dc) rq_ctx_free(e->dc);
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
    if (o->spec_used) e->spec_res = (double)e->parallel * (o->spec_kmax + 1) * ((double)e->sm.state + (double)e->dsm.state);
    if (e->nslot * per_slot > mem - e->spec_res) e->nslot = (int)((mem - e->spec_res) / per_slot);
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
        if (!e->dc || !e->dseg || !e->dsegrun || !e->dlogits) {
            snprintf(err, errlen, "out of memory (the draft model's compute context)");
            engine_free(e);
            return NULL;
        }
    }
    e->slot = calloc(e->nslot, sizeof(Slot));
    e->q = calloc(e->o.queue + e->parallel, sizeof(Job *));
    e->run = calloc(e->parallel, sizeof(Run *));
    e->seg = calloc(e->parallel, sizeof(RqSegment));
    e->segrun = calloc(e->parallel, sizeof(Run *));
    e->logits = malloc((size_t)(e->parallel + e->gmax) * e->V * sizeof(float));
    e->sseq = calloc(e->parallel, sizeof(RqSeq *));
    e->scfg = calloc(e->parallel, sizeof(RqSampling));
    e->stok = calloc(e->parallel, sizeof(int));
    e->srun = calloc(e->parallel, sizeof(Run *));
    e->vocab = con_vocab_new(m);
    if (!e->c || !e->slot || !e->q || !e->run || !e->seg || !e->segrun || !e->logits || !e->sseq || !e->scfg || !e->stok || !e->srun || !e->vocab ||
        !e->slg || !e->cand) {
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
        if ((rc = rq_pass_cost_measure(e->c, e->slot[0].seq, L, 3, &e->tcost)) != RQ_OK ||
            (e->dm && (rc = rq_pass_cost_measure(e->dc, e->slot[0].dseq, L, 3, &e->dcost)) != RQ_OK))
            fprintf(stderr, "runq serve: the speculation cost calibration failed (%s): estimates instead\n", rq_strerror(rc));
        if (e->hybrid) rq_seq_set_checkpoints(e->slot[0].seq, e->ckn);
        if (e->dhybrid) rq_seq_set_checkpoints(e->slot[0].dseq, e->ckn);
        e->t_cal = rq_now() - t0;
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

static void cost_text(char *b, size_t n, const RqPassCost *c) {
    snprintf(b, n, "%.2f ms + %.3f us x position per row (%s), f(2/4/8/16) %.2f / %.2f / %.2f / %.2f", c->t_w * 1e3, c->att * 1e6,
             c->calibrated ? "measured" : "estimated", rq_pass_cost(c, 2, 0) / c->t_w, rq_pass_cost(c, 4, 0) / c->t_w,
             rq_pass_cost(c, 8, 0) / c->t_w, rq_pass_cost(c, 16, 0) / c->t_w);
}

void engine_describe_spec(Engine *e, char *buf, size_t n) {
    // the cost model, and the choice for one sequence alone by position (the priors: the model's guesses accepted at 0.6,
    // a lookup match of 4 tokens)
    char t[200], d[200] = "";
    cost_text(t, sizeof(t), &e->tcost);
    if (e->dm) cost_text(d, sizeof(d), &e->dcost);
    int l = snprintf(buf, n, "costs: target %s%s%s (calibration %.1f s); one sequence alone", t, e->dm ? "; draft " : "", d, e->t_cal);
    const int Ls[] = { 512, 4096, 16384, 32768, 65536, 98304 };
    for (int i = 0; i < 6 && Ls[i] <= e->o.ctx && l < (int)n; i++) {
        RqSpecChoice c = rq_spec_choose(&e->tcost, e->dm ? &e->dcost : NULL, Ls[i], 3, 0.6, 3, rq_lookup_prior(4));
        if (c.src == RQ_SPEC_PLAIN) l += snprintf(buf + l, n - l, "%s at %d: plain", i ? "," : "", Ls[i]);
        else l += snprintf(buf + l, n - l, "%s at %d: %s %d (x%.2f)", i ? "," : "", Ls[i], c.src == RQ_SPEC_MODEL ? "model k" : "lookup",
                           c.n, c.gain);
    }
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
