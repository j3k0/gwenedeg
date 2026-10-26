/* runq serve's engine (serve.h): one thread runs the Jobs of a FIFO queue, one at a time, on one compute
 * context (the library runs one forward pass at a time per process). The connection threads only queue
 * Jobs and read their output, so /health and queued requests never wait on a running generation.
 *
 * Prefix KV reuse: a few cached sequences (slots, --slots). A job takes the slot sharing the longest
 * token prefix with its prompt (if that is a real reuse: >= 32 tokens or half the prompt; else the least
 * recently used slot), truncates it to that prefix and prefills only the rest. The positions and tokens
 * are the same as in a fresh sequence, and the forward passes do not depend on chunking, so the logits
 * are bit-identical to a fresh prefill (tests/server_test.py checks it through the logprobs).
 *
 * Generation, per job: prefill, then sample / decode until an end token, max tokens, a stop string, the
 * end of the context, a cancel (client gone) or the deadline. The RNG is seeded per request and draws
 * once per prompt token but the last, as the CLI's chat does for a first turn (rq_generate), however much
 * of the prompt was cached: a seeded request gives the same text as `runq -m chat -s SEED` on the same
 * single-turn conversation. With a json_schema constraint (constrain.c) the forced tokens are appended
 * without a forward pass of their own: they are fed with the next decision's step (one batched pass), or
 * never if the output is then complete.
 *
 * Step 10 replaces engine_main / run_job with a scheduler (several jobs decoding in one batched pass);
 * the Job interface stays.
 */
#ifndef _WIN32
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include "serve.h"

struct Engine {
    RqModel *m;
    RqCtx *c;
    RqSeq **slot;
    double *slot_used;
    EngineOpts o;
    ConVocab *vocab;
    int V, eos, think_start, think_end;
    unsigned char *allow;
    int *list;
    float *lcopy;
    pthread_t th;
    int started;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    Job **q;
    int qhead, qn;
    Job *running;
    int stop;
    double grace_until;
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
    pthread_condattr_t ca;
    pthread_condattr_init(&ca);
    pthread_condattr_setclock(&ca, CLOCK_MONOTONIC);
    pthread_cond_init(&s->cv, &ca);
    pthread_condattr_destroy(&ca);
    j->mu = s;
    j->cv = s;
    j->refs = 1;
    j->top_logprobs = -1;
    return j;
}

void job_lock(Job *j) { pthread_mutex_lock(&((JobSync *)j->mu)->mu); }
void job_unlock(Job *j) { pthread_mutex_unlock(&((JobSync *)j->mu)->mu); }

int job_wait(Job *j, double seconds) {
    JobSync *s = j->mu;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    long long ns = ts.tv_nsec + (long long)(seconds * 1e9);
    ts.tv_sec += ns / 1000000000LL;
    ts.tv_nsec = ns % 1000000000LL;
    return pthread_cond_timedwait(&s->cv, &s->mu, &ts) == 0;
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
    jb_raw(&o->text, b, n);
    pub_content(o, 0);
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

// ------------------------------------------------------------------------------------------ a job

static int pick_slot(Engine *e, const Job *j) {
    // the slot with the longest common prefix if it is a real reuse, else the least recently used
    int best = 0, bestL = -1, lru = 0;
    for (int i = 0; i < e->o.slots; i++) {
        int L = rq_seq_common_prefix(e->slot[i], j->prompt.v, j->prompt.n);
        if (L > bestL || (L == bestL && e->slot_used[i] > e->slot_used[best])) {
            best = i;
            bestL = L;
        }
        if (e->slot_used[i] < e->slot_used[lru]) lru = i;
    }
    if (e->o.slots == 1 || bestL >= 32 || (bestL > 0 && 2 * bestL >= j->prompt.n)) return best;
    return lru;
}

static int interrupted(Engine *e, Job *j, int *status) {
    // checked before each forward pass: client gone, deadline, shutdown past the grace period
    double now = rq_now();
    if (__atomic_load_n(&j->cancel, __ATOMIC_ACQUIRE)) { *status = 499; return 1; }
    if (j->deadline > 0 && now > j->deadline) { *status = 504; return 1; }
    pthread_mutex_lock(&e->mu);
    int stop = e->stop && now > e->grace_until;
    pthread_mutex_unlock(&e->mu);
    if (stop) { *status = 503; return 1; }
    return 0;
}

static void run_job(Engine *e, Job *j) {
    RqModel *m = e->m;
    job_lock(j);
    j->state = JOB_RUNNING;
    j->t_start = rq_now();
    job_unlock(j);
    int si = pick_slot(e, j), n = j->prompt.n, st = 0, err = RQ_OK, finish = FINISH_STOP;
    RqSeq *s = e->slot[si];
    j->slot = si;
    e->slot_used[si] = rq_now();
    const char *emsg = NULL;
    if (n < 1 || n >= rq_seq_capacity(s)) {
        job_finish(j, FINISH_NONE, 400, "context_length_exceeded", "the prompt does not fit in the context");
        return;
    }
    int L = j->no_cache ? 0 : rq_seq_common_prefix(s, j->prompt.v, n);
    if (L >= n) L = n - 1; // (the last prompt token is prefilled again: its logits)
    rq_seq_truncate(s, L);
    j->n_prompt = n;
    j->n_cached = L;
    rq_seq_seed(s, j->seed);
    if (j->temperature != 0) { // (rq_generate's draws for a prefill of the whole prompt: xorshift steps)
        unsigned long long *r = rq_seq_rng(s);
        for (int i = 0; i < n - 1; i++) {
            *r ^= *r >> 12;
            *r ^= *r << 25;
            *r ^= *r >> 27;
        }
    }

    Out o = { 0 };
    o.e = e;
    o.j = j;
    for (int i = 0; i < j->nstop; i++)
        if ((int)strlen(j->stop[i]) > o.maxstop) o.maxstop = (int)strlen(j->stop[i]);

    // a constrained reply starts with forced tokens (e.g. `{"tier": "`): they go into the prefill
    ConState cs;
    RqTokens pend = { 0 }, forced = { 0 }, pre = { 0 };
    if (j->con) {
        con_start(&cs, j->con);
        if ((err = con_forced(&cs, m, &forced)) != RQ_OK) goto out;
        if (cs.done) { // (one valid output: no forward pass at all)
            job_lock(j);
            for (int i = 0; i < forced.n && j->n_gen < j->max_new; i++) {
                out_token(&o, forced.v[i]);
                if (j->top_logprobs >= 0) logprobs_add(e, j, forced.v[i], NULL, NULL, 1);
                j->n_gen++;
            }
            job_unlock(j);
            finish = j->n_gen < forced.n ? FINISH_LENGTH : FINISH_STOP;
            goto out;
        }
        if (forced.n >= j->max_new || rq_seq_len(s) + n - L + forced.n > rq_seq_capacity(s))
            forced.n = 0, con_start(&cs, j->con); // (rare: no room; the loop below handles them)
    }
    if ((err = rq_tokens_push(&pre, j->prompt.v + L, n - L)) != RQ_OK || (err = rq_tokens_push(&pre, forced.v, forced.n)) != RQ_OK)
        goto out;

    // prefill (in chunks: a cancel is seen between them)
    float *logits = NULL;
    int chunk = rq_ctx_max_batch(e->c) > 1 ? rq_ctx_max_batch(e->c) : 64;
    for (int i = 0; i < pre.n && err == RQ_OK; i += chunk) {
        if (interrupted(e, j, &st)) goto out;
        int k = pre.n - i < chunk ? pre.n - i : chunk;
        err = rq_prefill(e->c, s, pre.v + i, k, i + k == pre.n ? RQ_LOGITS_LAST : RQ_LOGITS_NONE, NULL, &logits);
    }
    if (err != RQ_OK) goto out;
    j->t_first = rq_now();
    if (forced.n) {
        job_lock(j);
        for (int i = 0; i < forced.n; i++) {
            out_token(&o, forced.v[i]);
            if (j->top_logprobs >= 0) logprobs_add(e, j, forced.v[i], NULL, NULL, 1);
            j->n_gen++;
        }
        job_notify(j);
        job_unlock(j);
    }

    int cap = rq_seq_capacity(s);
    for (;;) {
        int nallow = 0;
        if (j->con) {
            forced.n = 0;
            if ((err = con_forced(&cs, m, &forced)) != RQ_OK) break;
            job_lock(j);
            for (int i = 0; i < forced.n && j->n_gen < j->max_new; i++) {
                out_token(&o, forced.v[i]);
                if (j->top_logprobs >= 0) logprobs_add(e, j, forced.v[i], NULL, NULL, 1);
                j->n_gen++;
                rq_tokens_push(&pend, &forced.v[i], 1);
            }
            job_notify(j);
            job_unlock(j);
            if (cs.done) { finish = FINISH_STOP; break; }
            if (j->n_gen >= j->max_new) { finish = FINISH_LENGTH; break; }
            nallow = con_mask(&cs, e->vocab, e->allow, e->list, e->V);
            if (nallow == 0) { err = RQ_ERR_STATE; emsg = "constrained decoding: no token continues the output"; break; }
        }
        if (pend.n > 0) { // feed the last sampled token (+ forced ones): the next logits
            if (rq_seq_len(s) + pend.n > cap) { finish = FINISH_LENGTH; break; }
            if (interrupted(e, j, &st)) break;
            err = pend.n == 1 ? rq_decode(e->c, s, pend.v[0], &logits)
                              : rq_prefill(e->c, s, pend.v, pend.n, RQ_LOGITS_LAST, NULL, &logits);
            if (err != RQ_OK) break;
            pend.n = 0;
        }
        if (j->top_logprobs >= 0) memcpy(e->lcopy, logits, e->V * sizeof(float));
        RqSampling cfg = { j->temperature, j->top_p, j->con ? e->allow : NULL };
        int tok = rq_sample(e->c, s, logits, &cfg);
        if (tok < 0) { err = tok; break; }
        job_lock(j);
        int end = j->con ? tok == e->eos : rq_is_end_token(m, tok);
        if (!end) {
            if (j->top_logprobs >= 0) logprobs_add(e, j, tok, e->lcopy, j->con ? e->allow : NULL, 0);
            out_token(&o, tok);
            j->n_gen++;
        }
        job_notify(j);
        job_unlock(j);
        for (int i = 0; i < nallow; i++) e->allow[e->list[i]] = 0;
        if (end) { finish = FINISH_STOP; break; }
        if (j->con) {
            int len;
            const char *b = rq_token_bytes(m, tok, &len);
            if (!con_advance(&cs, b, len)) { err = RQ_ERR_STATE; emsg = "constrained decoding: a disallowed token"; break; }
            if (cs.done) { finish = FINISH_STOP; break; }
        }
        if (o.stopped) { finish = FINISH_STOP; break; }
        if (j->n_gen >= j->max_new) { finish = FINISH_LENGTH; break; }
        if (rq_seq_len(s) >= cap) { finish = FINISH_LENGTH; break; }
        rq_tokens_push(&pend, &tok, 1);
    }
out:
    rq_tokens_free(&pend);
    rq_tokens_free(&forced);
    rq_tokens_free(&pre);
    job_lock(j);
    out_end(&o);
    job_unlock(j);
    jb_free(&o.text);
    if (st) {
        job_finish(j, FINISH_NONE, st, st == 499 ? "client_closed" : st == 504 ? "timeout" : "shutting_down",
                   st == 499 ? "the client closed the connection" : st == 504 ? "the request timed out" : "the server is shutting down");
    } else if (err != RQ_OK) {
        int ctxe = err == RQ_ERR_CONTEXT;
        job_finish(j, FINISH_NONE, ctxe ? 400 : 500, ctxe ? "context_length_exceeded" : "internal_error",
                   emsg ? emsg : rq_ctx_error(e->c)[0] ? rq_ctx_error(e->c) : rq_strerror(err));
    } else {
        job_finish(j, finish, 0, NULL, NULL);
    }
}

// ------------------------------------------------------------------------------------------ engine

static void *engine_main(void *arg) {
    Engine *e = arg;
    for (;;) {
        pthread_mutex_lock(&e->mu);
        while (e->qn == 0 && !e->stop) pthread_cond_wait(&e->cv, &e->mu);
        if (e->qn == 0) {
            pthread_mutex_unlock(&e->mu);
            break;
        }
        Job *j = e->q[e->qhead];
        e->qhead = (e->qhead + 1) % e->o.queue;
        e->qn--;
        e->running = j;
        pthread_mutex_unlock(&e->mu);
        run_job(e, j);
        pthread_mutex_lock(&e->mu);
        e->running = NULL;
        pthread_mutex_unlock(&e->mu);
        job_release(j);
    }
    return NULL;
}

void engine_free(Engine *e) {
    if (!e) return;
    for (int i = 0; e->slot && i < e->o.slots; i++) rq_seq_free(e->slot[i]);
    free(e->slot);
    free(e->slot_used);
    if (e->c) rq_ctx_free(e->c);
    con_vocab_free(e->vocab);
    free(e->allow);
    free(e->list);
    free(e->lcopy);
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
    if (e->o.slots < 1) e->o.slots = 1;
    if (e->o.queue < 1) e->o.queue = 1;
    RqModelInfo info;
    rq_model_info(m, &info);
    e->V = info.vocab_size;
    e->eos = info.eos;
    e->think_start = single_token(m, "<think>");
    e->think_end = single_token(m, "</think>");
    int rc;
    e->c = rq_ctx_new(m, o->batch, o->ctx, &rc);
    e->slot = calloc(e->o.slots, sizeof(RqSeq *));
    e->slot_used = calloc(e->o.slots, sizeof(double));
    e->q = calloc(e->o.queue, sizeof(Job *));
    e->allow = calloc(e->V, 1);
    e->list = malloc(e->V * sizeof(int));
    e->lcopy = malloc(e->V * sizeof(float));
    e->vocab = con_vocab_new(m);
    if (!e->c || !e->slot || !e->slot_used || !e->q || !e->allow || !e->list || !e->lcopy || !e->vocab) {
        snprintf(err, errlen, "out of memory (compute context)");
        engine_free(e);
        return NULL;
    }
    for (int i = 0; i < e->o.slots; i++) {
        e->slot[i] = rq_seq_new(m, o->ctx, o->kv_type, &rc);
        if (!e->slot[i]) {
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

int engine_submit(Engine *e, Job *j) {
    pthread_mutex_lock(&e->mu);
    int r = 0;
    if (e->stop) r = -2;
    else if (e->qn >= e->o.queue) r = -1;
    else {
        j->refs++; // (no other thread knows the job yet)
        j->t_submit = rq_now();
        j->state = JOB_QUEUED;
        e->q[(e->qhead + e->qn) % e->o.queue] = j;
        e->qn++;
        pthread_cond_signal(&e->cv);
    }
    pthread_mutex_unlock(&e->mu);
    return r;
}

int engine_queued(Engine *e, int *running) {
    pthread_mutex_lock(&e->mu);
    int n = e->qn;
    if (running) *running = e->running != NULL;
    pthread_mutex_unlock(&e->mu);
    return n;
}

const ConVocab *engine_vocab(Engine *e) { return e->vocab; }

void engine_stop(Engine *e, double grace) {
    if (!e) return;
    pthread_mutex_lock(&e->mu);
    e->stop = 1;
    e->grace_until = rq_now() + grace;
    while (e->qn > 0) { // queued jobs: rejected
        Job *j = e->q[e->qhead];
        e->qhead = (e->qhead + 1) % e->o.queue;
        e->qn--;
        job_finish(j, FINISH_NONE, 503, "shutting_down", "the server is shutting down");
        job_release(j);
    }
    pthread_cond_broadcast(&e->cv);
    pthread_mutex_unlock(&e->mu);
    if (e->started) pthread_join(e->th, NULL);
    e->started = 0;
}
#endif
