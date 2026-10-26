/* Chat template rendering (Qwen3's ChatML, as the CLI's multi-turn chat keeps the conversation in the
 * KV cache) and the generation loop (rq_generate). Only the public API underneath (runq.h), except
 * the model's templates.
 *
 * The conversation as token pieces (models/Qwen3-0.6B/tokenizer_config.json; the exporter's .template*
 * files hold its renderings):
 *   [system]  "<|im_start|>system\n" SYS "<|im_end|>\n"                       (= the start of
 *             .template.with-system, i.e. that file minus .template)
 *   turn i    P_i = "<|im_start|>user\n" U_i "<|im_end|>\n<|im_start|>assistant\n" [+ G]
 *             (= .template / .template.with-thinking; G = "<think>\n\n</think>\n\n" with thinking off)
 *             then the reply R_i (as generated: token ids, without the final <|im_end|>)
 *             then E = "<|im_end|>\n" (template: '<|im_end|>' + '\n' after every message)
 * In the history, a reply appears as H_i: thinking off: R_i as generated (the empty think block G
 * stays in P_i: HF's template drops it from earlier turns, the only difference from its rendering);
 * thinking on: R_i after its last </think>, leading newlines stripped, as the template renders
 * earlier assistant turns (it drops their reasoning). Text pieces are encoded separately; they are
 * separated by added tokens (<|im_start|>, <|im_end|>), so this is the same as encoding the whole
 * rendered string.
 * Variants (runq.h, RQ_CHAT_*): Instruct-2507 has no think block (G is empty in both modes, thinking is
 * treated as off everywhere: replies stay verbatim in the history, as its template renders them);
 * Thinking-2507's P_i ends with "<think>\n" with thinking on (G = "<think>\n\n</think>\n\n" with it off,
 * not in its template), history as the hybrid's, except that an answered turn's P_i loses the "<think>\n"
 * (rq_chat_user_history: HF renders earlier assistant turns without it; the reply is re-prefilled from its
 * answer anyway, as with the hybrid's thinking). */
#include "internal.h"

static int chat_mode(const RqModel *m, int thinking) {
    // the thinking mode a call renders: Instruct-2507 has none
    return m->chat == RQ_CHAT_INSTRUCT ? 0 : thinking != 0;
}

const char *rq_strerror(int err) {
    switch (err) {
        case RQ_OK: return "ok";
        case RQ_ERR_ARG: return "invalid argument";
        case RQ_ERR_NOMEM: return "out of memory";
        case RQ_ERR_CONTEXT: return "context full";
        case RQ_ERR_STATE: return "not possible in this state";
        case RQ_ERR_LOAD: return "cannot load the model";
        default: return "unknown error";
    }
}

double rq_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int render_tokens(const RqModel *m, const char *tmpl, const char *arg, RqTokens *out) {
    // the tokens of tmpl with its one %s replaced by arg, appended to out
    int n = snprintf(NULL, 0, tmpl, arg);
    char *s = n < 0 ? NULL : malloc((size_t)n + 1);
    if (!s) return RQ_ERR_NOMEM;
    snprintf(s, (size_t)n + 1, tmpl, arg);
    int e = rq_tokenize(m, s, n, out);
    free(s);
    return e;
}

static int tmpl_ok(const RqModel *m, int thinking) {
    return !m->has_tok ? RQ_ERR_STATE : m->tmpl_missing[thinking][0] ? RQ_ERR_LOAD : RQ_OK;
}

int rq_chat_system(const RqModel *m, const char *text, int thinking, RqTokens *out) {
    thinking = chat_mode(m, thinking);
    int e = tmpl_ok(m, thinking);
    return e != RQ_OK ? e : render_tokens(m, m->sys_tmpl[thinking], text, out);
}

int rq_chat_user(const RqModel *m, const char *text, int thinking, RqTokens *out) {
    thinking = chat_mode(m, thinking);
    int e = tmpl_ok(m, thinking);
    return e != RQ_OK ? e : render_tokens(m, m->user_tmpl[thinking], text, out);
}

int rq_chat_end(const RqModel *m, RqTokens *out) { return rq_tokenize(m, "<|im_end|>\n", -1, out); }

int rq_chat_user_history(const RqModel *m, const char *text, int thinking, RqTokens *out) {
    thinking = chat_mode(m, thinking);
    if (m->chat != RQ_CHAT_THINKING || !thinking) return rq_chat_user(m, text, thinking, out);
    int e = tmpl_ok(m, thinking);
    return e != RQ_OK ? e : render_tokens(m, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", text, out);
}

int rq_chat_assistant(const RqModel *m, int thinking, RqTokens *out) {
    if (!m->has_tok) return RQ_ERR_STATE;
    char s[64];
    snprintf(s, sizeof(s), "<|im_start|>assistant\n%s", chat_gen_suffix(m, chat_mode(m, thinking)));
    return rq_tokenize(m, s, -1, out);
}

int rq_chat_reply_history(const RqModel *m, const int *r, int n, int thinking, RqTokens *out) {
    // thinking: the reply as the template renders an earlier assistant turn: the text after the last
    // </think> (if any), leading "\n" stripped (content.split('</think>')[-1].lstrip('\n')); a token
    // of newlines + text is re-encoded without the newlines
    if (!chat_mode(m, thinking)) return rq_tokens_push(out, r, n);
    RqTokens te = { 0 };
    int e = rq_tokenize(m, "</think>", -1, &te);
    int think_end = e == RQ_OK && te.n == 1 ? te.v[0] : -1;
    rq_tokens_free(&te);
    if (e != RQ_OK) return e;
    int i = 0;
    for (int j = n - 1; j >= 0 && think_end >= 0; j--)
        if (r[j] == think_end) { i = j + 1; break; }
    if (i == 0) return rq_tokens_push(out, r, n); // no reasoning block: as generated
    for (; i < n; i++) {
        int len;
        const char *s = rq_token_bytes(m, r[i], &len);
        int k = (int)strspn(s, "\n");
        if (k == len) continue; // only newlines
        if (k > 0) { // newlines + text in one token: re-encode the text
            if ((e = rq_tokenize(m, s + k, -1, out)) != RQ_OK) return e;
            i++;
        }
        break;
    }
    return rq_tokens_push(out, r + i, n - i);
}

int rq_chat_render(const RqModel *m, const RqMessage *msgs, int n, int thinking, RqTokens *out) {
    // [system] + per user message P_i, per assistant message H_i (from its text) + E
    int i = 0, e = RQ_OK, expect = RQ_ROLE_USER, users = 0;
    thinking = chat_mode(m, thinking);
    if (n > 0 && msgs[0].role == RQ_ROLE_SYSTEM)
        e = rq_chat_system(m, msgs[i++].content, thinking, out);
    for (; i < n && e == RQ_OK; i++) {
        if (msgs[i].role != expect || !msgs[i].content) return RQ_ERR_ARG;
        if (expect == RQ_ROLE_USER) { // (answered: the history form)
            e = i + 1 < n ? rq_chat_user_history(m, msgs[i].content, thinking, out) : rq_chat_user(m, msgs[i].content, thinking, out);
            expect = RQ_ROLE_ASSISTANT;
            users++;
            continue;
        }
        const char *c = msgs[i].content, *p;
        if (thinking) // the text after the last </think>, leading newlines stripped
            for (const char *q = c; (p = strstr(q, "</think>")); q = p + 8)
                c = p + 8 + strspn(p + 8, "\n");
        e = rq_tokenize(m, c, -1, out);
        if (e == RQ_OK)
            e = rq_chat_end(m, out);
        expect = RQ_ROLE_USER;
    }
    return e == RQ_OK && users == 0 ? RQ_ERR_ARG : e;
}

int rq_generate(RqCtx *c, RqSeq *s, const int *tokens, int n, const RqSampling *cfg, int max_new,
                const RqCallbacks *cb, RqGenStats *st) {
    // the CLI chat's loop: prefill what is new, then sample, report, feed back; ends at an end token
    // (sampled, not fed: a next turn's prefill starts with it, so it is in the cache before the next
    // turn), at max_new tokens, at the end of the context (the token that does not fit is reported
    // but not fed), or when on_token asks to stop
    RqGenStats z;
    if (!st) st = &z;
    memset(st, 0, sizeof(*st));
    double t0 = rq_now();
    float *logits = NULL;
    int e = rq_prefill(c, s, tokens, n, RQ_LOGITS_LAST, NULL, &logits);
    if (e != RQ_OK) return e;
    st->n_prompt = n;
    if (cfg->temperature != 0)
        for (int i = 0; i < n - 1; i++)
            random_f32(rq_seq_rng(s));
    double t_prefill = rq_now();
    st->t_prefill = t_prefill - t0;
    const RqModel *m = s->m;
    while (1) {
        if (cb && cb->on_logits)
            cb->on_logits(cb->user, rq_seq_len(s) - 1, logits);
        int next = rq_sample(c, s, logits, cfg);
        if (next < 0) { e = next; break; }
        if (rq_is_end_token(m, next)) {
            st->stop = RQ_STOP_END;
            break;
        }
        if (cb && cb->on_token && cb->on_token(cb->user, next)) {
            st->stop = RQ_STOP_CALLBACK;
            break;
        }
        if (rq_seq_len(s) >= rq_seq_capacity(s)) {
            st->stop = RQ_STOP_CONTEXT;
            break;
        }
        int last = max_new > 0 && st->n_gen + 1 >= max_new;
        e = rq_decode(c, s, next, last ? NULL : &logits);
        if (e != RQ_OK) break;
        st->n_gen++;
        if (last) {
            st->stop = RQ_STOP_LIMIT;
            break;
        }
    }
    st->t_gen = rq_now() - t_prefill;
    return e;
}
