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
 * answer anyway, as with the hybrid's thinking). Qwen3.5 (models/Qwen3.5-0.8B/chat_template.jinja): with thinking
 * on, Thinking-2507's form ("<think>\n" opens the reply, answered turns lose it); off, the hybrid's (the empty
 * think block G); its template trims every message text (|trim: Python's str.strip()), so do rq_chat_system /
 * user / render here (rq_chat_trim). */
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

static int is_py_space(uint32_t c) {
    // Python's str.isspace() (what Jinja's trim strips): \t \n \v \f \r, U+001C..U+001F, space, U+0085, U+00A0,
    // U+1680, U+2000..U+200A, U+2028, U+2029, U+202F, U+205F, U+3000
    return (c >= 9 && c <= 13) || (c >= 0x1c && c <= 0x20) || c == 0x85 || c == 0xa0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}

static int utf8_at(const unsigned char *p, const unsigned char *end, uint32_t *c) {
    // the code point at p (its byte length), or 0 if p does not start a valid sequence of at most 3 bytes
    // (the spaces above are all below U+10000)
    if (p >= end) return 0;
    if (p[0] < 0x80) { *c = p[0]; return 1; }
    if ((p[0] & 0xe0) == 0xc0 && end - p >= 2 && (p[1] & 0xc0) == 0x80) { *c = (p[0] & 0x1fu) << 6 | (p[1] & 0x3f); return 2; }
    if ((p[0] & 0xf0) == 0xe0 && end - p >= 3 && (p[1] & 0xc0) == 0x80 && (p[2] & 0xc0) == 0x80) {
        *c = (p[0] & 0x0fu) << 12 | (p[1] & 0x3fu) << 6 | (p[2] & 0x3f);
        return 3;
    }
    return 0;
}

const char *rq_chat_trim(const RqModel *m, const char *text, size_t *len) {
    const unsigned char *p = (const unsigned char *)text, *end = p + strlen(text);
    uint32_t c;
    int k;
    if (m->chat == RQ_CHAT_QWEN35) {
        while ((k = utf8_at(p, end, &c)) && is_py_space(c)) p += k;
        for (const unsigned char *q = end; q > p;) { // the last code point: back over continuation bytes
            const unsigned char *r = q - 1;
            while (r > p && (*r & 0xc0) == 0x80 && q - r < 3) r--;
            if (!((k = utf8_at(r, q, &c)) && r + k == q && is_py_space(c))) break;
            end = q = r;
        }
    }
    *len = (size_t)(end - p);
    return (const char *)p;
}

static int render_tokens(const RqModel *m, const char *tmpl, const char *arg0, RqTokens *out) {
    // the tokens of tmpl with its one %s replaced by arg (rq_chat_trim'd), appended to out
    size_t al;
    const char *arg = rq_chat_trim(m, arg0, &al);
    char *cp = NULL;
    if (al != strlen(arg0)) { // (trimmed: a copy)
        if (!(cp = malloc(al + 1))) return RQ_ERR_NOMEM;
        memcpy(cp, arg, al);
        cp[al] = 0;
        arg = cp;
    }
    int n = snprintf(NULL, 0, tmpl, arg);
    char *s = n < 0 ? NULL : malloc((size_t)n + 1);
    if (!s) {
        free(cp);
        return RQ_ERR_NOMEM;
    }
    snprintf(s, (size_t)n + 1, tmpl, arg);
    free(cp);
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
    if ((m->chat != RQ_CHAT_THINKING && m->chat != RQ_CHAT_QWEN35) || !thinking) return rq_chat_user(m, text, thinking, out);
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

int rq_chat_checkpoint_pos(const RqModel *m, const int *tokens, int n) {
    RqTokens t = { 0 };
    int e = rq_tokenize(m, "<|im_start|>", -1, &t), id = e == RQ_OK && t.n == 1 ? t.v[0] : -1;
    rq_tokens_free(&t);
    for (int i = n - 1; i >= 0 && id >= 0; i--)
        if (tokens[i] == id) return i;
    return -1;
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
        size_t cl;
        const char *c = rq_chat_trim(m, msgs[i].content, &cl), *p, *ce = c + cl;
        if (thinking) // the text after the last </think>, leading newlines stripped
            for (const char *q = c; (p = strstr(q, "</think>")) && p + 8 <= ce; q = p + 8)
                c = p + 8 + strspn(p + 8, "\n");
        e = rq_tokenize(m, c, c < ce ? (long)(ce - c) : 0, out);
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
