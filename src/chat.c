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
 * user / render here (rq_chat_trim). Qwen3.8 (tests/golden/templates/qwen3.8.jinja, the 27B GGUF's, Unsloth's; Qwen's
 * own, qwen3.8-hf.jinja, renders the same but has no developer role / merging / 'high'): Qwen3.5's user turns and trimming; with thinking on, the reasoning-
 * effort instruction (chat_effort) opens the system block (alone without a system message: rq_chat_system(NULL));
 * leading system / developer messages are merged ("\n"); earlier replies keep their reasoning (the template's
 * preserve_thinking, on by default): H_i = G_i + R_i as generated (G_i: the generation suffix of P_i, which the
 * history form of the user turn leaves out: the template renders the think block with the reply, in both modes,
 * "<think>\n" REASONING "\n</think>\n\n" ANSWER), so the cache never diverges from the history. */
#include "internal.h"

static int chat_mode(const RqModel *m, int thinking) {
    // the thinking mode a call renders: Instruct-2507 has none
    return m->chat == RQ_CHAT_INSTRUCT ? 0 : thinking != 0;
}

// Qwen3.8's reasoning-effort instructions (its template's strings; medium: none)
#define QWEN38_XHIGH "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, " \
    "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."
#define QWEN38_LOW "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion " \
    "without unnecessary elaboration."

static const char *chat_effort(const RqModel *m, int thinking) {
    // the instruction a Qwen3.8 system block starts with, or NULL
    if (m->chat != RQ_CHAT_QWEN38 || !thinking || thinking == RQ_THINK_MEDIUM) return NULL;
    return thinking == RQ_THINK_LOW ? QWEN38_LOW : QWEN38_XHIGH;
}

int rq_chat_effort(const char *name) {
    if (!name) return -1;
    return !strcmp(name, "xhigh") || !strcmp(name, "high") ? RQ_THINK_ON : !strcmp(name, "medium") ? RQ_THINK_MEDIUM
         : !strcmp(name, "low") ? RQ_THINK_LOW : -1;
}

const char *rq_chat_effort_name(int thinking) {
    return !thinking ? "off" : thinking == RQ_THINK_MEDIUM ? "medium" : thinking == RQ_THINK_LOW ? "low" : "xhigh";
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
    if (m->chat == RQ_CHAT_QWEN35 || m->chat == RQ_CHAT_QWEN38) {
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

static int tok_parts(const RqModel *m, const char *const *part, const size_t *len, int n, RqTokens *out) {
    // the tokens of the concatenated parts (one piece)
    size_t t = 0;
    for (int i = 0; i < n; i++) t += len[i];
    char *s = malloc(t + 1);
    if (!s) return RQ_ERR_NOMEM;
    for (size_t i = 0, o = 0; i < (size_t)n; o += len[i], i++) memcpy(s + o, part[i], len[i]);
    s[t] = 0;
    int e = rq_tokenize(m, s, (long)t, out);
    free(s);
    return e;
}

static int qwen38_system(const RqModel *m, const char *text, size_t tl, int thinking, RqTokens *out) {
    // Qwen3.8's system block of the (trimmed) text, tl bytes (0: no system message): the effort instruction first
    const char *ins = chat_effort(m, thinking);
    if (!tl && !ins) return RQ_OK;
    const char *part[5] = { "<|im_start|>system\n", ins ? ins : "", ins && tl ? "\n\n" : "", text ? text : "", "<|im_end|>\n" };
    size_t len[5] = { strlen(part[0]), strlen(part[1]), strlen(part[2]), tl, strlen(part[4]) };
    return tok_parts(m, part, len, 5, out);
}

int rq_chat_system(const RqModel *m, const char *text, int thinking, RqTokens *out) {
    int mode = chat_mode(m, thinking);
    int e = tmpl_ok(m, mode);
    if (e != RQ_OK) return e;
    if (m->chat == RQ_CHAT_QWEN38) {
        size_t tl = 0;
        const char *t = text ? rq_chat_trim(m, text, &tl) : NULL;
        return qwen38_system(m, t, tl, thinking, out);
    }
    return text ? render_tokens(m, m->sys_tmpl[mode], text, out) : RQ_OK;
}

int rq_chat_user(const RqModel *m, const char *text, int thinking, RqTokens *out) {
    thinking = chat_mode(m, thinking);
    int e = tmpl_ok(m, thinking);
    return e != RQ_OK ? e : render_tokens(m, m->user_tmpl[thinking], text, out);
}

int rq_chat_end(const RqModel *m, RqTokens *out) { return rq_tokenize(m, "<|im_end|>\n", -1, out); }

int rq_chat_user_history(const RqModel *m, const char *text, int thinking, RqTokens *out) {
    thinking = chat_mode(m, thinking);
    if (m->chat == RQ_CHAT_QWEN38) thinking = 1; // (the assistant header alone in both modes: the reply's piece has the rest)
    if ((m->chat != RQ_CHAT_THINKING && m->chat != RQ_CHAT_QWEN35 && m->chat != RQ_CHAT_QWEN38) || !thinking)
        return rq_chat_user(m, text, thinking, out);
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
    // of newlines + text is re-encoded without the newlines. Qwen3.8: the generation suffix + the reply as generated
    if (m->chat == RQ_CHAT_QWEN38) {
        int e = rq_tokenize(m, chat_gen_suffix(m, chat_mode(m, thinking)), -1, out);
        return e != RQ_OK ? e : rq_tokens_push(out, r, n);
    }
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

static int qwen38_reply(const RqModel *m, const char *text, RqTokens *out) {
    // Qwen3.8: an earlier assistant message (without <|im_end|>) as its template renders reasoning_content R and content C:
    // "<think>\n" trim(R) "\n</think>\n\n" trim(C), from a text whose reasoning ends at its last </think> (R: the part
    // before it, after a <think> if any; no </think>: R empty, C the whole text)
    size_t tl, rl = 0, cl;
    const char *t = rq_chat_trim(m, text, &tl), *te = t + tl, *r = t, *re = t, *c = t, *p;
    for (const char *q = t; (p = strstr(q, "</think>")) && p + 8 <= te; q = p + 8) {
        re = p;
        c = p + 8;
    }
    if (re > t) {
        for (const char *q = t; (p = strstr(q, "<think>")) && p + 7 <= re; q = p + 7)
            r = p + 7;
        char *rc = malloc((size_t)(re - r) + 1);
        if (!rc) return RQ_ERR_NOMEM;
        memcpy(rc, r, (size_t)(re - r));
        rc[re - r] = 0;
        size_t off = (size_t)(rq_chat_trim(m, rc, &rl) - rc);
        free(rc);
        r += off;
    }
    char *cc = malloc((size_t)(te - c) + 1);
    if (!cc) return RQ_ERR_NOMEM;
    memcpy(cc, c, (size_t)(te - c));
    cc[te - c] = 0;
    size_t coff = (size_t)(rq_chat_trim(m, cc, &cl) - cc);
    free(cc);
    c += coff;
    const char *part[4] = { "<think>\n", r, "\n</think>\n\n", c };
    size_t len[4] = { 8, rl, 11, cl };
    return tok_parts(m, part, len, 4, out);
}

int rq_chat_reply(const RqModel *m, const char *text, int thinking, RqTokens *out) {
    if (!m->has_tok) return RQ_ERR_STATE;
    if (m->chat == RQ_CHAT_QWEN38) return qwen38_reply(m, text, out);
    size_t cl;
    const char *c = rq_chat_trim(m, text, &cl), *p, *ce = c + cl;
    if (chat_mode(m, thinking)) // the text after the last </think>, leading newlines stripped
        for (const char *q = c; (p = strstr(q, "</think>")) && p + 8 <= ce; q = p + 8)
            c = p + 8 + strspn(p + 8, "\n");
    return rq_tokenize(m, c, c < ce ? (long)(ce - c) : 0, out);
}

int rq_chat_render(const RqModel *m, const RqMessage *msgs, int n, int thinking, RqTokens *out) {
    // [system] + per user message P_i, per assistant message H_i (from its text) + E
    int i = 0, e = RQ_OK, expect = RQ_ROLE_USER, users = 0, effort = thinking;
    thinking = chat_mode(m, thinking);
    if (m->chat == RQ_CHAT_QWEN38) { // the leading system messages, merged; the effort instruction
        size_t tot = 0, l;
        for (; i < n && msgs[i].role == RQ_ROLE_SYSTEM; i++) {
            if (!msgs[i].content) return RQ_ERR_ARG;
            rq_chat_trim(m, msgs[i].content, &l);
            tot += l + 1;
        }
        char *sys = malloc(tot + 1);
        if (!sys) return RQ_ERR_NOMEM;
        size_t sl = 0;
        for (int j = 0; j < i; j++) {
            const char *t = rq_chat_trim(m, msgs[j].content, &l);
            if (!l) continue;
            if (sl) sys[sl++] = '\n';
            memcpy(sys + sl, t, l);
            sl += l;
        }
        sys[sl] = 0;
        e = qwen38_system(m, sys, sl, effort, out);
        free(sys);
    } else if (n > 0 && msgs[0].role == RQ_ROLE_SYSTEM)
        e = rq_chat_system(m, msgs[i++].content, thinking, out);
    for (; i < n && e == RQ_OK; i++) {
        if (msgs[i].role != expect || !msgs[i].content) return RQ_ERR_ARG;
        if (expect == RQ_ROLE_USER) { // (answered: the history form)
            e = i + 1 < n ? rq_chat_user_history(m, msgs[i].content, thinking, out) : rq_chat_user(m, msgs[i].content, thinking, out);
            expect = RQ_ROLE_ASSISTANT;
            users++;
            continue;
        }
        e = rq_chat_reply(m, msgs[i].content, thinking, out);
        if (e == RQ_OK)
            e = rq_chat_end(m, out);
        expect = RQ_ROLE_USER;
    }
    return e == RQ_OK && users == 0 ? RQ_ERR_ARG : e;
}

void rq_gen_start(RqGen *g, RqSeq *s, const RqSampling *cfg, int max_new, const RqCallbacks *cb, int n) {
    // the RNG: one draw per prefilled token but the last (a reply sampled after a prompt fed in one prefill
    // draws as one fed token by token, sampling after each, would)
    g->s = s;
    g->cfg = cfg;
    g->cb = cb;
    g->max_new = max_new;
    g->n_gen = 0;
    g->token = -1;
    g->stop = -1;
    if (cfg->temperature != 0)
        for (int i = 0; i < n - 1; i++)
            random_f32(rq_seq_rng(s));
}

int rq_gen_step(RqGen *g, RqCtx *c, int pos, float *logits) {
    // one token of the loop: report the logits, sample, then the stop checks in rq_generate's order
    RqSeq *s = g->s;
    const RqCallbacks *cb = g->cb;
    if (cb && cb->on_logits)
        cb->on_logits(cb->user, pos, logits);
    int next = rq_sample(c, s, logits, g->cfg);
    if (next < 0) return next;
    g->token = next;
    if (rq_is_end_token(s->m, next)) {
        g->stop = RQ_STOP_END;
        return RQ_GEN_DONE;
    }
    if (cb && cb->on_token && cb->on_token(cb->user, next)) {
        g->stop = RQ_STOP_CALLBACK;
        return RQ_GEN_DONE;
    }
    if (pos + 1 >= rq_seq_capacity(s)) {
        g->stop = RQ_STOP_CONTEXT;
        return RQ_GEN_DONE;
    }
    g->n_gen++;
    if (g->max_new > 0 && g->n_gen >= g->max_new) {
        g->stop = RQ_STOP_LIMIT;
        return RQ_GEN_LAST;
    }
    return RQ_GEN_NEXT;
}

unsigned long long rq_logits_hash(const float *logits, int n) {
    const unsigned char *b = (const unsigned char *)logits;
    unsigned long long h = 1469598103934665603ull;
    for (size_t i = 0; i < (size_t)n * sizeof(float); i++)
        h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

int rq_generate(RqCtx *c, RqSeq *s, const int *tokens, int n, const RqSampling *cfg, int max_new,
                const RqCallbacks *cb, RqGenStats *st) {
    // the CLI chat's loop: prefill what is new, then sample, report, feed back (rq_gen_step); ends at an
    // end token (sampled, not fed: a next turn's prefill starts with it, so it is in the cache before the
    // next turn), at max_new tokens, at the end of the context (the token that does not fit is reported
    // but not fed), or when on_token asks to stop
    RqGenStats z;
    if (!st) st = &z;
    memset(st, 0, sizeof(*st));
    double t0 = rq_now();
    float *logits = NULL;
    int e = rq_prefill(c, s, tokens, n, RQ_LOGITS_LAST, NULL, &logits);
    if (e != RQ_OK) return e;
    st->n_prompt = n;
    RqGen g;
    rq_gen_start(&g, s, cfg, max_new, cb, n);
    double t_prefill = rq_now();
    st->t_prefill = t_prefill - t0;
    while (1) {
        int r = rq_gen_step(&g, c, rq_seq_len(s) - 1, logits);
        if (r < 0) { e = r; break; }
        if (r == RQ_GEN_DONE) break;
        e = rq_decode(c, s, g.token, r == RQ_GEN_LAST ? NULL : &logits);
        if (e != RQ_OK) break;
        st->n_gen++;
        if (r == RQ_GEN_LAST) break;
    }
    if (e == RQ_OK) st->stop = g.stop; // (an error: RQ_STOP_END, as before)
    st->t_gen = rq_now() - t_prefill;
    return e;
}
