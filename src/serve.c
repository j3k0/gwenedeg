/* runq serve: the OpenAI-compatible layer (serve.h): routes, request validation, chat template
 * rendering, Jobs for the engine, JSON / SSE responses, OpenAI-style errors, one log line per request.
 *
 *   POST /v1/chat/completions   (also /chat/completions)
 *   GET  /v1/models, /v1/models/{id}   (also /models)
 *   GET  /health, /v1/health
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "serve.h"

#ifdef _WIN32
int serve_main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    fprintf(stderr, "runq serve: not available on Windows (the server uses POSIX sockets and threads)\n");
    return 1;
}

int serve_render(RqModel *m, const Js *body, RqTokens *out, int *thinking, char *err, size_t errlen) {
    (void)m;
    (void)body;
    (void)out;
    *thinking = 0;
    snprintf(err, errlen, "not available on Windows");
    return RQ_ERR_STATE;
}
#else
#include <unistd.h>
#include <strings.h>
#include <pthread.h>
#include "http.h"

typedef struct {
    RqModel *m;
    Engine *eng;
    RqModelInfo info;
    char model_id[256];
    const char *api_key;
    int ctx, verbose, def_max_tokens, def_topk;
    float def_temp, def_topp;
    double timeout;
    long long created;
    unsigned long long seq; // request counter (ids, seeds)
    pthread_mutex_t mu;
} App;

static void logline(App *a, HttpReq *r, int status, const Job *j, const char *note) {
    char ts[32];
    time_t t = time(NULL);
    struct tm tm;
    localtime_r(&t, &tm);
    strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tm);
    double now = rq_now();
    char extra[256] = "";
    if (j && j->t_start > 0)
        snprintf(extra, sizeof(extra), " prompt=%d cached=%d completion=%d queue=%.3fs prefill=%.3fs", j->n_prompt, j->n_cached,
                 j->n_gen, j->t_start - j->t_submit, j->t_first > 0 ? j->t_first - j->t_start : 0.0);
    fprintf(stderr, "[%s] %s %s %s %d%s total=%.3fs%s%s\n", ts, r->peer, r->method, r->target, status, extra, now - r->t0,
            note && note[0] ? " " : "", note ? note : "");
    (void)a;
}

// ------------------------------------------------------------------------------------------ errors

static const char *err_type(int status) {
    return status == 401 ? "authentication_error" : status == 404 ? "not_found_error" : status >= 500 ? "server_error" : "invalid_request_error";
}

static void error_json(JBuf *b, int status, const char *msg, const char *param, const char *code) {
    jb_cstr(b, "{\"error\": {\"message\": ");
    jb_strz(b, msg);
    jb_cstr(b, ", \"type\": ");
    jb_strz(b, err_type(status));
    jb_cstr(b, ", \"param\": ");
    if (param) jb_strz(b, param);
    else jb_cstr(b, "null");
    jb_cstr(b, ", \"code\": ");
    if (code && code[0]) jb_strz(b, code);
    else jb_cstr(b, "null");
    jb_cstr(b, "}}");
}

static void send_err(App *a, HttpConn *c, HttpReq *r, int status, const char *msg, const char *param, const char *code) {
    JBuf b = { 0 };
    error_json(&b, status, msg, param, code);
    const char *extra = status == 503 ? "Retry-After: 1\r\n" : status == 401 ? "WWW-Authenticate: Bearer\r\n" : NULL;
    http_respond(c, r, status, "application/json", b.p ? b.p : "", b.n, extra);
    logline(a, r, status, NULL, msg);
    jb_free(&b);
}

// ------------------------------------------------------------------------------------- GET routes

static void model_obj(App *a, JBuf *b) {
    jb_cstr(b, "{\"id\": ");
    jb_strz(b, a->model_id);
    jb_printf(b, ", \"object\": \"model\", \"created\": %lld, \"owned_by\": \"runq\", \"context_length\": %d}", a->created, a->ctx);
}

static void get_models(App *a, HttpConn *c, HttpReq *r, const char *id) {
    JBuf b = { 0 };
    if (id) {
        if (strcmp(id, a->model_id)) {
            char msg[512];
            snprintf(msg, sizeof(msg), "The model '%.200s' does not exist", id);
            send_err(a, c, r, 404, msg, "model", "model_not_found");
            return;
        }
        model_obj(a, &b);
    } else {
        jb_cstr(&b, "{\"object\": \"list\", \"data\": [");
        model_obj(a, &b);
        jb_cstr(&b, "]}");
    }
    http_respond(c, r, 200, "application/json", b.p, b.n, NULL);
    if (a->verbose) logline(a, r, 200, NULL, NULL);
    jb_free(&b);
}

static void get_health(App *a, HttpConn *c, HttpReq *r) {
    int running = 0, queued = engine_queued(a->eng, &running);
    JBuf b = { 0 };
    int st = http_shutting_down() ? 503 : 200;
    jb_printf(&b, "{\"status\": \"%s\", \"model\": ", st == 200 ? "ok" : "shutting down");
    jb_strz(&b, a->model_id);
    jb_printf(&b, ", \"running\": %d, \"queued\": %d}", running, queued);
    http_respond(c, r, st, "application/json", b.p, b.n, NULL);
    if (a->verbose) logline(a, r, st, NULL, NULL);
    jb_free(&b);
}

// ------------------------------------------------------------------------------ chat: the request

// (Msg, serve.h: role 0 system, 1 user, 2 assistant, 3 tool; reasoning: Qwen3.8's, or with tools Qwen3.5's too, else NULL)

typedef struct {
    int status;
    char msg[512];
    const char *param, *code;
} ReqErr;

static int bad(ReqErr *e, const char *param, const char *fmt, const char *arg) {
    e->status = 400;
    snprintf(e->msg, sizeof(e->msg), fmt, arg ? arg : "");
    e->param = param;
    return -1;
}

static char *content_text(const Js *c, const char *sep, ReqErr *e) {
    // a message's content: a string, an array of text parts (joined with sep: "\n"; with tools "", as the Qwen3.5 / 3.8
    // templates join them), or null (""); malloc'd
    JBuf b = { 0 };
    if (!c || c->t == JS_NULL) jb_cstr(&b, "");
    else if (c->t == JS_STR) jb_raw(&b, c->s, c->n);
    else if (c->t == JS_ARR) {
        jb_cstr(&b, "");
        for (int i = 0; i < c->n; i++) {
            const Js *p = c->v[i], *ty = json_get(p, "type"), *tx = json_get(p, "text");
            if (!json_str_eq(ty, "text") || !tx || tx->t != JS_STR) {
                jb_free(&b);
                bad(e, "messages", "only text content parts ({\"type\": \"text\", \"text\": ...}) are supported%s", NULL);
                return NULL;
            }
            if (i) jb_cstr(&b, sep);
            jb_raw(&b, tx->s, tx->n);
        }
    } else {
        bad(e, "messages", "message content must be a string or an array of text parts%s", NULL);
        return NULL;
    }
    if (b.oom) { jb_free(&b); bad(e, NULL, "out of memory%s", NULL); return NULL; }
    if (strlen(b.p) != b.n) {
        jb_free(&b);
        bad(e, "messages", "message content must not contain NUL characters%s", NULL);
        return NULL;
    }
    return b.p;
}

static int tok_text(const RqModel *m, const char *a, const char *b, const char *c, RqTokens *out) {
    // the tokens of a + b + c (one piece: the chat template's literal parts around a message)
    size_t n = strlen(a) + strlen(b) + strlen(c);
    char *s = malloc(n + 1);
    if (!s) return RQ_ERR_NOMEM;
    snprintf(s, n + 1, "%s%s%s", a, b, c);
    int e = rq_tokenize(m, s, (long)n, out);
    free(s);
    return e;
}

static const char *after_think(const char *s) {
    // an assistant message as the history keeps it: the text after the last </think>, leading newlines stripped
    for (const char *p; (p = strstr(s, "</think>"));) s = p + 8;
    return s + strspn(s, "\n");
}

static int render38(const RqModel *m, const Msg *msg, int n, int thinking, RqTokens *out) {
    // Qwen3.8's template (runq.h): the leading system / developer messages merged with "\n" (empty ones skipped) after
    // the effort instruction (parse_chat refused later ones, as its template does); every assistant message in its think
    // form (rq_chat_reply; its reasoning_content, if given, as the reasoning: "<think>" R "</think>" C), after the
    // assistant header of the user turn before it (rq_chat_user_history) or its own; other sequences as render's
    size_t tot = 1;
    int i = 0, e = RQ_OK, open = 0;
    for (; i < n && msg[i].role == 0; i++) tot += strlen(msg[i].text) + 1;
    char *sys = malloc(tot);
    if (!sys) return RQ_ERR_NOMEM;
    sys[0] = 0;
    for (int j = 0; j < i; j++)
        if (msg[j].text[0]) {
            if (sys[0]) strcat(sys, "\n");
            strcat(sys, msg[j].text); // (trimmed by parse_chat)
        }
    e = rq_chat_system(m, sys[0] ? sys : NULL, thinking, out);
    free(sys);
    for (; i < n && e == RQ_OK; i++) {
        const Msg *x = &msg[i];
        if (x->role == 1) {
            if (i == n - 1 || msg[i + 1].role == 2) {
                e = i == n - 1 ? rq_chat_user(m, x->text, thinking, out) : rq_chat_user_history(m, x->text, thinking, out);
                open = 1;
                continue;
            }
            e = tok_text(m, "<|im_start|>user\n", x->text, "<|im_end|>\n", out);
        } else {
            char *t = x->text;
            if (x->reasoning) { // (reasoning_content: the text rq_chat_reply splits back into it and the content)
                size_t l = strlen(x->reasoning) + strlen(x->text) + 16;
                if (!(t = malloc(l))) return RQ_ERR_NOMEM;
                snprintf(t, l, "<think>%s</think>%s", x->reasoning, x->text);
            }
            if (!open) e = tok_text(m, "<|im_start|>assistant\n", "", "", out);
            if (e == RQ_OK) e = rq_chat_reply(m, t, thinking, out);
            if (e == RQ_OK) e = rq_chat_end(m, out);
            if (t != x->text) free(t);
        }
        open = 0;
    }
    if (e == RQ_OK && !open) e = rq_chat_assistant(m, thinking, out);
    return e;
}

static int render(const RqModel *m, int chat, const Msg *msg, int n, int thinking, RqTokens *out) {
    // Qwen3's template in the CLI chat's canonical form (rq_chat_render: [system] then user / assistant
    // turns, the empty think block with thinking off kept in every user turn); other sequences as HF's
    // template renders them: a later system message as a system block, a user message not followed
    // by an assistant one as a plain block, an assistant message not after a user one with its own
    // header; ends with the assistant header (+ empty think block with thinking off). The variants (runq.h):
    // Instruct-2507 keeps assistant messages verbatim (its template does), no think block anywhere;
    // Thinking-2507's generation prompt opens the think block
    int i = 0, e = RQ_OK, open = 0; // open: the last piece ends with the assistant header
    if (chat == RQ_CHAT_QWEN38) return render38(m, msg, n, thinking, out);
    if (n > 0 && msg[0].role == 0) {
        e = rq_chat_system(m, msg[0].text, thinking, out);
        i = 1;
    }
    for (; i < n && e == RQ_OK; i++) {
        const Msg *x = &msg[i];
        if (x->role == 1) {
            if (i == n - 1 || msg[i + 1].role == 2) { // (answered: the history form, rq_chat_user_history)
                e = i == n - 1 ? rq_chat_user(m, x->text, thinking, out) : rq_chat_user_history(m, x->text, thinking, out);
                open = 1;
                continue;
            }
            e = tok_text(m, "<|im_start|>user\n", x->text, "<|im_end|>\n", out);
        } else if (x->role == 2) {
            const char *t = chat == RQ_CHAT_INSTRUCT ? x->text : after_think(x->text);
            e = open ? rq_tokenize(m, t, -1, out) : tok_text(m, "<|im_start|>assistant\n", t, "", out);
            if (e == RQ_OK) e = rq_chat_end(m, out);
        } else {
            e = tok_text(m, "<|im_start|>system\n", x->text, "<|im_end|>\n", out);
        }
        open = 0;
    }
    if (e == RQ_OK && !open) e = rq_chat_assistant(m, thinking, out);
    return e;
}

typedef struct {
    int stream, include_usage, logprobs, n_msgs;
    int tmode;        // tools, tool messages or earlier tool calls: rendered by tools_render (Qwen3.5 / 3.8)
    char model[256];
    Msg *msgs;
    ConSet *con;
    ToolSet *tools;
    int has_force, force_raw; // force_reply: given; a raw text (force.text) rather than a message
    Msg force;
} Chat;

static void chat_free(Chat *ch) {
    for (int i = 0; i < ch->n_msgs; i++) {
        free(ch->msgs[i].text);
        free(ch->msgs[i].reasoning);
        for (int k = 0; k < ch->msgs[i].ncalls; k++) {
            free(ch->msgs[i].calls[k].name);
            json_free(ch->msgs[i].calls[k].doc);
        }
        free(ch->msgs[i].calls);
    }
    free(ch->msgs);
    free(ch->force.text);
    free(ch->force.reasoning);
    for (int k = 0; k < ch->force.ncalls; k++) {
        free(ch->force.calls[k].name);
        json_free(ch->force.calls[k].doc);
    }
    free(ch->force.calls);
    con_free(ch->con);
    tools_free(ch->tools);
}

static int get_bool(const Js *root, const char *key, int *out, ReqErr *e) {
    const Js *v = json_get(root, key);
    if (!v || v->t == JS_NULL) return 0;
    if (v->t != JS_TRUE && v->t != JS_FALSE) return bad(e, key, "'%s' must be a boolean", key);
    *out = v->t == JS_TRUE;
    return 0;
}

static int get_num(const Js *root, const char *key, double lo, double hi, double *out, ReqErr *e) {
    const Js *v = json_get(root, key);
    if (!v || v->t == JS_NULL) return 0;
    if (v->t != JS_NUM || !(v->num >= lo && v->num <= hi)) {
        char m[160];
        snprintf(m, sizeof(m), "'%s' must be a number in [%g, %g]", key, lo, hi);
        return bad(e, key, "%s", m);
    }
    *out = v->num;
    return 0;
}

static int get_int(const Js *root, const char *key, long long lo, long long hi, long long *out, ReqErr *e) {
    const Js *v = json_get(root, key);
    long long x;
    if (!v || v->t == JS_NULL) return 0;
    if (!json_is_int(v, &x) || x < lo || x > hi) {
        char m[160];
        snprintf(m, sizeof(m), "'%s' must be an integer in [%lld, %lld]", key, lo, hi);
        return bad(e, key, "%s", m);
    }
    *out = x;
    return 0;
}

static int parse_calls(const Js *tc, Msg *x, ReqErr *e) {
    // an assistant message's tool_calls (an array or absent) into x->calls
    if (!tc || tc->t != JS_ARR || tc->n == 0) return 0;
    if (!(x->calls = calloc(tc->n, sizeof(MsgCall)))) return bad(e, NULL, "out of memory%s", NULL);
    for (int k = 0; k < tc->n; k++) {
        const Js *c = tc->v[k], *ty = json_get(c, "type"), *f = json_get(c, "function");
        const Js *nm = json_get(f, "name"), *ar = json_get(f, "arguments");
        if (c->t != JS_OBJ || (ty && ty->t != JS_NULL && !json_str_eq(ty, "function")) || !f || f->t != JS_OBJ)
            return bad(e, "messages", "each tool call must be {\"type\": \"function\", \"function\": {\"name\": ..., \"arguments\": ...}}%s", NULL);
        if (!nm || nm->t != JS_STR || nm->n == 0 || (int)strlen(nm->s) != nm->n)
            return bad(e, "messages", "a tool call's function.name must be a non-empty string%s", NULL);
        MsgCall *mc = &x->calls[x->ncalls++];
        if (!(mc->name = strdup(nm->s))) return bad(e, NULL, "out of memory%s", NULL);
        if (ar && ar->t == JS_STR) { // (OpenAI's form: a JSON text; "" or blank: none)
            char perr[200];
            size_t bl = strspn(ar->s, " \t\r\n");
            if ((int)bl < ar->n) {
                if (!(mc->doc = json_parse(ar->s, ar->n, perr, sizeof(perr))) || json_root(mc->doc)->t != JS_OBJ)
                    return bad(e, "messages", "a tool call's function.arguments must be a JSON object (or its text)%s", NULL);
                mc->args = json_root(mc->doc);
            }
        } else if (ar && ar->t == JS_OBJ) mc->args = ar;
        else if (ar && ar->t != JS_NULL)
            return bad(e, "messages", "a tool call's function.arguments must be a JSON object (or its text)%s", NULL);
    }
    return 0;
}

static int parse_force(App *a, const Js *root, Chat *ch, ReqErr *e) {
    // force_reply (a benchmark hook: scripts/replay_sessions.py): the reply as the model's raw text (a string), or an
    // assistant message (content, reasoning_content, tool_calls; Qwen3.5 / 3.8) that the reply is to be as the template
    // renders it; tokenized in render_req
    const Js *fr = json_get(root, "force_reply");
    if (!fr || fr->t == JS_NULL) return 0;
    ch->has_force = 1;
    if (fr->t == JS_STR) {
        if ((int)strlen(fr->s) != fr->n) return bad(e, "force_reply", "'force_reply' must not contain NUL characters%s", NULL);
        if (!(ch->force.text = strdup(fr->s))) return bad(e, NULL, "out of memory%s", NULL);
        ch->force_raw = 1;
        return 0;
    }
    if (fr->t != JS_OBJ) return bad(e, "force_reply", "'force_reply' must be a string or an assistant message object%s", NULL);
    if (a->info.chat != RQ_CHAT_QWEN35 && a->info.chat != RQ_CHAT_QWEN38)
        return bad(e, "force_reply", "'force_reply' as a message needs the Qwen3.5 / Qwen3.8 templates (a string works with any)%s", NULL);
    const Js *tc = json_get(fr, "tool_calls");
    if (tc && tc->t != JS_NULL && tc->t != JS_ARR) return bad(e, "force_reply", "'tool_calls' must be an array%s", NULL);
    if (parse_calls(tc, &ch->force, e)) return -1;
    if (!(ch->force.text = content_text(json_get(fr, "content"), "", e))) return -1;
    size_t tl;
    const char *t0 = rq_chat_trim(a->m, ch->force.text, &tl);
    memmove(ch->force.text, t0, tl);
    ch->force.text[tl] = 0;
    const Js *rc = json_get(fr, "reasoning_content");
    if (rc && rc->t == JS_STR && !(ch->force.reasoning = content_text(rc, "", e))) return -1;
    if ((ch->force.reasoning && strstr(ch->force.reasoning, "</think>")) || (ch->force.reasoning && strstr(ch->force.text, "</think>")))
        return bad(e, "force_reply", "a reply with reasoning_content must not contain </think>%s", NULL);
    ch->force.role = MSG_ASSISTANT;
    return 0;
}

static int parse_chat(App *a, const Js *root, Chat *ch, Job *j, ReqErr *e) {
    if (!root || root->t != JS_OBJ) return bad(e, NULL, "the request body must be a JSON object%s", NULL);
    const Js *v = json_get(root, "model");
    if (v && v->t == JS_STR) snprintf(ch->model, sizeof(ch->model), "%s", v->s);
    else snprintf(ch->model, sizeof(ch->model), "%s", a->model_id);

    // messages
    const Js *ms = json_get(root, "messages");
    if (!ms) return bad(e, "messages", "'messages' is required%s", NULL);
    if (ms->t != JS_ARR || ms->n == 0) return bad(e, "messages", "'messages' must be a non-empty array%s", NULL);
    // tool mode (tools_render): tools, a tool message or an earlier tool call; the Qwen3.5 / Qwen3.8 templates only
    int tvar = a->info.chat == RQ_CHAT_QWEN35 || a->info.chat == RQ_CHAT_QWEN38;
    const Js *tools = json_get(root, "tools");
    if (tools && tools->t != JS_NULL && tools->t != JS_ARR) return bad(e, "tools", "'tools' must be an array%s", NULL);
    int ntools = tools && tools->t == JS_ARR ? tools->n : 0;
    ch->tmode = ntools > 0;
    for (int i = 0; i < ms->n; i++) {
        const Js *m = ms->v[i], *role = json_get(m, "role"), *tc = json_get(m, "tool_calls");
        if (json_str_eq(role, "tool") || (json_str_eq(role, "assistant") && tc && tc->t == JS_ARR && tc->n > 0)) ch->tmode = 1;
    }
    if (ch->tmode && !tvar) {
        char msg[200];
        snprintf(msg, sizeof(msg), "tools / function calls are supported with the Qwen3.5 and Qwen3.8 chat templates only (this "
                 "model's: %s)", rq_chat_name(a->info.chat));
        return bad(e, ntools > 0 ? "tools" : "messages", "%s", msg);
    }
    const char *sep = ch->tmode ? "" : "\n";
    ch->msgs = calloc(ms->n, sizeof(Msg));
    if (!ch->msgs) return bad(e, NULL, "out of memory%s", NULL);
    for (int i = 0; i < ms->n; i++) {
        const Js *m = ms->v[i], *role = json_get(m, "role");
        if (!m || m->t != JS_OBJ || !role || role->t != JS_STR) return bad(e, "messages", "each message must be an object with a 'role'%s", NULL);
        int r = json_str_eq(role, "system") || json_str_eq(role, "developer") ? 0 : json_str_eq(role, "user") ? 1 : json_str_eq(role, "assistant") ? 2
              : json_str_eq(role, "tool") && tvar ? 3 : -1;
        if (r < 0) return bad(e, "messages", tvar ? "unsupported message role '%.40s' (system, developer, user, assistant, tool)"
                              : "unsupported message role '%.40s' (system, developer, user, assistant)", role->s);
        const Js *tc = json_get(m, "tool_calls");
        if (tc && tc->t != JS_NULL && (r != 2 || tc->t != JS_ARR)) return bad(e, "messages", "'tool_calls' must be an array, on an assistant message%s", NULL);
        if (parse_calls(tc, &ch->msgs[i], e)) return -1;
        const Js *c = json_get(m, "content");
        if (r != 2 && (!c || c->t == JS_NULL)) return bad(e, "messages", "a %s message needs 'content'", role->s);
        if (!(ch->msgs[i].text = content_text(c, sep, e))) return -1;
        size_t tl; // (Qwen3.5's template trims every message: rq_chat_trim, in place)
        const char *t0 = rq_chat_trim(a->m, ch->msgs[i].text, &tl);
        memmove(ch->msgs[i].text, t0, tl);
        ch->msgs[i].text[tl] = 0;
        ch->msgs[i].role = r;
        ch->n_msgs = i + 1;
        if (a->info.chat == RQ_CHAT_QWEN38 || ch->tmode) { // (its template: system / developer messages first; earlier reasoning kept)
            if (r == 0 && i > 0 && ch->msgs[i - 1].role != 0 && a->info.chat == RQ_CHAT_QWEN38)
                return bad(e, "messages", "system / developer messages must be at the beginning (Qwen3.8's chat template)%s", NULL);
            const Js *rc = json_get(m, "reasoning_content");
            if (r == 2 && rc && rc->t == JS_STR) {
                if (!(ch->msgs[i].reasoning = content_text(rc, sep, e))) return -1;
                if (strstr(ch->msgs[i].reasoning, "</think>") || strstr(ch->msgs[i].text, "</think>"))
                    return bad(e, "messages", "an assistant message with reasoning_content must not contain </think>%s", NULL);
            }
        }
    }
    if (parse_force(a, root, ch, e)) return -1;
    long long n = 1;
    if (get_int(root, "n", 1, 1000000, &n, e)) return -1;
    if (n != 1) return bad(e, "n", "only n = 1 is supported%s", NULL);
    const Js *fns = json_get(root, "functions");
    if (fns && fns->t != JS_NULL) return bad(e, "functions", "functions are not supported (use tools)%s", NULL);
    // tools: tool_choice auto (the default) or none (rendered, the reply not parsed); parallel_tool_calls false: one call
    int parse_calls = ntools > 0;
    const Js *tch = json_get(root, "tool_choice");
    if (tch && tch->t != JS_NULL) {
        if (json_str_eq(tch, "none")) parse_calls = 0;
        else if (!json_str_eq(tch, "auto"))
            return bad(e, "tool_choice", "tool_choice must be \"auto\" or \"none\" (\"required\" and a named function are not supported)%s", NULL);
    }
    if (ntools > 0) {
        char terr[300];
        if (!(ch->tools = tools_compile(tools, terr, sizeof(terr)))) return bad(e, "tools", "%s", terr);
        if (parse_calls) j->tools = ch->tools;
        int par = 1;
        if (get_bool(root, "parallel_tool_calls", &par, e)) return -1;
        j->single_call = !par;
        pthread_mutex_lock(&a->mu);
        unsigned long long k = ++a->seq;
        pthread_mutex_unlock(&a->mu);
        j->call_seed = (unsigned long long)time(NULL) * 1000003ULL + k * 0x9e3779b97f4a7c15ULL + (unsigned long long)(rq_now() * 1e6);
    }

    // sampling and limits
    double temp = a->def_temp, topp = a->def_topp;
    long long topk = a->def_topk; // (an extension, as vLLM's / llama.cpp's: -1 or 0 = off)
    if (get_num(root, "temperature", 0, 2, &temp, e) || get_num(root, "top_p", 0, 1, &topp, e) ||
        get_int(root, "top_k", -1, 1 << 30, &topk, e))
        return -1;
    j->temperature = (float)temp;
    j->top_p = (float)topp;
    j->top_k = topk > 0 ? (int)topk : 0;
    long long seed = 0, maxt = 0, maxc = 0;
    int has_seed = json_get(root, "seed") && json_get(root, "seed")->t != JS_NULL;
    if (get_int(root, "seed", -(1LL << 53), 1LL << 53, &seed, e)) return -1;
    pthread_mutex_lock(&a->mu);
    unsigned long long k = ++a->seq;
    pthread_mutex_unlock(&a->mu);
    if (has_seed && seed > 0) j->seed = (unsigned long long)seed; // (as runq -s: positive seeds as given)
    else {
        unsigned long long x = has_seed ? (unsigned long long)seed : (unsigned long long)time(NULL) * 1000003ULL + k + (unsigned long long)(rq_now() * 1e6);
        x += 0x9e3779b97f4a7c15ULL; // (splitmix64: a nonzero xorshift state)
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        j->seed = (x ^ (x >> 31)) | 1;
    }
    if (get_int(root, "max_tokens", 1, 1LL << 30, &maxt, e) || get_int(root, "max_completion_tokens", 1, 1LL << 30, &maxc, e)) return -1;
    j->max_new = (int)(maxc ? maxc : maxt ? maxt : a->def_max_tokens);
    const Js *stop = json_get(root, "stop");
    if (stop && stop->t != JS_NULL) {
        int ns = stop->t == JS_STR ? 1 : stop->t == JS_ARR ? stop->n : -1;
        if (ns < 0 || ns > 16) return bad(e, "stop", "'stop' must be a string or an array of at most 16 strings%s", NULL);
        j->stop = calloc(ns ? ns : 1, sizeof(char *));
        if (!j->stop) return bad(e, NULL, "out of memory%s", NULL);
        for (int i = 0; i < ns; i++) {
            const Js *s = stop->t == JS_STR ? stop : stop->v[i];
            if (s->t != JS_STR || s->n < 1 || s->n > 256 || (int)strlen(s->s) != s->n)
                return bad(e, "stop", "each stop sequence must be a non-empty string of at most 256 bytes%s", NULL);
            j->stop[j->nstop++] = strdup(s->s);
        }
    }
    if (get_bool(root, "stream", &ch->stream, e) || get_bool(root, "logprobs", &ch->logprobs, e)) return -1;
    const Js *so = json_get(root, "stream_options");
    if (so && so->t == JS_OBJ && get_bool(so, "include_usage", &ch->include_usage, e)) return -1;
    long long tl = -1;
    if (get_int(root, "top_logprobs", 0, 20, &tl, e)) return -1;
    if (tl >= 0 && !ch->logprobs) return bad(e, "top_logprobs", "'top_logprobs' needs 'logprobs': true%s", NULL);
    if (ch->logprobs && ch->stream) return bad(e, "logprobs", "logprobs are not supported with stream%s", NULL);
    j->top_logprobs = ch->logprobs ? (int)(tl > 0 ? tl : 0) : -1;
    int cache = 1;
    if (get_bool(root, "cache_prompt", &cache, e)) return -1;
    j->no_cache = !cache;

    // thinking: off unless reasoning_effort low / medium / high / xhigh or chat_template_kwargs.enable_thinking; Qwen3.8
    // (its template): on by default, the effort xhigh (= high), medium or low (RQ_THINK_*), also as
    // chat_template_kwargs.reasoning_effort (the template's variable: over the request's field)
    int q38 = a->info.chat == RQ_CHAT_QWEN38, thinking = q38, effort = RQ_THINK_ON;
    const Js *ck = json_get(root, "chat_template_kwargs");
    for (int k = 0; k < 2; k++) {
        const Js *re = k == 0 ? json_get(root, "reasoning_effort") : ck && ck->t == JS_OBJ ? json_get(ck, "reasoning_effort") : NULL;
        if (!re || re->t == JS_NULL) continue;
        if (json_str_eq(re, "none") || json_str_eq(re, "minimal")) thinking = 0;
        else if (re->t == JS_STR && rq_chat_effort(re->s) > 0) {
            thinking = 1;
            effort = rq_chat_effort(re->s);
        } else return bad(e, "reasoning_effort", "'reasoning_effort' must be none, minimal, low, medium, high or xhigh%s", NULL);
    }
    if (ck && ck->t == JS_OBJ && get_bool(ck, "enable_thinking", &thinking, e)) return -1;
    if (thinking && q38) thinking = effort;

    // response_format
    const Js *rf = json_get(root, "response_format");
    if (rf && rf->t != JS_NULL) {
        const Js *ty = json_get(rf, "type");
        if (rf->t != JS_OBJ || !ty || ty->t != JS_STR) return bad(e, "response_format", "'response_format' must be an object with a 'type'%s", NULL);
        if (json_str_eq(ty, "json_schema")) {
            if (j->tools) return bad(e, "response_format", "response_format json_schema cannot be combined with tools (tool_choice none: it can)%s", NULL);
            const Js *js = json_get(rf, "json_schema"), *sc = js ? json_get(js, "schema") : NULL;
            if (!sc) return bad(e, "response_format", "response_format json_schema needs json_schema.schema%s", NULL);
            char err[400];
            if (!(ch->con = con_compile(sc, err, sizeof(err))))
                return bad(e, "response_format", "unsupported json_schema: %s (runq serve constrains the output only to schemas "
                           "with a finite set of outputs: enum, const, boolean, null, bounded integer, objects of those)", err);
            j->con = ch->con;
            thinking = 0; // (a constrained reply has no reasoning block; Thinking-2507: the empty think block)
        } else if (json_str_eq(ty, "json_object")) {
            return bad(e, "response_format", "response_format json_object is not supported: use json_schema with a schema that "
                       "has a finite set of outputs (enum, const, boolean, objects of those)%s", NULL);
        } else if (!json_str_eq(ty, "text")) {
            return bad(e, "response_format", "unknown response_format type '%.40s'", ty->s);
        }
    }
    // the template variant: Instruct-2507 never thinks, Thinking-2507 always (but under a constraint)
    if (a->info.chat == RQ_CHAT_INSTRUCT) thinking = 0;
    else if (a->info.chat == RQ_CHAT_THINKING && !j->con) thinking = 1;
    j->thinking = thinking;
    return 0;
}

// ------------------------------------------------------------------------------ chat: the response

static void chunk_head(App *a, JBuf *b, const char *id, const Chat *ch) {
    jb_printf(b, "data: {\"id\": \"%s\", \"object\": \"chat.completion.chunk\", \"created\": %lld, \"model\": ", id, (long long)time(NULL));
    jb_strz(b, ch->model);
    jb_cstr(b, ", \"system_fingerprint\": \"runq\", \"choices\": [");
    (void)a;
}

static void usage_json(JBuf *b, const Job *j) {
    jb_printf(b, "{\"prompt_tokens\": %d, \"completion_tokens\": %d, \"total_tokens\": %d, "
              "\"prompt_tokens_details\": {\"cached_tokens\": %d}}", j->n_prompt, j->n_gen, j->n_prompt + j->n_gen, j->n_cached);
}

static void timings_json(JBuf *b, const Job *j) {
    // llama.cpp's "timings" (prompt_n: the tokens prefilled, cache_n: reused) and runq's: the slot, its length at
    // admission and its common prefix with the prompt (so a rollback to a checkpoint shows as common_tokens > cache_n)
    double pre = j->t_first > 0 ? j->t_first - j->t_start : 0, gen = j->t_first > 0 && j->t_end > j->t_first ? j->t_end - j->t_first : 0;
    jb_printf(b, "{\"prompt_n\": %d, \"prompt_ms\": %.3f, \"predicted_n\": %d, \"predicted_ms\": %.3f, \"cache_n\": %d, \"slot\": %d, "
              "\"slot_tokens\": %d, \"common_tokens\": %d, \"copied\": %s, \"queue_ms\": %.3f}", j->n_prompt - j->n_cached, pre * 1e3, j->n_gen,
              gen * 1e3, j->n_cached, j->slot, j->n_slot, j->n_common, j->copied ? "true" : "false",
              j->t_start > 0 ? (j->t_start - j->t_submit) * 1e3 : 0.0);
}

static const char *finish_name(const Job *j) {
    return j->finish == FINISH_LENGTH ? "length" : j->calls.n > 0 ? "tool_calls" : "stop";
}

static void tool_calls_json(JBuf *b, const Job *j, int from, int stream) {
    // "tool_calls": [...] items from..n (stream: with their index)
    jb_cstr(b, "\"tool_calls\": [");
    for (int i = from; i < j->calls.n; i++) {
        if (i > from) jb_cstr(b, ", ");
        if (stream) jb_printf(b, "{\"index\": %d, ", i);
        else jb_cstr(b, "{");
        jb_cstr(b, j->calls.v[i]);
        jb_cstr(b, "}");
    }
    jb_cstr(b, "]");
}

static void wait_job(HttpConn *c, Job *j, double *next_check) {
    // (under the job's lock) wait for news; every 100 ms: a client that went away cancels the job
    job_wait(j, 0.1);
    double now = rq_now();
    if (now < *next_check) return;
    *next_check = now + 0.1;
    job_unlock(j);
    int gone = http_peer_gone(c);
    job_lock(j);
    if (gone) __atomic_store_n(&j->cancel, 1, __ATOMIC_RELEASE);
}

static void run_stream(App *a, HttpConn *c, HttpReq *r, Chat *ch, Job *j, const char *id) {
    JBuf b = { 0 };
    size_t sent_c = 0, sent_r = 0;
    int status = 200, sent_t = 0;
    double next_check = 0;
    http_stream_start(c, r, 200, "text/event-stream");
    chunk_head(a, &b, id, ch);
    jb_cstr(&b, "{\"index\": 0, \"delta\": {\"role\": \"assistant\", \"content\": \"\"}, \"logprobs\": null, \"finish_reason\": null}]");
    if (ch->include_usage) jb_cstr(&b, ", \"usage\": null");
    jb_cstr(&b, "}\n\n");
    if (http_stream_write(c, b.p, b.n)) __atomic_store_n(&j->cancel, 1, __ATOMIC_RELEASE);
    job_lock(j);
    for (;;) {
        int done = j->state == JOB_DONE;
        if (j->content.n > sent_c || j->reasoning.n > sent_r) {
            jb_reset(&b);
            chunk_head(a, &b, id, ch);
            jb_cstr(&b, "{\"index\": 0, \"delta\": {");
            if (j->reasoning.n > sent_r) {
                jb_cstr(&b, "\"reasoning_content\": ");
                jb_str(&b, j->reasoning.p + sent_r, j->reasoning.n - sent_r);
                sent_r = j->reasoning.n;
            }
            if (j->content.n > sent_c) {
                if (b.p[b.n - 1] != '{') jb_cstr(&b, ", ");
                jb_cstr(&b, "\"content\": ");
                jb_str(&b, j->content.p + sent_c, j->content.n - sent_c);
                sent_c = j->content.n;
            }
            jb_cstr(&b, "}, \"logprobs\": null, \"finish_reason\": null}]");
            if (ch->include_usage) jb_cstr(&b, ", \"usage\": null");
            jb_cstr(&b, "}\n\n");
            job_unlock(j);
            if (http_stream_write(c, b.p, b.n)) __atomic_store_n(&j->cancel, 1, __ATOMIC_RELEASE);
            job_lock(j);
            continue;
        }
        if (j->calls.n > sent_t) { // (each call whole in one delta, after the content before it)
            jb_reset(&b);
            chunk_head(a, &b, id, ch);
            jb_cstr(&b, "{\"index\": 0, \"delta\": {");
            tool_calls_json(&b, j, sent_t, 1);
            sent_t = j->calls.n;
            jb_cstr(&b, "}, \"logprobs\": null, \"finish_reason\": null}]");
            if (ch->include_usage) jb_cstr(&b, ", \"usage\": null");
            jb_cstr(&b, "}\n\n");
            job_unlock(j);
            if (http_stream_write(c, b.p, b.n)) __atomic_store_n(&j->cancel, 1, __ATOMIC_RELEASE);
            job_lock(j);
            continue;
        }
        if (done) break;
        wait_job(c, j, &next_check);
    }
    jb_reset(&b);
    if (j->status) {
        status = j->status;
        jb_cstr(&b, "data: ");
        error_json(&b, j->status, j->errmsg, NULL, j->errcode);
        jb_cstr(&b, "\n\n");
    } else {
        chunk_head(a, &b, id, ch);
        jb_printf(&b, "{\"index\": 0, \"delta\": {}, \"logprobs\": null, \"finish_reason\": \"%s\"}]", finish_name(j));
        if (ch->include_usage) jb_cstr(&b, ", \"usage\": null");
        jb_cstr(&b, "}\n\n");
        if (ch->include_usage) {
            chunk_head(a, &b, id, ch);
            jb_cstr(&b, "], \"usage\": ");
            usage_json(&b, j);
            jb_cstr(&b, ", \"timings\": ");
            timings_json(&b, j);
            jb_cstr(&b, "}\n\n");
        }
    }
    jb_cstr(&b, "data: [DONE]\n\n");
    job_unlock(j);
    if (!http_stream_write(c, b.p, b.n)) http_stream_end(c);
    logline(a, r, status, j, status != 200 ? j->errmsg : "stream");
    jb_free(&b);
}

static void run_once(App *a, HttpConn *c, HttpReq *r, Chat *ch, Job *j, const char *id) {
    double next_check = 0;
    job_lock(j);
    while (j->state != JOB_DONE) wait_job(c, j, &next_check);
    job_unlock(j);
    if (j->status) {
        if (j->status == 499) { // (nobody to answer)
            logline(a, r, 499, j, j->errmsg);
            r->keep_alive = 0;
            return;
        }
        JBuf b = { 0 };
        error_json(&b, j->status, j->errmsg, NULL, j->errcode);
        http_respond(c, r, j->status, "application/json", b.p, b.n, j->status == 503 ? "Retry-After: 1\r\n" : NULL);
        logline(a, r, j->status, j, j->errmsg);
        jb_free(&b);
        return;
    }
    JBuf b = { 0 };
    jb_printf(&b, "{\"id\": \"%s\", \"object\": \"chat.completion\", \"created\": %lld, \"model\": ", id, (long long)time(NULL));
    jb_strz(&b, ch->model);
    jb_cstr(&b, ", \"system_fingerprint\": \"runq\", \"choices\": [{\"index\": 0, \"message\": {\"role\": \"assistant\", \"content\": ");
    if (j->calls.n > 0 && j->content.n == 0) jb_cstr(&b, "null"); // (as OpenAI: only tool calls)
    else jb_str(&b, j->content.p ? j->content.p : "", j->content.n);
    if (j->thinking) {
        jb_cstr(&b, ", \"reasoning_content\": ");
        jb_str(&b, j->reasoning.p ? j->reasoning.p : "", j->reasoning.n);
    }
    if (j->calls.n > 0) {
        jb_cstr(&b, ", ");
        tool_calls_json(&b, j, 0, 0);
    }
    jb_cstr(&b, "}, \"logprobs\": ");
    if (j->top_logprobs >= 0) {
        jb_cstr(&b, "{\"content\": [");
        if (j->logprobs.n) jb_raw(&b, j->logprobs.p, j->logprobs.n);
        jb_cstr(&b, "]}");
    } else
        jb_cstr(&b, "null");
    jb_printf(&b, ", \"finish_reason\": \"%s\"}], \"usage\": ", finish_name(j));
    usage_json(&b, j);
    jb_cstr(&b, ", \"timings\": ");
    timings_json(&b, j);
    jb_cstr(&b, "}");
    if (b.oom) {
        jb_free(&b);
        send_err(a, c, r, 500, "out of memory", NULL, NULL);
        return;
    }
    http_respond(c, r, 200, "application/json", b.p, b.n, NULL);
    if (a->verbose) fprintf(stderr, "[response] %s %.*s\n", r->peer, (int)(b.n < 65536 ? b.n : 65536), b.p);
    logline(a, r, 200, j, finish_name(j));
    jb_free(&b);
}

static int render_req(App *a, Chat *ch, Job *j, ReqErr *e) {
    // the prompt: tools_render in tool mode, else render; 0, or -1 with the error in e
    char msg[400] = "";
    int rc = ch->tmode ? tools_render(a->m, a->info.chat, ch->tools, ch->msgs, ch->n_msgs, j->thinking, &j->prompt, msg, sizeof(msg))
                       : render(a->m, a->info.chat, ch->msgs, ch->n_msgs, j->thinking, &j->prompt);
    if (rc == RQ_OK && ch->has_force) { // (force_reply: its tokens, as one text after the prompt)
        if (j->con || j->top_logprobs >= 0) return bad(e, "force_reply", "'force_reply' cannot be combined with json_schema or logprobs%s", NULL);
        JBuf t = { 0 };
        if (ch->force_raw) jb_cstr(&t, ch->force.text);
        else rc = tools_reply_text(a->m, a->info.chat, &ch->force, j->thinking, &t);
        if (rc == RQ_OK && t.oom) rc = RQ_ERR_NOMEM;
        if (rc == RQ_OK && t.n > 0) rc = rq_tokenize(a->m, t.p, (long)t.n, &j->force);
        jb_free(&t);
    }
    if (rc == RQ_OK) return 0;
    e->status = rc == RQ_ERR_NOMEM ? 500 : 400;
    if (msg[0]) snprintf(e->msg, sizeof(e->msg), "%s", msg);
    else snprintf(e->msg, sizeof(e->msg), "cannot render the conversation: %s", rq_strerror(rc));
    e->param = "messages";
    return -1;
}

int serve_render(RqModel *m, const Js *body, RqTokens *out, int *thinking, char *err, size_t errlen) {
    App a;
    memset(&a, 0, sizeof(a));
    pthread_mutex_init(&a.mu, NULL);
    a.m = m;
    rq_model_info(m, &a.info);
    a.def_temp = a.def_topp = 1;
    snprintf(a.model_id, sizeof(a.model_id), "render");
    Chat ch;
    memset(&ch, 0, sizeof(ch));
    ReqErr e;
    memset(&e, 0, sizeof(e));
    Job *j = job_new();
    int rc = RQ_ERR_NOMEM;
    if (j) {
        rc = parse_chat(&a, body, &ch, j, &e) == 0 && render_req(&a, &ch, j, &e) == 0 ? RQ_OK : RQ_ERR_ARG;
        if (rc == RQ_OK) rc = rq_tokens_push(out, j->prompt.v, j->prompt.n);
        else snprintf(err, errlen, "%s", e.msg);
        *thinking = j->thinking;
    }
    chat_free(&ch);
    job_release(j);
    pthread_mutex_destroy(&a.mu);
    return rc;
}

static void post_chat(App *a, HttpConn *c, HttpReq *r) {
    char err[300];
    if (a->verbose) fprintf(stderr, "[request] %s %s %.*s\n", r->peer, r->target, (int)(r->body_len < 65536 ? r->body_len : 65536), r->body ? r->body : "");
    JsDoc *doc = json_parse(r->body ? r->body : "", r->body_len, err, sizeof(err));
    if (!doc) {
        char msg[400];
        snprintf(msg, sizeof(msg), "invalid JSON in the request body: %s", err);
        send_err(a, c, r, 400, msg, NULL, "invalid_json");
        return;
    }
    Chat ch;
    memset(&ch, 0, sizeof(ch));
    ReqErr e;
    memset(&e, 0, sizeof(e));
    Job *j = job_new();
    if (!j) {
        json_free(doc);
        send_err(a, c, r, 500, "out of memory", NULL, NULL);
        return;
    }
    if (parse_chat(a, json_root(doc), &ch, j, &e) == 0 && render_req(a, &ch, j, &e) == 0) {
        if (j->prompt.n >= a->ctx) {
            e.status = 400;
            snprintf(e.msg, sizeof(e.msg), "This model's maximum context length is %d tokens. However, your messages resulted in %d tokens.", a->ctx,
                     j->prompt.n);
            e.param = "messages";
            e.code = "context_length_exceeded";
        } else {
            int room = a->ctx - j->prompt.n;
            if (j->max_new <= 0 || j->max_new > room) j->max_new = room;
            j->deadline = a->timeout > 0 ? rq_now() + a->timeout : 0;
        }
    }
    json_free(doc);
    if (e.status) {
        send_err(a, c, r, e.status, e.msg, e.param, e.code);
        chat_free(&ch);
        job_release(j);
        return;
    }
    int q = engine_submit(a->eng, j);
    if (q) {
        char msg[200];
        int running = 0, queued = engine_queued(a->eng, &running);
        if (q == -1) snprintf(msg, sizeof(msg), "server overloaded: %d requests queued, try again later", queued);
        send_err(a, c, r, 503, q == -1 ? msg : "the server is shutting down", NULL, q == -1 ? "overloaded" : "shutting_down");
        chat_free(&ch);
        job_release(j);
        return;
    }
    char id[64];
    pthread_mutex_lock(&a->mu);
    snprintf(id, sizeof(id), "chatcmpl-%llx%08llx", (unsigned long long)a->created, ++a->seq);
    pthread_mutex_unlock(&a->mu);
    if (ch.stream) run_stream(a, c, r, &ch, j, id);
    else run_once(a, c, r, &ch, j, id);
    chat_free(&ch); // (the engine is done with the job: its constraint can go)
    job_release(j);
}

// ------------------------------------------------------------------------------------------ routes

static int auth_ok(App *a, HttpReq *r) {
    if (!a->api_key) return 1;
    const char *h = r->auth;
    if (strncasecmp(h, "Bearer ", 7)) return 0;
    h += 7;
    size_t n = strlen(a->api_key), m = strlen(h);
    unsigned d = n != m;
    for (size_t i = 0; i < n; i++) d |= (unsigned char)a->api_key[i] ^ (unsigned char)(i < m ? h[i] : 0);
    return d == 0;
}

static void handle(void *app, HttpConn *c, HttpReq *r) {
    App *a = app;
    const char *p = r->path;
    int get = !strcmp(r->method, "GET") || !strcmp(r->method, "HEAD"), post = !strcmp(r->method, "POST");
    int is_health = !strcmp(p, "/health") || !strcmp(p, "/v1/health");
    int is_models = !strcmp(p, "/v1/models") || !strcmp(p, "/models");
    const char *model_id = !strncmp(p, "/v1/models/", 11) ? p + 11 : !strncmp(p, "/models/", 8) ? p + 8 : NULL;
    int is_chat = !strcmp(p, "/v1/chat/completions") || !strcmp(p, "/chat/completions");
    if (!is_health && !auth_ok(a, r)) {
        send_err(a, c, r, 401, "invalid or missing API key (Authorization: Bearer ...)", NULL, "invalid_api_key");
        return;
    }
    if (is_health && get) get_health(a, c, r);
    else if ((is_models || model_id) && get) get_models(a, c, r, model_id);
    else if (is_chat && post) post_chat(a, c, r);
    else if (is_health || is_models || model_id || is_chat) {
        char msg[128];
        snprintf(msg, sizeof(msg), "method %s not allowed here (use %s)", r->method, is_chat ? "POST" : "GET");
        send_err(a, c, r, 405, msg, NULL, "method_not_allowed");
    } else {
        char msg[300];
        snprintf(msg, sizeof(msg), "unknown endpoint %.200s %.60s", r->method, p);
        send_err(a, c, r, 404, msg, NULL, "not_found");
    }
}

// -------------------------------------------------------------------------------------------- main

static void usage(void) {
    fprintf(stderr,
            "Usage:   runq serve <checkpoint> [options]\n"
            "An OpenAI-compatible HTTP server (POST /v1/chat/completions, GET /v1/models, GET /health).\n"
            "Options:\n"
            "  --host <addr>       address to bind, default 127.0.0.1 (no TLS, no auth unless --api-key)\n"
            "  --port <int>        port, default 8080 (0: any free port)\n"
            "  --api-key <key>     require 'Authorization: Bearer <key>' (not on /health)\n"
            "  --name <id>         model id in /v1/models and responses, default: the file name\n"
            "  -c, --ctx <int>     context (positions) per sequence, default 8192\n"
            "  -k <type>           KV cache type f16|bf16|f32, default f16\n"
            "  --parallel <int>    requests running at once (batched in each forward pass), default 16\n"
            "  --slots <int>       sequences in the pool: running requests + prefix caches, default parallel + 4\n"
            "  --kv-mem <GiB>      memory of the pool at most (slots x (KV cache of ctx positions + recurrent state and its\n"
            "                      checkpoints)), default half the RAM, at most MemAvailable after the load\n"
            "  --mem <GiB>         refuse to load if the estimate (weights, runtime, one sequence) does not fit, default\n"
            "                      MemAvailable; 0 or force = no check\n"
            "  --page-cache <mode> auto|drop|keep: the model file's page cache after the load, default auto (dropped if the\n"
            "                      file is >= 1/4 of the RAM)\n"
            "  -b <int>            prompt tokens per pass when no request is decoding, default 512\n"
            "  --chunk <int>       prompt tokens per pass while others decode, default 64\n"
            "  --prefix-copy <int> copy a running request's prompt prefix if that saves this many tokens, default 32 (0: never)\n"
            "  --queue <int>       requests waiting at most (more: 503), default 64\n"
            "  --max-conns <int>   connections at most (more: 503), default 128\n"
            "  --max-body <bytes>  request body limit (more: 413), default 4194304\n"
            "  --timeout <s>       time limit per request incl. queueing (504), default 600, 0 = none\n"
            "  -t <float>          default temperature (requests without one), default 1.0\n"
            "  -p <float>          default top_p, default 1.0\n"
            "  --top-k <int>       default top_k, default 0 = off (-t / -p / --top-k not given: the model's recommended\n"
            "                      values if its GGUF has them, general.sampling.*)\n"
            "  -n <int>            default max tokens (requests without one), default 0 = up to the context\n"
            "  --threads <int>     threads per forward pass, default OMP_NUM_THREADS or nproc - nproc/8\n"
            "  -v, --verbose       log request bodies and responses\n");
    exit(EXIT_FAILURE);
}

int serve_main(int argc, char **argv) {
    // argv[0] = "serve"
    if (argc < 2 || argv[1][0] == '-') usage();
    const char *path = argv[1], *name = NULL, *host = "127.0.0.1", *key = NULL;
    int port = 8080, ctx = 8192, kv = RQ_KV_DEFAULT, batch = RQ_BATCH_MAX, slots = 0, queue = 64, conns = 128, threads = 0, verbose = 0, maxtok = 0;
    int parallel = 16, chunk = 64, copy_min = 32;
    long long maxbody = 4 << 20;
    double timeout = 600, temp = 1.0, topp = 1.0, kv_mem = 0, mem = 0;
    int page_cache = 0;
    int topk = 0, samp_given = 0; // (RQ_SAMPLING_* given: else the model's values, if any)
    for (int i = 2; i < argc; i++) {
        const char *f = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(f, "-v") || !strcmp(f, "--verbose")) { verbose = 1; continue; }
        if (!v) usage();
        i++;
        if (!strcmp(f, "--host")) host = v;
        else if (!strcmp(f, "--port")) port = atoi(v);
        else if (!strcmp(f, "--api-key")) key = v;
        else if (!strcmp(f, "--name")) name = v;
        else if (!strcmp(f, "-c") || !strcmp(f, "--ctx")) ctx = atoi(v);
        else if (!strcmp(f, "-k")) { if ((kv = rq_kv_parse(v)) < 0) usage(); }
        else if (!strcmp(f, "-b")) batch = atoi(v);
        else if (!strcmp(f, "--slots")) slots = atoi(v);
        else if (!strcmp(f, "--parallel")) parallel = atoi(v);
        else if (!strcmp(f, "--kv-mem")) kv_mem = atof(v);
        else if (!strcmp(f, "--mem")) {
            char *end;
            mem = strtod(v, &end);
            if (!strcmp(v, "force") || !strcmp(v, "off")) mem = -1;
            else if (end == v || *end || !(mem >= 0)) usage();
            else if (mem == 0) mem = -1;
        }
        else if (!strcmp(f, "--page-cache")) {
            page_cache = !strcmp(v, "drop") ? 1 : !strcmp(v, "keep") ? -1 : !strcmp(v, "auto") ? 0 : 2;
            if (page_cache == 2) usage();
        }
        else if (!strcmp(f, "--chunk")) chunk = atoi(v);
        else if (!strcmp(f, "--prefix-copy")) copy_min = atoi(v);
        else if (!strcmp(f, "--queue")) queue = atoi(v);
        else if (!strcmp(f, "--max-conns")) conns = atoi(v);
        else if (!strcmp(f, "--max-body")) maxbody = atoll(v);
        else if (!strcmp(f, "--timeout")) timeout = atof(v);
        else if (!strcmp(f, "-t")) { temp = atof(v); samp_given |= RQ_SAMPLING_TEMP; }
        else if (!strcmp(f, "-p")) { topp = atof(v); samp_given |= RQ_SAMPLING_TOPP; }
        else if (!strcmp(f, "--top-k")) { topk = atoi(v); samp_given |= RQ_SAMPLING_TOPK; }
        else if (!strcmp(f, "-n")) maxtok = atoi(v);
        else if (!strcmp(f, "--threads")) threads = atoi(v);
        else usage();
    }
    if (port < 0 || port > 65535 || ctx < 16 || slots < 0 || slots > 1024 || parallel < 1 || parallel > 256 || kv_mem < 0 || chunk < 1 ||
        copy_min < 0 || queue < 1 || conns < 1 || maxbody < 1024 || temp < 0 || temp > 2 || topp <= 0 || topp > 1 || topk < 0 || maxtok < 0 || batch < 1)
        usage();
    if (batch > RQ_BATCH_MAX) batch = RQ_BATCH_MAX;

    // signals first: every thread created from here on has SIGINT / SIGTERM blocked
    if (http_signals()) { fprintf(stderr, "runq serve: cannot install the signal handlers\n"); return 1; }
    const char *warn = rq_threads_init(threads);
    if (warn) fprintf(stderr, "%s\n", warn);
    char err[512];
    // (positions up to -c: the default cap of 40,960 would clamp a larger -c; the memory check counts one sequence of -c
    // positions, the pool is bounded by --kv-mem)
    RqModelOpts mo = { ctx, 0, 0, mem, kv == RQ_KV_F32 ? 4 : 2, page_cache };
    RqModel *m = rq_model_load(path, &mo, err, sizeof(err));
    if (!m) { fprintf(stderr, "%s\n", err); return 1; }
    App a;
    memset(&a, 0, sizeof(a));
    pthread_mutex_init(&a.mu, NULL);
    a.m = m;
    rq_model_info(m, &a.info);
    if (a.info.warning) fprintf(stderr, "warning: %s\n", a.info.warning);
    if (rq_chat_check(m, 0, err, sizeof(err)) != RQ_OK || rq_chat_check(m, 1, err, sizeof(err)) != RQ_OK) {
        fprintf(stderr, "runq serve: the chat templates are missing (%s)\n", err);
        return 1;
    }
    if (ctx > a.info.seq_len) ctx = a.info.seq_len;
    a.ctx = ctx;
    a.verbose = verbose;
    a.api_key = key;
    RqSampling ms = { (float)temp, (float)topp, NULL, topk };
    int mset = rq_model_sampling(m, &ms) & ~samp_given; // (the file's values for the options not given; silent)
    a.def_temp = (mset & RQ_SAMPLING_TEMP) && ms.temperature <= 2 ? ms.temperature : (float)temp;
    a.def_topp = mset & RQ_SAMPLING_TOPP ? ms.topp : (float)topp;
    a.def_topk = mset & RQ_SAMPLING_TOPK ? ms.top_k : topk;
    a.def_max_tokens = maxtok;
    a.timeout = timeout;
    a.created = (long long)time(NULL);
    if (name) snprintf(a.model_id, sizeof(a.model_id), "%s", name);
    else {
        const char *b = strrchr(path, '/');
        b = b ? b + 1 : path;
        size_t n = strlen(b);
        if (n > 5 && !strcmp(b + n - 5, ".gguf")) n -= 5;
        else if (n > 4 && !strcmp(b + n - 4, ".bin")) n -= 4;
        snprintf(a.model_id, sizeof(a.model_id), "%.*s", (int)n, b);
    }
    EngineOpts eo = { path, ctx, kv, batch, chunk, parallel, slots, kv_mem, copy_min, queue, verbose };
    double t0 = rq_now();
    a.eng = engine_start(m, &eo, err, sizeof(err));
    if (!a.eng) { fprintf(stderr, "runq serve: %s\n", err); return 1; }
    HttpOpts ho = { host, port, 16384, (size_t)maxbody, conns, 30.0, 120.0 };
    int bound = 0;
    int lfd = http_listen(&ho, &bound, err, sizeof(err));
    if (lfd < 0) { fprintf(stderr, "runq serve: %s\n", err); return 1; }
    char desc[512];
    engine_describe(a.eng, desc, sizeof(desc));
    fprintf(stderr, "runq serve: %s as '%s' (context %d of %d, KV %s, %d threads, chat template %s, vocab index %.0f ms)\nrunq serve: %s\n",
            path, a.model_id, ctx, a.info.ctx_max, rq_kv_name(kv), rq_threads(), rq_chat_name(a.info.chat), (rq_now() - t0) * 1e3, desc);
    fprintf(stderr, "runq serve: listening on http://%s%s%s:%d\n", strchr(host, ':') ? "[" : "", host, strchr(host, ':') ? "]" : "", bound);
    fflush(stderr);
    http_run(lfd, &ho, handle, &a);
    fprintf(stderr, "runq serve: shutting down\n");
    engine_stop(a.eng, 10.0);
    if (!http_wait_conns(5.0)) {
        fprintf(stderr, "runq serve: connections still open, exiting anyway\n");
        return 0;
    }
    engine_free(a.eng);
    rq_model_free(m);
    pthread_mutex_destroy(&a.mu);
    fprintf(stderr, "runq serve: stopped\n");
    return 0;
}
#endif
