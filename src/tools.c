/* runq serve: tool calls (OpenAI `tools`) for the Qwen3.5 and Qwen3.8 chat templates (serve.h).
 *
 * Rendering (tools_render): the conversation exactly as the templates render it with tools, `tool` messages and
 * earlier assistant `tool_calls` (models/Qwen3.5-0.8B/chat_template.jinja = tests/golden/templates/qwen3.5.jinja;
 * Qwen3.8: the 27B GGUF's, tests/golden/templates/qwen3.8.jinja, Unsloth's rewrite of Qwen's qwen3.8-hf.jinja):
 *   system   "<|im_start|>system\n" [effort instruction "\n\n" (Qwen3.8)] "# Tools\n\n...<tools>" ("\n" tojson(tool))*
 *            "\n</tools>" INSTRUCTIONS ["\n\n" system text] "<|im_end|>\n"   (no tools: the variant's usual block)
 *   user     "<|im_start|>user\n" text "<|im_end|>\n"
 *   assistant "<|im_start|>assistant\n" ["<think>\n" R "\n</think>\n\n"] C, then per call ("\n\n" before the first if C,
 *            "\n" before the others) "<tool_call>\n<function=NAME>\n" ("<parameter=K>\n" V "\n</parameter>\n")*
 *            "</function>\n</tool_call>", then "<|im_end|>\n". The think block: Qwen3.5 only after the last user query
 *            (a user message that is not a "<tool_response>...</tool_response>" turn), Qwen3.8 always (preserve_thinking)
 *   tool     ["<|im_start|>user" if the previous message is not a tool one] "\n<tool_response>\n" text
 *            "\n</tool_response>" ["<|im_end|>\n" if the next one is not a tool one]
 *   then the generation prompt "<|im_start|>assistant\n" + "<think>\n" (thinking) or "<think>\n\n</think>\n\n".
 * Every text is trimmed (|trim); tojson is transformers' (Python's json.dumps(ensure_ascii=False): ", " and ": ",
 * members in order, a repeated name once at its first place with its last value, Python's float repr); an argument
 * value V is the string itself, else tojson (Qwen3.8), or for Qwen3.5 Python's str() of a scalar (True, None, 1.5)
 * and tojson of an object / array. Arguments given as a JSON string (OpenAI's form) are parsed first, as vLLM does.
 * R / C (reasoning, content): Qwen3.5 from reasoning_content, else content split at "</think>" as its template does;
 * Qwen3.8 as runq's Qwen3.8 rendering (src/chat.c: reasoning_content, else split at the last "</think>").
 *
 * Parsing (tscan_*, tool_parse_calls): the reply's content (the text after the reasoning) is scanned as it is
 * generated: text before a "<tool_call>" is content (its trailing whitespace dropped: the template's "\n\n"), a
 * block "<tool_call>...</tool_call>" becomes one call per "<function=NAME>...</function>" inside, each
 * "<parameter=P>\nvalue\n</parameter>" an argument (one leading and one trailing "\n" removed; a value without its
 * </parameter> ends at the next <parameter= or </function>), converted to the JSON type of the tool's schema for P
 * (type / anyOf / oneOf: "null" if allowed, a string if allowed, else integer / number / boolean when the text is
 * one, object / array when it parses as JSON; anything else, an unknown parameter or tool: a string). After a
 * block, whitespace and further blocks may follow; any other text stops the generation (the template asks for "NO
 * suffix"). A malformed block (no <function=...>, text between the parts) is content if no call came before it;
 * a block cut by the end of the reply counts if its </function> was generated. The calls come out as OpenAI
 * tool_calls items: {"id": "call_...", "type": "function", "function": {"name": ..., "arguments": "<JSON>"}}, the
 * arguments as Python's json.dumps writes them (vLLM's form).
 */
#ifndef _WIN32
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include "serve.h"

#define TOOL_CALL "<tool_call>"
#define TOOL_CALL_END "</tool_call>"

// the template's text around the tool list (both variants)
#define TOOLS_HEAD "# Tools\n\nYou have access to the following functions:\n\n<tools>"
#define TOOLS_TAIL "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n" \
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n" \
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n" \
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified " \
    "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n" \
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural " \
    "language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question " \
    "like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>"

// ------------------------------------------------------------------------------ Python-style JSON

static int utf8_char(const unsigned char *p, size_t n) {
    // the length of the well-formed UTF-8 character at p, 0 if none
    if (n == 0) return 0;
    unsigned c = p[0];
    if (c < 0x80) return 1;
    int k = c >= 0xc2 && c <= 0xdf ? 2 : c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 0;
    if (!k || (size_t)k > n) return 0;
    for (int i = 1; i < k; i++)
        if ((p[i] & 0xc0) != 0x80) return 0;
    if (c == 0xe0 && p[1] < 0xa0) return 0;  // overlong
    if (c == 0xed && p[1] >= 0xa0) return 0; // surrogate
    if (c == 0xf0 && p[1] < 0x90) return 0;  // overlong
    if (c == 0xf4 && p[1] >= 0x90) return 0; // > U+10FFFF
    return k;
}

void py_str(JBuf *b, const char *s, size_t n) {
    // json.dumps(str, ensure_ascii=False): \" \\ \n \r \t \b \f, other controls \u00xx, the rest as is (invalid
    // UTF-8: U+FFFD per byte, as jb_str)
    const unsigned char *u = (const unsigned char *)s;
    jb_raw(b, "\"", 1);
    for (size_t i = 0; i < n;) {
        unsigned c = u[i];
        if (c >= 0x80) {
            int k = utf8_char(u + i, n - i);
            if (k) jb_raw(b, s + i, k);
            else jb_raw(b, "\xef\xbf\xbd", 3);
            i += k ? k : 1;
            continue;
        }
        i++;
        switch (c) {
            case '"': jb_raw(b, "\\\"", 2); break;
            case '\\': jb_raw(b, "\\\\", 2); break;
            case '\n': jb_raw(b, "\\n", 2); break;
            case '\r': jb_raw(b, "\\r", 2); break;
            case '\t': jb_raw(b, "\\t", 2); break;
            case '\b': jb_raw(b, "\\b", 2); break;
            case '\f': jb_raw(b, "\\f", 2); break;
            default:
                if (c < 0x20) jb_printf(b, "\\u%04x", c);
                else {
                    char ch = (char)c;
                    jb_raw(b, &ch, 1);
                }
        }
    }
    jb_raw(b, "\"", 1);
}

void py_float(JBuf *b, double x) {
    // Python's repr(float): the shortest digits that read back as x; fixed notation for decimal exponents -4..15
    // (at least one digit after the point), else d[.ddd]e+XX
    unsigned long long bits; // (not isinf / isnan: -Ofast assumes finite math)
    memcpy(&bits, &x, sizeof(bits));
    if ((bits >> 52 & 0x7ff) == 0x7ff) {
        jb_cstr(b, bits & 0xfffffffffffffULL ? "NaN" : bits >> 63 ? "-Infinity" : "Infinity");
        return;
    }
    char s[40];
    for (int p = 1; p <= 17; p++) {
        snprintf(s, sizeof(s), "%.*e", p - 1, x);
        if (strtod(s, NULL) == x) break;
    }
    // s: [-]d[.ddd]e[+-]XX
    char dig[24];
    int nd = 0, neg = s[0] == '-';
    const char *q = s + neg;
    for (; *q && *q != 'e'; q++)
        if (*q != '.') dig[nd++] = *q;
    int ex = atoi(q + 1);
    while (nd > 1 && dig[nd - 1] == '0') nd--;
    if (neg) jb_raw(b, "-", 1);
    if (ex >= -4 && ex < 16) {
        if (ex < 0) {
            jb_cstr(b, "0.");
            for (int i = 0; i < -ex - 1; i++) jb_raw(b, "0", 1);
            jb_raw(b, dig, nd);
        } else {
            for (int i = 0; i <= ex; i++) jb_raw(b, i < nd ? &dig[i] : "0", 1);
            jb_raw(b, ".", 1);
            if (nd > ex + 1) jb_raw(b, dig + ex + 1, nd - ex - 1);
            else jb_raw(b, "0", 1);
        }
    } else {
        jb_raw(b, dig, 1);
        if (nd > 1) {
            jb_raw(b, ".", 1);
            jb_raw(b, dig + 1, nd - 1);
        }
        jb_printf(b, "e%c%02d", ex < 0 ? '-' : '+', ex < 0 ? -ex : ex);
    }
}

static void py_num(JBuf *b, const Js *v) {
    // a JSON number as Python's json.loads reads it and json.dumps / str() write it: an int keeps its digits
    // ("-0": 0), a float its repr
    if (strpbrk(v->s, ".eE")) { py_float(b, v->num); return; }
    const char *s = v->s;
    if (s[0] == '-' && strspn(s + 1, "0") == strlen(s + 1)) s++;
    jb_cstr(b, s);
}

void py_json(JBuf *b, const Js *v) {
    // json.dumps(v, ensure_ascii=False) of what json.loads made of v (an object's repeated name: once, at its
    // first place, with its last value)
    switch (v->t) {
        case JS_NULL: jb_cstr(b, "null"); break;
        case JS_TRUE: jb_cstr(b, "true"); break;
        case JS_FALSE: jb_cstr(b, "false"); break;
        case JS_NUM: py_num(b, v); break;
        case JS_STR: py_str(b, v->s, v->n); break;
        case JS_ARR:
            jb_raw(b, "[", 1);
            for (int i = 0; i < v->n; i++) {
                if (i) jb_cstr(b, ", ");
                py_json(b, v->v[i]);
            }
            jb_raw(b, "]", 1);
            break;
        case JS_OBJ: {
            jb_raw(b, "{", 1);
            int first = 1;
            for (int i = 0; i < v->n; i++) {
                int seen = 0, last = i;
                for (int k = 0; k < i && !seen; k++) seen = v->kn[k] == v->kn[i] && !memcmp(v->k[k], v->k[i], v->kn[i]);
                if (seen) continue;
                for (int k = i + 1; k < v->n; k++)
                    if (v->kn[k] == v->kn[i] && !memcmp(v->k[k], v->k[i], v->kn[i])) last = k;
                if (!first) jb_cstr(b, ", ");
                first = 0;
                py_str(b, v->k[i], v->kn[i]);
                jb_cstr(b, ": ");
                py_json(b, v->v[last]);
            }
            jb_raw(b, "}", 1);
            break;
        }
    }
}

static void py_text(JBuf *b, const Js *v, int chat) {
    // an argument value in the history: a string as is; else Qwen3.8 tojson, Qwen3.5 str() of a scalar
    if (v->t == JS_STR) jb_raw(b, v->s, v->n);
    else if (chat == RQ_CHAT_QWEN38 || v->t == JS_OBJ || v->t == JS_ARR) py_json(b, v);
    else if (v->t == JS_NUM) py_num(b, v);
    else jb_cstr(b, v->t == JS_TRUE ? "True" : v->t == JS_FALSE ? "False" : "None");
}

// ------------------------------------------------------------------------------------------ tools

enum { T_STR = 1, T_INT = 2, T_NUM = 4, T_BOOL = 8, T_OBJ = 16, T_ARR = 32, T_NULL = 64 };

typedef struct { char *name; unsigned types; } ToolParam;
typedef struct { char *name; ToolParam *p; int np; } ToolDef;
struct ToolSet {
    ToolDef *t;
    int n;
    char *text; // the system block's tool section (TOOLS_HEAD ... TOOLS_TAIL)
};

static unsigned type_mask(const Js *s, int depth) {
    // the JSON types a parameter schema allows (type: a name or a list; anyOf / oneOf: their union); 0: unknown
    if (!s || s->t != JS_OBJ || depth > 8) return 0;
    unsigned m = 0;
    const Js *t = json_get(s, "type");
    for (int i = 0; t && i < (t->t == JS_ARR ? t->n : 1); i++) {
        const Js *x = t->t == JS_ARR ? t->v[i] : t;
        if (x->t != JS_STR) continue;
        m |= !strcmp(x->s, "string") ? T_STR : !strcmp(x->s, "integer") ? T_INT : !strcmp(x->s, "number") ? T_NUM
           : !strcmp(x->s, "boolean") ? T_BOOL : !strcmp(x->s, "object") ? T_OBJ : !strcmp(x->s, "array") ? T_ARR
           : !strcmp(x->s, "null") ? T_NULL : 0;
    }
    for (int k = 0; k < 2; k++) {
        const Js *a = json_get(s, k ? "oneOf" : "anyOf");
        for (int i = 0; a && a->t == JS_ARR && i < a->n; i++) m |= type_mask(a->v[i], depth + 1);
    }
    return m;
}

void tools_free(ToolSet *ts) {
    if (!ts) return;
    for (int i = 0; i < ts->n; i++) {
        for (int k = 0; k < ts->t[i].np; k++) free(ts->t[i].p[k].name);
        free(ts->t[i].p);
        free(ts->t[i].name);
    }
    free(ts->t);
    free(ts->text);
    free(ts);
}

int tools_count(const ToolSet *ts) { return ts ? ts->n : 0; }

ToolSet *tools_compile(const Js *tools, char *err, size_t errlen) {
    // OpenAI's tools: [{"type": "function", "function": {"name", "description"?, "parameters"?}}]
    if (!tools || tools->t != JS_ARR) {
        snprintf(err, errlen, "'tools' must be an array");
        return NULL;
    }
    ToolSet *ts = calloc(1, sizeof(ToolSet));
    if (!ts || !(ts->t = calloc(tools->n ? tools->n : 1, sizeof(ToolDef)))) {
        free(ts);
        snprintf(err, errlen, "out of memory");
        return NULL;
    }
    JBuf b = { 0 };
    jb_cstr(&b, TOOLS_HEAD);
    for (int i = 0; i < tools->n; i++) {
        const Js *t = tools->v[i], *ty = json_get(t, "type"), *f = json_get(t, "function"), *nm = json_get(f, "name");
        if (t->t != JS_OBJ || (ty && !json_str_eq(ty, "function"))) {
            snprintf(err, errlen, "tools[%d]: only function tools ({\"type\": \"function\", \"function\": {...}}) are supported", i);
            goto fail;
        }
        if (!f || f->t != JS_OBJ || !nm || nm->t != JS_STR || nm->n == 0 || (int)strlen(nm->s) != nm->n) {
            snprintf(err, errlen, "tools[%d]: function.name must be a non-empty string", i);
            goto fail;
        }
        const Js *pa = json_get(f, "parameters"), *pr = pa ? json_get(pa, "properties") : NULL;
        if (pa && pa->t != JS_OBJ && pa->t != JS_NULL) {
            snprintf(err, errlen, "tools[%d]: function.parameters must be a JSON schema object", i);
            goto fail;
        }
        ToolDef *d = &ts->t[ts->n++];
        if (!(d->name = strdup(nm->s))) goto oom;
        if (pr && pr->t == JS_OBJ && pr->n > 0) {
            if (!(d->p = calloc(pr->n, sizeof(ToolParam)))) goto oom;
            for (int k = 0; k < pr->n; k++) {
                if (!(d->p[k].name = strdup(pr->k[k]))) goto oom;
                d->p[k].types = type_mask(pr->v[k], 0);
                d->np = k + 1;
            }
        }
        jb_raw(&b, "\n", 1);
        py_json(&b, t);
    }
    jb_cstr(&b, TOOLS_TAIL);
    if (b.oom) goto oom;
    ts->text = b.p;
    return ts;
oom:
    snprintf(err, errlen, "out of memory");
fail:
    jb_free(&b);
    tools_free(ts);
    return NULL;
}

static const ToolParam *tool_param(const ToolSet *ts, const char *fn, size_t fl, const char *pn, size_t pl) {
    for (int i = 0; ts && i < ts->n; i++) {
        if (strlen(ts->t[i].name) != fl || memcmp(ts->t[i].name, fn, fl)) continue;
        for (int k = ts->t[i].np - 1; k >= 0; k--) // (a repeated property name: the last, as json.loads)
            if (strlen(ts->t[i].p[k].name) == pl && !memcmp(ts->t[i].p[k].name, pn, pl)) return &ts->t[i].p[k];
        return NULL;
    }
    return NULL;
}

// ------------------------------------------------------------------------------------------ rendering

static int is_ws(int c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; }

static int render_assistant(JBuf *b, const RqModel *m, int chat, const char *r, const char *c, const MsgCall *calls, int ncalls,
                            int think) {
    // an assistant message after its header: [the think block of reasoning r] content c, its calls, <|im_end|>\n
    if (think) {
        jb_cstr(b, "<think>\n");
        jb_cstr(b, r);
        jb_cstr(b, "\n</think>\n\n");
    }
    jb_cstr(b, c);
    size_t tl;
    rq_chat_trim(m, c, &tl);
    for (int k = 0; k < ncalls; k++) {
        const MsgCall *tc = &calls[k];
        jb_cstr(b, k ? "\n<tool_call>\n<function=" : tl ? "\n\n<tool_call>\n<function=" : "<tool_call>\n<function=");
        jb_cstr(b, tc->name);
        jb_cstr(b, ">\n");
        const Js *a = tc->args;
        for (int i = 0; a && i < a->n; i++) {
            int seen = 0, last = i; // (json.loads: a repeated name once, at its first place, with its last value)
            for (int j = 0; j < i && !seen; j++) seen = a->kn[j] == a->kn[i] && !memcmp(a->k[j], a->k[i], a->kn[i]);
            if (seen) continue;
            for (int j = i + 1; j < a->n; j++)
                if (a->kn[j] == a->kn[i] && !memcmp(a->k[j], a->k[i], a->kn[i])) last = j;
            jb_cstr(b, "<parameter=");
            jb_raw(b, a->k[i], a->kn[i]);
            jb_cstr(b, ">\n");
            py_text(b, a->v[last], chat);
            jb_cstr(b, "\n</parameter>\n");
        }
        jb_cstr(b, "</function>\n</tool_call>");
    }
    jb_cstr(b, "<|im_end|>\n");
    return b->oom ? -1 : 0;
}

static char *trimmed_copy(const RqModel *m, const char *s, size_t n) {
    // |trim (Python's strip, rq_chat_trim) of s[0..n): malloc'd
    char *t = malloc(n + 1);
    if (!t) return NULL;
    memcpy(t, s, n);
    t[n] = 0;
    size_t l;
    const char *p = rq_chat_trim(m, t, &l);
    memmove(t, p, l);
    t[l] = 0;
    return t;
}

int tools_render(const RqModel *m, int chat, const ToolSet *ts, const Msg *msg, int n, int thinking, RqTokens *out,
                 char *err, size_t errlen) {
    // the conversation as the template renders it (the header comment); the texts in msg are already trimmed
    int q38 = chat == RQ_CHAT_QWEN38, nsys = 0, e = RQ_OK;
    if (chat != RQ_CHAT_QWEN35 && !q38) {
        snprintf(err, errlen, "tools are supported with the Qwen3.5 and Qwen3.8 chat templates only (this model's: %s)", rq_chat_name(chat));
        return RQ_ERR_ARG;
    }
    if (n < 1) {
        snprintf(err, errlen, "no messages");
        return RQ_ERR_ARG;
    }
    // the leading system messages: Qwen3.8 merges them (Unsloth's template), Qwen3.5 has one at most
    for (; nsys < n && msg[nsys].role == MSG_SYSTEM && (q38 || nsys == 0); nsys++) {}
    for (int i = nsys; i < n; i++)
        if (msg[i].role == MSG_SYSTEM) {
            snprintf(err, errlen, "system / developer messages must be at the beginning (the %s chat template)", rq_chat_name(chat));
            return RQ_ERR_ARG;
        }
    // the last user query: the last user message that is not a tool response turn
    int lq = -1;
    for (int i = n - 1; i >= 0 && lq < 0; i--) {
        if (msg[i].role != MSG_USER) continue;
        const char *t = msg[i].text;
        size_t l = strlen(t);
        if (!(l >= 15 && !strncmp(t, "<tool_response>", 15) && l >= 16 && !strcmp(t + l - 16, "</tool_response>"))) lq = i;
    }
    if (lq < 0) {
        if (!q38) { // (Qwen3.5's template raises; Unsloth's Qwen3.8 one takes the last message)
            snprintf(err, errlen, "no user query found in the messages (the %s chat template)", rq_chat_name(chat));
            return RQ_ERR_ARG;
        }
        lq = n - 1;
    }
    JBuf b = { 0 };
    // the system block: rq_chat_system (the effort instruction first for Qwen3.8) of the tool section + the system text
    int nt = tools_count(ts);
    for (int i = 0; i < nsys; i++)
        if (msg[i].text[0]) {
            if (b.n) jb_raw(&b, "\n", 1);
            jb_cstr(&b, msg[i].text);
        }
    if (nt > 0) {
        JBuf s = { 0 };
        jb_cstr(&s, ts->text);
        if (b.n) {
            jb_cstr(&s, "\n\n");
            jb_raw(&s, b.p, b.n);
        }
        if (s.oom) e = RQ_ERR_NOMEM;
        else e = rq_chat_system(m, s.p, thinking, out);
        jb_free(&s);
    } else if (q38 || nsys > 0) {
        if (b.oom) e = RQ_ERR_NOMEM;
        else e = rq_chat_system(m, q38 && !b.n ? NULL : b.p ? b.p : "", thinking, out);
    }
    jb_reset(&b);
    for (int i = nsys; i < n && e == RQ_OK; i++) {
        const Msg *x = &msg[i];
        if (x->role == MSG_USER) {
            jb_cstr(&b, "<|im_start|>user\n");
            jb_cstr(&b, x->text);
            jb_cstr(&b, "<|im_end|>\n");
        } else if (x->role == MSG_ASSISTANT) {
            jb_cstr(&b, "<|im_start|>assistant\n");
            // reasoning R and content C: reasoning_content|trim; else from a content with </think>: Qwen3.5's template
            // (R: before the first </think>, after the last <think> there; C: after the last </think>, lstrip('\n')),
            // runq's Qwen3.8 form (R: before the last </think>, after the last <think> there; C: after it; both trimmed)
            const char *t = x->text, *last = strstr(t, "</think>"), *p, *c = t;
            char *rt = NULL, *ct = NULL;
            if (x->reasoning) rt = trimmed_copy(m, x->reasoning, strlen(x->reasoning));
            else if (last) {
                const char *e1 = last, *rs = t;
                while ((p = strstr(last + 8, "</think>"))) last = p;
                const char *re = q38 ? last : e1;
                for (const char *q = t; (p = strstr(q, "<think>")) && p + 7 <= re; q = p + 7) rs = p + 7;
                rt = trimmed_copy(m, rs, (size_t)(re - rs));
                if (q38) c = ct = trimmed_copy(m, last + 8, strlen(last + 8));
                else c = last + 8 + strspn(last + 8, "\n");
            } else rt = trimmed_copy(m, "", 0);
            if (!rt || (q38 && last && !x->reasoning && !ct)) e = RQ_ERR_NOMEM;
            else if (render_assistant(&b, m, chat, rt, c, x->calls, x->ncalls, q38 || i > lq) < 0) e = RQ_ERR_NOMEM;
            free(rt);
            free(ct);
        } else { // tool
            if (i > 0 && msg[i - 1].role != MSG_TOOL) jb_cstr(&b, "<|im_start|>user");
            jb_cstr(&b, "\n<tool_response>\n");
            jb_cstr(&b, x->text);
            jb_cstr(&b, "\n</tool_response>");
            if (i == n - 1 || msg[i + 1].role != MSG_TOOL) jb_cstr(&b, "<|im_end|>\n");
        }
    }
    jb_cstr(&b, thinking ? "<|im_start|>assistant\n<think>\n" : "<|im_start|>assistant\n<think>\n\n</think>\n\n");
    if (e == RQ_OK && b.oom) e = RQ_ERR_NOMEM;
    if (e == RQ_OK) e = rq_tokenize(m, b.p, (long)b.n, out);
    jb_free(&b);
    if (e != RQ_OK && !err[0]) snprintf(err, errlen, "cannot render the conversation: %s", rq_strerror(e));
    return e;
}

// ------------------------------------------------------------------------------------------ parsing

void strlist_free(StrList *l) {
    for (int i = 0; i < l->n; i++) free(l->v[i]);
    free(l->v);
    memset(l, 0, sizeof(*l));
}

static int strlist_push(StrList *l, char *s) {
    if (!s) return -1;
    if (l->n == l->cap) {
        int c = l->cap ? 2 * l->cap : 4;
        char **v = realloc(l->v, c * sizeof(char *));
        if (!v) {
            free(s);
            return -1;
        }
        l->v = v;
        l->cap = c;
    }
    l->v[l->n++] = s;
    return 0;
}

static const char *find(const char *s, const char *end, const char *pat) {
    // the first occurrence of pat in [s, end), or NULL
    size_t pl = strlen(pat);
    for (const char *p = s; p + pl <= end; p++) {
        p = memchr(p, pat[0], (size_t)(end - p));
        if (!p || p + pl > end) return NULL;
        if (!memcmp(p, pat, pl)) return p;
    }
    return NULL;
}

static const char *skip_ws(const char *p, const char *end) {
    while (p < end && is_ws((unsigned char)*p)) p++;
    return p;
}

static int starts(const char *p, const char *end, const char *pat) {
    size_t pl = strlen(pat);
    return (size_t)(end - p) >= pl && !memcmp(p, pat, pl);
}

static int json_int_text(const char *s, size_t n) {
    // -?(0|[1-9][0-9]*)
    size_t i = 0;
    if (i < n && s[i] == '-') i++;
    if (i >= n) return 0;
    if (s[i] == '0') return i + 1 == n;
    if (s[i] < '1' || s[i] > '9') return 0;
    while (i < n && s[i] >= '0' && s[i] <= '9') i++;
    return i == n;
}

static int json_num_text(const char *s, size_t n) {
    // RFC 8259's number
    size_t i = 0;
    if (i < n && s[i] == '-') i++;
    if (i >= n) return 0;
    if (s[i] == '0') i++;
    else if (s[i] >= '1' && s[i] <= '9') while (i < n && s[i] >= '0' && s[i] <= '9') i++;
    else return 0;
    if (i < n && s[i] == '.') {
        size_t k = ++i;
        while (i < n && s[i] >= '0' && s[i] <= '9') i++;
        if (i == k) return 0;
    }
    if (i < n && (s[i] == 'e' || s[i] == 'E')) {
        i++;
        if (i < n && (s[i] == '+' || s[i] == '-')) i++;
        size_t k = i;
        while (i < n && s[i] >= '0' && s[i] <= '9') i++;
        if (i == k) return 0;
    }
    return i == n;
}

static void arg_value(JBuf *b, unsigned types, const char *v, size_t n) {
    // a parameter's text as the JSON value of its schema's types (the header comment)
    const char *t = v, *te = v + n;
    while (t < te && is_ws((unsigned char)*t)) t++;
    while (te > t && is_ws((unsigned char)te[-1])) te--;
    size_t tl = (size_t)(te - t);
    if ((types & T_NULL) && tl == 4 && !memcmp(t, "null", 4)) { jb_cstr(b, "null"); return; }
    if (!(types & T_STR)) {
        if ((types & T_INT) && json_int_text(t, tl)) { jb_raw(b, t, tl); return; }
        if ((types & T_NUM) && json_num_text(t, tl)) { jb_raw(b, t, tl); return; }
        if ((types & T_BOOL) && ((tl == 4 && !strncasecmp(t, "true", 4)) || (tl == 5 && !strncasecmp(t, "false", 5)))) {
            jb_cstr(b, tl == 4 ? "true" : "false");
            return;
        }
        if (types & (T_OBJ | T_ARR)) {
            char e[64];
            JsDoc *d = json_parse(t, tl, e, sizeof(e));
            if (d) {
                py_json(b, json_root(d));
                json_free(d);
                return;
            }
        }
    }
    py_str(b, v, n);
}

static int parse_function(const ToolSet *ts, const char **pp, const char *end, JBuf *call, unsigned long long id) {
    // "<function=NAME>" (params) ["</function>"] at *pp (after whitespace): the call's JSON members into call; 0 or -1
    const char *p = skip_ws(*pp, end);
    if (!starts(p, end, "<function=")) return -1;
    p += 10;
    const char *gt = memchr(p, '>', (size_t)(end - p)), *nl = memchr(p, '\n', (size_t)(end - p));
    if (!gt || (nl && nl < gt)) return -1;
    const char *fn = skip_ws(p, gt), *fe = gt;
    while (fe > fn && is_ws((unsigned char)fe[-1])) fe--;
    if (fe == fn) return -1;
    p = gt + 1;
    JBuf args = { 0 };
    const char **seen = NULL;
    size_t *seen_l = NULL;
    int nseen = 0, ok = 0;
    jb_raw(&args, "{", 1);
    for (;;) {
        p = skip_ws(p, end);
        if (p >= end) { ok = 1; break; } // (no </function>: the caller decides whether that is complete)
        if (starts(p, end, "</function>")) { p += 11; ok = 1; break; }
        if (!starts(p, end, "<parameter=")) break;
        p += 11;
        gt = memchr(p, '>', (size_t)(end - p));
        nl = memchr(p, '\n', (size_t)(end - p));
        if (!gt || (nl && nl < gt)) break;
        const char *pn = skip_ws(p, gt), *pe = gt;
        while (pe > pn && is_ws((unsigned char)pe[-1])) pe--;
        if (pe == pn) break;
        const char *v = gt + 1, *ve = find(v, end, "</parameter>"), *next;
        if (ve) next = ve + 12;
        else { // (no </parameter>: up to the next parameter or the end of the function)
            const char *a = find(v, end, "<parameter="), *f = find(v, end, "</function>");
            ve = a && (!f || a < f) ? a : f ? f : end;
            next = ve;
        }
        p = next;
        if (ve > v && *v == '\n') v++;
        if (ve > v && ve[-1] == '\n') ve--;
        int dup = 0;
        for (int i = 0; i < nseen && !dup; i++) dup = seen_l[i] == (size_t)(pe - pn) && !memcmp(seen[i], pn, seen_l[i]);
        if (dup) continue; // (a repeated parameter: the first)
        const char **s2 = realloc(seen, (nseen + 1) * sizeof(char *));
        size_t *l2 = s2 ? realloc(seen_l, (nseen + 1) * sizeof(size_t)) : NULL;
        if (s2) seen = s2;
        if (!s2 || !l2) break;
        seen_l = l2;
        seen[nseen] = pn;
        seen_l[nseen++] = (size_t)(pe - pn);
        const ToolParam *tp = tool_param(ts, fn, (size_t)(fe - fn), pn, (size_t)(pe - pn));
        if (nseen > 1) jb_cstr(&args, ", ");
        py_str(&args, pn, (size_t)(pe - pn));
        jb_cstr(&args, ": ");
        arg_value(&args, tp ? tp->types : 0, v, (size_t)(ve - v));
    }
    jb_raw(&args, "}", 1);
    free(seen);
    free(seen_l);
    if (ok && !args.oom) {
        jb_printf(call, "\"id\": \"call_%016llx\", \"type\": \"function\", \"function\": {\"name\": ", id);
        jb_str(call, fn, (size_t)(fe - fn));
        jb_cstr(call, ", \"arguments\": ");
        jb_str(call, args.p, args.n);
        jb_cstr(call, "}");
        *pp = p;
    }
    jb_free(&args);
    return ok && !call->oom ? 0 : -1;
}

static unsigned long long call_id(unsigned long long seed, int k) {
    unsigned long long x = seed + 0x9e3779b97f4a7c15ULL * (unsigned long long)(k + 1);
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

int tool_parse_calls(const ToolSet *ts, const char *s, size_t n, int complete, StrList *calls, unsigned long long seed) {
    // the inside of a <tool_call> block: its calls appended to calls (ids from seed and their index); the number of
    // calls, or -1 if the block is malformed (calls unchanged). complete: the block was closed (else its last
    // function needs its </function>)
    const char *p = s, *end = s + n;
    int n0 = calls->n;
    for (;;) {
        p = skip_ws(p, end);
        if (p >= end && calls->n > n0) return calls->n - n0;
        JBuf c = { 0 };
        const char *q = p;
        if (parse_function(ts, &q, end, &c, call_id(seed, calls->n)) < 0 ||
            (!complete && !(q - p >= 11 && !memcmp(q - 11, "</function>", 11)))) {
            jb_free(&c);
            break;
        }
        p = q;
        if (strlist_push(calls, c.p) < 0) break;
    }
    while (calls->n > n0) free(calls->v[--calls->n]);
    return -1;
}

// the scanner's states
enum { S_TEXT, S_CALL, S_AFTER, S_STOP };

void tscan_init(ToolScan *s, const ToolSet *ts, int single, unsigned long long seed) {
    memset(s, 0, sizeof(*s));
    s->ts = ts;
    s->single = single;
    s->seed = seed;
}

void tscan_free(ToolScan *s) {
    jb_free(&s->hold);
    jb_free(&s->ws);
}

static void hold_drop(JBuf *h, size_t k) {
    if (k == 0) return;
    memmove(h->p, h->p + k, h->n - k);
    h->n -= k;
}

int tscan_push(ToolScan *s, const char *bytes, size_t n, int final, JBuf *content, StrList *calls) {
    // the reply's content bytes (final: the reply is over): the content that is surely not part of a call into
    // content, completed calls into calls; 1 once the generation should stop
    if (s->state == S_STOP) return 1;
    if (n) jb_raw(&s->hold, bytes, n);
    JBuf *h = &s->hold;
    const size_t tl = strlen(TOOL_CALL);
    for (;;) {
        const char *p = h->p, *end = h->p + h->n;
        if (s->state == S_TEXT) {
            const char *k = h->n ? find(p, end, TOOL_CALL) : NULL;
            if (k) {
                const char *e = k;
                while (e > p && is_ws((unsigned char)e[-1])) e--;
                jb_raw(content, p, (size_t)(e - p));
                jb_reset(&s->ws);
                jb_raw(&s->ws, e, (size_t)(k - e)); // (back into the content if the block is not a call)
                hold_drop(h, (size_t)(k - p) + tl);
                s->state = S_CALL;
                continue;
            }
            size_t keep = 0;
            if (!final) { // (a suffix that may start a call, and the whitespace before it)
                for (size_t l = tl - 1; l > 0 && !keep; l--)
                    if (h->n >= l && !memcmp(end - l, TOOL_CALL, l)) keep = l;
                while (keep < h->n && is_ws((unsigned char)end[-(long)keep - 1])) keep++;
            }
            if (h->n > keep) jb_raw(content, p, h->n - keep);
            hold_drop(h, h->n - keep);
            return 0;
        }
        if (s->state == S_CALL) {
            const char *k = h->n ? find(p, end, TOOL_CALL_END) : NULL;
            if (!k && !final) return 0;
            int got = tool_parse_calls(s->ts, p, (size_t)((k ? k : end) - p), k != NULL, calls, s->seed);
            if (got < 0) {
                if (s->ncalls == 0) { // (not a call: content)
                    if (s->ws.n) jb_raw(content, s->ws.p, s->ws.n);
                    jb_cstr(content, TOOL_CALL);
                    size_t l = k ? (size_t)(k - p) + strlen(TOOL_CALL_END) : h->n;
                    jb_raw(content, p, l);
                    hold_drop(h, l);
                    s->state = S_TEXT;
                    if (!k) return 0;
                    continue;
                }
                s->state = S_STOP;
                h->n = 0;
                return 1;
            }
            s->ncalls += got;
            if (!k || s->single) {
                s->state = S_STOP;
                h->n = 0;
                return 1;
            }
            hold_drop(h, (size_t)(k - p) + strlen(TOOL_CALL_END));
            s->state = S_AFTER;
            continue;
        }
        if (s->state == S_AFTER) {
            const char *q = h->n ? skip_ws(p, end) : p;
            hold_drop(h, (size_t)(q - p));
            p = h->p;
            end = p + h->n;
            if (h->n >= tl && !memcmp(p, TOOL_CALL, tl)) {
                hold_drop(h, tl);
                s->state = S_CALL;
                continue;
            }
            if (!final && (h->n == 0 || (h->n < tl && !memcmp(p, TOOL_CALL, h->n)))) return 0; // (empty, or the start of another call)
            s->state = S_STOP; // (the end of the reply, or text after the calls)
            h->n = 0;
            return 1;
        }
        return 1;
    }
}
#endif
