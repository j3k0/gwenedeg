/* JSON parser and writer for runq serve (json.h). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "json.h"

// ------------------------------------------------------------------------------------------- arena

typedef struct Blk { struct Blk *next; size_t used, cap; } Blk;
#define BLK_HDR ((sizeof(Blk) + 15) & ~(size_t)15)
struct JsDoc { Blk *blk; Js *root; };

static void *arena(JsDoc *d, size_t n) {
    n = (n + 15) & ~(size_t)15;
    Blk *b = d->blk;
    if (!b || b->cap - b->used < n) {
        size_t cap = n > 65536 ? n : 65536;
        Blk *nb = malloc(BLK_HDR + cap);
        if (!nb) return NULL;
        nb->next = b;
        nb->used = 0;
        nb->cap = cap;
        d->blk = b = nb;
    }
    void *p = (char *)b + BLK_HDR + b->used;
    b->used += n;
    return p;
}

void json_free(JsDoc *d) {
    if (!d) return;
    for (Blk *b = d->blk, *nx; b; b = nx) {
        nx = b->next;
        free(b);
    }
    free(d);
}

const Js *json_root(const JsDoc *d) { return d ? d->root : NULL; }

// ------------------------------------------------------------------------------------------- parser

typedef struct {
    const unsigned char *p, *s, *end;
    JsDoc *d;
    char *err;
    size_t errlen;
    int failed;
} P;

static void fail(P *x, const char *msg) {
    if (!x->failed && x->err && x->errlen)
        snprintf(x->err, x->errlen, "%s at byte %ld", msg, (long)(x->p - x->s));
    x->failed = 1;
}

static void ws(P *x) {
    while (x->p < x->end && (*x->p == ' ' || *x->p == '\t' || *x->p == '\n' || *x->p == '\r')) x->p++;
}

static int utf8_len_at(const unsigned char *s, size_t n) {
    // length of the well-formed UTF-8 character at s (n bytes available), 0 if ill-formed
    unsigned c = s[0];
    if (c < 0x80) return 1;
    if (c >= 0xc2 && c <= 0xdf) return n >= 2 && (s[1] & 0xc0) == 0x80 ? 2 : 0;
    if (c >= 0xe0 && c <= 0xef) {
        if (n < 3 || (s[1] & 0xc0) != 0x80 || (s[2] & 0xc0) != 0x80) return 0;
        if (c == 0xe0 && s[1] < 0xa0) return 0; // overlong
        if (c == 0xed && s[1] >= 0xa0) return 0; // surrogates
        return 3;
    }
    if (c >= 0xf0 && c <= 0xf4) {
        if (n < 4 || (s[1] & 0xc0) != 0x80 || (s[2] & 0xc0) != 0x80 || (s[3] & 0xc0) != 0x80) return 0;
        if (c == 0xf0 && s[1] < 0x90) return 0; // overlong
        if (c == 0xf4 && s[1] >= 0x90) return 0; // > U+10FFFF
        return 4;
    }
    return 0;
}

size_t utf8_valid_prefix(const char *s, size_t n) {
    size_t i = 0;
    while (i < n) {
        int k = utf8_len_at((const unsigned char *)s + i, n - i);
        if (!k) break;
        i += k;
    }
    return i;
}

static int put_utf8(unsigned char *o, unsigned cp) {
    if (cp < 0x80) { o[0] = cp; return 1; }
    if (cp < 0x800) { o[0] = 0xc0 | cp >> 6; o[1] = 0x80 | (cp & 0x3f); return 2; }
    if (cp < 0x10000) { o[0] = 0xe0 | cp >> 12; o[1] = 0x80 | (cp >> 6 & 0x3f); o[2] = 0x80 | (cp & 0x3f); return 3; }
    o[0] = 0xf0 | cp >> 18; o[1] = 0x80 | (cp >> 12 & 0x3f); o[2] = 0x80 | (cp >> 6 & 0x3f); o[3] = 0x80 | (cp & 0x3f);
    return 4;
}

static int hex4(P *x, unsigned *v) {
    if (x->end - x->p < 4) return 0;
    unsigned r = 0;
    for (int i = 0; i < 4; i++) {
        int c = x->p[i], d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
        if (d < 0) return 0;
        r = r * 16 + (unsigned)d;
    }
    x->p += 4;
    *v = r;
    return 1;
}

static char *pstring(P *x, int *len) {
    // at the opening quote; the decoded string (arena), its length in *len
    x->p++;
    const unsigned char *q = x->p;
    size_t raw = 0;
    while (q < x->end && *q != '"') { // bound the output size: the decoded string is never longer
        if (*q == '\\' && q + 1 < x->end) q++;
        q++;
    }
    raw = (size_t)(q - x->p);
    if (q >= x->end) { fail(x, "unterminated string"); return NULL; }
    if (raw > 0x7ffffff0) { fail(x, "string too long"); return NULL; }
    unsigned char *o = arena(x->d, raw + 1), *w = o;
    if (!o) { fail(x, "out of memory"); return NULL; }
    while (x->p < x->end && *x->p != '"') {
        unsigned c = *x->p;
        if (c < 0x20) { fail(x, "control character in string"); return NULL; }
        if (c != '\\') {
            int k = utf8_len_at(x->p, (size_t)(x->end - x->p));
            if (!k) { fail(x, "invalid UTF-8 in string"); return NULL; }
            memcpy(w, x->p, k);
            w += k;
            x->p += k;
            continue;
        }
        x->p++;
        c = *x->p++;
        switch (c) {
            case '"': *w++ = '"'; break;
            case '\\': *w++ = '\\'; break;
            case '/': *w++ = '/'; break;
            case 'b': *w++ = '\b'; break;
            case 'f': *w++ = '\f'; break;
            case 'n': *w++ = '\n'; break;
            case 'r': *w++ = '\r'; break;
            case 't': *w++ = '\t'; break;
            case 'u': {
                unsigned cp, lo;
                if (!hex4(x, &cp)) { x->p--; fail(x, "bad \\u escape"); return NULL; }
                if (cp >= 0xd800 && cp < 0xdc00) { // high surrogate: a low one must follow
                    if (x->end - x->p >= 6 && x->p[0] == '\\' && x->p[1] == 'u') {
                        const unsigned char *save = x->p;
                        x->p += 2;
                        if (hex4(x, &lo) && lo >= 0xdc00 && lo < 0xe000)
                            cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                        else {
                            x->p = save;
                            cp = 0xfffd;
                        }
                    } else
                        cp = 0xfffd;
                } else if (cp >= 0xdc00 && cp < 0xe000)
                    cp = 0xfffd; // lone low surrogate
                w += put_utf8(w, cp); // (at most 6 escape bytes -> at most 4 output bytes)
                break;
            }
            default: x->p--; fail(x, "bad escape in string"); return NULL;
        }
    }
    if (x->p >= x->end) { fail(x, "unterminated string"); return NULL; }
    x->p++;
    *w = 0;
    *len = (int)(w - o);
    return (char *)o;
}

static Js *node(P *x, JsType t) {
    Js *v = arena(x->d, sizeof(Js));
    if (!v) { fail(x, "out of memory"); return NULL; }
    memset(v, 0, sizeof(*v));
    v->t = t;
    return v;
}

typedef struct Item { Js *v; char *k; int kn; struct Item *next; } Item;

static Js *pvalue(P *x, int depth);

static Js *pcontainer(P *x, int depth, int obj) {
    Js *v = node(x, obj ? JS_OBJ : JS_ARR);
    if (!v) return NULL;
    x->p++;
    ws(x);
    Item *head = NULL, **tail = &head;
    int n = 0;
    if (x->p < x->end && *x->p == (obj ? '}' : ']')) {
        x->p++;
        return v;
    }
    for (;;) {
        Item *it = arena(x->d, sizeof(Item));
        if (!it) { fail(x, "out of memory"); return NULL; }
        it->next = NULL;
        it->k = NULL;
        it->kn = 0;
        ws(x);
        if (obj) {
            if (x->p >= x->end || *x->p != '"') { fail(x, "expected a member name"); return NULL; }
            if (!(it->k = pstring(x, &it->kn))) return NULL;
            ws(x);
            if (x->p >= x->end || *x->p != ':') { fail(x, "expected ':'"); return NULL; }
            x->p++;
        }
        if (!(it->v = pvalue(x, depth + 1))) return NULL;
        *tail = it;
        tail = &it->next;
        if (++n > 0x7fffff) { fail(x, "too many items"); return NULL; }
        ws(x);
        if (x->p < x->end && *x->p == ',') { x->p++; continue; }
        if (x->p < x->end && *x->p == (obj ? '}' : ']')) { x->p++; break; }
        fail(x, obj ? "expected ',' or '}'" : "expected ',' or ']'");
        return NULL;
    }
    v->n = n;
    v->v = arena(x->d, n * sizeof(Js *));
    if (obj) {
        v->k = arena(x->d, n * sizeof(char *));
        v->kn = arena(x->d, n * sizeof(int));
    }
    if (!v->v || (obj && (!v->k || !v->kn))) { fail(x, "out of memory"); return NULL; }
    int i = 0;
    for (Item *it = head; it; it = it->next, i++) {
        v->v[i] = it->v;
        if (obj) {
            v->k[i] = it->k;
            v->kn[i] = it->kn;
        }
    }
    return v;
}

static Js *pnumber(P *x) {
    const unsigned char *s = x->p;
    if (x->p < x->end && *x->p == '-') x->p++;
    if (x->p >= x->end || *x->p < '0' || *x->p > '9') { fail(x, "invalid number"); return NULL; }
    if (*x->p == '0') x->p++;
    else
        while (x->p < x->end && *x->p >= '0' && *x->p <= '9') x->p++;
    if (x->p < x->end && *x->p == '.') {
        x->p++;
        if (x->p >= x->end || *x->p < '0' || *x->p > '9') { fail(x, "invalid number"); return NULL; }
        while (x->p < x->end && *x->p >= '0' && *x->p <= '9') x->p++;
    }
    if (x->p < x->end && (*x->p == 'e' || *x->p == 'E')) {
        x->p++;
        if (x->p < x->end && (*x->p == '+' || *x->p == '-')) x->p++;
        if (x->p >= x->end || *x->p < '0' || *x->p > '9') { fail(x, "invalid number"); return NULL; }
        while (x->p < x->end && *x->p >= '0' && *x->p <= '9') x->p++;
    }
    size_t n = (size_t)(x->p - s);
    if (n > 512) { fail(x, "number too long"); return NULL; }
    Js *v = node(x, JS_NUM);
    if (!v) return NULL;
    v->s = arena(x->d, n + 1);
    if (!v->s) { fail(x, "out of memory"); return NULL; }
    memcpy(v->s, s, n);
    v->s[n] = 0;
    v->n = (int)n;
    v->num = strtod(v->s, NULL);
    return v;
}

static Js *pvalue(P *x, int depth) {
    if (depth > JSON_MAX_DEPTH) { fail(x, "nesting too deep"); return NULL; }
    ws(x);
    if (x->p >= x->end) { fail(x, "unexpected end of input"); return NULL; }
    switch (*x->p) {
        case '{': return pcontainer(x, depth, 1);
        case '[': return pcontainer(x, depth, 0);
        case '"': {
            Js *v = node(x, JS_STR);
            if (v && !(v->s = pstring(x, &v->n))) return NULL;
            return v;
        }
        case 't':
            if (x->end - x->p >= 4 && !memcmp(x->p, "true", 4)) { x->p += 4; return node(x, JS_TRUE); }
            break;
        case 'f':
            if (x->end - x->p >= 5 && !memcmp(x->p, "false", 5)) { x->p += 5; return node(x, JS_FALSE); }
            break;
        case 'n':
            if (x->end - x->p >= 4 && !memcmp(x->p, "null", 4)) { x->p += 4; return node(x, JS_NULL); }
            break;
        default:
            if (*x->p == '-' || (*x->p >= '0' && *x->p <= '9')) return pnumber(x);
    }
    fail(x, "invalid value");
    return NULL;
}

JsDoc *json_parse(const char *text, size_t len, char *err, size_t errlen) {
    JsDoc *d = calloc(1, sizeof(JsDoc));
    if (!d) {
        if (err && errlen) snprintf(err, errlen, "out of memory");
        return NULL;
    }
    P x = { (const unsigned char *)text, (const unsigned char *)text, (const unsigned char *)text + len, d, err, errlen, 0 };
    if (len >= 3 && !memcmp(text, "\xef\xbb\xbf", 3)) x.p += 3; // (a UTF-8 BOM: tolerated)
    d->root = pvalue(&x, 0);
    if (d->root) {
        ws(&x);
        if (x.p != x.end) fail(&x, "trailing characters after the JSON value");
    }
    if (x.failed || !d->root) {
        if (!x.failed) fail(&x, "invalid JSON");
        json_free(d);
        return NULL;
    }
    return d;
}

const Js *json_get(const Js *o, const char *key) {
    if (!o || o->t != JS_OBJ) return NULL;
    size_t kl = strlen(key);
    for (int i = o->n - 1; i >= 0; i--)
        if ((size_t)o->kn[i] == kl && !memcmp(o->k[i], key, kl)) return o->v[i];
    return NULL;
}

int json_is_int(const Js *v, long long *out) {
    if (!v || v->t != JS_NUM) return 0;
    double d = v->num;
    if (!(d >= -9007199254740992.0 && d <= 9007199254740992.0) || d != (double)(long long)d) return 0;
    if (out) *out = (long long)d;
    return 1;
}

int json_str_eq(const Js *v, const char *s) {
    return v && v->t == JS_STR && (size_t)v->n == strlen(s) && !memcmp(v->s, s, v->n);
}

// ------------------------------------------------------------------------------------------- writer

void jb_free(JBuf *b) {
    free(b->p);
    memset(b, 0, sizeof(*b));
}

void jb_reset(JBuf *b) {
    b->n = 0;
    if (b->p) b->p[0] = 0;
}

static int jb_grow(JBuf *b, size_t add) {
    if (b->oom) return 0;
    if (b->n + add + 1 <= b->cap) return 1;
    size_t cap = b->cap ? b->cap : 256;
    while (cap < b->n + add + 1) cap *= 2;
    char *p = realloc(b->p, cap);
    if (!p) { b->oom = 1; return 0; }
    b->p = p;
    b->cap = cap;
    return 1;
}

void jb_raw(JBuf *b, const char *s, size_t n) {
    if (!jb_grow(b, n)) return;
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = 0;
}

void jb_cstr(JBuf *b, const char *s) { jb_raw(b, s, strlen(s)); }

void jb_printf(JBuf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char tmp[256];
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if ((size_t)n < sizeof(tmp)) {
        jb_raw(b, tmp, n);
        return;
    }
    if (!jb_grow(b, n)) return;
    va_start(ap, fmt);
    vsnprintf(b->p + b->n, n + 1, fmt, ap);
    va_end(ap);
    b->n += n;
}

void jb_str(JBuf *b, const char *s, size_t n) {
    static const char hexd[] = "0123456789abcdef";
    if (!jb_grow(b, n + 2)) return;
    b->p[b->n++] = '"';
    const unsigned char *u = (const unsigned char *)s;
    size_t i = 0;
    while (i < n) {
        unsigned c = u[i];
        if (c >= 0x80) {
            int k = utf8_len_at(u + i, n - i);
            if (k) jb_raw(b, s + i, k);
            else jb_raw(b, "\xef\xbf\xbd", 3); // U+FFFD for each byte not part of a valid character
            i += k ? k : 1;
            continue;
        }
        i++;
        char e[8];
        switch (c) {
            case '"': jb_raw(b, "\\\"", 2); break;
            case '\\': jb_raw(b, "\\\\", 2); break;
            case '\n': jb_raw(b, "\\n", 2); break;
            case '\r': jb_raw(b, "\\r", 2); break;
            case '\t': jb_raw(b, "\\t", 2); break;
            case '\b': jb_raw(b, "\\b", 2); break;
            case '\f': jb_raw(b, "\\f", 2); break;
            default:
                if (c < 0x20 || c == 0x7f) {
                    memcpy(e, "\\u00", 4);
                    e[4] = hexd[c >> 4];
                    e[5] = hexd[c & 15];
                    jb_raw(b, e, 6);
                } else {
                    e[0] = (char)c;
                    jb_raw(b, e, 1);
                }
        }
    }
    jb_raw(b, "\"", 1);
}

void jb_strz(JBuf *b, const char *s) { jb_str(b, s, strlen(s)); }

void jb_value(JBuf *b, const Js *v, const char *colon, const char *comma) {
    switch (v->t) {
        case JS_NULL: jb_cstr(b, "null"); break;
        case JS_TRUE: jb_cstr(b, "true"); break;
        case JS_FALSE: jb_cstr(b, "false"); break;
        case JS_NUM: jb_raw(b, v->s, v->n); break;
        case JS_STR: jb_str(b, v->s, v->n); break;
        case JS_ARR:
            jb_raw(b, "[", 1);
            for (int i = 0; i < v->n; i++) {
                if (i) jb_cstr(b, comma);
                jb_value(b, v->v[i], colon, comma);
            }
            jb_raw(b, "]", 1);
            break;
        case JS_OBJ:
            jb_raw(b, "{", 1);
            for (int i = 0; i < v->n; i++) {
                if (i) jb_cstr(b, comma);
                jb_str(b, v->k[i], v->kn[i]);
                jb_cstr(b, colon);
                jb_value(b, v->v[i], colon, comma);
            }
            jb_raw(b, "}", 1);
            break;
    }
}
