/* Constrained decoding for response_format json_schema (serve.h): schemas with a finite set of outputs.
 *
 * 1. Schema -> the set of valid outputs, each in one canonical formatting: Python's json.dumps style
 *    on one line, `{"key": value, "key2": value2}` (": " and ", "; non-ASCII as UTF-8), which is what
 *    Qwen3 writes for a small object (asked for {"tier": ...} it answers {"tier": "MEDIUM"}). Supported:
 *    enum, const, boolean, null, integer with a small minimum..maximum range, objects of those (nested;
 *    properties in schema order; required ones always, optional ones present or absent; additional
 *    properties never), anyOf / oneOf of those, a single allOf, $ref to #/$defs/..., #/definitions/...
 *    (any local JSON pointer). Anything with an unbounded set of outputs (a string without enum, a
 *    number, an array, ...) is rejected with a message, as is a set of more than CON_MAX_STRINGS.
 * 2. The strings go into a byte trie. The output so far is a trie node.
 * 3. Per step: if all valid continuations share a first part (no decision before the next branch),
 *    its canonical tokens are forced: the longest common prefix of the tokenizations of the remaining
 *    suffixes of the candidate strings, so the forced tokens are what the tokenizer would produce for
 *    every candidate (e.g. `{"`, `tier`, `":`, ` "` before the tier label). They need no forward pass
 *    of their own: the caller feeds them with the next decision's step, or never if the string is then
 *    complete. Otherwise the step is a decision, masked at the byte level: every (non-added) vocab
 *    token whose bytes are a prefix of some remaining suffix is allowed. Byte-level rather than a trie
 *    of canonical token sequences: the model keeps all its ways of spelling a valid string (e.g. "S" +
 *    "IMPLE" as well as "SIM" + "PLE"), so the masked distribution is the model's distribution over
 *    the valid strings, not only over their canonical tokenizations; the cost is a trie walk per
 *    decision, bounded by the vocabulary's prefix structure (binary search in the sorted token bytes).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "serve.h"

#define CON_MAX_STRINGS 4096       // valid outputs at most
#define CON_MAX_BYTES (1 << 20)    // their total size
#define CON_MAX_DEPTH 32           // schema nesting / $ref chain depth
#define CON_MAX_RANGE 256          // integer minimum..maximum values

typedef struct { char *s; int n; } Str;
typedef struct { Str *v; int n, cap; size_t bytes; } StrSet;

typedef struct { int child, sibling, depth, rep; unsigned char byte, term; } Node;

struct ConSet {
    Str *s;
    int n;
    Node *node;
    int nnode;
};

// ------------------------------------------------------------------------------- schema -> strings

typedef struct {
    const Js *root;
    char *err;
    size_t errlen;
    int failed;
} Gen;

static int gfail(Gen *g, const char *path, const char *msg) {
    if (!g->failed) snprintf(g->err, g->errlen, "%s%s%s", msg, path[0] ? " at " : "", path);
    g->failed = 1;
    return -1;
}

static void set_free(StrSet *a) {
    for (int i = 0; i < a->n; i++) free(a->v[i].s);
    free(a->v);
    memset(a, 0, sizeof(*a));
}

static int set_add(Gen *g, StrSet *a, const char *s, size_t n) {
    if (a->n >= CON_MAX_STRINGS) return gfail(g, "", "the schema allows too many different outputs (more than 4096)");
    if (a->bytes + n > CON_MAX_BYTES) return gfail(g, "", "the schema's outputs are too long (more than 1 MB in total)");
    if (a->n == a->cap) {
        int cap = a->cap ? 2 * a->cap : 16;
        Str *v = realloc(a->v, cap * sizeof(Str));
        if (!v) return gfail(g, "", "out of memory");
        a->v = v;
        a->cap = cap;
    }
    char *c = malloc(n + 1);
    if (!c) return gfail(g, "", "out of memory");
    memcpy(c, s, n);
    c[n] = 0;
    a->v[a->n].s = c;
    a->v[a->n].n = (int)n;
    a->n++;
    a->bytes += n;
    return 0;
}

static int add_value(Gen *g, StrSet *a, const Js *v) {
    JBuf b = { 0 };
    jb_value(&b, v, ": ", ", ");
    int r = b.oom ? gfail(g, "", "out of memory") : set_add(g, a, b.p ? b.p : "", b.n);
    jb_free(&b);
    return r;
}

static const Js *resolve_ref(Gen *g, const Js *ref, const char *path) {
    // a local JSON pointer: "#", "#/$defs/Name", "#/definitions/Name", "#/a/b/0" (~0 ~1 escapes)
    if (ref->t != JS_STR || ref->n < 1 || ref->s[0] != '#') {
        gfail(g, path, "only local $ref (#/...) are supported");
        return NULL;
    }
    const Js *cur = g->root;
    const char *p = ref->s + 1, *end = ref->s + ref->n;
    if (p < end && *p != '/') { gfail(g, path, "bad $ref"); return NULL; }
    while (p < end && cur) {
        p++; // '/'
        const char *q = p;
        while (q < end && *q != '/') q++;
        char tok[256];
        size_t n = 0;
        for (const char *r = p; r < q; r++) {
            char c = *r;
            if (c == '~' && r + 1 < q && (r[1] == '0' || r[1] == '1')) c = *++r == '0' ? '~' : '/';
            if (n + 1 >= sizeof(tok)) { gfail(g, path, "$ref too long"); return NULL; }
            tok[n++] = c;
        }
        tok[n] = 0;
        if (cur->t == JS_OBJ) cur = json_get(cur, tok);
        else if (cur->t == JS_ARR) {
            char *e;
            long i = strtol(tok, &e, 10);
            cur = *e || i < 0 || i >= cur->n ? NULL : cur->v[i];
        } else cur = NULL;
        p = q;
    }
    if (!cur) gfail(g, path, "$ref target not found");
    return cur;
}

static int gen(Gen *g, const Js *sc, int depth, const char *path, StrSet *out);

static int gen_union(Gen *g, const Js *arr, int depth, const char *path, StrSet *out) {
    char sub[512];
    if (arr->t != JS_ARR || arr->n == 0) return gfail(g, path, "anyOf / oneOf must be a non-empty array");
    for (int i = 0; i < arr->n; i++) {
        snprintf(sub, sizeof(sub), "%s/%d", path, i);
        if (gen(g, arr->v[i], depth + 1, sub, out)) return -1;
    }
    return 0;
}

static int gen_object(Gen *g, const Js *sc, int depth, const char *path, StrSet *out) {
    const Js *props = json_get(sc, "properties"), *req = json_get(sc, "required"), *ap = json_get(sc, "additionalProperties");
    if (props && props->t != JS_OBJ) return gfail(g, path, "properties must be an object");
    if (req && req->t != JS_ARR) return gfail(g, path, "required must be an array");
    if (!props && !(ap && ap->t == JS_FALSE))
        return gfail(g, path, "an object without properties allows any object (not a finite set of outputs)");
    int np = props ? props->n : 0, rc = -1;
    StrSet *vals = calloc(np ? np : 1, sizeof(StrSet));
    char *opt = calloc(np ? np : 1, 1);
    int *idx = calloc(np ? np : 1, sizeof(int));
    JBuf b = { 0 };
    if (!vals || !opt || !idx) { gfail(g, path, "out of memory"); goto done; }
    for (int i = 0; i < np; i++) {
        char sub[512];
        snprintf(sub, sizeof(sub), "%s/properties/%.*s", path, props->kn[i] > 200 ? 200 : props->kn[i], props->k[i]);
        if (gen(g, props->v[i], depth + 1, sub, &vals[i])) goto done;
        opt[i] = 1;
        for (int r = 0; req && r < req->n; r++)
            if (req->v[r]->t == JS_STR && req->v[r]->n == props->kn[i] && !memcmp(req->v[r]->s, props->k[i], props->kn[i]))
                opt[i] = 0;
    }
    for (int r = 0; req && r < req->n; r++) {
        int found = 0;
        for (int i = 0; i < np && !found; i++)
            found = req->v[r]->t == JS_STR && req->v[r]->n == props->kn[i] && !memcmp(req->v[r]->s, props->k[i], props->kn[i]);
        if (!found) { gfail(g, path, "a required property is not in properties"); goto done; }
    }
    // every combination: property i takes value idx[i] (opt: idx == vals.n means absent), in order
    for (;;) {
        jb_reset(&b);
        jb_raw(&b, "{", 1);
        int first = 1;
        for (int i = 0; i < np; i++) {
            if (idx[i] == vals[i].n) continue; // (optional, absent)
            if (!first) jb_raw(&b, ", ", 2);
            first = 0;
            jb_str(&b, props->k[i], props->kn[i]);
            jb_raw(&b, ": ", 2);
            jb_raw(&b, vals[i].v[idx[i]].s, vals[i].v[idx[i]].n);
        }
        jb_raw(&b, "}", 1);
        if (b.oom) { gfail(g, path, "out of memory"); goto done; }
        if (set_add(g, out, b.p, b.n)) goto done;
        int i = np - 1;
        for (; i >= 0; i--) {
            if (++idx[i] < vals[i].n + opt[i]) break;
            idx[i] = 0;
        }
        if (i < 0) break;
    }
    rc = 0;
done:
    for (int i = 0; vals && i < np; i++) set_free(&vals[i]);
    free(vals);
    free(opt);
    free(idx);
    jb_free(&b);
    return rc;
}

static int gen_type(Gen *g, const Js *sc, const char *type, int depth, const char *path, StrSet *out) {
    if (!strcmp(type, "boolean")) return set_add(g, out, "true", 4) || set_add(g, out, "false", 5) ? -1 : 0;
    if (!strcmp(type, "null")) return set_add(g, out, "null", 4);
    if (!strcmp(type, "object")) return gen_object(g, sc, depth, path, out);
    if (!strcmp(type, "integer")) {
        const Js *mn = json_get(sc, "minimum"), *mx = json_get(sc, "maximum");
        const Js *xmn = json_get(sc, "exclusiveMinimum"), *xmx = json_get(sc, "exclusiveMaximum");
        long long lo = 0, hi = -1;
        int has_lo = 0, has_hi = 0;
        const Js *b4[4] = { mn, mx, xmn, xmx };
        for (int i = 0; i < 4; i++)
            if (b4[i] && b4[i]->t == JS_NUM && !(b4[i]->num > -1e12 && b4[i]->num < 1e12))
                return gfail(g, path, "integer bounds out of range");
        if (mn && mn->t == JS_NUM) { lo = (long long)mn->num; if ((double)lo < mn->num) lo++; has_lo = 1; }
        if (xmn && xmn->t == JS_NUM) { long long v = (long long)xmn->num; if ((double)v <= xmn->num) v++; if (!has_lo || v > lo) lo = v; has_lo = 1; }
        if (mx && mx->t == JS_NUM) { hi = (long long)mx->num; if ((double)hi > mx->num) hi--; has_hi = 1; }
        if (xmx && xmx->t == JS_NUM) { long long v = (long long)xmx->num; if ((double)v >= xmx->num) v--; if (!has_hi || v < hi) hi = v; has_hi = 1; }
        if (!has_lo || !has_hi || !(mn || xmn) || hi - lo >= CON_MAX_RANGE || lo < -1000000000LL || hi > 1000000000LL)
            return gfail(g, path, "an integer needs minimum and maximum at most 256 values apart");
        for (long long v = lo; v <= hi; v++) {
            char t[32];
            int n = snprintf(t, sizeof(t), "%lld", v);
            if (set_add(g, out, t, n)) return -1;
        }
        return 0;
    }
    char msg[160];
    snprintf(msg, sizeof(msg), "type \"%.40s\" without enum / const has no finite set of outputs", type);
    return gfail(g, path, msg);
}

static int gen(Gen *g, const Js *sc, int depth, const char *path, StrSet *out) {
    if (depth > CON_MAX_DEPTH) return gfail(g, path, "schema nested too deeply (or a recursive $ref)");
    if (sc->t == JS_TRUE) return gfail(g, path, "the schema true allows any value");
    if (sc->t != JS_OBJ) return gfail(g, path, "a schema must be an object");
    const Js *v;
    if ((v = json_get(sc, "$ref"))) {
        const Js *t = resolve_ref(g, v, path);
        if (!t) return -1;
        char sub[512];
        snprintf(sub, sizeof(sub), "%.200s", v->s);
        return gen(g, t, depth + 1, sub, out);
    }
    if ((v = json_get(sc, "enum"))) {
        if (v->t != JS_ARR || v->n == 0) return gfail(g, path, "enum must be a non-empty array");
        for (int i = 0; i < v->n; i++)
            if (add_value(g, out, v->v[i])) return -1;
        return 0;
    }
    if ((v = json_get(sc, "const"))) return add_value(g, out, v);
    if ((v = json_get(sc, "anyOf"))) return gen_union(g, v, depth, path, out);
    if ((v = json_get(sc, "oneOf"))) return gen_union(g, v, depth, path, out);
    if ((v = json_get(sc, "allOf"))) {
        if (v->t != JS_ARR || v->n != 1) return gfail(g, path, "allOf is only supported with one schema");
        return gen(g, v->v[0], depth + 1, path, out);
    }
    const Js *type = json_get(sc, "type");
    if (!type) {
        if (json_get(sc, "properties")) return gen_object(g, sc, depth, path, out);
        return gfail(g, path, "a schema without type / enum / const / properties allows any value");
    }
    if (type->t == JS_STR) return gen_type(g, sc, type->s, depth, path, out);
    if (type->t == JS_ARR && type->n > 0) {
        for (int i = 0; i < type->n; i++) {
            if (type->v[i]->t != JS_STR) return gfail(g, path, "bad type");
            if (gen_type(g, sc, type->v[i]->s, depth, path, out)) return -1;
        }
        return 0;
    }
    return gfail(g, path, "bad type");
}

static int str_cmp(const void *a, const void *b) {
    const Str *x = a, *y = b;
    int n = x->n < y->n ? x->n : y->n, c = memcmp(x->s, y->s, n);
    return c ? c : (x->n > y->n) - (x->n < y->n);
}

ConSet *con_compile(const Js *schema, char *err, size_t errlen) {
    Gen g = { schema, err, errlen, 0 };
    StrSet a = { 0 };
    if (gen(&g, schema, 0, "", &a) || a.n == 0) {
        if (!g.failed) snprintf(err, errlen, "the schema allows no output");
        set_free(&a);
        return NULL;
    }
    qsort(a.v, a.n, sizeof(Str), str_cmp); // (sorted, duplicates removed)
    int n = 0;
    for (int i = 0; i < a.n; i++) {
        if (n > 0 && !str_cmp(&a.v[n - 1], &a.v[i])) { free(a.v[i].s); continue; }
        a.v[n++] = a.v[i];
    }
    a.n = n;
    ConSet *cs = calloc(1, sizeof(ConSet));
    size_t maxnodes = a.bytes + 1;
    if (cs) cs->node = malloc(maxnodes * sizeof(Node));
    if (!cs || !cs->node) {
        snprintf(err, errlen, "out of memory");
        if (cs) free(cs->node);
        free(cs);
        set_free(&a);
        return NULL;
    }
    cs->s = a.v;
    cs->n = a.n;
    cs->nnode = 1;
    cs->node[0] = (Node){ -1, -1, 0, 0, 0, 0 };
    for (int i = 0; i < a.n; i++) { // the trie (children in byte order: the strings are sorted)
        int cur = 0;
        for (int k = 0; k < a.v[i].n; k++) {
            unsigned char c = (unsigned char)a.v[i].s[k];
            int ch = cs->node[cur].child, last = -1;
            while (ch >= 0 && cs->node[ch].byte != c) { last = ch; ch = cs->node[ch].sibling; }
            if (ch < 0) {
                ch = cs->nnode++;
                cs->node[ch] = (Node){ -1, -1, k + 1, i, c, 0 };
                if (last < 0) cs->node[cur].child = ch;
                else cs->node[last].sibling = ch;
            }
            cur = ch;
        }
        cs->node[cur].term = 1;
    }
    return cs;
}

void con_free(ConSet *cs) {
    if (!cs) return;
    for (int i = 0; i < cs->n; i++) free(cs->s[i].s);
    free(cs->s);
    free(cs->node);
    free(cs);
}

int con_count(const ConSet *cs) { return cs->n; }
const char *con_string(const ConSet *cs, int i, int *len) {
    *len = cs->s[i].n;
    return cs->s[i].s;
}

// ------------------------------------------------------------------------------------ token index

struct ConVocab {
    const RqModel *m;
    int *ids;    // usable tokens (non-empty bytes, not added tokens), sorted by their bytes
    int n;
    int eos;
};

static const RqModel *sort_model;
static int tok_cmp(const void *a, const void *b) {
    int la, lb;
    const char *x = rq_token_bytes(sort_model, *(const int *)a, &la), *y = rq_token_bytes(sort_model, *(const int *)b, &lb);
    int c = memcmp(x, y, la < lb ? la : lb);
    return c ? c : la != lb ? (la > lb) - (la < lb) : (*(const int *)a > *(const int *)b) - (*(const int *)a < *(const int *)b);
}

ConVocab *con_vocab_new(const RqModel *m) {
    RqModelInfo info;
    rq_model_info(m, &info);
    ConVocab *v = calloc(1, sizeof(ConVocab));
    if (!v || !(v->ids = malloc(info.vocab_size * sizeof(int)))) {
        free(v);
        return NULL;
    }
    v->m = m;
    v->eos = info.eos;
    for (int t = 0; t < info.vocab_size; t++) {
        int len;
        rq_token_bytes(m, t, &len);
        if (len > 0 && !rq_token_is_added(m, t)) v->ids[v->n++] = t;
    }
    sort_model = m; // (once, at startup, before any thread uses it)
    qsort(v->ids, v->n, sizeof(int), tok_cmp);
    return v;
}

void con_vocab_free(ConVocab *v) {
    if (!v) return;
    free(v->ids);
    free(v);
}

static int lower_bound(const ConVocab *v, const unsigned char *p, int n) {
    // first index whose token bytes are >= p[0..n)
    int lo = 0, hi = v->n;
    while (lo < hi) {
        int mid = (lo + hi) / 2, len;
        const char *s = rq_token_bytes(v->m, v->ids[mid], &len);
        int c = memcmp(s, p, len < n ? len : n);
        if (c < 0 || (c == 0 && len < n)) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// ------------------------------------------------------------------------------------------ state

void con_start(ConState *st, const ConSet *cs) {
    st->cs = cs;
    st->node = 0;
    st->done = cs->node[0].term && cs->node[0].child < 0;
}

int con_advance(ConState *st, const char *bytes, int len) {
    const ConSet *cs = st->cs;
    int cur = st->node;
    for (int k = 0; k < len; k++) {
        int ch = cs->node[cur].child;
        while (ch >= 0 && cs->node[ch].byte != (unsigned char)bytes[k]) ch = cs->node[ch].sibling;
        if (ch < 0) return 0;
        cur = ch;
    }
    st->node = cur;
    st->done = cs->node[cur].term && cs->node[cur].child < 0;
    return 1;
}

static void lcp_tokens(RqTokens *acc, int *have, const RqTokens *t) {
    if (!*have) {
        acc->n = 0;
        rq_tokens_push(acc, t->v, t->n);
        *have = 1;
        return;
    }
    int k = 0;
    while (k < acc->n && k < t->n && acc->v[k] == t->v[k]) k++;
    acc->n = k;
}

int con_forced(ConState *st, const RqModel *m, RqTokens *out) {
    const ConSet *cs = st->cs;
    const Node *nd = &cs->node[st->node];
    if (st->done || nd->term || nd->child < 0 || cs->node[nd->child].sibling >= 0)
        return RQ_OK; // complete, or a decision right here
    // the candidates: the strings with this node's prefix (rep: one of them)
    const Str *rep = &cs->s[nd->rep];
    int depth = nd->depth, ncand = 0, have = 0, e = RQ_OK;
    RqTokens acc = { 0 }, t = { 0 };
    for (int i = 0; i < cs->n; i++)
        ncand += cs->s[i].n >= depth && !memcmp(cs->s[i].s, rep->s, depth);
    if (ncand <= 64) {
        for (int i = 0; i < cs->n && e == RQ_OK; i++) {
            if (cs->s[i].n < depth || memcmp(cs->s[i].s, rep->s, depth)) continue;
            t.n = 0;
            e = rq_tokenize(m, cs->s[i].s + depth, cs->s[i].n - depth, &t);
            if (e == RQ_OK) lcp_tokens(&acc, &have, &t);
        }
    } else { // many candidates: the single-path run up to the branch, without its last token
        int cur = st->node, k = depth;
        while (!cs->node[cur].term && cs->node[cur].child >= 0 && cs->node[cs->node[cur].child].sibling < 0) {
            cur = cs->node[cur].child;
            k++;
        }
        e = rq_tokenize(m, rep->s + depth, k - depth, &acc);
        if (acc.n > 0) acc.n--;
    }
    for (int i = 0; i < acc.n && e == RQ_OK; i++) {
        int len;
        const char *b = rq_token_bytes(m, acc.v[i], &len);
        if (!con_advance(st, b, len)) { e = RQ_ERR_STATE; break; } // (cannot happen: the bytes are a common prefix)
        e = rq_tokens_push(out, &acc.v[i], 1);
    }
    rq_tokens_free(&acc);
    rq_tokens_free(&t);
    return e;
}

typedef struct { const ConSet *cs; const ConVocab *v; unsigned char *allow; int *list, max, n; unsigned char buf[1024]; } MaskW;

static void mask_walk(MaskW *w, int node, int len) {
    // the children of node, each extending the byte path buf[0..len)
    for (int ch = w->cs->node[node].child; ch >= 0; ch = w->cs->node[ch].sibling) {
        if (len >= (int)sizeof(w->buf)) return;
        w->buf[len] = w->cs->node[ch].byte;
        int n = len + 1, i = lower_bound(w->v, w->buf, n), more = 0;
        for (; i < w->v->n; i++) {
            int tl;
            const char *s = rq_token_bytes(w->v->m, w->v->ids[i], &tl);
            if (tl < n || memcmp(s, w->buf, n)) break;
            if (tl > n) { more = 1; break; } // (a longer token starts with this path: descend)
            int t = w->v->ids[i];
            if (!w->allow[t] && w->n < w->max) {
                w->allow[t] = 1;
                w->list[w->n++] = t;
            }
        }
        if (more) mask_walk(w, ch, n);
    }
}

int con_mask(const ConState *st, const ConVocab *v, unsigned char *allow, int *list, int maxlist) {
    MaskW w = { st->cs, v, allow, list, maxlist, 0, { 0 } };
    if (st->done) return 0;
    mask_walk(&w, st->node, 0);
    if (st->cs->node[st->node].term && v->eos >= 0 && !allow[v->eos] && w.n < maxlist) { // (a complete string that also continues)
        allow[v->eos] = 1;
        list[w.n++] = v->eos;
    }
    return w.n;
}
