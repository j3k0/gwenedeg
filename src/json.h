/* A small JSON parser (DOM, arena-allocated) and writer, for runq serve (src/serve.c).
 *
 * Parser: RFC 8259 JSON (no comments, NaN / Infinity, trailing commas or single quotes), UTF-8 input;
 * \uXXXX escapes incl. surrogate pairs (a lone surrogate becomes U+FFFD); raw control characters in
 * strings and invalid UTF-8 are errors; nesting depth limited (JSON_MAX_DEPTH). Strings are stored
 * decoded, NUL-terminated, with their byte length (they may contain NUL). Object members keep their
 * order; json_get returns the last member of a name (as Python's json does with duplicates).
 * Writer: a growable buffer; jb_str escapes a byte string as a JSON string and replaces invalid UTF-8
 * (bytes that are not part of a well-formed character, surrogates, overlongs) with U+FFFD, so the
 * output is always valid JSON in valid UTF-8 whatever the model generated.
 */
#ifndef RUNQ_JSON_H
#define RUNQ_JSON_H

#include <stddef.h>

#define JSON_MAX_DEPTH 64

typedef enum { JS_NULL, JS_FALSE, JS_TRUE, JS_NUM, JS_STR, JS_ARR, JS_OBJ } JsType;

typedef struct Js Js;
struct Js {
    JsType t;
    int n;         // string: byte length; array: items; object: members
    char *s;       // string: the decoded bytes (NUL-terminated); number: its text as written
    double num;    // number: its value (strtod)
    Js **v;        // array items / object member values
    char **k;      // object member names (decoded, NUL-terminated)
    int *kn;       // their byte lengths
};

typedef struct JsDoc JsDoc; // a parsed document: its arena

// Parses text[0..len). NULL on error, with a message ("... at byte N") in err
JsDoc *json_parse(const char *text, size_t len, char *err, size_t errlen);
const Js *json_root(const JsDoc *d);
void json_free(JsDoc *d);

const Js *json_get(const Js *obj, const char *key);       // member of an object (NULL if none / not an object)
int json_is_int(const Js *v, long long *out);              // a number with an integer value in +-2^53
int json_str_eq(const Js *v, const char *s);              // a string equal to s

// ------------------------------------------------------------------------------------------- writer
typedef struct {
    char *p;
    size_t n, cap;
    int oom; // an allocation failed (contents incomplete)
} JBuf;

void jb_free(JBuf *b);
void jb_reset(JBuf *b);
void jb_raw(JBuf *b, const char *s, size_t n);
void jb_cstr(JBuf *b, const char *s);                     // raw text (no quoting)
void jb_printf(JBuf *b, const char *fmt, ...);
void jb_str(JBuf *b, const char *s, size_t n);            // "..." escaped, invalid UTF-8 -> U+FFFD
void jb_strz(JBuf *b, const char *s);                     // jb_str of a NUL-terminated string
void jb_value(JBuf *b, const Js *v, const char *colon, const char *comma); // canonical serialization
size_t utf8_valid_prefix(const char *s, size_t n);         // length of the longest valid UTF-8 prefix

#endif
