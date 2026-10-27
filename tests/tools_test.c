/* Unit tests of runq serve's tool calls (src/tools.c): the reply parser (tscan_*, tool_parse_calls: content before
 * a call, several calls, multi-line and nested JSON parameters, typed arguments, unicode, malformed and cut blocks,
 * text after the calls, parallel_tool_calls false), each case fed whole and in pieces of 1, 2, 3, 5 and 7 bytes (the
 * scanner sees the reply as it is generated); Python's json.dumps (py_json, py_float: what transformers' tojson writes
 * in the tools' system block). No model needed. Prints PASS / FAIL lines (as tests/api_test.c), exit status 1 if any
 * failed; run by tests/check.py (group server). Build: make build/tools_test. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/serve.h"

static int n_fail, n_pass;

static void report(int ok, const char *name, const char *detail) {
    printf("%s %s%s%s\n", ok ? "PASS" : "FAIL", name, detail && detail[0] ? "  " : "", detail ? detail : "");
    if (ok) n_pass++;
    else n_fail++;
}

// the tools of the cases (as a client sends them)
static const char *TOOLS =
    "[{\"type\": \"function\", \"function\": {\"name\": \"read\", \"description\": \"Read a file\", \"parameters\": {\"type\": \"object\", "
    "\"properties\": {\"path\": {\"type\": \"string\"}, \"offset\": {\"type\": \"integer\"}, \"limit\": {\"type\": \"number\"}}, "
    "\"required\": [\"path\"]}}},"
    " {\"type\": \"function\", \"function\": {\"name\": \"bash\", \"parameters\": {\"type\": \"object\", \"properties\": {"
    "\"command\": {\"type\": \"string\"}, \"timeout\": {\"type\": [\"integer\", \"null\"]}, \"env\": {\"type\": \"object\"}, "
    "\"flags\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}}, \"dry\": {\"type\": \"boolean\"}, "
    "\"ratio\": {\"anyOf\": [{\"type\": \"number\"}, {\"type\": \"null\"}]}, \"any\": {\"description\": \"no type\"}}}}}]";

#define CALL(fn, body) "<tool_call>\n<function=" fn ">\n" body "</function>\n</tool_call>"
#define PARAM(k, v) "<parameter=" k ">\n" v "\n</parameter>\n"

typedef struct {
    const char *name, *reply;
    int single;           // parallel_tool_calls: false
    const char *content;  // the content after the final push
    const char *calls;    // the calls, "name arguments" per line
    int cut;              // the scanner stopped the generation before the end of the reply
} Case;

static const Case CASES[] = {
    { "plain", "Hello there. Nothing to call.", 0, "Hello there. Nothing to call.", "", 0 },
    { "trailing_newlines_kept", "Hi!\n\n", 0, "Hi!\n\n", "", 0 },
    { "one_call", CALL("read", PARAM("path", "notes/a.txt") PARAM("offset", "3")), 0, "",
      "read {\"path\": \"notes/a.txt\", \"offset\": 3}\n", 0 },
    { "content_then_call", "Let me look at it.\n\n" CALL("read", PARAM("path", "a.txt")), 0, "Let me look at it.",
      "read {\"path\": \"a.txt\"}\n", 0 },
    { "two_calls", CALL("read", PARAM("path", "a")) "\n" CALL("bash", PARAM("command", "ls")) "\n", 0, "",
      "read {\"path\": \"a\"}\nbash {\"command\": \"ls\"}\n", 0 },
    { "two_functions_one_block", "<tool_call>\n<function=read>\n" PARAM("path", "a") "</function>\n<function=read>\n"
      PARAM("path", "b") "</function>\n</tool_call>", 0, "", "read {\"path\": \"a\"}\nread {\"path\": \"b\"}\n", 0 },
    { "multiline_param", CALL("bash", PARAM("command", "cat <<EOF > x.c\nint main(void) {\n\treturn 0;\n}\nEOF\n")), 0, "",
      "bash {\"command\": \"cat <<EOF > x.c\\nint main(void) {\\n\\treturn 0;\\n}\\nEOF\\n\"}\n", 0 },
    { "typed_args", CALL("bash", PARAM("command", "  null ") PARAM("timeout", "null") PARAM("env", "{\"A\": \"1\", \"B\": [1, 2.50, true], \"A\": 2}")
                                 PARAM("flags", "[\"-x\",\"y\"]") PARAM("dry", "True") PARAM("ratio", "1.5e3") PARAM("any", "42")),
      0, "", "bash {\"command\": \"  null \", \"timeout\": null, \"env\": {\"A\": 2, \"B\": [1, 2.5, true]}, \"flags\": [\"-x\", \"y\"], "
             "\"dry\": true, \"ratio\": 1.5e3, \"any\": \"42\"}\n", 0 },
    { "typed_fallbacks", CALL("bash", PARAM("timeout", "3.0") PARAM("env", "{not json") PARAM("dry", "yes") PARAM("ratio", "null")
                                      PARAM("flags", "-x")) CALL("read", PARAM("offset", "-12") PARAM("limit", ".5")),
      0, "", "bash {\"timeout\": \"3.0\", \"env\": \"{not json\", \"dry\": \"yes\", \"ratio\": null, \"flags\": \"-x\"}\n"
             "read {\"offset\": -12, \"limit\": \".5\"}\n", 0 },
    { "unicode", "Je regarde \xc3\xa7" "a \xf0\x9f\x90\x9a\n" CALL("read", PARAM("path", "caf\xc3\xa9/\xe2\x80\xa8 \xf0\x9f\x90\x9a.txt")), 0,
      "Je regarde \xc3\xa7" "a \xf0\x9f\x90\x9a", "read {\"path\": \"caf\xc3\xa9/\xe2\x80\xa8 \xf0\x9f\x90\x9a.txt\"}\n", 0 },
    { "controls_escaped", CALL("bash", PARAM("command", "a\"b\\c\x01\x7f")), 0, "",
      "bash {\"command\": \"a\\\"b\\\\c\\u0001\x7f\"}\n", 0 },
    { "no_newlines", "<tool_call><function=read><parameter=path>a.txt</parameter><parameter=offset>7</parameter></function></tool_call>", 0,
      "", "read {\"path\": \"a.txt\", \"offset\": 7}\n", 0 },
    { "missing_param_end", "<tool_call>\n<function=read>\n<parameter=path>\na.txt\n<parameter=offset>\n5\n</function>\n</tool_call>", 0,
      "", "read {\"path\": \"a.txt\", \"offset\": 5}\n", 0 },
    { "repeated_param_first", CALL("read", PARAM("path", "a") PARAM("path", "b")), 0, "", "read {\"path\": \"a\"}\n", 0 },
    { "empty_args", CALL("bash", ""), 0, "", "bash {}\n", 0 },
    { "unknown_tool_strings", CALL("nope", PARAM("x", "1") PARAM("y", "{\"a\": 1}")), 0, "", "nope {\"x\": \"1\", \"y\": \"{\\\"a\\\": 1}\"}\n", 0 },
    { "text_after_call_stops", CALL("read", PARAM("path", "a")) "\n\nDone! I read it.", 0, "", "read {\"path\": \"a\"}\n", 1 },
    { "single_call", CALL("read", PARAM("path", "a")) "\n" CALL("read", PARAM("path", "b")), 1, "", "read {\"path\": \"a\"}\n", 1 },
    { "unterminated_block_ok", "Sure.\n<tool_call>\n<function=read>\n" PARAM("path", "a") "</function>\n", 0, "Sure.",
      "read {\"path\": \"a\"}\n", 0 },
    { "unterminated_block_cut", "Sure.\n<tool_call>\n<function=read>\n<parameter=path>\nnotes/", 0,
      "Sure.\n<tool_call>\n<function=read>\n<parameter=path>\nnotes/", "", 0 },
    { "malformed_no_function", "Use <tool_call>\nhello\n</tool_call> tags.", 0, "Use <tool_call>\nhello\n</tool_call> tags.", "", 0 },
    { "malformed_text_inside", "<tool_call>\n<function=read>\noops <parameter=path>\na\n</parameter>\n</function>\n</tool_call>\nok", 0,
      "<tool_call>\n<function=read>\noops <parameter=path>\na\n</parameter>\n</function>\n</tool_call>\nok", "", 0 },
    { "malformed_then_call", "<tool_call>x</tool_call> then " CALL("read", PARAM("path", "a")), 0, "<tool_call>x</tool_call> then",
      "read {\"path\": \"a\"}\n", 0 },
    { "malformed_after_call_stops", CALL("read", PARAM("path", "a")) "\n<tool_call>junk</tool_call>", 0, "", "read {\"path\": \"a\"}\n", 1 },
    { "partial_tag_text", "a <tool_ b <tool_call", 0, "a <tool_ b <tool_call", "", 0 },
    { "tag_name_spaces", "<tool_call>\n<function= read >\n<parameter= path >\na\n</parameter>\n</function>\n</tool_call>", 0, "",
      "read {\"path\": \"a\"}\n", 0 },
};

static int run_case(const ToolSet *ts, const Case *c, int piece, char *detail, size_t dl) {
    // feed the reply in pieces of `piece` bytes (0: whole), then the final push; compare
    ToolScan sc;
    tscan_init(&sc, ts, c->single, 12345);
    JBuf content = { 0 };
    StrList calls = { 0 };
    size_t n = strlen(c->reply), step = piece ? (size_t)piece : n ? n : 1;
    int cut = 0;
    for (size_t i = 0; i < n && !cut; i += step)
        cut = tscan_push(&sc, c->reply + i, i + step < n ? step : n - i, 0, &content, &calls);
    if (!cut) tscan_push(&sc, NULL, 0, 1, &content, &calls);
    JBuf got = { 0 };
    int ok = 1;
    for (int i = 0; i < calls.n; i++) { // '"id": "call_<16 hex>", "type": "function", "function": {"name": N, "arguments": A}'
        char wrap[8192];
        snprintf(wrap, sizeof(wrap), "{%s}", calls.v[i]);
        char err[128];
        JsDoc *d = json_parse(wrap, strlen(wrap), err, sizeof(err));
        const Js *r = d ? json_root(d) : NULL, *id = json_get(r, "id"), *f = json_get(r, "function");
        const Js *nm = json_get(f, "name"), *ar = json_get(f, "arguments");
        if (!d || !id || id->t != JS_STR || id->n != 21 || strncmp(id->s, "call_", 5) || !json_str_eq(json_get(r, "type"), "function") ||
            !nm || nm->t != JS_STR || !ar || ar->t != JS_STR) {
            ok = 0;
            jb_cstr(&got, "(bad item) ");
        } else {
            jb_cstr(&got, nm->s);
            jb_cstr(&got, " ");
            jb_raw(&got, ar->s, ar->n);
            jb_cstr(&got, "\n");
            JsDoc *ad = json_parse(ar->s, ar->n, err, sizeof(err)); // (the arguments are JSON)
            if (!ad || json_root(ad)->t != JS_OBJ) {
                ok = 0;
                jb_cstr(&got, "(arguments not a JSON object) ");
            }
            json_free(ad);
            for (int k = 0; k < i; k++)
                if (strstr(calls.v[k], id->s)) ok = 0; // (ids unique)
        }
        json_free(d);
    }
    const char *gc = content.p ? content.p : "", *gl = got.p ? got.p : "";
    ok &= !strcmp(gc, c->content) && !strcmp(gl, c->calls) && cut == c->cut;
    if (!ok) snprintf(detail, dl, "pieces of %d: content \"%s\" calls \"%s\" cut %d", piece, gc, gl, cut);
    jb_free(&got);
    jb_free(&content);
    strlist_free(&calls);
    tscan_free(&sc);
    return ok;
}

static void test_py(void) {
    // py_float == Python's repr(float); py_json == json.dumps(json.loads(text), ensure_ascii=False)
    static const struct { double x; const char *want; } F[] = {
        { 1.0, "1.0" }, { 1e16, "1e+16" }, { 1e-5, "1e-05" }, { 0.0001, "0.0001" }, { 123456789012345678.0, "1.2345678901234568e+17" },
        { 0.1, "0.1" }, { 1.0 / 3, "0.3333333333333333" }, { 2.5e-308, "2.5e-308" }, { 5e-324, "5e-324" },
        { 1.7976931348623157e308, "1.7976931348623157e+308" }, { -0.0, "-0.0" }, { 100.0, "100.0" }, { 1e15, "1000000000000000.0" },
        { 12345.678, "12345.678" }, { 1e22, "1e+22" }, { 0.30000000000000004, "0.30000000000000004" }, { 1.5e-7, "1.5e-07" },
        { 1152921504606846976.0, "1.152921504606847e+18" },
    };
    int bad = 0;
    char det[256] = "";
    for (size_t i = 0; i < sizeof(F) / sizeof(F[0]); i++) {
        JBuf b = { 0 };
        py_float(&b, F[i].x);
        if (strcmp(b.p, F[i].want)) {
            bad++;
            snprintf(det, sizeof(det), "%.17g -> %s, want %s", F[i].x, b.p, F[i].want);
        }
        jb_free(&b);
    }
    report(!bad, "tools/py_float", bad ? det : "18 values == Python's repr");
    const char *in = "{\"a\": 1, \"b\": [1.50, -0, 1E5, true, null, -12, 1e400], \"a\": \"x\\u0001\\u007f\\\"\\\\\\u00e9\\n\\u2028\", \"c\": {}}";
    const char *want = "{\"a\": \"x\\u0001\x7f\\\"\\\\\xc3\xa9\\n\xe2\x80\xa8\", \"b\": [1.5, 0, 100000.0, true, null, -12, Infinity], \"c\": {}}";
    char err[128];
    JsDoc *d = json_parse(in, strlen(in), err, sizeof(err));
    JBuf b = { 0 };
    if (d) py_json(&b, json_root(d));
    report(d && !strcmp(b.p, want), "tools/py_json", d ? b.p : err);
    jb_free(&b);
    json_free(d);
}

int main(void) {
    char err[256], det[16384];
    JsDoc *d = json_parse(TOOLS, strlen(TOOLS), err, sizeof(err));
    ToolSet *ts = d ? tools_compile(json_root(d), err, sizeof(err)) : NULL;
    report(ts && tools_count(ts) == 2, "tools/compile", ts ? "" : err);
    if (!ts) return 1;
    static const int pieces[] = { 0, 1, 2, 3, 5, 7 };
    for (size_t i = 0; i < sizeof(CASES) / sizeof(CASES[0]); i++) {
        int ok = 1;
        det[0] = 0;
        for (size_t p = 0; p < sizeof(pieces) / sizeof(pieces[0]) && ok; p++) ok = run_case(ts, &CASES[i], pieces[p], det, sizeof(det));
        char name[128];
        snprintf(name, sizeof(name), "tools/parse_%s", CASES[i].name);
        report(ok, name, ok ? "whole and in pieces of 1, 2, 3, 5, 7 bytes" : det);
    }
    // errors of tools_compile
    static const struct { const char *tools, *msg; } E[] = {
        { "[{\"type\": \"custom\", \"custom\": {\"name\": \"x\"}}]", "only function tools" },
        { "[{\"type\": \"function\", \"function\": {\"description\": \"no name\"}}]", "function.name" },
        { "[{\"type\": \"function\", \"function\": {\"name\": \"f\", \"parameters\": 3}}]", "JSON schema object" },
        { "{\"type\": \"function\"}", "must be an array" },
    };
    int bad = 0;
    for (size_t i = 0; i < sizeof(E) / sizeof(E[0]); i++) {
        JsDoc *x = json_parse(E[i].tools, strlen(E[i].tools), err, sizeof(err));
        ToolSet *t = x ? tools_compile(json_root(x), err, sizeof(err)) : NULL;
        if (t || !strstr(err, E[i].msg)) bad++;
        tools_free(t);
        json_free(x);
    }
    report(!bad, "tools/compile_errors", "custom tools, no name, bad parameters, not an array");
    test_py();
    tools_free(ts);
    json_free(d);
    printf("tools tests: %d passed, %d failed\n", n_pass, n_fail);
    return n_fail ? 1 : 0;
}
