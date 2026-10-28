/* The runq command line, on the library API (runq.h): chat (multi-turn, the conversation kept in
 * the KV cache), generate, and the test hooks (-m tokenize / logits / dequant, RUNQ_CHAT_DUMP).
 * Derived from qwen3.c's runq.c main(); all printing and exiting happens here. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
#endif
#include "runq.h"
#include "json.h" // (-m render: the messages as JSON)

static int max_new_tokens = 0; // -n: stop after this many generated tokens, 0 = unlimited
#define SPEC_K_DEFAULT 3       // -G with -D (the best fixed k for Qwen3-1.7B + 0.6B: docs/qwen38-speculative.md)

// speculative decoding (-D draft.gguf, -G k, --tau; --lookup N, --lookup-min L): the draft model, its context and
// sequence (m NULL: no draft model), the options (o.k 0 and o.lookup 0: plain decoding)
static struct { RqModel *m; RqCtx *c; RqSeq *s; RqSpecOpts o; } draft;
#define SPECULATIVE (draft.m || draft.o.lookup > 0)

static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
}

static void check(int e, RqCtx *c) {
    // a library error in a request path: its message, then exit as the CLI always did
    if (e == RQ_OK) return;
    const char *d = c ? rq_ctx_error(c) : "";
    die(e == RQ_ERR_NOMEM ? "malloc failed!" : d[0] ? d : rq_strerror(e));
}

static void push(RqTokens *t, const int *x, int n) { check(rq_tokens_push(t, x, n), NULL); }

static void print_token(const RqModel *m, int token) {
    printf("%s", rq_token_bytes(m, token, NULL)); // (raw bytes; a character split over tokens comes out whole)
}

static void report_timing(int n_prompt, double pf, int n_gen, double dc) {
    fprintf(stderr, "\n[timing] prompt: %d tok in %.3f s (%.1f tok/s) | generated: %d tok in %.3f s (%.1f tok/s)\n",
            n_prompt, pf, n_prompt / (pf > 0 ? pf : 1e-9), n_gen, dc, n_gen / (dc > 0 ? dc : 1e-9));
}

static void report_spec(const RqSpecStats *st) {
    // the speculative loop's counters (after report_timing); with the prompt lookup, its share after the totals
    char tau[32] = "", how[96] = "", lk[160] = "";
    if (draft.o.tau > 0) snprintf(tau, sizeof(tau), ", tau %g", draft.o.tau);
    if (draft.m) snprintf(how, sizeof(how), "k %d%s", draft.o.k, tau);
    if (draft.o.lookup > 0) {
        size_t l = strlen(how);
        snprintf(how + l, sizeof(how) - l, "%slookup %d (match >= %d)", l ? ", " : "", draft.o.lookup,
                 draft.o.lookup_min > 0 ? draft.o.lookup_min : RQ_LOOKUP_MIN_DEFAULT);
        snprintf(lk, sizeof(lk), " | lookup: %d passes, %d drafted, %d accepted (%.1f%%)", st->lookup_passes,
                 st->lookup_drafted, st->lookup_accepted, 100.0 * st->lookup_accepted / (st->lookup_drafted > 0 ? st->lookup_drafted : 1));
    }
    fprintf(stderr, "[timing] speculative: %s: %d passes, %d drafted, %d accepted (%.1f%%), %.2f tok/pass | draft %.3f s "
            "(%d steps), verify %.3f s%s\n", how, st->passes, st->drafted, st->accepted,
            100.0 * st->accepted / (st->drafted > 0 ? st->drafted : 1), (double)st->gen.n_gen / (st->passes > 0 ? st->passes : 1),
            st->t_draft, st->draft_steps, st->t_verify, lk);
}

// -m generate with a draft: the output of generate()'s loop (the prompt, then at most -n - 1 generated tokens, those
// that fit in the context), through rq_spec_run (no RNG draws for the prompt, as there)
typedef struct { const RqModel *m; int pos, cap, left; } GenOut;

static int on_gen_token(void *user, int token) {
    GenOut *g = user;
    if (g->pos >= g->cap || g->left == 0) return 1; // (generate() stops before printing it)
    print_token(g->m, token);
    fflush(stdout);
    g->pos++;
    if (g->left > 0) g->left--;
    return 0;
}

// ----------------------------------------------------------------------------
// generation loop (-m generate): the prompt printed, then the generated text (upstream's loop)

static void generate(RqModel *m, RqCtx *c, RqSeq *s, const RqSampling *cfg, char *prompt) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    RqTokens pt = { 0 };
    check(rq_tokenize(m, prompt, -1, &pt), NULL);
    int num_prompt_tokens = pt.n, *prompt_tokens = pt.v;
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "Please provide a prompt using -i <string> on the command line.\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    int seq_len = rq_seq_capacity(s);
    int next = 0;    // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    int n_gen = 0;
    double t0 = rq_now(), t_prefill = t0;
    float *pre_logits = NULL; // logits of the last prompt token, from the batched prefill

    if (SPECULATIVE && num_prompt_tokens <= seq_len) {
        // speculative: the prompt (both models), then the loop (rq_spec_run) prints what generate's loop would
        float *logits;
        check(rq_prefill(c, s, prompt_tokens, num_prompt_tokens, RQ_LOGITS_LAST, NULL, &logits), c);
        if (draft.m) {
            int nd = num_prompt_tokens < rq_seq_capacity(draft.s) ? num_prompt_tokens : rq_seq_capacity(draft.s);
            check(rq_prefill(draft.c, draft.s, prompt_tokens, nd, RQ_LOGITS_NONE, NULL, NULL), draft.c);
        }
        for (int i = 0; i < num_prompt_tokens; i++)
            print_token(m, prompt_tokens[i]);
        fflush(stdout);
        t_prefill = rq_now();
        GenOut go = { m, num_prompt_tokens, seq_len, max_new_tokens > 0 ? max_new_tokens - 1 : -1 };
        RqCallbacks cb = { on_gen_token, NULL, &go };
        RqGen g;
        rq_gen_start(&g, s, cfg, max_new_tokens > 1 ? max_new_tokens - 1 : 0, &cb, 1);
        RqSpecStats st;
        memset(&st, 0, sizeof(st));
        check(rq_spec_run(&g, c, logits, draft.c, draft.s, &draft.o, &st), c);
        printf("\n");
        report_timing(num_prompt_tokens, t_prefill - t0, go.pos - num_prompt_tokens, rq_now() - t_prefill);
        report_spec(&st);
        rq_tokens_free(&pt);
        return;
    }

    if (rq_ctx_max_batch(c) > 1 && num_prompt_tokens <= seq_len) {
        // batched prefill of the whole prompt, then resume the loop at its last token
        check(rq_prefill(c, s, prompt_tokens, num_prompt_tokens, RQ_LOGITS_LAST, NULL, &pre_logits), c);
        for (int i = 0; i < num_prompt_tokens - 1; i++)
            print_token(m, prompt_tokens[i]);
        fflush(stdout);
        pos = num_prompt_tokens - 1;
        token = prompt_tokens[pos];
    }

    while (pos < seq_len) {
        // forward the transformer to get logits for the next token (at position pos = the
        // sequence's length; only needed from the last prompt token on)
        float *logits = pre_logits;
        if (!logits)
            check(rq_decode(c, s, token, pos >= num_prompt_tokens - 1 ? &logits : NULL), c);
        pre_logits = NULL;

        // advance the state state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            if (pos == num_prompt_tokens - 1) t_prefill = rq_now();
            else n_gen++;
            next = rq_sample(c, s, logits, cfg);
            check(next < 0 ? next : RQ_OK, c);
        }
        pos++;
        if (max_new_tokens > 0 && n_gen >= max_new_tokens) break;

        // print the token as string, decode it with the Tokenizer object
        print_token(m, token);
        fflush(stdout);
        token = next;

        // data-dependent terminating condition: the BOS token delimits sequences
        if (pos >= num_prompt_tokens && rq_is_end_token(m, next))
            break;
    }
    printf("\n");
    double t_end = rq_now();
    report_timing(num_prompt_tokens, t_prefill - t0, n_gen, t_end - t_prefill);
    rq_tokens_free(&pt);
}

// ----------------------------------------------------------------------------
// chat loop (multi-turn, the conversation stays in the KV cache; rendering: chat.c)
//
// A new turn k is the canonical sequence  [system] + sum over kept turns i < k (P_i + H_i + E) + P_k
// (pieces: chat.c). Of that sequence, the longest prefix that is already in the cache is kept and the
// rest is prefilled from there: normally just E + P_k (the sampled <|im_end|> that ended the reply is
// the first token of the next prefill, so it is in the cache before the next turn), with -r 1 from
// the start of the previous answer (its reasoning is dropped and the answer re-prefilled).
// Context full: if the canonical sequence plus a reserve for the reply (-n, else seq_len / 4, at most
// seq_len / 2) does not fit in -c, the oldest turns are dropped until it is at most half of the
// rest (seq_len - reserve) / 2, keeping the system prompt (cached prefix) and the newest turns, and
// the rest is re-prefilled. A reply reaching the end of the context is cut there.
// RNG: one draw per prefilled token but the last, then one per sampled token (seeded output does
// not depend on -b; for a single turn the same draws as before batched prefill).

static char *read_line(FILE *f) {
    // one line of any length without its "\n" (or "\r\n"), malloc'd; NULL at EOF with nothing read
    size_t cap = 256, n = 0;
    char *s = malloc(cap);
    if (!s) die("malloc failed!");
    int c;
    while ((c = fgetc(f)) != EOF && c != '\n') {
        if (n + 1 >= cap && !(s = realloc(s, cap *= 2))) die("malloc failed!");
        s[n++] = (char)c;
    }
    if (c == EOF && n == 0) { free(s); return NULL; }
    if (n > 0 && s[n - 1] == '\r') n--;
    s[n] = 0;
    return s;
}

// test hook: RUNQ_CHAT_DUMP=<prefix> writes <prefix>.ids (the token ids in the KV cache at the end,
// space-separated) and <prefix>.logits (int32 n, int32 vocab, int32 pos[n], float32 logits[n][vocab]:
// every logits vector chat sampled from, pos = the position of the token that produced it; those of
// positions overwritten later, by a re-prefill, are dropped). tests/check.py compares them with
// -m logits on the decoded conversation
static struct { const char *prefix; int n, cap, vocab; int *pos; float *logits; } cdump;

static void cdump_record(int pos, const float *logits) {
    if (!cdump.prefix) return;
    if (cdump.n == cdump.cap) {
        cdump.cap = cdump.cap ? 2 * cdump.cap : 64;
        cdump.pos = realloc(cdump.pos, cdump.cap * sizeof(int));
        cdump.logits = realloc(cdump.logits, (size_t)cdump.cap * cdump.vocab * sizeof(float));
        if (!cdump.pos || !cdump.logits) die("malloc failed!");
    }
    cdump.pos[cdump.n] = pos;
    memcpy(cdump.logits + (size_t)cdump.n * cdump.vocab, logits, cdump.vocab * sizeof(float));
    cdump.n++;
}

static void cdump_truncate(int pos) {
    // positions >= pos are about to be overwritten
    while (cdump.n > 0 && cdump.pos[cdump.n - 1] >= pos)
        cdump.n--;
}

static void cdump_write(const int *kv, int n) {
    if (!cdump.prefix) return;
    size_t len = strlen(cdump.prefix) + 8;
    char *path = malloc(len);
    if (!path) die("malloc failed!");
    snprintf(path, len, "%s.ids", cdump.prefix);
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Couldn't open %s for writing\n", path); exit(EXIT_FAILURE); }
    for (int i = 0; i < n; i++)
        fprintf(f, i ? " %d" : "%d", kv[i]);
    fprintf(f, "\n");
    fclose(f);
    snprintf(path, len, "%s.logits", cdump.prefix);
    f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Couldn't open %s for writing\n", path); exit(EXIT_FAILURE); }
    int32_t header[2] = { cdump.n, cdump.vocab };
    int ok = fwrite(header, sizeof(int32_t), 2, f) == 2 && fwrite(cdump.pos, sizeof(int32_t), cdump.n, f) == (size_t)cdump.n
          && fwrite(cdump.logits, sizeof(float), (size_t)cdump.n * cdump.vocab, f) == (size_t)cdump.n * cdump.vocab;
    if (fclose(f) != 0 || !ok) { fprintf(stderr, "Error writing %s\n", path); exit(EXIT_FAILURE); }
    free(path);
}

typedef struct {
    RqTokens prompt; // P_i: user message + assistant header (+ empty think block with -r 0)
    RqTokens phist;  // P_i once answered (rq_chat_user_history: Thinking-2507 without its "<think>\n")
    RqTokens reply;  // R_i: the generated tokens, without the <|im_end|> that ended them
    RqTokens hist;   // H_i: the reply as part of the history
} ChatTurn;

typedef struct { const RqModel *m; RqTokens *reply; } ReplyOut;

static int on_reply_token(void *user, int token) {
    ReplyOut *r = user;
    print_token(r->m, token);
    fflush(stdout);
    push(r->reply, &token, 1);
    return 0;
}

static void on_reply_logits(void *user, int pos, const float *logits) {
    (void)user;
    cdump_record(pos, logits);
}

static void render_conversation(RqTokens *out, const RqTokens *sys, const ChatTurn *turns, int j0, int k, const RqTokens *close) {
    // [system] + turns j0..k-1 (prompt, history reply, <|im_end|>\n) + the prompt of turn k
    out->n = 0;
    push(out, sys->v, sys->n);
    for (int i = j0; i < k; i++) {
        push(out, turns[i].phist.v, turns[i].phist.n);
        push(out, turns[i].hist.v, turns[i].hist.n);
        push(out, close->v, close->n);
    }
    push(out, turns[k].prompt.v, turns[k].prompt.n);
}

static void chat(RqModel *m, RqCtx *c, RqSeq *s, const RqSampling *cfg, char *cli_user_prompt, char *system_prompt,
                 int enable_thinking) {
    int seq_len = rq_seq_capacity(s);
    RqTokens sys = { 0 }, close = { 0 }, want = { 0 };
    ChatTurn *turns = NULL;
    int n_turns = 0, j0 = 0; // turns[j0..] are in the conversation, older ones were dropped (context full)
    RqModelInfo info;
    rq_model_info(m, &info);

    check(rq_chat_system(m, system_prompt, enable_thinking, &sys), NULL); // (none: Qwen3.8's effort instruction, if any)
    check(rq_chat_end(m, &close), NULL);
    int reserve = max_new_tokens > 0 ? max_new_tokens : seq_len / 4;
    if (reserve > seq_len / 2) reserve = seq_len / 2;
    if (reserve < 1) reserve = 1;
    cdump.prefix = getenv("RUNQ_CHAT_DUMP");
    cdump.vocab = info.vocab_size;
#ifndef _WIN32
    int echo = !isatty(fileno(stdin)); // piped stdin: echo each user line, for a readable transcript
#else
    int echo = 0;
#endif

    for (int asked = 0;; asked++) {
        // the user message: -i (one turn only), else one line of stdin per turn (empty line or EOF ends)
        char *user;
        if (cli_user_prompt) {
            if (asked) break;
            user = strdup(cli_user_prompt);
            if (!user) die("malloc failed!");
        } else {
            printf("\n> ");
            fflush(stdout);
            user = read_line(stdin);
            if (!user || !user[0]) { free(user); break; }
            if (echo) printf("%s\n", user);
            fflush(stdout);
        }
        turns = realloc(turns, (size_t)(n_turns + 1) * sizeof(ChatTurn));
        if (!turns) die("malloc failed!");
        int k = n_turns;
        ChatTurn *tn = &turns[k];
        memset(tn, 0, sizeof(*tn));
        check(rq_chat_user(m, user, enable_thinking, &tn->prompt), NULL);
        check(rq_chat_user_history(m, user, enable_thinking, &tn->phist), NULL);
        free(user);

        // the canonical conversation; context full: drop the oldest turns (see above)
        int j = j0;
        render_conversation(&want, &sys, turns, j, k, &close);
        if (want.n + reserve > seq_len)
            while (j < k && want.n > (seq_len - reserve) / 2)
                render_conversation(&want, &sys, turns, ++j, k, &close);
        if (want.n > seq_len) {
            fprintf(stderr, "(message too long: %d tokens with the template%s, context %d (-c); skipped)\n", want.n,
                    sys.n ? " and the system prompt" : "", seq_len);
            rq_tokens_free(&tn->prompt);
            rq_tokens_free(&tn->phist);
            continue;
        }
        n_turns++;
        if (j > j0)
            fprintf(stderr, "[context] full (-c %d): dropped the %d oldest turn(s), kept %s%d earlier turn(s)\n", seq_len,
                    j - j0, sys.n ? "the system prompt and " : "", k - j);
        j0 = j;

        // prefill what is not in the cache yet (the longest common prefix stays), then the reply:
        // sample, print, feed back; ends at <|im_end|> (sampled, not fed yet: the next turn's prefill
        // starts with it), at -n tokens, or at the end of the context
        // (Qwen3.5: the recurrent state cannot be rewound: the cache goes back to a checkpoint <= L, e.g. this
        // conversation's last assistant header, where a reasoning reply's history form diverges, and the rest is
        // prefilled again; the next turn's checkpoint: this prompt's assistant header)
        int L = rq_seq_common_prefix(s, want.v, want.n);
        check(rq_seq_truncate(s, L), c);
        L = rq_seq_len(s);
        cdump_truncate(L);
        int ckp = rq_chat_checkpoint_pos(m, want.v, want.n);
        if (ckp >= L) check(rq_seq_checkpoint_at(s, ckp), c);
        ReplyOut ro = { m, &tn->reply };
        RqCallbacks cb = { on_reply_token, on_reply_logits, &ro };
        if (draft.m) { // (the draft follows the same conversation: its common prefix kept, the same checkpoint)
            check(rq_seq_truncate(draft.s, rq_seq_common_prefix(draft.s, want.v, want.n)), draft.c);
            if (ckp >= rq_seq_len(draft.s) && ckp <= rq_seq_capacity(draft.s)) check(rq_seq_checkpoint_at(draft.s, ckp), draft.c);
        }
        if ((info.chat == RQ_CHAT_THINKING || info.chat == RQ_CHAT_QWEN35 || info.chat == RQ_CHAT_QWEN38) && enable_thinking)
            printf("<think>\n"); // (Thinking-2507, Qwen3.5 / 3.8: the prompt opened the block, the reply starts inside it)
        RqSpecStats sst;
        RqGenStats st;
        if (SPECULATIVE) {
            check(rq_generate_spec(c, s, draft.c, draft.s, want.v + L, want.n - L, cfg, max_new_tokens, &draft.o, &cb, &sst), c);
            st = sst.gen;
        } else {
            check(rq_generate(c, s, want.v + L, want.n - L, cfg, max_new_tokens, &cb, &st), c);
        }
        if (st.stop == RQ_STOP_CONTEXT)
            fprintf(stderr, "\n(context window full (-c %d): reply cut)", seq_len);
        printf("\n");
        report_timing(st.n_prompt, st.t_prefill, st.n_gen, st.t_gen);
        if (SPECULATIVE) report_spec(&sst);
        check(rq_chat_reply_history(m, tn->reply.v, tn->reply.n, enable_thinking, &tn->hist), NULL);
    }
    cdump_write(rq_seq_tokens(s), rq_seq_len(s));
    for (int i = 0; i < n_turns; i++) {
        rq_tokens_free(&turns[i].prompt);
        rq_tokens_free(&turns[i].phist);
        rq_tokens_free(&turns[i].reply);
        rq_tokens_free(&turns[i].hist);
    }
    free(turns);
    rq_tokens_free(&sys);
    rq_tokens_free(&close);
    rq_tokens_free(&want);
    free(cdump.pos);
    free(cdump.logits);
}

// ----------------------------------------------------------------------------
// test hooks: -m tokenize, -m logits, -m dequant (see tests/)

static void tokenize(const RqModel *m, char *text, double t_load) {
    // encode the raw text (no chat template), print the token ids space-separated on stdout;
    // timings on stderr (encode only, and the tokenizer load before it)
    RqTokens t = { 0 };
    double t0 = rq_now();
    check(rq_tokenize(m, text, -1, &t), NULL);
    double t_enc = rq_now() - t0;
    for (int i = 0; i < t.n; i++)
        printf(i ? " %d" : "%d", t.v[i]);
    printf("\n");
    fprintf(stderr, "[tokenize] %zu bytes -> %d tokens, encode %.3f ms, tokenizer load %.1f ms\n",
            strlen(text), t.n, t_enc * 1e3, t_load * 1e3);
    rq_tokens_free(&t);
}

static void render_one(const RqModel *m, const Js *a, int thinking) {
    // one conversation (a JSON array of messages) rendered: its ids on a line of stdout
    RqMessage *msg = calloc(a->n + 1, sizeof(RqMessage));
    if (!msg) die("malloc failed!");
    for (int i = 0; i < a->n; i++) {
        const Js *r = json_get(a->v[i], "role"), *c = json_get(a->v[i], "content");
        if (!r || !c || c->t != JS_STR) die("-m render: each message needs a role and a string content");
        msg[i].role = json_str_eq(r, "system") || json_str_eq(r, "developer") ? RQ_ROLE_SYSTEM : json_str_eq(r, "user") ? RQ_ROLE_USER
                    : json_str_eq(r, "assistant") ? RQ_ROLE_ASSISTANT : -1;
        msg[i].content = c->s;
    }
    RqTokens t = { 0 };
    int e = rq_chat_render(m, msg, a->n, thinking, &t);
    if (e != RQ_OK) { fprintf(stderr, "-m render: %s\n", rq_strerror(e)); exit(EXIT_FAILURE); }
    for (int i = 0; i < t.n; i++)
        printf(i ? " %d" : "%d", t.v[i]);
    printf("\n");
    RqModelInfo info;
    rq_model_info(m, &info);
    if (info.chat == RQ_CHAT_QWEN38) // (the effort: Qwen3.8 only, the other lines unchanged)
        fprintf(stderr, "[render] chat template %s, thinking %d (%s): %d tokens\n", rq_chat_name(info.chat), thinking != 0,
                rq_chat_effort_name(thinking), t.n);
    else
        fprintf(stderr, "[render] chat template %s, thinking %d: %d tokens\n", rq_chat_name(info.chat), thinking != 0, t.n);
    rq_tokens_free(&t);
    free(msg);
}

int serve_render(RqModel *m, const Js *body, RqTokens *out, int *thinking, char *err, size_t errlen); // serve.c

static void render_request(RqModel *m, const Js *body) {
    // one chat completion request body rendered as runq serve renders it (tools, tool messages: serve.c, tools.c)
    RqTokens t = { 0 };
    char err[512] = "";
    int thinking = 0, e = serve_render(m, body, &t, &thinking, err, sizeof(err));
    if (e != RQ_OK) { fprintf(stderr, "-m render: %s\n", err[0] ? err : rq_strerror(e)); exit(EXIT_FAILURE); }
    for (int i = 0; i < t.n; i++)
        printf(i ? " %d" : "%d", t.v[i]);
    printf("\n");
    RqModelInfo info;
    rq_model_info(m, &info);
    fprintf(stderr, "[render] chat template %s, thinking %d (%s): %d tokens (request)\n", rq_chat_name(info.chat), thinking != 0,
            rq_chat_effort_name(thinking), t.n);
    rq_tokens_free(&t);
}

static void render_chat(const RqModel *m, const char *json, int thinking) {
    // -m render (test hook): the token ids of the conversation -i (a JSON array of {"role": system / user /
    // assistant, "content": text}) rendered by rq_chat_render with -r (and -e), space-separated on stdout; or of several
    // (-i: an array of {"messages": [...], "thinking": 0 / 1 (default -r), "reasoning_effort": xhigh / high / medium /
    // low (default -e)}) or {"request": a chat completion request body} (as runq serve renders it: tools), one line each
    char err[256];
    JsDoc *d = json_parse(json, strlen(json), err, sizeof(err));
    if (!d) { fprintf(stderr, "-m render: invalid JSON: %s\n", err); exit(EXIT_FAILURE); }
    const Js *a = json_root(d);
    if (a->t != JS_ARR) die("-m render: -i must be a JSON array of messages");
    if (a->n > 0 && (json_get(a->v[0], "messages") || json_get(a->v[0], "request"))) {
        for (int i = 0; i < a->n; i++) {
            if (json_get(a->v[i], "request")) {
                render_request((RqModel *)m, json_get(a->v[i], "request"));
                continue;
            }
            const Js *ms = json_get(a->v[i], "messages"), *th = json_get(a->v[i], "thinking");
            const Js *ef = json_get(a->v[i], "reasoning_effort");
            long long t = thinking != 0;
            int eff = thinking ? thinking : RQ_THINK_ON;
            if (!ms || ms->t != JS_ARR || (th && !json_is_int(th, &t))) die("-m render: bad conversation");
            if (ef && (ef->t != JS_STR || (eff = rq_chat_effort(ef->s)) < 0)) die("-m render: bad reasoning_effort");
            render_one(m, ms, t ? eff : 0);
        }
    } else {
        render_one(m, a, thinking);
    }
    json_free(d);
}

static int thinking_for(const RqModel *m, int thinking, int given) {
    // -r for the template variant: Instruct-2507 has no thinking mode, Thinking-2507 always thinks (a note
    // when -r asked for the other); thinking: 0 or the effort (RQ_THINK_*)
    RqModelInfo info;
    rq_model_info(m, &info);
    if (info.chat == RQ_CHAT_INSTRUCT && thinking) {
        fprintf(stderr, "note: this model (chat template %s) has no thinking mode: -r 1 ignored\n", rq_chat_name(info.chat));
        return 0;
    }
    if (info.chat == RQ_CHAT_THINKING && !thinking) {
        if (given) fprintf(stderr, "note: this model (chat template %s) always thinks: -r 0 ignored\n", rq_chat_name(info.chat));
        return RQ_THINK_ON;
    }
    return thinking;
}

static int thinking_default(const RqModel *m, int thinking, int given, int effort, int effort_given, int note) {
    // Qwen3.8: thinking on unless -r 0 (its template's default); -e for another template: a note (chat)
    RqModelInfo info;
    rq_model_info(m, &info);
    if (info.chat == RQ_CHAT_QWEN38) return given ? thinking : effort;
    if (effort_given && note)
        fprintf(stderr, "note: this model (chat template %s) has no reasoning effort: -e ignored\n", rq_chat_name(info.chat));
    return thinking != 0;
}

static void dump_logits(const RqModel *m, RqCtx *c, RqSeq *s, char *text, char *out_path) {
    // file format: int32 n_tokens, int32 vocab_size, int32 ids[n_tokens], float32 logits[n_tokens][vocab_size]
    // -b > 1: batched path (chunks of -b rows, the logits of every row), else one token at a time
    // through the decode path (rq_prefill chooses by the context's batch size)
    RqModelInfo info;
    rq_model_info(m, &info);
    int vocab_size = info.vocab_size, seq_len = rq_seq_capacity(s);
    RqTokens t = { 0 };
    check(rq_tokenize(m, text, -1, &t), NULL);
    int n_tokens = t.n;
    if (n_tokens < 1 || n_tokens > seq_len) {
        fprintf(stderr, "dump_logits: need 1..%d tokens, got %d\n", seq_len, n_tokens);
        exit(EXIT_FAILURE);
    }

    float *out = (float *)malloc((size_t)n_tokens * vocab_size * sizeof(float));
    if (!out) die("malloc failed!");
    check(rq_prefill(c, s, t.v, n_tokens, RQ_LOGITS_ALL, out, NULL), c);

    FILE *file = fopen(out_path, "wb");
    if (!file) { fprintf(stderr, "Couldn't open %s for writing\n", out_path); exit(EXIT_FAILURE); }
    int32_t header[2] = { n_tokens, vocab_size };
    int ok = fwrite(header, sizeof(int32_t), 2, file) == 2
          && fwrite(t.v, sizeof(int32_t), n_tokens, file) == (size_t)n_tokens
          && fwrite(out, sizeof(float), (size_t)n_tokens * vocab_size, file) == (size_t)n_tokens * vocab_size;
    if (fclose(file) != 0 || !ok) { fprintf(stderr, "Error writing %s\n", out_path); exit(EXIT_FAILURE); }
    free(out);
    rq_tokens_free(&t);
}

static void dump_dequant(const RqModel *m, const char *spec, const char *out_path) {
    // -m dequant -i NAME[:ROW0[:NROWS]] -o FILE (GGUF only) writes rows ROW0.. (default 0, 1 row) of the
    // matrix NAME (llama.cpp tensor name) dequantized from the file's blocks and, for the K-quants, also
    // from the repacked weights. FILE: int32 nrows, ncols, npacked (0 or 1), float32 [nrows][ncols]
    // from the blocks, then [nrows][ncols] from the repacked layout if npacked
    RqModelInfo info;
    rq_model_info(m, &info);
    if (!info.gguf) die("-m dequant: GGUF files only");
    size_t nlen = strcspn(spec, ":");
    long r0 = 0, nr = 1;
    if (spec[nlen]) {
        char *e;
        r0 = strtol(spec + nlen + 1, &e, 10);
        if (*e == ':') nr = strtol(e + 1, &e, 10);
        if (*e) { fprintf(stderr, "-m dequant: bad spec %s\n", spec); exit(EXIT_FAILURE); }
    }
    float *rows, *packed;
    int ncols, npacked;
    char err[256];
    int e = rq_dequant_rows(m, spec, nlen, r0, nr, &rows, &packed, &ncols, &npacked, err, sizeof(err));
    if (e == RQ_ERR_NOMEM) die("malloc failed!");
    if (e != RQ_OK) { fprintf(stderr, "-m dequant: %s\n", err); exit(EXIT_FAILURE); }
    FILE *f = fopen(out_path, "wb");
    if (!f) { fprintf(stderr, "Couldn't open %s for writing\n", out_path); exit(EXIT_FAILURE); }
    int32_t hdr[3] = { (int32_t)nr, ncols, npacked };
    size_t nv = (size_t)nr * ncols;
    int ok = fwrite(hdr, sizeof(int32_t), 3, f) == 3 && fwrite(rows, sizeof(float), nv, f) == nv &&
             (!npacked || fwrite(packed, sizeof(float), nv, f) == nv);
    if (fclose(f) != 0 || !ok) { fprintf(stderr, "Error writing %s\n", out_path); exit(EXIT_FAILURE); }
    free(rows);
    free(packed);
}

static void print_memory(const RqModel *m, int seq_len) {
    // --mem: the load's estimate (rq_model_mem) and what it was checked against
    RqModelMem mm;
    if (rq_model_mem(m, &mm) != RQ_OK) return;
    const double G = 1073741824.0;
    char st[96] = "";
    if (mm.seq.state)
        snprintf(st, sizeof(st), " + recurrent state %.1f MiB x (1 + %d checkpoints)", mm.seq.state / 1048576.0, mm.seq.checkpoints);
    fprintf(stderr, "[memory] estimate %.2f GiB: weights %.2f + runtime %.2f + a sequence of %d positions %.2f (KV cache %.2f%s); ",
            mm.total / G, mm.weights / G, mm.runtime / G, seq_len, mm.seq.total / G, mm.seq.kv / G, st);
    if (mm.budget_src) fprintf(stderr, "%s %.2f GiB\n", mm.budget_src == 2 ? "--mem" : "MemAvailable", mm.budget / G);
    else fprintf(stderr, "not checked\n");
}

// ----------------------------------------------------------------------------
// CLI

static void error_usage(void) {
    fprintf(stderr, "Usage:   runq <checkpoint> [options]\n");
    fprintf(stderr, "         checkpoint: a qwen3.c .bin (with its .tokenizer / .template* files) or a GGUF file (Q8_0, Q3_K_M .. Q5_K_M)\n");
    fprintf(stderr, "Example: runq Qwen3-0.6B-Q8_0.gguf -c 4096 -r 1\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1], default 0.9\n");
    fprintf(stderr, "  -K <int>    top-k: sample among the K most likely tokens (then top-p among them), 0 = off (default)\n");
    fprintf(stderr, "              (-t / -p / -K not given: the model's recommended values if its GGUF has them, general.sampling.*)\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -c <int>    context window size, 0 (default) = the model's maximum, at most %d; chat: holds the whole conversation,\n", RQ_CTX_DEFAULT_MAX);
    fprintf(stderr, "              when full the oldest turns are dropped (the system prompt is kept)\n");
    fprintf(stderr, "  -m <string> mode: generate|chat|tokenize|logits|dequant|render, default: chat\n");
    fprintf(stderr, "              tokenize: print the token ids of -i (raw text, no template)\n");
    fprintf(stderr, "              render (test hook): the token ids of the chat template rendering of -i, a JSON array of\n");
    fprintf(stderr, "              messages ({\"role\": \"user\", \"content\": \"...\"}), with -r; or of [{\"request\": BODY}, ...]:\n");
    fprintf(stderr, "              chat completion request bodies as runq serve renders them (tools)\n");
    fprintf(stderr, "              logits: write the logits at every position of -i to -o (binary)\n");
    fprintf(stderr, "  -i <string> input prompt (chat: one turn, then exit; without -i, one turn per stdin line,\n");
    fprintf(stderr, "              the conversation is kept until an empty line or EOF)\n");
    fprintf(stderr, "              dequant (GGUF, test hook): rows of the matrix -i NAME[:ROW0[:NROWS]] to -o (binary)\n");
    fprintf(stderr, "  -o <string> output file for -m logits / dequant\n");
    fprintf(stderr, "  -y <string> system prompt in chat mode, default is none\n");
    fprintf(stderr, "  -r <int>    reasoning mode, 0 (default) = no thinking, 1 = thinking (Qwen3-*-Instruct-2507: never,\n");
    fprintf(stderr, "              -Thinking-2507: always; Qwen3.8: 1 by default; the chat template variant is read from the GGUF)\n");
    fprintf(stderr, "  -e <string> reasoning effort with thinking (Qwen3.8's template): xhigh (default) | high (= xhigh) | medium | low\n");
    fprintf(stderr, "  -n <int>    max new tokens per response, 0 (default) = unlimited\n");
    fprintf(stderr, "  -b <int>    prefill chunk size (batched prompt processing), default %d, max %d;\n", RQ_BATCH_MAX, RQ_BATCH_MAX);
    fprintf(stderr, "              0 or 1 = one token at a time\n");
    fprintf(stderr, "  -k <string> KV cache type: f32|bf16|f16|bf16-dot|f16-dot, default %s (16-bit: half the memory and bandwidth);\n", rq_kv_name(RQ_KV_DEFAULT));
    fprintf(stderr, "              bf16-dot / f16-dot: that cache with the bf16 dot-product attention (q, K, V and the softmax weights\n");
    fprintf(stderr, "              rounded to bf16, the products as vdpbf16ps: ~1.7x the prefill attention at long context on AVX512_BF16\n");
    fprintf(stderr, "              CPUs; less accurate, see the README; bf16-dot is the faster one)\n");
    fprintf(stderr, "  --mem <GiB> refuse the load (before reading the weights) if its estimate (weights + runtime + the KV cache of -c\n");
    fprintf(stderr, "              positions) does not fit, default MemAvailable; 0 or force = no check; given: prints it ([memory])\n");
    fprintf(stderr, "  --page-cache auto|drop|keep  the model file's page cache after the load: auto (default) drops it if the file\n");
    fprintf(stderr, "              is >= 1/4 of the RAM\n");
    fprintf(stderr, "  -D <string> draft model for speculative decoding (generate, chat): a smaller model of the same vocabulary\n");
    fprintf(stderr, "              (e.g. Qwen3-0.6B for Qwen3-1.7B) guesses, the model checks; the output is exactly plain decoding's\n");
    fprintf(stderr, "  -G <int>    with -D: guesses per pass of the model, at most %d, default %d; 0 = plain decoding (no draft)\n", RQ_SPEC_K_MAX, SPEC_K_DEFAULT);
    fprintf(stderr, "  --tau <float> with -D: adaptive guess count, stop guessing at a draft top-1 probability < tau, default 0 (off)\n");
    fprintf(stderr, "  --lookup <int> prompt lookup (generate, chat): guess up to N tokens by copying what followed the latest match of\n");
    fprintf(stderr, "              the last tokens earlier in the context (no model; with -D: the draft model where none matches),\n");
    fprintf(stderr, "              at most %d, default 0 (off); the output is exactly plain decoding's\n", RQ_SPEC_K_MAX);
    fprintf(stderr, "  --lookup-min <int> with --lookup: the shortest match used, in tokens (>= 2), default %d\n", RQ_LOOKUP_MIN_DEFAULT);
    fprintf(stderr, "Threads: OMP_NUM_THREADS, default nproc - nproc/8 (Apple Silicon: the performance cores)\n");
    exit(EXIT_FAILURE);
}

int serve_main(int argc, char **argv); // serve.c: `runq serve <checkpoint> ...` (the HTTP server)

static void load_draft(const char *path, const RqModel *target, const RqModelOpts *o, int kv_type, int prefill_chunk) {
    // -D: the draft model (the same load options as the model), its context and sequence; refused unless its
    // vocabulary is the model's
    char err[512];
    if (!(draft.m = rq_model_load(path, o, err, sizeof(err)))) die(err);
    RqModelInfo info;
    rq_model_info(draft.m, &info);
    if (info.warning) fprintf(stderr, "warning: draft: %s\n", info.warning);
    if (rq_spec_check(target, draft.m, err, sizeof(err)) != RQ_OK) {
        fprintf(stderr, "-D %s: %s\n", path, err);
        exit(EXIT_FAILURE);
    }
    int e;
    if (!(draft.s = rq_seq_new(draft.m, 0, kv_type, &e))) die(e == RQ_ERR_NOMEM ? "malloc failed!" : "-D: unsupported draft model shape");
    if (!(draft.c = rq_ctx_new(draft.m, prefill_chunk, 0, &e))) die("malloc failed!");
}

int main(int argc, char *argv[]) {
    if (argc >= 2 && !strcmp(argv[1], "serve")) return serve_main(argc - 1, argv + 1);
    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    float temperature = 1.0f;   // 0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int top_k = 0;              // top-k: 0 = off
    int samp_given = 0;         // RQ_SAMPLING_* of -t / -p / -K given (else the model's values, if any)
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "chat";        // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode
    int enable_thinking = 0;    // 1 enables thinking
    int thinking_given = 0;     // -r given (Thinking-2507: a note for -r 0; Qwen3.8: on unless -r 0)
    int effort = RQ_THINK_ON;   // -e: the reasoning effort with thinking (Qwen3.8), RQ_THINK_*
    int effort_given = 0;
    int ctx_length = 0;         // context length
    char *out_path = NULL;      // output file for -m logits
    int prefill_chunk = RQ_BATCH_MAX; // -b: rows per batched prefill chunk (0 / 1: token by token)
    int kv_type = RQ_KV_DEFAULT;
    double mem = 0;             // --mem: GiB, < 0 no check, 0 MemAvailable
    int mem_given = 0, page_cache = 0;
    char *draft_path = NULL;    // -D: the draft model (speculative decoding)
    int spec_k = -1;            // -G: guesses per pass (-1: SPEC_K_DEFAULT)
    float spec_tau = 0;         // --tau
    int lookup = 0, lookup_min = 0; // --lookup, --lookup-min (0: RQ_LOOKUP_MIN_DEFAULT)

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (!strcmp(argv[i], "--mem")) {
            char *end;
            mem = strtod(argv[i + 1], &end);
            if (!strcmp(argv[i + 1], "force") || !strcmp(argv[i + 1], "off")) mem = -1;
            else if (end == argv[i + 1] || *end || !(mem >= 0)) error_usage();
            else if (mem == 0) mem = -1;
            mem_given = 1;
            continue;
        }
        if (!strcmp(argv[i], "--page-cache")) {
            const char *v = argv[i + 1];
            page_cache = !strcmp(v, "drop") ? 1 : !strcmp(v, "keep") ? -1 : !strcmp(v, "auto") ? 0 : 2;
            if (page_cache == 2) error_usage();
            continue;
        }
        if (!strcmp(argv[i], "--lookup") || !strcmp(argv[i], "--lookup-min")) {
            char *end;
            long v = strtol(argv[i + 1], &end, 10);
            int mn = argv[i][8] == '-';
            if (end == argv[i + 1] || *end || v < (mn ? 2 : 0) || v > (mn ? RQ_LOOKUP_LMAX : RQ_SPEC_K_MAX)) error_usage();
            *(mn ? &lookup_min : &lookup) = (int)v;
            continue;
        }
        if (!strcmp(argv[i], "--tau")) {
            char *end;
            spec_tau = strtof(argv[i + 1], &end);
            if (end == argv[i + 1] || *end || !(spec_tau >= 0 && spec_tau <= 1)) error_usage();
            continue;
        }
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); samp_given |= RQ_SAMPLING_TEMP; }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); samp_given |= RQ_SAMPLING_TOPP; }
        else if (argv[i][1] == 'K') { top_k = atoi(argv[i + 1]); samp_given |= RQ_SAMPLING_TOPK; }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'c') { ctx_length = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else if (argv[i][1] == 'r') { enable_thinking = atoi(argv[i + 1]); thinking_given = 1; }
        else if (argv[i][1] == 'e') {
            effort = rq_chat_effort(argv[i + 1]);
            if (effort < 0) error_usage();
            effort_given = 1;
        }
        else if (argv[i][1] == 'n') { max_new_tokens = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'o') { out_path = argv[i + 1]; }
        else if (argv[i][1] == 'b') { prefill_chunk = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'D') { draft_path = argv[i + 1]; }
        else if (argv[i][1] == 'G') {
            char *end;
            long k = strtol(argv[i + 1], &end, 10);
            if (end == argv[i + 1] || *end || k < 0 || k > RQ_SPEC_K_MAX) error_usage();
            spec_k = (int)k;
        }
        else if (argv[i][1] == 'k') {
            kv_type = rq_kv_parse(argv[i + 1]);
            if (kv_type < 0) error_usage();
        }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0) temperature = 0;
    if (topp < 0 || 1.0 < topp) topp = 0.9;
    if (top_k < 0) top_k = 0;
    if ((strcmp(mode, "tokenize") == 0 || strcmp(mode, "logits") == 0 || strcmp(mode, "render") == 0) && !prompt) { error_usage(); }
    if ((strcmp(mode, "logits") == 0 || strcmp(mode, "dequant") == 0) && (!out_path || !prompt)) { error_usage(); }
    if (prefill_chunk < 0) prefill_chunk = 0;
    if (prefill_chunk > RQ_BATCH_MAX) prefill_chunk = RQ_BATCH_MAX;
    if (ctx_length < 0) die("-c: the context size must be >= 0");
    if (spec_k > 0 && !draft_path) die("-G: speculative decoding needs a draft model (-D)");
    if (spec_k < 0) spec_k = SPEC_K_DEFAULT;
    enable_thinking = enable_thinking != 0 ? effort : 0; // (from here: 0 or the effort, RQ_THINK_*)

    char err[512];
    if (strcmp(mode, "tokenize") == 0 || strcmp(mode, "render") == 0) {
        // only the header and the tokenizer (GGUF: the metadata, the tensor data stays unread), on one
        // thread (one light process: the tests run many in parallel)
        rq_threads_init(1);
        double t0 = rq_now();
        RqModelOpts o = { 0, 1, 0 };
        RqModel *m = rq_model_load(checkpoint_path, &o, err, sizeof(err));
        if (!m) die(err);
        enable_thinking = thinking_default(m, enable_thinking, thinking_given, effort, effort_given, 0);
        // (-m render: the library's rendering for -r as given, e.g. Thinking-2507 with thinking off as runq serve
        // renders a constrained request; the chat forces the variant's mode: thinking_for)
        if (rq_chat_check(m, enable_thinking, err, sizeof(err)) == RQ_ERR_LOAD) {
            fprintf(stderr, "Couldn't load prompt template %s\n", err);
            exit(EXIT_FAILURE);
        }
        if (strcmp(mode, "render") == 0)
            render_chat(m, prompt, enable_thinking);
        else
            tokenize(m, prompt, rq_now() - t0);
        rq_model_free(m);
        return 0;
    }

    // threads: OMP_NUM_THREADS, else the default (rq_threads_init)
    const char *warn = rq_threads_init(0);
    if (warn) fprintf(stderr, "%s\n", warn);

    // the model (weights, tokenizer, templates); -m dequant: the weights only
    int dequant = strcmp(mode, "dequant") == 0;
    RqModelOpts o = { ctx_length, 0, dequant, mem, kv_type == RQ_KV_F32 ? 4 : 2, page_cache };
    RqModel *m = rq_model_load(checkpoint_path, &o, err, sizeof(err));
    if (!m) die(err);
    RqModelInfo info;
    rq_model_info(m, &info);
    if (info.warning) fprintf(stderr, "warning: %s\n", info.warning);
    if (mem_given) print_memory(m, info.seq_len);
    if (dequant) {
        dump_dequant(m, prompt, out_path);
        rq_model_free(m);
        return 0;
    }
    enable_thinking = thinking_default(m, enable_thinking, thinking_given, effort, effort_given, strcmp(mode, "chat") == 0);
    if (strcmp(mode, "chat") == 0)
        enable_thinking = thinking_for(m, enable_thinking, thinking_given);
    if (rq_chat_check(m, enable_thinking, err, sizeof(err)) == RQ_ERR_LOAD) {
        fprintf(stderr, "Couldn't load prompt template %s\n", err);
        exit(EXIT_FAILURE);
    }
    // -D (generate, chat; -G 0: plain decoding, not loaded)
    if (draft_path && spec_k > 0 && (strcmp(mode, "generate") == 0 || strcmp(mode, "chat") == 0)) {
        load_draft(draft_path, m, &o, kv_type, prefill_chunk);
        draft.o.k = spec_k;
        draft.o.tau = spec_tau;
    }
    if (lookup > 0 && (strcmp(mode, "generate") == 0 || strcmp(mode, "chat") == 0)) {
        draft.o.lookup = lookup;
        draft.o.lookup_min = lookup_min;
    }

    // the sequence (KV cache of -c positions) and the compute context (batched buffers for -b rows)
    int e;
    RqSeq *seq = rq_seq_new(m, 0, kv_type, &e);
    if (!seq && e == RQ_ERR_ARG) {
        fprintf(stderr, "unsupported model shape (heads %d / %d, head_dim %d, dim %d, hidden %d, GS %d)\n", info.n_heads,
                info.n_kv_heads, info.head_dim, info.dim, info.hidden_dim, info.group_size);
        exit(EXIT_FAILURE);
    }
    if (!seq) die("malloc failed!");
    RqCtx *ctx = rq_ctx_new(m, prefill_chunk, 0, &e);
    if (!ctx) die("malloc failed!");
    rq_seq_seed(seq, rng_seed);
    RqSampling cfg = { temperature, topp, NULL, top_k }, mcfg = cfg;
    int ms = rq_model_sampling(m, &mcfg) & ~samp_given; // (the file's values for the options not given; silent)
    if (ms & RQ_SAMPLING_TEMP) cfg.temperature = mcfg.temperature;
    if (ms & RQ_SAMPLING_TOPP) cfg.topp = mcfg.topp;
    if (ms & RQ_SAMPLING_TOPK) cfg.top_k = mcfg.top_k;

    if (!prompt)
        printf("hidden_size=%d, intermediate_size=%d, num_hidden_layers=%d, num_attention_heads=%d, num_kv_heads=%d, head_dim=%d, ctx_length=%d, vocab_size=%d, shared_classifier=%d, quantization_block_size=%d\n", info.dim, info.hidden_dim, info.n_layers, info.n_heads, info.n_kv_heads, info.head_dim, info.seq_len, info.vocab_size, info.shared_classifier, info.group_size);

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(m, ctx, seq, &cfg, prompt);
    } else if (strcmp(mode, "chat") == 0) {
        chat(m, ctx, seq, &cfg, prompt, system_prompt, enable_thinking);
    } else if (strcmp(mode, "logits") == 0) {
        dump_logits(m, ctx, seq, prompt, out_path);
    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        error_usage();
    }

    // -DRUNQ_PROFILE builds: time per batched-prefill phase
    const char *pn[16];
    double pt[16], pf_total = 0;
    int np = rq_ctx_profile(ctx, pn, pt, 16);
    for (int i = 0; i < np; i++) pf_total += pt[i];
    if (np) fprintf(stderr, "[profile] forward_batch total %.3f s\n", pf_total);
    for (int i = 0; i < np; i++)
        fprintf(stderr, "[profile]   %-18s %8.3f s  %5.1f%%\n", pn[i], pt[i], 100 * pt[i] / (pf_total > 0 ? pf_total : 1));

    // memory and file handles cleanup
    rq_ctx_free(ctx);
    rq_seq_free(seq);
    rq_model_free(m);
    if (draft.m) {
        rq_ctx_free(draft.c);
        rq_seq_free(draft.s);
        rq_model_free(draft.m);
    }
    return 0;
}
