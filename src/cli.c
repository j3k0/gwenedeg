/* CLI. */
#include "internal.h"

// ----------------------------------------------------------------------------
// timing / step limit (cwayn patch)

int max_new_tokens = 0; // -n: stop after this many generated tokens, 0 = unlimited

double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void report_timing(int n_prompt, double t0, double t_prefill, int n_gen, double t_end) {
    double pf = t_prefill - t0, dc = t_end - t_prefill;
    fprintf(stderr, "\n[timing] prompt: %d tok in %.3f s (%.1f tok/s) | generated: %d tok in %.3f s (%.1f tok/s)\n",
            n_prompt, pf, n_prompt / (pf > 0 ? pf : 1e-9), n_gen, dc, n_gen / (dc > 0 ? dc : 1e-9));
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int*)malloc(ENCODE_MAX_TOKENS(strlen(prompt)) * sizeof(int));
    encode(tokenizer, prompt, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "Please provide a prompt using -i <string> on the command line.\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    int next = 0;    // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    int n_gen = 0;
    double t0 = now_s(), t_prefill = t0;
    float *pre_logits = NULL; // logits of the last prompt token, from the batched prefill

    if (transformer->state.nb_max > 1 && num_prompt_tokens <= transformer->config.seq_len) {
        // batched prefill of the whole prompt, then resume the loop at its last token
        pre_logits = prefill(transformer, prompt_tokens, num_prompt_tokens, 0);
        for (int i = 0; i < num_prompt_tokens - 1; i++)
            printf("%s", decode(tokenizer, prompt_tokens[i]));
        fflush(stdout);
        pos = num_prompt_tokens - 1;
        token = prompt_tokens[pos];
    }

    while (pos < transformer->config.seq_len) {
        // forward the transformer to get logits for the next token
        // (only needed from the last prompt token on)
        float *logits = pre_logits ? pre_logits : forward(transformer, token, pos, pos >= num_prompt_tokens - 1);
        pre_logits = NULL;

        // advance the state state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            if (pos == num_prompt_tokens - 1) t_prefill = now_s();
            else n_gen++;
            next = sample(sampler, logits);
        }
        pos++;
        if (max_new_tokens > 0 && n_gen >= max_new_tokens) break;

        // print the token as string, decode it with the Tokenizer object
        printf("%s", decode(tokenizer, token));
        fflush(stdout);
        token = next;

        // data-dependent terminating condition: the BOS token delimits sequences
        if (pos >= num_prompt_tokens && (next == tokenizer->bos_token_id || next == tokenizer->eos_token_id))
            break;
    }
    printf("\n");
    report_timing(num_prompt_tokens, t0, t_prefill, n_gen, now_s());
    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// chat loop (cwayn: multi-turn, the conversation stays in the KV cache)
//
// The conversation is kept as token pieces, rendered with Qwen3's chat template (ChatML,
// models/Qwen3-0.6B/tokenizer_config.json; the exporter's .template* files hold its renderings):
//   [system]  "<|im_start|>system\n" SYS "<|im_end|>\n"                       (-y; = the start of
//             .template.with-system, i.e. that file minus .template)
//   turn i    P_i = "<|im_start|>user\n" U_i "<|im_end|>\n<|im_start|>assistant\n" [+ G]
//             (= .template / .template.with-thinking; G = "<think>\n\n</think>\n\n" with -r 0)
//             then the reply R_i as generated (token ids, without the final <|im_end|>)
//             then E = "<|im_end|>\n" (template: '<|im_end|>' + '\n' after every message)
// A new turn k is the canonical sequence  [system] + sum over kept turns i < k (P_i + H_i + E) + P_k,
// where H_i is the history form of R_i: -r 0: R_i as generated (and the empty think block G stays in
// P_i); -r 1: R_i after its last </think>, leading newlines stripped, as the template does for
// earlier assistant turns (it drops their reasoning). Of that sequence, the longest prefix that is
// already in the cache is kept and the rest is prefilled from there: normally just
// E + P_k (the sampled <|im_end|> that ended the reply is the first token of the next prefill, so
// it is in the cache before the next turn), with -r 1 from the start of the previous answer
// (its reasoning is dropped and the answer re-prefilled). Text pieces are encoded separately; they
// are separated by added tokens (<|im_start|>, <|im_end|>), so this is the same as encoding the
// whole rendered string.
// Context full: if the canonical sequence plus a reserve for the reply (-n, else seq_len / 4, at most
// seq_len / 2) does not fit in -c, the oldest turns are dropped until it is at most half of the
// rest (seq_len - reserve) / 2, keeping the system prompt (cached prefix) and the newest turns, and
// the rest is re-prefilled. A reply reaching the end of the context is cut there.
// RNG: one draw per prefilled token but the last, then one per sampled token (seeded output does
// not depend on -b; for a single turn the same draws as before batched prefill).

typedef struct { int *v; int n, cap; } IntVec;

static void iv_reserve(IntVec *a, size_t n) {
    if (n <= (size_t)a->cap) return;
    size_t cap = a->cap ? (size_t)a->cap : 256;
    while (cap < n) cap *= 2;
    a->v = realloc(a->v, cap * sizeof(int));
    if (!a->v || cap > 0x7fffffff) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    a->cap = (int)cap;
}

static void iv_append(IntVec *a, const int *x, int n) {
    if (n <= 0) return; // (x may be NULL then)
    iv_reserve(a, (size_t)a->n + n);
    memcpy(a->v + a->n, x, n * sizeof(int));
    a->n += n;
}

static void iv_push(IntVec *a, int x) { iv_append(a, &x, 1); }

static void encode_append(Tokenizer *t, IntVec *v, const char *text) {
    // append the tokens of text (raw, no template)
    iv_reserve(v, (size_t)v->n + ENCODE_MAX_TOKENS(strlen(text)));
    int n = 0;
    encode(t, (char *)text, v->v + v->n, &n);
    v->n += n;
}

static char *render1(const char *tmpl, const char *arg) {
    // tmpl with its one %s replaced by arg (malloc'd)
    int n = snprintf(NULL, 0, tmpl, arg);
    char *s = malloc((size_t)n + 1);
    if (!s) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    snprintf(s, (size_t)n + 1, tmpl, arg);
    return s;
}

static char *read_line(FILE *f) {
    // one line of any length without its "\n" (or "\r\n"), malloc'd; NULL at EOF with nothing read
    size_t cap = 256, n = 0;
    char *s = malloc(cap);
    if (!s) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    int c;
    while ((c = fgetc(f)) != EOF && c != '\n') {
        if (n + 1 >= cap && !(s = realloc(s, cap *= 2))) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
        s[n++] = (char)c;
    }
    if (c == EOF && n == 0) { free(s); return NULL; }
    if (n > 0 && s[n - 1] == '\r') n--;
    s[n] = 0;
    return s;
}

static float *feed_tokens(Transformer *tr, const int *tokens, int n, int pos0) {
    // tokens[0..n-1] at positions pos0.. (n >= 1, pos0 + n <= seq_len): batched prefill, or one
    // token at a time with -b 0/1; returns the logits of the last one
    if (tr->state.nb_max > 1)
        return prefill(tr, (int *)tokens, n, pos0);
    float *logits = NULL;
    for (int i = 0; i < n; i++)
        logits = forward(tr, tokens[i], pos0 + i, i == n - 1);
    return logits;
}

static void history_reply(Tokenizer *t, const IntVec *r, int think_end, IntVec *h) {
    // -r 1: the reply as the template renders an earlier assistant turn: the text after the last
    // </think> (if any), leading "\n" stripped (content.split('</think>')[-1].lstrip('\n'))
    h->n = 0;
    int i = 0;
    for (int j = r->n - 1; j >= 0 && think_end >= 0; j--)
        if (r->v[j] == think_end) { i = j + 1; break; }
    if (i == 0) { iv_append(h, r->v, r->n); return; } // no reasoning block: as generated
    for (; i < r->n; i++) {
        const char *s = t->vocab[r->v[i]];
        int m = (int)strspn(s, "\n");
        if (m == t->vocab_len[r->v[i]]) continue; // only newlines
        if (m > 0) encode_append(t, h, s + m), i++; // newlines + text in one token: re-encode the text
        break;
    }
    iv_append(h, r->v + i, r->n - i);
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
        if (!cdump.pos || !cdump.logits) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
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

static void cdump_write(const IntVec *kv) {
    if (!cdump.prefix) return;
    size_t len = strlen(cdump.prefix) + 8;
    char *path = malloc(len);
    snprintf(path, len, "%s.ids", cdump.prefix);
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Couldn't open %s for writing\n", path); exit(EXIT_FAILURE); }
    for (int i = 0; i < kv->n; i++)
        fprintf(f, i ? " %d" : "%d", kv->v[i]);
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
    IntVec prompt; // P_i: user message + assistant header (+ empty think block with -r 0)
    IntVec reply;  // R_i: the generated tokens, without the <|im_end|> that ended them
    IntVec hist;   // H_i: the reply as part of the history
} ChatTurn;

static void render_conversation(IntVec *out, const IntVec *sys, const ChatTurn *turns, int j0, int k, const IntVec *close) {
    // [system] + turns j0..k-1 (prompt, history reply, <|im_end|>\n) + the prompt of turn k
    out->n = 0;
    iv_append(out, sys->v, sys->n);
    for (int i = j0; i < k; i++) {
        iv_append(out, turns[i].prompt.v, turns[i].prompt.n);
        iv_append(out, turns[i].hist.v, turns[i].hist.n);
        iv_append(out, close->v, close->n);
    }
    iv_append(out, turns[k].prompt.v, turns[k].prompt.n);
}

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *cli_user_prompt, char *system_prompt,
          int enable_thinking) {
    int seq_len = transformer->config.seq_len;
    IntVec sys = { 0 }, close = { 0 }, kv = { 0 }, want = { 0 };
    ChatTurn *turns = NULL;
    int n_turns = 0, j0 = 0; // turns[j0..] are in the conversation, older ones were dropped (context full)

    // the system block template: .template.with-system = system block + .template
    char sys_tmpl[1024] = "<|im_start|>system\n%s<|im_end|>\n";
    size_t ls = strlen(tokenizer->system_prompt_template), lp = strlen(tokenizer->prompt_template);
    if (ls > lp && !strcmp(tokenizer->system_prompt_template + ls - lp, tokenizer->prompt_template)) {
        memcpy(sys_tmpl, tokenizer->system_prompt_template, ls - lp);
        sys_tmpl[ls - lp] = 0;
    }
    if (system_prompt) {
        char *s = render1(sys_tmpl, system_prompt);
        encode_append(tokenizer, &sys, s);
        free(s);
    }
    encode_append(tokenizer, &close, "<|im_end|>\n");
    IntVec te = { 0 };
    encode_append(tokenizer, &te, "</think>");
    int think_end = te.n == 1 ? te.v[0] : -1;
    free(te.v);
    int reserve = max_new_tokens > 0 ? max_new_tokens : seq_len / 4;
    if (reserve > seq_len / 2) reserve = seq_len / 2;
    if (reserve < 1) reserve = 1;
    cdump.prefix = getenv("RUNQ_CHAT_DUMP");
    cdump.vocab = transformer->config.vocab_size;
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
            user = render1("%s", cli_user_prompt);
        } else {
            printf("\n> ");
            fflush(stdout);
            user = read_line(stdin);
            if (!user || !user[0]) { free(user); break; }
            if (echo) printf("%s\n", user);
            fflush(stdout);
        }
        turns = realloc(turns, (size_t)(n_turns + 1) * sizeof(ChatTurn));
        if (!turns) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
        int k = n_turns;
        ChatTurn *tn = &turns[k];
        memset(tn, 0, sizeof(*tn));
        char *r = render1(tokenizer->prompt_template, user);
        encode_append(tokenizer, &tn->prompt, r);
        free(r);
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
            free(tn->prompt.v);
            continue;
        }
        n_turns++;
        if (j > j0)
            fprintf(stderr, "[context] full (-c %d): dropped the %d oldest turn(s), kept %s%d earlier turn(s)\n", seq_len,
                    j - j0, sys.n ? "the system prompt and " : "", k - j);
        j0 = j;

        // prefill what is not in the cache yet (the cache holds kv.v[0 .. kv.n), positions 0 .. kv.n-1)
        int L = 0;
        while (L < kv.n && L < want.n && kv.v[L] == want.v[L]) L++;
        cdump_truncate(L);
        kv.n = L;
        int n_new = want.n - L;
        double t0 = now_s();
        float *logits = feed_tokens(transformer, want.v + L, n_new, L);
        iv_append(&kv, want.v + L, n_new);
        if (sampler->temperature != 0)
            for (int i = 0; i < n_new - 1; i++)
                random_f32(&sampler->rng_state);
        double t_prefill = now_s();

        // the reply: sample, print, feed back; ends at <|im_end|> (sampled, not fed yet: the next
        // turn's prefill starts with it), at -n tokens, or at the end of the context
        int n_gen = 0;
        while (1) {
            cdump_record(kv.n - 1, logits);
            int next = sample(sampler, logits);
            if (next == (int)tokenizer->eos_token_id || next == (int)tokenizer->bos_token_id)
                break;
            printf("%s", decode(tokenizer, next));
            fflush(stdout);
            iv_push(&tn->reply, next);
            if (kv.n >= seq_len) {
                fprintf(stderr, "\n(context window full (-c %d): reply cut)", seq_len);
                break;
            }
            int last = max_new_tokens > 0 && n_gen + 1 >= max_new_tokens;
            logits = forward(transformer, next, kv.n, !last);
            iv_push(&kv, next);
            n_gen++;
            if (last)
                break;
        }
        printf("\n");
        report_timing(n_new, t0, t_prefill, n_gen, now_s());
        if (enable_thinking)
            history_reply(tokenizer, &tn->reply, think_end, &tn->hist);
        else
            iv_append(&tn->hist, tn->reply.v, tn->reply.n);
    }
    cdump_write(&kv);
    for (int i = 0; i < n_turns; i++) {
        free(turns[i].prompt.v);
        free(turns[i].reply.v);
        free(turns[i].hist.v);
    }
    free(turns);
    free(sys.v);
    free(close.v);
    free(kv.v);
    free(want.v);
    free(cdump.pos);
    free(cdump.logits);
}

// ----------------------------------------------------------------------------
// test hooks (cwayn patch): -m tokenize, -m logits (see tests/)

void tokenize(Tokenizer *tokenizer, char *text, double t_load) {
    // encode the raw text (no chat template), print the token ids space-separated on stdout;
    // timings on stderr (encode only, and the tokenizer load before it)
    int n_tokens = 0;
    int *tokens = (int *)malloc(ENCODE_MAX_TOKENS(strlen(text)) * sizeof(int));
    double t0 = now_s();
    encode(tokenizer, text, tokens, &n_tokens);
    double t_enc = now_s() - t0;
    for (int i = 0; i < n_tokens; i++)
        printf(i ? " %d" : "%d", tokens[i]);
    printf("\n");
    fprintf(stderr, "[tokenize] %zu bytes -> %d tokens, encode %.3f ms, tokenizer load %.1f ms\n",
            strlen(text), n_tokens, t_enc * 1e3, t_load * 1e3);
    free(tokens);
}

void forward_sequential(Transformer *transformer, int *tokens, int n_tokens, float *out) {
    // reference prefill: one token at a time at positions 0..n_tokens-1,
    // row pos of out (n_tokens, vocab_size) = logits after feeding tokens[pos]
    int vocab_size = transformer->config.vocab_size;
    for (int pos = 0; pos < n_tokens; pos++) {
        float *logits = forward(transformer, tokens[pos], pos, 1);
        memcpy(out + (size_t)pos * vocab_size, logits, vocab_size * sizeof(float));
    }
}

void forward_chunked(Transformer *transformer, int *tokens, int n_tokens, float *out) {
    // the same with the batched path: chunks of state.nb_max rows, logits of every row
    int vocab_size = transformer->config.vocab_size, nb = transformer->state.nb_max;
    for (int i = 0; i < n_tokens; i += nb) {
        int m = n_tokens - i < nb ? n_tokens - i : nb;
        forward_batch(transformer, tokens + i, m, i, LOGITS_ALL, out + (size_t)i * vocab_size);
    }
}

void dump_logits(Transformer *transformer, Tokenizer *tokenizer, char *text, char *out_path) {
    // file format: int32 n_tokens, int32 vocab_size, int32 ids[n_tokens], float32 logits[n_tokens][vocab_size]
    // -b > 1: batched path (forward_chunked), else one token at a time (forward_sequential)
    int n_tokens = 0, vocab_size = transformer->config.vocab_size;
    int *tokens = (int *)malloc(ENCODE_MAX_TOKENS(strlen(text)) * sizeof(int));
    encode(tokenizer, text, tokens, &n_tokens);
    if (n_tokens < 1 || n_tokens > transformer->config.seq_len) {
        fprintf(stderr, "dump_logits: need 1..%d tokens, got %d\n", transformer->config.seq_len, n_tokens);
        exit(EXIT_FAILURE);
    }

    float *out = (float *)malloc((size_t)n_tokens * vocab_size * sizeof(float));
    if (!out) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    if (transformer->state.nb_max > 1)
        forward_chunked(transformer, tokens, n_tokens, out);
    else
        forward_sequential(transformer, tokens, n_tokens, out);

    FILE *file = fopen(out_path, "wb");
    if (!file) { fprintf(stderr, "Couldn't open %s for writing\n", out_path); exit(EXIT_FAILURE); }
    int32_t header[2] = { n_tokens, vocab_size };
    int ok = fwrite(header, sizeof(int32_t), 2, file) == 2
          && fwrite(tokens, sizeof(int32_t), n_tokens, file) == (size_t)n_tokens
          && fwrite(out, sizeof(float), (size_t)n_tokens * vocab_size, file) == (size_t)n_tokens * vocab_size;
    if (fclose(file) != 0 || !ok) { fprintf(stderr, "Error writing %s\n", out_path); exit(EXIT_FAILURE); }
    free(out);
    free(tokens);
}

void dump_dequant(Transformer *tr, const char *spec, const char *out_path) {
    // test hook, GGUF only: -m dequant -i NAME[:ROW0[:NROWS]] -o FILE writes rows ROW0.. (default 0,
    // 1 row) of the matrix NAME (llama.cpp tensor name) dequantized from the file's blocks (embed_row:
    // ggml's expressions) and, for Q4_K / Q6_K, also from the repacked weights (kq_packed_row, the
    // same expressions: checks the repacking). FILE: int32 nrows, ncols, npacked (0 or 1), float32
    // [nrows][ncols] from the blocks, then [nrows][ncols] from the repacked layout if npacked
    Config *p = &tr->config;
    TransformerWeights *w = &tr->weights;
    if (!tr->gguf.map) { fprintf(stderr, "-m dequant: GGUF files only\n"); exit(EXIT_FAILURE); }
    int L = p->n_layers, dim = p->dim, hid = p->hidden_dim, qd = p->n_heads * p->head_dim, kvd = p->n_kv_heads * p->head_dim;
    struct { const char *fmt; QuantizedTensor *t; int n, d, k; } m[] = {
        { "token_embd.weight", w->q_tokens, 1, p->vocab_size, dim }, { "output.weight", w->wcls, 1, p->vocab_size, dim },
        { "blk.%d.attn_q.weight", w->wq, L, qd, dim }, { "blk.%d.attn_k.weight", w->wk, L, kvd, dim },
        { "blk.%d.attn_v.weight", w->wv, L, kvd, dim }, { "blk.%d.attn_output.weight", w->wo, L, dim, qd },
        { "blk.%d.ffn_gate.weight", w->w1, L, hid, dim }, { "blk.%d.ffn_down.weight", w->w2, L, dim, hid },
        { "blk.%d.ffn_up.weight", w->w3, L, hid, dim } };
    size_t nlen = strcspn(spec, ":");
    long r0 = 0, nr = 1;
    if (spec[nlen]) {
        char *e;
        r0 = strtol(spec + nlen + 1, &e, 10);
        if (*e == ':') nr = strtol(e + 1, &e, 10);
        if (*e) { fprintf(stderr, "-m dequant: bad spec %s\n", spec); exit(EXIT_FAILURE); }
    }
    QuantizedTensor *t = NULL;
    int d = 0, k = 0;
    for (size_t i = 0; i < sizeof(m) / sizeof(m[0]) && !t; i++) {
        if (i == 1 && p->shared_classifier) continue; // (no output.weight: tied)
        for (int l = 0; l < m[i].n && !t; l++) {
            char name[96];
            snprintf(name, sizeof(name), m[i].fmt, l);
            if (strlen(name) == nlen && !memcmp(name, spec, nlen)) { t = m[i].t + l; d = m[i].d; k = m[i].k; }
        }
    }
    if (!t || !t->b) { fprintf(stderr, "-m dequant: no matrix %.*s\n", (int)nlen, spec); exit(EXIT_FAILURE); }
    if (r0 < 0 || nr < 1 || r0 > d - nr) { fprintf(stderr, "-m dequant: rows %ld..%ld out of range (%d rows)\n", r0, r0 + nr - 1, d); exit(EXIT_FAILURE); }
    float *out = malloc((size_t)2 * nr * k * sizeof(float));
    if (!out) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    int32_t hdr[3] = { (int32_t)nr, k, t->kp != NULL };
    for (long r = 0; r < nr; r++) {
        embed_row(out + (size_t)r * k, t, (int)(r0 + r), k);
        if (t->kp)
            kq_packed_row(t->type, t->kp, (int)(r0 + r), k, out + (size_t)(nr + r) * k);
    }
    FILE *f = fopen(out_path, "wb");
    if (!f) { fprintf(stderr, "Couldn't open %s for writing\n", out_path); exit(EXIT_FAILURE); }
    size_t nv = (size_t)(1 + hdr[2]) * nr * k;
    int ok = fwrite(hdr, sizeof(int32_t), 3, f) == 3 && fwrite(out, sizeof(float), nv, f) == nv;
    if (fclose(f) != 0 || !ok) { fprintf(stderr, "Error writing %s\n", out_path); exit(EXIT_FAILURE); }
    free(out);
}

// ----------------------------------------------------------------------------
// CLI

void error_usage() {
    fprintf(stderr, "Usage:   runq <checkpoint> [options]\n");
    fprintf(stderr, "         checkpoint: a qwen3.c .bin (with its .tokenizer / .template* files) or a GGUF file (Q8_0, Q4_K_M)\n");
    fprintf(stderr, "Example: runq Qwen3-0.6B-Q8_0.gguf -c 4096 -r 1\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1], default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -c <int>    context window size, 0 (default) = max_seq_len; chat: holds the whole conversation,\n");
    fprintf(stderr, "              when full the oldest turns are dropped (the system prompt is kept)\n");
    fprintf(stderr, "  -m <string> mode: generate|chat|tokenize|logits|dequant, default: chat\n");
    fprintf(stderr, "              tokenize: print the token ids of -i (raw text, no template)\n");
    fprintf(stderr, "              logits: write the logits at every position of -i to -o (binary)\n");
    fprintf(stderr, "  -i <string> input prompt (chat: one turn, then exit; without -i, one turn per stdin line,\n");
    fprintf(stderr, "              the conversation is kept until an empty line or EOF)\n");
    fprintf(stderr, "              dequant (GGUF, test hook): rows of the matrix -i NAME[:ROW0[:NROWS]] to -o (binary)\n");
    fprintf(stderr, "  -o <string> output file for -m logits / dequant\n");
    fprintf(stderr, "  -y <string> system prompt in chat mode, default is none\n");
    fprintf(stderr, "  -r <int>    reasoning mode, 0 (default) = no thinking, 1 = thinking\n");
    fprintf(stderr, "  -n <int>    max new tokens per response, 0 (default) = unlimited\n");
    fprintf(stderr, "  -b <int>    prefill chunk size (batched prompt processing), default %d, max %d;\n", PREFILL_CHUNK_DEFAULT, B_MAX);
    fprintf(stderr, "              0 or 1 = one token at a time\n");
    fprintf(stderr, "  -k <string> KV cache type: f32|bf16|f16, default %s (16-bit: half the memory and bandwidth)\n", kv_type_name[KV_TYPE_DEFAULT]);
    fprintf(stderr, "Threads: OMP_NUM_THREADS, default nproc - nproc/8 (Apple Silicon: the performance cores)\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    float temperature = 1.0f;   // 0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "chat";        // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode
    int enable_thinking = 0;    // 1 enables thinking
    int ctx_length = 0;         // context length
    char *out_path = NULL;      // output file for -m logits

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'c') { ctx_length = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else if (argv[i][1] == 'r') { enable_thinking = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { max_new_tokens = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'o') { out_path = argv[i + 1]; }
        else if (argv[i][1] == 'b') { prefill_chunk = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'k') {
            kv_type = -1;
            for (int t = 0; t < 3; t++)
                if (strcmp(argv[i + 1], kv_type_name[t]) == 0) kv_type = t;
            if (kv_type < 0) error_usage();
        }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0) temperature = 0;
    if (topp < 0 || 1.0 < topp) topp = 0.9;
    if ((strcmp(mode, "tokenize") == 0 || strcmp(mode, "logits") == 0) && !prompt) { error_usage(); }
    if ((strcmp(mode, "logits") == 0 || strcmp(mode, "dequant") == 0) && (!out_path || !prompt)) { error_usage(); }
    if (prefill_chunk < 0) prefill_chunk = 0;
    if (prefill_chunk > B_MAX) prefill_chunk = B_MAX;

    if (strcmp(mode, "tokenize") == 0) {
        // only the header is needed (vocab_size): skip mapping and dequantizing the weights
        // (GGUF: the metadata, the tensor data stays unread)
        Config config;
        Tokenizer tokenizer;
        rq_set_threads(1); // (one light process: the tests run many in parallel)
        double t0 = now_s();
        if (file_is_gguf(checkpoint_path)) {
            Gguf g;
            gguf_open(&g, checkpoint_path);
            gguf_config(&g, &config);
            build_tokenizer_gguf(&tokenizer, &g, config.vocab_size, enable_thinking);
            munmap((void *)g.map, g.size);
            free(g.kv);
            free(g.t);
        } else {
            FILE *file = fopen(checkpoint_path, "rb");
            if (!file || fread(&config, sizeof(Config), 1, file) != 1 || config.magic_number != 0x616a6331) {
                fprintf(stderr, "Couldn't read qwen3.c checkpoint header from %s\n", checkpoint_path);
                exit(EXIT_FAILURE);
            }
            fclose(file);
            build_tokenizer(&tokenizer, checkpoint_path, config.vocab_size, enable_thinking);
        }
        tokenize(&tokenizer, prompt, now_s() - t0);
        free_tokenizer(&tokenizer);
        return 0;
    }

    // threads: OMP_NUM_THREADS, else the default nproc - nproc/8 (14 of 16 vCPUs). Measured on the
    // shared 16-vCPU dev VM (~3 CPUs busy with other work; Step 7a): decode 12 ~ 14 > 16 threads
    // (memory-bound: 12 already saturate the bandwidth), prefill 14 > 16 > 12 (~2,580 / ~2,290-2,500 /
    // ~2,420 tok/s at 439 tokens). All 16 work too (the phases tolerate a preempted thread; with
    // OpenMP's barriers they collapsed to ~2-20 tok/s), just not faster with other processes running.
    // Apple Silicon with efficiency cores: the performance cores only (rq_ncpu_perf). M1 Pro, 8 P + 2 E
    // (BENCHMARKS.md "Apple M1 Pro"): 9 or 10 threads instead of 8 decode 5-13% slower, prefill within 3%
    if (rq_env_threads()) {
        rq_set_threads(rq_env_threads());
    } else {
        int np = rq_ncpu(), npc = rq_ncpu_perf();
        rq_set_threads(npc ? npc : np - np / 8);
    }
    rq_config();

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path, ctx_length, prefill_chunk);
    if (strcmp(mode, "dequant") == 0) {
        dump_dequant(&transformer, prompt, out_path);
        free_transformer(&transformer);
        return 0;
    }

    // build the Tokenizer via the tokenizer .bin file (GGUF: from its metadata)
    Tokenizer tokenizer;
    if (transformer.gguf.map)
        build_tokenizer_gguf(&tokenizer, &transformer.gguf, transformer.config.vocab_size, enable_thinking);
    else
        build_tokenizer(&tokenizer, checkpoint_path, transformer.config.vocab_size, enable_thinking);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    if (!prompt)
        printf("hidden_size=%d, intermediate_size=%d, num_hidden_layers=%d, num_attention_heads=%d, num_kv_heads=%d, head_dim=%d, ctx_length=%d, vocab_size=%d, shared_classifier=%d, quantization_block_size=%d\n", transformer.config.dim, transformer.config.hidden_dim, transformer.config.n_layers, transformer.config.n_heads, transformer.config.n_kv_heads, transformer.config.head_dim, transformer.config.seq_len, transformer.config.vocab_size, transformer.config.shared_classifier, transformer.config.group_size);

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, enable_thinking);
    } else if (strcmp(mode, "logits") == 0) {
        dump_logits(&transformer, &tokenizer, prompt, out_path);
    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
#ifdef RUNQ_PROFILE
    double pf_total = 0;
    for (int i = 0; i < PF_N; i++) pf_total += pf_time[i];
    fprintf(stderr, "[profile] forward_batch total %.3f s\n", pf_total);
    for (int i = 0; i < PF_N; i++)
        fprintf(stderr, "[profile]   %-18s %8.3f s  %5.1f%%\n", pf_name[i], pf_time[i], 100 * pf_time[i] / (pf_total > 0 ? pf_total : 1));
#endif
    return 0;
}

