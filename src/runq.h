/* runq: Qwen3 inference on the CPU, as a library (the engine of the runq CLI, src/cli.c).
 *
 * Three kinds of objects (README.md, Architecture):
 *   RqModel  the model: hyperparameters, weights (repacked for the kernels at load), tokenizer, chat
 *            templates. Loaded once, read-only afterwards: any number of sequences and compute
 *            contexts may use it (one model per process for now: the activation group size is a
 *            process-wide setting of the kernels).
 *   RqSeq    a sequence: its own KV cache (capacity and element type chosen per sequence), the
 *            number of positions in it, the tokens at those positions, and an RNG state for sampling.
 *   RqCtx    a compute context: the activation / scratch buffers of the forward passes (decode and
 *            batched prefill, for up to max_batch rows), the threads' synchronization state, the
 *            logits of the last call, profiling. One forward pass at a time per context; the threads
 *            are a process-wide pool (rq_threads_init).
 * The forward passes take the sequence(s) explicitly: rq_decode (one token of one sequence),
 * rq_prefill (tokens of one sequence, in chunks), rq_forward_batch (one chunk of rows from several
 * sequences, each row at its own sequence's next position).
 *
 * Results are bit-identical for any thread count, chunk size and batch composition: a sequence's
 * logits do not depend on what it is batched with (tests/api_test.c). Nothing here prints or exits;
 * errors are returned as RQ_ERR_* codes (negative), with a message where it helps.
 */
#ifndef RUNQ_H
#define RUNQ_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    RQ_OK = 0,
    RQ_ERR_ARG = -1,       // invalid argument (token id, length, mode, unknown name, ...)
    RQ_ERR_NOMEM = -2,     // allocation failed
    RQ_ERR_CONTEXT = -3,   // the tokens do not fit in the sequence's capacity (context length)
    RQ_ERR_STATE = -4,     // not possible on this object (e.g. a model loaded without weights)
    RQ_ERR_LOAD = -5,      // model file missing, unreadable, corrupt or unsupported
};
const char *rq_strerror(int err);
double rq_now(void); // monotonic clock, seconds

// ---------------------------------------------------------------------------------------- threads
// Threads per job of the process-wide pool: n > 0, else OMP_NUM_THREADS, else nproc - nproc/8
// (Apple Silicon: the performance cores). Call before loading a model; later calls do not resize a
// started pool. Also reads the tuning knobs RUNQ_SPIN / RUNQ_SYNC (src/threads.h); returns a warning
// about them to show, or NULL
const char *rq_threads_init(int n);
int rq_threads(void);

// ------------------------------------------------------------------------------------------ model
typedef struct RqModel RqModel;

// The default context length cap (RqModelOpts.ctx_len 0): the model's maximum, at most this many positions
// (Qwen3's 40,960; the 2507 models' 262,144 would reserve 38.6 GB of KV cache address space per sequence)
#define RQ_CTX_DEFAULT_MAX 40960

typedef struct {
    int ctx_len;        // context length cap (0 = the model's maximum, at most RQ_CTX_DEFAULT_MAX; larger
                        // values up to the model's maximum are taken as given): the default sequence capacity
    int tokenizer_only; // only the header / metadata and the tokenizer (no weights, no forward passes)
    int no_tokenizer;   // only the weights (rq_dequant_rows)
} RqModelOpts;

// A qwen3.c .bin (with its .tokenizer / .template* side files) or a GGUF file (Q8_0, Q4_K_M), by
// magic. NULL on failure, with the reason in err (e.g. "GGUF <path>: tensor ... not found")
RqModel *rq_model_load(const char *path, const RqModelOpts *opts, char *err, size_t errlen);
void rq_model_free(RqModel *m);

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, head_dim, vocab_size;
    int seq_len;           // context length (the ctx_len cap applied)
    int ctx_max;           // the model's context length (metadata), before the cap
    int group_size;        // activation quantization group (64 .bin, 32 GGUF)
    int shared_classifier; // classifier tied to the token embeddings
    int gguf;              // loaded from a GGUF file
    int bos, eos;          // generation ends at either
    const char *warning;   // a load-time warning to show (e.g. weights not packable), or NULL
    int chat;              // chat template variant, RQ_CHAT_* (below)
    float rope_theta;      // RoPE base
} RqModelInfo;
void rq_model_info(const RqModel *m, RqModelInfo *info);

// --------------------------------------------------------------------------------------- sequence
typedef struct RqSeq RqSeq;

enum { RQ_KV_F32 = 0, RQ_KV_BF16 = 1, RQ_KV_F16 = 2, RQ_KV_DEFAULT = RQ_KV_F16 };
const char *rq_kv_name(int kv_type);   // "f32", "bf16", "f16"
int rq_kv_parse(const char *name);     // the type, or RQ_ERR_ARG

// capacity: positions (0 = the model's seq_len); the KV cache is address space, touched as used
RqSeq *rq_seq_new(const RqModel *m, int capacity, int kv_type, int *err);
void rq_seq_free(RqSeq *s);
void rq_seq_reset(RqSeq *s);                   // empty (position 0)
int rq_seq_truncate(RqSeq *s, int n);          // keep positions 0..n-1 (n <= the current length)
int rq_seq_len(const RqSeq *s);                // positions in the cache = the next token's position
int rq_seq_capacity(const RqSeq *s);
const int *rq_seq_tokens(const RqSeq *s);      // the tokens at positions 0..len-1
int rq_seq_common_prefix(const RqSeq *s, const int *tokens, int n); // longest common prefix length
// dst := the first n positions of src (KV entries and tokens; n <= rq_seq_len(src), n <= dst's capacity, the
// same model and KV type; dst's RNG unchanged): e.g. a cached prompt prefix shared by several sequences.
// The copied entries are the same bits, so dst continues exactly as src would
int rq_seq_copy(RqSeq *dst, const RqSeq *src, int n);
void rq_seq_seed(RqSeq *s, unsigned long long seed);
unsigned long long *rq_seq_rng(RqSeq *s);      // the sampling RNG state (xorshift)

// ---------------------------------------------------------------------------------- compute context
typedef struct RqCtx RqCtx;

#define RQ_BATCH_MAX 512 // rows per batched forward pass (prefill chunk), at most

// max_batch: rows per batched pass (<= RQ_BATCH_MAX; 0 or 1: no batched buffers, prefill token by
// token through the decode path). max_ctx: positions the attention scratch is sized for (0 = the
// model's seq_len; a longer sequence grows it at its first use)
RqCtx *rq_ctx_new(const RqModel *m, int max_batch, int max_ctx, int *err);
void rq_ctx_free(RqCtx *c);
int rq_ctx_max_batch(const RqCtx *c);
const char *rq_ctx_error(const RqCtx *c); // details of the last error ("" if none)
// -DRUNQ_PROFILE builds: wall time per batched-prefill phase type, summed over the calls; returns the
// number of entries (0 otherwise)
int rq_ctx_profile(const RqCtx *c, const char **names, double *seconds, int max);

// ---------------------------------------------------------------------------------- forward passes
enum { RQ_LOGITS_NONE = 0, RQ_LOGITS_LAST = 1, RQ_LOGITS_ALL = 2 };

// One token at the sequence's next position (the decode path: GEMV, attention split over threads per
// 256 positions). logits != NULL: the next-token logits, *logits = the context's buffer (vocab_size
// floats, valid until the next call); NULL: the KV cache only
int rq_decode(RqCtx *c, RqSeq *s, int token, float **logits);

// n >= 1 tokens at the sequence's next positions, in chunks of max_batch rows (max_batch <= 1: token
// by token through rq_decode). mode RQ_LOGITS_LAST: *logits = the last row's logits (the context's
// buffer); RQ_LOGITS_ALL: every row's logits into out (n x vocab_size floats), *logits = out
int rq_prefill(RqCtx *c, RqSeq *s, const int *tokens, int n, int mode, float *out, float **logits);

// One batched pass over the rows of several sequences (Step 10's multi-sequence decode, continuous
// batching): segment i appends its n tokens to seq at its next positions; a sequence at most once;
// at most max_batch rows in total. Each segment selects its logits rows (none, the last, all); the
// selected rows' logits go to out in row order (rows x vocab_size floats); with exactly one row
// selected and out == NULL, to the context's buffer. *logits (if not NULL) = where they went
typedef struct {
    RqSeq *seq;
    const int *tokens;
    int n;
    int logits; // RQ_LOGITS_NONE / LAST / ALL
} RqSegment;
int rq_forward_batch(RqCtx *c, const RqSegment *seg, int nseg, float *out, float **logits);

// ----------------------------------------------------------------------------------------- sampling
typedef struct {
    float temperature;          // 0 = greedy (argmax)
    float topp;                 // nucleus sampling in (0, 1); else the full distribution
    const unsigned char *allow; // optional token mask (vocab_size entries, nonzero = allowed): constrained decoding
} RqSampling;

// The next token from logits (modified in place when temperature > 0), with the sequence's RNG
// (one draw when temperature > 0). RQ_ERR_ARG if the mask allows no token
int rq_sample(RqCtx *c, RqSeq *s, float *logits, const RqSampling *cfg);
// The same for n rows at once, e.g. the logits of an rq_forward_batch pass: row i (logits + i * vocab_size,
// modified in place as by rq_sample) with seqs[i]'s RNG and cfg[i] (ncfg == n) or cfg[0] (ncfg == 1), on the
// thread pool; tokens[i] = rq_sample's result for that row (a token, or RQ_ERR_ARG). Returns RQ_OK, or an
// error for the call (RQ_ERR_ARG: n < 1 or ncfg not 1 / n; RQ_ERR_NOMEM)
int rq_sample_batch(RqCtx *c, RqSeq *const *seqs, float *logits, int n, const RqSampling *cfg, int ncfg, int *tokens);

// ------------------------------------------------------------------------------- tokens, tokenizer
typedef struct { int *v; int n, cap; } RqTokens; // growable token array (zero-initialize; rq_tokens_free)
int rq_tokens_push(RqTokens *t, const int *x, int n);
void rq_tokens_free(RqTokens *t);

// text (len bytes, len < 0: NUL-terminated) -> token ids appended to out (raw text, no template;
// added tokens such as <|im_start|> are recognized, NFC normalization as HF's tokenizer)
int rq_tokenize(const RqModel *m, const char *text, long len, RqTokens *out);
const char *rq_token_bytes(const RqModel *m, int token, int *len); // raw bytes (NUL-terminated too)
int rq_is_end_token(const RqModel *m, int token);                   // eos or bos
int rq_token_is_added(const RqModel *m, int token);                 // an added token (<|im_end|>, <think>, ...)

// Incremental detokenization for streaming: rq_detok_push appends a token's bytes and writes the
// part that ends at a UTF-8 character boundary (a character split over tokens comes out whole once
// complete; invalid bytes pass through); rq_detok_flush writes what is left. Return the bytes
// written, or RQ_ERR_ARG if cap is too small (the token's byte length + 8 always suffices)
typedef struct { unsigned char b[8]; int n; } RqDetok;
int rq_detok_push(const RqModel *m, RqDetok *d, int token, char *out, size_t cap);
int rq_detok_flush(RqDetok *d, char *out, size_t cap);

// --------------------------------------------------------------------------------------------- chat
// Qwen3's chat template (ChatML), as the CLI's chat keeps the conversation in the KV cache:
//   system block   "<|im_start|>system\n" SYS "<|im_end|>\n"
//   user turn      "<|im_start|>user\n" U "<|im_end|>\n<|im_start|>assistant\n"
//                  + "<think>\n\n</think>\n\n" with thinking off
//   reply          the assistant's text; with thinking on in the history form of HF's template (the
//                  part after the last </think>, leading newlines stripped)
//   end            "<|im_end|>\n" after every message
// (.bin files: the templates come from the .template* side files, GGUF: built in, identical)
// Variants (RqModelInfo.chat, detected from a GGUF's tokenizer.chat_template; .bin: hybrid):
//   RQ_CHAT_HYBRID    Qwen3 (0.6B .. 8B): as above, thinking on or off per call
//   RQ_CHAT_INSTRUCT  Qwen3-*-Instruct-2507: no think block ever; thinking is ignored (0), replies verbatim
//                     in the history
//   RQ_CHAT_THINKING  Qwen3-*-Thinking-2507: the user turn ends with "<think>\n" (the reply starts inside
//                     the reasoning, it has no opening tag; answered turns lose it: rq_chat_user_history);
//                     history as the hybrid with thinking. thinking 0
//                     gives the hybrid's empty think block instead, which is not in this model's template
//                     (runq serve uses it only for a constrained reply)
enum { RQ_CHAT_HYBRID = 0, RQ_CHAT_INSTRUCT = 1, RQ_CHAT_THINKING = 2 };
const char *rq_chat_name(int variant); // "qwen3", "instruct-2507", "thinking-2507"
enum { RQ_ROLE_SYSTEM = 0, RQ_ROLE_USER = 1, RQ_ROLE_ASSISTANT = 2 };
typedef struct { int role; const char *content; } RqMessage;

int rq_chat_system(const RqModel *m, const char *text, int thinking, RqTokens *out);
int rq_chat_user(const RqModel *m, const char *text, int thinking, RqTokens *out);
int rq_chat_end(const RqModel *m, RqTokens *out);
// a user turn as it stays in the history once answered: rq_chat_user's piece without Thinking-2507's
// "<think>\n" (it opened the reasoning, which the history drops; HF's template renders earlier turns so).
// The same as rq_chat_user for the other variants (the hybrid's empty think block stays, as the CLI keeps it)
int rq_chat_user_history(const RqModel *m, const char *text, int thinking, RqTokens *out);
// the generation prompt alone: "<|im_start|>assistant\n" + what the variant adds for this thinking mode (the
// end of rq_chat_user's piece; e.g. after a message sequence rq_chat_user cannot express)
int rq_chat_assistant(const RqModel *m, int thinking, RqTokens *out);
// the history form of a reply generated as tokens (thinking: without its reasoning block)
int rq_chat_reply_history(const RqModel *m, const int *reply, int n, int thinking, RqTokens *out);
// a whole conversation: an optional system message, then user / assistant alternating, ending with
// a user message (then the assistant header: the prompt to generate from) or an assistant one
int rq_chat_render(const RqModel *m, const RqMessage *msgs, int n, int thinking, RqTokens *out);
// RQ_OK if the templates of this thinking mode are available (else the missing file in err)
int rq_chat_check(const RqModel *m, int thinking, char *err, size_t errlen);

// --------------------------------------------------------------------------------------- generation
// Prefill tokens at the sequence's next positions, then sample and decode until an end token (not
// fed), max_new tokens (> 0) or a full context, as the CLI's chat does. With temperature > 0 the RNG
// draws once per prefilled token but the last, then once per sampled token. The callbacks (any may be
// NULL): on_logits before each sample (pos = the position of the token that produced them), on_token
// for each sampled token that is not an end token (return nonzero to stop: it is not fed then)
enum { RQ_STOP_END = 0, RQ_STOP_LIMIT = 1, RQ_STOP_CONTEXT = 2, RQ_STOP_CALLBACK = 3 };
typedef struct {
    int (*on_token)(void *user, int token);
    void (*on_logits)(void *user, int pos, const float *logits);
    void *user;
} RqCallbacks;
typedef struct {
    int n_prompt, n_gen;  // tokens prefilled; tokens generated and fed
    int stop;             // RQ_STOP_*
    double t_prefill, t_gen; // seconds
} RqGenStats;
int rq_generate(RqCtx *c, RqSeq *s, const int *tokens, int n, const RqSampling *cfg, int max_new,
                const RqCallbacks *cb, RqGenStats *st);

// ----------------------------------------------------------------------------------------- test hooks
// rows r0..r0+nr-1 of the matrix `name` (llama.cpp tensor name; GGUF only) dequantized from the file's
// blocks into rows (nr x *ncols), and for Q4_K / Q6_K also from the repacked layout into packed (if
// not NULL; *npacked = 1). rows / packed: malloc'd by the call
int rq_dequant_rows(const RqModel *m, const char *name, size_t name_len, long r0, long nr, float **rows,
                    float **packed, int *ncols, int *npacked, char *err, size_t errlen);

#ifdef __cplusplus
}
#endif
#endif // RUNQ_H
