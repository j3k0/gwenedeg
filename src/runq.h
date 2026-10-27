/* runq: Qwen3 (and Qwen3.5-0.8B) inference on the CPU, as a library (the engine of the runq CLI, src/cli.c).
 *
 * Three kinds of objects (README.md, Architecture):
 *   RqModel  the model: hyperparameters, weights (repacked for the kernels at load), tokenizer, chat
 *            templates. Loaded once, read-only afterwards: any number of sequences and compute
 *            contexts may use it. Several models can be loaded at once (a draft and a target model;
 *            tests/api_test.c --two: each gives the bits it gives alone) if they have the same
 *            activation group size, a process-wide setting of the kernels (GGUF 32, .bin 64:
 *            rq_model_load refuses another one while a model is loaded).
 *   RqSeq    a sequence: its own KV cache (capacity and element type chosen per sequence), the
 *            number of positions in it, the tokens at those positions, and an RNG state for sampling.
 *   RqCtx    a compute context: the activation / scratch buffers of the forward passes (decode and
 *            batched prefill, for up to max_batch rows), the threads' synchronization state, the
 *            logits of the last call, profiling. The threads are a process-wide pool (rq_threads_init)
 *            that runs one job at a time: one forward pass (or rq_sample_batch, rq_seq_copy, a load)
 *            at a time per process, so the passes of several contexts (models) alternate on one
 *            calling thread, never concurrently. A context and its sequences belong to one model.
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
// started pool, nor change the count once a compute context exists. Also reads the tuning knobs
// RUNQ_SPIN / RUNQ_SYNC (src/threads.h); returns a warning about them to show, or NULL
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
    double mem;         // the memory check, before any weight is read (rq_model_mem): the estimate must fit in this many
                        // GiB; 0 = in MemAvailable (Linux /proc/meminfo; no check where it is unknown), < 0 = no check
    int kv_bytes;       // bytes per KV cache value of the sequence the check counts (0: 2, f16 / bf16; 4: f32)
    int page_cache;     // the file's page cache after the repack (the loader reads it with pread): 0 = dropped
                        // if the file is >= 1/4 of the RAM, 1 = dropped, -1 = kept
} RqModelOpts;

// A qwen3.c .bin (with its .tokenizer / .template* side files) or a GGUF file (Q8_0, Q4_K_M), by
// magic. NULL on failure, with the reason in err (e.g. "GGUF <path>: tensor ... not found"; a model whose
// estimate does not fit: "... needs ~N GiB ...", RqModelOpts.mem)
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
    int linear_layers;     // Qwen3.5: linear-attention layers, whose recurrent state cannot be rewound (rq_seq_truncate,
                           // rq_seq_checkpoint_at); 0 for Qwen3
} RqModelInfo;
void rq_model_info(const RqModel *m, RqModelInfo *info);

// Memory estimates, in bytes (README.md, Memory). rq_model_load computes them from the tensor directory before
// reading any weight, and checks total against the budget (RqModelOpts.mem); a model loaded without weights
// has none (RQ_ERR_STATE)
typedef struct {
    size_t kv;          // KV cache (both halves, whole attention blocks) and the token array: address space, touched as used
    size_t state;       // Qwen3.5: the recurrent state of the linear-attention layers (0 for Qwen3)
    int checkpoints;    // copies of the state a sequence keeps at most (rq_seq_checkpoint_at; each allocated at its first use)
    size_t total;       // kv + (1 + checkpoints) x state
} RqSeqMem;
typedef struct {
    size_t weights;     // the weights in memory: repacked (or the mapped file a .bin on the generic kernels runs from), norms
    size_t runtime;     // the tokenizer, the file's metadata, a compute context of RQ_BATCH_MAX rows (rq_ctx_new)
    RqSeqMem seq;       // one sequence of the context length (RqModelOpts.ctx_len / kv_bytes)
    size_t total;       // weights + runtime + seq.total: what the check counts
    double budget;      // what it was checked against (bytes; 0: no check)
    int budget_src;     // 0 none, 1 MemAvailable, 2 RqModelOpts.mem
} RqModelMem;
int rq_model_mem(const RqModel *m, RqModelMem *mem);
// the memory of one sequence (as rq_seq_new(m, capacity, kv_type) allocates it, checkpoints included); RQ_ERR_ARG
// as rq_seq_new
int rq_seq_mem(const RqModel *m, int capacity, int kv_type, RqSeqMem *mem);
// MemAvailable of /proc/meminfo in bytes (Linux), or -1 where unknown
double rq_mem_available(void);

// --------------------------------------------------------------------------------------- sequence
typedef struct RqSeq RqSeq;

enum { RQ_KV_F32 = 0, RQ_KV_BF16 = 1, RQ_KV_F16 = 2, RQ_KV_DEFAULT = RQ_KV_F16 };
const char *rq_kv_name(int kv_type);   // "f32", "bf16", "f16"
int rq_kv_parse(const char *name);     // the type, or RQ_ERR_ARG

// capacity: positions (0 = the model's seq_len); the KV cache is address space, touched as used
RqSeq *rq_seq_new(const RqModel *m, int capacity, int kv_type, int *err);
void rq_seq_free(RqSeq *s);
void rq_seq_reset(RqSeq *s);                   // empty (position 0)
// keep positions 0..n-1 (n <= the current length). A model with linear-attention layers (Qwen3.5,
// RqModelInfo.linear_layers) cannot rewind their state: unless n is the current length, the sequence goes back
// to its longest checkpoint at or below n (rq_seq_checkpoint_at), else to 0. Callers read rq_seq_len afterwards
// and prefill the rest again (the results are then those of an uninterrupted run)
int rq_seq_truncate(RqSeq *s, int n);
int rq_seq_keep(const RqSeq *s, int n);        // the length rq_seq_truncate(s, n) would keep (n for Qwen3)
// Qwen3.5: keep the recurrent state at length pos (>= the current length) as a checkpoint, taken by the forward
// pass that reaches it (or now, if the sequence is there); several can be pending, inside one pass or later ones.
// The sequence keeps a set of them (rq_seq_set_checkpoints: 2 by default, ~19 MB each for the 0.8B, ~150 MiB for
// the 27B, allocated at first use); a new one takes a free slot, else the one at the lowest position. Checkpoints
// past a truncation are dropped; pending ones are forgotten by rq_seq_truncate / copy (dst) / reset. E.g. the
// start of the assistant turn, where the next request of a chat diverges. A no-op for Qwen3. RQ_ERR_ARG (pos below
// the length or past the capacity), RQ_ERR_NOMEM
int rq_seq_checkpoint_at(RqSeq *s, int pos);
// Qwen3.5: a checkpoint after each of rows pos..pos+n-1 of the next pass(es) (lengths pos + 1 .. pos + n; pos >= the
// length, 1 <= n <= RQ_CHECKPOINTS_MAX), e.g. the rows of a speculative verify segment, so that rq_seq_truncate
// lands exactly on any of them. They replace the previous call's (taken or pending; those of rq_seq_checkpoint_at
// stay): n + those slots keep them all. A no-op for Qwen3. RQ_ERR_ARG
int rq_seq_checkpoint_rows(RqSeq *s, int pos, int n);
// Qwen3.5: the number of checkpoint slots (1..RQ_CHECKPOINTS_MAX; default 2); fewer than in use: the checkpoints at
// the highest positions stay, the others' buffers are freed. RQ_ERR_ARG
#define RQ_CHECKPOINTS_MAX 32
int rq_seq_set_checkpoints(RqSeq *s, int n);
int rq_seq_len(const RqSeq *s);                // positions in the cache = the next token's position
int rq_seq_capacity(const RqSeq *s);
const int *rq_seq_tokens(const RqSeq *s);      // the tokens at positions 0..len-1
int rq_seq_common_prefix(const RqSeq *s, const int *tokens, int n); // longest common prefix length
// dst := the first n positions of src (KV entries and tokens; n <= rq_seq_len(src), n <= dst's capacity, the
// same model and KV type; dst's RNG unchanged): e.g. a cached prompt prefix shared by several sequences.
// The copied entries are the same bits, so dst continues exactly as src would. Qwen3.5: unless n is src's
// length, as rq_seq_truncate: src's longest checkpoint <= n, else 0 (rq_seq_len(dst) tells)
int rq_seq_copy(RqSeq *dst, const RqSeq *src, int n);
void rq_seq_seed(RqSeq *s, unsigned long long seed);
// the sampling RNG state (xorshift), read / write: a copy of it predicts the sequence's next draws without
// consuming them (rq_rng_f32 on the copy); e.g. sampling another model's logits with this sequence's next draw
// (TODO S10): save that sequence's state, put this one's copy there, rq_sample, restore
unsigned long long *rq_seq_rng(RqSeq *s);
// the next uniform draw in [0, 1) of an RNG state, advancing it: rq_sample's coin (one per sampled token at
// temperature > 0)
float rq_rng_f32(unsigned long long *state);

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
    int top_k;                  // sample among the top_k most likely tokens (then top-p among them); <= 0 or >=
                                // vocab_size: off (the full distribution / top-p as above)
} RqSampling;

// The next token from logits (modified in place when temperature > 0), with the sequence's RNG
// (one draw when temperature > 0). RQ_ERR_ARG if the mask allows no token. The order is exact (the same
// token for the same draw in every build): top-p takes the tokens by decreasing probability, ties by the
// lower id, until their sum exceeds topp; top-k (temperature > 0): the k largest logits (ties: the lower
// id), weights expf((l - max) x (1 / temperature)) summed in that order, top-p as the first prefix of them
// whose sum exceeds topp x the k's sum, then the draw u x that prefix's sum against the running sum
// (src/sampler.c, sample_topk; tests/sampler_ref.py replays it; this path leaves the logits as given)
int rq_sample(RqCtx *c, RqSeq *s, float *logits, const RqSampling *cfg);
// The model file's recommended settings (GGUF general.sampling.temp / top_p / top_k, as llama.cpp's converter
// writes them; e.g. Qwen3.8-27B: 1.0 / 0.95 / 20), for the options a user did not set: the fields the file has
// are written to *cfg (temperature, topp, top_k; the others untouched), and the return value says which:
// RQ_SAMPLING_TEMP | RQ_SAMPLING_TOPP | RQ_SAMPLING_TOPK (0: none; .bin files: none). Values out of range
// (temperature < 0, top_p outside (0, 1], top_k < 0) or of another type are ignored
enum { RQ_SAMPLING_TEMP = 1, RQ_SAMPLING_TOPP = 2, RQ_SAMPLING_TOPK = 4 };
int rq_model_sampling(const RqModel *m, RqSampling *cfg);
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
//   RQ_CHAT_QWEN35    Qwen3.5: thinking on or off per call, off by default in the CLI (as its template);
//                     on: rendered as Thinking-2507's ("<think>\n" opens the reply, answered turns lose it),
//                     off: as the hybrid's. Message texts are trimmed of surrounding whitespace (its template's
//                     |trim)
enum { RQ_CHAT_HYBRID = 0, RQ_CHAT_INSTRUCT = 1, RQ_CHAT_THINKING = 2, RQ_CHAT_QWEN35 = 3 };
const char *rq_chat_name(int variant); // "qwen3", "instruct-2507", "thinking-2507", "qwen3.5""
enum { RQ_ROLE_SYSTEM = 0, RQ_ROLE_USER = 1, RQ_ROLE_ASSISTANT = 2 };
typedef struct { int role; const char *content; } RqMessage;

// a message text as the variant's template renders it (Qwen3.5: without surrounding whitespace, Python's
// str.strip(); the others: as is): its start in text and *len bytes. rq_chat_system / user / render apply it
const char *rq_chat_trim(const RqModel *m, const char *text, size_t *len);
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
// where a chat prompt's recurrent state is worth a checkpoint (rq_seq_checkpoint_at; Qwen3.5): the position of its
// last <|im_start|> (the assistant header: the next request of the conversation diverges after it when the
// reply is rendered from its text, without its reasoning or cut at a stop string), or -1
int rq_chat_checkpoint_pos(const RqModel *m, const int *tokens, int n);
// a whole conversation: an optional system message, then user / assistant alternating, ending with
// a user message (then the assistant header: the prompt to generate from) or an assistant one
int rq_chat_render(const RqModel *m, const RqMessage *msgs, int n, int thinking, RqTokens *out);
// RQ_OK if the templates of this thinking mode are available (else the missing file in err)
int rq_chat_check(const RqModel *m, int thinking, char *err, size_t errlen);

// --------------------------------------------------------------------------------------- generation
// Prefill tokens at the sequence's next positions, then sample and decode until an end token (not
// fed), max_new tokens (> 0) or a full context, as the CLI's chat does. The RNG pattern (temperature
// != 0): one draw per prefilled token but the last (rq_gen_start), then one per sampled token (rq_gen_step:
// every token sampled, fed or not: the end token, a token on_token refused or that found no room too). The
// callbacks (any may be NULL): on_logits before each sample (pos = the position of the token that produced
// them; the hook for equivalence tests: rq_logits_hash of each), on_token for each sampled token that is
// not an end token (return nonzero to stop: it is not fed then)
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

// rq_generate's loop one token at a time, for a loop that feeds the tokens itself, e.g. a speculative one: it
// verifies [last, d1..dk] in one pass (RQ_LOGITS_ALL) and steps through the rows in order while the sampled
// token equals the guess. rq_generate is built on these two calls, so such a loop gets its tokens, RNG draws
// and callbacks by construction:
//   rq_prefill(c, s, tokens, n, RQ_LOGITS_LAST, NULL, &logits);   // (or any chunking: the same bits)
//   RqGen g;
//   rq_gen_start(&g, s, cfg, max_new, cb, n);
//   for (int pos = rq_seq_len(s) - 1;; pos++) {
//       int r = rq_gen_step(&g, c, pos, logits);   // logits: those of the token at pos
//       if (r < 0) error; if (r == RQ_GEN_DONE) break;
//       rq_decode(c, s, g.token, r == RQ_GEN_LAST ? NULL : &logits);   // feed it at pos + 1
//       if (r == RQ_GEN_LAST) break;
//   }
// rq_gen_step: on_logits(pos), rq_sample (one draw at temperature > 0), then the checks in this order: an end
// token (RQ_GEN_DONE, stop RQ_STOP_END), on_token asks to stop (DONE, RQ_STOP_CALLBACK), no room for the token
// at pos + 1 (DONE, RQ_STOP_CONTEXT), the max_new-th token (RQ_GEN_LAST: feed it, its logits are not needed;
// stop RQ_STOP_LIMIT); else RQ_GEN_NEXT: feed g.token, then step on its logits. A token is not fed after
// RQ_GEN_DONE. Errors (< 0): those of rq_sample. The sequence's length is not read (pos is), so the rows of a
// verify pass that are already in the cache can be stepped through; the callbacks see what rq_generate's would
enum { RQ_GEN_DONE = 0, RQ_GEN_NEXT = 1, RQ_GEN_LAST = 2 };
typedef struct {
    RqSeq *s;
    const RqSampling *cfg;
    const RqCallbacks *cb;
    int max_new;  // as rq_generate's (0: no limit)
    int n_gen;    // tokens returned to feed so far (RQ_GEN_NEXT / LAST)
    int token;    // the last sampled token
    int stop;     // RQ_STOP_* once done, else -1
} RqGen;
// the state for a reply after n prompt tokens were prefilled: RNG draws for them (n - 1 at temperature != 0)
void rq_gen_start(RqGen *g, RqSeq *s, const RqSampling *cfg, int max_new, const RqCallbacks *cb, int n);
int rq_gen_step(RqGen *g, RqCtx *c, int pos, float *logits);

// a hash of a logits vector's bits (FNV-1a over the bytes; n floats): equal vectors, equal hashes. For tests
// comparing generation paths through on_logits (e.g. speculative == plain decoding)
unsigned long long rq_logits_hash(const float *logits, int n);

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
