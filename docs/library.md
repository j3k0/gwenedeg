# The C library and the source map

For people embedding the engine or reading the source. The engine is a C library, `src/runq.h`; the CLI
(`src/cli.c`) and the server use only this API. The library has no dependencies, never prints or exits, and
reports errors as `RQ_ERR_*` codes with messages. This text came from the README's Library API section (the
"Architecture" that `src/runq.h` and the Makefile refer to).

The engine is a C library, `src/runq.h`; the CLI (`src/cli.c`) and the server use only this API. It
has three kinds of objects:

- **RqModel** (`rq_model_load`): hyperparameters, weights (repacked for the kernels at load), tokenizer,
  and chat templates. It is read-only after loading and shared by any number of sequences.
- **RqSeq** (`rq_seq_new`): one sequence with its own KV cache (capacity and type per sequence), tokens
  and sampling RNG. `rq_seq_truncate`, `rq_seq_common_prefix` and `rq_seq_copy` handle prefix reuse;
  `rq_seq_save` / `rq_seq_load` write a sequence to a file and read it back (whole or a prefix; the same bits;
  an MTP view's kept hidden state too when it is at its last row and the file is loaded whole, TODO S9c.2),
  `rq_seq_file_info` reads a saved one's tokens.
- **RqCtx** (`rq_ctx_new`): a compute context (activation buffers for up to `max_batch` rows, logits).
  The threads are one process-wide pool (`rq_threads_init`) running one pass at a time: the passes of
  several contexts (e.g. two models with the same activation group size) alternate on one calling thread.

The forward passes are `rq_decode` (one token), `rq_prefill` (a sequence's tokens in chunks) and
`rq_forward_batch` (one pass over rows of several sequences, each at its own position: prefill chunks
next to decode rows; short segments, e.g. decode rows or a speculative verify pass of a few rows of one sequence,
take the decode-rows worker: 4 rows of one sequence cost ~1.2-1.3x a decode step, docs/notes/qwen38-speculative.md). On top of them sit `rq_sample` / `rq_sample_batch` (temperature, top-p, an optional
token mask), `rq_generate` (the chat loop with per-token callbacks; built on `rq_gen_start` / `rq_gen_step`, one
token at a time, for loops that feed the tokens themselves), `rq_generate_spec` (the same with a draft model:
speculative decoding, the output exactly `rq_generate`'s; `rq_spec_run` is its loop alone, `rq_prefill_spec` the prompt
with the draft following; `rq_mtp_load` makes a qwen35 file's MTP block a model of its own, a view sharing the target's
weights and tokenizer, fed the target's final hidden states from `RqSegment.hidden` through its `RqSegment.h_in`: as the
draft of `rq_generate_spec` it follows the target in lockstep, TODO S9; a caller that batches the view's rows itself
(runq serve, TODO S9c) uses `rq_decode_hidden` (`rq_decode` with the hidden state and a view's h input) and
`rq_mtp_hin` / `rq_mtp_final` / `rq_mtp_keep` / `rq_mtp_kept` (the view's h inputs from the target's hidden rows or the
state kept with it, its final rows, the kept state: the loop's own rule); `rq_spec_prior` is auto's prior per draft;
`RqSpecOpts.guess_hook` replaces the guesses in tests), `rq_tokenize`, `rq_detok_push`
(incremental, UTF-8 safe) and `rq_chat_render`. Errors come back as `RQ_ERR_*` codes and messages; the
library never prints or exits. `RqModelOpts.mem` / `.page_cache` set the load's memory check and page-cache policy
([usage.md](usage.md#memory-check-at-load)); `rq_model_mem` and `rq_seq_mem` return the estimates (the server sizes its slot pool with the latter;
`RqModelMem.read`, the weights a one-row pass reads, is what `rq_pass_cost_default` prices: a model's weights, an MTP
view's block plus the target's classifier rows it computes).

```c
rq_threads_init(0);
RqModel *m = rq_model_load("Qwen3-0.6B-Q8_0.gguf", NULL, err, sizeof(err));
RqCtx *c = rq_ctx_new(m, 512, 0, &e);
RqSeq *s = rq_seq_new(m, 4096, RQ_KV_DEFAULT, &e);
RqMessage msg[] = { { RQ_ROLE_USER, "Why is the sky blue?" } };
RqTokens t = { 0 };
rq_chat_render(m, msg, 1, 0, &t);
RqSampling cfg = { 0.7f, 0.9f, NULL };
RqCallbacks cb = { on_token, NULL, user };   // on_token: e.g. rq_detok_push + write
rq_generate(c, s, t.v, t.n, &cfg, 256, &cb, &stats);

// batched decode: one row per sequence, every weight read shared by the N rows
RqSegment sg[N];
for (int i = 0; i < N; i++) sg[i] = (RqSegment) { seq[i], &tok[i], 1, RQ_LOGITS_LAST };
rq_forward_batch(c, sg, N, logits, NULL);         // logits: N x vocab_size floats
rq_sample_batch(c, seq, logits, N, cfg, N, tok);  // the next token of each, per-row settings
```

| file | contents |
|---|---|
| `src/runq.h` | the public API |
| `src/model.c` | RqModel: .bin / GGUF loading, weight conversion and repacking, chat templates |
| `src/gguf.c` | the GGUF reader (untrusted input: every count, length and offset checked) |
| `src/forward.c` | RqSeq, RqCtx and all numeric code of the forward passes: decode, batched pass, attention, matmul dispatch, quantization, norms, RoPE, KV cache |
| `src/tokenizer.c` | BPE tokenizer matching HF (added tokens, NFC, Qwen2 pre-tokenizer), detokenization |
| `src/sampler.c`, `src/chat.c` | sampling; chat template rendering and the generation loop |
| `src/spec.c` | speculative decoding: the draft's (or the prompt lookup's) guesses verified in one pass of the model (`rq_generate_spec`) |
| `src/threads.c`, `src/threads.h` | the thread pool and its phase synchronization |
| `src/kernels_avx512.h`, `src/kernels_neon.h`, `src/kquants.h` | SIMD kernels; the K-quant formats and the exact arithmetic every kernel follows |
| `src/iquants.h`, `src/iq_grids.h` | Q2_K and the i-quants (IQ1_S .. IQ4_XS): formats, repacked layouts, arithmetic, generic and AVX-512 kernels; the codebooks (generated by `tools/gen_iq_grids.py`) |
| `src/unicode_tables.h`, `src/unicode_nfc.h` | Unicode classes and NFC tables, generated by `tools/gen_unicode_tables.py` |
| `src/cli.c` | the `runq` command line and its test hooks |
| `src/serve.c`, `src/engine.c`, `src/http.c`, `src/constrain.c`, `src/tools.c`, `src/json.c` | `runq serve` (not part of the library): OpenAI layer, scheduler and slot pool, HTTP/1.1, constrained decoding, tool calls, JSON |

Every kernel family follows one arithmetic definition: int8 activations in groups of 32 (GGUF) or 64
(`.bin`), with the float folds in a fixed order and without FMA contraction. The kernels are
interchangeable bit for bit, but logits differ slightly from fp32 engines; the tests bound the
differences against HF. The activation group size is a process-wide kernel setting: load one model,
or models with the same group size, per process. The changes relative to qwen3.c, step by step, the design notes
and the experiments that were not kept are in [DEVELOPMENT.md](DEVELOPMENT.md); the full development log
with every measurement is in the git history ([History](DEVELOPMENT.md#history)). Open work:
[TODO.md](../TODO.md), [IDEAS.md](../IDEAS.md).
