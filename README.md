# Gwenedeg

Gwenedeg runs Qwen3 dense language models, and the dense Qwen3.5 models (0.8B and 4B tested), on the CPU, in plain C with no external dependencies. It is a small
inference library (`src/runq.h`), a command line, and an OpenAI-compatible HTTP server. **The binary is
called `runq`**, as in qwen3.c, the project it grew from. The library keeps the name too (`runq.h`,
`rq_` prefix).

- **Models**: Qwen3-0.6B, 1.7B, 4B and 8B, plus Qwen3-4B-Instruct-2507 / Thinking-2507, and Qwen3.5-0.8B and 4B (the
  hybrid architecture: Gated DeltaNet linear attention + gated attention; 0.8B Q8_0 and Q4_K_M, 4B Q8_0). It loads GGUF
  **Q8_0**, **Q4_K_M** and **Q5_K_M** files directly (weights, tokenizer and hyperparameters come from the file; chat templates are built
  in), and also qwen3.c's Q8_0 `.bin` export.
- **Kernels**: AVX-512 VNNI on x86-64, NEON (dot product) on AArch64 / Apple Silicon, and generic C
  everywhere else. The SIMD kernels give bit-identical results to the generic ones.
- **Deterministic**: logits are bit-identical whatever the thread count, the prefill chunk size or the
  batch composition. Batched prefill equals token-by-token, and a batched decode row equals decoding
  that sequence alone. The test suite checks all of this.
- **Fast prefill and decode**: int8 GEMM / GEMV on weights repacked at load, a tiled f16 KV cache, its
  own thread pool. On a 16-vCPU AMD EPYC Genoa VM, Qwen3-0.6B prefills ~2,000-2,700 tok/s and decodes
  ~200-350 tok/s (below).
- **Server** (`runq serve`): OpenAI chat completions with streaming and continuous batching (requests
  share each forward pass), prefix KV-cache reuse across requests, and constrained decoding for
  JSON schemas with a finite set of outputs. That last feature makes it a fast LLM judge for
  [LiteLLM](https://github.com/BerriAI/litellm)'s complexity auto-router.
- **Exact tokenizer**: byte-level BPE matching Hugging Face `tokenizers` on Qwen's tokenizer, including
  NFC normalization (tested against HF).

Platforms: Linux (gcc) is the main platform. macOS on Apple Silicon (Apple clang) builds and runs with
NEON kernels. The Windows code paths inherited from qwen3.c are kept but untested, and the server is
POSIX only.

## Quick start

Requirements: `make` and gcc (Linux) or Xcode's command-line tools (macOS). Python 3 (standard
library only) runs the model download script.

```sh
make                                      # -> ./runq   (make generic: ./runq-generic, make debug: ./runq-debug)
scripts/download_models.py q8gguf         # models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf (639 MB), pinned + checksummed
G=models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf

./runq $G -c 4096                         # interactive chat: one message per line, empty line / EOF ends
./runq $G -c 4096 -t 0 -n 200 -i "Why is the sky blue?"     # one turn, greedy, at most 200 tokens
./runq $G -c 4096 -r 1                    # with thinking (Qwen3's reasoning mode)
./runq serve $G                           # OpenAI-compatible server on http://127.0.0.1:8080/v1

curl -s localhost:8080/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Hi!"}], "max_tokens": 32}'
```

`scripts/download_models.py --list` shows every downloadable set (table below). Files go to
`models/` (or `--dest DIR`, `$GWENEDEG_MODELS`). Each file is checked by sha256 or git blob hash at a
pinned Hugging Face revision.

CLI options (`./runq` without arguments prints them all):

| option | |
|---|---|
| `-c N` | context size in tokens; default: the model's maximum, at most 40,960. Pass it: the KV cache is reserved for the whole context (as address space, touched only as used) |
| `-t T`, `-p P`, `-s S` | temperature (default 1.0, 0 = greedy), top-p (0.9), seed |
| `-n N` | max new tokens per answer (default unlimited) |
| `-i TEXT`, `-y TEXT` | a single prompt (one turn, then exit); the system prompt |
| `-r 0\|1` | thinking off (default) / on |
| `-m MODE` | `chat` (default), `generate` (raw completion), `tokenize`; test hooks `logits`, `dequant`, `render` |
| `-k f16\|bf16\|f32` | KV cache type, default f16 (half the memory of f32, as accurate) |
| `-b N` | prefill chunk size, default 512 (`-b 1`: token by token) |
| `OMP_NUM_THREADS` | threads; default nproc - nproc/8 (Apple Silicon: the performance cores) |

Chat keeps the whole conversation in the KV cache, so each turn prefills only the new tokens. Earlier
answers drop their `<think>` block, as Qwen3's template does. When the context is full, the oldest turns
are dropped (the system prompt is kept) and the rest is re-prefilled. A `[timing]` line on stderr gives
prefill and decode speed. Details: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## Models

| set (`download_models.py`) | file | size | notes |
|---|---|---|---|
| `q8gguf` | `Qwen3-0.6B-Q8_0.gguf` (unsloth) | 639 MB | closest to the fp32 model, best for prompt-heavy work |
| `q4km` | `Qwen3-0.6B-Q4_K_M.gguf` (unsloth) | 397 MB | ~1.5x faster decode, ~40% less RAM, +4.9% perplexity |
| `q5km` | `Qwen3-0.6B-Q5_K_M.gguf` (unsloth) | 444 MB | between the two: 1-cos vs HF 5.4e-3 (Q8_0 6.3e-4, Q4_K_M 1.6e-2) |
| `1.7b-q8`, `1.7b-q4km` | `Qwen3-1.7B-Q8_0.gguf`, `-Q4_K_M.gguf` | 1,834 / 1,107 MB | 1.7B Q8_0: the smallest good LiteLLM judge |
| `4b-q8`, `4b-q4km` | `Qwen3-4B-Q8_0.gguf`, `-Q4_K_M.gguf` | 4,280 / 2,497 MB | |
| `4b-2507i-q4km` | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | 2,497 MB | the best 4B for chat (never thinks) |
| `4b-2507t-q4km` | `Qwen3-4B-Thinking-2507-Q4_K_M.gguf` | 2,497 MB | always thinks |
| `8b-q4km` | `Qwen3-8B-Q4_K_M.gguf` | 5,028 MB | quality over speed |
| `3.5-0.8b-q8` | `Qwen3.5-0.8B-Q8_0.gguf` (unsloth) | 812 MB | Qwen3.5: hybrid layers, 248k vocabulary, thinking off by default ([docs/qwen35.md](docs/qwen35.md)) |
| `3.5-0.8b-q4km` | `Qwen3.5-0.8B-Q4_K_M.gguf` (unsloth) | 533 MB | Q4_K / Q5_K / Q6_K / Q8_0 mix: decode ~1.27x the Q8_0 file (188 vs 148 tok/s), prefill ~0.88x, peak RSS 662 vs 963 MB; vs HF fp32 pooled cos 0.9946, top-1 57/62 |
| `3.5-4b-q8` | `Qwen3.5-4B-Q8_0.gguf` (unsloth) | 4,482 MB | Qwen3.5-4B: 32 value heads on 16 key heads in its linear layers; as fast as Qwen3-4B Q8_0, perplexity 10.59 (Qwen3-4B: 15.65) |
| `hf`, `1.7b-hf`, `3.5-0.8b-hf`, `3.5-4b-hf` | `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-4B` (bf16 safetensors) | 1.5 / 4.1 / 1.8 / 9.3 GB | for the `.bin` export and the test goldens |

Measured on the 16-vCPU AMD EPYC Genoa VM (shared, noisy: best of 2, interleaved, 2026-10-26; prefill of
451 tokens, decode of 128 tokens after a short prompt, perplexity on the 4,142-token
`bench/ppl_beagle_ch1.txt`):

| file | peak RSS (1,500 tok, `-c 4096`) | prefill, 8 / 12 threads | decode, 8 / 12 threads | 8 sequences batched, 8 / 12 | perplexity |
|---|---|---|---|---|---|
| Qwen3-0.6B Q8_0 | 0.93 GB | 1,813 / 2,287 tok/s | 207 / 165 | 814 / 910 | 27.60 (HF fp32 27.57) |
| Qwen3-0.6B Q4_K_M | 0.67 GB | 1,721 / 2,338 | 278 / 373 | 1,005 / 1,096 | 28.95 |
| Qwen3-1.7B Q8_0 | 2.1 GB | 790 / 1,013 | 78 / 78 | 381 / 458 | 18.90 (HF 18.84) |
| Qwen3-1.7B Q4_K_M | 1.4 GB | 680 / 943 | 101 / 112 | 466 / 529 | 19.09 |
| Qwen3-4B Q8_0 | 4.7 GB | 313 / 412 | 29 / 31 | 157 / 201 | 15.65 |
| Qwen3-4B Q4_K_M | 2.8 GB | 276 / 385 | 47 / 51 | 177 / 226 | 16.26 |
| Qwen3-4B-Instruct-2507 Q4_K_M | 2.8 GB | 284 / 401 | 56 / 56 | 173 / 237 | 11.86 |
| Qwen3-4B-Thinking-2507 Q4_K_M | 2.8 GB | 280 / 375 | 51 / 58 | 179 / 233 | 13.61 |
| Qwen3-8B Q4_K_M | 4.9 GB | 158 / 219 | 30 / 30 | 91 / 119 | 11.53 |
| Qwen3.5-0.8B Q8_0 (\*) | 0.96 GB | 1,506 / 2,039 | 141 / 182 | | 18.86 (512-token windows: 22.729, HF 22.728) |
| Qwen3.5-4B Q8_0 (\*\*) | 4.8 GB | 308 / 443 | 31 / 30 | | 10.59 (512-token windows: 12.645) |

(\*) Another session: prefill 2026-10-27 (`scripts/bench_models.py`, 451 tokens; Qwen3-0.6B Q8_0 alongside: 1,832 /
2,511), decode 2026-10-26 (0.6B alongside: 209 / 230). Its decode barely slows down with the context (160 / 174 tok/s
after 1,500 tokens): only 6 of its 24 layers have a KV cache, at 12 KB per token. Prefill is ~0.8x the 0.6B's (it was
~0.64x before the linear attention's recurrence was blocked and tuned, docs/qwen35.md, Performance).
(\*\*) 2026-10-27, `scripts/bench_models.py -r 3`, load 13-15 (with the run's own threads); Qwen3-4B Q8_0 alongside:
314 / 440 and 31 / 31, the same within noise (after 1,500 tokens 26 / 32 vs 32 / 32).

Decode is memory-bound and moves with the machine's load (8B ranged 12-30 tok/s); startup takes 0.1-0.2 s
up to 1.7B, ~0.5 s for 4B and ~1 s for 8B. Q4_K_M is the better trade at 4B / 8B (decode ~1.6x Q8_0 at
~60% of the RAM). Q8_0 is the more accurate one: at 0.6B, Q4_K_M has ~25x the logits error and answers
"Sydney" for the capital of Australia.

**Chat templates** are detected from the file's `tokenizer.chat_template`, which is not interpreted
(Qwen3's three variants and Qwen3.5's are built in). The hybrid Qwen3 template (0.6B to 8B) follows `-r 0` / `-r 1`.
Instruct-2507 never thinks, and Thinking-2507 always does; `-r` is ignored for those, with a note. Qwen3.5 follows
`-r` too, off by default as in its template. The renderings equal HF's `apply_chat_template` (tested).

**Qwen3.5** keeps a fixed-size recurrent state per sequence (19 MB for 0.8B) besides its small KV cache. That state
cannot be rewound, so a chat that edits its history (a thinking reply's reasoning, dropped from the next turn) goes
back to a checkpoint the CLI and the server keep at each prompt's assistant header and prefills the rest again, with
the same results as a fresh run.

**qwen3.c's `.bin`** (Q8_0, group 64) works too, exported from the HF weights with qwen3.c's exporter.
The tests use it, and it is the format for comparing with upstream: `scripts/download_models.py hf &&
scripts/setup_venv.sh && scripts/export_q8.sh` (a `.venv` with CPU torch) gives `models/Qwen3-0.6B-q8.bin`.
`make models` does the same.

Not supported: other GGUF types (Q4_0, Q4_1, IQ*, F16 / BF16, Q2_K / Q3_K, and the Unsloth "UD" files that use
them: rejected with a clean error; a "UD" file made only of Q4_K / Q5_K / Q6_K / Q8_0 loads), Qwen3-14B and larger (untested),
the MoE models (`qwen3moe`, `qwen35moe`: Qwen3.5-35B-A3B and up, Qwen3.6-35B-A3B), images, Qwen3-Embedding / Reranker,
RoPE scaling (YaRN), and Hadamard-folded GGUFs (`prism.hadamard.*` metadata: rejected). The other dense Qwen3.5 / 3.6
sizes should load but are untested: 2B (as many value heads as key heads, like 0.8B), 9B, 27B and Qwen3.6-27B (2x / 3x
as many value heads, like 4B; the 27B shape is covered by a synthetic test model). Files with an MTP block
(`nextn` tensors, e.g. bartowski's) or F32 `ssm_alpha` / `ssm_beta` do not load: use Unsloth's or LM Studio's.

## Performance

Full tables, methodology and raw logs: [BENCHMARKS.md](BENCHMARKS.md) and [bench/](bench/). Two machines,
not comparable with each other:

- **AMD EPYC Genoa (Zen 4), 16 vCPUs** (KVM VM, 30 GB RAM, shared with other workloads: run-to-run
  noise ±15-30%, tables report the best runs):
  - Against upstream qwen3.c, Qwen3-0.6B Q8_0 `.bin`, one session (2026-10-25), 8 / 12 threads:
    prefill of 451 tokens 2,058 / 2,758 tok/s, where qwen3.c prefills token by token at ~99 tok/s
    (short prompt); decode 148 / 170 tok/s against qwen3.c's 101 / 106.
  - Later steps ([docs/DEVELOPMENT.md](docs/DEVELOPMENT.md#benchmark-notes-per-step)): decode of the Q8_0
    GGUF ~190 / ~210 tok/s at 8 / 12 threads, of Q4_K_M ~280 / ~350. After a 3,000-token prompt, decode is ~130 / ~150 (Q8_0 GGUF) and ~160 / ~180
    (Q4_K_M) with the f16 KV cache.
  - Batched decode (0.6B Q4_K_M, short contexts, 8 / 12 threads): 8 sequences 1,003 / 1,214 tok/s in
    aggregate, 16 sequences 1,091 / 1,449 (single sequence: 326 / 366).
  - `runq serve`, 0.6B, 12 threads, 16 concurrent clients streaming 128-token replies: 1,185 (Q4_K_M) /
    1,223 (Q8_0) tok/s in aggregate, time to first token ~0.1 s.
- **Apple M1 Pro** (8 performance + 2 efficiency cores, 32 GB, macOS), 0.6B, 8 threads: prefill of 451
  tokens ~1,700 tok/s with the Q8_0 `.bin` and ~1,300 with Q4_K_M; decode ~140-155 and ~190-230 tok/s.
  Sessions vary by ~15% on that desktop.

Quality (0.6B, perplexity relative to HF transformers fp32): Q8_0 GGUF +0.19%, Q8_0 `.bin` +0.85%,
Q4_K_M +4.9% (llama.cpp-based mynah-slm on the same Q4_K_M file: +4.7%).

## Server

`runq serve MODEL` serves one model over HTTP/1.1. By default it binds 127.0.0.1:8080 with no
authentication (`--api-key KEY` requires a bearer token). It has no TLS, so keep it on localhost or a
trusted network.

- Endpoints: `POST /v1/chat/completions`, `GET /v1/models`, `GET /health`. Supported: system / user /
  assistant messages, `max_tokens`, `temperature`, `top_p`, `seed`, `stop`, `stream` (SSE),
  `logprobs`, thinking (`reasoning_effort` or `chat_template_kwargs.enable_thinking`, returned as
  `reasoning_content`), `response_format` `json_schema`, and `usage` with cached prompt tokens. Errors
  come back as OpenAI error objects.
- **Continuous batching**: every running request (up to `--parallel` 16) contributes its next token or a
  prompt chunk to one batched forward pass per step. Streams keep flowing while long prompts prefill,
  and each request's output is bit-identical to running it alone.
- **Prefix reuse**: a new request takes the idle KV slot with the longest common prefix and prefills
  only the rest, or copies the prefix from a running request. A second chat turn, or a fixed system
  prompt, is not prefilled again.
- **Constrained decoding** for `json_schema` with a finite set of outputs (`enum`, `const`, booleans,
  bounded integers, objects of those, `anyOf` / `oneOf`, `$ref`). Forced tokens are not decoded at all,
  so a classification is a single prefill.

As the judge of LiteLLM's complexity router (`auto_router/complexity_router`, `classifier_type: llm`,
tested with LiteLLM 1.105), runq answers `{"tier": "..."}` in ~15-27 ms with the rubric cached (0.6B,
8 threads). Qwen3-1.7B Q8_0 is the smallest good judge: 88% on the project's 80-prompt eval set at
~50 ms per classification.

```yaml
model_list:
  - model_name: smart-router
    litellm_params:
      model: auto_router/complexity_router
      complexity_router_config:
        classifier_type: llm
        classifier_context_window_size: 0     # judge the current message only (shorter prompts)
        classifier_llm_config:
          model: runq-judge
          timeout_ms: 10000                    # default 3000; a timeout opens a 30 s circuit breaker
        tiers: { SIMPLE: small-model, MEDIUM: small-model, COMPLEX: big-model, REASONING: big-model }
  - model_name: runq-judge
    litellm_params:
      model: openai/runq                       # any name
      api_base: http://127.0.0.1:8080/v1
      api_key: none
      temperature: 0
```

All flags, the endpoint details, behaviour under load and the measurements are in
[docs/server.md](docs/server.md). `scripts/litellm_e2e.sh` runs the whole chain against a scratch
LiteLLM proxy, and `scripts/eval_router.py` measures judge accuracy.

## Library API

The engine is a C library, `src/runq.h`; the CLI (`src/cli.c`) and the server use only this API. It
has three kinds of objects:

- **RqModel** (`rq_model_load`): hyperparameters, weights (repacked for the kernels at load), tokenizer,
  and chat templates. It is read-only after loading and shared by any number of sequences.
- **RqSeq** (`rq_seq_new`): one sequence with its own KV cache (capacity and type per sequence), tokens
  and sampling RNG. `rq_seq_truncate`, `rq_seq_common_prefix` and `rq_seq_copy` handle prefix reuse.
- **RqCtx** (`rq_ctx_new`): a compute context (activation buffers for up to `max_batch` rows, logits).
  The threads are one process-wide pool (`rq_threads_init`).

The forward passes are `rq_decode` (one token), `rq_prefill` (a sequence's tokens in chunks) and
`rq_forward_batch` (one pass over rows of several sequences, each at its own position: prefill chunks
next to decode rows). On top of them sit `rq_sample` / `rq_sample_batch` (temperature, top-p, an optional
token mask), `rq_generate` (the chat loop with per-token callbacks), `rq_tokenize`, `rq_detok_push`
(incremental, UTF-8 safe) and `rq_chat_render`. Errors come back as `RQ_ERR_*` codes and messages; the
library never prints or exits.

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
| `src/threads.c`, `src/threads.h` | the thread pool and its phase synchronization |
| `src/kernels_avx512.h`, `src/kernels_neon.h`, `src/kquants.h` | SIMD kernels; the K-quant formats and the exact arithmetic every kernel follows |
| `src/unicode_tables.h`, `src/unicode_nfc.h` | Unicode classes and NFC tables, generated by `tools/gen_unicode_tables.py` |
| `src/cli.c` | the `runq` command line and its test hooks |
| `src/serve.c`, `src/engine.c`, `src/http.c`, `src/constrain.c`, `src/json.c` | `runq serve` (not part of the library): OpenAI layer, scheduler and slot pool, HTTP/1.1, constrained decoding, JSON |

Every kernel family follows one arithmetic definition: int8 activations in groups of 32 (GGUF) or 64
(`.bin`), with the float folds in a fixed order and without FMA contraction. The kernels are
interchangeable bit for bit, but logits differ slightly from fp32 engines; the tests bound the
differences against HF. The activation group size is a process-wide kernel setting: load one model,
or models with the same group size, per process. The changes relative to qwen3.c, step by step, the design notes
and the experiments that were not kept are in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md); the full development log
with every measurement is in the git history ([History](docs/DEVELOPMENT.md#history)). Open work:
[TODO.md](TODO.md), [IDEAS.md](IDEAS.md).

## Testing

```sh
scripts/download_models.py                # default sets: hf (HF bf16), q4km, q8gguf
scripts/setup_venv.sh && scripts/export_q8.sh   # .venv (CPU torch) and models/Qwen3-0.6B-q8.bin
make test                                 # the regression suite
make test-models                          # optional: every downloaded bigger model (minutes)
```

`make test` runs 481 checks in ~28-36 s on the 16-vCPU VM, 644 with the Qwen3.5-0.8B Q8_0 file
(`scripts/download_models.py 3.5-0.8b-q8`; also the source of two synthetic Qwen3.5 models with more value heads
than key heads), 686 with its Q4_K_M file too (`3.5-0.8b-q4km`; ~70-80 s). It needs numpy (from `.venv` or the system `python3`) and the 0.6B
`.bin`; the GGUF groups are skipped if their files are missing. It covers:
tokenizer ids against HF; logits against HF fp32 (cosine, top-1 / top-5); batched against sequential,
thread-count and generic-against-SIMD bit-identity; Q4_K_M and Q5_K dequantization against gguf-py; the KV cache
types; greedy-output goldens; chat-template rendering against HF's `apply_chat_template`; the library
API; the server over HTTP; synthetic GGUF models with the bigger models' shapes; and Qwen3.5-0.8B (its tokenizer,
logits against HF fp32, the same invariants, chats that roll back to a state checkpoint) and synthetic Qwen3.5 models
with the 4B / 27B head layout. Golden references
come from HF transformers and are committed under `tests/golden/`. See [tests/README.md](tests/README.md).

Benchmarks: `make bench` (decode, thread sweep), `make bench-prefill`, `scripts/bench_all.sh` (the runq
tables of BENCHMARKS.md, ~8 min), `scripts/bench_batch.sh`, `scripts/bench_serve.py`,
`scripts/bench_models.py`, `scripts/ppl.py` (perplexity). `scripts/fetch_mynah.sh` fetches and builds
mynah-slm, an optional reference engine for comparison (not part of this repository).

## Limitations

- Qwen3 dense models and dense Qwen3.5 (0.8B and 4B tested; `general.architecture` qwen3 / qwen35, gpt2 tokenizer with the qwen2 /
  qwen35 pre-tokenizer). GGUF Q8_0 / Q4_K_M / Q5_K_M (any mix of Q8_0, Q4_K, Q5_K and Q6_K tensors) and qwen3.c's `.bin`. No other
  quantization types, no MoE, no RoPE scaling, no images, no Jinja interpreter (the templates are built in).
- Fast paths: x86-64 with AVX-512 VNNI (selected at compile time with `-march=native`) and AArch64 with
  the dot-product extension. Other CPUs, including x86 without AVX-512 VNNI, use the generic C kernels:
  the same results, but for Q4_K_M / Q5_K_M ~10x slower prefill and ~3-4x slower decode.
- The NEON port was tested on macOS only. On arm64 the tests that compare generated text against the
  goldens recorded on x86 fail (8 of them), because the compilers round some Q8_0 sums differently.
  Everything else passes there, but the goldens need an arm64 baseline, and the Qwen3 1.7B-8B fixes have
  not been re-run on the Mac yet (docs/DEVELOPMENT.md).
- `runq serve`: no TLS, no tools / function calling, no images, `n` = 1, no `json_object` or free-form
  string constraints; the KV memory is a fixed pool of slots (no paging).
- CPU only; one model per process.

## Credits and license

Gwenedeg is MIT-licensed ([LICENSE](LICENSE)). Third-party code, data and formats (qwen3.c, ggml / llama.cpp formats, Unicode data, Qwen chat templates and test data, LiteLLM's rubric) are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

- The engine started from [qwen3.c](https://github.com/adriancable/qwen3.c) by Adrian Cable (MIT), itself
  based on Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) (MIT). An unmodified copy of
  qwen3.c is kept in [third_party/qwen3.c](third_party/qwen3.c) for diffs and for its `.bin` exporter.
- The GGUF format and the Q8_0 / Q4_K / Q6_K block formats and dequantization expressions follow
  [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT); the implementation here is our own.
- The Unicode tables are generated from the Unicode Character Database (Unicode License v3). The
  tokenizer and NFC behaviour match Hugging Face [tokenizers](https://github.com/huggingface/tokenizers),
  which is used only to generate and check test data.
- The Qwen3 models are by the Qwen team (Alibaba Cloud), Apache-2.0. **No model weights are included**;
  `scripts/download_models.py` fetches them from Hugging Face. The tests contain Qwen's chat templates
  and reference outputs computed with the models.
- The test suite contains LiteLLM's classifier rubric (MIT, Berri AI). The perplexity text is from Charles
  Darwin's *The Voyage of the Beagle* (public domain, via Project Gutenberg).
- [mynah-slm](https://github.com/mynah-org/mynah-slm) is only fetched by a script for benchmark
  comparisons; it is not included.

The notices and license texts of these components are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
