# cwayn

Personal experiment: running Qwen3-0.6B from plain C on the CPU.
The engine is a small C library (`src/runq.h`, see Architecture below) with the `runq` command line on
top, derived from qwen3.c. Roadmap and status: [TODO.md](TODO.md).

## Quick start

```sh
scripts/download_models.py q8gguf     # Qwen3-0.6B-Q8_0.gguf (639 MB), pinned + checksummed
make                                  # -> ./runq  (make debug -> ./runq-debug, make generic -> ./runq-generic)

G=models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf
./runq $G -c 4096                     # interactive chat
./runq $G -c 4096 -t 0 -n 200 -i "Why is the sky blue?"
```

On macOS (Apple Silicon) `make` uses Apple clang (no OpenMP, no extra packages); every format and
KV cache type has NEON kernels there (see Benchmarks). On an M1 Pro, 451-token prefill / decode:
Q8_0 .bin ~1,700 / ~140-155 tok/s, Q4_K_M ~1,300 / ~190-230 (sessions vary by ~15% on that desktop).

The GGUF file is all runq needs: weights, hyperparameters, tokenizer (from its metadata) and the
chat templates (built in). The Q4_K_M GGUF (`scripts/download_models.py q4km`, 397 MB) works the same
way. Which file to use (measured 2026-10-25 on the 16-vCPU dev box, TODO.md Steps 6a, 6b-1, 6b-2, 7a, 7b):

| | Q8_0 GGUF (639 MB) | Q4_K_M GGUF (397 MB) | Q8_0 .bin (export) |
|---|---|---|---|
| perplexity vs HF fp32 (Beagle ch. I) | **+0.19%** | +4.9% | +0.85% |
| logits vs HF: pooled 1-cos, top-1 (84 positions) | **6.1e-4, 79** | 1.6e-2, 69 | 9.0e-4, 76 |
| prefill 451 tokens, 8 / 12 threads | 1,870 / 2,540 tok/s | 1,670 / 2,140 | **1,960 / 2,690** |
| decode, 8 / 12 threads (best of a noisy shared box; Step 7a) | ~190 / ~210 tok/s | **~280 / ~350** | ~200 / ~250 |
| decode after a 3,000-token prompt, 8 / 12 threads (Step 7b) | ~130 / ~150 tok/s | **~160 / ~180** | |
| peak RSS, startup (12 threads) | ~670 MB, 0.15 s | **~410 MB, 0.12 s** | ~625 MB, 0.13 s |

- Quality first, and prompt-heavy work (long documents, many short answers): the **Q8_0 GGUF**. It is
  the closest to the fp32 model by far; the .bin is less accurate and needs the Python export, so it
  is only worth it for the tests and the comparison with qwen3.c.
- Generation speed or memory first (long answers, `-r 1` reasoning, small machines): **Q4_K_M**.
  Decode is memory-bound, and Q4_K_M reads 0.40 GB per token instead of 0.67, so it generates ~1.5x
  faster with ~40% less RAM; prefill is ~10-15% slower (more arithmetic per weight). The price is the
  4-bit weights: ~25x the logits error of Q8_0 and +4.9% perplexity (the same for llama.cpp's
  reference engine, mynah-slm: +4.7%), e.g. it answers "Sydney" for the capital of Australia.

Alternatively, qwen3.c's own format, exported from the HF weights (needs the Python venv; the tests
use both):

```sh
scripts/download_models.py            # default sets: Qwen/Qwen3-0.6B (bf16), Q4_K_M and Q8_0 GGUFs
scripts/setup_venv.sh                 # .venv with CPU torch, for the exporter (and the test goldens)
scripts/export_q8.sh                  # -> models/Qwen3-0.6B-q8.bin (Q8_0, ~630 MB) + .tokenizer + .template*
# (or: make models, which runs download + export if the .bin is missing)
./runq models/Qwen3-0.6B-q8.bin -c 4096

make test                             # regression suite, ~30 s (tests/README.md)
make bench                            # thread sweep, prefill/decode tok/s (scripts/bench_qwen3c.sh)
make bench-prefill                    # prefill tok/s vs prompt length and -b (scripts/bench_prefill.sh)
scripts/bench_all.sh                  # all runq speed tables of BENCHMARKS.md (~8 min)
scripts/fetch_mynah.sh                # optional reference engine, then scripts/bench_mynah.sh
```

## Chat

`-m chat` (the default) without `-i` reads one user message per stdin line and keeps the whole
conversation in the KV cache; an empty line or EOF ends it. `-i` is a single turn, then exit.
Piped input works the same way (each line echoed after the `> ` prompt), e.g. with `-t 0` for a
deterministic transcript:

```sh
printf 'My name is Ada.\nWhat is my name?\n' | ./runq models/Qwen3-0.6B-q8.bin -c 4096 -t 0
```

- Each turn prefills only what is new: the `<|im_end|>` that ended the previous answer, `\n`, and
  the next user turn of Qwen3's chat template (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`,
  plus `<think>\n\n</think>\n\n` with `-r 0`). The `[timing]` line per turn counts those new tokens.
- Thinking (`-r 1`): as in Qwen3's template, earlier answers stay in the history without their
  `<think>...</think>` block: at the next turn the cache is cut back to where the answer started and
  the visible answer is re-prefilled (usually a few dozen tokens; the reasoning is usually much
  longer, so this also saves context). With `-r 0` the empty think block of the template stays in
  the cache before each answer (HF's template drops it from earlier turns; removing it would mean
  re-prefilling every answer, for a 4-token difference). Otherwise the cache holds exactly what the
  template renders (checked against HF's `apply_chat_template`).
- Context full (`-c`): when the conversation plus a reserve for the answer (`-n`, else `-c`/4, at
  most `-c`/2) does not fit, the oldest turns are dropped until it takes at most half of the rest,
  keeping the system prompt (`-y`, stays cached) and the most recent turns, and the rest is
  re-prefilled (~2,000 tok/s). stderr says `[context] full ... dropped N oldest turn(s)`. An answer
  that reaches the end of the context is cut; a single message that does not fit is skipped.
- `-n` limits each answer; `-y` sets the system prompt for the whole conversation.

KV cache: `-k f16` (the default since Step 7b), `-k bf16` or `-k f32`. The 16-bit caches take half the
memory (28 layers x 2 x 1,024 values per position: 115 KB instead of 229 KB; 0.47 instead of 0.94 GB at
`-c 4096`; peak RSS after 3,000 tokens ~1.15 instead of ~1.37 GB for the Q8_0 GGUF, ~0.89 instead of ~1.23
for Q4_K_M) and decode long contexts faster (3,000 tokens, Q4_K_M, 8 threads: ~160 vs ~136 tok/s with
`-k f32`, ~100 before Step 7b). f16 is as close to the fp32 model as the f32 cache (KL divergence from HF fp32 +0.2%,
within the noise); bf16 is a bit further (+2.6%: |K| reaches ~500, and bf16 keeps 8 significant bits to
f16's 11) and prefills long prompts ~3% faster. All three pass the same logits thresholds; the attention
arithmetic is fp32 on the exactly converted values (TODO.md Step 7b).

Threads: without `OMP_NUM_THREADS`, runq uses nproc - nproc/8 threads (Apple Silicon with efficiency
cores: the performance cores, see the M1 Pro notes under Benchmarks; 14 on the 16-vCPU dev box).
Decode is memory-bound: 12-14 threads saturate the bandwidth there (~190-210 tok/s Q8_0 GGUF,
~280-350 Q4_K_M), 8 give ~90% of it; prefill scales to 14 (~2,580 tok/s at 439 tokens vs ~2,400 at 12).
All CPUs work too since Step 7a (16 of 16 vCPUs: decode ~140-170 tok/s with other processes running;
with OpenMP barriers it collapsed to ~2-20), but are not faster on a busy machine. Long prompts: decode
splits each kv head's attention over threads per 256 positions (3,000 tokens at 12 threads, Q8_0 GGUF:
~70 -> ~100 tok/s in Step 7a, ~150 since Step 7b's cache layout and f16). Pass `-c` (context size):
without it the KV cache is sized for the full 40,960-token context, 4.7 GB of address space (9.4 GB with
`-k f32`), touched only as far as the conversation goes.

## Architecture

Since Step 8 (TODO.md) the engine is a library, `src/runq.h`, and the CLI (`src/cli.c`) only uses that
API. Three kinds of objects:

- **RqModel** (`rq_model_load`): hyperparameters, weights (repacked for the kernels at load),
  tokenizer, the chat templates of both thinking modes. Loaded once, read-only afterwards, shared by
  any number of sequences and contexts. Load errors (missing / corrupt / unsupported file, no memory)
  come back as a message, nothing exits.
- **RqSeq** (`rq_seq_new`): one sequence: its own KV cache (capacity and type `RQ_KV_F16 / BF16 / F32`
  per sequence), its length and the tokens at its positions, an RNG for sampling;
  `rq_seq_truncate` / `rq_seq_common_prefix` for prefix reuse.
- **RqCtx** (`rq_ctx_new`): a compute context: activation and scratch buffers for up to `max_batch`
  rows, the threads' synchronization state, the logits buffer, profiling. The threads are one
  process-wide pool (`rq_threads_init`), so one forward pass runs at a time per process.

The forward passes take the sequence(s) explicitly: `rq_decode` (one token: the GEMV path, attention
split over threads per 256 positions), `rq_prefill` (a sequence's tokens in chunks of `max_batch`),
`rq_forward_batch` (one pass over rows of several sequences, each at its own next position: a prefill
chunk next to decode rows of other sequences, logits of the selected rows). Results are bit-identical
whatever the thread count, chunk size or batch composition (`tests/api_test.c` checks sequences
interleaved and batched together against each alone). On top: `rq_sample` (per-call temperature /
top-p, the sequence's RNG, an optional token mask for constrained decoding), `rq_generate` (the chat
loop: prefill, sample, decode; callbacks per token and per logits vector; timing in a struct),
`rq_tokenize`, `rq_detok_push` (incremental, never splits a UTF-8 character: streaming),
`rq_chat_render` (system / user / assistant messages in Qwen3's template, thinking on or off, as the
CLI's chat keeps the conversation). Errors are `RQ_ERR_*` codes (e.g. `RQ_ERR_CONTEXT` for a prompt
longer than the sequence's capacity); the library never prints or exits.

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
```

| file | contents |
|---|---|
| `src/runq.h` | the public API |
| `src/model.c` | RqModel: .bin / GGUF loading, weight conversion and repacking (`prepare_weights`), chat templates |
| `src/gguf.c` | the GGUF reader (untrusted input: every count, length and offset checked) |
| `src/forward.c` | RqSeq, RqCtx and all numeric code of the forward passes: decode step, batched pass, attention and matmul dispatch, quantization, norms, RoPE, KV cache |
| `src/tokenizer.c` | BPE tokenizer matching HF (added tokens, NFC, Qwen2 pre-tokenizer), detokenization |
| `src/sampler.c`, `src/chat.c` | sampling; chat template rendering and the generation loop |
| `src/threads.c`, `src/threads.h` | the thread pool and the phase synchronization |
| `src/kernels_avx512.h`, `src/kernels_neon.h`, `src/kquants.h` | SIMD kernels, K-quant formats and arithmetic definitions |
| `src/internal.h` | the structures shared by the modules |
| `src/cli.c` | the `runq` command line and its test hooks |

Each module is its own translation unit (objects in `build/native`, `build/generic`, `build/debug`).
All numeric code of the forward passes stays in `forward.c`, so the `-Ofast` code generation of the
exactly defined arithmetic is the one measured and tested before; the split was checked byte for byte
against the single-file binary (`tests/compare_bins.sh`) and is as fast. The activation group size
(64 for the .bin, 32 for GGUF) is a process-wide setting of the kernels: one model, or models of the
same group size, per process.

## Models (`scripts/download_models.py --list`)

| set | source | file(s) |
|---|---|---|
| `hf` | `Qwen/Qwen3-0.6B` @ `c1899de` | bf16 safetensors + tokenizer (input for the .bin export, HF test goldens) |
| `q4km` | `unsloth/Qwen3-0.6B-GGUF` @ `50968a4` | `Qwen3-0.6B-Q4_K_M.gguf` (imatrix; runq loads it directly) |
| `q8gguf` | `unsloth/Qwen3-0.6B-GGUF` @ `50968a4` | `Qwen3-0.6B-Q8_0.gguf` (runq loads it directly) |

Revisions are pinned. Each file is verified by sha256 (LFS files) or git blob sha1 (small files).

GGUF vs .bin: both Q8_0. The GGUF has 32-weight groups with fp16 scales (fp32 once loaded,
exactly), the .bin 64-weight groups with fp32 scales; runq quantizes the activations to int8 in
groups of the same size. The GGUF is closer to HF: perplexity +0.19% vs +0.85% over fp32, logits
pooled 1-cos 6.1e-4 vs 9.0e-4 (TODO.md, Step 6a). Prefill is ~4-8% slower (twice as many
float folds in the GEMM), decode ~5% (the packed weights are ~6% larger: fp32 scales per 32).

## Layout and third-party code

- `src/*.c`: our engine (library + CLI, see Architecture), from qwen3.c's `runq.c` (MIT, see `LICENSE`).
  Changes so far:
  `-n <int>` max new tokens, plus a `[timing]` line on stderr (prefill and decode tok/s);
  no classifier matmul for prompt tokens whose logits are unused;
  batched prefill (`-b <int>` tokens per chunk, default 512, `-b 1` = token by token):
  `forward_batch()`, bit-identical to the token-by-token path;
  AVX-512 VNNI kernels for the GEMM (prefill) and GEMV (decode) on weights repacked at load,
  and an AVX-512 attention (shared by decode and prefill) on a tiled KV cache
  (`src/kernels_avx512.h`, compile-time selection; `-DRUNQ_GENERIC` / `make generic` or a CPU
  without AVX-512 gives the generic C kernels, bit-identical results);
  NEON kernels for Q8_0, Q4_K / Q6_K and the attention (f32 / bf16 / f16 KV cache) on AArch64 with
  the dot product extension (`src/kernels_neon.h`; tested on macOS only), bit-identical to the generic ones
  (`make kq-neon-test`);
  RoPE cos/sin once per position;
  threading (`src/threads.h`, Step 7a): a decode step / prefill chunk is one job on a persistent
  thread pool, in phases of work items that end when the work is done (not when every thread has
  arrived; waiting threads spin, then sleep on a futex), so a preempted thread no longer stalls the
  others (16 threads on 16 vCPUs used to collapse); 4 phases per decoded layer (113 sync points per
  token, was 198 OpenMP barriers); the attention defined over blocks of 256 positions, decode splitting
  long contexts over threads per block (same results for decode, prefill and any thread count);
  KV cache types (`-k f16|bf16|f32`, Step 7b): 16-bit caches rounded to nearest even at the store and
  converted exactly at the load (the fp32 attention arithmetic unchanged), in blocks of 256 positions
  with each kv head's tiles contiguous; the activation quantizer vectorized (same results);
  SwiGLU NaN fix (`-Ofast` expf overflow, it did corrupt some last-layer activations);
  a rewritten tokenizer matching HF (added tokens, Qwen2 pre-tokenizer regex, byte-level BPE
  with a hash map and a merge heap; ~1 ms for 2,000 tokens) with HF's NFC normalizer (quick check
  fast path; same Unicode data as HF tokenizers);
  multi-turn chat with the conversation in the KV cache (see Chat above);
  faster sampling (top-p: vectorized candidate scan + a heap popped only for the nucleus; argmax
  vectorized; same tokens for the same seed);
  token embedding rows dequantized at lookup from the Q8_0 table (upstream kept a 622 MB float
  copy) and the weight repacking done from `pread` in one parallel loop: startup ~0.5 -> ~0.06 s,
  peak RSS 1.36 -> ~0.63-0.74 GB;
  test hooks `-m tokenize`, `-m logits -o <file>` and `RUNQ_CHAT_DUMP` (see `tests/README.md`);
  GGUF Q8_0 loading (format detected by magic): a bounds-checked GGUF v2/v3 reader (every
  count, length and offset validated; corrupt files give a clean error), the hyperparameters from
  the `qwen3.*` metadata, the tensors by llama.cpp name, the Q8_0 blocks (group 32, fp16 scales)
  converted at load for the VNNI kernels (generalized to group size 32 or 64) or the generic ones,
  the tokenizer from `tokenizer.ggml.*` (GPT-2 byte-level tokens, merge ranks as scores), Qwen3's
  chat templates built in (identical to the exporter's `.template*` files);
  GGUF Q4_K_M (Q4_K / Q6_K tensors, type dispatch per tensor): weights repacked at load into 16-row
  interleaved blocks with the 4 / 6-bit data kept packed (398 MB read per decoded token vs 670 MB for
  Q8_0), activations int8 per 32, AVX-512 VNNI GEMV (at the memory-bandwidth limit: decode ~1.5x
  the Q8_0 GGUF) and GEMM (weight tile unpacked once per thread into u8, ~1.17x the Q8_0 GEMM time),
  bit-identical to the generic C kernels, embedding rows dequantized with ggml's expressions
  (bit-identical to gguf-py), test hook `-m dequant`;
  the library API (Step 8: model / sequence / compute context, multi-sequence batched passes, errors
  returned instead of `exit()`, no printing in the library), split into modules.
- `src/kquants.h`: the Q4_K / Q6_K block formats, their dequantization, the repacked layouts (Q4P,
  Q6P) and the generic kernels; the arithmetic of a K-quant matmul output is defined there.
- `src/kernels_avx512.h`: the AVX-512 VNNI Q8_0 GEMM / GEMV (group size 64 or 32) and the weight
  repacking, the Q4_K / Q6_K GEMV / GEMM on the Q4P / Q6P layouts, and the fp32 attention with the KV
  cache layouts (fp32 / bf16 / f16; layouts, arithmetic and bit-identity argument in its comments).
- `src/threads.h`: the thread pool and the phase synchronization (design and measurements in its
  header comment; OpenMP's parallel region instead with `-DRUNQ_NO_POOL` and on Windows).
- `src/unicode_tables.h`: \p{L}/\p{N}/\s ranges for the pre-tokenizer, generated by
  `tools/gen_unicode_tables.py` (Python's unicodedata, Unicode 15.1).
- `src/unicode_nfc.h`: NFC tables (combining classes, decompositions, compositions, quick check),
  `tools/gen_unicode_tables.py --nfc`: Unicode 15.1 with the overrides that reproduce HF
  tokenizers' older (9.0) normalization data (108 newer marks ccc 0, U+11938); see TODO.md Step 5.
- `third_party/qwen3.c/`: pristine https://github.com/adriancable/qwen3.c @ `55643ff`
  (2025-07-05), kept for diffs (upstream's `runq.c`; ours is `src/*.c` since Step 8, the single file
  before it in git history). Only its
  `export.py`/`model.py` are used, by `scripts/export_q8.sh` (the .bin path; GGUF needs neither).
  qwen3.c itself started from Andrej Karpathy's llama2.c.
- `third_party/mynah-slm/`: not committed. `scripts/fetch_mynah.sh` clones
  https://github.com/mynah-org/mynah-slm @ `cde1599` (2026-09-22) and builds it (needs
  `libopenblas-dev`). Reference and benchmarks only.

## Benchmarks

Full tables, methodology and baselines: [BENCHMARKS.md](BENCHMARKS.md) (one session, 2026-10-25,
`d76e4cc`, 16-vCPU EPYC Genoa VM, shared). Headline, Q8_0, greedy, best of 6-9 runs:

| threads | prefill 451 / 1,500 / 3,000 tokens | decode tok/s |
|---|---|---|
| 8 | 2,058 / 1,570 / 1,132 tok/s | 148 |
| 12 (the default then; 14 since Step 7a) | 2,758 / 1,976 / 1,531 tok/s | 170 |

Same session: upstream qwen3.c decodes at 101 / 106 tok/s (8 / 12 threads) with ~99 tok/s token-by-token
prefill; mynah-slm at ~34 (Q8_0 and Q4_K_M). Perplexity vs HF fp32: runq Q8_0 +0.8%, Q4_K_M +4.7%
(mynah-slm); runq on the Q8_0 GGUF +0.19% (TODO.md Step 6a, measured later), on the Q4_K_M GGUF
+4.9% (Step 6b-1). Q4_K_M with the AVX-512 kernels (Step 6b-2, not in BENCHMARKS.md yet): see the
table under Quick start and TODO.md. Step 7a (threading, not in BENCHMARKS.md yet; same box, later
session, interleaved with the Step 6b-2 binary, best / median of 8): decode Q8_0 GGUF 8 / 12 / 16 threads
192 / 168, 210 / 182, ~140-170 / ~115-160 tok/s (before: 198 / 137, 179 / 118, collapse); prefill
within ~5% at 8-14 threads, no collapse at 16 (TODO.md Step 7a). Step 7b (KV cache, not in BENCHMARKS.md yet;
interleaved with the Step 7a binary, best of 5, load 8-13), Q8_0 GGUF / Q4_K_M, f16 cache: decode after a
3,000-token prompt 8 threads 90 -> 128 / 103 -> 158 tok/s, 12 threads 114 -> 147 / 120 -> 180; after 1,500
tokens 8 threads 130 -> 164 / 153 -> 220; prefill 450 / 1,500 / 3,000 tokens within -0..+7% (TODO.md Step 7b).
`scripts/bench_all.sh` reruns the runq tables (~8 min).

Apple M1 Pro (8 P + 2 E cores, branch `mac-m1-pro`, 8 threads = the default there (the performance
cores), separate sessions, BENCHMARKS.md "Apple M1 Pro"), prefill at 451 / 1,500 tokens and decode
(256 new tokens), tok/s: Q8_0 .bin 1,691 / 1,211 / 140 (generic C kernels: 659 / 441 / 134);
Q4_K_M 1,327 / 968 / 190 (generic: 156 / 138 / 60), both before the rebase onto Step 7a. After it,
against the old build in one session: decode +4-7%, prefill +0-5%; 9 or 10 threads decode 5-13%
slower than 8. After the rebase onto Step 7b (f16 cache by default, NEON attention for it): decode
vs the f32 cache +8-10% / +17-22% / +23-29% after 451 / 1,500 / 3,000-token prompts, prefill within
~4%. Not comparable with the EPYC table above (other CPU).

## Known limitations

- Q8_0 (.bin or GGUF) and Q4_K_M GGUF (any mix of Q8_0 / Q4_K / Q6_K matrices). Other GGUF types
  (F16, Q5_K, IQ*, ...) are rejected with a clean error. Without AVX-512 VNNI (or `make generic`)
  every format runs on the generic C kernels, with the same results; for Q4_K_M they are ~10x
  slower in prefill and ~3-4x in decode than the AVX-512 ones. On AArch64 (with `sdot`) every
  format and the attention have NEON kernels (same results; on an M1 Pro vs generic: Q4_K_M 7-8.5x
  prefill and 3x decode, Q8_0 2.6-2.7x prefill and +4% decode). x86 CPUs without AVX-512 VNNI
  still use the generic kernels. The NEON attention takes every KV cache type (`-k f32|bf16|f16`):
  the 16-bit elements are converted to fp32 in registers, exactly, so the results are the generic
  build's.
- `tests/run_tests.sh` on arm64: everything passes except the tests that depend on the exact
  generated text, recorded with the x86 build; they also fail with the unmodified source built on
  the Mac. Last full run (`95ccbb7`, rebased onto `caf833e`): 184 passed, 8 failed: seven `greedy/*`
  goldens (`chat_system`, `generate`, `generate@b5`, `f32_generate`, `bf16_generate`, `seeded_chat`,
  `seeded_generate`) and `chat/multiturn_golden` (a transcript golden). Unmodified `origin/main`
  (`989f712`) built on the Mac fails 9 (the same 8, at other characters, and `chat/thinking`): its
  generic C code is not exact under clang (`ATT_EXACT` / `NO_FP_CONTRACT` are gcc `optimize`
  attributes; this branch adds clang pragmas, `RQ_FP_EXACT`), so it generates other text. The GGUF
  Q8_0 tests are skipped (file not downloaded); the logits against HF and the batched / thread /
  kernel bit-identity checks pass.
  The goldens need an arm64 re-baseline (`--update-greedy` into a separate file).
- GGUF: Qwen3 (`general.architecture` qwen3, RoPE theta 1e6, RMSNorm eps 1e-6) with the gpt2 /
  qwen2 tokenizer only; `tokenizer.chat_template` (Jinja) is not interpreted, Qwen3's templates
  are built in.
