# Gwenedeg: developer notes

Details moved out of the [README](../README.md) when the project (then called cwayn) was prepared for
publication: what changed relative to qwen3.c, the 0.6B file comparison, chat / KV cache / thread details,
benchmark notes, the platform test status, and design notes with the experiments that were measured and not
kept. "Step N" refers to the development log ([History](#history)); open work is in [TODO.md](../TODO.md).
"The box" / "the dev box" is the development machine: a shared 16-vCPU AMD EPYC Genoa (Zen 4) KVM VM (see
[BENCHMARKS.md](../BENCHMARKS.md), Setup).

## Qwen3-0.6B: which file to use

The GGUF file is all runq needs: weights, hyperparameters, tokenizer (from its metadata) and the
chat templates (built in). The Q4_K_M GGUF (`scripts/download_models.py q4km`, 397 MB) works the same
way. Which file to use (measured 2026-10-25 on the 16-vCPU dev box, Steps 6a, 6b-1, 6b-2, 7a, 7b):

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

GGUF vs .bin: both Q8_0. The GGUF has 32-weight groups with fp16 scales (fp32 once loaded,
exactly), the .bin 64-weight groups with fp32 scales; runq quantizes the activations to int8 in
groups of the same size. The GGUF is closer to HF: perplexity +0.19% vs +0.85% over fp32, logits
pooled 1-cos 6.1e-4 vs 9.0e-4 (Step 6a). Prefill is ~4-8% slower (twice as many
float folds in the GEMM), decode ~5% (the packed weights are ~6% larger: fp32 scales per 32).

## Chat, KV cache and threads (details)

`-m chat` (the default) without `-i` reads one user message per stdin line and keeps the whole
conversation in the KV cache; an empty line or EOF ends it. `-i` is a single turn, then exit.
Piped input works the same way (each line echoed after the `> ` prompt), e.g. with `-t 0` for a
deterministic transcript:

```sh
printf 'My name is Ada.\nWhat is my name?\n' | ./runq models/Qwen3-0.6B-q8.bin -c 4096 -t 0
```

- Each turn prefills only what is new: the `<|im_end|>` that ended the previous answer, `\n`, and
  the next user turn of Qwen3's chat template (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`,
  plus `<think>\n\n</think>\n\n` with `-r 0`; the 2507 models' variants: README, Chat templates). The `[timing]` line per
  turn counts those new tokens.
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
arithmetic is fp32 on the exactly converted values (Step 7b).

Threads: without `OMP_NUM_THREADS`, runq uses nproc - nproc/8 threads (Apple Silicon with efficiency
cores: the performance cores, see the M1 Pro notes under Benchmarks; 14 on the 16-vCPU dev box).
Decode is memory-bound: 12-14 threads saturate the bandwidth there (~190-210 tok/s Q8_0 GGUF,
~280-350 Q4_K_M), 8 give ~90% of it; prefill scales to 14 (~2,580 tok/s at 439 tokens vs ~2,400 at 12).
All CPUs work too since Step 7a (16 of 16 vCPUs: decode ~140-170 tok/s with other processes running;
with OpenMP barriers it collapsed to ~2-20), but are not faster on a busy machine. Long prompts: decode
splits each kv head's attention over threads per 256 positions (3,000 tokens at 12 threads, Q8_0 GGUF:
~70 -> ~100 tok/s in Step 7a, ~150 since Step 7b's cache layout and f16). Pass `-c` (context size):
without it the KV cache is sized for the model's full context, at most 40,960 tokens (Qwen3's; the 2507 models'
262,144 would be 38.6 GB): 4.7 GB of address space for 0.6B (9.4 GB with `-k f32`), touched only as far as the
conversation goes.

## Layout and changes from qwen3.c

- `src/*.c`: our engine (library + CLI, see the README's Architecture), from qwen3.c's `runq.c` (MIT, see `LICENSE`).
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
  returned instead of `exit()`, no printing in the library), split into modules;
  `runq serve` (Step 9): an OpenAI-compatible HTTP server with prefix KV reuse and constrained decoding
  for finite JSON schemas (a hand-written HTTP/1.1 server and JSON parser / writer, no dependencies);
  batched decode of several sequences (Step 10a): small-batch AVX-512 kernels, a decode-rows worker,
  batched sampling, bit-identical to decoding each sequence alone; continuous batching in `runq serve`
  (Step 10b).
- `src/kquants.h`: the Q4_K / Q5_K / Q6_K block formats, their dequantization, the repacked layouts (Q4P, Q5P,
  Q6P) and the generic kernels; the arithmetic of a K-quant matmul output is defined there.
- `src/kernels_avx512.h`: the AVX-512 VNNI Q8_0 GEMM / GEMV (group size 64 or 32) and the weight
  repacking, the Q4_K / Q5_K / Q6_K GEMV / GEMM / small-batch kernels on the Q4P / Q5P / Q6P layouts, and the fp32 attention with the KV
  cache layouts (fp32 / bf16 / f16; layouts, arithmetic and bit-identity argument in its comments).
- `src/threads.h`: the thread pool and the phase synchronization (design and measurements in its
  header comment; OpenMP's parallel region instead with `-DRUNQ_NO_POOL` and on Windows).
- `src/unicode_tables.h`: \p{L}/\p{N}/\s ranges for the pre-tokenizer, generated by
  `tools/gen_unicode_tables.py` (Python's unicodedata, Unicode 15.1).
- `src/unicode_nfc.h`: NFC tables (combining classes, decompositions, compositions, quick check),
  `tools/gen_unicode_tables.py --nfc`: Unicode 15.1 with the overrides that reproduce HF
  tokenizers' older (9.0) normalization data (108 newer marks ccc 0, U+11938); see
  [tests/README.md](../tests/README.md#tokenizer-fuzz-test-on-demand-not-in-make-test) (`--nfc-all`).
- `third_party/qwen3.c/`: pristine https://github.com/adriancable/qwen3.c @ `55643ff`
  (2025-07-05), kept for diffs (upstream's `runq.c`; ours is `src/*.c` since Step 8, the single file
  before it in git history). Only its
  `export.py`/`model.py` are used, by `scripts/export_q8.sh` (the .bin path; GGUF needs neither).
  qwen3.c itself started from Andrej Karpathy's llama2.c.
- `third_party/mynah-slm/`: not committed. `scripts/fetch_mynah.sh` clones
  https://github.com/mynah-org/mynah-slm @ `cde1599` (2026-09-22) and builds it (needs
  `libopenblas-dev`). Reference and benchmarks only.

## Benchmark notes (per step)

Full tables, methodology and baselines: [BENCHMARKS.md](../BENCHMARKS.md) (one session, 2026-10-25,
`d76e4cc`, 16-vCPU EPYC Genoa VM, shared). Headline, Q8_0, greedy, best of 6-9 runs:

| threads | prefill 451 / 1,500 / 3,000 tokens | decode tok/s |
|---|---|---|
| 8 | 2,058 / 1,570 / 1,132 tok/s | 148 |
| 12 (the default then; 14 since Step 7a) | 2,758 / 1,976 / 1,531 tok/s | 170 |

Same session: upstream qwen3.c decodes at 101 / 106 tok/s (8 / 12 threads) with ~99 tok/s token-by-token
prefill; mynah-slm at ~34 (Q8_0 and Q4_K_M). Perplexity vs HF fp32: runq Q8_0 +0.8%, Q4_K_M +4.7%
(mynah-slm); runq on the Q8_0 GGUF +0.19% (Step 6a, measured later), on the Q4_K_M GGUF
+4.9% (Step 6b-1). Q4_K_M with the AVX-512 kernels (Step 6b-2, not in BENCHMARKS.md yet): see the
table under "Qwen3-0.6B: which file to use" above. Step 7a (threading, not in BENCHMARKS.md yet; same box, later
session, interleaved with the Step 6b-2 binary, best / median of 8): decode Q8_0 GGUF 8 / 12 / 16 threads
192 / 168, 210 / 182, ~140-170 / ~115-160 tok/s (before: 198 / 137, 179 / 118, collapse); prefill
within ~5% at 8-14 threads, no collapse at 16. Step 7b (KV cache, not in BENCHMARKS.md yet;
interleaved with the Step 7a binary, best of 5, load 8-13), Q8_0 GGUF / Q4_K_M, f16 cache: decode after a
3,000-token prompt 8 threads 90 -> 128 / 103 -> 158 tok/s, 12 threads 114 -> 147 / 120 -> 180; after 1,500
tokens 8 threads 130 -> 164 / 153 -> 220; prefill 450 / 1,500 / 3,000 tokens within -0..+7%.
`scripts/bench_all.sh` reruns the runq tables (~8 min).

Apple M1 Pro (8 P + 2 E cores, branch `mac-m1-pro`, 8 threads = the default there (the performance
cores), separate sessions, BENCHMARKS.md "Apple M1 Pro"), prefill at 451 / 1,500 tokens and decode
(256 new tokens), tok/s: Q8_0 .bin 1,691 / 1,211 / 140 (generic C kernels: 659 / 441 / 134);
Q4_K_M 1,327 / 968 / 190 (generic: 156 / 138 / 60), both before the rebase onto Step 7a. After it,
against the old build in one session: decode +4-7%, prefill +0-5%; 9 or 10 threads decode 5-13%
slower than 8. After the rebase onto Step 7b (f16 cache by default, NEON attention for it): decode
vs the f32 cache +8-10% / +17-22% / +23-29% after 451 / 1,500 / 3,000-token prompts, prefill within
~4%. Not comparable with the EPYC table above (other CPU).

## Platform notes and test status

- Q8_0 (.bin or GGUF) and Q4_K_M / Q5_K_M GGUF (any mix of Q8_0 / Q4_K / Q5_K / Q6_K matrices). Other GGUF types
  (F16, IQ*, Q4_0, ...) are rejected with a clean error. Without AVX-512 VNNI (or `make generic`)
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
- GGUF: Qwen3 (`general.architecture` qwen3: the dense models, any RoPE theta without scaling, RMSNorm eps 1e-6)
  with the gpt2 / qwen2 tokenizer only; `tokenizer.chat_template` (Jinja) is not interpreted: Qwen3's templates are
  built in, the variant (hybrid / Instruct-2507 / Thinking-2507) is detected from it (README, Chat templates).
- The Step 11 fixes (RoPE theta, rmsnorm, the untied embedding) are shared C code, but untested on the Mac (NEON,
  clang): its goldens need a re-run.

## Design notes and rejected experiments

Measured alternatives that were not kept, plus the limits they ran into, where the source comments do not already
record them (those cover the kernel layouts and arithmetic, `src/kquants.h` and `src/kernels_avx512.h`; the
thread-pool design, `src/threads.h`; the per-thread ranges vs shared-counter claims, `src/forward.c`; huge pages,
`src/model.c`; byte-level constrained decoding, `src/constrain.c`). Same box, so the usual ±15-30% noise applies;
the step names the log entry with the full measurement.

Kernels
- Q8_0 GEMV with software prefetch (2 KB / 8 KB ahead) was slower: 1 thread 46 -> 42 / 38 tok/s, 4 threads 116 ->
  107 / 92 (Step 3a). The hardware prefetchers do best with one sequential stream per thread, which is also why
  Q6P keeps each sub-block's high bits next to its low bits (`src/kquants.h`).
- Ceilings: the Q8_0 GEMM runs at ~85% of the measured `vpdpbusd` peak (2.9 TOPS at 8 threads, 372 GOPS per
  thread, register-only microbenchmark). The rest is the float fold (cvt, 2 mul, add per accumulator and group,
  two of them on `vpdpbusd`'s ports): ~89% at most with groups of 64, ~80% with 32. The K-quant GEMM reaches
  63-74% (6-7 fold ops per 8 `vpdpbusd`, Step 6b-2). The fold order is the bit-identity definition, so it cannot
  simply be fused into FMAs.
- K-quant GEMM: unpacking Q4_K in registers per 6-row tile, instead of once per thread into the scratch tile, was
  ~8% slower (w1 + w3 80-81 vs 74-75 ms at 451 tokens, 8 threads). Register tile widths (Q8_0 C = 4-8 and NB =
  24-128; Q4_K C = 4-8; Q6_K C = 2-6) were all within noise. Generic K-quant kernels with more rows per unpack
  (`KQ_NC` 4 -> 8 / 16) were only 7-10% faster, so this was not changed.

Attention
- Block size: `ATT_P` 128 and 512 are as accurate as 256 against a double-precision attention (within ~5%). 512
  costs the same in prefill and gives fewer decode work items at mid contexts, so 256 was kept (Step 7a).
- Prefill rows per work item: 3 / 6 / 8 / 12 were within noise at first (Step 3b). Later, 12 rows (24 vectors)
  measured worse: 66 vs 48 ms at 451 tokens, 2.24 vs 1.58 s at 3,000. `ATT_ROWS` stays at 6 (Step 7b).
- bf16 scores: converting both halves of a K pair at once spilled registers (+15% prefill attention over fp32);
  loading the 64-byte pair per dim, as the fp32 loop does, gives -2%.
- Prefill attention at long prompts is compute-bound, not bandwidth-bound: from 1 to 8 threads it speeds up 6.96x,
  the same with the half-size bf16 cache. The exp is only ~3% of it (Step 7b).

Prefill "rest" (norms, SwiGLU, RoPE, K / V stores; ~11% of a 451-token prefill, 5% at 3,000)
- Fusing w1 + w3 + SwiGLU + quantize into one GEMM phase was slower (79-81 ms vs 63-64 + 9-10 ms at 451 tokens):
  partial-line writes scattered across rows, and bigger column groups unbalance the threads. Reverted
  (`15f697d`).
- These phases are memory-traffic bound: the rows were written by other cores' GEMM tiles. A 451-token norm phase
  takes ~0.3 ms against ~60 us of arithmetic per thread.
- SwiGLU's `expf` (libmvec) is ~0.75% of a 451-token prefill. A polynomial would save ~0.4% and change the
  numerics, so it was not done.

Threads
- With OpenMP barriers, 16 threads on the 16 vCPUs collapsed (decode ~15-22 tok/s). No libgomp setting fixed it:
  `OMP_WAIT_POLICY=passive` gave ~40 tok/s, `GOMP_SPINCOUNT` 1e3 / 3e4 / 3e6 ~40 / ~9 / ~54, and
  `OMP_PROC_BIND=close` ~20-30 (Step 3b). Hence the phase synchronization on our own pool (`src/threads.h`).
  Spinning only 100 rounds (~3 us) before sleeping was ~30% slower: sleeps and wake-ups every phase.
- Batched decode worker (Step 10a), measured and not kept:
  - the attention items of each kv head placed after the next head's q / k / v items: ~5-10% slower;
  - the norms as the first items of the next phase: one sync point less, but the w1 / w3 items then come from a
    shared counter, ~10% slower at 8 rows;
  - the norms as a phase epilogue for more than 2 rows: within noise at 4, ~10% slower at 8 and 16.
  - q / k / v items of 32 / 64 / 128 weight rows were within noise. Above 64 rows per pass, the prefill path's
    GEMM is the faster one: ~10% at 96-256 rows, equal at 48-64.

Sampling and the KV cache type
- At `-t > 0`, the sampler's remaining cost is the temperature scale and the softmax over the vocabulary (~80-110
  us per token, already vectorized by gcc). Splitting it over threads while keeping gcc's exact summation order
  was not done: ~2% of decode at ~190 tok/s.
- The default cache is f16, not bf16: bf16 missed the "< 0.1% perplexity" bar. A paired per-token NLL over 49k
  tokens gave bf16 vs f32 +0.11% ± 0.08 (95%), and KL +2.6%, consistently. f16 is within the noise of f32.

Server
- `--chunk` (prompt tokens per pass while others decode), 8 streams + 4 prompts of ~2,000 tokens, 0.6B Q4_K_M, 12
  threads, values 32 / 64 / 128: the long prompts' time to first token was 1,762 / 1,386 / 1,224 ms, the streams'
  token gaps p99 48 / 66 / 122 ms and at most 147 / 71 / 144 ms. 64 was kept.
- `--parallel` 16 vs 32, with 32 clients: 1,139-1,206 vs 1,218-1,250 tok/s, time to first token p50 1.7-1.8 vs
  0.22-0.24 s. 16 was kept for memory: 32 slots of 8,192 positions would take 29 GiB on the 30 GB box.
- Byte-level masking: the non-canonical first tokens (e.g. `S` + `IMPLE` rather than `SIM` + `PLE`) carry at most
  0.2% (0.6B) / 1.1% (1.7B) of the probability mass on the router eval set. It is small, but a trie of
  canonical token sequences would drop it.

## History

Until 2026-10-26, [TODO.md](../TODO.md) was the development log: one section per step with its design decisions,
every measurement (before / after tables, profiles), the test results and the deliberate-breakage checks. It now
lists open work only; the full log is in git: `git show b3e48a3:TODO.md`. The steps and their main commits:

| step | what | commits |
|---|---|---|
| 0a, 0b | repository layout; the test suite (tokenizer and logits vs HF, greedy goldens) | `87e08aa`, `43d5977` |
| 1, 1b | no classifier for prompt tokens; HF-exact tokenizer (pre-tokenizer, BPE with a merge heap) | `6003108`, `da82485` |
| 2 | batched prefill (generic GEMM), bit-identical to token by token; the SwiGLU NaN fix | `e737c02`, `7b2b034` |
| 3a, 3b | AVX-512 VNNI GEMM / GEMV on repacked weights; AVX-512 attention on a tiled KV cache, threading | `21fb5bd`, `219b072`, `cc1c164` |
| 4 | chat history in the KV cache, faster sampling, startup ~0.5 -> ~0.06 s | `9ffdb96`, `39f4cc8`, `86d88d0` |
| 5 | NFC normalizer matching HF tokenizers | `271f2e7` |
| 6a, 6b-1, 6b-2 | GGUF Q8_0 loading; Q4_K_M (layouts, arithmetic, generic kernels); AVX-512 K-quant kernels | `b4180af`, `ca07a0c`, `546033a` |
| 7a, 7b | phase synchronization on a thread pool, attention blocks of 256; 16-bit KV cache (f16 default) | `779b99f`, `fed707e`, `52928cf`, `989f712` |
| 8 | modules and the library API (`src/runq.h`) | `dfed67f`, `1f8bf3b` |
| 9 | `runq serve` (OpenAI API, prefix reuse, constrained decoding) | `69078ea` |
| 10a, 10b | batched decode of several sequences; continuous batching in `runq serve` | `302e0b2`, `38b71d9` |
| 11 | the other Qwen3 dense models (1.7B-8B, 2507), fixes and tests | `fb2b9fb`, `6e1f251` |
| 12 | Qwen3.5-0.8B: Gated DeltaNet + gated attention, recurrent-state checkpoints ([qwen35.md](qwen35.md)) | `99dfa79`, `8d3be20` |
| 13 | Q5_K (Unsloth's Qwen3.5-0.8B Q4_K_M, Qwen3 Q5_K_M files): Q5P layout, generic / AVX-512 / NEON kernels | `fc7fca6` |

The macOS / NEON port (`67d7296` onwards) was developed on a separate branch and merged after Step 7b.
