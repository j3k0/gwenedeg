# TODO

Open work on Gwenedeg: concrete next steps only. Speculative directions are in [IDEAS.md](IDEAS.md), current
limitations in the [README](README.md#limitations), and finished work in the git history (the step-by-step
development log that this file used to hold: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md#history)).

The rules for every change: logits stay bit-identical across paths (batched and sequential, any thread count, SIMD
and generic kernels, a request batched or alone), and a change of numerics must pass the logits-vs-HF thresholds
(`make test`, [tests/README.md](tests/README.md)). Numbers below are from the shared 16-vCPU AMD EPYC Genoa VM of
[BENCHMARKS.md](BENCHMARKS.md) (run-to-run noise ±15-30%) unless noted.

## Platforms

- [ ] **macOS / NEON after Steps 8-11.** The full suite last ran on the Mac before the library split (Step 8); since
  then the code has only been built there (the server's portability fix, `32826f8`). Untested under clang
  `-ffast-math`: the Step 11 fixes in shared C code (RoPE as `expf(log_theta * x)`, rmsnorm's
  `ss * (1.0f / size)`, `src/forward.c`), the batched-decode changes (NEON keeps the GEMM path: no small-batch
  kernels, `bd_max` 0) and `tests/server_test.py`.
  Done when `make && make test`, `make kq-neon-test` and the server and synth groups pass on an Apple Silicon Mac,
  apart from the known greedy goldens (next item).
- [ ] **arm64 baseline for the text goldens.** On arm64, the tests that compare generated text with goldens
  recorded on x86 fail (8 at the last run: seven `greedy/*` and `chat/multiturn_golden`), because the compilers
  round some float code differently; the logits and bitwise checks all pass. Needs per-architecture greedy
  goldens (`--update-greedy` into a separate file, chosen by `uname -m`). Done when `make test` is green on the
  Mac.
- [ ] **NEON small-batch kernels.** Batched decode on AArch64 runs the prefill GEMM, whose per-call tile unpacking
  makes passes of 2-8 rows slow. M1 Pro, 0.6B Q4_K_M, 8 threads (2026-10-25, before the 16-bit KV cache): one
  decode step on the GEMV took 5.4 ms, a GEMM pass of 2 or 3 rows 7.6-8.2 ms. The AVX-512 version (`smb512_rows`
  in `src/kernels_avx512.h`) is 2.3x the GEMM at 1 row and 1.4x at 4. Port it with the same folds, then enable
  the decode-rows worker (`bd_max`). Done when `kq-neon-test` and the api digests are bit-identical and 8
  sequences give >= 3x the single-sequence rate.
- [ ] **x86 without AVX-512 VNNI** (AVX2, AVX-VNNI) runs the generic C kernels: Q4_K_M prefill is ~10x slower and
  decode ~3-4x. AVX2 kernels would have to keep the integer sums exact and the folds in the order of
  `src/kquants.h` and `dot_q8`, so they stay bit-identical to the generic build. Note: `-march=x86-64-v3` builds
  already fail the greedy goldens, because the non-kernel float code is vectorized at another width (they pass the
  logits, batched and thread tests).
- [ ] **Windows.** The code paths inherited from qwen3.c (`src/win.c`, the OpenMP region instead of the pool) are
  untested (no Windows toolchain was available), and `runq serve` is a stub there. A MinGW or MSVC build, plus the
  tests that need no server.

## Performance

- [ ] **Long-context decode: find the real bound.** After 3,000 tokens, decode moves 130-180 GB/s effective (Q8_0
  GGUF, f16 cache: 1.0 GB per token at ~128 tok/s, 8 threads), which is more than the 72-130 GB/s that the
  simple read test (`bench/bw.c`) measures on this VM. Re-measure the bandwidth limit, with a better test such
  as several streams per thread, then profile whether bandwidth or the attention items (8 kv heads x blocks of
  256 positions, after the q / k / v items) bound it.
- [ ] **KV cache with f16 K and bf16 V.** In f16, `vcvtph2ps` costs ~0-4% of prefill attention (1-3% of a
  1,500-3,000-token prefill), while bf16 converts with a shift (almost free). Measured KL divergence from HF fp32
  on 45.6k tokens: bf16 V alone +0.7%, bf16 K alone +4.4% (on 5 of the 13 pieces), both +2.6%, f16 +0.2%
  (noise). A mixed cache would halve the conversion cost. Done when the logits thresholds pass, the KL stays
  within the noise of f16, and the bitwise invariants hold for the new type (add it to the `kv` group).
- [ ] **Prefill attention balance.** Prefill runs all blocks of a query vector in one work item (one item per
  (row block, kv head)); only single-row blocks are split per 256 positions. Items per (row block, kv head, block
  range) would balance long prompts better. Separately, the blocks still cost ~4-5% of the attention at 3,000
  tokens (against a scratch build with 4,096-position blocks: 1.69 vs 1.62-1.64 s). The arithmetic, including the
  combine order of `att_combine`, must stay as it is.

## Server (`runq serve`)

- [ ] **Subset classifier at constrained decisions.** A decision step computes all 151,936 classifier rows, but only
  the allowed tokens matter (~13 for LiteLLM's four tiers). The last-row classifier costs 1.2 ms (8 threads; 1.8 ms
  at 4) of a 16.6 ms cached classification step (20 new tokens after 260 cached), ~7%. It needs a kernel in
  `src/forward.c` that computes selected rows whose bits equal the full classifier's (bitwise test against
  `RQ_LOGITS_LAST`), and the logprobs renormalized over the allowed set, as now.
- [ ] **`json_object` and free-form strings.** `response_format: json_object`, and `json_schema` with `string` (no
  enum), `number` or arrays, get a 400 today (`src/constrain.c` needs a finite set of outputs). A
  byte-level JSON grammar (estimated at ~400 lines) would run beside the trie, with the same byte-level masking
  and the same forced-token logic. Done when every output parses and validates (fuzzed schemas in
  `tests/server_test.py`).
- [ ] **`logprobs` while streaming** (rejected now): send each chunk's tokens with their logprobs in the SSE deltas,
  as OpenAI does.
- [ ] **Test the per-job fallback after a failed pass.** When a batched `rq_forward_batch` fails, `src/engine.c`
  re-runs it segment by segment, so that only the failing job gets its 500 (the passes validate everything before
  computing). No test reaches this path. It needs a fault-injection hook (e.g. an environment variable that fails
  one segment) and a test showing that the other jobs' outputs are unchanged, bitwise.

## Models and formats

- [ ] **More GGUF types**: Q5_K (Q5_K_M and the Unsloth "UD" files), Q4_0 / Q4_1, IQ4_XS, F16 / BF16 weights. Each
  needs its block format, a repacked layout and generic, AVX-512 and NEON kernels that follow one arithmetic
  definition, as in `src/kquants.h`. Tests: dequantization bitwise against gguf-py (`-m dequant`), logits against
  the file's dequantized-weights reference (`tests/make_model_ref.py`), and the bitwise groups. Q6_K-only files
  should already load, but are untested.
- [ ] **Qwen3-14B and larger dense models**: probably work (the Step 11 audit found no other size assumptions), but
  are untested (~9 GB at Q4_K_M). Done when `tests/run_tests.sh --models` covers one.
- [ ] **YaRN RoPE scaling**: contexts past 40,960 tokens for the hybrid Qwen3 models (Qwen documents YaRN up to
  131,072). The 2507 models need none up to 262,144. Files with RoPE scaling are rejected today.
- [ ] **Top-k sampling and per-model sampling defaults**, e.g. 2507: temperature 0.7, top-p 0.8, top-k 20 (Qwen's
  `generation_config.json`; runq has no top-k). The sampler's exact-order rules (`src/sampler.c`) apply.
- [ ] **Qwen3-Embedding / Qwen3-Reranker** (0.6B: the same architecture, vocabulary 151,669): a pooled-embedding
  output mode and the reranker's yes / no score, in the library and as `/v1/embeddings` (and a rerank endpoint).

## Library

- [ ] **Per-model activation group size.** The group size (32 for GGUF, 64 for the `.bin`) is process-wide, so
  `rq_model_load` refuses a second model with another one. Making it a model field touches every kernel call.
  The -Ofast code generation of `src/forward.c` must not change: check with `tests/compare_bins.sh`.
- [ ] **`rq_chat_render` with thinking on** drops the reasoning of every earlier assistant message. HF's template
  keeps it for assistant messages after the last user message, which only matters when a client continues an
  assistant turn. Add the case to `golden/chat_render.json`.

## Benchmarks and docs

- [ ] **A fresh BENCHMARKS.md session.** It is still the 2026-10-25 session (`d76e4cc`, before the Q4_K_M kernels,
  the thread pool, the f16 KV cache, batched decode and the server); later numbers are in the README and
  [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). Rerun `scripts/bench_all.sh`, `scripts/bench_models.py`,
  `scripts/bench_batch.sh` and `scripts/bench_serve.py` with one binary in one session, against qwen3.c and
  mynah-slm as before, and update its text.
