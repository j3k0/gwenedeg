# TODO

Open work on Gwenedeg: concrete next steps only. Speculative directions are in [IDEAS.md](IDEAS.md), current
limitations in the [README](README.md#limitations), and finished work in the git history (the step-by-step
development log that this file used to hold: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md#history)).

The rules for every change: logits stay bit-identical across paths (batched and sequential, any thread count, SIMD
and generic kernels, a request batched or alone), and a change of numerics must pass the logits-vs-HF thresholds
(`make test`, [tests/README.md](tests/README.md)). Numbers below are from the shared 16-vCPU AMD EPYC Genoa VM of
[BENCHMARKS.md](BENCHMARKS.md) (run-to-run noise ±15-30%) unless noted.

## Qwen3.8-27B with speculative decoding (the current target)

(The use it serves: the next section, agentic development at 96k context.)
Decode Qwen3.8-27B (`qwen35`, 64 layers, r = 3, untied, an MTP block) at ~13-18 tok/s on the dev box with
Qwen3.5-0.8B as the draft, the output **exactly** plain decoding's (tokens, greedy and seeded, and each emitted
token's logits bits). Facts, measurements and decisions: [docs/qwen38-speculative.md](docs/qwen38-speculative.md).
Added rule for this section: speculative output == plain output (tokens + a hash of each emitted token's logits).
The preparation is done (2026-10-27: the regression net `tests/compare_bins.sh REF_DIR NEW_DIR` over every model
family incl. Qwen3.5 and the synthetic r = 2 / 3 / 27B-shaped models; two models in one process; checkpoint sets and
`rq_seq_checkpoint_rows`; the memory estimate, `--mem` and the page-cache drop; `rq_gen_start` / `rq_gen_step`, top-k
and `general.sampling.*`; the file's specifics: the MTP block validated and skipped, F32 `ssm_alpha` / `ssm_beta` in
fp32, the 3.8 chat template with its reasoning effort; the verify-pass cost: small segments through the decode-rows
worker, a 4-row segment 1.20-1.29x a decode step), and so is S1, the loop (`rq_generate_spec` / `rq_spec_run`,
`src/spec.c`; CLI `-D` / `-G` / `--tau`; exact on Qwen3 and Qwen3.5, tested in `make test`'s `spec` group; Qwen3-1.7B
+ 0.6B Q4_K_M: 1.15-1.5x at k 3, [docs](docs/qwen38-speculative.md#the-loop-s1)), and so is S7, the 27B bring-up
(Unsloth's Q4_K_M: 16.3 GiB of RSS, the load estimate within 0.3%; logits vs its numpy reference cos 0.9999, top-1
62/62; plain 8.7 tok/s, with the 0.8B Q4_K_M draft 13.2-14.1 tok/s thinking on, 16.6-17.2 off, 8 threads, every output
plain decoding's; a checkpoint per verify row 2.2 ms (S3's cost); the opt-in `3.8-27b-q4km` of the `models` group;
[docs](docs/qwen38-speculative.md#the-27b-bring-up-s7)), and so is S8, the server (2026-10-28: `runq serve --draft FILE`
and / or `--lookup N`, the guesses verified in the engine's batched passes, the draft's steps in its own batched passes,
Qwen3.5's checkpoints per verify row transient next to the slot's G5 budget; outputs exactly plain decoding's,
`server_test.py --spec` in the `spec` group; the auto policy of G7; `timings.speculative`;
[docs](docs/server.md#speculative-decoding-todo-s8-g7)): each step below keeps `make test` green and compare_bins
identical where it promises no output change. **Disk and RAM:** ~29 GB free with the 27B Q4_K_M and Q3_K_M downloaded (`models/Qwen3.8-27B-GGUF`, sha256
checked; 2026-10-28), `models/` ~59 GB; `/tmp` is a 16 GB tmpfs (RAM: no models or venvs there); check `df` / `MemAvailable`
before every download or 27B run, keep >= 15 GB of disk free (the 4B / 1.7B HF checkpoints were removed: they only
regenerate committed goldens, re-download their pinned sets), and use `flock /tmp/cwayn-bench.lock` for timings
(several agents share the box).

Steps:

- [ ] **S5. The speculative loop on Qwen3.5.** Target Qwen3.5-4B Q8_0 + draft 0.8B (~6 GB), a checkpoint per row on both.
  Expect ~1.3-1.6x (a 0.8B guess costs ~18% of a 4B token). The loop already handles hybrid sequences (S1: a
  checkpoint per verify row on the target and per draft step on the draft, k + 2 slots, the target's checkpoints
  restored afterwards; `spec_api/qwen35/*` on the 0.8B Q8_0 + Q4_K_M pair, every rollback exact, and the CLI's
  `spec_cli/q35_*`). Left: `api_test --spec` and the CLI cases on the 4B + 0.8B by hand (both builds), the
  tok/s and acceptance there (with the copy cost per row, `bench_verify -c`), and the default k / tau for it (and for
  the 27B: `-G 6 --tau 0.4` was +4-7% over the default `-G 3` there). Done when outputs equal
  plain decoding bitwise (greedy, seeded, thinking on / off) and tok/s + acceptance are recorded.
- [ ] **S9. The MTP drafter** (`blk.64`, same file; validated at load into `RqModel.mtp`, `MtpBlock`: pointers into
  the mapping, to convert / repack when the drafter is built). The target's last hidden state + the next token's embedding ->
  `eh_proj` -> one full-attention layer with its own KV cache -> the shared output (~9-10 ms per guess for the 1 GB
  classifier; a vocabulary subset is fine, exactness does not care which guesses). Needs the final hidden state per
  row from the verify pass (a new `rq_forward_batch` output). Measure acceptance first; keep whichever draft gives
  more tok/s (the 0.8B Q4_K_M at S7: ~6.5 ms per guess, 13.2-14.1 tok/s thinking on, 16.6-17.2 off).
- [ ] **S10. Seed-coupled drafting at temperature > 0.** The draft samples its guess with the target's next uniform
  draw u_j (known from a copy of the target's RNG: `rq_seq_rng` + `rq_rng_f32`) and the same settings, instead of its
  argmax: more accepted guesses (up to sum min(p, q)), output unchanged. Measure first with the API's `top_logprobs`
  = 20 (`scripts/spec_accept.py gen --top-logprobs 20`). Top-k is in the sampler (`rq_sample`'s exact order,
  `src/runq.h`). The guess is `draft_pick` in `src/spec.c` (S1: argmax; at temperature 1, k 3, the 1.7B accepted 64%
  of the 0.6B's guesses against 77% greedy).

RAM for the 27B runs (decided 2026-10-27): the parcolab, billing and triominos-server containers are stopped
for the runs (`docker compose -p NAME stop`; `start` brings them back), which leaves ~24 GB available: the Q4_K_M
file (measured at S7: 16.3 GiB of RSS alone, 17.4 GiB with the draft and its checkpoints; `api_test` on it alone
22.4 GiB) at short and medium contexts; the Q3_K_M (`3.8-27b-q3km`, 13.3 GB of RSS) for the 96k sessions with the
draft (G3, next section). Keep other jobs small while a 27B run is going.

## Agentic development at 96k context (pi), on Qwen3.8-27B

The use the 27B work is for: an agent harness such as pi (`openai-completions` provider: streamed chat completions
with tools, `reasoning_effort` / `chat_template_kwargs`, usage in the stream) driving `runq serve` with Qwen3.8-27B
and its draft. A context window of **96k tokens** is the minimum. A session grows by **2-3k tokens per turn** (tool
results, file contents) followed by a reply (reasoning, then a tool call or a code edit). When the context nears
the window, pi compacts: it summarizes the old messages in a separate request (the conversation serialized as text,
reasoning omitted, tool results cut to 2,000 characters: no prefix to reuse) and continues with the system prompt +
the summary + the last `keepRecentTokens`. Every item keeps the project's rules (bitwise paths; speculative == plain).

Where it stands (measured 2026-10-28 on the 27B: G3's gate, G0's replay and long-context costs, G2's retrieval, all done; the setup and its numbers in docs/server.md, "Qwen3.8-27B
with pi"; the measurements in docs/qwen38-speculative.md, "Q3_K_M: the quality gate", "Long-context costs", "Agent
sessions"):
- **Memory: fits.** The agentic file is the **Q3_K_M** (G3's gate: top-1 = the API model's 96.5 / 94.5% thinking off /
  on against the Q4_K_M's 98.4 / 96.9%, perplexity +2.7%, KL 0.029 from the Q4_K_M on pi sessions' replies): runq's
  estimate at `-c 98304` with 3 checkpoints 19.45 GiB + the 0.8B draft at that context 2.03 + the verify rows'
  checkpoints 0.66: ~22.1 GiB of ~24 available. The Q4_K_M: ~25.1 GiB with the draft (no), ~23.0 without it (~1 GiB
  to spare).
- **Prefill per turn** (16 threads, `-k bf16-dot`): P / 67.5 + 0.19 us x P x (L + P / 2) s, within 0-11% of every replayed
  prefill: a 2.5k-token chunk at ~90k ~80 s (8 threads ~140 s); a 2,560-token chunk at 98k 86 s (f16: 123 s).
- **Decode**: plain 6.9 tok/s at 98k (16 threads; 5.7 at 8), 8.5-9.3 at 16-32k; with speculation in the server 8.1-13.9
  tok/s at 21-81k (the lookup; the replay's probes), 15-23 at 2-3k (the draft; a live pi session).
- **Compaction** (one at 84k, `keepRecentTokens` 8,000): ~12.4 min (a 30k-token summary request 8 min, the next request
  from 0 4 min, its system block included: one slot).
- **Tools:** a live pi session on the 27B (read / bash / edit / write, thinking medium) finished a small coding task (a
  function, its tests, the test run: 7 / 7 pass) in 6 requests and ~80 s.

Goal, against the measurements: a 96k session with the draft fits with >= 2 GiB to spare (~2 GiB: met, barely); a
2.5k-token turn at 90k prefills in ~1 min (~80 s at 16 threads); decode >= 8 tok/s at 90k with speculation (8.1-8.7 at
66-81k when the lookup matches, 6.9 plain at 98k: not on reasoning text); no turn re-prefills more than its new tokens
plus the last reply (met: 46 of 52 tool turns exact, none beyond).

- [ ] **G3 leftovers: memory beyond the Q3_K_M.** (The quality gate is done: the Q3_K_M is the agentic file, the
  Q4_K_M the reference and the choice where it fits; docs/qwen38-speculative.md, G3.) The ~2 GiB to spare are thin on a
  shared box, and the Q4_K_M at 96k with its draft needs ~3 GiB more:
  - An 8-bit KV cache: V first (its bf16 rounding was harmless), then K, which needs per-tile scales: |K| reaches
    ~500 and bf16 K alone already cost +4.4% KL against HF fp32. Each variant judged by the KL measurement of the 16-bit
    types and G2's checks (its baseline: the 16-bit numbers of "KV cache with f16 K and bf16 V", Performance);
    generic, AVX-512 and NEON kernels on one definition. Halves the KV (3 GiB at 96k) and its read per token (at 98k
    the decode step's attention part reads 6.4 GB in ~63 ms at 16 threads: ~100 GB/s, the bound there).
  - The draft's own cost: its sequence at 96k (1.23 GiB) and its attention there. Its guesses never change the output,
    so it can run on a shorter recent window (measure the acceptance lost). At long context the server used only the
    lookup in the replay (G7 (a)): if a live session confirms the draft adds nothing past ~32k, a draft window of ~32k
    frees ~0.8 GiB.
  - The checkpoint budget: 150 MiB each for the 27B; `--checkpoints 3` is enough for Qwen3.8 (the header, the system
    block, the reply's end).
- [ ] **G4 leftovers: prefill attention against a long cache.** (a) and (b) are in (`082511f`, `-k bf16-dot` as an
  option; docs/qwen38-speculative.md, "G4", "G4 (b)"). On the 27B (G0's run, the Q3_K_M): the 2,560-token chunk at 98k
  123 -> 86 s at 16 threads, 208 -> 148 s at 8 (1.41-1.43x the whole chunk); G2's retrieval at 88k passes with it. Left
  for the goal's ~1 min: the matmul part (38 of the 86 s at 16 threads; the Q3_K kernels' prefill is ~6% slower than
  the Q4_K_M's), an fp32-free softmax (a change of definition), and the KL at 64k-98k positions (the long group's golden
  keeps 576 ids per position). Not the default: `bf16-dot` stays opt-in (docs: the setting for the 27B's sessions).
- [ ] **G5 leftovers: re-prefill and compaction.** G5 is done (docs/server.md and docs/qwen38-speculative.md,
  "Agent sessions"). Measured on the 27B (Q3_K_M, 16 threads): every tool turn prefills its new tokens, or those plus
  the previous reply's last few; a compaction at 84k with `keepRecentTokens` 8,000 costs ~12.4 min. Left: (a) the reply's
  own tokens for its span of the next prompt when the text matches (no re-prefill after a non-canonical sampled token
  or an empty reasoning; but the prompt is then not the canonical tokenization and the output depends on what was
  cached: opt-in only, if at all); (b) a pinned system-block cache outside the slot pool for one-slot setups (the 27B
  at 96k has one slot: the summary request takes it and the next request prefills the system block again, ~2 of the
  12.4 min; ~0.7 GiB for 8k tokens, which the memory above does not have yet); (c) pi-side: a
  `session_before_compact` extension summarizing without reasoning or with a smaller model (pi has no per-task model
  setting); `keepRecentTokens` 8,000 is in the documented setup.
- [ ] **G6 leftovers: prompt lookup.** Prompt-lookup drafting is in (`--lookup N [--lookup-min M]`, with or without
  `-D`; exact; docs/qwen38-speculative.md, "Prompt lookup": ~1.5 tokens per pass on agentic replies, 2.8-3.9x on edits
  that return the prompt's code; in the server and chosen per pass since S8 / G7; on the 27B's replay 47% of its guesses
  accepted at 21-81k). Left: tool schemas whose old / new text are plain string parameters (pi's `edits` array is
  JSON-escaped, so multi-line old text does not match the raw file text in the context: a pi-side change or a custom
  edit tool).
- [ ] **G7 leftovers: speculation at long context.** In (2026-10-28, docs/qwen38-speculative.md, "In the server (S8)
  and the choice by position (G7)"): `runq serve --spec auto` and `runq --spec auto`. On the 27B (Q3_K_M, `bf16-dot`, 16
  threads): `bench_long -D`'s choice from the measured passes, the model k 3 up to 32k (x1.38-1.49), k 1 at 64-98k
  (x1.24 / x1.17), a 4-token lookup match 3 guesses (x1.47-2.10); in the server the replay's probes (256 tokens) at
  21-81k used the lookup only (805 / 1,699 accepted, 8.1-13.9 tok/s), a live session at 2-3k the draft (15-23 tok/s).
  Left: (a) the draft at 64-98k in a live session (the probes' `max_tokens` of 256 never paid for its catch-up of
  21-81k tokens; pi's 16,384 should: measure its gain there, and the acceptance on the 27B's own agentic replies (G0
  (c)'s proxies were other models' replies)); (b) the calibration: the attention slope at the default 8,192 positions
  is within the noise of the 27B's ~90 ms step (4 `bench_long` runs: 0.38-2.40 us per position for the first row against
  ~0.6 measured), and twice (once at `--spec-calibrate 32768`) the measurement fell under 10% of the estimate and the
  estimate was used (2.40 us: KV at 60 GB/s + 0.3 TFLOPS, ~4x the 27B's); calibrate at the target's context, take the
  slope from several positions, or fit the estimate's constants per box; and the online correction scales each model's
  whole cost by one ratio, so f(R)'s shape and the slope stay as calibrated; (c) the draft's prompt catch-up runs with
  the prefill (+~20 ms TTFT on short prompts with the 0.6B draft, more on long ones): defer it, or bound it by the
  expected reply length as the decode-time catch-up is; (d) adaptive length in auto (the draft's top-1 probability per
  guess, as tau, instead of the geometric p only); (e) S10's seed-coupled draft would raise p at temperature > 0, which
  the policy would then pick up.

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
  256 positions, after the q / k / v items) bound it. The 27B (G0, 2026-10-28, `bench_long`, the Q3_K_M, `bf16-dot`):
  a decode step at 98k is 146 ms at 16 threads against 83 at 512 positions, 175 / 100 ms at 8: the 6.4 GB of KV read in
  ~63 / ~75 ms, ~100 / ~86 GB/s (with the weights' 13 GB, ~133 / ~110 GB/s per step): the read is the bound there.
- [ ] **KV cache with f16 K and bf16 V.** In f16, `vcvtph2ps` costs ~0-4% of prefill attention (1-3% of a
  1,500-3,000-token prefill), while bf16 converts with a shift (almost free). Measured KL divergence from HF fp32
  on 45.6k tokens: bf16 V alone +0.7%, bf16 K alone +4.4% (on 5 of the 13 pieces), both +2.6%, f16 +0.2%
  (noise). A mixed cache would halve the conversion cost. Done when the logits thresholds pass, the KL stays
  within the noise of f16, and the bitwise invariants hold for the new type (add it to the `kv` group).
- [ ] **Prefill attention balance.** Prefill runs all blocks of a query vector in one work item (one item per
  (row block, kv head)); only single-row blocks are split per 256 positions. Items per (row block, kv head, block
  range) would balance long prompts better. (Measured for G4, a 512-row pass at 98k on synth-27b: the threads are
  busy 98.3% of the attention phase at 8 threads, 94.8% at 16; the passes that would gain are short ones, e.g. a
  turn's last few hundred rows: 100 rows = 17 row blocks x 4 kv heads on the 27B, 68 items for 16 threads.)
  Separately, the blocks still cost ~4-5% of the attention at 3,000 tokens (against a scratch build with
  4,096-position blocks: 1.69 vs 1.62-1.64 s). The arithmetic, including the combine order of `att_combine`, must
  stay as it is.

## Server (`runq serve`)

- [ ] **Tool calls, after G1.** (a) Stream a call's arguments as they are generated: each call goes out whole when its
  `</function>` arrives, so a long `write` / `edit` call shows no progress until it is complete (OpenAI streams the
  arguments' JSON in pieces; the typed conversion needs a parameter's end, a string parameter could stream its text).
  (b) `tool_choice: "required"` / a named function (a 400 today): force `<tool_call>\n<function=` (+ the name) as
  constrained decoding forces tokens; `none` could also mask the `<tool_call>` token (today the model may still write
  the XML, returned as content). (c) Constrain the parameter values by the tool's schema while decoding (enums,
  booleans, integers: `src/constrain.c`'s byte-level masking inside a `<parameter=...>` block; free strings need the
  JSON grammar below). (d) Qwen3's older templates (Qwen3 0.6B-8B, 2507: JSON `{"name": ..., "arguments": ...}` inside
  `<tool_call>`): another renderer and parser, a 400 today. (A multi-turn pi session on the 27B finished its edit task
  on 2026-10-28: docs/server.md, "Qwen3.8-27B with pi".)

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

## Qwen3.5 (hybrid: Gated DeltaNet + gated attention)

Qwen3.5-0.8B (Q8_0 and Unsloth's Q4_K_M) and Qwen3.5-4B (Q8_0: more value heads than key heads) run and are tested
(`make test`'s `q35` group, the 4B in the `models` group); design, results and measurements in
[docs/qwen35.md](docs/qwen35.md). Untested but expected to load: 2B, 9B, 27B / Qwen3.6-27B (Q8_0 or Unsloth's Q4_K_M /
Q6_K files; 9B Q8_0 is 9.5 GB) and the 4B's Q4_K_M (Qwen3.8-27B Q4_K_M, the same architecture: tested, S7). The rules stay: `make test` green, Qwen3's outputs byte-identical
(`tests/compare_bins.sh`), the bitwise invariants on the new paths.

- [ ] **NEON check of the recurrence.** The blocked recurrence (`dn_prep` / `dn_rec` / `dn_out`, docs/qwen35.md) is
  plain C vectorized along v (no AVX-512 specifics; `DN_UNROLL` is `_Pragma("clang loop unroll(full)")` for clang),
  so it should be bit-identical to the generic build on arm64 too, and as fast as the new layouts allow; untested
  (the q35 group's AVX-512 vs generic checks, `tests/check_models.py`, on the Mac; then `-DRUNQ_PROFILE` for the
  split). Speed left: `dn_prep` scales poorly past 8 threads and `dn_rec` is ~2x its hot-cache cost (docs/qwen35.md).
- [ ] **MoE** (not now: the Qwen3.8-27B section first) (`qwen35moe`: Qwen3.5 / 3.6-35B-A3B and up; images are out of scope for this project, ideas:
  [IDEAS.md](IDEAS.md#images-qwen35-vision)).
  - Qwen3.6-35B-A3B and a tiny test model: router `ffn_gate_inp` (F32), experts
    `ffn_{gate,up,down}_exps` (3-D tensors), a shared expert `ffn_*_shexp` with its sigmoid gate
    `ffn_gate_inp_shexp`; softmax, top-8, always renormalized (as `qwen3moe`, which has no shared expert). The 35B's
    Q4_K_M file is 22 GB: runq copies and repacks all weights into RAM today, so the experts (19.5 GB) would be read
    in place from the mmap'd file (`ffn_down_exps` is Q5_K, supported now). Estimate: ~2.6 GB read per token, 30-40 tok/s
    with everything cached, a few tok/s on the 30 GB dev box. Reference: a numpy forward that dequantizes only the
    routed experts (HF's 67 GB bf16 is out of reach). Its linear layers have 16 key / 32 value heads, as 4B (done).
    Order: a tiny MoE (random-init `qwen3_5_moe`, converted by llama.cpp's converter: docs/qwen35.md design 8 did the
    same for the head mapping), then the 35B.

## Models and formats

- [x] **Q2_K and the i-quants** (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS; 2026-10-30): Underdog Saluki
  27B 1.0 (`saluki-27b`, 7.9 GB) and the Unsloth "UD" files made of them load and run; [docs/iquants.md](docs/iquants.md).
- [ ] **i-quant prefill**: a tiled GEMM for the types of `src/iquants.h` (each group is decoded once per 8 activation rows
  today, `iq_rows`; no tiles, no small-batch kernels): Saluki prefills at ~0.72x the 27B Q3_K_M's tok/s (30 / 54 vs
  42 / 74 at 8 / 16 threads). Decode the 16-row block's codes once into a scratch tile (as `kq_unpack_tile`) and run
  more activation rows per pass; maybe 32-row tiles sharing the activation broadcasts.
- [x] **NEON kernels for the i-quants** (2026-10-30; `iqn_*` in `src/iquants.h`, `make iq-neon-test`): bit-identical to
  `iq_dot`; Saluki on an M1 Pro (8 threads): decode 0.6 -> 4.7 tok/s, prefill ~1 -> 15.8 tok/s (docs/iquants.md).
- [ ] **NEON i-quants, faster.** GEMV is instruction-bound (~6 IPC at ~1 instruction per weight, the codebook gathers
  and the sign masks; 97% of Saluki's decode on an M1 Pro is these kernels). Open: (a) IQ3_S, 0.095 ns/weight vs IQ3_XXS's
  0.067: its index bit 8 costs ~2.5 scalar instructions per entry; building the 9-bit indices in NEON and moving them over
  (umov, or a store and loads) was slower; storing them as u16 at load would need a layout of its own (+0.5 bit per
  weight) or a change of the shared one; (b) a NEON prefill GEMM (decode a 16-row block once into a scratch tile for all
  activation rows, as `kqn_unpack_tile`), with the AVX-512 one of the item above; (c) gcc on AArch64 Linux (Graviton) is
  untested (built and checked with Apple clang only).
- [ ] **The Saluki MTP variant** (`Kujira/Underdog-Saluki-27B-1.0-MTP-GGUF`: Saluki + an MTP block) once S9 runs MTP
  blocks: a self-drafting 7.9 GB 27B.
- [ ] **More GGUF types**: Q4_0 / Q4_1, IQ4_NL, F16 / BF16 weights (Unsloth's Qwen3.5-0.8B UD-IQ3_XXS has F16 matrices).
  Each needs its block format, a repacked layout and generic, AVX-512 and NEON kernels that follow one
  arithmetic definition, as in `src/kquants.h` / `src/iquants.h` (Q5_K and Q3_K, done, are the templates: a new bit
  plane on Q4_K's fold, resp. 3-bit values with a bit plane on Q6_K's; the i-quants: an integer scale per sub-block). Tests:
  dequantization bitwise against gguf-py (`-m dequant`, `tests/make_dequant_golden.py`), logits against the file's
  dequantized-weights reference (`tests/make_model_ref.py`), and the bitwise groups. Q6_K-only files should already
  load, but are untested.
- [ ] **Qwen3-14B and larger dense models**: probably work (the Step 11 audit found no other size assumptions), but
  are untested (~9 GB at Q4_K_M). Done when `tests/run_tests.sh --models` covers one.
- [ ] **YaRN RoPE scaling**: contexts past 40,960 tokens for the hybrid Qwen3 models (Qwen documents YaRN up to
  131,072). The 2507 models need none up to 262,144. Files with RoPE scaling are rejected today.
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

## Distribution and model files

- [ ] **A model directory convention** (today: an explicit path, `./models`, or `$GWENEDEG_MODELS`, which only
  `scripts/download_models.py` reads; the binaries read neither). Proposal, XDG rather than a `~/.gwenedeg/` of our
  own: the cache dir is `${HF_HOME:-${XDG_CACHE_HOME:-~/.cache}/huggingface}/hub` (HF's own, see below) and
  `${XDG_CACHE_HOME:-~/.cache}/gwenedeg/models` second, with `$GWENEDEG_MODELS` overriding both for set
  resolutions; a bare set name on the command line (`runq qwen3.5-0.8b-q8 -i hi`) resolves as: a path, a file in
  `$GWENEDEG_MODELS`, then `./models`, then the cache dirs in order. In the HF cache, a set finds its file in
  `models--<org>--<repo>/snapshots/<rev>/<file>` under any revision (verify the content: the LFS symlink points to
  `../blobs/<sha256>`, whose name is the hash; else hash it against the pinned one), so anything the user already
  pulled with `hf download` is used as-is, offline and without re-verification of blobs named by their hash.
  `runq models` lists what is there (sets found in each location, files, sizes, and what each needs to download).
  Fetching stays out of the binaries (no TLS in them): `scripts/download_models.py` keeps the pinned sets and uses
  the `hf` CLI when available (`hf download <repo> <file> --revision <sha>`, resumable, writes the HF cache; it
  verifies nothing, so still check the sha256 before linking), else plain curl + sha256sum into the gwenedeg cache
  dir; the same logic as a `scripts/fetch.sh` (no Python needed). Done when a bare set name resolves in every
  location, `hf`-downloaded files are found and used without copying, `runq models` prints the table, and the docs
  (`README.md`, `scripts/download_models.py --list`) agree.
