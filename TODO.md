# TODO

Open work on Gwenedeg: concrete next steps only. Speculative directions are in [IDEAS.md](IDEAS.md), current
limitations in the [README](README.md#limitations), and finished work in the git history (the step-by-step
development log that this file used to hold: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md#history)).

The rules for every change: logits stay bit-identical across paths (batched and sequential, any thread count, SIMD
and generic kernels, a request batched or alone), and a change of numerics must pass the logits-vs-HF thresholds
(`make test`, [tests/README.md](tests/README.md)). Numbers below are from the shared 16-vCPU AMD EPYC Genoa VM of
[BENCHMARKS.md](BENCHMARKS.md) (run-to-run noise ±15-30%) unless noted.

## Qwen3.8-27B with speculative decoding (the current target)

Decode Qwen3.8-27B (`qwen35`, 64 layers, r = 3, untied, an MTP block) at ~13-18 tok/s on the dev box with
Qwen3.5-0.8B as the draft, the output **exactly** plain decoding's (tokens, greedy and seeded, and each emitted
token's logits bits). Facts, measurements and decisions: [docs/qwen38-speculative.md](docs/qwen38-speculative.md).
Added rule for this section: speculative output == plain output (tokens + a hash of each emitted token's logits).
Order: the preparation items (P1-P5) first, each a no-behaviour-change step checked by the regression net of P1, then
the steps (S). **Disk and RAM:** ~44 GB free, `models/` 42 GB, `/tmp` is a 16 GB tmpfs (RAM: no models or venvs
there); check `df` / `MemAvailable` before every download or 27B run, keep >= 15 GB of disk free (58 GB free after
removing the 4B / 1.7B HF checkpoints, which only regenerate committed goldens: re-download their pinned sets), and use
`flock /tmp/cwayn-bench.lock` for timings (several agents share the box).

Preparation (no output changes; `make test` and `tests/compare_bins.sh` byte-identical after each):

- [ ] **P1. The regression net covers Qwen3.5.** `tests/compare_bins.sh` has only Qwen3 jobs; the Qwen3.5 changes so
  far were compared by hand (40-180 files each time). Add the Qwen3.5-0.8B Q8_0 / Q4_K_M files and the synthetic
  r = 2 / 3 models (`tests/make_synth_q35.py`): the `-m logits` grid (-b 1 / 7 / 64 / 512, 3 / 8 threads, both
  builds), greedy and seeded generate, a `-r 1` chat that rolls back to a checkpoint (`RUNQ_CHAT_DUMP`), and
  `api_test --digest` (batched passes). Skipped if a file is missing; `--quick` stays short. Done when one command
  checks every model family against a base commit, and a deliberately broken DeltaNet build is caught.
- [ ] **P2. Two models in one process, tested.** The pool, the group size `GS` and `rq_nt` are process-wide
  (`src/threads.c`, `src/model.c`); a draft and a target will alternate passes on one pool. An `api_test` case:
  load two models (Qwen3-0.6B + 1.7B; Qwen3.5-0.8B + a synthetic r = 3 model), interleave their prefills, decodes and
  batched passes: each context's logits bitwise == the model alone. Also check the error paths (a second model with
  another GS, freeing one while the other runs). Done when it passes on both builds.
- [ ] **P3. Checkpoints as a set, not two fixed slots.** `RQ_SEQ_CK` = 2 and one pending `ck_at` per pass
  (`src/internal.h`, `ck_plan` / `ck_done` in `src/forward.c`). Refactor to a per-sequence set of n slots (a context
  setting, default 2: today's behaviour) with a sorted list of pending positions per pass, each (layer, head) item
  copying its part after each of them; `rq_seq_truncate` / `copy` / `keep` unchanged for the current callers.
  Prepares S3 (a checkpoint per verified row). Done when the q35 group, the API tests and P1 are byte-identical,
  plus an `api_test` case with 3 pending positions in one prefill chunk (truncate to each continues bitwise).
- [ ] **P4. Memory accounting and the page cache.** (a) `runq serve`'s per-slot budget (`src/engine.c`) counts only
  the KV cache: add the recurrent state and its checkpoints (~0.45 GB per 27B slot). (b) Print a load-time estimate
  (repacked weights, KV reserve, state + checkpoints per sequence) and refuse to load past `MemAvailable` unless
  forced (`--mem`), so a 27B run cannot OOM the box's other services. (c) After the repack, drop the file's page
  cache (`posix_fadvise(POSIX_FADV_DONTNEED)`; the loader `pread`s). (d) Record load peak RSS for the 8B Q4_K_M and
  the 4B Q8_0 as the baseline (tests/README or BENCHMARKS). Done when the estimates are within ~5% of the measured
  RSS and outputs are unchanged.
- [ ] **P5. Sampling as a reusable step, then top-k.** Factor `rq_generate`'s per-token body (sample with the
  sequence's RNG, the stop / limit checks, the `on_logits` hook) into one function that the speculative loop calls
  too, so its RNG pattern (one draw per prefilled token but the last, one per emitted token) is shared by
  construction, not copied; expose a logits-hash hook in `src/runq.h` for the equivalence tests. Then top-k and the
  per-model defaults from `general.sampling.*` (Qwen3.8: 1.0 / 0.95 / 20; the older item "Top-k sampling and
  per-model sampling defaults" is folded in here), with the sampler's exact-order rules. Done when the refactor is
  byte-identical (P1) and top-k matches a numpy reference of the sampler on recorded logits.

Steps:

- [ ] **S0. Keep the measurement tooling.** Move `logs/specmeas-2026-10-27/{prompts,gen,measure}.py` into
  `scripts/spec_accept.py` (endpoint, key file and models configurable; the 3.8 rendering with the reasoning-effort
  line). Measure: the 0.8B Q4_K_M as the draft; adaptive draft length (stop when the draft's top-1 probability < tau:
  changes the guesses, never the output); real prompts and longer reasoning. Done when it reproduces the table of
  docs/qwen38-speculative.md from the saved JSON.
- [ ] **S1. The speculative loop on Qwen3** (no recurrent state). Target Qwen3-1.7B, draft Qwen3-0.6B.
  `rq_generate_spec(ctx_t, seq_t, ctx_d, seq_d, tokens, n, cfg, max_new, k, cb, st)`: draft k tokens greedily on
  `seq_d`, verify `[last, d1..dk]` as one segment on `seq_t` (`RQ_LOGITS_ALL`), sample the rows in order with P5's
  step, accept while sample == draft (the first mismatch's sample is the next token), truncate both sequences.
  Refuse a draft whose vocabulary or token bytes (hash) differ. CLI `-D draft.gguf -K k`; stats in `[timing]`. Done
  when, greedy and seeded, thinking on / off: tokens == `rq_generate`'s, each emitted token's logits hash == plain
  decoding's, `-K 0` is plain decoding; in `make test` on the 0.6B pair.
- [ ] **S2. Verify-pass cost** (measure first). A segment of 2-8 rows of one sequence takes the prefill path
  (`rq_forward_batch`: `n == nseg` selects the decode rows). Time it against one decode step and against the same
  number of decode rows of different sequences. If small segments are slow, route them through the small-batch
  kernels, keeping causal attention inside the segment (`batch_mixed`-style bitwise tests). Done when f(rows) is
  recorded and a 4-row segment costs <= ~1.3x one decode step (Qwen3-1.7B, Qwen3.5-4B).
- [ ] **S3. Rollback for hybrid models: a checkpoint per verified row** (on P3). `rq_seq_checkpoint_rows(s, pos, n)`:
  the next pass checkpoints after each of rows pos..pos+n-1; `rq_seq_truncate(s, L)` lands exactly on any of them.
  Cost: 27B ~150 MiB per row (0.6 GB at k = 3, ~4-5 ms of copying per ~130 ms pass); 0.8B 18.9 MB per token. Later,
  only if profiles show it: swap state buffers instead of copying back, or keep each row's DeltaNet inputs (~2 MB per
  row for the 27B) and replay from the pre-pass checkpoint. Done when truncating to any position inside the last
  segment continues bitwise like an uninterrupted run (`api_test`, 0.8B and 4B).
- [ ] **S5. The speculative loop on Qwen3.5.** Target Qwen3.5-4B Q8_0 + draft 0.8B (~6 GB), S3's rollback on both.
  Expect ~1.3-1.6x (a 0.8B guess costs ~18% of a 4B token). Done when outputs equal plain decoding bitwise (greedy,
  seeded, thinking on / off) and tok/s + acceptance are recorded.
- [ ] **S6. Qwen3.8 specifics** (on `tests/make_synth_q35.py`'s `synth-q35-27b-mtp.gguf` and a tiny random-init
  `qwen3_5` with an MTP layer, converted by llama.cpp's converter in a scratch venv off `/tmp`). Loader: `block_count`
  65 with `nextn_predict_layers` 1 = 64 main layers + the MTP block (loaded for S9, else skipped) instead of today's
  refusal (`load_model`; the `q35/synth/reject_mtp` check becomes a load whose logits == `synth-q35-27b.gguf`'s,
  bitwise); and F32 `ssm_alpha` / `ssm_beta` matrices ([48, 5120] in the 27B file, Q8_0 in the 0.8B's:
  `gguf_matrices` refuses type 0 today; an fp32 matvec, checked on a synthetic model with F32 ones; the file layout:
  docs/qwen38-speculative.md). Chat variant for 3.8, detected by its text (`reasoning_effort`): thinking on by
  default, the effort's system line merged with the system / developer message, `-r` plus an effort option, the server's
  `reasoning_effort`; the template in `tests/golden/templates/` (license noted). Done when `-m render` equals HF's
  `apply_chat_template` for thinking off / on x each effort, with and without a system message.
- [ ] **S7. 27B bring-up** (after P4; Q4_K_M). Download the pinned file with `hf` (into
  `models/`, sha256 checked); load: peak RSS, page cache dropped. Logits vs `tests/ref_qwen35.py` on the dequantized
  weights (layer by layer, a few GB; a classifier subset); teacher-forced top-1 vs the API's model on S0's outputs
  (a sanity check). Speed: plain decode (~120 ms per token), then with the 0.8B draft (thinking on: ~13-14 tok/s;
  off: ~17). Not in `make test` (an opt-in group, like `models`). Done when the
  numbers are in the README and docs, and the 27B's speculative output equals plain decoding bitwise.
- [ ] **S8. Server.** `runq serve --draft FILE`: the draft rows join the job's segment in the batched pass, k per
  request; speculate less as the batch fills. Done when `server_test.py` shows equal outputs with and without the
  draft, and `scripts/bench_serve.py` numbers at 1 / 4 clients.
- [ ] **S9. The MTP drafter** (`blk.64`, same file). The target's last hidden state + the next token's embedding ->
  `eh_proj` -> one full-attention layer with its own KV cache -> the shared output (~9-10 ms per guess for the 1 GB
  classifier; a vocabulary subset is fine, exactness does not care which guesses). Needs the final hidden state per
  row from the verify pass (a new `rq_forward_batch` output). Measure acceptance first; keep whichever draft gives
  more tok/s.
- [ ] **S10. Seed-coupled drafting at temperature > 0.** The draft samples its guess with the target's next uniform
  draw u_j (known from a copy of the target's RNG) and the same settings, instead of its argmax: more accepted
  guesses (up to sum min(p, q)), output unchanged. Measure first with the API's `top_logprobs` = 20 on S0's outputs.
  Needs P5's top-k and the same candidate order in both models.

RAM for the 27B runs (decided 2026-10-27): the parcolab, billing and triominos-server containers are stopped
for the runs (`docker compose -p NAME stop`; `start` brings them back), which leaves ~24 GB available: the Q4_K_M
file (~19.5 GB RSS with the draft and checkpoints), no Q3_K needed. Keep other jobs small while a 27B run is going.

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

## Qwen3.5 (hybrid: Gated DeltaNet + gated attention)

Qwen3.5-0.8B (Q8_0 and Unsloth's Q4_K_M) and Qwen3.5-4B (Q8_0: more value heads than key heads) run and are tested
(`make test`'s `q35` group, the 4B in the `models` group); design, results and measurements in
[docs/qwen35.md](docs/qwen35.md). Untested but expected to load: 2B, 9B, 27B / Qwen3.6-27B (Q8_0 or Unsloth's Q4_K_M /
Q6_K files; 9B Q8_0 is 9.5 GB) and the 4B's Q4_K_M. The rules stay: `make test` green, Qwen3's outputs byte-identical
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
- [ ] **Checkpoints, measured**: `RQ_SEQ_CK` = 2 per sequence (18.9 MB each) at the prompt's assistant header
  (docs/qwen35.md). Not measured on real traffic: how often `runq serve`'s requests roll back further (a
  re-prefill of the whole conversation), and whether a checkpoint at the end of each reply would pay off.

## Models and formats

- [ ] **More GGUF types**: Q4_0 / Q4_1, Q3_K / Q2_K (the Unsloth "UD" files of the small bit widths), IQ4_XS, F16 / BF16
  weights. Each needs its block format, a repacked layout and generic, AVX-512 and NEON kernels that follow one
  arithmetic definition, as in `src/kquants.h` (Q5_K, done, is the template: a new bit plane on Q4_K's fold). Tests:
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
