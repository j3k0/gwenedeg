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
[docs](docs/qwen38-speculative.md#the-27b-bring-up-s7)): each step below keeps `make test` green and compare_bins
identical where it promises no output change. **Disk and RAM:** ~40 GB free with the 27B Q4_K_M downloaded (`models/Qwen3.8-27B-GGUF`, sha256
checked), `models/` 45 GB; `/tmp` is a 16 GB tmpfs (RAM: no models or venvs there); check `df` / `MemAvailable`
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
- [ ] **S8. Server.** `runq serve --draft FILE`: the draft rows join the job's segment in the batched pass, k per
  request; speculate less as the batch fills. (`rq_spec_run` is one sequence's loop on its own passes: the engine
  needs the same round split into steps of its scheduler: draft rows, then the verify segment in the shared pass,
  then `rq_gen_step` per row and the truncations, as `src/spec.c` does them.) Done when `server_test.py` shows equal outputs with and without the
  draft, and `scripts/bench_serve.py` numbers at 1 / 4 clients.
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
22.4 GiB), no Q3_K needed. Keep other jobs small while a 27B run is going.

## Agentic development at 96k context (pi), on Qwen3.8-27B

The use the 27B work is for: an agent harness such as pi (`openai-completions` provider: streamed chat completions
with tools, `reasoning_effort` / `chat_template_kwargs`, usage in the stream) driving `runq serve` with Qwen3.8-27B
and its draft. A context window of **96k tokens** is the minimum. A session grows by **2-3k tokens per turn** (tool
results, file contents) followed by a reply (reasoning, then a tool call or a code edit). When the context nears
the window, pi compacts: it summarizes the old messages in a separate request (the conversation serialized as text,
reasoning omitted, tool results cut to 2,000 characters: no prefix to reuse) and continues with the system prompt +
the summary + the last ~20k tokens (`keepRecentTokens`). Every item keeps the project's rules (bitwise paths;
speculative == plain).

Where it stands (estimates from measured rates: S7, S2 and `IDEAS.md`'s ~590 GFLOPS of prefill attention at 8
threads; G0 replaces them):
- **Memory:** runq's own estimate for one 27B sequence at `-c 98304` is 22.38 GiB (64 KiB of f16 KV per position:
  6 GiB at 96k), plus the 0.8B Q4_K_M draft (2.03 GiB at the same `-c`) and 3 more checkpoints for speculation at
  k = 3 (0.44 GiB): ~24.9 GiB against ~23.9 GiB available with the other services stopped. **Does not fit yet.**
- **Prefill per turn:** a 2.5k-token chunk at ~90k positions costs ~55 s of matrix multiplies (45 tok/s at 8
  threads; ~33 s at 16) plus ~150 s of attention (2.5k x 90k positions x 393 kFLOP / ~0.59 TFLOPS): **~3 min**.
- **Decode at ~90k:** each token reads ~17 GB of weights + ~6 GB of KV and does ~35 GFLOP of attention: ~4-5 tok/s
  plain (8.5 at short context). Speculation's gain shrinks there: a verify segment shares the KV read, but its
  attention work grows with its rows.
- **Compaction:** the summary request prefills the serialized history from scratch (measured at 96k: ~18k tokens,
  ~32% of the summarized context), then the next request re-prefills the summary and the kept ~20k after the system
  block (its checkpoint since G5): ~28-33 min on the 27B at 8 threads, ~17-21 at 16 (G5 below).
- **Tools:** done (G1, docs/server.md): `runq serve` renders tools and tool history as the Qwen3.5 / 3.8 templates do and
  returns the XML calls as OpenAI `tool_calls`; pi's agent loop (read / bash, thinking on) ran against the 0.8B. Not yet
  run against the 27B (its first pi session belongs to G0's replay).

Goal (provisional, G0 sets it from measurements): a 96k session with the draft fits with >= 2 GiB to spare; a
2.5k-token turn at 90k positions prefills in ~1 min; decode >= 8 tok/s at 90k with speculation; no turn re-prefills
more than its new tokens plus the last reply.

- [ ] **G0. Measure the agentic turn first.** (a) A replay benchmark: pi's recorded sessions (`~/.pi/agent/sessions`,
  local only, never committed) replayed against `runq serve` request by request, recording per turn the reused
  prefix, the prefilled tokens, the prefill and decode times and the speculation stats. (b) Long-context costs
  without hours of prefill: decode, verify passes of 2-8 rows (`build/bench_verify`) and the prefill of a
  2.5k-token chunk at 16k / 32k / 64k / 96k positions on a KV cache filled with synthetic values (the timing does
  not depend on them), 8 and 16 threads; this answers "Long-context decode: find the real bound" (Performance) for
  the 27B. (c) Draft acceptance on the sessions' real replies (`scripts/spec_accept.py`, teacher-forced): the 0.8B
  draft, prompt lookup (G6), both. Done when docs/qwen38-speculative.md has a turn-latency model checked against the
  replay, and the goal above has real numbers.
  Status (2026-10-27): (b)'s tool is there, `build/bench_long` on `rq_seq_fill_synthetic` (a library test hook), run
  on the 0.8B, the 4B (to 32k) and the 27B-shaped synthetic model; docs/qwen38-speculative.md, "Long-context costs",
  has the numbers and a turn-latency model extrapolated to the 27B (decode ~4.8 / 5.6 tok/s at 98k with 8 / 16
  threads, a 2.5k-chunk prefill ~250 / ~145 s, ~200 / ~111 s since G4; a verify row at 98k adds ~80-90% of a step's
  attention, so k = 3 passes cost ~2.2x a step there, ~2.0x since G4). Left of (b): the 27B run itself, alone on the
  box (the command is in that section).
  (a) **done** (2026-10-28): `scripts/replay_sessions.py` (+ `scripts/replay_pi.mjs`: pi's own code builds each
  request from the session, pi's compaction runs at the server's window, replies forced to the recorded ones through
  `force_reply` or generated with `--mode live`; per request the reused prefix, prefilled / new tokens, divergences
  and rollbacks from the response's `timings`, times; docs/server.md, "Agent sessions"), run on the 0.8B at 32k and
  96k on two sessions (G5's numbers). Left of (a): the speculation stats per turn (S8: the server has no draft yet)
  and the 27B's own run.. (c) done on proxies (2026-10-27, `scripts/lookup_accept.py`, docs/qwen38-speculative.md, "Prompt
  lookup"): pi's sessions rendered as Qwen3.8 (4,508 replies, 2.0 M tokens before position 98k: other models'
  replies, the 27B's are not recorded), teacher-forced: the prompt lookup alone 1.56 tokens per pass (k 3; syntax 2.9,
  edits' old text 2.6, reasoning / prose 1.3), the 0.8B draft 2.40 (k 3; on a 95k-token subset up to 32k positions),
  combined 2.38 (lookup only for matches >= 8 tokens: no gain over the model); in the cost model at 98k positions the
  lookup alone is the only setting faster than plain (5.2 vs 4.8 tok/s; the 0.8B at k 3 4.6). Left: the same on the
  27B's own agentic replies (G0 (a)'s replay records them).
- [ ] **G2. Correctness at long context** (the 27B's retrieval left). Done on Qwen3.5 (2026-10-27, docs/qwen35.md,
  "Long context"): the opt-in `long` group (`tests/run_tests.sh --long-context`, `tests/check_long.py`,
  `build/long_ctx`) checks the 0.8B Q8_0 at 14 positions up to 98,303 and the 4B Q8_0 at 6 up to 32,767 against
  `tests/ref_qwen35.py`'s memory-frugal forward (exact softmax, numpy fp32; goldens in `tests/golden/long`): the 0.8B's
  cos >= 0.9996 on the full vocabulary, the 4B's >= 0.9999 on the golden's ids, top-1 everywhere, no drift with the
  position; the verify / decode paths == prefill and the generic build == native, bitwise; a fact planted ~84k tokens
  back retrieved (0.8B; the 4B ~25k back). RoPE's fp32 angles (as HF's) and the f16 cache's range (|K| <= 22, |V| <=
  52) checked; no bug found. Left: the retrieval on the 27B, alone on the box (~1-1.5 h of prefill; ~21.5 GiB with
  the KV cache of its 88k-token prompt): `OMP_NUM_THREADS=16 TMPDIR=/var/tmp .venv/bin/python tests/check_long.py
  --retrieval models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf`. Not run: HF fp32 at long context (the numpy
  reference on the HF weights matches it at short context).
- [ ] **G3. Memory: a 96k session with its draft fits.** First lever: **Q3_K_M weights** (Unsloth
  `Qwen3.8-27B-Q3_K_M.gguf` @ `4a03c64`, 13.82 GB; from its header: Q3_K 6.1 GiB, Q4_K 5.2, Q6_K 1.0, Q5_K 0.3: 3-bit
  `ffn_gate` / `ffn_up`, `attn_gate`, attention q / k and `token_embd`; 4-bit DeltaNet `attn_qkv` / `ssm_out`, attention
  v / output, `ffn_down`; Q6_K `output`): ~12.4 GiB of weights in RAM vs 15.3, so a 96k session + draft + speculation
  needs ~21.9 GiB (~2 GiB spare); decode reads ~13.0 GB per token instead of ~16.1 (est. ~+20% at short context,
  ~+10-15% at 90k); prefill unchanged (attention-bound). Q3_K in runq: **done** (`2f7b8f3`, 2026-10-27: the Q3P
  layout, +1.8% over the file's blocks; Q6_K's fold on 3-bit values; generic / AVX-512 / NEON / small-batch kernels;
  the embedding-row lookup; `-m dequant` == gguf-py bitwise on all 78 Q3_K tensors of Unsloth's Qwen3.5-0.8B Q3_K_M,
  which is in the q35 group: vs HF fp32 pooled cos 0.981, top-1 54/62 (its Q4_K_M 0.9946, 57/62), vs its
  dequantized-weights reference 0.99984, every bitwise check; compare_bins vs 5d93873 identical but the usage line).
  From the 27B file's header: 12.39 GiB of repacked weights vs 15.32 for Q4_K_M (Q3_K 5.71 GiB, Q4_K 5.31, Q6_K 0.98,
  Q5_K 0.30; token_embd read from the mapping: no test file has a Q3_K token_embd, its rows go through kq_dequant_row,
  the code `-m dequant` checks; the 27B's models-group entry will cover it). Open: the **quality gate** vs Q4_K_M before
  it becomes the agentic file: teacher-forced top-1 vs the API's model on the saved answers (Q4_K_M: 716/720 thinking
  off, 579/588 on), KL between the two files on recorded pi sessions, draft acceptance, perplexity. Q4_K_M stays the
  reference. Not Q3_K_S (nearly all 3-bit for 1.2 GiB more), nor the UD / IQ files (codebook formats: new kernels),
  nor 2-bit.
  Second lever, optional after Q3_K_M (from `IDEAS.md`, "int8 (or 8-bit float) KV cache"):
  - An 8-bit KV cache: V first (its bf16 rounding was harmless), then K, which needs per-tile scales: |K| reaches
    ~500 and bf16 K alone already cost +4.4% KL against HF fp32. Each variant judged by the KL measurement of the 16-bit
    types and G2's checks (its baseline: the 16-bit numbers of "KV cache with f16 K and bf16 V", Performance);
    generic, AVX-512 and NEON kernels on one definition. Halves the KV (3 GiB at 96k) and its read per token.
  - The draft's own cost: its KV cache at 96k (~1.1 GiB) and its attention there. Its guesses never change the
    output, so it can run on a shorter recent window (measure the acceptance lost, G0).
  - The checkpoint budget: 150 MiB each for the 27B (the prompt's, the speculative ones, a reply's end: G5). Since G5
    `runq serve` keeps `--checkpoints` (default 4: 0.59 GiB per slot on the 27B) and counts them; Qwen3.8 needs 3
    (the header, the system block, the `</think>` / the reply's end; the turn checkpoint is Qwen3.5's).
  Done when runq's estimate for a 96k session with the draft and speculation leaves >= 2 GiB of the available memory.
- [ ] **G4. Prefill attention against a long cache** (from `IDEAS.md`, "bf16 dot products in prefill attention").
  A turn's chunk is ~2.5k queries against a ~90k-position cache: ~2.5x the matrix multiplies' time. (a) Spread the
  work: the queries x KV blocks over all threads with the decode path's split / combine (see "Prefill attention
  balance", Performance). (b) `vdpbf16ps`: it doubles the dot-product rate (prefill attention runs at ~590 GFLOPS of
  a ~900 FMA peak at 8 threads), but q and the softmax weights become bf16, so the logits-vs-HF test and the KL
  measurement decide; the instruction's exact rounding (two FMAs per pair, DAZ / FTZ, the odd element first) needs
  a C twin for the generic build, verified per CPU. Done when G0's 2.5k-chunk-at-90k prefill is ~2x faster, the
  thresholds hold, and the paths stay bitwise.
  Status (2026-10-28, `082511f`, docs/qwen38-speculative.md, "G4"): (a) **~1.3x, bit-identical**. The loss was the
  memory traffic per item, not the balance (threads 98% busy at 8, 95% at 16 on synth-27b) nor the inner loop (99 of
  111 GFLOPS per core from L2): `att512_kv` now sweeps a long range 2 blocks at a time (K / V once per item, from L2
  for every group) in groups of <= 6 vectors (~108 GFLOPS per core). synth-27b, 2,560-token chunk at 98k: 12.56 ->
  9.49 s at 8 threads (attention 12.0 -> 9.0 s, ~520 -> ~700 GFLOPS of a measured ~880 FMA peak), 7.01 -> 5.20 s at
  16; the 0.8B at 98k 26.0 -> 20.3 s; verify segments at 98k -5-9%; the 27B estimate for the chunk at 98k ~250 ->
  ~200 s (8 threads), ~143 -> ~111 s (16). (a) as written (items per block range with a shared combine) not done:
  ~2-5% of balance to win at 512-row passes, and its partial results would have to live across threads (~60 MB per
  split row block of the 27B at 98k); short passes (a turn's last few hundred rows) would gain more.
  (b) **done as an option** (2026-10-28, docs/qwen38-speculative.md, "G4 (b)"): KV types `bf16-dot` / `f16-dot`
  (`-k bf16-dot`), the attention's products as `vdpbf16ps` computes them (q, K, the softmax weights, V rounded to bf16;
  the lane's two FMAs, DAZ / FTZ, the odd element first), a C twin in the generic build equal to the instruction on this
  CPU (`build/bf16dp_test`: edge cases and 6 x 2^24 random lanes, bit for bit), every path of a sequence on the same
  arithmetic (bitwise: batched == sequential, threads, builds, api digests, long_ctx). The attention part of the
  2,560-token chunk **~1.72x** (synth-27b at 98k 8.99 -> 5.22 s at 8 threads, 4.89 -> 2.82 s at 16; the 0.8B 18.3 ->
  10.7 s; `f16-dot` ~1.5x), verify segments at 98k -24-32%, decode -9-16%; the 27B estimate for the chunk at 98k
  ~200 -> ~140 s (8 threads), ~112 -> ~79 s (16), a turn of 2,500 tokens + a 500-token reply at ~90k ~4.7 -> ~3.7 min
  (8 threads). Not 2x: one core reaches 1.9x with its K / V in L2 (the softmax's exp and the conversions stay fp32,
  the scores' chunk sums go through memory), the rest is K / V streamed at twice the rate (prefetched, ~10% left).
  Accuracy: KL from HF fp32 (scripts/kl_kv.py, pieces of 3,500 tokens of prose, docs and code, relative to the f32
  cache): Qwen3-0.6B (45k tokens) f16 +2.1%, bf16 +3.9%, bf16-dot +5.4% (higher on all 13 pieces), f16-dot +5.9%;
  Qwen3-1.7B (28k) f16 +0.5%, bf16 +2.8%, bf16-dot +2.9% (8 of 8); Qwen3.5-0.8B (45k) f16 -0.7%, bf16-dot +0.3%
  (8 / 13); Qwen3.5-4B (21k) f16 +0.9%, bf16-dot -3.4% (4 / 6; a swinging piece). The logits thresholds hold everywhere
  (the kv group, q35_dot, synth-27b, the models group with --models-kv on the 1.7B / 4B files, 65 checks, the long
  group on the 0.8B with --long-kv: 36/36, pooled cos 0.999927 vs f16's 0.999936 in the same session). So **opt-in**,
  not the default: beyond f16's noise on Qwen3 (attention in every layer: about a bf16 cache's cost), within it on
  Qwen3.5 (1 layer in 4, as the 27B's 16 of 64): the setting for the 27B's long sessions, once checked on the 27B.
  Left for G0's 2x on the whole chunk: its matmul part (57 of the 27B's ~140 s at 8 threads), an fp32-free softmax (a
  change of definition), and the KL at 64k-98k positions (the long group's golden keeps 576 ids per position).
- [ ] **G5 leftovers: re-prefill and compaction.** G5 is done (docs/server.md and docs/qwen38-speculative.md,
  "Agent sessions": checkpoints at the system block's end, the assistant header, Qwen3.5's turn start, `</think>` and the
  reply's end, `--checkpoints`, slot affinity, SSE keep-alive; on the 0.8B replay every tool turn prefills its new
  tokens, or those plus the previous reply). Left: (a) the reply's own tokens for its span of the next prompt when the
  text matches (no re-prefill after a non-canonical sampled token or an empty reasoning; but the prompt is then not the
  canonical tokenization and the output depends on what was cached: opt-in only, if at all); (b) a pinned system-block
  cache outside the slot pool for one-slot setups (the 27B at 96k: the summary request takes the only slot; ~0.7 GiB for
  8k tokens); (c) pi-side, documented for users: `keepRecentTokens` ~8k (~7 min less per compaction on the 27B) and a
  `session_before_compact` extension summarizing without reasoning or with a smaller model (pi has no per-task model
  setting). A compaction costs ~28-33 min on the 27B at 8 threads (~6 normal turns): measure it on the 27B (G0).
- [ ] **G6 leftovers: prompt lookup.** Prompt-lookup drafting is in (`--lookup N [--lookup-min M]`, with or without
  `-D`; exact; docs/qwen38-speculative.md, "Prompt lookup": ~1.5 tokens per pass on agentic replies, 2.8-3.9x on edits
  that return the prompt's code; with the 0.8B draft at short context it adds nothing; at long context it is the
  cheap one). Left: in the server (S8); G7's per-position choice between the lookup and the model; tool schemas whose
  old / new text are plain string parameters (pi's `edits` array is JSON-escaped, so multi-line old text does not match
  the raw file text in the context: a pi-side change or a custom edit tool).
- [ ] **G7. Speculation at long context.** A verify segment of k + 1 rows shares the KV read but multiplies the
  attention work (k = 3: 4x), so the best k falls as the context grows. Choose k (and tau) per pass from the
  position and the measured costs (G0), in the CLI and the server. Done when speculation is never slower than plain
  decoding at any context length in G0's benchmark, and its gain at 90k is recorded.

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
  `<tool_call>`): another renderer and parser, a 400 today. (e) A multi-turn pi session on the 27B (G0): on the 0.8B the
  loop runs (read, bash and edit calls, nested `edits` arrays included, parsed and executed, results sent back, each
  turn prefilling only its new tokens), but the model is too weak to finish an edit task (wrong paths, loops).

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

- [ ] **More GGUF types**: Q4_0 / Q4_1, Q2_K (the Unsloth "UD" files of the small bit widths), IQ4_XS, F16 / BF16
  weights. Each needs its block format, a repacked layout and generic, AVX-512 and NEON kernels that follow one
  arithmetic definition, as in `src/kquants.h` (Q5_K and Q3_K, done, are the templates: a new bit plane on Q4_K's
  fold, resp. 3-bit values with a bit plane on Q6_K's). Tests:
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
