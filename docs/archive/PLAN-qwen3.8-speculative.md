# Plan (archived): Qwen3.8-27B with speculative decoding

> **Superseded working plan, archived after implementation.** Steps 0-4b and 6-8 are done. S5's hybrid loop works,
> with the 4B + 0.8B validation, measurements and tuning left; S9's MTP drafter is implemented and measured, with
> follow-ups left; S10's seed-coupled drafting is not implemented. The IQ/UD and RAM notes below are also historical.
> See [TODO.md](../../TODO.md) for open work, [docs/notes/qwen38-speculative.md](../notes/qwen38-speculative.md) for
> the dated implementation record, and [docs/usage.md](../usage.md) / [docs/server.md](../server.md) for current behavior.
> The original text is retained unchanged below.

Working notes, not committed. Delete or fold into TODO.md / docs/qwen35.md once the work is done.

**Goal:** decode Qwen3.8-27B on the CPU at ~13-18 tok/s on the 16-vCPU box (plain decode: ~8, estimated),
using Qwen3.5-0.8B as the draft model. The output must be **exactly** what plain decoding gives: the same tokens,
greedy and seeded, and the same logits bits for every emitted token. Later: the 27B's own MTP layer as the draft.

## What we know (verified 2026-10-27)

**The model** (HF `config.json`, GGUF header):
- Dense hybrid (`qwen35`), 64 layers: 48 Gated DeltaNet + 16 gated full attention (every 4th).
- Dimensions: dim 5120, FFN 17,408.
- Attention: 24 query heads, 4 KV heads, head size 256, RoPE on 64 dims (theta 1e7).
- DeltaNet: 16 key heads and **48 value heads** of 128 (ratio 3), conv kernel 4.
- Vocabulary: 248,320 tokens, **untied** (`token_embd` Q4_K, `output` Q6_K). The same tokenizer as
  Qwen3.5-0.8B: the 0.8B re-tokenized every non-thinking 27B answer to the API's ids.
- **MTP layer included:** `block_count` = 65 and `nextn_predict_layers` = 1.
  - `blk.64` is a full-attention layer plus `nextn.eh_proj` [10240 -> 5120], `nextn.enorm`, `nextn.hnorm` and
    `nextn.shared_head_norm`.
  - It shares `token_embd` and `output` with the main model.

**The file:** unsloth `Qwen3.8-27B-Q4_K_M.gguf` **at revision `4a03c640833598c0fb5ddd11e846135e906eb764`**.
- Size and checksum: 17.1 GB, sha256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Types: Q4_K 11.4 GB, Q6_K 4.5, Q5_K 1.0, Q8_0 0.06, F32 0.1. **All are supported by runq today.**
- Files to avoid:
  - `main` now only has `UD-Q4_K_M` (16.5 GB), which uses IQ4_XS / Q3_K / IQ4_NL / IQ3_S. Not supported.
  - `Q8_0` (29.0 GB) does not fit this box.
  - `MTP/mtp-...gguf` (a standalone MTP with Q3_K tensors) is not needed.

**Memory:**
- Weights: 17.1 GB, repacked into anonymous memory. The loader reads the file with `pread`, so the file
  passes through the page cache.
- Per sequence:
  - KV cache: 16 layers x 4 x 256 x 2 x 2 B = 64 KiB per position (f16). That is 2.5 GiB of address space at
    40,960 positions, touched only as used.
  - DeltaNet state: 48 x (48 x 128 x 128 x 4 B + 3 x 10,240 x 4 B) = **~150 MiB**, the same per checkpoint.
- Total with the 0.8B and a few checkpoints: ~19-20 GB.

**The chat template differs from Qwen3.5's** (`logs/specmeas-2026-10-27/tmpl_38.jinja`):
- Thinking is **on** by default.
- With thinking on, it adds a system message with reasoning instructions: effort `xhigh` by default, `medium`
  adds no text, `low` adds a short one. This makes the prompt +40 tokens.
- System and developer messages are merged.
- runq's `RQ_CHAT_QWEN35` renders none of this.

**Draft acceptance, measured:**
- Method: Qwen3.8-27B through the Hetzner inference API (`https://inference.hetzner.com/api/v1`, key in
  `hetzner.key`; HTTP 429 above ~4 parallel requests, 2 is fine). Each answer is teacher-forced through
  Qwen3.5-0.8B Q8_0 with `runq -m logits`, using the exact 3.8 prompt rendering (prompt lengths equal the API's).
- Prompts: 30, in 6 kinds. Answers up to 400 tokens, or 500 with thinking (the start of the reasoning only).
- Scripts and raw data: `logs/specmeas-2026-10-27/` (git-ignored).

| | top-1 match | tokens per pass, k = 3 (loop simulated) | est. tok/s (best k) |
|---|---|---|---|
| thinking off, temperature 0 | 83% | 2.97 | ~17.7 (k = 3) |
| thinking off, temperature 1 | 80% | 2.81 | ~16.7 |
| thinking on, temperature 0 | 70% | 2.38 | ~14.2 |
| thinking on, temperature 1 | 65% | 2.18 | ~13.0 |

- By kind, with thinking off at temperature 0: chat 71%, summaries 73%, code / edits 88-89%, math 92%,
  extraction 93%. Best k: 3 for chat and summaries (~15-16 tok/s), 6 for code and math (~21-23 tok/s).
- Temperature 1 is measured as an exact match against one sampled output: the rule we keep, and a lower
  bound for rejection sampling.
- The tok/s figures are a **cost model**, not measurements:
  - one 27B decode step ≈ 120 ms (16.8 GB at ~140 GB/s);
  - a verify pass of k + 1 rows ≈ 1.03-1.9x one step for k = 1-8 (interpolated from 8 batched rows ≈ 1.8-2x
    one row on the 4B / 8B, README);
  - one 0.8B step ≈ 6 ms, and k + 1 draft steps per pass.

**runq today:**
- **Two models in one process:** allowed when they have the same activation group size; both GGUFs use 32
  (`src/model.c`).
- **Embeddings:** GGUF embedding rows are dequantized at lookup, so there is no fp32 table (5 GB for the 27B).
- **Exactness:** batched passes are bit-identical to sequential ones. So a verify row's logits equal plain
  decoding's at that position, and **exact-match acceptance reproduces plain decoding** by construction.
- **Verify pass:** `rq_forward_batch` with one segment of k + 1 rows and `RQ_LOGITS_ALL` gives every row's
  logits. IDEAS.md notes that this segment takes the prefill path (9 phases per layer).
- **Rollback:** `RQ_SEQ_CK` = 2 checkpoints per sequence, with one pending `ck_at` per pass. That is not enough
  to roll back to any row of a verify segment.
- **Sampling:** no top-k yet (Qwen3.8's defaults are temperature 1.0, top-p 0.95, top-k 20, in the GGUF's
  `general.sampling.*`).

## Decisions (2026-10-27)

1. **Everything runs on this box** (a development box). The 27B file is chosen by what fits in RAM (below).
2. **Exact output only:** speculative == plain decoding, token for token, for a given seed. Rejection sampling is
   not needed. With a greedy draft it accepts exactly as often as exact match (the 27B's probability of the
   guessed token, either way), so the measured table is already what it would give. Its extra gain needs a
   *sampled* draft, and that gain can be had while staying exact (step 10).
3. **Follow Qwen3.8's template:** thinking on by default, effort `xhigh`. The thinking rows of the table are the
   ones that count.
4. **Quantization types:** Q4_K_M or lower; IQ / UD types maybe later.

## Which 27B file fits

Measured now: 14.7 GB available (31.3 GB total, other services ~16.6 GB, e.g. a 6.1 GB uvicorn), 40 GB of disk free.
Our process needs the weights plus ~1.5-2.5 GB: the 0.8B draft (0.66 GB as Q4_K_M), the 27B's state and k + 1
checkpoints (~0.75 GB at k = 3), the KV cache (64 KiB per position) and the scratch buffers.

| file (unsloth @ `4a03c64`) | size | types | runq support | RSS est. | fits now? |
|---|---|---|---|---|---|
| Q4_K_M | 17.1 GB | Q4_K, Q5_K, Q6_K, Q8_0 | yes | ~19-19.5 GB | no |
| Q4_K_S | 16.1 GB | Q4_K, Q5_K, Q6_K, Q8_0 | yes | ~18-18.5 GB | no |
| Q3_K_M | 13.8 GB | Q3_K 6.7 GB + Q4_K 5.6 + Q6_K / Q5_K | needs Q3_K | ~15.5-16 GB | no, ~1.5 GB short |
| Q3_K_S | 12.6 GB | Q3_K 10.5 GB + Q4_K 0.9 + Q6_K | needs Q3_K | ~14.5 GB | at the limit |
| UD-Q3_K_XL | 13.4 GB | IQ4_XS 7.0 GB + IQ3_S + Q5_K | no (IQ types) | | |

(Unsloth has no plain Q3_K_L / Q3_K_XL for this model. lmstudio-community's Q4_K_M, 16.8 GB, is Q4_K / Q6_K only.)

So: either ~2-5 GB more RAM is freed for the runs (then Q4_K_S / Q4_K_M work today), or we add **Q3_K** (step 4b)
and use Q3_K_M with ~1.5 GB freed, or Q3_K_S as is (no margin: an OOM would hit the other services). Q3 also decodes
faster: ~13.4 GB read per token, ~96 ms, ~10.4 tok/s plain, against ~120 ms at Q4_K_M. Its quality cost (perplexity,
and acceptance against the 3-bit target) is measured in step 7.

## Constraints

- **This box:** see above. ≤ 8 threads while the other agent benchmarks. Downloads of the chosen file (13-17 GB)
  are fine now (40 GB free).
- **The rules of TODO.md apply:** bit-identical paths and the logits thresholds. Added rule: speculative output ==
  plain output (tokens, plus a hash of each emitted token's logits), greedy and seeded.

## Steps

**0. Keep the measurement tooling.**
- Move `logs/specmeas-2026-10-27/{prompts,gen,measure}.py` into `scripts/spec_accept.py`, with the endpoint and
  models configurable and the 3.8 rendering (incl. the reasoning-effort system line).
- Use it to measure:
  - the 0.8B Q4_K_M as the draft (533 MB, decode 1.27x the Q8_0's);
  - adaptive draft length: stop drafting when the draft's top-1 probability < τ. This only changes which guesses
    are made, never the output;
  - your real prompts, and longer reasoning.
- Done when it reproduces the table above from the saved JSON.

**1. The speculative loop on Qwen3** (no recurrent state, so no rollback problem).
- Models: target Qwen3-1.7B, draft Qwen3-0.6B (both local, same 151,936-token tokenizer).
- Library: `rq_generate_spec(ctx_t, seq_t, ctx_d, seq_d, tokens, n, cfg, max_new, k, cb, st)`.
  - Draft k tokens greedily with `rq_decode` on `seq_d`.
  - Verify `[last, d1..dk]` as one segment on `seq_t`.
  - Sample the rows in order with `seq_t`'s RNG (`rq_sample`), accepting while the sampled token == the draft.
    The first mismatch's sample is the next token.
  - Truncate both sequences to the accepted length.
  - Replicate `rq_generate`'s RNG pattern: one draw per prefilled token but the last, then one per emitted token.
- Load check: the vocabulary size and a hash of all token bytes must be equal, else refuse the draft.
- CLI: `-D draft.gguf -K k`. Stats (passes, drafted, accepted, draft / verify time) in `[timing]`.
- Done when, on the test prompts (greedy and seeded, thinking on / off): the tokens equal `rq_generate`'s, each
  emitted token's logits hash (`on_logits`) equals plain decoding's, and `-K 0` is plain decoding.

**2. Verify-pass cost** (measure first).
- Time a segment of 2-8 rows of one sequence (prefill path) against one decode step, and against the same number
  of decode rows of different sequences (the small-batch kernels, `smb512_rows`).
- If small segments are slow on the prefill path, route them through the small-batch kernels. Keep the prefill
  path's causal attention inside the segment, with `batch_mixed`-style bitwise tests.
- Done when the curve f(rows) is recorded and a 4-row segment costs ≤ ~1.3x one decode step (Qwen3-1.7B,
  Qwen3.5-4B).

**3. Rollback for hybrid models: a checkpoint per verified row.**
- API: `rq_seq_checkpoint_rows(s, pos, n)`. The next pass takes a checkpoint after each of rows pos..pos+n-1;
  each (layer, head) item copies its part, as it does now for the single checkpoint. Slots are allocated on first
  use, and `rq_seq_truncate(s, L)` lands exactly on any of them.
- Cost:
  - 27B: ~150 MiB per row, so 0.6 GB at k = 3, and ~4-5 ms of copying per ~130 ms pass.
  - 0.8B draft: 18.9 MB per drafted token (~0.15 ms).
- Later, only if the copies show up in profiles: swap state buffers instead of copying back. Or keep each row's
  DeltaNet inputs (q / k / v / gates, ~2 MB per row for the 27B) and replay the recurrence from the pre-pass
  checkpoint.
- Done when truncating to any position inside the last segment continues bitwise like an uninterrupted run
  (an `api_test` case), on the 0.8B and the 4B.

**4. Larger Qwen3.5 shapes: value heads = r x key heads** (TODO.md, "The other sizes", generalized from r = 2 to
any r; the 27B has r = 3).
- GGUF value head h uses q / k head `h % nk`.
- `dn_prep`'s q / k conv + SiLU + l2norm run per (row, key head); the v part and the gates per (row, value head);
  the checkpoint copy likewise. Reject `prism.hadamard.*`.
- Test models:
  - Qwen3.5-4B Q8_0 (local, r = 2) against `ref_qwen35.py` run layer by layer on the HF weights;
  - a tiny random-init `qwen3_5` model with r = 3 and an MTP layer (for step 6), made with llama.cpp's
    converter (not in the repo: needs a llama.cpp checkout and gguf-py).
- Done when the 4B passes the logits thresholds, the tiny model matches its reference, and the bitwise groups
  pass.

**4b. Q3_K weights** (only if the Q3 files are chosen; TODO.md, "More GGUF types").
- Block format (256 weights: 2-bit low + 1-bit high plane, 16 sub-blocks with 6-bit scales), a repacked layout,
  and generic + AVX-512 kernels on one arithmetic definition in `src/kquants.h` (NEON later), as Q5_K was done.
- Tests: dequantization bitwise against gguf-py (`-m dequant`), logits against the file's dequantized-weights
  reference, the bitwise groups. Test file: a small Q3_K_M (e.g. Qwen3-0.6B or Qwen3.5-0.8B Q3_K_M, < 0.5 GB).

**5. The speculative loop on Qwen3.5.**
- Models: target Qwen3.5-4B Q8_0 (4.48 GB) + draft 0.8B, ~6 GB in total, which fits the budget. Uses step 3's
  rollback for both sequences.
- Expect a smaller gain than on the 27B: a 0.8B guess costs ~18% of a 4B token, against ~5% of a 27B token.
  Estimate ~1.3-1.6x.
- Done when the outputs match plain decoding bitwise (greedy, seeded, thinking on / off) and tok/s plus
  acceptance are recorded.

**6. Qwen3.8 specifics** (doable with the tiny model and the header facts above).
- Loader: `block_count` 65 with `nextn_predict_layers` 1 means 64 main layers plus the MTP block. Load the MTP
  block, or skip it, instead of failing on the layer pattern.
- Chat template variant for 3.8: detect it by its text (it contains `reasoning_effort`). Thinking on by default;
  the effort's system line, merged with the system / developer message; `-r` plus an effort option, and the
  server's `reasoning_effort`.
- Goldens: `make_chat_golden.py` with the 3.8 template against HF's `apply_chat_template`.
- Sampling defaults from `general.sampling.*`, and top-k itself (the TODO.md item).
- Done when `-m render` matches HF for thinking off / on x each effort, with and without a system message.

**7. 27B bring-up** (this box, with the file chosen above).
- Download the pinned file and check its sha256.
- Load it: record peak RSS, and drop the file's page cache after the repack (`posix_fadvise` DONTNEED), so the
  box's other services keep theirs.
- Logits: against `ref_qwen35.py` on the dequantized weights, layer by layer. Also the teacher-forced top-1
  agreement with the API's FP8 model on step 0's outputs: a sanity check, not a threshold.
- Speed: plain decode (expect ~120 ms per token at Q4_K_M, ~96 at Q3_K_M), then speculation with the 0.8B.
  Expect, with thinking on (the default): ~13-14 tok/s at Q4_K_M, ~16-17 at Q3_K_M (if acceptance holds against
  the 3-bit target); with thinking off ~17 / ~21.
- Q3 only: perplexity against Q4_K_M (or the API's FP8 model's top-1 on teacher-forced text), and the draft's
  acceptance against the 3-bit target (`spec_accept.py` on runq's own outputs).
- Done when the numbers are in the README and docs/qwen35.md and the 27B's speculative output equals plain
  decoding bitwise.

**8. Server.**
- `runq serve --draft FILE`: the draft rows join the job's segment in the batched pass (IDEAS.md), with k per
  request.
- Speculate less as the batch fills: the gain shrinks with concurrency.

**9. MTP drafter** (no extra download: `blk.64` is in the same file).
- How it drafts: the target's last hidden state + the next token's embedding -> `eh_proj` -> one full-attention
  layer with its own KV cache -> the shared output.
- Cost: the 1 GB output matrix makes ~9-10 ms per guess. Drafting over a vocabulary subset would cut that;
  exactness doesn't care which guesses are made.
- Needs: the target's final hidden state per row from the verify pass (a new output of `rq_forward_batch`).
- Measure acceptance first. Published: 3.7-5.0 tokens per check of 7 (GPU, temperature 1, rejection sampling).
  Keep whichever draft gives more tok/s.

**10. Seed-coupled drafting at temperature > 0** (more accepted guesses, outputs unchanged).
- The idea: plain decoding samples position j with uniform draw u_j from the sequence's RNG (xorshift: one draw
  per emitted token, so u_j is known in advance from a copy of the RNG state). The draft samples its guess with
  the same u_j and the same settings, instead of taking its argmax. When the two distributions are similar, the
  guess lands on the token the 27B will pick more often than the argmax does: up to Σ min(p, q), like rejection
  sampling with a sampled draft. The 27B's token is still sampled exactly as in plain decoding, so the output
  is unchanged.
- Measure first: the API's `top_logprobs` = 20 gives the 27B's whole sampling distribution under top-k 20.
  Compare greedy vs coupled guesses on the step-0 outputs.
- Depends on top-k in the sampler (step 6) and on both models sorting candidates the same way.

**Not now:**
- DFlash 2: a 2B block-diffusion drafter with the best published acceptance, but a new architecture and a 4 GB
  BF16 download.
- Multi-VM splits: ~10-30% over one-VM speculation at typical acceptance.
- Rejection sampling: not needed (decision 2).
- IQ / UD quantization types (IQ4_XS, IQ3_S, IQ4_NL): maybe later (decision 4).

## Open question

RAM for the 27B runs: can ~2-5 GB be freed while we test (Q4_K_S / Q4_K_M, no new kernels), or do we add Q3_K and
use Q3_K_M / Q3_K_S?
