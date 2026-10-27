# Qwen3.8-27B with speculative decoding: facts and design

Status (2026-10-27): **planned, not implemented.** The steps are in [TODO.md](../TODO.md) (section "Qwen3.8-27B with
speculative decoding"); this file holds what was verified and the design decisions, so that the two cannot drift.
The architecture is Qwen3.5's (`qwen35`, [qwen35.md](qwen35.md)).

**Goal:** decode Qwen3.8-27B on the CPU at ~13-18 tok/s on the 16-vCPU dev box (plain decode: ~8, estimated), with
Qwen3.5-0.8B as the draft model. The output is **exactly** plain decoding's: the same tokens, greedy and seeded, and
the same logits bits for every emitted token. Later: the 27B's own MTP layer as the draft.

## The model (HF `config.json`, GGUF header; verified 2026-10-27)

- Dense hybrid (`qwen35`), 64 layers: 48 Gated DeltaNet + 16 gated full attention (every 4th). dim 5120, FFN 17,408.
- Attention: 24 query heads, 4 KV heads (GQA 6), head size 256, RoPE on 64 dims (theta 1e7).
- DeltaNet: 16 key heads and **48 value heads** of 128 (r = 3; r = 2 is done and tested on Qwen3.5-4B, r = 3 on a
  synthetic model, qwen35.md design 8), conv kernel 4.
- Vocabulary 248,320, **untied** (`token_embd` Q4_K, `output` Q6_K). The same tokenizer as Qwen3.5-0.8B: the 0.8B
  re-tokenized every non-thinking 27B answer of the measurement below to the API's ids.
- **An MTP layer in the file:** `block_count` = 65 and `nextn_predict_layers` = 1. `blk.64` is a full-attention
  layer plus `nextn.eh_proj` [10240 -> 5120], `nextn.enorm`, `nextn.hnorm`, `nextn.shared_head_norm`; it shares
  `token_embd` and `output` with the main model. runq's loader ignores unknown tensors today, but its layer pattern
  is computed from `block_count`, so the file needs the nextn handling (TODO S6) before it loads correctly.
- Sampling defaults in the GGUF (`general.sampling.*`): temperature 1.0, top-p 0.95, top-k 20.

**The file:** unsloth `Qwen3.8-27B-Q4_K_M.gguf` at revision `4a03c640833598c0fb5ddd11e846135e906eb764`, 17.1 GB,
sha256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`. Types: Q4_K 11.4 GB, Q6_K 4.5, Q5_K 1.0,
Q8_0 0.06, F32 0.1, all supported by runq. Avoid: `main`'s `UD-Q4_K_M` (16.5 GB: IQ4_XS / Q3_K / IQ4_NL / IQ3_S),
`Q8_0` (29.0 GB, does not fit), `MTP/mtp-...gguf` (a standalone MTP with Q3_K tensors; not needed).

| file (unsloth @ `4a03c64`) | size | types | runq | RSS est. |
|---|---|---|---|---|
| Q4_K_M | 17.1 GB | Q4_K, Q5_K, Q6_K, Q8_0 | yes | ~19-19.5 GB |
| Q4_K_S | 16.1 GB | Q4_K, Q5_K, Q6_K, Q8_0 | yes | ~18-18.5 GB |
| Q3_K_M | 13.8 GB | Q3_K 6.7 GB + Q4_K 5.6 + Q6_K / Q5_K | needs Q3_K | ~15.5-16 GB |
| Q3_K_S | 12.6 GB | Q3_K 10.5 GB + Q4_K 0.9 + Q6_K | needs Q3_K | ~14.5 GB |
| UD-Q3_K_XL | 13.4 GB | IQ4_XS 7.0 GB + IQ3_S + Q5_K | no (IQ types) | |

(No plain Q3_K_L / Q3_K_XL from Unsloth for this model; lmstudio-community's Q4_K_M, 16.8 GB, is Q4_K / Q6_K only.)
Q3 also decodes faster: ~13.4 GB read per token, ~96 ms, ~10.4 tok/s plain, against ~120 ms at Q4_K_M.

## Memory and disk

- Weights: repacked into anonymous memory (17.1 GB for Q4_K_M); the loader `pread`s the file, so it also passes
  through the page cache (drop it after the repack: TODO P4). Embedding rows are dequantized at lookup (no 5 GB fp32
  table).
- Per sequence: KV cache 16 layers x 4 x 256 x 2 x 2 B = 64 KiB per position (f16; 2.5 GiB of address space at
  40,960 positions, touched only as used); DeltaNet state 48 x (48 x 128 x 128 x 4 B + 3 x 10,240 x 4 B) =
  **~150 MiB**, the same per checkpoint. A checkpoint per verified row (S3): ~0.6 GB at k = 3.
- Total with the 0.8B draft (0.66 GB as Q4_K_M) and a few checkpoints: ~19-20 GB at Q4_K_M.
- The box (2026-10-27): 31.3 GB RAM. With the parcolab, billing and triominos-server containers stopped (decided
  for the 27B runs; they used ~7.5 GB, 6.2 of it a uvicorn) ~24 GB is available: **Q4_K_M** it is, no Q3_K. Disk:
  58 GB free, `models/` 29 GB; `/tmp` is a 16 GB tmpfs (RAM): never put model files or venvs there.
- `runq serve`'s slot budget counts only the KV cache today (`src/engine.c`), not the recurrent state and its
  checkpoints: ~0.45 GB per 27B slot uncounted (TODO P4).

## Chat template (`tmpl_38.jinja`, differs from Qwen3.5's)

- Thinking is **on** by default. With thinking on it adds a system message with reasoning instructions: effort
  `xhigh` by default, `medium` adds no text, `low` a short one (+~40 tokens). System and developer messages are
  merged. `RQ_CHAT_QWEN35` renders none of this: a new variant (TODO S6).

## Draft acceptance (measured 2026-10-27)

Method: Qwen3.8-27B through an OpenAI-compatible inference API (HTTP 429 above ~4 parallel requests; the key file is
git-ignored); each answer teacher-forced through Qwen3.5-0.8B Q8_0 with `runq -m logits`, with the exact 3.8 prompt
rendering (prompt lengths equal the API's). 30 prompts in 6 kinds; answers up to 400 tokens, 500 with thinking (the
start of the reasoning only). Scripts and raw data: `logs/specmeas-2026-10-27/` (git-ignored; TODO S0 keeps them).

| | top-1 match | tokens per pass, k = 3 (loop simulated) | est. tok/s (best k) |
|---|---|---|---|
| thinking off, temperature 0 | 83% | 2.97 | ~17.7 (k = 3) |
| thinking off, temperature 1 | 80% | 2.81 | ~16.7 |
| thinking on, temperature 0 | 70% | 2.38 | ~14.2 |
| thinking on, temperature 1 | 65% | 2.18 | ~13.0 |

By kind (thinking off, temperature 0): chat 71%, summaries 73%, code / edits 88-89%, math 92%, extraction 93%. Best
k: 3 for chat and summaries (~15-16 tok/s), 6 for code and math (~21-23). Temperature 1 is an exact match against one
sampled output (the rule kept; a lower bound for rejection sampling). The tok/s are a **cost model**: a 27B decode
step ~120 ms (16.8 GB at ~140 GB/s); a verify pass of k + 1 rows ~1.03-1.9x one step for k = 1-8 (interpolated from
8 batched rows ~1.8-2x one row on the 4B / 8B); a 0.8B step ~6 ms, k + 1 draft steps per pass.

## Decisions

1. **Everything runs on this box.** The 27B file is chosen by what fits in RAM (above).
2. **Exact output only:** speculative == plain decoding, token for token, for a given seed; no rejection sampling.
   With a greedy draft, exact-match acceptance accepts exactly as often as rejection sampling would (the 27B's
   probability of the guessed token, either way), so the table is what it would give; its extra gain needs a
   *sampled* draft, which can stay exact (seed coupling, TODO S10).
3. **Follow Qwen3.8's template:** thinking on by default, effort `xhigh`; the thinking rows of the table count.
4. **Quantization:** Q4_K_M or lower; IQ / UD types maybe later.

## Why exactness is cheap here

Batched passes are bit-identical to sequential ones (the repo's invariant), so a verify row's logits equal plain
decoding's at that position, and exact-match acceptance reproduces plain decoding by construction, provided the
loop draws from the target sequence's RNG exactly as `rq_generate` does (one draw per prefilled token but the last,
then one per emitted token) and rolls the target back exactly (KV: truncate; DeltaNet: a checkpoint per verified
row). The draft's state never affects the output, only the acceptance rate.

## Not now

DFlash 2 (a 2B block-diffusion drafter: a new architecture, a 4 GB BF16 download); multi-VM splits (~10-30% over
one-VM speculation at typical acceptance); rejection sampling (decision 2); IQ / UD types (decision 4).
