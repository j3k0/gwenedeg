# Qwen3.8-27B with speculative decoding: facts and design

Status (2026-10-27): **in progress.** The file's specifics are in (the MTP block, F32 `ssm_alpha` / `ssm_beta`, the
3.8 chat template: TODO S6, checked on synthetic files); the speculative loop and the 27B bring-up are open. The steps
are in [TODO.md](../TODO.md) (section "Qwen3.8-27B with
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
  `token_embd` and `output` with the main model. runq loads the 64 main layers (`block_count` - `nextn_predict_layers`)
  and validates the block without reading it ([loader](#loader-the-mtp-block-and-the-f32-matrices)). Details:
  [file layout](#file-layout).
- Sampling defaults in the GGUF (`general.sampling.*`): temperature 1.0, top-p 0.95, top-k 20 (runq's defaults for
  the options not given, CLI and server: `rq_model_sampling`).

**The file:** unsloth `Qwen3.8-27B-Q4_K_M.gguf` at revision `4a03c640833598c0fb5ddd11e846135e906eb764`, 17.1 GB,
sha256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`. Types: Q4_K 11.4 GB, Q6_K 4.5, Q5_K 1.0,
Q8_0 0.06, F32 0.1, all supported by runq's kernels; `ssm_alpha` / `ssm_beta` are F32 *matrices* here (an fp32 path:
[loader](#loader-the-mtp-block-and-the-f32-matrices)). Avoid: `main`'s `UD-Q4_K_M` (16.5 GB: IQ4_XS / Q3_K / IQ4_NL / IQ3_S),
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

### File layout

The header and tensor directory of the Q4_K_M file (read with `tests/ggufmini.py`, no tensor data; 2026-10-27): GGUF v3,
51 metadata keys, 866 tensors, alignment 32, data at 10,996,704. Architecture keys (`qwen35.*`): `block_count` 65,
`nextn_predict_layers` 1 (u32), `context_length` 262,144, `embedding_length` 5120, `feed_forward_length` 17,408,
`attention.head_count` 24 / `head_count_kv` 4, `key_length` = `value_length` 256, `rope.dimension_count` 64,
`rope.dimension_sections` [11, 11, 10, 0], `rope.freq_base` 1e7, `layer_norm_rms_epsilon` 1e-6, `ssm.conv_kernel` 4,
`ssm.state_size` 128, `ssm.group_count` 16, `ssm.time_step_rank` 48, `ssm.inner_size` 6144, `full_attention_interval` 4.
`general.sampling.top_k` (i32 20), `top_p` (f32 0.95), `temp` (f32 1.0): the defaults for the options not given. Keys
runq reads besides the usual: `nextn_predict_layers` (the MTP block), `tokenizer.ggml.bos_token_id` 248044 (absent in the
0.8B's: `<|endoftext|>`, the token runq takes without it, so nothing changes: a stop token, never prepended); not read: `general.*`,
`quantize.imatrix.*`; the tokenizer: `gpt2` / pre `qwen35`, 248,320 tokens, 247,587 merges, eos 248046, pad 248055;
the 3.8 chat template (9,993 bytes). Tensors (row-major [rows, columns]; llama.cpp's order: `output`, `output_norm`,
`token_embd`, then per block sorted by name):

| tensors | shape | types |
|---|---|---|
| `token_embd` / `output` | [248320, 5120] | Q4_K / Q6_K |
| linear layers (48: l % 4 != 3) `attn_qkv` | [10240, 5120] | Q6_K (24: 0-2, 4-6, 10, 13, 16, ...) / Q4_K (24) |
| `attn_gate` (z) / `ssm_out` | [6144, 5120] / [5120, 6144] | Q4_K / Q5_K |
| `ssm_alpha`, `ssm_beta` | [48, 5120] | **F32** (the 0.8B's: Q8_0) |
| `ssm_conv1d` [10240, 4], `ssm_a` / `ssm_dt.bias` [48], `ssm_norm` [128] | | F32 |
| attention layers (16: 3, 7, .., 63) `attn_q` ([q / gate] per head) | [12288, 5120] | Q4_K |
| `attn_k` / `attn_v` | [1024, 5120] | Q4_K / Q6_K (9) or Q4_K (8) |
| `attn_output` | [5120, 6144] | Q4_K |
| `attn_q_norm` / `attn_k_norm` [256], `attn_norm` / `post_attention_norm` [5120] | | F32 |
| every block: `ffn_gate` / `ffn_up`, `ffn_down` | [17408, 5120], [5120, 17408] | Q4_K, Q6_K (33) or Q4_K (32) |
| `blk.64` (MTP): the attention layer's tensors (`attn_v`, `ffn_down` Q6_K) + `nextn.eh_proj` | [5120, 10240] | Q8_0 |
| `nextn.enorm`, `nextn.hnorm`, `nextn.shared_head_norm` | [5120] | F32 |

No `nextn.embed_tokens` / `nextn.shared_head_head`: the MTP block uses `token_embd` and `output`.
`tests/make_synth_q35.py` builds this layout at a small scale (`synth-q35-27b.gguf`, dim 768, GQA 6, r = 3, the types
per role; `-mtp`: + the MTP block; `-f32`: + F32 alpha / beta, the 3.8 template and the bos key), checked by the `q35`
group ([tests/README.md](../tests/README.md)).

### Loader: the MTP block and the F32 matrices

(TODO S6, 2026-10-27; on the synthetic files: running the real one is S7's.) The real file's tensor directory passes
every check: `runq Qwen3.8-27B-Q4_K_M.gguf --mem 0.1` maps the 64 main layers (F32 alpha / beta taken), validates
`blk.64` and refuses at the memory check, before any tensor data is read (0.04 s, 12 MiB peak RSS): "needs ~18.50 GiB
(weights 15.32, runtime 0.24, a sequence of 40960 positions: KV cache 2.50 + recurrent state 150 MiB x (1 + 2
checkpoints))", the script's estimate below.

- **Main layers** = `block_count` - `nextn_predict_layers` (`gguf_config`): the layer pattern, the linear layers, the
  KV cache, the state and the memory estimate count those only; the forward passes are unchanged. Checked:
  `synth-q35-27b-mtp.gguf`'s logits == `synth-q35-27b.gguf`'s, bitwise (`q35/synth/mtp_logits/*`).
- **The MTP block** (`MtpBlock`, `src/internal.h`; `gguf_map_mtp`, `src/model.c`): a qwen35 file with exactly one
  block, `blk.<n_layers>`: the attention layer's 7 matrices (Q8_0 / Q4_K / Q5_K / Q6_K, the shapes of the main
  attention layers), `nextn.eh_proj` (dim, 2 dim) and the 7 F32 norms (`attn_norm`, `post_attention_norm`,
  `attn_q_norm` / `attn_k_norm`, `nextn.enorm` / `hnorm` / `shared_head_norm`) are checked from the tensor directory
  (present, shape, type, data inside the file) and kept as pointers into the mapping: **not read, not repacked, not
  in the memory estimate** (the 27B's: ~0.6 GB as repacked weights + an attention KV cache of its own). Decision: the
  drafter (TODO S9) converts them when it is built, so that a run without it pays nothing; the validation catches a
  bad file now. Other layouts (several blocks, a `qwen3` file with `nextn_predict_layers`) are refused at load
  ("MTP (multi-token prediction) blocks: ... is not supported"); a tokenizer-only load (`-m tokenize`, `-m render`)
  never looks at the block.
- **F32 `ssm_alpha` / `ssm_beta`** (`gguf_matrix`: F32 accepted for these two roles only, every other matrix still
  needs Q8_0 / Q4_K / Q5_K / Q6_K): copied at load into the weight buffer (`prepare_weights`, an inf / NaN refused as
  a corrupt file, the page cache dropped as for the others) and applied to the **fp32** rmsnorm'd activations
  (`f32_rows`, `src/forward.c`), not the int8 ones the quantized matrices take: as llama.cpp and
  `tests/ref_qwen35.py` do. The summation order is the definition for every path and build: 16 partial sums
  (column mod 16) in column order, product then sum (no FMA), then a fixed tree (8, 4, 2, 1); under `KQ_EXACT` /
  `RQ_FP_EXACT`, gcc vectorizes it as one zmm (vmulps + vaddps, a ymm / xmm tree), the same operations in the
  generic build. Decode and the batched decode rows split the two matrices into items of 16 rows (6 for the 27B's 48:
  ~320 KB each, in parallel, before the heads' preps); batched prefill takes them as blocks of 4 rows over all the
  chunk's rows in the GEMM phase. Q8_0 alpha / beta (every other Qwen3.5 file) run exactly as before. Checked on
  `synth-q35-27b-f32.gguf` (`q35/synth-q35-27b-f32/*`): vs `tests/ref_qwen35.py` cos mean 0.9984 / 0.9976 / 0.9966
  (en, code, long2), min 0.9791, top-1 290/338 (Q8_0 file: 0.9984 / 0.9979 / 0.9966, 0.9803, 281/338); bitwise
  -b 1 / 7 / 64 / 512, 4 vs 3 threads, AVX-512 vs generic, `api_test` 28/28 (batched decode rows == alone,
  checkpoints) and its digests across builds. Cost for the 27B (estimated): 2 x 48 x 5120 x 4 B = 1.9 MB per linear
  layer, 94 MB per token (0.6% of the 16.8 GB read per token).

## Memory and disk

- Weights: repacked into anonymous memory; the loader `pread`s the file, so it also passes through the page cache,
  which it drops matrix by matrix as they are repacked for a file of >= 1/4 of the RAM (`--page-cache`; measured on
  the 8B with `drop`: 4.7 GB of cache freed, +0.35 s on a 6 s cold load). The untied `token_embd` stays in the mapping: its rows are
  dequantized at lookup (no 5 GB fp32 table).
- The load-time estimate (`--mem`; computed from the tensor directory before any weight is read, refused past
  MemAvailable; within 3% of the peak RSS with the context full for Qwen3-0.6B to 8B and Qwen3.5-0.8B / 4B,
  [tests/README.md](../tests/README.md#memory)), for the 27B from its tensor directory with the same rules
  (2026-10-27, by script; runq's own on the file: TODO S7; the MTP block is not counted, it is not loaded): repacked weights
  **15.3 GiB** (the 64 main layers and `output`; K-quants +1-3% repacked; `ssm_alpha` / `ssm_beta` counted as F32),
  runtime ~0.3 GiB, one sequence at the default `-c` 40,960: KV 2.5 GiB + state 150 MiB x 3: **~18.6 GiB** (~20 GB);
  at `-c 8192`: ~16.6 GiB.
- Per sequence: KV cache 16 layers x 4 x 256 x 2 x 2 B = 64 KiB per position (f16; 2.5 GiB of address space at
  40,960 positions, touched only as used); DeltaNet state 48 x (48 x 128 x 128 x 4 B + 3 x 10,240 x 4 B) =
  **~150 MiB**, the same per checkpoint. A checkpoint per verified row (S3): ~0.6 GB at k = 3.
- Total with the 0.8B draft (0.66 GB as Q4_K_M) and a few checkpoints: ~19-20 GB at Q4_K_M.
- The box (2026-10-27): 31.3 GB RAM. With the parcolab, billing and triominos-server containers stopped (decided
  for the 27B runs; they used ~7.5 GB, 6.2 of it a uvicorn) ~24 GB is available: **Q4_K_M** it is, no Q3_K. Disk:
  58 GB free, `models/` 29 GB; `/tmp` is a 16 GB tmpfs (RAM): never put model files or venvs there.
- `runq serve`'s slot budget (`--kv-mem`, default half the RAM and at most MemAvailable after the load) counts a slot's
  KV cache, recurrent state and its 2 checkpoints: 0.94 GiB per 27B slot at `-c 8192` (0.5 GiB KV + 0.44 GiB), shown
  in the startup line.

## Chat template (`tmpl_38.jinja`, differs from Qwen3.5's)

- Thinking is **on** by default. With thinking on it adds a system message with reasoning instructions: effort
  `xhigh` by default, `medium` adds no text, `low` a short one (+~40 tokens). System and developer messages are
  merged. Earlier assistant turns keep their reasoning (`preserve_thinking`, on by default): `<think>\n` reasoning
  `\n</think>\n\n` answer, also with thinking off.
- runq: `RQ_CHAT_QWEN38` (TODO S6), detected in a qwen35 file whose template has `reasoning_effort` (the 27B GGUF's
  = `tests/golden/templates/qwen3.8.jinja`, Unsloth's rewrite; Qwen's own, `qwen3.8-hf.jinja`, is detected too; it
  lacks the developer role, the merging and `high`). CLI: thinking on unless `-r 0`, `-e xhigh|high|medium|low`;
  `runq serve`: on by default, `reasoning_effort` / `chat_template_kwargs` ([server.md](server.md)); the library:
  the effort in the `thinking` argument (`RQ_THINK_*`, `src/runq.h`). The CLI's history keeps each reply as
  generated, which is what the template renders: a second turn prefills only `<|im_end|>\n` + its user turn (no
  rollback to a checkpoint, unlike Qwen3.5 with thinking). `-m render` == HF's `apply_chat_template` in 40 of 45
  golden cases (thinking off / xhigh / medium / low / high x 9 conversations); the 5 others (a reply with a reasoning
  block) equal HF's with the reasoning as `reasoning_content`: runq splits an assistant text at its last `</think>`.
  On the real file, `-m render` reads the header only (0.10 s, 26.5 MiB peak RSS; the 54 golden ids).

## Draft acceptance (measured 2026-10-27)

Method: Qwen3.8-27B through an OpenAI-compatible inference API (HTTP 429 above ~4 parallel requests; the key file is
git-ignored); each answer teacher-forced through Qwen3.5-0.8B Q8_0 with `runq -m logits`, with the exact 3.8 prompt
rendering (prompt lengths equal the API's). 30 prompts in 6 kinds; answers up to 400 tokens, 500 with thinking (the
start of the reasoning only). Tool: `scripts/spec_accept.py` (below); raw data: `logs/specmeas-2026-10-27/`
(git-ignored).

| | top-1 match | tokens per pass, k = 3 (loop simulated) | est. tok/s (best k) |
|---|---|---|---|
| thinking off, temperature 0 | 83% | 2.97 | ~17.7 (k = 3) |
| thinking off, temperature 1 | 80% | 2.81 | ~16.8 |
| thinking on, temperature 0 | 70% | 2.38 | ~14.2 |
| thinking on, temperature 1 | 65% | 2.18 | ~13.0 |

By kind (thinking off, temperature 0): chat 71%, summaries 73%, code / edits 88-89%, math 92%, extraction 93%. Best
k: 3 for chat and summaries (~15-16 tok/s), 6 for code and math (~21-23). Temperature 1 is an exact match against one
sampled output (the rule kept; a lower bound for rejection sampling). The tok/s are a **cost model**: a 27B decode
step ~120 ms (16.8 GB at ~140 GB/s); a verify pass of k + 1 rows ~1.03-1.9x one step for k = 1-8 (interpolated from
8 batched rows ~1.8-2x one row on the 4B / 8B); a 0.8B step ~6 ms, k + 1 draft steps per pass.

### Rerunning it: `scripts/spec_accept.py`

```sh
P=".venv/bin/python -I scripts/spec_accept.py"; L=logs/specmeas-2026-10-27
$P gen $L/out_think_t0.json --think --max 500 --key-file hetzner.key   # 30 requests, 2 at a time; --temp 1: seeded
$P measure $L/out_think_t0.json models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf   # -> $L/meas_<draft>_<gen>.json
$P report --by-kind $L/meas_Qwen3.5-0.8B-Q8_0_out_*.json [--draft-ms 6] [--tau 0.4,0.5 --kmax 6,8]
```

`gen` defaults to the run above (endpoint, model `Qwen3.8-27B`, 2 parallel requests, max 400 tokens, `--set base`
the 30 prompts; `--set long` 3 prompts that reason at length; `--top-logprobs N` for TODO S10). `measure` renders
Qwen3.8's template itself (runq's `-m render` gives the same since TODO S6) (thinking on: the `xhigh` effort line, `--effort`; `--template tmpl_38.jinja` checks it
against the model's own template with jinja2), checks the prompt length against the API's and saves per answer token
the target id, the draft's argmax, its top-1 probability and the target token's probability (~0.5 s per answer at 8
threads; the logits files, ~1 MB per position, go to `--tmp`). `report` simulates the loop and applies the cost
model. Reproduction (2026-10-27, measure rerun with the 0.8B Q8_0): every per-answer top-1 of the original logs is
identical (120 answers), so are the table's top-1 and tokens per pass (all k, all kinds). The original tok/s were
worked out by hand from the ranges above; the script's default verify factors (k = 1..8: 1.03, 1.12, 1.20, 1.35,
1.47, 1.60, 1.75, 1.90) reproduce them: 17.7 / 16.8 / 14.2 / 13.0 at k = 3 (16.7 before: 2.81 rounded / 168 ms),
chat / summaries 14.9 / 15.7 at k = 3, code 21.1 and math 23.5 at k = 6. They are a guess until TODO S2 measures
the verify pass.

### Q4_K_M draft, long reasoning, adaptive draft length (2026-10-27)

The 0.8B decode step (`scripts/bench_models.py -t 8 -r 3`, under the bench lock, two runs, load 12-16 from other
jobs): Q8_0 134-153 tok/s short, 134-144 after 1,500 tokens (6.5-7.5 ms); Q4_K_M 186-210 / 193-200 (4.8-5.4 ms),
1.34-1.50x faster; peak RSS 963 / 663 MB. Below: 7.0 ms vs 5.1 ms per draft step, the other parameters as above.
"Long": 3 new prompts (`--set long`: probability, an LRU cache in C, a logic puzzle; thinking on, up to 1,500
tokens, 3,522 answer tokens). Adaptive: stop drafting at the first position where the draft's top-1 probability <
tau, at most k_max (it changes the guesses, never the output); the teacher-forced probabilities are exact up to the
first wrong guess, after it they only decide how many wasted guesses are paid for.

| | Q8_0: top-1, k = 3, tok/s | Q4_K_M: top-1, k = 3, tok/s | adaptive, tau 0.5, k_max 8: Q8_0 / Q4_K_M |
|---|---|---|---|
| thinking off, temperature 0 | 83%, 2.97, 17.3 | 82%, 2.94, 17.9 | 18.4 / 18.9 |
| thinking off, temperature 1 | 80%, 2.81, 16.4 | 79%, 2.79, 17.0 | 17.3 / 18.0 |
| thinking on, temperature 0 | 70%, 2.38, 13.8 | 70%, 2.38, 14.5 | 14.7 / 15.1 |
| thinking on, temperature 1 | 65%, 2.18, 12.7 | 65%, 2.18, 13.3 | 13.7 / 14.1 |
| thinking on, long (3 prompts) | 70%, 2.38, 13.8 | 69%, 2.35, 14.3 | 14.6 / 15.1 |

- **Q4_K_M as the draft:** the same acceptance (top-1 within a point, tokens per pass within 0.03), ~+4% tok/s from
  the faster step, 300 MB less RAM. Use it.
- **Long reasoning** accepts like the first 500 tokens (70%, 2.38 per pass); deeper into the reasoning it does not get
  worse (code: 41% in the first 500 tokens, 77% after; math 80% / 94%).
- **Adaptive length: +5-6%** over the best fixed k (3) with one setting (tau 0.4-0.5, k_max 6-8) in every row:
  thinking off, more tokens per pass for about the same guesses (Q4_K_M: 3.49 per pass for 3.5 guesses, k = 3: 2.94
  for 3); thinking on, the same tokens per pass with fewer guesses (2.42 for 2.1). Per kind (Q4_K_M, best tau /
  k_max per kind): thinking off code +10%, math +8%, extraction +11% (tau 0.6, k_max 12: long runs of confident
  guesses), chat and summaries +0-2%; thinking on +7-15% for code, math and extraction, +2-6% for the rest. With a
  linear verify cost (1.03 + 0.124 (k - 1)) the gain is +8-11%: fixed k pays for long passes where they fail.
  Worth it: a loop option (TODO S1).

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
then one per sampled token) and rolls the target back exactly (KV: truncate; DeltaNet: a checkpoint per verified
row). The draft's state never affects the output, only the acceptance rate. The first part is shared by
construction: `rq_generate` is `rq_gen_start` + `rq_gen_step` per token (`src/runq.h`: sample, the stop / limit
checks, `on_logits`), which the speculative loop calls for each verified row with the row's position; `api_test`'s
`generate_step` runs that loop with an oracle draft (wrong guesses included) and checks tokens, RNG states and
logits hashes against `rq_generate`. `rq_seq_rng` + `rq_rng_f32` on a copy give the next uniform draw without
consuming it (S10).

## Not now

DFlash 2 (a 2B block-diffusion drafter: a new architecture, a 4 GB BF16 download); multi-VM splits (~10-30% over
one-VM speculation at typical acceptance); rejection sampling (decision 2); IQ / UD types (decision 4).
