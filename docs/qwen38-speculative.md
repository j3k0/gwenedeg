# Qwen3.8-27B with speculative decoding: facts and design

Status (2026-10-27): **the 27B runs, with the 0.8B draft, exactly.** In: the speculative loop (S1: `rq_generate_spec`
in `src/spec.c`, the CLI's `-D` / `-G` / `--tau`, exact on Qwen3 and Qwen3.5 targets and drafts,
[below](#the-loop-s1)); the verify pass through the decode-rows worker (S2); the file's specifics (the MTP block, F32
`ssm_alpha` / `ssm_beta`, the 3.8 chat template: S6); the bring-up on the real file (S7, [below](#the-27b-bring-up-s7)):
16.3 GiB of RSS, logits within cos 0.9999 of the numpy reference, 8.7 tok/s plain, 13-14 tok/s with the draft and
thinking on, 16.5-17.5 off (8 threads), every output equal to plain decoding's. Open: the MTP drafter and the server. The open steps are in
[TODO.md](../TODO.md) (section "Qwen3.8-27B with speculative decoding"); this file holds what was verified and the
design decisions, so that the two cannot drift.
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
| Q3_K_M | 13.8 GB | Q3_K 6.7 GB + Q4_K 5.6 + Q6_K / Q5_K | yes (Q3_K since TODO G3) | ~15.5-16 GB |
| Q3_K_S | 12.6 GB | Q3_K 10.5 GB + Q4_K 0.9 + Q6_K | yes (Q3_K since TODO G3) | ~14.5 GB |
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
  (2026-10-27, by script; the MTP block is not counted, it is not loaded): repacked weights
  **15.3 GiB** (the 64 main layers and `output`; K-quants +1-3% repacked; `ssm_alpha` / `ssm_beta` counted as F32),
  runtime ~0.3 GiB, one sequence at the default `-c` 40,960: KV 2.5 GiB + state 150 MiB x 3: **~18.6 GiB** (~20 GB);
  at `-c 8192`: ~16.6 GiB. runq's own (S7): 18.65 GiB at 40,960 (runtime 0.39), 16.51 at 8,192, 16.25 at 4,096;
  measured: within 0.3% of the peak RSS ([S7](#the-27b-bring-up-s7)).
- Per sequence: KV cache 16 layers x 4 x 256 x 2 x 2 B = 64 KiB per position (f16; 2.5 GiB of address space at
  40,960 positions, touched only as used); DeltaNet state 48 x (48 x 128 x 128 x 4 B + 3 x 10,240 x 4 B) =
  **~150 MiB**, the same per checkpoint. A checkpoint per verified row (S3): ~0.6 GB at k = 3.
- Total with the 0.8B draft (0.66 GB as Q4_K_M) and a few checkpoints: ~19-20 GB at Q4_K_M (measured at S7: 17.4 GiB
  of RSS with k + 2 = 10 checkpoint slots on each model).
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
| thinking off, temperature 0 | 83% | 2.97 | ~17.1 (k = 3) |
| thinking off, temperature 1 | 80% | 2.81 | ~16.2 (k = 3) |
| thinking on, temperature 0 | 70% | 2.38 | ~13.7 (k = 3) |
| thinking on, temperature 1 | 65% | 2.18 | ~12.6 (k = 3) |

By kind (thinking off, temperature 0): chat 71%, summaries 73%, code / edits 88-89%, math 92%, extraction 93%. Best
k: 3 for chat and summaries (~14.4 / 15.2 tok/s), 4-5 for code, math and extraction (~19-21.5). Temperature 1 is an
exact match against one sampled output (the rule kept; a lower bound for rejection sampling). The tok/s are a **cost
model**: a 27B decode step ~120 ms (16.8 GB at ~140 GB/s); a verify pass of k + 1 rows 1.14-2.52x one step for k =
1-8 (measured on Qwen3-8B Q4_K_M, the 27B's proxy: [Verify-pass cost](#verify-pass-cost-todo-s2-measured-2026-10-27));
a 0.8B step ~6 ms, k + 1 draft steps per pass. (With the verify factors guessed before S2, 1.03-1.9x, the same
data gave ~17.7 / 16.8 / 14.2 / 13.0 tok/s, best k 6 for code and math at ~21-23.)

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
worked out by hand from guessed verify factors (k = 1..8: 1.03, 1.12, 1.20, 1.35, 1.47, 1.60, 1.75, 1.90; `--verify`
with them reproduces 17.7 / 16.8 / 14.2 / 13.0 at k = 3, chat / summaries 14.9 / 15.7, code 21.1 and math 23.5 at
k = 6). The script's defaults are now the measured factors (TODO S2: 1.14, 1.16, 1.25, 1.41, 1.62, 1.83, 2.15,
2.52), which give the table above.

### Q4_K_M draft, long reasoning, adaptive draft length (2026-10-27)

The 0.8B decode step (`scripts/bench_models.py -t 8 -r 3`, under the bench lock, two runs, load 12-16 from other
jobs): Q8_0 134-153 tok/s short, 134-144 after 1,500 tokens (6.5-7.5 ms); Q4_K_M 186-210 / 193-200 (4.8-5.4 ms),
1.34-1.50x faster; peak RSS 963 / 663 MB. Below: 7.0 ms vs 5.1 ms per draft step, the other parameters as above.
"Long": 3 new prompts (`--set long`: probability, an LRU cache in C, a logic puzzle; thinking on, up to 1,500
tokens, 3,522 answer tokens). Adaptive: stop drafting at the first position where the draft's top-1 probability <
tau, at most k_max (it changes the guesses, never the output); the teacher-forced probabilities are exact up to the
first wrong guess, after it they only decide how many wasted guesses are paid for.

| | Q8_0: top-1, k = 3, tok/s | Q4_K_M: top-1, k = 3, tok/s | adaptive, tau 0.4, k_max 4: Q8_0 / Q4_K_M |
|---|---|---|---|
| thinking off, temperature 0 | 83%, 2.97, 16.7 | 82%, 2.94, 17.3 | 17.2 / 17.7 |
| thinking off, temperature 1 | 80%, 2.81, 15.8 | 79%, 2.79, 16.4 | 16.3 / 16.8 |
| thinking on, temperature 0 | 70%, 2.38, 13.4 | 70%, 2.38, 13.9 | 14.0 / 14.5 |
| thinking on, temperature 1 | 65%, 2.18, 12.3 | 65%, 2.18, 12.8 | 13.1 / 13.5 |
| thinking on, long (3 prompts) | 70%, 2.38, 13.4 | 69%, 2.35, 13.8 | 14.1 / 14.5 |

(tok/s with the measured verify factors of TODO S2; with the guessed ones of before, k = 3: Q8_0 17.3 / 16.4 / 13.8 /
12.7 / 13.8, Q4_K_M 17.9 / 17.0 / 14.5 / 13.3 / 14.3, adaptive tau 0.5, k_max 8: Q4_K_M 18.9 / 18.0 / 15.1 / 14.1 /
15.1.)

- **Q4_K_M as the draft:** the same acceptance (top-1 within a point, tokens per pass within 0.03), ~+4% tok/s from
  the faster step, 300 MB less RAM. Use it.
- **Long reasoning** accepts like the first 500 tokens (70%, 2.38 per pass); deeper into the reasoning it does not get
  worse (code: 41% in the first 500 tokens, 77% after; math 80% / 94%).
- **Adaptive length: +3-5%** over the best fixed k (3) with one setting (tau 0.3-0.4, k_max 4) in every row (Q4_K_M:
  +2.6% / +2.7% thinking off, +4.1% / +5.0% / +5.3% thinking on): thinking off, more tokens per pass for about the
  same guesses (3.10 per pass for 2.9 guesses at tau 0.4, k = 3: 2.94 for 3); thinking on, the same tokens per pass
  with fewer guesses (2.35 for 2.1). Per kind (Q4_K_M, best tau / k_max per kind): thinking off +1-5% (math tau 0.6,
  k_max 6 +3.5%, extraction tau 0.5, k_max 6 +4.6%; summaries -3%: k = 3 is better), thinking on +2-6%. The
  measured verify cost rises steeply past 5 rows (the K-quant kernels become compute-bound, below), so long passes
  that fail cost more and a cap of 4 is best; with the guessed, nearly linear costs it was +5-6% at tau 0.4-0.5,
  k_max 6-8. Still worth it: a loop option (TODO S1).

## Verify-pass cost (TODO S2, measured 2026-10-27)

A verify pass is one segment of k + 1 rows of one sequence with every row's logits (`RQ_LOGITS_ALL`). f(r) = the
median time of such a segment of r rows / the median decode step (`rq_decode`) of the same sequence, at 500 and
2,000 positions of context. Tool: `build/bench_verify` (`make bench-verify`, tests/bench_verify.c): each rep runs
every configuration once in turn (dec, the segments of r = 1..9 rows, r decode rows of r different sequences),
truncating back after each pass; `-A "ENV=..."` runs a second context with other settings in the same process,
paired pass by pass. 8 threads, f16 KV, 21 reps (31 for the second 8B run), under the bench lock, load 2-8 from other
jobs. "Before" is the paired context with `RUNQ_BD=64,2,8,1 RUNQ_SMB=32,32,32,32,0`: the routing and kernels of
`848cf01` (a first run of the `848cf01` build itself, load 3-10, gave the same picture). Decode step (ms, median):
1.7B Q8_0 14.2 / 17.2 (500 / 2,000), Qwen3.5-4B Q8_0 34.6 / 34.0, 8B Q4_K_M 33.5 / 35.4, 0.6B Q8_0 5.3 / 6.4, Qwen3.5-0.8B
Q8_0 8.4 / 8.4.

| model, positions | f(2) | f(3) | f(4) | f(5) | f(6) | f(7) | f(8) | f(9) |
|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B Q8_0, 500: before | 1.20 | 1.26 | 1.34 | 1.41 | 1.46 | 1.53 | 1.62 | 1.84 |
| after | 1.07 | 1.13 | **1.24** | 1.30 | 1.34 | 1.44 | 1.52 | 1.68 |
| 2,000: before | 1.12 | 1.18 | 1.28 | 1.36 | 1.40 | 1.56 | 1.68 | 1.82 |
| after | 1.05 | 1.11 | **1.21** | 1.31 | 1.39 | 1.58 | 1.62 | 1.79 |
| Qwen3.5-4B Q8_0, 500: before | 1.14 | 1.26 | 1.29 | 1.40 | 1.53 | 1.57 | 1.66 | 1.78 |
| after | 1.07 | 1.12 | **1.20** | 1.29 | 1.38 | 1.54 | 1.59 | 1.70 |
| 2,000: before | 1.23 | 1.34 | 1.43 | 1.52 | 1.60 | 1.80 | 1.78 | 1.93 |
| after | 1.11 | 1.23 | **1.29** | 1.36 | 1.45 | 1.54 | 1.62 | 1.82 |
| Qwen3-8B Q4_K_M, 500: before | 1.31 | 1.41 | 1.49 | 1.62 | 1.74 | 1.90 | 2.19 | 2.58 |
| after | 1.14 | 1.17 | **1.28** | 1.43 | 1.61 | 1.83 | 2.14 | 2.51 |
| 2,000: before | 1.28 | 1.45 | 1.58 | 1.72 | 1.90 | 2.12 | 2.46 | 2.85 |
| after | 1.14 | 1.22 | **1.34** | 1.55 | 1.83 | 2.10 | 2.43 | 2.85 |
| Qwen3-0.6B Q8_0, 500: before | 1.22 | 1.31 | 1.48 | 1.59 | 1.71 | 1.81 | 1.96 | 2.19 |
| after | 1.18 | 1.29 | 1.42 | 1.55 | 1.69 | 1.86 | 1.95 | 2.19 |
| 2,000: before | 1.20 | 1.37 | 1.64 | 1.85 | 2.04 | 2.29 | 2.55 | 2.81 |
| after | 1.18 | 1.38 | 1.64 | 1.92 | 2.10 | 2.41 | 2.52 | 2.75 |
| Qwen3.5-0.8B Q8_0, 500: before | 1.15 | 1.23 | 1.34 | 1.45 | 1.61 | 1.66 | 1.69 | 1.80 |
| after | 1.05 | 1.17 | 1.23 | 1.34 | 1.43 | 1.50 | 1.55 | 1.68 |
| 2,000: before | 1.36 | 1.53 | 1.64 | 1.80 | 1.96 | 2.01 | 2.06 | 2.21 |
| after | 1.12 | 1.23 | 1.32 | 1.44 | 1.49 | 1.66 | 1.71 | 1.85 |

- **Before:** a segment of 2-9 rows took the prefill path (9-10 phases per layer). It was not slower than the same
  number of decode rows of different sequences (the rows path: within 5% at 500 positions, faster at 2,000, where
  the rows of one sequence read its K / V once per row block), but 4 rows cost 1.28-1.43x a step on the 1.7B and
  4B, 1.49-1.58x on the 8B Q4_K_M. Profile (`-DRUNQ_PROFILE`, 1.7B, 4 rows, 500 positions): the GEMMs' extra
  arithmetic ~3 ms, the per-row phases (norms, SwiGLU, RoPE) ~1.1 ms of phases of their own.
- **Change 1, the route:** segments of at most `bd_seg` rows (16; `RUNQ_BD=max,epi,split,seg`) take the decode-rows
  worker, with attention items per (row block of the segment, kv head[, block of 256 positions]) and the DeltaNet
  recurrence over each segment's rows in order (`dn_rec_run`, as prefill): 4 phases per layer instead of 9-10.
  Paired, less time per pass: Qwen3 Q8_0 mostly 0-7% (the 1.7B; the 0.6B at 2,000 positions, bound by its
  attention there, -4% to +5%), Qwen3.5 2-14%, the 0.8B at 2,000 positions 18-24% (the prefill path's 5 phases per
  linear layer), the 8B Q4_K_M 1-4%. Longer segments: still 1-4% at 8-16 rows, within noise or up to 3% slower at
  24-32 (hence 16).
- **Change 2, prefetch in the K-quant small-batch kernels** (`SMB_PF` 4,096 bytes ahead; `RUNQ_SMB=...,pf`): at 2-5
  rows the hardware prefetchers leave the kernels waiting on memory between their arithmetic. The prefetch alone,
  paired (8B Q4_K_M, 500 positions): -10 / -12 / -5 / -1% at 2 / 4 / 6 / 9 rows, 0 at 1, the same for decode rows
  of different sequences; 1,024-8,192 bytes about the same. Both changes on the 8B: -11 to -17% at 2-5 rows, -8% at
  6, 0-4% at 7-9. Q8_0: nothing gained (within noise, up to +5% at 1-2 rows), so its kernel does not prefetch.
- **Past 4-5 rows the K-quant kernels are compute-bound:** on the 8B, rows 6-9 cost ~10 ms each (~1/3 of a step;
  the passes' times are the same in every run, ~34 multiply-adds per cycle and thread vs ~43 for Q8_0), so f rises
  by ~0.3 per row there, ~0.1 below. Faster K-quant folds would be the next lever for k >= 4.
- **Exactness:** every row's logits are bitwise those of decoding the tokens one by one, on both routes, both builds
  and any thread count (`api_test`: `batch_verify` / `batch_verify_mixed` and their `_prefill` twins, with a
  checkpoint per row on the hybrid models; the same digests with the `848cf01` library); `tests/compare_bins.sh`
  against `848cf01`: 799 jobs identical.
- **Checkpoints per row** (`bench_verify -c`, a first look for TODO S3, 500 positions): 0.8B 4 / 9 rows +1.2 / +2.9
  ms (~0.3 ms per 18.9 MB checkpoint), 4B +1.7 / +7.6 ms (~0.4-0.8 ms per row).

**The 27B Q4_K_M, extrapolated:** verify(k) = f(k + 1) of the 8B Q4_K_M at 500 positions, the mean of two runs:
**1.14, 1.16, 1.25, 1.41, 1.62, 1.83, 2.15, 2.52** for k = 1..8 (`scripts/spec_accept.py`'s defaults now). Why the
8B: its weights are Q4_K and Q6_K like most of the 27B's (Q4_K 11.4 GB, Q6_K 4.5, Q5_K 1.0, the same kernels: Q5_K
is Q4_K's plus a bit plane), so the bytes per multiply-add, i.e. the ratio of a row's arithmetic to the memory-bound
step, are about the same (0.61 bytes per weight in both files); both scale with the model, so f(r) carries over (the 8B's
33 ms step : ~10 ms per compute-bound row ~ the 27B's ~120 ms : ~35 ms). 500 positions, because the 27B's KV cache
is much smaller per weight byte (64 KiB per position for 16.8 GB of weights, the 8B 144 KiB for 4.7 GB: its 500
positions are the 27B's ~4,000). The hybrid layers add little per row: the Qwen3.5-4B's f(r) is at or below the dense
1.7B's at the same Q8_0 (the recurrence per extra row is ~150 M multiply-adds for the 27B, against ~27 G in its
matmuls). Not included: the checkpoint copies per row (~150 MiB each: measured ~2.2 ms, 2% of a step, at S7).
The 27B itself (S7, [below](#the-27b-bring-up-s7)): f(2..5) = 1.10, 1.15, 1.25, 1.34 at 500 positions, the 8B's within
0.01-0.07.
Against the guess (1.03-1.90) verify is +0-4% at k = 1-3 and +19-33% at k = 6-8: the best fixed k stays 3, the
estimates drop by 3-4% (table above: ~13-17 tok/s with the Q4_K_M draft, ~13.5-17.7 adaptive).

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

## The loop (S1)

`src/spec.c`, API in `src/runq.h` (speculative decoding): `rq_generate_spec(c, s, dc, ds, tokens, n, cfg, max_new,
&opts, cb, &stats)` is `rq_generate` with a draft (`RqSpecOpts`: `k` guesses at most, `tau`; k 0 = `rq_generate`);
`rq_spec_run` is its loop alone, after the caller's prefill and `rq_gen_start` (the CLI's generate mode, which draws
nothing for its prompt); `rq_spec_check` compares the vocabularies (size, and an FNV hash of every token's length and
bytes: ~4 ms for Qwen3's 151,936 tokens, 7 ms for Qwen3.5's 248,320; checked by every call too). One round, the target
at length P with its sampled token t not fed:

1. **Draft**: the draft sequence keeps its longest common prefix with the target's tokens and is fed the rest + t
   (one prefill: after an all-accepted pass, the last guess it never fed; after a rejection, nothing but t, the cut
   being a truncation), then guesses greedily (argmax; the token mask of `cfg.allow` respected), each fed for the
   next, at most k, fewer before `max_new` (the pass's last row gives one more token) or the end of either context;
   adaptive: the first guess whose top-1 probability is < tau stops it (the softmax sum stops once it shows p < tau).
2. **Verify**: `[t, d1..dn]` in one `rq_prefill(RQ_LOGITS_ALL)` (no guess: one decode step).
3. **Accept**: `rq_gen_step` on rows 0..n: on_logits, the sample (the target sequence's RNG, one draw per row), the
   stop checks; while the sample equals the next row's token, that row is valid; the first mismatch's sample (or row
   n's, the bonus) is the next t (`RQ_GEN_LAST`: fed alone without logits; an accepted last token is already in);
   the target is truncated after the last row used.

Qwen3.5: `rq_seq_checkpoint_rows` on the target's verify rows (P .. P + n - 1) and on the draft's (t and the guesses
but the last), k + 2 slots each during the loop, so both truncations land exactly (`RqSpecStats.refed` counts
positions fed again: 0 in every test). Afterwards the target's checkpoints are as `rq_generate` leaves them (the rows'
dropped, the slot count back): a chat's next turn then truncates to the same checkpoint as after plain decoding, so it
prefills the same tokens and draws the same RNG values (one per prefilled token but the last). The draft keeps its
slots. The CLI's chat keeps the draft's sequence across turns (the same common prefix and assistant-header checkpoint
as the target's); the draft sequence is only ever a source of guesses.

Equivalence (all bitwise, `tests/check_spec.py`, `make test`'s `spec` group, 44 checks): `api_test --spec` compares
`rq_generate_spec` with k = 1 / 3 / 8 and k 8 + tau 0.5 against `rq_generate` on 9 cases (chat turns with thinking off /
on, raw code; greedy, seeded T 1 / top-p 0.95 / top-k 20 and T 0.8 / top-p 0.9; an end token, `max_new` and the end of
the context inside verify passes, a draft context shorter than the target's or than the prompt, two calls with both sequences cut back in
between): the tokens, the RNG state after each, every sampled logits vector's position and hash, the stops, the cache
and one more decode step's logits; on the 0.6B Q8_0 + Q4_K_M pair (both builds), the Qwen3.5-0.8B Q8_0 + Q4_K_M pair
and, by hand, Qwen3-1.7B Q8_0 + 0.6B Q8_0 / Q4_K_M (11/11 each). The CLI against itself without `-D`: stdout, stderr
but the `[timing]` lines, and the chat's `RUNQ_CHAT_DUMP` (the cache's ids and every sampled logits vector) for 12
generate / chat cases (thinking on / off, seeded, `-n` 1 / 2 / 13, `-c` 37 / 41 / 100 / 160: the context's end inside a
pass, turns dropped, a reply cut; `-b 1`), on the 0.6B pair (and the generic build, the Qwen3.5 pair) and by hand on
both 1.7B pairs; `-G 0 -D` is plain decoding byte for byte (stderr included). A scratch build that kept one rejected row
fails 2 of the api cases. `tests/compare_bins.sh` against 848cf01 (full, 792 jobs): identical but for the 16 usage-text
jobs (the 4 lines of `-D` / `-G` / `--tau`).

**Speed** (2026-10-27, `runq -m chat -n 256`, 8 threads, under the bench lock, load 4.5-10 from other jobs; median of 3
runs, [min-max]; acceptance = accepted / drafted guesses, then tokens per target pass). Target Qwen3-1.7B Q8_0, draft
Qwen3-0.6B Q4_K_M unless noted; prompts: code ("Write a Python function that parses a CSV file ...", greedy), chat
("What are the main differences between a cat and a dog as pets?"), think ("How many prime numbers are there between 1
and 50?", `-r 1`), code at `-t 1 -p 0.95 -K 20 -s 1`:

| tok/s | code | chat | think | code, T 1 | mean |
|---|---|---|---|---|---|
| plain | 78 [67-84] | 70 [64-86] | 78 [60-90] | 75 [70-78] | 75.5 |
| `-G 1` | 95, 85%, 1.86 | 74, 65%, 1.65 | 96, 94%, 1.94 | 99, 83%, 1.84 | 90.9 |
| `-G 2` | 102, 83%, 2.67 | 86, 56%, 2.13 | 108, 83%, 2.64 | 99, 71%, 2.42 | 99.0 |
| **`-G 3`** (default) | 111, 77%, 3.32 | 88, 45%, 2.35 | 119, 80%, 3.37 | 87, 64%, 2.91 | **101.5** |
| `-G 4` | 110, 74%, 3.94 | 80, 41%, 2.64 | 108, 76%, 3.94 | 90, 57%, 3.28 | 96.8 |
| `-G 5` | 122, 70%, 4.49 | 67, 36%, 2.78 | 118, 72%, 4.49 | 105, 55%, 3.71 | 103.0 |
| `-G 6` | 110, 64%, 4.83 | 67, 32%, 2.88 | 121, 69%, 5.02 | 96, 49%, 3.88 | 98.5 |
| `-G 6 --tau 0.4` | 115, 67%, 4.65 | 72, 47%, 2.72 | 100, 72%, 4.65 | 99, 55%, 3.46 | 96.4 |
| `-G 8 --tau 0.5` | 99, 69%, 5.02 | 80, 54%, 2.61 | 114, 76%, 4.49 | 92, 59%, 3.76 | 96.0 |
| Q8_0 draft, `-G 4` | 91, 68%, 3.71 | 68, 41%, 2.61 | 105, 77%, 4.00 | 77, 59%, 3.32 | 85.0 |
| Q8_0 draft, `-G 8 --tau 0.5` | 93, 67%, 5.12 | 70, 62%, 2.64 | 83, 75%, 4.74 | 78, 55%, 3.71 | 81.0 |

- **1.15-1.5x** at `-G 3` (code 1.42x, chat 1.26x, thinking 1.53x, code at T 1 1.16x); k 2-5 within the noise of each
  other on average (chat prefers 2-3, code and thinking 3-6). An earlier session (2 runs, load 8-10) gave the same order:
  `-G 3` 112-115 / 65-72 / 120 on code / chat / think, `-G 4` 116-121 / 64-78 / 124, adaptive (k 8, tau 0.5) 97-124 /
  78-84 / 122, plain 62-67 / 66-75 / 77.
- **The draft is expensive here**: a 0.6B Q4_K_M step costs ~4.2 ms (Q8_0 5.6 ms) against ~13 ms for a 1.7B decode
  step, a ratio of ~0.32 (the 27B with the 0.8B: ~0.04-0.06), so every rejected guess costs a third of a target token;
  the Q8_0 draft accepts the same and is 15-20% slower overall: Q4_K_M as the draft, as for the 27B.
- **Adaptive length** did not beat the best fixed k here (`-G 8 --tau 0.5` 96 vs 101.5): it helps where guesses are
  cheap relative to the target (the 27B's cost model, above), not at a 0.32 draft / target cost ratio. On the 27B it
  is the best setting (S7, below).
- **The verify pass** (S2 measures it properly): from these runs' stats, a pass of 2 / 4 / 5 rows took ~15.1 / 17.5 /
  18.9 ms, 1.16x / 1.35x / 1.45x a plain step (13.0 ms including sampling): the cost model's factors (1.03 / 1.20 /
  1.35) are optimistic for 1.7B. Temperature 1 accepts less (64% vs 77% at k 3), as the acceptance table predicted.

## The 27B bring-up (S7)

The pinned file (`scripts/download_models.py 3.8-27b-q4km`, sha256 checked), measured 2026-10-27 with the parcolab,
billing and triominos containers stopped (MemAvailable 24-25 GiB before each run; one 27B process at a time).

**Memory and load.** `build/mem_probe FILE 4096` (8 threads, a cold file: 10 MiB of it in the page cache before):
load 19.6 s; estimate 16,621.7 MiB (weights 15,686.7, runtime 230.2, a sequence of 4,096 positions 704.9: KV 256 +
state 149.6 x 3); the load's peak RSS 15,735.6 MiB (weights + runtime 1.2% above it); the file's page cache after
the load 32 MiB (`auto` dropped it); peak RSS with the context full 16,676.2 MiB: **the estimate within -0.3%**. The
CLI's peak RSS (1,500 + 128 tokens, `-c 4096`) 16,325-16,336 MiB; the load takes 20 s at 8 threads, 15 s at 12, 12 s
at 16 (cold every time: the cache is dropped, so each run reads the 17 GB again). With the draft (`api_test --spec`:
both models, k + 2 = 10 checkpoint slots on each): 17.4 GiB. Avoid: `api_test` on the file alone peaks at 22.4 GiB
(its batched-decode checks hold up to 17 sequences, each with its 150 MiB state and checkpoints), so the models group
skips it there; it passes by hand (32/32, 374 s).

**Accuracy.** `tests/make_model_ref.py 3.8-27b-q4km` (`tests/ref_qwen35.py` on the file's dequantized weights, layer
by layer, the MTP block skipped: 214 s at 8 BLAS threads, ~2 GB of anonymous memory, the rest mapped file pages) vs
runq (`-b 64`): cos mean 0.99994 / 0.99996 / 0.99997 (chat, code, en), min 0.99981; long2 (positions 248..296)
0.99966, min 0.99886; pooled 0.99992 over 59 positions, top-1 62/62 (the models group's `ref` thresholds: 0.995 /
0.97 / 0.98 / top-1 0.75 / pooled 0.997). Teacher-forced through runq's 27B (`scripts/spec_accept.py measure` with the
27B as the "draft") on the API's answers of the measurement above (`logs/specmeas-2026-10-27`, one answer per kind):
runq's top-1 = the API model's token at **716 / 720** positions (99.4%) with thinking off (6 answers), **579 / 588**
(98.5%) with thinking on (4 answers); every answer re-tokenized to the API's ids. Sanity: "Canberra" in generate and
chat, thinking off and on; a two-turn chat remembers the name and colour of turn 1.

**Bitwise** (the models group, `tests/run_tests.sh --models --model-tags 3.8-27b-q4km`: 61 checks, 419 s): `-b 7` /
`-b 64` vs `-b 1` (en, all), `-b 512` vs `-b 64` (long2, 297 tokens), 8 vs 3 threads, AVX-512 vs generic (en, `-b 1` /
`-b 7`): bit-identical; `api_test` (batched decode rows == alone, verify segments with a checkpoint per row, ...)
32/32 by hand.

**Plain speed** (`scripts/bench_models.py`, under the bench lock, best of 2 at 8 / 12 threads, 1 at 16; load 9-17
including the run's own threads, 6.6 before):

| threads | prefill 488 / 1,532 tokens | decode short / after 1,500 tokens | peak RSS |
|---|---|---|---|
| 8 | 45 / 44 tok/s | 8.5 / 8.7 tok/s | 16,325 MiB |
| 12 | 65 / 64 | 10.0 / 10.6 | 16,329 |
| 16 | 76 / 75 | 10.3 / 9.5 | 16,336 |

The decode step is ~118 ms at 8 threads, the cost model's ~120 ms. `bench_verify -L 500` (11 reps, two runs): a
segment of 2 / 3 / 4 / 5 rows 1.10 / 1.15 / 1.25 / 1.34x a decode step (the 8B Q4_K_M's, the model's: 1.14 / 1.16 /
1.25 / 1.41); 2-5 decode rows of different sequences 1.13-1.14 / 1.21-1.25 / 1.30-1.38 / 1.49-1.53x.

**Speculative** (target the 27B Q4_K_M, draft Qwen3.5-0.8B Q4_K_M; `runq -c 4096 -n 256 -r 1|0 -i PROMPT [-D DRAFT -G k
[--tau T]]`, 8 threads, under the bench lock, load 7-11; prompts of `spec_accept.py`'s kinds: chat "What are the main
differences between TCP and UDP?", code "Write a C function that reverses a singly linked list in place.", math the
train's average speed, summ the bike-lane article in three bullet points; greedy, best of 2 runs for plain, `-G 3` /
`4`, tau 0.3 / 0.4, 1 run for the others; tok/s, then guesses accepted and tokens per pass):

| thinking on | chat | code | math | summ | mean | vs plain |
|---|---|---|---|---|---|---|
| plain | 8.6 | 8.2 | 8.9 | 9.0 | 8.7 | |
| `-G 2` | 12.1 (52%, 2.04) | 10.7 (44%, 1.88) | 15.4 (78%, 2.56) | 13.0 (60%, 2.20) | 12.8 | 1.47x |
| `-G 3` | 12.3 (43%, 2.30) | 10.2 (33%, 1.98) | 16.2 (71%, 3.12) | 13.9 (57%, 2.70) | 13.2 | 1.51x |
| `-G 4` | 11.1 (36%, 2.45) | 9.5 (28%, 2.12) | 16.7 (67%, 3.68) | 13.6 (48%, 2.90) | 12.7 | 1.46x |
| `-G 5` | 10.0 (30%, 2.48) | 8.8 (25%, 2.23) | 16.1 (61%, 4.07) | 11.9 (41%, 3.05) | 11.7 | 1.34x |
| `-G 6` | 9.3 (27%, 2.60) | 7.8 (21%, 2.25) | 16.2 (56%, 4.38) | 12.8 (42%, 3.50) | 11.5 | 1.32x |
| `-G 4 --tau 0.3` | 12.4 (56%, 2.43) | 11.3 (47%, 2.02) | 17.3 (72%, 3.68) | 14.5 (59%, 2.90) | 13.9 | 1.59x |
| **`-G 6 --tau 0.4`** | 12.6 (58%, 2.43) | 11.4 (51%, 2.03) | 17.8 (68%, 4.30) | 14.6 (56%, 3.22) | **14.1** | **1.62x** |
| `-G 8 --tau 0.5` | 12.6 (66%, 2.30) | 10.4 (50%, 1.88) | 16.2 (68%, 4.47) | 12.7 (56%, 2.70) | 13.0 | 1.49x |

| thinking off | chat | code | math | summ | mean | vs plain |
|---|---|---|---|---|---|---|
| plain | 8.2 | 8.3 | 9.1 | 9.1 | 8.7 | |
| `-G 2` | 13.3 (62%, 2.23) | 13.6 (87%, 2.74) | 14.3 (89%, 2.78) | 12.8 (64%, 2.28) | 13.5 | 1.56x |
| `-G 3` | 13.1 (54%, 2.59) | 19.3 (85%, 3.55) | 19.9 (89%, 3.66) | 13.9 (55%, 2.65) | 16.6 | 1.91x |
| `-G 4` | 12.9 (46%, 2.81) | 19.1 (78%, 4.11) | 20.4 (85%, 4.41) | 13.2 (48%, 2.92) | 16.4 | 1.89x |
| `-G 5` | 11.9 (41%, 3.01) | 17.0 (72%, 4.59) | 20.2 (79%, 4.92) | 13.5 (47%, 3.35) | 15.7 | 1.80x |
| `-G 6` | 10.8 (36%, 3.08) | 17.8 (67%, 5.03) | 22.0 (82%, 5.82) | 13.1 (43%, 3.56) | 15.9 | 1.83x |
| `-G 4 --tau 0.3` | 13.2 (51%, 2.72) | 18.0 (80%, 4.11) | 20.2 (85%, 4.41) | 13.8 (53%, 2.78) | 16.3 | 1.88x |
| **`-G 6 --tau 0.4`** | 13.3 (54%, 2.75) | 18.7 (74%, 5.03) | 22.0 (86%, 5.82) | 14.7 (59%, 3.26) | **17.2** | **1.98x** |
| `-G 8 --tau 0.5` | 13.5 (63%, 2.61) | 19.8 (82%, 5.57) | 21.4 (85%, 6.74) | 15.6 (70%, 3.26) | 17.6 | 2.02x |

- **The target is met at 8 threads:** 13.2-14.1 tok/s with thinking on (1.5-1.6x), 16.6-17.6 off (1.9-2.0x); the
  goal was ~13-18. Adaptive length (tau 0.4, k_max 6) is the best single setting, +4-7% over `-G 3` (the CLI's
  default stays `-G 3`, no tau: the per-model defaults are TODO S5's).
- **Sampled** (the GGUF's defaults T 1, top-p 0.95, top-k 20, `-s 42`; 1 run): thinking on plain 8.6, `-G 3` 12.8,
  tau 0.3 / k 4 13.5, tau 0.4 / k 6 13.6 (1.59x); off 8.5, 15.7, 16.2, 17.1 (2.03x).
- **12 threads** (chat and code, greedy, 1 run, load 11-15): thinking on plain 10.3, `-G 3` 13.8, tau 0.4 / k 6 15.3;
  off 9.6, 18.5, 20.6 (code 24.8).
- **Against the API-based estimate:** the tokens per pass are the teacher-forced simulation's. `spec_accept.py`'s
  simulation at k = 3 on the API's answers to the same 4 prompts (their first 256 tokens): thinking off 2.56 / 3.29 /
  3.71 / 2.65 per pass, measured 2.59 / 3.55 / 3.66 / 2.65; on 2.30 / 1.91 / 3.14 / 2.42, measured 2.30 / 1.98 / 3.12 /
  2.70. Its tok/s for them (17.6 off, 13.8 on, k = 3) are 4-6% above the measured 16.6 / 13.2: in the loop a k = 3
  verify pass takes 167-171 ms, ~1.4x the CLI's plain step (the model: 1.25x, without the checkpoint copies and the
  per-row sampling); the draft's ~20 ms per pass is the model's (4 x 5.1 ms).
- **Exactness:** every speculative run printed plain decoding's text: 128 speculative runs of the tables (greedy and
  sampled, both thinking modes, 8 and 12 threads) against their plain runs, stdout byte for byte; `spec`'s CLI cases on
  the pair (`check_spec.cli_case`, 8 threads): `chat_greedy` (3 turns, thinking on, `-G 1 / 3 / 8`, `-G 8 --tau 0.5`),
  `chat_seeded_think` and `gen_seeded_topk` (`-G 3`, `-G 8 --tau 0.5`): stdout, stderr but the `[timing]` lines and
  the chat dumps (the cache's ids and every sampled logits vector) bitwise; `api_test --spec` on the pair: 12/12 (every
  case at k = 1 / 3 / 8 and tau 0.5: tokens, RNG states, logits hashes, the cache; no position fed again; 178 s).
- **The checkpoint copies (S3):** `bench_verify -L 500 -R 1,2,3,4,5 -r 11`, two runs without and two with `-c`
  alternated: the best pass of 3 / 4 / 5 rows +6.2 / +8.9 / +11.9 ms with a checkpoint per row (the medians are
  noisier: +2.7 / +2.1 / +10.7), **~2.2 ms per row** (150 MiB at ~70 GB/s), 1.8% of a decode step: ~7 ms of a k = 3
  verify pass (4%). Not worth swapping buffers or replaying the DeltaNet inputs for (the ideas S3 kept for later).

## Long-context costs (TODO G0 (b), measured 2026-10-27)

What a turn of an agentic session costs at a context of L positions, without prefilling L tokens first (the 27B at
~45 tok/s plus its attention: hours for 96k). **Tool:** `build/bench_long` (`make bench-long`, `tests/bench_long.c`):
the sequence is filled by `rq_seq_fill_synthetic` (a test hook of the library, `src/runq.h`: pseudo-random KV entries
at positions 0..L-1 in the cache's type, the pages touched as a prefill touches them, a pseudo-random recurrent state;
no pass does data-dependent work, so the values do not change the timing), then per L: `dec` (a decode step with its
logits), `seg r` (a verify segment of r rows, every row's logits: speculation with r - 1 guesses) and `pre` (a
2,560-token chunk prefilled in passes of 512 rows, the last row's logits: a turn's new tokens), each pass followed by
a truncation back to L (Qwen3.5: through a checkpoint at L), reps rotated over the configurations. Checked against a
real context: Qwen3.5-0.8B at 16,384 positions, `bench_verify -L 16384` (a real prefill of the lighthouse text) vs
`bench_long -L 16384`, 8 threads: decode 30.9 vs 28.9 ms (median; best 21.2 vs 20.1), segments of 2 / 4 / 8 rows
34.2 / 50.1 / 73.3 vs 35.0 / 48.0 / 62.4 ms: the same within that run's noise (load 26-30).

Measured (under the bench lock, f16 KV, 9 reps for dec / seg, 2 for pre; loadavg 2-15 from other jobs; medians, ms;
`pre` in s):

| model, threads | L | dec | seg 2 | seg 4 | seg 8 | pre 2,560 |
|---|---|---|---|---|---|---|
| 0.8B Q8_0, 8 | 512 | 6.0 | 7.0 | 8.5 | 12.0 | 1.93 |
| | 16,384 | 8.8 | 10.9 | 15.3 | 25.1 | 5.41 |
| | 32,768 | 11.4 | 14.6 | 21.6 | 36.9 | 9.18 |
| | 65,536 | 14.6 | 20.8 | 33.3 | 62.0 | 17.6 |
| | 98,304 | 19.3 | 27.5 | 45.5 | 86.1 | 26.2 |
| 0.8B Q8_0, 16 | 512 | 6.8 | 8.1 | 10.1 | 12.3 | 1.25 |
| | 32,768 | 9.0 | 12.0 | 16.8 | 26.9 | 5.33 |
| | 98,304 | 14.8 | 19.5 | 30.5 | 54.4 | 15.0 |
| 4B Q8_0, 8 | 512 | 29.6 | 33.6 | 38.8 | 51.2 | 8.94 |
| | 16,384 | 42.0 | 47.6 | 57.2 | 82.9 | 18.4 |
| | 32,768 | 38.6 | 49.3 | 67.9 | 113.5 | 29.0 |
| 4B Q8_0, 16 | 512 | 27.5 | 29.0 | 31.3 | 38.1 | 5.57 |
| | 32,768 | 35.4 | 40.3 | 50.3 | 74.7 | 16.6 |
| synth-27b, 8 | 512 | 1.62 | 1.87 | 2.43 | 4.33 | 0.55 |
| | 16,384 | 2.55 | 3.44 | 5.40 | 9.87 | 2.23 |
| | 32,768 | 3.37 | 5.20 | 8.48 | 15.96 | 4.21 |
| | 65,536 | 5.34 | 8.80 | 14.80 | 28.62 | 8.35 |
| | 98,304 | 7.20 | 12.02 | 21.12 | 40.76 | 12.69 |
| synth-27b, 16 | 512 | 1.87 | 2.33 | 2.92 | 4.01 | 0.36 |
| | 16,384 | 2.89 | 3.43 | 5.04 | 8.02 | 1.31 |
| | 32,768 | 2.61 | 4.30 | 6.25 | 10.96 | 2.37 |
| | 65,536 | 4.14 | 6.26 | 10.06 | 18.37 | 4.66 |
| | 98,304 | 6.86 | 9.48 | 15.79 | 27.27 | 7.16 |

(The 4B only to 32k: at 98k it needs ~7.7 GB, more than this run's share of the box. Its 16k row at 8 threads is
noisy: best 31 ms.) `synth-27b` is `build/synth-q35/synth-q35-27b.gguf` (`tests/make_synth_q35.py`): 2 attention
layers with the 27B's attention shape, 12 query / 2 kv heads of 256 (6 query heads per kv head, as 24 / 4), and tiny
matmuls; so `X(L) - X(512)` is the attention of 4 (layer, kv head) units, the 27B has 16 x 4 = 64 of them.

- **Decode at long context is attention-bound well before 96k.** The 0.8B's step triples (6.0 -> 19.3 ms); its
  attention reads 12 KiB per position (1.2 GB at 98k, ~91 GB/s) at ~370 GFLOPS; the 27B-shaped one at 98k does
  ~430 GFLOPS (24.6 kFLOP per position) while reading ~72 GB/s: at the box's practical limits on both counts.
- **Verify rows do not share the attention:** at 98k each extra row of a segment adds ~80-90% of a decode step's
  attention (synth-27b, 8 threads: rows 2 / 4 / 8 add 10.2 / 18.7 / 36.4 ms of attention against 5.6 for one); only
  the weights' read is shared. So f(r) grows with L: synth-27b f(4) 1.50 at 512, 2.93 at 98k.
- **Prefill attention** of a 2,560-token chunk runs at ~510 GFLOPS at 8 threads (synth-27b: 6.3 TFLOP in 12.1 s at
  98k), ~1.8x that at 16 threads.

**The 27B, extrapolated** (not measured: the 27B run is below). Per pass: the 27B's own short-context cost (S7: decode
118 ms at 8 threads, 97 ms at 16; prefill 45 / 76 tok/s; verify f(2) / f(4) = 1.10 / 1.25, f(8) ~2.15 as the 8B
Q4_K_M's) + 16 x synth-27b's attention, `X(L) - X(512)`. The attention part should be an upper bound: with 4 kv heads
per layer instead of 2 the 27B's passes have twice the work items per phase (better balance), and its KV cache per
position (64 KiB) is 16 x the synth's exactly.

| L | decode ms (tok/s), 8 thr | 16 thr | verify pass 2 / 4 / 8 rows, 8 thr (x decode) | prefill 2,560 tokens, 8 thr | 16 thr |
|---|---|---|---|---|---|
| 512 | 118 (8.5) | 97 (10.3) | 130 / 148 / 254 (1.10 / 1.25 / 2.15) | 57 s | 34 s |
| 16,384 | 133 (7.5) | 113 (8.8) | 155 / 195 / 342 (1.17 / 1.47 / 2.57) | 84 s | 49 s |
| 32,768 | 146 (6.8) | 109 (9.2) | 183 / 244 / 440 (1.25 / 1.67 / 3.01) | 116 s | 66 s |
| 65,536 | 178 (5.6) | 133 (7.5) | 241 / 345 / 642 (1.36 / 1.95 / 3.62) | 182 s | 103 s |
| 98,304 | 207 (4.8) | 177 (5.6) | 292 / 447 / 837 (1.41 / 2.16 / 4.04) | 251 s | 143 s |

(16 threads, decode: synth-27b's medians are noisy at 16k-32k, hence 32k below 16k.) **Turn-latency model**, a turn
at position L with P new tokens and a reply of G tokens, plain decoding: T(L) = P / R_mm + 16 A_pre(P, L) + G x
(t_w + 16 A_dec(L)), with R_mm the 27B's prefill matmul rate (45 / 76 tok/s), A_pre / A_dec the synth's attention
times above (both about linear in L: at 8 threads A_dec ~0.057 us per position, A_pre ~0.048 us per (query, key)
pair) and t_w = 118 / 97 ms. A turn of 2,500 tokens + a 500-token reply at ~90k: **~5.6 min at 8 threads (~235 s +
~100 s), ~3.6 min at 16 (~133 s + ~83 s)**; at 32k: ~3.2 / ~2.0 min. With
speculation at 98k (k = 3: a 4-row pass of ~447 ms + the 0.8B draft's 3 steps at 98k, ~60 ms; 2.0-3.7 tokens per
pass as at S7): ~4-7 tok/s against 4.8 plain, k = 1 about the same: the goal's ">= 8 tok/s at 90k with
speculation" needs cheaper attention per row first (G4's bf16 dot products for prefill and verify, an 8-bit KV for
the decode read; G7's k by position). To be checked against the 27B itself and the replay (G0 (a)).

**The 27B run (later, alone on the box: the other services stopped, MemAvailable >= ~23.5 GiB):**

```
make bench-long
flock /tmp/cwayn-bench.lock sh -c 'for t in 8 16; do OMP_NUM_THREADS=$t build/bench_long \
    models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf -L 512,16384,32768,65536,98304 -R 2,4,8 -P 2560 -r 5 -p 1; done' \
    | tee logs/bench-long-27b.txt
```

Its memory: weights 15.3 GiB + KV of 100,865 positions (6.2 GiB, 64 KiB each) + the attention scratch (~0.8 GiB at 16
threads) + 2 checkpoints (0.3): ~22.7 GiB; runq's load-time check refuses it if MemAvailable does not cover its
estimate (`-m GiB` overrides; with less memory, `-L ...,90112`). ~15 min per thread count (one 2.5k chunk at 98k is
~2.5-4 min).

## Prompt lookup (TODO G6, G0 (c), measured 2026-10-27)

Agentic replies copy from their context: an edit's old text, paths, identifiers, a tool result echoed. Prompt lookup
drafts without a model: at a pass whose last sampled token is t, find the latest earlier occurrence of the sequence's
last tokens (t included) and guess the tokens that followed it. Exact as any draft (the verify loop is unchanged).

**Acceptance, offline** (`scripts/lookup_accept.py`, teacher-forced; the loop simulated as `spec_accept.py` does,
the same cost model: a 27B step 120 ms, verify(k) = 1.14 ... 2.52 for k = 1..8 (S2), a 0.8B Q4_K_M draft step 5.1 ms;
`--cost long`: the 98k-position extrapolation of "Long-context costs", a step 207 ms, verify(1 / 3 / 7) = 1.41 / 2.16 /
4.04, interpolated, a draft step 19 ms; a lookup guess costs no draft step). Data: pi's recorded sessions
(`~/.pi/agent/sessions`, local, nothing committed but these aggregates): 202 sessions, each one's main branch
rendered as runq serve renders Qwen3.8 (`tmpl_38.jinja`: the effort line, the tools' block (pi's four core tools,
approximated: the sessions do not record the schemas), every assistant message with its reasoning, tool calls in the
XML form, a non-string argument as JSON, e.g. edit's `edits` array; checked against the jinja template on 40
sessions: identical) and tokenized with the Qwen3.5 tokenizer: 206 streams (a compaction starts a new one), 16.5 M
tokens, 9,079 replies. Kept: the 4,508 replies that end before position 98,304 (the 27B's window), 2.01 M reply
tokens, labeled by kind: reasoning 512k, prose 289k, the template's and the XML's syntax 168k, tool-call arguments
648k, edits' old text 42k and new text 124k, write's content 224k. **A proxy:** these are other models' replies
(DeepSeek, GLM, Kimi, Claude, ... through pi), not the 27B's; their copying is the task's, so it should carry over,
their wording not. The lookup rule is src/spec.c's (below): `build/teacher_force -l` runs `rq_lookup_draft` at every
position, identical to the simulation's numpy rule at 98.5% of the 2.0 M positions (the rest: the 64-candidate
chain limit).

Tokens per target pass / est. tok/s (short-context cost model; plain 1.00 / 8.3), the lookup alone (`>= m`: the
longest match of up to 16 tokens, used when it has >= m; `n only`: the latest occurrence of the last n tokens):

| lookup, all 2.01 M tokens | all | reasoning | prose | syntax | args | old text | new text | write |
|---|---|---|---|---|---|---|---|---|
| n = 1 only, k = 4 | 1.50 / 9.1 | 1.31 / 7.9 | 1.28 / 7.8 | 1.90 / 11.5 | 1.68 / 10.2 | 1.82 / 10.8 | 1.66 / 10.0 | 1.49 / 9.0 |
| n = 3 only, k = 4 | 1.46 / 10.8 | 1.20 / 9.1 | 1.19 / 9.1 | 3.05 / 19.6 | 1.62 / 11.8 | 2.44 / 16.2 | 1.59 / 11.5 | 1.41 / 10.4 |
| >= 2, k = 1 | 1.33 / 10.2 | 1.19 / 9.2 | 1.18 / 9.2 | 1.78 / 13.1 | 1.41 / 10.8 | 1.70 / 12.6 | 1.41 / 10.7 | 1.31 / 10.1 |
| **>= 2, k = 3** | **1.56 / 11.4** | 1.29 / 9.5 | 1.28 / 9.5 | 2.91 / 19.9 | 1.73 / 12.6 | 2.57 / 17.8 | 1.72 / 12.3 | 1.52 / 11.1 |
| >= 2, k = 8 | 1.69 / 7.9 | 1.33 / 6.4 | 1.31 / 6.5 | 4.30 / 16.0 | 1.95 / 9.1 | 3.54 / 14.1 | 1.92 / 8.4 | 1.63 / 7.5 |
| >= 2, k = 16 | 1.73 / 4.5 | 1.33 / 3.5 | 1.32 / 3.7 | 5.09 / 9.5 | 2.02 / 5.3 | 4.09 / 8.5 | 1.98 / 4.6 | 1.67 / 4.2 |
| >= 4, k = 3 | 1.35 / 10.7 | 1.13 / 9.1 | 1.12 / 9.1 | 2.54 / 18.3 | 1.48 / 11.5 | 2.19 / 16.2 | 1.45 / 11.3 | 1.30 / 10.3 |
| >= 8, k = 8 | 1.23 / 9.5 | 1.05 / 8.5 | 1.05 / 8.5 | 2.25 / 13.7 | 1.34 / 10.0 | 1.97 / 12.9 | 1.30 / 9.8 | 1.18 / 9.2 |

A match exists at 71% of the positions (>= 2 tokens; 55% of the passes at k = 3), but most are short and wrong: at k
= 3, 34% of the lookup's guesses are accepted. By the match's length (the longest-first rule up to 16, all
positions): 2 tokens, 0.99 tokens accepted on average out of what it can guess; 4: 2.8; 8: 5.9; >= 16 (12% of the
positions): 15.6 (most of them long copies). Choosing k from the match length (the best k per length under the cost
model: 2-5) gains only 0.3 tok/s over a fixed k = 3: the verify cost of long passes (2.52x a step at k = 8, the
K-quant kernels' compute-bound rows) caps what a long copy is worth.

The 0.8B Q4_K_M teacher-forced on the same contexts (`lookup_accept.py draft`, `build/teacher_force`: the argmax at
every reply position after the whole context; 40 random streams' replies up to position 32,768, 700k positions fed,
95k reply tokens, 40 min at 6 threads), and both combined (the lookup's guesses when its match has >= m tokens, else
the model's); tokens per pass / est. tok/s, short context then 98k positions:

| 95k tokens of the sessions | all | reasoning | prose | syntax | args | old text | new text | all, 98k |
|---|---|---|---|---|---|---|---|---|
| plain | 1.00 / 8.3 | | | | | | | 1.00 / 4.8 |
| lookup >= 2, k = 3 | 1.50 / 11.2 | 1.30 / 9.8 | 1.26 / 9.5 | 2.74 / 18.9 | 1.61 / 12.0 | 2.18 / 15.4 | 1.62 / 11.8 | 1.50 / 4.7 |
| lookup >= 3, k = 3 | 1.39 / 10.8 | | | | | | | 1.39 / **5.2** |
| lookup >= 4, k = 3 | 1.31 / 10.4 | | | | | | | 1.31 / **5.2** |
| model k = 3 | 2.40 / 14.1 | 2.07 / 12.2 | 2.04 / 12.0 | 3.52 / 20.7 | 2.59 / 15.2 | 3.25 / 19.1 | 2.88 / 16.9 | 2.40 / 4.6 |
| model k = 6, tau 0.4 | 2.56 / 14.2 | 2.02 / 12.4 | 1.98 / 12.2 | 5.28 / 22.3 | 2.95 / 15.0 | 4.86 / 20.3 | 3.56 / 17.1 | 2.56 / 5.0 |
| lookup >= 2, k 3 + model k 3 | 2.16 / 13.4 | | | | | | | 2.16 / 4.4 |
| lookup >= 4, k 3 + model k 3 | 2.32 / 14.1 | | | | | | | 2.32 / 4.6 |
| lookup >= 8, k 3 + model k 3 | 2.38 / 14.2 | 2.05 / 12.1 | 2.03 / 12.0 | 3.55 / 22.4 | 2.56 / 15.4 | 3.17 / 19.3 | 2.81 / 16.8 | 2.38 / 4.6 |
| lookup >= 8, k 8 + model k 6, tau 0.4 | 2.57 / 13.9 | 2.01 / 12.0 | 1.97 / 12.0 | 5.94 / 21.9 | 2.97 / 14.5 | 4.59 / 18.0 | 3.52 / 16.5 | 2.57 / 4.9 |

On the 27B's own answers (`logs/specmeas-2026-10-27`, the 0.8B Q4_K_M's teacher-forced guesses of `spec_accept.py
measure`; short prompts, nothing agentic; `lookup_accept.py report --answers GEN:MEAS`): thinking off (6,524 tokens)
lookup >= 2, k = 3 1.20 per pass (9.5 tok/s; code 1.29, the edits of the prompt's code 1.24, extraction 1.40),
model k = 3 2.94 (17.3),
combined >= 8 / k 3 2.93 (17.3), >= 2 2.67 (16.1); thinking on (9,265) 1.27 (10.0), 2.38 (13.9), 2.36 (14.0), 2.23 (13.5);
the 3 long reasonings (3,522) 1.30 (9.9), 2.35 (13.8), 2.33 (13.8).

- **The lookup alone: ~1.5 tokens per pass on agentic replies** (1.2-1.3 on the 27B's chat-style answers), ~+35% at
  short context (11.4 vs 8.3 tok/s in the cost model), half of the 0.8B draft's gain (14.1). Where it shines: the
  template's and the XML's syntax (2.9 per pass), edits' old text (2.6), tool arguments (1.7); reasoning and prose
  1.3, new code (write) 1.5. k = 3 with matches of >= 2 tokens is the best fixed setting; a unigram match is worse
  than none past k = 1.
- **Combined, it does not beat the draft at short context:** the 0.8B guesses copies as well (old text 3.25 per pass
  vs the lookup's 2.18; it reads the same context), so a lookup pass replaces a better model pass; with long matches
  only (>= 8) the combination equals the model alone (14.2 vs 14.1). Lookup's value with a draft is its cost: none.
- **At long context it is the best speculation measured:** at 98k (a verify row ~80-90% of a step's attention, a
  draft step 19 ms) the model draft at k = 3 is slower than plain (4.6 vs 4.8 tok/s; tau 0.4: 5.0), the lookup alone
  (>= 3-4 tokens, k = 3) 5.2 (+8%), with no draft model in memory (2 GiB at 96k: G3). A per-position choice (G7)
  should use the lookup there, the model at short context.
- **Edits' arguments are JSON:** pi's `edit` takes an `edits` array, so its old / new text is a JSON string (escaped
  newlines and quotes) inside `<parameter=edits>`, while the file's text in the tool results is raw: the lookup
  matches the old text line by line, not across lines (2.6 per pass). A tool schema with string parameters
  (`old_text` / `new_text`, written raw in the XML) would let whole blocks match.

**In the loop** (`src/spec.c`, `RqSpecOpts.lookup` / `.lookup_min`; CLI `--lookup N [--lookup-min M]`, with or
without `-D`): the index is a hash table of chains over the pairs of consecutive tokens of the target's sequence
(`RqLookup`: `head[hash]` the latest position, `prev[i]` the previous one with the same hash; 1.4 MiB at 100k
positions, built over the prompt when `rq_spec_run` starts, then extended by the accepted tokens: O(1) per token). A
lookup walks the chain of (last token, t) from the most recent, at most 64 candidates, and keeps the longest match
backwards (up to 16 tokens; the most recent among equals), which is the choice of trying the longest n-gram first (n
= 16 .. 2) while the chain reaches it; it guesses up to N of the tokens after it if the match has >= M tokens
(default 2), stopping at a token the sampling mask forbids. Else the draft model guesses (with `-D`), else the pass
is a plain step. The model's sequence is not fed during lookup passes: it catches up by common prefix on its next
round (one prefill). The verify / accept / rollback steps are unchanged; the target gets kmax + 2 checkpoint slots
(kmax = max(k, N)). `RqSpecStats` counts the lookup's passes, guesses and accepted ones; the CLI's `[timing]
speculative` line adds them. `rq_lookup_*` are in the API for tools that replay the loop (`tests/teacher_force.c
-l`).

**Equivalence** (`make test`'s `spec` group): `api_test --spec` runs every case with the lookup alone (4 guesses;
30 with matches of >= 4) and combined (lookup 8 with >= 3 + k 3; lookup 3 + k 8 tau 0.5) besides the model settings:
tokens, RNG states, every sampled logits vector's hash, stops and the cache == `rq_generate`, on the 0.6B Q8_0 + Q4_K_M
pair (and the generic build, `--quick`) and the Qwen3.5-0.8B pair (hybrid rollbacks: every one on its checkpoint);
two new cases copy a block of their prompt (greedy; seeded T 0.6 top-k 20 over two calls): the lookup alone guesses
4.2 / 4.8 tokens per pass there (`spec_lookup_copy`, >= 3 required); `spec_lookup_index`: `rq_lookup_draft` ==
a brute-force scan at 71,814 positions of synthetic sequences. The CLI cases run `--lookup 4` (no `-D`) and `-G 3
--lookup 8 --lookup-min 3` against plain (stdout, stderr but timing, chat dumps), and a generation that copies its
prompt (`gen_copy`: 7.8 tokens per pass with either setting, all 40 guesses accepted).

**Speed** (2026-10-28, `scripts/bench_lookup.py`, `runq -m chat -t 0 -n 384 -c 4096`, 8 threads, under the bench lock,
**load 18-25 from other jobs**: the plain rates are ~40% below their usual, the ratios are what to read; 2 reps,
median [min-max]; accepted / guessed, tokens per pass, lookup passes / passes; every output == plain's). Prompts:
`edit_py` (rewrite a 19-line Python function with a parameter renamed and None -> 0.0, "return the full function"),
`edit_c` (rename a field in a 36-line C ring buffer and change one function, "return the complete code"), `chat`
("What are the main differences between TCP and UDP?", nothing to copy).

| Qwen3-1.7B Q8_0, draft 0.6B Q4_K_M | edit_py | edit_c | chat | mean |
|---|---|---|---|---|
| plain | 46.4 [40-53] | 36.3 [35-38] | 50.5 [43-58] | 44.4 |
| `-G 3` | 63.4 [53-73] (91%, 3.73) | 57.7 [57-59] (93%, 3.79) | 58.9 [55-63] (55%, 2.67) | 60.0 |
| `--lookup 3` | 108.6 [85-132] (95%, 3.50) | 97.1 [83-112] (84%, 3.24) | 63.6 [56-71] (42%, 1.22) | 89.8 |
| `--lookup 8` | 129.5 [122-137] (89%, 6.46) | 132.3 [125-139] (71%, 5.54) | 57.6 [53-62] (22%, 1.25) | 106.4 |
| `-G 3 --lookup 3` | 102.3 [92-113] (94%, 3.82, 40/44) | 92.8 [81-105] (84%, 3.51, 75/82) | 55.4 [51-60] (50%, 2.49, 36/154) | 83.5 |
| `-G 3 --lookup 8 --lookup-min 4` | 136.0 [125-147] (88%, 7.30, 19/23) | 134.6 [111-158] (88%, 6.86, 31/42) | 51.1 [45-57] (53%, 2.74, 8/140) | 107.2 |

| Qwen3.5-0.8B Q8_0 (no draft) | edit_py | edit_c | chat | mean |
|---|---|---|---|---|
| plain | 86.6 [82-91] | 73.9 [65-83] | 84.1 [77-91] | 81.5 |
| `--lookup 3` | 233.3 [218-249] (98%, 3.93) | 198.1 [194-202] (100%, 4.00) | 95.3 [79-111] (35%, 1.20) | 175.6 |
| `--lookup 8` | 336.9 [298-376] (95%, 8.62) | 271.8 [245-298] (97%, 8.75) | 86.7 [78-95] (16%, 1.22) | 231.8 |
| `--lookup 8 --lookup-min 4` | 268.7 [239-299] (99%, 8.23) | 324.6 [287-362] (100%, 8.48) | 88.3 [82-95] (15%, 1.07) | 227.2 |

- **Edits that return the code: 2.8-3.9x** with the lookup alone at k = 8 (5.5-8.8 tokens per pass), against 1.4-1.6x
  for the 0.6B draft at `-G 3`: the copy runs longer than any draft's k, and a lookup guess costs nothing (the 0.6B's
  step is a third of the 1.7B's). Combined (`-G 3 --lookup 8 --lookup-min 4`) is the best on the 1.7B's edits (the
  model fills in where the copy breaks off), below it on chat (51 vs 59 tok/s; the runs' spread is ~±12%).
- **Nothing to copy (chat):** the lookup alone is within the noise of plain (1.2 tokens per pass; its guesses there
  cost verify rows, not draft steps); with the draft, lookups with short matches (min 2) take passes from the model
  (`-G 3 --lookup 3`: 55 vs 59 tok/s): with `-D`, give the lookup a longer minimum match (4-8), as the offline
  numbers say.
- These replies are mostly copies (the prompts ask for the full code), so they are the lookup's best case; the
  agentic sessions above have ~1.5 tokens per pass from it.

Rerun:

```sh
P=".venv/bin/python -I scripts/lookup_accept.py"; S=/var/tmp/streams.pkl   # (local data: not in the repo)
$P sessions $S --check-template logs/specmeas-2026-10-27/tmpl_38.jinja                       # ~30 s
$P report $S --by-kind --engine [--cost long]                                              # ~1 min
make teacher-force && $P draft $S models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf -o /var/tmp/draft.npz \
    --max-pos 32768 --budget 700000 --threads 6                                            # ~40 min
$P report $S --draft /var/tmp/draft.npz --by-kind [--cost long]
L=logs/specmeas-2026-10-27; $P report --by-kind --answers $L/out_nothink.json:$L/meas_Qwen3.5-0.8B-Q4_K_M_out_nothink.json
```

## Not now

DFlash 2 (a 2B block-diffusion drafter: a new architecture, a 4 GB BF16 download); multi-VM splits (~10-30% over
one-VM speculation at typical acceptance); rejection sampling (decision 2); IQ / UD types (decision 4).
