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
  `token_embd` and `output` with the main model. runq's layer pattern is computed from `block_count`, so the loader
  refuses a file with `nextn_predict_layers` > 0 (a clear message, before any weight is read; the tokenizer alone
  loads) until the nextn handling (TODO S6). Details: [file layout](#file-layout).
- Sampling defaults in the GGUF (`general.sampling.*`): temperature 1.0, top-p 0.95, top-k 20 (runq's defaults for
  the options not given, CLI and server: `rq_model_sampling`).

**The file:** unsloth `Qwen3.8-27B-Q4_K_M.gguf` at revision `4a03c640833598c0fb5ddd11e846135e906eb764`, 17.1 GB,
sha256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`. Types: Q4_K 11.4 GB, Q6_K 4.5, Q5_K 1.0,
Q8_0 0.06, F32 0.1, all supported by runq's kernels, but `ssm_alpha` / `ssm_beta` are F32 *matrices* here, which the
loader does not take yet (TODO S6). Avoid: `main`'s `UD-Q4_K_M` (16.5 GB: IQ4_XS / Q3_K / IQ4_NL / IQ3_S),
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
runq does not read: `nextn_predict_layers` (only to refuse the file), `tokenizer.ggml.bos_token_id` 248044 (absent in the 0.8B's), `general.*`,
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

No `nextn.embed_tokens` / `nextn.shared_head_head`: the MTP block uses `token_embd` and `output`. Two things the loader
must learn (TODO S6): the main layers are `block_count` - `nextn_predict_layers` (today: refused), and F32 matrices
for `ssm_alpha` / `ssm_beta` (today: "has ggml type 0, need Q8_0 ..."). `tests/make_synth_q35.py` builds this layout
at a small scale (`synth-q35-27b.gguf`, dim 768, GQA 6, r = 3, the types per role; `-mtp`: + the MTP block), checked by
the `q35` group ([tests/README.md](../tests/README.md)).

## Memory and disk

- Weights: repacked into anonymous memory; the loader `pread`s the file, so it also passes through the page cache,
  which it drops matrix by matrix as they are repacked for a file of >= 1/4 of the RAM (`--page-cache`; measured on
  the 8B with `drop`: 4.7 GB of cache freed, +0.35 s on a 6 s cold load). The untied `token_embd` stays in the mapping: its rows are
  dequantized at lookup (no 5 GB fp32 table).
- The load-time estimate (`--mem`; computed from the tensor directory before any weight is read, refused past
  MemAvailable; within 3% of the peak RSS with the context full for Qwen3-0.6B to 8B and Qwen3.5-0.8B / 4B,
  [tests/README.md](../tests/README.md#memory)), for the 27B from its tensor directory with the same rules
  (2026-10-27, by script; the file is refused at its MTP block before the check, TODO S6): repacked weights
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
  merged. `RQ_CHAT_QWEN35` renders none of this: a new variant (TODO S6).

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
Qwen3.8's template itself (thinking on: the `xhigh` effort line, `--effort`; `--template tmpl_38.jinja` checks it
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
matmuls). Not included: the checkpoint copies per row (TODO S3: ~150 MiB each, estimated ~2-4 ms, 2-3% of a step).
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

## Not now

DFlash 2 (a 2B block-diffusion drafter: a new architecture, a 4 GB BF16 download); multi-VM splits (~10-30% over
one-VM speculation at typical acceptance); rejection sampling (decision 2); IQ / UD types (decision 4).
