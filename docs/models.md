# Models and files

Which models and GGUF files runq loads, how to fetch them, and what is not supported. For measurements of each file
(speed, memory, perplexity) see [BENCHMARKS.md](../BENCHMARKS.md#all-model-files-on-the-epyc-vm-2026-10-26-to-2026-10-30);
how to run them, [usage.md](usage.md).

runq reads the model's weights, tokenizer and hyperparameters from the GGUF file (chat templates are built in), and
qwen3.c's Q8_0 `.bin` export. `scripts/download_models.py --list` shows every downloadable set; each file is checked
by sha256 or git blob hash at a pinned Hugging Face revision. The IQ3_S-mtp file of ISTA-DASLab used in the
llama.cpp / ik_llama.cpp comparison (Qwen3.8-27B, 12.1 GB, with its MTP block) is not one of the sets.

## Download sets

| set (`download_models.py`) | file | size | notes |
|---|---|---|---|
| `q8gguf` | `Qwen3-0.6B-Q8_0.gguf` (unsloth) | 639 MB | the closest to the fp32 model of the Qwen3-0.6B files (perplexity +0.19% over HF fp32, [DEVELOPMENT.md](DEVELOPMENT.md#qwen3-06b-which-file-to-use)) |
| `q4km` | `Qwen3-0.6B-Q4_K_M.gguf` (unsloth) | 397 MB | ~1.5x faster decode than the Q8_0 file, ~40% less RAM, +4.9% perplexity over HF fp32 (same table) |
| `q5km` | `Qwen3-0.6B-Q5_K_M.gguf` (unsloth) | 444 MB | between the two: 1-cos vs HF 5.4e-3 (Q8_0 6.3e-4, Q4_K_M 1.6e-2) |
| `1.7b-q8`, `1.7b-q4km` | `Qwen3-1.7B-Q8_0.gguf`, `-Q4_K_M.gguf` | 1,834 / 1,107 MB | as the LiteLLM router's judge (the paragraph on the judge in [server.md](server.md)): 88% on the project's 80-prompt eval set at ~50 ms per classification; the 0.6B scores 51-71% |
| `4b-q8`, `4b-q4km` | `Qwen3-4B-Q8_0.gguf`, `-Q4_K_M.gguf` | 4,280 / 2,497 MB | |
| `4b-2507i-q4km` | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | 2,497 MB | Instruct-2507: never thinks (`-r` is ignored, with a note) |
| `4b-2507t-q4km` | `Qwen3-4B-Thinking-2507-Q4_K_M.gguf` | 2,497 MB | Thinking-2507: always thinks (`-r` is ignored, with a note) |
| `8b-q4km` | `Qwen3-8B-Q4_K_M.gguf` | 5,028 MB | the largest dense Qwen3 file tested |
| `3.5-0.8b-q8` | `Qwen3.5-0.8B-Q8_0.gguf` (unsloth) | 812 MB | Qwen3.5: hybrid layers, 248k vocabulary, thinking off by default ([docs/qwen35.md](qwen35.md)) |
| `3.5-0.8b-q4km` | `Qwen3.5-0.8B-Q4_K_M.gguf` (unsloth) | 533 MB | Q4_K / Q5_K / Q6_K / Q8_0 mix: decode ~1.27x the Q8_0 file (188 vs 148 tok/s), prefill ~0.88x, peak RSS 662 vs 963 MB; vs HF fp32 pooled cos 0.9946, top-1 57/62 |
| `3.5-0.8b-q3km` | `Qwen3.5-0.8B-Q3_K_M.gguf` (unsloth) | 470 MB | Q3_K for `attn_q` / `attn_k` / `attn_gate` / `ffn_gate` / `ffn_up` (the 27B Q3_K_M file's 3-bit roles), Q4_K / Q5_K / Q6_K / Q8_0 for the rest: vs its Q4_K_M decode 211 / 209 vs 206 / 192 tok/s, prefill ~0.98x, peak RSS 601 vs 663 MB; vs HF fp32 pooled cos 0.981, top-1 54/62 (Q4_K_M 0.9946, 57/62) |
| `3.5-4b-q8` | `Qwen3.5-4B-Q8_0.gguf` (unsloth) | 4,482 MB | Qwen3.5-4B: 32 value heads on 16 key heads in its linear layers; as fast as Qwen3-4B Q8_0, perplexity 10.59 (Qwen3-4B: 15.65) |
| `3.8-27b-q4km` | `Qwen3.8-27B-Q4_K_M.gguf` (unsloth) | 17,107 MB | Qwen3.8-27B (thinks by default; its MTP block is the draft of `-D mtp`, TODO S9): ~16-16.5 GB of RAM, 8.7 tok/s plain, 13-17 with `-D` Qwen3.5-0.8B ([docs/notes/qwen38-speculative.md](notes/qwen38-speculative.md)) |
| `3.8-27b-q3km` | `Qwen3.8-27B-Q3_K_M.gguf` (unsloth) | 13,819 MB | the same model in 3-bit (Q3_K) for its FFN gate / up, attention q / k / gate and embeddings: ~13.3 GB of RAM, the file for 96k agentic sessions with the draft ([server.md](server.md#qwen38-27b-with-pi)); vs the Q4_K_M: top-1 = the API model's 96.5 / 94.5% (thinking off / on; Q4_K_M 98.4 / 96.9%), perplexity +2.7%, decode +17-22% at 8 threads |
| `saluki-27b` | `Underdog-Saluki-27B-1.0-IQ2-mix.gguf` (ConwayResearch) | 7,898 MB | Qwen3.8-27B as a Q2_K / IQ1_S .. IQ4_XS mix tuned for tool calling (thinks by default; Qwen's own Qwen3.8 template): 8.4 GB of RAM, decode 13-15 tok/s at 16 threads (the Q3_K_M file: 9.6-12.3, 2026-10-30), prefill ~0.72x the Q3_K_M's, perplexity +18% over it ([docs/iquants.md](iquants.md)) |
| `hf`, `1.7b-hf`, `3.5-0.8b-hf`, `3.5-4b-hf` | `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-4B` (bf16 safetensors) | 1.5 / 4.1 / 1.8 / 9.3 GB | for the `.bin` export and the test goldens |

## Qwen3.5

Qwen3.5 keeps a fixed-size recurrent state per sequence (19 MB for 0.8B) besides its small KV cache. That state
cannot be rewound, so a chat that edits its history (a thinking reply's reasoning, dropped from the next turn) goes
back to a checkpoint the CLI and the server keep at each prompt's assistant header and prefills the rest again, with
the same results as a fresh run.

Design notes: [qwen35.md](qwen35.md).

## qwen3.c's `.bin`

qwen3.c's `.bin` (Q8_0, group 64) works too, exported from the HF weights with qwen3.c's exporter.
The tests use it, and it is the format for comparing with upstream: `scripts/download_models.py hf &&
scripts/setup_venv.sh && scripts/export_q8.sh` (a `.venv` with CPU torch) gives `models/Qwen3-0.6B-q8.bin`.
`make models` does the same.

## Not supported, and untested

Not supported: other GGUF types (Q4_0, Q4_1, IQ4_NL, TQ1_0 / TQ2_0, F16 / BF16 matrices, and the files that use them:
rejected with a clean error; Unsloth's "UD" files made of Q2_K / Q3_K .. Q6_K / the i-quants / Q8_0 load), Qwen3-14B and larger (untested),
the MoE models (`qwen3moe`, `qwen35moe`: Qwen3.5-35B-A3B and up, Qwen3.6-35B-A3B), images, Qwen3-Embedding / Reranker,
RoPE scaling (YaRN), and Hadamard-folded GGUFs (`prism.hadamard.*` metadata: rejected). The other dense Qwen3.5 / 3.6
sizes should load but are untested: 2B (as many value heads as key heads, like 0.8B), 9B, 27B and Qwen3.6-27B (2x / 3x
as many value heads, like 4B; the 27B shape is covered by synthetic test models). **Qwen3.8-27B** (Unsloth's Q4_K_M,
docs/notes/qwen38-speculative.md) loads its 64 layers; the MTP block after them (`nextn_predict_layers` 1,
`blk.64`: an attention layer + `nextn.*`) is checked (names, shapes, types) at load and read by `-D mtp` /
`--draft mtp`, and its F32 (BF16 in ISTA's IQ3_S-mtp file: widened to F32 at
load) `ssm_alpha` / `ssm_beta` matrices are applied in fp32; tested on the real file (logits vs its numpy reference cos
0.9999, top-1 62/62; the bitwise invariants; the models group's `3.8-27b-q4km`, opt-in; its Q3_K_M too, `3.8-27b-q3km`:
cos 0.99983, top-1 62/62; a fact ~84k tokens back retrieved from an 88k-token prompt). Other MTP layouts (several
blocks, `qwen3` files) are refused with a clear message.
