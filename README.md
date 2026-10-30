# Gwenedeg

Gwenedeg runs Qwen3 dense language models, and the dense Qwen3.5 models (0.8B and 4B tested), on the CPU, in plain C with no external dependencies. It is a small
inference library (`src/runq.h`), a command line, and an OpenAI-compatible HTTP server. **The binary is
called `runq`**, as in qwen3.c, the project it grew from. The library keeps the name too (`runq.h`,
`rq_` prefix).

- **Models**: Qwen3-0.6B, 1.7B, 4B and 8B, plus Qwen3-4B-Instruct-2507 / Thinking-2507, and Qwen3.5-0.8B and 4B (the
  hybrid architecture: Gated DeltaNet linear attention + gated attention; 0.8B Q8_0, Q4_K_M and Q3_K_M, 4B Q8_0), Qwen3.8-27B, and
  **Underdog Saluki 27B** (Qwen3.8-27B in 7.9 GB, an i-quant mix tuned for tool calling). It loads GGUF
  **Q8_0**, **Q4_K_M**, **Q5_K_M**, **Q3_K_M** and the **i-quant / Q2_K** mixes (IQ1_S .. IQ4_XS) directly (weights, tokenizer and hyperparameters come from the file; chat templates are built
  in), and also qwen3.c's Q8_0 `.bin` export.
- **Kernels**: AVX-512 VNNI on x86-64, NEON (dot product) on AArch64 / Apple Silicon, and generic C
  everywhere else. The SIMD kernels give bit-identical results to the generic ones.
- **Deterministic**: logits are bit-identical whatever the thread count, the prefill chunk size or the
  batch composition. Batched prefill equals token-by-token, and a batched decode row equals decoding
  that sequence alone. The test suite checks all of this.
- **Fast prefill and decode**: int8 GEMM / GEMV on weights repacked at load, a tiled f16 KV cache, its
  own thread pool. On a 16-vCPU AMD EPYC Genoa VM, Qwen3-0.6B prefills ~2,000-2,700 tok/s and decodes
  ~200-350 tok/s (below).
- **Server** (`runq serve`): OpenAI chat completions with streaming and continuous batching (requests
  share each forward pass), prefix KV-cache reuse across requests, tool calls (Qwen3.5 / Qwen3.8: agent
  harnesses such as pi run their loop against it), and constrained decoding for JSON schemas with a finite
  set of outputs. That last feature makes it a fast LLM judge for
  [LiteLLM](https://github.com/BerriAI/litellm)'s complexity auto-router.
- **Exact tokenizer**: byte-level BPE matching Hugging Face `tokenizers` on Qwen's tokenizer, including
  NFC normalization (tested against HF).

Platforms: Linux (gcc) is the main platform. macOS on Apple Silicon (Apple clang) builds and runs with
NEON kernels. The Windows code paths inherited from qwen3.c are kept but untested, and the server is
POSIX only.

## Quick start

Requirements: `make` and gcc (Linux) or Xcode's command-line tools (macOS). Python 3 (standard
library only) runs the model download script.

```sh
make                                      # -> ./runq   (make generic: ./runq-generic, make debug: ./runq-debug)
scripts/download_models.py q8gguf         # models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf (639 MB), pinned + checksummed
G=models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf

./runq $G -c 4096                         # interactive chat: one message per line, empty line / EOF ends
./runq $G -c 4096 -t 0 -n 200 -i "Why is the sky blue?"     # one turn, greedy, at most 200 tokens
./runq $G -c 4096 -r 1                    # with thinking (Qwen3's reasoning mode)
./runq serve $G                           # OpenAI-compatible server on http://127.0.0.1:8080/v1

curl -s localhost:8080/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Hi!"}], "max_tokens": 32}'
```

`scripts/download_models.py --list` shows every downloadable set (table below). Files go to
`models/` (or `--dest DIR`, `$GWENEDEG_MODELS`). Each file is checked by sha256 or git blob hash at a
pinned Hugging Face revision.

CLI options (`./runq` without arguments prints them all):

| option | |
|---|---|
| `-c N` | context size in tokens; default: the model's maximum, at most 40,960. Pass it: the KV cache is reserved for the whole context (as address space, touched only as used) |
| `-t T`, `-p P`, `-s S` | temperature (default 1.0, 0 = greedy), top-p (0.9), seed |
| `-K N` | top-k: sample among the N most likely tokens, then top-p among them (default 0 = off). `-t` / `-p` / `-K` not given: the model's recommended values if its GGUF has them (`general.sampling.*`, e.g. Qwen3.8-27B 1.0 / 0.95 / 20; none of the Qwen3 / Qwen3.5 files here do) |
| `-n N` | max new tokens per answer (default unlimited) |
| `-i TEXT`, `-y TEXT` | a single prompt (one turn, then exit); the system prompt |
| `-r 0\|1` | thinking off (default; Qwen3.8: on) / on |
| `-e xhigh\|high\|medium\|low` | reasoning effort with thinking (Qwen3.8's template; default xhigh, high = xhigh) |
| `-m MODE` | `chat` (default), `generate` (raw completion), `tokenize`; test hooks `logits`, `dequant`, `render` |
| `-k f16\|bf16\|f32\|bf16-dot\|f16-dot` | KV cache type, default f16 (half the memory of f32, as accurate); `bf16-dot`: a bf16 cache with the bf16 dot-product attention, ~1.7x faster attention at long context, opt-in (below) |
| `-b N` | prefill chunk size, default 512 (`-b 1`: token by token) |
| `--mem GiB` | memory budget of the load check (below), default MemAvailable; `0` or `force`: no check. Given, prints the estimate (a `[memory]` line) |
| `--page-cache auto\|drop\|keep` | the model file's page cache after the load; `auto` (default) drops it if the file is >= 1/4 of the RAM |
| `-D FILE` | speculative decoding (generate and chat): a smaller draft model of the same vocabulary (e.g. Qwen3-0.6B for Qwen3-1.7B, Qwen3.5-0.8B for Qwen3.5) guesses, the model verifies; the output is exactly plain decoding's (below) |
| `-G K`, `--tau T` | with `-D`: at most K guesses per pass of the model (default 3, at most 30; `-G 0`: plain decoding, the draft not loaded); adaptive length: stop guessing at the first guess whose draft top-1 probability is < T (default 0 = off, always K) |
| `--lookup N`, `--lookup-min M` | prompt lookup (generate and chat), with or without `-D`: guess up to N tokens (at most 30) by copying what followed the latest, longest match (>= M tokens, default 2; up to 16 counted) of the last tokens earlier in the context; no model, exact as any draft; with `-D` the draft model guesses where the lookup finds no match (below) |
| `OMP_NUM_THREADS` | threads; default nproc - nproc/8 (Apple Silicon: the performance cores) |

**Memory.** Before reading any weight, the loader estimates what the run needs from the file's tensor directory:
the repacked weights, the runtime (tokenizer, metadata, a compute context of 512 rows) and one sequence of `-c`
positions (its whole KV cache, and for Qwen3.5 the recurrent state with its 2 checkpoints; `runq serve`: its
`--checkpoints`, default 4). It refuses the file
with a one-line message and exit status 1 when that exceeds MemAvailable (Linux, the host's: a cgroup limit is not read; no check elsewhere
unless `--mem` gives a budget): a smaller `-c` lowers it, `--mem 0` loads anyway. The estimate is within 3% of the peak RSS
of a run that fills the context (Qwen3-0.6B to 8B, Qwen3.5-0.8B and 4B, [tests/README.md](tests/README.md#memory)).
The weights are read with `pread`, so the file also passes through the page cache: for a big file (>= 1/4 of the
RAM, e.g. Qwen3.8-27B) the loader drops it matrix by matrix as they are repacked (`posix_fadvise` DONTNEED;
+0.35 s on the 8B's cold load, +0.6 s warm, when forced with `--page-cache drop`), for smaller ones it stays to make the next load fast
(~1 s warm vs ~6 s cold for the 8B here).

Chat keeps the whole conversation in the KV cache, so each turn prefills only the new tokens. Earlier
answers drop their `<think>` block, as Qwen3's template does. When the context is full, the oldest turns
are dropped (the system prompt is kept) and the rest is re-prefilled. A `[timing]` line on stderr gives
prefill and decode speed. Details: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

**Speculative decoding** (`-D DRAFT`, generate and chat): a smaller model of the same vocabulary guesses up to `-G`
tokens greedily, the model checks them in one batched pass and keeps the guesses its own sampling agrees with, plus
one token of its own. Batched passes are bit-identical to decode steps, and the model samples every token with its
own RNG exactly as plain decoding does, so the output is plain decoding's, greedy or seeded, whatever the draft: the
draft only changes the speed. A `[timing] speculative` line follows each `[timing]` line (passes, guesses drafted /
accepted, tokens per pass, the draft's and the verify passes' time). Qwen3-1.7B Q8_0 with Qwen3-0.6B Q4_K_M as the
draft, 8 threads, 256 tokens, `-G 3` (2026-10-27, the shared VM at load 4-10): 111 tok/s on code, 88 on chat, 119 with
thinking, 87 on code at temperature 1, against 70-78 plain (1.15-1.5x; 45-80% of the guesses accepted). A Q8_0 draft
guesses the same but its step is slower (5.6 vs 4.2 ms). Qwen3.5 targets and drafts work too (rolled back through
state checkpoints). **Qwen3.8-27B Q4_K_M** with Qwen3.5-0.8B Q4_K_M as the draft, 8 threads, 256 tokens, the mean of
chat, code, math and summary prompts: 8.7 tok/s plain; thinking on 13.2 at `-G 3`, 14.1 at `-G 6 --tau 0.4` (1.6x),
thinking off 16.6 / 17.2 (2.0x; code and math 19-22); 12 threads: up to 15.3 / 20.6. In `runq serve` too (below).
`--spec auto` (opt-in in the CLI) chooses the source and the guess count per pass from a cost model calibrated at the
start (TODO G7).
Measurements: [docs/qwen38-speculative.md](docs/qwen38-speculative.md#the-27b-bring-up-s7).

**Prompt lookup** (`--lookup N`, no model needed): replies that copy from their context (code being edited, paths,
tool output) are guessed from the context itself, verified in the same loop (so the output is still plain decoding's).
On edits that return the prompt's code, `--lookup 8` decodes 2.8-3.9x faster (Qwen3-1.7B, Qwen3.5-0.8B; the 0.6B draft
at `-G 3`: 1.4-1.6x); with nothing to copy it is about plain speed. With `-D`, the draft guesses where the lookup
finds nothing: `-G 3 --lookup 8 --lookup-min 4` (a longer minimum match: the draft guesses short matches better).
On recorded agentic sessions it guesses ~1.5 tokens per pass alone (the 0.8B draft 2.4): [docs/qwen38-speculative.md](docs/qwen38-speculative.md#prompt-lookup-todo-g6-g0-c-measured-2026-10-27).

## Models

| set (`download_models.py`) | file | size | notes |
|---|---|---|---|
| `q8gguf` | `Qwen3-0.6B-Q8_0.gguf` (unsloth) | 639 MB | closest to the fp32 model, best for prompt-heavy work |
| `q4km` | `Qwen3-0.6B-Q4_K_M.gguf` (unsloth) | 397 MB | ~1.5x faster decode, ~40% less RAM, +4.9% perplexity |
| `q5km` | `Qwen3-0.6B-Q5_K_M.gguf` (unsloth) | 444 MB | between the two: 1-cos vs HF 5.4e-3 (Q8_0 6.3e-4, Q4_K_M 1.6e-2) |
| `1.7b-q8`, `1.7b-q4km` | `Qwen3-1.7B-Q8_0.gguf`, `-Q4_K_M.gguf` | 1,834 / 1,107 MB | 1.7B Q8_0: the smallest good LiteLLM judge |
| `4b-q8`, `4b-q4km` | `Qwen3-4B-Q8_0.gguf`, `-Q4_K_M.gguf` | 4,280 / 2,497 MB | |
| `4b-2507i-q4km` | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | 2,497 MB | the best 4B for chat (never thinks) |
| `4b-2507t-q4km` | `Qwen3-4B-Thinking-2507-Q4_K_M.gguf` | 2,497 MB | always thinks |
| `8b-q4km` | `Qwen3-8B-Q4_K_M.gguf` | 5,028 MB | quality over speed |
| `3.5-0.8b-q8` | `Qwen3.5-0.8B-Q8_0.gguf` (unsloth) | 812 MB | Qwen3.5: hybrid layers, 248k vocabulary, thinking off by default ([docs/qwen35.md](docs/qwen35.md)) |
| `3.5-0.8b-q4km` | `Qwen3.5-0.8B-Q4_K_M.gguf` (unsloth) | 533 MB | Q4_K / Q5_K / Q6_K / Q8_0 mix: decode ~1.27x the Q8_0 file (188 vs 148 tok/s), prefill ~0.88x, peak RSS 662 vs 963 MB; vs HF fp32 pooled cos 0.9946, top-1 57/62 |
| `3.5-0.8b-q3km` | `Qwen3.5-0.8B-Q3_K_M.gguf` (unsloth) | 470 MB | Q3_K for `attn_q` / `attn_k` / `attn_gate` / `ffn_gate` / `ffn_up` (the 27B Q3_K_M file's 3-bit roles), Q4_K / Q5_K / Q6_K / Q8_0 for the rest: vs its Q4_K_M decode 211 / 209 vs 206 / 192 tok/s, prefill ~0.98x, peak RSS 601 vs 663 MB; vs HF fp32 pooled cos 0.981, top-1 54/62 (Q4_K_M 0.9946, 57/62) |
| `3.5-4b-q8` | `Qwen3.5-4B-Q8_0.gguf` (unsloth) | 4,482 MB | Qwen3.5-4B: 32 value heads on 16 key heads in its linear layers; as fast as Qwen3-4B Q8_0, perplexity 10.59 (Qwen3-4B: 15.65) |
| `3.8-27b-q4km` | `Qwen3.8-27B-Q4_K_M.gguf` (unsloth) | 17,107 MB | Qwen3.8-27B (thinks by default; its MTP block is not used): ~16-16.5 GB of RAM, 8.7 tok/s plain, 13-17 with `-D` Qwen3.5-0.8B ([docs/qwen38-speculative.md](docs/qwen38-speculative.md)) |
| `3.8-27b-q3km` | `Qwen3.8-27B-Q3_K_M.gguf` (unsloth) | 13,819 MB | the same model in 3-bit (Q3_K) for its FFN gate / up, attention q / k / gate and embeddings: ~13.3 GB of RAM, the file for 96k agentic sessions with the draft (below, Agentic development); vs the Q4_K_M: top-1 = the API model's 96.5 / 94.5% (thinking off / on; Q4_K_M 98.4 / 96.9%), perplexity +2.7%, decode +17-22% at 8 threads |
| `saluki-27b` | `Underdog-Saluki-27B-1.0-IQ2-mix.gguf` (ConwayResearch) | 7,898 MB | Qwen3.8-27B as a Q2_K / IQ1_S .. IQ4_XS mix tuned for tool calling (thinks by default; Qwen's own Qwen3.8 template): 8.4 GB of RAM, decode 13-15 tok/s at 16 threads (the fastest 27B here), prefill ~0.72x the Q3_K_M's, perplexity +18% over it ([docs/iquants.md](docs/iquants.md)) |
| `hf`, `1.7b-hf`, `3.5-0.8b-hf`, `3.5-4b-hf` | `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-4B` (bf16 safetensors) | 1.5 / 4.1 / 1.8 / 9.3 GB | for the `.bin` export and the test goldens |

Measured on the 16-vCPU AMD EPYC Genoa VM (shared, noisy: best of 2, interleaved, 2026-10-26; prefill of
451 tokens, decode of 128 tokens after a short prompt, perplexity on the 4,142-token
`bench/ppl_beagle_ch1.txt`):

| file | peak RSS (1,500 tok, `-c 4096`) | prefill, 8 / 12 threads | decode, 8 / 12 threads | 8 sequences batched, 8 / 12 | perplexity |
|---|---|---|---|---|---|
| Qwen3-0.6B Q8_0 | 0.93 GB | 1,813 / 2,287 tok/s | 207 / 165 | 814 / 910 | 27.60 (HF fp32 27.57) |
| Qwen3-0.6B Q4_K_M | 0.67 GB | 1,721 / 2,338 | 278 / 373 | 1,005 / 1,096 | 28.95 |
| Qwen3-1.7B Q8_0 | 2.1 GB | 790 / 1,013 | 78 / 78 | 381 / 458 | 18.90 (HF 18.84) |
| Qwen3-1.7B Q4_K_M | 1.4 GB | 680 / 943 | 101 / 112 | 466 / 529 | 19.09 |
| Qwen3-4B Q8_0 | 4.7 GB | 313 / 412 | 29 / 31 | 157 / 201 | 15.65 |
| Qwen3-4B Q4_K_M | 2.8 GB | 276 / 385 | 47 / 51 | 177 / 226 | 16.26 |
| Qwen3-4B-Instruct-2507 Q4_K_M | 2.8 GB | 284 / 401 | 56 / 56 | 173 / 237 | 11.86 |
| Qwen3-4B-Thinking-2507 Q4_K_M | 2.8 GB | 280 / 375 | 51 / 58 | 179 / 233 | 13.61 |
| Qwen3-8B Q4_K_M | 4.9 GB | 158 / 219 | 30 / 30 | 91 / 119 | 11.53 |
| Qwen3.5-0.8B Q8_0 (\*) | 0.96 GB | 1,506 / 2,039 | 141 / 182 | | 18.86 (512-token windows: 22.729, HF 22.728) |
| Qwen3.5-4B Q8_0 (\*\*) | 4.8 GB | 308 / 443 | 31 / 30 | | 10.59 (512-token windows: 12.645) |
| Qwen3.8-27B Q4_K_M (\*\*\*) | 15.9 GB | 45 / 65 | 8.5 / 10.0 | | 9.800 (512-token windows) |
| Qwen3.8-27B Q3_K_M (\*\*\*\*) | 13.0 GB | 42 / 68 (16 threads) | 9.6 / 9.8 (16 threads) | | 10.069 (512-token windows) |
| Underdog Saluki 27B IQ2-mix (\*\*\*\*\*) | 8.45 GB | 30 / 54 (16 threads) | 9.9 / 13.3 (16 threads) | | 11.841 (512-token windows) |

(\*) Another session: prefill 2026-10-27 (`scripts/bench_models.py`, 451 tokens; Qwen3-0.6B Q8_0 alongside: 1,832 /
2,511), decode 2026-10-26 (0.6B alongside: 209 / 230). Its decode barely slows down with the context (160 / 174 tok/s
after 1,500 tokens): only 6 of its 24 layers have a KV cache, at 12 KB per token. Prefill is ~0.8x the 0.6B's (it was
~0.64x before the linear attention's recurrence was blocked and tuned, docs/qwen35.md, Performance).
(\*\*) 2026-10-27, `scripts/bench_models.py -r 3`, load 13-15 (with the run's own threads); Qwen3-4B Q8_0 alongside:
314 / 440 and 31 / 31, the same within noise (after 1,500 tokens 26 / 32 vs 32 / 32).
(\*\*\*) 2026-10-27, `scripts/bench_models.py`, best of 2, load 9-14 (with the run's own threads); prefill of 488 tokens
(Qwen3.8's template adds its reasoning instruction); 16 threads: 76 / 10.3; after 1,500 tokens 8.7 / 10.6; startup
20 s at 8 threads (the 17 GB file read cold: its page cache is dropped after each load).
(\*\*\*\*) 2026-10-28, `scripts/bench_models.py -t 8,16`, best of 2, load 10-19, the Q4_K_M alongside: 45 / 73 tok/s of
prefill, 7.9 / 9.1 of decode at 8 / 16 threads; perplexity of both with `scripts/ppl.py -w 512`, docs/qwen38-speculative.md, G3.
(\*\*\*\*\*) 2026-10-30, `scripts/bench_models.py -t 8,16 -r 2`, the Q3_K_M alongside (42 / 74 prefill, 9.8 / 12.3 decode
at 8 / 16 threads; 13.3 GB); decode after 1,500 tokens 9.8 / 14.3; startup 2.3 s at 16 threads; docs/iquants.md.

Decode is memory-bound and moves with the machine's load (8B ranged 12-30 tok/s); startup takes 0.1-0.2 s
up to 1.7B, ~0.5 s for 4B and ~1 s for 8B. Q4_K_M is the better trade at 4B / 8B (decode ~1.6x Q8_0 at
~60% of the RAM). Q8_0 is the more accurate one: at 0.6B, Q4_K_M has ~25x the logits error and answers
"Sydney" for the capital of Australia.

**Chat templates** are detected from the file's `tokenizer.chat_template`, which is not interpreted
(Qwen3's three variants, Qwen3.5's and Qwen3.8's are built in). The hybrid Qwen3 template (0.6B to 8B) follows `-r 0` / `-r 1`.
Instruct-2507 never thinks, and Thinking-2507 always does; `-r` is ignored for those, with a note. Qwen3.5 follows
`-r` too, off by default as in its template. Qwen3.8 (a qwen35 file whose template has `reasoning_effort`) thinks by
default (`-r 0` turns it off); with thinking on, the system block starts with the reasoning-effort instruction of
`-e` (xhigh by default, medium has none, low a short one), merged with `-y`; earlier replies keep their reasoning,
as its template renders them by default. The renderings equal HF's `apply_chat_template` (tested; the documented
differences: [tests/README.md](tests/README.md)).

**Qwen3.5** keeps a fixed-size recurrent state per sequence (19 MB for 0.8B) besides its small KV cache. That state
cannot be rewound, so a chat that edits its history (a thinking reply's reasoning, dropped from the next turn) goes
back to a checkpoint the CLI and the server keep at each prompt's assistant header and prefills the rest again, with
the same results as a fresh run.

**qwen3.c's `.bin`** (Q8_0, group 64) works too, exported from the HF weights with qwen3.c's exporter.
The tests use it, and it is the format for comparing with upstream: `scripts/download_models.py hf &&
scripts/setup_venv.sh && scripts/export_q8.sh` (a `.venv` with CPU torch) gives `models/Qwen3-0.6B-q8.bin`.
`make models` does the same.

Not supported: other GGUF types (Q4_0, Q4_1, IQ4_NL, TQ1_0 / TQ2_0, F16 / BF16 matrices, and the files that use them:
rejected with a clean error; Unsloth's "UD" files made of Q2_K / Q3_K .. Q6_K / the i-quants / Q8_0 load), Qwen3-14B and larger (untested),
the MoE models (`qwen3moe`, `qwen35moe`: Qwen3.5-35B-A3B and up, Qwen3.6-35B-A3B), images, Qwen3-Embedding / Reranker,
RoPE scaling (YaRN), and Hadamard-folded GGUFs (`prism.hadamard.*` metadata: rejected). The other dense Qwen3.5 / 3.6
sizes should load but are untested: 2B (as many value heads as key heads, like 0.8B), 9B, 27B and Qwen3.6-27B (2x / 3x
as many value heads, like 4B; the 27B shape is covered by synthetic test models). **Qwen3.8-27B** (Unsloth's Q4_K_M,
docs/qwen38-speculative.md) loads its main model: the MTP block after its 64 layers (`nextn_predict_layers` 1,
`blk.64`: an attention layer + `nextn.*`) is checked (names, shapes, types) but not read nor used yet, and its F32
`ssm_alpha` / `ssm_beta` matrices are applied in fp32; tested on the real file (logits vs its numpy reference cos
0.9999, top-1 62/62; the bitwise invariants; the models group's `3.8-27b-q4km`, opt-in; its Q3_K_M too, `3.8-27b-q3km`:
cos 0.99983, top-1 62/62; a fact ~84k tokens back retrieved from an 88k-token prompt). Other MTP layouts (several
blocks, `qwen3` files) are refused with a clear message.

## Performance

Full tables, methodology and raw logs: [BENCHMARKS.md](BENCHMARKS.md) and [bench/](bench/). Two machines,
not comparable with each other:

- **AMD EPYC Genoa (Zen 4), 16 vCPUs** (KVM VM, 30 GB RAM, shared with other workloads: run-to-run
  noise ±15-30%, tables report the best runs):
  - Against upstream qwen3.c, Qwen3-0.6B Q8_0 `.bin`, one session (2026-10-25), 8 / 12 threads:
    prefill of 451 tokens 2,058 / 2,758 tok/s, where qwen3.c prefills token by token at ~99 tok/s
    (short prompt); decode 148 / 170 tok/s against qwen3.c's 101 / 106.
  - Later steps ([docs/DEVELOPMENT.md](docs/DEVELOPMENT.md#benchmark-notes-per-step)): decode of the Q8_0
    GGUF ~190 / ~210 tok/s at 8 / 12 threads, of Q4_K_M ~280 / ~350. After a 3,000-token prompt, decode is ~130 / ~150 (Q8_0 GGUF) and ~160 / ~180
    (Q4_K_M) with the f16 KV cache.
  - Batched decode (0.6B Q4_K_M, short contexts, 8 / 12 threads): 8 sequences 1,003 / 1,214 tok/s in
    aggregate, 16 sequences 1,091 / 1,449 (single sequence: 326 / 366).
  - `runq serve`, 0.6B, 12 threads, 16 concurrent clients streaming 128-token replies: 1,185 (Q4_K_M) /
    1,223 (Q8_0) tok/s in aggregate, time to first token ~0.1 s.
- **Apple M1 Pro** (8 performance + 2 efficiency cores, 32 GB, macOS), 0.6B, 8 threads: prefill of 451
  tokens ~1,700 tok/s with the Q8_0 `.bin` and ~1,300 with Q4_K_M; decode ~140-155 and ~190-230 tok/s.
  Sessions vary by ~15% on that desktop.

Quality (0.6B, perplexity relative to HF transformers fp32): Q8_0 GGUF +0.19%, Q8_0 `.bin` +0.85%,
Q4_K_M +4.9% (llama.cpp-based mynah-slm on the same Q4_K_M file: +4.7%).

**Long context: `-k bf16-dot`** (opt-in; `runq serve -k bf16-dot` too). The attention's two dot products as the
`vdpbf16ps` instruction computes them, on q, K, the softmax weights and V rounded to bf16: twice the dot-product rate
on CPUs with AVX512_BF16 (AMD Zen 4, Intel Sapphire Rapids; elsewhere the same results at the fp32 rate; the
instruction's exact rounding was verified on Zen 4, `make bf16dp-test` checks another CPU). On the EPYC VM, a
2,560-token chunk prefilled at 98k positions, 8 threads: its attention ~1.72x faster (the 27B-shaped synthetic model
8.99 -> 5.22 s, Qwen3.5-0.8B 18.3 -> 10.7 s), verify passes of speculative decoding ~20-30% cheaper, decode ~10-15%;
for Qwen3.8-27B that is ~200 -> ~140 s per such chunk (estimated,
[docs/qwen38-speculative.md](docs/qwen38-speculative.md#long-context-costs-todo-g0-b-measured-2026-10-27), G4 (b)).
Bit-exactness is kept: the same results for any batch, chunk size, thread count and build. The price is accuracy: KL
divergence from HF fp32 (28-45k tokens of prose and code) +0.3% on Qwen3.5-0.8B (within the noise of the default
cache: f16 -0.7%), but +5.4% on Qwen3-0.6B and +2.9% on Qwen3-1.7B (about a bf16 cache's +3.9% / +2.8%; f16 +2.1% /
+0.5%): Qwen3 has attention in every layer, Qwen3.5 / 3.8 in one of four. `f16-dot` keeps the f16 cache and rounds it
again when read (~1.5x: slower, and no more accurate than `bf16-dot`).

## Server

`runq serve MODEL` serves one model over HTTP/1.1. By default it binds 127.0.0.1:8080 with no
authentication (`--api-key KEY` requires a bearer token). It has no TLS, so keep it on localhost or a
trusted network.

- Endpoints: `POST /v1/chat/completions`, `GET /v1/models`, `GET /health`. Supported: system / user /
  assistant messages, `max_tokens`, `temperature`, `top_p`, `seed`, `stop`, `stream` (SSE),
  `logprobs`, thinking (`reasoning_effort` or `chat_template_kwargs.enable_thinking`, returned as
  `reasoning_content`; Qwen3.8: on by default, `reasoning_effort` picks its effort instruction, docs/server.md), `response_format` `json_schema`, and `usage` with cached prompt tokens. Errors
  come back as OpenAI error objects.
- **Tool calls** (Qwen3.5 and Qwen3.8): `tools` (`tool_choice` auto / none), `tool` messages and earlier
  `tool_calls` rendered exactly as the model's chat template does (tested against HF's
  `apply_chat_template`); the model's XML calls come back as OpenAI `tool_calls` with JSON arguments typed by
  each tool's schema (`finish_reason: tool_calls`; streamed as `tool_calls` deltas). pi runs its agent loop
  (read / bash) against it (docs/server.md, Tool calls).
- **Continuous batching**: every running request (up to `--parallel` 16) contributes its next token or a
  prompt chunk to one batched forward pass per step. Streams keep flowing while long prompts prefill,
  and each request's output is bit-identical to running it alone.
- **Speculative decoding** (`--draft FILE`, `--lookup N`): the guesses of a draft model of the same vocabulary, or
  copied from the context, verified in the same batched passes; the draft's steps in its own batched passes. Every
  output is plain decoding's (seeded, streamed, with logprobs, tool calls and stop strings; tested request by request).
  By default (`--spec auto`) the source and the guess count are chosen per pass from a cost model calibrated at the
  start, the position and the batch: the draft model at short context, the lookup at long context, less as the batch
  fills, plain steps when no guess pays off; per request `speculative`, counters in `timings`
  ([docs/server.md](docs/server.md#speculative-decoding-todo-s8-g7)).
- **Prefix reuse**: a new request takes the idle KV slot with the longest common prefix and prefills
  only the rest, or copies the prefix from a running request. A second chat turn, or a fixed system
  prompt, is not prefilled again.
- **Constrained decoding** for `json_schema` with a finite set of outputs (`enum`, `const`, booleans,
  bounded integers, objects of those, `anyOf` / `oneOf`, `$ref`). Forced tokens are not decoded at all,
  so a classification is a single prefill.

As the judge of LiteLLM's complexity router (`auto_router/complexity_router`, `classifier_type: llm`,
tested with LiteLLM 1.105), runq answers `{"tier": "..."}` in ~15-27 ms with the rubric cached (0.6B,
8 threads). Qwen3-1.7B Q8_0 is the smallest good judge: 88% on the project's 80-prompt eval set at
~50 ms per classification.

```yaml
model_list:
  - model_name: smart-router
    litellm_params:
      model: auto_router/complexity_router
      complexity_router_config:
        classifier_type: llm
        classifier_context_window_size: 0     # judge the current message only (shorter prompts)
        classifier_llm_config:
          model: runq-judge
          timeout_ms: 10000                    # default 3000; a timeout opens a 30 s circuit breaker
        tiers: { SIMPLE: small-model, MEDIUM: small-model, COMPLEX: big-model, REASONING: big-model }
  - model_name: runq-judge
    litellm_params:
      model: openai/runq                       # any name
      api_base: http://127.0.0.1:8080/v1
      api_key: none
      temperature: 0
```

All flags, the endpoint details, behaviour under load and the measurements are in
[docs/server.md](docs/server.md). `scripts/litellm_e2e.sh` runs the whole chain against a scratch
LiteLLM proxy, and `scripts/eval_router.py` measures judge accuracy.

**Agentic development with Qwen3.8-27B (pi)** at a 96k context on the 16-vCPU / 31 GB VM (~24 GiB free):

```sh
OMP_NUM_THREADS=16 runq serve models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q3_K_M.gguf -c 98304 -k bf16-dot \
    --draft models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf --slots 1 --parallel 1 --checkpoints 3 \
    --spec-calibrate 32768 --timeout 0 --name qwen3.8-27b --port 18080
```

with pi's `openai-completions` provider (`contextWindow` 98304, `maxTokens` 16384, `reasoning` true; compaction
`keepRecentTokens` 8000; the entry and settings: [docs/server.md](docs/server.md#qwen38-27b-with-pi)). ~22.1 GiB with the
draft (the Q4_K_M would need ~25). Measured (2026-10-28): a live pi session finished a small coding task (a function,
its tests, the test run) in 6 requests and ~80 s, decoding at 15-23 tok/s; a recorded 54-reply session replayed to 84k:
every tool turn prefills only its new tokens (2,600 tokens at 72k: 70 s), decode 8-14 tok/s at 21-81k with
speculation, a compaction ~12 min; decode at 98k 6.9 tok/s plain, a 2,560-token chunk there 86 s.

## Library API

The engine is a C library, `src/runq.h`; the CLI (`src/cli.c`) and the server use only this API. It
has three kinds of objects:

- **RqModel** (`rq_model_load`): hyperparameters, weights (repacked for the kernels at load), tokenizer,
  and chat templates. It is read-only after loading and shared by any number of sequences.
- **RqSeq** (`rq_seq_new`): one sequence with its own KV cache (capacity and type per sequence), tokens
  and sampling RNG. `rq_seq_truncate`, `rq_seq_common_prefix` and `rq_seq_copy` handle prefix reuse.
- **RqCtx** (`rq_ctx_new`): a compute context (activation buffers for up to `max_batch` rows, logits).
  The threads are one process-wide pool (`rq_threads_init`) running one pass at a time: the passes of
  several contexts (e.g. two models with the same activation group size) alternate on one calling thread.

The forward passes are `rq_decode` (one token), `rq_prefill` (a sequence's tokens in chunks) and
`rq_forward_batch` (one pass over rows of several sequences, each at its own position: prefill chunks
next to decode rows; short segments, e.g. decode rows or a speculative verify pass of a few rows of one sequence,
take the decode-rows worker: 4 rows of one sequence cost ~1.2-1.3x a decode step, docs/qwen38-speculative.md). On top of them sit `rq_sample` / `rq_sample_batch` (temperature, top-p, an optional
token mask), `rq_generate` (the chat loop with per-token callbacks; built on `rq_gen_start` / `rq_gen_step`, one
token at a time, for loops that feed the tokens themselves), `rq_generate_spec` (the same with a draft model:
speculative decoding, the output exactly `rq_generate`'s; `rq_spec_run` is its loop alone), `rq_tokenize`, `rq_detok_push`
(incremental, UTF-8 safe) and `rq_chat_render`. Errors come back as `RQ_ERR_*` codes and messages; the
library never prints or exits. `RqModelOpts.mem` / `.page_cache` set the load's memory check and page-cache policy
(Memory, above); `rq_model_mem` and `rq_seq_mem` return the estimates (the server sizes its slot pool with the latter).

```c
rq_threads_init(0);
RqModel *m = rq_model_load("Qwen3-0.6B-Q8_0.gguf", NULL, err, sizeof(err));
RqCtx *c = rq_ctx_new(m, 512, 0, &e);
RqSeq *s = rq_seq_new(m, 4096, RQ_KV_DEFAULT, &e);
RqMessage msg[] = { { RQ_ROLE_USER, "Why is the sky blue?" } };
RqTokens t = { 0 };
rq_chat_render(m, msg, 1, 0, &t);
RqSampling cfg = { 0.7f, 0.9f, NULL };
RqCallbacks cb = { on_token, NULL, user };   // on_token: e.g. rq_detok_push + write
rq_generate(c, s, t.v, t.n, &cfg, 256, &cb, &stats);

// batched decode: one row per sequence, every weight read shared by the N rows
RqSegment sg[N];
for (int i = 0; i < N; i++) sg[i] = (RqSegment) { seq[i], &tok[i], 1, RQ_LOGITS_LAST };
rq_forward_batch(c, sg, N, logits, NULL);         // logits: N x vocab_size floats
rq_sample_batch(c, seq, logits, N, cfg, N, tok);  // the next token of each, per-row settings
```

| file | contents |
|---|---|
| `src/runq.h` | the public API |
| `src/model.c` | RqModel: .bin / GGUF loading, weight conversion and repacking, chat templates |
| `src/gguf.c` | the GGUF reader (untrusted input: every count, length and offset checked) |
| `src/forward.c` | RqSeq, RqCtx and all numeric code of the forward passes: decode, batched pass, attention, matmul dispatch, quantization, norms, RoPE, KV cache |
| `src/tokenizer.c` | BPE tokenizer matching HF (added tokens, NFC, Qwen2 pre-tokenizer), detokenization |
| `src/sampler.c`, `src/chat.c` | sampling; chat template rendering and the generation loop |
| `src/spec.c` | speculative decoding: the draft's (or the prompt lookup's) guesses verified in one pass of the model (`rq_generate_spec`) |
| `src/threads.c`, `src/threads.h` | the thread pool and its phase synchronization |
| `src/kernels_avx512.h`, `src/kernels_neon.h`, `src/kquants.h` | SIMD kernels; the K-quant formats and the exact arithmetic every kernel follows |
| `src/iquants.h`, `src/iq_grids.h` | Q2_K and the i-quants (IQ1_S .. IQ4_XS): formats, repacked layouts, arithmetic, generic and AVX-512 kernels; the codebooks (generated by `tools/gen_iq_grids.py`) |
| `src/unicode_tables.h`, `src/unicode_nfc.h` | Unicode classes and NFC tables, generated by `tools/gen_unicode_tables.py` |
| `src/cli.c` | the `runq` command line and its test hooks |
| `src/serve.c`, `src/engine.c`, `src/http.c`, `src/constrain.c`, `src/tools.c`, `src/json.c` | `runq serve` (not part of the library): OpenAI layer, scheduler and slot pool, HTTP/1.1, constrained decoding, tool calls, JSON |

Every kernel family follows one arithmetic definition: int8 activations in groups of 32 (GGUF) or 64
(`.bin`), with the float folds in a fixed order and without FMA contraction. The kernels are
interchangeable bit for bit, but logits differ slightly from fp32 engines; the tests bound the
differences against HF. The activation group size is a process-wide kernel setting: load one model,
or models with the same group size, per process. The changes relative to qwen3.c, step by step, the design notes
and the experiments that were not kept are in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md); the full development log
with every measurement is in the git history ([History](docs/DEVELOPMENT.md#history)). Open work:
[TODO.md](TODO.md), [IDEAS.md](IDEAS.md).

## Testing

```sh
scripts/download_models.py                # default sets: hf (HF bf16), q4km, q8gguf
scripts/setup_venv.sh && scripts/export_q8.sh   # .venv (CPU torch) and models/Qwen3-0.6B-q8.bin
make test                                 # the regression suite
make test-models                          # optional: every downloaded bigger model (minutes)
```

`make test` runs 481 checks in ~28-36 s on the 16-vCPU VM, 644 with the Qwen3.5-0.8B Q8_0 file
(`scripts/download_models.py 3.5-0.8b-q8`; also the source of two synthetic Qwen3.5 models with more value heads
than key heads), 686 with its Q4_K_M file too (`3.5-0.8b-q4km`; ~70-80 s), 27 more with its Q3_K_M file (`3.5-0.8b-q3km`). It needs numpy (from `.venv` or the system `python3`) and the 0.6B
`.bin`; the GGUF groups are skipped if their files are missing. It covers:
tokenizer ids against HF; logits against HF fp32 (cosine, top-1 / top-5); batched against sequential,
thread-count and generic-against-SIMD bit-identity; Q4_K_M, Q5_K, Q3_K, Q2_K and i-quant dequantization against gguf-py; the KV cache
types; greedy-output goldens; chat-template rendering against HF's `apply_chat_template`; the library
API; speculative decoding against plain decoding (tokens and logits bits; in the server, request by request); the server over HTTP; synthetic GGUF models with the bigger models' shapes; and Qwen3.5-0.8B (its tokenizer,
logits against HF fp32, the same invariants, chats that roll back to a state checkpoint) and synthetic Qwen3.5 models
with the 4B / 27B head layout (one of them in all nine i-quant / Q2_K types, as Underdog Saluki 27B). Golden references
come from HF transformers and are committed under `tests/golden/`. See [tests/README.md](tests/README.md).

Benchmarks: `make bench` (decode, thread sweep), `make bench-prefill`, `scripts/bench_all.sh` (the runq
tables of BENCHMARKS.md, ~8 min), `scripts/bench_batch.sh`, `scripts/bench_serve.py`,
`scripts/bench_models.py`, `scripts/ppl.py` (perplexity), `scripts/spec_accept.py` (draft acceptance for
speculative decoding), `scripts/lookup_accept.py` (prompt-lookup and draft acceptance on recorded agentic sessions),
`scripts/bench_lookup.py` (tok/s of the lookup on code-editing prompts; all three:
[docs/qwen38-speculative.md](docs/qwen38-speculative.md)). `scripts/fetch_mynah.sh` fetches
and builds mynah-slm, an optional reference engine for comparison (not part of this repository).

## Limitations

- Qwen3 dense models and dense Qwen3.5 (0.8B and 4B tested; `general.architecture` qwen3 / qwen35, gpt2 tokenizer with the qwen2 /
  qwen35 pre-tokenizer). GGUF Q8_0 / Q3_K_M / Q4_K_M / Q5_K_M and the i-quant mixes (any mix of Q8_0, Q2_K .. Q6_K, IQ1_S, IQ1_M,
  IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ4_XS tensors; F32 only for Qwen3.5's `ssm_alpha` / `ssm_beta`, as in
  Qwen3.8-27B) and qwen3.c's `.bin`. The i-quants have AVX-512 kernels (fastest with AVX512-VBMI), NEON ones and the
  generic ones; their prefill is ~0.72x the K-quants' on AVX-512 (docs/iquants.md). No other quantization types, no MoE, no RoPE scaling,
  no images, no Jinja interpreter (the templates are built in: Qwen3's, Qwen3.5's, Qwen3.8's). Qwen3.8-27B: its MTP block
  is not used (not a draft yet: TODO S9); it needs ~17 GB of free RAM (18.65 GiB estimated at the default `-c` 40,960).
- Fast paths: x86-64 with AVX-512 VNNI (selected at compile time with `-march=native`) and AArch64 with
  the dot-product extension. Other CPUs, including x86 without AVX-512 VNNI, use the generic C kernels:
  the same results, but for the K-quant files ~10x slower prefill and ~3-4x slower decode.
- The NEON port was tested on macOS only. On arm64 the tests that compare generated text against the
  goldens recorded on x86 fail (8 of them), because the compilers round some Q8_0 sums differently.
  Everything else passes there, but the goldens need an arm64 baseline, and the Qwen3 1.7B-8B fixes have
  not been re-run on the Mac yet (docs/DEVELOPMENT.md).
- `runq serve`: no TLS, no images, `n` = 1, no `json_object` or free-form string constraints; tool calls only
  with the Qwen3.5 / Qwen3.8 templates (not Qwen3's), `tool_choice` auto / none only; the KV memory is a fixed
  pool of slots (no paging).
- CPU only; the CLI and the server run one model per process, and its draft model for speculative decoding (the library
  can hold several with the same activation group size, tested by `api_test --two`).

## Credits and license

Gwenedeg is MIT-licensed ([LICENSE](LICENSE)). Third-party code, data and formats (qwen3.c, ggml / llama.cpp formats, Unicode data, Qwen chat templates and test data, LiteLLM's rubric) are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

- The engine started from [qwen3.c](https://github.com/adriancable/qwen3.c) by Adrian Cable (MIT), itself
  based on Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) (MIT). An unmodified copy of
  qwen3.c is kept in [third_party/qwen3.c](third_party/qwen3.c) for diffs and for its `.bin` exporter.
- The GGUF format and the Q8_0 / Q4_K / Q6_K block formats and dequantization expressions follow
  [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT); the implementation here is our own.
- The Unicode tables are generated from the Unicode Character Database (Unicode License v3). The
  tokenizer and NFC behaviour match Hugging Face [tokenizers](https://github.com/huggingface/tokenizers),
  which is used only to generate and check test data.
- The Qwen3 models are by the Qwen team (Alibaba Cloud), Apache-2.0. **No model weights are included**;
  `scripts/download_models.py` fetches them from Hugging Face. The tests contain Qwen's chat templates
  and reference outputs computed with the models.
- The test suite contains LiteLLM's classifier rubric (MIT, Berri AI). The perplexity text is from Charles
  Darwin's *The Voyage of the Beagle* (public domain, via Project Gutenberg).
- [mynah-slm](https://github.com/mynah-org/mynah-slm) is only fetched by a script for benchmark
  comparisons; it is not included.

The notices and license texts of these components are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
