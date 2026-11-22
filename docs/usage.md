# Using gwenedeg: command line, memory, speculative decoding, KV cache types

For users of the `gwenedeg` binary. The server has its own reference, [server.md](server.md); the models and their
files are in [models.md](models.md); the C library is in [library.md](library.md). The tables here come from the
README as it was before the documentation was reorganised (2026-11-21); every measurement keeps its date and setup.

## Quick start and download

Requirements: `make` and gcc (Linux) or Xcode's command-line tools (macOS). Python 3 (standard
library only) runs the model download script.

```sh
make                                      # -> ./gwenedeg   (make generic: ./gwenedeg-generic, make debug: ./gwenedeg-debug)
scripts/download_models.py q8gguf         # models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf (639 MB), pinned + checksummed
G=models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf

./gwenedeg $G -c 4096                         # interactive chat: one message per line, empty line / EOF ends
./gwenedeg $G -c 4096 -t 0 -n 200 -i "Why is the sky blue?"     # one turn, greedy, at most 200 tokens
./gwenedeg $G -c 4096 -r 1                    # with thinking (Qwen3's reasoning mode)
./gwenedeg serve $G                           # OpenAI-compatible server on http://127.0.0.1:8080/v1

curl -s localhost:8080/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Hi!"}], "max_tokens": 32}'
```

`scripts/download_models.py --list` shows every downloadable set ([models.md](models.md)). Files go to
`models/` (or `--dest DIR`, `$GWENEDEG_MODELS`). Each file is checked by sha256 or git blob hash at a
pinned Hugging Face revision.

## Command line options

CLI options (`./gwenedeg` without arguments prints them all):

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
| `-k f16\|bf16\|f32\|bf16-dot\|f16-dot\|q8\|q4\|q4k4` (+ `-dot`) | KV cache type, default f16 (half the memory of f32, as accurate); `bf16-dot`: a bf16 cache with the bf16 dot-product attention, ~1.7x faster attention at long context, opt-in (below); `q8` / `q4` / `q4k4`: 8-bit, 8-bit K + 4-bit V, 4-bit K + V (0.53 / 0.41 / 0.28 of f16's memory; `q4k4` for Qwen3.5 / 3.8 only; below) |
| `-b N` | prefill chunk size, default 512 (`-b 1`: token by token) |
| `--mem GiB` | memory budget of the load check (below), default MemAvailable; `0` or `force`: no check. Given, prints the estimate (a `[memory]` line) |
| `--page-cache auto\|drop\|keep` | the model file's page cache after the load; `auto` (default) drops it if the file is >= 1/4 of the RAM |
| `--repack[=LIST]`, `--no-repack` | store the IQ3_S / IQ3_XXS matmul weights as 4-bit codes at load (LIST: `IQ3_S`, `IQ3_XXS`; default all): lossless, bit-identical output, less decode work per weight for +0.75 / +1.0 bit per such weight of RAM (ISTA's IQ3_S 27B: +1.37 GiB). Off by default; AVX-512 builds ([docs/iquants.md](iquants.md#--repack-4-bit-codes)) |
| `--repack-info` | what `--repack` costs and gains on this machine with the current threads, per type: loads and times the model with each choice in turn (a decode step, a 4-row pass), then exits |
| `-D FILE` | speculative decoding (generate and chat): a smaller draft model of the same vocabulary (e.g. Qwen3-0.6B for Qwen3-1.7B, Qwen3.5-0.8B for Qwen3.5) guesses, the model verifies; the output is exactly plain decoding's (below) |
| `-D mtp`, `--mtp-vocab N` | the model file's own MTP block as the draft (multi-token prediction: Qwen3.8-27B's `blk.64`; a draft file named `mtp`: `-D ./mtp`); `--mtp-vocab`: its guesses among the tokens 0..N-1 and the added ones only (a part of the classifier: faster guesses), default 100000 if the vocabulary is larger, 0 = all (below) |
| `-G K`, `--tau T` | with `-D`: at most K guesses per pass of the model (default 3, at most 30; `-G 0`: plain decoding, the draft not loaded); adaptive length: stop guessing at the first guess whose draft top-1 probability is < T (default 0 = off, always K) |
| `--lookup N`, `--lookup-min M` | prompt lookup (generate and chat), with or without `-D`: guess up to N tokens (at most 30) by copying what followed the latest, longest match (>= M tokens, default 2; up to 16 counted) of the last tokens earlier in the context; no model, exact as any draft; with `-D` the draft model guesses where the lookup finds no match (below) |
| `OMP_NUM_THREADS` | threads; default nproc - nproc/8 (Apple Silicon: the performance cores) |

## Memory check at load

Before reading any weight, the loader estimates what the run needs from the file's tensor directory:
the repacked weights, the runtime (tokenizer, metadata, a compute context of 512 rows) and one sequence of `-c`
positions (its whole KV cache, and for Qwen3.5 the recurrent state with its 2 checkpoints; `gwenedeg serve`: its
`--checkpoints`, default 4). It refuses the file
with a one-line message and exit status 1 when that exceeds MemAvailable (Linux, the host's: a cgroup limit is not read; no check elsewhere
unless `--mem` gives a budget): a smaller `-c` lowers it, `--mem 0` loads anyway. The estimate is within 3% of the peak RSS
of a run that fills the context (Qwen3-0.6B to 8B, Qwen3.5-0.8B and 4B, [tests/README.md](../tests/README.md#memory)).
The weights are read with `pread`, so the file also passes through the page cache: for a big file (>= 1/4 of the
RAM, e.g. Qwen3.8-27B) the loader drops it matrix by matrix as they are repacked (`posix_fadvise` DONTNEED;
+0.35 s on the 8B's cold load, +0.6 s warm, when forced with `--page-cache drop`), for smaller ones it stays to make the next load fast
(~1 s warm vs ~6 s cold for the 8B here).

## Chat

Chat keeps the whole conversation in the KV cache, so each turn prefills only the new tokens. Earlier
answers drop their `<think>` block, as Qwen3's template does. When the context is full, the oldest turns
are dropped (the system prompt is kept) and the rest is re-prefilled. A `[timing]` line on stderr gives
prefill and decode speed. Details: [docs/DEVELOPMENT.md](DEVELOPMENT.md).

## Chat templates and thinking

The chat templates are detected from the file's `tokenizer.chat_template`, which is not interpreted
(Qwen3's three variants, Qwen3.5's and Qwen3.8's are built in). The hybrid Qwen3 template (0.6B to 8B) follows `-r 0` / `-r 1`.
Instruct-2507 never thinks, and Thinking-2507 always does; `-r` is ignored for those, with a note. Qwen3.5 follows
`-r` too, off by default as in its template. Qwen3.8 (a qwen35 file whose template has `reasoning_effort`) thinks by
default (`-r 0` turns it off); with thinking on, the system block starts with the reasoning-effort instruction of
`-e` (xhigh by default, medium has none, low a short one), merged with `-y`; earlier replies keep their reasoning,
as its template renders them by default. The renderings equal HF's `apply_chat_template` (tested; the documented
differences: [tests/README.md](../tests/README.md)).

## Speculative decoding and prompt lookup

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
thinking off 16.6 / 17.2 (2.0x; code and math 19-22); 12 threads: up to 15.3 / 20.6. In `gwenedeg serve` too ([server.md](server.md#speculative-decoding-todo-s8-g7)).
`--spec auto` (opt-in in the CLI) chooses the source and the guess count per pass from a cost model calibrated at the
start (TODO G7).
Measurements: [docs/notes/qwen38-speculative.md](notes/qwen38-speculative.md#the-27b-bring-up-s7).

**The MTP drafter** (`-D mtp`, TODO S9): a file with a multi-token-prediction block (Qwen3.8-27B's `blk.64`, e.g.
ISTA-DASLab's IQ3_S-mtp, Unsloth's Q4_K_M) guesses with it: one attention layer fed the model's own final hidden state
and the next token, through the model's classifier (by default `--mtp-vocab 100000`: a part of it, 99.9% of agentic
replies' tokens). It follows the model in lockstep (its rows over the prompt and after every pass), so no second model,
no second 64-layer state per sequence; the output is still plain decoding's. On the 27B with thinking on it is ahead of
the 0.8B draft: IQ3_S-mtp 12.8 vs 11.1 tok/s at each one's best fixed settings (`-D mtp -G 6 --tau 0.5` vs `-G 6 --tau
0.4`; 1.6x plain), Unsloth's Q4_K_M 17.1 vs 14.1 (2.0x plain); with thinking off the two are even (TODO S9d,
[docs/notes/qwen38-speculative.md](notes/qwen38-speculative.md#the-27bs-acceptance-and-the-go-todo-s9d2-2026-11-09)).
`gwenedeg serve --draft mtp` runs it too (TODO S9c.3: every request followed by the view, its guesses batched across
requests; [docs/server.md](server.md#speculative-decoding-todo-s8-g7)). Measured on the 27B in the server (S9c.G,
2026-11-10, ISTA's IQ3_S-mtp, 14 threads, `--parallel 4 --admit bound`, against the 0.8B draft at `-G 5 --tau 0.4`):
`--draft mtp -G 6 --tau 0.5` +14% tok/s at one client with thinking on, +18% with it off, +1% at 4 clients, outputs
identical ([docs/server.md](server.md#speculative-decoding-todo-s8-g7); [the gated run](notes/qwen38-speculative.md#in-the-server-s9c)).

**Prompt lookup** (`--lookup N`, no model needed): replies that copy from their context (code being edited, paths,
tool output) are guessed from the context itself, verified in the same loop (so the output is still plain decoding's).
On edits that return the prompt's code, `--lookup 8` decodes 2.8-3.9x faster (Qwen3-1.7B, Qwen3.5-0.8B; the 0.6B draft
at `-G 3`: 1.4-1.6x); with nothing to copy it is about plain speed. With `-D`, the draft guesses where the lookup
finds nothing: `-G 3 --lookup 8 --lookup-min 4` (a longer minimum match: the draft guesses short matches better).
On recorded agentic sessions it guesses ~1.5 tokens per pass alone (the 0.8B draft 2.4): [docs/notes/qwen38-speculative.md](notes/qwen38-speculative.md#prompt-lookup-todo-g6-g0-c-measured-2026-10-27).

## KV cache types

`-k TYPE` (the CLI and `gwenedeg serve`). Every type is bit-exact across batches, threads and builds; they differ in memory,
speed and distance from the fp32 model:

| `-k` | KV cache | memory per position | notes |
|---|---|---|---|
| `f16` (default) | 16-bit floats | half of `f32` | as accurate as `f32` (KL from HF fp32 +0.2%, within the noise; [DEVELOPMENT.md](DEVELOPMENT.md#chat-kv-cache-and-threads-details)) |
| `bf16` | bf16 | half of `f32` | a bit further from fp32 (KL +2.6% on Qwen3-0.6B); prefills long prompts ~3% faster |
| `f32` | fp32 | the reference size | |
| `bf16-dot`, `f16-dot` | bf16 / f16 | as `bf16` / `f16` | the attention's dot products as `vdpbf16ps` (below); opt-in |
| `q8`, `q4`, `q4k4` (each also as `-dot`) | Walsh-Hadamard rotated, 8-bit / 8-bit K + 4-bit V / 4-bit K + V, per 32 dims | 0.53 / 0.41 / 0.28 of `f16`'s | `q4k4`: Qwen3.5 / 3.8 only; [kv-quant.md](kv-quant.md) |

**Long context: `-k bf16-dot`** (opt-in; `gwenedeg serve -k bf16-dot` too). The attention's two dot products as the
`vdpbf16ps` instruction computes them, on q, K, the softmax weights and V rounded to bf16: twice the dot-product rate
on CPUs with AVX512_BF16 (AMD Zen 4, Intel Sapphire Rapids; elsewhere the same results at the fp32 rate; the
instruction's exact rounding was verified on Zen 4, `make bf16dp-test` checks another CPU). On the EPYC VM, a
2,560-token chunk prefilled at 98k positions, 8 threads: its attention ~1.72x faster (the 27B-shaped synthetic model
8.99 -> 5.22 s, Qwen3.5-0.8B 18.3 -> 10.7 s), verify passes of speculative decoding ~20-30% cheaper, decode ~10-15%;
for Qwen3.8-27B that is ~200 -> ~140 s per such chunk (estimated,
[docs/notes/qwen38-speculative.md](notes/qwen38-speculative.md#long-context-costs-todo-g0-b-measured-2026-10-27), G4 (b)).
Bit-exactness is kept: the same results for any batch, chunk size, thread count and build. The price is accuracy: KL
divergence from HF fp32 (28-45k tokens of prose and code) +0.3% on Qwen3.5-0.8B (within the noise of the default
cache: f16 -0.7%), but +5.4% on Qwen3-0.6B and +2.9% on Qwen3-1.7B (about a bf16 cache's +3.9% / +2.8%; f16 +2.1% /
+0.5%): Qwen3 has attention in every layer, Qwen3.5 / 3.8 in one of four. `f16-dot` keeps the f16 cache and rounds it
again when read (~1.5x: slower, and no more accurate than `bf16-dot`).

**Smaller caches: `-k q8`, `q4`, `q4k4`** (opt-in, each also as `-dot` with the bf16 dot-product attention): K and V
Walsh-Hadamard rotated and quantized per 32 dims of a position, 8-bit (`q8`), 8-bit K + 4-bit V (`q4`), 4-bit both
(`q4k4`), bit-exact across batches, threads and builds like the other types. Qwen3.8-27B at 98k: KV 6.0 GiB (f16) ->
3.19 / 2.44 / 1.69 GiB, so the Q4_K_M with its draft fits where f16 did not. KL from HF fp32: `q8` within the 16-bit
types' range on Qwen3.5-0.8B (+5%; Qwen3-0.6B +56%, below the Q8_0 weights' own), `q4` ~5-6x the baseline, `q4k4` ~7x
on Qwen3.5 but unusable on Qwen3 (4-bit K: top-1 50%). Decode at 64k on Saluki ~5-20% faster with the `-dot` types,
`q4k4-dot` the most (the weights dominate a step). Measurements and definition: [docs/kv-quant.md](kv-quant.md).

## Limitations

- Qwen3 dense models and dense Qwen3.5 (0.8B and 4B tested; `general.architecture` qwen3 / qwen35, gpt2 tokenizer with the qwen2 /
  qwen35 pre-tokenizer). GGUF Q8_0 / Q3_K_M / Q4_K_M / Q5_K_M and the i-quant mixes (any mix of Q8_0, Q2_K .. Q6_K, IQ1_S, IQ1_M,
  IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ4_XS tensors; F32 or BF16 (widened to F32 at load) only for Qwen3.5's `ssm_alpha` / `ssm_beta`, as in
  Qwen3.8-27B) and qwen3.c's `.bin`. The i-quants have AVX-512 kernels (fastest with AVX512-VBMI), NEON ones and the
  generic ones; their prefill is ~0.72x the K-quants' on AVX-512 (docs/iquants.md). No other quantization types, no MoE, no RoPE scaling,
  no images, no Jinja interpreter (the templates are built in: Qwen3's, Qwen3.5's, Qwen3.8's). Qwen3.8-27B: it needs ~17 GB of free RAM
  (18.65 GiB estimated at the default `-c` 40,960 for the Q4_K_M file); its MTP block is the draft of `-D mtp` / `--draft mtp`.
- Fast paths: x86-64 with AVX-512 VNNI (selected at compile time with `-march=native`) and AArch64 with
  the dot-product extension. Other CPUs, including x86 without AVX-512 VNNI, use the generic C kernels:
  the same results, but for the K-quant files ~10x slower prefill and ~3-4x slower decode.
- The NEON port was tested on macOS only. On arm64 the tests that compare generated text against the
  goldens recorded on x86 fail (8 of them), because the compilers round some Q8_0 sums differently.
  Everything else passes there, but the goldens need an arm64 baseline, and the Qwen3 1.7B-8B fixes have
  not been re-run on the Mac yet (docs/DEVELOPMENT.md).
- `gwenedeg serve`: no TLS, no images, `n` = 1, no `json_object` or free-form string constraints; tool calls only
  with the Qwen3.5 / Qwen3.8 templates (not Qwen3's), `tool_choice` auto / none only; the KV memory is a fixed
  pool of slots (no paging).
- CPU only; the CLI and the server run one model per process, and its draft model for speculative decoding (the library
  can hold several with the same activation group size, tested by `api_test --two`).
