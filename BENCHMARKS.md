# Gwenedeg benchmarks

Qwen3-0.6B on the CPU: Gwenedeg (Q8_0, AVX-512 VNNI, batched prefill) vs upstream qwen3.c and
mynah-slm, plus perplexity. All numbers in the Results, Comparison and Quality sections are from
**one session, 2026-10-25 04:08-05:05 UTC, commit `d76e4cc`**, same machine and conditions. Raw
output of every run: [bench/results-2026-10-25.txt](bench/results-2026-10-25.txt). Earlier
measurements are only in [History](#history-earlier-measurements) and are labelled there.
(The project was called cwayn when these sessions ran; the raw logs in `bench/` have the new name and
generic paths, nothing measured was changed.) The binary was called `runq` until 2026-11-22 (after qwen3.c's
`runq.c`, where it started); older raw data and logs use that name, and the commands below use `gwenedeg`.

The [Qwen3.8-27B: Gwenedeg vs llama.cpp vs ik_llama.cpp section](#qwen38-27b-iq3_s-gwenedeg-vs-llamacpp-vs-ik_llamacpp-server-workloads-2026-11-22)
is a different model, comparison and session (2026-11-22) and is not comparable with the tables above either. The
[Qwen3.8-27B vs ik_llama.cpp section](#qwen38-27b-iq3_s-vs-ik_llamacpp-server-workloads-2026-11-21) is the same model
a day earlier, before the i-quant tiled GEMM: its gwenedeg numbers are superseded by the 2026-11-22 ones.

The [i-quant tiled GEMM section](#qwen38-27b-iq3_s-prefill-with-the-i-quant-tiled-gemm-2026-11-21) is a
before / after of one change on that model (2026-11-21, a later window), not comparable with the 0.6B tables either;
its "not re-run" estimate is now measured in the 2026-11-22 section.

The [all-models table](#all-model-files-on-the-epyc-vm-2026-10-26-to-2026-10-30) and the [batched-decode
table](#batched-decode-aggregate-toks-step-10a-2026-10-26) are from other days on the same VM (dated in each) and are not
comparable with the tables above either.

The [Apple M1 Pro section](#apple-m1-pro-macos-neon) is a different machine and different sessions
(2026-10-25, 14:56-22:30 UTC, branch `mac-m1-pro`) and is not comparable with the tables above
(other CPU; Q4_K_M as well as Q8_0).

## Setup

| | |
|---|---|
| CPU | AMD EPYC Genoa (Zen 4), 16 vCPUs (KVM VM, 1 thread per core, 32 MB L3), 30 GB RAM |
| sharing | the VM runs other users' services; 1-min load 2-6 before each block, up to ~19 during some (see Methodology) |
| memory read bandwidth, this session | 1 / 4 / 8 / 12 threads: 33-35 / 49-71 / 60-73 / 86-115 GB/s (min-max of 3-4 runs at the start and end; `bench/bw.c`) |
| compiler | gcc 14.2.0 (Debian 14.2.0-19), `-Ofast -fopenmp -march=native -D_FILE_OFFSET_BITS=64 -lm` (`make`; since 2026-11-23 `make` has no `-fopenmp`: only `omp_set_num_threads` in `threads.c` used it, and every object file but that one is byte-identical, so these numbers stand) |
| gwenedeg | `d76e4cc` (`src/runq.c` + `src/kernels_avx512.h`); defaults: `-b 512`, threads min(12, nproc - nproc/4) = 12 |
| qwen3.c | upstream engine = `src/runq.c` at `87e08aa` (upstream `55643ff` + only the `-n` / `[timing]` patch), same flags |
| mynah-slm | `cde1599` (`scripts/fetch_mynah.sh`, its own Makefile) |
| HF reference | transformers 4.53.0 / torch 2.7.0, fp32, eager attention (`.venv`) |

| model file | bytes | used by |
|---|---|---|
| `models/Qwen3-0.6B-q8.bin` (Q8_0, group 64, tied classifier; `scripts/export_q8.sh`) | 633,495,808 | gwenedeg, qwen3.c |
| `models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf` (unsloth @ `50968a4`, group 32) | 639,447,744 | mynah-slm |
| `models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf` (unsloth @ `50968a4`, imatrix) | 396,705,472 | mynah-slm |
| `models/Qwen3-0.6B/model.safetensors` (bf16, `Qwen/Qwen3-0.6B` @ `c1899de`) | 1,503,300,328 | HF reference (run in fp32) |

## Results: gwenedeg

**Prefill tok/s** vs prompt length and threads (`-b 512`, chat, greedy):

| prompt tokens | 1 thread | 4 | 8 | 12 |
|---|---|---|---|---|
| 451 | 308 | 1,098 | **2,058** | **2,758** |
| 1,500 | 222 | 826 | **1,570** | **1,976** |
| 3,000 | 157 | 582 | **1,132** | **1,531** |

(The 1-thread 3,000-token cell takes ~20 s per run, so it was kept.)

**Chunk size** (`-b`), 1,500 tokens, 8 threads:

| `-b` | 1 (token by token) | 64 | 128 | 256 | 512 (default) |
|---|---|---|---|---|---|
| prefill tok/s | 192 | 1,323 | 1,386 | 1,496 | **1,570** |

**Decode tok/s** vs threads (`scripts/bench_qwen3c.sh`: 33-token chat prompt, 256 new tokens):

| threads | 1 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| decode tok/s | 46.6 | 138.3 | **148.1** | **169.8** | 19.9 (collapse) |
| prefill tok/s (33 tokens) | 326 | 951 | 1,564 | 1,934 | 267 |

**Decode vs context length** (new tokens after a long prompt; the model stops by itself after
84 / 102 tokens):

| prompt | mean position | KV read per token | 8 threads | 12 threads |
|---|---|---|---|---|
| 33 tokens (table above) | ~161 | 37 MB | 148.1 | 169.8 |
| 1,500 tokens | ~1,542 | 354 MB | 117.5 | 97.0 |
| 3,000 tokens | ~3,051 | 700 MB | 84.1 | 80.8 |

**Effective decode bandwidth** (bytes per token = 0.633 GB of Q8_0 weights incl. the tied classifier
+ the fp32 KV cache read up to the current position, 28 layers x 8 KV heads x 128 x 2 x 4 B =
224 KiB per position):

| case | GB per token | tok/s | effective GB/s | measured read bandwidth, same threads (max of session) |
|---|---|---|---|---|
| 1 thread, short | 0.670 | 46.6 | 31 | 35 |
| 4 threads, short | 0.670 | 138.3 | 93 | 71 |
| 8 threads, short | 0.670 | 148.1 | 99 | 73 |
| 12 threads, short | 0.670 | 169.8 | 114 | 115 |
| 8 threads, 1,500 context | 0.987 | 117.5 | 116 | 73 |
| 8 threads, 3,000 context | 1.333 | 84.1 | 112 | 73 |
| 12 threads, 3,000 context | 1.333 | 80.8 | 108 | 115 |
| qwen3.c, 8 threads, short | 0.670 | 101.4 | 68 | 73 |

Decode runs at ~100-115 GB/s effective from 4 threads up, i.e. memory-bound; the simple
bandwidth test (one streaming loop, `bench/bw.c`) under-reads this machine at 4-8 threads (it gave
87-100 GB/s at 8 threads in earlier sessions, 60-73 today), so treat its column as a noisy lower
bound, not a roofline. At 3,000 tokens of context the fp32 KV cache (0.70 GB) is read more than the
weights, so decode speed halves: a 16-bit KV cache was the next decode lever (done afterwards: f16 is the default now, see [docs/usage.md](docs/usage.md#kv-cache-types)).

**Startup** (`-c 4096 -t 0 -n 1 -i "Why is the sky blue?"`, page cache warm, process start to exit
after the first generated token):

| | wall | peak RSS |
|---|---|---|
| gwenedeg, 12 threads (default), 5 runs | 0.75-0.80 s (prefill of the 18-token prompt 12-28 ms) | 1,361 MB |
| gwenedeg, 8 threads, 3 runs | 0.76-0.78 s | 1,361 MB |
| gwenedeg without `-c` (40,960 context) | 0.79 s | 1,361 MB |
| gwenedeg `-m tokenize` (no weights) | 0.025 s | 11 MB |

Startup is dominated by page faults: the dequantized fp32 embedding table (~620 MB) and the repacked
VNNI weights (~630 MB, ~55 ms to pack). mynah-slm reports a 0.14-0.15 s model load (mmap'd GGUF).

## Comparison (same session)

| | qwen3.c (upstream) Q8_0 | mynah-slm Q8_0 | mynah-slm Q4_K_M | **gwenedeg** Q8_0 |
|---|---|---|---|---|
| decode, 1 thread | 21.4 | | | 46.6 |
| decode, 4 threads | 70.9 | 33.8 | 28.4 | 138.3 |
| decode, 8 threads | 101.4 | 33.6 | 32.6 | 148.1 |
| decode, 12 threads | 105.5 | 34.5 | 34.3 | 169.8 |
| prefill, short bench prompt (~33 tokens), 8 threads | 98.7 | 57.5 | 47.4 | 1,564 |
| prefill, short bench prompt, 12 threads | 80.4 | 53.8 | 45.0 | 1,934 |
| prefill, 1,500 tokens, 8 threads | not run (see below) | 23.6 | 21.5 | 1,570 |
| token-by-token reference: gwenedeg `-b 1`, 1,500 tokens, 8 threads | | | | 192 |

- Decode: `scripts/bench_qwen3c.sh` (`GWENEDEG_BIN=` for qwen3.c) and `scripts/bench_mynah.sh`, same prompt
  text, greedy, 256 new tokens, best of 3. Prefill is the short prompt of the same runs.
- qwen3.c's tokenizer is quadratic (minutes for a 450-token prompt, outside its timing but
  impractical), so it was only run with the short prompt. Its prefill is token by token with the
  classifier on every token; gwenedeg `-b 1` is Gwenedeg's own token-by-token path (VNNI GEMV + AVX-512
  attention), labelled as such, not qwen3.c.
- mynah-slm 1,500 tokens: `mynah-slm run -p "$(scripts/bench_prefill.sh -P 1500)" --ctx 4096 -n 16
  -t 8 --temp 0 --think off --no-stream` (1,500 prompt tokens in its count too), 2 runs Q8_0 (17.2,
  23.6), 1 run Q4_K_M; ~65-90 s per run, so not repeated more.
- mynah-slm decode is flat from 4 threads; its Q4_K_M (0.40 GB) is no faster than Q8_0 here, so it
  is compute-bound (~13 GB/s), not bandwidth-bound.

## Qwen3.8-27B (IQ3_S): Gwenedeg vs llama.cpp vs ik_llama.cpp, server workloads (2026-11-22)

The same model, prompts and harness as the section below, a different day: mainline **llama.cpp** added, **gwenedeg
re-measured at current main** (the numbers below were of `13d55a5`, before the i-quant tiled GEMM), and ik_llama.cpp's
`--short` and `--long` re-run with the same commands so all three columns come from one day. Raw data, commands and
tuning: [bench/llamacpp-2026-11-22/](bench/llamacpp-2026-11-22/).

| | |
|---|---|
| CPU | AMD EPYC Genoa (Zen 4), 16 vCPUs, 32 MB L3, 30 GB RAM, AVX-512 VNNI / VBMI / BF16 (the same KVM VM as below; only the daytime gwenedeg server stopped for the day, other services running: 1-min load 1-20, see Caveats) |
| model | `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (3.44 bpw, 12.1 GB, with its MTP block), thinking on and off |
| gwenedeg | `50ba538` (main + the i-quant tiled GEMM), gcc 14.2.0; `gwenedeg serve --ctx 98304 --draft mtp -G 6 --tau 0.5 --repack -k q4-dot --slots 6 --parallel 4 --admit bound`, 14 threads, `--kv-disk` in a temporary directory (the production arguments) |
| llama.cpp | `225a5345bbb0` (master of 2026-11-22 03:25Z), Release, `-DGGML_NATIVE=ON`, gcc 14.2.0 (`-march=native`; `ggml_cpu_has_avx512 / _vnni / _vbmi / _bf16` all 1); `llama-server -t 14 -fa on -ctk q8_0 -ctv q4_0 -ub 512 -b 2048 -lm none --no-repack --ctx-checkpoints 4 -cram 512 --jinja`, `-np 1` / `-np 4` at `-c 98304`, 14 threads; every run repeated with `-ctk f16 -ctv f16` (the same flags and context otherwise; `--long` at `-c 49152`), which is faster for mainline everywhere. Tuned with `llama-bench` (threads, ubatch, repacking, mmap) and a server draft-length probe: [tune.md](bench/llamacpp-2026-11-22/tune.md) |
| ik_llama.cpp | `ac8d544b` (2026-11-19), the build and commands of the section below: `llama-server -fa on -ctk q8_0 -ctv q4_0 -rtr -ub 1024 -b 2048 -t 14 --spec-type mtp:n_max=6,p_min=0.5 --ctx-checkpoints 4 --jinja -cram 512`, `-c 16384` (`--short`) / `-c 49152` (`--long`), 1 slot + MTP |
| equal settings | 14 threads; KV cache 8-bit K + 4-bit V (gwenedeg `q4-dot` = mainline / ik `q8_0` K / `q4_0` V: equal bit widths, not the same quantizer), and mainline also with an f16 cache, its faster configuration; greedy and seeded (T 1, top-p 0.95, top-k 20, seed 42); `scripts/s9c_bench.py` unchanged, `--server-cmd` for the other engines; 22 GB memory cap, no swap, one model process at a time, the bench lock held |
| speculative decoding | gwenedeg: MTP, up to 6 guesses, tau 0.5. Mainline runs this file's MTP block (`--spec-type draft-mtp`, also with several slots; the drafts are accepted), but every draft length of 1-6 decoded **slower** than no drafting on this CPU, so its runs below have **no MTP**; ik: MTP as before |
| prompts | as below: `bench/s9c/`, two LiveCodeBench-shaped prompts and two agent-shaped ones, `max_tokens` 512; a 31,755-token cold prompt + 2 follow-up turns of ~2,500 new tokens |

**Short requests** (tok/s per request, mean of 8 at one client; 4 clients: aggregate tokens / the round's wall time):

| | gwenedeg | llama.cpp, 1 slot, q8_0 / q4_0 | llama.cpp, 4 slots, q8_0 / q4_0 | llama.cpp, 1 slot, f16 | llama.cpp, 4 slots, f16 | ik_llama.cpp, 1 slot + MTP |
|---|---|---|---|---|---|---|
| thinking on, 1 client | **17.5** | 3.5 | 4.4 | 5.2 | 4.7 | 9.6 |
| thinking on, 4 clients (greedy / seeded) | **23.0 / 23.0** | 2.7 / 2.7 (queued behind one slot) | 4.7 / 4.2 | 4.2 / 4.0 (queued) | 4.7 / 5.7 | 8.0 / 6.6 (queued) |
| thinking off, 1 client | **21.2** | 3.7 | 3.7 | 4.9 | 4.9 | 11.3 |
| thinking off, 4 clients (greedy / seeded) | **27.0 / 25.4** | 3.0 / 3.0 (queued) | 3.3 / 3.7 | 3.6 / 3.5 (queued) | 4.8 / 5.8 | 7.6 / 9.6 (queued) |
| prefill, 0.8-2.5k-token prompts, tok/s | 101-111 | 48-58 | 49-55 | 68-78 | 63-74 | **117-125** |
| peak RSS (GiB) | 17.4 (98k context) | 14.7 (98k) | 15.9 (98k) | 17.8 (98k) | 19.0 (98k) | 17.7 (16k) |

Mainline's best measured configuration per row, the one the ratios below use: f16 with 1 slot for one client (thinking
on 5.2; off 4.86 / 4.88 with 1 / 4 slots) and for the short prefill; f16 with 4 slots for 4 clients. The f16 runs are
13:13-14:27Z, the q8_0 / q4_0 ones 07:43-09:28Z (similar load, 14-18).

**Long context** (the 31,755-token cold prompt, then two turns adding ~2,500 tokens each; thinking on, greedy, 512 new
tokens):

| | gwenedeg | llama.cpp, q8_0 / q4_0 cache | llama.cpp, f16 / f16 cache (`-c 49152`) | ik_llama.cpp, q8_0 / q4_0 |
|---|---|---|---|---|
| cold prefill (31,755 tokens) | 391 s (81 tok/s) | 5,057 s (6.3 tok/s) | 682 s (47 tok/s) | **309 s (103 tok/s)** |
| decode, cold turn | **14.0 tok/s** | 1.56 | 3.83 | 2.54 |
| decode, follow-up turns | **14.8 / 14.5 tok/s** | 1.52 / 1.46 | 3.81 / 3.88 | 2.05 / 2.16 |
| prefill of a follow-up turn's ~2.5k new tokens | 37.4 / 39.4 s (the 31-34k cached prefix reused) | 924 / 1,006 s (a checkpoint restored: the same `cache_n` as gwenedeg) | 83.9 / 91.5 s | **35.0 / 36.0 s** (checkpoint restored) |
| a follow-up turn, wall | **72 / 75 s** | 1,261 / 1,357 s | 218 / 224 s | 285 / 273 s |
| peak RSS (GiB) | 16.0 (98k context) | 14.4 (98k) | 15.0 (49k) | 17.7 (49k) |

Mainline's quantized-cache attention is its own bottleneck at depth (`llama-bench`, pp512 / tg32 at an 8k depth: q8_0 /
q4_0 14.3 / 3.40, q8_0 / q8_0 17.6 / 3.25, f16 / f16 62.7 / 4.48 tok/s; [depth.md](bench/llamacpp-2026-11-22/depth.md)): with the equal-bit-width cache its
2,048-token batches fall from 45 tok/s to 3 tok/s over the 32k prompt. The f16 column is therefore mainline's better
configuration for this session, and the one the long-context ratios below use.

**Quality, the same tokens and context** (the first-chunk perplexity of the 2026-11-21 protocol, 8-bit K / 4-bit V in
all three): prose gwenedeg 9.483, mainline 9.480, ik 9.493 (2026-11-21); code gwenedeg 7.133, mainline 7.141, ik 7.145:
mainline within 0.03-0.11% of gwenedeg, ik within 0.1-0.2% — the same quality.

**`llama-bench` micro line** (pp512 / tg128, 14 threads, q8_0 K / q4_0 V, no drafting; 06:10-06:13Z): mainline 68.5 /
4.86, ik 124.2 / 5.13. Gwenedeg's equivalent is its CLI on a 524-token prompt, 128 new tokens, plain decoding (`--repack`,
`q4-dot`, 2 runs): **107-110 / 13.8-15.1**.

**Reading** (against mainline's best measured configuration per row, f16 cache throughout): Gwenedeg decodes **3.4-4.4x**
faster than mainline on short requests at one client and **4.1-5.7x** with 4 clients (against its 4 slots), **3.7-3.9x**
faster at 32-37k context, and prefills short prompts **1.3-1.5x** faster per identical request and the cold 32k prompt
**1.7x** faster; a follow-up turn on the 32k session finishes 3.0x sooner. Against mainline with the equal-bit-width
q8_0 / q4_0 cache the factors are 4.0-8.1x (short decode), 1.8-2.4x (short prefill), 12.9x (cold 32k) and 17-18x
(a follow-up turn). Against ik_llama.cpp,
runs of the same day: Gwenedeg decodes 1.8-1.9x faster at one client, 2.6-3.6x with 4 clients, 5.5-7.2x at 32k context,
and a follow-up turn finishes 3.7-4.0x sooner; **ik prefills faster**, less so than before the tiled GEMM: Gwenedeg's
short-prompt prefill is **0.79-0.95x** of ik's per identical request (101-111 vs 117-125 tok/s; about 0.7x, 81-84 vs
110-118, on 2026-11-21), its cold 32k prefill **0.79x** (391 s vs 309 s; 471 vs 319 s on 2026-11-21), a follow-up
turn's 2.5k tokens 0.91-0.94x (37-39 s vs 35-36 s). Quality is equal.

**Caveats**
- One run per cell, a shared VM (1-min load 1-20 over the day, other users' jobs; each run's start / end load in
  `bench/llamacpp-2026-11-22/runs.log`, the minute-by-minute log in `loadavg.txt`): differences of a few tens of
  percent are noise, the 3-9x ones are not. Gwenedeg's short decode today (17.5 / 21.2) is 6% below its 2026-11-21 number
  (18.6 / 21.8), its short prefill up 1.24x, its cold 32k prefill down 17% (471 s to 391 s). ik's re-run moved by
  similar amounts (short decode 7.8 / 11.2 to 9.6 / 11.3, cold 32k 319 s to 309 s).
- Gwenedeg's `--long` ran with `--turns 3` (one more follow-up than the others, 74 s wall, in its JSON); turns 0-2 are
  compared.
- llama.cpp and ik_llama.cpp were configured by us from their docs and tuned with `llama-bench`; their authors may know
  better settings. Mainline's MTP slowdown is measured, not explained (a k+1-token verification pass costs about as
  much as k+1 single-token passes here); its quantized-cache slowdown at depth shows in `llama-bench` alone.
- Equal KV bit widths, not the same quantizer, between gwenedeg and ik and mainline's q8_0 / q4_0 columns; mainline's f16
  columns (its best) hold 2.6x Gwenedeg's KV bits. Mainline's quality was measured with q8_0 / q4_0 only (f16 can only be
  as good or better).
- Mainline is not slower for lack of context: its short-request servers ran at 98,304 positions with the full model,
  with either cache (peak RSS 14.7-19.0 GiB). ik could not (its OOMs of 2026-11-21); its 16k / 48k contexts do not
  change its speed at those depths.
- Not measured: vLLM's CPU backend, other engines, other CPUs.

## Qwen3.8-27B (IQ3_S) vs ik_llama.cpp, server workloads (2026-11-21)

A different session, model and comparison from everything above: not comparable with the 0.6B tables. One engine
against another on the **same GGUF file**, the same prompts, requests and measurement script, one box, one window.
Raw per-request data, commands and logs' numbers: [bench/ikllama-2026-11-21/](bench/ikllama-2026-11-21/).

| | |
|---|---|
| CPU | AMD EPYC Genoa (Zen 4), 16 vCPUs, 32 MB L3, 30 GB RAM (the same KVM VM as above; only the daytime gwenedeg server was stopped for the window, other services kept running: 1-min load 0.5-6.5, see Caveats) |
| model | `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (ISTA-DASLab, 3.44 bpw, 12.1 GB, with its MTP block), thinking on and off |
| gwenedeg | `13d55a5`, gcc 14.2.0, `gwenedeg serve --ctx 98304 --draft mtp -G 6 --tau 0.5 --repack -k q4-dot --slots 6 --parallel 4 --admit bound` (the production arguments; `--kv-disk` in a temporary directory) |
| ik_llama.cpp | `ac8d544b` (upstream `06a141f3` of 2026-11-19 + 2 commits unrelated to this model), `-DGGML_NATIVE=ON`, Release, AVX-512 / VNNI active (checked: `HAVE_FANCY_SIMD`); `llama-server -fa on -ctk q8_0 -ctv q4_0 -rtr -ub 1024 -b 2048 -t 14 --spec-type mtp:n_max=6,p_min=0.5 --ctx-checkpoints 4 --jinja`, 1 slot (its MTP stage is removed with more than one slot) |
| equal settings | 14 threads; KV cache 8-bit K + 4-bit V (gwenedeg `q4-dot` = ik `q8_0` K / `q4_0` V: not the same quantizer, the same bit widths; see Caveats); MTP drafting with up to 6 tokens, p_min 0.5; greedy and seeded (T 1, top-p 0.95, top-k 20, seed 42) requests; `scripts/s9c_bench.py` unchanged for both (`--server-cmd` starts ik's server) |
| prompts | `bench/s9c/`: two LiveCodeBench-shaped prompts, two agent-shaped ones (system block + task), `max_tokens` 512; a 32k-token source file + 2 pi-shaped follow-up turns of ~2,500 new tokens |

**Short requests** (tok/s per request, mean of 8 at one client; 4 clients: aggregate tokens / the round's wall time):

| | gwenedeg | ik_llama.cpp, 1 slot + MTP | ik_llama.cpp, 4 slots, no MTP |
|---|---|---|---|
| thinking on, 1 client | **18.6** | 7.8 | 5.5 |
| thinking on, 4 clients (greedy / seeded) | **23.1 / 23.7** | 7.1 / 6.0 (queued behind one slot) | 8.1 / 4.2 |
| thinking off, 1 client | **21.8** | 11.2 | 5.6 |
| thinking off, 4 clients (greedy / seeded) | **26.1 / 26.6** | 8.3 / 8.6 (queued) | 4.5 / 6.0 |
| prefill, 0.8-2.5k-token prompts, tok/s | 81-84 | **110-118** | |
| peak RSS (GiB) | 18.4 | 17.7 (16k context) | 14.6 |

**Long context** (a 31,755-token cold prompt, then two turns adding ~2,500 tokens each; thinking on, greedy, 512 new tokens):

| | gwenedeg | ik_llama.cpp (1 slot + MTP, `-c 49152`) |
|---|---|---|
| cold prefill (31,755 tokens) | 471 s (67 tok/s) | **319 s (100 tok/s)** |
| decode, cold turn | **13.1 tok/s** | 2.4 tok/s |
| decode, follow-up turns | **14.8 / 15.9 tok/s** | 1.9 / 2.0 tok/s |
| prefill of a follow-up turn's ~2.5k new tokens | 44.2 / 46.7 s (the 31-34k cached prefix reused) | 39.6 / 41.9 s (checkpoint restored) |
| a follow-up turn, wall | **76 / 81 s** | 316 / 301 s |

**Quality, the same tokens and context** (the same file; perplexity over the second half of the first 2,048 / 3,072-token
window, `llama-perplexity` against Gwenedeg's `-m logits`, both with the 8-bit K / 4-bit V cache; the token ids of both
engines' tokenizers are identical): prose (Beagle ch. I) gwenedeg 9.483, ik 9.493 (+0.1%); code (the 32k source's head)
gwenedeg 7.133, ik 7.145 (+0.2%): within the +-0.07 standard error of 1,000-1,500 scored tokens, i.e. the same quality.
(`llama-perplexity`'s printed final estimate also averages in a second window, which gwenedeg was not run on, so it is not
the number to compare.)

**ik_llama.cpp tuning, `llama-bench` on the same file** (pp2048 / tg64, tok/s): `-t 14 -ub 2048`: 148 / 6.0; `-t 16 -ub
2048`: 160 / 4.6; `-t 14 -ub 512`: 135 / 5.9; `-rtr 0` and `-rtr 1` equal within noise on this file; decoding at a
16k-token depth: 4.3 tok/s with a 4-bit V cache, 4.8 with 8-bit. 14 threads decode faster than 16 (other services on the
box); `-ub 2048` helps prefill by ~8-10% and was not usable at 98k context in a 20 GB cap (see below), `-ub 1024` was used.

**Reading**: on this CPU, with this model, gwenedeg decodes **1.9-2.4x** faster on short requests at one client (3.1-4.0x with 4
clients, against ik's one slot) and **5.5-8x** faster at 32k context (13.1 / 2.4 on the cold turn, 14.8 / 1.9 and 15.9 /
2.0 on the follow-ups), serves 4 clients at once with MTP (ik: one slot with MTP, or four without), and finishes a 2.5k-token follow-up
turn on a 32k session ~4x sooner; ik_llama.cpp prefills **1.35-1.5x** faster (cold 32k prompt 319 s vs 471 s). Quality is
equal. Neither gap was profiled: ik's decode without MTP (`llama-bench` tg64, 5.4-6.0 tok/s) is already below Gwenedeg's
plain decode of earlier sessions, and its MTP decode drops sharply with depth (2-2.4 tok/s at 32k); its prefill is
the part gwenedeg could learn from.

Superseded for gwenedeg: [the 2026-11-22 section](#qwen38-27b-iq3_s-gwenedeg-vs-llamacpp-vs-ik_llamacpp-server-workloads-2026-11-22)
re-ran gwenedeg at current main (with the tiled GEMM: short-prompt prefill 101-111 tok/s here to 81-84 below), re-ran this
section's ik commands the same day, and added mainline llama.cpp. This section stays for ik's setup, its OOMs at 98k
context and the session-to-session spread.

**Caveats**
- One run per cell, a shared box (1-min load 0.5-6.5 over the window, from other users' jobs; the same window for both
  engines but not simultaneous): differences of a few tens of percent are noise, the 2-8x ones are not. Gwenedeg's numbers
  here are higher than the S9c ones in `docs/notes/qwen38-speculative.md` (a busier box then).
- ik_llama.cpp was configured by us, from its README and `--help` (flags above, tuned with `llama-bench`); its authors
  may know better settings. Its support for Qwen3.8's hybrid layers and MTP is recent, and it is a fork under heavy
  development (the build is of 2026-11-19).
- Equal KV bit widths, not the same quantizer: Gwenedeg's `q4-dot` is Hadamard-rotated per-32 8-bit K / 4-bit V with a bf16
  dot-product attention; ik's `q8_0` / `q4_0` are ggml's types. The perplexity above says the effect on quality is
  within noise on these texts; that is not a long-context quality test.
- At 98,304 context ik_llama.cpp's server was OOM-killed twice in a 20 GB cap (it reserves the KV cache at the full
  context up front, 2.65 GiB K+V here, plus a 2 GiB compute buffer at `-ub 2048` and an 8 GiB default prompt cache,
  `-cram`); the runs above use 16k (short) and 48k (long) contexts and `-cram 512`, which do not change the speed at
  those depths.
- Not measured: mainline llama.cpp, vLLM's CPU backend, other engines; a plain (no MTP, 1 client) decode of gwenedeg side by
  side with ik's (`llama-bench` tg64 gives ik's 5.4-6.0 tok/s; Gwenedeg's plain decode on this file was 7.2-8.1 tok/s in
  a loaded window of an earlier session, `docs/notes/qwen38-speculative.md`); other CPUs (a different AVX-512 implementation or
  AVX2 would change both engines' numbers).

## Qwen3.8-27B (IQ3_S) prefill with the i-quant tiled GEMM (2026-11-21)

One change measured before / after on the same file as the section above: the prefill GEMM for Q2_K and the i-quants
(branch `pp-tiled-iq`; docs/iquants.md, "Batched rows"; the work and its micro-benchmarks: PP-RESULTS.md). Raw log:
[bench/pp-window-2026-11-21/](bench/pp-window-2026-11-21/), produced by `scripts/pp_window.sh`.

| | |
|---|---|
| CPU | the same AMD EPYC Genoa VM (16 vCPUs, 32 MB L3, 30 GB RAM); window 16:51-17:09Z, the live gwenedeg server stopped, other services running (1-min load 4.8-7.9 before the runs, 10-16 during them, mostly the runs' own threads) |
| model | `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` with `--repack` (IQ3_S, IQ3_XXS as 4-bit codes: weights 12.23 GiB), as in production |
| builds | base `fbe8c51` (main) and new `c601176` (main + the tiled GEMM), gcc 14.2.0, `-Ofast -march=native` |
| workload | the CLI, chat mode, a 2,521-token prompt (the bench paragraph repeated), `-b 512`, `-n 1`, tok/s of the `[timing]` line, 2 runs per cell alternating base / new |

**Prefill tok/s** (2 runs; best vs best):

| threads | base | new | |
|---|---|---|---|
| 8 | 49.8 / 49.3 | 67.3 / 66.3 | 1.35x |
| **14** (production) | 83.4 / 81.3 | **101.1 / 103.3** | **1.24x** |
| 16 | 86.5 / 79.4 | 109.3 / 107.7 | 1.26x |
| 14, without `--repack` | 67.9 | 101.3 | 1.49x |

- **Exactness:** the logits of a ~300-token prompt are bit-identical, base `-b 512` == new `-b 512` == `-b 1` ==
  `-b 77` at 3 threads == `--repack` (sha256 `ff19abe701acd1e1...`).
- **Where the time goes** (`-DGWENEDEG_PROFILE`, 14 threads): the GEMM phases are 92.7% of the base prefill and 90.4% of the
  new one; they gained 1.32x (w1+w3 1.43x, wo 1.31x, w2 1.22x, qkv 1.21x). Attention, the delta-net recurrence and the
  norms are ~7% at this length, unchanged.
- **Chunk size:** `-b` 256 / 512 / 1024 / 2048 (4,113 tokens, 14 threads): 99.9 / 103.9 / 100.1 / 95.4 tok/s.
- **Reading:** at 14 threads Gwenedeg's short-prompt prefill moves from ik_llama.cpp's 0.7x to ~0.9x on this file (CLI
  numbers here vs the server numbers of the section above, 110-118 tok/s; not the same harness, so an estimate; the
  next day's server re-run gives 0.79-0.95x).
  `--repack` no longer changes prefill; it still speeds up decode.

**Caveats**
- Two runs per cell on a shared box: base's runs spread 2-9%, so the speed-ups are good to about ±5%.
- The CLI, not the server. The server workloads were re-run the next day ([the 2026-11-22
  section](#qwen38-27b-iq3_s-gwenedeg-vs-llamacpp-vs-ik_llamacpp-server-workloads-2026-11-22)): short-prompt prefill
  101-111 tok/s against ik's 117-125 (0.79-0.95x per request), the cold 32k prompt 391 s against ik's 319 s below. At
  32k the attention share is larger and the gain smaller.
- The GEMM phases gained 1.32x where the 8-thread micro-benchmark predicts ~1.5x for this type mix; not yet explained
  (PP-RESULTS.md lists the likely causes).

## All model files on the EPYC VM (2026-10-26 to 2026-10-30)

A different set of sessions from the Results above (other days, other loads: not comparable with the 2026-10-25 tables;
this is the table the README carried until the documentation was reorganised on 2026-11-21, moved here unchanged except
for the document paths in the footnotes). Same VM as Setup. Where a footnote gives another date, that cell is from it.

Measured on the 16-vCPU AMD EPYC Genoa VM of Setup (shared, noisy: best of 2, interleaved, 2026-10-26; prefill of
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
prefill, 7.9 / 9.1 of decode at 8 / 16 threads; perplexity of both with `scripts/ppl.py -w 512`, docs/notes/qwen38-speculative.md, G3.
(\*\*\*\*\*) 2026-10-30, `scripts/bench_models.py -t 8,16 -r 2`, the Q3_K_M alongside (42 / 74 prefill, 9.8 / 12.3 decode
at 8 / 16 threads; 13.3 GB); decode after 1,500 tokens 9.8 / 14.3; startup 2.3 s at 16 threads; docs/iquants.md.

Decode is memory-bound and moves with the machine's load (8B ranged 12-30 tok/s); startup takes 0.1-0.2 s
up to 1.7B, ~0.5 s for 4B and ~1 s for 8B. Q4_K_M is the better trade at 4B / 8B (decode ~1.6x Q8_0 at
~60% of the RAM). Q8_0 is the more accurate one: at 0.6B, Q4_K_M has ~25x the logits error and answers
"Sydney" for the capital of Australia.

## Batched decode, aggregate tok/s (Step 10a, 2026-10-26)

From the development log of Step 10a (full tables: `git show b3e48a3:TODO.md`), quoted for the README's former
"batched decode" line. Aggregate tok/s of N sequences decoded in one batched pass, 50 positions per sequence, best of 4
interleaved rounds per binary (2026-10-26, load 6-12), same VM; "before" = the binary before Step 10a, "now" = after it
(commit `302e0b2`); the "decode" column is one sequence through `rq_decode`:

| | decode | N=1 | N=2 | N=4 | N=8 | N=16 |
|---|---|---|---|---|---|---|
| Q4_K_M 8 threads, before | 315 | 186 | 330 | 526 | 758 | 942 |
| Q4_K_M 8 threads, now | 326 | 311 | 515 | 813 | **1,003** (3.1x decode) | 1,091 |
| Q4_K_M 12 threads, before | 345 | 215 | 378 | 619 | 897 | 1,164 |
| Q4_K_M 12 threads, now | 366 | 332 | 552 | 965 | **1,214** (3.3x) | 1,449 |
| Q8_0 GGUF 8 threads, before | 207 | 212 | 372 | 627 | 931 | 1,105 |
| Q8_0 GGUF 8 threads, now | 210 | 216 | 340 | 647 | **976** (4.6x) | 1,146 |
| Q8_0 GGUF 12 threads, before | 195 | 180 | 371 | 690 | 978 | 1,343 |
| Q8_0 GGUF 12 threads, now | 223 | 252 | 403 | 705 | **1,107** (5.0x) | 1,446 |

(Qwen3-0.6B. The Q8_0 cells where "now" is lower than "before" are within the run-to-run noise of +-15-30% on this box.
The server's end-to-end throughput with concurrent clients is in [docs/server.md](docs/server.md): 0.6B Q4_K_M, 12
threads, 16 clients streaming 128-token replies, 1,185 tok/s in aggregate.)

## Apple M1 Pro (macOS, NEON)

Experimental port, branch `mac-m1-pro`: `gwenedeg` built with Apple clang (`make`), NEON `sdot` kernels
for the Q4_K / Q6_K matrices, the attention (all KV cache types) and Q8_0, all in `src/kernels_neon.h`.
Four sessions on 2026-10-25: Q4_K_M 14:56-15:05 UTC
([bench/results-mac-m1-pro-2026-10-25.txt](bench/results-mac-m1-pro-2026-10-25.txt)); Q8_0 with
Q4_K_M rerun for comparison 16:11-16:20 UTC
([bench/results-mac-m1-pro-q8-2026-10-25.txt](bench/results-mac-m1-pro-q8-2026-10-25.txt)); both
formats after the rebase onto Step 7a, against the build from before it, 20:59-21:07 UTC
([bench/results-mac-m1-pro-step7a-2026-10-25.txt](bench/results-mac-m1-pro-step7a-2026-10-25.txt));
the KV cache types after the rebase onto Step 7b's f16 default and the NEON 16-bit attention,
22:17-22:30 UTC ([bench/results-mac-m1-pro-kv16-2026-10-25.txt](bench/results-mac-m1-pro-kv16-2026-10-25.txt)).

| | |
|---|---|
| CPU | Apple M1 Pro, 8 performance + 2 efficiency cores, 32 GB, macOS 26.6.2 |
| sharing | a desktop in use (iTerm, browser, ...): 1-min load before / after the runs 3 / 9.6 (first session), 6.4 / 8.7 (second), 4.3 / 11.1 (third), 10.6 / 12.8 (fourth: right after a test run, Spotlight indexing); the "after" values include the benchmark's own threads |
| compiler | Apple clang 21.0.0, `-O3 -ffast-math -ffp-contract=on -mcpu=native` (`make`; no OpenMP, the pthread pool) |
| models | `Qwen3-0.6B-Q4_K_M.gguf` (unsloth @ `50968a4`, 396,705,472 bytes); `Qwen3-0.6B-q8.bin` (633,495,808 bytes, group 64); greedy, `-b 512`, `-c 4096` |
| gwenedeg | first session `bb69e5f`; second `a2ab409` + the Q8_0 kernels, committed unchanged as `14169f4` (both on base `92fe41f`); third `33069be` (on `10211a3`); fourth `59d1e34` (on `989f712`). The branch has been rebased since each of them (last onto `caf833e`, Step 7b complete), so these commits are no longer on it; the Mac's clone keeps them as tags `mac-m1-pro-pre-step7a` = `f4b8a68`, `mac-m1-pro-on-10211a3` = `2b152b6`, `mac-m1-pro-on-989f712` = `83e133c`. The fourth session's code is `95ccbb7` on the current branch |

### Q4_K_M (first session)

Best of 3 runs (decode: 256 new tokens, the short bench prompt; prefill: `scripts/bench_prefill.sh`,
chat-mode prompts of 451 and 1,500 tokens):

| build, threads | prefill 451 tok | prefill 1,500 tok | decode |
|---|---|---|---|
| generic C kernels, 8 | 156 | 138 | 60 |
| NEON, 4 | 724 | 547 | 160 |
| NEON, 6 | 1,040 | 784 | 179 |
| **NEON, 8** | **1,327** | **968** | **190** |
| NEON, 10 (8 P + 2 E cores) | 1,332 | 982 | 171 |

- NEON vs generic at 8 threads: prefill 8.5x (451 tokens) and 7.0x (1,500 tokens), decode 3.2x.
  Both builds give bit-identical logits (`make test`'s kernels group, and a 688-position comparison
  of all logits, prefill and token by token).
- 8 threads (the performance cores) is the best setting. It was the default in the first two
  sessions (min(12, nproc - nproc/4) = 8 for 10 CPUs); Step 7a's nproc - nproc/8 would give 9, so
  since the rebase the default on Apple Silicon is the performance cores (third session below). The
  two efficiency cores add ~0-1% to prefill and cost ~10% of decode.
- Run-to-run spread, 8 threads: 1-2% (prefill 451), 1% (decode); one 1,500-token run was disturbed
  (278 vs 959-968 tok/s) and is not the reported best. The decode column of `bench_prefill.sh` (15
  tokens) is lower than the 256-token one (164 vs 190 at 8 threads): different prompt and length.
- Where the time goes (one profiled run, 8 threads, `-DGWENEDEG_PROFILE`, 1,500 tokens): GEMMs ~55%,
  attention ~36%. The GEMMs run at ~47% of the `sdot` peak, limited by the exact float fold per group
  (6 rounded operations per 8 `sdot`s), and register-tile shapes (1x6 to 4x3) changed them by less
  than 4%. Decode falls from 190 to ~100 tok/s at 1,500 tokens: the fp32 KV cache then adds about
  340 MB of reads per token to the ~400 MB of weights.
### Q8_0 .bin (second session, Q4_K_M rerun alongside)

Best of 3 runs, 8 threads unless noted, same scripts:

| build, threads | prefill 451 tok | prefill 1,500 tok | decode |
|---|---|---|---|
| Q8_0, generic C kernels, 8 | 659 | 441 | 134 |
| Q8_0, NEON, 6 | 1,327 | 943 | 134 |
| **Q8_0, NEON, 8** | **1,691** | **1,211** | 140 |
| Q8_0, NEON, 10 | 1,671 | 1,220 | 136 |
| Q4_K_M, NEON, 8 (rerun) | 1,332 | 1,009 | **191** |

- Q8_0 NEON vs generic at 8 threads: prefill 2.6x / 2.7x, decode +4%. The generic Q8_0 loop was
  already well auto-vectorized by clang, and decode is bandwidth-bound in both: ~0.67 GB read per
  token at 140 tok/s is ~94 GB/s, ~84% of the ~112 GB/s an 8-thread streaming read measured on this
  machine (a 1 GB NEON read loop, not in the repo).
- Q8_0 vs Q4_K_M, both NEON: prefill +27% (451) / +20% (1,500): no unpacking and 4 float ops per
  fold instead of 6; decode -27%: ~0.67 vs ~0.40 GB per token. Q8_0 for prompt-heavy work, Q4_K_M
  for long answers (and Q8_0 is the more accurate one, see Quality).
- The Q4_K_M rerun matches the first session within 1-4% (prefill 1,332 vs 1,327, 1,009 vs 968;
  decode 191 vs 190).
- Startup with the repacking at load: 0.09 s instead of 0.05 s (generic, mapping only); peak RSS
  ~620 MB both. Run spread at 8 threads 1-2%, except one 1,500-token run at 10 threads (886 vs
  1,213-1,220). Load 6.4 before, 8.7 after (desktop in use).
- Bit-identical to the generic build: `make test`'s kernels group (the .bin through both builds),
  a 688-position comparison of all logits (`-b 512` and `-b 1`), and `make kq-neon-test` (random
  data, group 32 and 64). The GGUF Q8_0 (group 32) was not downloaded, so only the unit test
  covers group 32. Rounding: the NEON fold follows `dot_q8` as written, `((float) isum * ws) * xs`;
  gcc -Ofast on x86 reassociates it to `(ws * xs) * isum` (the AVX-512 kernels follow that), so arm64
  and x86 Q8_0 logits differ in the last bits, as they did with the generic kernels.

### After the rebase onto Step 7a (third session)

`origin/main` moved on during the port, from `92fe41f` to `10211a3`: Step 7a (the thread pool with
phase sync, the decode GEMVs on per-thread item ranges, default threads nproc - nproc/8), then
`-k f32|bf16|f16` and a vectorized `quantize`. The branch was rebased onto it (three small
conflicts: the attention signatures, a test label, README text), and the default thread count on
Apple Silicon became the performance cores (`33069be`: `hw.perflevel0.logicalcpu` when there is
more than one core type; 8 here instead of 9). The build from before the rebase (the old branch tip
`f4b8a68`, the code of `14169f4`) and the rebased one ran interleaved, two rounds, best of 3 per
cell, same scripts (round 1 / round 2):

| model, build, 8 threads | prefill 451 tok | prefill 1,500 tok | decode |
|---|---|---|---|
| Q4_K_M, before | 1,294 / 1,304 | 992 / 971 | 179 / 180 |
| **Q4_K_M, after** | **1,324 / 1,318** | **994 / 988** | **188 / 193** |
| Q8_0 .bin, before | 1,622 / 1,662 | 1,113 / 1,156 | 132 / 133 |
| **Q8_0 .bin, after** | **1,666 / 1,665** | **1,164 / 1,203** | **137 / 137** |

- Step 7a on this Mac: decode +5-7% (Q4_K_M) and +4% (Q8_0); prefill +0-5%, mostly within the
  1-2% run-to-run spread except Q8_0 at 1,500 tokens (+4-5% in both rounds).
- The pre-rebase build measured lower than in the first two sessions (decode 179-180 vs 190-191 for
  Q4_K_M, 132-133 vs 140 for Q8_0; prefill up to 8% lower); cause not determined (desktop in use).
  Compare within a session.
- Threads, rebased build, decode: Q4_K_M 188 / 193 at 8, 175 / 175 at 9, 164 / 168 at 10; Q8_0
  137 / 137, 130 / 129, 128 / 127. So 9 threads (Step 7a's default here) cost 7-9% (Q4_K_M) and 5-6%
  (Q8_0) of decode, 10 threads 7-13%; prefill is within 3% at 8, 9 and 10 threads. Hence the
  performance cores as the default; without `OMP_NUM_THREADS` the process runs 8 threads (checked).
- 16-bit KV caches, rebased build, Q4_K_M, 8 threads, a 1,500-token chat prompt then 128 new tokens
  (best of 3, round 1 / round 2):

  | `-k` | prefill | decode |
  |---|---|---|
  | f32 (then the default) | 995 / 970 | 97 / 98 |
  | bf16 | 205 / 203 | 67 / 65 |
  | f16 | 104 / 102 | 39 / 39 |

  The NEON attention handles only the f32 cache; with bf16 / f16 runq.c's generic attention runs (it
  converts each tile to fp32, then the same arithmetic): prefill ~5x / ~10x slower, decode -31-34% /
  -60%. Logits are bit-identical to the generic build with both (688 positions, `-b 512` and `-b 1`).
  A NEON version (the exact conversion in registers, then the existing kernels) is not written; it
  would halve the ~340 MB of cache reads per decoded token at 1,500 positions. The M1 Pro has no bf16
  instructions (`hw.optional.arm.FEAT_BF16` 0); the conversion does not need them (a 16-bit shift).
  (Since `59d1e34` the NEON attention covers bf16 / f16 too: fourth session.)
- `make test` on the rebased build: 162 passed, 8 failed, the same 8 as unmodified `origin/main`
  built on this Mac (tests that depend on the generated text; docs/usage.md, Limitations).
- Disturbed runs, not the reported bests: Q8_0 before, round 2, 9 threads, 1,500 tokens (prefill 373
  vs 1,154-1,159); Q8_0 after, round 2, 10 threads, 1,500 tokens (decode 60 vs 77).

### KV cache types: f16 default, NEON 16-bit attention (fourth session)

`origin/main` moved on again, to `989f712`: Step 7b's cache layout (per block of 256 positions, each
kv head's 16 tiles contiguous) and f16 as the default KV cache (Step 7b on the dev box: KL to HF
fp32 on 45k tokens +0.2% for f16, +2.6% for bf16). The branch was rebased onto it (the NEON f32
attention adapted to the block layout in the port commit), then `59d1e34` added the NEON attention
for the bf16 / f16 caches: the 16-bit elements converted to fp32 in registers (exact), then the same
arithmetic, bit-identical to the generic build. Builds, interleaved, two rounds, 8 threads, best of 3
(round 1 / round 2): **new** `59d1e34`; **rebased** `4426981` (the 16-bit caches on the generic
attention; one run per cell, none at 3,000 tokens: ~1 min each); **before** the rebase `2b152b6`
(Step 7a, f32). Prompts: `bench_qwen3c.sh`'s short one then 256 new tokens; `bench_prefill.sh -P`
451 / 1,500 / 3,000 tokens then 128 new tokens (decode after a long prompt).

Decode, tok/s:

| model, build, `-k` | short | 451 | 1,500 | 3,000 |
|---|---|---|---|---|
| Q4_K_M, before, f32 | 218 / 218 | 174 / 173 | 110 / 109 | 70 / 71 |
| Q4_K_M, rebased, f32 | 227 / 223 | 188 / 188 | 130 / 130 | 88 / 89 |
| Q4_K_M, rebased, f16 (generic attention) | 160 / 153 | 94 / 91 | 42 / 41 | |
| Q4_K_M, new, f32 | 225 / 225 | 188 / 186 | 129 / 129 | 90 / 89 |
| Q4_K_M, new, bf16 | 232 / 231 | 207 / 204 | 158 / 158 | 115 / 115 |
| **Q4_K_M, new, f16 (default)** | **234 / 230** | **207 / 205** | **157 / 158** | **113 / 115** |
| Q8_0 .bin, before, f32 | 148 / 150 | 125 / 127 | 89 / 89 | 62 / 62 |
| Q8_0 .bin, rebased, f32 | 152 / 153 | 132 / 134 | 101 / 101 | 76 / 75 |
| Q8_0 .bin, rebased, f16 (generic attention) | 108 / 121 | 71 / 82 | 36 / 41 | |
| Q8_0 .bin, new, f32 | 151 / 152 | 132 / 135 | 100 / 102 | 75 / 75 |
| Q8_0 .bin, new, bf16 | 155 / 157 | 144 / 146 | 117 / 118 | 93 / 93 |
| **Q8_0 .bin, new, f16 (default)** | **155 / 158** | **144 / 146** | **117 / 119** | **93 / 93** |

Prefill, tok/s (the short prompt is ~25 tokens: left out):

| model, build, `-k` | 451 | 1,500 | 3,000 |
|---|---|---|---|
| Q4_K_M, before, f32 | 1,331 / 1,229 | 906 / 893 | 480 / 618 |
| Q4_K_M, rebased, f32 | 1,295 / 1,186 | 958 / 912 | 544 / 659 |
| Q4_K_M, rebased, f16 (generic attention) | 286 / 276 | 101 / 99 | |
| Q4_K_M, new, f32 | 1,379 / 1,159 | 959 / 910 | 649 / 652 |
| Q4_K_M, new, bf16 | 1,275 / 1,167 | 948 / 894 | 610 / 659 |
| **Q4_K_M, new, f16 (default)** | **1,257 / 1,167** | **945 / 889** | **580 / 665** |
| Q8_0 .bin, before, f32 | 1,272 / 1,787 | 883 / 1,158 | 660 / 721 |
| Q8_0 .bin, rebased, f32 | 1,186 / 1,824 | 857 / 1,214 | 704 / 789 |
| Q8_0 .bin, rebased, f16 (generic attention) | 223 / 324 | 88 / 103 | |
| Q8_0 .bin, new, f32 | 1,151 / 1,799 | 813 / 1,312 | 638 / 780 |
| Q8_0 .bin, new, bf16 | 1,161 / 1,774 | 817 / 1,258 | 662 / 795 |
| **Q8_0 .bin, new, f16 (default)** | **1,162 / 1,784** | **826 / 1,266** | **697 / 795** |

- f16 vs f32 cache, new build, decode: +2-4% (short), +8-10% (451), +17-22% (1,500), +23-29%
  (3,000 tokens): the cache is half the bytes (229 -> 115 KB per position; at 3,000 positions 688 ->
  344 MB read per token, next to ~400 MB of Q4_K_M weights). Less than a pure bandwidth model gives
  (~1.3x at 1,500, ~1.45x at 3,000, the estimate before this session). bf16 decodes as fast as f16.
- Prefill with the 16-bit caches: -4% to +2% vs f32 in round 2 (round 1 was disturbed: Q8_0 ~35%
  lower for every build, Q4_K_M at 3,000 tokens too). The NEON conversion vs the generic attention
  (rebased build, f16): prefill 4-12x, decode 1.3-3.9x.
- Step 7b's layout alone (rebased vs before, f32): decode +2-4% (short) to +22-27% (3,000 tokens).
  With the f16 default on top, decode vs before the rebase: Q4_K_M +6-8% (short), +19% (451),
  +43-45% (1,500), +61-63% (3,000); Q8_0 +5%, +15%, +32-33%, +50-51%.
- Session to session: the before build decodes 218 here (short prompt, Q4_K_M) against 188-193 for
  the same code in the third session: 13-16% between sessions on this desktop (cause not determined;
  on AC power, Low Power Mode off). Only the comparisons within a round are meaningful.
- After the final rebase onto `caf833e` (Step 7b's generic-attention speedup and docs; the NEON path
  unchanged): Q4_K_M, 3,000-token prompt, default settings (f16, 8 threads), two interleaved rounds
  of 3: prefill 745-760 vs 746-774 tok/s, decode 104-111 vs 107-113 for the build measured above.
- Tests on the new build: `make kq-neon-test` (84 attention cases: bf16 / f16 vs the fp32 cache with
  the same values, bit-identical); `make test` 184 passed, 8 failed (text vs the x86 goldens, README
  Known limitations); 688-position logits, both models, `-k f32 / bf16 / f16`, `-b 512` and `-b 1`:
  NEON == generic.

### Both

- The oMLX figures reported for this Mac (`mlx-community/Qwen3-0.6B-4bit`, ~1,500 tok/s
  prefill / ~150 tok/s decode) were not measured here and their prompt length and settings are
  unknown, so no comparison is made; the same model needs `mlx_lm` on the same prompts for one.

## Quality

### Perplexity

Text: Charles Darwin, *The Voyage of the Beagle*, start of Chapter I (Project Gutenberg #944,
public domain), committed as [bench/ppl_beagle_ch1.txt](bench/ppl_beagle_ch1.txt): lines 129-770 of
`pg944.txt` (2025-04-28 update, sha256 `e13d5717...`), CR removed, the first 19 paragraphs (until >=
4,000 tokens), 17,077 bytes, **4,142 tokens** (HF tokenizer; Gwenedeg's and mynah-slm's tokenizations
are identical, checked for every window). sha256 `6ac0515c1da2cfff...`.

Protocols, both teacher-forced on raw text (no template; Qwen3 has no BOS), PPL = exp(mean NLL):
- **full**: the whole text as one sequence, every token after the first scored (4,141
  predictions). This is exactly `mynah-slm ppl -f TEXT` (one token at a time, f32 KV cache).
- **win512**: 8 non-overlapping windows of 512 tokens (last 46 tokens dropped), each from position
  0, 511 predictions each (4,088).

Command: `.venv/bin/python -I scripts/ppl.py bench/ppl_beagle_ch1.txt -t 8` (HF fp32, gwenedeg `-m
logits`, mynah-slm `ppl` per window file; ~10 min, mostly mynah-slm).

| engine | weights | PPL full | vs HF | PPL win512 | vs HF | NLL full (nats) |
|---|---|---|---|---|---|---|
| HF transformers fp32 | bf16 -> fp32 | **27.571** | | **31.984** | | 3.31677 |
| mynah-slm | Q8_0 GGUF (group 32, fp32 activations) | 27.580 | +0.03% | 32.039 | +0.17% | 3.31709 |
| gwenedeg | Q8_0 (group 64, int8 activations) | 27.805 | +0.85% | 32.206 | +0.69% | 3.32522 |
| mynah-slm | Q4_K_M GGUF | 28.864 | **+4.7%** | 34.185 | **+6.9%** | 3.36258 |

- Q4_K_M costs ~5-7% perplexity over fp32 (+0.046 nats per token on the full text), 5-10x the loss
  of Gwenedeg's Q8_0 and far more than mynah-slm's Q8_0, and it is not faster on this CPU (Comparison).
- Gwenedeg's Q8_0 is ~0.7-0.8% behind mynah-slm's Q8_0: Gwenedeg also quantizes the activations to int8
  per 64-group (the VNNI path, as in upstream qwen3.c) and uses 64- instead of 32-weight groups.
  mynah-slm's own int8-activation mode (`--fast`, not available in `ppl`) is documented at ~1.4%.
- Windowed PPL is higher (less context per prediction); the ranking is the same in both protocols.
- Famous public-domain text may be partly memorized by the model; this lowers all numbers alike and
  does not affect the comparison.

### Logits vs HF (test suite)

`tests/run_tests.sh -v --only logits` at `d76e4cc` (5 prompts, 15-24 tokens, metrics as in
[tests/README.md](tests/README.md)); batched (`-b 64`) and token-by-token give the same numbers:

| prompt | n | cos pos 0 | cos mean (pos >= 1) | cos min | top-1 | top-5 |
|---|---|---|---|---|---|---|
| en | 15 | 0.8316 | 0.99824 | 0.99521 | 13/15 | 0.907 |
| code | 24 | 0.8323 | 0.99929 | 0.99832 | 22/24 | 0.958 |
| zh | 15 | 0.9064 | 0.99961 | 0.99942 | 14/15 | 0.920 |
| chat | 15 | 0.8967 | 0.99879 | 0.99599 | 13/15 | 0.933 |
| count | 15 | 0.8653 | 0.99945 | 0.99916 | 14/15 | 0.893 |
| pooled | 84 | | **0.99910** | | **76/84** | |

## Methodology

- **[timing] line** (gwenedeg and the qwen3.c build): `prompt: N tok in X s` = prefill, from after
  tokenization to the logits of the last prompt token and the first sampled token (time to first
  token without load and tokenization); `generated: M tok` = the following M decode steps. mynah-slm
  prints its own `prefill` / `decode` tok/s.
- Greedy (`-t 0`), chat mode (template, no thinking), `-c 4096`, page cache warm (the scripts `cat`
  the model first). **Best of 3** per cell unless noted; this session ran every gwenedeg speed table
  twice (main pass + `scripts/bench_all.sh`), decode 3x, and the tables show the best of all runs
  of the same configuration (6-12 runs per cell; the -b 512 / 1,500 / 8-thread cell appears in both
  prefill tables). Every run is in the raw file.
- Load: `/proc/loadavg` before every block; if the 1-min load was above 6, wait in 30 s steps (9
  blocks waited 30-150 s). 1-min load at the start of blocks: 2.1-6.0. The load printed after a
  block includes our own threads (a 12- or 16-thread run adds ~12-16), e.g. 14-19 after the decode
  and `bench_all` blocks.
- Commands (from the repo root, `make` first):

```sh
cat /proc/loadavg
for t in 1 4 8 12; do OMP_NUM_THREADS=$t ./bw 1024; done      # gcc -O2 -fopenmp -march=native bench/bw.c -o bw
scripts/bench_all.sh                                          # all gwenedeg tables below, ~8 min, or:
ALL=1 scripts/bench_prefill.sh -l "450 1500 3000" -b 512 -t "1 4 8 12" -r 3   # prefill grid
ALL=1 scripts/bench_prefill.sh -l 1500 -b "1 64 128 256 512" -t 8 -r 3        # chunk sweep
ALL=1 REPS=3 scripts/bench_qwen3c.sh models/Qwen3-0.6B-q8.bin 1 4 8 12 16     # decode
OMP_NUM_THREADS=8 ./gwenedeg models/Qwen3-0.6B-q8.bin -c 4096 -t 0 -n 256 -i "$(scripts/bench_prefill.sh -P 3000)"  # decode at 3,000
# baselines
git show 87e08aa:src/runq.c > /tmp/q/runq.c
gcc -Ofast -fopenmp -march=native -D_FILE_OFFSET_BITS=64 /tmp/q/runq.c -lm -o /tmp/q/qwen3c
ALL=1 REPS=3 GWENEDEG_BIN=/tmp/q/qwen3c scripts/bench_qwen3c.sh models/Qwen3-0.6B-q8.bin 1 4 8 12
scripts/fetch_mynah.sh
ALL=1 REPS=3 scripts/bench_mynah.sh models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf 4 8 12   # and -Q4_K_M
# quality
.venv/bin/python -I scripts/ppl.py bench/ppl_beagle_ch1.txt -t 8
tests/run_tests.sh -v --only logits
```

`ALL=1` (added to the three bench scripts in this session) appends every run's prefill/decode
tok/s, which is where the spread figures come from. `bench_prefill.sh` decode column = only 15
tokens, so decode vs context uses `-n 256` runs instead.

## History (earlier measurements)

Not from this session: from README.md / TODO.md at the time of each step, same VM, other days and
loads (+-20-30%). 8 threads unless noted; prefill in chat mode, `-c 4096`.

| step | prefill ~450 tok | prefill 1,500 tok | prefill 3,000 tok | decode |
|---|---|---|---|---|
| start: qwen3.c (2026-10-24) | ~85-99 | (tokenizer: impractical) | | ~100 |
| Step 1: no classifier for prompt tokens | 117.7 | | | ~105 |
| Step 1b: HF-exact O(n log n) tokenizer | (enables long prompts) | | | |
| Step 2: batched prefill, generic C GEMM (`-b 128`) | 125 (`-b 1`) -> 295 | 81 -> 233 | | ~107 |
| Step 3a: AVX-512 VNNI GEMM / GEMV (`-b 128`) | 287 -> 1,179 | 233 -> 541 | 276 (`-b 256`) | 121 -> 139 |
| Step 3b: AVX-512 attention, threading (`-b 512`) | 1,983 | 1,481 | 1,067 | ~184-195 |
| Step 3b, 12 threads | 2,569 | 2,018 | 1,551 | ~159-185 |
| **this session** (8 / 12 threads) | 2,058 / 2,758 | 1,570 / 1,976 | 1,132 / 1,531 | 148 / 170 |

## Caveats

- **16 threads collapse** (decode 5-20 tok/s, prefill of 33 tokens 17-267 tok/s): OpenMP spin
  barriers (7 per layer in decode) with one thread per vCPU; whenever another process preempts one
  thread, all others spin at the next barrier. Not VM throttling (0% steal). Hence the default of 12
  threads; 8 leaves more headroom on a shared box at a modest cost.
- **Run-to-run variance** on this shared VM, (max - min) / max over the 6-9 runs of a cell: 1-5% at 1
  and 4 threads in prefill, median 12% over all prefill cells, up to 38% (451 tokens, 12 threads);
  decode 14% at 1 thread, 26-45% at 4-12 threads, 73% at 16. Baselines over 3 runs: qwen3.c 1-15%,
  mynah-slm 2-13%. Tables report the best run, i.e. the least-disturbed one.
- **Differences from earlier numbers**: prefill matches Step 3b within noise (-2% to +7%). Decode at 8
  threads is lower (148 vs 184-195): this session's bandwidth test at 8 threads was also lower (60-73
  vs 87-100 GB/s), i.e. more memory traffic from other tenants; decode is memory-bound. Startup is
  ~0.77 s vs ~0.45 s earlier, but the Step 3a binary (`64c0983`) measured the same 0.77-0.80 s in
  this session, so it is the machine state (memory pressure, swap in use), not a code change.
