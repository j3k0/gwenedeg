# Benchmarks

Qwen3-0.6B on the CPU: cwayn `runq` (Q8_0, AVX-512 VNNI, batched prefill) vs upstream qwen3.c and
mynah-slm, plus perplexity. All numbers in the Results, Comparison and Quality sections are from
**one session, 2026-10-25 04:08-05:05 UTC, commit `d76e4cc`**, same machine and conditions. Raw
output of every run: [bench/results-2026-10-25.txt](bench/results-2026-10-25.txt). Earlier
measurements are only in [History](#history-earlier-measurements) and are labelled there.

## Setup

| | |
|---|---|
| CPU | AMD EPYC Genoa (Zen 4), 16 vCPUs (KVM VM, 1 thread per core, 32 MB L3), 30 GB RAM |
| sharing | the VM runs other users' services; 1-min load 2-6 before each block, up to ~19 during some (see Methodology) |
| memory read bandwidth, this session | 1 / 4 / 8 / 12 threads: 33-35 / 49-71 / 60-73 / 86-115 GB/s (min-max of 3-4 runs at the start and end; `bench/bw.c`) |
| compiler | gcc 14.2.0 (Debian 14.2.0-19), `-Ofast -fopenmp -march=native -D_FILE_OFFSET_BITS=64 -lm` (`make`) |
| runq | `d76e4cc` (`src/runq.c` + `src/kernels_avx512.h`); defaults: `-b 512`, threads min(12, nproc - nproc/4) = 12 |
| qwen3.c | upstream engine = `src/runq.c` at `87e08aa` (upstream `55643ff` + only the `-n` / `[timing]` patch), same flags |
| mynah-slm | `cde1599` (`scripts/fetch_mynah.sh`, its own Makefile) |
| HF reference | transformers 4.53.0 / torch 2.7.0, fp32, eager attention (`.venv`) |

| model file | bytes | used by |
|---|---|---|
| `models/Qwen3-0.6B-q8.bin` (Q8_0, group 64, tied classifier; `scripts/export_q8.sh`) | 633,495,808 | runq, qwen3.c |
| `models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf` (unsloth @ `50968a4`, group 32) | 639,447,744 | mynah-slm |
| `models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf` (unsloth @ `50968a4`, imatrix) | 396,705,472 | mynah-slm |
| `models/Qwen3-0.6B/model.safetensors` (bf16, `Qwen/Qwen3-0.6B` @ `c1899de`) | 1,503,300,328 | HF reference (run in fp32) |

## Results: runq

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
weights, so decode speed halves: a bf16 KV cache would be the next decode lever (TODO.md).

**Startup** (`-c 4096 -t 0 -n 1 -i "Why is the sky blue?"`, page cache warm, process start to exit
after the first generated token):

| | wall | peak RSS |
|---|---|---|
| runq, 12 threads (default), 5 runs | 0.75-0.80 s (prefill of the 18-token prompt 12-28 ms) | 1,361 MB |
| runq, 8 threads, 3 runs | 0.76-0.78 s | 1,361 MB |
| runq without `-c` (40,960 context) | 0.79 s | 1,361 MB |
| runq `-m tokenize` (no weights) | 0.025 s | 11 MB |

Startup is dominated by page faults: the dequantized fp32 embedding table (~620 MB) and the repacked
VNNI weights (~630 MB, ~55 ms to pack). mynah-slm reports a 0.14-0.15 s model load (mmap'd GGUF).

## Comparison (same session)

| | qwen3.c (upstream) Q8_0 | mynah-slm Q8_0 | mynah-slm Q4_K_M | **runq** Q8_0 |
|---|---|---|---|---|
| decode, 1 thread | 21.4 | | | 46.6 |
| decode, 4 threads | 70.9 | 33.8 | 28.4 | 138.3 |
| decode, 8 threads | 101.4 | 33.6 | 32.6 | 148.1 |
| decode, 12 threads | 105.5 | 34.5 | 34.3 | 169.8 |
| prefill, short bench prompt (~33 tokens), 8 threads | 98.7 | 57.5 | 47.4 | 1,564 |
| prefill, short bench prompt, 12 threads | 80.4 | 53.8 | 45.0 | 1,934 |
| prefill, 1,500 tokens, 8 threads | not run (see below) | 23.6 | 21.5 | 1,570 |
| token-by-token reference: runq `-b 1`, 1,500 tokens, 8 threads | | | | 192 |

- Decode: `scripts/bench_qwen3c.sh` (`RUNQ=` for qwen3.c) and `scripts/bench_mynah.sh`, same prompt
  text, greedy, 256 new tokens, best of 3. Prefill is the short prompt of the same runs.
- qwen3.c's tokenizer is quadratic (minutes for a 450-token prompt, outside its timing but
  impractical), so it was only run with the short prompt. Its prefill is token by token with the
  classifier on every token; runq `-b 1` is runq's own token-by-token path (VNNI GEMV + AVX-512
  attention), labelled as such, not qwen3.c.
- mynah-slm 1,500 tokens: `mynah-slm run -p "$(scripts/bench_prefill.sh -P 1500)" --ctx 4096 -n 16
  -t 8 --temp 0 --think off --no-stream` (1,500 prompt tokens in its count too), 2 runs Q8_0 (17.2,
  23.6), 1 run Q4_K_M; ~65-90 s per run, so not repeated more.
- mynah-slm decode is flat from 4 threads; its Q4_K_M (0.40 GB) is no faster than Q8_0 here, so it
  is compute-bound (~13 GB/s), not bandwidth-bound.

## Quality

### Perplexity

Text: Charles Darwin, *The Voyage of the Beagle*, start of Chapter I (Project Gutenberg #944,
public domain), committed as [bench/ppl_beagle_ch1.txt](bench/ppl_beagle_ch1.txt): lines 129-770 of
`pg944.txt` (2025-04-28 update, sha256 `e13d5717...`), CR removed, the first 19 paragraphs (until >=
4,000 tokens), 17,077 bytes, **4,142 tokens** (HF tokenizer; runq's and mynah-slm's tokenizations
are identical, checked for every window). sha256 `6ac0515c1da2cfff...`.

Protocols, both teacher-forced on raw text (no template; Qwen3 has no BOS), PPL = exp(mean NLL):
- **full**: the whole text as one sequence, every token after the first scored (4,141
  predictions). This is exactly `mynah-slm ppl -f TEXT` (one token at a time, f32 KV cache).
- **win512**: 8 non-overlapping windows of 512 tokens (last 46 tokens dropped), each from position
  0, 511 predictions each (4,088).

Command: `.venv/bin/python -I scripts/ppl.py bench/ppl_beagle_ch1.txt -t 8` (HF fp32, runq `-m
logits`, mynah-slm `ppl` per window file; ~10 min, mostly mynah-slm).

| engine | weights | PPL full | vs HF | PPL win512 | vs HF | NLL full (nats) |
|---|---|---|---|---|---|---|
| HF transformers fp32 | bf16 -> fp32 | **27.571** | | **31.984** | | 3.31677 |
| mynah-slm | Q8_0 GGUF (group 32, fp32 activations) | 27.580 | +0.03% | 32.039 | +0.17% | 3.31709 |
| runq | Q8_0 (group 64, int8 activations) | 27.805 | +0.85% | 32.206 | +0.69% | 3.32522 |
| mynah-slm | Q4_K_M GGUF | 28.864 | **+4.7%** | 34.185 | **+6.9%** | 3.36258 |

- Q4_K_M costs ~5-7% perplexity over fp32 (+0.046 nats per token on the full text), 5-10x the loss
  of runq's Q8_0 and far more than mynah-slm's Q8_0, and it is not faster on this CPU (Comparison).
- runq's Q8_0 is ~0.7-0.8% behind mynah-slm's Q8_0: runq also quantizes the activations to int8
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

- **[timing] line** (runq and the qwen3.c build): `prompt: N tok in X s` = prefill, from after
  tokenization to the logits of the last prompt token and the first sampled token (time to first
  token without load and tokenization); `generated: M tok` = the following M decode steps. mynah-slm
  prints its own `prefill` / `decode` tok/s.
- Greedy (`-t 0`), chat mode (template, no thinking), `-c 4096`, page cache warm (the scripts `cat`
  the model first). **Best of 3** per cell unless noted; this session ran every runq speed table
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
scripts/bench_all.sh                                          # all runq tables below, ~8 min, or:
ALL=1 scripts/bench_prefill.sh -l "450 1500 3000" -b 512 -t "1 4 8 12" -r 3   # prefill grid
ALL=1 scripts/bench_prefill.sh -l 1500 -b "1 64 128 256 512" -t 8 -r 3        # chunk sweep
ALL=1 REPS=3 scripts/bench_qwen3c.sh models/Qwen3-0.6B-q8.bin 1 4 8 12 16     # decode
OMP_NUM_THREADS=8 ./runq models/Qwen3-0.6B-q8.bin -c 4096 -t 0 -n 256 -i "$(scripts/bench_prefill.sh -P 3000)"  # decode at 3,000
# baselines
git show 87e08aa:src/runq.c > /tmp/q/runq.c
gcc -Ofast -fopenmp -march=native -D_FILE_OFFSET_BITS=64 /tmp/q/runq.c -lm -o /tmp/q/runq-upstream
ALL=1 REPS=3 RUNQ=/tmp/q/runq-upstream scripts/bench_qwen3c.sh models/Qwen3-0.6B-q8.bin 1 4 8 12
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
- The user-reported Mac figures (~1,500 tok/s prefill / ~150 tok/s decode with oMLX on an Apple
  Silicon Mac, model and quantization unknown) are other hardware and were not measured here.
