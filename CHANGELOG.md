# Changelog

Behaviour changes and performance, per release. A release is a git tag; the numbers a page or a README quotes are the
ones of the release they name. Each entry lists what changed **and what moved in the published figures**, so the site
and the README can be updated without re-measuring anything.

Numbers live in [README.md](README.md#headline-numbers) (current, one set) and [BENCHMARKS.md](BENCHMARKS.md) (every
table, labelled with its release). Measurements and setups behind a figure are linked from each.

## Unreleased (after v0.2)

- **Prefill attention against a long cache is 5-7% faster** (`cb7f5cf`, ATT_ROWS 6 -> 10: more query rows share each
  attention item's K/V expansion). Output bit-identical (compare_bins 792/792). Measured on Qwen3.8-27B ISTA IQ3_S,
  14 threads, a 2,560-token chunk: -4.8% at 64k, -6.6% at 98k; no change on short prompts. Not re-measured for the README
  rows (0.8-2.5k prompts: unaffected; the cold 31.8k prompt: the chunks average ~16k of context, so the gain is smaller
  and was not measured). Details: PLAN-KVQ.md / branch `kvq` docs/notes/kvq.md. The q4-i8 int8-attention experiment
  did not beat `q4-dot` (0.83-0.95x per prefill item) and is not on main.

## v0.2 — 2026-11-25

One activation scale per 256 values replaces per 32 as the default for the quantized files that have kernels for it.

**Changed (user-visible)**

- **The activation format is per file.** A GGUF file whose matmul weights are mostly Q2_K, i-quants or Q3_K .. Q6_K now
  runs with one activation scale per 256 values by default; Q8_0, F32 / BF16, NariQ and `.bin` files are unchanged, and
  so is any file whose hidden size is not a multiple of 256. The choice is read from the file's tensor table, never from
  the build, so the same command gives the same bits on every build and every CPU.
- `--act256` still forces one scale per 256, the new `--no-act256` forces per 32. `RqModelInfo.act256` and the
  `gwenedeg serve` banner report the choice ("activations per 256 (the file's choice)").
- Faster prefill, unchanged decode, and a small quality cost — the tables below. For a file of a K-quant / i-quant mix,
  this is a behaviour change: its output is not what v0.1 produced.

**Added**

- Per-256 kernels for every i-quant type and Q2_K (AVX-512), and for Q3_K .. Q6_K (AVX-512, rows and tiled). The
  definition, and the kernel of the generic build, is `kqf_dot` in `src/kquants.h`.
- `GWENEDEG_ACT256=0|1`: pins the format when no flag is given. A test switch: the suite exports `0` so its goldens
  keep their per-32 meaning; `q35/act256_auto` checks the auto rule itself on 11 files.

**Fixed**

- The generic build was 3x slower with the flag (`kqf_dot` unpacked to int32 and multiplied signed by signed). It is now
  at parity with the per-32 kernel (`kqf_dot` / `kq_dot` 0.70-1.10 per type).

**Numbers** (measured on the same files and prompts on both sides; setups: [docs/v4h.md](docs/v4h.md) sections 12-14)

| prefill, tok/s | v0.1 | v0.2 | |
|---|---:|---:|---|
| Qwen3.5-0.8B Q3_K_M, 1,500 tok, 8 threads | 2,304 | 2,529 | 1.10x |
| Qwen3.5-0.8B Q4_K_M, 1,500 tok, 8 threads | 2,399 | 2,502 | 1.04x |
| Qwen3.8-27B Q3_K_M, 1,540 tok, 14 threads | 98.8 | 111.6 | 1.13x |
| Qwen3.8-27B Q3_K_M, 8,040 tok, 14 threads | 96.5 | 111.3 | 1.15x |
| Qwen3.8-27B IQ3_S (`--repack`, the server's file), 1,540 tok | 99.9 | 115.0 | 1.15x |

Decode is unchanged within the noise (27B Q3_K_M, 128 tokens: 9.8 / 10.2 / 10.8 tok/s at v0.1, 10.9 / 10.6 / 9.6 at
v0.2). The 27B server workloads of the headline table were not re-run: its prefill rows carry the 1.12-1.17x measured on
that file (see the website notes below).

**Quality** (measured against the per-32 definition of the same file)

- 27B Q3_K_M, 16,384 tokens (8k wiki + 8k code): KL 0.00215 / 0.00119, top-1 agreement 97.77% / 99.51%, perplexity
  x1.0014 / x0.9994.
- 27B IQ3_S: perplexity x1.004 (prose) / x1.000 (code) — the file the headline table measures.
- Qwen3.5-0.8B Q4_K_M / Q3_K_M: KL +0.0025 / +0.0012 (wiki), +0.0006 / +0.0008 (code) against the file's own error
  (1-8% of it).
- Qwen3-0.6B Q4_K_M against HF fp32: 1 - cos 0.0159 -> 0.0177 pooled over 79 positions (+11%); top-1 68/84 either way.
  Small files pay proportionally more than 27B ones.

**Tests**

- Suite: 1,953 passed, 5 failed on this box under load (3 known disk/load-dependent server checks, 2 timing checks that
  pass on a rerun of the server group, 198/198).
- `build/flat_bench`: 199 cases, every kernel bit-identical to `kqf_dot`.
- `--act256` runs on the 27B synthetic model (full `api_test`, 34/34) and the real 0.8B Q3_K_M (bit-identical across
  batch sizes, thread counts, AVX-512 vs generic, and the API's batched passes).

## v0.1 — 2026-11-24

The first tag: the state every benchmark table in the repository was measured on. One activation scale per 32 values
(Q8_0 and the quantized formats alike), i-quant and Q2_K tiled prefill GEMM, speculative decoding with an MTP drafter,
quantized KV caches with a disk tier, memory-aware admission, the OpenAI-compatible server.

Its numbers are the "v0.1" columns of [BENCHMARKS.md](BENCHMARKS.md) and of the v0.2 entry above. The per-32 default is
still reachable with `--no-act256`.

---

# Publishing the numbers (fovea.cc/gwenedeg)

What a page that quotes this project needs from a release. Take current figures from
[README.md](README.md#headline-numbers) only; [BENCHMARKS.md](BENCHMARKS.md) is the archive and its tables are labelled
with the release they belong to.

## Headline table, v0.2

| tok/s unless noted | gwenedeg | llama.cpp | ik_llama.cpp |
|---|---|---|---|
| decode, 1 client, thinking on / off | 17.5 / 21.2 | 5.2 / 4.9 | 9.6 / 11.3 |
| decode, 4 clients at once (aggregate), thinking on / off | 23.0 / 26.2 | 5.2 / 5.3 (4 slots) | 7.3 / 8.6 (queued) |
| decode in a 32-37k-token session | 14.0-14.8 | 3.8-3.9 | 2.1-2.5 |
| a pi-shaped follow-up turn (+2.5k tokens) on a 32k session, wall | 72-75 s | 218-224 s | 273-285 s |
| prefill, 0.8-2.5k-token prompts | 113-130 | 68-78 | 117-125 |
| prefill, a cold 31.8k-token prompt | 334-349 s | 682 s | 309 s |
| perplexity on the same tokens (prose / code) | 9.52 / 7.13 | 9.480 / 7.141 | 9.493 / 7.145 |

Qwen3.8-27B, the same IQ3_S file of 12.1 GB, 14 threads, 16-vCPU AMD EPYC Genoa (Zen 4) VM, one day of runs. Compared
with the v0.1 headline: **prefill 101-111 tok/s and 391 s became 113-130 tok/s and 334-349 s**, and perplexity
9.483 / 7.133 became 9.52 / 7.13 (the per-256 format costs 0.4% on prose, 0.0% on code). Everything else is unchanged.

## Claims that move with it

| claim | v0.1 | v0.2 |
|---|---|---|
| prefill vs llama.cpp, short prompts | 1.3-1.5x faster | 1.4-1.9x faster |
| prefill vs llama.cpp, cold 32k prompt | 1.7x faster | 2.0x faster |
| prefill vs ik_llama.cpp, short prompts | ik faster: 0.79-0.95x | 0.90-1.11x (parity) |
| prefill vs ik_llama.cpp, cold 32k | ik faster: 0.79x | 0.89-0.92x |
| quality vs the other engines | perplexities within 0.2% | within 0.5% |
| decode vs both | 3.4-4.4x (one client), 4.1-5.7x (four), 3.7-3.9x / 5.5-7.2x at 32k | unchanged |

If the page describes the activation format, it is now: one scale per 256 values by default for K-quant and i-quant
GGUF files, one per 32 for everything else (`--no-act256` forces per 32, `--act256` forces per 256). Q8_0 / F32 / NariQ
outputs are identical to v0.1.

## Facts to keep attached

- The machine is a shared 16-vCPU KVM VM; run-to-run noise is ±15-30%, so quote ranges, not single runs.
- The 27B table is one run per cell, from one day (2026-11-22), and the two engines were configured from their
  documentation — their authors may know better settings.
- gwenedeg's two prefill cells in that table are that run scaled by the 1.12-1.17x that the per-256 format measured on
  the same file; the other engines' cells are as run. Re-running the server workloads is what would replace them.
- Small-model figures: 1,813 / 2,287 tok/s prefill for Qwen3-0.6B Q8_0 and the qwen3.c comparison are unaffected by
  v0.2 (Q8_0 is not a quantized-activation file). The K-quant rows of BENCHMARKS.md's all-model table are v0.1 and are
  now 4-15% conservative on prefill; label them as v0.1 or leave them out.
- Apple M1 Pro / NEON numbers predate v0.2 and are v0.1.

## For the next release

Add an entry with: what changed for a user, the numbers that moved (old and new, same file and prompt), the quality
measurement, and the test result — then bump the tag and point the site at the new entry. If a figure was not re-run,
say which measurement it was scaled from; never let a scaled figure stand as a measured one.
