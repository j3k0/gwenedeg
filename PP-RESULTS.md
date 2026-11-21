# Faster prefill for the i-quants: results (branch `pp-tiled-iq`, 2026-11-21)

**In short.** I added a tiled prefill GEMM for Q2_K and all the i-quants on AVX-512 (`iqx_gemm_item`, `src/iquants.h`).
It decodes a 32-row weight tile once per chunk into a per-thread scratch (exact u8 codes plus their integer scales),
then runs a 2-block x 6-row vpdpbusd register tile over every activation row. The output stays bit-identical (same
integers, same fp32 fold, same order). On the 27B-shaped micro-benchmark (17408 x 5120, 512 rows, 8 threads) the
i-quant GEMM got **1.43-2.11x faster per type**: IQ3_S 2.09-2.11x, IQ3_XXS 1.92-2.00x, IQ4_XS 1.43-1.48x, and
--repack's codes 1.43-1.58x. All one-scale types now take ~0.0040 ns per weight and row. Weighted by the ISTA 27B's
type mix, that is **~1.5x on the i-quant GEMM with production `--repack`** (1.8x without it). The synthetic i-quant model
prefills 6.8-7.3k -> 8.7-9.3k tok/s (**+27-29%**, 1,500 tokens, 8 threads). End-to-end numbers on the 27B need a
measurement window: one command, `scripts/pp_window.sh` (below).

Commits on `pp-tiled-iq`, on top of `eaf34f6` (not pushed, not merged):

| commit | what |
|---|---|
| `179abf3` | the tiled GEMM (`iqx_unpack_tile`, `iqx_tile`, `iqx_gemm_item`), `gemm_plan` / `gemm_item` dispatch (from `IQX_MIN` = 24 rows), `GEMM_SCRATCH`, `tests/kernels/iq_gemm_test.c` (test + micro-benchmark, `make iq-gemm-test`), wired into `make test` (q35 group: `q35/iq_gemm_test`) |
| `e905a38` | `RQ_BATCH_MAX` can be overridden at build time (for the `-b` 1024/2048 measurements; default still 512); the benchmark checks its own outputs are bit-identical |
| `78e030e` | `scripts/pp_window.sh`: the 27B measurement script |
| `39dedf8` | docs/iquants.md ("Batched rows": layout, arithmetic, per-type table, crossover), TODO.md |

Box: AMD EPYC Genoa VM, 16 vCPUs, 1 MB L2 per core, 32 MB L3, gcc 14.2.0 `-Ofast -march=native`. The live server ran
throughout. Every benchmark ran under `nice -n 10` with `OMP_NUM_THREADS=8`. The 1-minute load average is given with
each number: it was 2.8-4.3 for most runs and 8.4-8.7 for the final re-run. Builds used `-j4`. No model over 2 GB was
loaded; the 27B was never touched. Temp files are in `/var/tmp`, which is disk; `/tmp` is tmpfs here.

## 1. Measure first

**Micro-benchmark**: `build/iq_gemm_test -b TYPES [NB [D [K]]]` (new, `tests/kernels/iq_gemm_test.c`).
- Shape: by default D 17408 x K 5120, the 27B's ffn_gate / ffn_up shape: 38-49 MB of packed weights per matrix, more
  than L3, so the weights come from DRAM as in the real model.
- Rows: NB activation rows, default 512 (the prefill chunk).
- Threads: OpenMP with `OMP_NUM_THREADS` threads.
- It compares two schedules that mirror `forward.c`:
  - before: 16-row blocks over all rows (`iq_rows`), claimed a few at a time;
  - after: (tile, chunk) items, all chunks of a tile claimed together, so a tile is decoded once per claim.
- Measurement: best of 5. It prints ms per call and ns per weight per activation row (wall time x threads / (D K NB)).
  It also checks that the two schedules' outputs are bit-identical at that size.
- Test mode (no arguments): every type, both decodes, 1 .. 77 rows, 143 cases bit-identical to `iq_dot`.

**Baseline** (before any change; load 2.8-2.9; ns per weight and row):

| Q2_K | IQ2_XXS | IQ2_XS | IQ2_S | IQ3_XXS | IQ3_S | IQ1_S | IQ1_M | IQ4_XS | IQ3_XXS/r | IQ3_S/r |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0069 | 0.0068 | 0.0073 | 0.0080 | 0.0077 | 0.0083 | 0.0060 | 0.0064 | 0.0056 | 0.0057 | 0.0060 |

(`/r` = `--repack`'s 4-bit codes.) The codebook decode costs up to +48% over IQ4_XS, which has no codebook. That fits
the report's finding: each weight block is decoded again for every 8 activation rows.

**Phase split** (`-DRUNQ_PROFILE`, `synth-q35-iq.gguf`: the test suite's 27B-shaped synthetic model with all nine
types, dim 768, 8 layers; 1,500 tokens, `-b 512`, 8 threads, load 3.2): GEMM phases 68%:
- qkv 23%, w1+w3 22%, w2 13%, wo 10%
- attention 12%
- delta-net 13% (recurrence 9%)
- norms / quantisation ~7%

The model is small, so its weights sit in L2/L3. That flatters the old kernel; the 27B's share needs the window
(step 3 of the script).

**Peak**: a hand-written loop of 16 independent vpdpbusd measures **440 GMAC/s per core** (`/var/tmp/pp-scripts/peak2.c`).

## 2. Cheap wins in `iq512_blk` / `iq_rows`: nothing kept

- **One accumulator for the one-scale types.** One vpdpbusd chain over both halves, started at -B (X0 + X1): the same
  int32, one set1 and one add fewer. Measured at 1 / 4 / 8 / 16 / 512 rows, the difference was within the noise
  (±10% run to run at these sizes). It makes the GEMV's dependent chain 8 long instead of 4, a risk for decode.
  **Reverted.**
- **`D * xs` cannot be hoisted.** It depends on (weight row, super-block) and on (activation row, group): one product
  per (row, group, activation row) whatever the loop order. The bias set1 is per (activation row, group): the tiled
  kernel computes it once per 32 weight rows instead of per 16.
- **32 rows per call sharing each broadcast** is built into the tiled kernel (2 blocks x C rows). In `iq_rows` (fewer
  than 24 rows: verify passes, batched decode) it would need 16 weight registers plus 2 x NC accumulators and spill at
  NC = 8. Not done; see "Not done".

## 3. The tiled GEMM

Design (docs/iquants.md, "Batched rows", has the full text):

- **Work items.** `gemm_plan` marks Q2_K / i-quant jobs tiled when the phase has at least `IQX_MIN` = 24 rows (AVX-512
  only, `RUNQ_IQX`). Items are (32-row tile, chunk of `q8p_nb` rows), as for the K-quants. `iqx_gemm_item` decodes the
  tile once per claim (`*last`).
- **Scratch layout**, 704 bytes per (16-row block, group of 32): 8 x 64 u8 codes v + B, S0 and S1 as int32, D as fp32.
  `iqx_unpack_tile` produces the u8 codes by calling `iq512_group`, the same decode `iq_rows` uses, so the VBMI and
  gather paths give the same bytes.
- **Kernel** `iqx_tile`, 2 blocks x C rows:
  - One-scale types (IQ2_XXS, IQ3_XXS, IQ3_S, IQ1_S, IQ4_XS and the `/r` codes): one chain of 8 vpdpbusd from
    -B (X0 + X1), then I = S0 * t (one vpmulld), C = 6.
  - Two-scale types (IQ2_XS, IQ2_S, IQ1_M): two chains of 4, I = S0 t0 + S1 t1, C = 5.
  - Fold: `kq6_fold`, i.e. `val + (D * xs) * (float) I`, each op rounded, in group order: `iq512_blk`'s operations.
- **Q2_K** (C = 4):
  - Its codes are stored premultiplied by their sub-block scale (v S <= 45 fits u8), so one chain gives
    S0 T0 + S1 T1 directly: the same int32 by distributivity, with no lossy step.
  - Its mins term M0 X0 + M1 X1 is one `vpdpwssd` on int16 pairs (M <= 15, |X| <= 2032): exact.
  - Before these two changes Q2_K was 0.92x the old path; now 1.5-1.6x.
- **Scratch size:** 2 x (k / 32) x 832 bytes per thread (`IQX_SCRATCH`). `GEMM_SCRATCH` = the larger of this and the
  K-quant tile; it is 18% above the K-quant scratch, also for K-quant-only models (+0.1 MB per thread at k = 12288).

**Micro-benchmark, 17408 x 5120, 512 rows, 8 threads** (final tree, load 8.5-8.7; the first post-change run at load
3.5-4.3 gave the same numbers within ±5%: docs/iquants.md has that one):

| type | iq_rows | tiled | speed-up |
|---|---|---|---|
| IQ3_S | 0.0084 | 0.0040 | **2.09x** |
| IQ3_XXS | 0.0081 | 0.0041 | **2.00x** |
| IQ4_XS | 0.0061 | 0.0041 | 1.48x |
| IQ3_S/r, IQ3_XXS/r (`--repack`) | 0.0060, 0.0058 | 0.0040, 0.0040 | 1.50x, 1.43x |
| IQ2_XXS | 0.0071 | 0.0041 | 1.76x |
| IQ2_S, IQ2_XS | 0.0083, 0.0075 | 0.0048, 0.0048 | 1.73x, 1.58x |
| IQ1_S, IQ1_M | 0.0065, 0.0072 | 0.0040, 0.0048 | 1.63x, 1.49x |
| Q2_K | 0.0078 | 0.0053 | 1.49x (1.60x at load 3.5) |

Other measurements at load 3-5:
- **ffn_down shape** (5120 x 17408): IQ3_S 2.02x, IQ3_XXS 1.77x, IQ4_XS 1.37x, IQ2_XXS 1.66x (0.0042-0.0045 ns).
- **Crossover** vs `iq_rows`:

  | rows | 4 | 8 | 16 | 24 | 32 | 64 | 128 |
  |---|---|---|---|---|---|---|---|
  | IQ3_S | 0.83x | 0.87x | 1.20x | 1.33x | 1.54x | 1.76x | 1.95x |
  | IQ2_S / IQ4_XS / Q2_K | 0.75-0.83x | 0.70-0.87x | 0.94-1.02x | 1.02-1.06x | 1.11-1.15x | 1.20-1.35x | 1.36-1.46x |

  Hence `IQX_MIN` = 24.
- **Tuning tried:**
  - C (rows per register tile): one-scale C 4-8 all ~0.0040; two-scale C 2 / 3 / 4 / 5 / 6: 0.0060 / 0.0055 / 0.0053 / 0.0047 / 0.0048.
  - Rows per chunk (`Q8P_NB` 16-128): flat.
  - K-blocking, i.e. accumulating in `out` per 2,048-column block so the tile slice and the activations stay in L2 (exact:
    fp32 stores and reloads): at most ~3% on ffn_down, ~5% worse at K 5120 with 64-group blocks. **Not kept.**
- **Where the time goes now:** a diagnostic build without the fp32 fold (wrong results) runs the one-scale types at
  0.0031 ns (multi-threaded) / 0.0028 ns (single thread, 81% of the vpdpbusd peak). The kernel reaches ~60% of the
  peak. The per-32 fold, which bit-exactness fixes (vpmulld, cvt, 2 muls, add per 8 vpdpbusd), costs ~25%. ik's
  kernel folds once per 256 columns; ops per 512 MACs are now about level with ik's (~13 incl. the fold vs its ~12).
- **`--repack` and prefill:** both layouts decode to the same scratch tile, so prefill speed is the same either way.
  `--repack` still speeds up decode.

**Small-model prefill** (`synth-q35-iq.gguf`, ~1,500-token prompt, `-b 512`, 8 threads, best of 7 alternating runs,
`/var/tmp/pp-scripts/pf.sh`):

| | base (`eaf34f6`) | tiled | |
|---|---|---|---|
| load 4.1 | 6,779 tok/s | 8,744 tok/s | +29% |
| load 8.4-8.7 (final tree) | 7,342 | 9,343 | +27% |
| same, `--repack` | 7,433 | 9,499 | +28% |

Phase split after the change (1,500 tokens, load 3.8):

| phase | before | after |
|---|---|---|
| qkv | 0.055 s | 0.041 s |
| wo | 0.025 s | 0.014 s |
| w1+w3 | 0.050 s | 0.033 s |
| w2 | 0.031 s | 0.019 s |
| attention, recurrence | unchanged | unchanged |

On this small model the GEMM share falls from 68% to 61%. Its qkv phase gains least: the matrices are small (fewer
tiles than ideal for 8 threads), and it includes the Q8_0 alpha / beta (12 x 768, generic path).

**Estimate for the ISTA 27B** (weights by type from the task: IQ3_S 31%, IQ4_XS 21%, IQ3_XXS 20%, IQ2_S 9%, IQ2_XS 3%,
Q2_K 3%, IQ2_XXS 1.4%, IQ1_M 0.3%), time ∝ share x ns from the table:
- With production `--repack`: the i-quant GEMM takes 0.371 instead of 0.563, **1.52x faster**.
- Without `--repack`: 0.371 instead of 0.683, 1.84x.
- End to end, if the i-quant GEMM is a fraction g of the 27B's prefill: 1 / (1 - g + g / 1.52). That is 1.21x at
  g = 0.5, 1.26x at 0.6, 1.32x at 0.7, against the measured ik gap of 1.35-1.5x.
- g is the open number: the window's step 3 measures it.

## 4. `-b` 512 vs 1024 vs 2048

- **Micro-benchmark** (8 threads, load 2.9-5.4), ns per weight and row:

  | rows | 256 | 512 | 1024 | 2048 |
  |---|---|---|---|---|
  | IQ3_S | 0.0042 | 0.0041 | 0.0038 | 0.0040 |
  | IQ4_XS | 0.0042 | 0.0040 | 0.0039 | 0.0040 |
  | IQ2_S | 0.0049 | 0.0047 | 0.0047 | 0.0048 |

  The per-row cost of the tiled GEMM is flat from 256 rows up.
- **Synth model, 4,100 tokens** (an `-DRQ_BATCH_MAX=2048` build, 8 threads, best of 5, load 4.4-4.9): 7,741 / 7,882 /
  7,874 / 7,736 tok/s at `-b` 256 / 512 / 1024 / 2048.
- **Conclusion:** no reason to raise `RQ_BATCH_MAX` for the GEMM. The 27B (64 layers, ~600 barriers per chunk) could
  still gain a little from fewer barriers; the window's step 4 measures it.

## Tests

**Run before each commit**, and again on the final tree:
- `make iq-gemm-test`: 143 cases, every type plus the `/r` codes, 1 .. 77 rows, K up to 17408, the VBMI and the gather
  decodes, vs `iq_dot` and `iq_rows`. All ok.
- `build/iq_repack_test`: all ok.
- The `q35` group: `tests/check.py --only q35`, `GWENEDEG_TEST_PROCS=3`, `TMPDIR` on `/var/tmp`, the Qwen3.5-0.8B Q8_0 /
  Q4_K_M from the main tree read-only, the synth models copied into the worktree's `build/synth-q35`.
  - **639/639** after the GEMM commit; **640/640** on the final tree (now including `q35/iq_gemm_test`).
  - It covers on `synth-q35-iq` and `synth-q35-ista-mtp`:
    - `-b 7` / `-b 64` / `-b 512` vs `-b 1`;
    - 4 vs 3 threads;
    - AVX-512 vs generic;
    - `RUNQ_IQ_GATHER=1`, `--repack`, `--repack=IQ3_S` vs default;
    - the MTP paths and api_test;
    - and the 0.8B Q8_0 / Q4_K_M batched / thread / kernel checks (the K-quant path: `GEMM_SCRATCH`).
- Manual: `synth-q35-iq` logits sha256 identical for `-b 1`, `-b 64`, `-b 512`, generic `-b 512`, `--repack -b 333`,
  and 3 threads `-b 100`. `scripts/pp_window.sh`'s dry run (below) also printed identical sha256 for base vs new.

**Not run:** the default `make test` groups that need `models/Qwen3-0.6B-q8.bin` (tokenizer, logits, batched, threads,
kernels, greedy, chat, gguf, q4, kv, api, server, synth, spec). The worktree has no `models/`, the task only cleared
the Qwen3.5-0.8B files, and the full suite starts ~25 processes (12-16 GB). These groups do not reach the changed code:
it is gated on `iq_type()` / `RUNQ_IQX`, except the scratch size, which the q35 group's K-quant checks cover. Run
`make test` once before merging.

## Needs a measurement window

One command, from the worktree, with the live server stopped (≈ 13 GB free RAM needed):

```sh
cd /opt/projects/tools/gwenedeg-pp && scripts/pp_window.sh
# = scripts/pp_window.sh /opt/projects/tools/gwenedeg/models/Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf eaf34f6
```

- **What it does.** It builds base (`eaf34f6`) and this tree in a temp dir (plain, `-DRUNQ_PROFILE`,
  `-DRQ_BATCH_MAX=2048`; `-j8`), then:
  1. **Bit-identity on the 27B.** sha256 of `-m logits` on a ~300-token prompt: base `-b 512`, new `-b 512`, `-b 1`,
     `-b 77` at 3 threads, and `--repack`. All must be equal.
  2. **Prefill tok/s**, base vs new, ~2,500 tokens, `-b 512 --repack`, at 8 / 14 / 16 threads, best of 2. Also 14
     threads without `--repack`.
  3. **Phase split** (`RUNQ_PROFILE`), base vs new, 14 threads: gives g above and what remains (recurrence,
     attention, norms).
  4. **`-b` 256 / 512 / 1024 / 2048**, ~4,100 tokens, 14 threads.
- **Cost.** Expected duration ~20-25 min: each 27B run is a ~5-10 s load plus 20-45 s of prefill; step 1's `-b 1` run
  is ~40 s. Peak RSS ~12.5-13.5 GB, one runq at a time.
- **Output.** Results go to `bench/pp-window-DATE/log.txt` (stderr in `stderr.txt`).
- **Options.** Env `TH` (step 2's thread counts), `TP` (threads of the other steps, default 14), `REPS`, `JOBS`,
  `SKIP="1 4"`.
- **Dry run.** Tested end to end on `synth-q35-iq.gguf` (`JOBS=4 TH=8 TP=8 REPS=1`, 3 min). Bit-identity all equal
  (`175bded2dd8e714e`); prefill 6,713 -> 8,354 tok/s with `--repack`, 6,248 -> 8,503 without.

**Follow-up, optional (~10 min):** the server workload of BENCHMARKS.md's 27B section, with the new binary, to compare
the "prefill, 0.8-2.5k-token prompts" row (runq 81-84 vs ik 110-118 tok/s) and the 32k cold prefill (471 s vs 319 s).
Command: `scripts/s9c_bench.py` exactly as in `bench/ikllama-2026-11-21/` with `--runq` pointing at this tree's
`runq`; the long-context case takes ~10 min, and at 98k ctx the peak RSS is ~18.4 GB.

## Not done, and why

- **Fold once per 256 columns, and ik's int8 re-quantised tile.** Both change the fp32 rounding or the weights: they
  break bit-exactness. The fold per 32 is now the main cost of the one-scale kernel (~25%).
- **Pre-scaled codes for the other types** (the Q2_K trick). S·v does not fit u8 for them: IQ3_S ±465, IQ1_M ±135,
  IQ2 ±1333. With int16 codes and vpdpwssd it would be exact but at half the MACs per instruction, so slower.
- **S·T in fp32 instead of vpmulld.** It is exact here: every |S·T| < 2^24 (IQ4_XS ≤ 16.5M), so `cvt(S)·cvt(T)` equals
  `cvt(S·T)`. But it replaces one op with another and changes the instruction sequence. The task's rule asks for the
  same integer and float operations, so I left it out. It is worth a try only if a measurement shows vpmulld is the
  bottleneck.
- **A two-scale kernel without the second vpmulld.** The int16-pair vpdpwssd trick needs T to fit int16. That holds for
  Q2_K and IQ1_M (|T| ≤ 18,288), not for IQ2_XS / IQ2_S (87k). Gain ≤ 0.3% of the 27B's weights for IQ1_M.
- **Step-2 variants in `iq_rows`** (fewer than 24 rows: MTP verify passes, batched decode): the single-chain change
  gave no measurable gain. Sharing broadcasts across 2 blocks spills at 8 rows. The decode path was out of scope; it
  would need its own measurement with `--repack-info` / `spec_sweep`.
- **K-blocking for k = 17408.** At most ~3%, losses at k = 5120 (§3). Not kept.
- **NEON.** No tiled GEMM there: it still uses `iq_rows` with 32 rows per pass, unchanged and untested here (no
  AArch64 box). `GEMM_SCRATCH` falls back to `KQX_SCRATCH` there.
- **Non-GEMM phases** (delta-net recurrence, fp32 attention, norms). Out of this task's plan. After this change they
  are a larger share of the prefill. The report's item 6 (a chunked recurrence, bf16 attention) changes rounding and is
  ruled out. Step 3 of the window will show how big they are on the 27B.
- **Raising `RQ_BATCH_MAX`.** It is now overridable at build time only. The measurements so far show no gain (§4).
