# TODO

Goal: fast Qwen3-0.6B inference in C on CPU, based on qwen3.c (Q8_0).
Baseline (16-vCPU EPYC Genoa, 8 threads): prefill ~85 tok/s, decode ~100 tok/s.
Target: prefill several hundred to 1000+ tok/s; decode limited by memory bandwidth.
Now (Step 3b, 8 threads, -b 512): prefill ~1,980 tok/s (451-token prompt), ~1,480 (1,500 tokens),
~1,070 (3,000 tokens); decode ~180-195. 12 threads (the default): prefill ~2,570 / ~2,020 / ~1,550.

Status: `[ ]` todo · `[~]` in progress · `[x]` done

## Step 0a: repo cleanup and preparation
- [x] git init; first commit
- [x] Our working copy of the engine goes to `src/runq.c`; `third_party/qwen3.c` reverts to pristine upstream (for diffs)
- [x] mynah-slm: not vendored; fetched at a pinned commit by a script (reference/benchmark only)
- [x] Top-level Makefile: `make` (build), `make test`, `make bench`
- [x] Scripts and README updated for the new layout

## Step 0b: fast test suite (target: under 30 s total)
- [x] Test hooks in runq: dump token ids (tokenizer) and raw logits per prompt position
- [x] Golden references generated once from HF transformers (fp32), committed (compact)
- [x] Tokenizer test: ids match HF on varied strings (ASCII, Unicode, code, whitespace, special tokens)
- [x] Logits test: Q8_0 logits vs HF fp32, cosine and top-1 agreement within tolerance
- [x] Greedy regression test: fixed prompts, `-t 0`, exact output vs our own golden
- [x] `make test` runs all of them (~15 s; details and baseline in `tests/README.md`)

## Step 1: skip the classifier during prefill
- [x] Compute logits only for the last prompt token (and during generation)
  (`forward(..., compute_logits)`; chat still advances the RNG per skipped prompt token, so
  seeded `-t > 0` output is unchanged)
- [x] Tests pass; measure prefill gain: 8 threads, 443-token chat prompt 99.4 -> 117.7 tok/s
  (+18%, best of 3); bench prompt (short) 91.8 -> 121.2 tok/s; decode unchanged (~105)

## Step 1b: tokenizer (blocks long-prompt benchmarks)
Encoding was O(n² · vocab): 109 tokens took 24 s and 217 tokens 104 s, because of a linear `strcmp` scan of 151,936 entries per lookup.
- [x] Hash-map vocab lookup (open addressing, 2^19 slots); BPE per chunk with a heap of merge
  candidates, O(k log k). Merge priority = the `.tokenizer` scores (tiktoken style: any adjacent
  pair whose concatenation is a token, best score first): verified identical to HF's
  `model.merges` BPE on all 151,643 vocab strings + 300k random token concatenations, so no
  extra merges file is needed
- [x] Qwen2 pre-tokenizer regex split, hand-written (`pretok_next`), with \p{L}/\p{N}/\s tables
  (`src/unicode_tables.h` from `tools/gen_unicode_tables.py`, Unicode 15.1, 807 ranges; `\s` =
  White_Space as in HF's Oniguruma). Added tokens are split out first (leftmost-longest), as in HF.
  The 5 XFAIL cases pass
- [x] Byte fallback: BPE starts from the 256 single-byte tokens, so nothing is ever dropped
  (invalid UTF-8 too); the stdout warning is gone
- [x] Tokenizer goldens 27 -> 69 cases (2 XFAIL: no NFC normalizer, see Known bugs);
  `tests/fuzz_tokenizer.py`: 2,000 random strings + 13 real files (seed 1) and 5,000 (seed 2),
  0 mismatches vs HF
- [x] Target met: 2,000 tokens encode in 0.3-1.3 ms (README text); before: 28 tokens 0.74 s,
  109 tokens 11.0 s on the same text (217 / 2,000 tokens not re-measured: ~1 min / hours).
  A 30,000-byte chunk without spaces: ~5-13 ms. Tokenizer load ~8-20 ms. `-m tokenize` no
  longer loads the weights, so the tokenizer tests take 0.2 s and `make test` ~8 s

## Step 2: batched prefill
- [x] `forward_batch(tokens, n, pos0, logits_mode)`: chunks of `-b` tokens (default 64, max
  B_MAX = 256; `-b 0/1` = the old token-by-token path), activations as (n, dim) row matrices in
  separate RunState buffers (decode buffers unchanged); logits none / last row / all rows
- [x] Per-row RMSNorm, quantize, QK-norm, RoPE (own position); `matmul_batch()` GEMM in generic C:
  parallel over blocks of 4 weight rows, each block reused from L1 for all n activation rows,
  `dot_q8_x4` kernel (each activation value loaded once for 4 rows, ~35% faster than 4 x
  `dot_q8`); K/V GEMMs write straight into the cache; causal attention parallel over (row, head)
  items, ordered by kv head so the threads re-read one K/V slice from L1/L2 (2.8x faster than
  plain (row, head) dynamic scheduling at 1,500 tokens)
- [x] Test: batched logits **bit-identical** to sequential, any chunk size (-b 7/64/256 checked)
  and thread count (1/8/13), because both paths call the same kernels and the 4-row GEMM is built
  without FP contraction (`NO_FP_CONTRACT`; with an FMA, `greedy/chat_system` flips). HF logits
  test also on the batched path; greedy goldens unchanged; seeded `-t 0.8` output identical to
  before (chat still advances the RNG per prompt token). `make test` ~18 s
- [x] Used in generate and chat. Prefill tok/s, 8 threads, chat, best of 3 (2026-10-25):
  451 tokens 124.8 (-b 1) -> 251.9 (16) / 281.2 (32) / 289.4 (64) / 295.2 (128);
  1,500 tokens 81.4 -> 205.4 / 223.2 / 231.4 / 232.8 (2.3x / 2.9x). At -b 128: 4 threads 155 / 124,
  12 threads 406 / 324, 16 threads collapse (73 / 87). Decode unchanged (8 threads ~107)
- [x] GEMM ~290-310 GOPS at 8 threads (int8 ops, 0.88 GOP per token), 36-39 per thread: gcc
  already auto-vectorizes `dot_q8` into `vpdpwssd` (VNNI on 16-bit, after sign extension) plus a
  horizontal sum per 64-group, about 10% of the int8 VNNI peak
- [x] Fixed on the way: SwiGLU NaN in the shipped int8 path (see Known bugs)

## Step 3a: AVX-512 VNNI int8 kernels (GEMM, GEMV)
Before (Step 2, `-DRUNQ_PROFILE`, 8 threads, -b 128): 451 tokens: GEMMs 86% (qkv 25, wo 11, w1+w3 34,
w2 16), attention 10%, rest 4%; 1,500 tokens: GEMMs 68%, attention 29%, rest 3%.
- [x] GEMM kernel (`vpdpbusd`) with cache tiling, for prefill: `src/kernels_avx512.h`, `q8p_gemm`.
  Weights repacked once at load ("Q8P"): per 16-row block and 64-group, 16 x 64 bytes where dword
  lane r holds 4 consecutive bytes of row r (stored as u8 = w + 128), then the 16 row scales; 1088
  bytes per (block, group), same size as Q8_0. So the 16 lanes are 16 output rows and there are no
  horizontal reductions at all (instead of a 16 -> 1 reduction per group, ~3 shuffle/add ops per
  `vpdpbusd` for a row-major 4x4 tile). Activations stay s8, broadcast 4 bytes at a time (embedded
  broadcast operand); the +128 offset is removed by starting each group accumulator at
  -128 * (activation group sum), computed per call (no weight-side sums, no extra memory or decode
  traffic). Register tile 2 blocks (32 rows) x 6 activation rows: 12 int32 + 12 float zmm
  accumulators, float outputs stay in registers over all groups, no spills. Work items = (32-row
  tile, 64 activation rows), static; the weight tile (34 KB at k=1024, 102 KB at k=3072) is reused
  from L1/L2 for every 6 rows. Tile width C = 4..8 and NB = 24..128 rows all within noise (+-4%).
  Result: 0.30 -> 2.42-2.48 TOPS at 8 threads (8x), ~85% of the measured `vpdpbusd` peak (2.9 TOPS,
  372 GOPS per thread, register-only microbenchmark); 12 threads 3.2 TOPS (85% of 3.78). The rest
  is mostly the float fold (cvt, 2 mul, add per accumulator and group, 2 of them on the same pipes
  as `vpdpbusd`: ceiling ~89%)
- [x] GEMV kernel for decode (`q8p_gemv`): same layout, one 16-row block at a time with two
  accumulator chains (even/odd steps), each thread streams one contiguous range. Decode 8 threads
  121 -> 139 tok/s, 4 threads 77 -> 115, 12 threads 125 -> 142, 1 thread 25 -> 45 (2026-10-25, same
  session). Software prefetch (2 KB / 8 KB ahead) was slower (1 thread 46 -> 42 / 38; 4 threads
  116 -> 107 / 92); transparent huge pages for the packed weights: no measurable difference, so not
  used (with THP defrag=madvise the faults can stall on compaction, see the KV cache item)
- [x] Measure against the memory-bandwidth limit: ~0.633 GB of packed weights + ~34 MB of KV cache
  per token (bench_qwen3c, ~150 positions on average): 8 threads ~92 GB/s, 4 threads ~76 GB/s, 1
  thread ~30 GB/s; the read-bandwidth test gave 35 / 72 / 100 / 114 GB/s at 1 / 4 / 8 / 12 threads
  the same day (87 GB/s at 8 in an earlier session). So decode is at the limit up to 8 threads; at
  12 it gets ~94 of ~114 GB/s (threading overhead, Step 3b)
- [x] Bit-identity: per-group int32 sums exact; float fold per output exactly as `dot_q8`
  (`val + (ws * xs) * (float) isum`, groups in order; checked in the -Ofast asm of the generic
  code), written with explicit-rounding intrinsics (`_mm512_mul_round_ps` / `_add_round_ps`), which
  gcc can neither contract into an FMA nor reassociate. VNNI GEMM, VNNI GEMV and the generic kernels
  give byte-identical logits; the logits are byte-identical to the previous commit's binary;
  greedy goldens unchanged. Deliberate breakage: an FMA in the fold gives max |diff| 2.1 (caught by
  `kernels/*`, not by `batched/*`: both paths share the kernel). Also checked: Q8P_RB=1 (odd block
  counts; Qwen3-0.6B has none), C = 1 / 3 / 8, NB = 5, an ASan+UBSan build (-b 1 / 7 / 13 / 64): all
  identical, no reports
- [x] Portable fallback: compile-time selection (`__AVX512F__ && __AVX512BW__ && __AVX512VNNI__`
  and not `-DRUNQ_GENERIC`); `make generic` -> `./runq-generic` (same flags + `-DRUNQ_GENERIC`);
  `make test` compares both bitwise (`kernels/*`). `-march=x86-64` and `x86-64-v3` builds compile
  without warnings; x86-64-v3 (AVX2) passes the logits, batched-bitwise and thread tests, but not the
  greedy goldens: the
  non-kernel float code (rmsnorm, softmax, attention, vectorized at 8 instead of 16 lanes) rounds
  differently. Pre-existing, the previous commit built the same way gives the same text
- [x] Weight repacking cost: load +~55 ms (0.40 -> 0.45 s for a 1-token run), peak RSS 1,316 ->
  1,362 MB: the packed copy is one anonymous mapping and the mapped Q8_0 pages of each tensor are
  dropped (`MADV_DONTNEED`) right after packing it
- [x] Found on the way: the KV cache is calloc'd and was first touched inside the K/V GEMMs, whose
  page faults serialize in the kernel (~55 ms of a 451-token prefill, half of "gemm qkv"). Now
  pre-faulted per chunk with `MADV_POPULATE_WRITE`, one call per (layer, K/V) range in parallel
  (~7 ms). `MADV_HUGEPAGE` on the cache instead made it 2-3x slower (compaction on fault)
- [x] Results (2026-10-25, 8 threads, -b 128; load 3-10, shared machine): prefill 451 tokens 287 ->
  1,179 tok/s (4.1x), 1,500 tokens 233 -> 541 (2.3x); 12 threads 389 -> 1,559 and 319 -> 748.
  Profile now: 451 tokens: GEMMs 43% (qkv 12.8, wo 5.7, w1+w3 16.9, w2 7.7), attention 42%, rest 15%
  (norm+quantize+residual+KV pre-fault 5.4, qk-norm+RoPE 6.6, swiglu+quantize 2.8); 1,500 tokens:
  GEMMs 19%, attention 74%, rest 7%

## Step 3b: attention and threading
Before (Step 3a, 8 threads, -b 128): 451 tokens: GEMMs 43%, attention 42%, rest 15% (qk-norm+RoPE
6.6%); 1,500 tokens: attention 74% (fp32, ~126 GFLOPS).
- [x] RoPE: `rope_row` computes cos/sin once per position (per chunk in `forward_batch`, per token in
  `forward`) with the original expressions (scalar `expf` + `sincosf`, as gcc compiled them before):
  logits byte-identical. qk-norm+RoPE 27 -> 4 ms at 451 tokens (6.8% -> 1.1%)
- [x] Attention, AVX-512 (`kernels_avx512.h`, `att512_*`), one definition for decode and prefill:
  KV cache in tiles of 16 positions x kv head, K transposed (one zmm = one dim of 16 positions),
  V row-major. `attention_kv` takes all query vectors of one kv head (decode: the 2 query heads;
  prefill work item: 6 rows x 2 heads, dynamic schedule, longest row blocks first) in groups of
  up to 12: scores for 16 positions per FMA, register tile vectors x tiles (<= 24 accumulators,
  no horizontal sums), softmax with a Cephes-style `exp` (FMA polynomial + exponent add), V as a
  small GEMM tile (vectors x 32-128 dims), each vector's causal length respected exactly. Rows per
  item 3 / 6 / 8 / 12 all within noise. Result: 1,500 tokens 2.0 -> 0.39 s, 3,000 tokens 1.54 s,
  ~670 GFLOPS at 8 threads (was ~126; FMA peak ~900)
- [x] Bit-identity, option (a): the arithmetic is defined once (header comment of the attention
  in `kernels_avx512.h`): score = sum of fma chains over 4 chunks of 32 dims (one 128-long chain:
  ~10% larger deviation from a double-precision attention; chunks: 1-cos 4.7e-4 vs 5.3e-4 for the
  old code, on a 288-token prompt), max, `exp_att`, 16 partial sums + fixed tree, one division, V
  fma chain in position order, times 1/sum. Explicit-rounding intrinsics; the generic C
  `attention_vec` does the same operations (`fp-contract=off`, no reassociation; also verified at
  -O1). Decode == prefill == generic, any thread count: all `batched/*`, `kernels/*`, `threads/*`
  bitwise. Cost of the chunked scores: ~3%. Logits vs HF: thresholds unchanged and met (pooled
  cos 0.99910, top-1 76/84, as before); greedy `chat_system` and `generate` flipped a token and
  were regenerated. New test prompt `long` (187 tokens, 12 tiles, 3 chunks): batched vs
  sequential, VNNI vs generic, threads 8 vs 3. Sanity: a linear instead of tree sum in the C
  version gives max |diff| 1.3-1.5 (caught by `kernels/*`). ASan+UBSan (-b 1 / 7 / 13 / 64, both
  builds, chat and generate on 288 tokens): no reports
- [x] Decode threading: `forward()` is one parallel region per token (was ~230 fork/joins), 7
  barriers per layer: [single] residual + rmsnorm + quantize, [for] q / k / v per head with its
  QK-norm + RoPE + cache store, [for] attention per kv head + quantize, [for] wo, [single], [for]
  w1 + w3 + SwiGLU + quantize per 64 hidden rows, [for] w2. Same functions per output (byte-identical
  logits). Decode 8 threads ~141-150 -> ~184-195 tok/s, 12 threads ~138-142 -> ~159-185, 4 threads
  116 -> 140 (busy box, +-30% run to run): ~120 GB/s effective at 8 threads, so the Step 3a
  "bandwidth limit" was mostly fork/join overhead
- [x] Prefill threading: `matmul_batch_n` / `q8p_gemm_n` run q + k + v (and w1 + w3) as one
  parallel loop; `q8p_nb` halves the activation rows per GEMM item while there are fewer than ~6
  items per thread (wo / w2 at 12 threads: 64 -> 128 items); attention + quantize in one region;
  residual add fused into the next layer's norm loop. Within noise at 8-12 threads. Dynamic GEMM
  scheduling: no gain at 8-14 threads and no help at 16, so static
- [x] 16 threads: still collapse (decode ~15-22 tok/s, prefill 451 tokens ~220-540 vs ~2,570 at 12).
  `OMP_WAIT_POLICY=passive` ~40, `GOMP_SPINCOUNT` 1e3 / 3e4 / 3e6: ~40 / ~9 / ~54, `OMP_PROC_BIND=close`
  ~20-30: with 16 spinning threads on 16 vCPUs and other processes running (load 3-5), every
  barrier waits for a preempted thread. Unfixable with spin barriers on this shared VM: default
  threads when `OMP_NUM_THREADS` is unset = min(12, nproc - nproc/4) (12 here; libgomp's default
  was all 16)
- [x] Chunk size: -b 64 / 128 / 256 / 512 / 1024 at 8 threads: 1,500 tokens 1,237 / 1,370 / 1,467 /
  1,547 / ~1,526, 3,000 tokens 911 / 1,018 / 1,093 / 1,112 / ~1,136: default 512, B_MAX 512
  (~58 KB of chunk buffers per row, ~30 MB)
- [x] Decode: the KV cache tile of a new position (every 16 tokens) is pre-faulted for all layers
  (`kv_populate`), no page faults inside the K/V stores
- [x] Profile now (8 threads, -b 512): 451 tokens 0.237 s: GEMMs 67% (qkv 19.0, wo 8.6, w1+w3 27.3,
  w2 12.2), attention+quantize 18.9%, norm+quant+resid 6.1, swiglu+quant 4.3, qk-norm+rope+kv 2.3,
  classifier 1.3; 1,500 tokens 0.98 s: GEMMs 50%, attention 40%, rest 10%; 3,000 tokens 2.74 s:
  attention 57%, GEMMs 36%

## Known bugs
Found by the tests.
- [ ] Tokenizer: no NFC normalizer (HF's tokenizer.json normalizer is NFC); the input is assumed
  to be NFC already. Decomposed text (`e` + U+0301) and NFC singletons (U+2000 -> U+2002,
  U+212B -> U+00C5, CJK compatibility ideographs) tokenize differently from HF. Tests: XFAIL
  `tokenizer/{nfc_decomposed,nfc_singleton}`; the fuzzer only generates NFC strings.
- [x] SwiGLU `1/(1+expf(-x))` under `-Ofast`: for x < -88, `expf` = inf and the vectorized
  reciprocal (`vrcp14ps` + Newton step, 0 * inf) gives NaN. This *did* happen in the shipped int8
  path: gate values down to ~-100 in the last layer on the `chat` and `zh` logits prompts and on
  longer texts. The NaN was quantized to 0, but it broke `quantize()`'s vectorized max (`vmaxps`
  drops a lane's running max), so the scale of that 64-group came out too small (e.g. 0.073
  instead of 0.114) and its larger values overflowed the int8 cast (UB, wrapped). Fixed by
  clamping the exponent to 80 (exact for x >= -80; below, silu is < 1e-33 either way). Logits
  vs HF improve slightly (zh top-5 0.880 -> 0.893, chat cos mean 0.99888 -> 0.99891); greedy
  goldens unchanged; en/code/count logits bit-identical.
- [x] Minor: `quantize()` divided by a zero scale on an all-zero group (NaN cast to int8, UB): now
  writes q = 0. (Written as an early `continue`: rewriting the `x / scale` loop itself changes
  its -Ofast code generation and the rounding of some values.)

## Later / ideas
- [ ] Chat history: stop resetting to position 0 every turn (keep the KV cache across turns)
- [ ] 16-thread collapse: needs barriers that tolerate a preempted thread (e.g. own spin-then-yield
  barriers, or work stealing with no global barrier per phase); OpenMP knobs did not help
- [ ] Attention: bf16 K/V cache with `vdpbf16ps` (half the bytes, 2x the FMA rate; changes the
  numerics, the logits vs HF test decides); a split over positions for decode at long contexts
  (only 8 kv heads = 8 work items, idle threads at 12+); ~670 GFLOPS now, FMA peak ~900
- [ ] Decode: fewer barriers (the two `single` sections could be done redundantly per thread on
  private copies of x); finer q / k / v items at 12 threads (32 heads over 12 threads: 3 vs 2.67)
- [ ] Prefill rest (14% at 451 tokens): norm / quantize / SwiGLU loops as separate parallel
  regions per phase; SwiGLU's `expf` (libmvec) is ~4%
- [ ] Perplexity comparison Q4_K_M vs Q8_0 (small text sample)
- [ ] GGUF loading (Q8_0 / Q4_K_M) without the Python export step
- [ ] Faster sampling (top-p currently sorts the whole vocab)
- [ ] Startup ~0.45 s per process (~55 ms of it the VNNI weight repacking), mostly page faults for the dequantized embedding table (~620 MB
  of floats; parallelizing the dequantize didn't help): look up rows from the Q8_0 table instead
