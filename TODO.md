# TODO

Goal: fast Qwen3-0.6B inference in C on CPU, based on qwen3.c (Q8_0).
Baseline (16-vCPU EPYC Genoa, 8 threads): prefill ~85 tok/s, decode ~100 tok/s.
Target: prefill several hundred to 1000+ tok/s; decode limited by memory bandwidth.
Now (Step 3b, 8 threads, -b 512): prefill ~1,980 tok/s (451-token prompt), ~1,480 (1,500 tokens),
~1,070 (3,000 tokens); decode ~180-195. 12 threads (the default): prefill ~2,570 / ~2,020 / ~1,550.
Startup (Step 4): ~0.06 s per process, peak RSS ~0.63-0.74 GB; chat keeps the conversation.
GGUF (Step 6a): `./runq Qwen3-0.6B-Q8_0.gguf` directly (no export, no side files); closer to HF than
the .bin (perplexity +0.19% vs +0.85%), prefill ~4-8% slower.

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
- [x] Tokenizer goldens 27 -> 69 cases (2 XFAIL then: no NFC normalizer, done in Step 5);
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
  dropped (`MADV_DONTNEED`) right after packing it (Step 4: now read with `pread`, ~23 ms)
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

## Step 4: usability
- [x] Chat history across turns (was: every user turn restarted at position 0). The conversation
  stays in the KV cache, kept as token pieces (system block, per turn the user prompt P_i, the
  reply as generated, `<|im_end|>\n`) in Qwen3's chat template; each turn prefills only the part of
  the canonical sequence not in the cache yet (longest common prefix): normally the `<|im_end|>`
  sampled at the end of the previous answer (so it is in the cache before the next turn), `\n` and
  the new user turn, e.g. 22 tokens in 0.016 s. Pieces are separated by added tokens, so encoding
  them separately = encoding the whole rendering (checked). `-r 1`: earlier answers lose their
  `<think>` block, as in HF's template: the cache is cut back to the answer start and the visible
  answer re-prefilled (2+3 example: 16 tokens instead of 117 kept). `-r 0`: the empty think block
  stays before each earlier answer (the only difference from HF's rendering; checked with
  `apply_chat_template`). Context full: drop the oldest turns until the conversation is at most
  half of (`-c` - reserve), reserve = `-n` or `-c`/4 (<= `-c`/2); system prompt kept (cached
  prefix), the rest re-prefilled; a reply reaching `-c` is cut; a single message too long for `-c`
  is skipped. stdin: one turn per line (any length), EOF or an empty line ends (EOF used to repeat
  the last line forever), piped lines are echoed. `-i` unchanged (one turn; greedy goldens and
  seeded `-t 0.8` single-turn output byte-identical to before)
- [x] Tests (`chat/*`, ~6 s): 3-turn transcript golden; turn 2 remembers "Ada" / "green";
  `RUNQ_CHAT_DUMP` hook (cache ids + every sampled logits vector): the cache text equals the
  template rendering built in Python, re-encodes to the same ids, and the logits are bit-identical
  to a fresh `-m logits` run on the whole conversation, for 3 turns, `-r 1` (stripped reasoning)
  and context full (`-c 160`, 6 turns, 3 dropped). Breakage caught: follow-up prefill at pos0 + 1,
  `\n` after `<|im_end|>` missing. `-b 1` / `7` / `512`: same transcripts. ASan+UBSan: multi-turn
  with `-c 256` (cut replies, dropped turns, a skipped message), `-r 1`, `-b 1/7/512`: clean (one
  `memcpy(NULL, ..., 0)` in the new code fixed)
- [x] Faster sampling. The "sorts the whole vocab" premise was off: top-p already kept only the
  candidates with p >= (1 - p) / (n - 1) (llama2.c), median ~40 of 151,936 at -t 0.8 -p 0.9 (max
  ~1,300; ~44k at -t 2). The cost was the scalar candidate scan (~90 us), qsort (up to 5 ms at -t 2)
  and, at -t 0, a scalar argmax (~125 us). Now: candidate scan one AVX-512 compare per 16 values
  (generic build: blocks of 64 with a vectorized max), a max-heap of the candidates popped only until
  the nucleus is complete (O(n0 + K log n0)), argmax as a vectorized max + first-occurrence search.
  Identical results: same order (prob desc, index asc on ties: the old qsort is glibc's stable merge
  sort at these sizes, checked on glibc 2.41; ties do occur in nuclei at -t 2, and a reversed tie
  order changes the output), same sums in the same order. Checked: 120 seeded runs (12 seeds x
  -t/-p 0.8/0.9, 1.0/0.95, 1.5/0.5, 0.6/0.99, 2.0/0.9, chat and generate, 80 tokens) byte-identical
  to the old binary, also the generic build; a harness on 112 real logits vectors x 10 coins per
  setting: same tokens; argmax vs the old loop on those + 200k random arrays with ties / signed
  zeros. Per call (harness, hot cache): top-p part 102 -> 13 us (-t 0.8 -p 0.9), 141 -> 23
  (1.0/0.95), 283 -> 49 (1.5/0.5), 5,066 -> 1,530 (2.0/0.9). In the engine (decode, 8 or 12
  threads; logits cold, written by the classifier threads): -t 0 126 -> 21 us per token, -t 0.8
  -p 0.9 ~220-270 -> ~130, -t 1 -p 0.9 ~320 -> ~150, i.e. ~0.1 ms of ~5.3 ms per token (~+2% decode
  at ~190 tok/s; end-to-end tok/s on the shared box, load 4-11, varied by +-20% and could not
  resolve it). The rest at -t > 0 is the temperature scale and the softmax (~110 us cold, ~80 hot),
  which gcc already vectorizes (libmvec 16-wide `expf`, vector max / sum / scale); making it
  faster would mean splitting the exp pass over threads with the sum kept in gcc's exact order: not
  done. Tests: seeded goldens `greedy/seeded_chat` (-t 0.8 -p 0.9) and `greedy/seeded_generate`
  (-t 1.5 -p 0.95), generated with the old binary
- [x] Startup and RAM. Before (12 threads, `-c 4096`): ~0.50 s per process, peak RSS 1,361 MB; in
  process ~420 ms of loading: dequantizing the whole token embedding table to floats (622 MB,
  ~300 ms, nearly all page faults) and packing the weights (~118 ms). Now:
  - the embedding rows are dequantized at lookup from the Q8_0 table (`embed_row`, upstream's
    `q * s` expression: same floats; logits files byte-identical to the previous binary, VNNI and
    generic, -b 1 / 7 / 64). The float table was only used for the lookup (the classifier is the
    shared Q8_0 table, packed). With the shared classifier its mapped pages are never dropped now
    (see below), the looked-up rows are simply mapped
  - packing: one parallel loop over (matrix, ~64 KB of rows) items, the source rows read with
    `pread` (copied from the page cache, never mapped: no source page faults and no per-tensor
    `MADV_DONTNEED`, which alone cost ~40 ms in TLB shootdowns / unmapping), the destination
    pre-faulted with parallel `MADV_POPULATE_WRITE` (2 MB per call, ~8 ms instead of ~40 ms of
    faults inside the copy). Pack ~90-118 -> ~23 ms (12 threads; ~35 at 8)
  - result: process wall (`-n 1 -i hi`, chat, 12 threads) ~0.50 -> ~0.06 s; time to first token,
    1-token prompt (`-m generate`, first stdout byte): 12 threads 476 -> 46 ms (median of 9), 8
    threads 503 -> 63; peak RSS 1,361 -> ~625-735 MB (618 MiB packed weights + runtime state; the
    spread: this VM has THP=always, and when free huge pages exist the first KV tile of each layer
    gets a 2 MB page, 56 x 2 MB). Generic build: first token 351 -> 35 ms, RSS 1,251-1,264 -> ~730
    MB. `-m tokenize` unchanged (~11 ms, 10 MB). `make test` 25-32 -> ~15 s. Left in a 1-token
    run (12 threads, in process ~50 ms): pack ~23 ms (now ~2 GB of memory traffic: copy + write),
    tokenizer load ~8.5 ms (151,936 hash inserts), first forward ~9 ms (incl. KV tile faults),
    exit ~4 ms

## Step 5: NFC normalizer in the tokenizer
- [x] tokenizer.json's normalizer is NFC. Its added tokens are all `"normalized": false`, so HF
  matches them on the raw text and normalizes each piece between them separately (a mark after
  `<|im_end|>` never composes with what precedes the token): runq does the same in
  `encode_segment()`, before the pre-tokenizer
- [x] Unicode version: HF tokenizers 0.21.4 links `unicode-normalization-alignments` 0.1.12, whose
  data is Unicode 9.0. Measured (`tools/gen_unicode_tables.py --check-hf`, 1 s): its NFC equals
  Python 3.13's (Unicode 15.1) on every single code point, on all 662,596 ordered pairs of the
  pre-10.0 combining marks and on all primary composites, except for characters added since:
  108 combining marks (U+1DF6..1DFA, U+0D3B, U+1AC0.., U+1E130.., ...) are ccc 0 in HF (never
  reordered), and U+11938 (13.0) has no decomposition (U+11935 + U+11930 do not compose). The
  tables are 15.1 data with exactly these overrides (runq must give HF's ids); `--unicode-only`
  writes pure 15.1 tables (a scratch build with them equals `unicodedata` everywhere)
- [x] Tables (`src/unicode_nfc.h`, `gen_unicode_tables.py --nfc`, 42.6 KB): 2,941 sorted entries
  (cp | NFC_QC << 21 | ccc << 23, binary search) for every code point with ccc != 0, NFC_QC !=
  Yes or a canonical decomposition, a 16-bit index into a pool of full decompositions (3,266
  code points), 940 composition pairs as sorted 64-bit keys, and two bitmaps over 64-code-point
  blocks (any entry / an entry with ccc or QC) so most lookups need no search (CJK, Hangul
  syllables, most letters). Hangul (de)composition is algorithmic. NFC_QC derived: No = not
  NFC-stable alone, Maybe = second of a pair (incl. Hangul V / T jamo)
- [x] Fast path: `nfc_check()`, the standard quick check (all NFC_QC = Yes, ccc in order) with an
  ASCII skip loop: 0.27 ns/byte on ASCII, 0.7-1.7 ns/byte on Chinese / Russian / Vietnamese /
  Korean NFC text (~2% of encode). Otherwise only the part from the last boundary (ccc 0 and
  Yes) before the first failure is normalized (`nfc_normalize`: decompose, stable sort of each
  mark run by ccc (insertion sort, counting sort from 32 marks), compose; 2.5-10 ns/byte)
- [x] Output can be up to 3x the input bytes (U+1D160: 4 bytes -> 12) and have more tokens than
  input bytes, so `encode()` callers allocate `ENCODE_MAX_TOKENS(n)` = 3n + 3 (was n + 3: a heap
  overflow otherwise, e.g. 50 x U+1D160 = 200 bytes -> 300 tokens) and the BPE scratch grows
- [x] Invalid UTF-8 (no HF reference: a Python str cannot hold it): passed through unchanged; each
  invalid byte acts like U+FFFD (a starter composing with nothing), so nothing is reordered or
  composed across it; the valid text around it is normalized as usual
- [x] Tests: the 2 XFAIL cases pass; goldens 69 -> 81 (decomposed Latin, mark order incl.
  blocking, Hangul jamo, Vietnamese, singletons, exclusions, Greek, 600-mark runs, expansion,
  added tokens next to marks, HF's old-data cases) + `tokenizer/nfc_invalid_utf8` (6 byte
  strings: token bytes == the valid runs NFC'd separately). Fuzzer: no NFC-only restriction, new
  generators (marks in random order, jamo, NFD/NFKD words, QC=No characters, mark runs), files
  also in NFD, normalized text compared with HF's normalizer: seeds 1 and 2, 5,000 random
  (~2,400 not NFC) + 28 file pieces each (1.25M HF tokens): 0 mismatches. `--nfc-all`: every
  code point alone (1,112,062), all 850,084 mark pairs, 11,322 composition cases, 13,167 Hangul
  pairs, 100,000 random strings: 0 differences from HF; from `unicodedata` only on strings with
  override code points. Breakage (scratch builds): no reordering, Maybe as Yes, no blocking,
  composing across invalid bytes, unstable counting sort: all caught by `make test` (the fuzzer
  on no blocking: 12% mismatches, minimized to 3 characters). ASan+UBSan: tokenizer suite,
  `--nfc-all`, 2,000-case fuzz, edge inputs: clean
- [x] Speed (same session, load ~4): NFC input unchanged within noise (old speed_2k text
  0.62-0.81 -> 0.62-0.76 ms, README+TODO 32 KB 3.7-4.1 ms both); NFD input +0.1-0.4 ms per 20 KB.
  `make test` ~23 s on the busy box (84 tokenizer checks, 0.4 s)

## Step 6a: load GGUF Q8_0 models directly
`models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf` (unsloth, GGUF v3, 310 tensors, 32 metadata entries, no
`general.alignment`, no `output.weight`, no bos id) works like the .bin: chat, generate, `-m logits`,
`-m tokenize`; the format is detected by magic.
- [x] Reader (`gguf_parse`, ~300 lines with the config and lookups): v2 / v3 header, metadata (all
  value types; string arrays walked; nested arrays rejected), tensor table, alignment (default 32,
  power of two), data offset. Untrusted input: every read goes through `gguf_take` (overflow-safe
  bounds check); counts checked against the remaining bytes (and <= 65,536 entries) before
  allocating; tensor shapes (1-4 dims, product < 2^56), data size by ggml type (block sizes incl.
  the k-quants), offset alignment and range; duplicate names rejected; strings never assumed
  NUL-terminated. Errors name the file and the entry and exit 1. The file is mapped; tensor data
  is read only where used
- [x] Config from `qwen3.*` (block_count, embedding_length, feed_forward_length, head_count(_kv),
  key/value_length, context_length) and checked against the engine: architecture qwen3, RoPE
  freq_base == 1e6 and rms epsilon == 1e-6 (the constants in `rope_row` / `rmsnorm`; making them
  parameters could change -Ofast code generation of the .bin path), key_length == value_length,
  heads % kv heads, no RoPE scaling; vocab = rows of `token_embd.weight`; tied classifier when
  `output.weight` is absent. The engine's shape checks (`malloc_run_state`) as before
- [x] Tensors by llama.cpp name: `token_embd`, `blk.N.attn_{q,k,v,output}`, `attn_{q,k}_norm`,
  `attn_norm`, `ffn_norm`, `ffn_gate` = w1, `ffn_up` = w3, `ffn_down` = w2, `output_norm`,
  `output` (optional); type (Q8_0 matrices, F32 norms) and shape checked (ne[0] is the fastest
  dimension: a (d, k) matrix is { k, d }, the .bin's row-major layout). No Q/K row permutation: the
  converter permutes for the llama architecture only, and the logits match HF (below). Norms copied
  into one buffer in the .bin layout (forward pass unchanged); embedding rows dequantized from the
  blocks at lookup (`q * d`, d the fp16 scale as a float: the values ggml computes)
- [x] Kernels for group size 32 without requantizing: `kernels_avx512.h` takes the group size as a
  parameter (Q8P block of 16 rows x 32 = 512 bytes + 16 fp32 scales = 576 bytes per group; GEMM
  tile and GEMV instantiated for 64 and 32; activation correction per 32 bytes with a 256-bit sum);
  the generic kernels already had a run-time GS. `prepare_weights` (was `pack_weights`) converts
  the Q8_0 blocks once at load, fp16 -> fp32 exact (`fp16_to_f32`, subnormals too), in the same
  parallel pread loop: VNNI build into Q8P with GS 32, generic build into the .bin's q / s arrays;
  inf / NaN scales rejected. Activations quantized in groups of 32 (GS). Same float fold per group
  (`val + (ws * xs) * isum`). Checked: the .bin path's logits files byte-identical to the previous
  binary (VNNI and generic, -b 1 / 7 / 64, 73 tokens) and all .bin tests unchanged; GGUF: VNNI ==
  generic == batched (-b 7 / 64) == sequential, 1 / 3 / 8 threads, all byte-identical
- [x] Tokenizer from the metadata (`build_tokenizer_gguf`; gpt2 model + qwen2 pre-tokenizer
  required): normal tokens decoded from GPT-2's byte-to-unicode form to raw bytes, control (3) and
  user-defined (4) tokens = the 26 added tokens (raw text), unused (5: 267 `[PAD...]` fillers up to
  151,936) empty as the ids missing from the .tokenizer file; merge of rank r -> score -r for the
  token it produces (the .tokenizer's -log(r + 1) has the same order: same BPE); eos from the file,
  bos (absent) = `<|endoftext|>` as config.json (generation stops at both, as with the .bin).
  Checked (scratch Python with the `gguf` package): all 151,936 tokens decode to exactly the
  .tokenizer bytes, the merges equal tokenizer.json's, merge r produces token 256 + r (so scores
  from the rank and from the id coincide), no normal multi-byte token without a merge. Tokenizer
  goldens: all 84 checks pass on the GGUF tokenizer; `fuzz_tokenizer.py --model <gguf>` seeds 1 and
  2 (5,030 cases each, 1.28M HF tokens): 0 mismatches; `--nfc-all`: 0 differences. Load ~30-48 ms
  on one thread, ~20-37 at 12 (.bin ~15-25 ms): decoding 151,936 tokens, 151k hash inserts and 151k
  merge lookups (parallel outside `-m tokenize`, which stays single-threaded: the tests and the
  fuzzer run many at once). The fuzzer takes ~20 s instead of ~12
- [x] Chat templates: the 4 Qwen3 renderings built in (`QWEN3_TMPL_*`), byte-identical to the
  exporter's `.template*` files (test `gguf_template/*`: same chat token ids as the .bin run for
  -r 0 / 1, with and without -y). The file's `tokenizer.chat_template` (Jinja) is not used
- [x] Accuracy vs HF fp32 (`tests/run_tests.sh -v --only logits,gguf`; batched == sequential):

  | prompt | n | cos pos 0 .bin / GGUF | cos mean .bin / GGUF | cos min .bin / GGUF | top-1 .bin / GGUF | top-5 .bin / GGUF |
  |---|---|---|---|---|---|---|
  | en | 15 | 0.8316 / 0.9188 | 0.99824 / 0.99876 | 0.99521 / 0.99769 | 13 / 14 | 0.907 / 0.947 |
  | code | 24 | 0.8323 / 0.9804 | 0.99929 / 0.99951 | 0.99832 / 0.99899 | 22 / 23 | 0.958 / 0.967 |
  | zh | 15 | 0.9064 / 0.9763 | 0.99961 / 0.99972 | 0.99942 / 0.99921 | 14 / 14 | 0.920 / 0.907 |
  | chat | 15 | 0.8967 / 0.9670 | 0.99879 / 0.99936 | 0.99599 / 0.99764 | 13 / 14 | 0.933 / 0.933 |
  | count | 15 | 0.8653 / 0.9385 | 0.99945 / 0.99950 | 0.99916 / 0.99926 | 14 / 14 | 0.893 / 0.933 |
  | pooled | 84 | | 0.99910 / 0.99939 | | 76 / 79 | |

  Pooled 1-cos 9.0e-4 -> 6.1e-4 (-32%), top-1 76 -> 79 / 84, and position 0 (the massive-activation
  token) much better: a ~6500 value costs less precision in a 32-group than in a 64-group
- [x] Perplexity (`scripts/ppl.py bench/ppl_beagle_ch1.txt -t 8 --runq-model ...`, new option; the
  .bin run reproduces BENCHMARKS.md exactly): runq GGUF Q8_0 full **27.623** (+0.19% vs HF fp32
  27.571), win512 32.040 (+0.18% vs 31.984); runq .bin 27.805 (+0.85%) / 32.206 (+0.69%); mynah-slm
  Q8_0 (fp32 activations) 27.580 / 32.039. Group 32 with fp16 scales removes ~80% of runq's gap to
  HF; the rest is the int8 activations
- [x] mynah-slm cross-check (`mynah-slm run -m <gguf> --temp 0 -n 64 -t 8` vs `runq <gguf> -t 0 -n 64`,
  chat with thinking off on both, and `--raw` vs `-m generate`): 4 of 6 chat prompts identical
  (capital of Australia, hash table, primes > 100, translation; the .bin gives the same 4), 2 diverge
  after ~21 characters (sky blue: "due to" vs "because of"; haiku line 2). Raw generation: "The
  three laws of thermodynamics are" identical over all 64 tokens, "Once upon a time..." for 154
  characters; the .bin leaves both after 23-33 characters. (Exact match not expected: mynah-slm
  keeps the activations in float)
- [x] Speed (same session, 2026-10-25, load 2-9, best of 6; prefill `bench_prefill.sh -m ... -b 512`,
  decode `bench_qwen3c.sh`, 256 tokens, best and median of 15 over two sessions):

  | | .bin | GGUF |
  |---|---|---|
  | prefill 451 tokens, 8 / 12 threads | 1,980 / 2,661 tok/s | 1,832 / 2,559 (-7.5% / -3.8%) |
  | prefill 1,500 tokens, 8 / 12 threads | 1,526 / 2,102 tok/s | 1,461 / 2,005 (-4.3% / -4.6%) |
  | decode 8 threads, best / median | 197 / 142 tok/s | 187 / 137 |
  | decode 12 threads, best / median | 180 / 153 tok/s | 194 / 151 |
  | process wall, `-n 1 -i hi`, 12 / 8 threads (median of 9, x2) | 0.110-0.114 / 0.129-0.133 s | 0.129-0.141 / 0.152-0.154 s |
  | peak RSS | 620-632 MB | 667-679 MB |
  | `-m tokenize` (wall, RSS) | 23 ms, 11 MB | 48 ms, 17 MB |
  | generic build: wall, RSS (12 threads) | 0.115 s, 736 MB | 0.164 s, 666 MB |

  Prefill: the float fold runs once per 32 instead of 64 weights (in the GEMM, 8 instead of 16
  `vpdpbusd` per fold: ceiling ~80% instead of ~89% of the int8 peak). Decode: the packed weights
  are 670 vs 633 MB (fp32 scale per 32 weights: 1.125 vs 1.0625 bytes per weight), so up to ~6%
  slower at the bandwidth limit; not resolvable on the shared box (run-to-run +-30%). Startup +15-30
  ms (tokenizer from the metadata), RSS +47 MB (the larger packed weights + ~6 MB of mapped
  metadata). The generic build converts the GGUF into anonymous memory instead of mapping the
  weights: lower RSS than the .bin there
- [x] Robustness: ASan+UBSan builds (VNNI and generic, `-O1`): GGUF chat (multi-turn, -r 1, -y),
  generate at -b 1 / 7 / 512, logits, tokenize: no reports (it found one bug before the commit: the
  .bin path's `QuantizedTensor.b` left uninitialized by `init_quantized_tensors`, harmless only
  because fresh malloc memory was zero). A scratch harness (151 corrupted / truncated files x
  tokenize and generate, both sanitizer builds): magic, version, counts (huge, off by one, 0), key /
  string / array lengths, value types (unknown, nested array, wrong type), array counts, invalid
  token / merge strings, every config key out of range or unsupported, tensor dims, shapes, types
  (F16, Q4_K, unknown), offsets (huge, misaligned, past the end, overlapping), duplicate and
  missing tensors, NaN / inf weight scales, truncation at 22 offsets from 0 to size - 1, 60
  random byte flips: 61 runs fine (data-area flips), 241 clean errors, 0 crashes or sanitizer
  reports
- [x] Tests (`make test` 238 checks, ~25 s on the busy box, was ~21 s): `gguf/*` group (tokenizer
  goldens, logits vs HF at -b 64 on 5 prompts and -b 1 on 2, batched vs sequential and threads 8 vs
  3 bitwise, VNNI vs generic on -b 1 / 7 / 64, the templates) and greedy goldens `gguf_chat`,
  `gguf_generate`; skipped if the GGUF file is missing. `scripts/download_models.py` now fetches
  the Q8_0 GGUF by default
- Notes for Step 6b (Q4_K_M, `Qwen3-0.6B-Q4_K_M.gguf`, 397 MB): tensor types: Q4_K for attn_q /
  attn_k / attn_output / ffn_gate / ffn_up (all 28 layers) and attn_v / ffn_down in 14 layers; Q6_K
  for attn_v / ffn_down in the other 14 and for `token_embd` (the tied classifier, 151,936 x 1024);
  norms F32. Q4_K: super-blocks of 256 (144 bytes: fp16 d, fp16 dmin, 12 bytes of 6-bit scales and
  mins for 8 sub-blocks of 32, 128 bytes of nibbles), w = d * sc * q - dmin * m. Q6_K: 256 (210
  bytes: 4 low + 2 high bits per value, 16 int8 scales per 16 values, fp16 d), w = d * sc * (q - 32).
  Kernels: keep the 4 / 6-bit data packed (decode is bandwidth-bound: ~0.40 vs 0.67 GB per token)
  and unpack to u8 in registers for `vpdpbusd` (u8 x s8 fits: q in 0..15 / 0..63); the offsets
  (dmin * m, -32) go through the activation group sums, as the +128 of Q8P does now; per sub-block
  int32 sums folded with d * sc (activation groups of 32, plus sums per 16 for Q6_K). Requantizing
  to Q8_0 at load would be simpler but loses accuracy and the bandwidth gain. Embedding rows: Q6_K
  row dequantized at lookup. mynah-slm's Q4_K_M perplexity is +4.7% over fp32, so the point would
  be decode speed and RAM

## Known bugs
Found by the tests.
- [x] Tokenizer: no NFC normalizer (HF's tokenizer.json normalizer is NFC); the input was assumed
  to be NFC already. Decomposed text (`e` + U+0301) and NFC singletons (U+2000 -> U+2002,
  U+212B -> U+00C5, CJK compatibility ideographs) tokenized differently from HF (XFAIL
  `tokenizer/{nfc_decomposed,nfc_singleton}`). Fixed in Step 5: NFC with HF's Unicode data, 0
  differences from HF on every code point and 10k fuzz cases; both cases pass.
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

## Plan for the remaining items (in order)
- [x] Step 4 (usability): chat history across turns, faster sampling, startup (embedding rows from the Q8_0 table): done, see above
- [x] Step 5: NFC normalizer in the tokenizer: done, see above
- [x] Step 6a: GGUF Q8_0 loading (no export step, no side files): done, see above
- Step 6b: GGUF Q4_K_M (Q4_K / Q6_K kernels, see the notes in Step 6a)
- Step 7: performance leftovers (decode barriers / attention split, bf16 KV cache, 16-thread barriers, prefill rest)

## Later / ideas
- [ ] 16-thread collapse: needs barriers that tolerate a preempted thread (e.g. own spin-then-yield
  barriers, or work stealing with no global barrier per phase); OpenMP knobs did not help
- [ ] Attention: bf16 K/V cache with `vdpbf16ps` (half the bytes, 2x the FMA rate; changes the
  numerics, the logits vs HF test decides); a split over positions for decode at long contexts
  (only 8 kv heads = 8 work items, idle threads at 12+); ~670 GFLOPS now, FMA peak ~900
- [ ] Decode: fewer barriers (the two `single` sections could be done redundantly per thread on
  private copies of x); finer q / k / v items at 12 threads (32 heads over 12 threads: 3 vs 2.67)
- [ ] Prefill rest (14% at 451 tokens): norm / quantize / SwiGLU loops as separate parallel
  regions per phase; SwiGLU's `expf` (libmvec) is ~4%
- [x] Perplexity comparison Q4_K_M vs Q8_0 (2026-10-25, `scripts/ppl.py`, BENCHMARKS.md): Voyage of
  the Beagle ch. I, 4,142 tokens, one sequence (= `mynah-slm ppl`): HF fp32 27.571, mynah-slm Q8_0
  27.580 (+0.03%), runq Q8_0 27.805 (+0.85%: int8 activations, group 64), mynah-slm Q4_K_M 28.864
  (+4.7%); 512-token windows: 31.984 / 32.039 / 32.206 / 34.185 (+6.9% for Q4_K_M). Step 6a: runq
  on the Q8_0 GGUF (group 32) 27.623 (+0.19%), windows 32.040 (+0.18%)
- [x] GGUF loading without the Python export step: Q8_0 done (Step 6a); Q4_K_M is Step 6b
