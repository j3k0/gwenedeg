# TODO

Goal: fast Qwen3-0.6B inference in C on CPU, based on qwen3.c (Q8_0).
Baseline (16-vCPU EPYC Genoa, 8 threads): prefill ~85 tok/s, decode ~100 tok/s.
Target: prefill several hundred to 1000+ tok/s; decode limited by memory bandwidth.
Now (Step 3b, 8 threads, -b 512): prefill ~1,980 tok/s (451-token prompt), ~1,480 (1,500 tokens),
~1,070 (3,000 tokens); decode ~180-195. 12 threads (the default): prefill ~2,570 / ~2,020 / ~1,550.
Startup (Step 4): ~0.06 s per process, peak RSS ~0.63-0.74 GB; chat keeps the conversation.
GGUF (Step 6a): `./runq Qwen3-0.6B-Q8_0.gguf` directly (no export, no side files); closer to HF than
the .bin (perplexity +0.19% vs +0.85%), prefill ~4-8% slower.
Q4_K_M (Step 6b-1/2): `./runq Qwen3-0.6B-Q4_K_M.gguf` (perplexity +4.9% vs HF, mynah-slm +4.7%; Q8_0
GGUF +0.19%): AVX-512 kernels, decode ~1.5x the Q8_0 GGUF (~200-300 tok/s, bandwidth-bound: 0.40 vs 0.67
GB per token), prefill ~0.85-0.9x (~1,670 / 2,140 tok/s at 451 tokens, 8 / 12 threads), RSS ~0.41 GB.
Threads (Step 7a): own phase synchronization on a thread pool (no OpenMP at run time); 16 threads on
the 16 vCPUs no longer collapse (decode ~140-170 tok/s, was ~2-20; prefill ~2,300-2,500 at 439
tokens, was ~100-600); decode 8 / 12 threads ~190 / ~210 tok/s best (Q8_0 GGUF; Q4_K_M ~280 / ~350),
~+20-50% median; long contexts split per 256-position block (3,000 tokens, 12 threads: ~70 -> ~100
tok/s). Default threads nproc - nproc/8 (14 here).
KV cache (Step 7b): f16 by default (`-k f16|bf16|f32`): half the memory, as accurate as fp32 (KL from HF
+0.2%); with the new cache layout decode after 3,000 tokens 90 -> 128 tok/s (Q8_0 GGUF, 8 threads; Q4_K_M
103 -> 158), prefill unchanged within -0..+7%.
Architecture (Step 8): a library (`src/runq.h`: model / sequence / compute context, multi-sequence batched
passes, errors returned, no printing) in modules, the CLI on top; outputs byte-identical, same speed.
Server (Step 9): `runq serve`, OpenAI-compatible (chat completions incl. streaming, models, health), prefix
KV reuse across requests (bit-identical), constrained decoding for finite JSON schemas; as LiteLLM's
complexity-router judge ~20 ms per classification with the rubric cached (0.6B; 1.7B ~55 ms, 4B ~150 ms).
Batched decode (Step 10a): N sequences decoding together in one `rq_forward_batch` pass, bit-identical to each
alone: Q4_K_M, short contexts, best of 4 rounds, 8 / 12 threads: N=8 1,003 / 1,214 tok/s in aggregate (3.1x /
3.3x of single-sequence decode at 326 / 366; was 758 / 897), N=16 1,091 / 1,449; Q8_0 GGUF N=8 976 / 1,107
(4.6x / 5.0x). Long contexts are KV-bandwidth bound (2,000 positions: ~350-480 tok/s for any N >= 8).
Continuous batching (Step 10b): `runq serve` runs every request in one batched pass per step (up to `--parallel`
16; prompts prefilled in 64-token chunks next to decoding requests), outputs bit-identical to each request
alone: 0.6B, 12 threads, 128-token replies, 16 clients ~1,185 / 1,223 tok/s in aggregate (Q4_K_M / Q8_0; one at
a time before: ~300 / ~200), time to first token ~0.1 s (was 3.5-5.7 s); LiteLLM-judge-like load 1.3-1.4x
(prefill-bound); streams keep flowing (<= ~70-90 ms gaps) while a 2,000-token prompt is prefilled.

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

## Step 6b-1: GGUF Q4_K_M, correct (loader, repacked layouts, generic kernels, tests)
`models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf` (unsloth, imatrix, 397 MB) works like the Q8_0 files:
chat, generate, `-m logits`, `-m tokenize`. Generic C kernels in every build for now (speed: Step 6b-2).
- [x] Tensor types (gguf-py): Q4_K for attn_q / attn_k / attn_output / ffn_gate / ffn_up in all 28
  layers and attn_v / ffn_down in layers 3 4 6 7 9 10 12 13 15 16 18 19 21 22; Q6_K for attn_v /
  ffn_down in layers 0 1 2 5 8 11 14 17 20 23-27 and `token_embd` (= the tied classifier); norms F32.
  381.7M Q4_K + 214.3M Q6_K values. Block formats checked against ggml-common.h / ggml-quants.c
  (written out in `src/kquants.h`'s header comment, incl. Q4_K's nibble order and Q6_K's ql / qh
  interleaving)
- [x] Loader: type per tensor (`gguf_matrices`): every matrix role may be Q8_0, Q4_K or Q6_K (attn_v /
  ffn_down mix Q4_K and Q6_K across layers, and the fused q / k / v GEMM mixes them within a layer:
  `matmul_batch_n` sends the K-quant jobs to `kq_gemm_n`, the Q8_0 ones as before), norms F32, any
  other type a clean error naming tensor and type; K-quants need rows % 16 and columns % 256 (block
  sizes already checked by the reader); fp16 d / dmin inf / NaN rejected at load (every block, in the
  parallel conversion). The Q8_0 / .bin paths are byte-identical (logits files of the previous
  binaries, both builds, -b 1 / 7 / 64, .bin and Q8_0 GGUF; all existing goldens unchanged)
- [x] Repacked layouts Q4P / Q6P (`src/kquants.h`, built at load by `kq_pack_blocks` in the parallel
  pread loop of `prepare_weights`, both builds): 16 rows interleaved per 256-column super-block, so a
  zmm's 16 dword lanes hold 16 rows' 4-value chunks once unpacked (vpdpbusd u8 operand), the 4 / 6-bit
  data kept packed: Q4P 2368 bytes per 16 rows x 256 (fp16 d / dmin, 6-bit sc / m unpacked to bytes,
  per sub-block 4 x 64 bytes of nibbles: low = step 2c, high = step 2c + 1), Q6P 3392 bytes (fp32 d:
  fills what would be padding, int8 scales, ql nibbles as Q4P, qh: 4 crumbs per byte = the 4 steps of
  a sub-block). 148 / 212 bytes per row and super-block vs 144 / 210 in the file (+2.8% / +1.0%; the
  unpacked sc / m bytes avoid decoding get_scale_min_k4 across lanes). Bytes per decoded token (all
  matmul weights incl. the classifier): **398.1 MB** vs 390.5 MB of tensor data in the file (+2.0%)
  and 670.5 MB for the Q8_0 GGUF (Q8P): 0.59x. Load 0.28 -> 0.11 s once the repacking wrote every byte
  once from the file bytes (was: per-value decode + read-modify-write); peak RSS 418 MB (Q8_0 GGUF 667)
- [x] Arithmetic (defined once, `src/kquants.h` header comment; the generic `q4p_dot` / `q6p_dot`
  follow it, compiled without contraction / reassociation, and 6b-2's AVX-512 kernels must too):
  activations int8 per 32 (the same quantized row for every matrix it feeds), plus per group
  xf = xs * (float) sum(xq) (`kq_act_prep`). Per output row, groups in order, every op rounded:
  Q4_K: isum = sum q * xq (int32 exact), ds = d * sc, dm = dmin * m (exact in fp32),
  val = val + ((ds * xs) * (float) isum - dm * xf); Q6_K: t_j = sum q * xq - 32 * sum xq per 16
  (int32 exact), I = sc_j0 * t_j0 + sc_j1 * t_j1 (exact, |I| < 2^24), val = val + (d * xs) * (float) I.
  All float operands are per row (lanes) or per group (broadcast): no horizontal operation. Q4_K's
  mins and Q6_K's -32 go through the activation sums exactly
- [x] Activation grouping: groups of 32 for Q6_K too (two weight scales per group, combined in int32):
  the fused q / k / v with mixed types needs one quantization of the row, and 16-groups would double
  the float folds; no accuracy cost measurable at this level (runq vs the Q4-weight reference below
  is as close as Q8_0 vs HF)
- [x] Embedding rows (Q6_K) dequantized at lookup from the file's blocks with ggml's expressions
  (`kq_dequant_row`; d * sc and (d * sc) * q are exact in fp32, so the values are bit-identical to
  ggml / gguf-py whatever the compiler contracts)
- [x] Dequantization check (`-m dequant -i NAME:ROW0:N -o FILE`, new hook: rows from the blocks and
  from the repacked layout): `tests/make_dequant_golden.py` (needs `pip install gguf`) wrote 2 rows of
  blk.0.attn_q, blk.3.ffn_down (Q4_K, 1024 / 3072 columns), blk.0.attn_v, blk.27.ffn_down, token_embd
  (Q6_K) by gguf-py (43 KB); all **bit-identical**, both sources
- [x] Accuracy (`tests/run_tests.sh -v --only q4`; -b 64 == -b 1). The Q4_K_M weights alone are ~25x
  the Q8_0 error: an fp32 HF model with the GGUF's weights dequantized by gguf-py ("Q4 reference",
  new golden `tests/golden/logits_q4`, `make_golden.py --q4ref`) is as far from HF (pooled cos 0.98459,
  top-1 67/84) as runq. runq vs that reference is at the Q8_0-vs-HF level (int8 activations only):

  | prompt | n | vs HF: cos pos 0 | cos mean | cos min | top-1 | top-5 | vs Q4 ref: cos pos 0 | cos mean | cos min | top-1 | top-5 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | en | 15 | 0.3609 | 0.96701 | 0.93967 | 13 | 0.720 | 0.9981 | 0.99883 | 0.99677 | 14 | 0.987 |
  | code | 24 | 0.3609 | 0.98676 | 0.96326 | 18 | 0.842 | 0.9985 | 0.99974 | 0.99943 | 23 | 0.983 |
  | zh | 15 | 0.4518 | 0.99174 | 0.98791 | 14 | 0.760 | 0.9994 | 0.99980 | 0.99971 | 14 | 0.947 |
  | chat | 15 | 0.4595 | 0.98174 | 0.94176 | 11 | 0.733 | 0.9990 | 0.99949 | 0.99784 | 15 | 0.947 |
  | count | 15 | 0.3549 | 0.99109 | 0.98384 | 13 | 0.840 | 0.9977 | 0.99973 | 0.99948 | 15 | 0.973 |
  | pooled | 84 | | 0.98402 (1-cos 1.6e-2) | | 69 | | | 0.99954 (1-cos 4.6e-4) | | 81 | |

  (Q8_0 GGUF vs HF: pooled 1-cos 6.1e-4, top-1 79.) Position 0 (the massive-activation token) is
  where 4-bit weights hurt most (cos 0.36-0.46 vs HF, the reference too). Thresholds: vs HF loose
  sanity bounds, ~2x the baseline error (cos mean >= 0.94, every pos >= 0.85, pos 0 >= 0.10, top-1 /
  top-5 >= 0.5, pooled cos >= 0.97, top-1 >= 0.65); vs the reference as for Q8_0 vs HF (~3-5x:
  cos mean >= 0.995, pos >= 0.98, pos 0 >= 0.99, top-1 / top-5 >= 0.75, pooled cos >= 0.998, top-1 >= 0.9)
- [x] Calibration by breakage (scratch builds, generic kernels; `tests/run_tests.sh --only q4`): all
  13 structural breakages fail every q4 logits check (14 of 14 batched + the sequential one): wrong
  nibble order, Q4_K scale / min swapped, wrong Q4_K sub-block scale (j ^ 1), Q4_K min term dropped,
  Q4_K min from the neighbour group's xf, one value per Q4_K sub-block dropped, Q6_K without the -32,
  Q6_K both halves with the first sub-block's scale, Q6_K qh crumbs reversed, Q6_K d of the next row,
  one value per Q6_K sub-block dropped (the mildest: pooled cos vs HF 0.719, top-1 40/84), plus two
  in the shared decoders (get_scale_min_k4's high bits, Q6_K qh shift) which also fail 2 resp. 3
  `q4_dequant` cases. All others: pooled cos vs HF <= 0.14, top-1 0/84. Precision-level changes pass the
  logits checks (vs the reference: d * sc truncated to bf16 in the Q4_K fold 0.99943, Q6_K d to bf16
  0.99954, Q6_K -31 instead of -32 in 1 of 16 sub-blocks 0.99921; baseline 0.99954); the first two
  flip `greedy/q4_generate`; in 6b-2, `q4_kernels/*` (bitwise) catches any such deviation of a kernel
- [x] Perplexity (`scripts/ppl.py bench/ppl_beagle_ch1.txt -t 8 --runq-model <Q4_K_M>`): runq Q4_K_M
  full **28.911** (+4.86% vs HF fp32 27.571; mynah-slm Q4_K_M 28.864, +4.69%), win512 **34.190**
  (+6.90% vs 31.984; mynah-slm 34.185); runq Q8_0 GGUF 27.623 / 32.040. The int8 activations add
  ~0.16% on top of the 4-bit weights here (vs mynah-slm's fp32 activations), as for Q8_0
- [x] mynah-slm cross-check (`mynah-slm run -m <Q4_K_M> --temp 0 -n 64 -t 8 --think off` vs
  `runq <Q4_K_M> -t 0 -n 64`; `--raw` vs `-m generate`): chat, 5 of 6 prompts identical over 64 tokens
  (capital of Australia: both "**Sydney**", the Q4_K_M model's answer; hash table, primes > 100,
  translation, haiku), "Why is the sky blue?" diverges after 36 characters; raw generation diverges
  after ~12 ("The three laws of thermodynamics are") and ~33 ("Once upon a time...") characters
- [x] Speed with the generic kernels, for reference (load 5-7, best of 3, 451-token prompt, decode
  128 tokens): prefill 168 / 233 tok/s at 8 / 12 threads (Q8_0 GGUF, VNNI: 1,870 / 2,438), decode
  55-58 / 75 tok/s (Q8_0 GGUF: 157-177 / 151-200). Profile (451 tokens, 8 threads): GEMM qkv 33%,
  w1 + w3 28%, w2 25%, wo 10%, the rest 3%; the generic Q6_K kernel takes ~2x the time per weight of
  the Q4_K one (qkv with Q6_K attn_v in 14 layers is slower than w1 + w3 with 1.5x the weights)
- [x] Robustness: ASan+UBSan builds (`-O1 -g -fsanitize=address,undefined`, AVX-512 and generic) on
  Q4_K_M: generate at -b 1 / 7 / 512, chat (-r 1, -y, multi-turn, context full at -c 96), logits,
  dequant: no reports. Scratch harness, 108 corrupted files x (tokenize, generate), both sanitizer
  builds: header (version, counts), tensor table (types Q5_K / F16 / unknown / Q8_0 / Q4_K <-> Q6_K
  for matrices and norms, ne0 not % 256 / wrong / huge, ne1 +-16 / huge, n_dims, offsets huge /
  misaligned / near the end / 0), data (Q4_K d / dmin inf, -inf, NaN at the first / last / a middle
  block; Q6_K d inf / NaN incl. in the classifier; scale bytes 0xff / int8 -128; d = 65504 with all
  quants 0xff), 15 random header bit flips, 10 random data byte flips, 17 truncations from 0 to
  size - 1: 132 clean errors, 84 normal runs, 0 crashes or sanitizer reports per build
- [x] Tests (`make test` 265 checks, ~30-33 s on the busy box, load 3-6; was 238 / ~25 s): `q4` group
  (~4 s: dequant vs gguf-py, logits vs HF and vs the Q4 reference at -b 64 on 5 prompts and -b 1 on en,
  -b 7 / 64 vs -b 1, threads 8 vs 3, AVX-512 build vs generic on -b 1 / 7) and greedy goldens
  `q4_chat`, `q4_generate` (24 tokens each); skipped if the file is missing (`Q4=` to skip)

## Step 6b-2: AVX-512 kernels for Q4_K / Q6_K (speed)
AVX-512 VNNI GEMV and GEMM on the Q4P / Q6P layouts (`src/kernels_avx512.h`, after the Q8 kernels),
bit-identical to the generic `q4p_dot` / `q6p_dot` (the definition in `src/kquants.h`). Before (generic
kernels, Step 6b-1): prefill 451 tokens 168 / 233 tok/s at 8 / 12 threads, decode 55-58 / 75.
- [x] GEMV (decode, `kq4_gemv_rows` / `kq6_gemv_rows`): one 16-row block at a time, streamed, the
  4 / 6-bit data unpacked in registers. Q4_K: per sub-block 4 loads of 64 bytes, each 2 `vpdpbusd`
  steps (`and 0x0f`; `srli_epi16 4` + `and`), two accumulator chains; ds = cvtph2ps(d) * cvt(zx(sc)),
  dm likewise, then the fold (6 float ops). Q6_K: per sub-block 2 ql loads + 1 qh load, q of step t =
  ternary-logic select (low nibble from ql, bits 4-5 from qh shifted by 4 - 2 t on 16-bit lanes) & 0x3f,
  4 `vpdpbusd` from t_j = c16 (-32 * the 16 activation values' sum, `kq512_c16`, precomputed per row in
  `matvec_prep`: RunState `xq_c16` / `hq_c16`), I += mullo(t_j, sc_j) (vpmovsxbd), fold (4 ops)
- [x] Q6P layout change on the way (same size, same arithmetic): each sub-block's qh chunk now follows
  its 2 ql chunks (320 + 192 j), one sequential stream. With qh in a separate region 2 KB after the ql
  chunks the Q6P GEMV read memory at 13.4 GB/s on one thread vs 31-33 for Q4P / Q8P and a plain read
  loop (in L2: 54 GB/s, so not compute): the hardware prefetchers lose two interleaved short streams.
  Software prefetch of the next super-block only got 20-22 GB/s. Packer, generic kernel, dequant hook
  (`kq_packed_row`) and the AVX-512 kernels changed together; logits unchanged, `q4_dequant/*` pass
- [x] GEMV bandwidth (microbenchmark over 256 MB, the box shared, best of 5; a plain read loop the
  same minute as the reference): 1 thread Q4P 31-34, Q6P 28-31, Q8P 32, read 29-35 GB/s; 8-12 threads
  all kernels within the run-to-run spread of the read loop (72-130 GB/s that day). In decode (below):
  ~435 MB per token for Q4_K_M (398.1 MB weights + ~37 MB KV cache at ~161 positions on average) ->
  86 / 123 / 131 GB/s effective at 4 / 8 / 12 threads (best runs); Q8_0 GGUF ~707 MB per token: 98 /
  127 / 141 GB/s. So both are at the bandwidth limit and the speed ratio is about the byte ratio
  (1.43-1.58x measured, 1.63x in bytes)
- [x] GEMM (prefill, `kq512_gemm_n`): work items (2-block = 32-row tile, chunk of activation rows) as
  `q8p_gemm_n` (static, `q8p_nb` chunk sizes); q / k / v and w1 / w3 with mixed Q4_K / Q6_K types in
  one parallel loop (`matmul_batch_n` sends the K-quant jobs there, Q8_0 jobs, if any, to `q8p_gemm_n`).
  Each thread unpacks its weight tile once into a per-thread scratch ("KQS": per block and group of 32
  columns 8 steps x 64 bytes of u8 q in the Q8P step layout; Q4_K + ds, dm as fp32 = 640 bytes, Q6_K +
  sc_j0, sc_j1 as int32 + d = 704 bytes; 40-135 KB per tile) and reuses it for all its chunks of that
  tile (static schedule: a thread's items are consecutive, so each tile is unpacked once per thread);
  the register tile then reads u8 weights like the Q8 GEMM, no unpacking in the inner loop. Register
  tiles: Q4_K 2 blocks x 6 rows (12 int32 + 12 float accumulators), Q6_K 2 x 4 (t_j, I and the floats
  are live together: 3 x 8). xf and c16 per activation row in the prologue (`kq_act_prep`, `kq512_c16`)
- [x] Unpack strategy measured (451 tokens, 8 threads, `-DRUNQ_PROFILE`, min of 3, twice): scratch tile
  w1+w3 74-75 ms, wo 26; Q4_K unpacked in registers per 6-row tile instead (`and`, `srli` + `and` per
  64-byte load, ds / dm recomputed per tile): w1+w3 80-81, wo 27 (~8% slower: 6 extra ALU ops per 24
  `vpdpbusd`). Not kept. Tile widths Q4 C = 4 / 5 / 7 / 8 and Q6 C = 2 / 3 / 5 / 6: all within noise
  (+-5%), kept 6 / 4. Thread balance (instrumented scratch build): per-thread busy time within ~5% in
  every GEMM, prologue < 1 ms per call type
- [x] GEMM throughput (8 threads, 451 tokens, profile below): 1.8 (qkv, with Q6_K v in half the layers)
  to 2.15 TOPS (w2; w1+w3 2.1), 63-74% of the measured `vpdpbusd` peak (2.9 TOPS); Q8_0 GGUF 2.2-2.4.
  The difference is the fold: 6 (Q4_K) /
  ~7 (Q6_K: 2 mullo + add + 4) ops per 8 `vpdpbusd` vs 4 for Q8_0 group 32. GEMMs Q4_K_M / Q8_0 GGUF:
  196 / 168 ms at 451 tokens (1.17x), 633 / 545 ms at 1,500 tokens (1.16x)
- [x] Bit-identity: integer sums exact (any order), the folds per (block, row, group) as defined, with
  explicit-rounding intrinsics (`kq4_fold`, `kq6_fold`). AVX-512 == generic byte for byte on every
  checked case (GEMV, GEMM at -b 7 / 64 / 512, 1-row chunks, 3 / 8 threads, a 119-token prompt in 17
  chunks, the 187-token `long` prompt at -b 1 / 7 / 64 / 512); logits byte-identical to the 6b-1 binaries (which ran the generic kernels in both builds);
  greedy goldens `q4_*` and all others unchanged. Deliberate breakage (scratch builds): an FMA in the
  Q4_K fold (`fmsub` for (ds xs) isum - b) or in the Q6_K fold (`fmadd` into val): max |diff| 1.58 /
  1.52, all four `q4_kernels/*` fail (and `greedy/q4_generate`); `q4_batched/*` / `q4_threads/*` cannot
  see it (both sides use the same kernel)
- [x] Tests (`make test` 269 checks, ~30-33 s on the busy box, load 4-6; was 265 / ~30 s): `q4_kernels/*`
  en -b 1 (GEMV incl. the Q6_K classifier), en -b 7 (tiles 6 + 1 / 4 + 3, a 1-row chunk), all -b 7 (13
  chunks), all -b 64 (2 chunks, tiles 6 + 4 / 4 + 2, the all-rows classifier GEMM) vs the generic build,
  which runs in the background at 4 threads (~3 s of generic runs: ~1 s per 86 tokens with all-rows
  logits, mostly the classifier); `q4_batched` + all -b 7 / 64; `q4_threads` on all -b 7 / 64 (8 vs 3:
  other chunk sizes and partitions). The q4 group: 2.9 -> ~5 s with these (the AVX-512 side alone ~4 s,
  half of it process startup + repacking). The generic kernels' `KQ_NC` 4 -> 8 / 16 (unpack amortized
  over more rows) was only 7-10% faster: not changed
- [x] ASan+UBSan (`-O1 -g -fsanitize=address,undefined`), AVX-512 and generic builds, Q4_K_M: generate
  at -b 1 / 7 / 512 on a 119-token prompt, logits -b 1 / 7 / 512 (3 and 8 threads), chat multi-turn at
  -b 1 / 512 and -c 160 -r 1 -y -b 7 (context full), dequant: no reports; the sanitizer builds agree
  with each other and across -b / threads byte for byte
- [x] Speed (2026-10-25, same session, load 4-8 (shared box, other tenants' memory traffic); prefill
  `scripts/bench_prefill.sh -m <model> -l "450 1500" -b 512 -t "8 12"`, best of 3; decode
  `scripts/bench_qwen3c.sh <model> 4 8 12` (256 tokens, 3 runs) plus 7 interleaved rounds of the same
  runs (the 3 models alternated), best of the 7-10 runs per cell):

  | | .bin Q8_0 | GGUF Q8_0 | GGUF Q4_K_M |
  |---|---|---|---|
  | prefill 451 tokens, 8 / 12 threads | 1,961 / 2,693 tok/s | 1,874 / 2,540 | 1,670 / 2,142 (0.89x / 0.84x of GGUF Q8_0) |
  | prefill 1,500 tokens, 8 / 12 threads | 1,542 / 2,029 | 1,448 / 2,028 | 1,317 / 1,843 (0.91x / 0.91x) |
  | decode 4 / 8 / 12 threads (best) | 136 / 170 / 232 tok/s | 138 / 179 / 199 | 198 / 283 / 302 (1.43x / 1.58x / 1.52x) |
  | decode, effective GB/s (4 / 8 / 12) | 91 / 114 / 155 | 98 / 127 / 141 | 86 / 123 / 131 |
  | process wall `-n 1 -i hi`, 12 / 8 threads (median of 9) | 0.133 / 0.148 s | 0.147 / 0.175 | 0.120 / 0.139 |
  | peak RSS | 620-632 MB | 667-673 MB | 408-419 MB |

  Decode run to run: +-30% (e.g. Q4_K_M at 12 threads 188-302 tok/s); within one interleaved round
  Q4_K_M / Q8_0 GGUF was 1.1-2.0x, median ~1.5x. GB/s: bytes per token = packed weights (Q4_K_M 398.1
  MB, Q8_0 GGUF 670.5, .bin 633) + ~37 MB of KV cache (33-token prompt + 256 generated: ~161 positions
  on average x 229 KB). Prefill is ~10-16% slower than Q8_0 GGUF (the GEMM
  folds, above), decode ~1.5x faster (bytes), startup ~25-35 ms faster (fewer bytes to repack), RSS -38%
- [x] Profile (`-DRUNQ_PROFILE`, 8 threads, -b 512, min of 3 per phase): 451 tokens 274 ms: GEMMs 71.5%
  (qkv 58 ms 21.2%, wo 26 9.5%, w1+w3 75 27.4%, w2 37 13.5%), attention+quant 45 16.4%, norm+quant+resid
  16 5.8%, swiglu+quant 11 4.0%, qk-norm+rope+kv 5 1.8%, classifier 1; 1,500 tokens 1.13 s: GEMMs 56%
  (qkv 182, wo 86, w1+w3 241, w2 124 ms), attention 386 ms 34%, rest 10%. 12 threads: 196 ms (GEMMs 139,
  attention 32) / 846 ms (GEMMs 481, attention 272). (Q8_0 GGUF, 8 threads: 245 ms, GEMMs 168 / 1.04 s,
  GEMMs 545)

## Step 7a: threading (barriers, decode work split)
Before (Step 6b-2): decode = one OpenMP parallel region per token with 7 barriers per layer (198 per
token + the fork / join); prefill = ~9 OpenMP parallel regions and ~10 more barriers per layer; 16
threads on the 16 vCPUs collapsed (decode ~2-20 tok/s, prefill ~100-600 at 451 tokens; every barrier
waits for a preempted thread while the others spin on CPUs it could use); decode attention had 8 work
items (one per kv head), idle threads at 12+ and slow at long contexts.
- [x] Phase synchronization instead of barriers (`src/threads.h`): a job (one decode step, one prefill
  chunk) is a sequence of phases of work items; a phase ends when its *work* is done, not when every
  thread has arrived: the thread finishing the last item runs the phase's serial epilogue (if any) and
  publishes the phase count, the others wait for it. A thread preempted between phases holds nothing
  up (it finds the phases done and catches up); only one preempted inside an item does. Waiting: spin
  with `pause` for 4,000 rounds (~110 us on Zen 4, ~28 ns per round), then sleep on a futex (Linux;
  elsewhere yield); the publisher wakes only if someone sleeps (registered first, seq_cst on both
  sides, so no lost wake-up). Spin 1,000 / 4,000 / 16,000 rounds: within noise at 4-16 threads for
  decode (best and median of 6), 4,000 ~5% faster than 1,000 for prefill at 8-12 threads; 100 (~3 us)
  ~30% slower (sleeps and wake-ups every phase). Work distribution, measured: GEMV / GEMM phases give
  each thread its own contiguous range of items (long weight streams, as a static schedule; 64-bit
  range words, CAS, epoch-tagged so nothing is reset per job), taken from the front a chunk at a time,
  then single items stolen from the back of the others' ranges (a preempted thread's work gets done,
  and the end of a phase balances item by item). Shared-counter claims of single GEMM tiles were ~15%
  slower in prefill (shorter streams) and of 1-4 blocks ~10-20% slower in decode at 4-12 threads.
  Phases whose items wait on other items (decode qkv -> attention) claim in order from one shared
  counter: with ranges and stealing all threads can end up waiting in stolen items whose dependencies
  sit unclaimed in their own ranges (an intermediate build deadlocked so at 4 threads); the prefill
  attention too (items of unequal cost, longest first). `RUNQ_SPIN=decode[,prefill]`,
  `RUNQ_SYNC=static` (no stealing, i.e. a barrier per phase; OpenMP region builds only) for tuning
- [x] Thread pool instead of `#pragma omp parallel` per job: libgomp's fork / join is itself a barrier
  every thread must reach; with the phases inside an OpenMP region decode at 16 threads got ~70-80
  tok/s vs ~150-170 on the pool. Persistent pthread workers + the caller; job protocol: fn / ctx, open
  the job (state), bump gen (futex word); a worker enters only while the job is open, the caller closes
  it after its own share and waits only for the workers inside (one that has not started yet skips the
  job: its range is stolen). Found by measuring: gen bumped *before* the state was opened let workers
  spinning on gen see a closed job and skip it, every other token with most of the threads (decode
  ~20-40% slower at 4-12 threads, every second token 2-3x as long; per-token timing histograms): fixed
  in `0380a83`. Weight loading (`prepare_weights`) and the GGUF merge lookups run on the pool too, so
  no OpenMP threads exist at run time (`-DRUNQ_NO_POOL` and Windows: the OpenMP region; a build
  without `-fopenmp` still runs threaded on the pool). Thread count: `rq_set_threads`
  (`OMP_NUM_THREADS` or the default)
- [x] Fewer sync points. Decode: 4 phases per layer instead of 7 barriers: [q, k, v per head + QK-norm
  + RoPE + cache store; attention per (kv head, block)] in one phase, each attention item waiting on a
  dependency counter for the 4 items of its kv head (the attention output is quantized into its own
  buffer `aq`, as `xq` may still be read); [wo; epilogue: residual + rmsnorm + quantize]; [w1 + w3 +
  SwiGLU + quantize]; [w2; epilogue: residual + the next layer's rmsnorm + quantize]; layer 0's norm
  before the job. **113 sync points per token instead of 198** (4 x 28 + the classifier; +1 every 16
  tokens for the KV tile pre-fault, now a phase of the job). Prefill: one job per chunk, 9 phases per
  layer (8 in layer 0) instead of ~19 OpenMP sync points (9 regions + 10 barriers); the GEMM activation
  prep (VNNI corrections, K-quant sums) is computed per row where the row is quantized, the attention
  output quantized by the last kv head of each row block (dependency counter), LOGITS_LAST / ALL in
  the job. All outputs computed by the same functions per item: logits byte-identical to Step 6b-2
- [x] Attention over blocks of ATT_P = 256 positions (`src/kernels_avx512.h` header, option (a)): per
  block the max, the exponentials, their sum (16 lanes + the fixed tree) and the weighted V; with more
  than one block M = max m_b, c_b = exp_att(m_b - M), L = fma chain of c_b * l_b, out = (fma chain of
  c_b * o_b) * (1 / L), blocks in order (`att_combine`). One block (<= 256 positions) is exactly the Step
  3b definition: every shorter prompt gives the same bits as before (all HF and greedy goldens
  unchanged). The block size is a constant, so the arithmetic depends on neither the thread count nor
  the chunk size: decode, prefill, the generic C version and all thread counts stay bit-identical
  (tests below). Decode splits a long context: one work item per (kv head, block) (3,000 positions: 96
  items instead of 8), the last of a head's items to finish combines (flash-decoding). Prefill runs
  all blocks of its vectors in one item. Accuracy vs a double-precision attention (scratch build;
  1,488-token prompt, mean 1-cos of the logits): positions >= 256 2.81e-4 vs 2.77e-4 with the Step 3b
  definition (the ranges [256, 512) / [512, 1024) / [1024, 1488): 4.06 / 2.86 / 2.07e-4 vs 3.49 / 3.04
  / 2.08e-4), ATT_P 128 / 512 the same within ~5%: equally good. Against HF fp32 past 256 (new golden):
  cos mean 0.99927, min 0.99791, top-1 13/13. Prefill cost: with one kernel call per block the
  attention at 3,000 tokens was ~27% slower (each block restarted the 4 strided V passes: prefetcher
  ramp-ups, ~5% single-threaded but 27% with 8 threads competing for L3 / DRAM); with the passes
  running over all blocks (accumulators restart per block, one long stream) ~10% (kernel
  microbenchmark 3-5%), i.e. ~4-6% of a 1,500-3,000-token prefill, nothing below 256 tokens. ATT_P 512:
  the same prefill cost, fewer decode items at mid contexts: 256 kept
- [x] Results (2026-10-25, the shared 16-vCPU VM, ~2.5-4.5 other CPUs busy, measured between runs;
  generate mode, greedy, interleaved with the Step 6b-2 binary `0cea116`, best / median of 8 (decode)
  or best of 5-6 (prefill); `/tmp` scripts like `scripts/bench_qwen3c.sh`):

  | decode tok/s, best / median | 4 threads | 8 | 12 | 14 | 16 |
  |---|---|---|---|---|---|
  | Q8_0 GGUF before | 145 / 101 | 198 / 137 | 179 / 118 | 136 / 121 | 1.3-23 (collapse) |
  | Q8_0 GGUF now | 142 / 108 | 192 / 168 | 210 / 182 | 206 / 171 | 141-172 / 116-160 |
  | Q4_K_M before | 171 / 154 | 237 / 212 | 296 / 234 | 298 / 195 | 1.5-28 |
  | Q4_K_M now | 193 / 169 | 277 / 249 | 351 / 265 | 319 / 276 | 218 / 196 |
  | Q8_0 .bin before / now (6 runs) | | 145 / 124 -> 199 / 180 | 207 / 153 -> 252 / 188 | 178 / 123 -> 218 / 185 | |

  | prefill tok/s (best) | 8 threads | 12 | 14 | 16 |
  |---|---|---|---|---|
  | Q8_0 GGUF, 439 tokens, before / now | 1,860 / 1,880 | 2,350 / 2,370-2,420 | 2,456 / 2,580 | 115-631 / 2,290-2,500 |
  | Q8_0 GGUF, 1,488 tokens, before / now | 1,477 / 1,384 | 1,937 / 1,853 | 2,000 / 2,019 | 308 / 1,922 |
  | Q4_K_M, 439 tokens, before / now | 1,681 / 1,688 | 2,297 / 2,290 | | 214 / 2,162 |

  | decode after a long prompt (Q8_0 GGUF, 128 tokens), best / median | 8 threads | 12 | 16 (now) |
  |---|---|---|---|
  | 488 tokens, before / now | 141 / 115 -> 148 / 124 | 128 / 103 -> 162 / 143 | 125 / 115 |
  | 1,488 tokens, before / now | 109 / 87 -> 119 / 111 | 105 / 100 -> 131 / 126 | 112 / 106 |
  | 2,988 tokens, before / now | 85 / 80 -> 86 / 80 | 71 / 66 -> 102 / 101 | 84 / 83 |

  With 4 extra busy threads (a scratch CPU hog; ~6.8 CPUs busy besides runq): 8 threads 150 / 133 ->
  169 / 162 tok/s, 12 threads ~11 -> 143 / 120, 16 threads -> 118 / 104. Decode is memory-bound: 12-14
  threads saturate the bandwidth and 16 only add contention with the other processes; 4 threads are
  bandwidth-per-thread limited (best equal, median +7%). The attention split helps from ~1,000
  positions on at 12+ threads (8 items were 8 threads' worth of work); at 8 threads and 3,000 tokens
  the 0.7 GB of KV cache per token keeps it at the bandwidth limit either way (7b: bf16 cache).
  Prefill: within ~+-5% at 8-14 threads (1,500 tokens -4..-6%: the attention blocks), no collapse at 16
- [x] Default threads (no `OMP_NUM_THREADS`): nproc - nproc/8 = 14 on the 16-vCPU VM (was min(12, nproc
  - nproc/4) = 12): decode 12 ~ 14 > 16, prefill 14 > 16 > 12 (2,580 / 2,290-2,500 / 2,370-2,420 at 439
  tokens, 2,019 / 1,922 / 1,853 at 1,488). All 16 work, but leave room for the other processes
- [x] Tests (`make test` 278 checks, ~23-26 s on the busy box, load 4-12; was 269 / ~27-33 s): new
  `logits_long/long2` (HF fp32 golden at every 4th position 248..296 of a 297-token prompt,
  `make_golden.py --long`, 111 KB), `batched/long2_b64` / `_b512` vs -b 1, `threads/bitwise_long2_b64_t1`
  / `_t3` / `_t16`, `kernels/long2_b64`, `chat/long_context` (411-token conversation decoded at 16
  threads vs the batched prefill, bitwise) and `chat/long_context_generic` (the generic build's run,
  bitwise). Breakage (scratch builds): the combination without the rescale fails `logits_long` (cos mean
  0.397) and `chat/long_context`; another rounding of the l_b sum in the generic build only fails
  `kernels/long2_b64` (max |diff| 0.54) and `chat/long_context_generic`. Faster: every `-m logits` run
  of the selected groups is started up front, 3 at a time (`prefetch_all`), the generic build's at 4
  threads in the background; greedy and chat cases 3 at a time
- [x] Sanitizers (scratch builds): ASan + UBSan (`-O1 -g -fsanitize=address,undefined`, AVX-512 and
  generic) and ThreadSanitizer (`-O1 -g -fsanitize=thread` without `-fopenmp`: the pool runs loading,
  prefill and decode, all instrumented; libgomp is not, hence no OpenMP build): 17 runs each (Q8_0 /
  Q4_K_M GGUF generate + logits at 3 / 8 / 16 threads on a 274-token prompt, -b 1 / 7 / 13 / 64 / 512,
  the .bin, chat with context full, chat -r 1 -t 0.8 on Q4_K_M): no reports; the TSan build's -b 1 and
  -b 7 logits agree bit for bit
- Notes for Step 7b: bf16 K/V cache (`vdpbf16ps`; halves the 0.7 GB per token at 3,000 positions that
  keeps 8-thread decode at the bandwidth limit; numerics change, the logits tests decide); prefill rest
  (norm / quantize / SwiGLU ~14% at 451 tokens, SwiGLU's `expf` ~4%); the prefill attention could use
  the blocks for balance (one item per (row block, kv head, block range)) and the remaining ~10% block
  overhead; decode at 4 threads is per-thread-bandwidth bound

## Step 7b: 16-bit KV cache, long-context attention, prefill rest
Before (Step 7a): fp32 KV cache, 229 KB per position (28 layers x K, V x 1,024 floats): after 3,000 tokens
decode reads ~0.7 GB of KV per token, more than the Q4_K_M weights (0.4 GB): decode after a 2,988-token
prompt ~86 tok/s at 8 threads (Q8_0 GGUF) vs ~190 at short contexts; prefill at 1,500-3,000 tokens
40-57% attention; prefill "rest" ~12% at 451 tokens.
- [x] KV cache type `-k f32|bf16|f16` (option (i) of the plan: the stored values change, the attention
  arithmetic does not). Store: round to nearest even from the fp32 K / V rows (`f32_bf16`, `f32_f16`:
  integer / exact float operations in C, the same in every build; equal to `vcvtneps2bf16` /
  `vcvtps2ph` for every non-denormal input, checked over all 2^32 floats; f16 saturates at 65504
  instead of rounding to inf, measured |K| <= ~500, |V| <= ~100). Load: exact conversion, then the
  fp32 operations of the Step 3b / 7a definition. bf16 tiles pair dims (K: dims 2p, 2p + 1 of a
  position in one dword; V: dims 32c + l and 32c + 16 + l), so one 64-byte load gives two zmm of fp32
  by a shift and a mask; f16 tiles keep the fp32 layout and load with `vcvtph2ps`. The generic C
  attention converts each (kv head, block) to fp32 tiles once per call (`kv_block_f32`: `bf16_f32`,
  `fp16_to_f32`), then runs unchanged (~1.1x the fp32 time; converting per vector and tile was 1.8x).
  All invariants hold per type: batched == sequential, any thread count, AVX-512 == generic (tests
  below; also checked on a 1,500-token prompt at -b 1 / 7 / 64 / 512, 3 / 8 / 12 / 16 threads). -k f32
  is bit-identical to Step 7a (30 logits files: 3 models, -b 1 / 64 / 512, both builds)
- [x] Option (ii) (`vdpbf16ps`) not done: prefill attention is compute-bound (below), so 2x the dot
  product rate is the one big lever left, but it needs q and the softmax weights rounded to bf16 too
  (bf16 K alone already costs +2.6% KL, mostly from K: |K| ~500), and the instruction's rounding /
  denormal behaviour (two FMAs per pair, DAZ / FTZ, odd pair first per Intel's pseudocode) would have
  to be emulated in C and verified against this AMD core, with no guarantee for other CPUs
- [x] Accuracy, KL(HF fp32 || runq) per token (scratch script, Q8_0 GGUF, 13 pieces of ~3,500 tokens:
  Voyage of the Beagle ch. I, node.js `stream.md` / `fs.md` docs; 45.6k tokens, positions up to 3,500):
  f32 0.001745 mean, **f16 +0.2%** (7 pieces higher, 5 lower: noise), **bf16 +2.6%** (10 higher, 2 equal,
  1 lower), V only in bf16 +0.7%, K only in bf16 +4.4% (5 pieces), Step 7a's arithmetic change (the
  256-position blocks) +0.1% for scale. Top-1 agreement with HF 0.9758 / 0.9754 / 0.9761 (f32 / bf16 /
  f16, 24.5k tokens incl. the legal text). A legal text
  (AGPL-3.0, much repetition) is chaotic: any rounding change, f32 variants included, moves its pieces by
  +-1-10%, so it is left out of these numbers. Logits vs HF (`make test` thresholds unchanged, all met):

  | pooled cos (pos >= 1) / top-1 | f32 | bf16 | f16 |
  |---|---|---|---|
  | .bin Q8_0, 5 prompts (-b 64) | 0.99910 / 76 | 0.99900 / 76 | 0.99905 / 75 |
  | .bin, long2 positions 248-296: cos mean, min | 0.99927, 0.99791 | 0.99909, 0.99525 | 0.99921, 0.99695 |
  | GGUF Q8_0, 5 prompts | 0.99939 / 79 | 0.99940 / 79 | 0.99937 / 78 |
  | Q4_K_M vs HF / vs the Q4 reference | 0.98402 / 0.99954 | 0.98440 / 0.99960 | 0.98412 / 0.99959 |

- [x] Perplexity (`scripts/ppl.py bench/ppl_beagle_ch1.txt -t 8 --runq-model ...`, full / 512-windows;
  f32 = Step 7a, which moved the .bin from 27.805 to 27.720 and the GGUFs by +0.03 / +0.12% vs Step 6):

  | | f32 | bf16 | f16 |
  |---|---|---|---|
  | Q8_0 GGUF | 27.630 / 32.058 | 27.547 / 32.018 (-0.30 / -0.12%) | 27.602 / 32.025 (-0.10 / -0.10%) |
  | Q4_K_M | 28.945 / 34.166 | 28.963 / 34.231 (+0.06 / +0.19%) | 28.947 / 34.198 (+0.01 / +0.09%) |
  | .bin Q8_0 | 27.720 / 32.208 | 27.788 / 32.197 (+0.25 / -0.03%) | 27.807 / 32.160 (+0.31 / -0.15%) |

  On 4,142 tokens any rounding change moves these by ~0.1-0.3% either way (the Step 7a blocks: -0.3%
  for the .bin), so they cannot resolve the KV types; paired per-token NLL over 49k tokens (Beagle +
  AGPL + stream.md, 14 pieces): bf16 vs f32 +0.11% +-0.08 (95%), one AGPL piece +1.7%
- [x] Default: **f16**. bf16 misses the "< 0.1%" bar (+0.11% perplexity on 49k tokens, +2.6% KL,
  consistently), f16 is within the noise of f32 everywhere, and both 16-bit types give the same memory
  and decode gains. f16 costs ~0-4% of prefill attention for `vcvtph2ps` (bf16's shift / mask ~0) and
  1-3% of a 1,500-3,000-token prefill. Greedy goldens: 7 texts flip with f16 (`chat_system`, `generate`,
  `gguf_generate`, `q4_generate`, the two seeded ones, the multi-turn transcript; all read fine, and
  the Q8_0 GGUF already answered "I'm Ada" in turn 2 with f32) and were regenerated; `-k f32` reproduces
  every old golden exactly (checked; `f32_generate` keeps one), `bf16_generate` is new
- [x] Memory: KV cache at `-c 4096` 0.94 GB (f32) -> 0.47 GB (bf16 / f16), address space, touched as
  used; default `-c` 0 = 40,960 positions: 9.4 -> 4.7 GB of address space (unchanged default: nothing is
  resident until used, and `-c` is still the right knob). Peak RSS (12 threads, 128 generated tokens):
  short prompt Q8_0 GGUF ~780 MB, Q4_K_M ~520 MB for every type (2 MB transparent huge pages of the first
  block per layer); after a 3,000-token prompt Q8_0 GGUF 1,373 (f32, = Step 7a) -> 1,150 MB (bf16, f16),
  Q4_K_M 1,228 -> 892 MB
- [x] KV cache layout: blocks of ATT_P = 256 positions, in a block each kv head's 16 tiles contiguous
  (were: tile-major, the 8 heads' tiles alternating). With 4-8 KB tiles, every tile of a head was a new
  page 32-64 KB further on, the hardware prefetchers restarted per tile. Measured first with one range
  per head (2 MB per head and layer at -c 4096): decode after 3,000 tokens (Q8_0 GGUF, 8 threads) f32 82
  -> 95, f16 101 -> 131 tok/s (best of 3, same session), prefill +3-5%; but its pre-fault took ~2x as
  long (8 ranges per layer, no huge pages: +8 ms at 451 tokens). Per block of 256 the positions of all
  heads are contiguous, pre-faulted as before, and the decode gain stays (f16, 3,000 tokens, 8 / 12
  threads: 112 / 149 vs 139 / 121 best, 110 / 136 vs 110 / 121 median: noise). kv_len is rounded up to
  whole blocks. Same arithmetic (bit-identical)
- [x] bf16 scores: converting both halves of a pair at once spilled registers (prefill attention +15%
  over fp32); loading the 64-byte pair per dim, as the fp32 loop does: -2%
- [x] Prefill attention at long prompts (8 threads, 1,500 tokens, -DRUNQ_PROFILE): compute-bound, not
  bandwidth-bound: 1 -> 8 threads 3.05 -> 0.438 s (6.96x), the same for bf16 (half the bytes); ~590
  GFLOPS (66% of the ~900 FMA peak), PV ~710 GFLOPS (79%, 42% of the time), scores + softmax ~500 (the
  exp only ~3%: a scratch build without it). The block overhead is ~4-5% now (ATT_P 4,096 scratch build
  at 3,000 tokens: 1.62-1.64 vs 1.69 s fp32; Step 7a's ~10% was measured before its V-pass rework). Rows
  per item: 12 rows (24 vectors, register tiles 24 x 1 / DW 1): 66 vs 48 ms at 451 tokens, 2.24 vs 1.58 s
  at 3,000: worse; 6 kept. Attention at 3,000 tokens: 1,674 -> 1,558 ms (f32, the layout), 1,654 (f16)
- [x] Prefill rest. `quantize()` was scalar: its int8 stores aliased `GS` and `qx` (reloaded per value,
  no vectorization), ~8 instructions per value, 0.6 us per 1,024 values; the same operations (the ones
  -Ofast made of upstream's expressions: scale = wmax * (float)(1/127), x * (127 / wmax), round half away
  as trunc(v + copysign(0.49999997, v))) with local pointers vectorize: 0.21 us; outputs byte-identical
  (30 logits files vs the Step 7a binaries, 1.2e9 random values incl. ties vs the old code). Fusing w1 +
  w3 + SwiGLU + quantize into one GEMM phase (epilogue per (32-column group, 64-row chunk) item): slower
  (79-81 ms vs 63-64 + 9-10 at 451 tokens: scattered partial-line writes of hq / scales / corrections
  across rows; bigger column groups unbalance the threads), reverted. The rest is memory traffic, not
  arithmetic: a 451-token norm phase ~0.3 ms vs ~60 us of compute per thread (the rows were written by
  other cores' GEMM tiles). SwiGLU's `expf` (libmvec): 1.1 of 1.5 us per 3,072-value row, ~0.75% of a
  451-token prefill: a polynomial would save ~0.4% and change the numerics: not done. 16-bit stores
  branch-free (vectorized V rows). Rest (norm + swiglu + rope / K/V store) at 451 / 1,500 / 3,000 tokens:
  28 / 92 / 181 -> 27 / 76 / 156 ms
- [x] Results (2026-10-25, the shared 16-vCPU VM, load 8-13 throughout (other tenants); chat mode, greedy,
  -n 128, `-c 4096`, interleaved with the Step 7a binary `dd47f07`, best / median of 5; prompts of 451 /
  1,500 / 3,000 tokens from `bench_prefill.sh -P`):

  | Q8_0 GGUF | prefill 451 / 1,500 / 3,000, 8 threads | 12 threads | decode after 451 / 1,500 / 3,000, 8 threads | 12 threads |
  |---|---|---|---|---|
  | Step 7a | 1,873 / 1,426 / 1,029 | 2,403 / 1,938 / 1,383 | 178 / 130 / 90 | 161 / 140 / 114 |
  | -k f32 | 1,870 / 1,440 / 1,069 | 2,588 / 1,996 / 1,475 | 194 / 146 / 132 | 197 / 143 / 134 |
  | -k bf16 | 1,828 / 1,446 / 1,058 | 2,568 / 2,010 / 1,491 | 207 / 162 / 136 | 244 / 181 / 158 |
  | -k f16 (default) | 1,869 / 1,425 / 1,028 | 2,591 / 1,949 / 1,434 | 194 / 164 / 128 | 243 / 177 / 147 |

  | Q4_K_M | prefill 451 / 1,500 / 3,000, 8 threads | 12 threads | decode after 451 / 1,500 / 3,000, 8 threads | 12 threads |
  |---|---|---|---|---|
  | Step 7a | 1,666 / 1,288 / 980 | 2,214 / 1,786 / 1,340 | 233 / 153 / 103 | 233 / 169 / 120 |
  | -k f32 | 1,664 / 1,327 / 1,005 | 2,302 / 1,816 / 1,394 | 246 / 168 / 136 | 292 / 191 / 158 |
  | -k bf16 | 1,635 / 1,314 / 996 | 2,306 / 1,809 / 1,390 | 276 / 219 / 167 | 306 / 240 / 193 |
  | -k f16 (default) | 1,671 / 1,297 / 972 | 2,346 / 1,827 / 1,364 | 273 / 220 / 158 | 299 / 227 / 180 |

  Medians (decode, 3,000 tokens, 8 threads): Q8_0 87 -> 113 / 129 / 127, Q4_K_M 91 -> 122 / 160 / 148
  (f32 / bf16 / f16). The layout gives most of the f32 gain; the 16-bit cache adds ~5-20% (more for
  Q4_K_M, whose weights are the smaller part). Prefill (f16): within +-1% at 8 threads, +1-8% at 12
- [x] Profile (`-DRUNQ_PROFILE`, Q8_0 GGUF, 8 threads, -b 512, min of 3 per phase; Step 7a -> f16):
  451 tokens 0.241 -> 0.246 s: GEMMs 68% (qkv 19.1, wo 9.3, w1+w3 26.4, w2 13.0), attention 20.7%,
  norm+quant+resid 4.1, swiglu+quant 4.5, qk-norm+rope+kv 2.4; 1,500 tokens 1.060 -> 1.048 s: attention
  41.5%, GEMMs 51%, rest 7.3%; 3,000 tokens 2.94 -> 2.90 s (f32 2.78): attention 57.0%, GEMMs 37.5%,
  rest 5.3%
- [x] Tests (`make test` 300 checks, ~25 s on the busy box at load 9-14; was 278 / ~19-26 s):
  everything runs on the f16 default (incl. `chat/long_context*`: decode at 16 threads past one block
  vs prefill and the generic build); new group `kv` for -k f32 and -k bf16 (.bin): logits vs HF on en,
  code (+ pooled) and long2 (past 256), batched (en -b 7, long2 -b 64) and threads (long2 -b 64, 8 vs 3)
  bitwise, AVX-512 vs generic (en -b 1, long2 -b 64) bitwise; greedy `f32_generate`, `bf16_generate`
- [x] Sanitizers (scratch builds): ASan + UBSan (`-O1 -g -fsanitize=address,undefined`, AVX-512 and
  generic) and ThreadSanitizer (no `-fopenmp`: the pool runs everything), 22 runs each: per KV type
  logits at -b 64 / 7 / 1 / 512 on a 310-token prompt (2 blocks) at 8 / 3 / 16 / 8 threads (Q8_0 GGUF,
  Q4_K_M), generate at -b 13 / 12 threads, chat with context full (-c 160, 6 turns, .bin, -b 7); chat -r 1
  -t 0.8 (Q4_K_M): no reports
- Notes / open: f16's `vcvtph2ps` is the remaining cost of the default (bf16's conversion is ~free but
  less accurate); a mixed cache (K f16, V bf16: V's rounding is harmless, +0.7% KL) would halve it.
  Decode after 3,000 tokens now moves 130-180 GB/s effective (Q8_0 GGUF: f16 1.0 GB per token at 128
  tok/s, f32 1.36 GB at 132), above the 72-130 GB/s of Step 3a's read test: the bandwidth limit of this VM
  needs re-measuring before deciding what bounds long-context decode

## Step 8: cleanup / architecture (library API, modules)
Goal: make Steps 9 (`runq serve`) and 10 (multi-sequence decode) natural without changing any result.
Before: one file `src/runq.c` (3,916 lines): `Transformer` = config + weights + one `RunState` (activations,
batched buffers, the KV cache) + the GGUF mapping; globals for the group size, K-quant flag, `-b`, `-k`, `-n`;
`forward(tok, pos)` / `forward_batch(tokens, n, pos0)` for consecutive positions of the one sequence; chat /
generate print and sample inside the loops; 66 `exit()` calls (62 in runq.c, 4 in threads.h).
- [x] Modules, one translation unit each (`dfed67f`, then reworked): `model.c` (loading .bin / GGUF, weight
  conversion / repacking, chat templates), `gguf.c` (reader), `forward.c` (sequences, contexts and every
  numeric part: decode step, batched pass, attention / matmul dispatch, quantization, norms, RoPE, KV cache),
  `tokenizer.c` (+ detokenization), `sampler.c`, `chat.c` (template rendering, generation loop), `threads.c`
  (the pool; `threads.h` keeps the inline phase synchronization, its globals are single instances now),
  `cli.c`, `internal.h`, the public `runq.h`. Separate TUs rather than a unity build: all numeric code of the
  forward passes stays together in `forward.c` (the exact arithmetic depends on gcc -Ofast's code
  generation in its context, e.g. `quantize()`, Step 7b), and the mechanical split alone was checked first:
  byte-identical outputs (compare_bins.sh, 360 jobs), speed within noise. The one place the context mattered
  elsewhere: `sample()` keeps its own function (noinline, the configuration through a pointer): inlined into
  a caller with a local configuration, -Ofast may turn the per-value temperature division into a
  multiplication by the reciprocal (other bits). Makefile: objects per flavour (`build/native`, `generic`,
  `debug`, rebuilt when the flags change); a clean native build still takes ~20 s (`forward.c` with the
  kernels, ~22 s for the single file before), a change outside `forward.c` rebuilds in ~1-3 s
- [x] The three objects (`src/runq.h`): **RqModel** (config, weights, tokenizer, the templates of both thinking
  modes; read-only once loaded, shared), **RqSeq** (its own KV cache: capacity and KV type per sequence; length;
  the tokens at its positions; RNG), **RqCtx** (activation / scratch buffers for `max_batch` rows, attention
  scratch sized for a context length and grown for a longer sequence, synchronization, logits, profiling, the
  per-thread conversion buffer the generic attention used to malloc per call). The forward passes take the
  sequence(s): `rq_decode`, `rq_prefill` (chunks, or token by token without batch buffers), and
  **`rq_forward_batch`**: rows of several sequences in one pass, in segments (consecutive positions of one
  sequence); per-row position (RoPE, causal length), per-segment KV cache in the QK-norm + RoPE + store items
  (a cache tile of one segment) and in the attention items (row blocks never straddle a segment; sorted
  longest first, for one segment the old order), logits of selected rows (none / last / all per segment: the
  decode GEMV for a single last row, as before, else the classifier GEMM over the compacted rows). So Step 10's
  multi-sequence decode is already possible (B segments of one row) and bit-identical to separate runs
  (tests below); what it lacks is speed (below)
- [x] API around it: `rq_model_load` (opts: context cap, tokenizer only, weights only), `rq_seq_*` (new /
  free / reset / truncate / common prefix / seed), `rq_sample` (per-call temperature / top-p, the sequence's
  RNG, an optional token mask for constrained decoding: disallowed logits set to a large finite negative
  value, no infinities under -Ofast), `rq_generate` (the chat loop as a function: prefill, RNG draws per
  prefilled token, sample, end tokens, max tokens, context full; callbacks per token (may stop) and per logits
  vector; stats and timing in a struct), `rq_tokenize` (with length: NUL bytes allowed), `rq_detok_push` /
  `_flush` (incremental, never splits a UTF-8 character), `rq_chat_system / user / end / reply_history /
  render` (messages -> tokens in the CLI chat's canonical form, thinking on / off), `rq_dequant_rows` (hook).
  No printing in the library: the load warning and the RUNQ_SYNC note are returned; the CLI prints the same
  `[timing]` / `[tokenize]` / `[profile]` lines from the stats
- [x] Errors: request paths return `RQ_ERR_*` (prompt past the capacity: `RQ_ERR_CONTEXT`, nothing prefilled;
  bad token / segment / mode: `RQ_ERR_ARG`; allocation: `RQ_ERR_NOMEM`, incl. inside encode / NFC / BPE) with
  a message (`rq_ctx_error`); load errors (every GGUF check, .bin header / size (new: a truncated .bin was a
  crash), tokenizer, allocation in the parallel workers) come back from `rq_model_load` as the message the CLI
  printed before, after freeing what was allocated (load_fail / gguf_fail longjmp to the loader; checked
  with ASan on truncated / corrupt files incl. a failure after the weights were loaded); `rq_begin` /
  `rq_sync_init` return errors; the pool keeps the threads it got if `pthread_create` fails. **exit(): 66 ->
  16, all in cli.c** (library: 0). CLI unchanged (same flags, modes, outputs, defaults); the only visible
  difference: `-c < 0` says so instead of "malloc failed!"
- [x] Cleanup: comments name the modules; the per-call malloc in the generic attention, globals for
  `-b` / `-k` / `-n` / the K-quant flag, `decode()`, the `Sampler` construction, `DecodeJob.token`, the
  duplicated template strings removed; `rq_model_free` / `rq_ctx_free` free everything (ASan leak check clean)
- [x] Verification (2026-10-26): `tests/compare_bins.sh` (new, reusable: two builds' outputs byte for byte,
  360 jobs: 3 models x both builds x f16 / bf16 / f32 x `-b 1 / 7 / 64 / 512` x a 5-token and a 280-token prompt
  for `-m logits`; greedy and seeded generate / chat, multi-turn on stdin with `-r 0 / 1`, a system prompt,
  context full, a cut reply, `-b 1 / 7`, the chat dumps, tokenize, dequant, error cases) against copies of the
  pre-refactor binaries (`e6ca1ec` = /tmp/cw11/runq-base): all identical after each commit. `make test` 300 ->
  333 checks (new group `api`, below), ~29 s at load 8-12. ASan + UBSan (scratch, both builds): generate at
  -b 7 / 512, logits -b 1 / 64 (-k bf16), chat with context full (-c 96) and -r 1 -k f32 -t 0.8, 16-thread
  decode past 256 positions, on the 3 models; tokenize, dequant, the API test on the 3 models (both builds);
  8 loads of truncated / corrupt files: ~65 runs, no reports. Builds: -DRUNQ_PROFILE (profile lines as
  before), no -fopenmp, -DRUNQ_NO_POOL, -march=x86-64-v3, -O0 compile and give the same text; -march=x86-64
  compiles; the NEON / macOS (clang) and Windows paths could not be compiled here (no toolchain): same
  guards and macros, to be checked on the Mac
- [x] Tests (`tests/api_test.c`, group `api`, 11 checks per model, .bin / Q8_0 GGUF / Q4_K_M, ~1.3 s each in the
  background from the start, so +0 s): sequences interleaved == alone, chunks of 2 sequences in one pass (all
  rows / last / none), decode rows of 3 sequences per pass (one at positions 250+), a chunk at 256..276,
  truncate + re-prefill == fresh, all bitwise; `rq_generate` == a hand-written loop; error codes; chat
  rendering == the CLI's pieces; detokenization; the token mask. Breakage (scratch): one sequence's cache for
  every attention row block, or position 0 for later segments: the batch checks fail
- [x] Speed (2026-10-26, interleaved with the pre-refactor binary, best of 6 (decode: best / median), chat,
  `-c 4096 -n 128`, load 9-12; prefill numbers in tok/s):

  | | prefill 451 tok, 8 / 12 thr | prefill 1,500 tok, 8 / 12 | decode after 451, 8 thr | 12 thr |
  |---|---|---|---|---|
  | Q8_0 GGUF, before | 1,875 / 2,356 | 1,391 / 1,933 | 201 / 153 | 190 / 165 |
  | Q8_0 GGUF, Step 8 | 1,856 / 2,526 | 1,428 / 1,982 | 185 / 162 | 220 / 194 |
  | Q4_K_M, before | 1,700 / 2,308 | 1,318 / 1,840 | 280 / 228 | 294 / 241 |
  | Q4_K_M, Step 8 | 1,681 / 2,354 | 1,317 / 1,816 | 297 / 249 | 294 / 245 |

  Within the noise everywhere (decode +-10% run to run on the shared box). Startup (`-n 1 -i hi`, 12 threads,
  median of 9): 0.095 / 0.072 / 0.077 s (Q8_0 GGUF / Q4_K_M / .bin) for both binaries
- Ready for Step 9: per request, `rq_chat_render` (thinking off) -> tokens; prefix reuse across requests with
  `rq_seq_common_prefix` + `rq_seq_truncate` on a cached sequence (several sequences = several prefix
  slots); `rq_generate` with the request's sampling, max tokens, `on_token` for stop strings / streaming
  (`rq_detok_push`), usage from the stats; constrained decoding through `RqSampling.allow` (the caller may
  change the mask from `on_token` between steps; forced runs of tokens can go through `rq_prefill`); errors
  as codes for HTTP statuses. Not there: HTTP / JSON, the schema -> token trie, a queue. One forward pass at
  a time per process (the pool is process-wide and not reentrant): the server serializes model calls
- Ready for Step 10: `rq_forward_batch` over rows of different sequences, each with its own cache and
  position, bit-identical to separate runs; sequences can join / leave per pass. Not there: small-B kernels
  (decode rows run through the prefill GEMM, whose per-call tile unpacking makes B = 2-8 slow, IDEAS.md), the
  per-256-position attention split for decode rows in a batch (one item per (row block, kv head): long
  contexts at small B use few threads), per-sequence sampling in parallel, a scheduler
- Open: the group size is process-wide (one model, or models of the same group size, per process:
  `rq_model_load` refuses another); `rq_chat_render` with thinking on drops the reasoning of every
  assistant message (HF keeps it for messages after the last user message; only matters when continuing an
  assistant message); `rq_detok_push` passes invalid UTF-8 through (a JSON writer must replace it)

## Step 9: `runq serve`, an OpenAI-compatible HTTP server
Goal: a local OpenAI-compatible endpoint, usable as LiteLLM's complexity-router judge. One binary: `runq serve
<model> [flags]` (a subcommand dispatched first thing in cli.c's `main`; nothing else in the CLI changed).
- [x] Layers (`src/serve.h`), each its own file, not part of the library: `http.c` (HTTP/1.1), `serve.c` (the
  OpenAI layer), `engine.c` (the engine thread and queue), `constrain.c` (constrained decoding), `json.c`: ~3,000
  lines of C with the headers. Threads: the main thread accepts (poll on the socket + a self-pipe for SIGINT / SIGTERM); one
  detached thread per connection (at most `--max-conns` 128, more get a 503; SIGINT / SIGTERM blocked in them);
  one engine thread running `Job`s from a FIFO ring (`--queue` 64 waiting, more get 503 `overloaded`). The Job is
  the interface between the layers: in, the prompt tokens, sampling, seed, max tokens, stop strings, constraint,
  logprobs, deadline; out, under the job's mutex + condition variable, the content / reasoning text so far, the
  logprobs, usage, finish reason or error; a cancel flag the connection sets when its client goes away (checked
  by the engine before every forward pass). serve.c / http.c never call the model; Step 10 replaces
  `engine_main` / `run_job` with a scheduler behind `engine_submit` and the same Job
- [x] HTTP (`http.c`, POSIX sockets): request line + headers up to 16 KB (431), bodies by Content-Length only (a
  `Transfer-Encoding` gets 411, before the body), `--max-body` 4 MB (413), `Expect: 100-continue`, keep-alive
  (HTTP/1.1 default, `Connection: close` honoured, HTTP/1.0 keep-alive on request), sequential pipelined requests,
  30 s to finish a started request (408), 120 s idle keep-alive, responses with Content-Length or chunked (SSE),
  `MSG_NOSIGNAL` + `SO_SNDTIMEO`; client gone = `poll` POLLRDHUP / HUP or a 0-byte peek. Shutdown: stop
  accepting, reject the queue (503), the running job may finish for 10 s, idle connections close, everything
  freed (a second signal exits at once). Windows: `serve.c` compiles to a stub that says the server is not
  available (http.c / engine.c are `#ifndef _WIN32`; json.c / constrain.c are plain C)
- [x] JSON (`json.c`, hand-written): DOM in an arena, strict RFC 8259 (no NaN / comments / trailing commas;
  raw control characters and invalid UTF-8 in strings are errors), `\uXXXX` with surrogate pairs (a lone
  surrogate becomes U+FFFD), depth 64, member order kept, duplicates: the last wins; the writer escapes and
  replaces every byte that is not part of a well-formed UTF-8 character (overlongs, surrogates, truncated
  sequences: `rq_detok_push` passes them through) with U+FFFD, so responses are always valid JSON / UTF-8
- [x] OpenAI layer (`serve.c`): `POST /v1/chat/completions` (+ `/chat/completions`), `GET /v1/models`,
  `/v1/models/{id}`, `/health` (+ `/v1/health`); 404 / 405 otherwise; `--api-key` (Bearer, constant-time
  compare, not on /health). Messages system / developer / user / assistant, content a string or text parts
  (joined with "\n"; images, tools, tool calls, `n > 1`, NUL characters: 400). Rendering: Qwen3's template in
  the CLI chat's canonical form (`rq_chat_system` / `rq_chat_user` / `rq_chat_end` pieces: the empty think block
  stays in every user turn with thinking off), so the server's greedy text equals `runq -m chat`'s; sequences the
  CLI cannot produce as HF's template renders them (a later system message, two user messages in a row, an
  assistant message first); earlier assistant messages lose their `<think>` block; the prompt ends with the
  assistant header. Sampling: `temperature` 0-2 (0 greedy), `top_p`, `seed` (positive seeds as `runq -s`; others
  and none: splitmix64 of the value / time + a counter), server defaults `-t 1 -p 1` (OpenAI's); `max_tokens` /
  `max_completion_tokens` (default and cap: the rest of the context; a prompt that does not fit: 400
  `context_length_exceeded` with OpenAI's message); `stop` (up to 16, held back while a match is possible, cut
  before it; whole UTF-8 characters); `stream` (SSE chunks with `role`, `content` / `reasoning_content` deltas,
  a final chunk with `finish_reason`, `stream_options.include_usage`: `usage: null` on each chunk and a last one
  with `choices: []`; errors after the start: a `data: {"error": ...}` event; then `[DONE]`); `logprobs` /
  `top_logprobs` (0-20, not with stream): log-softmax of the raw logits (no temperature), renormalized over the
  allowed tokens under a constraint, forced tokens 0; thinking (`reasoning_effort` low / medium / high,
  `chat_template_kwargs.enable_thinking`; off by default and under a constraint): the `<think>` block's text as
  `reasoning_content` (leading / trailing newlines stripped), the rest as `content` (leading newlines stripped);
  `cache_prompt: false` (llama.cpp's name): no prefix reuse. Usage: prompt, completion (sampled + forced, not the
  end token), total, `prompt_tokens_details.cached_tokens`. Errors: OpenAI objects (`message`, `type`, `param`,
  `code`) with 400 / 401 / 404 / 405 / 411 / 413 / 431 / 499 (logged only) / 500 / 503 (`Retry-After: 1`) / 504.
  Log: one line per request on stderr (time, client, method, path, status, prompt / cached / completion tokens,
  queue, prefill, total); `--verbose` adds the request and response bodies (up to 64 KB)
- [x] Engine (`engine.c`): `--slots` 4 cached sequences of `-c` 8192 positions (address space; 115 KB per
  position touched). A job takes the slot with the longest common token prefix if it saves >= 32 tokens or half
  the prompt (else the least recently used slot: LiteLLM's judge calls and the routed tier calls then keep one
  slot each, as the e2e run shows), truncates it (`rq_seq_truncate`; the whole prompt cached: its last token is
  prefilled again, for the logits) and prefills the rest in chunks of `-b` (a cancel is seen between chunks).
  The RNG is seeded per request and drawn once per prompt token but the last however much was cached, as
  `rq_generate` does for a first turn: a seeded request == `runq -m chat -s` (tested). Generation: sample (the
  sampler's token mask under a constraint), the end token ends it (not fed), else it is fed with the next step
  (`rq_decode`, or one `rq_prefill` together with forced tokens); a reply cut by `max_tokens` does not feed its
  last token. Bit-identity of reuse: the same positions and tokens, and the forward passes do not depend on the
  chunking, so a cached prefix gives the same logits as a fresh prefill (tested through 9-digit logprobs)
- [x] Constrained decoding (`constrain.c`), `response_format: json_schema` with a finite set of outputs: the schema
  -> every valid output as canonical text -> a byte trie. Supported: `enum` (any JSON values), `const`,
  `boolean`, `null`, `integer` with `minimum` / `maximum` (exclusive too, <= 256 values), objects (nested,
  `properties` in schema order, `required` / optional, additional properties never), `anyOf` / `oneOf`, one
  `allOf`, `$ref` (any local JSON pointer: `#/$defs/..`, `#/definitions/..`; recursion depth 32), annotations
  ignored; <= 4096 outputs, 1 MB; anything else is a 400 naming the part (e.g. `type "string" without enum /
  const has no finite set of outputs at /properties/a`). Formatting: Python's `json.dumps` (`": "`, `", "`, one
  line, non-ASCII as UTF-8): asked for `{"tier": ...}` Qwen3-0.6B writes exactly `{"tier": "MEDIUM"}` (in a
  ```` ```json ```` fence: the fence is not part of the output). Unconstrained, with LiteLLM's prompt, it answers a
  bare `MEDIUM` (the prompt does not ask for JSON), so the schema has to come from the decoder. Per step: if all
  candidate outputs share the next bytes, the longest common prefix of their suffixes' canonical tokenizations is
  forced (over 64 candidates: the single-path run's tokens minus the last); else a decision, masked at the byte
  level: every vocab token (not added tokens) whose bytes keep the output a prefix of a valid string, found by a
  trie walk with binary search in the vocabulary sorted by bytes (pruned where no token continues; the end token
  too where a complete output can also continue, e.g. enum [1, 12]). Why byte level and not a trie of canonical
  token sequences: the model keeps every way of spelling a valid output (`M` + `EDIUM` and `MED` + `IUM`; `S`,
  `SI`, `SIM`), so the masked distribution is its distribution over the valid strings; measured, the
  non-canonical first tokens carry at most 0.2% (0.6B) / 1.1% (1.7B) of the mass on the eval set. Forced tokens
  get no forward pass of their own: the initial ones (`{"`, `tier`, `":`, ` "`) go into the prompt's prefill, the
  ones after a decision are fed with the next decision's pass, and the last run (`PLE`, `"}`) is never fed; a
  single-output schema runs no forward pass at all. A TierClassification classification is thus one prefill (+
  the decision's sampling). Temperature / top-p work under the mask (the sampler's large finite negative logits)
- [x] Measured, not done: computing only the allowed classifier rows at a decision (~13 of 151,936 for the tiers).
  The last-row classifier costs 1.2 ms (8 threads; 1.8 at 4) of a 16.6 ms cached classification step (20 new
  tokens after 260), ~7%; worth it only with a subset-classifier kernel in `forward.c` (new numeric code under
  the bit-identity tests), so left for later. `json_object` (a JSON grammar over bytes, ~400 lines): not done, 400
- [x] Tests (`tests/server_test.py`, group `server`, 48 checks, details in tests/README.md): the server and its 5
  CLI reference runs at 2 threads in the background from the start; `make test` 333 -> 381 checks, ~30 -> ~32 s
  (interleaved runs at load 8-22 with and without the group: 32.4 / 31.6 / 31.6 vs 29.4 / 29.9 / 30.3 s). The
  cost is CPU contention (~19 core-seconds of decoding next to the logits prefetch), so the generation lengths
  were cut to what the checks need (~8 s alone at 2 threads); at 1 thread it is cheaper (~16 core-s) but takes
  ~12 s and sometimes became the critical path (35.7 s once)
- [x] Sanitizers (scratch builds, `tests/server_test.py --slow`): ASan + UBSan, AVX-512 (Q8_0 GGUF, .bin, Q4_K_M)
  and generic (Q8_0 GGUF): 48 / 48 each, no reports, leak check clean at exit; ThreadSanitizer (no OpenMP: pool,
  engine and connection threads instrumented): 48 / 48, clean after one fix (the shutdown flag written by the
  signal handler and read by the connection threads: now atomic). The fuzz check (150 mutated / random requests)
  runs in all of them
- [x] End to end (`scripts/litellm_e2e.sh`): an own LiteLLM 1.105.0 proxy from `/opt/litellm/venv-1.105.0` (read
  only; `env -i`, scratch HOME, random master key, no database, telemetry and remote cost map off, free localhost
  ports) with `auto_router/complexity_router`, `classifier_type: llm`, `classifier_context_window_size: 0`, the
  judge and the four tier models all on one runq serve (tier models by model name), 8 prompts through the openai
  client. What LiteLLM sends: `{"messages": [{"role": "system", "content": <rubric, 1,142 chars>}, {"role": "user",
  "content": "\nClassify this message:\nHi there!"}], "model": "judge", "response_format": {"type": "json_schema",
  "json_schema": {"schema": {"description": ..., "properties": {"tier": {"enum": ["SIMPLE", "MEDIUM", "COMPLEX",
  "REASONING"], "title": "Tier", "type": "string"}}, "required": ["tier"], "title": "TierClassification", "type":
  "object", "additionalProperties": false}, "name": "TierClassification", "strict": true}}, "temperature": 0}`
  (temperature from the model_list entry; the enum is inline, not a `$ref`; both tested). Its prompt and
  response_format are byte-identical to `classification_system_prompt(0)` + `_build_classifier_user_payload` +
  `type_to_response_format_param(TierClassification)` from LiteLLM's code. LiteLLM facts that differ from IDEAS.md:
  the rubric is the system message (stable: cacheable), the caller's text is the user message (`Caller system
  prompt, quoted as task context:` if any, prior turns with the default context window 3, then `Classify this
  message:`); the reply is parsed from its first JSON object (`extract_classifier_json`, then the label); timeout
  `timeout_ms` 3,000 by default with `asyncio.wait_for`, no retries, a timeout opens a 30 s circuit breaker
  (fallback: the heuristic scorer); rubrics legacy (the default) / chat / agentic / business. Result: every request
  classified and routed to its tier's model; judge latency (runq's log, 8 threads, Q8_0 GGUF) 176-182 ms for the
  first (265 tokens, nothing cached), then 15-27 ms with the 253-token rubric prefix cached; end to end through the
  proxy ~120-235 ms (incl. LiteLLM and the tier model's 24-token answer). The openai package (2.54.0, from the
  venv): models, chat, streaming with usage, `beta.chat.completions.parse` with a Pydantic `Literal` model, errors
  (`BadRequestError`), keep-alive: all work
- [x] Judge quality (`scripts/eval_router.py`, `scripts/router_eval.jsonl`: 80 prompts written for this, 20 per tier
  following the legacy rubric's definitions, incl. lookups that look long (a trip story ending in a currency
  question, an email ending in "what day", pasted Rust code + "what language") and hard ones that look short (`Is P
  equal to NP?`, `Is 2^61 - 1 prime?`, the bat and ball); the request built with LiteLLM's own code and checked
  against a logged one; temperature 0; label probabilities = the masked softmax at the decision step summed per
  label). 8 threads, 2026-10-26:

  | model | legacy | chat | agentic | business | p(gold), legacy | latency first / median (legacy) |
  |---|---|---|---|---|---|---|
  | Qwen3-0.6B Q8_0 GGUF | 51% | 62% | 64% | 64% | 0.53 | 172 / 26 ms |
  | Qwen3-0.6B Q4_K_M | 51% | 68% | 71% | 60% | 0.54 | 185 / 27 ms |
  | Qwen3-1.7B Q8_0 | 88% | 82% | 86% | 79% | 0.84 | 399 / 51 ms |
  | Qwen3-4B Q4_K_M | 91% | 88% | 92% | 79% | 0.90 | 963 / 139 ms |

  (The longer rubrics cost more cold: chat / agentic / business 298 / 334 / 528 ms first for 0.6B, the median
  only +3-8 ms.) 0.6B is not a usable judge: with the legacy rubric it never says REASONING (19 of 20 go to
  COMPLEX) and rates most lookups MEDIUM (13 of 20; SIMPLE recall 25%); its probabilities are confident when
  wrong (`What is the capital of Canada?` MEDIUM 0.71). LiteLLM's warning ("rated hard prompts SIMPLE") does not
  reproduce under constrained decoding: 0 of 40 COMPLEX / REASONING prompts rated SIMPLE for any 0.6B setup; its
  errors are within one tier 94-98% of the time, so as a two-way splitter (SIMPLE + MEDIUM vs COMPLEX +
  REASONING) it gets 69-75 of 80 right (Q8_0 legacy 73; 1.7B 70-75, 4B 77-79), the only use it supports. Q4_K_M is no
  worse than Q8_0 here. 1.7B Q8_0 is the smallest usable one (88% legacy, REASONING recall 75%, ~50-60 ms); 4B
  Q4_K_M is the best (88-92% except business) at ~140-160 ms. The business rubric moves REASONING to "committing
  to a decision", so its lower scores partly measure our legacy-based labels
- [x] Verification: `make test` green (381); `tests/compare_bins.sh -j 4 /tmp/cwref .`: all 360 jobs identical to
  the pre-Step-8 binaries (296 s); CLI speed, interleaved with /tmp/cwref/runq, best of 6, 8 threads, Q8_0 GGUF,
  312-token chat prompt, `-n 128`: prefill 1,953 / 1,986 tok/s (before / now), decode 180 / 190 (load 6-10; within
  the noise)
- Open for Step 10: the engine runs one job at a time: concurrent requests queue (6 concurrent 60-token
  generations: 2.3-2.7 s instead of ~0.4 s each); continuous batching = several jobs' decode rows in one
  `rq_forward_batch` per step, prefill chunks of new jobs next to them, each job's sampling / constraint / stop
  logic per row (the Job keeps its interface; `run_job` becomes per-step state), needs the small-B kernels for
  speed. Slots then become the batch's sequences (one per running job + idle cached ones); slot choice could
  prefer a prefix shared by several queued jobs. Also: a decision step's subset classifier (above);
  `json_object` / free-form strings in schemas (a byte-level JSON grammar); logprobs with streaming; prefill of a
  long prompt blocks the queue (chunked: a cancel is seen, but other jobs wait: interleave chunks with decode
  rows in Step 10); tools / function calling are not planned

## Step 10a: multi-sequence (batched) decode in the library
Goal: N independent sequences decoding together, one row each per `rq_forward_batch` pass, as fast as the
weights allow, each sequence's results bit-identical to decoding it alone. Before (Step 8): such passes
already worked and were bit-identical, but ran as a prefill chunk: the GEMM (K-quants: each 32-row weight
tile unpacked into a per-thread scratch per pass), 9 phases per layer, one attention item per (row, kv head),
the KV pre-fault of every segment every pass, sampling row by row.
- [x] Benchmark first (`tests/bench_batch.c`, `make bench-batch`, `scripts/bench_batch.sh`): one base sequence
  prefilled with L tokens is copied into the N sequences (`rq_seq_copy`, new API: the first n positions of a
  sequence into another, the same bits; also the scheduler's way to share a cached prompt prefix); each rep
  truncates them to L and decodes STEPS tokens together (different first tokens), logits of every row, greedy
  (`-t`: top-p 0.9) sampling; best / median rep; `-d` adds `rq_decode` (N = 1, the decode path); `-P CHUNK` adds
  a prompt's first chunk of CHUNK tokens to every pass (mixed passes); a `-DRUNQ_PROFILE` build prints the time
  per phase type and pass (`scripts/bench_batch.sh -p`)
- [x] Baseline profile (Q4_K_M, 8 threads, 50 positions, ms per pass): N = 1: 6.7 (decode path: 3.9): GEMMs
  qkv 1.0 + wo 0.54 + w1/w3 1.38 + w2 0.94 (tile unpacking: ~2x the GEMV), classifier 1.68 (the GEMM over 1
  row), norm / rope / swiglu phases 0.9 (5 extra phases per layer, ~5 us each); N = 8: 10.4 (GEMMs 5.7,
  classifier 2.1, attention 0.9 for 50 positions, rest 1.5); at 2,000 positions the attention 2.6 (N = 1: 8
  items for 8 threads) to 14.7 ms (N = 8: 1.8 GB of KV cache per pass, ~120 GB/s: at the bandwidth limit)
- [x] Small-batch kernels (`kernels_avx512.h`, `smb512_rows`; forward.c `rows_mm`, `smb_use`): per 16-row block
  of W, streamed once by the thread owning it (per-thread block ranges, as the GEMV), the 4 / 6-bit data
  unpacked in registers once per group (Q4_K: 8 steps; Q6_K: 4 steps per sub-block) and used for up to SMB_C =
  8 activation rows (more rows: the block again from L1/L2, 8 at a time); two vpdpbusd chains per row (Q8_0
  too: with one chain per row at 3-8 rows the latency bound it below the GEMM tile); no scratch, no tile
  unpacking. The folds are the existing functions (`q8p_fold`, `kq4_fold`, `kq6_fold`) per (block, row, group)
  in group order, the integer sums exact: every row's result is the GEMV's / GEMM's / generic kernel's, whatever
  it is batched with. One thread, L2-resident 512 x 1024 matrix, GMAC/s (`/tmp` microbenchmark):

  | rows | 1 | 2 | 4 | 8 | 16 | 24 | 32 | 48 | 64 |
  |---|---|---|---|---|---|---|---|---|---|
  | Q8_0 small-batch / GEMM | 97 / 95 | 165 / 175 | 183 / 183 | 183 / 182 | 159 / 183 | 183 / 184 | 183 / 183 | 169 / 184 | 183 / 184 |
  | Q4_K small-batch / GEMM | 95 / 41 | 124 / 73 | 142 / 101 | 151 / 125 | 152 / 142 | 152 / 149 | 152 / 151 | 152 / 157 | 152 / 158 |
  | Q6_K small-batch / GEMM | 69 / 30 | 101 / 57 | 125 / 83 | 130 / 105 | 130 / 121 | 130 / 127 | 129 / 130 | 130 / 134 | 129 / 135 |

  Crossovers: K-quants faster up to ~24 rows (2.3x at 1 row, 1.4x at 4), equal at 32, ~4% slower at 48-64; Q8_0
  equal (the Q8 GEMM tile streams the packed weights without unpacking: it already was a small-batch kernel).
  Dispatch: `rq_forward_batch`'s GEMM phases use them for at most 32 rows per weight type (`smb_max`,
  `RUNQ_SMB=Q8,Q4,Q6` to tune), the batched decode worker always. NEON keeps its GEMM (no small-batch kernels
  there yet, `RUNQ_SMB` undefined; same results by definition); the generic build runs the decode GEMV per row
- [x] `batch_decode_worker` (forward.c), for passes whose segments are all single rows, N <= 64 (`BD_MAX`;
  `RUNQ_BD=max,epi,split`): the decode step's structure for N rows, 4 phases per layer: [q / k / v in items of
  64 weight rows over all rows (8 per kv head; the last of a kv head's items to finish runs its QK-norm + RoPE
  and K / V stores for every row, into each row's own sequence cache at its own position); attention items
  per (row, kv head) or (row, kv head, block of 256), each waiting for its kv head (dependency counters); the
  last kv head of a row quantizes it for wo] [wo; residual + rmsnorm + quantize per row] [w1 + w3 + SwiGLU +
  quantize per 64 hidden rows, all rows] [w2; residual + the next rmsnorm], then the classifier (small-batch,
  the rows with logits). The per-row norms are the phase epilogue up to `bd_epi` = 2 rows, else a phase of
  their own (epilogue at 4: within noise; at 8 / 16: ~10% slower). The KV tile pre-fault only for rows starting
  a tile (was every segment, every pass: 56 madvise calls per row). Measured and not kept: the attention items
  of each kv head after the next head's q / k / v items (~5-10% slower at N = 8); the norms as the first items
  of the next phase with the matmul items waiting on a counter (one sync point less per norm, but the w1 / w3
  items then come from one shared counter instead of per-thread ranges: ~10% slower at N = 8); q / k / v items
  of 32 / 64 / 128 rows: within noise; 64 rows per pass the limit (above it the prefill path's GEMM: at 96-256
  rows ~10% faster, at 48-64 equal)
- [x] Attention for decode rows: a row's attention split over its blocks of ATT_P = 256 positions (one item per
  (row, kv head, block), the last block to finish combines with `att_combine`: the decode path's definition and
  order, so the same bits) when the pass has fewer than `bd_split` (8) x threads (row, kv head) items; else one
  item per (row, kv head) over all its blocks (attention_kv, as the prefill). At 2,000 positions, Q4_K_M,
  aggregate tok/s, 8 threads N = 1 / 2 / 4: never split 133-188 / 193-258 / 262-324, split as above 177-199 /
  256-260 / 304-322; from N = 8 within noise (the KV traffic is at the bandwidth limit either way). Mixed passes
  (`prefill_worker`): the same split for single-row row blocks (`prefill_plan`), row blocks ordered by the cost of
  their items; the QK-norm + RoPE + K / V store items per (cache tile, kv head) instead of per tile (a 16-row
  chunk ran on one thread: 1.5 -> 0.75 ms of a 16-row pass at 12 threads); the KV pre-fault only for segments
  that start a tile
- [x] Sampling: `rq_sample_batch` (new API): the rows of a pass on the pool (one row per claim, a candidate scratch
  per thread), each with its sequence's RNG and its own settings; the same tokens and RNG states as `rq_sample` row
  by row (tested). Greedy argmax is ~20 us per row (not visible at N <= 16); at -t 0.8 -p 0.9 the softmax over the
  vocabulary per row: N = 64 pass + sampling 63-64 -> 56 ms (8 threads), N = 8 9.5-10 -> 8.1-9.2
- [x] Results, aggregate tok/s, best of 4 interleaved rounds per binary (2026-10-26, load 6-12; 50 positions per
  sequence; "decode" = `rq_decode`, one sequence):

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

  The 3x target at N = 8 is met at 8-12 threads for Q4_K_M, with less margin than the Mac proxy suggested for
  the EPYC: Q4_K_M decode is already fast (0.4 GB per token), and at N = 8 the pass is compute-bound at 8
  threads (~0.55 ms per row of arithmetic: the K-quant folds, 6 float ops per 8 vpdpbusd, cost about as much
  as the dot products; the small-batch kernels run at ~80% of the measured vpdpbusd peak with them). Per pass
  at N = 8 (profile, 8 threads): q / k / v + attention 2.9 ms, w1 / w3 2.0, classifier 1.9, w2 1.0, wo 0.7,
  norms 0.4. Q8_0 gains more (its decode reads 0.67 GB per token); for it the small-batch kernel is no faster
  than the GEMM tile, the gain is the phases.
  Full table, one interleaved run per (model, threads) with both binaries (load 8-16, so single runs; tok/s
  before / **now**, contexts of 50 / 500 / 2,000 positions per sequence; per-sequence tok/s = aggregate / N):

  | Q4_K_M: threads, context | decode | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 | N=64 |
  |---|---|---|---|---|---|---|---|---|
  | 8, 50 | 165 / 242 | 132 / **247** | 272 / **424** | 487 / **714** | 725 / **1028** | 935 / **1126** | 1047 / **1165** | 1150 / **1158** |
  | 8, 500 | 233 / 261 | 172 / **253** | 287 / **390** | 422 / **443** | 586 / **591** | 658 / **698** | 724 / **813** | 782 / **775** |
  | 8, 2000 | 162 / 192 | 134 / **195** | 196 / **235** | 236 / **326** | 333 / **306** | 338 / **364** | | |
  | 12, 50 | 234 / 258 | 183 / **250** | 356 / **482** | 567 / **945** | 813 / **1292** | 1040 / **1454** | 1406 / **1569** | 1499 / **1644** |
  | 12, 500 | 210 / 274 | 147 / **230** | 312 / **434** | 487 / **678** | 670 / **851** | 855 / **914** | 850 / **971** | 1012 / **1075** |
  | 12, 2000 | 212 / 178 | 128 / **172** | 225 / **218** | 308 / **314** | 328 / **412** | 422 / **468** | | |
  | 14, 50 | 222 / 269 | 179 / **255** | 299 / **399** | 501 / **821** | 800 / **1285** | 1142 / **1475** | 1448 / **1565** | 1504 / **1645** |
  | 14, 500 | 285 / 318 | 163 / **288** | 244 / **441** | 529 / **635** | 735 / **888** | 901 / **1016** | 1038 / **972** | 1108 / **1148** |
  | 14, 2000 | 244 / 203 | 134 / **192** | 228 / **319** | 319 / **387** | 384 / **358** | 460 / **486** | | |

  | Q8_0 GGUF: threads, context | decode | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 | N=64 |
  |---|---|---|---|---|---|---|---|---|
  | 8, 50 | 236 / 211 | 219 / **204** | 386 / **371** | 641 / **548** | 946 / **842** | 1179 / **1209** | 1245 / **1325** | 1312 / **1433** |
  | 8, 500 | 179 / 181 | 185 / **206** | 318 / **338** | 500 / **519** | 702 / **737** | 773 / **806** | 819 / **902** | 870 / **831** |
  | 8, 2000 | 164 / 139 | 158 / **147** | 193 / **199** | 279 / **242** | 288 / **309** | 372 / **402** | | |
  | 12, 50 | 198 / 223 | 210 / **232** | 366 / **415** | 627 / **615** | 982 / **831** | 1132 / **1243** | 1532 / **1514** | 1727 / **1760** |
  | 12, 500 | 162 / 220 | 157 / **205** | 333 / **354** | 483 / **447** | 688 / **638** | 818 / **883** | 923 / **970** | 1101 / **1118** |
  | 12, 2000 | 145 / 174 | 104 / **166** | 188 / **261** | 264 / **326** | 378 / **390** | 448 / **452** | | |
  | 14, 50 | 180 / 164 | 156 / **171** | 327 / **328** | 497 / **690** | 632 / **1136** | 1054 / **1492** | 1408 / **1800** | 1666 / **1944** |
  | 14, 500 | 176 / 240 | 161 / **251** | 292 / **359** | 474 / **576** | 685 / **528** | 912 / **751** | 970 / **905** | 1075 / **1083** |
  | 14, 2000 | 169 / 147 | 108 / **142** | 179 / **253** | 330 / **334** | 379 / **403** | 416 / **467** | | |

  (2,000 positions x N = 32 / 64: skipped, 7.4 / 14.9 GB of KV cache.) Run-to-run noise on this box is +-15-30%
  (the Q8_0 cells where "now" is lower, e.g. 8 threads N = 4 / 8, are faster in the best-of-4 table above).
  Long contexts are KV-bandwidth bound: each row reads 115 KB per position (f16) per token, 230 MB at 2,000
  positions vs 0.4 GB of Q4_K_M weights shared by the pass, so batching gains at most ~2x there (aggregate
  ~350-480 tok/s from N = 8 on, 8-14 threads).
- [x] Mixed passes (a prompt's first chunk of C tokens next to N decode rows at 500 positions; Q4_K_M, best of 5,
  `bench_batch -P C`): ms per pass / prefill tok/s:

  | C \ N | 0 (prefill only) | 1 | 4 | 8 | 16 |
  |---|---|---|---|---|---|
  | 16, 8 / 12 threads | 11.4 / 9.6 ms (1,403 / 1,669 tok/s) | 13.6 / 12.1 | 17.3 / 14.4 | 21.5 / 18.1 | 32.1 / 22.5 |
  | 64 | 37.1 / 27.2 (1,724 / 2,350) | 39.3 / 30.4 | 41.7 / 32.8 | 45.9 / 36.4 | 55.8 / 42.4 |
  | 128 | 69.2 / 53.7 (1,850 / 2,383) | 73.2 / 53.5 | 74.8 / 54.7 | 79.4 / 58.8 | 88.2 / 67.0 |
  | 256 | 140 / 106 (1,828 / 2,414) | 145 / 106 | 147 / 108 | 152 / 113 | 161 / 121 |
  | decode rows alone | | 4.2 / 3.3 | 7.8 / 5.5 | 13.0 / 10.7 | 22.2 / 19.8 |

  A decode row next to a chunk adds ~0.6-1.3 ms to the pass (a decode-only pass of 8 rows: 1.3-1.6 ms per row),
  but every decode row then advances one token per pass of the chunk's length
- [x] Prefill and single-sequence decode unchanged (interleaved with the previous binary, best of 4, chat
  `-c 4096 -n 128`, load 7-11): prefill 451 / 1,500 tokens Q8_0 GGUF 8 threads 1,873 / 1,415 -> 1,883 / 1,421,
  12 threads 2,511 / 1,976 -> 2,605 / 1,964; Q4_K_M 8 threads 1,739 / 1,314 -> 1,736 / 1,321, 12 threads 2,269 /
  1,830 -> 2,374 / 1,812 tok/s; decode (`rq_decode`, untouched) within noise (`bench_batch -d`, 6 rounds, Q4_K_M 8
  threads: best 310 / 281 vs 298 / 284 at 50 / 500 positions). Again after the per-(tile, kv head) RoPE items
  (best of 3-5): prefill within noise (e.g. Q8_0 GGUF 1,500 tokens 12 threads 1,876 -> 1,905, Q4_K_M 1,500 tokens 8
  threads 1,307 -> 1,328)
- [x] Bit-identity: every output is computed whole by one thread with the functions of the decode / prefill
  paths (the small-batch kernels with the folds of the GEMV), so a sequence's logits do not depend on the other
  rows, the row count, the path (batch_decode_worker, prefill_worker with the small-batch kernel or the GEMM),
  the attention split or the thread count. `tests/compare_bins.sh -j 4 /tmp/cwref .` (CLI outputs, 360 jobs, incl.
  prefill chunks of 7 rows that now use the small-batch kernels): all identical, after each commit. Builds with
  -DRUNQ_NO_POOL, without -fopenmp, -march=x86-64-v3, -O0: the api tests pass (each its own digests: other
  non-kernel float code); -DRUNQ_PROFILE -Wall -Wextra: no warnings. NEON / Windows not compiled here (no
  toolchain): the small-batch code is under `RUNQ_SMB` (AVX-512 only), the decode-rows worker off on NEON
  (`bd_max` 0), so both keep their previous paths
- [x] Tests (`tests/api_test.c`, group `api`; `make test` 381 -> 427 checks): N = 1, 2, 3, 5, 8, 9, 17 sequences
  (prefixes of 8-32 positions of one prompt, 2 steps) decoded together == each alone through `rq_decode`
  (bitwise, via 64-bit hashes of the logits), N = 3 and 9 at positions 248-260 (across the first attention block,
  split and not split at 4 threads), 65 sequences in one pass (Q4_K_M: more than the decode-rows worker takes, the
  prefill path's GEMM), mixed passes (a 20-row prefill chunk of a new sequence with all logits + 5 decode rows,
  and a 2-row chunk + 2 rows: split attention in prefill_worker; with long or short rows), `rq_sample_batch` ==
  `rq_sample` (greedy, top-p, full distribution, per-row settings; RNG states), a bf16 KV cache (.bin), copy
  errors; per model (.bin group 64, GGUF Q8_0 group 32, Q4_K_M). Every batched pass also prints a DIGEST line:
  `check.py` runs `api_test --digest` at 1, 8 and 16 threads (Q4_K_M; the main runs use 3 / 4 threads) and the
  generic build (`build/api_test-generic --digest-small`: the short passes, .bin and Q4_K_M; checked once by hand
  with `--long` on all three models: identical) and compares them (5 checks). Breakage (scratch builds): an FMA in the small-batch fold of Q8_0, Q4_K or Q6_K
  fails 14 api checks of each affected model (batch_prefill / batch_decode / past_block / n1..n17 / long /
  mixed). Time: the api runs ~4 s per model at 3-4 threads in the background (was ~1.3), the digests ~6 s on
  one thread and ~2-3 s generic; to stay near ~32 s, the sequential (`-b 1`) logits runs of `check.py` now use 4
  threads instead of 8 (token by token they are memory-bound; the three `long2 -b 1` runs took ~8 s each next to
  the other jobs at 8 threads, ~3 s at 4): `make test` 31.5-34 s vs 29.6-35.9 s for the Step 9 suite,
  interleaved, at load 8-22 (the box's load dominates: within ~1-2 s)
- [x] Sanitizers (scratch builds): ASan + UBSan (`-O1 -g -fsanitize=address,undefined`): api_test on the 3 models
  (all checks, `--n65 --kv`, digests at 1 and 16 threads, the generic build's), bench_batch (N up to 70, 20 /
  600 positions, top-p sampling, 3 and 16 threads, `RUNQ_BD=0`), the CLI (logits -b 1 / 7 / 64 / 512 on a 390-token
  prompt, identical; generate -b 7; chat at -c 200): no reports, no leaks (one leak in api_test's digest mode
  fixed). ThreadSanitizer (no OpenMP: the pool): api_test on Q4_K_M and the .bin at 6 threads, bench_batch at 16
  threads (N up to 24, 600 positions, sampling) and 5 threads through prefill_worker: no reports. Again after the
  per-(tile, kv head) RoPE items: api_test (.bin `--n65 --kv`, Q4_K_M) and `bench_batch -P 37` (mixed passes)
  under both: clean
- Notes for Step 10b (the scheduler in `runq serve`):
  - One pass per step: every running job's next token as a one-row segment (`RQ_LOGITS_LAST`), in one
    `rq_forward_batch`; then `rq_sample_batch` with each job's sampling settings and token mask (rows are
    independent: constrained jobs mix with free ones). A sequence may appear once per pass. The context's
    `max_batch` must cover the rows (decode rows + a prefill chunk); passes of single rows up to 64 take the
    decode-rows worker, anything with a multi-row segment the prefill path
  - Batch sizes: aggregate throughput keeps growing to N ~ 16 at 8 threads and ~32 at 12-14 (short contexts:
    ~1,100 / ~1,450-1,650 tok/s), per-sequence speed falls (N = 8: ~130-160 tok/s each, N = 16 ~70-90, N = 32 ~35-50);
    for a judge / router workload of short replies, take every queued job up to ~16-32 rows. At long contexts
    the KV traffic dominates (115 KB per position and row per token): at 2,000 positions little is gained past
    N = 4-8
  - Prompts: prefill in chunks interleaved with the decode rows; a chunk of C tokens takes ~C / 1,700-2,400 s and
    stalls the decode rows for that long (table above). For interactive streams chunks of 32-64 (decode rows at
    ~25-45 tok/s each meanwhile), for throughput 256-512 when nothing is decoding. A prompt's last chunk with
    `RQ_LOGITS_LAST` gives its first token's logits in the same pass
  - Memory per sequence: KV 115 KB per position (f16 / bf16; f32 229 KB), address space up to the capacity, touched
    per 16-position tile (1.8 MB for all layers); a context with `max_batch` 64 ~4 MB of activation buffers (512:
    ~30 MB); the split attention's partials grow with rows x blocks (~1 MB at 14 rows x 2,000 positions)
  - Prefix sharing: `rq_seq_copy` copies a cached prefix into another slot (bit-identical continuation), e.g. the
    judge's rubric for several concurrent requests
  - Threads: 12-14 for throughput (the batched passes are compute-bound at 8 threads from N ~ 8)

## Step 10b: continuous batching in `runq serve`
Goal: concurrent requests share the forward passes (as vLLM / llama.cpp's server), each request's output
unchanged. Before (Step 9): one engine thread ran one Job at a time from a FIFO (6 concurrent 60-token requests
2.3-2.7 s instead of ~0.4 s each); a long prefill blocked everything behind it.
- [x] Scheduler (`engine.c`; the Job interface, serve.c and http.c unchanged but for the flags, the startup line and
  the `[response]` log line naming the client). Job states queued -> running (prefilling -> decoding) -> done. Each
  engine step: (1) queued jobs whose client left / whose deadline passed are finished (499 / 504); (2) admission,
  FIFO, while fewer than `--parallel` run (a job waiting for a shared prefix keeps its place, later ones may pass
  it); (3) one `rq_forward_batch` over every running job: the decode rows first (each decoding job's last sampled
  token, plus a constrained job's forced tokens as a multi-row segment with the last row's logits), then prompt
  tokens within a budget (`--chunk` 64 while any job decodes, `-b` 512 when none does), shared by the prefilling
  jobs: first up to an equal share each (FIFO), then the rest FIFO, so a short prompt never waits for a long
  one's whole prefill; a prompt's last chunk asks for its last row's logits (its first token is sampled in the
  same step); (4) `rq_sample_batch` over the rows with logits, each with its job's temperature / top-p / token
  mask and its own sequence's RNG; then per job what Step 9's loop did after a sample (output, logprobs, stop
  strings, end token, max tokens, context end, constraint advance, the next forced tokens and mask). A single
  decode row runs `rq_decode` (the decode path). Interruptions (client gone, deadline, shutdown past the grace
  period) are checked per job before every step; a failed pass is re-run segment by segment (the passes
  validate everything before computing), so only the failing job gets its 500. The engine holds no lock during a
  pass; connection threads see output under each job's mutex as before
- [x] Flags (`runq serve`): `--parallel` 16 (jobs running at once; 10a: aggregate throughput grows to N ~ 16 at 8
  threads, ~32 at 12-14, per-sequence speed falls: 16 keeps ~75 tok/s per stream), `--slots` (default parallel +
  4: the running jobs' sequences plus idle prefix caches), `--kv-mem` GiB (default half the RAM: slots x `-c` x KV
  bytes per position at most; slots, then parallel, lowered to fit; reported at startup: `16 running at most, 20
  slot(s) of 4096 positions (KV 115 KB per position, 0.44 GiB per slot, 8.8 GiB at most, touched as used; --kv-mem
  15.3 GiB)`), `-b` 512 / `--chunk` 64 (prompt tokens per pass without / with decoding jobs), `--prefix-copy` 32
  (0: never), `--queue` 64 = jobs waiting beyond the free running places (a burst of `parallel` requests never
  sees a 503 while the engine is mid-pass). With the defaults (`-c 8192`) this 30 GB box gets 17 slots (0.88 GiB each, 14.9 GiB at most)
- [x] KV memory and prefix reuse: a pool of sequences; a running job owns one, idle ones keep their tokens. A new
  job takes the idle slot with the longest common prefix if it saves >= 32 tokens or half the prompt (Step 9's
  rule), else the least recently used idle slot (LRU eviction); if a busy slot shares a prefix >= `--prefix-copy`
  tokens longer, it is copied (`rq_seq_copy`) into the LRU idle slot; if a job being prefilled will share a
  longer prefix than anything cached, the new job waits until that part is in its slot and copies it then (a
  burst of LiteLLM classifications prefills the 253-token rubric once instead of 16 times). Measured: a cold burst
  of 16 classifications at once (0.6B Q4_K_M, 12 threads) 1.76-1.79 s with `--prefix-copy 0` (each prefills the
  rubric) vs 0.61-0.68 s with copies, then 0.41-0.42 s after making the copy fast: `rq_seq_copy` now runs on the
  thread pool and pre-faults the destination tiles (`kv_populate_one`, as the prefill does): 253 positions into a
  fresh sequence 14.5 -> 1.6 ms, into a touched one 2.1 -> 0.6 ms (a prefill of them: ~110 ms). Context and
  errors per job: a request's tokens are capped to its sequence (max_tokens <= the rest of the context, forced
  runs checked before feeding), others are not affected
- [x] Measured while tuning (0.6B Q4_K_M, 12 threads, `scripts/bench_serve.py --only mix`: 8 streams of 256-token
  replies + 4 prompts of ~2,000 tokens): `--chunk` 32 / 64 / 128: the long prompts' time to first token 1,762 /
  1,386 / 1,224 ms, the streams' token gaps p99 48 / 66 / 122 ms, max 147 / 71 / 144 ms, the streams' tok/s
  meanwhile 423 / 412 / 395: 64 kept. Before the budget was shared (FIFO only), with `--chunk 32` the max gap was
  825 ms: when every stream ended at once, nothing decoded, so the long prompt took the full `-b` 512 budget and
  the streams' new short prompts waited for its whole prefill. `--parallel` 16 vs 32 at 32 clients (128 tokens
  each): 1,139-1,206 vs 1,218-1,250 tok/s, time to first token p50 1.7-1.8 s (half of them queued) vs 0.22-0.24 s;
  16 kept as the default (memory: 32 slots of 8,192 positions would be 29 GiB here)
- [x] Benchmark (`scripts/bench_serve.py`, new: stdlib Python, a thread and a connection per client, a fresh
  server per load): streamed requests, a distinct short prompt each (`Write a long, detailed story about ...`, 22
  tokens), 128-token replies, temperature 0; 2 requests per client up to 4 clients, else 1; before = `b0d9f35`'s
  binary (Step 10a: the FIFO engine) built in a scratch worktree, interleaved, 2 rounds, best by aggregate tok/s,
  12 threads, load 13-19 (2026-10-26). Aggregate tok/s, request latency p50, time to first token (first content
  delta) p50 / p95:

  | Qwen3-0.6B Q4_K_M | 1 | 2 | 4 | 8 | 16 | 32 clients |
  |---|---|---|---|---|---|---|
  | before: tok/s | 265 | 250 | 290 | 289 | 298 | 297 |
  | after: tok/s | **309** | **460** (1.8x) | **740** (2.6x) | **914** (3.2x) | **1,185** (4.0x) | **1,169** (3.9x) |
  | latency p50 before / after (s) | 0.46 / 0.41 | 0.96 / 0.56 | 1.64 / 0.69 | 2.14 / 1.12 | 3.97 / 1.72 | 6.95 / 3.45 |
  | TTFT p50 / p95 before (ms) | 15 / 28 | 525 / 605 | 1,233 / 1,290 | 1,781 / 3,099 | 3,546 / 6,052 | 6,584 / 12,444 |
  | TTFT p50 / p95 after (ms) | 16 / 26 | 28 / 30 | 41 / 55 | 82 / 92 | 97 / 103 | 1,671 / 1,787 |
  | tok/s per request after | 323 | 239 | 195 | 123 | 78 | 76 |

  | Qwen3-0.6B Q8_0 GGUF | 1 | 2 | 4 | 8 | 16 | 32 clients |
  |---|---|---|---|---|---|---|
  | before: tok/s | 201 | 217 | 219 | 198 | 193 | 205 |
  | after: tok/s | **184** | **299** (1.4x) | **577** (2.6x) | **911** (4.6x) | **1,223** (6.3x) | **1,219** (5.9x) |
  | latency p50 before / after (s) | 0.63 / 0.67 | 1.21 / 0.89 | 2.20 / 0.89 | 3.20 / 1.12 | 6.45 / 1.67 | 10.15 / 3.30 |
  | TTFT p50 / p95 before (ms) | 14 / 30 | 628 / 633 | 1,660 / 1,845 | 2,615 / 4,517 | 5,747 / 9,641 | 9,508 / 18,099 |
  | TTFT p50 / p95 after (ms) | 17 / 25 | 25 / 31 | 40 / 55 | 78 / 90 | 79 / 103 | 1,643 / 1,741 |

  One client: the same within noise (4 interleaved rounds of 4 requests, Q8_0: 165-197 before, 175-185 after).
  32 clients: 16 run, 16 wait (`--parallel`), hence the TTFT. The decode rate at 16 streams (~75-80 tok/s each,
  ~1,200-1,280 in all) is the library's (`bench_batch -N 16`, 12 threads, the same hour: 1,124-1,157): the server
  adds no measurable overhead per token. Mixed load (8 streams of 256-token replies, 4 prompts of ~2,000 tokens
  one after the other): before, a long prompt waited for the queue and then stalled every stream for its whole
  prefill; after, it is prefilled 64 tokens per pass next to the streams:

  | mixed load, 2 rounds | long prompt TTFT p50 / max (ms) | stream token gaps p50 / p99 / max (ms) | stream tok/s meanwhile |
  |---|---|---|---|
  | Q4_K_M before | 7,353-7,368 / 7,532-7,826 | 3 / 8 / 7,515-7,772 | 261-262 |
  | Q4_K_M after | 1,370-1,409 / 1,410-1,438 | 8-9 / 64-70 / 69-88 | 414-427 |
  | Q8_0 before | 10,428-10,931 / 10,833-12,083 | 5 / 11-12 / 11,206-11,694 | 175-183 |
  | Q8_0 after | 1,293-1,304 / 1,305-1,322 | 10 / 62-63 / 72 | 411-416 |

  LiteLLM-judge-like load: classifications as LiteLLM 1.105 sends them (the legacy rubric as the system message,
  `\nClassify this message:\n` + one of the 80 prompts of `scripts/router_eval.jsonl`, the TierClassification
  schema, temperature 0), 64 per configuration after one warm-up (the rubric cached), then requests/s and
  latency p50 / p95 (ms); first a cold burst on the fresh server (16 at once: wall / p50 / p95). Best of 2 rounds,
  final binary, interleaved, 12 threads:

  | judge load | 1 client | 4 | 8 | 16 | cold burst of 16 (ms) |
  |---|---|---|---|---|---|
  | 0.6B Q4_K_M before | 45.4/s, 19 / 39 | 46.6/s, 79 / 139 | 45.3/s, 174 / 230 | 46.5/s, 334 / 407 | 473-525 / 298-330 / 412-460 |
  | 0.6B Q4_K_M after | **47.6/s**, 18 / 39 | **62.0/s**, 56 / 132 | **64.4/s**, 122 / 211 | **62.4/s**, 253 / 289 | 413-427 / 405-415 / 410-418 |
  | 0.6B Q8_0 before | 49.3/s, 17 / 38 | 51.0/s, 77 / 130 | 52.0/s, 157 / 224 | 51.2/s, 311 / 363 | 457-482 / 298-313 / 429-452 |
  | 0.6B Q8_0 after | **51.0/s**, 18 / 41 | **64.5/s**, 55 / 111 | **68.3/s**, 114 / 193 | **68.9/s**, 233 / 272 | 408-414 / 390-402 / 395-408 |
  | 1.7B Q8_0 before | 22.8/s, 39 / 84 | 24.4/s, 151 / 283 | 23.9/s, 320 / 527 | 23.6/s, 676 / 781 | 954-1,002 / 614-665 / 896-905 |
  | 1.7B Q8_0 after | **24.7/s**, 37 / 76 | **29.6/s**, 121 / 243 | **33.7/s**, 246 / 357 | **30.5/s**, 517 / 607 | 795-821 / 786-800 / 792-808 |

  The judge load gains much less than generation (1.3-1.4x): a classification is ~30-60 uncached prompt tokens
  (the user message, the template, the forced `{"tier": "`) and no decode step at all (the decision is sampled
  from the prompt's last row, the rest is forced and never fed), so it is prefill-bound already; batching helps
  because several small prefills make one larger pass (a 16-row pass runs at ~1,400-1,700 tok/s, 64 rows at
  ~1,700-2,350: Step 10a's mixed-pass table). In a cold burst every job completes in the same pass, so p50 rises
  (~300 -> ~400 ms) while the wall time and p95 fall
- [x] Bit-identity: each job has its own sequence (KV cache, RNG), every row's logits are what its sequence
  alone would give (runq.h; Step 10a) and `rq_sample_batch` == `rq_sample` row by row, so a request's tokens do not
  depend on what it is batched with, the chunking of its prompt or the pass it joined. Tests (`tests/server_test.py`,
  group `batch`; the server at `--parallel 4 --slots 6 --chunk 16`): 7 requests at once (constrained greedy with
  20 logprobs, two sampled classifications, a sampled nested schema, free greedy with logprobs, a stream with a
  stop string, seeded with logprobs) == each alone (text and logprobs, 9 significant digits); three requests
  (greedy with a system prompt, seeded, thinking) joining a running stream == `runq -m chat`; a client leaving a
  long sampled stream while three others run: theirs == `runq -m chat`, the cancel logged 499; four requests
  sharing a new 90-token system prompt at once (wait + copy: cached 91 / 1 / 91 / 91) == the same with
  `cache_prompt: false`. The queue / cancel / prefix-reuse checks of Step 9 still pass (with 4 running).
  Breakage: one job's sampling settings for every row (`rq_sample_batch(..., ncfg = 1)`) fails the batch checks.
  `make test` 427 -> 430 checks (server 48 -> 52: +4 batch; kernels -1, below)
- [x] `make test` time: ~33 s -> ~28.5 s (interleaved, the same checks in the previous order vs now: 32.5 / 33.0 /
  32.6 vs 28.6 / 28.3 / 28.7 s at load 18-31; Step 10a's suite 31.5-36 s). The suite is bound by CPU time (~360
  core-seconds in ~310 processes; each runq start costs ~0.5-0.9 core-s of weight loading at 8 threads), and a
  timeline of the processes' CPU (a wrapper logging rusage) showed the first ~16 s at 13-15 busy cores but the
  last ~5 s at 1-5: the greedy runs, the chat cases, the GGUF template and Q4_K_M dequant runs only started at
  their group's turn, and the API tests ran on one worker in a ~31 s serial chain. Now all of them start with
  the suite (their results are checked in the usual order), the API tests on two workers. Also dropped: the
  generic build's `long -b 64` run (`kernels/long_b64`; `long2` = `long` + `all` + `code` takes the same paths and
  is checked too, ~6 core-s) and `batch/concurrent_eq_cli` (the join / cancel checks run the same requests
  concurrently). No other coverage change
- [x] Sanitizers (scratch builds of all sources: `-O1 -g -fsanitize=address,undefined -fopenmp`; `-fsanitize=thread`
  without OpenMP, the pool / engine / connection threads instrumented; reports to files with `log_path`):
  `server_test.py --slow` 52-53 / 52-53 under ASan + UBSan on the Q8_0 GGUF, Q4_K_M and the .bin, and under TSan
  on the Q8_0 GGUF; plus `bench_serve.py` against both (ASan: 8 / 24 clients at `--chunk 16 --slots 18`, a cold
  judge burst of 16 with copies, the mixed load; TSan: 12 clients at `--parallel 8 --queue 8`, a cold burst of 8):
  no reports, no leaks at exit
- [x] Verification: `tests/compare_bins.sh -j 4 /tmp/cwref .` (CLI outputs vs the pre-Step-8 binaries, 360 jobs):
  all 360 jobs identical (294 s; `rq_seq_copy`, the only library change, is not used by the CLI). Builds: `-Wall -Wextra` on engine.c / serve.c: no new warnings
- [x] End to end (`scripts/litellm_e2e.sh`, own scratch LiteLLM 1.105.0 proxy as in Step 9, now with a second pass
  of 16 requests at once through the proxy): Q8_0 GGUF judge and tier models on one
  runq serve (8 threads, defaults: 16 running, 17 slots of 8,192 positions within the default `--kv-mem` 15.3 GiB),
  8 prompts one at a time as before (judge 161 ms cold, then 15-26 ms with the rubric cached; end to end 99-185 ms),
  then 16 at once: all 16 classified and routed (tier-complex x4, tier-medium x8, tier-simple x4), every tier ==
  the sequential pass's for the same prompt, 715-835 ms for all 16 end to end (164-828 ms each), judge latency
  p50 / max 93-166 / 186-210 ms (two runs); LiteLLM's classifier prompt and response_format still byte-identical
  to its own builders
- Open: the KV pool is fixed (no paging, no preemption: a request keeps its slot to the end; `--kv-mem` bounds
  the worst case, so with long contexts fewer requests run); a shared prefix is copied, not shared (paged KV
  would let concurrent requests read one copy of the rubric); in a cold burst all deferred jobs finish in one
  pass (p50 up, p95 down); the decision step of a classification still computes the full classifier row (Step 9's
  subset classifier); `json_object` / free strings in schemas; logprobs with streaming; per-request priorities;
  the HTTP layer is one thread per connection (fine up to `--max-conns` 128)

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
- [x] Step 6b-1: GGUF Q4_K_M correct (loader, repacked layouts, arithmetic, generic kernels, tests): done, see above
- [x] Step 6b-2: AVX-512 kernels for Q4_K / Q6_K (GEMV at the bandwidth limit, GEMM ~1.17x Q8_0's): done, see above
- [x] Step 7a: threading (phase sync on a thread pool, 113 sync points per decode step, no 16-thread
  collapse, attention blocks of 256 positions with the decode split over them): done, see above
- [x] Step 7b: 16-bit KV cache (f16 default, bf16, f32), cache layout per (block, head), quantizer
  vectorized, prefill attention / rest analysed: done, see above

Steps 4-7 are done. Next plan (2026-10-26), from IDEAS.md (multi-sequence decode, `runq serve` as the
LiteLLM auto-router judge) and the user's request: clean up first, then both features, then other models.
IDEAS.md facts outdated since it was written: the KV cache is f16 by default (115 KB per position, not
229), Step 7 has landed and the Mac port is merged (NEON Q8_0 kernels exist too). The LiteLLM proxy on
this box runs 1.105.0 (`/opt/litellm/venv-1.105.0`; never touch the running proxy or its config).
- [x] Step 8: cleanup / architecture for batched decode and the server, no change in results:
  model (weights, read-only, shared) vs sequence state (KV cache, position, tokens, sampler) vs
  scratch; a request-level API (tokens in, logits / tokens out, callbacks instead of printing);
  errors returned instead of `exit()` in request paths; runq.c split into modules; docs updated: done,
  see above
- [x] Step 9: `runq serve`: OpenAI-compatible HTTP server (chat completions incl. streaming, models,
  prefix KV reuse across requests, `response_format` json_schema with finite outputs by constrained
  decoding, FIFO), tests, end-to-end with a scratch LiteLLM instance, judge-quality evaluation: done, see above
- [x] Step 10a: multi-sequence (batched) decode in the library: small-batch kernels, a decode-rows
  worker, attention for decode rows, batched sampling, benchmarks, tests: done, see above
- [x] Step 10b: continuous batching in `runq serve` (a scheduler: every running job's decode row and prompt
  chunks in one pass per step, a slot pool with prefix reuse / copies, benchmarks, bit-identity tests): done,
  see above
- [ ] Step 11 (next): other models that should run on this engine (Qwen3 dense family: 1.7B, 4B, 8B; other
  GGUF quant types of them), pinned downloads, correctness vs HF, speed. (`runq serve` open items after Step 10b:
  the end of its section: paged / shared KV, priorities, the subset classifier, json_object)

## Later / ideas
- [x] 16-thread collapse: phase synchronization that waits for the work, not for every thread, on a
  thread pool, spin then futex (Step 7a)
- [x] 16-bit K/V cache (Step 7b: f16 default, bf16; the arithmetic stays fp32 on converted values).
  The split over positions for decode: done (Step 7a, blocks of 256)
- [ ] Prefill attention with `vdpbf16ps` (2x the dot-product rate; prefill attention is compute-bound at
  ~590-710 GFLOPS of ~900): needs q and the softmax weights in bf16 (accuracy: the logits vs HF test and
  KL would decide; bf16 K alone is +2.6% KL) and a C twin of the instruction's exact rounding verified
  on the target CPUs (Step 7b notes)
- [ ] KV cache K f16 + V bf16 (halves the f16 conversion cost of the default; V's bf16 rounding +0.7% KL)
- [ ] Decode at long contexts: re-measure the VM's bandwidth limit, then decide whether the bandwidth or
  the attention items (8 kv heads x blocks, after the qkv items) bound it (Step 7b notes)
- [x] Decode: fewer barriers (Step 7a: 4 phases per layer, epilogues instead of `single` sections);
  q / k / v items at 12 threads: shared-counter claims of single heads, no idle tail
- [x] Prefill rest (Step 7b): the quantizer vectorized (same results); the phases are memory-traffic
  bound (rows written by other cores), a fused SwiGLU epilogue was slower; SwiGLU's `expf` is ~0.75%
  of a 451-token prefill (not ~4%: that was the whole phase): left. ~11% at 451 tokens, 5% at 3,000
- [x] Perplexity comparison Q4_K_M vs Q8_0 (2026-10-25, `scripts/ppl.py`, BENCHMARKS.md): Voyage of
  the Beagle ch. I, 4,142 tokens, one sequence (= `mynah-slm ppl`): HF fp32 27.571, mynah-slm Q8_0
  27.580 (+0.03%), runq Q8_0 27.805 (+0.85%: int8 activations, group 64), mynah-slm Q4_K_M 28.864
  (+4.7%); 512-token windows: 31.984 / 32.039 / 32.206 / 34.185 (+6.9% for Q4_K_M). Step 6a: runq
  on the Q8_0 GGUF (group 32) 27.623 (+0.19%), windows 32.040 (+0.18%)
- [x] GGUF loading without the Python export step: Q8_0 done (Step 6a); Q4_K_M Step 6b-1 (runq on it:
  perplexity 28.911 / win512 34.190, mynah-slm 28.864 / 34.185)
