# TODO

Goal: fast Qwen3-0.6B inference in C on CPU, based on qwen3.c (Q8_0).
Baseline (16-vCPU EPYC Genoa, 8 threads): prefill ~85 tok/s, decode ~100 tok/s.
Target: prefill several hundred to 1000+ tok/s; decode limited by memory bandwidth.

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

## Step 2: batched prefill
- [ ] `forward_batch(tokens, n, pos)`: chunked (32 to 128 tokens), activations as matrices
- [ ] Batched quantize, GEMM (generic C first), per-row RMSNorm, QK-norm and RoPE, causal attention inside the chunk, KV cache writes
- [ ] Test: batched logits match sequential logits (near-exact)
- [ ] Use it in generate and chat; measure

## Step 3: AVX-512 VNNI int8 kernels
- [ ] GEMM kernel (`vpdpbusd`) with cache tiling, for prefill
- [ ] GEMV kernel for decode
- [ ] Threading: fewer barriers per token, parallelize over rows
- [ ] Keep a portable fallback (non-AVX-512); tests compare both
- [ ] Measure against the memory-bandwidth limit (~87 GB/s at 8 threads)

## Known bugs
Found by the Step 0b tests; not fixed yet.
- [ ] Tokenizer: no pre-tokenizer (Qwen's split regex), BPE merges run over the whole text.
  A whitespace run before a word is merged whole, while HF splits the last space off onto the
  word (`"   leading"`, indented code, `"a\t\tc"`). Tests: XFAIL `tokenizer/{leading_spaces,
  multi_spaces,tabs,code_c,code_python}`. Encoding is also O(n^2 * vocab) (linear `str_lookup`).
- [ ] SwiGLU `1/(1+expf(-x))` under `-Ofast` (finite-math): `expf` overflow gives NaN. Not seen in
  the shipped int8 path yet, but a scratch build with float activations got whole NaN logits
  rows; clamping the exponent fixed it. Risk for the Step 2/3 kernels (the logits test checks
  for NaN).
- [ ] Minor: `quantize()` divides by a zero scale on an all-zero group (NaN cast to int8 is UB);
  `encode()` prints its unknown-character warning to stdout and leaves an uninitialized token.

## Later / ideas
- [ ] Chat history: stop resetting to position 0 every turn (keep the KV cache across turns)
- [ ] Fix the 16-thread collapse (OpenMP spin barriers under oversubscription)
- [ ] Perplexity comparison Q4_K_M vs Q8_0 (small text sample)
- [ ] GGUF loading (Q8_0 / Q4_K_M) without the Python export step
- [ ] Faster sampling (top-p currently sorts the whole vocab)
