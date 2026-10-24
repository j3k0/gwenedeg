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
- [ ] Test hooks in runq: dump token ids (tokenizer) and raw logits per prompt position
- [ ] Golden references generated once from HF transformers (fp32), committed (compact)
- [ ] Tokenizer test: ids match HF on varied strings (ASCII, Unicode, code, whitespace, special tokens)
- [ ] Logits test: Q8_0 logits vs HF fp32, cosine and top-1 agreement within tolerance
- [ ] Greedy regression test: fixed prompts, `-t 0`, exact output vs our own golden
- [ ] `make test` runs all of them

## Step 1: skip the classifier during prefill
- [ ] Compute logits only for the last prompt token (and during generation)
- [ ] Tests pass; measure prefill gain

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

## Later / ideas
- [ ] Chat history: stop resetting to position 0 every turn (keep the KV cache across turns)
- [ ] Fix the 16-thread collapse (OpenMP spin barriers under oversubscription)
- [ ] Perplexity comparison Q4_K_M vs Q8_0 (small text sample)
- [ ] GGUF loading (Q8_0 / Q4_K_M) without the Python export step
- [ ] Faster sampling (top-p currently sorts the whole vocab)
