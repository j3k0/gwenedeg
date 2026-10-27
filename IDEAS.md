# Ideas

Possible directions for Gwenedeg that nobody has committed to: each would need a measurement or a design decision
before it becomes work. Concrete next steps are in [TODO.md](TODO.md). Numbers are from the 16-vCPU AMD EPYC Genoa
VM of [BENCHMARKS.md](BENCHMARKS.md) with Qwen3-0.6B unless noted. Every idea has to keep the project's rule: a
sequence's logits never depend on the path, the thread count or what it is batched with.

## Speculative decoding by prompt lookup

Draft the next few tokens by finding the last n-gram of the sequence earlier in the context (prompt lookup: no
draft model), then verify them in one pass: the sequence's last token plus the k drafted tokens as one segment of
k + 1 rows, with logits for every row. Accept drafts while they equal the token sampled at their position.

- Why: decode is memory-bound, so a pass over a few rows costs much less than that many single-row steps. Rough
  figures from the batched-decode measurements (0.6B Q4_K_M, 8 threads, 500 positions): 4 decode rows of
  different sequences take ~1.9x one row (7.8 vs 4.2 ms). Rows of one sequence share its KV cache, so should be
  cheaper still. Measure first: a multi-row segment takes the prefill path (9 phases per layer, not the decode
  path's 4).
- Exactness for free: the verification rows' logits are bit-identical to plain decoding (batched == sequential).
  If every position is sampled with the sequence's own RNG in order, and the pass stops at the first mismatch,
  then greedy and seeded outputs equal non-speculative decoding exactly. The price is a lower acceptance rate than
  rejection sampling.
- Gain depends on how repetitive the output is (code edits, quoting, extraction, JSON echoing the prompt). In
  `runq serve` the draft rows would simply enlarge the job's segment in the batched pass.
- Qwen3.5 ships its own draft model: a multi-token-prediction head (`mtp.*` in the HF checkpoint, 15 tensors; not in
  the Unsloth GGUFs, docs/qwen35.md). On a hybrid model a rejected draft also needs the recurrent state from before
  it: a checkpoint per verification pass (18.9 MB for 0.8B), or verification rows that keep the state per position.

## A classification endpoint that returns label probabilities

`runq serve` already answers a constrained classification in one prefill: forced tokens are never decoded, and the
decision step's masked softmax gives the probability of each label (`scripts/eval_router.py` sums it per label).
An endpoint could return that distribution directly, without generated text, e.g. a request with a prompt and a
list of choices, answering with per-choice probabilities. LiteLLM's auto-router docs describe one such protocol
("System One", `POST /v1/systemone`); check its current specification before implementing. Combined with the
subset classifier of TODO.md, such a call would cost little more than the prefill of the new tokens.

## Shared (paged) KV cache between slots

A shared prompt prefix (e.g. a judge's rubric) is copied into each request's slot today (`rq_seq_copy`: 253
positions in 0.6-1.6 ms), and the slot pool is fixed (`--slots` x `-c`, bounded by `--kv-mem`; no paging, no
preemption). Copying is cheap in time, so sharing would gain memory: 115 KB per position for the 28-layer models,
147 KB for the 36-layer ones (f16). The cache is already laid out in blocks of 256 positions, so a natural design
is reference-counted, copy-on-write blocks with per-sequence block tables. That would allow more concurrent
requests for the same memory, and preemption (a slot swapped out under memory pressure). The attention reads the
same values in the same order, so results would stay bit-identical. The cost is an indirection per block in every
attention path (AVX-512, NEON and generic).

## Scheduling: request priorities

Admission is FIFO, up to `--parallel` running requests. A priority (or separate limits) could let short
interactive calls, such as router classifications, pass long generations, and could decide which request yields
its slot when memory runs out (with the previous idea). Related effect: in a cold burst, all requests deferred
behind a shared prefix finish in the same pass, so the median latency rises (~300 -> ~400 ms for 16 at once)
while the wall time and p95 fall.

## int8 (or 8-bit float) KV cache

At long contexts, batched decode is bound by KV traffic: each row reads 115 KB per position per token, so at
2,000 positions aggregate throughput stops at ~350-480 tok/s from 8 sequences on, whatever the batch size. An
8-bit cache would halve that again. Accuracy is the question: |K| reaches ~500, and bf16 K alone (8 significant
bits) already costs +4.4% KL against HF fp32. Per-tile scales, or 8 bits for V only (V's rounding was harmless in
bf16), would be the things to try, with the KL measurement used for the 16-bit types.

## bf16 dot products in prefill attention (`vdpbf16ps`)

Prefill attention is compute-bound at long prompts (~590 GFLOPS of a ~900 FMA peak at 8 threads; PV ~710, scores
and softmax ~500), and `vdpbf16ps` doubles the dot-product rate. But q and the softmax weights would have to be
rounded to bf16 too: the logits-vs-HF test and the KL measurement would decide whether that is acceptable. The
instruction's exact rounding and denormal behaviour (two FMAs per pair, DAZ / FTZ, the odd element first in
Intel's pseudocode) would need a C twin for the generic build, verified on each target CPU. Without that, AVX-512
and generic results would differ.

## Qwen3 MoE models

The `qwen3moe` architecture (e.g. Qwen3-30B-A3B) adds a router and expert FFNs. Decode reads only the active
experts' weights, which suits a memory-bound CPU engine. Batched passes would group rows by expert, and each row's
result must stay independent of the grouping (the expert matmuls per row as now). The files are large (~18 GB at
Q4_K_M for 30B-A3B), and the GGUF loader, the weight repacking and the tests would all grow.

## Images (Qwen3.5 vision)

Out of scope for the project (TODO.md), noted in case it is wanted. The Qwen3.5 / 3.6 models are multimodal. In the
GGUF ecosystem the vision part is a separate **mmproj** file (0.19 GB for 0.8B to 0.93 GB for 27B; architecture
`clip`, projector `qwen3vl_merger`); the text GGUF is unchanged. It holds a ViT (depth 12-27, patch 16, temporal patch
2, learned position embedding interpolated + 2-D RoPE, gelu-tanh MLP) and a 2x2 patch merger to the text width.
The merged patch embeddings replace the `<|image_pad|>` tokens (the template wraps each image in
`<|vision_start|><|image_pad|><|vision_end|>`; the processor expands the pad to (H/32) x (W/32) tokens).

Positions are 3-D (interleaved MRoPE, sections 11 / 11 / 10): an image token i at `t = p0, x = p0 + i % nx,
y = p0 + i / nx`, and the image advances the position by `max(nx, ny)`; text has t = h = w, which is today's plain
partial RoPE. Work: an image decoder + resize / normalize; the ViT + merger forward (its own kernels: patch conv as a
matmul, bidirectional attention, 2-D RoPE, LayerNorm with bias, gelu); a position counter per sequence apart from
the index (through `rope_row`, the KV cache, the checkpoints and the sequence API); embeddings as input in the batch
API; the server's OpenAI image content parts; the template's image items; HF goldens with images. Several weeks.
Notes: `docs/qwen35.md` (the text model only), llama.cpp's `mtmd` / `clip` and `conversion/qwen3vl.py` as the
reference for the layout.
