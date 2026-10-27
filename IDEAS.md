# Ideas

Possible directions for Gwenedeg that nobody has committed to: each would need a measurement or a design decision
before it becomes work. Concrete next steps are in [TODO.md](TODO.md) (prompt-lookup drafting, an 8-bit KV cache and bf16
dot products in prefill attention moved there: the section on agentic development at 96k context). Numbers are from the 16-vCPU AMD EPYC Genoa
VM of [BENCHMARKS.md](BENCHMARKS.md) with Qwen3-0.6B unless noted. Every idea has to keep the project's rule: a
sequence's logits never depend on the path, the thread count or what it is batched with.

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
