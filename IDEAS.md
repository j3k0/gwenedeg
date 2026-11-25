# Ideas

Possible directions for Gwenedeg that nobody has committed to: each would need a measurement or a design decision
before it becomes work. Concrete next steps are in [TODO.md](TODO.md) (prompt-lookup drafting, an 8-bit KV cache and bf16
dot products in prefill attention moved there: the section on agentic development at 96k context). Numbers are from the 16-vCPU AMD EPYC Genoa
VM of [BENCHMARKS.md](BENCHMARKS.md) with Qwen3-0.6B unless noted. Every idea has to keep the project's rule: a
sequence's logits never depend on the path, the thread count or what it is batched with.

## A classification endpoint that returns label probabilities

`gwenedeg serve` already answers a constrained classification in one prefill: forced tokens are never decoded, and the
decision step's masked softmax gives the probability of each label (`scripts/eval_router.py` sums it per label).
An endpoint could return that distribution directly, without generated text, e.g. a request with a prompt and a
list of choices, answering with per-choice probabilities. LiteLLM's auto-router docs describe one such protocol
("System One", `POST /v1/systemone`); check its current specification before implementing. Combined with the
subset classifier of TODO.md, such a call would cost little more than the prefill of the new tokens.

## Shared (paged) KV cache between slots

A shared prompt prefix (e.g. a judge's rubric) is copied into each request's slot today (`rq_seq_copy`: 253
positions in 0.6-1.6 ms), and the slot pool is fixed (`--slots` x `-c`, bounded by `--kv-mem`; no paging; since this was written
`--admit bound` / `chunk` admit requests on leases and `chunk` can swap a request out to disk, [docs/memory-admission.md](docs/memory-admission.md),
but blocks are still not shared between slots). Copying is cheap in time, so sharing would gain memory: 115 KB per position for the 28-layer models,
147 KB for the 36-layer ones (f16). The cache is already laid out in blocks of 256 positions, so a natural design
is reference-counted, copy-on-write blocks with per-sequence block tables. That would allow more concurrent
requests for the same memory, and preemption (a slot swapped out under memory pressure). The attention reads the
same values in the same order, so results would stay bit-identical. The cost is an indirection per block in every
attention path (AVX-512, NEON and generic).

## Scheduling: request priorities

Admission is FIFO, up to `--parallel` running requests (`--admit chunk --admit-yield N` adds a round-robin of the running
places, [docs/memory-admission.md](docs/memory-admission.md); there is still no priority). A priority (or separate limits) could let short
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

## V4H: a VNNI-native 4-bit weight and activation format

Full analysis, layout, arithmetic proof and prototype plan: [docs/v4h.md](docs/v4h.md).

**Prototyped 2026-11-24** ([docs/v4h.md section 9](docs/v4h.md#9-prototype-results-2026-11-24)). On Qwen3.5-0.8B against
BF16, V4H with an imatrix has KL 0.092 / 0.050 (wiki / code): half llama.cpp IQ3_S's 0.180 / 0.115, but worse than
IQ4_XS's 0.061 / 0.041. The activation subscale is unnecessary: even one activation scale per 256 costs KL ≤ 0.004. The
kernels (`make v4h-bench`) are bit-identical to the definition. A tile decoded to `s q` bytes runs 1.34-1.38x faster than
IQ3_S/r, but the existing IQ3_S/r and IQ4_XS tiles with one activation scale per 256 already get 1.17-1.25x. Next step: that
activation format for the existing types, measured on the 27B. V4H itself only with a better quantizer.
[section 10](docs/v4h.md#10-flat-activations-for-the-existing-types-kernel-measurement-2026-11-24) measured it per type
(`make flat-bench`): 1.21-1.23x on the 27B file's i-quant GEMMs at 512 rows, weighted by type, small batches unchanged.
[section 11](docs/v4h.md#11-implemented-as---act256-2026-11-24): implemented as the opt-in `gwenedeg --act256` (commit
6a35a07), all paths bit-identical; Qwen3.5-0.8B prefill 1.14x / 1.11x, KL vs the default <= +0.004. On the 27B (2026-11-24,
the day server stopped): prefill 1.16-1.18x at 1.5k and 8k prompts (103 / 100 -> 122 / 117 tok/s), decode
unchanged, KL vs the default 0.0030 / 0.0014, PPL x1.004 / x1.00.
[section 12](docs/v4h.md#12-flat-activations-for-the-k-quants-kernel-measurement-2026-11-25): the K-quants, kernels only
(`build/flat_bench -kb`, bit-identical to their definition). At 512 rows: Q3_K 1.22-1.29x (int16-pair scales), Q6_K
1.14-1.20x, Q4_K / Q5_K 0.98-1.10x. On Zen 4 their flat form saves only 8 of 88 multiply-class ops. Weighted: the 27B
Q3_K_M 1.16-1.19x (passes the 1.15 gate), the Q4_K_M 1.02-1.12x (fails).
[section 13](docs/v4h.md#13-the-k-quant-kernels-in-the-engine-2026-11-25): in the engine (`--act256` on K-quant files;
NEON: the generic kernel). Qwen3.5-0.8B prefill: Q3_K_M 1.10x, Q4_K_M 1.04x. The flag's quality cost is unchanged and
per file: KL against the Q8_0 file +1-2% (Q3_K_M) / +6-8% (Q4_K_M) of the file's own. Qwen3.8-27B Q3_K_M (the day server
stopped): prefill 1.13-1.15x at 1.5k and 8k tokens, decode unchanged.

`vpdpbusd` still needs one byte per weight, so a lower bit width alone cannot raise its compute peak. A format co-designed with the
kernel could nevertheless beat the i-quants by making nibble expansion and scaling cheaper. V4H uses 16-row x 256-column
superblocks: 4-bit codes `u = q + 8` (`q = -8 .. 7`), an fp16 `D` per row and superblock, and a 4-bit unsigned subscale `s`
per row and group of 32. Activations are int8 `z = -127 .. 127`, with an fp32 `A` per 256 values and a u8 subscale `a` per 32.
The definition is `w = D s q`, `x = A a z`, hence `y += D A * sum_g(s_g a_g * sum_j(q_j z_j))`. Weight storage is 2048 + 64 +
32 = 2144 bytes per 16 x 256 values, the same 4.1875 bits/weight and physical D / scales / nibbles arrangement as IQ3_S/r.
This is a new quantization, not a lossless repack, and needs a quantizer plus KL / perplexity evaluation.

The nibbles decode directly with `and`, `shift`, `and`, without IQ3_S/r's `vpshufb`, codebooks or sign handling; the fixed bias
uses the activation sums as now. For each group of 32, `T = sum(q z)`, `C = s a`, and an int32 accumulator takes `I += C T`;
there is one float fold per 256 columns instead of eight. The ranges fit the other VNNI instruction exactly: `|T| <= 32512`,
`C <= 3825`, and `|I| < 1 billion`. With C's high 16 bits zero, `vpdpwssd(I, T, C)` multiplies the complete signed low-16-bit
T by C and accumulates it, with no narrowing or packing (the high-T product is multiplied by zero). Compare that with
`vpmulld + vpaddd`: the former uses fewer instructions but may compete for the same execution ports as `vpdpbusd`.

The target is the measured per-32-fold overhead in docs/iquants.md: one-scale tiled GEMMs run at 0.0038-0.0039 ns per weight
and activation row, versus 0.0031 in the diagnostic without the fold, so about 1.25x kernel speed is the plausible headroom,
not another 4x. Direct nibbles should matter most to GEMV and small batches; tiled prefill already amortizes decoding. At the
memory-bound end V4H has the same traffic as IQ3_S/r and should not beat it merely by having a simpler kernel. V4-flat (no
subscales, one scale per 256 or per row) is the speed-ceiling experiment; V3H (the same hierarchy with 3-bit codes, 3.1875
bits/weight) is the DRAM-bound-decode variant, with more unpacking and unknown quality.

Prototype from `iq512_group` / `iqx_tile` in `src/iquants.h`, with the Q8P register-tile rules in `src/kernels_avx512.h`. Benchmark
both integer scale-and-accumulate forms at 1, 4, 8, 64 and 512 activation rows against IQ3_S/r. Every build needs the same
arithmetic definition and rounded float operations. Open: quantization quality, the activation subscale representation, whether
`vpdpwssd` wins on Zen 4, and whether any end-to-end gain survives the memory-bound phases.
