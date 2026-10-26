# Qwen3.5 support: design notes and plan

Status (2026-10-26): **prepared, not implemented** (the step list with "done when" criteria is in TODO.md, Q35-1 .. 7). runq loads `general.architecture == qwen3` only. This file
records what Qwen3.5 is, what was verified, the design decisions for runq, and the steps. Test model:
**Qwen3.5-0.8B**. It has the same layer types as the larger Qwen3.5 / 3.6 models, but its FFN is dense instead of
MoE, so it takes over the role Qwen3-0.6B has now as the fast golden-logit model for the new layers.

What is ready:

- `models/Qwen3.5-0.8B` (HF safetensors, bf16, with the tokenizer files) and `models/Qwen3.5-0.8B-GGUF`
  (Unsloth `Q8_0`, 812 MB, and `Q4_K_M`, 533 MB).
- `tests/ref_qwen35.py`: a numpy reference forward pass (the spec), on the GGUF's dequantized weights or on the
  HF checkpoint. On the HF weights it matches transformers 5.21 fp32 to 2-6e-5 max abs logit error
  (KL ~1e-11, top-1 100%) on 12-22-token prompts, so the token-by-token recurrence below equals HF's chunked
  prefill up to rounding.
- Goldens in the `tests/check_models.py` format: `tests/golden/models/3.5-0.8b-hf` (HF fp32,
  `make_golden.py --model-dir`) and `3.5-0.8b-q8` (the Q8_0 file's dequantized-weights reference,
  `make_model_ref.py`, which dispatches on the architecture). They are not wired into the `models` group yet
  (runq would fail to load the file). Q8_0 reference vs HF: 1-cos mean 0.9-2.8e-4, min cos 0.9991, top-1
  100% on en / code / chat / long2.
- `.venv-hf5` (git-ignored): transformers 5.21 / torch 2.7.0 CPU. The test venv `.venv` stays at transformers
  4.53, which does not know `qwen3_5`. The existing goldens were made with 4.53, and regenerating
  `1.7b-q8` after the `make_model_ref.py` change gave byte-identical files.

## The architecture (0.8B)

24 layers, dim 1024, SwiGLU FFN 3584, vocab 248,320, tied embedding / classifier, RMSNorm eps 1e-6. The blocks
keep the Qwen3 pre-norm shape: `x += mixer(rmsnorm(x)); x += ffn(rmsnorm(x))`. The mixer depends on the
layer: Gated DeltaNet in layers with `(l + 1) % 4 != 0` (18 of 24), gated full attention in layers 3, 7, ..., 23.

**Gated DeltaNet** (linear attention; 16 key heads = 16 value heads of 128):

```
qkv  = W_qkv h                     6144 = [q 16x128 | k 16x128 | v 16x128]
qkv  = silu(causal depthwise conv over the last 4 positions)    kernel (6144, 4) f32, no bias
q, k = l2norm per head (x / sqrt(sum x^2 + 1e-6)); q *= 1/sqrt(128)
beta = sigmoid(W_beta h)          (16)
g    = ssm_a * softplus(W_alpha h + ssm_dt)   (16), ssm_a = -exp(A_log) < 0
per head, state S (128 k x 128 v), per token:
    S *= exp(g);  delta = beta * (v - S^T k);  S += k delta^T;  o = S^T q
o    = rmsnorm(o per head, ssm_norm) * silu(W_z h)      (W_z: 2048 rows)
out  = W_out o                     (1024 x 2048)
```

**Gated attention** (8 query heads / 2 kv heads of 256): `W_q` gives `[q | gate]` per head (4096 rows). QK-norm
as in Qwen3. **Partial RoPE**: only the first 64 dims of each head rotate (NeoX pairs `(j, j + 32)`,
theta 1e7), and dims 64..255 pass through. Output: `W_o (attn * sigmoid(gate))`. The config declares
MRoPE (sections 11 / 11 / 10, interleaved). For text, the three position ids are equal, so it reduces to the
plain partial RoPE above. Images would need the 3-D positions, which are out of scope.

**Not part of the text model:** the vision tower (153 tensors) and the MTP head (15 tensors, `mtp.*`;
speculative decoding, see IDEAS.md). The Unsloth GGUFs contain neither: 24 blocks, no `nextn` tensors.

## The GGUF layout (verified against the safetensors)

Metadata `qwen35.*`: `block_count` 24, `embedding_length` 1024, `feed_forward_length` 3584,
`attention.head_count` 8 / `head_count_kv` 2, `attention.key_length` / `value_length` 256,
`rope.dimension_count` 64, `rope.freq_base` 1e7, `rope.dimension_sections` [11, 11, 10, 0],
`ssm.conv_kernel` 4, `ssm.state_size` 128 (head k dim), `ssm.group_count` 16 (k heads),
`ssm.time_step_rank` 16 (v heads), `ssm.inner_size` 2048 (v dim), `full_attention_interval` 4,
`context_length` 262,144. Tokenizer: `gpt2` model, `tokenizer.ggml.pre` = `qwen35`, eos 248046
(`<|im_end|>`), padding 248055.

Per linear layer: `attn_qkv` (1024 -> 6144), `attn_gate` (W_z, -> 2048), `ssm_alpha` / `ssm_beta`
(-> 16 each), `ssm_out` (2048 -> 1024), `ssm_conv1d` (F32, 4 x 6144), `ssm_a` (F32, 16), `ssm_dt.bias` (F32, 16),
`ssm_norm` (F32, 128). Per attention layer: `attn_q` (-> 4096: per head 256 q then 256 gate), `attn_k`,
`attn_v` (-> 512), `attn_output` (2048 -> 1024), `attn_q_norm` / `attn_k_norm` (256). Every layer:
`attn_norm`, `post_attention_norm` (the FFN norm; not `ffn_norm`), `ffn_gate` / `ffn_up` / `ffn_down`.

Value transforms by llama.cpp's converter, checked bit-exactly (`maxabs 0`) on layer 0 / 3:
- The norms are stored as `1 + w`: `attn_norm`, `post_attention_norm`, `output_norm`, `attn_q_norm`,
  `attn_k_norm`. HF's `Qwen3_5RMSNorm` computes `x * (1 + w)`, so runq's `x * w` on the stored weight is
  right as it is. `ssm_norm` has no `+ 1` in either.
- `ssm_a = -exp(A_log)` (1e-7 rounding).
- With 16 = 16 heads (0.8B), there is no head reordering. When `num_v_heads > num_k_heads` (larger models),
  the converter reorders the v heads. Check this when going past 0.8B.

Tensor types: the Unsloth **Q8_0** file is Q8_0 + F32 only, so runq's kernels cover it. The Unsloth **Q4_K_M**
uses Q4_K / Q6_K, plus **Q5_K** for `attn_qkv` and `ssm_out` of every linear layer (36 tensors), and Q8_0 for
`ssm_alpha` / `ssm_beta`. It needs Q5_K, which is the TODO item "More GGUF types". **Q8_0 comes first.**

## Design decisions

**1. Exact recurrence everywhere; no chunked prefill.** HF and llama.cpp prefill with the chunked
(WY / UT-transform) form of the delta rule, and decode with the recurrence. In fp32 these round differently.
runq's rule is that batched and sequential logits are bit-identical, so prefill runs the same per-token
recurrence as decode. It is cheap: per token and layer, the recurrence costs 16 heads x 128 x 128 x ~8 flops,
~2 MFLOP, so 38 MFLOP over 18 layers. The matmuls cost ~1.5 GFLOP per token, which makes the recurrence ~2.5%.
Work item: one (sequence, head) for all the tokens of a chunk, in order. The head's 64 KB state stays in that
core's L2 for the whole chunk. Arithmetic order, fixed so that SIMD widths agree: vectorize along the v index
(contiguous), and reduce over k sequentially (`S^T k`, `S^T q`: for each v lane, the sum in k order). Each
lane's sum is then the same at any vector width, and AVX-512, NEON and generic agree bitwise, as the
attention kernels do (`ATT_EXACT` / `ATT_FMA` rules). The per-head scalars (`exp(g)`, `beta`, the l2norm
sums) are computed by one function shared by every path.

**2. Per-sequence state next to a smaller KV cache.** In `RqSeq`, per linear layer: the conv state (the last
3 positions x 6144 f32, 72 KB) and S (16 x 128 x 128 f32, 1 MiB). Over 18 layers that is **18.9 MiB per
sequence**, independent of the length. The KV cache covers the 6 attention layers only:
6 x 2 kv heads x 256 x 2 (K, V) x 2 bytes = **12 KB per token** at f16 (Qwen3-0.6B: 112 KB). The 40,960-token
default context therefore needs 0.5 GB instead of 4.6 GB, and the full 262,144 needs 3.2 GB. Index the cache by
attention layer (0..5), not by model layer. The state stays fp32 (HF keeps `mamba_ssm_dtype` float32), and
the f16 / bf16 cache types apply to K / V only.

**3. Rollback needs checkpoints.** Several places assume a sequence can return to any earlier length:
`rq_seq_truncate(n < len)`, `rq_seq_copy(dst, src, n < src->len)`, the CLI's chat reuse
(`rq_seq_common_prefix` + truncate, `src/cli.c`), and the server's prefix cache and slot copy
(`src/engine.c`). For the DeltaNet layers this is false: the state at position n is gone once n + 1 is
processed. Plan:
- A sequence keeps a few **state checkpoints** (position + a copy of the conv + S state, ~20 MB each).
  `rq_seq_checkpoint(s)` saves at `len`, and the library saves one by itself at the end of every prefill
  (before generation), which is where the next request of a chat diverges: the template drops the earlier
  reasoning, or the client edits the last turn.
- `rq_seq_truncate(s, n)` on a hybrid model goes back to the largest checkpoint at a position <= n. It
  returns the length actually kept, and the caller re-prefills the rest (`rq_seq_common_prefix` users need
  that number). `rq_seq_copy(dst, src, n)` works the same way: it copies the KV prefix and the checkpoint
  state. Positions 0 and `len` always work. This is an API change (truncate can keep less than asked for),
  so check every caller. A model without linear layers keeps the exact behaviour.
- Checkpoints per sequence: a small LRU (e.g. 4, an `RqCtxConfig` field). llama.cpp handles hybrid and SWA
  models the same way ("context checkpoints").

**4. Batched decode across sequences:** the matmuls stay row-batched as now. The recurrence items are
(sequence, head) over each sequence's own state, so a sequence alone and in a batch run the same per-head
code on the same inputs, and the "a request batched or alone" invariant holds by construction.

**5. Tokenizer:** the `qwen35` pre-tokenizer regex differs from `qwen2` in one way: letters run as
`[\p{L}\p{M}]+` instead of `\p{L}+`, and the punctuation class excludes `\p{M}` too
(`[^\s\p{L}\p{M}\p{N}]`), so combining marks join words (Devanagari, Thai, decomposed accents left after NFC).
Needed: a `UCLS_M` class in `tools/gen_unicode_tables.py` / `src/unicode_tables.h`, and `pretok_next` taking
the regex variant from `tokenizer.ggml.pre`. The vocabulary grows to 248,320 (the hash table is sized from
it). Tests: `golden/tokenizer.json`'s cases with the Qwen3.5 tokenizer, as a second golden file, plus
`fuzz_tokenizer.py` against it.

**6. Chat template:** a new variant `RQ_CHAT_QWEN35`. The template contains `enable_thinking`, so today's
detection would call it *hybrid*, which is wrong. Differences from hybrid: thinking is **off** by default (the
generation prompt ends with `<think>\n\n</think>\n\n`), and thinking on opens `<think>\n` as Thinking-2507
does. Earlier assistant turns are rendered without their reasoning, as in hybrid. Detect it by
`general.architecture == qwen35`, or by the template text: `enable_thinking is defined and enable_thinking
is true`. Tools use a new XML call format, which does not matter while the server rejects tools.
`-m render` goldens: `tests/make_chat_golden.py` with the Qwen3.5 template (`.venv-hf5`).

**7. Other config:** a `rope_dim` (64, from `rope.dimension_count`) for `rope_row` / `qk_norm_rope`, which
rotate `rope_dim` dims with pairs `(j, j + rope_dim/2)` and leave the rest untouched. With rope_dim =
head_dim, Qwen3 must stay bit-identical. A per-layer type array, from `full_attention_interval`, or from the
`attention.recurrent_layers` array when present. `head_dim` 256 is fine for the attention kernels, which only
need multiples of 16 / 32. The default context cap stays at 40,960.

## Expected performance (estimates, to be measured)

Decode reads every weight once per token. Q8_0 total ~810 MB, of which the tied classifier is ~270 MB
(248,320 x 1,024: a third of the file). Qwen3-0.6B Q8_0 reads ~640 MB, so expect ~0.8x its decode rate
(~150 tok/s at 8 threads on the dev box) and a lower long-context slowdown, since the KV cache is 1/9 the size
and covers 6 layers only. The classifier share makes the TODO item "subset classifier at constrained
decisions" more valuable here. Prefill: about the same matmul rate per weight byte as 0.6B, plus the ~2.5%
recurrence.

## Steps

The step list, with "done when" criteria for each step, is in [TODO.md](../TODO.md#qwen35-08b-hybrid-gated-deltanet--gated-attention)
(Q35-1 .. Q35-7): loader / tokenizer / chat template, the sequential forward pass on Q8_0, batched prefill and
the bitwise invariants, state checkpoints for rollback, tests in the suite, Q5_K for the Q4_K_M file, performance.

## Commands

```
.venv-hf5/bin/python tests/make_golden.py --model-dir models/Qwen3.5-0.8B --out-dir tests/golden/models/3.5-0.8b-hf
.venv-hf5/bin/python tests/make_model_ref.py 3.5-0.8b-q8 models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf
```

Re-creating `.venv-hf5`: `python3 -m venv .venv-hf5 && .venv-hf5/bin/pip install torch==2.7.0 --index-url
https://download.pytorch.org/whl/cpu && .venv-hf5/bin/pip install "transformers>=5.2" safetensors numpy jinja2`
(5.21.0 used). Downloads: `https://huggingface.co/Qwen/Qwen3.5-0.8B` (config, tokenizer files, the
safetensors) and `https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF` (`Qwen3.5-0.8B-Q8_0.gguf`,
`Qwen3.5-0.8B-Q4_K_M.gguf`).
