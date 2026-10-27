# Qwen3.5: design notes

Status (2026-10-27): **Qwen3.5-0.8B Q8_0 and Unsloth's Q4_K_M, and Qwen3.5-4B Q8_0 are supported** (GGUF
`general.architecture == qwen35`; 4B since Step 14: more value heads than key heads, design 8 below): loader, the
`qwen35` pre-tokenizer, the chat template variant, the hybrid forward pass in every path (decode, batched prefill,
batched decode rows) with the bitwise invariants, recurrent-state checkpoints for truncate / copy, the CLI chat and
`runq serve`, and the tests (`tests/check_models.py` group `q35` in `make test`, the file in the `models` group).
Open: the NEON check of the recurrence, the MoE models ([TODO.md](../TODO.md)); 2B (16 / 16 heads, as 0.8B), 9B and 27B
(2x / 3x as many value heads as key heads, as 4B) should load but are untested (Qwen3.8-27B, the same architecture
at r = 3, is tested: [qwen38-speculative.md](qwen38-speculative.md#the-27b-bring-up-s7)).
This file records what Qwen3.5 is, what was verified, and the design. Test model: **Qwen3.5-0.8B**: the same layer
types as the larger Qwen3.5 / 3.6 models, with a dense FFN instead of MoE.

Results (2026-10-26, 16-vCPU dev box):

- Logits vs HF transformers 5.21 fp32 (`tests/golden/models/3.5-0.8b-hf`, en / code / chat / long2, 62 positions):
  cos mean 0.9996-0.9998 per prompt, min 0.9986, top-1 62/62, pooled 0.99976; vs the Q8_0 file's dequantized-weights
  reference (`3.5-0.8b-q8`): pooled 0.99986. Perplexity on `bench/ppl_beagle_ch1.txt`, 512-token windows: runq Q8_0
  22.729, HF fp32 22.728 (full text: runq 18.86; HF's reference kernels ran out of memory there).
- Bit-identical: `-b 1 / 7 / 64 / 512`, 8 vs 3 threads, AVX-512 vs generic, KV f16 / bf16 / f32, a batched decode
  row vs the sequence alone (`api_test`), a CLI chat that rolls back to a checkpoint vs a single-shot run.
- Speed (8 / 12 threads, `scripts/bench_models.py`, 2026-10-27, load 3-6 from another agent): prefill of 451 / 1,500
  tokens 1,506 / 1,461 and 2,039 / 2,029 tok/s (Qwen3-0.6B Q8_0 in the same session: 1,832 / 1,396 and 2,511 / 1,974:
  0.82x / 1.05x at 8 threads, 0.81x / 1.03x at 12; before the blocked recurrence 1,170 / 1,127 and 1,568 / 1,590),
  decode 162 / 174 tok/s (0.6B: 180 / 219; 141 / 182 in the quiet session of 2026-10-26, 0.6B 209 / 230), after a
  1,500-token prompt 142 / 160 (0.6B: 161 / 194: the 6-layer KV cache barely slows it down); peak RSS 964 MB (1,500 +
  128 tokens, `-c 4096`), startup 0.1 s. The recurrence takes ~17% of prefill (`-DRUNQ_PROFILE`): see Performance
  below.
- Tokenizer: `tests/golden/tokenizer_qwen35.json` (90 cases, 23 of them pre-tokenized differently from Qwen3's
  regex) and `fuzz_tokenizer.py` (5,000 random strings + the repo's text, 1.4 M tokens): equal to HF.

Q4_K_M (Unsloth, 533 MB; Q5_K support added 2026-10-27, Step 13; measured on the same box, load 5-7, 8 threads, best of 3,
interleaved with the Q8_0 file):

- Logits vs HF fp32 (`3.5-0.8b-hf`, 62 positions): cos means 0.9924 / 0.9959 / 0.9958 / 0.9925 (chat, code, en, long2),
  min 0.9677, position 0 0.9855-0.9957, pooled 0.9946, top-1 57/62. That is the 4-bit weights' own error (Qwen3-0.6B
  Q4_K_M: pooled 0.984, top-1 68/84; its Q8_0: 0.9994): vs the file's dequantized-weights reference (`3.5-0.8b-q4km`,
  `make_model_ref.py`) runq is at pooled cos 0.99986, every position's cos >= 0.9996 except chat's (0.9990), top-1 62/62,
  the same distance as the Q8_0 file from its reference (0.99986). So the int8 activations and kernels add nothing
  visible; the rest is quantization.
- Bit-identical, as the Q8_0 file: `-b 1 / 7 / 64 / 512`, 8 vs 3 threads, AVX-512 vs generic (`-b 1 / 7 / 64`; the
  batched passes of `api_test --digest-small` too), a batched decode row vs the sequence alone; the NEON kernels vs the
  generic ones on random Q5_K blocks (`make kq-neon-test`, run under clang + qemu-user: the same for all three types).
- Speed (prefill 448 / 1,492 tokens, decode short / after 1,500 tokens, tok/s; `scripts/bench_models.py -t 8 -r 3`):
  Q8_0 1,210 / 1,167, 148 / 139; Q4_K_M 1,065 / 1,039, 188 / 189. Decode 1.27x (the 18 linear layers' `attn_qkv` and
  `ssm_out` are Q5_K: ~5.5 bits; the FFN, `attn_gate` Q4_K; the tied 248k x 1,024 classifier Q6_K; 0.53 GB read per
  token vs 0.81), prefill 0.88x (more arithmetic per weight, as for Qwen3). Peak RSS 662 MB vs 963 MB.
- Where the Q5_K tensors are (36): `blk.N.attn_qkv.weight` (6,144 x 1,024) and `blk.N.ssm_out.weight` (1,024 x 2,048) of
  the 18 linear layers; Q4_K: all `ffn_gate` / `ffn_up`, `attn_gate`, the full-attention `attn_q` / `attn_k` /
  `attn_output`, half of `ffn_down` and 2 `attn_v`; Q6_K: the other `ffn_down`, 4 `attn_v` and `token_embd`; Q8_0:
  `ssm_alpha` / `ssm_beta`.

Q3_K_M (Unsloth, 470 MB; Q3_K support added 2026-10-27, TODO G3: the 27B's Q3_K_M file uses the same 3-bit roles):

- Tensor types: Q3_K (78) `attn_q` / `attn_k` (the 6 attention layers), `attn_gate` (18 linear layers), every
  `ffn_gate` / `ffn_up`; Q4_K `attn_qkv`, `ssm_out`, `attn_output`, most `attn_v` / `ffn_down`; Q5_K 3 of those;
  Q6_K the tied `token_embd`; Q8_0 `ssm_alpha` / `ssm_beta`.
- Logits vs HF fp32 (`3.5-0.8b-hf`, 62 positions): cos means 0.9701 / 0.9864 / 0.9815 / 0.9822 (chat, code, en, long2),
  min 0.8577 (chat), position 0 0.9379-0.9918, pooled 0.98089, top-1 54/62: ~3.5x the Q4_K_M file's 1 - cos (0.9946,
  57/62). All of it is the 3-bit weights': vs the file's dequantized-weights reference (`3.5-0.8b-q3km`) runq is at
  pooled cos 0.99984, top-1 62/62 (the Q4_K_M file: 0.99986). Greedy chat answers as the Q4_K_M file does.
- Bit-identical: `-m dequant` of all 78 Q3_K tensors (236,544 rows, from the blocks and from the repacked layout) ==
  gguf-py; `-b 1 / 7 / 64 / 512`, 8 vs 3 threads, AVX-512 vs generic (`-b 1 / 7 / 64`, `api_test --digest-small`), a
  batched decode row vs the sequence alone; NEON vs generic on random Q3_K blocks at k = 1,024 / 3,072
  (`make kq-neon-test` under clang + qemu-user).
- Speed (prefill 448 / 1,492 tokens, decode short / after 1,500 tokens, tok/s; `scripts/bench_models.py -t 8 -r 3`,
  interleaved with the Q4_K_M file, load 3-5): Q3_K_M 1,409 / 1,322, 211.5 / 208.9; Q4_K_M 1,443 / 1,355, 206.0 / 191.8;
  peak RSS 601 vs 663 MB. Small gains on this file: its matrices read 459 MB per token vs 521, the Q3_K ones only 104
  (the tied 248k x 1,024 Q6_K classifier is 209). The Q3_K GEMV runs as fast per weight as Q4_K's on one thread (a
  scratch benchmark at the 27B's `ffn_gate` shape, 17,408 x 5,120: 1.88 vs 1.89 ms; 8 threads 0.29 vs 0.49 ms), so on
  the 27B the 3-bit matrices are read at memory speed as the 4-bit ones are.

Qwen3.5-4B Q8_0 (Unsloth, 4.48 GB; Step 14, 2026-10-27):

- Logits vs the HF-weights reference (`tests/golden/models/3.5-4b-hf`: **`tests/ref_qwen35.py` on HF's bf16
  safetensors, not transformers**, whose fp32 forward needs ~17 GB; that reference matches transformers to 2-6e-5 on
  0.8B and on tiny models with nv = 2 / 3 nk, see design 8): cos means 0.99985 / 0.99991 / 0.99988 / 0.99934 (chat,
  code, en, long2), min 0.99849, position 0 >= 0.9999, top-1 62/62, pooled 0.99982 (0.8B Q8_0 vs HF: 0.99976); vs the
  Q8_0 file's own reference (`3.5-4b-q8`): pooled 0.99987, min 0.99887, top-1 62/62. The two references agree to cos
  >= 0.999 per position, top-1 100%.
- Greedy "capital of Australia": "Australia does not have a single capital city; instead, it has three capitals:
  Canberra (the capital of the Commonwealth of" -- the model's own answer: the same 24 tokens from a greedy decode of
  the HF-weights reference (the closest call, token 20, by a margin of 0.03).
- Bit-identical: `-b 1 / 7 / 64 / 512`, 8 vs 3 threads, AVX-512 vs generic (`-b 1 / 7 / 64`, and the batched passes of
  `api_test --digest-small`), `api_test` 25/25 (batched decode rows == alone, checkpoints); `runq serve`'s greedy reply ==
  the CLI's. (`tests/run_tests.sh --models --model-tags 3.5-4b-q8`: 36 checks, ~100 s.)
- Speed (`scripts/bench_models.py -t 8,12 -r 3`, load 13-15 incl. the run's own threads; prefill of 448 / 1,492 tokens,
  decode short / after 1,500 tokens, tok/s): 8 threads 308 / 298, 30.6 / 26.4; 12 threads 443 / 421, 29.5 / 31.9.
  Qwen3-4B Q8_0 alongside: 314 / 281, 30.7 / 31.7 and 440 / 397, 30.8 / 31.5: the same (they read ~4.3-4.5 GB per token;
  the 4B's linear layers have 24 x 32 heads of state, 50 MB per sequence, its 8 attention layers 32 KB per token at
  f16). Peak RSS 4,813 MB (1,500 + 128 tokens, `-c 4096`; Qwen3-4B 4,653), startup 0.7-0.8 s. Perplexity on
  `bench/ppl_beagle_ch1.txt`: 10.592 (full text), 12.645 (512-token windows); Qwen3-4B Q8_0: 15.65.

Tools: `models/Qwen3.5-0.8B` (HF safetensors, bf16, with the tokenizer files) and `models/Qwen3.5-0.8B-GGUF`
(Unsloth `Q8_0`, 812 MB, and `Q4_K_M`, 533 MB): `scripts/download_models.py qwen35`; `models/Qwen3.5-4B` (9.3 GB) and
`models/Qwen3.5-4B-GGUF` (`Q8_0`, 4.48 GB): `qwen35-4b`. `tests/ref_qwen35.py`: a numpy
reference forward pass (the spec), on the GGUF's dequantized weights or on the HF checkpoint; on the HF weights it
matches transformers 5.21 fp32 to 2-6e-5 max abs logit error (KL ~1e-11, top-1 100%), so the token-by-token
recurrence below equals HF's chunked prefill up to rounding. `.venv-hf5` (git-ignored): transformers 5.21 / torch
2.7.0 CPU for the HF goldens; the test venv `.venv` stays at transformers 4.53, which does not know `qwen3_5` (but
loads its tokenizer and renders its template: the tokenizer and chat goldens use it).

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
plain partial RoPE above. Images would need the 3-D positions, which are out of scope for the project ([IDEAS.md](../IDEAS.md#images-qwen35-vision)).

**Not part of the text model:** the vision tower (153 tensors) and the MTP head (15 tensors, `mtp.*`;
a draft model for speculative decoding: TODO.md, S9 for Qwen3.8's). The Unsloth GGUFs contain neither: 24 blocks, no `nextn` tensors.

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
- With 16 = 16 heads (0.8B, 2B), there is no head reordering. With `num_v_heads` = r x `num_k_heads` (4B, 9B: r = 2;
  27B: r = 3), the converter reorders the value heads from HF's grouped order to a tiled one: GGUF value head j =
  HF value head (j % nk) r + j / nk, i.e. **GGUF value head j uses key head j % nk** (HF: i // r). Reordered: the v
  rows of `attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_a`, `ssm_dt.bias`, the v channels of `ssm_conv1d`,
  the input columns of `ssm_out`; not the q / k rows and channels, nor `ssm_norm`. Checked on the 4B Q8_0 file,
  layers 0 / 1 / 4 / 28, whole tensors: max |GGUF - HF| under the tiled permutation 0 (F32 tensors) to 2e-3 (Q8_0
  rounding), under the identity 0.17-48. Design 8.

Tensor types: the Unsloth **Q8_0** file is Q8_0 + F32 only, so runq's kernels cover it. The Unsloth **Q4_K_M**
uses Q4_K / Q6_K, plus **Q5_K** for `attn_qkv` and `ssm_out` of every linear layer (36 tensors), and Q8_0 for
`ssm_alpha` / `ssm_beta`. Q5_K is supported since Step 13 (`src/kquants.h`: the Q5P layout, and the Q5_K arithmetic
= Q4_K's fold on 5-bit values); the Q8_0 file came first. Q3_K (the **Q3_K_M** file, TODO G3) since 2026-10-27: the Q3P
layout, Q6_K's fold on 3-bit values. Qwen3.8-27B's Q4_K_M has `ssm_alpha` / `ssm_beta` as **F32**
matrices: accepted for those two only, applied in fp32 to the fp32 activations with a fixed summation order (TODO S6:
[qwen38-speculative.md](qwen38-speculative.md#loader-the-mtp-block-and-the-f32-matrices)), and an MTP block after its
64 layers (`nextn_predict_layers` 1), validated but not loaded.

## Design decisions

**1. Exact recurrence everywhere; no chunked prefill.** HF and llama.cpp prefill with the chunked
(WY / UT-transform) form of the delta rule, and decode with the recurrence. In fp32 these round differently.
runq's rule is that batched and sequential logits are bit-identical, so prefill runs the same per-token
recurrence as decode. It is cheap: per token and layer, the recurrence costs 16 heads x 128 x 128 x ~8 flops,
~2 MFLOP, so 38 MFLOP over 18 layers. The matmuls cost ~1.5 GFLOP per token, which makes the recurrence ~2.5%.
(That estimate was wrong in practice: the recurrence is memory-bound, see Performance.) Implemented in
`src/forward.c` as three steps, the same functions in every path: `dn_prep` (one token of one value head: the
causal conv + SiLU, the l2norm of q / k and the gates; the conv state, so the rows of a segment in order), `dn_rec`
(the delta rule on a block of 32 v columns of the head's state, a 128 x 32 block of 16 KB that stays in L1; the
state's rows in order), and `dn_out` (the gated RMSNorm over the head's dv outputs, once all its blocks are done).
Each v lane's recurrence is independent of the others, so the blocks give exactly the bits of a whole head. Work
items: decode, one alpha / beta item, then one `dn_prep` item per value head (after it: a dependency counter) and
one `dn_rec` item per (head, block) (after its head's prep; the last block of a head runs `dn_out` and quantizes):
1 + 16 + 64 items; batched prefill, three phases: `dn_prep` per (segment, head, q / k / v group), `dn_rec` per
(segment, head, block) over the segment's rows in order, `dn_out` + the row's quantization per row; batched decode
rows, the decode's items over all rows (each row on its own sequence). Arithmetic order, fixed so that SIMD widths
agree: elementwise along the channels / the v index
(vectorized along them), the reductions over k sequential per v lane (`S^T k`, `S^T q`), the l2 / rms sums in 16
lanes + a fixed tree, `exp` by the attention's polynomial, the per-head scalars (`exp(g)`, `beta`: libm `expf` /
`log1pf`) by one non-inlined scalar function.

One trap, found by the AVX-512 vs generic comparison: under `-Ofast` gcc turns a **vectorized float division** into
a reciprocal approximation plus a Newton step (`vrcp14ps`), whatever `no-reciprocal-math` says (it depends on
`unsafe-math-optimizations`). Whether the sigmoid loops (`1 / (1 + e)`) were vectorized depended on inlining, which
differed between the two builds, so they rounded differently (max logit difference 0.5). The DeltaNet code uses
`DN_EXACT` (`ATT_EXACT` + `no-unsafe-math-optimizations`; vector divisions are then exact `vdivps`) and its own
copy of the exp polynomial (`dn_exp`, so that it inlines under the same attributes). The attention's code divides
only in scalar code.

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
processed. Implemented (llama.cpp handles hybrid and SWA models the same way, "context checkpoints"):
- A sequence keeps a **set of state checkpoints** (position + a copy of the conv + S state, 18.9 MB each for the
  0.8B, ~150 MiB for the 27B, allocated at first use): `rq_seq_set_checkpoints(s, n)` slots, 1..32, default 2
  (`RQ_SEQ_CK`; fewer than in use: the highest positions stay, the other buffers are freed). The setting is per
  sequence, not per context: the slots belong to the sequence, and `rq_seq_checkpoint_at` / truncate / copy run
  without a context. `rq_seq_checkpoint_at(s, pos)` asks for one at a position >= the length: the forward pass that
  reaches it takes it, or it is taken at once if the sequence is there. Several can be pending (a sorted list, in
  one pass or later ones): before a pass, `ck_plan` gives each pending position the pass reaches a slot (a
  `CkPass` per sequence: length, row, buffer, ascending), and each work item copies its part right after each of
  those rows (`dn_prep_part` its channel group's conv state, looked up by position; `dn_rec_run` its block of S, a
  cursor over the rows in the loop), so checkpoints can sit anywhere inside a prefill chunk, a batched pass of
  several sequences or a decode step. A new checkpoint takes a free slot, else the one at the lowest position;
  with more pending in one pass than slots, the pass's own lowest ones give way (the highest stay). With nothing
  pending the loops test one counter, as before one pointer: the speed is unchanged (0.8B Q8_0, 8 threads,
  `scripts/bench_models.py`, 3 interleaved ABBA rounds vs the commit before, the box at load ~20: best of 6,
  prefill of 448 / 1,492 tokens 936 / 960 vs 853 / 965 tok/s, decode short / after 1,500 72.8 / 75.8 vs 69.6 /
  74.6 tok/s; the medians within the noise as well), and the outputs are byte-identical (`tests/compare_bins.sh`:
  792 jobs, every family).
- `rq_seq_checkpoint_rows(s, pos, n)` (for speculative decoding, TODO S3): checkpoints after each of rows
  pos..pos+n-1 of the next pass(es), flagged as rows: the next call frees them (taken or pending) before adding its
  own, so a verify segment of k + 1 rows per pass needs k + 2 slots to keep the prompt's checkpoint as well (the
  lowest, which plain lowest-first replacement would evict first). Truncating to any accepted length inside the
  segment lands exactly. A draft's decode steps work the same way (one pending per pass).
  Pending positions are forgotten by `rq_seq_truncate` (any n), `rq_seq_copy` (dst) and `rq_seq_reset`: they were
  for the tokens that were to follow (before, a new `rq_seq_checkpoint_at` replaced the one pending position: the
  callers, the CLI chat and the engine, ask for one right after truncating or copying, so nothing changes for
  them). Tests: `api_test`'s `checkpoint_sets`.
- `rq_seq_truncate(s, n)` on a hybrid model, n < len, goes back to the largest checkpoint at a position <= n (its
  state copied back), else to 0; checkpoints past the new length are dropped. Callers read `rq_seq_len` afterwards
  and re-prefill the rest; `rq_seq_keep(s, n)` tells in advance what would be kept. `rq_seq_copy(dst, src, n)`
  likewise: src's current state if n is its length, else its best checkpoint <= n, else 0. The return codes did
  not change; a model without linear layers keeps the exact behaviour (Qwen3's outputs are byte-identical,
  `tests/compare_bins.sh`). The state of a position-0 token is zero (`dn_prep` / `dn_rec` clear it), so a reset needs no work.
- Where: `rq_chat_checkpoint_pos` is the prompt's last `<|im_start|>` (the assistant header). The CLI chat and the
  server's engine ask for a checkpoint there for each prompt: a thinking chat's next turn diverges right after it
  (the history drops `<think>\n` and the reasoning), as does a reply cut at a stop string or re-tokenized
  differently from its text. A non-thinking chat's next turn usually starts exactly at the cache's length (no
  rollback). The engine scores slots by what they can keep (`rq_seq_keep`) and does not wait for another job's
  prefix being prefilled (its state is gone by then).

**4. Batched decode across sequences:** the matmuls stay row-batched as now. The recurrence items are
(sequence, head) over each sequence's own state, so a sequence alone and in a batch run the same per-head
code on the same inputs, and the "a request batched or alone" invariant holds by construction.

**5. Tokenizer:** the `qwen35` pre-tokenizer regex differs from `qwen2` in one way: letters run as
`[\p{L}\p{M}]+` instead of `\p{L}+`, and the punctuation class excludes `\p{M}` too
(`[^\s\p{L}\p{M}\p{N}]`), so combining marks join words (Devanagari, Thai, decomposed accents left after NFC).
Implemented as a `UCLS_M` class (`tools/gen_unicode_tables.py`, `src/unicode_tables.h`), which `uchar_at` reads
as a letter for `qwen35` and as "other" for `qwen2`: that reproduces both regexes exactly (the optional prefix
`[^\r\n\p{L}\p{N}]` of the letter alternative admits a mark, but a mark is taken by the letter run anyway).
Tests: `tests/golden/tokenizer_qwen35.json` (`make_golden.py --tokenizer-qwen35`: `tokenizer.json`'s cases plus
combining-mark ones, ids from the Qwen3.5 tokenizer) and `fuzz_tokenizer.py --model ... --hf-dir models/Qwen3.5-0.8B`.

**6. Chat template:** a new variant `RQ_CHAT_QWEN35` ("qwen3.5"), detected by the architecture (the template
contains `enable_thinking`, which the text detection would call *hybrid*). Thinking is **off** by default (the
generation prompt ends with `<think>\n\n</think>\n\n`; the CLI's `-r` defaults to 0, the server's requests too);
thinking on opens `<think>\n` as Thinking-2507 does, and answered turns lose it (`rq_chat_user_history`), so a
reply's history form is Thinking-2507's; with thinking off, the hybrid's (the empty think block stays in the
history, the documented difference from HF). Its template trims every message (`|trim`, Python's `str.strip()`):
`rq_chat_trim`, applied by `rq_chat_system` / `user` / `render` and the server. Tools use a new XML call format,
which does not matter while the server rejects tools. `-m render` goldens: the `qwen3.5` cases of
`tests/golden/chat_render.json` (`make_chat_golden.py`, with a whitespace case for the trim), 9 of 12 equal to
HF's `apply_chat_template` (the other 3: thinking off with history, by design).

**7. Other config:** `RqModel.rope_dim` (64, from `rope.dimension_count`) for `rope_row` / `qk_norm_rope`, which
rotate `rope_dim` dims with pairs `(j, j + rope_dim/2)` and leave the rest untouched (with rope_dim = head_dim,
Qwen3's arithmetic is unchanged). The layer maps `kv_layer` / `lin_layer` (from `full_attention_interval`): the
attention weights, norms and the KV cache are indexed by attention layer, the linear ones by linear layer.
`head_dim` 256 is fine for the attention kernels. The default context cap stays at 40,960.

**8. More value heads than key heads (4B, Step 14).** 4B has 16 key heads and 32 value heads per linear layer (9B
the same, 27B 16 / 48). In the GGUF, value head h uses the q / k of key head h % nk (above). So `dn_rec`'s q / k
offsets are those of key head h % nk, and everything else stays indexed by h as before (z, alpha / beta, `ssm_a` /
`ssm_dt`, the v channels, S, the outputs' slot in `ssm_out`). The q / k conv state, though, belongs to the key head:
it must advance once per token, not once per value head (with two value heads each running the q / k conv, the
second would shift the state again and read the first one's in-place results; and the two items race). So
`dn_prep`'s channel groups are now q and k per key head and v (+ the gates) per value head:
- decode and batched decode rows: the item of value head h computes its v / z rows and, if h < nk, the q / k rows of
  key head h, then `dn_prep` on those groups; `dn_rec` of head h waits for the preps of h and of h % nk (two counters
  of the existing ones). With nk == nv this is exactly the old schedule;
- prefill: the items of a segment are the groups (`dn_prep_item`): q, k, v of heads 0 .. nk - 1 in turn (the old
  order when nk == nv), then the v of the other value heads: 3 nk + nv - nk per segment;
- checkpoints: each group copies its own conv state, so the q / k part once per key head; S per (value head, block).
The arithmetic of each group did not change, so 0.8B's outputs are byte-identical to before (the logits at `-b 1 / 7 /
64 / 512`, 3 / 8 threads, both builds, Q8_0 and Q4_K_M; Qwen3 via `tests/compare_bins.sh`: 360 jobs identical).
Load checks: nv % nk == 0, and files with `prism.hadamard.*` keys (Hadamard-folded weights, llama.cpp #29600: their
`ssm_out` keeps HF's grouped order) are rejected.

How the mapping was verified, end to end: tiny random-init `qwen3_5` models (transformers 5.21: 8 layers, dim 128, 2
key heads with 4 resp. 6 value heads, the matrices snapped to the Q8_0 grid so that the conversion is exact),
converted by llama.cpp's `convert_hf_to_gguf.py` (master 7aca0d2, 2026-10-26; `--no-mtp --outtype q8_0`). Against
transformers' fp32 logits (its torch fallback of the chunked delta rule): `ref_qwen35.py` on the HF weights (h // r)
max |diff| 4-9e-5, on the GGUF (h % nk) 4e-5-1e-4, on the GGUF with h // r cos means 0.91-0.95; runq on the GGUF cos >=
0.9991 at every position (means 0.99986-0.99993), top-1 100%. (Too big to commit: the 248k-token embedding; and the
converter and transformers 5.x are not test dependencies.) In `make test`: two synthetic models cut out of the 0.8B
file with 4 / 8 and 2 / 6 heads (`tests/make_synth_q35.py`), runq vs `ref_qwen35.py` on them and the bitwise
invariants; scratch builds with HF's mapping in `dn_rec`, or with the q / k conv redone by every value head, fail it.

## Performance

Estimated before: decode ~0.8x Qwen3-0.6B's rate (the Q8_0 file reads ~810 MB per token vs ~640 MB; the tied
classifier is a third of it: 248,320 x 1,024), a lower long-context slowdown, prefill at the matmul rate plus a
~2.5% recurrence. Measured: decode as estimated (141 vs 209 tok/s at 8 threads; 160 vs 182 after 1,500 tokens).

Prefill was ~0.64x the 0.6B's instead of ~0.8x (451 tokens, 8 threads): the recurrence took ~48% of it
(`-DRUNQ_PROFILE`, 1,492-token prompt, -b 512: 0.62-0.69 s of 1.29-1.43 s at 8 threads, 0.42 of 0.89 at 16).
`dn_head` alone (one thread, a micro-benchmark) took ~5 us per token and head, ~4 of them in two passes over the
64 KB state (read and write each): bound by the cache bandwidth, not by the flops; in prefill ~12 us per token and
head at 8 threads, with only 16 work items per segment. Done (the design above): v blocks of 32 columns,
64 items per segment and the state of a block in L1, bit-identical. What measuring found, in order of effect:
- The FMA chains over k (128 dependent FMAs per v lane for `S^T k`, again for `S^T q`) are latency-bound: with 32
  lanes, 2 independent chains of 512-bit vectors. They have to live in registers (the j loops unrolled,
  `DN_UNROLL`): as arrays gcc kept them in memory and every step waited for a store forward.
- The second pass of row t and the first pass of row t + 1 are fused (`dn_p21`): the first reads each S row right
  after the second wrote it, so the same floats go in, and 4 chains run side by side (hot-cache micro-benchmark:
  ~1,750 -> ~880 TSC cycles per block and row). The decay `exp(g) * S` is recomputed in the second pass instead of
  being stored by the first (the same product of the same floats): one write pass less.
- 4K aliasing: the conv state and taps were rows of the whole layer (6144 floats = 24 KB apart), so the 8-9 streams
  of the conv's index i fell on one set of the 8-way L1 (`dn_conv` took ~900 cycles per 128 channels, ~4x its
  work). The state and the taps are now stored per channel group (a head's q, k or v channels as contiguous
  rows; the loader transposes into it), S block-major: `dn_prep` 0.17 -> 0.09 s.
- The inputs of a head are 512 B pieces of rows 33 KB apart that other cores' GEMM items wrote: each line was a
  miss waiting for the previous one. Prefetching the rows ahead in `dn_prep` / `dn_rec` gave ~20%.

Result (8 threads, 1,492 tokens, interleaved, medians; load 3-12): prefill 1,099 -> 1,428 tok/s (Qwen3-0.6B Q8_0
alongside: 1,373), 1,470 -> 1,949 at 12 threads, 1,644 -> 2,129 at 16; `bench_models.py` at 451 / 1,500 tokens: 0.82x /
1.05x of the 0.6B at 8 threads, 0.81x / 1.03x at 12 (was 0.64x / 0.81x); decode unchanged. Profile at 8 threads (1,492
tokens, 1.0 s): the recurrence is `dn_prep` 0.08 s + `dn_rec` 0.15 s + `dn_out` 0.016 s (was ~0.55-0.6 s of the
"attention+quant" bucket, whose remaining 0.09 s is the 6 attention layers); at 16 threads (0.67 s) 0.06-0.07 + 0.10 +
0.013-0.026. What is left of it: `dn_prep` scales poorly past 8 threads (48 items per segment, each one sequential
over the rows: the conv's SiLU is a vectorized exp polynomial + a division per channel), and `dn_rec` is still
~2x its hot-cache cost (the q / k rows are read from the other cores' caches by each of a head's 4 blocks; a compact
per-head copy written by `dn_prep` could help).

## Long context (TODO G2, 2026-10-27)

Nothing else in the tests goes past ~2,000 positions. The `long` group (`tests/run_tests.sh --long-context`,
`tests/check_long.py`, opt-in) checks runq at up to 98,304 positions against `tests/ref_qwen35.py` on the file's
dequantized weights, through `build/long_ctx` (`tests/long_ctx.c`, the library API: the CLI's hooks take neither a
400 KB prompt nor 98k rows of logits).

- **The text:** 18 files of this repository at the fixed commit `5d93873` (`git show`: README, docs, `src/*.c`,
  `tests/check.py`, the Beagle chapter, interleaved prose / C / Python, no repeat), tokenized by runq: 244,198
  tokens, the first 98,304 used (sha256 of the ids in the golden).
- **The reference:** `ref_qwen35.forward_long` (`tests/make_long_ref.py`): the same numpy fp32 forward, rows in
  chunks of 2,048 (the DeltaNet state and the conv inputs carried across chunks), full attention per layer as all
  rows' K / V first, then the queries in blocks of 128 with an exact softmax over every key up to the block's last
  position (no split, no combine: the opposite of runq's per-256-position partial results), the last layer on the
  sampled rows only, the logits on the full vocabulary. `--check-short` compares it with `forward` (700 tokens, `rc`
  256): max |diff| 2.5e-5. Cost on the 0.8B at 98,304 tokens: **1,813 s** at 8 BLAS threads (load ~10-25; ~31 s per
  DeltaNet layer, its token-by-token recurrence, ~150-240 s per attention layer), ~1-2 GB. The golden
  (`tests/golden/long/3.5-0.8b-q8.npz`, 35 KB): per position the top 64 ids / logits and a fixed subset of 512 ids.

Runtime of runq for the same rows (`long_ctx -V`, 8 threads, load ~10): 524 s (the prefill up to each position plus
the two extra paths at each).

**Qwen3.5-0.8B Q8_0 vs the reference** (full vocabulary; KL = KL(ref || runq) of the softmax):

| position | 1,023 | 4,095 | 8,191 | 16,383 | 24,575 | 32,767 | 40,959 | 49,151 | 57,343 | 65,535 | 73,727 | 81,919 | 90,111 | 98,303 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cos | 0.99987 | 0.99990 | 0.99986 | 0.99987 | 0.99982 | 0.99990 | 0.99992 | 0.99974 | 0.99959 | 0.99991 | 0.99991 | 0.99975 | 0.99989 | 0.99971 |
| max \|diff\| | 0.20 | 0.23 | 0.30 | 0.28 | 0.30 | 0.32 | 0.27 | 0.28 | 0.44 | 0.29 | 0.40 | 0.30 | 0.29 | 0.44 |
| KL | 5e-4 | 2e-6 | 7e-4 | 9e-4 | 6e-5 | 2e-7 | 1e-4 | 1e-4 | 4e-7 | 8e-6 | 7e-6 | 8e-8 | 3e-4 | 4e-6 |

Top-1 equal at all 14 positions, top-5 5/5 at 13 (4/5 at 57,343). The agreement does not degrade with the
position: cos over the golden's 576 ids 0.99997 at 1k, 0.99989-0.99995 at 90-98k (the check's bound: 0.995 per
position), max |diff| 0.2-0.3 at every distance with two 0.44s. What was checked along the way:

- **Split / combine:** past 256 positions every path combines per-block partial results (`att_combine`); at 98k
  that is 384 blocks per query. Against the reference's exact softmax, nothing accumulates (above). Bitwise, at all
  14 positions: the last row of a 64-row prefill chunk == a 4-row verify segment (the decode-rows worker, its
  attention items per (row block, kv head)) == a decode step (`rq_decode`: each kv head's 384 blocks split over the
  threads, combined by the last item).
- **RoPE at large positions** (theta 1e7, 64 of 256 dims): runq's frequencies `expf((float)ln(theta) * -j / 32)`
  carry the fp32 rounding of ln(1e7) (relative error up to 6.7e-7, HF's `1 / theta ** (2j / 64)` 7e-8), but the
  angle's error is that of the fp32 product `pos * freq` in both: at 98,303 max 2.6e-3 rad from the exact angle for
  runq and for HF (runq - HF 2.0e-3), at 262,143 6.1e-3 for both (cos error 3.2e-3 / 2.4e-3). Not worth a change
  (a double-precision angle would beat HF's own computation and change every model's bits).
- **The f16 cache's range:** max |K| 22.3, |V| 37.2 over the 98k tokens (the reference's, per attention layer: K
  12.4-22.3, V 5.7-37.2): far from f16's 65,504, and f16's relative step (2^-11) is the precision the short tests
  already measure.
- **Retrieval:** a fact ("the access code of the old lighthouse at Penmarc'h is PLUM-7316-HERON") after 4,000 tokens
  of the corpus, the corpus up to 88,000 tokens, then the question in a chat turn (thinking off): an 88,092-token
  prompt; the 0.8B's greedy reply is `PLUM-7316-HERON` (prefill + 24 decode steps at ~88k: 602 s at 8 threads,
  load ~25).
- **Qwen3.5-4B Q8_0** (32 value heads on 16 key heads) on the first 32,768 tokens (runq at 98k would need ~7.7 GB:
  more than this run's share of the box; reference 1,233 s at 8 BLAS threads, `golden/long/3.5-4b-q8.npz`): cos over
  the golden's ids 0.99998 / 0.99993 / 0.99997 / 0.99996 / 0.99990 / 0.99990 at 1k / 4k / 8k / 16k / 24k / 32k, top-1
  6/6, max |diff| 0.16-0.41; the three paths bitwise equal, the generic build too (to 16k); max |K| 19.6, |V| 52.5;
  the fact retrieved from a 28,860-token prompt.
- No bug found: the long-context paths give the reference's logits and the invariants' bits.

## Commands

```
.venv-hf5/bin/python tests/make_golden.py --model-dir models/Qwen3.5-0.8B --out-dir tests/golden/models/3.5-0.8b-hf
.venv-hf5/bin/python tests/make_model_ref.py 3.5-4b-hf --hf models/Qwen3.5-4B      # ref_qwen35.py on HF bf16, ~30 s
.venv-hf5/bin/python tests/make_model_ref.py 3.5-4b-q8 models/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q8_0.gguf
tests/run_tests.sh --models --model-tags 3.5-4b-q8                     # the 4B's checks (~100 s)
.venv-hf5/bin/python tests/make_model_ref.py 3.5-0.8b-q8 models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf
.venv/bin/python tests/make_golden.py --tokenizer-qwen35          # golden/tokenizer_qwen35.json
.venv/bin/python tests/make_chat_golden.py                        # golden/chat_render.json (incl. qwen3.5)
.venv/bin/python tests/fuzz_tokenizer.py --model models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf --hf-dir models/Qwen3.5-0.8B
.venv-hf5/bin/python scripts/ppl.py bench/ppl_beagle_ch1.txt --hf --hf-dir models/Qwen3.5-0.8B --no-full \
    --runq-model models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf
.venv-hf5/bin/python tests/make_model_ref.py 3.5-0.8b-q4km models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf
python3 tests/make_dequant_golden.py --q5                         # golden/dequant/q5_k.npz (needs gguf)
tests/run_tests.sh --only q35                                     # the q35 group alone (~45 s, ~85 s with the Q4_K_M file)
```

Re-creating `.venv-hf5`: `python3 -m venv .venv-hf5 && .venv-hf5/bin/pip install torch==2.7.0 --index-url
https://download.pytorch.org/whl/cpu && .venv-hf5/bin/pip install "transformers>=5.2" safetensors numpy jinja2`
(5.21.0 used). Downloads: `https://huggingface.co/Qwen/Qwen3.5-0.8B` (config, tokenizer files, the
safetensors) and `https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF` (`Qwen3.5-0.8B-Q8_0.gguf`,
`Qwen3.5-0.8B-Q4_K_M.gguf`), `Qwen/Qwen3.5-4B` @ 851bf6e and `unsloth/Qwen3.5-4B-GGUF` @ e87f176
(`Qwen3.5-4B-Q8_0.gguf`). The tiny models of design 8 were made by a scratch script (transformers 5.21) and
converted with llama.cpp's converter in a scratch venv; neither is kept.
