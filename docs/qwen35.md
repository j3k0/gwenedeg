# Qwen3.5: design notes

Status (2026-10-26): **Qwen3.5-0.8B Q8_0 is supported** (GGUF `general.architecture == qwen35`): loader, the
`qwen35` pre-tokenizer, the chat template variant, the hybrid forward pass in every path (decode, batched prefill,
batched decode rows) with the bitwise invariants, recurrent-state checkpoints for truncate / copy, the CLI chat and
`runq serve`, and the tests (`tests/check_models.py` group `q35` in `make test`, the file in the `models` group).
Open: Q5_K for the Unsloth Q4_K_M file, the recurrence's speed, the bigger Qwen3.5 / 3.6 models ([TODO.md](../TODO.md)).
This file records what Qwen3.5 is, what was verified, and the design. Test model: **Qwen3.5-0.8B**: the same layer
types as the larger Qwen3.5 / 3.6 models, with a dense FFN instead of MoE.

Results (2026-10-26, 16-vCPU dev box):

- Logits vs HF transformers 5.21 fp32 (`tests/golden/models/3.5-0.8b-hf`, en / code / chat / long2, 62 positions):
  cos mean 0.9996-0.9998 per prompt, min 0.9986, top-1 62/62, pooled 0.99976; vs the Q8_0 file's dequantized-weights
  reference (`3.5-0.8b-q8`): pooled 0.99986. Perplexity on `bench/ppl_beagle_ch1.txt`, 512-token windows: runq Q8_0
  22.729, HF fp32 22.728 (full text: runq 18.86; HF's reference kernels ran out of memory there).
- Bit-identical: `-b 1 / 7 / 64 / 512`, 8 vs 3 threads, AVX-512 vs generic, KV f16 / bf16 / f32, a batched decode
  row vs the sequence alone (`api_test`), a CLI chat that rolls back to a checkpoint vs a single-shot run.
- Speed (8 / 12 threads, `scripts/bench_models.py`): prefill of 448 tokens 1,117 / 1,580 tok/s (Qwen3-0.6B Q8_0 in
  the same session: 1,817 / 2,494), decode 141 / 182 tok/s (0.6B: 209 / 230), after a 1,500-token prompt 160 / 174
  (0.6B: 182 / 183: the 6-layer KV cache barely slows it down); peak RSS 964 MB (1,500 + 128 tokens, `-c 4096`),
  startup 0.1 s. The recurrence takes ~45% of prefill (`-DRUNQ_PROFILE`, the "attention+quant" bucket): see
  Performance below.
- Tokenizer: `tests/golden/tokenizer_qwen35.json` (90 cases, 23 of them pre-tokenized differently from Qwen3's
  regex) and `fuzz_tokenizer.py` (5,000 random strings + the repo's text, 1.4 M tokens): equal to HF.

Tools: `models/Qwen3.5-0.8B` (HF safetensors, bf16, with the tokenizer files) and `models/Qwen3.5-0.8B-GGUF`
(Unsloth `Q8_0`, 812 MB, and `Q4_K_M`, 533 MB): `scripts/download_models.py qwen35`. `tests/ref_qwen35.py`: a numpy
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
(That estimate was wrong in practice: the recurrence is memory-bound, see Performance.) Implemented as `dn_head`
in `src/forward.c`: one token of one value head, the same function in every path. Work items: decode, one per
value head (after an alpha / beta item: a dependency counter); batched prefill, one (segment, head) for all the
segment's rows in order (the head's 64 KB state stays in that core's cache); batched decode rows, one per head over
all rows. Arithmetic order, fixed so that SIMD widths agree: elementwise along the channels / the v index
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
- A sequence keeps `RQ_SEQ_CK` = 2 **state checkpoints** (position + a copy of the conv + S state, 18.9 MB each,
  allocated at first use). `rq_seq_checkpoint_at(s, pos)` asks for one at a position >= the length: the forward
  pass that reaches it takes it (each (layer, head) work item copies its part right after that token, so a
  checkpoint can sit inside a prefill chunk), or it is taken at once if the sequence is there. A new one replaces
  the one at the lowest position.
- `rq_seq_truncate(s, n)` on a hybrid model, n < len, goes back to the largest checkpoint at a position <= n (its
  state copied back), else to 0; checkpoints past the new length are dropped. Callers read `rq_seq_len` afterwards
  and re-prefill the rest; `rq_seq_keep(s, n)` tells in advance what would be kept. `rq_seq_copy(dst, src, n)`
  likewise: src's current state if n is its length, else its best checkpoint <= n, else 0. The return codes did
  not change; a model without linear layers keeps the exact behaviour (Qwen3's outputs are byte-identical,
  `tests/compare_bins.sh`). The state of a position-0 token is zero (`dn_head` clears it), so a reset needs no work.
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

## Performance

Estimated before: decode ~0.8x Qwen3-0.6B's rate (the Q8_0 file reads ~810 MB per token vs ~640 MB; the tied
classifier is a third of it: 248,320 x 1,024), a lower long-context slowdown, prefill at the matmul rate plus a
~2.5% recurrence. Measured: decode as estimated (141 vs 209 tok/s at 8 threads; 160 vs 182 after 1,500 tokens).
Prefill is ~0.6x the 0.6B's instead of ~0.8x: the recurrence phase takes ~45% of it (`-DRUNQ_PROFILE`, 712-token
prompt, -b 512: 0.28 s of 0.61 s at 8 threads, 0.20 of 0.43 at 16). `dn_head` alone (one thread, a micro-benchmark)
takes ~5 us per token and head, ~4 of them in the two passes over the 64 KB state (read and write each): it is
bound by the cache bandwidth, not by the flops; in prefill, ~11 us per token and head at 8 threads. With one work
item per head, a segment has only 16 items. Next (TODO.md): split each head's v columns into blocks (each v lane's
recurrence is independent: the same bits; a 128 x 32 block, 16 KB, stays in L1, and a segment gets 64 items), with
the conv / l2norm and the gated norm as phases of their own; the decay folded into one pass.

## Commands

```
.venv-hf5/bin/python tests/make_golden.py --model-dir models/Qwen3.5-0.8B --out-dir tests/golden/models/3.5-0.8b-hf
.venv-hf5/bin/python tests/make_model_ref.py 3.5-0.8b-q8 models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf
.venv/bin/python tests/make_golden.py --tokenizer-qwen35          # golden/tokenizer_qwen35.json
.venv/bin/python tests/make_chat_golden.py                        # golden/chat_render.json (incl. qwen3.5)
.venv/bin/python tests/fuzz_tokenizer.py --model models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf --hf-dir models/Qwen3.5-0.8B
.venv-hf5/bin/python scripts/ppl.py bench/ppl_beagle_ch1.txt --hf --hf-dir models/Qwen3.5-0.8B --no-full \
    --runq-model models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf
tests/run_tests.sh --only q35                                     # the q35 group alone (~45 s)
```

Re-creating `.venv-hf5`: `python3 -m venv .venv-hf5 && .venv-hf5/bin/pip install torch==2.7.0 --index-url
https://download.pytorch.org/whl/cpu && .venv-hf5/bin/pip install "transformers>=5.2" safetensors numpy jinja2`
(5.21.0 used). Downloads: `https://huggingface.co/Qwen/Qwen3.5-0.8B` (config, tokenizer files, the
safetensors) and `https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF` (`Qwen3.5-0.8B-Q8_0.gguf`,
`Qwen3.5-0.8B-Q4_K_M.gguf`).
