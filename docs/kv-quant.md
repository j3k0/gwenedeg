# 8-bit and 4-bit KV cache (`-k q8`, `q4`, `q4k4`; TODO G3)

`-k` (runq, `runq serve`, `rq_seq_new`'s kv_type) takes three quantized cache types, each with the fp32 attention or,
as `-dot`, the bf16 dot-product attention of `bf16-dot`:

| `-k` | K | V | bytes / value (with scales) | KV at 98,304 positions, Qwen3.8-27B | for |
|---|---|---|---|---|---|
| `f16` (default) | f16 | f16 | 2 | 6.00 GiB | |
| `q8`, `q8-dot` | 8-bit | 8-bit | 1.06 | 3.19 GiB | every model: KL within the 16-bit types' range on Qwen3.5 / 3.8 |
| `q4`, `q4-dot` | 8-bit | 4-bit | 0.81 | 2.44 GiB | every model, when memory matters more than the last digits |
| `q4k4`, `q4k4-dot` | 4-bit | 4-bit | 0.56 | 1.69 GiB | Qwen3.5 / 3.8 only (Saluki, the 27B): **unusable on Qwen3** |

What it buys on the 16-vCPU / 31 GB box, the loader's estimate at `-c 98304 --slots 1 --checkpoints 3`, before the
draft's sequence (which shrinks the same way): Qwen3.8-27B Q4_K_M 22.37 GiB (f16) -> 19.56 (`q8-dot`) / 18.81 (`q4-dot`) /
18.06 (`q4k4-dot`); Saluki (IQ2-mix) 14.69 -> 11.87 / 11.12 / 10.37.

## Definition

Source of truth: forward.c, "8-bit / 4-bit KV cache". Per attention layer, position and kv head, the key row (after
QK-norm + RoPE) and the value row are **rotated**, x' = H D x (D: fixed signs, H: the Walsh-Hadamard matrix of order
head_dim, unnormalized), then **quantized per group of 32 dims** from that group's values alone:

- 8-bit: d = bf16 of (max |x'| / 127), its magnitude rounded up; n = rint(x' / d) in -127..127.
- 4-bit: m = the x' of the largest |x'|, d = bf16 of (m / -8), rounded up in magnitude; n = rint(x' / d) clamped to
  -8..7 (Q4_0's 16 levels).

The value read back is n x d, exact in fp32. The query is rotated alike, q' = H D q x 2^-log2(head_dim) (q'.x' = q.x;
the power of two is exact), and the attention's output comes back as o = D H o' x 2^-log2(head_dim). `q8` / `q4` /
`q4k4` run the fp32 attention (as `-k f32`) on the values read back, the `-dot` types the bf16 dot-product attention on
bf16 of them (a second rounding, as `f16-dot`'s). head_dim must be a power of two (Qwen3: 128, Qwen3.5 / 3.8: 256).

What this keeps:

- **The project's rule**: a position's bytes depend on its own rows only, so prefill chunks, decode, batches,
  truncation + re-prefill, copies and the disk tier store the same bytes; the logits are bit-identical for any `-b`,
  thread count and build (AVX-512 == generic C, tested; NEON runs the generic conversion).
- **The kernels**: each attention block's tiles are dequantized into the thread's scratch (the f16-dot pattern), then
  the existing fp32 / bf16-dot kernels run on it. Tile layouts: forward.c; the AVX-512 conversions in kernels_avx512.h
  (`att512_q_f32`, `att512_q_dot`) equal the C one (`kvq_block_f32`) bit for bit.

## Quality

KL divergence of runq's next-token distributions from the model's HF checkpoint in fp32, teacher-forced on prose and
code (`scripts/kl_kv.py`; the Q8_0 GGUF, so the baseline f32 / f16 is the weights' own error).

**Qwen3-0.6B** (attention in all 28 layers; 3 pieces, 10,338 tokens, 2026-10-31):

| `-k` | KL | vs f32 | top-1 = HF's |
|---|---|---|---|
| f32 | 0.00235 | | 97.2% |
| f16 | 0.00237 | +1% | 97.4% |
| bf16-dot | 0.00246 | +5% | 97.3% |
| q8 | 0.00366 | +56% | 96.8% |
| q8-dot | 0.00378 | +61% | 96.5% |
| q4 | 0.0115 | x4.9 | 94.5% |
| q4-dot | 0.0115 | x4.9 | 94.3% |
| q4k4 | **1.51** | x640 | **50%** |

**Qwen3.5-0.8B** (attention in 6 of 24 layers, as Qwen3.8-27B's 16 of 64; 3 pieces, 4,154 tokens, 2026-10-31):

| `-k` | KL | vs f32 | top-1 = HF's | perplexity vs f32 |
|---|---|---|---|---|
| f32 | 0.00125 | | 97.5% | |
| f16 | 0.00124 | -1% | 97.8% | -0.1% |
| bf16-dot | 0.00126 | +0% | 97.7% | +0.05% |
| q8 | 0.00131 | +5% | 97.8% | -0.1% |
| q8-dot | 0.00130 | +4% | 97.7% | -0.1% |
| q4 | 0.00774 | x6.2 | 94.7% | +0.9% |
| q4-dot | 0.00777 | x6.2 | 94.3% | +0.9% |
| q4k4 | 0.00924 | x7.4 | 93.7% | +1.1% |
| q4k4-dot | 0.00926 | x7.4 | 93.9% | +1.2% |

So on this family `q8` is free (within the 16-bit types' noise), and the 4-bit types cost ~1% perplexity, most of it
V's; 4-bit K adds ~20% to `q4`'s KL. For scale: Saluki's own IQ2-mix weights cost +18% perplexity over the Q3_K_M.

Why the K / V split: K and V cost very differently, and in opposite ways in the two families. A simulation (scratch
build: the f32 cache with each row quantized and dequantized as above, fp32 scales; KL as above):

| quantized | Qwen3-0.6B (6,859 tokens) | Qwen3.5-0.8B (4,500 tokens) |
|---|---|---|
| none (f32) | 0.00244 | 0.00125 |
| K 4-bit | 1.65 | 0.00267 |
| V 4-bit | 0.0101 | 0.00764 |
| K 8-bit + V 4-bit (`q4`) | 0.0114 | 0.00768 |
| K 6-bit / 5-bit | 0.0232 / 0.094 | 0.00133 / |
| K 4-bit, groups of 16 / asymmetric (min-max) | 1.50 / 1.00 | |

4-bit K breaks Qwen3 (sharp attention in every layer: the score error of 4-bit K is a few logits), and neither smaller
groups nor an asymmetric grid save it; V tolerates 4 bits in both. Hence `q4` keeps K at 8 bits everywhere, and
`q4k4` is the opt-in for the hybrid models, where 4-bit K costs little.

Not measured on the 27B itself (no HF checkpoint of Qwen3.8-27B here): Qwen3.5-0.8B, the same architecture with the same
1-in-4 attention layers and head_dim 256, is the proxy for Saluki and the Qwen3.8-27B files. A check on the 27B: KL
between its f16 and quantized caches (runq vs runq, `scripts/kl_quant.py`'s method with two `-k` instead of two files).

**The rotation** (Qwen3-0.6B, `q8` without it, 2 pieces): KL 0.0222 / 0.0174 vs 0.0043 / 0.0034 rotated: ~5x. K's
outlier dims (|K| up to ~500 in a few dims, TODO G3) otherwise set their group's scale; the rotation spreads them over
the head. (Without it, 4-bit is unusable everywhere: Qwen3-0.6B KL 2.5-3.2.)

In `make test` (group kv, the Qwen3-0.6B `.bin`): `q8` passes the Q8_0 thresholds vs HF; `q8-dot` and the 4-bit types
have their own, set from their measured values (tests/check.py, `KV_TH`: `q8-dot` one position at cos 0.978, `q4` pooled
cos 0.9949); all are bit-exact across `-b`, threads and builds. `q4k4` is measured here, not in that group (Qwen3).

## Speed

Saluki 27B (IQ2-mix), 16 threads, `build/bench_long -L 16384,65536 -R 4 -r 5` (synthetic fill), best of two passes of
5, the box at load ~13-15 from other work (so +-10%):

| `-k` | decode at 64k, ms | verify segment of 4 rows at 64k, ms |
|---|---|---|
| f16 | 126-131 | 260-279 |
| bf16-dot | 132-142 | 226-246 |
| q8 | 136-141 | 265-278 |
| q8-dot | 125 | 232-238 |
| q4-dot | 123-134 | 220-242 |
| q4k4-dot | 103-123 | 210-243 |

At 16k every type decodes in ~91-110 ms: Saluki's step is its ~8 GB of weights. At 64k the quantized `-dot` types are
~5-20% faster than f16 per decode step, `q4k4-dot` the most; the non-dot `q8` is not faster (the fp32 attention, plus the
conversion). The memory is the solid gain; the speed one grows with the context.

Where the speed is left: each block is dequantized into scratch before the attention reads it (~as many instructions
as the attention's own at decode). A fused kernel (K as int8 against a quantized q with VNNI, the 32-dim scales per
ATT_DC chunk; V's scale folded into the softmax weight) would read the quantized bytes once; that and NEON's own
conversion are left (TODO G3).

## Usage

```sh
# Qwen3.8-27B Q4_K_M at 98k: the loader's estimate 19.56 GiB (f16: 22.37) + the draft (its KV shrinks alike)
OMP_NUM_THREADS=16 runq serve models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf -c 98304 -k q8-dot \
    --draft models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf --slots 1 --parallel 1 --checkpoints 3 \
    --spec-calibrate 32768 --timeout 0 --name qwen3.8-27b --port 18080
# Saluki at 98k, speed first: 10.37 GiB (f16: 14.69) + the draft
OMP_NUM_THREADS=16 runq serve models/Underdog-Saluki-27B-1.0-GGUF/Underdog-Saluki-27B-1.0-IQ2-mix.gguf -c 98304 -k q4k4-dot \
    --draft models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf --slots 1 --parallel 1 --checkpoints 3 \
    --spec-calibrate 32768 --timeout 0 --name saluki-27b --port 18080
```

Rerun the measurements:

```sh
.venv-hf5/bin/python scripts/kl_kv.py --hf models/Qwen3.5-0.8B --gguf models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf \
    --kv f32,f16,q8,q8-dot,q4,q4-dot,q4k4,q4k4-dot --pieces 3 --tokens 1500   # (~1.5 GB per temp file: $TMPDIR)
OMP_NUM_THREADS=16 build/bench_long MODEL -L 16384,65536 -R 4 -r 5 -p 0 -C 0 -k q4k4-dot
```
