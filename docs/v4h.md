# V4H: a VNNI-native 4-bit weight and activation format

Status: **prototyped 2026-11-24 on Qwen3.5-0.8B and in a kernel micro-benchmark; not integrated** (results:
[section 9](#9-prototype-results-2026-11-24)). V4H is a new quantization, not a lossless repack of an existing GGUF type.
The short version is in [IDEAS.md](../IDEAS.md#v4h-a-vnni-native-4-bit-weight-and-activation-format). The starting points
are the Q8P layout in `src/kernels_avx512.h` and the i-quant layouts and kernels in `src/iquants.h`; current measurements
are in [iquants.md](iquants.md).

## Summary

AVX-512 VNNI cannot consume packed 2-, 3- or 4-bit weights directly. Every `vpdpbusd` still needs one byte per weight, so a
smaller bitstream does not increase the instruction's MAC count. The useful optimization is therefore to co-design the weight
codes, scale representation, activation quantization and kernel so that:

1. packed weights expand to VNNI-ready bytes with only masks and shifts;
2. each vector lane remains an independent output row, with no horizontal reductions;
3. fine-grained scales are applied in integer arithmetic; and
4. floating-point conversion and scaling happen once per 256 inputs, not once per 32.

V4H does this with linear 4-bit codes and hierarchical integer subscales. Its weights occupy **4.1875 bits/weight**, exactly the
same as the current IQ3_S/r and IQ3_XXS/r repacks. Unlike those formats, its nibbles need no `vpshufb`, and its 16-bit VNNI form,
`vpdpwssd`, can apply the combined weight/activation subscales without packing or narrowing the dot-product results.

The realistic prefill target is the already-measured float-fold overhead: current one-scale tiled GEMMs take 0.0038-0.0039 ns
per weight and activation row, versus 0.0031 in a diagnostic with no fold. That is about **1.25x kernel headroom**, not another
4x. Direct nibble decoding may help GEMV and small batches more. At a memory-bound decode limit, V4H has the same weight traffic
as IQ3_S/r and should not win merely because its kernel is simpler.

## 1. What the instructions dictate

`vpdpbusd` multiplies unsigned bytes by signed bytes. In its 512-bit form it computes 64 products and accumulates them into 16
int32 dword lanes, four products per lane. The Q8P arrangement is consequently the natural one for Gwenedeg:

- a dword lane is one of 16 output rows;
- its four bytes are four consecutive input columns;
- a weight vector covers 16 rows x 4 columns;
- an activation dword is broadcast to all lanes; and
- each lane accumulates one output, so no horizontal reduction is needed.

For signed weight codes `q`, store `u = q + B` as unsigned bytes and correct the result with activation sums:

```
sum(u * z) = sum(q * z) + B * sum(z)
sum(q * z) = sum(u * z) - B * sum(z)
```

The activation sums are prepared once per activation row and reused for every weight row. This is cheaper and simpler than
sign-extending every decoded weight. It is also why the existing kernels use unsigned biased weights and signed activations,
rather than swapping the operands.

Reducing the stored bit width still reduces memory traffic, but it cannot reduce the number of `vpdpbusd` instructions per
weight: after decoding, 64 weights always need one 512-bit instruction. Trying to feed packed nibbles directly would treat each
byte as `lo + 16 * hi`; the products and sums overlap, so the two dot products cannot generally be recovered. Bit-plane tricks
change the decoding cost, not this fundamental VNNI throughput limit.

## 2. V4H representation

Use a superblock of 16 output rows and 256 input columns, divided into eight groups of 32 columns.

### Weights

For output row `r`, superblock `b`, group `g` and column `j`:

```
q[r,j] in -8 .. 7
u[r,j] = q[r,j] + 8                    # stored as a 4-bit code, 0 .. 15
s[r,b,g] in 0 .. 15                    # unsigned 4-bit integer subscale
D[r,b]                                # fp16 super-scale
w[r,j] = D[r,b] * s[r,b,g] * q[r,j]
```

`D` is per output row and 256 columns. `s` retains a finer, per-row scale every 32 columns without introducing a floating-point
scale into the inner loop.

### Activations

For one activation row, superblock `b` and group `g`:

```
z[j] in -127 .. 127                     # signed int8, deliberately excluding -128
a[b,g] in 0 .. 255                      # unsigned 8-bit integer subscale
A[b]                                    # fp32 super-scale
x[j] = A[b] * a[b,g] * z[j]
```

Both sides therefore have a floating-point super-scale and an integer fine scale. Their product is what allows eight groups to
share one final float fold:

```
T[r,b,g] = sum over j in g of q[r,j] * z[j]
C[r,b,g] = s[r,b,g] * a[b,g]
I[r,b]   = sum over g in b of C[r,b,g] * T[r,b,g]
y[r]    += (D[r,b] * A[b]) * float(I[r,b])
```

The activation subscale representation is a design choice, not yet a validated quantization algorithm. One possible construction
starts with the usual desired group scales `sigma_g = max(abs(x_g)) / 127`, chooses `A = max_g(sigma_g) / 255`, then chooses
`a_g = ceil(sigma_g / A)` (clamped to 1 .. 255), so `A * a_g >= sigma_g` and quantization does not clip. The zero block needs its
own trivial handling. Whether this scale discretization preserves model quality is an empirical question. An 8-bit `a` costs only
0.25 bit per activation and is still cheaper than an fp32 scale per 32 values.

The weight quantizer likewise remains to be designed. It should start from a higher-precision model; requantizing an IQ3 file
would not be lossless and would make the quality result difficult to interpret.

## 3. Weight storage and kernel layout

The in-memory layout can reuse the sizes and ordering of IQ3_S/r. Per 16-row block and 256-column superblock:

| Offset | Size | Contents |
|---:|---:|---|
| 0 | 32 bytes | `D[16]`, one fp16 per output row |
| 32 | 64 bytes | eight 4-bit `s` values per output row |
| 96 | 2048 bytes | the 16 x 256 4-bit `u` codes |
| **Total** | **2144 bytes** | **4.1875 bits/weight** |

As in the existing repacks, a 16-row block is contiguous over all input superblocks, so GEMV streams it sequentially. The scales
can use one dword per row, with nibble `g` holding group `g`'s scale.

For each group of 32, the 256 code bytes are four 64-byte chunks. In chunk `c = 0 .. 3`, dword lane `r` and byte `j = 0 .. 3`:

- the low nibble is column `32*g + 8*c + j` of row `r`;
- the high nibble is column `32*g + 8*c + 4 + j` of row `r`.

A chunk load therefore gives two VNNI-ready vectors after nibble extraction:

```c
__m512i p  = load64(chunk);
__m512i w0 = and(p, set1_epi8(15));
__m512i w1 = and(srli_epi16(p, 4), set1_epi8(15));
```

Per 16 rows x 32 columns this takes four packed loads and 12 mask/shift instructions, producing eight vectors. IQ3_S/r performs
the same extraction but additionally needs one `vpshufb` per output vector to map its sign/magnitude codebook entries to biased
values. V4H's codes are already the biased values, so those eight shuffles disappear.

## 4. Dot product and bias correction

For each group of 32 activations, prepare `X = sum(z)`. Initialize the integer accumulator to `-8 * X`, then run the eight
`vpdpbusd` steps over the decoded `u` vectors:

```
P = -8 * X
for step in 0 .. 7:
    P = vpdpbusd(P, u[step], broadcast4(z + 4*step))
T = P
```

Thus `T = sum((q + 8)z) - 8 sum(z) = sum(qz)`. The current per-16 activation sums can be combined to produce `X`; no per-weight
work is needed.

For one activation row and one 16-row weight block, eight `vpdpbusd` instructions cover 16 x 32 = 512 weights, as in the current
one-scale i-quant kernels. For several activation rows the decoded vectors are reused. A tiled prefill kernel can either retain
the current decode-to-scratch strategy or decode into registers when the register tile permits it; that choice is orthogonal to
the file format.

## 5. Applying subscales with `vpdpwssd`

The other AVX-512 VNNI operation, `vpdpwssd`, multiplies adjacent signed int16 pairs and accumulates their sum into each int32
dword. V4H can use it directly on `T` and `C`:

```c
C = mullo_epi16(S, set1_epi16(a));  # S and a are in the low words; high words are zero
I = dpwssd(I, T, C);
```

No `vpmovdw`, pack, permutation or mask is required. For each dword lane, the operation is effectively:

```
I += int16(T low word)  * int16(C low word)
   + int16(T high word) * int16(C high word)
```

The second product is zero because `C`'s high word is zero. The first product is the complete signed `T * C`, because the bounds
below guarantee that `T` fits signed int16. Sign extension in `T`'s high word is irrelevant.

### Exact bounds

The apparently arbitrary activation restriction to `-127 .. 127` is what makes this work for all inputs, including the asymmetric
4-bit endpoint `q = -8`:

```
max positive T = 32 * (-8) * (-127) =  32512
min negative T = 32 * (-8) *   127  = -32512
```

Both fit int16 (`-32768 .. 32767`). Using activation `-128` would permit a positive 32768 and break that guarantee.

The combined scale and superblock accumulator also fit:

```
C = s * a <= 15 * 255 = 3825                    # fits positive signed int16
|I| <= 8 * 32512 * 3825 = 994,867,200           # fits int32
```

These are worst-case bounds over eight groups, not probabilistic assumptions. The non-saturating `vpdpwssd` is therefore exact.

The alternative is `I = vpaddd(I, vpmulld(T, C))`. It uses two instructions instead of one but may avoid competition for the
VNNI execution ports. This is a microarchitectural benchmark question, not something the instruction counts alone settle.

After all eight groups, convert `I` to fp32 and fold with `D * A`, in a defined superblock order. Very large `I` values need not
be exactly representable as fp32; the int32-to-fp32 conversion is part of the format's arithmetic definition, just as the current
conversions are. Generic, AVX-512 and NEON implementations must perform the same integer arithmetic and explicitly rounded float
operations. This proposal does not relax the project's bit-identity rule.

## 6. Expected effects by workload

### Large tiled prefill

The current tiled path decodes a weight tile once and reuses it across activation rows. Consequently, removing V4H's LUT shuffles
is not the main prefill opportunity. The opportunity is replacing eight per-32 float folds with one per-256 fold.

The measured reference points on the 16-vCPU Zen 4 VM are:

- IQ3_S/r tiled: 0.0038 ns per weight and activation row;
- other one-scale types: about 0.0039;
- diagnostic with the float fold removed, producing wrong results: 0.0031.

The fold is therefore about 25% of current runtime, or a 1.25x speed ratio if it were entirely free. V4H still pays for integer
subscale combination, loads and tile decoding, so 0.0031 is a target/headroom indicator, not a prediction. An end-to-end prefill
gain would be smaller and depends on phases that do not use this kernel.

### GEMV and small batches

Decoding is not amortized as heavily here. IQ3_S/r already improved single-thread GEMV from 0.043 to 0.0117 ns/weight by replacing
the IQ3_S codebook decode with 4-bit codes and a shuffle. V4H removes that shuffle as well and reduces the fold frequency, so it
has a plausible further gain. No numeric gain is claimed before a microbenchmark.

### Memory-bound decode

V4H and IQ3_S/r both use 4.1875 bits/weight. When all threads are limited by DRAM, their simpler or more complex arithmetic does
not change the traffic, so V4H should not be expected to beat IQ3_S/r in that regime. The original IQ3_S file is smaller still
(3.4375 bits/weight), and can be preferable when its decoding is no longer the bottleneck. This is why a custom format is not
automatically a universal decode win.

## 7. Alternatives and variants

### V4-flat

Remove `s` and `a`, leaving one scale per 256 columns, or eventually one per whole row. This is the speed ceiling: nibble decode,
`vpdpbusd`, bias correction and a final float fold, with no `vpdpwssd` or integer scale multiply. A 256-column weight scale gives
4.0625 bits/weight; a whole-row scale approaches 4 bits/weight.

The risk is quantization quality, especially activation outliers. Input-channel equalization, folding fixed channel scales into
RMSNorm and the following matrices, Hadamard rotations, or quantization-aware training could make larger scale groups practical.
Those are layer-level co-design choices and have their own online cost; they are not prerequisites for V4H's hierarchical version.

### V3H

Use the same hierarchy with 3-bit codes. With a 4-bit weight subscale every 32 and an fp16 super-scale every 256, the nominal size
is `3 + 4/32 + 16/256 = 3.1875` bits/weight. This can reduce DRAM traffic relative to V4H and IQ3_S, but it requires more
unpacking and does not increase VNNI's compute rate. It is a decode/bandwidth experiment with additional quality risk, not the
first prefill prototype.

### V2 and binary formats

Lower widths reduce memory further but require more extraction work per VNNI-ready vector. They do not reduce the number of
`vpdpbusd` operations per weight. They make sense only if memory savings dominate and model quality survives.

### Non-uniform 4-bit codebooks

A 16-entry `vpshufb` table can represent arbitrary levels and may improve quality per bit. IQ4_XS and IQ3_S/r already demonstrate
the technique. V4H deliberately trades that freedom for raw linear nibbles and simpler scaling. A non-uniform V4H variant remains
possible, but its decode cost should be expected to resemble the existing 4-bit-code kernels.

### Expanded int8 weights

Q8P has the simplest possible hot loop and no nibble decoding, but roughly doubles weight traffic relative to V4H. It is useful as
a VNNI compute-floor reference, not as the low-bit storage answer.

## 8. Prototype and decision plan

1. **Scalar arithmetic first.** Implement a small V4H reference and tests for the bias correction, subscale combination and float
   fold. Include the extrema `q = -8`, `z = -127`, `s = 15`, `a = 255`, and verify the int16/int32 bounds directly.
2. **Microkernel before a model converter.** Generate synthetic packed V4H tiles and compare `vpdpwssd` with `vpmulld + vpaddd`.
   Start from `iq512_group` and `iqx_tile` in `src/iquants.h`, retaining Q8P's 16-lane output arrangement.
3. **Benchmark the regimes separately.** Use 1, 4, 8, 64 and 512 activation rows; hot-cache and DRAM-resident weights; relevant
   thread counts. Compare against IQ3_S, IQ3_S/r, IQ4_XS and Q2_K on the same shapes and session. Useful existing shapes are
   1024 x 5120 for GEMV and 17408 x 5120 at 512 rows for tiled prefill.
4. **Measure quantization quality.** Build a quantizer from a higher-precision checkpoint, then compare KL/perplexity for V4H and
   V4-flat. Treat the activation subscale choice as part of the quantizer, not just kernel metadata.
5. **Integrate only if both sides win.** A production format needs its file/block definition, loader and packing, generic and NEON
   arithmetic, dequantization, model-quality results, and the normal bitwise tests across paths, thread counts and batches.

Open questions:

- Does `vpdpwssd` beat `vpmulld + vpaddd` on Zen 4 after accounting for execution-port pressure?
- How much of the diagnostic no-fold headroom survives the integer subscale work?
- Does direct nibble decoding materially improve one-row and small-batch GEMV?
- Does the 8-bit activation subscale preserve quality, and how should `A` and `a` be chosen?
- Can uniform 4-bit weights with 4-bit subscales match the required model quality without QAT?
- Does any gain survive complete forward passes and memory-bound phases?

The design is worth testing because it attacks measured overhead with the same storage cost as the existing fast repack. It is not
yet evidence that a custom model format will be faster end to end or accurate enough to ship.

## 9. Prototype results (2026-11-24)

Steps 1-4 of section 8, at small scale. The short answer: **most of the prefill gain needs only a different activation format,
not V4H.** With one activation scale per 256 values, the existing IQ3_S/r and IQ4_XS tiles already reach 1.2x, with no new
quantization. V4H adds about 1.1x on top of that, at the cost of a new quantizer and, on the 0.8B model, worse quality than
IQ4_XS for the same bits.

Tools (none is used by gwenedeg itself):

- `tools/v4h_quant.py`: the V4H weight quantizer (numpy). It reads a BF16 GGUF and writes the V4H weights into a copy of a
  Q8_0 GGUF of the same model, the *carrier*: each Q8_0 block holds `q` and `d = fp16(D s)`, which dequantizes to `D s q`
  (relative error at most 2^-11 from rounding the product). So llama.cpp and gwenedeg can measure V4H's quality before a
  file type exists. For each group of 32 it searches candidate scales, weighting values as IQ4_XS does: the imatrix times
  `sqrt(sigma2 + x^2)`, or `x^2` without an imatrix. Each candidate is refit by least squares. It then searches `D` per
  superblock, with the subscales rounded and `D` refit. Variants: `v4h`, `flat256` (V4-flat) and `s6` (6-bit subscales,
  an ablation).
- `GWENEDEG_ACTQ_SIM=v4h|flat` (`forward.c` `quantize_sim`, off by default): a diagnostic that quantizes activations as a
  V4H kernel would. `v4h` uses `A` per 256 and `a_g = ceil(sigma_g / A)`; `flat` uses one scale per 256 (llama.cpp's Q8_K).
  The result is written as the usual per-32 scales, so every existing kernel computes the same products.
- `tools/v4h_kl.py`: KL of gwenedeg runs against a reference run, using `build/long_ctx -A`, as `scripts/kl_quant.py` does.
- `tests/kernels/v4h_bench.c` (`make v4h-bench`): the packing and the scalar definition of section 2, plus kernels that are
  bit-identical to it (checked on 1 to 77 rows and on the extremes `q = -8`, `s = 15`, `z = -127 / 127`, `|T| = 32512`).
  It also has a micro-benchmark against IQ3_S/r and IQ4_XS, run in the same process.

### Quality, Qwen3.5-0.8B

Source: unsloth's BF16 GGUF. All files were made with llama-quantize `--pure` (llama.cpp `225a534`), with `token_embd` (the
tied output) and `ssm_alpha` / `ssm_beta` at Q8_0. The V4H carriers use the same tensor set: every other matrix, 484M
weights. The imatrix is unsloth's `imatrix_unsloth.gguf_file`. KL is against BF16 (llama-perplexity `--kl-divergence`,
512-token chunks): wikitext-2 test, 48 chunks (12k scored tokens), and llama.cpp C++ source, 32 chunks (8k).

| file | bits/weight | wiki KL | wiki PPL / BF16 | wiki same top-1 | code KL | code PPL / BF16 |
|---|---:|---:|---:|---:|---:|---:|
| Q8_0 | 8.5 | 0.0012 | 1.002 | 97.8% | 0.0007 | 1.001 |
| Q4_K, imatrix | 4.5 | 0.048 | 1.055 | 88.6% | 0.030 | 1.041 |
| IQ4_XS, imatrix | 4.25 | 0.061 | 1.062 | 86.9% | 0.041 | 1.066 |
| **V4H, imatrix** | 4.1875 | **0.092** | 1.176 | 83.6% | **0.050** | 1.055 |
| V4H, 6-bit subscales | 4.4375 | 0.090 | 1.169 | 83.9% | 0.049 | 1.053 |
| V4H, no imatrix | 4.1875 | 0.114 | 1.178 | 82.3% | 0.064 | 1.073 |
| Q4_0, imatrix | 4.5 | 0.097 | 1.181 | 84.1% | 0.054 | 1.071 |
| V4-flat, imatrix | 4.0625 | 0.171 | 1.289 | 78.8% | 0.112 | 1.144 |
| IQ3_S, imatrix | 3.4375 (4.1875 as /r) | 0.180 | 1.112 | 78.6% | 0.115 | 1.115 |

- V4H falls between IQ3_S and IQ4_XS. At IQ3_S/r's in-memory size it has half llama.cpp IQ3_S's KL. It has 1.5x / 1.2x
  IQ4_XS's KL (wiki / code), although IQ4_XS costs only 1.5% more bits. Quality per bit therefore depends on IQ4_XS's
  non-uniform levels, not on the subscale width: 4-bit subscales cost almost nothing compared with 6-bit ones
  (0.092 vs 0.090).
- V4-flat is as bad as IQ3_S. One weight scale per 256 is not practical without the rotation or equalization of section 7.
- The uniform symmetric formats (V4H, Q4_0, V4-flat) lose much more wikitext perplexity than their KL suggests: +18% for
  V4H, against +11% for IQ3_S at twice V4H's KL. The mean probability of the text's token drops by 1.8 points, against
  0.8 for IQ4_XS. This is unexplained; the quantizer's least-squares scale refit, which shrinks scales, is a suspect.
- These are not the 27B's numbers. The 0.8B model is more sensitive to quantization. ISTA's 27B IQ3_S file also comes from
  a GSQ/RCO quantizer, which is much better than llama-quantize; V4H was quantized round-to-nearest with an imatrix.

Activation formats, measured in gwenedeg (`tools/v4h_kl.py`; KL against the Q8_0 file with the current per-32
activations; 8,192 tokens each of wikitext and code):

| weights | activations | wiki KL | code KL |
|---|---|---:|---:|
| Q8_0 | V4H (`A` per 256, u8 `a` per 32) | 0.0010 | 0.0003 |
| Q8_0 | flat (one scale per 256) | 0.0036 | 0.0010 |
| V4H | per 32 / V4H / flat | 0.1032 / 0.1019 / 0.1036 | 0.0271 / 0.0271 / 0.0277 |
| IQ4_XS / Q4_K / IQ3_S (imatrix) | per 32 | 0.057 / 0.047 / 0.180 | 0.021 / 0.015 / 0.064 |

Next to 4-bit weight error, the activation format is invisible, even with one scale per 256 (llama.cpp's CPU i-quants and
K-quants use that format). The activation subscale `a` of section 2 is therefore unnecessary.

### Kernels (`build/v4h_bench -b NB D K 7`, best of 7 interleaved, the day server running: load 2-5)

Kernels compared:

- *rows*: GEMV and small batches from the packed blocks. V4H: nibbles by `and` / shift, `T_g` by `vpdpbusd`, then
  `I += s_g T_g` by `vpdpwssd` or `vpmulld + vpaddd`, and one fold per 256.
- *tiled-u*: the proposal's tile. The `u` codes are decoded to scratch; per group, `T_g` and one `vpdpwssd`; one fold per 256.
- ***tiled-p***: the tile decoded once to the signed bytes `s q`. This needs `|s q| <= 120`, which only V4H's ranges give.
  `s u` takes one `vpmullw` on byte pairs, since `s u <= 225` cannot carry into the next byte; then `8 s` is subtracted.
  The activations are stored as u8 `z + 128`, and a bias of `-128 sum(s q)` per row is precomputed at decode. Per
  superblock this leaves 64 `vpdpbusd` and one fold, with no per-group work at all. It uses a 2 x 5 register tile (6 and
  up spill).
- *tiled-flat*: the control, with no new format. It uses the existing IQ3_S/r or IQ4_XS scratch (`iqx_unpack_tile`) and
  flat activations. Per group it computes `T` and `I += S T` with `vpmulld` (`|T|` exceeds int16 for these types), with one
  fold per 256. Checked against `iq_rows` with `xs = A`: equal up to float rounding (relative 2e-7).

ns per weight and activation row (x threads); in parentheses, speed relative to IQ3_S/r's faster path:

| shape, threads | rows | IQ3_S/r | IQ4_XS | IQ3_S/r flat | IQ4_XS flat | V4H rows | V4H tiled-u | **V4H tiled-p** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 x 5120, 1 thr | 512 | 0.00374 | 0.00378 | 0.00301 (1.24) | 0.00300 (1.25) | 0.00410 | 0.00302 (1.24) | **0.00271 (1.38)** |
| | 64 | 0.00369 | 0.00367 | 0.00312 (1.18) | 0.00302 (1.22) | 0.00369 | 0.00300 (1.23) | **0.00270 (1.37)** |
| | 8 | 0.00491 (rows) | 0.00447 (rows) | 0.00558 | 0.00532 | **0.00388 (1.27)** | 0.00465 | 0.00460 |
| | 1 | 0.00923 | 0.00763 | | | **0.00712 (1.30)** | | |
| 17408 x 5120, 8 thr | 512 | 0.00393 | 0.00397 | 0.00331 (1.19) | 0.00336 (1.17) | 0.00467 | 0.00330 (1.19) | **0.00292 (1.34)** |
| | 64 | 0.00454 | 0.00457 | 0.00377 (1.20) | 0.00379 (1.20) | 0.00486 | 0.00367 (1.24) | **0.00339 (1.34)** |
| | 8 | 0.00830 (rows) | 0.00875 (rows) | 0.00981 | 0.00977 | 0.00759 (1.09) | 0.00711 (1.17) | 0.00736 (1.13) |
| | 1 | 0.03645 | 0.03620 | | | **0.02806 (1.30)** | | |
| 65536 x 5120, 8 thr (DRAM) | 1 | 0.04028 | 0.04027 | | | 0.03884 (1.04) | | |

- **The activation format accounts for most of the prefill gain.** IQ3_S/r and IQ4_XS with flat activations reach
  1.17-1.25x. That is about the whole headroom of section 6, and it would apply to ISTA's existing 27B file.
- V4H's own gain comes from the premultiplied tile: 1.34-1.38x against IQ3_S/r, or 1.10-1.13x over the flat-activation
  control. The proposal's tile with `vpdpwssd` (tiled-u) gives no more than the control. Whether `vpdpwssd` or
  `vpmulld + vpaddd` is faster depends on the case and changes between runs (open question 1: no consistent difference).
- GEMV and small batches gain 1.1-1.3x when the weights are in cache (no `vpshufb`, a fold per 256). In DRAM-bound decode
  the gain is 1.04x, as section 6 predicted.

### What this means for the plan

1. **Flat activations for the existing formats come first.** Use one activation scale per 256, accumulate the integer
   subscales per group, and fold once per 256, in the tiled GEMM, the rows kernels and the generic and NEON definitions.
   This is a new arithmetic definition, not a new quantization. Expected: 1.2x on the tiled GEMM kernel for the ISTA IQ3_S
   file, at an activation cost of KL +0.001-0.004 on the 0.8B model. It must be measured on the 27B: end-to-end prefill,
   and KL against the current definition.
2. **V4H only if a 27B V4H file can match ISTA IQ3_S's quality.** Its remaining advantage over item 1 is about 1.1x on
   prefill and 1.1-1.3x on cache-resident GEMV. On the 0.8B model its quality per bit is worse than IQ4_XS. Better rounding
   (GPTQ-style error feedback), and the unexplained perplexity loss above, come before any 27B quantization. Quantizing the
   27B needs its BF16 checkpoint (about 54 GB) and a reference run that does not fit beside the day server.

## 10. Flat activations for the existing types: kernel measurement (2026-11-24)

This is item 1 of the plan above, as a micro-benchmark only: nothing in `forward.c` changes yet.
`tests/kernels/flat_bench.c` (`make flat-bench`) covers every i-quant type of ISTA's 27B file. The definition, `iqf_dot`
(generic C), is the current one, except for where the fold happens:

```
I = sum over the 8 groups of (S_0 T_0 + S_1 T_1)            int32, exact: |I_g| < 2^23 (IQ4_XS < 2^24), so |I| < 2^27
val = val + (D * A[s]) * (float) I                          one fp32 scale A per 256 activations, each operation rounded
Q2_K: M = sum over the groups of (m_0 X_0 + m_1 X_1),  val = val + ((D * A) * (float) I - (Dm * A) * (float) M)
```

The codes, `D`, `S`, `c16`, the decode (`iq512_group`) and the tile scratch (`iqx_unpack_tile`) are today's. The AVX-512
rows kernel (`iqf_blk`: GEMV, verify and batched decode) and the tiled kernel (`iqf_tile`) are bit-identical to `iqf_dot`
for all 11 types (the file's 8, plus IQ1_S and the non-repacked IQ3_S / IQ3_XXS). The test covers 1 to 77 rows, k up to
17408, activations at ±127, and both decodes, VBMI and gather: 143 cases. A deliberate off-by-one in the tiled kernel
fails 261 comparisons.

Q2_K needs the activation prep to store each group's pair `X_0 | X_1 << 16`, one int32 per 32 activations. With the pair
rebuilt from `c16` inside the kernel, as today's Q2_K tile does, the mins term costs the whole gain (0.98x). With the
stored pair, summed in the group loop, the gain is 1.28x. Today's kernel could use the stored pairs too, so part of Q2_K's
gain is not from the activation format; Q2_K is 2.7% of the weights.

ns per weight and activation row (x threads), best of 5, day server running (load 2-4). Ratio = today's faster path /
the flat format's faster path:

| type (share of the 27B file) | today rows | today tiled | flat tiled | 512 rows, 17408 x 5120 | 512 rows, 5120 x 17408 | 512 rows, 1 thr, 1024 x 5120 | 24 rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| IQ3_S/r (31.0%) | 0.00615 | 0.00403 | 0.00327 | 1.23 | 1.22 | 1.23 | 1.14 |
| IQ4_XS (20.8%) | 0.00600 | 0.00417 | 0.00332 | 1.25 | 1.23 | 1.23 | 1.18 |
| IQ3_XXS/r (19.8%) | 0.00565 | 0.00398 | 0.00329 | 1.21 | 1.20 | 1.23 | 1.11 |
| IQ2_S (9.1%) | 0.00751 | 0.00466 | 0.00399 | 1.17 | 1.17 | 1.18 | 1.09 |
| IQ2_XS (2.8%) | 0.00712 | 0.00477 | 0.00396 | 1.20 | 1.17 | 1.19 | 1.10 |
| Q2_K (2.7%) | 0.00666 | 0.00449 | 0.00351 | 1.28 | 1.29 | 1.32 | 1.24 |
| IQ2_XXS (1.4%) | 0.00648 | 0.00394 | 0.00319 | 1.24 | 1.25 | 1.23 | 1.18 |
| IQ1_M (0.3%) | 0.00647 | 0.00456 | 0.00385 | 1.19 | 1.17 | 1.19 | 1.09 |
| **weighted by share (87.9%)** | | | | **1.225** | **1.213** | **1.222** | **1.137** |

The first three numeric columns are the 512-row, 8-thread, 17408 x 5120 run.

Weighted totals for the other batch sizes (8 threads, 17408 x 5120): 64 rows 1.17x; 16 rows 1.07x (with the tiled
kernel; with today's rule of rows below `IQX_MIN` = 24, 0.96x); 8 rows 1.02x; 4 rows 1.07-1.20x; 1 row 1.05x. In cache
(1 thread, 1024 x 5120): 8 rows 1.10x, 1 row 1.02x. Below 24 rows the same configuration varies by 10-20% between runs
with the day server up. Passes of 4, 6 or 8 rows in the rows kernel give 0.93-1.2x with no consistent order, so verify
and batched decode should be counted as unchanged. At 16 rows the flat tiled kernel beats both rows kernels, so `IQX_MIN`
may move down to about 16.

What it means:

- **Prefill.** The i-quant GEMMs gain 1.21-1.23x for the file's mix, on both shapes and on one thread. The 27B's prefill is
  about 90% GEMM phases (PP-RESULTS.md), and 12% of the weights are K-quants (Q4_K, Q6_K) that this does not cover. The
  arithmetic gives about 1.17x end-to-end. The last kernel change delivered 1.32x where the micro-benchmark said 1.5x, so
  expect roughly **1.1-1.15x end-to-end prefill**, to be measured.
- **Decode, verify and small batches:** no reliable change.
- **Next (step 3):** the definition in `forward.c`'s matmul paths as an opt-in mode. That means: activations quantized with
  one scale per 256, plus the X pairs; `iqf_dot` as the generic definition; the rows and tiled kernels; NEON (`iqn_*`:
  untested here, no AArch64 machine); and the bitwise tests across paths, thread counts and batch sizes. Then step 4: on
  the 27B, prefill speed and KL against today's definition, with the day server stopped.

## 11. Implemented as `--act256` (2026-11-24)

`RqModelOpts.act256` (`gwenedeg --act256`, `gwenedeg serve --act256`, opt-in) implements the activation change, including
where section 10's benchmark could not check it:

- **Activation prep.** `forward.c quantize256`: quantize()'s operations with one scale per 256, the scale written to each
  of its groups of GS, so every kernel still reads per-group scales as before (only the Q2_K / i-quant kernels under the
  flag take it as the super-block's). 24 call sites: the decode row, its FFN and attention parts, the batched rows,
  batched prefill, the verify row, the hidden rows, an MTP view's input row and the trellis side buffers.
- **Kernels.** `iquants.h`: `iqf_dot` (the definition, and the generic and NEON builds' kernel), `iqf_rows` (the AVX-512
  rows kernel) and `iqf_tile` / `iqf_gemm_item` (the tiled GEMM on the existing scratch). Dispatch per matrix on
  `QuantizedTensor.act256`, so a model runs mixed: IQ3_S / IQ3_XXS / IQ4_XS / Q2_K / IQ2_* / IQ1_* flat, Q4_K / Q6_K /
  Q8_0 / NariQ per 32.
- **Where a part is not a whole super-block.** An attention group (kv_mul * hd) or a linear-attention head (dv) that is
  not a multiple of 256 (Qwen3.5's are 128, and most of Qwen3's attention groups) would otherwise quantize a partial
  super-block: those phases quantize their whole row once in the phase epilogue instead (a few % slower than a whole
  super-block there, not measured; Qwen3-0.6B's kv groups are 128 too). The FFN items are whole super-blocks
  (ff_rows = 256 under the flag).
- **Rejected inputs** (`RqModelOpts.act256 = 0` with a warning): a NariQ file (trellis matrices cannot take it) and a
  hidden size not a multiple of 256. Not implemented for the flag: Q4_K / Q6_K flat tiles (the K-quant GEMMs keep the
  per-32 folds; 12% of ISTA's 27B weights), and the `--repack-info` timings do not load with the flag set.
- **Decode (`quantize` on one row at a time)** is the same function with n = dim: whole rows, no partial super-blocks.

Quality and speed:

- `tests/kernels/flat_bench.c`: the prototype kernels of section 10 are gone; the production `iqf_*` ones are tested
  (143 cases) and give the same times. Weighted by ISTA's file's types: 1.20-1.23x at 512 rows, 1.13x at 24.
- Qwen3.5-0.8B, 1,500-token prompt, `-b 512`, 8 threads, day server up, best of 5 interleaved: IQ3_S 2377 -> 2707 tok/s
  (1.14x), IQ4_XS 2414 -> 2667 (1.11x). End-to-end on the 27B should be less (its non-i-quant phases do not change);
  not measured yet (the day server would have to be stopped).
- KL vs the default on the same files (`tools/v4h_kl.py`, gwenedeg itself): IQ3_S +0.0036 wiki / +0.0014 code, IQ4_XS
  +0.0028 / +0.0007, Q4_K +0.0021 / +0.0005, Q8_0 +0.0037 / +0.0010. About 2-7% of those files' total KL. Q8_0's
  +0.0037 is the activation change alone, on 8-bit weights.

The test suite (`tests/run_tests.sh`, TMPDIR on disk, the day server up): 1,902 passed, 3 failed; those three fail the
same way in an HEAD worktree (server admission order and the KV-disk watcher: environment-sensitive). The q35 group's
`act256` runs (synth-q35-iq with all nine i-quant types, synth-q35-ista-mtp, synth-q35-27b; batched, threads, AVX-512
vs generic, api_test) all pass with the same bitwise invariants as the default, at thresholds for their lower
cosine vs the numpy reference (TH_Q35_ACT256).

On the 27B (ISTA's `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`, `--repack`, `-k q4-dot`, 14 threads, the day server stopped,
2026-11-24; `scripts/bench_prefill.sh -P`'s prompts, best / median of 3 interleaved runs):

| | default | `--act256` | ratio |
|---|---:|---:|---:|
| prefill 1,500 tok | 106.1 / 103.1 tok/s | 123.4 / 121.9 tok/s | 1.16 / 1.18 |
| prefill 8,000 tok | 100.8 / 99.1 tok/s | 117.7 / 116.2 tok/s | 1.17 / 1.17 |
| decode 32 tok | 13.1 - 13.2 tok/s | 12.3 - 13.2 tok/s | 1.0 |

KL of `--act256` vs the default on the same file (`tools/v4h_kl.py`, `build/long_ctx -A`, 8,192 tokens each of wikitext
and llama.cpp source): 0.0030 (p99 0.030, top-1 97.2%, PPL x1.004) and 0.0014 (p99 0.024, top-1 99.4%, PPL x1.00) --
about the same as on the 0.8B files, and 1.5-2% of the file's KL against its higher-precision reference.

**Scope of "the default".** The production server config runs `--act256` since 2026-11-24 (`launch-server` ista-iq3s),
like `--repack` before it: a config-level choice. The library default stays per 32, and so do the test goldens: under
the flag **every** file's activations change (quantize_act covers all matmul activations, Q8_0 included), so flipping
the library default would invalidate every golden in the suite (greedy, logits, long-context, chat) and break the
project's rule that the default definition is bit-exact -- the reason PP-RESULTS.md rejected "fold once per 256" when
it was considered silently. `--no-act256` (the old per-32 definition) remains one flag away, and the suite tests both
definitions (the act256 runs of the q35 group). Making it the library default would be its own project: regenerating
all goldens, re-deriving the HF-comparison thresholds, and dropping the per-32 reference the docs describe.

Open:

- The attention / linear-attention epilogue quantizes: measure their cost on a real workload.
- A user-visible way to compare (e.g. the logit KL of `--act256` vs the default on one prompt).
