# V4H: a VNNI-native 4-bit weight and activation format

Status: **design proposal, not implemented or benchmarked**. V4H is a new quantization, not a lossless repack of an existing
GGUF type. The short, unscheduled version is in [IDEAS.md](../IDEAS.md#v4h-a-vnni-native-4-bit-weight-and-activation-format).
The starting points are the Q8P layout in `src/kernels_avx512.h` and the i-quant layouts and kernels in `src/iquants.h`;
current measurements are in [iquants.md](iquants.md).

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
