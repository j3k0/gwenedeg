# NariQ: trellis-coded Qwen3.8-27B (2026-11-24)

Gwenedeg loads and runs [nari-labs/Qwen3.8-27B-NariQ-GGUF](https://huggingface.co/nari-labs/Qwen3.8-27B-NariQ-GGUF)'s
3.0 bpw file (`scripts/download_models.py nariq-27b`, `./launch-server nariq`): Qwen3.8-27B quantized with QTIP's
bitshift trellis and exllamav3's integer codebook, calibrated on coding-agent traces (tech report
[arXiv 2611.26157](https://arxiv.org/abs/2611.26157)). Nari's own runtime is a CUDA-first llama.cpp fork
([nari-labs/nari-llama.cpp](https://github.com/nari-labs/nari-llama.cpp), tag `b12431-nariv0`), whose CPU path is a
reference-grade fallback (decode to fp32, then an fp32 GEMV: ~0.25 ns per weight on this Zen 4, ~2 tok/s for the 27B).

```sh
scripts/download_models.py nariq-27b       # 10.1 GB, pinned revision, sha256-checked
gwenedeg models/Qwen3.8-27B-NariQ-GGUF/Qwen3.8-27B-NariQ-3.0bpw.gguf -r 1
gwenedeg models/Qwen3.8-27B-NariQ-GGUF/Qwen3.8-27B-NariQ-3.0bpw.gguf -D models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf
```

## The file

GGUF types `TRL1` .. `TRL6` (fork-only ids 44 .. 49, meaningful only with `trellis.version = 1`): 16 x 16 tiles of
`bits` 32-bit words, output-tile-major. A coded linear is
`y = scatter(post * H_rows(V (s16 * H_cols(pre * gather(x)))))`, V the integer grid (each value the sum of the four
6-bit fields of `state * 0x83DCD12D` minus 126, state a 16-bit window of the tile's circular bit stream), `s16` one
scale per 16 inputs shared by every row, `H` normalized block Sylvester Hadamards; [src/trellis.h](../src/trellis.h)
has the exact definition. Side tensors per matrix: `trellis_s16` (F16), `trellis_pre` / `trellis_post` (F32),
optional `trellis_pin` / `trellis_pout` (I32 gathers); `trellis.desc.<name>` the transforms.

What the 3.0 bpw file holds (from its header): every matrix coded, 3.008 bpw over all its bytes (codes 2.99),
39 MB of side vectors, the norms F32; widths by parameters TRL2 24%, TRL3 50%, TRL4 23%, TRL5 2.5%, TRL6 0.3%
(`ssm_alpha` / `ssm_beta` TRL5, `attn_k` / `attn_v` mostly 5, `ssm_out` / `attn_output` 3.6-3.7 on average, FFN ~3);
a 2-bit `token_embd`; the classifier in 16 frequency bands (`output.bandNN.weight`, 13 of them 2-bit) and
`output.vocab_gather`. Transforms: input side blocks of 128 (`ffn_down` 1024, `ssm_out` / `attn_output` 2048),
output side 128 .. 4096 or none (alpha / beta); input gathers on `ssm_out` / `attn_output`, output gathers on the input
projections. No MTP block.

Taken: those layouts. Refused at load: whole-slice (Paley) Hadamards, several fused parts, a banded embedding (the
2.0 bpw file's), files mixing coded and uncoded matrices, coded files with an MTP block. Untrusted input: shapes,
types, descriptors checked from the directory; the side vectors checked finite and the gathers as permutations
(model.c `trl_load_sides`, after the memory check).

## Arithmetic and kernels

Activations: per matrix, its own input transform (gather, `pre`, the Hadamard, `s16`) of the fp32 row, then
gwenedeg's int8 quantization (groups of 32). The GEMV per output row: per group the exact int32 sum of `v * q`, then
`y = y + (float) I * s_group` in group order. Every build computes these bits: the generic kernel (`trl_value` per
weight) and the AVX-512 ones:

- one row (decode): per tile the code words in two (5-6 bits: three) registers; for each quad of columns
  (`2g, 2g+1, 2g+8, 2g+9` of a row: consecutive ring positions, whose windows lie in one 32-bit window for bits <= 5)
  two `vpermt2d` + `vpshldvd`, then per column a shift and mask, `vpmulld`, shift, mask: the four u8 fields of
  `state * mult` (lane = output row), fed to `vpdpbusd` against the activation byte replicated in a dword (the lane sums
  `(v + 126) q`; the bias removed with the group's activation sum). Specialized per bit width (the column loop unrolled
  with the tables' offsets known: ~25% faster than one function for every width).
- several rows (verify segments, prefill): each 16-row block decoded once into a per-thread scratch as u8 `v + 126`,
  four consecutive columns per dword lane (the fields summed by a `vpdpbusd` against ones), then an int8 GEMM over the
  rows (64 products per `vpdpbusd`): 1.5x the one-row kernel per row at 8 rows, 4x at 256.

Single thread, ns per weight (Zen 4; 17408 x 5120): fork-style decode-to-fp32 + fp32 GEMV 0.25; the field kernel
without quads 0.055; with quads TRL2 / 3 / 4 / 5 / 6: 0.044 / 0.045 / 0.046 / 0.056 / 0.059; in the model (codes
streamed from DRAM, 4 KB pages) TRL3 0.052. GEMM 0.0039 ns per multiply-add at 256 rows. gwenedeg's IQ3_S kernel: 0.037
([iquants.md](iquants.md)). The decode is compute-bound on this CPU: the codes cost more to decode per weight than
IQ3_S's, which the smaller file (10.1 vs 12.1 GB) does not make up.

## Passes

The forward passes run a coded matrix whole, never by row ranges (its outputs mix through the output Hadamard and the
scatter): before each phase that reads matrix outputs (the input projections, `wo` / `ssm_out`, `w1` + `w3`, `w2`, the
classifier) a coded pass of three phases (forward.c `trl_pass`): A the activation side of every row (items of whole
Hadamard blocks), B the GEMV / GEMM of 16-row blocks (per-thread ranges: contiguous code streams), C the output side
(Hadamard, `post`, scatter) into the buffers the phase reads; `matvec_rows` and the GEMMs skip coded matrices. Decode
(`decode_worker`), the batched prefill (`prefill_worker`) and batched decode / verify (`batch_decode_worker`) all do
this; the classifier's 16 bands scatter straight into the logits (`vocab_gather` inverted at load), 8 rows per pass.
The embedding row is decoded from the mapping at lookup (`trl_embed_row`).

## Verification

- Against Nari's runtime: the fork built CPU-only, its logits for the same token ids
  (`/tmp/nari-review/scripts/oracle_logits.c`, a 234-token chat with code and French): KL(fork || gwenedeg) mean
  7.2e-5, median 5e-6, max 2.0e-3, top-1 99.6% (one near-tie). The fork's own CUDA gate vs its F32 reference is
  mean KL <= 1e-3; its prefill vs decode differ by 1.7e-4. The difference here is gwenedeg's int8 activations.
- Bitwise: the decode path == the batched prefill at `-b` 512 / 37 / 8 / 2; the generic build == the AVX-512 one
  (34 tokens, batched); every kernel change since the first run reproduced its logits bit for bit; speculative
  decoding with the 0.8B draft at `-G` 3 / 6 == plain decoding; `gwenedeg serve` with 4 parallel requests.
- Not in the suite yet: a synthetic NariQ model (tests/make_synth_q35.py's shapes with coded matrices, a numpy
  reference of the format) for check.py's bitwise groups.

## Measured (2026-11-24, 16-vCPU Zen 4, the day server stopped, other users' load 4-13)

| | NariQ 3.0 bpw | ISTA IQ3_S-mtp (--repack) |
|---|---|---|
| weights in RAM | 9.4 GiB | 12.2 GiB |
| decode, plain, 14 threads | 6.6-6.9 tok/s | 10.8-11.2 tok/s |
| decode, 0.8B draft -G 3 | 8.6-8.7 tok/s (2.37 tok/pass) | (its MTP block: 14-15 in the server) |
| prefill, 1,959 tokens | 81.7 tok/s | 102.6 tok/s |

Quality (Nari's numbers): agent KL from BF16 0.0063 at 3.0 bpw, against 0.0239 for the best published file at or
below 3.24 bpw; GSQ-RCO IQ3_S (3.50 bpw), the file of the day server, is not in their table. Measured here against it
(2026-11-24): KL from Unsloth's Q8_0 (as a stand-in for BF16, run by the fork's llama.cpp CPU build from the mapping:
29 GB on this 30 GB machine) on four texts written by hand (not by the model: higher KLs than Nari's, which counts
model-written tokens only), both quants run by gwenedeg:

| text (positions) | NariQ KL mean / median | IQ3_S KL mean / median | top-1 vs Q8_0 |
|---|---|---|---|
| coding-agent transcript, tool calls (857) | **0.127** / **0.0009** | 0.182 / 0.0010 | 91.4 / 88.7% |
| C source, src/threads.c (796) | **0.023** / **0.009** | 0.032 / 0.010 | 95.2 / 93.7% |
| English technical prose, docs/iquants.md (1148) | 0.074 / **0.038** | **0.067** / 0.040 | 86.3 / 86.8% |
| French + Breton chat (537) | 0.165 / 0.042 | **0.085** / **0.018** | 83.1 / 91.1% |

(Agent row: the 73 positions where the two engines disagree on the same weights by more than KL 0.05 left out: a few
positions are very sensitive to rounding, and there every run differs, even llama.cpp's Q8_0 from its own IQ3_S, by
KL up to 20; with them the means are 0.29 / 0.36. The ranking is the same with both quants run by llama.cpp.) NariQ is
closer to the reference on code and agent traces (its calibration data), even with English prose, and twice as far on
French: its classifier is 13 of 16 bands at 2 bits and its embedding 2-bit (Nari's 2.5 bpw file loses 2.2 points on
MGSM, a significant drop, while their report calls it level). gwenedeg vs llama.cpp on the IQ3_S file: median KL 1e-4,
mean 0.048 (those sensitive positions; gwenedeg's perplexity on the transcript 3.96, llama.cpp's 4.02): worth a look.

Second round (2026-11-24, the same reference and engines): 16 code excerpts of ~2,800 characters (~650-1,100 tokens:
C, TypeScript, Rust, Python, JavaScript, PHP, Go, Java, shell; local repositories and vendored open-source files) and
model-written answers (an LRU cache in Python, a C off-by-one bug, two tool-call turns; written greedily by each quant,
both scored on all), per-position KL from Q8_0, the ratio NariQ / IQ3_S of the mean KL per text with a 95% moving-block
bootstrap interval (blocks of 32 positions), pooled as the geometric mean of the per-text ratios with a bootstrap over
texts:

| set | NariQ / IQ3_S mean KL | texts NariQ lower | per-text CI excludes 1 |
|---|---|---|---|
| code, both quants run by gwenedeg | 0.94 [0.85, 1.04] | 10 of 16 | better 6, worse 1 |
| code, both run by llama.cpp | 0.90 [0.82, 1.00] | 11 of 16 | |
| code, engine-sensitive positions left out | 0.94 [0.85, 1.05] | 10 of 16 | |
| model-written answers (4 of ~450 tokens) | 0.79-0.90 each, ~0.85 pooled | 4 of 4 | none alone |

Per code file the ratio runs from 0.71 (src/threads.c, the first round's only code text: its -30% was the favorable
end) to 1.37 (a PHP file); medians per file are within ~20% either way. So on code NariQ 3.0 is about level with the
IQ3_S file (6-10% lower KL on average, not significant over 16 files), on model-written coding answers ~15% closer
(consistent but small samples), on the hand-written agent transcript of the first round 30% closer (one text), and on
French twice as far. By the rule of thumb of 2-4x KL per bit per weight, level-to-15% is 0-0.25 bit: NariQ matches
the 3.44 bpw file with 0.45 bpw less, not a quality step up.

## Open

- Decode at full threads: ~20% of the GEMV phase waits at its end (load from other processes, 16 vs 14 threads);
  three phases per coded pass (~770 per token): the output side could move into the phases that read it.
- MTP: the file has none; ISTA's MTP block (blk.64 of the IQ3_S-mtp file) as a drafter fed NariQ's hidden states is
  untried (an MTP view with a coded target needs the banded classifier subset).
- The 2.5 / 2.0 bpw files: 2.5 should load (same layouts, untested); 2.0 has a banded embedding (refused).
