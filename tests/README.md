# Tests

`make test` (or `tests/run_tests.sh`) builds `./runq` and `./runq-generic` if needed (in parallel)
and runs everything in about 21-25 s on the 16-vCPU dev box (~37 runq processes at ~0.45 s startup
each; the box is shared, so the time varies). It needs `models/Qwen3-0.6B-q8.bin`, plus numpy (`.venv/bin/python`
if present, else the system `python3`). Torch is only needed to regenerate the HF goldens.

```sh
tests/run_tests.sh                    # all tests, OMP_NUM_THREADS=8 unless set
tests/run_tests.sh -v                 # + per-position logits metrics and a summary table
tests/run_tests.sh --only logits      # subset: tokenizer,logits,batched,threads,kernels,greedy,chat
RUNQ=/path/to/other-binary tests/run_tests.sh   # test another build (no make; kernels test skipped)
RUNQ=./runq RUNQ_GENERIC=./runq-generic tests/run_tests.sh   # ... with the kernels test
tests/run_tests.sh --update-greedy    # regenerate the greedy goldens (incl. the chat transcript)
.venv/bin/python tests/make_golden.py # regenerate the HF goldens (tokenizer + logits, ~15 s)
.venv/bin/python tests/make_golden.py --tokenizer-only   # only golden/tokenizer.json (~2 s)
.venv/bin/python tests/fuzz_tokenizer.py                 # on-demand tokenizer fuzz vs HF (~6 s)
```

Exit status is non-zero if any test fails. XFAIL = known engine bug, reported but not failing;
XPASS = an XFAIL case now passes (remove its mark in `make_golden.py`).

## Engine hooks (`src/runq.c`)

- `-m tokenize -i TEXT`: token ids of the raw text (no chat template), space-separated on stdout.
  Only the checkpoint header and the tokenizer are loaded (no weights). stderr gets
  `[tokenize] B bytes -> N tokens, encode X ms, tokenizer load Y ms` (encode time only).
- `-m logits -i TEXT -o FILE [-b N]`: writes `int32 n, int32 vocab, int32 ids[n],
  float32 logits[n][vocab]` (row i = logits after token i). `-b 1` feeds the tokens one at a
  time through `forward()` (`forward_sequential`); `-b N > 1` (default 512) uses the batched
  prefill, `forward_batch()` in chunks of N tokens with the logits of every row
  (`forward_chunked`). The tests always pass `-b` explicitly.
- `RUNQ_CHAT_DUMP=PREFIX` (chat mode): at exit, writes `PREFIX.ids` (the token ids in the KV
  cache, i.e. the conversation as the model saw it, space-separated) and `PREFIX.logits`
  (`int32 n, int32 vocab, int32 pos[n], float32 logits[n][vocab]`: every logits vector chat
  sampled from, `pos` = position of the token that produced it; records of positions that a
  re-prefill overwrote are dropped, so they always match the final cache).
## What is checked

| test | what | golden |
|---|---|---|
| `tokenizer/*` | runq ids == HF `encode(text, add_special_tokens=False)` for 69 strings: ASCII, whitespace runs of 1-8 chars before words/punctuation/digits/newlines, trailing whitespace, `\r\n`/`\r`, tab/space indentation, digits (long numbers, mixed with letters, non-ASCII digits), punctuation runs, contractions (any case, `'ſ`), C/Python code, NBSP, ideographic and other Unicode spaces, U+001C/zero-width (not `\s`), control chars, emoji (ZWJ, flags, skin tones, keycaps), rare CJK (Ext. B-I), Arabic, Hindi, Korean, Russian, Greek etc., math, two 3,000-char strings without spaces, added tokens (adjacent to text, back to back, partial ones), 1-char strings. 2 XFAIL: NFC normalization (not implemented) | `golden/tokenizer.json` |
| `tokenizer/speed_*` | encode time inside runq (stderr line) < 100 ms for ~2,000 tokens of the golden texts and for one 30,000-byte chunk (no spaces); measured ~0.3-1.3 ms and ~5-13 ms | |
| `logits/<prompt>` | runq Q8_0 logits vs HF fp32 for 5 prompts (en, code, zh, chat template, counting), 15-24 tokens; ids must match HF first | `golden/logits/*.npz` |
| `logits/cos_pooled`, `logits/top1_pooled` | the same metrics pooled over all prompts | |
| `logits_batched/*` | the same test (same thresholds) with the batched prefill, `-b 64` (one chunk here, the same code as the default `-b 512`) | `golden/logits/*.npz` |
| `batched/<prompt>_b7`, `_b64` | batched logits vs sequential (`-b 1`) on every logits prompt; `-b 7` gives 3-4 chunks with boundaries mid-prompt. `all_b64`: all 5 prompts concatenated (86 tokens), 2 chunks at `-b 64`; `long_b64`: `all` twice + `en` (187 tokens, 12 KV tiles, 3 chunks: attention over several K/V tiles and causal row blocks in later chunks). Max abs diff <= 1e-3 and cos >= 0.99999 per row (full vocab), finite; reports "bit-identical" (currently all 12) | sequential run |
| `threads/bitwise`, `threads/bitwise_b7` | logits file for `en` is byte-identical with OMP_NUM_THREADS=8 and 1, sequential and `-b 7` | |
| `threads/bitwise_long_b64` | `long` at `-b 64`: OMP_NUM_THREADS=8 vs 3 byte-identical (other partitions of the dynamically scheduled attention work items) | |
| `kernels/en_b1`, `en_b7`, `all_b64`, `long_b64` | logits file byte-identical for `./runq` (AVX-512: VNNI GEMM/GEMV and the AVX-512 attention) and `./runq-generic` (`-DRUNQ_GENERIC`, generic C kernels and C attention, same flags otherwise): GEMV incl. classifier and decode attention (`-b 1`), GEMM with 6 + 1 row tiles and a 1-row chunk (`-b 7`), 2 chunks with 6 + 4 row tiles and the all-rows classifier GEMM (`all`, `-b 64`), 3 chunks with attention over 12 tiles (`long`) | `./runq-generic` |
| `greedy/*` | exact stdout of `-t 0 -c 4096` runs: chat, chat with thinking (`-r 1`), chat with system prompt (`-y`), generate; 40-48 tokens. These use the default batched prefill; `generate@b5` = generate in 3 chunks (same golden) | `golden/greedy/*.txt` (our engine) |
| `greedy/seeded_chunks` | seeded sampling (`-t 0.8 -s 42`, chat): same output with `-b 1` and `-b 5` (chat advances the RNG once per prompt token, as before batched prefill) | |
| `chat/multiturn_golden` | exact stdout of a 3-turn chat piped on stdin (`-t 0 -n 32`) | `golden/greedy/chat_multiturn.txt` |
| `chat/multiturn_memory` | the turn-2 reply to "What is my name and favourite colour?" mentions `Ada` and `green` (told in turn 1) | |
| `chat/multiturn`, `chat/thinking`, `chat/context_full` | multi-turn runs with `RUNQ_CHAT_DUMP`: (1) the decoded KV cache equals the expected chat-template rendering, built in Python from the user lines and the replies on stdout (ChatML, `<|im_end|>\n` after every message, the empty think block before each answer with `-r 0`; with `-r 1` earlier answers without their reasoning, as HF's template renders them); (2) re-encoding that text gives the same ids (piecewise encoding == whole-string encoding); (3) every sampled logits vector is **bit-identical** to the same row of a fresh `-m logits` run on the whole conversation (incremental prefill at `pos0 > 0` and decode vs one batched prefill from 0). `thinking`: `-r 1`, 2 turns, turn 1's reasoning dropped (re-prefill of its answer). `context_full`: `-c 160 -n 30 -y ...`, 6 turns: the oldest turns are dropped (stderr `[context] full`), the system prompt kept | |

Logits metrics, per position, over the union of a fixed 4096-id vocab subset and HF's top-64 ids:
cosine similarity, top-1 agreement (runq argmax vs HF top-1), top-5 overlap. Position 0 has its
own loose threshold: the first token carries a ~6500 "massive activation" in the residual stream
(also in HF fp32), which Q8_0 handles poorly, so its logits are much noisier (cos 0.83..0.91).

The logits test against HF is the correctness gate. Greedy is only a cheap tripwire: any change
in rounding (e.g. a new kernel summing in a different order) can legitimately change the text. On a
`REGRESSION (greedy)` with all `logits/*` passing, check the outputs still read fine and run
`--update-greedy`.

## Baseline and thresholds

Measured 2026-10-24 (Q8_0 engine at the commit adding this suite, unchanged from upstream numerics):

| prompt | n | cos pos 0 | cos mean (pos >= 1) | cos min (pos >= 1) | top-1 | top-5 |
|---|---|---|---|---|---|---|
| en | 15 | 0.8316 | 0.99833 | 0.99607 | 13/15 | 0.893 |
| code | 24 | 0.8323 | 0.99920 | 0.99838 | 21/24 | 0.942 |
| zh | 15 | 0.9064 | 0.99958 | 0.99930 | 14/15 | 0.880 |
| chat | 15 | 0.8967 | 0.99888 | 0.99663 | 14/15 | 0.920 |
| count | 15 | 0.8653 | 0.99941 | 0.99901 | 14/15 | 0.907 |
| pooled | 84 | | 0.99909 | | 76/84 = 0.905 | |

Since the SwiGLU NaN fix (2026-10-25, see `../TODO.md` Known bugs; it changed the last layer of
`zh` and `chat`): zh top-5 0.893, chat cos mean 0.99891, pooled cos mean 0.99910; the rest is
unchanged. The batched path gives exactly the same numbers.
Since the Step 3b attention (pooled numbers unchanged, per-prompt values moved slightly, e.g. en
top-5 0.907, code top-1 22/24): current per-prompt table in `../BENCHMARKS.md` (Quality).

Thresholds (`tests/check.py`): per prompt cos mean >= 0.995, every position >= 1 cos >= 0.98,
pos 0 cos >= 0.75, top-1 >= 0.75, top-5 >= 0.75; pooled cos mean >= 0.997, pooled top-1 >= 0.85.
Logits must be finite. That is about 3x the baseline error (1 - cos) for means, 5x per position.

Sanity check by deliberate breakage (scratch builds, not committed), all caught:

| breakage | pooled cos (pos >= 1) | per-prompt cos mean | pooled top-1 |
|---|---|---|---|
| RoPE theta 1e4 instead of 1e6 (subtlest) | 0.9876 | 0.977 .. 0.995 (4 of 5 fail) | 0.88 |
| RoPE pairs interleaved instead of split halves | | 0.80 .. 0.91 | 0.57 |
| matmul skips its last 64-group | | 0.81 .. 0.96 | 0.76 |
| no QK-norm / KV written at pos+1 / V read from t-1 / no final norm / attention scale 1/d / rmsnorm eps 1e-2 | | < 0.71 | < 0.16 |
| activation quantizer truncates instead of rounding (precision loss only) | 0.99713 | en 0.99432 fails, others pass | 0.94 |

Thread invariance: matmul/matmul_batch (generic and VNNI) parallelize over output rows and attention over
(kv head, block of query rows) items; each query vector's result does not depend on the other
vectors in its item, so results are bit-identical for any thread count and any grouping (tested 1
vs 8, 3 vs 8; also checked 13).

Batched vs sequential: `forward_batch()` gives bit-identical logits to `forward()` because both
call the same numeric kernels (`dot_q8`, `qk_norm_rope`, `attention_kv`, `swiglu`, `rmsnorm`,
`quantize`) per row, and the 4-row GEMM kernel `dot_q8_x4` is compiled without FP contraction
(`NO_FP_CONTRACT`, gcc only) so it rounds like `dot_q8`. With contraction (an FMA in the scale
accumulation) the logits differ slightly and `greedy/chat_system` flips a token. A future kernel
that sums in another order may legitimately need the 1e-3 tolerance (or a looser one, justified);
real bugs are far off. Deliberate breakage (scratch builds, all caught):

| breakage in forward_batch | max abs diff | min cos | caught by |
|---|---|---|---|
| causal mask leaks one position (row b sees b+1) | 22.4 .. 32.1 | <= 0.41 | every `batched/*` |
| K RoPE at the chunk-local position (b, not pos0+b) | 17.3 / 32.1 | 0.47 / -0.42 | `_b7`, `all_b64` |
| chunk K written one position early (pos0-1) | 13 .. 33.7 | <= 0.68 | `_b7`, `all_b64` (single-chunk `_b64` pass) |

VNNI vs generic kernels (`kernels/*`): the AVX-512 VNNI GEMM and GEMV (`src/kernels_avx512.h`)
keep the per-group int32 sums exact and the float fold of `dot_q8` (`val + (ws * xs) * (float)
isum`, groups in order, each operation rounded, via explicit-rounding intrinsics), so they are
bit-identical to the generic kernels, and the logits are byte-identical to the binary before
them. The `batched/*` tests cannot see a kernel rounding change (both paths use the same kernel);
`kernels/*` can: an FMA in the VNNI fold gives max |diff| 2.1 on all three cases (scratch build).
Also checked once (scratch builds, not in `make test`): tile variants (`-DQ8P_RB=1`, the odd
block-count tail that Qwen3-0.6B never hits; `-DQ8P_C=1/3/8`, `-DQ8P_NB=5`) byte-identical, and an
ASan+UBSan build (`-O1 -g -fsanitize=address,undefined`, -b 1/7/13/64): no reports.

Multi-turn chat (`chat/*`): the conversation in the KV cache was also checked once against HF's
`apply_chat_template` (transformers 4.53.0): identical text with `-r 1`; with `-r 0` the only
difference is the empty `<think>\n\n</think>\n\n` that runq keeps before each earlier answer (see
`../README.md`). Deliberate breakage (scratch builds), caught: a follow-up prefill one position too
far (`pos0 + 1`): max |diff| 6-25 on all three cases, and the turn-2 reply forgets the name; the
`\n` after `<|im_end|>` dropped from the history: the template-rendering check on all three.

Builds without AVX-512 (e.g. `-march=x86-64-v3`) use the generic kernels and pass the logits,
batched and thread tests, but not the greedy goldens: the rest of the float code (rmsnorm,
softmax, attention) is vectorized at a different width and rounds differently. The greedy
goldens belong to AVX-512 (`-march=native` here) builds.

Attention (Step 3b): the attention is defined once (arithmetic and tile layout in the
`src/kernels_avx512.h` header comment): scores as fma chains over chunks of 32 dims, a clamped
Cephes-style `exp`, 16 partial sums with a fixed reduction tree, one division, V accumulated by
an fma chain in position order. Decode (`forward`, 2 query vectors per kv head) and prefill
(`forward_batch`, 2 x 6 vectors per work item, causal per row) run the same per-vector
operations, so `batched/*` stay bit-identical, and the generic C version (`attention_vec`,
compiled with FP contraction and reassociation off) does the same, so `kernels/*` stay
bit-identical (a scratch build with the C reduction tree replaced by a linear sum: max |diff| 1.3-1.5,
caught by all `kernels/*`). The change of arithmetic from the old attention (expf, one dot product
per position) changed the logits slightly: against a double-precision attention reference
(scratch build) the new logits are closer (1-cos 4.7e-4 vs 5.3e-4 on a 288-token prompt) and the
HF thresholds are unchanged and met (pooled cos 0.99910, top-1 76/84 as before), but two greedy
goldens flip a token (`chat_system`, `generate`: both texts read fine) and were regenerated
(`--update-greedy`, both with the same new text on the generic build).

## Golden data

`make_golden.py` documents the formats. HF reference: `Qwen3ForCausalLM` from
`models/Qwen3-0.6B` in float32 on CPU (eager attention), one forward pass per prompt,
transformers 4.53.0 / torch 2.7.0; regenerating gives byte-identical files. Total size ~0.75 MB.
The logits prompts avoid multi-space indentation (written when runq still had no pre-tokenizer;
kept so the logits goldens stay unchanged).

Known engine bugs found by these tests are listed in `../TODO.md` ("Known bugs").

## Tokenizer fuzz test (on demand, not in `make test`)

`.venv/bin/python tests/fuzz_tokenizer.py [-n 2000] [--seed 1] [--jobs 8] [--show 5] [--no-files]`
compares `runq -m tokenize` with the HF tokenizer on `-n` random strings (a mix of the character
classes of the goldens, added tokens and broken ones, and random code points from all planes)
plus real text (repo docs/sources, the HF model README and LICENSE, split at 64 KiB for argv).
Strings are NFC-normalized first (runq has no normalizer) and contain no NUL. It prints the
mismatch rate, minimizes the first `--show` mismatches (delta debugging) and exits 1 on any.
Result 2026-10-25: seed 1, 2,013 cases (232k HF tokens), and seed 2, 5,013 cases (492k tokens):
0 mismatches. Sanity check: a scratch build with `\s+(?!\S)` disabled gives 27% mismatches,
minimized to `'  k'` and `'\t\t4'`.
