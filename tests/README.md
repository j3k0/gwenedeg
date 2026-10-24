# Tests

`make test` (or `tests/run_tests.sh`) builds `./runq` if needed and runs everything in about
15 s on the 16-vCPU dev box. It needs `models/Qwen3-0.6B-q8.bin`, plus numpy (`.venv/bin/python`
if present, else the system `python3`). Torch is only needed to regenerate the HF goldens.

```sh
tests/run_tests.sh                    # all tests, OMP_NUM_THREADS=8 unless set
tests/run_tests.sh -v                 # + per-position logits metrics and a summary table
tests/run_tests.sh --only logits      # subset: tokenizer,logits,threads,greedy
RUNQ=/path/to/other-binary tests/run_tests.sh   # test another build (no make)
tests/run_tests.sh --update-greedy    # regenerate the greedy goldens
.venv/bin/python tests/make_golden.py # regenerate the HF goldens (tokenizer + logits, ~15 s)
```

Exit status is non-zero if any test fails. XFAIL = known engine bug, reported but not failing;
XPASS = an XFAIL case now passes (remove its mark in `make_golden.py`).

## Engine hooks (`src/runq.c`)

- `-m tokenize -i TEXT`: token ids of the raw text (no chat template), space-separated.
- `-m logits -i TEXT -o FILE`: feeds the tokens one at a time (`forward_sequential`) and writes
  `int32 n, int32 vocab, int32 ids[n], float32 logits[n][vocab]` (row i = logits after token i).
  `dump_logits()` is where a batched-prefill path will plug in for the same comparison.

## What is checked

| test | what | golden |
|---|---|---|
| `tokenizer/*` | runq ids == HF `encode(text, add_special_tokens=False)` for 27 strings: ASCII, whitespace runs, tabs/newlines, digits, punctuation, contractions, C/Python code, French/German/Chinese/Japanese, emoji, a long word, URL, special tokens (`<\|im_start\|>`, `<think>`), stray `<`/`>`, 1-char strings | `golden/tokenizer.json` |
| `logits/<prompt>` | runq Q8_0 logits vs HF fp32 for 5 prompts (en, code, zh, chat template, counting), 15-24 tokens; ids must match HF first | `golden/logits/*.npz` |
| `logits/cos_pooled`, `logits/top1_pooled` | the same metrics pooled over all prompts | |
| `threads/bitwise` | logits file for `en` is byte-identical with OMP_NUM_THREADS=8 and 1 | |
| `greedy/*` | exact stdout of `-t 0 -c 4096` runs: chat, chat with thinking (`-r 1`), chat with system prompt (`-y`), generate; 40-48 tokens | `golden/greedy/*.txt` (our engine) |

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

Thread invariance: matmul parallelizes over output rows and attention over heads, so results are
bit-identical for any thread count (verified 1 vs 8).

## Golden data

`make_golden.py` documents the formats. HF reference: `Qwen3ForCausalLM` from
`models/Qwen3-0.6B` in float32 on CPU (eager attention), one forward pass per prompt,
transformers 4.53.0 / torch 2.7.0; regenerating gives byte-identical files. Total size ~0.7 MB.
The logits prompts avoid multi-space indentation because of the tokenizer bug below.

Known engine bugs found by these tests are listed in `../TODO.md` ("Known bugs").
