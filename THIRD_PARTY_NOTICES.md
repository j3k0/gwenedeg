# Third-party notices

Gwenedeg itself is under the MIT license in [LICENSE](LICENSE). This file lists the third-party code,
data and formats in or behind this repository, what they cover here, and their licenses. No model
weights are included.

## qwen3.c (and llama2.c)

- Source: https://github.com/adriancable/qwen3.c @ `55643ff` (see `third_party/qwen3.c/UPSTREAM.md`).
- Used for: the engine in `src/` started as qwen3.c's `runq.c` (since then rewritten and split into
  modules), and `src/win.c` / `src/win.h` come from it. `third_party/qwen3.c/` is an unmodified copy
  (apart from `UPSTREAM.md`), kept for diffs and for its `.bin` exporter (`export.py`, `model.py`).
- License: MIT, Copyright (c) 2025 Adrian Cable. The full text is in
  [third_party/qwen3.c/LICENSE](third_party/qwen3.c/LICENSE), and the notice is repeated in
  [LICENSE](LICENSE). qwen3.c started from Andrej Karpathy's llama2.c (https://github.com/karpathy/llama2.c,
  MIT).

## ggml / llama.cpp (formats)

- Source: https://github.com/ggml-org/llama.cpp (ggml).
- Used for: the GGUF file format (`src/gguf.c`, `tests/ggufmini.py`), the ggml tensor type numbers and
  llama.cpp's tensor and metadata names, and the Q8_0 / Q4_K / Q6_K block formats with their
  dequantization expressions (`src/kquants.h`: `get_scale_min_k4`'s unpacking of the 6-bit scales,
  `dequantize_row_q4_K` / `_q6_K`). These are reimplemented to be bit-compatible; no ggml source
  file is included.
- License: MIT.

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Unicode Character Database

- Used for: `src/unicode_tables.h` and `src/unicode_nfc.h`, which `tools/gen_unicode_tables.py`
  generates from Unicode 15.1.0 data (through Python's `unicodedata`, plus the White_Space list of
  `PropList.txt`): general categories, canonical combining classes, decompositions and compositions.
- License: Unicode License v3, full text in [licenses/Unicode-3.0.txt](licenses/Unicode-3.0.txt).

## Qwen3 and Qwen3.5 (models, tokenizers, chat templates)

- Source: the Qwen team's models on Hugging Face (`Qwen/Qwen3-*`, `Qwen/Qwen3.5-0.8B`; the GGUF files are
  unsloth's conversions of them). `scripts/download_models.py` downloads them; none are included.
- Used for: `tests/golden/templates/*.jinja` (Qwen3's and Qwen3.5's chat templates, verbatim); the chat templates
  built into `src/model.c`; and the test references in `tests/golden/` (token ids, logits and texts computed
  with the Qwen3 / Qwen3.5 models and tokenizers, and chat renderings).
- License: Apache License 2.0, full text in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).

## Hugging Face transformers / tokenizers

Used only to generate and check test data (`tests/make_*.py`, `tests/fuzz_tokenizer.py`, in a local
venv); no code of theirs is included. The tokenizer in `src/tokenizer.c` is written to give the same ids.
Both are Apache-2.0.

## LiteLLM

- Source: https://github.com/BerriAI/litellm, version 1.105.0.
- Used for: the complexity router's classifier rubric and response format, copied verbatim into
  `tests/server_test.py` (`RUBRIC`, `TIER_RF`; `scripts/bench_serve.py` imports them). The LiteLLM
  scripts (`scripts/litellm_e2e.sh`, `scripts/eval_router.py`) import it from a user-provided install.
  `scripts/router_eval.jsonl` (the labelled prompts) was written for this project.
- License: MIT (code outside LiteLLM's `enterprise/` directory, which this does not use).

```
MIT License

Copyright (c) 2023 Berri AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## The Voyage of the Beagle

`bench/ppl_beagle_ch1.txt`, the perplexity text, is the start of Chapter I of Charles Darwin's *The
Voyage of the Beagle*. It is public domain, taken from Project Gutenberg eBook #944 with the Project
Gutenberg header and license removed (BENCHMARKS.md, Quality).

## mynah-slm

https://github.com/mynah-org/mynah-slm is not included. `scripts/fetch_mynah.sh` clones and builds it
(into the git-ignored `third_party/mynah-slm/`), only to compare benchmarks.
