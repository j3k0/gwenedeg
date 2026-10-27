# Gwenedeg tests

`make test` (or `tests/run_tests.sh`) builds `./runq`, `./runq-generic`, `build/api_test` and
`build/api_test-generic` if needed (in parallel) and runs everything in about 28-30 s on the 16-vCPU dev box,
a shared AMD EPYC Genoa VM (481 checks; 644 with the Qwen3.5-0.8B file of the `q35` group (Step 14: incl. 31 checks of
two synthetic Qwen3.5 models cut out of it), 704 with the Unsloth Q4_K_M file next to it (~80 s at load 10-18; incl.
18 checks of the 27B-shaped synthetic models cut out of it, TODO P6: within the noise, 97 s vs 144 s for the 686 checks
before them, load 27-35); TODO P5: 717 with its generation step (89 s, load ~12-20), 733 with top-k vs numpy and the
sampling defaults (the top-k replay: ~13 s of one background worker); the box
is shared, so the time varies: Step 11's 475 checks 28.6-29.8 s vs 26.6-28.0 s for the 430
of Step 10b, interleaved, load 8-34; 28.3-28.9 s at load 18-31 in Step 10b, where the same checks in the
previous order took 32.5-33.0 s; Step 10a's 427 checks took 31.5-34 s at load 8-22).
The suite is bound by CPU time (~360 core-seconds, ~310 processes, each runq start ~0.5-0.9 core-s of loading
at 8 threads), so everything starts as early as it can: every `-m logits` run the selected groups need, 3 at a
time, the generic build's at 4 threads in the background (`prefetch_all` in `check.py`), the sequential
(`-b 1`) ones at 4 threads unless a test names a thread count (`SEQ_THREADS`: token by token they are
memory-bound, and at 8 threads next to the other jobs the 297-token `long2` runs took ~8 s each); from the
start too, on their own workers: the server test, the API tests (2 workers), the greedy runs and the chat
cases (3 at a time; their outputs do not depend on the thread count or the timing), the GGUF template and
Q4_K_M dequant runs. The tests then check the results in their usual group order (in group order these
processes ran at the end with half the CPUs idle, Step 10b). It needs `models/Qwen3-0.6B-q8.bin`, plus numpy
(`.venv/bin/python` if present, else the system `python3`); the `gguf` group and the `gguf_*` greedy
goldens also `models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf` (`scripts/download_models.py q8gguf`;
skipped with a note if it is missing, `GGUF=` to skip), the `q4` group and the `q4_*` greedy goldens
`models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf` (`scripts/download_models.py q4km`; `Q4=` to skip), the `q35` group
`models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf` (`scripts/download_models.py 3.5-0.8b-q8`; `Q35=` to skip) and, if
`Qwen3.5-0.8B-Q4_K_M.gguf` is next to it (`3.5-0.8b-q4km`), the same checks on that file (Q5_K).
`tests/run_tests.sh --models --model-tags 3.5-4b-q8` runs the models group on some FAMILY tags only. Qwen3.8-27B
(`3.8-27b-q4km`, ~17 GB of RAM) runs only when MemAvailable covers runq's estimate + 1.5 GiB (else skipped with a
note; `GWENEDEG_TEST_MEM=GiB` stands in for MemAvailable); give it `TMPDIR=/var/tmp` where `/tmp` is a RAM tmpfs (its
logits files, ~1 GB).
Torch is only needed to regenerate the HF goldens, the `gguf` package (`pip install gguf`, e.g. in a
scratch venv) only for the Q4_K_M / Q5_K dequantization goldens.

```sh
tests/run_tests.sh                    # all tests, OMP_NUM_THREADS=8 unless set
tests/run_tests.sh -v                 # + per-position logits metrics and a summary table
tests/run_tests.sh --only logits      # subset: tokenizer,logits,batched,threads,kernels,greedy,chat,gguf,q4,kv,api,server,synth,q35
tests/run_tests.sh --models           # the bigger Qwen3 models (group models, not in make test; = make test-models, ~4.5 min + 7 min for the 27B)
TMPDIR=/var/tmp tests/run_tests.sh --models --model-tags 3.8-27b-q4km   # Qwen3.8-27B alone (61 checks, ~7 min)
tests/run_tests.sh --models --update-models   # ... and rewrite their greedy goldens (golden/models/greedy)
GGUF=other.gguf tests/run_tests.sh    # the gguf group on another Q8_0 GGUF file (GGUF= skips it)
Q4= tests/run_tests.sh                # without the Q4_K_M group
RUNQ=/path/to/other-binary tests/run_tests.sh   # test another build (no make; kernels and api tests skipped)
RUNQ=./runq RUNQ_GENERIC=./runq-generic tests/run_tests.sh   # ... with the kernels test
tests/run_tests.sh --update-greedy    # regenerate the greedy goldens (incl. the chat transcript)
.venv/bin/python tests/make_golden.py # regenerate the HF goldens (tokenizer + logits, ~15 s)
.venv/bin/python tests/make_golden.py --tokenizer-only   # only golden/tokenizer.json (~2 s)
.venv/bin/python tests/make_golden.py --tokenizer-qwen35  # only golden/tokenizer_qwen35.json (Qwen3.5's tokenizer, ~2 s)
.venv/bin/python tests/make_golden.py --q4ref --gguf-py DIR   # only golden/logits_q4 (~20 s; DIR holds the gguf package)
.venv/bin/python tests/make_golden.py --long  # only golden/logits_long (positions past 256 of long2, ~5 s)
python3 tests/make_dequant_golden.py  # golden/dequant/q4_k_m.npz (needs numpy + gguf, no torch; ~1 s)
python3 tests/make_dequant_golden.py --q5   # golden/dequant/q5_k.npz (Q5_K rows of the Qwen3.5 Q4_K_M file)
.venv/bin/python tests/make_chat_golden.py   # golden/chat_render.json (HF apply_chat_template of the 5 templates, ~3 s)
.venv/bin/python tests/make_golden.py --model-dir models/Qwen3-1.7B --out-dir tests/golden/models/1.7b-hf  # HF fp32 (~8 GB)
.venv/bin/python tests/make_model_ref.py TAG FILE.gguf   # golden/models/TAG: dequantized-weights reference (numpy, 15-95 s)
.venv-hf5/bin/python tests/make_model_ref.py 3.5-0.8b-q8 FILE.gguf   # a qwen35 file: tests/ref_qwen35.py (docs/qwen35.md; ~40 s)
python3 tests/make_synth.py [--force] # build/synth: the synthetic models of the synth group (check.py runs it; ~1 s)
.venv/bin/python tests/fuzz_tokenizer.py                 # on-demand tokenizer fuzz vs HF (~12 s)
.venv/bin/python tests/fuzz_tokenizer.py --model models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf   # GGUF tokenizer (~20 s)
.venv/bin/python tests/fuzz_tokenizer.py --nfc-all       # exhaustive NFC normalizer check vs HF (~8 s)
.venv/bin/python tools/gen_unicode_tables.py --check-hf  # HF's NFC data vs Unicode 15.1 (~1 s)
tests/compare_bins.sh REF_DIR NEW_DIR # two builds' outputs byte for byte, every model family (below; ~25 min, --quick ~2)
build/api_test MODEL [--n65] [--kv]   # the library API tests alone (make api-test; ~4 s per model at 4 threads)
build/api_test MODEL --digest         # only its batched passes' digests (compare thread counts / builds)
build/api_test --two MODEL_A MODEL_B [--gs MODEL_GS]   # two models in one process (api_pair, q35/two; ~3-5 s)
build/api_test --spec TARGET DRAFT [--quick] [--other MODEL]   # speculative decoding == plain (spec_api; ~10-30 s)
.venv/bin/python tests/sampler_ref.py --runq ./runq --api-test build/api_test --api-test build/api_test-generic --model models/Qwen3-0.6B-q8.bin   # top-k vs numpy (sampler/*; ~13 s)
build/bench_batch MODEL [-N 1,8] [-L 50,2000] [-P 64] [-d]   # batched decode benchmark (make bench-batch; not a test)
build/bench_verify MODEL [-L 500,2000] [-R 1,2,4] [-c] [-A "RUNQ_BD=64,2,8,1"]   # verify-pass cost: an r-row segment of one sequence vs a decode step vs r decode rows (make bench-verify; not a test)
tests/run_tests.sh --long-context     # group long alone (opt-in): Qwen3.5 at up to 98k positions vs golden/long (~35 min at 8 threads)
.venv/bin/python tests/check_long.py --retrieval FILE.gguf   # only the retrieval check (any qwen35 file, no golden; e.g. the 27B)
.venv/bin/python tests/make_long_ref.py 3.5-0.8b-q8 FILE.gguf   # golden/long/TAG: the long-context reference (~30 min; -n 32768 for the 4B: ~21 min)
build/long_ctx MODEL -I IDS -P p1,p2 -o OUT [-V] [-g N]   # logits at positions of a long token file, the verify / decode paths there, greedy generation (make long-ctx)
build/bench_long MODEL [-L 16384,32768,65536,98304] [-R 2,4,8] [-P 2560]   # decode / verify / prefill-chunk times at long context on a synthetic cache (make bench-long; not a test)
tests/server_test.py --runq ./runq --model MODEL [--threads 4] [--slow] [-v]   # runq serve alone (~6 s; 2 threads ~11 s)
scripts/bench_serve.py --runq ./runq --model MODEL [--only gen,judge,mix]   # runq serve under concurrent clients (not a test)
```

Exit status is non-zero if any test fails. XFAIL = known engine bug, reported but not failing;
XPASS = an XFAIL case now passes (remove its mark in `make_golden.py`).

## Engine hooks (`src/cli.c`)

All hooks work the same on a GGUF file (tokenizer and templates from the file and built in).

- `-m tokenize -i TEXT`: token ids of the raw text (no chat template), space-separated on stdout.
  Only the checkpoint header and the tokenizer are loaded (no weights; GGUF: the metadata), on one
  thread. stderr gets `[tokenize] B bytes -> N tokens, encode X ms, tokenizer load Y ms` (encode
  time only).
- `-m logits -i TEXT -o FILE [-b N]`: writes `int32 n, int32 vocab, int32 ids[n],
  float32 logits[n][vocab]` (row i = logits after token i). `-b 1` feeds the tokens one at a
  time through the decode path (`rq_prefill` on a context without batch buffers: `rq_decode` per
  token); `-b N > 1` (default 512) uses the batched prefill, `rq_forward_batch` in chunks of N tokens
  with the logits of every row. The tests always pass `-b` explicitly.
- `RUNQ_CHAT_DUMP=PREFIX` (chat mode): at exit, writes `PREFIX.ids` (the token ids in the KV
  cache, i.e. the conversation as the model saw it, space-separated) and `PREFIX.logits`
  (`int32 n, int32 vocab, int32 pos[n], float32 logits[n][vocab]`: every logits vector chat
  sampled from, `pos` = position of the token that produced it; records of positions that a
  re-prefill overwrote are dropped, so they always match the final cache).
- `-m dequant -i NAME[:ROW0[:NROWS]] -o FILE` (GGUF only): rows of the matrix NAME (llama.cpp tensor
  name, e.g. `blk.0.attn_q.weight`, `token_embd.weight`) dequantized from the file's blocks (the
  embedding lookup's code) and, for Q4_K / Q6_K, also from the repacked Q4P / Q6P weights the
  matmul kernels read. `int32 nrows, ncols, npacked, float32 blocks[nrows][ncols]`, then
  `float32 repacked[nrows][ncols]` if npacked is 1.
- `-m render -i JSON [-r 0|1] [-e EFFORT]`: the token ids of `rq_chat_render` (the chat template as the CLI's chat and
  `runq serve` render it) for the messages `-i` (`[{"role": "system" | "developer" | "user" | "assistant", "content":
  ...}]`), or for several conversations (`[{"messages": [...], "thinking": 0 | 1, "reasoning_effort": "low"}, ...]`,
  one line of ids each); stderr `[render] chat template <variant>, thinking T: N tokens` (Qwen3.8: `thinking T
  (EFFORT): N tokens`; thinking on unless `-r 0`, as the chat). Tokenizer only (no weights), so it also
  works on tokenizer-only GGUF files (`make_synth.py`'s `synth-tok-*.gguf`). Unlike the chat, it renders
  Thinking-2507 with `-r 0` as asked (the library's form for a constrained reply in `runq serve`).
## What is checked

| test | what | golden |
|---|---|---|
| `tokenizer/*` | runq ids == HF `encode(text, add_special_tokens=False)` for 81 strings: ASCII, whitespace runs of 1-8 chars before words/punctuation/digits/newlines, trailing whitespace, `\r\n`/`\r`, tab/space indentation, digits (long numbers, mixed with letters, non-ASCII digits), punctuation runs, contractions (any case, `'ſ`), C/Python code, NBSP, ideographic and other Unicode spaces, U+001C/zero-width (not `\s`), control chars, emoji (ZWJ, flags, skin tones, keycaps), rare CJK (Ext. B-I), Arabic, Hindi, Korean, Russian, Greek etc., math, two 3,000-char strings without spaces, added tokens (adjacent to text, back to back, partial ones), 1-char strings; NFC (`nfc_*`, 13): decomposed Latin, marks in non-canonical order and blocked ones, Hangul jamo (L+V, L+V+T, LV+T), decomposed Vietnamese and Greek, singletons (U+212B, U+2126, U+2000, CJK compatibility), composition exclusions (U+0958, U+FB2C, U+1D160), 600-mark runs, expansion (more tokens than input bytes), marks right after added tokens (normalized separately, as in HF), marks HF's older Unicode data treats as ccc 0 (`nfc_hf_compat`) | `golden/tokenizer.json` |
| `tokenizer/nfc_invalid_utf8` | 6 strings with invalid UTF-8 next to marks, Hangul jamo and expanding characters: the concatenated token bytes equal the valid runs NFC-normalized separately with the invalid bytes kept in between (runq's rule, no HF reference: see the fuzz section) | Python `unicodedata` |
| `tokenizer/speed_*` | encode time inside runq (stderr line) < 100 ms for ~2,000 tokens of the golden texts and for one 30,000-byte chunk (no spaces); measured ~0.3-1.3 ms and ~5-13 ms | |
| `logits/<prompt>` | runq Q8_0 logits vs HF fp32 for 5 prompts (en, code, zh, chat template, counting), 15-24 tokens; ids must match HF first | `golden/logits/*.npz` |
| `logits/cos_pooled`, `logits/top1_pooled` | the same metrics pooled over all prompts | |
| `logits_batched/*` | the same test (same thresholds) with the batched prefill, `-b 64` (one chunk here, the same code as the default `-b 512`) | `golden/logits/*.npz` |
| `batched/<prompt>_b7`, `_b64` | batched logits vs sequential (`-b 1`) on every logits prompt; `-b 7` gives 3-4 chunks with boundaries mid-prompt. `all_b64`: all 5 prompts concatenated (86 tokens), 2 chunks at `-b 64`; `long_b64`: `all` twice + `en` (187 tokens, 12 KV tiles, 3 chunks: attention over several K/V tiles and causal row blocks in later chunks). Max abs diff <= 1e-3 and cos >= 0.99999 per row (full vocab), finite; reports "bit-identical" (currently all 12) | sequential run |
| `threads/bitwise`, `threads/bitwise_b7` | logits file for `en` is byte-identical with OMP_NUM_THREADS=8 and 1, sequential and `-b 7` | |
| `threads/bitwise_long_b64` | `long` at `-b 64`: OMP_NUM_THREADS=8 vs 3 byte-identical (other partitions of the dynamically scheduled attention work items) | |
| `logits_long/long2` | `long2` (297 tokens: `long` + `all` + `code`), `-b 64`, vs HF fp32 at every 4th position from 248 on, i.e. past the first attention block of 256 positions (Step 7a: per-block partial results, combined): Q8_0 thresholds (cos mean, every position, top-1, top-5); measured cos mean 0.99927, min 0.99791, top-1 13/13 | `golden/logits_long/long2.npz` |
| `batched/long2_b64`, `long2_b512` | `long2` at `-b 64` (5 chunks) and `-b 512` (one chunk) vs `-b 1` (sequential decode: from position 256 on, the attention of each kv head split over threads per block), bitwise | sequential run |
| `threads/bitwise_long2_b64_t1`, `_t3`, `_t16` | `long2` at `-b 64`: OMP_NUM_THREADS=8 vs 1, 3, 16 byte-identical | |
| `kernels/en_b1`, `en_b7`, `all_b64`, `long2_b64` | logits file byte-identical for `./runq` (AVX-512: VNNI GEMM/GEMV and the AVX-512 attention) and `./runq-generic` (`-DRUNQ_GENERIC`, generic C kernels and C attention, same flags otherwise): GEMV incl. classifier and decode attention (`-b 1`), GEMM with 6 + 1 row tiles and a 1-row chunk (`-b 7`), 2 chunks with 6 + 4 row tiles and the all-rows classifier GEMM (`all`, `-b 64`), 5 chunks with attention over 19 tiles, causal row blocks in later chunks and 2 blocks of 256 positions (`long2`; `long`, a prefix of it, was a case too until Step 10b). The generic runs (all groups) go in the background at 4 threads | `./runq-generic` |
| `greedy/*` | exact stdout of `-t 0 -c 4096` runs: chat, chat with thinking (`-r 1`), chat with system prompt (`-y`), generate; 40-48 tokens. These use the default batched prefill; `generate@b5` = generate in 3 chunks (same golden) | `golden/greedy/*.txt` (our engine) |
| `greedy/seeded_chunks` | seeded sampling (`-t 0.8 -s 42`, chat): same output with `-b 1` and `-b 5` (chat advances the RNG once per prompt token, as before batched prefill) | |
| `greedy/seeded_chat`, `greedy/seeded_generate` | exact stdout of seeded sampling: the `-b 1` run of `seeded_chunks` (`-t 0.8 -p 0.9 -s 42`) and generate at `-t 1.5 -p 0.95 -s 7` (~1-2k top-p candidates per token, nuclei up to ~100 tokens); goldens written by the binary before the top-p heap / vectorized argmax (Step 4), which give the same tokens | `golden/greedy/seeded_*.txt` |
| `chat/multiturn_golden` | exact stdout of a 3-turn chat piped on stdin (`-t 0 -n 32`) | `golden/greedy/chat_multiturn.txt` |
| `chat/multiturn_memory` | the turn-2 reply to "What is my name and favourite colour?" mentions `Ada` and `green` (told in turn 1) | |
| `gguf_tokenizer/*` | the same 84 tokenizer checks on the tokenizer runq builds from the GGUF metadata (token bytes for `nfc_invalid_utf8` from the .bin's `.tokenizer`: identical) | `golden/tokenizer.json` |
| `gguf_logits/{en,code}`, `gguf_logits_batched/*` | logits vs HF fp32 on the GGUF Q8_0 file, same thresholds: sequential (`-b 1`) on en and code, batched (`-b 64`) on all 5 prompts, pooled metrics | `golden/logits/*.npz` |
| `gguf_batched/*`, `gguf_threads/bitwise_b7`, `gguf_kernels/*` | GGUF (group size 32 kernels), byte-identical logits: `-b 7` / `-b 64` vs `-b 1` (en, code), OMP_NUM_THREADS 8 vs 3 (en `-b 7`), VNNI vs generic on en `-b 1` (GEMV), en `-b 7` (GEMM 6 + 1 rows), all `-b 64` (2 chunks, 6 + 4 row tiles, all-rows classifier) | sequential / other build |
| `gguf_template/*` | the built-in chat templates (GGUF) give the same chat token ids as the .bin's `.template*` side files, `-r 0` / `-r 1`, with and without `-y`, and decode to the side file text | `.template*` files |
| `gguf_mem/refused`, `gguf_mem/estimate`, `gguf_mem/bad_value` | the load-time memory check (README, Memory) on the GGUF Q8_0 file: `--mem 0.1` refuses it with one stderr line ("needs ~N GiB (weights ..., runtime ..., a sequence of 40960 positions: KV cache 4.38), more than the budget of 0.10 GiB (--mem): not loaded ...") and exit status 1, nothing on stdout; `--mem 64 -c 1024` prints the `[memory]` line with the KV cache of 1,024 positions (28 layers x 1,024 values x 2 x 2 bytes x 1,024 = 0.11 GiB); `--mem lots` is a usage error. That the refusal comes before any weight is read: by hand below ([Memory](#memory)) | |
| `greedy/gguf_chat`, `greedy/gguf_generate` | exact stdout of `-t 0` chat / generate on the GGUF file (same prompts as `chat`, `generate`) | `golden/greedy/gguf_*.txt` |
| `q4_dequant/*` | Q4_K_M: 2 rows each of blk.0.attn_q, blk.3.ffn_down (Q4_K, 1024 / 3072 columns), blk.0.attn_v, blk.27.ffn_down, token_embd (Q6_K), across a 16-row block boundary, `-m dequant`: from the blocks and from the repacked layout, both bit-identical to gguf-py's `dequantize` | `golden/dequant/q4_k_m.npz` |
| `q4_logits_batched/*` | Q4_K_M logits vs HF fp32 (-b 64, 5 prompts, pooled), loose Q4 thresholds (below) | `golden/logits/*.npz` |
| `q4_ref_batched/*`, `q4_ref/en` | Q4_K_M logits vs the Q4 reference: HF fp32 with the GGUF's weights dequantized by gguf-py (-b 64 on 5 prompts + pooled; -b 1 on en), tight thresholds | `golden/logits_q4/*.npz` |
| `q4_batched/en_b7`, `all_b7`, `all_b64`, `q4_threads/bitwise_all_b7`, `_all_b64` | Q4_K_M byte-identical logits: -b 7 / 64 vs -b 1 (`all`: 13 / 2 chunks), OMP_NUM_THREADS 8 vs 3 (other GEMM chunk sizes and partitions) | sequential run |
| `q4_kernels/en_b1`, `en_b7`, `all_b7`, `all_b64` | Q4_K_M logits byte-identical for `./runq` (AVX-512 VNNI K-quant GEMV / GEMM, `src/kernels_avx512.h`) and `./runq-generic` (`q4p_dot` / `q6p_dot`, the definition in `src/kquants.h`): GEMV incl. the Q6_K classifier (`-b 1`), GEMM tiles of 6 + 1 rows (Q4_K) / 4 + 3 (Q6_K) and a 1-row chunk (`en -b 7`), 13 chunks (`all -b 7`), 2 chunks with tiles 6 + 4 / 4 + 2 and the all-rows classifier GEMM (`all -b 64`). The generic runs go in the background at 4 threads (the slow part: ~1 s per 86 tokens with all-rows logits) | `./runq-generic` |
| `greedy/q4_chat`, `greedy/q4_generate` | exact stdout of `-t 0 -n 24` chat / generate on the Q4_K_M file | `golden/greedy/q4_*.txt` |
| `chat/multiturn`, `chat/thinking`, `chat/context_full` | multi-turn runs with `RUNQ_CHAT_DUMP`: (1) the decoded KV cache equals the expected chat-template rendering, built in Python from the user lines and the replies on stdout (ChatML, `<|im_end|>\n` after every message, the empty think block before each answer with `-r 0`; with `-r 1` earlier answers without their reasoning, as HF's template renders them); (2) re-encoding that text gives the same ids (piecewise encoding == whole-string encoding); (3) every sampled logits vector is **bit-identical** to the same row of a fresh `-m logits` run on the whole conversation (incremental prefill at `pos0 > 0` and decode vs one batched prefill from 0). `thinking`: `-r 1`, 2 turns, turn 1's reasoning dropped (re-prefill of its answer). `context_full`: `-c 160 -n 30 -y ...`, 6 turns: the oldest turns are dropped (stderr `[context] full`), the system prompt kept | |
| `kv_f32_*`, `kv_bf16_*` | the KV cache types other than the default f16 (Step 7b), on the .bin with `-k f32` / `-k bf16`: `kv_*_logits/{en,code}` + pooled vs HF fp32 (`-b 64`, the Q8_0 thresholds), `kv_*_logits_long/long2` (positions 248..296, past one attention block), `kv_*_batched/en_b7`, `long2_b64` vs `-b 1`, `kv_*_threads/bitwise_long2_b64` (8 vs 3 threads), `kv_*_kernels/en_b1`, `long2_b64` (AVX-512 vs generic), all bitwise | `golden/logits*`, other runs |
| `greedy/f32_generate`, `greedy/bf16_generate` | exact stdout of `generate` (the `generate` prompt) with `-k f32` (the text the default gave before Step 7b) and `-k bf16` | `golden/greedy/{f32,bf16}_generate.txt` |
| `chat/long_context`, `chat/long_context_generic` | the same checks on a 411-token conversation (a 374-token system prompt, `-n 12`) decoded at OMP_NUM_THREADS=16: the reply's positions are past one attention block, so decode splits each kv head's attention over threads per block and combines (Step 7a); its sampled logits equal the batched prefill's (`-m logits`, 8 threads) bit for bit, and the generic build's run (4 threads: C attention, generic kernels) gives the same cache ids and logits file | `-m logits`, `./runq-generic` |
| `api_bin/*`, `api_gguf/*`, `api_q4/*`, `api_pair/*` | the library API (`src/runq.h`) per model, `tests/api_test.c`, 30-32 checks each: `interleaved_bitwise` (2 sequences prefilled and decoded alternately on one context == each alone), `batch_prefill_bitwise` (chunks of 2 sequences in one `rq_forward_batch`: all rows / last row / no logits), `batch_decode_bitwise` (decode rows of 3 sequences per pass, one at positions 250+: classifier GEMM over the selected rows), `batch_past_block_bitwise` (a chunk at positions 256..276), `truncate_reprefill_bitwise` (truncate + re-prefill == fresh prefill), `generate_loop` (`rq_generate` == a hand-written sample / decode loop, stats, the reply limit), `generate_step` (TODO P5: `rq_gen_start` / `rq_gen_step`, the step `rq_generate` is built on, == `rq_generate` in three loops: a hand loop on the step, the RNG pattern written out with `rq_sample` (n - 1 draws after the prompt, one per sample; `rq_rng_f32` on a copy of the state = the next draw, not consumed), and a verify-pass loop of TODO S1's shape (an oracle draft, every 4th guess wrong, k = 3: `[t, d1..d3]` in one all-rows prefill, stepped row by row while the sample equals the guess, then truncated; Qwen3.5: back to the checkpoint at the pass's start + the accepted rows prefilled again); equal: the tokens, the RNG state after each sample, each sampled logits vector's position and `rq_logits_hash`, the stop, the cache's tokens and the next decode step's logits; greedy, seeded top-p and the full distribution, limits 40 and 9. Caught: n instead of n - 1 prompt draws (the seeded cases), no re-prefill after a Qwen3.5 truncation (every case)), `errors` (prompt past the capacity: `RQ_ERR_CONTEXT` with nothing prefilled, bad token, a sequence twice in a pass, logits rows without a buffer, a batch larger than the context's), `chat_render` (a message list == the CLI's pieces, thinking off / on), `detok_utf8` (incremental detokenization: whole UTF-8 characters, the text back), `sample_mask` (token mask: greedy and sampled), `model_sampling` (`rq_model_sampling`: only the fields the file has are written; the detail shows them), `load_error`; batched decode (Step 10a): `batch_decode_n1` .. `_n17` (N sequences, prefixes of 8-32 positions of one prompt copied with `rq_seq_copy`, 2 tokens decoded together, one row each per pass == each alone through `rq_decode`, compared via 64-bit hashes of the logits: the decode-rows worker and the small-batch kernels, 1 / 2-8 / more than 8 rows), `batch_decode_long_n3` / `_n9` (positions 248-260: across the first attention block, split and not split), `batch_decode_n65` (Q4_K_M: 65 sequences in one pass, the prefill path's GEMM), `batch_mixed` / `_short` (a 20-row prefill chunk with all logits + 5 decode rows, a 2-row chunk + 2 decode rows: the decode-rows path since TODO S2), `batch_verify` (TODO S2: verify segments, k + 1 rows of one sequence with every row's logits: 4 rows at position 20, also with the last row's logits only, and 9 rows at 250-258; each row == the tokens decoded one by one through `rq_decode`; with `rq_seq_checkpoint_rows` before the pass, truncating after each row (the last first) lands exactly there and the next token's logits are the reference's: on Qwen3.5 the checkpoint per row), `batch_verify_mixed` (a 4-row segment + 2 decode rows in one pass; a 6-row segment at 252-257 + a decode row at 257: few attention items, so the long rows' attention is split over its blocks), both on the default context (the decode-rows worker: segments of at most 16 rows) and as `_prefill` on a context made with `RUNQ_BD=64,2,8,1` (the prefill path, as before TODO S2: there the single row's attention is split); DIGEST lines `batch_verify`, `_long`, `_mixed`, `_mixed_long` (+ `_prefill`), `batch_sample` (`rq_sample_batch` == `rq_sample` per row: tokens and RNG states, per-row settings incl. top-k), `batch_decode_long_bf16` (.bin), `seq_copy_errors`; and `batch_digest_t1` / `_t8` / `_t16` (Q4_K_M: `api_test --digest` at those thread counts, every batched pass's hash == the main run at 4 threads) and `batch_digest_generic` (`build/api_test-generic --digest-small` at 3 threads: the short passes of the .bin and Q4_K_M; `--long` adds the long ones, ~3x the time). `api_pair/*` (TODO P2: a draft and a target model of speculative decoding share the process's thread pool and activation group size): `api_test --two Q8_0.gguf Q4_K_M.gguf --gs Qwen3-0.6B-q8.bin`, 6 checks, ~3-5 s. Each model's script (13 steps: an all-rows prefill, a 2-chunk prefill with a checkpoint at 40, a decode step, a 20-row prefill chunk between 2 decode rows, 3 decode rows, a truncation (Qwen3.5: to the checkpoint) + re-prefill, another next to a decode row, `rq_seq_copy` + decode, 4 decode rows + seeded `rq_sample_batch`, greedy `rq_generate`; then a decode step, a mixed pass and a seeded `rq_generate`) is first run alone in a fresh process (fork), then both models are loaded in one process and their first 10 steps interleaved (A first on even steps, B on odd ones): `two_interleaved_a` / `_b` (every step's hash of its logits, tokens, lengths and RNG states == alone), `two_free_other` (A freed with its context and sequences, B's last 3 steps == alone), `two_cross_model` (a sequence of one model on the other's context: `rq_decode`, `rq_prefill`, `rq_forward_batch`, `rq_seq_copy` refused with nothing run), `two_gs_refused` (the .bin, group size 64 vs 32, refused with "group size ... one group size per process" while A is loaded, mid-run and after A is freed; the steps after it still == alone), `two_gs_after_free` (both freed: the .bin loads and decodes). DIGEST lines `two_a` / `two_b` are the same on `build/api_test-generic` (checked by hand: the generic run takes ~10-18 s). Also by hand: Qwen3-0.6B + Qwen3-1.7B Q8_0, Qwen3.5-0.8B Q8_0 + Q4_K_M, both builds; the Qwen3.5 pair of the default suite is `q35/two/*`. A scratch build whose refused load left the group size at 64 fails (B no longer loads). Runs in the background from the start (3 threads for the .bin, 4 for the others). Breakage (scratch builds): one sequence's KV cache for every attention row block, or the first position for later segments: the batch checks fail; an FMA in the small-batch kernels' fold (Q8_0, Q4_K or Q6_K): 14 checks of each affected model fail; every row of an attention row block over its last row's positions, both paths (not causal; TODO S2): 9 checks fail, `batch_verify*` among them; no checkpoint in a multi-row segment's recurrence on the decode-rows path: `batch_verify` / `_mixed` and `checkpoint_sets` (Qwen3.5) | other runs of the same binary; the generic build |
| `sampler/*` | (TODO P5; group `api`) the sampler's top-k (`src/sampler.c`, `sample_topk`) replayed in numpy draw for draw, `tests/sampler_ref.py`: `runq -m logits` rows of a short prompt (the .bin) + 4 rows with ties (rounded to 1/2, the top 40 equal, all equal, 30 values left by a mask), sampled by `api_test MODEL --sample FILE` (a plan on stdin: settings, row, RNG states) in both builds. 9 settings (`topk_ref/t<T>_p<P>_k<K>`: the 27B's 1.0 / 0.95 / 20, Qwen's 0.6 / 0.95 / 20 and 0.7 / 0.8 / 20, top-k alone, k = 1 / 2 / 40 / 500 / 5000, top-p 0.5 ... 1), ~3,300-4,200 draws each: 150 seeded ones per row, plus RNG states chosen (xorshift inverted) so that the draw lands within 2 steps of 2^-24 of each of the first 8 running-sum boundaries, where a one-ulp difference changes the token; every token must equal the reference's, and `topk_builds`: AVX-512 == generic. The reference mirrors the float32 operations: candidates by logit, ties by the lower id; weights `expf((x - max) * (1 / T))` with the C library's `expf` (ctypes); the sum in candidate order; top-p as the first prefix sum > topp * sum; `r = coin * prefix sum` against the running sum. Caught (reference variants vs the C): ties by the higher id (every setting), the sum in reverse order (2 settings), `x / T` divisions (4), numpy's `exp` (6), float64 weights (6); with the seeded draws alone only the tie order and the reversed sum (5 draws) showed. Found on the way: `-Ofast` compiles most of `sample()`'s division of the vocabulary by T as a multiplication by 1 / T (top-k does not use it). ~12-15 s of one background worker | `tests/sampler_ref.py` |
| `spec_api/*`, `spec_cli/*` | (TODO S1; `tests/check_spec.py`, in the background from the start, 3 runs at a time at 4 threads) speculative decoding (`src/spec.c`) == plain decoding. `spec_api/qwen3/*`: `api_test --spec` on the 0.6B Q8_0 (target) + Q4_K_M (draft), 11 checks: per case `rq_generate` on fresh sequences, then `rq_generate_spec` with k = 1, 3, 8 and k = 8 with tau 0.5, equal: the tokens, the RNG state after each, every sampled logits vector's position and `rq_logits_hash`, the stops, the cache's tokens and the logits of one more decode step. Cases: a chat turn greedy (an end token inside a verify pass) and seeded (T 1, top-p 0.95, top-k 20), thinking on greedy / seeded (T 0.8, top-p 0.9) to `max_new` 45 / 37, raw code greedy / seeded to the end of a context of prompt + 29 (and a draft context of prompt + 12: fewer guesses near its end), a draft context shorter than the prompt (no guesses), two calls on the same sequences (after the first reply both are cut back 3 positions, as a chat's next turn, the draft left as it was: it follows by common prefix); the detail gives accepted / drafted and the passes per setting. `spec_k0` (k 0 = `rq_generate`, no draft), `spec_vocab` (`rq_spec_check`: the draft and the target itself accepted, ~4-7 ms per check; Qwen3.5-0.8B's vocabulary refused). `spec_api/qwen3_generic/*`: the generic build, `--quick` (4 cases, k 3 and adaptive). `spec_api/qwen35/*` (if `Qwen3.5-0.8B-Q4_K_M.gguf` is next to the Q8_0 file): the 0.8B Q8_0 + its Q4_K_M as the draft, `--quick`, + `spec_rollback_exact` (every rollback of both sequences landed on its checkpoint: `RqSpecStats.refed` 0). `spec_cli/*`: `runq -D DRAFT -G k [--tau]` vs the same command without `-D`: stdout byte for byte, stderr but the `[timing]` lines, the exit status, one `[timing] speculative` line per reply, and in chat mode the `RUNQ_CHAT_DUMP` files (the cache's ids, every sampled logits vector bitwise): `gen_*` (generate: greedy, seeded top-k, `-n 2` / `-n 1`, the context's end at `-c 37` / `-c 41 -b 1`), `chat_*` (3 piped turns greedy, thinking seeded, a system prompt with `-n 13`, `-c 160` with 6 turns (the oldest dropped), a reply cut at the end of `-c 100`, one `-i` turn with thinking), `*_generic` (one generate and one chat case on runq-generic), `q35_*` (the 0.8B pair: a thinking chat whose second turn rolls both sequences back to the checkpoint, a seeded generate), `g0` (`-G 0 -D` == no `-D`, stderr included), `refused_vocab` (`-D` a Qwen3.5 file for a Qwen3 model: exit status 1, "not usable as a draft"), `refused_no_draft` (`-G 3` without `-D`), `bad_tau`. Breakage (a scratch build keeping one rejected row: the target cut at `keep + 1`): `spec_think_seeded_limit` and `spec_think_turn2_greedy` fail. By hand (2026-10-27): Qwen3-1.7B Q8_0 + 0.6B Q8_0 and + 0.6B Q4_K_M, `api_test --spec` 11/11 each and the 12 CLI cases 12/12 each; Qwen3.5-0.8B pair, the full `api_test --spec` 12/12 | `rq_generate`, runs without `-D` |
| `synth/*` | (Step 11) synthetic GGUF models built by `tests/make_synth.py` from the 0.6B Q4_K_M file into `build/synth` (cached; rebuilt when the source, the script or the templates change, ~1 s), for the shapes of the bigger Qwen3 models that the 0.6B files do not have. `synth-q4.gguf` (176 MB): 3 layers, dim 768 (not a power of two, as 4B's 2560), 8 query / 2 kv heads (GQA 4, q dim 1024 != dim), ffn 1280, an untied Q4_K `token_embd` and a Q6_K `output.weight` (as Qwen3-8B Q4_K_M), RoPE theta 5e6 and context 262,144 (as the 2507 models), the Instruct-2507 template; every matrix whole super-blocks of 0.6B's (layers 0 / 3 / 27: Q6_K / Q4_K / Q6_K attn_v and ffn_down). `synth/load`: shapes, the default context capped at 40,960; `synth/chat_instruct_template`: the chat's KV cache ids == `-m render` (no think block), `-r 1` ignored with a note; `synth/logits_ref/{en,code,long2}` + `top1_pooled`: logits (`-b 64`) vs `tests/ref_forward.py` (numpy fp32 on the file's weights dequantized by `tests/ggufmini.py`, bit-identical to gguf-py; a 4,096-row classifier subset): cos mean >= 0.99, every position >= 0.90, top-1 >= 0.6 pooled (measured 0.9993 / 0.9982 / 0.9958, min 0.962, top-1 282/336: runq's int8 activations on odd cut-out weights; a theta of 1e6 instead of 5e6 gives 0.98 / 0.87 / 0.62 and top-1 45/336); `synth/batched/{en_b7,en_b64,long2_b64}` (vs `-b 1`: the decode path, 297 tokens past one attention block), `synth/threads/long2_b64` (4 vs 3 threads), `synth/kernels/en_b1,b7,b64` (AVX-512 vs generic), all bitwise; `synth/api` (`api_test` on the file: batched decode rows == alone with GQA 4 / dim 768, mixed passes, ...) and `synth/api_digest_threads`; `synth/render_<variant>/r<0,1>_<case>` (30): `-m render` on `synth-tok-{qwen3,instruct-2507,thinking-2507}.gguf` (tokenizer-only files with the HF templates of `golden/templates`) == `golden/chat_render.json`, 23 of them equal to HF's `apply_chat_template` (the others: runq's documented forms, the hybrid with thinking off keeping the empty think block in earlier turns, Thinking-2507 with thinking off). Breakage (scratch builds, all caught): rmsnorm's `ss / size` (the Step 11 bug: 4 `batched/*` + 3 `kernels/*` fail), the embedding decoded by `kp` (NaN: `logits_ref/*`), RoPE theta 1e6 (`logits_ref/*`). ~2.4 s alone, ~1.3-2 s of the suite (in the background from the start) | `tests/ref_forward.py`, `golden/chat_render.json`, other runs |
| `long/<tag>/*` | (opt-in: `tests/run_tests.sh --long-context`, `tests/check_long.py`; TODO G2) Qwen3.5 files with a golden in `golden/long` (`3.5-0.8b-q8`: 98,304 tokens, 14 positions; `3.5-4b-q8`: 32,768, 6), through `build/long_ctx` (the library API; `make long-ctx`). The text: 18 files of this repository at the fixed commit `5d93873` (`git show`), tokenized by runq, its first n tokens. `ids` (the count and sha256 of the ids == the golden's: the text and the tokenizer unchanged); `ref_p<P>`: the logits after token P (1,023 .. 98,303) vs `tests/ref_qwen35.py`'s `forward_long` on the file's dequantized weights (numpy fp32, exact softmax per query, `tests/make_long_ref.py`; the golden keeps the top 64 and 512 fixed ids per position): cos over those ids >= 0.995 and top-1 (or a reference margin < 0.5) per position, `cos_pooled` >= 0.999, `top1_pooled` >= 0.85 (measured 0.8B: cos 0.99989-0.99997 over the ids, 0.9996-0.9999 on the full vocabulary, top-1 14/14); `verify_p<P>` (every position): the last row of a 64-row prefill chunk == a 4-row verify segment (`rq_forward_batch`, every row's logits) == a decode step (`rq_decode`, the attention split per 256 positions over the threads and combined), bitwise; `generic_p<P>` (P <= 16,383): `build/long_ctx-generic` == native, bitwise; `retrieval`: a fact planted after 4,000 tokens of the corpus, the corpus up to 88,000 tokens (a golden of n positions: n - 4,000), a question in a chat turn (thinking off): the greedy reply (<= 24 tokens) contains it. 56 checks, 2,106 s at 8 threads (load ~5-15; 2,887 s at ~30): the 0.8B ~18 min (its `-V` run 515 s, the generic one 126 s, the retrieval 407 s), the 4B ~18 min (232 / 640 / 185 s; ~5.6 GB of RAM) | `golden/long/*.npz`, other paths, `build/long_ctx-generic` |
| `models/<tag>/*` | (opt-in: `tests/run_tests.sh --models`, `make test-models`; Step 11) every downloaded file of `scripts/download_models.py family` (+ `4b-2507t-q4km`, `3.5-0.8b-q4km`, `3.5-4b-q8`), skipped with a note if missing; Qwen3.8-27B Q4_K_M (`3.8-27b-q4km`, TODO S7): 61 checks, 419 s, skipped with a note when MemAvailable is below runq's estimate (18.65 GiB) + 1.5 GiB (`BIG`; `GWENEDEG_TEST_MEM` overrides it), without `api` (its batched-decode checks peak at 22.4 GiB there: by hand, 32/32), `server/greedy_thinking` as for Thinking-2507 (thinking on by default), `logits_ref/*` vs `golden/models/3.8-27b-q4km` (pooled cos 0.99992, top-1 62/62); Qwen3.5-4B Q8_0 (`3.5-4b-q8`, Step 14): 36 checks, ~100 s, incl. `kernels/en_b64` and `api_digest_generic`; ~4.5 min for the 7 files (8 threads), ~13 s for the 0.6B Q5_K_M (`q5km`, Step 13: Q5_K / Q6_K, `logits_ref/*` vs `golden/models/0.6b-q5km`: pooled cos 0.99970, top-1 58/61). Per file: `render/*` (the file's own template variant detected, its `golden/chat_render.json` cases); `greedy` (`-t 0 -n 24` chat, "capital of Australia" == `golden/models/greedy/<tag>.txt`); `logits_ref/{en,code,chat,long2}` + `cos_pooled` vs the file's dequantized-weights reference (`make_model_ref.py`; thresholds as Q4_K_M vs its reference for 0.6B: cos mean 0.995, positions 0.97, pos0 0.98, top-1 0.75, pooled 0.997); for Qwen3-1.7B also `logits_q8_hf/*` / `logits_q4_hf/*` vs HF fp32 (`golden/models/1.7b-hf`; Q4_K_M: no per-position bound, see below); bitwise: `batched/en_b7, en_b64, all_b64` (vs `-b 1`), `batched/long2_b512` (vs `-b 64`), `threads/all_b64` (8 vs 3), `kernels/en_b1, en_b7` (AVX-512 vs generic; the Q5_K files also `en_b64` and `api_digest_generic`: `api_test --digest-small` of both builds, 4 batched passes, equal); `dequant/*` (Q5_K files: `-m dequant` rows vs gguf-py, `golden/dequant/q5_k.npz`: 6 cases, 4 of Q5_K, 1 Q6_K, 1 Q4_K, from the blocks and the repacked layout, bit-identical); `api` (`api_test` on the file: batched decode rows == alone, mixed passes, digests); `server/*` (`runq serve --port 0`: `/v1/models` id, the chat template variant in its log, a greedy completion == the CLI's text (Thinking-2507: a non-empty `reasoning_content`; also Qwen3.8), a `json_schema` classification within its enum) | `golden/models/*`, other runs |
| `q35/*` | (default suite, if the Qwen3.5-0.8B Q8_0 file is there: `--q35`, run_tests.sh's `Q35`; `tests/check_models.py`) the hybrid architecture ([docs/qwen35.md](../docs/qwen35.md)), in the background from the start (~45 s of the suite's wall time instead of ~28 s): `q35/tokenizer/*` (`golden/tokenizer_qwen35.json`, 90 cases: the `qwen35` pre-tokenizer's `\p{M}`; token bytes from the GGUF's metadata), `q35/chat/*` (CLI chats with `RUNQ_CHAT_DUMP`: `q35_thinking` (`-r 1`: turn 2's history drops turn 1's reasoning and `<think>\n`, so the cache goes back to the checkpoint at turn 1's assistant header and prefills the rest; the KV cache == the template rendering and the sampled logits == `-m logits` on the whole conversation, bitwise), `q35_multiturn` (+ the generic build, bitwise), `q35_context_full` (oldest turns dropped)), and the `models/<tag>/*` checks of the file, and of `Qwen3.5-0.8B-Q4_K_M.gguf` if it is next to it (tag `3.5-0.8b-q4km`, 42 checks, ~20 s: Unsloth's Q4_K_M, the Q5_K kernels (`attn_qkv`, `ssm_out`); `dequant/*` Q5_K rows == gguf-py bitwise; `logits_q35q4_hf/*` vs HF fp32: cos means 0.9924 / 0.9959 / 0.9958 / 0.9925 (chat, code, en, long2), min 0.9677, pooled 0.9946, top-1 57/62, thresholds ~2x that error (`TH["q35q4_hf"]`); `logits_ref/*` vs its dequantized-weights reference `3.5-0.8b-q4km`: pooled 0.99986, top-1 62/62; the same bitwise checks incl. `kernels/en_b64`); tag `3.5-0.8b-q8`: render (12 cases), greedy, `logits_q8_hf/*` vs `golden/models/3.5-0.8b-hf`, `logits_ref/*` vs `3.5-0.8b-q8`, batched / threads / kernels bitwise, `api` (28 checks: `checkpoints_bitwise`; `checkpoint_sets` (TODO P3: 3 pending positions in one 64-row prefill chunk with the default 2 slots (the highest 2 stay), then 6 slots: 3 in one chunk, 2 at a chunk boundary (the last row of a chunk and the first of the next), one in a later chunk, truncated to each (highest first) and continued 5 rows == the reference rows; 7 pending + 1 taken on 6 slots (the lowest replaced, also inside the pass), `rq_seq_set_checkpoints` down to 3 (the highest stay); one batched pass of 3 sequences (a 20-row chunk with 4 pending, a 5-row chunk with 3, a decode row with 1), each continued from each; `rq_seq_checkpoint_rows` as a speculative loop uses it: the prompt's checkpoint + 5 verify segments of 4 rows truncated to 1-4 accepted rows (every row's logits == the reference, each length kept exactly, the prompt's checkpoint stays), 3 decode steps with one each, argument errors; two scratch builds, the S copy or the conv-state copy of the first pending row only, fail it; by hand also on Qwen3.5-4B Q8_0, 26/26 in 75 s); and the batched-decode cases build their sequences by prefill: a copy to an earlier length keeps only a checkpoint), server); Step 14, more value heads than key heads (Qwen3.5-4B / 9B / 27B's linear layers): `q35/synth-q35-r2/*` and `q35/synth-q35-r3/*` (`tests/make_synth_q35.py`, cut out of the Q8_0 file, cached in `build/synth-q35`, ~87 MB each: 4 key / 8 value heads and 2 / 6, dim 256, 6 layers): `logits_ref/{en,code,long2}` + `top1_pooled` vs `tests/ref_qwen35.py` on the file (value head h uses key head h % nk; a 4,096-id classifier subset; measured cos mean 0.9996-0.9999, min 0.9971, top-1 0.87-1.0: thresholds 0.999 / 0.99 / 0.8; with HF's mapping h // r instead 0.60-0.89), `batched/*` (`en` -b 7 / 64, `all` -b 64, `long2` -b 64 vs -b 1, -b 512 vs 64), `threads/long2_b64` (4 vs 3), `kernels/*` (AVX-512 vs generic: `en` -b 1 / 7, `long2` -b 64), `api` (r2 only: batched decode rows == alone, checkpoints), `api_digest_generic`; `q35/two/*` (`api_test --two` on the 0.8B Q8_0 file and `synth-q35-r3.gguf`: two hybrid models in one process, their prefills, batched passes and truncations to checkpoints interleaved == each alone, one freed while the other goes on; 4 checks, ~3-5 s, as `api_pair/*` above); the logits files (0.3 GB for `long2`) are reduced to a hash and the subset at once. Two scratch builds with a bug each failed: HF's mapping in `dn_rec` (`logits_ref` cos 0.60-0.88), and the q / k conv redone by every value head (no longer bitwise -b 1 vs batched). `q35/synth/reject_{nv,hadamard,nextn}`: metadata-only files with 6 value heads on 4 key heads, with a `prism.hadamard.*` key and with `nextn_predict_layers` = `block_count` are rejected with their messages. TODO P6, the Qwen3.8-27B's layout at a small scale (if `Qwen3.5-0.8B-Q4_K_M.gguf` is next to the Q8_0 file; `make_synth_q35.py --src-q4`, its docstring; the 27B's header: [docs/qwen38-speculative.md](../docs/qwen38-speculative.md#file-layout)): `q35/synth-q35-27b/*` (330 MB: 8 layers, attention every 4th, dim 768 (not a power of two), FFN 2560, 12 / 2 heads of 256 (GQA 6), 4 key / 12 value heads (r = 3), an untied Q4_K `token_embd` + Q6_K `output`, Q4_K / Q5_K / Q6_K matrices per role as in the 27B's Q4_K_M file (`attn_qkv` Q6_K / Q4_K, `ssm_out` Q5_K, `attn_v` / `ffn_down` Q6_K / Q4_K); `ssm_alpha` / `ssm_beta` stay Q8_0: the 27B's are F32): the same checks as `synth-q35-r2` incl. the full `api` (28/28; `model_sampling` set 7: its general.sampling.* keys) and `sampling_defaults` (the keys as the CLI's defaults for the options not given: seeded `-m generate` without `-p` / `-K` == with `-p 0.95 -K 20`, stdout and stderr, != with `-p 0.9 -K 0`); `logits_ref` measured cos mean 0.9984 / 0.9979 / 0.9966 (en, code, long2), min 0.9803, top-1 281/338: thresholds 0.993 / 0.95 / 0.7 (`TH_Q35_SYNTH_KQ`; int8 activations on cut-out K-quant weights, as the `synth` group's); q heads on kv head h % 2 instead of h // 6 gives 0.31-0.40, value heads grouped 0.25-0.38. TODO S6, the 27B's MTP block: `q35/synth/mtp_logits/{b64,b1,b7_gen}`: `synth-q35-27b-mtp.gguf`, the same + the MTP block as the 27B lays it out (`block_count` 9, `nextn_predict_layers` 1, `blk.8` = a full-attention layer + `nextn.eh_proj` / `enorm` / `hnorm` / `shared_head_norm`) loads its 8 main layers (the block validated, not read) and its `-m logits` file == `synth-q35-27b.gguf`'s, bitwise (`-b 64` and `-b 1`, AVX-512; `-b 7` generic); `q35/synth/mtp_tokenizer_only`: `-m tokenize` on it. The 27B's F32 `ssm_alpha` / `ssm_beta`: `q35/synth-q35-27b-f32/*` (`synth-q35-27b-mtp.gguf` with them F32, the Q8_0 rows dequantized: the same values, applied in fp32 by `f32_rows`; Qwen3.8's template and the 27B's `tokenizer.ggml.bos_token_id`), the same checks as `synth-q35-27b` incl. the full `api` (28/28) and `api_digest_generic` (decode, batched prefill and batched decode rows, AVX-512 == generic): `logits_ref` cos mean 0.9984 / 0.9976 / 0.9966, min 0.9791, top-1 290/338 (the Q8_0 file 0.9984 / 0.9979 / 0.9966, 0.9803, 281/338: the F32 rows' fp32 activations change little next to the int8 ones of every other matrix); `q35/synth-q35-27b-f32/chat_qwen38`: a two-turn CLI chat with `-e low -y "Be brief."`: the first prompt == `-m render -e low` of [system, user] (46 tokens: the low instruction + the system prompt in one block), the second turn prefills only `<|im_end|>\n` + its user turn (15 tokens: Qwen3.8's history keeps the reply as generated, no checkpoint rollback), each reply printed after `<think>`; `q35/synth-q35-27b-f32/server38/*` (9): `runq serve` on it, each request's `usage.prompt_tokens` == `-m render` of its messages with its thinking / effort (the default: on, xhigh, and the greedy reasoning == the CLI's; `reasoning_effort` low / medium / none, `chat_template_kwargs.reasoning_effort` / `enable_thinking: false`, system + developer merged, an assistant `reasoning_content` kept), a late system message: 400. `q35/synth/render_qwen3.8/*`, `render_qwen3.8-hf/*` (45 each): `-m render` on `synth-tok-qwen3.8.gguf` / `synth-tok-qwen3.8-hf.gguf` (tokenizer-only, the 0.8B's tokenizer = the 27B's, with Unsloth's resp. Qwen's 3.8 template: the variant is detected from either) == `golden/chat_render.json`'s `qwen3.8` cases. `q35/serve_slots` (TODO P4): `runq serve` on the 0.8B with `-c 2048 --slots 8 --kv-mem 0.25` starts with 3 slots of 81.8 MiB, each counting its KV cache (24 MiB), its recurrent state and the state's 2 checkpoints (19.3 MiB each), all recomputed here from the file's metadata (the KV cache alone would allow 10; the line's slot size and state must match within 0.06 MiB) | `golden/tokenizer_qwen35.json`, `golden/models/3.5-0.8b-*`, other runs |
| `server/*` | `runq serve` over HTTP (`tests/server_test.py`, Python stdlib only, no network beyond 127.0.0.1, no LiteLLM), on the Q8_0 GGUF (else the .bin), 54 checks; the server (`--port 0 -c 2048 --parallel 4 --slots 6 --chunk 16 --queue 4 --max-body 65536`: continuous batching of up to 4 requests, prompts prefilled 16 tokens per pass next to decoding ones) and five `runq -m chat` reference runs at 2 threads, in the background from the start. `basic/*`: `/health`, `/v1/models`, a model by id (404 for another), 6 requests on one keep-alive connection. `chat/*`: greedy completions == the CLI's stdout bitwise (single turn, a system prompt, a second turn with the history re-sent as messages, whose `cached_tokens` must be the first turn's prompt + reply), seeded `-t 0.8 -p 0.9 -s 42` == the CLI's, and with `top_k: 3` (`-t 1.2 -p 0.95`) == `runq -K 3`'s, thinking (`reasoning_effort`): `reasoning_content` == the CLI's `<think>` text. `cache/*`: the same request repeated (all but one prompt token cached), one sharing only the system prompt, and `cache_prompt: false`: identical text and logprobs (9 significant digits: bit-identical logits). `stream/*`: SSE == non-streaming text, finish reason, usage chunk, `[DONE]`; stop strings (stream and not); thinking streamed. `schema/*`: LiteLLM 1.105's `TierClassification` request (rubric + `response_format` verbatim) at temperature 1 over 8 seeds, the same schema as `$ref` into `$defs` (greedy: the decision step's alternatives are allowed tokens only, forced tokens logprob 0, tokens concatenate to the text), the rubric prefix cached on a later request, a nested schema (object, boolean, integer range, optional property, `anyOf` null, `const`; 6 seeds; each output validates and is `json.dumps`-formatted), a single-output schema (all forced). `batch/*` (continuous batching; every output must be what the request gives alone, bitwise): `mixed_constrained_free_stream`: 7 requests at once (the `$ref` classification greedy with 20 logprobs, two sampled classifications, a sampled nested schema, the library request with logprobs, a stream with a stop string, a seeded one with logprobs) == their results alone (text and logprobs) / the CLI's; `join_mid_generation`: a streamed greedy request runs, three requests (system prompt, seeded, thinking) join after its first token: all == the CLI's; `cancel_one_others_unchanged`: a client leaves a long sampled stream after 4 tokens while three requests run: those == the CLI's, the cancel logged 499; `shared_prefix_bitwise`: four requests sharing a new 90-token system prompt at once (the first prefills it, the others wait for it and copy it from its slot: `cached_tokens` >= 60 for three) == the same four with `cache_prompt: false` (text and logprobs). `errors/*`: 18 OpenAI error objects (400 malformed JSON, invalid UTF-8, missing / empty messages, bad role, image part, `n: 2`, string schema, bad `$ref`, `json_object`, temperature, top_k, max_tokens, tools, context too long with `context_length_exceeded`; 404, 405, 413), 411 on a chunked upload, 400 / 505 on bad request lines. `queue/*`: 4 concurrent requests == each alone, `/health` answers while they run, 10 concurrent with 4 running and `--queue 4` give 503 `overloaded`, a client disconnecting (stream and not) cancels its generation (the next request is served at once; logged 499). `fuzz/server_alive`: 150 mutated / random requests on raw sockets, then `/health`. `shutdown/sigterm`: exit status 0 | the `runq -m chat` runs |

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

Since the SwiGLU NaN fix (2026-10-25, `7b2b034`: `expf` overflow under `-Ofast`; it changed the last layer of
`zh` and `chat`): zh top-5 0.893, chat cos mean 0.99891, pooled cos mean 0.99910; the rest is
unchanged. The batched path gives exactly the same numbers.
Since the Step 3b attention (pooled numbers unchanged, per-prompt values moved slightly, e.g. en
top-5 0.907, code top-1 22/24): current per-prompt table in `../BENCHMARKS.md` (Quality).
The GGUF Q8_0 file (Step 6a, group 32 with fp16 scales) is closer to HF under the same thresholds:
pooled cos mean 0.99939 (1-cos 6.1e-4 vs 9.0e-4), top-1 79/84, pos 0 cos 0.92-0.98 (per-prompt table
in Step 6a of the development log: [History](../docs/DEVELOPMENT.md#history)).

Q4_K_M (Step 6b-1; per-prompt table in the development log): vs HF, pooled cos 0.98402 (1-cos 1.6e-2, ~25x
Q8_0), top-1 69/84, per-prompt cos mean 0.967-0.992, every position >= 0.940, pos 0 0.35-0.46. That
is the 4-bit weights' own error: HF fp32 with the GGUF's weights dequantized (the "Q4 reference",
`make_golden.py --q4ref`) gives pooled cos 0.98459, top-1 67/84 against HF. runq vs the Q4 reference:
pooled cos 0.99954 (1-cos 4.6e-4), top-1 81/84, cos mean >= 0.99883, min >= 0.99677, pos 0 >= 0.9977:
the int8 activations only, the Q8_0-vs-HF level. So the HF comparison is a loose sanity bound and the
reference comparison the sensitive one.

Thresholds (`tests/check.py`): per prompt cos mean >= 0.995, every position >= 1 cos >= 0.98,
pos 0 cos >= 0.75, top-1 >= 0.75, top-5 >= 0.75; pooled cos mean >= 0.997, pooled top-1 >= 0.85.
Logits must be finite. That is about 3x the baseline error (1 - cos) for means, 5x per position.
Q4_K_M vs HF (`TH_Q4_HF`, ~2x its baseline error): cos mean >= 0.94, every position >= 0.85, pos 0 >=
0.10, top-1 and top-5 >= 0.5, pooled cos >= 0.97, pooled top-1 >= 0.65. Q4_K_M vs the Q4 reference
(`TH_Q4_REF`, as Q8_0 vs HF, ~3-5x): cos mean >= 0.995, every position >= 0.98, pos 0 >= 0.99, top-1
and top-5 >= 0.75, pooled cos >= 0.998, pooled top-1 >= 0.9.

Sanity check by deliberate breakage (scratch builds, not committed), all caught:

| breakage | pooled cos (pos >= 1) | per-prompt cos mean | pooled top-1 |
|---|---|---|---|
| RoPE theta 1e4 instead of 1e6 (subtlest) | 0.9876 | 0.977 .. 0.995 (4 of 5 fail) | 0.88 |
| RoPE pairs interleaved instead of split halves | | 0.80 .. 0.91 | 0.57 |
| matmul skips its last 64-group | | 0.81 .. 0.96 | 0.76 |
| no QK-norm / KV written at pos+1 / V read from t-1 / no final norm / attention scale 1/d / rmsnorm eps 1e-2 | | < 0.71 | < 0.16 |
| activation quantizer truncates instead of rounding (precision loss only) | 0.99713 | en 0.99432 fails, others pass | 0.94 |

Q4_K_M breakage (scratch builds, generic kernels, `--only q4`): all 13 structural kernel bugs fail
all 15 Q4 logits checks, vs HF and vs the reference: nibble order swapped, Q4_K scale / min swapped,
wrong Q4_K sub-block scale (j ^ 1), Q4_K min term dropped or from the neighbour group, one value per
Q4_K / Q6_K sub-block dropped (the mildest: pooled cos vs HF 0.719), Q6_K without its -32, Q6_K both
halves with one scale, Q6_K qh crumbs reversed, Q6_K d of the next row; also two decoder bugs
(get_scale_min_k4's high bits, the Q6_K qh shift), which fail `q4_dequant/*` too. Precision-level
changes pass the logits checks (d * sc or d rounded to bf16 in a fold: pooled cos vs the reference
0.99943 / 0.99954; a -31 instead of -32 in 1 of 16 Q6_K sub-blocks: 0.99921; baseline 0.99954); the
bf16 ones flip `greedy/q4_generate`. Kernel rounding deviations are the job of `q4_kernels/*` (bitwise):
an FMA in the AVX-512 Q4_K fold (`fmsub` for (ds xs) isum - dm xf) or Q6_K fold (`fmadd` into the
output; scratch builds, Step 6b-2) gives max |diff| 1.58 / 1.52 and fails all four `q4_kernels/*` (and
`greedy/q4_generate`), while `q4_batched/*` and `q4_threads/*` pass (both sides use the same kernel).
Also checked once: an ASan+UBSan build on Q4_K_M (generate / logits at -b 1 / 7 / 512, chat with
context full, 3 and 8 threads; AVX-512 and generic): no reports.

Thread invariance: the GEMV / GEMM work items are blocks of output rows (and of activation rows),
the attention items (kv head, block of query rows) in prefill and (kv head, block of 256 positions)
in decode; each output is computed whole by one item, the same way whichever thread runs it (Step 7a:
threads claim items from shared counters or steal them, src/threads.h), and each query vector's
result does not depend on the other vectors in its item. So results are bit-identical for any thread
count and any interleaving (tested 1 / 3 / 16 vs 8; also checked 5 and 13).

Batched vs sequential: the batched pass (`rq_forward_batch`) gives bit-identical logits to the decode
step (`rq_decode`; `forward_batch()` / `forward()` before Step 8) because both
call the same numeric kernels (`dot_q8`, `qk_norm_rope`, `attention_kv`, `swiglu`, `rmsnorm`,
`quantize`) per row, and the 4-row GEMM kernel `dot_q8_x4` is compiled without FP contraction
(`NO_FP_CONTRACT`, gcc only) so it rounds like `dot_q8`. With contraction (an FMA in the scale
accumulation) the logits differ slightly and `greedy/chat_system` flips a token. A future kernel
that sums in another order may legitimately need the 1e-3 tolerance (or a looser one, justified);
real bugs are far off. Deliberate breakage (scratch builds, all caught):

| breakage in the batched pass | max abs diff | min cos | caught by |
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

Attention blocks (Step 7a): the attention is defined over blocks of 256 positions (per block the max,
exponentials, sum and weighted V, then a fixed-order combination; `src/kernels_avx512.h` header). Up
to 256 positions this is exactly the Step 3b definition below, so every shorter prompt (all the HF
goldens, the greedy goldens) gives the same bits as before. Past it: `logits_long/long2` against HF,
`batched/long2_*`, `threads/bitwise_long2_*`, `kernels/long2_b64` and `chat/long_context*`. Breakage
(scratch builds): the combination without the rescale (c_b = 1) fails `logits_long/long2` (cos mean
0.397) and `chat/long_context`; a sum of the l_b without fma in the generic build only fails
`kernels/long2_b64` (max |diff| 0.54) and `chat/long_context_generic`.

KV cache types (Step 7b): the default is the f16 cache (`-k f16`), so every test above runs on it; the
greedy and seeded goldens were regenerated for it (7 texts changed, all read fine) and `-k f32` still
reproduces every earlier golden exactly (checked once; `greedy/f32_generate` keeps one). The logits
thresholds are the same for all three types (measured in Step 7b: pooled cos .bin f32 / bf16 /
f16 0.99910 / 0.99900 / 0.99905, long2 cos mean 0.99927 / 0.99909 / 0.99921). The 16-bit caches round at
the store (integer operations in C, the same in both builds) and convert exactly at the load, so the
AVX-512 and generic attentions still compute the same fp32 operations: `kernels/*` and `kv_*_kernels/*`
are bitwise.

Sanitizers (Step 7b, scratch builds): ASan + UBSan (AVX-512 and generic) and ThreadSanitizer (no OpenMP),
22 runs each over the three KV types (logits at -b 1 / 7 / 64 / 512 on a 2-block prompt, 3 / 8 / 16
threads, Q8_0 GGUF and Q4_K_M, generate, chat with context full, chat -r 1 -t 0.8): no reports.

Sanitizers (Step 7a, scratch builds, not in `make test`): ASan + UBSan (`-O1 -g -fsanitize=address,undefined`,
AVX-512 and generic) and ThreadSanitizer (`-O1 -g -fsanitize=thread`, built without `-fopenmp`, so the
thread pool runs everything: loading, prefill, decode; libgomp is not instrumented): 17 runs each
(Q8_0 / Q4_K_M GGUF generate and logits at 3 / 8 / 16 threads on a 274-token prompt, -b 1 / 7 / 13 /
64 / 512, the .bin, chat with context full, chat -r 1 -t 0.8): no reports.

Attention (Step 3b): the attention is defined once (arithmetic and tile layout in the
`src/kernels_avx512.h` header comment): scores as fma chains over chunks of 32 dims, a clamped
Cephes-style `exp`, 16 partial sums with a fixed reduction tree, one division, V accumulated by
an fma chain in position order. Decode (`rq_decode`, 2 query vectors per kv head) and prefill
(`rq_forward_batch`, 2 x 6 vectors per work item, causal per row) run the same per-vector
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
`golden/logits_q4/*.npz` (`make_golden.py --q4ref`, same prompts, format and vocab subset): the same HF
model with every weight replaced by the Q4_K_M GGUF's tensor (`token_embd` -> embed_tokens, tied
lm_head; norms F32 as stored) dequantized by gguf-py 0.19.0; regenerating gives byte-identical files.
`golden/dequant/q4_k_m.npz` (`make_dequant_golden.py`, 43 KB): tensor names, first rows, the rows by
`gguf.quants.dequantize` (float32) and their types.
`golden/templates/*.jinja` (Step 11): the `chat_template` of the HF repos' `tokenizer_config.json` (Apache-2.0):
`qwen3.jinja` (Qwen/Qwen3-0.6B .. 14B, identical in all; e.g. Qwen/Qwen3-4B @ 1cfa9a7), `instruct-2507.jinja`
(Qwen/Qwen3-4B-Instruct-2507 @ cdbee75), `thinking-2507.jinja` (Qwen/Qwen3-4B-Thinking-2507 @ 768f209); TODO S6:
`qwen3.8.jinja`, the 27B GGUF's `tokenizer.chat_template` (unsloth/Qwen3.8-27B-GGUF @ 4a03c64, the file runq reads:
Unsloth's rewrite with the developer role, merged system messages and `high` = `xhigh`), `qwen3.8-hf.jinja`
(Qwen/Qwen3.8-27B @ 1d4bf0f, `chat_template.jinja`; Apache-2.0, the model card's license).
`golden/chat_render.json` (`make_chat_golden.py`, ~95 KB): 48 cases (4 variants x thinking 0 / 1 x 6 conversations:
one turn, a system prompt, multi-turn, an earlier reply with a reasoning block, Unicode, surrounding whitespace),
each with runq's canonical rendering (text, ids) and whether it equals HF's `apply_chat_template` (transformers
4.53.0); the `qwen3.5` cases with `golden/templates/qwen3.5.jinja` (Qwen/Qwen3.5-0.8B @ 2fc0636) and that model's
tokenizer. The 45 `qwen3.8` cases: thinking off, and on with the default effort (no `reasoning_effort`: xhigh), `medium`,
`low` and `high` (= xhigh), x the 6 conversations + 3 of system messages (`developer`, `system` + `developer` merged, an
empty one): 40 equal HF's rendering with Unsloth's template; the other 5 (the reply with a reasoning block, in each
mode) equal it with the reasoning passed as `reasoning_content` (`hf_reasoning_content`; runq's intended form: an
assistant text's `</think>` splits it into reasoning and answer, the CLI's replies are so; HF given the text as
`content` alone puts an empty think block in front of it); 28 also equal Qwen's own template's (`hf_qwen`; it raises
for `developer`, several system messages and `high`). `golden/tokenizer_qwen35.json` (`make_golden.py --tokenizer-qwen35`): `tokenizer.json`'s 81 cases plus 9
with combining marks, ids from the Qwen3.5 tokenizer (23 of the 90 pre-tokenize differently from Qwen3's regex).
`golden/models/` (Step 11, ~1.1 MB): per file tag the logits goldens of `en`, `code`, `chat` (every position) and
`long2` (positions 248, 256, .., 296) with a 1,024-id vocab subset: `1.7b-hf` (HF fp32 Qwen3-1.7B,
`make_golden.py --model-dir`), the others the file's dequantized-weights reference (`make_model_ref.py`:
`tests/ref_forward.py`, fp32 numpy, layer by layer, ~1 GB peak for 8B; checked on 0.6B Q4_K_M against
`golden/logits_q4`: cos 1.0, max |diff| 0.008 = the goldens' float16 rounding); `greedy/<tag>.txt` the
greedy texts (`--models --update-models`). Also `3.5-0.8b-hf` (HF fp32, transformers 5.21 in `.venv-hf5`) and
`3.5-0.8b-q8` (`tests/ref_qwen35.py` on the Q8_0 file), for Qwen3.5-0.8B ([docs/qwen35.md](../docs/qwen35.md)): the Q8_0
reference vs HF has 1-cos mean 0.9-2.8e-4, min cos 0.9991, top-1 100%, so the `q8_hf` and `ref` thresholds apply as
for Qwen3. Qwen3.5-4B (Step 14): `3.5-4b-q8` (the Q8_0 file's reference) and `3.5-4b-hf`: **not transformers** (its fp32
forward needs ~17 GB) but `tests/ref_qwen35.py` on the HF bf16 safetensors (`make_model_ref.py 3.5-4b-hf --hf
models/Qwen3.5-4B`, layer by layer, pread, 1.3 GB peak, 30 s), in HF's value head order (head i uses key head i // 2); that
reference reproduces transformers 5.21 fp32 on 0.8B (top logits within 2-4e-5) and on tiny random-init `qwen3_5` models
with 2 / 3 value heads per key head (1e-4; docs/qwen35.md). The two 4B references agree to cos >= 0.999 per position
(subset), top-1 100%. Qwen3.8-27B (TODO S7): `3.8-27b-q4km`, `tests/ref_qwen35.py` on the Q4_K_M file (`make_model_ref.py
3.8-27b-q4km FILE` with `.venv-hf5`: 214 s at 8 BLAS threads, ~2 GB of anonymous memory, the MTP block skipped; no
HF golden: its bf16 weights are 54 GB). Thresholds vs HF fp32 for 1.7B (measured, golden vs golden): the Q8_0
reference is within 0.9985-0.9998 (mean per prompt, min 0.9955, pos0 0.994), so `q8_hf` = the 0.6B Q8_0
thresholds; the Q4_K_M reference has cos mean 0.890 on `en` (position 8 at -0.137: the whole logit vector shifted,
top-1 still right), 0.977-0.997 on the others, pos0 0.95, top-1 56/61, pooled 0.967: so `q4_hf` has no per-position
bound, cos mean >= 0.85, pos0 >= 0.90, top-1 >= 0.75, pooled >= 0.95 (runq: 0.928 / 0.995-0.998, pooled 0.979).
`golden/long/` (TODO G2, ~35 / ~20 KB per file; `make_long_ref.py TAG FILE [-n N]`): the long-context reference of the
`long` group, `ref_qwen35.forward_long` on the file's dequantized weights (rows in chunks of 2,048, full attention as
all K / V then blocks of 128 queries with an exact softmax; ~1-3.5 GB): `n`, the sha256 of the ids (`ids_hash`), the
corpus commit, `positions`, per position the top 64 ids / logits and 512 fixed ids' logits (fp32), the max |K| / |V|
per attention layer (`kv_max`) and the run's seconds. `3.5-0.8b-q8`: 98,304 tokens, 1,813 s at 8 BLAS threads;
`3.5-4b-q8`: 32,768 tokens, 1,233 s (the 4B at 98k would need ~7.7 GB in runq alone: more than this box's share).

Engine bugs found by these tests so far (the missing NFC normalizer, the SwiGLU NaN, a division by a zero
scale in `quantize()`) are fixed; open problems go to [`../TODO.md`](../TODO.md).

## Tokenizer fuzz test (on demand, not in `make test`)

`.venv/bin/python tests/fuzz_tokenizer.py [-n 5000] [--seed 1] [--jobs 8] [--show 5] [--no-files]`
compares `runq -m tokenize` with the HF tokenizer on `-n` random strings (a mix of the character
classes of the goldens, added tokens and broken ones, random code points from all planes, and
for the normalizer: marks in random order, Hangul jamo runs, NFD/NFKD words, NFC_QC=No characters,
long mark runs; about half the strings are not NFC) plus real text (repo docs/sources, the HF
model README and LICENSE, split at 64 KiB for argv), as is and in NFD. Strings contain no NUL.
Besides the ids it checks the normalized text directly: the concatenated token bytes must equal
HF's NFC normalizer output. It prints the mismatch rate, minimizes the first `--show` mismatches
(delta debugging) and exits 1 on any.
Result (Step 5, 2026-10-25): seeds 1 and 2, 5,028 cases each (~2,400 not NFC, 1.25M HF tokens):
0 mismatches, normalized text identical. GGUF tokenizer (Step 6a, `--model <gguf>`; `--vocab` = the
.bin whose `.tokenizer` gives the token bytes): seeds 1 and 2, 5,030 cases each, 1.28M HF tokens: 0
mismatches; `--nfc-all`: 0 differences. Sanity checks: a scratch build with `\s+(?!\S)` disabled
gives 27% mismatches, minimized to `'  k'` and `'\t\t4'`; one ignoring NFC blocking 12%, minimized
to `'a\u0367\u0308'`.

`--nfc-all` checks the normalizer alone (tokens -> bytes) against HF's NFC normalizer and Python's
`unicodedata.normalize('NFC')`: every code point alone (1,112,062, all but surrogates, NUL and
`\n`), all 850,084 ordered pairs of combining marks after a base, 11,322 composition cases (NFD of
every decomposable character, plus marks; every composition pair, with a mark in between), all
13,167 Hangul L+V / LV+T / L+V+T cases, and 20 x `-n` random strings; joined with `\n` into 64 KiB
arguments. Result: 0 differences from HF; from `unicodedata` only on strings with a code point of
the HF compatibility overrides (64,176 mark pairs, 5 compositions, ~2.5% of the random strings).
HF tokenizers 0.21.4 normalizes with Unicode 9.0 data (`unicode-normalization-alignments` 0.1.12):
108 combining marks added later are ccc 0 for it and U+11938 does not decompose
(`tools/gen_unicode_tables.py --check-hf` verifies this list); `src/unicode_nfc.h` applies the same
overrides, and a scratch build with the pure 15.1 tables (`--nfc --unicode-only`) gives exactly the
opposite (0 differences from `unicodedata`, the same counts from HF).

Invalid UTF-8 has no HF reference (a Python str cannot hold it). runq keeps every invalid byte
(byte-level fallback, as before) and its normalizer does not reorder or compose across one (each
acts like U+FFFD, a starter composing with nothing); `tokenizer/nfc_invalid_utf8` tests that.

## Server tests under sanitizers (on demand)

`tests/server_test.py --slow` relaxes the latency bounds for slow builds. Step 9 (2026-10-26): ASan + UBSan
(`-O1 -g -fsanitize=address,undefined -fopenmp -march=native`, all sources compiled together, AVX-512 and
`-DRUNQ_GENERIC`) on the Q8_0 GGUF, the AVX-512 build also on the .bin and Q4_K_M: 48 / 48 each, no
reports (incl. the leak check at the server's exit after SIGTERM); ThreadSanitizer (`-O1 -g
-fsanitize=thread`, no `-fopenmp`: the thread pool, the engine thread and the connection threads all
instrumented): 48 / 48, no reports (after making the shutdown flag atomic: it found the signal handler's
write racing the connection threads' reads). The ASan builds take ~40 s (AVX-512) and ~6 min (generic), TSan ~4 min.
Step 10b (continuous batching; the same builds, reports to files with `ASAN_OPTIONS` / `UBSAN_OPTIONS` /
`TSAN_OPTIONS` `log_path=...`, so nothing in the server's log can be missed): ASan + UBSan on the Q8_0 GGUF,
Q4_K_M and the .bin, TSan on the Q8_0 GGUF: 52-53 / 52-53 each incl. the `batch` group, and
`scripts/bench_serve.py` against both builds for more concurrency than the test has (ASan: 8 and 24 clients at
`--chunk 16 --slots 18`, a cold judge burst of 16 (prefix waits and copies), the mixed load with long prompts;
TSan: 12 clients at `--parallel 8 --queue 8`, a cold burst of 8): no reports, no leaks at exit (TSan's
`server_test.py --slow` takes ~5.5 min).

## Memory

The load-time estimate (README, Memory; `src/model.c`, `model_mem`) against the measured memory, with
`build/mem_probe MODEL CTX [--cold] [--page-cache auto|drop|keep]` (`make mem-probe`, `tests/mem_probe.c`): it loads
the model, then fills a sequence of CTX positions (Qwen3.5: with its checkpoints taken), and prints the estimate's
parts, the load's peak RSS and the peak RSS with the context full. 2026-10-27, 8 threads, CTX 2048:

| file | weights + runtime + sequence (estimate, MiB) | load peak RSS | peak RSS, context full | estimate vs it |
|---|---|---|---|---|
| Qwen3-0.6B Q8_0 | 639.7 + 60.8 + 224.0 = 924.5 | 663 MiB | 939.7 MiB | -1.6% |
| Qwen3-0.6B Q4_K_M | 379.9 + 62.8 + 224.0 = 666.7 | 403 | 676.6 | -1.5% |
| Qwen3-0.6B `.bin` | 604.1 + 53.8 + 224.0 = 881.9 | 615 | 892.4 | -1.2% |
| Qwen3.5-0.8B Q8_0 | 808.6 + 97.8 + 81.8 (KV 24.0, state 19.3 x 3) = 988.2 | 842 | 977.1 | +1.1% |
| Qwen3.5-0.8B Q4_K_M | 506.4 + 100.0 + 81.8 = 688.2 | 538 | 671.1 | +2.5% |
| Qwen3-8B Q4_K_M | 4,556.2 + 126.3 + 288.0 = 4,970.5 | 4,583 | 5,008.9 | -0.8% |
| Qwen3.5-4B Q8_0 | 4,514.9 + 149.8 + 214.8 (KV 64.0, state 50.2 x 3) = 4,879.5 | 4,551 | 4,874.7 | +0.1% |
| Qwen3.8-27B Q4_K_M (CTX 4096, TODO S7) | 15,686.7 + 230.2 + 704.9 (KV 256.0, state 149.6 x 3) = 16,621.7 | 15,736 | 16,676.2 | -0.3% |

The runtime part counts a compute context of 512 rows at CTX positions, which a load alone does not allocate yet
(weights + runtime is 2-13% above the load's peak). At 1,024 and 8,192 positions the 0.6B / 0.8B files gave -1.8% to
+2.6%. The KV cache and the compute context are address space touched as used; with transparent huge pages
(`always` here) a layer's KV cache grows by 2 MB at a time, so a short run already has ~112 MB of it for the 0.6B.

The 27B (a cold load of 19.6 s): 32 MiB of its 17.1 GB file left in the page cache after the load (`auto` dropped
it); filling its 4,096 positions takes ~100 s (prefill at ~40 tok/s).

The refusal comes before any weight is read: `runq Qwen3-8B-Q4_K_M.gguf --mem 0.1 -i hi` with the file's page
cache dropped first exits in 23 ms with a peak RSS of 7.5 MiB, and 5.8 MB of the 5.0 GB file in the page cache
(`fincore`: the header and metadata only).

**Load baseline** (the 8B Q4_K_M and the 4B Q8_0; `runq MODEL -m generate -i Hi -n 1 -t 0`, 8 threads, wall time of
the whole process and its peak RSS (`getrusage`), the file's page cache before and after (`fincore`); cold: the
file's cache dropped before; under `flock /tmp/cwayn-bench.lock`, 2026-10-27; 979d5b1 = before this change):

| file | build | cold load | warm load | file in the page cache after |
|---|---|---|---|---|
| Qwen3-8B Q4_K_M (5.03 GB) | 979d5b1 | 5.86 s, 4,594 MiB | 1.17-1.19 s, 4,668-4,675 MiB | 4.68 GB cold (the untied `token_embd` is not read), 5.03 warm |
| | now (`auto`: kept) | 5.82 s, 4,598 MiB | 1.03-1.05 s, 4,585 MiB | the same |
| | now, `--page-cache drop` | 6.17 s, 4,596 MiB | 1.66-1.79 s, 4,666-4,749 MiB | 0.01 GB (`Cached` in /proc/meminfo -4.7 GB) |
| Qwen3.5-4B Q8_0 (4.48 GB) | 979d5b1 | 5.16 s, 4,617 MiB | 1.12-1.23 s, 4,603-4,613 MiB | 4.48 GB |
| | now (`auto`: kept) | 5.39 s, 4,605 MiB | 1.17-1.20 s, 4,603-4,652 MiB | 4.48 GB |
| | now, `--page-cache drop` | 5.49 s, 4,606 MiB | 1.32-1.37 s, 4,603-4,605 MiB | 0.02 GB (`Cached` -4.5 GB) |

The peak RSS varies by up to ~160 MiB between identical runs (both builds). Dropping costs ~0.4 s per 5 GB (the
kernel freeing 1.2 M pages) and makes the next load cold (~6 s instead of ~1 s): `auto` drops only files of at least
1/4 of the RAM (8 GB here: the 27B). Dropping per 64 KB work item instead of per matrix cost +1.1 s on the 8B's
cold load (each `posix_fadvise` call drains the LRU caches of every CPU).

## Comparing two builds (`tests/compare_bins.sh`, on demand)

`tests/compare_bins.sh [-j JOBS] [-t THREADS] [--quick] [--q35-4b] [--no-api] REF_DIR NEW_DIR` runs the same
command lines with `REF_DIR/runq`, `REF_DIR/runq-generic` and the two of `NEW_DIR` and compares the outputs byte for
byte: the check for refactorings, where nothing may change. 792 jobs, per model family (a family whose file is
missing is skipped, with a note in the summary):
- Qwen3 (`bin`, `q8`, `q4`: Qwen3-0.6B .bin, Q8_0, Q4_K_M; 118 jobs each): `-m logits` files for both builds x KV
  f16 / bf16 / f32 x `-b 1 / 7 / 64 / 512` x a 5-token and a 280-token prompt (past one attention block); greedy
  generate (short / long prompt, `-b 1 / 7`, default `-c`, a prompt longer than `-c`, `-n 1`); seeded sampling
  (`-t 0.8 -s 7` generate and chat, `-t 1.5 -p 0.95`, `-b 1`, multi-turn); multi-turn chat on stdin (a system prompt,
  `-r 1`, context full at `-c 160`, a reply cut at `-c 48`, `-b 1 / 7`, a message too long), with the
  `RUNQ_CHAT_DUMP` files; `-m tokenize`; `-m dequant`; error cases (prompt too long for `-m logits`, empty prompt,
  bad `-k`, unknown mode, missing file, usage).
- Qwen3.5 (`q35-q8`, `q35-q4`: Qwen3.5-0.8B Q8_0, Q4_K_M; `synth-r2`, `synth-r3`: the synthetic models with 2 / 3
  value heads per key head; 108 jobs each): the `-m logits` grid (both builds x KV f16 / bf16 x 3 / 8 threads x
  `-b 1 / 7 / 64 / 512` x the two prompts); greedy and seeded generate; `-r 1` chats whose next turn rolls the
  recurrent state back to the checkpoint at the previous assistant header (greedy, `-b 7`, bf16 KV, seeded with a
  system prompt, context full at `-c 160`) and `-r 0` ones, with the `RUNQ_CHAT_DUMP` files; `api_test --digest`
  and `api_test-generic --digest-small` (the batched passes' logits hashes, section 9 of `api_test.c`).
  The synthetic models are data, not code: `tests/make_synth_q35.py` of the script's own tree builds them (cached in
  `build/synth-q35`), the same files serve both sides, and their hashes are part of the `CMP_CACHE` key.
- `--q35-4b`: 7 jobs on Qwen3.5-4B Q8_0 (4.5 GB: one at a time, 8 threads; AVX-512 build only).
- The `api_test` jobs need `api_test` and `api_test-generic` in both directories, as `DIR/api_test` or
  `DIR/build/api_test` (`make api-test`; for a copy of a commit's binaries, copy them next to its `runq`); else they
  are skipped with a note. `--no-api` leaves them out.

It compares stdout, stderr (the numbers of `[timing]` / `[tokenize]` lines masked), the exit status and the output
files (logits files, dumps: hashed and deleted at once; a Qwen3.5 logits file of the long prompt is 277 MB, so up to
`-j` of them sit in `$TMPDIR` at a time). It prints a summary per family (jobs, identical / how many differ), then
the differing jobs' hashes. `CMP_CACHE=DIR` keeps the reference results (keyed by the binaries, the synthetic models
and the job list), so later comparisons against the same reference only run the new binaries: ~25 min without the
cache, ~7 with it (`-j 3`, 4 threads per job; the generic build's Qwen3.5 logits jobs are the bulk); `--quick` (the
AVX-512 builds, `-b 1 / 64`, 6 jobs + one `--digest-small` per Qwen3.5 family: 172 jobs) ~1.5 min.
Step 8 used it against copies of the binaries before the refactoring (all identical). Checked when the Qwen3.5
families were added: main against itself identical (full and `--quick`); a build with one DeltaNet gate one ulp off
(`dn_gates`' exp(g)) differs in 352 Qwen3.5 jobs (every logits file, dump and digest; greedy text unchanged) and no
Qwen3 job, `--quick` too (25 of 28); one with rmsnorm's scale one ulp off differs in every family (614 jobs).
