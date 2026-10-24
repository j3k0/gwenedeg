# cwayn

Personal experiment: running Qwen3-0.6B from plain C on the CPU.
The engine is `src/runq.c`, derived from qwen3.c. Roadmap and status: [TODO.md](TODO.md).

## Quick start

```sh
scripts/download_models.py            # Qwen/Qwen3-0.6B (bf16) + Q4_K_M GGUF, pinned + checksummed
scripts/setup_venv.sh                 # .venv with CPU torch, for the exporter only
scripts/export_q8.sh                  # -> models/Qwen3-0.6B-q8.bin (Q8_0, ~630 MB)
# (or: make models, which runs download + export if the .bin is missing)
make                                  # -> ./runq  (make debug -> ./runq-debug)

OMP_NUM_THREADS=8 ./runq models/Qwen3-0.6B-q8.bin -c 4096   # interactive chat
OMP_NUM_THREADS=8 ./runq models/Qwen3-0.6B-q8.bin -c 4096 -t 0 -n 200 -i "Why is the sky blue?"
make test                             # regression suite, ~15 s (tests/README.md)
make bench                            # thread sweep, prefill/decode tok/s (scripts/bench_qwen3c.sh)
scripts/fetch_mynah.sh                # optional reference engine, then scripts/bench_mynah.sh
```

Use `OMP_NUM_THREADS=8` (see benchmarks below). Always pass `-c` (context size): without it the KV cache is sized for the full
40,960-token context, about 9 GB of address space.

## Models (`scripts/download_models.py --list`)

| set | source | file(s) |
|---|---|---|
| `hf` | `Qwen/Qwen3-0.6B` @ `c1899de` | bf16 safetensors + tokenizer (input for the Q8_0 export) |
| `q4km` | `unsloth/Qwen3-0.6B-GGUF` @ `50968a4` | `Qwen3-0.6B-Q4_K_M.gguf` (imatrix) |
| `q8gguf` | `unsloth/Qwen3-0.6B-GGUF` @ `50968a4` | `Qwen3-0.6B-Q8_0.gguf` |

Revisions are pinned. Each file is verified by sha256 (LFS files) or git blob sha1 (small files).

## Layout and third-party code

- `src/runq.c`: our engine, from qwen3.c's `runq.c` (MIT, see `LICENSE`). Changes so far:
  `-n <int>` max new tokens, plus a `[timing]` line on stderr (prefill and decode tok/s);
  no classifier matmul for prompt tokens whose logits are unused;
  test hooks `-m tokenize` and `-m logits -o <file>` (see `tests/README.md`).
- `third_party/qwen3.c/`: pristine https://github.com/adriancable/qwen3.c @ `55643ff`
  (2025-07-05), kept for diffs (`diff -u third_party/qwen3.c/runq.c src/runq.c`). Only its
  `export.py`/`model.py` are used, by `scripts/export_q8.sh`. qwen3.c itself started from
  Andrej Karpathy's llama2.c.
- `third_party/mynah-slm/`: not committed. `scripts/fetch_mynah.sh` clones
  https://github.com/mynah-org/mynah-slm @ `cde1599` (2026-09-22) and builds it (needs
  `libopenblas-dev`). Reference and benchmarks only.

## Benchmarks (2026-10-24, 16-vCPU EPYC Genoa VM, greedy, 256 new tokens, best of 3)

Decode tok/s (prefill similar):

| threads | qwen3.c Q8_0 | mynah-slm Q8_0 | mynah-slm Q4_K_M |
|---|---|---|---|
| 1 | 21.5 | 17.3 | 14.3 |
| 4 | 68 | 32.5 | 30.0 |
| 8 | **~100** | 35.5 | 34.4 |
| 12 | ~95 | 34.8 | 34.5 |
| 16 | **~11** (collapse) | 30.3 | 30.4 |

- Since TODO Step 1 (no classifier for prompt tokens), prefill is ahead of decode: at 8 threads
  ~118 tok/s on a 443-token prompt (was ~99), ~121 on the short bench prompt (was ~92).
- qwen3.c at 16 threads collapses. This is not VM throttling (0% steal time), but OpenMP
  spin-barrier oversubscription: with every vCPU spinning, any other process preempting one
  thread stalls all of them at every barrier (hundreds per token). `OMP_WAIT_POLICY=passive`
  is worse at every thread count (~20 tok/s). Use 8 threads, which also leaves headroom.
- mynah-slm `--fast` (int8 activations) made no measurable difference here.
- At about 100 tok/s, qwen3.c reads ~63 GB/s, below the measured ~87 GB/s at 8 threads,
  so it isn't purely bandwidth-bound yet.
- With mynah-slm, Q4_K_M is no faster than Q8_0 on this CPU.
- Greedy Q8_0 output is identical from qwen3.c and mynah-slm (cross-check of both engines).

## Known limitations of qwen3.c

- Chat mode keeps no history: each user turn restarts at position 0.
- Q8_0 only; there is no GGUF loader.
