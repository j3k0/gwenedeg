# Commands, 2026-11-22

Every measurement ran under `run.sh LABEL CMD...`: `flock /tmp/cwayn-bench.lock systemd-run --user --scope -p
MemoryMax=22G -p MemorySwapMax=0 CMD`, the 1-min load average and MemAvailable logged at its start and end
(`runs.log`; the load every minute over the day: `loadavg.txt`). One 27B process at a time; the daytime runq server
was stopped for the day (from 05:41Z). Paths below are relative to the repository / the build trees.

## Builds

- llama.cpp: `git clone https://github.com/ggml-org/llama.cpp` (master `225a5345bbb0`, 2026-11-22T03:25Z, "server :
  fit non-causal batches in one ubatch (#32713)"), `cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
  -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF` (cmake 4.4.5, gcc 14.2.0: `-march=native`), `cmake --build build -j14 --target
  llama-server llama-bench llama-perplexity llama-cli`. `ggml_cpu_has_avx512 / _vnni / _vbmi / _bf16` all 1 in the
  build (checked with a 5-line program linked against `libggml-cpu`).
- ik_llama.cpp: the build of the 2026-11-21 section, `ac8d544b` (2026-11-19), used read-only.
- runq: this branch at `50ba538` (main, with the i-quant tiled GEMM), `make` (gcc 14.2.0, `-Ofast -fopenmp
  -march=native`).

## Server runs (`scripts/s9c_bench.py`, unchanged)

```
# runq (OMP_NUM_THREADS=14), the production arguments
scripts/s9c_bench.py --model MODEL --serve-args "--ctx 98304 --draft mtp -G 6 --tau 0.5 --repack -k q4-dot --slots 6 \
  --parallel 4 --admit bound --kv-disk TMPDIR --kv-disk-max 10 --kv-disk-free 14 -v" --short --think 1,0 --label runq-short
scripts/s9c_bench.py ... (the same) --long 32768 --turns 3 --label runq-long      # turns 0-3; 0-2 compared

# llama.cpp, 1 slot (lc1.cmd) / 4 slots (lc4.cmd), q8_0 K / q4_0 V
scripts/s9c_bench.py --model MODEL --server-cmd "$(cat lc1.cmd)" --short --think 1,0 --label lc1-short
scripts/s9c_bench.py --model MODEL --server-cmd "$(cat lc4.cmd)" --short --think 1,0 --label lc4-short
# the same two with f16 K / V (lc1f16s.cmd, lc4f16s.cmd: -ctk f16 -ctv f16, all else equal, -c 98304), 13:13-14:27Z
scripts/s9c_bench.py --model MODEL --server-cmd "$(cat lc1f16s.cmd)" --short --think 1,0 --label lc1f16-short
scripts/s9c_bench.py --model MODEL --server-cmd "$(cat lc4f16s.cmd)" --short --think 1,0 --label lc4f16-short
scripts/s9c_bench.py --model MODEL --server-cmd "$(cat lc1.cmd)" --long 32768 --turns 2 --label lc1-long
# llama.cpp, f16 K / V (lc1f16.cmd, -c 49152), the 32k session only
scripts/s9c_bench.py --model MODEL --server-cmd "$(cat lc1f16.cmd)" --long 32768 --turns 2 --label lc1f16-long

# ik_llama.cpp, 1 slot + MTP (ik1.cmd, as on 2026-11-21), re-run the same day
scripts/s9c_bench.py --model MODEL --server-cmd "$(cat ik1.cmd)" --short --think 1,0 --label ik1-short
scripts/s9c_bench.py --model MODEL --server-cmd "$(cat ik1-long.cmd)" --long 32768 --turns 2 --label ik1-long   # -c 49152
```

The JSONs here keep every 10th of the script's 1 s memory samples (`samples`; the peaks in `peak` are the full
run's); everything else is as written. The servers' own logs (with the request bodies) are not kept: the `.out`
files are the script's per-request summaries.

`lc1-short-aborted.*`: the first llama.cpp `--short` run with MTP (`--spec-type draft-mtp --spec-draft-n-max 6
--spec-draft-p-min 0.5`), stopped after 2 requests (3.6-4.0 tok/s; see tune.md, MTP draft length).

## Micro line (pp512 / tg128, 14 threads)

```
llama.cpp/build/bin/llama-bench -m MODEL -fa on -ctk q8_0 -ctv q4_0 -t 14 -p 512 -n 128 -r 2 -v                       # micro512.out
ik_llama.cpp/build/bin/llama-bench -m MODEL -fa 1 -ctk q8_0 -ctv q4_0 -t 14 -ub 1024 -b 2048 -rtr -p 512 -n 128 -r 2 -v  # micro512-ik.out
# runq: the CLI, a 524-token prompt (the first 1,850 bytes of the Beagle text + " Summarize.", chat template), plain decoding
OMP_NUM_THREADS=14 ./runq MODEL -c 1024 -r 0 -t 0 -n 128 -b 512 --repack -k q4-dot -i "PROMPT"            # micro-runq1/2.out
```

## Perplexity (the texts and protocol of the 2026-11-21 section)

```
llama.cpp/build/bin/llama-perplexity -m MODEL -f q_prose.txt -c 2048 -b 2048 -ub 512 -t 14 -fa on -ctk q8_0 -ctv q4_0 -lm none --no-repack
llama.cpp/build/bin/llama-perplexity -m MODEL -f q_code.txt  -c 3072 -b 3072 -ub 512 -t 14 -fa on -ctk q8_0 -ctv q4_0 -lm none --no-repack
```

The number compared is the first chunk's (`[1]...`, the second half of the first window), as for ik_llama.cpp; runq's
from its `-m logits` dump with `ppl_cmp.py` (2026-11-21, `bench/ikllama-2026-11-21/cmp.txt`; not re-run).

## Tuning and depth

`probe_spec.py` (the MTP draft-length probe), `tune1/2/3.out`, `depth-*.out`: tune.md, depth.md. `summ.py`: the
summary numbers of the JSONs (prefill of short prompts = the first three >= 700-token one-client requests: thinking
on, greedy, lcb1 / lcb2 / code; the metric of the 2026-11-21 tables).
