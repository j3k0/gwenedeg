# llama.cpp (mainline `225a534`) tuning on Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf, 2026-11-22

`llama-bench`, `-fa on -ctk q8_0 -ctv q4_0` unless noted, one or a few repetitions on the shared VM (1-min load in
brackets: start / end of the run; ~14 of it is the benchmark's own threads). llama-bench counts the MTP block in the
model's size (11.28 GiB, 27.32 B); ik_llama.cpp does not (10.95 GiB, 26.90 B): the same file.

## Threads x ubatch (tune1: pp2048 / tg64, 2 reps, 05:47-06:05Z, load 3.7 / 20.7)

| threads | -ub | pp2048 | tg64 |
|---|---|---|---|
| 14 | 512 | 50.4 | 4.22 |
| 14 | 1024 | 50.3 | 4.74 |
| 14 | 2048 | 51.4 | 4.81 |
| 16 | 512 | 42.8 | 1.70 |
| 16 | 1024 | 49.6 | 3.67 |
| 16 | 2048 | 49.7 | 0.96 |

14 threads (16 collide with the other services' threads: tg drops to 1-4 tok/s). `-ub` 512 / 1024 / 2048 equal within
noise: `-ub 512` (the default, the smallest compute buffer at 98k context).

## Weight repacking, load mode (tune2: 14 threads, -ub 2048, 2 reps, load 17.9 / 15.2; tune3: -ub 512, 3 reps, load 13.3 / 17.7)

| run | load mode | --repack | pp2048 | tg64 |
|---|---|---|---|---|
| tune2 | mmap | 0 | 55.3 | 5.00 |
| tune2 | mmap | 1 | 51.9 | 4.88 |
| tune3 | mmap | 0 | 48.9 | 3.77 |
| tune3 | mmap | 1 | 48.2 | 3.96 |
| tune3 | none (no mmap) | 0 | 52.3 | 4.92 |
| tune3 | none (no mmap) | 1 | 47.5 | 4.23 |

Repacking is a no-op on this file: mainline's CPU repack covers Q4_0, Q4_K, Q2_K, Q5_K, Q6_K, Q8_0, Q1_0, IQ4_NL and
MXFP4 (`ggml/src/ggml-cpu/repack.cpp`), and the file holds IQ3_S (144 tensors), IQ3_XXS (78), IQ4_XS (96), IQ2_S / XS /
XXS, IQ1_M, BF16 and F32 only; the differences above are noise. `--no-repack` (no effect, kept for the record). No mmap was the best cell: `-lm none` (the weights read into anonymous memory; peak RSS 14-16
GiB in the server runs). `--mlock` not tried (no swap in the scope: nothing to lock against).

## MTP draft length (server, one request: bench/s9c lcb1, thinking on, greedy, 256 tokens; 07:31-07:43Z, load 13.9-17.7)

Base: `llama-server -m {model} -c 98304 -np 1 -t 14 -fa on -ctk q8_0 -ctv q4_0 -ub 512 -b 2048 -lm none --no-repack
--ctx-checkpoints 4 -cram 512 --jinja`, plus:

| setting | prompt tok/s | decode tok/s | drafted | accepted |
|---|---|---|---|---|
| no MTP | 57.3 | **4.46** | - | - |
| `--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-p-min 0.5` | 57.8 | 4.11 | 117 | 106 |
| n_max 2, p_min 0.5 | 48.6 | 3.59 | 163 | 131 |
| n_max 3, p_min 0.5 | 50.0 | 3.30 | 198 | 129 |
| n_max 3, p_min 0 (`--spec-type draft-mtp` alone: the defaults) | 58.0 | 3.39 | 300 | 155 |
| n_max 4, p_min 0.5 | 52.1 | 3.90 | 193 | 150 |
| n_max 6, p_min 0.5 (runq's / ik's setting) | 59.1 | 3.34 | 275 | 156 |

Mainline runs this file's MTP block (`draft-mtp`, also with several slots), and the drafts are accepted (57-91%), but
every draft length decodes slower than no drafting: a verification pass of k+1 tokens costs about as much as k+1
single-token passes here (e.g. n_max 6 in an aborted first `--short` run: mean accepted length 2.4-2.9, 3.6-4.0
tok/s against 4.5 plain). So the runs use **no MTP**.

## Cache type at depth (see depth.md)

Any quantized KV cache makes prefill and decode at depth much slower than f16 (pp512 at an 8k depth: q8_0/q4_0 14.3,
q8_0/q8_0 17.6, f16/f16 62.7 tok/s): the equal-bit-width runs use q8_0/q4_0 as the protocol says, and the 32k session was
run a second time with f16/f16 (`-c 49152`) as mainline's better configuration.

## Chosen

`llama-server -m {model} -c 98304 -np 1|4 -t 14 -fa on -ctk q8_0 -ctv q4_0 -ub 512 -b 2048 -lm none --no-repack
--ctx-checkpoints 4|2 -cram 512 --jinja` (`commands.md`).
