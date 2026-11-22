# llama.cpp `225a534`: KV cache type vs depth, 2026-11-22

`llama-bench -m {model} -fa on -t 14 -ub 512 -lm none --repack 0 -p 512 -n 32 -d 0,8192 -r 1`, one repetition each,
11:46-11:58Z, 1-min load 12.6-15.4 (mostly the benchmark's own 14 threads).

| -ctk / -ctv | pp512 | tg32 | pp512 @ d8192 | tg32 @ d8192 |
|---|---|---|---|---|
| q8_0 / q4_0 (the equal-bit-width setting) | 71.1 | 5.17 | **14.3** | 3.40 |
| q8_0 / q8_0 | | | 17.6 | 3.25 |
| f16 / f16 | 78.2 | 4.49 | **62.7** | 4.48 |

With a quantized cache the CPU flash-attention path slows down steeply with depth: prefill at an 8k depth is 4.4x
slower than with f16, and in the 32k server session (`lc1-long.json`) the 2,048-token batches fell from 45 tok/s (the
first 2,090 tokens) to 3.0 tok/s (28.7k to 30.8k) (5,057 s for the 31,755-token prompt). Hence a second `--long` run with an f16 cache
(`lc1f16-long.json`, `-c 49152`: f16 K+V for 98k positions would add ~6 GiB, too close to the 22 GB cap).
