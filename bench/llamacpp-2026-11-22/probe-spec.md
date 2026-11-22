base: llama-server -m {model} -c 98304 -np 1 -t 14 -fa on -ctk q8_0 -ctv q4_0 -ub 512 -b 2048 -lm none --no-repack --ctx-checkpoints 4 -cram 512 --jinja --host 127.0.0.1 --port 18733

| setting | prompt tok/s | decode tok/s | drafted | accepted | 1-min load |
|---|---|---|---|---|---|
| no MTP | 57.3 | 4.46 | - | - | 13.86 |
| mtp n_max 1 | 57.8 | 4.11 | 117 | 106 | 15.77 |
| mtp n_max 2 | 48.6 | 3.59 | 163 | 131 | 17.66 |
| mtp n_max 3 | 50.0 | 3.30 | 198 | 129 | 17.13 |
| mtp n_max 3 p_min 0 (defaults) | 58.0 | 3.39 | 300 | 155 | 16.20 |
| mtp n_max 4 | 52.1 | 3.90 | 193 | 150 | 16.29 |
| mtp n_max 6 | 59.1 | 3.34 | 275 | 156 | 16.67 |
