#!/usr/bin/env bash
# Benchmark ./runq (src/runq.c) decode/prefill speed across thread counts.
# Usage: scripts/bench_qwen3c.sh [model.bin] [threads...]
#   default model: models/Qwen3-0.6B-q8.bin, default threads: 1 2 4 8 12
#   (16 threads collapses: OpenMP spin-barrier oversubscription, see README)
# Env: N=new tokens per run (256), REPS=runs per thread count, best kept (3), RUNQ=binary
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNQ="${RUNQ:-$ROOT/runq}"
MODEL="${1:-$ROOT/models/Qwen3-0.6B-q8.bin}"; shift || true
THREADS=("${@:-1 2 4 8 12}"); THREADS=(${THREADS[@]})
N="${N:-256}"; REPS="${REPS:-3}"
PROMPT="Write a detailed, multi-paragraph essay about the history of the printing press and its impact on Europe."

[ -x "$RUNQ" ] || make -C "$ROOT" >/dev/null
cat "$MODEL" > /dev/null   # warm the page cache

printf "%-8s %12s %12s\n" threads "prefill t/s" "decode t/s"
for t in "${THREADS[@]}"; do
  best_d=0; best_p=0
  for _ in $(seq "$REPS"); do
    line=$(OMP_NUM_THREADS=$t "$RUNQ" "$MODEL" -c 4096 -t 0 -n "$N" -i "$PROMPT" 2>&1 >/dev/null | grep '\[timing\]')
    p=$(sed -E 's/.*prompt: [0-9]+ tok in [0-9.]+ s \(([0-9.]+) tok\/s\).*/\1/' <<<"$line")
    d=$(sed -E 's/.*generated: [0-9]+ tok in [0-9.]+ s \(([0-9.]+) tok\/s\).*/\1/' <<<"$line")
    best_d=$(awk -v a="$best_d" -v b="$d" 'BEGIN{print (b>a)?b:a}')
    best_p=$(awk -v a="$best_p" -v b="$p" 'BEGIN{print (b>a)?b:a}')
  done
  printf "%-8s %12s %12s\n" "$t" "$best_p" "$best_d"
done
