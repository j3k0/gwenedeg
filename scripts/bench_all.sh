#!/usr/bin/env bash
# All runq speed tables of BENCHMARKS.md in one go (~8 min on the 16-vCPU dev box; baselines and
# perplexity are separate: bench_qwen3c.sh with RUNQ=..., bench_mynah.sh, ppl.py).
# Usage: scripts/bench_all.sh [-r REPS]      (default 3: best of 3 per cell, every run listed)
#   1. prefill t/s vs prompt length (450 / 1,500 / 3,000 tokens) and threads (1 4 8 12), -b 512
#   2. chunk-size sweep, 1,500 tokens, 8 threads: -b 1 (token by token) 64 128 256 512
#   3. decode t/s vs threads (1 4 8 12 16): bench_qwen3c.sh (short prompt, 256 new tokens)
#   4. decode t/s after a 1,500 / 3,000-token prompt (up to 256 new tokens), 8 and 12 threads
# /proc/loadavg is printed before each block (it includes our own threads after a run).
# Env: RUNQ=binary (default ./runq).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNQ="${RUNQ:-$ROOT/runq}"; export RUNQ
MODEL="$ROOT/models/Qwen3-0.6B-q8.bin"
REPS=3
while getopts "r:h" o; do
  case $o in r) REPS=$OPTARG ;; *) sed -n '2,11p' "$0"; exit 1 ;; esac
done
[ -x "$RUNQ" ] || make -C "$ROOT" -s runq
load() { echo "[loadavg $(date +%T)] $(cat /proc/loadavg)"; }
echo "runq $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '?'), $(nproc) CPUs, REPS=$REPS"

echo; echo "## 1. prefill vs prompt length and threads (-b 512)"; load
ALL=1 "$ROOT/scripts/bench_prefill.sh" -l "450 1500 3000" -b 512 -t "1 4 8 12" -r "$REPS"

echo; echo "## 2. chunk-size sweep (1,500 tokens, 8 threads)"; load
ALL=1 "$ROOT/scripts/bench_prefill.sh" -l 1500 -b "1 64 128 256 512" -t 8 -r "$REPS"

echo; echo "## 3. decode vs threads (bench_qwen3c.sh, 256 new tokens)"; load
ALL=1 REPS=$REPS "$ROOT/scripts/bench_qwen3c.sh" "$MODEL" 1 4 8 12 16

echo; echo "## 4. decode after a long prompt (-n 256; generated tokens:t/s per run)"; load
for len in 1500 3000; do
  prompt=$("$ROOT/scripts/bench_prefill.sh" -P "$len")
  for t in 8 12; do
    all=""; best=0
    for _ in $(seq "$REPS"); do
      line=$(OMP_NUM_THREADS=$t "$RUNQ" "$MODEL" -c 4096 -t 0 -n 256 -i "$prompt" 2>&1 >/dev/null | grep '\[timing\]')
      g=$(sed -E 's/.*generated: ([0-9]+) tok.*/\1/' <<<"$line")
      d=$(sed -E 's/.*generated: [0-9]+ tok in [0-9.]+ s \(([0-9.]+) tok\/s\).*/\1/' <<<"$line")
      best=$(awk -v a="$best" -v b="$d" 'BEGIN{print (b>a)?b:a}'); all+=" $g:$d"
    done
    printf "prompt %-5s threads %-3s decode t/s %7s  all:%s\n" "$len" "$t" "$best" "$all"
  done
done
echo; load
