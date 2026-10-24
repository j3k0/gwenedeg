#!/usr/bin/env bash
# Benchmark third_party/mynah-slm decode/prefill speed across thread counts.
# Usage: scripts/bench_mynah.sh [model.gguf] [threads...]
#   default model: models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf, threads: 1 2 4 8 12 16
# Env: N=new tokens per run (256), REPS=runs per thread count, best kept (3),
#      EXTRA="--fast" etc. extra flags passed to `mynah-slm run`
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/third_party/mynah-slm/mynah-slm"
MODEL="${1:-$ROOT/models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf}"; shift || true
THREADS=("${@:-1 2 4 8 12 16}"); THREADS=(${THREADS[@]})
N="${N:-256}"; REPS="${REPS:-3}"; EXTRA="${EXTRA:-}"
PROMPT="Write a detailed, multi-paragraph essay about the history of the printing press and its impact on Europe."

[ -x "$BIN" ] || "$ROOT/scripts/fetch_mynah.sh" >/dev/null
cat "$MODEL" > /dev/null   # warm the page cache

printf "%-8s %12s %12s\n" threads "prefill t/s" "decode t/s"
for t in "${THREADS[@]}"; do
  best_d=0; best_p=0
  for _ in $(seq "$REPS"); do
    # shellcheck disable=SC2086
    line=$("$BIN" run -m "$MODEL" -p "$PROMPT" --temp 0 -n "$N" -t "$t" --no-stream --think off $EXTRA 2>&1 >/dev/null | grep 'decode')
    p=$(sed -E 's/.*prefill ([0-9.]+) tok\/s.*/\1/' <<<"$line")
    d=$(sed -E 's/.*decode ([0-9.]+) tok\/s.*/\1/' <<<"$line")
    best_d=$(awk -v a="$best_d" -v b="$d" 'BEGIN{print (b>a)?b:a}')
    best_p=$(awk -v a="$best_p" -v b="$p" 'BEGIN{print (b>a)?b:a}')
  done
  printf "%-8s %12s %12s\n" "$t" "$best_p" "$best_d"
done
