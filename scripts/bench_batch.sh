#!/usr/bin/env bash
# Batched decode benchmark over models and thread counts: tests/bench_batch.c (N sequences decoding
# together, one row each per rq_forward_batch pass; aggregate / per-sequence tok/s per context and N).
# Usage: scripts/bench_batch.sh [-b BIN] [-t "8 12 14"] [-M "MODEL ..."] [-p] [bench_batch options ...]
#   -b BIN   the benchmark binary (default build/bench_batch, built with make bench-batch)
#   -t LIST  thread counts (default "8 12 14")
#   -M LIST  models (default: the Q8_0 GGUF and the Q4_K_M GGUF of models/Qwen3-0.6B-GGUF)
#   -p       a -DRUNQ_PROFILE build (built into /tmp/cwayn-prof; per-phase times per pass)
#   the rest goes to bench_batch: -N 1,2,4,8 -L 50,500,2000 -s STEPS -r REPS -k KV -t TEMP -m MAXGB -d
# The box is shared: compare builds by interleaving runs (two -b invocations in a loop), best of N.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BIN=build/bench_batch
THREADS="8 12 14"
MODELS="models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf"
PROF=0
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -b) BIN="$2"; shift 2 ;;
        -t) THREADS="$2"; shift 2 ;;
        -M) MODELS="$2"; shift 2 ;;
        -p) PROF=1; shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
if [[ $PROF == 1 ]]; then
    make -s -j8 B=/tmp/cwayn-prof CFLAGS="-Ofast -fopenmp -march=native -DRUNQ_PROFILE" /tmp/cwayn-prof/bench_batch
    BIN=/tmp/cwayn-prof/bench_batch
elif [[ $BIN == build/bench_batch ]]; then
    make -s -j8 bench-batch
fi
for m in $MODELS; do
    for t in $THREADS; do
        OMP_NUM_THREADS=$t "$BIN" "$m" "${ARGS[@]}"
    done
done
