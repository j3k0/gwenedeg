#!/usr/bin/env bash
# cwayn fast regression suite: tokenizer + logits vs HF (sequential and batched prefill),
# batched vs sequential logits, thread invariance, VNNI vs generic kernels, greedy tripwire.
# Usage: tests/run_tests.sh [--update-greedy] [other tests/check.py options, e.g. -v, --only logits]
# Env:   OMP_NUM_THREADS (default 8), RUNQ (binary to test, default ./runq, built with make),
#        RUNQ_GENERIC (generic-kernel build to compare with, default ./runq-generic when RUNQ is
#        not set; with RUNQ set and RUNQ_GENERIC unset the kernels test is skipped).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="models/Qwen3-0.6B-q8.bin"
if [[ ! -f "$MODEL" ]]; then
    echo "error: $MODEL not found (run: make models)" >&2
    exit 1
fi

if [[ -z "${RUNQ:-}" ]]; then
    make -s -j2 runq runq-generic
    RUNQ="$ROOT/runq"
    RUNQ_GENERIC="${RUNQ_GENERIC:-$ROOT/runq-generic}"
fi

if [[ -x .venv/bin/python ]]; then
    PY=(.venv/bin/python -I)
else
    PY=(python3 -I)
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
exec "${PY[@]}" tests/check.py --runq "$RUNQ" --generic "${RUNQ_GENERIC:-}" --model "$MODEL" "$@"
