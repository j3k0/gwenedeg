#!/usr/bin/env bash
# Gwenedeg fast regression suite: tokenizer + logits vs HF (sequential and batched prefill),
# batched vs sequential logits, thread invariance, VNNI vs generic kernels, greedy tripwire, the
# library API (tests/api_test.c, built as build/api_test).
# Usage: tests/run_tests.sh [--update-greedy] [other tests/check.py options, e.g. -v, --only logits]
#        tests/run_tests.sh --models   the bigger Qwen3 models only (group models: every downloaded family file,
#                                      scripts/download_models.py family; slow, minutes; make test-models)
#        tests/run_tests.sh --long-context   the long-context checks only (group long, tests/check_long.py: Qwen3.5
#                                      at up to 98k positions; ~15-20 min; TMPDIR=/var/tmp where /tmp is small)
# Env:   OMP_NUM_THREADS (default 8), RUNQ (binary to test, default ./runq, built with make),
#        RUNQ_GENERIC (generic-kernel build to compare with, default ./runq-generic when RUNQ is
#        not set; with RUNQ set and RUNQ_GENERIC unset the kernels test is skipped; with RUNQ set the
#        api test is skipped too: it links the library objects of this tree),
#        GGUF (GGUF Q8_0 file for the gguf tests, default models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf;
#        skipped if missing), Q4 (GGUF Q4_K_M file for the q4 tests, default
#        models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf; skipped if missing).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="models/Qwen3-0.6B-q8.bin"
if [[ ! -f "$MODEL" ]]; then
    echo "error: $MODEL not found (run: make models)" >&2
    exit 1
fi
GGUF="${GGUF-models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf}"
if [[ -n "$GGUF" && ! -f "$GGUF" ]]; then
    echo "note: $GGUF not found, skipping the GGUF tests (scripts/download_models.py q8gguf)" >&2
    GGUF=""
fi
Q4="${Q4-models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf}"
if [[ -n "$Q4" && ! -f "$Q4" ]]; then
    echo "note: $Q4 not found, skipping the Q4_K_M tests (scripts/download_models.py q4km)" >&2
    Q4=""
fi

Q35="${Q35-models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf}"
if [[ -n "$Q35" && ! -f "$Q35" ]]; then
    echo "note: $Q35 not found, skipping the Qwen3.5 tests (scripts/download_models.py 3.5-0.8b-q8)" >&2
    Q35=""
fi

API_TEST=""
if [[ -z "${RUNQ:-}" ]]; then
    make -s -j8 runq runq-generic api-test tools-test
    RUNQ="$ROOT/runq"
    RUNQ_GENERIC="${RUNQ_GENERIC:-$ROOT/runq-generic}"
    API_TEST="$ROOT/build/api_test"
fi

if [[ -x .venv/bin/python ]]; then
    PY=(.venv/bin/python -I)
else
    PY=(python3 -I)
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
ARGS=()
LONG=0
for x in "$@"; do
    if [[ "$x" == "--models" ]]; then ARGS+=(--only models)
    elif [[ "$x" == "--long-context" ]]; then ARGS+=(--only long); LONG=1
    else ARGS+=("$x"); fi
done
set -- "${ARGS[@]+"${ARGS[@]}"}"
if [[ $LONG == 1 ]]; then make -s -j8 long-ctx; fi
exec "${PY[@]}" tests/check.py --runq "$RUNQ" --generic "${RUNQ_GENERIC:-}" --model "$MODEL" --gguf "$GGUF" --q4 "$Q4" \
    --api-test "$API_TEST" --q35 "$Q35" "$@"
