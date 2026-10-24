#!/usr/bin/env bash
# Convert the downloaded HF weights (models/Qwen3-0.6B) to qwen3.c's Q8_0 format.
# Output: models/Qwen3-0.6B-q8.bin (+ .tokenizer and .template* side files).
# Prereqs: scripts/download_models.py hf && scripts/setup_venv.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:-$ROOT/models/Qwen3-0.6B}"
OUT="${2:-$ROOT/models/Qwen3-0.6B-q8.bin}"
cd "$ROOT/third_party/qwen3.c"   # export.py imports model.py from here (kept pristine)
PYTHONDONTWRITEBYTECODE=1 "$ROOT/.venv/bin/python" export.py "$OUT" "$SRC"
