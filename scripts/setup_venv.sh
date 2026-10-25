#!/usr/bin/env bash
# Create .venv with the Python deps needed by third_party/qwen3.c/export.py.
# Uses the CPU-only torch wheel (much smaller than the default CUDA build).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
"${PYTHON:-python3}" -m venv .venv   # (PYTHON=python3.12: numpy 2.3 needs Python >= 3.11)
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q --index-url https://download.pytorch.org/whl/cpu torch==2.7.0
.venv/bin/pip install -q transformers==4.53.0 numpy==2.3.1 Jinja2==3.1.6 safetensors
.venv/bin/python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"
