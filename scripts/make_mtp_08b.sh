#!/usr/bin/env bash
# Build the opt-in test model models/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-mtp-Q8_0.gguf: Qwen3.5-0.8B with its trained
# MTP block kept (blk.24, nextn_predict_layers 1; TODO S9.0b). Unsloth's 0.8B GGUFs drop the block.
# Steps: download the HF checkpoint at a fixed revision into an empty /var/tmp dir, convert it with llama.cpp's
# converter and its own gguf-py (no --no-mtp), print the sha256, delete the download.
# Prereqs: a llama.cpp checkout at cf34d31 (LLAMA_CPP), .venv (scripts/setup_venv.sh). The converter also needs
# sentencepiece and protobuf: they go into a throwaway venv under /var/tmp that layers on .venv (never into .venv).
# Needs ~2.6 GB of disk while it runs (1.75 GB download + 0.81 GB output); /tmp is a RAM tmpfs, so /var/tmp is used.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_CPP="${LLAMA_CPP:-/home/jeko/JC/llama.cpp}"
REV=2fc06364715b967f1860aea9cf38778875588b17      # Qwen/Qwen3.5-0.8B, 2026-03-02 ("Update README.md")
WORK="${WORK:-/var/tmp/s9-hf}"
HF="$WORK/Qwen3.5-0.8B"
VENV="$WORK/venv"
OUT="${1:-$ROOT/models/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-mtp-Q8_0.gguf}"

[ "$(git -C "$LLAMA_CPP" rev-parse --short=7 HEAD)" = cf34d31 ] || { echo "llama.cpp is not at cf34d31" >&2; exit 1; }
[ -e "$WORK" ] && { echo "$WORK exists: remove it first (the download goes into a new, empty dir)" >&2; exit 1; }
mkdir -p "$HF" "$(dirname "$OUT")"

# the throwaway venv: .venv's packages (torch, transformers, numpy, safetensors) + sentencepiece, protobuf
"$ROOT/.venv/bin/python" -m venv "$VENV"
"$ROOT/.venv/bin/python" -I -c 'import site; print(site.getsitepackages()[0])' \
    > "$("$VENV/bin/python" -I -c 'import site; print(site.getsitepackages()[0])')/dotvenv.pth"
"$VENV/bin/pip" install -q 'sentencepiece>=0.1.98,<0.3.0' 'protobuf>=4.21.0,<5.0.0'

(cd "$WORK" && "$ROOT/.venv/bin/python" -I -c '
import sys
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.5-0.8B", revision=sys.argv[1], local_dir=sys.argv[2],
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])' "$REV" "$HF")

cd "$LLAMA_CPP"   # the converter imports its conversion/ package from here
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$LLAMA_CPP/gguf-py" "$VENV/bin/python" convert_hf_to_gguf.py "$HF" \
    --outtype q8_0 --outfile "$OUT"
sha256sum "$OUT"
rm -rf "$WORK"
