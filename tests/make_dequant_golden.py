#!/usr/bin/env python3
"""Goldens for the K-quant dequantization check (tests/golden/dequant/q4_k_m.npz, q5_k.npz).

A few rows of Q4_K and Q6_K tensors of Qwen3-0.6B-Q4_K_M.gguf (q4_k_m.npz) and of Q5_K tensors (and a
Q4_K / Q6_K one) of the Unsloth Qwen3.5-0.8B-Q4_K_M.gguf (q5_k.npz, `--q5`), dequantized to float32 by
gguf-py (gguf.quants.dequantize, the numpy port of ggml's dequantize_row_q4_K / _q5_K / _q6_K).
check.py compares runq's `-m dequant` output (rows from the file's blocks, as for the embedding lookup,
and from the repacked Q4P / Q5P / Q6P layouts the matmul kernels read) with these bit for bit.

Needs the `gguf` package (pip install gguf; written with gguf 0.19.0) and numpy, e.g. in a scratch
venv:  python3 -m venv /tmp/g && /tmp/g/bin/pip install gguf && /tmp/g/bin/python tests/make_dequant_golden.py
The runq tests themselves only need numpy.
"""
import argparse
import os

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf")
OUT = os.path.join(ROOT, "tests", "golden", "dequant", "q4_k_m.npz")
MODEL5 = os.path.join(ROOT, "models", "Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf")
OUT5 = os.path.join(ROOT, "tests", "golden", "dequant", "q5_k.npz")

# (tensor, first row, rows): each crosses a 16-row block boundary of the repacked layouts
CASES = [
    ("blk.0.attn_q.weight", 15, 2),     # Q4_K, 1024 columns
    ("blk.3.ffn_down.weight", 1007, 2),  # Q4_K, 3072 columns (attn_v / ffn_down are Q4_K in layer 3)
    ("blk.0.attn_v.weight", 15, 2),     # Q6_K, 1024 columns
    ("blk.27.ffn_down.weight", 511, 2),  # Q6_K, 3072 columns
    ("token_embd.weight", 151871, 2),   # Q6_K: embedding rows and the tied classifier
]
# Qwen3.5-0.8B-Q4_K_M.gguf: attn_qkv / ssm_out of the linear layers are Q5_K (1024 and 2048 columns)
CASES5 = [
    ("blk.0.attn_qkv.weight", 15, 2),     # Q5_K, 1024 columns, across a 16-row block boundary
    ("blk.17.attn_qkv.weight", 6143, 1),  # Q5_K, the last row (6144 rows)
    ("blk.2.ssm_out.weight", 1007, 2),    # Q5_K, 2048 columns
    ("blk.21.ssm_out.weight", 0, 1),      # Q5_K
    ("blk.3.attn_v.weight", 15, 2),       # Q6_K, 1024 columns
    ("blk.3.ffn_down.weight", 1007, 2),   # Q4_K, 3584 columns
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model")
    ap.add_argument("-o", "--out")
    ap.add_argument("--q5", action="store_true", help="the Qwen3.5 Q5_K cases (q5_k.npz) instead of the 0.6B ones")
    a = ap.parse_args()
    cases = CASES5 if a.q5 else CASES
    a.model = a.model or (MODEL5 if a.q5 else MODEL)
    a.out = a.out or (OUT5 if a.q5 else OUT)
    r = GGUFReader(a.model)
    tensors = {t.name: t for t in r.tensors}
    out = {"names": np.array([c[0] for c in cases]), "row0": np.array([c[1] for c in cases], dtype=np.int32)}
    for i, (name, r0, n) in enumerate(cases):
        t = tensors[name]
        rows = dequantize(t.data[r0:r0 + n], t.tensor_type).astype(np.float32)  # (n, ncols)
        assert rows.shape == (n, int(t.shape[0])), rows.shape
        out[f"rows{i}"] = rows
        out[f"type{i}"] = np.array(t.tensor_type.name)
        print(f"{name} rows {r0}..{r0 + n - 1}: {t.tensor_type.name}, {rows.shape[1]} columns")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f"wrote {os.path.relpath(a.out, ROOT)} ({os.path.getsize(a.out)} bytes)")


if __name__ == "__main__":
    main()
