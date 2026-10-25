#!/usr/bin/env python3
"""Golden for the K-quant dequantization check (tests/golden/dequant/q4_k_m.npz).

A few rows of Q4_K and Q6_K tensors of Qwen3-0.6B-Q4_K_M.gguf, dequantized to float32 by gguf-py
(gguf.quants.dequantize, the numpy port of ggml's dequantize_row_q4_K / _q6_K). check.py compares
runq's `-m dequant` output (rows from the file's blocks, as for the embedding lookup, and from the
repacked Q4P / Q6P layouts the matmul kernels read) with these bit for bit.

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

# (tensor, first row, rows): each crosses a 16-row block boundary of the repacked layouts
CASES = [
    ("blk.0.attn_q.weight", 15, 2),     # Q4_K, 1024 columns
    ("blk.3.ffn_down.weight", 1007, 2),  # Q4_K, 3072 columns (attn_v / ffn_down are Q4_K in layer 3)
    ("blk.0.attn_v.weight", 15, 2),     # Q6_K, 1024 columns
    ("blk.27.ffn_down.weight", 511, 2),  # Q6_K, 3072 columns
    ("token_embd.weight", 151871, 2),   # Q6_K: embedding rows and the tied classifier
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("-o", "--out", default=OUT)
    a = ap.parse_args()
    r = GGUFReader(a.model)
    tensors = {t.name: t for t in r.tensors}
    out = {"names": np.array([c[0] for c in CASES]), "row0": np.array([c[1] for c in CASES], dtype=np.int32)}
    for i, (name, r0, n) in enumerate(CASES):
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
