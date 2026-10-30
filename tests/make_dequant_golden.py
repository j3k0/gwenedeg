#!/usr/bin/env python3
"""Goldens for the K-quant dequantization check (tests/golden/dequant/q4_k_m.npz, q5_k.npz, q3_k.npz).

A few rows of Q4_K and Q6_K tensors of Qwen3-0.6B-Q4_K_M.gguf (q4_k_m.npz), of Q5_K tensors (and a
Q4_K / Q6_K one) of the Unsloth Qwen3.5-0.8B-Q4_K_M.gguf (q5_k.npz, `--q5`) and of Q3_K tensors (and a
Q4_K / Q5_K / Q6_K one) of the Unsloth Qwen3.5-0.8B-Q3_K_M.gguf (q3_k.npz, `--q3`), dequantized to float32
by gguf-py (gguf.quants.dequantize, the numpy port of ggml's dequantize_row_q3_K / _q4_K / _q5_K / _q6_K).
check.py compares runq's `-m dequant` output (rows from the file's blocks, as for the embedding lookup,
and from the repacked Q3P / Q4P / Q5P / Q6P layouts the matmul kernels read) with these bit for bit.

Needs the `gguf` package (pip install gguf; written with gguf 0.19.0) and numpy, e.g. in a scratch
venv:  python3 -m venv /tmp/g && /tmp/g/bin/pip install gguf && /tmp/g/bin/python tests/make_dequant_golden.py
The runq tests themselves only need numpy.

--iq: tests/golden/dequant/iq_blocks.npz, for Q2_K and the i-quants (src/iquants.h): per type the raw blocks and gguf-py's
float32 of (a) the first 1024 values of row 17 of the first tensor of that type in Underdog Saluki 27B's file
(--model, default models/Underdog-Saluki-27B-1.0-GGUF/...: all nine types) and (b) 4 x 256 random block bytes (seed 7;
finite fp16 scales): every bit pattern of the formats. tests/check_models.py checks tests/ggufmini.py's numpy
dequantization against these, bitwise (runq is checked against ggufmini on the files: iq_synth_run, the models group).
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
MODEL3 = os.path.join(ROOT, "models", "Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q3_K_M.gguf")
OUT3 = os.path.join(ROOT, "tests", "golden", "dequant", "q3_k.npz")
MODEL_IQ = os.path.join(ROOT, "models", "Underdog-Saluki-27B-1.0-GGUF", "Underdog-Saluki-27B-1.0-IQ2-mix.gguf")
OUT_IQ = os.path.join(ROOT, "tests", "golden", "dequant", "iq_blocks.npz")
IQ = {10: 84, 16: 66, 17: 74, 18: 98, 19: 50, 21: 110, 22: 82, 23: 136, 29: 56}  # ggml type: block bytes


def make_iq(model, out_path):
    from gguf import GGMLQuantizationType as QT
    r = GGUFReader(model)
    out, names, types = {}, [], []
    rng = np.random.default_rng(7)
    first = {}
    for t in r.tensors:
        ty = int(t.tensor_type)
        if ty in IQ and ty not in first and len(t.shape) == 2:
            first[ty] = t
    for ty in sorted(IQ):
        bb = IQ[ty]
        t = first[ty]
        real = np.ascontiguousarray(np.asarray(t.data[17:18])[:, :4 * bb])          # 4 super-blocks of row 17
        rnd = rng.integers(0, 256, size=(16, bb), dtype=np.uint8)
        h = (0x1000 + rng.integers(0, 0x3000, size=16)).astype(np.uint16)          # fp16 scales 2^-11 .. 2^1
        hb = h.view(np.uint8).reshape(16, 2)
        if ty == 10:
            rnd[:, 80:82], rnd[:, 82:84] = hb, hb[::-1]
        elif ty == 29:  # IQ1_M: the scale's 4 nibbles in the top nibbles of the 4 scale words
            sc = rnd[:, 48:56].copy().view(np.uint16)
            for j in range(4):
                sc[:, j] = (sc[:, j] & 0x0FFF) | (((h >> (4 * j)) & 15) << 12)
            rnd[:, 48:56] = sc.view(np.uint8)
        else:
            rnd[:, 0:2] = hb
        for kind, raw in (("real", real), ("random", rnd.reshape(4, 4 * bb))):
            i = len(names)
            rows = dequantize(raw, QT(ty)).astype(np.float32)
            out[f"raw{i}"], out[f"rows{i}"] = raw, rows
            names.append(f"{QT(ty).name}_{kind}")
            types.append(ty)
            print(f"{names[-1]}: {rows.shape}" + (f" ({t.name} row 17)" if kind == "real" else ""))
    out["names"], out["type"] = np.array(names), np.array(types, dtype=np.int32)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **out)
    print(f"wrote {os.path.relpath(out_path, ROOT)} ({os.path.getsize(out_path)} bytes)")

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
# Qwen3.5-0.8B-Q3_K_M.gguf: attn_q / attn_k / attn_gate / ffn_gate / ffn_up are Q3_K (1024 columns; the 27B's
# Q3_K_M file has the same roles plus token_embd)
CASES3 = [
    ("blk.0.ffn_gate.weight", 15, 2),     # Q3_K, across a 16-row block boundary
    ("blk.23.ffn_up.weight", 3583, 1),    # Q3_K, the last row (3584 rows)
    ("blk.3.attn_q.weight", 2047, 2),     # Q3_K, attention q + gate rows (4096)
    ("blk.3.attn_k.weight", 496, 16),     # Q3_K, the last 16-row block (512 rows)
    ("blk.0.attn_gate.weight", 1, 3),     # Q3_K, the linear layers' z
    ("blk.0.ffn_down.weight", 1007, 2),   # Q5_K, 3584 columns
    ("blk.3.attn_output.weight", 15, 2),  # Q4_K, 2048 columns
    ("token_embd.weight", 248318, 2),     # Q6_K: embedding rows and the tied classifier
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model")
    ap.add_argument("-o", "--out")
    ap.add_argument("--q5", action="store_true", help="the Qwen3.5 Q5_K cases (q5_k.npz) instead of the 0.6B ones")
    ap.add_argument("--q3", action="store_true", help="the Qwen3.5 Q3_K cases (q3_k.npz) instead of the 0.6B ones")
    ap.add_argument("--iq", action="store_true", help="Q2_K and the i-quants (iq_blocks.npz, from Underdog Saluki 27B's file)")
    a = ap.parse_args()
    if a.iq:
        make_iq(a.model or MODEL_IQ, a.out or OUT_IQ)
        return
    cases = CASES3 if a.q3 else CASES5 if a.q5 else CASES
    a.model = a.model or (MODEL3 if a.q3 else MODEL5 if a.q5 else MODEL)
    a.out = a.out or (OUT3 if a.q3 else OUT5 if a.q5 else OUT)
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
