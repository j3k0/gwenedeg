#!/usr/bin/env python3
"""Synthetic Qwen3.5 GGUF models with more value heads than key heads (num_v_heads = r num_k_heads, as Qwen3.5-4B / 9B
(r = 2) and 27B (r = 3)), cut out of Qwen3.5-0.8B-Q8_0.gguf (16 key heads = 16 value heads), for the q35 group of the
default suite (tests/check_models.py; numpy only; cached in build/synth-q35, ~0.3 s when up to date).

synth-q35-r2.gguf (nk 4, nv 8) and synth-q35-r3.gguf (nk 2, nv 6), ~87 MB each: 6 layers (0.8B's layers 0..5: linear
0, 1, 2, 4, 5 and attention 3), dim 256, FFN 512, head dims 128, the 0.8B's attention (8 / 2 heads of 256), its
tokenizer, template and tied Q8_0 embedding (the first 256 columns). Per linear layer: the q and k rows / conv channels
of key heads 0 .. nk - 1, the v rows / channels, z rows, alpha / beta rows, ssm_a / ssm_dt of value heads
0 .. nv - 1, ssm_out's input columns of those heads; every matrix cut to its first 256 (or 512) columns (whole Q8_0
blocks). The file's value head h then uses key head h % nk (the tiled order of llama.cpp's converter, which every
qwen35 GGUF has: docs/qwen35.md): tests/ref_qwen35.py's GgufWeights reads it so, and that reading was checked against
transformers on tiny random models converted by llama.cpp's convert_hf_to_gguf.py. Not a language model, but real
weight statistics in the shapes of the bigger models' linear layers.
bad-q35-nv.gguf, bad-q35-hadamard.gguf (< 1 KB): metadata only (the source's hyperparameters, no tokenizer, a token_embd
  entry without data), with 6 value heads on 4 key heads, resp. a prism.hadamard.* key (Hadamard-folded weights):
  runq must reject them while reading the configuration.

Usage: python3 tests/make_synth_q35.py [--src Qwen3.5-0.8B-Q8_0.gguf] [--out build/synth-q35] [--force]
Prints the output directory; exit status 2 if the source is missing.
"""
import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ggufmini as G  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "models", "Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q8_0.gguf")
OUT = os.path.join(ROOT, "build", "synth-q35")
DIM, FFN, LAYERS = 256, 512, 6
MODELS = {"synth-q35-r2.gguf": (4, 8), "synth-q35-r3.gguf": (2, 6)}  # (key heads, value heads)
BAD = ["bad-q35-nv.gguf", "bad-q35-hadamard.gguf"]


def rows(f, name, ranges, k):
    """the rows of the ranges [(r0, r1)] of a source tensor, its first k columns (whole blocks): bytes"""
    ne, typ, _ = f.tensors[name]
    tb = f.tensor_bytes(name)
    nb = G.row_bytes(typ, k)
    assert k <= ne[0] and all(0 <= a < b <= tb.shape[0] for a, b in ranges), (name, ranges, k)
    return np.ascontiguousarray(np.concatenate([tb[a:b, :nb] for a, b in ranges])).tobytes()


def build(f, path, nk, nv):
    A = "qwen35."
    src_nk, dk = int(f.kv[A + "ssm.group_count"]), int(f.kv[A + "ssm.state_size"])
    dv = int(f.kv[A + "ssm.inner_size"]) // int(f.kv[A + "ssm.time_step_rank"])
    assert int(f.kv[A + "ssm.time_step_rank"]) == src_nk and nv % nk == 0 and nv <= src_nk
    hd, H, KVH = int(f.kv[A + "attention.key_length"]), int(f.kv[A + "attention.head_count"]), int(f.kv[A + "attention.head_count_kv"])
    over = {"general.name": "gwenedeg-synth-q35 (cut from Qwen3.5-0.8B-Q8_0)", A + "block_count": LAYERS,
            A + "embedding_length": DIM, A + "feed_forward_length": FFN, A + "ssm.group_count": nk,
            A + "ssm.time_step_rank": nv, A + "ssm.inner_size": nv * dv}
    kvs = []
    for key, raw in f.kv_raw:
        if key.startswith("quantize.imatrix"):
            continue
        if key in over:
            typ = struct.unpack_from("<I", raw)[0]
            raw = G.raw_kv(typ, over[key])
        kvs.append((key, raw))
    V = f.tensors["token_embd.weight"][0][1]
    F32 = G.F32
    t = [("token_embd.weight", (DIM, V), G.Q8_0, rows(f, "token_embd.weight", [(0, V)], DIM)),
         ("output_norm.weight", (DIM,), F32, rows(f, "output_norm.weight", [(0, 1)], DIM))]
    qk = [(0, nk * dk), (src_nk * dk, src_nk * dk + nk * dk)]  # q, k rows / channels of key heads 0 .. nk - 1
    v0 = 2 * src_nk * dk
    for l in range(LAYERS):
        s = f"blk.{l}."
        t.append((s + "attn_norm.weight", (DIM,), F32, rows(f, s + "attn_norm.weight", [(0, 1)], DIM)))
        t.append((s + "post_attention_norm.weight", (DIM,), F32, rows(f, s + "post_attention_norm.weight", [(0, 1)], DIM)))
        if s + "attn_qkv.weight" in f.tensors:  # linear attention
            cd = 2 * nk * dk + nv * dv
            ch = qk + [(v0, v0 + nv * dv)]
            t.append((s + "attn_qkv.weight", (DIM, cd), G.Q8_0, rows(f, s + "attn_qkv.weight", ch, DIM)))
            t.append((s + "attn_gate.weight", (DIM, nv * dv), G.Q8_0, rows(f, s + "attn_gate.weight", [(0, nv * dv)], DIM)))
            t.append((s + "ssm_alpha.weight", (DIM, nv), G.Q8_0, rows(f, s + "ssm_alpha.weight", [(0, nv)], DIM)))
            t.append((s + "ssm_beta.weight", (DIM, nv), G.Q8_0, rows(f, s + "ssm_beta.weight", [(0, nv)], DIM)))
            t.append((s + "ssm_out.weight", (nv * dv, DIM), G.Q8_0, rows(f, s + "ssm_out.weight", [(0, DIM)], nv * dv)))
            K = f.tensors[s + "ssm_conv1d.weight"][0][0]
            t.append((s + "ssm_conv1d.weight", (K, cd), F32, rows(f, s + "ssm_conv1d.weight", ch, K)))
            t.append((s + "ssm_a", (nv,), F32, rows(f, s + "ssm_a", [(0, 1)], nv)))
            t.append((s + "ssm_dt.bias", (nv,), F32, rows(f, s + "ssm_dt.bias", [(0, 1)], nv)))
            t.append((s + "ssm_norm.weight", (dv,), F32, rows(f, s + "ssm_norm.weight", [(0, 1)], dv)))
        else:
            for role, d, k in (("attn_q", 2 * H * hd, DIM), ("attn_k", KVH * hd, DIM), ("attn_v", KVH * hd, DIM),
                               ("attn_output", DIM, H * hd)):
                t.append((s + role + ".weight", (k, d), G.Q8_0, rows(f, s + role + ".weight", [(0, d)], k)))
            for role in ("attn_q_norm", "attn_k_norm"):
                t.append((s + role + ".weight", (hd,), F32, rows(f, s + role + ".weight", [(0, 1)], hd)))
        for role, d, k in (("ffn_gate", FFN, DIM), ("ffn_up", FFN, DIM), ("ffn_down", DIM, FFN)):
            t.append((s + role + ".weight", (k, d), G.Q8_0, rows(f, s + role + ".weight", [(0, d)], k)))
    G.write_gguf(path, kvs, t)


def build_bad(f, path, extra):
    """a metadata-only file: the source's architecture keys (no tokenizer) with overrides / additions extra"""
    kvs, seen = [], set()
    for key, raw in f.kv_raw:
        if key.startswith(("general.architecture", "qwen35.")):
            if key in extra:
                raw = G.raw_kv(struct.unpack_from("<I", raw)[0], extra[key][1])
                seen.add(key)
            kvs.append((key, raw))
    kvs += [(k, G.raw_kv(*v)) for k, v in extra.items() if k not in seen]
    ne = f.tensors["token_embd.weight"][0]
    G.write_gguf(path, kvs, [("token_embd.weight", ne, 255, None)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(a.src):
        print(f"missing {a.src}", file=sys.stderr)
        return 2
    h = hashlib.sha256()
    for p in [os.path.abspath(__file__), os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggufmini.py")]:
        with open(p, "rb") as fh:
            h.update(fh.read())
    st = os.stat(a.src)
    stamp = {"code": h.hexdigest(), "src": os.path.abspath(a.src), "size": st.st_size, "mtime": int(st.st_mtime)}
    sp = os.path.join(a.out, "stamp.json")
    if not a.force and os.path.exists(sp) and all(os.path.exists(os.path.join(a.out, n)) for n in list(MODELS) + BAD):
        with open(sp) as fh:
            if json.load(fh) == stamp:
                print(a.out)
                return 0
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(sp):
        os.remove(sp)
    f = G.GgufFile(a.src)
    for name, (nk, nv) in MODELS.items():
        build(f, os.path.join(a.out, name), nk, nv)
    build_bad(f, os.path.join(a.out, BAD[0]), {"qwen35.ssm.group_count": (G.GV_U32, 4), "qwen35.ssm.time_step_rank": (G.GV_U32, 6),
                                               "qwen35.ssm.inner_size": (G.GV_U32, 6 * 128)})
    build_bad(f, os.path.join(a.out, BAD[1]), {"prism.hadamard.gdn_v_grouped": (G.GV_BOOL, True)})
    with open(sp, "w") as fh:
        json.dump(stamp, fh)
    print(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
