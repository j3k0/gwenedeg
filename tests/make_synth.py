#!/usr/bin/env python3
"""Synthetic GGUF models for the default test suite (tests/check_models.py, group synth), cut out of
Qwen3-0.6B-Q4_K_M.gguf: the shapes of the bigger Qwen3 models that 0.6B does not have, at a fraction
of the cost (numpy only; run by check.py, cached in build/synth, ~0.2 s when up to date).

synth-q4.gguf (~165 MB): a 3-layer "model" with
  dim 768 (not a power of two, as 4B's 2560: rmsnorm's 1 / dim, the K-quant super-block counts),
  8 query / 2 kv heads (GQA 4, as 4B / 8B; q dim 1024 != dim), ffn 1280 (5 super-blocks),
  an untied Q4_K token_embd and a Q6_K output.weight (as Qwen3-8B Q4_K_M), RoPE theta 5e6 and
  context_length 262,144 (as the 2507 models: the default context cap), the Instruct-2507 chat template,
  Q4_K and Q6_K matrices mixed as in Q4_K_M (layers 0, 3 and 27 of 0.6B: attn_v / ffn_down Q6_K, Q4_K, Q6_K).
  Every matrix is a block of whole super-blocks (256 columns) of the corresponding 0.6B matrix (rows 0..d-1,
  super-blocks 0..k/256-1), the norms the first values of 0.6B's; token_embd's rows are 3 consecutive Q4_K
  super-blocks from a pool of 0.6B's Q4_K matrices, output.weight the first 768 columns of 0.6B's Q6_K
  token_embd. Not a language model, but real weight statistics and every code path of those shapes.
synth-tok-{qwen3,instruct-2507,thinking-2507}.gguf (~7.5 MB each): tokenizer-only files (the 0.6B
  tokenizer metadata, the hyperparameters, a token_embd entry of an unknown type without data) with the HF
  chat templates of tests/golden/templates: `runq -m render` / `-m tokenize` load them.

Usage: python3 tests/make_synth.py [--src Qwen3-0.6B-Q4_K_M.gguf] [--out build/synth] [--force]
Prints the output directory; exit status 2 if the source is missing.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ggufmini as G  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "models", "Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf")
OUT = os.path.join(ROOT, "build", "synth")
TEMPLATES = os.path.join(ROOT, "tests", "golden", "templates")

DIM, HEADS, KV_HEADS, HD, FFN = 768, 8, 2, 128, 1280
LAYERS = [0, 3, 27]  # 0.6B layers: attn_v / ffn_down Q6_K, Q4_K, Q6_K
THETA, CTX = 5e6, 262144
OVERRIDE = {  # metadata of the synthetic model (the rest copied from the source)
    "general.name": (G.GV_STR, "gwenedeg-synth (cut from Qwen3-0.6B-Q4_K_M)"),
    "qwen3.block_count": (G.GV_U32, len(LAYERS)),
    "qwen3.context_length": (G.GV_U32, CTX),
    "qwen3.embedding_length": (G.GV_U32, DIM),
    "qwen3.feed_forward_length": (G.GV_U32, FFN),
    "qwen3.attention.head_count": (G.GV_U32, HEADS),
    "qwen3.attention.head_count_kv": (G.GV_U32, KV_HEADS),
    "qwen3.rope.freq_base": (G.GV_F32, THETA),
}


def cut(f, name, d, k):
    """rows 0..d-1, columns 0..k-1 (whole blocks) of a source matrix: (type, bytes)"""
    ne, typ, _ = f.tensors[name]
    vals, nbytes = G.BLOCK[typ]
    tb = f.tensor_bytes(name)
    assert d <= (ne[1] if len(ne) > 1 else 1) and k <= ne[0] and k % vals == 0, (name, d, k, ne)
    return typ, np.ascontiguousarray(tb[:d, :k // vals * nbytes]).tobytes()


def metadata(f, template, extra=None):
    kv = dict(OVERRIDE, **(extra or {}))
    out, seen = [], set()
    for key, raw in f.kv_raw:
        if key == "tokenizer.chat_template":
            continue
        if key in kv:
            out.append((key, G.raw_kv(*kv[key])))
            seen.add(key)
        else:
            out.append((key, raw))
    for key in kv:
        if key not in seen:
            out.append((key, G.raw_kv(*kv[key])))
    out.append(("tokenizer.chat_template", G.raw_kv(G.GV_STR, template)))
    return out


def template(name):
    with open(os.path.join(TEMPLATES, name + ".jinja"), encoding="utf-8") as fh:
        return fh.read()


def build_model(f, path):
    V = f.tensors["token_embd.weight"][0][1]
    tensors = []
    # token_embd: untied Q4_K rows from a pool of the source's Q4_K super-blocks
    sb_row = DIM // 256
    need, pool = V * sb_row, []
    for l in range(28):
        for role in ("attn_q", "attn_k", "attn_output", "ffn_gate", "ffn_up"):
            name = f"blk.{l}.{role}.weight"
            if f.tensors[name][1] == G.Q4_K:
                pool.append(f.tensor_bytes(name).reshape(-1, 144))
        if sum(len(p) for p in pool) >= need:
            break
    emb = np.concatenate(pool)[:need]
    tensors.append(("token_embd.weight", (DIM, V), G.Q4_K, emb.tobytes()))
    tensors.append(("output_norm.weight", (DIM,), G.F32, cut(f, "output_norm.weight", 1, DIM)[1]))
    t, b = cut(f, "token_embd.weight", V, DIM)  # Q6_K, the first 3 super-blocks of each row
    tensors.append(("output.weight", (DIM, V), t, b))
    qd, kvd = HEADS * HD, KV_HEADS * HD
    for i, l in enumerate(LAYERS):
        s = f"blk.{l}."
        for role, d, k in (("attn_q", qd, DIM), ("attn_k", kvd, DIM), ("attn_v", kvd, DIM), ("attn_output", DIM, qd),
                           ("ffn_gate", FFN, DIM), ("ffn_up", FFN, DIM), ("ffn_down", DIM, FFN)):
            t, b = cut(f, s + role + ".weight", d, k)
            tensors.append((f"blk.{i}.{role}.weight", (k, d), t, b))
        for role, n in (("attn_norm", DIM), ("ffn_norm", DIM), ("attn_q_norm", HD), ("attn_k_norm", HD)):
            tensors.append((f"blk.{i}.{role}.weight", (n,), G.F32, cut(f, s + role + ".weight", 1, n)[1]))
    G.write_gguf(path, metadata(f, template("instruct-2507")), tensors)


def build_tok(f, path, tmpl):
    V = f.tensors["token_embd.weight"][0][1]
    G.write_gguf(path, metadata(f, template(tmpl)), [("token_embd.weight", (DIM, V), 255, None)])


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
    for p in [os.path.abspath(__file__), os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggufmini.py")] + \
             [os.path.join(TEMPLATES, t + ".jinja") for t in ("qwen3", "instruct-2507", "thinking-2507")]:
        with open(p, "rb") as fh:
            h.update(fh.read())
    st = os.stat(a.src)
    stamp = {"code": h.hexdigest(), "src": os.path.abspath(a.src), "size": st.st_size, "mtime": int(st.st_mtime)}
    sp = os.path.join(a.out, "stamp.json")
    names = ["synth-q4.gguf"] + [f"synth-tok-{t}.gguf" for t in ("qwen3", "instruct-2507", "thinking-2507")]
    if not a.force and os.path.exists(sp) and all(os.path.exists(os.path.join(a.out, n)) for n in names):
        with open(sp) as fh:
            if json.load(fh) == stamp:
                print(a.out)
                return 0
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(sp):
        os.remove(sp)
    f = G.GgufFile(a.src)
    build_model(f, os.path.join(a.out, "synth-q4.gguf"))
    for t in ("qwen3", "instruct-2507", "thinking-2507"):
        build_tok(f, os.path.join(a.out, f"synth-tok-{t}.gguf"), t)
    with open(sp, "w") as fh:
        json.dump(stamp, fh)
    print(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
