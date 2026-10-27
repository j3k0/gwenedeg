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
bad-q35-nv.gguf, bad-q35-hadamard.gguf, bad-q35-nextn.gguf (< 1 KB): metadata only (the source's hyperparameters, no
  tokenizer, a token_embd entry without data), with 6 value heads on 4 key heads, resp. a prism.hadamard.* key
  (Hadamard-folded weights), resp. nextn_predict_layers = block_count: runq must reject them while reading the
  configuration.

The 27B's layout at a small scale (if the 0.8B Q4_K_M file is there, --src-q4; docs/qwen38-speculative.md, file layout):
synth-q35-27b.gguf (330 MB): 8 layers (linear 0, 1, 2, 4, 5, 6, attention 3, 7: every 4th), dim 768 (not a power of
  two; 27B 5120), FFN 2560 (27B 17408), attention 12 / 2 heads of 256 (GQA 6, as 24 / 4), DeltaNet 4 key / 12 value
  heads of 128 (r = 3, as 16 / 48), an untied Q4_K token_embd and a Q6_K output, the 27B's types per role: attn_q / k /
  output / gate, ffn_gate / up Q4_K; attn_v and ffn_down Q6_K or Q4_K; attn_qkv Q6_K or Q4_K (the 0.8B's are Q5_K:
  rows of its Q6_K token_embd resp. of a Q4_K ffn_up); ssm_out Q5_K. Not the 27B's: ssm_alpha / ssm_beta stay Q8_0
  (the 27B's are F32 matrices, which runq does not load yet), the 0.8B's tokenizer and template. Cut out of the 0.8B
  Q4_K_M file in whole super-blocks: the first 768 columns (rows 0 .. d - 1 of the matrices with dim rows), an
  attention layer's q heads 0..7 from one source layer (with its 2 kv heads) and 8..11 from another (attn_q rows,
  attn_output columns side by side), the q / k channels of key heads 0..3 and the v channels of value heads 0..11 as
  in build(); token_embd: 3 consecutive Q4_K super-blocks per row from a pool of the source's Q4_K matrices; output:
  the source's Q6_K token_embd, its first 768 columns. The general.sampling.* keys of the 27B (runq's defaults
  for the options not given: the q35 group's sampling_defaults check).
synth-q35-27b-mtp.gguf: the same + the MTP block as the 27B lays it out: block_count 9, qwen35.nextn_predict_layers 1,
  blk.8 a full-attention layer (attn_* , ffn_*, the norms) + nextn.eh_proj [2 dim -> dim] Q8_0, nextn.enorm / hnorm /
  shared_head_norm F32, sharing token_embd / output. Today runq refuses it (TODO S6 makes it load).

Usage: python3 tests/make_synth_q35.py [--src Qwen3.5-0.8B-Q8_0.gguf] [--src-q4 Qwen3.5-0.8B-Q4_K_M.gguf]
                                      [--out build/synth-q35] [--force]
Prints the output directory; exit status 2 if the source is missing. Deterministic (byte-identical rebuilds).
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
BAD = ["bad-q35-nv.gguf", "bad-q35-hadamard.gguf", "bad-q35-nextn.gguf"]


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


# ------------------------------------------------------------------------------------------ the 27B's shapes

SRC_Q4 = os.path.join(ROOT, "models", "Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf")
M27 = "synth-q35-27b.gguf"
M27_MTP = "synth-q35-27b-mtp.gguf"
D27, FFN27, H27, KVH27, NK27, NV27, L27 = 768, 2560, 12, 2, 4, 12, 8  # GQA 6, r = 3; layers 3, 7 attention
QKV27 = {0: G.Q6_K, 1: G.Q4_K, 2: G.Q6_K, 4: G.Q4_K, 5: G.Q6_K, 6: G.Q4_K}  # attn_qkv per linear layer (27B: Q6_K / Q4_K)
ATT27 = {3: (3, 11), 7: (15, 19), 8: (23, 7)}  # attention layer: the source layers of q heads 0..7, 8..11 (8: the MTP block)


def cut_cols(f, srcs, d):
    """rows 0..d-1 of source matrices, each its first k columns (whole blocks), side by side: srcs [(name, k)] -> bytes"""
    parts = []
    for name, k in srcs:
        ne, typ, _ = f.tensors[name]
        assert d <= f.tensor_bytes(name).shape[0] and k <= ne[0], (name, d, k)
        parts.append(f.tensor_bytes(name)[:d, :G.row_bytes(typ, k)])
    return np.ascontiguousarray(np.concatenate(parts, axis=1)).tobytes()


def build_27b(f8, f4, path, mtp):
    """the 27B's layout and types at a small scale, cut out of the 0.8B Q4_K_M file (f4; eh_proj: Q8_0, from f8)"""
    A = "qwen35."
    T = lambda name: f4.tensors[name][1]  # noqa: E731
    dk = int(f4.kv[A + "ssm.state_size"])
    src_nk, hd = int(f4.kv[A + "ssm.group_count"]), int(f4.kv[A + "attention.key_length"])
    dv = int(f4.kv[A + "ssm.inner_size"]) // int(f4.kv[A + "ssm.time_step_rank"])
    over = {"general.name": "gwenedeg-synth-q35-27b (cut from Qwen3.5-0.8B-Q4_K_M)", A + "block_count": L27 + mtp,
            A + "embedding_length": D27, A + "feed_forward_length": FFN27, A + "attention.head_count": H27,
            A + "attention.head_count_kv": KVH27, A + "ssm.group_count": NK27, A + "ssm.time_step_rank": NV27,
            A + "ssm.inner_size": NV27 * dv}
    kvs = []
    for key, raw in f4.kv_raw:
        if key.startswith("quantize.imatrix"):
            continue
        if key in over:
            raw = G.raw_kv(struct.unpack_from("<I", raw)[0], over[key])
        kvs.append((key, raw))
        if key == "general.type":  # (the 27B's sampling defaults, at its place: rq_model_sampling)
            kvs += [("general.sampling.top_k", G.raw_kv(G.GV_I32, 20)), ("general.sampling.top_p", G.raw_kv(G.GV_F32, 0.95)),
                    ("general.sampling.temp", G.raw_kv(G.GV_F32, 1.0))]
        if key == A + "attention.value_length" and mtp:
            kvs.append((A + "nextn_predict_layers", G.raw_kv(G.GV_U32, 1)))
    V = f4.tensors["token_embd.weight"][0][1]
    F32, sb = G.F32, D27 // 256
    # token_embd: untied Q4_K, each row 3 consecutive super-blocks of a pool of the source's Q4_K matrices
    pool, need = [], V * sb
    for l in range(23, -1, -1):
        for role in ("ffn_gate", "ffn_up", "attn_gate"):
            name = f"blk.{l}.{role}.weight"
            if name in f4.tensors and T(name) == G.Q4_K:
                pool.append(f4.tensor_bytes(name).reshape(-1, 144))
        if sum(len(p) for p in pool) >= need:
            break
    emb = np.concatenate(pool)[:need]
    assert len(emb) == need
    t = [("output.weight", (D27, V), G.Q6_K, cut_cols(f4, [("token_embd.weight", D27)], V)),  # Q6_K: the 0.8B's classifier
         ("output_norm.weight", (D27,), F32, cut_cols(f4, [("output_norm.weight", D27)], 1)),
         ("token_embd.weight", (D27, V), G.Q4_K, emb.tobytes())]
    qk = [(0, NK27 * dk), (src_nk * dk, src_nk * dk + NK27 * dk)]  # the q, k channels of key heads 0 .. NK27 - 1
    v0 = 2 * src_nk * dk
    cd, vd = 2 * NK27 * dk + NV27 * dv, NV27 * dv
    for l in range(L27 + mtp):
        s, blk = f"blk.{l}.", []
        if l in ATT27:  # q heads 0..7 of layer a (its kv heads), 8..11 of layer b
            a, b = (f"blk.{x}." for x in ATT27[l])
            blk += [("attn_k.weight", (D27, KVH27 * hd), T(a + "attn_k.weight"), cut_cols(f4, [(a + "attn_k.weight", D27)], KVH27 * hd)),
                    ("attn_k_norm.weight", (hd,), F32, cut_cols(f4, [(a + "attn_k_norm.weight", hd)], 1)),
                    ("attn_norm.weight", (D27,), F32, cut_cols(f4, [(a + "attn_norm.weight", D27)], 1)),
                    ("attn_output.weight", (H27 * hd, D27), T(a + "attn_output.weight"),
                     cut_cols(f4, [(a + "attn_output.weight", 8 * hd), (b + "attn_output.weight", (H27 - 8) * hd)], D27)),
                    ("attn_q.weight", (D27, 2 * H27 * hd), T(a + "attn_q.weight"),
                     rows(f4, a + "attn_q.weight", [(0, 16 * hd)], D27) + rows(f4, b + "attn_q.weight", [(0, 2 * (H27 - 8) * hd)], D27)),
                    ("attn_q_norm.weight", (hd,), F32, cut_cols(f4, [(a + "attn_q_norm.weight", hd)], 1)),
                    ("attn_v.weight", (D27, KVH27 * hd), T(a + "attn_v.weight"), cut_cols(f4, [(a + "attn_v.weight", D27)], KVH27 * hd))]
            assert T(a + "attn_output.weight") == T(b + "attn_output.weight") and T(a + "attn_q.weight") == T(b + "attn_q.weight")
        else:
            a = s
            qt = QKV27[l]
            if qt == G.Q6_K:  # rows of the source's Q6_K token embedding (attn_qkv is Q5_K in the 0.8B)
                qkv = rows(f4, "token_embd.weight", [(20000 * (l + 1), 20000 * (l + 1) + cd)], D27)
            else:             # rows of a Q4_K ffn_up of another layer
                qkv = rows(f4, f"blk.{l + 12}.ffn_up.weight", [(0, cd)], D27)
            ch = qk + [(v0, v0 + vd)]
            K = f4.tensors[s + "ssm_conv1d.weight"][0][0]
            blk += [("attn_gate.weight", (D27, vd), T(s + "attn_gate.weight"), rows(f4, s + "attn_gate.weight", [(0, vd)], D27)),
                    ("attn_norm.weight", (D27,), F32, cut_cols(f4, [(s + "attn_norm.weight", D27)], 1)),
                    ("attn_qkv.weight", (D27, cd), qt, qkv)]
            blk += [("ssm_a", (NV27,), F32, cut_cols(f4, [(s + "ssm_a", NV27)], 1)),
                    ("ssm_alpha.weight", (D27, NV27), T(s + "ssm_alpha.weight"), rows(f4, s + "ssm_alpha.weight", [(0, NV27)], D27)),
                    ("ssm_beta.weight", (D27, NV27), T(s + "ssm_beta.weight"), rows(f4, s + "ssm_beta.weight", [(0, NV27)], D27)),
                    ("ssm_conv1d.weight", (K, cd), F32, rows(f4, s + "ssm_conv1d.weight", ch, K)),
                    ("ssm_dt.bias", (NV27,), F32, cut_cols(f4, [(s + "ssm_dt.bias", NV27)], 1)),
                    ("ssm_norm.weight", (dv,), F32, cut_cols(f4, [(s + "ssm_norm.weight", dv)], 1)),
                    ("ssm_out.weight", (vd, D27), T(s + "ssm_out.weight"), cut_cols(f4, [(s + "ssm_out.weight", vd)], D27))]
        blk += [("ffn_down.weight", (FFN27, D27), T(a + "ffn_down.weight"), cut_cols(f4, [(a + "ffn_down.weight", FFN27)], D27)),
                ("ffn_gate.weight", (D27, FFN27), T(a + "ffn_gate.weight"), rows(f4, a + "ffn_gate.weight", [(0, FFN27)], D27)),
                ("ffn_up.weight", (D27, FFN27), T(a + "ffn_up.weight"), rows(f4, a + "ffn_up.weight", [(0, FFN27)], D27)),
                ("post_attention_norm.weight", (D27,), F32, cut_cols(f4, [(a + "post_attention_norm.weight", D27)], 1))]
        if l == L27:  # the MTP block: + nextn.* (eh_proj [2 dim -> dim] Q8_0; the norms), sharing token_embd / output
            blk += [("nextn.eh_proj.weight", (2 * D27, D27), G.Q8_0, cut_cols(f8, [("blk.23.attn_output.weight", 2 * D27)], D27)),
                    ("nextn.enorm.weight", (D27,), F32, cut_cols(f4, [("blk.23.attn_norm.weight", D27)], 1)),
                    ("nextn.hnorm.weight", (D27,), F32, cut_cols(f4, [("blk.23.post_attention_norm.weight", D27)], 1)),
                    ("nextn.shared_head_norm.weight", (D27,), F32, cut_cols(f4, [("output_norm.weight", D27)], 1))]
        t += [(s + n, ne, ty, data) for n, ne, ty, data in sorted(blk)]
    G.write_gguf(path, kvs, t)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--src-q4", default=SRC_Q4, help="the 0.8B Q4_K_M file (the 27B-shaped models; skipped if missing)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(a.src):
        print(f"missing {a.src}", file=sys.stderr)
        return 2
    q4 = a.src_q4 if os.path.exists(a.src_q4) else None
    if not q4:
        print(f"note: missing {a.src_q4}: no {M27} / {M27_MTP}", file=sys.stderr)
    h = hashlib.sha256()
    for p in [os.path.abspath(__file__), os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggufmini.py")]:
        with open(p, "rb") as fh:
            h.update(fh.read())
    st = os.stat(a.src)
    stamp = {"code": h.hexdigest(), "src": os.path.abspath(a.src), "size": st.st_size, "mtime": int(st.st_mtime)}
    if q4:
        st = os.stat(q4)
        stamp.update(src_q4=os.path.abspath(q4), size_q4=st.st_size, mtime_q4=int(st.st_mtime))
    names = list(MODELS) + BAD + ([M27, M27_MTP] if q4 else [])
    sp = os.path.join(a.out, "stamp.json")
    if not a.force and os.path.exists(sp) and all(os.path.exists(os.path.join(a.out, n)) for n in names):
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
    build_bad(f, os.path.join(a.out, BAD[2]), {"qwen35.nextn_predict_layers": (G.GV_U32, int(f.kv["qwen35.block_count"]))})
    for n in (M27, M27_MTP):
        if os.path.exists(os.path.join(a.out, n)):
            os.remove(os.path.join(a.out, n))
    if q4:
        f4 = G.GgufFile(q4)
        build_27b(f, f4, os.path.join(a.out, M27), False)
        build_27b(f, f4, os.path.join(a.out, M27_MTP), True)
    with open(sp, "w") as fh:
        json.dump(stamp, fh)
    print(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
