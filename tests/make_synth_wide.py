#!/usr/bin/env python3
"""A full-width synthetic Qwen3.8-27B file for timing the MTP drafter without loading the 27B (TODO S9w; build/bench_mtp,
docs/qwen38-speculative.md "The MTP drafter (S9)"; numpy only). Not part of the default suite.

build/synth-wide/synth-q38-wide-cls4.gguf (~2.7 GB) or -cls6.gguf (~3.0 GB): the 27B's hyper-parameters, read from the
IQ3_S-mtp file's header (metadata only, no tensor data: dim 5120, FFN 17408, 24 / 4 heads of 256, DeltaNet 16 key / 48
value heads of 128, rope, eps), with
- a 4-layer trunk of the 27B's layer pattern (full_attention_interval 4: linear 0, 1, 2, attention 3; the 27B's 64 layers
  are 16 times this), the types of Unsloth's Q4_K_M per role and in its mix (attn_qkv Q6_K / Q4_K / Q6_K, ffn_down
  Q6_K / Q4_K / Q4_K / Q6_K, attn_v Q6_K, ssm_out Q5_K, the other matrices Q4_K; ssm_alpha / ssm_beta F32);
- nextn_predict_layers 1 and the MTP block blk.4 all Q6_K, nextn.eh_proj (2 dim -> dim) Q6_K too (IQ3_S-mtp's);
- token_embd Q4_K and the classifier output (248,320 rows) Q4_K (-cls4, as IQ3_S-mtp) or Q6_K (-cls6, as Unsloth's);
- the tokenizer and template of Qwen3.5-0.8B-Q8_0.gguf (the same 248,320-token vocabulary: checked against the 27B
  header's tokens, merges and token types).
The K-quant blocks are random bytes (seeded per tensor: deterministic) with a fixed fp16 d / dmin (and Q6_K's scales
in -16..15) so that the weights are ~0.03 in size: finite activations; the norms are 1, the F32 tensors small random
values. Not a language model: the passes' cost does not depend on the values. Written tensor by tensor, so the
memory stays small. Refuses when / would have less than 15 GB free after writing.

Usage: python3 tests/make_synth_wide.py --cls 4|6 [--hdr 27B.gguf] [--tok Qwen3.5-0.8B-Q8_0.gguf] [--out build/synth-wide]
                                        [--force]
Prints the file's path (an existing file is kept unless --force).
"""
import argparse
import os
import shutil
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ggufmini as G  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HDR = os.path.join(ROOT, "models", "Qwen3.8-27B-GSQ-RCO-GGUF", "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf")
TOK = os.path.join(ROOT, "models", "Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q8_0.gguf")
OUT = os.path.join(ROOT, "build", "synth-wide")
TRUNK = 4  # linear 0, 1, 2, attention 3 (full_attention_interval 4)
QKV = {0: G.Q6_K, 1: G.Q4_K, 2: G.Q6_K}  # attn_qkv per linear layer
DOWN = {0: G.Q6_K, 1: G.Q4_K, 2: G.Q4_K, 3: G.Q6_K}  # ffn_down per layer
KEEP_FREE = 15e9  # bytes free on the output's file system after writing
# fp16 d (and dmin) per type: weights of ~0.03 rms (Q4_K: d sc q - dmin m, sc / m 0..63, q 0..15: mean 0 at
# dmin = 7.5 d; Q5_K: q 0..31, dmin = 15.5 d; Q6_K: d sc (q - 32), sc -16..15)
SCALE = {G.Q4_K: (1.0e-4, 7.5e-4), G.Q5_K: (7.0e-5, 1.085e-3), G.Q6_K: (1.5e-4, None)}


def f16_bytes(x):
    return np.frombuffer(np.float16(x).tobytes(), np.uint8)


def rng_of(name):
    return np.random.default_rng(zlib.crc32(name.encode()) ^ 0x5157)


def kq_chunks(name, typ, nrows, ncols, rows_per=4096):
    """random K-quant rows of a (nrows, ncols) matrix, in chunks of rows_per rows: bytes"""
    rng = rng_of(name)
    vals, bb = G.BLOCK[typ]
    nb = ncols // vals
    d, dmin = SCALE[typ]
    for r0 in range(0, nrows, rows_per):
        n = min(rows_per, nrows - r0) * nb
        b = np.frombuffer(rng.bytes(n * bb), np.uint8).reshape(n, bb).copy()
        if typ == G.Q6_K:  # ql[128], qh[64], scales[16] (int8), d
            b[:, 192:208] = ((b[:, 192:208] & 31).astype(np.int16) - 16).astype(np.int8).view(np.uint8)
            b[:, 208:210] = f16_bytes(d)
        else:  # Q4_K / Q5_K: d, dmin, scales[12], ...
            b[:, 0:2] = f16_bytes(d)
            b[:, 2:4] = f16_bytes(dmin)
        yield b.tobytes()


def f32_data(name, n, kind):
    rng = rng_of(name)
    if kind == "ones":
        x = np.ones(n, np.float32)
    elif kind == "neg":  # ssm_a = -exp(A_log)
        x = -np.exp(rng.uniform(-1.0, 1.5, n))
    else:
        x = rng.normal(0.0, kind, n)
    return [np.asarray(x, "<f4").tobytes()]


def tensors(hp, cls):
    """[(name, ne, type, nbytes, chunks())] in the real files' order"""
    dim, ffn, H, KVH, hd, V = hp["dim"], hp["ffn"], hp["H"], hp["KVH"], hp["hd"], hp["V"]
    nk, nv, dk, inner, K = hp["nk"], hp["nv"], hp["dk"], hp["inner"], hp["K"]
    cd = 2 * nk * dk + inner
    out = []

    def mat(name, k, d, typ):  # (ne = (columns, rows), as GGUF stores them)
        out.append((name, (k, d), typ, d * G.row_bytes(typ, k), lambda: kq_chunks(name, typ, d, k)))

    def vec(name, ne, kind):
        n = int(np.prod(ne))
        out.append((name, ne, G.F32, 4 * n, lambda: f32_data(name, n, kind)))

    mat("output.weight", dim, V, G.Q4_K if cls == 4 else G.Q6_K)
    vec("output_norm.weight", (dim,), "ones")
    mat("token_embd.weight", dim, V, G.Q4_K)
    for l in range(TRUNK + 1):
        s = f"blk.{l}."
        if l < TRUNK and (l + 1) % TRUNK:  # linear attention (DeltaNet)
            mat(s + "attn_gate.weight", dim, inner, G.Q4_K)
            vec(s + "attn_norm.weight", (dim,), "ones")
            mat(s + "attn_qkv.weight", dim, cd, QKV[l])
            mat(s + "ffn_down.weight", ffn, dim, DOWN[l])
            mat(s + "ffn_gate.weight", dim, ffn, G.Q4_K)
            mat(s + "ffn_up.weight", dim, ffn, G.Q4_K)
            vec(s + "post_attention_norm.weight", (dim,), "ones")
            vec(s + "ssm_a", (nv,), "neg")
            vec(s + "ssm_alpha.weight", (dim, nv), 0.01)
            vec(s + "ssm_beta.weight", (dim, nv), 0.01)
            vec(s + "ssm_conv1d.weight", (K, cd), 0.2)
            vec(s + "ssm_dt.bias", (nv,), 0.5)
            vec(s + "ssm_norm.weight", (dk,), "ones")
            mat(s + "ssm_out.weight", inner, dim, G.Q5_K)
            continue
        mtp = l == TRUNK  # the MTP block: all Q6_K
        if mtp:
            mat(s + "nextn.eh_proj.weight", 2 * dim, dim, G.Q6_K)
        q = lambda t: G.Q6_K if mtp else t  # noqa: E731
        mat(s + "ffn_down.weight", ffn, dim, q(DOWN.get(l, G.Q6_K)))
        mat(s + "ffn_gate.weight", dim, ffn, q(G.Q4_K))
        mat(s + "ffn_up.weight", dim, ffn, q(G.Q4_K))
        mat(s + "attn_k.weight", dim, KVH * hd, q(G.Q4_K))
        mat(s + "attn_output.weight", H * hd, dim, q(G.Q4_K))
        mat(s + "attn_q.weight", dim, 2 * H * hd, q(G.Q4_K))
        mat(s + "attn_v.weight", dim, KVH * hd, q(G.Q6_K))
        vec(s + "attn_norm.weight", (dim,), "ones")
        vec(s + "post_attention_norm.weight", (dim,), "ones")
        vec(s + "attn_k_norm.weight", (hd,), "ones")
        vec(s + "attn_q_norm.weight", (hd,), "ones")
        if mtp:
            for n in ("shared_head_norm", "enorm", "hnorm"):
                vec(s + f"nextn.{n}.weight", (dim,), "ones")
    return out


def write(path, kvs, ts, alignment=32):
    """GGUF v3, the tensors' data streamed (G.write_gguf's layout)"""
    head = [b"GGUF", struct.pack("<IQQ", 3, len(ts), len(kvs))]
    for k, raw in kvs:
        kb = k.encode()
        head.append(struct.pack("<Q", len(kb)) + kb + raw)
    off = 0
    for name, ne, typ, nbytes, _ in ts:
        nb = name.encode()
        head.append(struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(ne)) + struct.pack(f"<{len(ne)}Q", *ne)
                    + struct.pack("<IQ", typ, off))
        off += nbytes + (-nbytes) % alignment
    h = b"".join(head)
    tmp = path + ".part"
    try:  # (a failed or interrupted write: no partial multi-GB file left behind)
        with open(tmp, "wb") as f:
            f.write(h)
            f.write(b"\0" * ((-len(h)) % alignment))
            for name, ne, typ, nbytes, chunks in ts:
                n = 0
                for c in chunks():
                    f.write(c)
                    n += len(c)
                assert n == nbytes, (name, n, nbytes)
                f.write(b"\0" * ((-nbytes) % alignment))
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cls", type=int, choices=(4, 6), required=True, help="the classifier's type: Q4_K or Q6_K")
    ap.add_argument("--hdr", default=HDR, help="the 27B file whose header gives the hyper-parameters")
    ap.add_argument("--tok", default=TOK, help="the file whose tokenizer and template are copied")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    path = os.path.join(a.out, f"synth-q38-wide-cls{a.cls}.gguf")
    if os.path.exists(path) and not a.force:
        print(path)
        return 0
    for p in (a.hdr, a.tok):
        if not os.path.exists(p):
            print(f"missing {p}", file=sys.stderr)
            return 2
    A = "qwen35."
    fh, ft = G.GgufFile(a.hdr), G.GgufFile(a.tok)
    kv = fh.kv
    hp = {"dim": int(kv[A + "embedding_length"]), "ffn": int(kv[A + "feed_forward_length"]),
          "H": int(kv[A + "attention.head_count"]), "KVH": int(kv[A + "attention.head_count_kv"]),
          "hd": int(kv[A + "attention.key_length"]), "V": int(fh.tensors["token_embd.weight"][0][1]),
          "nk": int(kv[A + "ssm.group_count"]), "nv": int(kv[A + "ssm.time_step_rank"]), "dk": int(kv[A + "ssm.state_size"]),
          "inner": int(kv[A + "ssm.inner_size"]), "K": int(kv[A + "ssm.conv_kernel"])}
    assert int(kv[A + "full_attention_interval"]) == TRUNK and int(kv.get(A + "nextn_predict_layers", 0)) == 1, "not the 27B's layout"
    # the tokenizer: the 0.8B's, the same vocabulary as the 27B's
    for k in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges"):
        assert ft.kv[k] == fh.kv[k], f"{k} differs from {a.hdr}'s"
    assert np.array_equal(ft.kv["tokenizer.ggml.token_type"], fh.kv["tokenizer.ggml.token_type"]), "token types differ"
    assert len(ft.kv["tokenizer.ggml.tokens"]) == hp["V"]
    kvs = [("general.architecture", G.raw_kv(G.GV_STR, "qwen35")),
           ("general.name", G.raw_kv(G.GV_STR, f"gwenedeg-synth-q38-wide-cls{a.cls} (random weights in Qwen3.8-27B's dimensions)"))]
    for key, raw in fh.kv_raw:
        if key.startswith(A):
            if key == A + "block_count":
                raw = G.raw_kv(struct.unpack_from("<I", raw)[0], TRUNK + 1)
            kvs.append((key, raw))
    kvs += [(k, raw) for k, raw in ft.kv_raw if k.startswith("tokenizer.")]
    ts = tensors(hp, a.cls)
    total = sum(t[3] for t in ts)
    os.makedirs(a.out, exist_ok=True)
    free = shutil.disk_usage(a.out).free
    if free - total < KEEP_FREE:
        print(f"{path}: {total / 1e9:.2f} GB would leave {(free - total) / 1e9:.1f} GB free (< {KEEP_FREE / 1e9:.0f}): not written",
              file=sys.stderr)
        return 3
    write(path, kvs, ts)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
