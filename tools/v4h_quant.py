#!/usr/bin/env python3
"""V4H weight quantizer, quality prototype (docs/v4h.md): BF16 / F16 / F32 GGUF -> a Q8_0 *carrier* GGUF.

V4H per output row and 256-column superblock: an fp16 super-scale D, per group of 32 an unsigned 4-bit subscale s (0..15)
and 32 codes q in -8..7; w = D * s * q. No kernel or file type exists yet, so the quantized weights are written into a Q8_0
file of the same model (TEMPLATE: llama-quantize's Q8_0 of the same source, same tensor set): each Q8_0 block of 32 gets
d = fp16(D * s) and the int8 values q. Q8_0 then dequantizes to D * s * q up to the fp16 rounding of the product (relative
<= 2^-11, far below the 4-bit error), so any engine that loads Q8_0 (llama.cpp, gwenedeg) evaluates V4H's weight quality.
Only the tensors TEMPLATE-compatible and selected (--roles; default every 2-D weight that llama-quantize quantized at the
model's main type, i.e. not token_embd / ssm_alpha / ssm_beta) are replaced; the rest stay as in TEMPLATE.

Variants (--variant):
  v4h      the format above, 4.1875 bits/weight
  flat256  V4-flat: one fp16 scale per row and 256 columns, no subscales, 4.0625 bits/weight
  s6       ablation: V4H with 6-bit subscales (0..63; not int16-safe in the kernel, only to measure the 4-bit cost)

Quantizer (llama.cpp-style, numpy): per-value importance weights w = imatrix[col] * sqrt(sigma2 + x^2) with an imatrix
(sigma2 = 2 * mean(x^2) of the superblock), else x^2 + small (IQ4_XS's convention); per group of 32 a search over
candidate scales with a least-squares refit (the best weighted squared error), then per superblock a search over D (fp16)
around max(d_g) / smax with the subscales rounded, q requantized and D refit by least squares.

Usage: .venv/bin/python -I tools/v4h_quant.py SRC.gguf TEMPLATE_Q8_0.gguf OUT.gguf [--imatrix IM.gguf] [--variant v4h]
(gguf-py on PYTHONPATH: llama.cpp's gguf-py directory.)
"""
import argparse
import shutil
import sys
import time

import numpy as np
import gguf

SB, G = 256, 32  # superblock and group sizes


def f16(x):
    return np.asarray(x, np.float32).astype(np.float16).astype(np.float32)


def load_f32(t):
    """a 2-D GGUF tensor as float32 (rows = ne1, cols = ne0)"""
    tt = t.tensor_type
    a = np.asarray(t.data)
    if tt == gguf.GGMLQuantizationType.BF16:
        a = (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif tt == gguf.GGMLQuantizationType.F16:
        a = a.view(np.float16).astype(np.float32)
    elif tt == gguf.GGMLQuantizationType.F32:
        a = a.astype(np.float32)
    else:
        raise SystemExit(f"{t.name}: source type {tt.name} (need BF16 / F16 / F32)")
    ne0, ne1 = int(t.shape[0]), int(t.shape[1])
    return a.reshape(ne1, ne0)


def load_imatrix(path):
    if not path:
        return {}
    r = gguf.GGUFReader(path)
    s2, cnt = {}, {}
    for t in r.tensors:
        if t.name.endswith(".in_sum2"):
            s2[t.name[:-8]] = np.asarray(t.data, np.float32).reshape(-1)
        elif t.name.endswith(".counts"):
            cnt[t.name[:-7]] = float(np.asarray(t.data, np.float32).reshape(-1)[0])
    return {k: v / max(cnt.get(k, 1.0), 1.0) for k, v in s2.items()}


def importance(x, qw):
    """x (n, 8, 32) superblocks; qw None or (n, 8, 32) imatrix values of each column"""
    if qw is None:
        return x * x + 1e-12
    sigma2 = 2.0 * (x * x).mean(axis=(1, 2), keepdims=True)
    return qw * np.sqrt(sigma2 + x * x) + 1e-12


def group_scales(x, w, qmin=-8, qmax=7):
    """best per-group scale d >= 0 for codes in qmin..qmax (x, w: (n, 8, 32)) -> d (n, 8)"""
    amax = np.abs(x).max(axis=2)  # (n, 8)
    best_e = np.full(amax.shape, np.inf, np.float32)
    best_d = np.zeros(amax.shape, np.float32)
    for div in np.arange(6.0, 9.51, 0.125):
        d0 = amax / div
        inv = np.where(d0 > 0, 1.0 / np.where(d0 > 0, d0, 1), 0)
        q = np.clip(np.rint(x * inv[..., None]), qmin, qmax)
        sxq = (w * x * q).sum(2)
        sqq = (w * q * q).sum(2)
        d = np.where(sqq > 0, sxq / np.where(sqq > 0, sqq, 1), 0)
        d = np.maximum(d, 0)
        e = (w * (x - d[..., None] * q) ** 2).sum(2)
        better = e < best_e
        best_e = np.where(better, e, best_e)
        best_d = np.where(better, d, best_d)
    return best_d


def quant_with(x, w, D, smax, sub=True, qmin=-8, qmax=7):
    """given D (n,) fp16-exact: subscales s (n, 8), codes q (n, 8, 32), weighted error (n,)"""
    if sub:
        dg = quant_with.dg
        s = np.clip(np.rint(dg / np.where(D > 0, D, 1)[:, None]), 0, smax)
        s = np.where(D[:, None] > 0, s, 0)
    else:
        s = np.ones(x.shape[:2], np.float32)
    sc = D[:, None] * s  # (n, 8)
    inv = np.where(sc > 0, 1.0 / np.where(sc > 0, sc, 1), 0)
    q = np.clip(np.rint(x * inv[..., None]), qmin, qmax)
    e = (w * (x - sc[..., None] * q) ** 2).sum(axis=(1, 2))
    return s, q, e


def quant_sb(x, w, variant):
    """x, w (n, 8, 32) -> D (n,) fp16 values, s (n, 8), q (n, 8, 32)"""
    sub = variant != "flat256"
    smax = 63 if variant == "s6" else 15
    if sub:
        dg = group_scales(x, w)
        quant_with.dg = dg
        base = dg.max(axis=1) / smax
    else:
        xs, ws = x.reshape(-1, 1, SB), w.reshape(-1, 1, SB)
        base = group_scales(xs, ws)[:, 0]
    best = None
    for f in np.arange(0.80, 1.201, 0.025):
        D = f16(base * f)
        s, q, _ = quant_with(x, w, D, smax, sub)
        # least-squares refit of D for these s, q; requantize once
        sq = s[..., None] * q
        num, den = (w * x * sq).sum(axis=(1, 2)), (w * sq * sq).sum(axis=(1, 2))
        D2 = f16(np.where(den > 0, num / np.where(den > 0, den, 1), D))
        D2 = np.maximum(D2, 0)
        for Dc in (D, D2):
            s, q, e = quant_with(x, w, Dc, smax, sub)
            if best is None:
                best = [Dc, s, q, e]
            else:
                m = e < best[3]
                best[0] = np.where(m, Dc, best[0])
                best[1] = np.where(m[:, None], s, best[1])
                best[2] = np.where(m[:, None, None], q, best[2])
                best[3] = np.where(m, e, best[3])
    return best[0], best[1], best[2]


def to_q8_0(D, s, q):
    """carrier Q8_0 blocks (n*8 blocks of 34 bytes) for superblocks D (n,), s (n, 8), q (n, 8, 32)"""
    n = D.shape[0]
    d = (D[:, None] * s).astype(np.float32).astype(np.float16)  # (n, 8)
    out = np.empty((n * 8, 34), np.uint8)
    out[:, 0:2] = d.reshape(-1).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.reshape(-1, 32).astype(np.int8).view(np.uint8)
    return out


def default_role(name):
    return name.endswith(".weight") and not any(k in name for k in ("token_embd", "output.weight", "ssm_alpha",
                                                                  "ssm_beta", "norm", "conv1d"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("template")
    ap.add_argument("out")
    ap.add_argument("--imatrix", default="")
    ap.add_argument("--variant", default="v4h", choices=["v4h", "flat256", "s6"])
    ap.add_argument("--roles", default="", help="comma-separated substrings of tensor names (default: all matrices)")
    ap.add_argument("--rows", type=int, default=256, help="rows per numpy chunk")
    a = ap.parse_args()
    im = load_imatrix(a.imatrix)
    src = gguf.GGUFReader(a.src)
    srct = {t.name: t for t in src.tensors}
    shutil.copyfile(a.template, a.out)
    tpl = gguf.GGUFReader(a.out)
    roles = [r for r in a.roles.split(",") if r]
    t0 = time.time()
    nw, err, ref = 0, 0.0, 0.0
    with open(a.out, "r+b") as fo:
        for t in tpl.tensors:
            if t.tensor_type != gguf.GGMLQuantizationType.Q8_0 or len(t.shape) != 2:
                continue
            if not (any(r in t.name for r in roles) if roles else default_role(t.name)):
                continue
            ne0, ne1 = int(t.shape[0]), int(t.shape[1])
            if ne0 % SB:
                print(f"skip {t.name}: {ne0} columns", file=sys.stderr)
                continue
            W = load_f32(srct[t.name])
            assert W.shape == (ne1, ne0), (t.name, W.shape)
            qw_row = im.get(t.name)
            if a.imatrix and qw_row is None:
                print(f"warning: no imatrix entry for {t.name}", file=sys.stderr)
            blocks = []
            for r0 in range(0, ne1, a.rows):
                x = W[r0:r0 + a.rows].reshape(-1, 8, G)
                qw = None
                if qw_row is not None:
                    qw = np.broadcast_to(qw_row.reshape(1, -1), (min(a.rows, ne1 - r0), ne0)).reshape(-1, 8, G)
                w = importance(x, qw)
                D, s, q = quant_sb(x, w, a.variant)
                deq = (f16((D[:, None] * s)))[..., None] * q
                err += float(((x - deq) ** 2).sum())
                ref += float((x * x).sum())
                blocks.append(to_q8_0(D, s, q))
            data = np.concatenate(blocks).reshape(-1)
            assert data.size == t.n_bytes, (t.name, data.size, t.n_bytes)
            fo.seek(t.data_offset)
            fo.write(data.tobytes())
            nw += ne0 * ne1
            print(f"{t.name:32s} {ne1}x{ne0}  rel rms err so far {np.sqrt(err / ref):.4f}  {time.time() - t0:.0f}s",
                  file=sys.stderr)
    print(f"{a.variant}: {nw} weights quantized, relative RMS error {np.sqrt(err / ref):.4f}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
