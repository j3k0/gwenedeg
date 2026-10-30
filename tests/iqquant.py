"""A simple quantizer to Q2_K and the i-quants (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS), numpy only:
the synthetic test models' weights (tests/make_synth_q35.py: synth-q35-iq.gguf). Not llama.cpp's quantizers (no
importance matrix, no search over scales): every block it writes is a valid block of its format (ggml's layouts,
ggml-common.h), dequantized by tests/ggufmini.py and runq exactly as llama.cpp would; the values are close enough to the
source weights to keep a forward pass's statistics realistic (relative RMS error ~0.1 .. 0.6 depending on the type).

Per group of values the scale maps the largest magnitude to the largest code; each value is rounded to the nearest value
of its type's alphabet, and a grid type's chunk (8 or 4 values) takes the grid entry nearest to that pattern (a table
over all alphabet patterns: 3^8 for IQ2 / IQ1, 8^4 for IQ3, built once). The signs that ggml stores as 7 bits with even
parity (IQ2_XXS / IQ2_XS / IQ3_XXS): the smallest value of a chunk with odd parity takes the other sign.

quantize(typ, x): x float32 (rows, cols), cols % 256 == 0 -> uint8 (rows, cols / 256 * block bytes).
"""
import numpy as np

import ggufmini as G

_NEAR = {}


def _f16(v):
    """fp16 of float32 values (round to nearest), as (uint16 bits, the float32 value)"""
    h = np.asarray(v, np.float32).astype(np.float16)
    return h.view(np.uint16), h.astype(np.float32)


def _nearest_entry(name, alpha, nv):
    """for every pattern of nv codes (base len(alpha), code j of value j) the index of the nearest entry of the grid
    `name` (squared distance of the values; ties: the lowest index)"""
    key = (name, nv)
    if key not in _NEAR:
        g = G.grids()[name + "_b"].astype(np.int64)
        a = np.array(alpha, np.int64)
        A = len(alpha)
        codes = (np.arange(A ** nv)[:, None] // (A ** np.arange(nv))[None, :]) % A   # (patterns, nv)
        pv = a[codes]                                                                  # pattern values
        # (integers, exact: the same argmin on every machine, so the synthetic files rebuild byte for byte)
        dist = (pv ** 2).sum(1)[:, None] - 2 * np.matmul(pv, g.T) + (g ** 2).sum(1)[None, :]
        _NEAR[key] = np.argmin(dist, axis=1).astype(np.int64)
    return _NEAR[key]


def _codes(t, alpha):
    """the nearest alphabet index of every value of t (alpha ascending)"""
    a = np.asarray(alpha, np.float32)
    mids = (a[1:] + a[:-1]) / 2
    return np.searchsorted(mids, t).astype(np.int64)


def _pattern(c, A):
    """codes (..., nv) -> pattern index"""
    return (c * (A ** np.arange(c.shape[-1], dtype=np.int64))).sum(-1)


def _grid_pick(t, name, alpha):
    """t (..., nv) target values -> grid entry indices (...)"""
    nv = t.shape[-1]
    return _nearest_entry(name, alpha, nv)[_pattern(_codes(t, alpha), len(alpha))]


def _parity_fix(neg, mag):
    """neg (..., 8) bool, mag (..., 8): flip the sign of the smallest value of chunks with an odd count of negatives
    (ksigns: the 8th sign is the parity of the other 7)"""
    odd = (neg.sum(-1) & 1) == 1
    j = np.argmin(mag, axis=-1)
    flip = np.zeros_like(neg)
    np.put_along_axis(flip, j[..., None], odd[..., None], axis=-1)
    return neg ^ flip


def _scales(amax, top, maxs, mult, add):
    """per sub-block scales s in 0..maxs with (mult * s + add) * D ~ amax / top: (D float32 per super-block (fp16-rounded),
    its fp16 bits, s int64)"""
    want = amax / top                                                     # (nb, nsub)
    dbits, D = _f16(want.max(1) / (mult * maxs + add))
    Ds = np.where(D == 0, np.float32(1), D)[:, None]
    s = np.clip(np.rint((want / Ds - add) / mult), 0, maxs).astype(np.int64)
    return D, dbits, s


def quantize(typ, x):
    x = np.asarray(x, np.float32)
    nrows, ncols = x.shape
    assert ncols % 256 == 0
    nb = nrows * ncols // 256
    xb = x.reshape(nb, 256)
    bb = G.BLOCK[typ][1]
    out = np.zeros((nb, bb), np.uint8)
    mag, neg = np.abs(xb), xb < 0
    if typ == G.Q2_K:
        xs = xb.reshape(nb, 16, 16)
        lo, hi = np.minimum(xs.min(2), 0), xs.max(2)
        dbits, d = _f16((hi - lo).max(1) / 3 / 15)
        mbits, dm = _f16((-lo).max(1) / 15)
        dS, dmS = np.where(d == 0, 1, d)[:, None], np.where(dm == 0, 1, dm)[:, None]
        sc = np.clip(np.rint((hi - lo) / 3 / dS), 0, 15).astype(np.int64)
        m = np.clip(np.rint(-lo / dmS), 0, 15).astype(np.int64)
        step = (d[:, None] * sc)[:, :, None]
        q = np.where(step > 0, np.rint((xs + (dm[:, None] * m)[:, :, None]) / np.where(step > 0, step, 1)), 0)
        q = np.clip(q, 0, 3).astype(np.uint8).reshape(nb, 256)
        out[:, 0:16] = (sc | m << 4).astype(np.uint8)
        qv = q.reshape(nb, 2, 4, 32)                       # value 128 n + 32 p + l -> qs[32 n + l] bits 2 p
        out[:, 16:80] = (qv << (2 * np.arange(4, dtype=np.uint8)).reshape(1, 1, 4, 1)).sum(2).astype(np.uint8).reshape(nb, 64)
        out[:, 80:82] = dbits.reshape(nb, 1).view(np.uint8)
        out[:, 82:84] = mbits.reshape(nb, 1).view(np.uint8)
    elif typ == G.IQ4_XS:
        kv = G.grids()["kvalues_iq4nl"].astype(np.float32)
        xs = xb.reshape(nb, 8, 32)
        D, dbits, s = _scales(np.abs(xs).max(2), 127, 31, 1, 0)
        ls = 32 + s
        dl = (D[:, None] * s.astype(np.float32))[:, :, None]
        t = xs / np.where(dl == 0, 1, dl)
        q = np.argmin(np.abs(t[..., None] - kv), axis=-1).astype(np.uint8)     # (nb, 8, 32)
        out[:, 0:2] = dbits.reshape(nb, 1).view(np.uint8)
        sh = ((ls >> 4) << (2 * np.arange(8))).sum(1).astype(np.uint16)
        out[:, 2:4] = sh.reshape(nb, 1).view(np.uint8)
        out[:, 4:8] = ((ls[:, 0::2] & 15) | (ls[:, 1::2] & 15) << 4).astype(np.uint8)
        out[:, 8:136] = (q[:, :, :16] | q[:, :, 16:] << 4).reshape(nb, 128)
    elif typ in (G.IQ2_XXS, G.IQ2_XS, G.IQ2_S):
        alpha, name = [8, 25, 43], {G.IQ2_XXS: "iq2xxs_grid", G.IQ2_XS: "iq2xs_grid", G.IQ2_S: "iq2s_grid"}[typ]
        per16 = typ != G.IQ2_XXS
        m = mag.reshape(nb, 16 if per16 else 8, 16 if per16 else 32)
        D, dbits, s = _scales(m.max(2), 43 / 8, 15, 2, 1)                    # db = d (2 s + 1) / 8 (D = d)
        d8 = D / 8
        db = (d8[:, None] * (2 * s + 1).astype(np.float32))
        db = np.repeat(db, 16 if per16 else 32, axis=1).reshape(nb, 32, 8)    # per chunk of 8
        t = mag.reshape(nb, 32, 8) / np.where(db == 0, 1, db)
        idx = _grid_pick(t, name, alpha)                                       # (nb, 32 chunks)
        ng = neg.reshape(nb, 32, 8)
        if typ != G.IQ2_S:
            ng = _parity_fix(ng, mag.reshape(nb, 32, 8))
        sbits = (ng.astype(np.int64) << np.arange(8)).sum(-1)                  # (nb, 32) sign byte of the chunk
        out[:, 0:2] = dbits.reshape(nb, 1).view(np.uint8)
        if typ == G.IQ2_XXS:
            s7 = (sbits & 127).reshape(nb, 8, 4)
            a1 = ((s7 << (7 * np.arange(4))).sum(-1) | s << 28).astype(np.uint32)   # (nb, 8)
            g8 = np.zeros((nb, 8, 8), np.uint8)
            g8[:, :, 0:4] = idx.reshape(nb, 8, 4).astype(np.uint8)
            g8[:, :, 4:8] = a1.reshape(nb, 8, 1).view(np.uint8)
            out[:, 2:66] = g8.reshape(nb, 64)
        else:
            sc = s.reshape(nb, 8, 2)
            scb = (sc[:, :, 0] | sc[:, :, 1] << 4).astype(np.uint8)
            if typ == G.IQ2_XS:
                q = (idx | (sbits & 127) << 9).astype(np.uint16)
                out[:, 2:66] = q.view(np.uint8).reshape(nb, 64)
                out[:, 66:74] = scb
            else:
                out[:, 2:34] = (idx & 255).astype(np.uint8)
                out[:, 34:66] = sbits.astype(np.uint8)
                out[:, 66:74] = ((idx.reshape(nb, 8, 4) >> 8) << (2 * np.arange(4))).sum(-1).astype(np.uint8)
                out[:, 74:82] = scb
    elif typ in (G.IQ3_XXS, G.IQ3_S):
        xxs = typ == G.IQ3_XXS
        alpha = [4, 12, 20, 28, 36, 44, 52, 62] if xxs else [1, 3, 5, 7, 9, 11, 13, 15]
        top = alpha[-1]
        # IQ3_XXS: db = D (2 s + 1) / 4; IQ3_S: db = D (2 s + 1)
        D, dbits, s = _scales(mag.reshape(nb, 8, 32).max(2), top / 4 if xxs else top, 15, 2, 1)
        unit = D / 4 if xxs else D
        db = np.repeat(unit[:, None] * (2 * s + 1).astype(np.float32), 32, axis=1).reshape(nb, 64, 4)
        t = mag.reshape(nb, 64, 4) / np.where(db == 0, 1, db)
        idx = _grid_pick(t, "iq3xxs_grid" if xxs else "iq3s_grid", alpha)     # (nb, 64 steps)
        ng = neg.reshape(nb, 32, 8)
        if xxs:
            ng = _parity_fix(ng, mag.reshape(nb, 32, 8))
        sbits = (ng.astype(np.int64) << np.arange(8)).sum(-1)                  # (nb, 32 chunks)
        out[:, 0:2] = dbits.reshape(nb, 1).view(np.uint8)
        out[:, 2:66] = (idx & 255).astype(np.uint8)
        if xxs:
            s7 = (sbits & 127).reshape(nb, 8, 4)
            a = ((s7 << (7 * np.arange(4))).sum(-1) | s << 28).astype(np.uint32)
            out[:, 66:98] = a.view(np.uint8).reshape(nb, 32)
        else:
            out[:, 66:74] = ((idx.reshape(nb, 8, 8) >> 8) << np.arange(8)).sum(-1).astype(np.uint8)
            out[:, 74:106] = sbits.astype(np.uint8)
            out[:, 106:110] = (s[:, 0::2] | s[:, 1::2] << 4).astype(np.uint8)
    elif typ in (G.IQ1_S, G.IQ1_M):
        # w = D (2 s + 1) (grid + delta), delta = +-1/8 (IQ1_S per 32 values, IQ1_M per 8): the scale from the largest
        # magnitude (-> 1.125), the delta's sign from the chunk's / group's mean
        m1 = typ == G.IQ1_M
        sub = 16 if m1 else 32
        D, dbits, s = _scales(mag.reshape(nb, 256 // sub, sub).max(2), 1.125, 7, 2, 1)
        dl = np.repeat(D[:, None] * (2 * s + 1).astype(np.float32), sub, axis=1).reshape(nb, 32, 8)
        t = xb.reshape(nb, 32, 8) / np.where(dl == 0, 1, dl)
        if m1:
            dneg = t.mean(-1) < 0                                              # (nb, 32 chunks)
        else:
            dneg = np.repeat(t.reshape(nb, 8, 32).mean(-1) < 0, 4, axis=1)     # (nb, 32)
        t = t - np.where(dneg, -0.125, 0.125)[..., None]
        idx = _grid_pick(t, "iq1s_grid", [-1, 0, 1])                          # (nb, 32)
        hi3 = (idx >> 8) & 7
        if not m1:
            out[:, 0:2] = dbits.reshape(nb, 1).view(np.uint8)
            out[:, 2:34] = (idx & 255).astype(np.uint8)
            qh = (hi3.reshape(nb, 8, 4) << (3 * np.arange(4))).sum(-1) | s << 12 | dneg.reshape(nb, 8, 4)[:, :, 0].astype(np.int64) << 15
            out[:, 34:50] = qh.astype(np.uint16).view(np.uint8).reshape(nb, 16)
        else:
            out[:, 0:32] = (idx & 255).astype(np.uint8)
            nib = hi3 | dneg.astype(np.int64) << 3                             # (nb, 32 chunks)
            out[:, 32:48] = (nib[:, 0::2] | nib[:, 1::2] << 4).astype(np.uint8)
            s2 = s.reshape(nb, 8, 2)                                           # (group, sub-block of 16)
            sc = np.zeros((nb, 4), np.int64)
            for ib in range(8):
                sc[:, ib // 2] |= (s2[:, ib, 0] | s2[:, ib, 1] << 3) << (6 * (ib % 2))
            db = dbits.astype(np.int64)
            for j in range(4):
                sc[:, j] |= ((db >> (4 * j)) & 15) << 12
            out[:, 48:56] = sc.astype(np.uint16).view(np.uint8).reshape(nb, 8)
    else:
        raise ValueError(f"no quantizer for ggml type {typ}")
    return out.reshape(nrows, ncols // 256 * bb)
