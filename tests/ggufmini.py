"""A minimal GGUF reader / writer and numpy dequantizer for the tests (numpy only; no gguf-py).

GgufFile(path): the metadata (raw byte spans and decoded scalars / strings / arrays) and the tensor table of
a GGUF v2 / v3 file (layout: src/gguf.c), the tensor data as numpy views of a memory map.
write_gguf(path, kvs, tensors): a GGUF v3 file from (key, raw value bytes) pairs (raw_kv / copied spans)
and (name, ne, ggml type, data bytes) tuples.
dequant(type, data, nrows, ncols): Q8_0 / Q2_K .. Q6_K / the i-quants (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
IQ3_S, IQ4_XS) / F32 rows to float32 with ggml's expressions (dequantize_row_q8_0 / _q3_K / .. / _iq4_xs: the same float32
operations as gguf-py, bit for bit; checked against tests/golden/dequant; the i-quant codebooks are read from
src/iq_grids.h). Used by tests/make_synth.py (the synthetic models), tests/ref_forward.py (the numpy reference forward
pass) and tests/check_models.py.
"""
import os
import re
import struct

import numpy as np

GV_U8, GV_I8, GV_U16, GV_I16, GV_U32, GV_I32, GV_F32, GV_BOOL, GV_STR, GV_ARR, GV_U64, GV_I64, GV_F64 = range(13)
_SCALAR = {GV_U8: "<B", GV_I8: "<b", GV_U16: "<H", GV_I16: "<h", GV_U32: "<I", GV_I32: "<i", GV_F32: "<f",
           GV_BOOL: "<?", GV_U64: "<Q", GV_I64: "<q", GV_F64: "<d"}
F32, F16, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K = 0, 1, 8, 11, 12, 13, 14
Q2_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ3_S, IQ2_S, IQ4_XS, IQ1_M = 10, 16, 17, 18, 19, 21, 22, 23, 29
IQ_TYPES = (Q2_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ3_S, IQ2_S, IQ4_XS, IQ1_M)  # (src/iquants.h)
BLOCK = {F32: (1, 4), F16: (1, 2), Q8_0: (32, 34), Q3_K: (256, 110), Q4_K: (256, 144), Q5_K: (256, 176),
         Q6_K: (256, 210), Q2_K: (256, 84), IQ2_XXS: (256, 66), IQ2_XS: (256, 74), IQ3_XXS: (256, 98), IQ1_S: (256, 50),
         IQ3_S: (256, 110), IQ2_S: (256, 82), IQ4_XS: (256, 136), IQ1_M: (256, 56)}  # values, bytes
TYPE_NAME = {F32: "F32", F16: "F16", Q8_0: "Q8_0", Q3_K: "Q3_K", Q4_K: "Q4_K", Q5_K: "Q5_K", Q6_K: "Q6_K", Q2_K: "Q2_K",
             IQ2_XXS: "IQ2_XXS", IQ2_XS: "IQ2_XS", IQ3_XXS: "IQ3_XXS", IQ1_S: "IQ1_S", IQ3_S: "IQ3_S", IQ2_S: "IQ2_S",
             IQ4_XS: "IQ4_XS", IQ1_M: "IQ1_M"}


def row_bytes(t, ncols):
    v, b = BLOCK[t]
    assert ncols % v == 0
    return ncols // v * b


class GgufFile:
    def __init__(self, path):
        self.path = path
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        b = memoryview(self.mm)
        self.buf = b
        assert bytes(b[:4]) == b"GGUF", "not a GGUF file"
        self.version, n_t, n_kv = struct.unpack_from("<IQQ", b, 4)
        pos = 24
        self.kv = {}      # key -> decoded value
        self.kv_raw = []  # (key, raw bytes of type + value), in file order
        self.alignment = 32
        for _ in range(n_kv):
            klen = struct.unpack_from("<Q", b, pos)[0]
            key = bytes(b[pos + 8:pos + 8 + klen]).decode()
            pos += 8 + klen
            start = pos
            val, pos = self._value(pos)
            self.kv[key] = val
            self.kv_raw.append((key, bytes(b[start:pos])))
            if key == "general.alignment":
                self.alignment = int(val)
        self.tensors = {}  # name -> (ne tuple, type, offset in the data section)
        self.order = []
        for _ in range(n_t):
            nlen = struct.unpack_from("<Q", b, pos)[0]
            name = bytes(b[pos + 8:pos + 8 + nlen]).decode()
            pos += 8 + nlen
            nd = struct.unpack_from("<I", b, pos)[0]
            ne = struct.unpack_from(f"<{nd}Q", b, pos + 4)
            pos += 4 + 8 * nd
            typ, off = struct.unpack_from("<IQ", b, pos)
            pos += 12
            self.tensors[name] = (tuple(ne), typ, off)
            self.order.append(name)
        self.data_off = pos + (-pos) % self.alignment

    def _value(self, pos):
        b = self.buf
        typ = struct.unpack_from("<I", b, pos)[0]
        pos += 4
        if typ == GV_ARR:
            et, n = struct.unpack_from("<IQ", b, pos)
            pos += 12
            if et == GV_STR:
                out = []
                for _ in range(n):
                    ln = struct.unpack_from("<Q", b, pos)[0]
                    out.append(bytes(b[pos + 8:pos + 8 + ln]))
                    pos += 8 + ln
                return out, pos
            fmt = _SCALAR[et]
            sz = struct.calcsize(fmt)
            arr = np.frombuffer(b, dtype=np.dtype(fmt), count=n, offset=pos)
            return arr, pos + n * sz
        if typ == GV_STR:
            ln = struct.unpack_from("<Q", b, pos)[0]
            return bytes(b[pos + 8:pos + 8 + ln]).decode("utf-8", "surrogateescape"), pos + 8 + ln
        fmt = _SCALAR[typ]
        return struct.unpack_from(fmt, b, pos)[0], pos + struct.calcsize(fmt)

    def tensor_bytes(self, name):
        """the tensor's data as a uint8 array of shape (rows, row bytes), rows = prod(ne[1:])"""
        ne, typ, off = self.tensors[name]
        rows = int(np.prod(ne[1:])) if len(ne) > 1 else 1
        rb = row_bytes(typ, ne[0])
        a = self.data_off + off
        return np.asarray(self.mm[a:a + rows * rb]).reshape(rows, rb)

    def rows(self, name, r0=0, n=None):
        """rows r0..r0+n-1 dequantized (float32, (n, ne[0]))"""
        ne, typ, _ = self.tensors[name]
        tb = self.tensor_bytes(name)
        n = tb.shape[0] - r0 if n is None else n
        return dequant(typ, tb[r0:r0 + n], n, ne[0])

    def take_rows(self, name, idx):
        ne, typ, _ = self.tensors[name]
        tb = self.tensor_bytes(name)
        return dequant(typ, tb[np.asarray(idx)], len(idx), ne[0])


def raw_kv(typ, value):
    """the raw bytes (type + value) of a metadata entry: a scalar, a str (GV_STR) or a list of str (GV_ARR)"""
    if typ == GV_STR:
        s = value.encode() if isinstance(value, str) else value
        return struct.pack("<IQ", GV_STR, len(s)) + s
    if typ == GV_ARR:  # (strings only)
        out = [struct.pack("<IIQ", GV_ARR, GV_STR, len(value))]
        for s in value:
            out.append(struct.pack("<Q", len(s)) + s)
        return b"".join(out)
    return struct.pack("<I", typ) + struct.pack(_SCALAR[typ], value)


def write_gguf(path, kvs, tensors, alignment=32):
    """kvs: [(key, raw value bytes)]; tensors: [(name, ne, type, bytes or None)]. A tensor of an unknown type
    with None data gets offset 0 and no data (runq's tokenizer-only loads accept it)."""
    head = [b"GGUF", struct.pack("<IQQ", 3, len(tensors), len(kvs))]
    for k, raw in kvs:
        kb = k.encode()
        head.append(struct.pack("<Q", len(kb)) + kb + raw)
    off, offs = 0, []
    for name, ne, typ, data in tensors:
        offs.append(off)
        if data is not None:
            off += len(data)
            off += (-off) % alignment
    for (name, ne, typ, data), o in zip(tensors, offs):
        nb = name.encode()
        head.append(struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(ne)) + struct.pack(f"<{len(ne)}Q", *ne)
                    + struct.pack("<IQ", typ, o if data is not None else 0))
    h = b"".join(head)
    with open(path, "wb") as f:
        f.write(h)
        f.write(b"\0" * ((-len(h)) % alignment))
        for name, ne, typ, data in tensors:
            if data is None:
                continue
            f.write(data)
            f.write(b"\0" * ((-len(data)) % alignment))


def _f16(u8pair):
    return u8pair.copy().view(np.float16).astype(np.float32)


def _q4k_scale_min(s):
    """the 6-bit scales / mins of the 8 sub-blocks from the 12 packed bytes (ggml's get_scale_min_k4) ->
    (sc, mn), each (nb, 8) int32; Q4_K and Q5_K share them"""
    nb = s.shape[0]
    sc = np.empty((nb, 8), np.int32)
    mn = np.empty((nb, 8), np.int32)
    for j in range(8):
        if j < 4:
            sc[:, j] = s[:, j] & 63
            mn[:, j] = s[:, j + 4] & 63
        else:
            sc[:, j] = (s[:, j + 4] & 0xF) | ((s[:, j - 4] >> 6) << 4)
            mn[:, j] = (s[:, j + 4] >> 4) | ((s[:, j] >> 6) << 4)
    return sc, mn


_GRIDS = None


def grids():
    """the codebooks of src/iq_grids.h as numpy arrays (name -> array; the uint64 / uint32 grids also as their bytes,
    name + "_b": (entries, 8 or 4) uint8, iq1s_grid's as int8), parsed once"""
    global _GRIDS
    if _GRIDS is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "iq_grids.h")
        src = open(path, encoding="utf-8").read()
        dts = {"uint64_t": np.uint64, "uint32_t": np.uint32, "uint16_t": np.uint16, "uint8_t": np.uint8, "int8_t": np.int8}
        g = {}
        for m in re.finditer(r"static const (\w+) (\w+)\[(\d+)\][^=]*= \{(.*?)\};", src, re.S):
            ctype, name, n, body = m.groups()
            vals = [int(v, 0) for v in re.findall(r"-?0x[0-9a-fA-F]+|-?\d+", body)]
            assert len(vals) == int(n), name
            g[name] = np.array(vals, dtype=dts[ctype])
        for name in ("iq2xxs_grid", "iq2xs_grid", "iq2s_grid", "iq3xxs_grid", "iq3s_grid", "iq1s_grid"):
            a = g[name]
            g[name + "_b"] = a.view(np.uint8).reshape(len(a), a.dtype.itemsize)
        g["iq1s_grid_b"] = g["iq1s_grid_b"].view(np.int8)
        g["ksigns"] = np.array([s | (bin(s).count("1") & 1) << 7 for s in range(128)], dtype=np.uint8)
        _GRIDS = g
    return _GRIDS


def _bits8(b):
    """(..., ) uint8 -> (..., 8) bits, bit j at j"""
    return (b[..., None] >> np.arange(8, dtype=np.uint8)) & 1


def _sgn(bits):
    return np.where(bits != 0, np.float32(-1.0), np.float32(1.0))


def _dequant_iq(typ, data, nrows, ncols):
    """Q2_K and the i-quants: ggml's dequantize_row_* (every product exact in fp32: the grouping does not matter; Q2_K's
    final subtraction is rounded once, as in ggml)"""
    G = grids()
    blk = data.reshape(-1, BLOCK[typ][1])
    nb = blk.shape[0]
    u16 = lambda a: np.ascontiguousarray(a).view(np.uint16)  # noqa: E731
    u32 = lambda a: np.ascontiguousarray(a).view(np.uint32)  # noqa: E731
    if typ == Q2_K:
        sc = blk[:, 0:16].astype(np.int32)
        qs = blk[:, 16:80].reshape(nb, 2, 1, 32)
        d, dmin = _f16(blk[:, 80:82]).reshape(-1, 1), _f16(blk[:, 82:84]).reshape(-1, 1)
        q = ((qs >> (2 * np.arange(4, dtype=np.uint8)).reshape(1, 1, 4, 1)) & 3).reshape(nb, 16, 16).astype(np.float32)
        dl = (d * (sc & 15).astype(np.float32)).reshape(nb, 16, 1)
        ml = (dmin * (sc >> 4).astype(np.float32)).reshape(nb, 16, 1)
        return (dl * q - ml).reshape(nrows, ncols)
    if typ in (IQ2_XXS, IQ2_XS, IQ2_S):
        d = _f16(blk[:, 0:2]).reshape(nb, 1, 1)
        if typ == IQ2_XXS:
            g8 = blk[:, 2:66].reshape(nb, 8, 8)
            idx = g8[:, :, 0:4].astype(np.int64)                                     # (nb, 8 groups, 4 chunks)
            a1 = u32(g8[:, :, 4:8]).reshape(nb, 8, 1)
            sb = G["ksigns"][(a1 >> (7 * np.arange(4, dtype=np.uint32))) & 127]       # (nb, 8, 4)
            db = (d * (np.float32(0.5) + (a1 >> 28).astype(np.float32)) * np.float32(0.25))  # (nb, 8, 1)
            gr = G["iq2xxs_grid_b"][idx].astype(np.float32)                         # (nb, 8, 4, 8)
            y = db[..., None] * gr * _sgn(_bits8(sb))
            return y.reshape(nrows, ncols)
        if typ == IQ2_XS:
            qs = u16(blk[:, 2:66]).reshape(nb, 8, 4).astype(np.int64)
            idx, sb, scb = qs & 511, G["ksigns"][qs >> 9], blk[:, 66:74]
            gr = G["iq2xs_grid_b"][idx].astype(np.float32)
        else:
            qs, signs, qh, scb = blk[:, 2:34], blk[:, 34:66], blk[:, 66:74], blk[:, 74:82]
            l2 = 2 * np.arange(4, dtype=np.int64)
            idx = qs.reshape(nb, 8, 4).astype(np.int64) | ((qh.astype(np.int64)[:, :, None] >> l2) & 3) << 8
            sb = signs.reshape(nb, 8, 4)
            gr = G["iq2s_grid_b"][idx].astype(np.float32)
        nib = ((scb[:, :, None] >> np.array([0, 0, 4, 4], dtype=np.uint8)) & 15).astype(np.float32)  # (nb, 8, 4): l / 2
        db = d * (np.float32(0.5) + nib) * np.float32(0.25)
        y = db[..., None] * gr * _sgn(_bits8(sb))
        return y.reshape(nrows, ncols)
    if typ in (IQ3_XXS, IQ3_S):
        d = _f16(blk[:, 0:2]).reshape(nb, 1, 1)
        qs = blk[:, 2:66].reshape(nb, 8, 8).astype(np.int64)                           # (nb, group, step t)
        if typ == IQ3_XXS:
            ss = u32(blk[:, 66:98]).reshape(nb, 8, 1)
            sb = G["ksigns"][(ss >> (7 * np.arange(4, dtype=np.uint32))) & 127]         # (nb, 8, 4 chunks)
            db = d * (np.float32(0.5) + (ss >> 28).astype(np.float32)) * np.float32(0.5)
            gr = G["iq3xxs_grid_b"][qs].astype(np.float32)                            # (nb, 8, 8 steps, 4)
        else:
            qh = blk[:, 66:74].astype(np.int64)
            idx = qs | ((qh[:, :, None] >> np.arange(8, dtype=np.int64)) & 1) << 8
            sb = blk[:, 74:106].reshape(nb, 8, 4)
            scb = blk[:, 106:110]
            nib = ((scb[:, :, None] >> np.array([0, 4], dtype=np.uint8)) & 15).reshape(nb, 8, 1).astype(np.float32)
            db = d * (np.float32(1.0) + np.float32(2.0) * nib)
            gr = G["iq3s_grid_b"][idx].astype(np.float32)
        y = db[..., None] * gr.reshape(nb, 8, 4, 8) * _sgn(_bits8(sb))
        return y.reshape(nrows, ncols)
    if typ in (IQ1_S, IQ1_M):
        if typ == IQ1_S:
            d = _f16(blk[:, 0:2]).reshape(nb, 1, 1)
            qs = blk[:, 2:34].reshape(nb, 8, 4).astype(np.int64)
            qh = u16(blk[:, 34:50]).reshape(nb, 8, 1).astype(np.int64)
            idx = qs | ((qh >> (3 * np.arange(4, dtype=np.int64))) & 7) << 8
            dl = d * (2 * ((qh >> 12) & 7) + 1).astype(np.float32)                    # (nb, 8, 1)
            delta = np.where(qh & 0x8000, np.float32(-0.125), np.float32(0.125))        # (nb, 8, 1)
        else:
            sc = u16(blk[:, 48:56]).astype(np.int64)                                   # (nb, 4)
            dh = ((sc[:, 0] >> 12) | ((sc[:, 1] >> 8) & 0x00F0) | ((sc[:, 2] >> 4) & 0x0F00) | (sc[:, 3] & 0xF000))
            d = dh.astype(np.uint16).view(np.float16).astype(np.float32).reshape(nb, 1, 1)
            qs = blk[:, 0:32].reshape(nb, 8, 4).astype(np.int64)
            qh = blk[:, 32:48].reshape(nb, 8, 2).astype(np.int64)
            q = (qh[:, :, :, None] >> np.array([0, 4], dtype=np.int64)).reshape(nb, 8, 4)  # nibble l % 2 of qh[2 ib + l / 2]
            idx = qs | (q & 7) << 8
            delta = np.where(q & 8, np.float32(-0.125), np.float32(0.125))            # (nb, 8, 4)
            ib = np.arange(8, dtype=np.int64)
            s3 = (sc[:, ib // 2][:, :, None] >> (6 * (ib % 2)[:, None] + 3 * (np.arange(4) // 2)[None, :])) & 7  # (nb, 8, 4)
            dl = d * (2 * s3 + 1).astype(np.float32)
        gr = G["iq1s_grid_b"][idx].astype(np.float32)                                  # (nb, 8, 4, 8)
        y = dl[..., None] * (gr + delta[..., None])
        return y.reshape(nrows, ncols)
    if typ == IQ4_XS:
        d = _f16(blk[:, 0:2]).reshape(nb, 1)
        sh = u16(blk[:, 2:4]).reshape(nb, 1).astype(np.int64)
        sl = blk[:, 4:8].astype(np.int64)
        ib = np.arange(8, dtype=np.int64)
        ls = ((sl[:, ib // 2] >> (4 * (ib % 2))) & 15) | ((sh >> (2 * ib)) & 3) << 4          # (nb, 8)
        dl = (d * (ls - 32).astype(np.float32)).reshape(nb, 8, 1)
        qs = blk[:, 8:136].reshape(nb, 8, 16)
        kv = G["kvalues_iq4nl"].astype(np.float32)
        q = np.concatenate([kv[qs & 15], kv[qs >> 4]], axis=2)                           # (nb, 8, 32)
        return (dl * q).reshape(nrows, ncols)
    raise ValueError(f"ggml type {typ} not supported here")


def dequant(typ, data, nrows, ncols):
    """rows of a tensor (uint8 (nrows, row bytes)) -> float32 (nrows, ncols), ggml's expressions"""
    data = np.ascontiguousarray(data, dtype=np.uint8)
    if typ in IQ_TYPES:
        return _dequant_iq(typ, data, nrows, ncols)
    if typ == F32:
        return data.view(np.float32).reshape(nrows, ncols).copy()
    if typ == F16:
        return data.view(np.float16).astype(np.float32).reshape(nrows, ncols)
    if typ == Q8_0:
        blk = data.reshape(-1, 34)
        d = _f16(blk[:, 0:2]).reshape(-1, 1)
        q = blk[:, 2:].view(np.int8).astype(np.float32)
        return (q * d).reshape(nrows, ncols)
    if typ == Q3_K:
        blk = data.reshape(-1, 110)
        nb = blk.shape[0]
        hm = blk[:, 0:32]
        qs = blk[:, 32:96]
        s = blk[:, 96:108].astype(np.int32)
        d = _f16(blk[:, 108:110]).reshape(-1, 1, 1)
        # 16 6-bit scales (ggml's kmask shuffle): low 4 bits from s[0..7] (j < 8: low nibble of s[j], else the
        # high nibble of s[j - 8]), high 2 bits from s[8 + j % 4] >> 2 (j / 4)
        j = np.arange(16)
        lo = np.where(j < 8, s[:, j % 8] & 0xF, s[:, j % 8] >> 4)
        sc = lo | (((s[:, 8 + j % 4] >> (2 * (j // 4))) & 3) << 4)
        dl = d * (sc - 32).astype(np.float32).reshape(nb, 16, 1)  # dl = d * (sc - 32), exact
        # value i = 128 n + 32 p + l: q2 = (qs[32 n + l] >> 2 p) & 3, minus 4 unless bit i / 32 of hmask[l] is set
        q2 = (qs.reshape(nb, 2, 1, 32) >> (2 * np.arange(4, dtype=np.uint8)).reshape(1, 1, 4, 1)) & 3
        h = (hm.reshape(nb, 1, 32) >> np.arange(8, dtype=np.uint8).reshape(1, 8, 1)) & 1
        q = q2.reshape(nb, 8, 32).astype(np.int8) - ((1 - h) << 2).astype(np.int8)
        return (dl * q.reshape(nb, 16, 16).astype(np.float32)).reshape(nrows, ncols)
    if typ == Q4_K:
        blk = data.reshape(-1, 144)
        nb = blk.shape[0]
        d = _f16(blk[:, 0:2]).reshape(-1, 1)
        dmin = _f16(blk[:, 2:4]).reshape(-1, 1)
        sc, mn = _q4k_scale_min(blk[:, 4:16].astype(np.int32))
        qs = blk[:, 16:].reshape(nb, 4, 32)
        d_sc = (d * sc.astype(np.float32))  # (nb, 8): d1 = d * sc (float32 products)
        m_mn = (dmin * mn.astype(np.float32))
        out = np.empty((nb, 8, 32), np.float32)
        lo = (qs & 0xF).astype(np.float32)
        hi = (qs >> 4).astype(np.float32)
        for g in range(4):  # 64 values per step: low nibbles (scale 2g), then high (2g + 1)
            out[:, 2 * g] = d_sc[:, 2 * g:2 * g + 1] * lo[:, g] - m_mn[:, 2 * g:2 * g + 1]
            out[:, 2 * g + 1] = d_sc[:, 2 * g + 1:2 * g + 2] * hi[:, g] - m_mn[:, 2 * g + 1:2 * g + 2]
        return out.reshape(nrows, ncols)
    if typ == Q5_K:
        blk = data.reshape(-1, 176)
        nb = blk.shape[0]
        d = _f16(blk[:, 0:2]).reshape(-1, 1)
        dmin = _f16(blk[:, 2:4]).reshape(-1, 1)
        sc, mn = _q4k_scale_min(blk[:, 4:16].astype(np.int32))
        qh = blk[:, 16:48]
        qs = blk[:, 48:176]
        d_sc = (d * sc.astype(np.float32))
        m_mn = (dmin * mn.astype(np.float32))
        # 5 bits per value: the nibbles (as Q4_K) and bit 4 = bit j of qh[l] for value 32 j + l
        lo = ((qs.reshape(nb, 4, 1, 32) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)) & 0x0F).reshape(nb, 8, 32)
        hi = (qh.reshape(nb, 1, 32) >> np.arange(8, dtype=np.uint8).reshape(1, 8, 1)) & 1
        q = (lo | (hi << np.uint8(4))).astype(np.float32)
        out = d_sc.reshape(nb, 8, 1) * q - m_mn.reshape(nb, 8, 1)
        return out.reshape(nrows, ncols)
    if typ == Q6_K:
        blk = data.reshape(-1, 210)
        nb = blk.shape[0]
        ql = blk[:, 0:128].astype(np.int32)
        qh = blk[:, 128:192].astype(np.int32)
        sc = blk[:, 192:208].view(np.int8).astype(np.float32)
        d = _f16(blk[:, 208:210]).reshape(-1, 1)
        out = np.empty((nb, 256), np.float32)
        for n in range(2):  # 128 values per half
            L, H, S = ql[:, 64 * n:64 * n + 64], qh[:, 32 * n:32 * n + 32], sc[:, 8 * n:8 * n + 8]
            q1 = ((L[:, :32] & 0xF) | (((H >> 0) & 3) << 4)) - 32
            q2 = ((L[:, 32:] & 0xF) | (((H >> 2) & 3) << 4)) - 32
            q3 = ((L[:, :32] >> 4) | (((H >> 4) & 3) << 4)) - 32
            q4 = ((L[:, 32:] >> 4) | (((H >> 6) & 3) << 4)) - 32
            for k, q in enumerate((q1, q2, q3, q4)):
                isc = np.repeat(S[:, 2 * k:2 * k + 2], 16, axis=1)  # is = l / 16: + 0 / 2 / 4 / 6
                out[:, 128 * n + 32 * k:128 * n + 32 * k + 32] = (d * isc) * q.astype(np.float32)
        return out.reshape(nrows, ncols)
    raise ValueError(f"ggml type {typ} not supported here")
