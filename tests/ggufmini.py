"""A minimal GGUF reader / writer and numpy dequantizer for the tests (numpy only; no gguf-py).

GgufFile(path): the metadata (raw byte spans and decoded scalars / strings / arrays) and the tensor table of
a GGUF v2 / v3 file (layout: src/gguf.c), the tensor data as numpy views of a memory map.
write_gguf(path, kvs, tensors): a GGUF v3 file from (key, raw value bytes) pairs (raw_kv / copied spans)
and (name, ne, ggml type, data bytes) tuples.
dequant(type, data, nrows, ncols): Q8_0 / Q4_K / Q6_K / F32 rows to float32 with ggml's expressions
(dequantize_row_q8_0 / _q4_K / _q6_K: the same float32 operations as gguf-py, bit for bit; checked against
tests/golden/dequant). Used by tests/make_synth.py (the synthetic models), tests/ref_forward.py (the numpy
reference forward pass) and tests/check_models.py.
"""
import struct

import numpy as np

GV_U8, GV_I8, GV_U16, GV_I16, GV_U32, GV_I32, GV_F32, GV_BOOL, GV_STR, GV_ARR, GV_U64, GV_I64, GV_F64 = range(13)
_SCALAR = {GV_U8: "<B", GV_I8: "<b", GV_U16: "<H", GV_I16: "<h", GV_U32: "<I", GV_I32: "<i", GV_F32: "<f",
           GV_BOOL: "<?", GV_U64: "<Q", GV_I64: "<q", GV_F64: "<d"}
F32, F16, Q8_0, Q4_K, Q6_K = 0, 1, 8, 12, 14
BLOCK = {F32: (1, 4), F16: (1, 2), Q8_0: (32, 34), Q4_K: (256, 144), Q6_K: (256, 210)}  # values, bytes
TYPE_NAME = {F32: "F32", F16: "F16", Q8_0: "Q8_0", Q4_K: "Q4_K", Q6_K: "Q6_K"}


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


def dequant(typ, data, nrows, ncols):
    """rows of a tensor (uint8 (nrows, row bytes)) -> float32 (nrows, ncols), ggml's expressions"""
    data = np.ascontiguousarray(data, dtype=np.uint8)
    if typ == F32:
        return data.view(np.float32).reshape(nrows, ncols).copy()
    if typ == F16:
        return data.view(np.float16).astype(np.float32).reshape(nrows, ncols)
    if typ == Q8_0:
        blk = data.reshape(-1, 34)
        d = _f16(blk[:, 0:2]).reshape(-1, 1)
        q = blk[:, 2:].view(np.int8).astype(np.float32)
        return (q * d).reshape(nrows, ncols)
    if typ == Q4_K:
        blk = data.reshape(-1, 144)
        nb = blk.shape[0]
        d = _f16(blk[:, 0:2]).reshape(-1, 1)
        dmin = _f16(blk[:, 2:4]).reshape(-1, 1)
        s = blk[:, 4:16].astype(np.int32)
        qs = blk[:, 16:].reshape(nb, 4, 32)
        sc = np.empty((nb, 8), np.int32)
        mn = np.empty((nb, 8), np.int32)
        for j in range(8):  # get_scale_min_k4
            if j < 4:
                sc[:, j] = s[:, j] & 63
                mn[:, j] = s[:, j + 4] & 63
            else:
                sc[:, j] = (s[:, j + 4] & 0xF) | ((s[:, j - 4] >> 6) << 4)
                mn[:, j] = (s[:, j + 4] >> 4) | ((s[:, j] >> 6) << 4)
        d_sc = (d * sc.astype(np.float32))  # (nb, 8): d1 = d * sc (float32 products)
        m_mn = (dmin * mn.astype(np.float32))
        out = np.empty((nb, 8, 32), np.float32)
        lo = (qs & 0xF).astype(np.float32)
        hi = (qs >> 4).astype(np.float32)
        for g in range(4):  # 64 values per step: low nibbles (scale 2g), then high (2g + 1)
            out[:, 2 * g] = d_sc[:, 2 * g:2 * g + 1] * lo[:, g] - m_mn[:, 2 * g:2 * g + 1]
            out[:, 2 * g + 1] = d_sc[:, 2 * g + 1:2 * g + 2] * hi[:, g] - m_mn[:, 2 * g + 1:2 * g + 2]
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
