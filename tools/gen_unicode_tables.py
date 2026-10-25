#!/usr/bin/env python3
"""Generate src/unicode_tables.h (pre-tokenizer classes) and src/unicode_nfc.h (NFC normalizer).

Usage: python3 tools/gen_unicode_tables.py > src/unicode_tables.h          (any Python 3; stdlib only)
       python3 tools/gen_unicode_tables.py --nfc > src/unicode_nfc.h
       python3 tools/gen_unicode_tables.py --nfc --unicode-only > x.h   (no HF compat overrides)
       .venv/bin/python tools/gen_unicode_tables.py --check-hf         (verify the overrides, needs tokenizers)

The Qwen2/Qwen3 pre-tokenizer regex (tokenizer.json) needs three classes:
  L   \\p{L}  general category Lu, Ll, Lt, Lm, Lo
  N   \\p{N}  general category Nd, Nl, No
  WS  \\s     Unicode White_Space (PropList.txt), which is what HF tokenizers' regex engine
             (Oniguruma) uses for \\s. Not Python's str.isspace(), which also has U+001C..U+001F
             (checked against HF: "x\\x1c\\x1cy" splits like punctuation, "x\\u3000\\u3000y" like space).
Everything else is "other". The table is a sorted list of disjoint ranges, each with one class,
looked up by binary search (ASCII has a fast path in runq.c).
The Unicode version is that of the Python running this script (written into the header); HF
tokenizers 0.21.4 agrees with Unicode 15.1 on the 15.1 additions (e.g. CJK Ext. I is \\p{L}).

NFC (--nfc): tokenizer.json's normalizer. HF tokenizers 0.21.4 normalizes with the Rust crate
unicode-normalization-alignments 0.1.12, whose data is older (Unicode 9.0): checked against it
(--check-hf), its NFC equals Unicode 15.1's on every single code point, on all ordered pairs of
the older combining marks and on all primary composites, except for characters added in 10.0+:
  HF_CCC0          combining marks that HF treats as ccc 0 (never reordered)
  HF_NO_DECOMP     U+11938 DIVES AKURU VOWEL SIGN O (13.0): no decomposition, so 11935 + 11930
                   does not compose
The tables apply these overrides (runq must match HF's ids), unless --unicode-only.
Tables: entries for every code point with ccc != 0, NFC_QC != Yes or a canonical decomposition
(Hangul syllables are algorithmic, not listed), sorted, binary search; full decompositions in a
pool; primary composites as sorted (first, second) keys; two bitmaps over 64-code-point blocks
(any entry / an entry with ccc != 0 or NFC_QC != Yes) so that most lookups need no search.
NFC_QC is derived: No = canonically decomposable and not a primary composite (NFC(c) != c),
Maybe = second character of a composition pair (incl. Hangul V and T jamo).
"""
import sys
import unicodedata

# Combining marks with ccc != 0 in Unicode 15.1 that HF tokenizers 0.21.4 treats as ccc 0 (all
# added in Unicode 10.0-15.1), found and verified with --check-hf.
HF_CCC0 = [(0x07FD, 0x07FD), (0x0898, 0x089F), (0x08CA, 0x08D3), (0x09FE, 0x09FE), (0x0C3C, 0x0C3C),
           (0x0D3B, 0x0D3C), (0x0EBA, 0x0EBA), (0x1715, 0x1715), (0x1ABF, 0x1ACE), (0x1DF6, 0x1DFA),
           (0xA82C, 0xA82C), (0x10D24, 0x10D27), (0x10EAB, 0x10EAC), (0x10EFD, 0x10EFF),
           (0x10F46, 0x10F50), (0x10F82, 0x10F85), (0x11070, 0x11070), (0x1133B, 0x1133B),
           (0x1145E, 0x1145E), (0x11839, 0x1183A), (0x1193D, 0x1193E), (0x11943, 0x11943),
           (0x119E0, 0x119E0), (0x11A34, 0x11A34), (0x11A47, 0x11A47), (0x11A99, 0x11A99),
           (0x11D42, 0x11D42), (0x11D44, 0x11D45), (0x11D97, 0x11D97), (0x11F41, 0x11F42),
           (0x16FF0, 0x16FF1), (0x1E08F, 0x1E08F), (0x1E130, 0x1E136), (0x1E2AE, 0x1E2AE),
           (0x1E2EC, 0x1E2EF), (0x1E4EC, 0x1E4EF)]
HF_NO_DECOMP = [0x11938]
HANGUL_S = range(0xAC00, 0xAC00 + 11172)
HANGUL_VT = list(range(0x1161, 0x1176)) + list(range(0x11A8, 0x11C3))

# PropList.txt White_Space (stable since Unicode 6.3, when U+180E was removed).
WHITE_SPACE = [(0x0009, 0x000D), (0x0020, 0x0020), (0x0085, 0x0085), (0x00A0, 0x00A0),
               (0x1680, 0x1680), (0x2000, 0x200A), (0x2028, 0x2029), (0x202F, 0x202F),
               (0x205F, 0x205F), (0x3000, 0x3000)]
CLS = {"L": 1, "N": 2, "WS": 3}


def classify(cp, ws):
    if cp in ws:
        return "WS"
    cat = unicodedata.category(chr(cp))
    if cat[0] == "L":
        return "L"
    if cat[0] == "N":
        return "N"
    return None


def nfc_data(hf_compat):
    """ccc, full canonical decompositions, composition pairs and NFC_QC as runq applies them"""
    ccc0 = {c for lo, hi in HF_CCC0 for c in range(lo, hi + 1)} if hf_compat else set()
    nodec = set(HF_NO_DECOMP) if hf_compat else set()
    cps = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]
    ccc = {c: unicodedata.combining(chr(c)) for c in cps if unicodedata.combining(chr(c)) and c not in ccc0}
    assert all(unicodedata.combining(chr(c)) for c in ccc0), "HF_CCC0 entry without ccc"
    dm = {}  # one level
    for c in cps:
        d = unicodedata.decomposition(chr(c))
        if d and not d.startswith("<") and c not in nodec:
            dm[c] = [int(x, 16) for x in d.split()]
    assert not any(c in HANGUL_S for c in dm)

    def full(c):
        return [y for x in dm[c] for y in (full(x) if x in dm else [x])] if c in dm else [c]
    dec = {c: full(c) for c in dm}
    comp = {}  # (first, second) -> primary composite
    for c, d in dm.items():
        if len(d) == 2 and unicodedata.normalize("NFC", chr(c)) == chr(c):
            assert ccc.get(d[0], 0) == 0 and ccc.get(c, 0) == 0, hex(c)
            comp[tuple(d)] = c
    qc = {}
    for c in dm:
        if unicodedata.normalize("NFC", chr(c)) != chr(c):
            qc[c] = 2  # No
    for _, b in comp:
        assert qc.get(b, 1) == 1
        qc[b] = 1  # Maybe
    for b in HANGUL_VT:
        qc[b] = 1
    # size bounds that runq.c relies on: NFC output <= 3x the input bytes, <= 2 code points per byte
    u8 = lambda cs: len("".join(map(chr, cs)).encode())
    for c, d in dec.items():
        assert u8(d) <= 3 * u8([c]) and len(d) <= 2 * u8([c]) and len(d) <= 7, hex(c)
    for (a, b), c in comp.items():
        assert u8([c]) <= u8([a]) + u8([b])
    return ccc, dec, comp, qc


def write_nfc(out, hf_compat):
    ccc, dec, comp, qc = nfc_data(hf_compat)
    keys = sorted(set(ccc) | set(dec) | set(qc))
    pool, pool_at, decw = [], {}, []
    for c in keys:
        if c not in dec:
            decw.append(0)
            continue
        t = tuple(dec[c])
        if t not in pool_at:
            pool_at[t] = len(pool)
            pool.extend(t)
        decw.append(pool_at[t] << 3 | len(t))
    assert len(pool) < 1 << 13
    any_blk, qc_blk = [0] * (0x110000 >> 11), [0] * (0x110000 >> 11)
    for c in keys:
        any_blk[c >> 11] |= 1 << (c >> 6 & 31)
        if ccc.get(c, 0) or qc.get(c, 0):
            qc_blk[c >> 11] |= 1 << (c >> 6 & 31)
    combo = sorted((a << 21 | b) << 21 | c for (a, b), c in comp.items())

    py = ".".join(map(str, sys.version_info[:3]))
    ver = unicodedata.unidata_version + (" + HF tokenizers compat" if hf_compat else "")
    out.write("// Generated by tools/gen_unicode_tables.py --nfc, do not edit.\n")
    out.write(f"// Python {py}, unicodedata Unicode {unicodedata.unidata_version}"
              + ("; HF tokenizers 0.21.4 overrides (see the generator): "
                 f"{sum(hi - lo + 1 for lo, hi in HF_CCC0)} marks ccc 0, U+11938 not decomposed.\n"
                 if hf_compat else ", no HF overrides.\n"))
    out.write("// NFC data: nfc_cp[] = cp | qc << 21 | ccc << 23 (qc 0 Yes, 1 Maybe, 2 No), sorted; nfc_dec[] = full\n"
              "// canonical decomposition, 0 or offset << 3 | length into nfc_pool[]; nfc_comp[] = (first << 21 |\n"
              "// second) << 21 | composite, sorted; nfc_*_blk = bit per 64-cp block: any entry / ccc or qc entry.\n"
              "// Hangul syllables are algorithmic (not listed); the V and T jamo are listed as Maybe.\n")
    out.write("#ifndef CWAYN_UNICODE_NFC_H\n#define CWAYN_UNICODE_NFC_H\n\n#include <stdint.h>\n\n")
    out.write(f'#define UNICODE_NFC_VERSION "{ver}"\n')
    out.write(f"#define NFC_MIN_CP 0x{keys[0]:X}\n#define NFC_N {len(keys)}\n#define NFC_NCOMP {len(combo)}\n\n")

    def arr(decl, vals, fmt, per):
        out.write(decl + " = {\n")
        for i in range(0, len(vals), per):
            out.write("    " + ",".join(fmt(v) for v in vals[i:i + per]) + ",\n")
        out.write("};\n")
    arr(f"static const uint32_t nfc_any_blk[{len(any_blk)}]", any_blk, lambda v: f"0x{v:X}", 12)
    arr(f"static const uint32_t nfc_qc_blk[{len(qc_blk)}]", qc_blk, lambda v: f"0x{v:X}", 12)
    arr("static const uint32_t nfc_cp[NFC_N]", [c | qc.get(c, 0) << 21 | ccc.get(c, 0) << 23 for c in keys],
        lambda v: f"0x{v:X}", 10)
    arr("static const uint16_t nfc_dec[NFC_N]", decw, str, 16)
    arr(f"static const uint32_t nfc_pool[{len(pool)}]", pool, lambda v: f"0x{v:X}", 12)
    arr("static const uint64_t nfc_comp[NFC_NCOMP]", combo, lambda v: f"0x{v:X}ull", 6)
    out.write("\n#endif\n")
    size = 4 * len(any_blk) * 2 + 4 * len(keys) + 2 * len(keys) + 4 * len(pool) + 8 * len(combo)
    print(f"NFC: {len(keys)} entries ({len(ccc)} ccc, {len(dec)} decompositions, {sum(1 for v in qc.values() if v == 2)} "
          f"QC No, {sum(1 for v in qc.values() if v == 1)} Maybe), pool {len(pool)}, {len(combo)} compositions, "
          f"{size} bytes (Unicode {ver})", file=sys.stderr)


def check_hf():
    """compare HF tokenizers' NFC with Unicode NFC: singles, mark pairs, composites; derive HF_CCC0"""
    from tokenizers import __version__ as tv
    from tokenizers.normalizers import NFC
    hf = NFC().normalize_str

    def bat(strs):  # one call per batch, '\n' blocks normalization between the strings
        out = []
        for i in range(0, len(strs), 50000):
            r = hf("\n".join(strs[i:i + 50000])).split("\n")
            assert len(r) == len(strs[i:i + 50000])
            out += r
        return out
    cps = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF and c != 10]
    nfc = lambda s: unicodedata.normalize("NFC", s)
    single = [chr(c) for c in cps]
    d1 = [s for s, o in zip(single, bat(single)) if o != nfc(s)]
    marks = [c for c in cps if unicodedata.combining(chr(c))]
    probe = ["%" + x + y for c in marks for m in "\u0301\u0316\u0334" for x, y in ((chr(c), m), (m, chr(c)))]
    po = bat(probe)
    ccc0 = {c for i, c in enumerate(marks) if po[6 * i:6 * i + 6] == probe[6 * i:6 * i + 6]}
    old = [c for c in marks if c not in ccc0]
    pairs = ["%" + chr(x) + chr(y) for x in old for y in old]
    d2 = [s for s, o in zip(pairs, bat(pairs)) if o != nfc(s)]
    _, _, comp, _ = nfc_data(False)
    d3 = sorted(c for ((a, b), c), o in zip(comp.items(), bat([chr(a) + chr(b) for a, b in comp])) if o != chr(c))
    want = {c for lo, hi in HF_CCC0 for c in range(lo, hi + 1)}
    print(f"tokenizers {tv}, Unicode {unicodedata.unidata_version}: {len(d1)} single code point diffs, "
          f"{len(ccc0)} marks HF treats as ccc 0 ({'= HF_CCC0' if ccc0 == want else 'DIFFERS from HF_CCC0'}), "
          f"{len(d2)} diffs on {len(pairs)} pairs of the other marks, composites not composed: "
          f"{[hex(c) for c in d3]} ({'= HF_NO_DECOMP' if d3 == HF_NO_DECOMP else 'DIFFERS'})")
    return 0 if not d1 and not d2 and ccc0 == want and d3 == HF_NO_DECOMP else 1


def main():
    if "--check-hf" in sys.argv:
        return check_hf()
    if "--nfc" in sys.argv:
        return write_nfc(sys.stdout, "--unicode-only" not in sys.argv)
    ws = {c for lo, hi in WHITE_SPACE for c in range(lo, hi + 1)}
    # sanity: White_Space == str.isspace() minus the 4 information separators U+001C..U+001F
    isspace = {c for c in range(0x110000) if chr(c).isspace()}
    assert isspace - ws == set(range(0x1C, 0x20)) and not ws - isspace, "unexpected White_Space set"

    ranges = []  # (lo, hi, cls)
    for cp in range(0x110000):
        c = classify(cp, ws)
        if c is None:
            continue
        if ranges and ranges[-1][2] == c and ranges[-1][1] == cp - 1:
            ranges[-1][1] = cp
        else:
            ranges.append([cp, cp, c])

    out = sys.stdout
    py = ".".join(map(str, sys.version_info[:3]))
    out.write("// Generated by tools/gen_unicode_tables.py, do not edit.\n")
    out.write(f"// Python {py}, unicodedata Unicode {unicodedata.unidata_version}.\n")
    out.write("// Code point classes for the Qwen2 pre-tokenizer regex: UCLS_L = \\p{L} (Lu Ll Lt Lm Lo),\n")
    out.write("// UCLS_N = \\p{N} (Nd Nl No), UCLS_WS = \\s (Unicode White_Space), anything else UCLS_OTHER.\n")
    out.write("// Sorted, disjoint ranges: hi_cls = hi | cls << 24.\n")
    out.write("#ifndef CWAYN_UNICODE_TABLES_H\n#define CWAYN_UNICODE_TABLES_H\n\n#include <stdint.h>\n\n")
    out.write(f'#define UNICODE_TABLES_VERSION "{unicodedata.unidata_version}"\n')
    out.write("enum { UCLS_OTHER = 0, UCLS_L = 1, UCLS_N = 2, UCLS_WS = 3 };\n\n")
    out.write(f"#define UCLS_NRANGES {len(ranges)}\n")
    out.write("static const uint32_t ucls_lo[UCLS_NRANGES] = {\n")
    for i in range(0, len(ranges), 10):
        out.write("    " + ",".join(f"0x{r[0]:X}" for r in ranges[i:i + 10]) + ",\n")
    out.write("};\nstatic const uint32_t ucls_hi_cls[UCLS_NRANGES] = {\n")
    for i in range(0, len(ranges), 8):
        out.write("    " + ",".join(f"0x{r[1] | CLS[r[2]] << 24:X}" for r in ranges[i:i + 8]) + ",\n")
    out.write("};\n\n#endif\n")
    print(f"{len(ranges)} ranges (Unicode {unicodedata.unidata_version})", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
