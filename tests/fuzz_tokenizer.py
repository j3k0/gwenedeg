#!/usr/bin/env python3
"""On-demand tokenizer fuzz test: runq -m tokenize vs the HF tokenizer (not in the fast suite).

Usage: .venv/bin/python tests/fuzz_tokenizer.py [-n 5000] [--seed 1] [--jobs 8] [--show 5]
       .venv/bin/python tests/fuzz_tokenizer.py --nfc-all     (exhaustive NFC check, see below)
       .venv/bin/python tests/fuzz_tokenizer.py --model models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf
                                                     (the tokenizer runq builds from GGUF metadata)
Needs transformers (.venv) and models/Qwen3-0.6B (local files only) plus models/Qwen3-0.6B-q8.bin
(its .tokenizer file gives the token bytes, --vocab; a GGUF file's tokens decode to the same bytes).

Cases:
  random  -n strings built from a mix of character classes (ASCII words, digits, punctuation,
          whitespace runs incl. \\r\\n, tabs, NBSP and other Unicode spaces, contractions in any
          case, emoji sequences (ZWJ, flags, skin tones, keycaps), CJK incl. rare extension
          blocks, Arabic, Devanagari, Hangul, Cyrillic, Greek, math, combining marks, added
          tokens and broken ones, and random code points from all planes), and for the NFC
          normalizer: base + combining marks in random order, Hangul jamo runs (L V T, LV + T),
          NFD forms of words (Latin, Vietnamese, Greek, Cyrillic, Devanagari, Hangul, kana),
          NFC_QC=No characters (singletons, composition exclusions), long mark runs.
          Not normalized: HF applies NFC itself.
  files   real text: the repo's docs and sources and the HF model README (whole files, split
          into <= 64 KiB pieces at line ends because of the argv size limit), as they are and
          in NFD.
Strings contain no NUL (argv). Besides the ids, the normalized text is compared directly: the
concatenated bytes of runq's tokens must equal HF's NFC normalizer output (byte-level BPE is
lossless). Each mismatch is minimized (delta debugging on characters, as long as it still
mismatches) and the first --show are printed. Exit status 1 if any case mismatches.

--nfc-all: runq's normalizer (tokens -> bytes) vs HF's NFC normalizer and vs Python's
unicodedata.normalize('NFC') on every code point alone (all of U+0001..U+10FFFF but surrogates
and \\n), all ordered pairs of combining marks after a base, every composition pair and the NFD
of every decomposable character, all Hangul L+V and LV+T pairs, and random strings. Strings
are joined with \\n (blocks normalization) into <= 64 KiB arguments. Differences from
unicodedata are expected only for strings with a code point of the HF compatibility overrides
(tools/gen_unicode_tables.py: marks added after Unicode 9.0 that HF treats as ccc 0, U+11938);
they are counted separately. Exit status 1 on any difference from HF.
"""
import argparse
import concurrent.futures as cf
import os
import random
import string
import subprocess
import sys
import time
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen3-0.6B")
FILES = ["README.md", "TODO.md", "tests/README.md", "src/runq.c", "src/win.c", "tests/check.py",
         "tests/make_golden.py", "tests/fuzz_tokenizer.py", "tools/gen_unicode_tables.py",
         "scripts/download_models.py", "third_party/qwen3.c/export.py",
         "models/Qwen3-0.6B/README.md", "models/Qwen3-0.6B/LICENSE"]
MAX_ARG = 64 * 1024

SPACES = [" ", "  ", "   ", "    ", "        ", "\t", "\t\t", " \t", "\t ", "\n", "\n\n", "\r\n",
          "\r", " \n", "\n ", "  \n  ", "\u00a0", "\u00a0\u00a0", "\u3000", "\u2002", "\u2009",
          "\u202f", "\u205f", "\u2028", "\u2029", "\x85", "\x0b", "\x0c", "\u1680", " \u00a0 "]
CONTRACTIONS = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "'T", "'RE", "'Ve", "'LL", "'lL",
                "'D", "'M", "'\u017f", "'", "''", "n't", "N'T", "'x"]
EMOJI = ["\U0001F600", "\U0001F355", "\U0001F680", "\u2764\ufe0f", "\U0001F44D\U0001F3FD",
         "\U0001F44B\U0001F3FB", "\U0001F468\u200d\U0001F469\u200d\U0001F467\u200d\U0001F466",
         "\U0001F469\U0001F3FD\u200d\U0001F4BB", "\U0001F3F3\ufe0f\u200d\U0001F308",
         "\U0001F1FA\U0001F1F8", "\U0001F1EF\U0001F1F5", "\U0001F1E9\U0001F1EA", "1\ufe0f\u20e3",
         "#\ufe0f\u20e3", "\u270c\ufe0f", "\U0001F9D1\u200d\U0001F91D\u200d\U0001F9D1", "\u00a9\ufe0f",
         "\U0001F3F4\U000E0067\U000E0062\U000E0073\U000E0063\U000E0074\U000E007F"]
SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>",
            "</tool_call>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|vision_pad|>",
            "<|im_start", "im_end|>", "<|", "|>", "<", ">", "</", "<think", "think>", "<br>", "<a>"]
RANGES = {  # name: list of (lo, hi) code point ranges
    "latin_ext": [(0xC0, 0x24F)], "greek": [(0x370, 0x3FF)], "cyrillic": [(0x400, 0x4FF)],
    "arabic": [(0x600, 0x6FF)], "devanagari": [(0x900, 0x97F)], "thai": [(0xE00, 0xE7F)],
    "hangul": [(0xAC00, 0xD7A3)], "kana": [(0x3040, 0x30FF)], "cjk": [(0x4E00, 0x9FFF)],
    "cjk_rare": [(0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0x2A700, 0x2EE5D), (0x30000, 0x323AF),
                 (0xF900, 0xFAFF)],
    "math": [(0x2200, 0x22FF), (0x2190, 0x21FF), (0x1D400, 0x1D7FF), (0x2070, 0x209F), (0xB2, 0xBE)],
    "symbols": [(0x2000, 0x206F), (0x2500, 0x25FF), (0x20A0, 0x20CF), (0xFF00, 0xFFEF), (0xA1, 0xBF)],
    "combining": [(0x300, 0x36F), (0x1AB0, 0x1AFF), (0x20D0, 0x20FF), (0xFE00, 0xFE0F)],
    "numbers": [(0x660, 0x669), (0x966, 0x96F), (0x2160, 0x2188), (0x2460, 0x249B), (0x3007, 0x3007),
                (0xFF10, 0xFF19), (0x1D7CE, 0x1D7FF)],
    "controls": [(0x01, 0x1F), (0x7F, 0x9F)],
    "any_bmp": [(0x01, 0xD7FF), (0xE000, 0xFFFF)],
    "any": [(0x01, 0xD7FF), (0xE000, 0x10FFFF)],
    "jamo": [(0x1100, 0x1112), (0x1161, 0x1175), (0x11A8, 0x11C2), (0x1113, 0x11FF), (0xA960, 0xA97C),
             (0xD7B0, 0xD7FB)],
}
ALL_CPS = [c for c in range(1, 0x110000) if not 0xD800 <= c <= 0xDFFF]
MARKS = [c for c in ALL_CPS if unicodedata.combining(chr(c))]  # ccc != 0 (Unicode 15.1)
COMMON_MARKS = list(range(0x300, 0x370)) + [0x323, 0x328, 0x31B, 0x93C, 0x94D, 0x5B0, 0x5BC, 0x5C1, 0x64B,
                                           0x651, 0x3099, 0x309A, 0x1DC0, 0x20D7, 0x0E38, 0x0E48]
DECOMPOSABLE = [c for c in ALL_CPS if unicodedata.decomposition(chr(c))
                and not unicodedata.decomposition(chr(c)).startswith("<")]
QC_NO = [c for c in DECOMPOSABLE if unicodedata.normalize("NFC", chr(c)) != chr(c)]  # NFC_QC = No
NFD_WORDS = ["café", "naïve", "Ångström", "Nguyễn", "người", "được", "Tiếng Việt", "phở", "ặ", "ễ",
             "λόγος", "Ἀθῆναι", "ΐΰ", "ᾅ", "Йод", "ёлка", "ї", "क\u093c", "ड\u093cी", "한국어", "각", "ガギグ", "パ",
             "ǖ", "ṩ", "Å", "ñ", "ö", "ą\u0301"]
BASES = "aeiouAEOUnsqxz%1 कאاあᄀ가"


def gen_marks(rng):
    # a base (or nothing) and 1-6 combining marks in random (often non-canonical) order
    pool = COMMON_MARKS if rng.random() < 0.7 else MARKS
    base = rng.choice(BASES) if rng.random() < 0.9 else ""
    return base + "".join(chr(rng.choice(pool)) for _ in range(rng.randint(1, 6)))


def gen_nfc(rng):
    k = rng.randint(0, 5)
    if k == 0:
        return gen_marks(rng)
    if k == 1:  # Hangul: L V, L V T, LV + T, LV + V, stray V / T
        L, V, T = chr(0x1100 + rng.randrange(19)), chr(0x1161 + rng.randrange(21)), chr(0x11A8 + rng.randrange(27))
        lv = chr(0xAC00 + rng.randrange(399) * 28)
        return rng.choice([L + V, L + V + T, lv + T, lv + V, V + T, T, L + L + V, L + V + T + T, lv + chr(0x11A7)])
    if k == 2:
        return unicodedata.normalize(rng.choice(["NFD", "NFKD"]), rng.choice(NFD_WORDS))
    if k == 3:
        return "".join(chr(rng.choice(QC_NO)) for _ in range(rng.randint(1, 3)))
    if k == 4:
        return unicodedata.normalize("NFD", chr(rng.choice(DECOMPOSABLE))) + gen_marks(rng)[1:]
    return rng.choice("ae") + "".join(chr(rng.choice(COMMON_MARKS)) for _ in range(rng.randint(20, 300)))

WORDS = ["the", "The", "hello", "world", "token", "Qwen", "llama", "x", "a", "I", "naïve", "café",
         "über", "Straße", "日本語", "中文", "привет", "مرحبا", "नमस\u094dत\u0947", "안녕", "λόγος", "int", "def",
         "return", "printf", "self", "None", "http", "www", "com", "README", "camelCase", "snake_case",
         "ABC", "xyz", "\u017f", "ǅ", "Ⅻ", "ﬁ"]


def rand_from_ranges(rng, rs):
    lo, hi = rng.choice(rs)
    return chr(rng.randint(lo, hi))


def gen_piece(rng, kind):
    if kind == "word":
        return rng.choice(WORDS)
    if kind == "ascii":
        return "".join(rng.choice(string.ascii_letters) for _ in range(rng.randint(1, 12)))
    if kind == "digits":
        return "".join(rng.choice(string.digits + ".,_-e") if i else rng.choice(string.digits)
                       for i in range(rng.randint(1, 15)))
    if kind == "punct":
        return "".join(rng.choice(string.punctuation) for _ in range(rng.randint(1, 5)))
    if kind == "space":
        return "".join(rng.choice(SPACES) for _ in range(rng.randint(1, 3)))
    if kind == "contraction":
        return rng.choice(CONTRACTIONS)
    if kind == "emoji":
        return "".join(rng.choice(EMOJI) for _ in range(rng.randint(1, 3)))
    if kind == "special":
        return rng.choice(SPECIALS)
    if kind == "nfc":
        return gen_nfc(rng)
    if kind in RANGES:
        return "".join(rand_from_ranges(rng, RANGES[kind]) for _ in range(rng.randint(1, 6)))
    raise ValueError(kind)


KINDS = ["word"] * 6 + ["ascii"] * 4 + ["digits"] * 3 + ["punct"] * 3 + ["space"] * 6 + \
        ["contraction"] * 2 + ["emoji"] * 2 + ["special"] + ["nfc"] * 6 + list(RANGES)


def gen_case(rng):
    # half the cases favour a few kinds (runs of similar text), half mix everything
    kinds = rng.sample(KINDS, rng.randint(2, 4)) if rng.random() < 0.5 else KINDS
    n = rng.choice([1, 2, 3, 5, 10, 20, 40, 80])
    s = "".join(gen_piece(rng, rng.choice(kinds)) for _ in range(n))
    return s.replace("\0", "")


def file_cases():
    out = []
    for rel in FILES:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8", newline="") as f:
            text = f.read()
        for form in ("", "NFD"):
            t = unicodedata.normalize(form, text) if form else text
            pieces, cur = [], ""
            for line in t.splitlines(keepends=True):
                if cur and len((cur + line).encode()) > MAX_ARG:
                    pieces.append(cur)
                    cur = ""
                cur += line
            if cur:
                pieces.append(cur)
            for i, p in enumerate(pieces):
                out.append((f"file:{rel}" + (f"#{i}" if len(pieces) > 1 else "") + (f" ({form})" if form else ""), p))
    return out


def load_vocab(model):
    """token bytes from the .tokenizer file (as tests/check.py)"""
    with open(model + ".tokenizer", "rb") as f:
        data = f.read()
    vocab, off = [], 12
    while off + 8 <= len(data):
        n = int.from_bytes(data[off + 4:off + 8], "little", signed=True)
        vocab.append(data[off + 8:off + 8 + n])
        off += 8 + n
    return vocab


class Runner:
    def __init__(self, a):
        from tokenizers.normalizers import NFC
        from transformers import AutoTokenizer
        self.a = a
        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        self.hf_nfc = NFC().normalize_str
        self.vocab = load_vocab(a.vocab)

    def hf(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    def text(self, ids):
        return b"".join(self.vocab[i] for i in ids)

    def runq(self, s):
        p = subprocess.run([self.a.runq, self.a.model, "-m", "tokenize", "-i", s], capture_output=True)
        if p.returncode != 0:
            raise RuntimeError(f"runq failed on {s!r}: {p.stderr.decode(errors='replace')[-300:]}")
        return [int(x) for x in p.stdout.split()]

    def mismatch(self, s):
        return self.hf(s) != self.runq(s)

    def minimize(self, s, budget=300):
        # ddmin-style: drop chunks of characters while the (NFC) string still mismatches
        n = 2
        while len(s) > 1 and budget > 0:
            step = max(1, len(s) // n)
            reduced = False
            for i in range(0, len(s), step):
                cand = s[:i] + s[i + step:]
                if not cand:
                    continue
                budget -= 1
                if self.mismatch(cand):
                    s, n, reduced = cand, max(n - 1, 2), True
                    break
                if budget <= 0:
                    break
            if not reduced:
                if step == 1:
                    break
                n = min(len(s), n * 2)
        return s


def hf_override_cps():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import gen_unicode_tables as g
    return {chr(c) for lo, hi in g.HF_CCC0 for c in range(lo, hi + 1)} | {chr(c) for c in g.HF_NO_DECOMP}


def nfc_all(r, a):
    """exhaustive normalizer check (see the module docstring)"""
    rng = random.Random(a.seed)
    sets = {}
    sets["single"] = [chr(c) for c in ALL_CPS if c != 10]
    sets["mark_pairs"] = ["a" + chr(x) + chr(y) for x in MARKS for y in MARKS]
    comp = []
    for c in DECOMPOSABLE:
        d = unicodedata.normalize("NFD", chr(c))
        comp += [d, d + "\u0323", d + "\u0301", "\u0301" + d]
        p = [int(x, 16) for x in unicodedata.decomposition(chr(c)).split()]
        if len(p) == 2:
            comp += [chr(p[0]) + chr(p[1]), chr(p[0]) + "\u0323" + chr(p[1]), chr(p[0]) + "\u0334" + chr(p[1])]
    sets["compositions"] = comp
    sets["hangul"] = [chr(0x1100 + l) + chr(0x1161 + v) for l in range(19) for v in range(21)] + \
        [chr(0xAC00 + i * 28) + chr(t) for i in range(399) for t in range(0x11A7, 0x11C4)] + \
        [chr(0x1100 + l) + chr(0x1161 + v) + chr(0x11A8 + t) for l in range(19) for v in range(21) for t in (0, 13, 26)]
    pool = MARKS + COMMON_MARKS * 8 + [ord(c) for c in BASES] + QC_NO + DECOMPOSABLE + \
        list(range(0x1100, 0x1113)) + list(range(0x1161, 0x1176)) + list(range(0x11A8, 0x11C3))
    sets["random"] = ["".join(chr(rng.choice(pool) if rng.random() < 0.9 else rng.choice(ALL_CPS)) for _ in
                              range(rng.randint(1, 12))).replace("\n", "") for _ in range(a.n * 20)]
    sets["random"] = [s for s in sets["random"] if s]
    over = hf_override_cps()
    bad = 0
    for name, strs in sets.items():
        t0 = time.time()
        batches, cur, size = [], [], 0
        for s in strs:
            b = len(s.encode()) + 1
            if cur and size + b > MAX_ARG:
                batches.append(cur)
                cur, size = [], 0
            cur.append(s)
            size += b
        if cur:
            batches.append(cur)
        with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
            outs = list(ex.map(lambda bt: r.text(r.runq("\n".join(bt))).split(b"\n"), batches))
        n_hf = n_py = n_py_exp = 0
        for bt, out in zip(batches, outs):
            hf = r.hf_nfc("\n".join(bt)).split("\n")
            assert len(out) == len(bt) == len(hf), (len(out), len(bt), len(hf))
            for s, o, h in zip(bt, out, hf):
                if o != h.encode():
                    n_hf += 1
                    if n_hf <= a.show:
                        print(f"  {name}: {s.encode('unicode_escape')} runq {o.decode(errors='replace').encode('unicode_escape')}"
                              f" hf {h.encode('unicode_escape')}")
                py = unicodedata.normalize("NFC", s)
                if o != py.encode():
                    if over & set(s + py + unicodedata.normalize("NFD", s)):
                        n_py_exp += 1
                    else:
                        n_py += 1
        bad += n_hf + n_py
        print(f"{name:12s} {len(strs):8d} strings in {len(batches):3d} runs: {n_hf} diffs vs HF, "
              f"{n_py} vs unicodedata (+ {n_py_exp} with HF override code points) ({time.time() - t0:.1f} s)")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-n", type=int, default=5000, help="number of random cases (--nfc-all: x20)")
    ap.add_argument("--nfc-all", action="store_true", help="exhaustive normalizer check instead")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--show", type=int, default=5, help="minimize and print this many mismatches")
    ap.add_argument("--no-files", action="store_true", help="skip the real-text cases")
    ap.add_argument("--runq", default=os.path.join(ROOT, "runq"))
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "Qwen3-0.6B-q8.bin"), help=".bin or GGUF checkpoint")
    ap.add_argument("--vocab", default=os.path.join(ROOT, "models", "Qwen3-0.6B-q8.bin"),
                    help=".bin checkpoint whose .tokenizer file gives the token bytes")
    a = ap.parse_args()

    t0 = time.time()
    r = Runner(a)
    if a.nfc_all:
        ret = nfc_all(r, a)
        print(f"\n({time.time() - t0:.1f} s)")
        return ret
    rng = random.Random(a.seed)
    cases = [(f"random#{i}", gen_case(rng)) for i in range(a.n)]
    if not a.no_files:
        cases += file_cases()
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        got = list(ex.map(lambda c: r.runq(c[1]), cases))
    bad, bad_nfc = [], []
    n_tok = n_chars = n_nonnfc = 0
    for (name, s), ids in zip(cases, got):
        want = r.hf(s)
        n_tok += len(want)
        n_chars += len(s)
        n_nonnfc += s != unicodedata.normalize("NFC", s)
        if ids != want:
            bad.append((name, s, want, ids))
        if r.text(ids) != r.hf_nfc(s).encode():
            bad_nfc.append(name)
    n_rand = sum(1 for c in cases if c[0].startswith("random"))
    print(f"{len(cases)} cases ({n_rand} random, {len(cases) - n_rand} file pieces), "
          f"{n_chars} chars, {n_tok} HF tokens, {n_nonnfc} cases not NFC, seed {a.seed}")
    print(f"mismatches: {len(bad)} / {len(cases)} = {len(bad) / max(1, len(cases)):.4%}")
    print(f"normalized text != HF NFC: {len(bad_nfc)} {bad_nfc[:5]}")
    for name, s, want, ids in bad[:a.show]:
        m = r.minimize(s)
        print(f"\n{name}: {len(s)} chars, minimized to {m!r}")
        print(f"  hf  : {r.hf(m)}  {[r.tok.convert_ids_to_tokens(i) for i in r.hf(m)]}")
        print(f"  runq: {r.runq(m)}  {[r.tok.convert_ids_to_tokens(i) for i in r.runq(m)]}")
    print(f"\n({time.time() - t0:.1f} s)")
    return 1 if bad or bad_nfc else 0


if __name__ == "__main__":
    sys.exit(main())
