#!/usr/bin/env python3
"""On-demand tokenizer fuzz test: runq -m tokenize vs the HF tokenizer (not in the fast suite).

Usage: .venv/bin/python tests/fuzz_tokenizer.py [-n 2000] [--seed 1] [--jobs 8] [--show 5]
Needs transformers (.venv) and models/Qwen3-0.6B (local files only) plus models/Qwen3-0.6B-q8.bin.

Cases:
  random  -n strings built from a mix of character classes (ASCII words, digits, punctuation,
          whitespace runs incl. \\r\\n, tabs, NBSP and other Unicode spaces, contractions in any
          case, emoji sequences (ZWJ, flags, skin tones, keycaps), CJK incl. rare extension
          blocks, Arabic, Devanagari, Hangul, Cyrillic, Greek, math, combining marks, added
          tokens and broken ones, and random code points from all planes).
  files   real text: the repo's docs and sources and the HF model README (whole files, split
          into <= 64 KiB pieces at line ends because of the argv size limit).
Strings are NFC-normalized before use (runq has no NFC normalizer; documented, tested as XFAIL
in the fast suite), and contain no NUL (argv). Each mismatch is minimized (delta debugging on
characters, as long as it still mismatches) and the first --show are printed.
Exit status 1 if any case mismatches.
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
}
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
    if kind in RANGES:
        return "".join(rand_from_ranges(rng, RANGES[kind]) for _ in range(rng.randint(1, 6)))
    raise ValueError(kind)


KINDS = ["word"] * 6 + ["ascii"] * 4 + ["digits"] * 3 + ["punct"] * 3 + ["space"] * 6 + \
        ["contraction"] * 2 + ["emoji"] * 2 + ["special"] + list(RANGES)


def gen_case(rng):
    # half the cases favour a few kinds (runs of similar text), half mix everything
    kinds = rng.sample(KINDS, rng.randint(2, 4)) if rng.random() < 0.5 else KINDS
    n = rng.choice([1, 2, 3, 5, 10, 20, 40, 80])
    s = "".join(gen_piece(rng, rng.choice(kinds)) for _ in range(n))
    return unicodedata.normalize("NFC", s.replace("\0", ""))


def file_cases():
    out = []
    for rel in FILES:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8", newline="") as f:
            text = f.read()
        if text != unicodedata.normalize("NFC", text):
            print(f"note: {rel} is not NFC, normalizing")
            text = unicodedata.normalize("NFC", text)
        pieces, cur = [], ""
        for line in text.splitlines(keepends=True):
            if cur and len((cur + line).encode()) > MAX_ARG:
                pieces.append(cur)
                cur = ""
            cur += line
        if cur:
            pieces.append(cur)
        for i, p in enumerate(pieces):
            out.append((f"file:{rel}" + (f"#{i}" if len(pieces) > 1 else ""), p))
    return out


class Runner:
    def __init__(self, a):
        from transformers import AutoTokenizer
        self.a = a
        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

    def hf(self, s):
        return self.tok.encode(s, add_special_tokens=False)

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
                if not cand or cand != unicodedata.normalize("NFC", cand):
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-n", type=int, default=2000, help="number of random cases")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--show", type=int, default=5, help="minimize and print this many mismatches")
    ap.add_argument("--no-files", action="store_true", help="skip the real-text cases")
    ap.add_argument("--runq", default=os.path.join(ROOT, "runq"))
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "Qwen3-0.6B-q8.bin"))
    a = ap.parse_args()

    t0 = time.time()
    rng = random.Random(a.seed)
    cases = [(f"random#{i}", gen_case(rng)) for i in range(a.n)]
    if not a.no_files:
        cases += file_cases()
    r = Runner(a)
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        got = list(ex.map(lambda c: r.runq(c[1]), cases))
    bad = []
    n_tok = n_chars = 0
    for (name, s), ids in zip(cases, got):
        want = r.hf(s)
        n_tok += len(want)
        n_chars += len(s)
        if ids != want:
            bad.append((name, s, want, ids))
    n_rand = sum(1 for c in cases if c[0].startswith("random"))
    print(f"{len(cases)} cases ({n_rand} random, {len(cases) - n_rand} file pieces), "
          f"{n_chars} chars, {n_tok} HF tokens, seed {a.seed}")
    print(f"mismatches: {len(bad)} / {len(cases)} = {len(bad) / max(1, len(cases)):.4%}")
    for name, s, want, ids in bad[:a.show]:
        m = r.minimize(s)
        print(f"\n{name}: {len(s)} chars, minimized to {m!r}")
        print(f"  hf  : {r.hf(m)}  {[r.tok.convert_ids_to_tokens(i) for i in r.hf(m)]}")
        print(f"  runq: {r.runq(m)}  {[r.tok.convert_ids_to_tokens(i) for i in r.runq(m)]}")
    print(f"\n({time.time() - t0:.1f} s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
