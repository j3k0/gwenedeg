#!/usr/bin/env python3
"""cwayn regression checks (numpy only). Normally run via tests/run_tests.sh.

Tests (see tests/README.md):
  tokenizer  runq -m tokenize vs HF ids              (tests/golden/tokenizer.json)
  logits     runq -m logits vs HF fp32 logits         (tests/golden/logits/*.npz)
  threads    logits bit-identical for 1 vs N OpenMP threads
  greedy     runq -t 0 output vs our own golden text  (tests/golden/greedy/*.txt)

Logits metrics, per position, over U = (fixed 4096-id vocab subset) + (HF top-64 ids):
  cos   cosine similarity of runq vs HF logits on U
  top1  runq argmax over U == HF top-1
  top5  |runq top-5 over U  &  HF top-5| / 5
Position 0 is reported separately: Qwen3 puts a ~6500 "massive activation" in the residual
stream of the first token from layer ~3 on (HF fp32 too), and Q8_0 (64-wide groups, weights
and activations) loses most of the other values next to it, so pos-0 logits are much noisier.

Baseline, Q8_0 engine at the commit adding this suite (2026-10-24, OMP_NUM_THREADS=8):
  prompt  n   cos pos0  cos mean  cos min   top1    top5     (cos mean/min over pos >= 1)
  en      15  0.8316    0.99833   0.99607   13/15   0.893
  code    24  0.8323    0.99920   0.99838   21/24   0.942
  zh      15  0.9064    0.99958   0.99930   14/15   0.880
  chat    15  0.8967    0.99888   0.99663   14/15   0.920
  count   15  0.8653    0.99941   0.99901   14/15   0.907
  pooled: cos mean 0.99909 (79 positions), top1 76/84 = 0.905
Deliberate breakage (scratch builds, all fail): wrong RoPE theta 1e4 (the subtlest: pooled cos
0.9876, per-prompt cos mean 0.977..0.995), interleaved RoPE pairs (0.80..0.91), no QK-norm,
KV cache written at pos+1, V read from t-1, no final norm, attention scale 1/d, rmsnorm eps
1e-2, matmul dropping its last group (all <= 0.96). Rounding -> truncation in the activation
quantizer (a pure precision loss) fails logits/en (cos mean 0.99432) and greedy.
"""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "tests", "golden")

# Thresholds: ~3x the baseline error (1 - cos) for the means, ~5x for single positions.
COS_MEAN_MIN = 0.995   # per prompt, mean over positions >= 1
COS_POS_MIN = 0.980    # every single position >= 1
COS_POS0_MIN = 0.75    # position 0 (massive-activation token, inherently noisy under Q8_0)
TOP1_MIN = 0.75        # per prompt, fraction of positions
TOP1_ALL_MIN = 0.85    # all prompts pooled
COS_ALL_MIN = 0.997    # all prompts pooled, mean over positions >= 1
TOP5_MIN = 0.75        # per prompt, mean overlap

CTX = "4096"
GREEDY_CASES = [
    # name, extra args (all run with -t 0 -c 4096 -s 1)
    ("chat", ["-m", "chat", "-n", "40", "-i", "What is the capital of Australia? Answer in one sentence."]),
    ("chat_thinking", ["-m", "chat", "-r", "1", "-n", "48", "-i", "What is 17 * 23?"]),
    ("chat_system", ["-m", "chat", "-n", "40", "-y", "You are a pirate. Always answer like a pirate.",
                     "-i", "How are you today?"]),
    ("generate", ["-m", "generate", "-n", "40", "-i", "Once upon a time, in a small village by the sea,"]),
]
THREADS_PROMPT = "en"


class Results:
    def __init__(self):
        self.passed, self.failed, self.xfailed, self.xpassed = [], [], [], []

    def report(self, ok, name, detail="", xfail=None):
        if xfail:
            if ok:
                self.xpassed.append(name)
                print(f"XPASS {name}  (marked xfail: {xfail}; remove the mark?)")
            else:
                self.xfailed.append(name)
                print(f"XFAIL {name}  ({xfail})")
        elif ok:
            self.passed.append(name)
            print(f"PASS  {name}" + (f"  {detail}" if detail else ""))
        else:
            self.failed.append(name)
            print(f"FAIL  {name}" + (f"  {detail}" if detail else ""))
        sys.stdout.flush()


def run(cmd, threads=None, **kw):
    env = dict(os.environ)
    if threads is not None:
        env["OMP_NUM_THREADS"] = str(threads)
    return subprocess.run(cmd, env=env, capture_output=True, **kw)


def runq_tokenize(a, text):
    p = run([a.runq, a.model, "-c", "64", "-m", "tokenize", "-i", text], threads=1)
    if p.returncode != 0:
        raise RuntimeError(f"runq -m tokenize failed: {p.stderr.decode(errors='replace')[-500:]}")
    return [int(x) for x in p.stdout.split()]


def runq_logits(a, text, path, threads=None):
    p = run([a.runq, a.model, "-c", CTX, "-m", "logits", "-i", text, "-o", path], threads=threads)
    if p.returncode != 0:
        raise RuntimeError(f"runq -m logits failed: {p.stderr.decode(errors='replace')[-500:]}")
    raw = np.fromfile(path, dtype=np.int32, count=2)
    n, vocab = int(raw[0]), int(raw[1])
    ids = np.fromfile(path, dtype=np.int32, count=2 + n)[2:]
    logits = np.fromfile(path, dtype=np.float32, offset=4 * (2 + n))
    assert logits.size == n * vocab, f"{path}: bad size"
    return ids, logits.reshape(n, vocab)


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


# ----------------------------------------------------------------------------- tokenizer

def test_tokenizer(a, res):
    with open(os.path.join(GOLDEN, "tokenizer.json"), encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        outs = list(ex.map(lambda c: runq_tokenize(a, c["text"]), cases))
    for c, got in zip(cases, outs):
        want = c["ids"]
        ok = got == want
        detail = ""
        if not ok:
            i = first_diff(got, want)
            detail = (f"text={c['text']!r} first diff at index {i}\n"
                      f"      hf  : {want[max(0, i - 3):i + 6]}\n      runq: {got[max(0, i - 3):i + 6]}")
        res.report(ok, f"tokenizer/{c['name']}", detail, c.get("xfail"))
        if not ok and c.get("xfail") and a.verbose:
            print("      " + detail)


# ----------------------------------------------------------------------------- logits

def logit_metrics(g, logits):
    """Per-position metrics of runq logits (n, vocab) vs the golden npz g."""
    sub_ids, sub, top_ids, top_logits = g["subset_ids"], g["subset"], g["top_ids"], g["top_logits"]
    cos, top1, top5 = [], [], []
    for i in range(len(g["ids"])):
        ids = np.concatenate([top_ids[i], sub_ids])
        ref = np.concatenate([top_logits[i], sub[i].astype(np.float32)])
        ids, first = np.unique(ids, return_index=True)  # keep the fp32 top-64 value for duplicates
        ref = ref[first].astype(np.float64)
        got = logits[i, ids].astype(np.float64)
        cos.append(float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref))))
        top1.append(int(ids[np.argmax(got)]) == int(top_ids[i, 0]))
        rq5 = set(ids[np.argsort(-got, kind="stable")[:5]].tolist())
        top5.append(len(rq5 & set(top_ids[i, :5].tolist())) / 5)
    return np.array(cos), np.array(top1), np.array(top5)


def golden_prompts():
    d = os.path.join(GOLDEN, "logits")
    order = ["en", "code", "zh", "chat", "count"]
    names = sorted((f[:-4] for f in os.listdir(d) if f.endswith(".npz")),
                   key=lambda n: (order.index(n) if n in order else len(order), n))
    return [(n, os.path.join(d, n + ".npz")) for n in names]


def test_logits(a, res, tmp):
    all_top1, all_cos, rows = [], [], []
    for name, path in golden_prompts():
        g = np.load(path)
        text = str(g["text"])
        out = os.path.join(tmp, f"logits_{name}.bin")
        ids, logits = runq_logits(a, text, out)
        tname = f"logits/{name}"
        if ids.tolist() != g["ids"].tolist():
            i = first_diff(ids.tolist(), g["ids"].tolist())
            res.report(False, tname, f"token ids differ from HF at index {i} (tokenizer bug?), cannot compare\n"
                       f"      hf  : {g['ids'].tolist()}\n      runq: {ids.tolist()}")
            continue
        if not np.isfinite(logits).all():
            bad = np.where(~np.isfinite(logits).all(axis=1))[0].tolist()
            res.report(False, tname, f"non-finite logits at positions {bad}")
            continue
        cos, top1, top5 = logit_metrics(g, logits)
        all_top1.extend(top1.tolist())
        c0, cr = cos[0], cos[1:]
        all_cos.extend(cr.tolist())
        rows.append((name, len(ids), c0, cr.mean(), cr.min(), int(top1.sum()), top5.mean()))
        why = []  # written as `not (x >= t)` so that NaN fails
        if not cr.mean() >= COS_MEAN_MIN: why.append(f"cos mean < {COS_MEAN_MIN}")
        if not cr.min() >= COS_POS_MIN: why.append(f"cos min < {COS_POS_MIN} at pos {1 + int(cr.argmin())}")
        if not c0 >= COS_POS0_MIN: why.append(f"cos pos0 < {COS_POS0_MIN}")
        if not top1.mean() >= TOP1_MIN: why.append(f"top1 < {TOP1_MIN}")
        if not top5.mean() >= TOP5_MIN: why.append(f"top5 < {TOP5_MIN}")
        detail = (f"n={len(ids)} cos pos0 {c0:.4f}, pos1+ mean {cr.mean():.5f} min {cr.min():.5f} | "
                  f"top1 {int(top1.sum())}/{len(top1)} | top5 {top5.mean():.3f}")
        res.report(not why, tname, detail + ("  <- " + "; ".join(why) if why else ""))
        if a.verbose:
            for i in range(len(cos)):
                print(f"      pos {i:2d} cos {cos[i]:.5f} top1 {int(top1[i])} top5 {top5[i]:.1f}")
    if all_cos:
        c = float(np.mean(all_cos))
        res.report(bool(c >= COS_ALL_MIN), "logits/cos_pooled",
                   f"mean cos over {len(all_cos)} positions (pos >= 1) = {c:.5f} (min {COS_ALL_MIN})")
    if all_top1:
        t = float(np.mean(all_top1))
        res.report(bool(t >= TOP1_ALL_MIN), "logits/top1_pooled",
                   f"{int(np.sum(all_top1))}/{len(all_top1)} = {t:.3f} (min {TOP1_ALL_MIN})")
    if a.verbose and rows:
        print("      prompt  n   cos pos0  cos mean  cos min   top1    top5")
        for r in rows:
            print(f"      {r[0]:<7} {r[1]:<3} {r[2]:.4f}    {r[3]:.5f}   {r[4]:.5f}   {r[5]:>2}/{r[1]:<4} {r[6]:.3f}")


def test_threads(a, res, tmp):
    g = np.load(os.path.join(GOLDEN, "logits", THREADS_PROMPT + ".npz"))
    text = str(g["text"])
    n_a = int(os.environ.get("OMP_NUM_THREADS", "8"))
    n_b = 1 if n_a != 1 else 8
    pa, pb = os.path.join(tmp, "threads_a.bin"), os.path.join(tmp, "threads_b.bin")
    a_path = os.path.join(tmp, f"logits_{THREADS_PROMPT}.bin")
    if os.path.exists(a_path):  # reuse the logits-test output (made with n_a threads)
        pa = a_path
    else:
        runq_logits(a, text, pa, threads=n_a)
    runq_logits(a, text, pb, threads=n_b)
    with open(pa, "rb") as fa, open(pb, "rb") as fb:
        same = fa.read() == fb.read()
    detail = f"{THREADS_PROMPT}: OMP_NUM_THREADS={n_a} vs {n_b} " + ("bit-identical" if same else "DIFFER")
    if not same:
        xa = np.fromfile(pa, dtype=np.float32, offset=4 * (2 + len(g["ids"])))
        xb = np.fromfile(pb, dtype=np.float32, offset=4 * (2 + len(g["ids"])))
        detail += f" (max abs diff {np.abs(xa - xb).max():.3g})"
    res.report(same, "threads/bitwise", detail)


# ----------------------------------------------------------------------------- greedy

def greedy_output(a, args):
    p = run([a.runq, a.model, "-t", "0", "-c", CTX, "-s", "1"] + args)
    if p.returncode != 0:
        raise RuntimeError(f"runq failed: {p.stderr.decode(errors='replace')[-500:]}")
    return p.stdout.decode("utf-8", errors="replace")


def test_greedy(a, res):
    d = os.path.join(GOLDEN, "greedy")
    os.makedirs(d, exist_ok=True)
    for name, args in GREEDY_CASES:
        path = os.path.join(d, name + ".txt")
        got = greedy_output(a, args)
        if a.update_greedy:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(got)
            print(f"UPDATED greedy/{name} -> {os.path.relpath(path, ROOT)}")
            continue
        if not os.path.exists(path):
            res.report(False, f"greedy/{name}", "no golden; run tests/run_tests.sh --update-greedy")
            continue
        with open(path, encoding="utf-8", newline="") as f:
            want = f.read()
        if got == want:
            res.report(True, f"greedy/{name}")
        else:
            i = first_diff(got, want)
            res.report(False, f"greedy/{name}",
                       f"REGRESSION (greedy): output differs at char {i}: "
                       f"golden {want[i:i + 20]!r} vs runq {got[i:i + 20]!r}\n"
                       f"      --- golden ---\n{want}\n      --- runq ---\n{got}\n"
                       f"      (if the numerics changed on purpose and logits/* pass: "
                       f"tests/run_tests.sh --update-greedy)")


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runq", default=os.path.join(ROOT, "runq"))
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "Qwen3-0.6B-q8.bin"))
    ap.add_argument("--only", default="tokenizer,logits,threads,greedy",
                    help="comma-separated subset of tests to run")
    ap.add_argument("--update-greedy", action="store_true", help="regenerate the greedy goldens")
    ap.add_argument("--jobs", type=int, default=8, help="parallel runq processes for the tokenizer test")
    ap.add_argument("-v", "--verbose", action="store_true", help="per-position logits metrics")
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "8")

    t0 = time.time()
    res = Results()
    only = set(a.only.split(","))
    if a.update_greedy:
        only = {"greedy"}
    with tempfile.TemporaryDirectory(prefix="cwayn-test-") as tmp:
        for name, fn in [("tokenizer", lambda: test_tokenizer(a, res)),
                         ("logits", lambda: test_logits(a, res, tmp)),
                         ("threads", lambda: test_threads(a, res, tmp)),
                         ("greedy", lambda: test_greedy(a, res))]:
            if name in only:
                t = time.time()
                print(f"== {name}")
                try:
                    fn()
                except Exception as e:  # a crashing runq is a failure, not a traceback
                    res.report(False, f"{name}/error", str(e))
                print(f"   ({time.time() - t:.1f} s)")
    dt = time.time() - t0
    print(f"\nSummary: {len(res.passed)} passed, {len(res.failed)} failed, "
          f"{len(res.xfailed)} xfail, {len(res.xpassed)} xpass, {dt:.1f} s "
          f"(OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']})")
    for n in res.failed:
        print(f"  FAILED: {n}")
    return 1 if res.failed else 0


if __name__ == "__main__":
    sys.exit(main())
