#!/usr/bin/env python3
"""cwayn regression checks (numpy only). Normally run via tests/run_tests.sh.

Tests (see tests/README.md):
  tokenizer  runq -m tokenize vs HF ids              (tests/golden/tokenizer.json)
             + encode speed (~2,000 tokens, one 30,000-byte chunk), from runq's stderr timing
  logits     runq -m logits vs HF fp32 logits         (tests/golden/logits/*.npz)
             sequential path (-b 1) and batched prefill (-b 64), same thresholds
  batched    batched logits (-b 7, -b 64) vs sequential (-b 1): near-exact (bit-identical now)
  threads    logits bit-identical for 1 vs N OpenMP threads (sequential and batched)
  kernels    logits bit-identical for the AVX-512 VNNI kernels (runq) and the generic C ones
             (runq-generic, -DRUNQ_GENERIC): GEMV (-b 1) and GEMM (-b 7, -b 64) paths
  greedy     runq -t 0 output vs our own golden text  (tests/golden/greedy/*.txt), with the
             default batched prefill (and -b 5 once); seeded -t 0.8 chat equal for -b 1 and 5

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

# Batched vs sequential logits (same engine, same numerics): currently bit-identical. Deliberate
# breakage (scratch builds) gives max |diff| 17..32 and min cos <= 0.47: causal mask leaking one
# position, K RoPE at the chunk-local position, chunk K written one position early.
BATCH_CHUNKS = [7, 64]   # 7: several chunks with boundaries mid-prompt; 64: the default
BATCH_MAX_ABS = 1e-3     # logits are O(10)
BATCH_COS_MIN = 0.99999  # per row, full vocab
BATCH_LONG = "all"       # extra prompt: all logits prompts concatenated (86 tokens, 2 chunks at -b 64)
SEQ, BATCH = 1, 64       # -b of the sequential path and of the default batched path

CTX = "4096"
TOKENIZE_MAX_MS = 100.0  # encode time limit for the speed checks (measured ~1 ms and ~13 ms)
GREEDY_CASES = [
    # name, extra args (all run with -t 0 -c 4096 -s 1)
    ("chat", ["-m", "chat", "-n", "40", "-i", "What is the capital of Australia? Answer in one sentence."]),
    ("chat_thinking", ["-m", "chat", "-r", "1", "-n", "48", "-i", "What is 17 * 23?"]),
    ("chat_system", ["-m", "chat", "-n", "40", "-y", "You are a pirate. Always answer like a pirate.",
                     "-i", "How are you today?"]),
    ("generate", ["-m", "generate", "-n", "40", "-i", "Once upon a time, in a small village by the sea,"]),
    # the same golden with several prefill chunks (generate prints the prompt tokens itself)
    ("generate@b5", ["-m", "generate", "-n", "40", "-b", "5", "-i", "Once upon a time, in a small village by the sea,"]),
]
# seeded sampling (-t 0.8 -s 42): output must not depend on -b (chat advances the RNG once per
# prompt token, as before batched prefill)
SEEDED_ARGS = ["-m", "chat", "-n", "40", "-t", "0.8", "-s", "42", "-c", CTX, "-i", "Tell me about cats."]
SEEDED_CHUNKS = [1, 5]  # token by token vs 2 chunks (-b 64 runs the same code)
THREADS_PROMPT = "en"
# VNNI vs generic kernels: (prompt, -b). -b 1: GEMV (decode path, incl. the classifier);
# -b 7 on en (15 tokens): GEMM tiles of 6 + 1 rows, last chunk 1 row; -b 64 on all (86 tokens,
# chunks 64 + 22): tiles of 6 + 4 rows, LOGITS_ALL classifier GEMM
KERNEL_CASES = [(BATCH_LONG, SEQ), (THREADS_PROMPT, BATCH_CHUNKS[0]), (BATCH_LONG, BATCH)]


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


def runq_tokenize(a, text, timing=False):
    p = run([a.runq, a.model, "-c", "64", "-m", "tokenize", "-i", text], threads=1)
    if p.returncode != 0:
        raise RuntimeError(f"runq -m tokenize failed: {p.stderr.decode(errors='replace')[-500:]}")
    ids = [int(x) for x in p.stdout.split()]
    if not timing:
        return ids
    # stderr: "[tokenize] B bytes -> N tokens, encode X ms, tokenizer load Y ms"
    line = [l for l in p.stderr.decode().splitlines() if l.startswith("[tokenize]")][-1]
    f = line.replace(",", " ").split()
    return ids, float(f[f.index("encode") + 1]), float(f[f.index("load") + 1])


def runq_logits(a, text, path, threads=None, chunk=SEQ):
    p = run([a.runq, a.model, "-c", CTX, "-m", "logits", "-b", str(chunk), "-i", text, "-o", path], threads=threads)
    if p.returncode != 0:
        raise RuntimeError(f"runq -m logits failed: {p.stderr.decode(errors='replace')[-500:]}")
    raw = np.fromfile(path, dtype=np.int32, count=2)
    n, vocab = int(raw[0]), int(raw[1])
    ids = np.fromfile(path, dtype=np.int32, count=2 + n)[2:]
    logits = np.fromfile(path, dtype=np.float32, offset=4 * (2 + n))
    assert logits.size == n * vocab, f"{path}: bad size"
    return ids, logits.reshape(n, vocab)


def logits_file(a, tmp, name, text, chunk, threads=None):
    """runq -m logits for (prompt, chunk, threads), cached in tmp across tests."""
    t = threads or int(os.environ["OMP_NUM_THREADS"])
    path = os.path.join(tmp, f"logits_{name}_b{chunk}_t{t}.bin")
    if not os.path.exists(path):
        runq_logits(a, text, path, threads=t, chunk=chunk)
    return path


def read_logits(path):
    raw = np.fromfile(path, dtype=np.int32, count=2)
    n = int(raw[0])
    return np.fromfile(path, dtype=np.int32, count=2 + n)[2:], \
        np.fromfile(path, dtype=np.float32, offset=4 * (2 + n)).reshape(n, int(raw[1]))


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

    # speed: ~2,000 tokens of varied text (the short golden cases, repeated), and one long
    # pre-tokenizer chunk (no spaces); the encode time is measured inside runq, without the load
    pool = [c for c in cases if len(c["text"]) < 1000 and not c.get("xfail")]
    parts, n = [], 0
    while n < 2000:
        for c in pool:
            parts.append(c["text"])
            n += len(c["ids"])
            if n >= 2000:
                break
    for name, text in [("speed_2k", "\n".join(parts)), ("speed_long_chunk", "Supercalifragilistic" * 1500)]:
        ids, ms, load_ms = runq_tokenize(a, text, timing=True)
        res.report(ms < TOKENIZE_MAX_MS, f"tokenizer/{name}",
                   f"{len(text.encode())} bytes -> {len(ids)} tokens: encode {ms:.2f} ms "
                   f"(max {TOKENIZE_MAX_MS:.0f}), tokenizer load {load_ms:.1f} ms")


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


def test_logits(a, res, tmp, chunk=SEQ, prefix="logits"):
    all_top1, all_cos, rows = [], [], []
    for name, path in golden_prompts():
        g = np.load(path)
        text = str(g["text"])
        ids, logits = read_logits(logits_file(a, tmp, name, text, chunk))
        tname = f"{prefix}/{name}"
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
        res.report(bool(c >= COS_ALL_MIN), f"{prefix}/cos_pooled",
                   f"mean cos over {len(all_cos)} positions (pos >= 1) = {c:.5f} (min {COS_ALL_MIN})")
    if all_top1:
        t = float(np.mean(all_top1))
        res.report(bool(t >= TOP1_ALL_MIN), f"{prefix}/top1_pooled",
                   f"{int(np.sum(all_top1))}/{len(all_top1)} = {t:.3f} (min {TOP1_ALL_MIN})")
    if a.verbose and rows:
        print("      prompt  n   cos pos0  cos mean  cos min   top1    top5")
        for r in rows:
            print(f"      {r[0]:<7} {r[1]:<3} {r[2]:.4f}    {r[3]:.5f}   {r[4]:.5f}   {r[5]:>2}/{r[1]:<4} {r[6]:.3f}")


def batched_prompts():
    texts = [(name, str(np.load(path)["text"])) for name, path in golden_prompts()]
    return texts + [(BATCH_LONG, " ".join(t for _, t in texts))]


def test_batched(a, res, tmp):
    n_same = n_all = 0
    for name, text in batched_prompts():
        ids_s, seq = read_logits(logits_file(a, tmp, name, text, SEQ))
        for chunk in (BATCH_CHUNKS if name != BATCH_LONG else [BATCH]):  # long: 2 chunks at -b 64
            n_all += 1
            tname = f"batched/{name}_b{chunk}"
            ids_b, bat = read_logits(logits_file(a, tmp, name, text, chunk))
            if ids_b.tolist() != ids_s.tolist() or bat.shape != seq.shape:
                res.report(False, tname, "token ids / shape differ from the sequential run")
                continue
            if not np.isfinite(bat).all():
                bad = np.where(~np.isfinite(bat).all(axis=1))[0].tolist()
                res.report(False, tname, f"non-finite logits at positions {bad}")
                continue
            same = np.array_equal(bat.view(np.uint32), seq.view(np.uint32))
            n_same += same
            x, y = seq.astype(np.float64), bat.astype(np.float64)
            d = np.abs(x - y).max(axis=1)
            cos = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
            ok = bool(d.max() <= BATCH_MAX_ABS and cos.min() >= BATCH_COS_MIN)
            n_chunks = -(-len(ids_s) // chunk)
            detail = f"n={len(ids_s)}, {n_chunks} chunk(s): " + ("bit-identical" if same else
                     f"max |diff| {d.max():.3g} (pos {int(d.argmax())}), min cos {cos.min():.7f}")
            if not ok:
                detail += f"  <- max {BATCH_MAX_ABS}, cos {BATCH_COS_MIN}"
            res.report(ok, tname, detail)
    print(f"      {n_same}/{n_all} batched runs bit-identical to the sequential path")


def test_threads(a, res, tmp):
    g = np.load(os.path.join(GOLDEN, "logits", THREADS_PROMPT + ".npz"))
    text = str(g["text"])
    n_a = int(os.environ.get("OMP_NUM_THREADS", "8"))
    n_b = 1 if n_a != 1 else 8
    for chunk, tname in [(SEQ, "threads/bitwise"), (BATCH_CHUNKS[0], f"threads/bitwise_b{BATCH_CHUNKS[0]}")]:
        pa = logits_file(a, tmp, THREADS_PROMPT, text, chunk, threads=n_a)
        pb = logits_file(a, tmp, THREADS_PROMPT, text, chunk, threads=n_b)
        with open(pa, "rb") as fa, open(pb, "rb") as fb:
            same = fa.read() == fb.read()
        detail = f"{THREADS_PROMPT} -b {chunk}: OMP_NUM_THREADS={n_a} vs {n_b} " + ("bit-identical" if same else "DIFFER")
        if not same:
            xa, xb = read_logits(pa)[1], read_logits(pb)[1]
            detail += f" (max abs diff {np.abs(xa - xb).max():.3g})"
        res.report(same, tname, detail)


# ----------------------------------------------------------------------------- kernels

def test_kernels(a, res, tmp):
    if not a.generic:
        print("      skipped: no generic build (--generic / RUNQ_GENERIC)")
        return
    texts = dict(batched_prompts())
    gen = argparse.Namespace(**{**vars(a), "runq": a.generic})
    gtmp = os.path.join(tmp, "generic")
    os.makedirs(gtmp, exist_ok=True)
    for name, chunk in KERNEL_CASES:
        pv = logits_file(a, tmp, name, texts[name], chunk)    # VNNI build (shared cache)
        pg = logits_file(gen, gtmp, name, texts[name], chunk)
        with open(pv, "rb") as fv, open(pg, "rb") as fg:
            same = fv.read() == fg.read()
        n = len(read_logits(pv)[0])
        detail = f"{name} (n={n}) -b {chunk}: " + ("bit-identical" if same else "DIFFER")
        if not same:
            xv, xg = read_logits(pv)[1], read_logits(pg)[1]
            if xv.shape == xg.shape:
                d = np.abs(xv.astype(np.float64) - xg)
                detail += f" (max abs diff {d.max():.3g} at pos {int(d.max(axis=1).argmax())})"
        res.report(same, f"kernels/{name}_b{chunk}", detail)


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
        path = os.path.join(d, name.split("@")[0] + ".txt")  # name@variant: same golden
        if a.update_greedy and "@" in name:
            continue
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
    if a.update_greedy:
        return
    outs = {}
    for chunk in SEEDED_CHUNKS:
        p = run([a.runq, a.model, "-b", str(chunk)] + SEEDED_ARGS)
        if p.returncode != 0:
            raise RuntimeError(f"runq failed: {p.stderr.decode(errors='replace')[-500:]}")
        outs[chunk] = p.stdout
    ref = outs[SEEDED_CHUNKS[0]]
    bad = [c for c in SEEDED_CHUNKS if outs[c] != ref]
    res.report(not bad, "greedy/seeded_chunks",
               f"chat -t 0.8 -s 42, -b {'/'.join(map(str, SEEDED_CHUNKS))}: " +
               ("identical" if not bad else f"-b {bad} differ from -b {SEEDED_CHUNKS[0]}"))


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runq", default=os.path.join(ROOT, "runq"))
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "Qwen3-0.6B-q8.bin"))
    ap.add_argument("--generic", default="", help="generic-kernel build for the kernels test (empty: skip)")
    ap.add_argument("--only", default="tokenizer,logits,batched,threads,kernels,greedy",
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
                         ("logits", lambda: (test_logits(a, res, tmp),
                                             test_logits(a, res, tmp, BATCH, "logits_batched"))),
                         ("batched", lambda: test_batched(a, res, tmp)),
                         ("threads", lambda: test_threads(a, res, tmp)),
                         ("kernels", lambda: test_kernels(a, res, tmp)),
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
