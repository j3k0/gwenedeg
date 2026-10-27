#!/usr/bin/env python3
"""The sampler's top-k path (src/sampler.c, sample_topk) replayed in numpy, draw for draw (TODO P5).

  tests/sampler_ref.py --runq ./runq --api-test build/api_test [--api-test build/api_test-generic] --model MODEL
                       [--draws N] [--tmp DIR] [-v]

Recorded logits: `runq -m logits` on a short prompt (every row), plus rows with ties made from them (values
rounded to 1/2, the top 40 set equal, all equal, a masked-like row). `api_test MODEL --sample FILE T:P:K:N ...`
samples each row N times per setting, draw j of row r with the RNG seeded 1000003 r + j + 1; this script
computes the same draws from the same logits and seeds and requires the very same token for every draw, in
each build. The reference mirrors the C's float32 operations: the candidates = the k largest logits x, ties
by the lower index, weights expf((x - max) * (1 / T)) with the C library's expf (ctypes), their sum in
candidate order, top-p as the first prefix whose sum exceeds topp * sum, the draw r = coin * that
prefix's sum against the running sum; the coin is xorshift's (random_f32). Prints PASS / FAIL lines; exit
status 1 if any failed.
"""
import argparse
import ctypes
import ctypes.util
import os
import subprocess
import sys
import tempfile

import numpy as np

F32 = np.float32
MASK = (1 << 64) - 1
_libm = ctypes.CDLL(ctypes.util.find_library("m"))
_libm.expf.restype = ctypes.c_float
_libm.expf.argtypes = [ctypes.c_float]

# T:P:K settings: the 27B's defaults (1.0 / 0.95 / 20), Qwen's thinking (0.6 / 0.95 / 20) and non-thinking
# (0.7 / 0.8 / 20) recommendations, top-k alone, k = 1 / 2, a low top-p, large k
SETTINGS = ["1.0:0.95:20", "0.6:0.95:20", "0.7:0.8:20", "1.0:1.0:20", "1.5:0.5:40", "1.0:1.0:1", "1.0:1.0:2",
            "2.0:0.9:500", "0.8:1.0:5000"]
PROMPT = "The capital of France is Paris. A short list of fruits: apples, pears,"


M_XS = 0x2545F4914F6CDD1D
M_XS_INV = pow(M_XS, -1, 1 << 64)


def coin(seed):
    """the first random_f32 of a sequence seeded with seed (xorshift, src/sampler.c)"""
    s = seed & MASK
    s ^= s >> 12
    s ^= (s << 25) & MASK
    s ^= s >> 27
    u = ((s * M_XS) & MASK) >> 32
    return F32(u >> 8) / F32(16777216.0)


def _unshift(y, k, left):
    """x from y = x ^ (x >> k) (or << k), 64 bits"""
    x = y
    for i in range(1, 64 // k + 1):
        x ^= ((y << (k * i)) & MASK) if left else y >> (k * i)
    return x


def seed_for(c):
    """an RNG state whose next draw is c / 2^24 (0 <= c < 2^24): xorshift's step inverted"""
    s = ((c << 40) * M_XS_INV) & MASK  # the state after the step: its product's top 24 bits are c
    return _unshift(_unshift(_unshift(s, 27, False), 25, True), 12, False)


def boundary_seeds(prep, nb=8, spread=2):
    """seeds whose coins put r = coin * W within a few steps of the first nb running-sum boundaries: the draws
    that a one-ulp difference in a weight, the sum or the order would change"""
    cand, w, m, wsum = prep
    out, cdf = [], F32(0)
    for j in range(min(m, nb)):
        cdf = F32(cdf + w[j])
        c0 = int(float(cdf) / float(wsum) * 16777216.0)
        out += [seed_for(c) for c in range(c0 - spread, c0 + spread + 1) if 0 <= c < 1 << 24]
    return out


def expf(x):
    return F32(_libm.expf(float(x)))


def prepare(row, temp, topp, k):
    """the part of sample_topk that does not depend on the coin: (candidate ids, weights, m, prefix sum W)"""
    x, inv = row, F32(1) / F32(temp)
    assert x.dtype == np.float32
    if k >= len(x):
        raise ValueError("k >= vocab: top-k is off")
    kth = np.partition(x, len(x) - k)[len(x) - k]  # the k-th largest value: candidates are among x >= kth
    cand = np.nonzero(x >= kth)[0]
    cand = cand[np.lexsort((cand, -x[cand]))][:k]  # value descending, then index ascending
    mx = x[cand[0]]
    w = [expf(F32(F32(x[i] - mx) * inv)) for i in cand]
    s = F32(0)
    for v in w:
        s = F32(s + v)
    m, wsum = k, s
    if 0 < topp < 1:
        t, cum = F32(F32(topp) * s), F32(0)
        for j, v in enumerate(w):
            cum = F32(cum + v)
            if cum > t:
                m, wsum = j + 1, cum
                break
    return cand, w, m, wsum


def draw(prep, u):
    cand, w, m, wsum = prep
    r, cdf, last = F32(u * wsum), F32(0), 0
    for j in range(m):
        cdf = F32(cdf + w[j])
        if r < cdf:
            return int(cand[j])
        if w[j] > 0:
            last = j
    return int(cand[last])


def read_logits(path):
    with open(path, "rb") as f:
        n, v = np.frombuffer(f.read(8), dtype=np.int32)
        ids = np.frombuffer(f.read(4 * n), dtype=np.int32)
        lg = np.frombuffer(f.read(4 * n * v), dtype=np.float32).reshape(n, v)
    return ids, lg


def write_logits(path, lg):
    n, v = lg.shape
    with open(path, "wb") as f:
        f.write(np.array([n, v], dtype=np.int32).tobytes())
        f.write(np.zeros(n, dtype=np.int32).tobytes())
        f.write(np.ascontiguousarray(lg, dtype=np.float32).tobytes())


def tie_rows(lg):
    """rows with ties at the top: rounded to 1/2; the top 40 of a row all equal to its max; all equal; one with
    most values at -1e30 (as a token mask leaves them) and 30 allowed"""
    a = (np.round(lg[3] * 2) / 2).astype(np.float32)
    b = lg[5].copy()
    top = np.argsort(-b, kind="stable")[:40]
    b[top] = b[top[0]]
    c = np.zeros_like(lg[0])
    d = np.full_like(lg[0], -1e30)
    keep = np.random.default_rng(1).choice(len(d), 30, replace=False)
    d[keep] = lg[7][keep]
    return np.stack([a, b, c, d])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runq", required=True)
    ap.add_argument("--api-test", action="append", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--draws", type=int, default=150, help="draws per row and setting")
    ap.add_argument("--tmp", default=None)
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    fails = 0

    def report(ok, name, detail):
        nonlocal fails
        print(f"{'PASS' if ok else 'FAIL'} {name}  {detail}", flush=True)
        fails += not ok

    with tempfile.TemporaryDirectory(prefix="sampler-ref-", dir=a.tmp) as tmp:
        path = os.path.join(tmp, "rec.logits")
        env = dict(os.environ, OMP_NUM_THREADS="2")
        p = subprocess.run([a.runq, a.model, "-m", "logits", "-i", PROMPT, "-o", path], capture_output=True, env=env)
        if p.returncode != 0:
            report(False, "sampler/logits", f"runq -m logits: exit status {p.returncode}: {p.stderr.decode(errors='replace')[-300:]}")
            return 1
        _, lg = read_logits(path)
        rows = np.concatenate([lg, tie_rows(lg)])
        write_logits(path, rows)
        # the plan: per setting and row, a.draws seeded draws (1000003 r + j + 1) + the boundary draws
        plan, refs = [], {}
        for s in SETTINGS:
            t, pp, k = (float(x) for x in s.split(":"))
            for r in range(len(rows)):
                pr = prepare(rows[r], F32(t), F32(pp), int(k))
                seeds = [1000003 * r + j + 1 for j in range(a.draws)] + boundary_seeds(pr)
                plan.append(f"{t:g} {pp:g} {int(k)} {r} " + " ".join(map(str, seeds)))
                refs[(s, r)] = ([draw(pr, coin(x)) for x in seeds], len(seeds) - a.draws)
        outs = {}
        for exe in a.api_test:
            q = subprocess.run([exe, a.model, "--sample", path], input=("\n".join(plan) + "\n").encode(), capture_output=True,
                               env=env)
            got = [[int(t) for t in l.split()[2:]] for l in q.stdout.decode(errors="replace").splitlines() if l.startswith("SAMPLE ")]
            if q.returncode != 0 or len(got) != len(plan):
                report(False, f"sampler/run_{os.path.basename(exe)}", f"exit status {q.returncode}, {len(got)} of {len(plan)} "
                       f"lines: {(q.stdout + q.stderr).decode(errors='replace')[-300:]}")
                return 1
            outs[exe] = {key: g for key, g in zip(refs, got)}
        for s in SETTINGS:
            t, pp, k = s.split(":")
            keys = [key for key in refs if key[0] == s]
            n = sum(len(refs[key][0]) for key in keys)
            nb = sum(refs[key][1] for key in keys)
            distinct = {(key[1], x) for key in keys for x in refs[key][0]}
            bad = {exe: [(key[1], j) for key in keys for j, (x, y) in enumerate(zip(refs[key][0], got[key])) if x != y]
                   for exe, got in outs.items()}
            ok = all(not b for b in bad.values())
            first = next(((os.path.basename(exe), b[0]) for exe, b in bad.items() if b), None)
            detail = f"{n} draws ({nb} at running-sum boundaries; {len(rows)} rows incl. 4 with ties; {len(distinct)} " \
                     f"distinct (row, token)) == numpy, " + ", ".join(os.path.basename(e) for e in outs) + \
                     ("" if ok else f"; {sum(len(b) for b in bad.values())} differ, first {first}")
            report(ok, f"sampler/topk_ref/t{float(t):g}_p{float(pp):g}_k{k}", detail)
        if len(outs) > 1:
            exes = list(outs)
            same = all(outs[e] == outs[exes[0]] for e in exes[1:])
            report(same, "sampler/topk_builds", f"{' == '.join(os.path.basename(e) for e in exes)}: every draw")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
