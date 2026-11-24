#!/usr/bin/env python3
"""KL divergence of gwenedeg runs against a reference run on token-id files (docs/v4h.md: the activation format test).

Each run is `build/long_ctx MODEL -I IDS -A` (every position's logits streamed through a FIFO) with its own environment
(GWENEDEG_ACTQ_SIM=v4h|flat for the activation formats). The reference keeps per position its top --topk log-probs;
a test run is scored against them as scripts/kl_quant.py does:
  KL(p || q) ~= sum_{v in top-K(p)} p_v (log p_v - log q_v) + p_tail log(p_tail / q_tail)   (a lower bound, tight)
plus the top-1 agreement and both runs' NLL of the next token (perplexity ratio).

Usage: .venv/bin/python -I tools/v4h_kl.py IDS1[,IDS2...] REF_MODEL[:ENV=V] TEST_MODEL[:ENV=V][:ARG=-F] [TEST ...] [-t 8]
  (ids: int32 files from `build/long_ctx MODEL -T TEXT -o IDS`; each file is one sequence)
"""
import argparse
import os
import subprocess
import sys
import tempfile
import threading

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOCK = 64


def parse(spec):
    # MODEL[:ENV=V ...][:ARG=-F]: environment variables, and long_ctx arguments (ARG; -F: --act256)
    model, *envs = spec.split(":")
    kv = [e.split("=", 1) for e in envs]
    return model, {k: v for k, v in kv if k != "ARG"}, [v for k, v in kv if k == "ARG"]


RUN_KV = "f16"


def run(model, env_extra, ids_path, n, t, consume, args=()):
    tmp = tempfile.mkdtemp()
    fifo = os.path.join(tmp, "l.fifo")
    os.mkfifo(fifo)
    env = dict(os.environ, OMP_NUM_THREADS=str(t), **env_extra)
    p = subprocess.Popen([os.path.join(ROOT, "build", "long_ctx"), model, "-I", ids_path, "-A", "-o", fifo, "-k", RUN_KV, *args],
                         env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    err = []
    th = threading.Thread(target=lambda: err.append(p.stderr.read()))
    th.start()
    with open(fifo, "rb", buffering=1 << 22) as f:
        nn, v = np.frombuffer(f.read(8), np.int32)
        assert nn == n
        f.read(4 * n)
        r = 0
        while r < n:
            k = min(BLOCK, n - r)
            consume(r, np.frombuffer(f.read(4 * v * k), np.float32).reshape(k, v))
            r += k
    rc = p.wait()
    th.join()
    os.remove(fifo)
    os.rmdir(tmp)
    if rc:
        sys.exit(f"long_ctx failed: {err[0].decode(errors='replace')[-400:]}")


def lsm(x):
    m = x.max(axis=1, keepdims=True)
    y = x - m
    return y - np.log(np.exp(y).sum(axis=1, keepdims=True, dtype=np.float64)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids")
    ap.add_argument("ref")
    ap.add_argument("tests", nargs="+")
    ap.add_argument("-t", type=int, default=8)
    ap.add_argument("--topk", type=int, default=1024)
    ap.add_argument("-k", default="f16", help="KV cache type of the runs")
    a = ap.parse_args()
    K = a.topk
    global RUN_KV
    RUN_KV = a.k
    seqs = [np.fromfile(p, np.int32) for p in a.ids.split(",")]
    refs = []
    rm, renv, rargs = parse(a.ref)
    for path, ids in zip(a.ids.split(","), seqs):
        n = len(ids)
        top = np.zeros((n, K), np.int32)
        lp = np.zeros((n, K), np.float32)
        nll = np.zeros(n, np.float64)
        amax = np.zeros(n, np.int32)

        def c(r0, blk):
            L = lsm(blk)
            idx = np.argpartition(-L, K, axis=1)[:, :K]
            top[r0:r0 + len(blk)] = idx
            lp[r0:r0 + len(blk)] = np.take_along_axis(L, idx, 1)
            amax[r0:r0 + len(blk)] = L.argmax(1)
            nx = ids[r0 + 1:r0 + 1 + len(blk)]
            nll[r0:r0 + len(nx)] = -L[np.arange(len(nx)), nx]
        run(rm, renv, path, n, a.t, c, rargs)
        refs.append((top, lp, nll, amax))
    print(f"reference {a.ref}: {sum(len(s) for s in seqs)} tokens, PPL " +
          " / ".join(f"{np.exp(r[2][:-1].mean()):.4f}" for r in refs))
    for spec in a.tests:
        m, env, args = parse(spec)
        out = []
        for (path, ids), (top, lp, rnll, ramax) in zip(zip(a.ids.split(","), seqs), refs):
            n = len(ids)
            kl = np.zeros(n)
            same = np.zeros(n, bool)
            nll = np.zeros(n)

            def c(r0, blk):
                L = lsm(blk)
                sl = slice(r0, r0 + len(blk))
                p = np.exp(lp[sl].astype(np.float64))
                q = np.exp(np.take_along_axis(L, top[sl], 1).astype(np.float64))
                pt = np.clip(1 - p.sum(1), 1e-12, 1)
                qt = np.clip(1 - q.sum(1), 1e-12, 1)
                kl[sl] = (p * (lp[sl] - np.log(q))).sum(1) + pt * np.log(pt / qt)
                same[sl] = L.argmax(1) == ramax[sl]
                nx = ids[r0 + 1:r0 + 1 + len(blk)]
                nll[r0:r0 + len(nx)] = -L[np.arange(len(nx)), nx]
            run(m, env, path, n, a.t, c, args)
            out.append((kl[:-1], same[:-1], nll[:-1], rnll[:-1]))
        cols = []
        for kl, same, nll, rnll in out:
            se = kl.std() / np.sqrt(len(kl))
            cols.append(f"KL {kl.mean():.5f}±{se:.5f} p99 {np.percentile(kl, 99):.3f} top1 {100 * same.mean():.2f}% "
                        f"PPL x{np.exp(nll.mean() - rnll.mean()):.4f}")
        print(f"{spec}\n    " + "\n    ".join(cols), flush=True)


if __name__ == "__main__":
    main()
