#!/usr/bin/env python3
"""KL divergence between two GGUF files of one model on agentic text (TODO G3: Qwen3.8-27B Q3_K_M vs Q4_K_M):
KL(ref || test) per position, teacher-forced on pi's recorded sessions rendered as runq serve renders Qwen3.8
(scripts/lookup_accept.py sessions: local data, nothing of it is written here but aggregate numbers).

Each model runs alone (one big process at a time: build/long_ctx -A, every row's logits streamed through a FIFO, so
neither 1 MB per position of logits nor two 27B processes at once): the reference first, keeping per position its
top --topk ids and log-probabilities, its argmax and the log-probability of the text's next token; then the test file,
scored against that:
  KL(p || q) ~= sum_{v in top-K(p)} p_v (log p_v - log q_v) + p_tail log(p_tail / q_tail)
(p_tail / q_tail: the mass outside p's top K; by the log-sum inequality a lower bound of the full KL, tight when
p_tail is small: printed), the top-1 agreement and both files' NLL of the text: on the reply positions (the predicted
token belongs to an assistant message; by kind) and on the sampled context positions by turn (system / user, tool
results included / assistant), with the per-position medians and percentiles. (The means of the system block are
carried by a few positions of the template's tool instructions, prompt-only text where the quantized files are
near-certain and disagree with each other and with fp32: Qwen3.5-0.8B, Q4_K_M there p 0.998 for the text's token where
HF fp32 and Q8_0 give 0.001; read the replies and the user / tool turns.)

Usage: .venv/bin/python -I scripts/kl_quant.py STREAMS.pkl REF.gguf TEST.gguf [--streams 4] [--max-pos 16384]
           [--min-replies 2000] [--every 8] [--topk 1024] [--seed 1] [-t 16] [-k f16] [--out RESULT.json]
(make long-ctx; numpy + tokenizers. --out: also RESULT.npz, per scored position: KL, reply / role / kind, top-1, NLLs.) Streams: --streams of those with >= --max-pos tokens and >= --min-replies reply tokens
among their first --max-pos (one per session), in a random order (--seed), their first --max-pos tokens each. Scored:
every reply position and every --every-th other one (the rest is fed, not scored: the numpy side is the bottleneck).
Temp files (the FIFO, the ids, the reference's top-K: ~8 B x K per scored position) in $TMPDIR.
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KINDS = ["think", "prose", "syntax", "args", "old", "new", "write"]
ROLES = ["system", "user", "assistant"]  # of a context position: the turn it is in (tool results are user turns)
BLOCK = 64


def rows(path, n_expect):
    """yield (row0, float32 logits block) from long_ctx -A's output (int32 N, V, ids[N], float32 [N][V])"""
    with open(path, "rb", buffering=1 << 22) as f:
        n, v = np.frombuffer(f.read(8), np.int32)
        assert n == n_expect, (n, n_expect)
        f.read(4 * n)
        r = 0
        while r < n:
            k = min(BLOCK, n - r)
            b = f.read(4 * v * k)
            if len(b) != 4 * v * k:
                raise RuntimeError(f"short read at row {r}")
            yield r, np.frombuffer(b, np.float32).reshape(k, v)
            r += k


def run_model(a, gguf, ids, tmp, consume):
    """run long_ctx on ids, feeding each block of logits rows to consume(row0, block); returns the stderr tail"""
    idf, fifo = os.path.join(tmp, "s.ids"), os.path.join(tmp, "logits.fifo")
    np.asarray(ids, np.int32).tofile(idf)
    if os.path.exists(fifo):
        os.remove(fifo)
    os.mkfifo(fifo)
    env = dict(os.environ, OMP_NUM_THREADS=str(a.t))
    p = subprocess.Popen([a.long_ctx, gguf, "-I", idf, "-A", "-o", fifo, "-k", a.k], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    err = []
    th = threading.Thread(target=lambda: err.append(p.stderr.read()))
    th.start()
    try:
        for r0, blk in rows(fifo, len(ids)):
            consume(r0, blk)
    finally:
        rc = p.wait()
        th.join()
        os.remove(fifo)
    if rc:
        sys.exit(f"long_ctx failed ({rc}): {err[0].decode(errors='replace')[-400:]}")
    return err[0].decode(errors="replace")


def log_softmax(x):
    # float32 over the vocabulary (the numpy side is the bottleneck), the sum in float64
    m = x.max(axis=1, keepdims=True)
    y = x - m
    return y - np.log(np.exp(y).sum(axis=1, keepdims=True, dtype=np.float64)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("streams")
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--streams", dest="n", type=int, default=4)
    ap.add_argument("--min-replies", type=int, default=2000)
    ap.add_argument("--every", type=int, default=8, help="score every Nth non-reply position")
    ap.add_argument("--max-pos", type=int, default=16384)
    ap.add_argument("--topk", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-t", type=int, default=16)
    ap.add_argument("-k", default="f16", help="KV cache type of both runs")
    ap.add_argument("--long-ctx", default=os.path.join(ROOT, "build", "long_ctx"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    with open(a.streams, "rb") as f:
        streams = pickle.load(f)
    order = np.random.RandomState(a.seed).permutation(len(streams))
    sel, files = [], set()
    for i in order:
        st = streams[i]
        if len(st["ids"]) >= a.max_pos and int((st["kind"][:a.max_pos] >= 0).sum()) >= a.min_replies and \
                st["file"] not in files and len(sel) < a.n:
            sel.append(int(i))
            files.add(st["file"])
    K = a.topk
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(ROOT, "models", "Qwen3.5-0.8B", "tokenizer.json"))
    im_start = tok.token_to_id("<|im_start|>")
    keep = []
    res = {"ref": os.path.basename(a.ref), "test": os.path.basename(a.test), "max_pos": a.max_pos, "topk": K,
           "kv": a.k, "streams": []}
    print(f"# KL({res['ref']} || {res['test']}): {len(sel)} streams x {a.max_pos} positions, top-{K}, KV {a.k}",
          flush=True)
    with tempfile.TemporaryDirectory(prefix="kl_quant-", dir=os.environ.get("TMPDIR", "/var/tmp")) as tmp:
        for si in sel:
            ids = streams[si]["ids"][:a.max_pos].astype(np.int32)
            kind = streams[si]["kind"][:a.max_pos]
            N = len(ids)
            nxt = np.append(ids[1:], 0)
            rep_next = np.append(kind[1:], -1) >= 0
            role, cur = np.zeros(N, np.int8), 0
            for i in np.nonzero(ids[:-1] == im_start)[0]:
                r = tok.decode([int(ids[i + 1])]).strip()
                role[i:] = ROLES.index(r) if r in ROLES else 1
            scored = (rep_next | (np.arange(N) % a.every == 0)) & (np.arange(N) < N - 1)
            top_i = np.zeros((N, K), np.int32)
            top_l = np.zeros((N, K), np.float32)
            am_r, nll_r = np.zeros(N, np.int32), np.zeros(N)

            def ref_block(r0, blk):
                rr = np.nonzero(scored[r0:r0 + len(blk)])[0]
                if not len(rr):
                    return
                lp = log_softmax(blk[rr])
                ti = np.argpartition(-lp, K, axis=1)[:, :K]
                top_i[r0 + rr] = ti
                top_l[r0 + rr] = np.take_along_axis(lp, ti, axis=1)
                am_r[r0 + rr] = lp.argmax(axis=1)
                nll_r[r0 + rr] = -lp[np.arange(len(rr)), nxt[r0 + rr]]
            t0 = time.time()
            run_model(a, a.ref, ids, tmp, ref_block)
            t1 = time.time()
            kl, tail, am_t, nll_t = np.zeros(N), np.zeros(N), np.zeros(N, np.int32), np.zeros(N)

            def test_block(r0, blk):
                rr = np.nonzero(scored[r0:r0 + len(blk)])[0]
                if not len(rr):
                    return
                lq = log_softmax(blk[rr])
                k = len(rr)
                ti, lp = top_i[r0 + rr], top_l[r0 + rr].astype(np.float64)
                lqk = np.take_along_axis(lq, ti, axis=1).astype(np.float64)
                p = np.exp(lp)
                pt = np.clip(1.0 - p.sum(axis=1), 0.0, 1.0)
                qt = np.clip(1.0 - np.exp(lqk).sum(axis=1), 1e-300, 1.0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    tl = np.where(pt > 0, pt * (np.log(np.maximum(pt, 1e-300)) - np.log(qt)), 0.0)
                kl[r0 + rr] = (p * (lp - lqk)).sum(axis=1) + tl
                tail[r0 + rr] = pt
                am_t[r0 + rr] = lq.argmax(axis=1)
                nll_t[r0 + rr] = -lq[np.arange(k), nxt[r0 + rr]]
            run_model(a, a.test, ids, tmp, test_block)
            t2 = time.time()
            del top_i, top_l
            m = scored & ~rep_next  # the sampled non-reply positions (predictions of the context's tokens)
            rep = scored & rep_next
            row = {"stream": si, "tokens": N, "s_ref": round(t1 - t0), "s_test": round(t2 - t1)}
            for name, mk in (("context", m), ("reply", rep)) + tuple(
                    (kn, scored & (np.append(kind[1:], -1) == i)) for i, kn in enumerate(KINDS)):
                n = int(mk.sum())
                if not n:
                    continue
                row[name] = {"n": n, "kl": float(kl[mk].mean()), "kl_p99": float(np.percentile(kl[mk], 99)),
                             "top1": float((am_r[mk] == am_t[mk]).mean()), "nll_ref": float(nll_r[mk].mean()),
                             "nll_test": float(nll_t[mk].mean()), "tail": float(tail[mk].mean())}
            # by position band
            for lo, hi in ((0, 4096), (4096, 8192), (8192, 16384), (16384, 1 << 30)):
                mk = rep & (np.arange(N) >= lo) & (np.arange(N) < hi)
                if mk.any():
                    row[f"pos{lo}"] = {"n": int(mk.sum()), "kl": float(kl[mk].mean()),
                                       "top1": float((am_r[mk] == am_t[mk]).mean())}
            for ri, rn in enumerate(ROLES):
                mk = m & (role == ri)
                if mk.any():
                    row["ctx_" + rn] = {"n": int(mk.sum()), "kl": float(kl[mk].mean()), "kl_median": float(np.median(kl[mk])),
                                        "top1": float((am_r[mk] == am_t[mk]).mean()), "nll_ref": float(nll_r[mk].mean()),
                                        "nll_test": float(nll_t[mk].mean()), "tail": float(tail[mk].mean())}
            res["streams"].append(row)
            keep.append((kl[scored], rep_next[scored], role[scored], np.append(kind[1:], -1)[scored],
                         (am_r == am_t)[scored], nll_r[scored], nll_t[scored]))
            r, rr = row["context"], row.get("reply", {})
            print(f"  stream {si} ({N} tokens, {row['s_ref']} + {row['s_test']} s): replies {rr.get('n', 0)}: KL "
                  f"{rr.get('kl', 0):.5f} top-1 {rr.get('top1', 0):.4f} NLL {rr.get('nll_ref', 0):.4f} -> {rr.get('nll_test', 0):.4f}"
                  f" (tail {rr.get('tail', 0):.1e}); context (1 in {a.every}) {r['n']}: KL {r['kl']:.5f} top-1 {r['top1']:.4f}",
                  flush=True)
    S = res["streams"]
    print("\n# pooled (positions weighted):")
    for name in ["context"] + ["ctx_" + r for r in ROLES] + ["reply"] + KINDS:
        rs = [s[name] for s in S if name in s]
        n = sum(r["n"] for r in rs)
        if not n:
            continue
        w = lambda key: sum(r[key] * r["n"] for r in rs) / n
        tot = {"n": n, "kl": w("kl"), "top1": w("top1"), "nll_ref": w("nll_ref"), "nll_test": w("nll_test"),
               "tail": w("tail")}
        res[name] = tot
        print(f"  {name:13s} n {n:6d}  KL {tot['kl']:.5f}  top-1 {tot['top1']:.4f}  NLL {tot['nll_ref']:.4f} -> "
              f"{tot['nll_test']:.4f} ({(np.exp(tot['nll_test'] - tot['nll_ref']) - 1) * 100:+.2f}% PPL)  tail {tot['tail']:.1e}")
    for lo in (0, 4096, 8192, 16384):
        rs = [s[f"pos{lo}"] for s in S if f"pos{lo}" in s]
        n = sum(r["n"] for r in rs)
        if n:
            print(f"  replies at positions >= {lo:5d}: n {n:6d}  KL {sum(r['kl'] * r['n'] for r in rs) / n:.5f}  top-1 "
                  f"{sum(r['top1'] * r['n'] for r in rs) / n:.4f}")
    if keep:
        kl, rp, ro, kd, t1, nr, nt = (np.concatenate(x) for x in zip(*keep))
        print("  per-position KL, median / p90 / p99 / max (the means are carried by few positions: e.g. the template's "
              "memorized tool instructions in the system block, where both files are near-certain and disagree):")
        for name, mk in [("reply", rp)] + [(f"ctx_{r}", ~rp & (ro == i)) for i, r in enumerate(ROLES)]:
            if mk.any():
                q = np.percentile(kl[mk], [50, 90, 99])
                res[name + "_pct"] = [float(x) for x in q] + [float(kl[mk].max())]
                print(f"    {name:13s} {q[0]:.5f} / {q[1]:.5f} / {q[2]:.4f} / {kl[mk].max():.2f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        if keep:  # per scored position: numbers only (no token of the sessions)
            np.savez(os.path.splitext(a.out)[0] + ".npz", kl=kl, reply=rp, role=ro, kind=kd, top1=t1, nll_ref=nr, nll_test=nt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
