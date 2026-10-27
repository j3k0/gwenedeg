#!/usr/bin/env python3
"""The long-context reference (tests/golden/long/<tag>.npz) of tests/check_long.py (TODO G2): tests/ref_qwen35.py's
forward_long (numpy fp32 on the GGUF's weights dequantized with ggml's expressions; exact softmax per query in blocks
of 128 queries, rows in chunks of 2,048: ~2 GB for the 0.8B at 98k) on the first n tokens of check_long's corpus,
the logits at check_long.POSITIONS (full vocabulary), stored as the top 64 ids / logits and a fixed subset of 512 ids
per position (~40 KB). Also stores the max |K| / |V| of each attention layer (the f16 cache's range) and the run's time.

Usage: .venv/bin/python tests/make_long_ref.py TAG FILE.gguf [-n 98304] [--rows OUT.npy] [--check-short]
  (build/long_ctx tokenizes: `make long-ctx`; OMP_NUM_THREADS / OPENBLAS_NUM_THREADS bound numpy's BLAS threads)
  --rows: also the full-vocabulary rows (for analysis, not committed). --check-short: forward_long == forward on a
  short prompt (the chunked code against the plain one) and exit.
Measured (Qwen3.5-0.8B Q8_0, 98,304 tokens, 8 threads): see docs/qwen35.md, "Long context".
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import check_long as L  # noqa: E402
import ggufmini as G  # noqa: E402
import ref_qwen35 as R35  # noqa: E402

TOPK = 64


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tag")
    ap.add_argument("model")
    ap.add_argument("-n", type=int, default=L.LONG_N)
    ap.add_argument("--rows", default="")
    ap.add_argument("--check-short", action="store_true")
    a = ap.parse_args()
    w = R35.GgufWeights(G.GgufFile(a.model))
    tmp = os.environ.get("TMPDIR", "/var/tmp")
    ids = L.corpus_ids(os.path.join(ROOT, "build", "long_ctx"), a.model, tmp)[:a.n]
    if a.check_short:
        n = 700
        t = time.time()
        full = R35.forward(w, [ids[:n].tolist()])[0]
        pos = np.array([0, 255, 256, 257, 300, 511, 699])
        lng = R35.forward_long(w, ids[:n], pos, rc=256)
        d = np.abs(full[pos] - lng).max()
        print(f"forward_long vs forward, {n} tokens, rc 256: max |diff| {d:.3g}, top-1 equal "
              f"{(full[pos].argmax(1) == lng.argmax(1)).all()} ({time.time() - t:.0f} s)")
        return 0 if d < 1e-3 else 1
    assert len(ids) == a.n, f"the corpus has {len(ids)} tokens"
    pos = np.array([p for p in L.POSITIONS if p < a.n], np.int32)
    t = time.time()
    stats = {}
    rows = R35.forward_long(w, ids, pos, verbose=True, stats=stats)
    dt = time.time() - t
    if a.rows:
        np.save(a.rows, rows)
    top_ids = np.argsort(-rows, axis=1, kind="stable")[:, :TOPK].astype(np.int32)
    top_logits = np.take_along_axis(rows, top_ids, axis=1).astype(np.float32)
    sub = L.subset_ids(rows.shape[1])
    os.makedirs(L.GOLDEN, exist_ok=True)
    out = os.path.join(L.GOLDEN, f"{a.tag}.npz")
    np.savez_compressed(out, n=np.int32(a.n), ids_hash=np.array(L.ids_hash(ids)), corpus_rev=np.array(L.CORPUS_REV),
                        positions=pos, top_ids=top_ids, top_logits=top_logits, subset_ids=sub,
                        subset=rows[:, sub].astype(np.float32), kv_max=np.array(sorted(stats.items()), dtype=object).astype(str),
                        seconds=np.float32(dt))
    print(f"{out}: {len(pos)} positions, {dt:.0f} s; max |K| / |V| per attention layer: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
