"""Group `long` (opt-in: tests/run_tests.sh --long-context; TODO G2): correctness at long context on Qwen3.5-0.8B Q8_0
(and other qwen35 files with a golden), through build/long_ctx (tests/long_ctx.c: the library API; the CLI's hooks cannot
take ~100k tokens).

The text: files of this repository at a fixed commit (CORPUS_REV, read with `git show`, so the text never changes; prose,
C and Python interleaved, no repeats), tokenized by runq, the first LONG_N = 98,304 tokens. Checks:
  long/<tag>/ids            the token ids == the golden's (count + hash: the text and the tokenizer)
  long/<tag>/ref_p<P>       logits at sampled positions P (1k .. 96k) vs tests/ref_qwen35.py on the file's dequantized
                            weights (tests/make_long_ref.py: numpy fp32, chunked attention; golden/long/<tag>.npz holds the
                            top-64 and a fixed 512-id subset of each row): cos over those ids, top-1, max |diff|
  long/<tag>/cos_pooled, top1_pooled
  long/<tag>/verify_p<P>    at each position the same row's logits by three paths, bitwise: the last row of a 64-row
                            prefill chunk, a 4-row verify segment (rq_forward_batch, every row's logits: the decode-rows
                            worker) and a decode step (rq_decode: per-block split of the attention over the threads, then
                            att_combine)
  long/<tag>/generic_p<P>   positions <= GENERIC_MAX on build/long_ctx-generic (C attention, generic kernels) == native
  long/<tag>/retrieval      a fact planted ~4k tokens into a ~88k-token chat prompt, asked at its end: the greedy reply
                            (thinking off) contains it (and the decode steps at ~88k: the split / combine of decode)
Thresholds (TH): per position cos >= 0.995 over the golden's ids and the same top-1 (or a reference margin < 0.5), pooled
cos >= 0.999 and top-1 >= 0.85 (measured: cos 0.99988-0.99998, top-1 20/20; runq's int8 activations). Run alone: .venv/bin/python tests/check_long.py [--tags TAG,...] (56 checks, ~35 min at 8 threads with both
goldens: the 0.8B ~18 min, the 4B ~18 min); --retrieval FILE: only the retrieval check, on any qwen35 file.
"""
import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GOLDEN = os.path.join(HERE, "golden", "long")

CORPUS_REV = "5d93873aa3fed6189c10341bcd49e6c689f860c1"
CORPUS = ["README.md", "docs/qwen35.md", "src/forward.c", "docs/qwen38-speculative.md", "src/model.c", "TODO.md",
          "bench/ppl_beagle_ch1.txt", "src/engine.c", "docs/DEVELOPMENT.md", "src/tokenizer.c", "BENCHMARKS.md",
          "src/serve.c", "docs/server.md", "src/cli.c", "IDEAS.md", "src/spec.c", "tests/check.py", "src/chat.c"]
LONG_N = 98304
# sampled positions: short (the existing tests' range), then every 8k to 96k, and the last
POSITIONS = [1023, 4095, 8191, 16383, 24575, 32767, 40959, 49151, 57343, 65535, 73727, 81919, 90111, 98303]
GENERIC_MAX = 16383                   # build/long_ctx-generic: positions up to here (its C attention is slower)
SUBSET = 512                          # fixed random ids per golden row (+ the reference's top 64)
# per-position thresholds over the golden's ids (Q8_0 vs its dequantized-weights reference: the models group's
# logits_ref bounds are cos mean 0.995 / every position 0.97 / top-1 0.75 for Q4_K_M; the 0.8B Q8_0 measures
# ~0.9999 at short context)
TH = dict(cos_pos=0.995, cos_pooled=0.999, top1=0.85, top1_margin=0.5)
FACT = ("By the way, for anyone who reads this far: the access code of the old lighthouse at Penmarc'h is "
        "PLUM-7316-HERON. Keep it in mind.")
QUESTION = ("What is the access code of the old lighthouse at Penmarc'h, mentioned once in the text above? "
            "Answer with the code only.")
FACT_AT = 4000     # tokens of corpus before the fact
RETR_N = 88000     # tokens of corpus in the retrieval prompt (+ the fact, the question, the template)
MODELS = {"3.5-0.8b-q8": "models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf",
          "3.5-4b-q8": "models/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q8_0.gguf"}


def corpus_text():
    """the corpus: CORPUS at CORPUS_REV (git show), each file after a header line"""
    parts = []
    for f in CORPUS:
        p = subprocess.run(["git", "-C", ROOT, "show", f"{CORPUS_REV}:{f}"], capture_output=True)
        if p.returncode:
            raise RuntimeError(f"git show {CORPUS_REV[:7]}:{f}: {p.stderr.decode(errors='replace').strip()} "
                               "(the long-context text needs the repository's history)")
        parts.append(f"\n\n===== {f} =====\n\n" + p.stdout.decode())
    return "".join(parts).lstrip()


def tokenize(tool, model, text, tmp):
    tf, idf = os.path.join(tmp, "text.txt"), os.path.join(tmp, "text.ids")
    with open(tf, "w", encoding="utf-8") as f:
        f.write(text)
    p = subprocess.run([tool, model, "-T", tf, "-o", idf], capture_output=True)
    if p.returncode:
        raise RuntimeError(f"long_ctx -T: {p.stderr.decode(errors='replace')[-400:]}")
    return np.fromfile(idf, dtype=np.int32)


def corpus_ids(tool, model, tmp):
    return tokenize(tool, model, corpus_text(), tmp)


def ids_hash(ids):
    return hashlib.sha256(np.asarray(ids, np.int32).tobytes()).hexdigest()[:16]


def subset_ids(vocab, seed=1234):
    return np.sort(np.random.default_rng(seed).choice(vocab, SUBSET, replace=False)).astype(np.int32)


def read_rows(path):
    np_, v = np.fromfile(path, dtype=np.int32, count=2)
    pos = np.fromfile(path, dtype=np.int32, count=2 + np_)[2:]
    return pos, np.fromfile(path, dtype=np.float32, offset=4 * (2 + np_)).reshape(np_, v)


def run_tool(cmd, threads, log=None):
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, env=dict(os.environ, OMP_NUM_THREADS=str(threads)))
    if log is not None:
        log.append(f"{os.path.basename(cmd[0])} {time.time() - t:.0f} s")
    if p.returncode:
        raise RuntimeError(f"{' '.join(cmd[:3])}...: {p.stderr.decode(errors='replace')[-600:]}")
    return p.stdout.decode(errors="replace")


def metrics(g, j, row):
    """cos over the golden row's ids (top 64 + subset), top-1 (runq's full-vocab argmax == the reference's, or the
    reference's margin between them < top1_margin), max |diff| over those ids"""
    ids = np.concatenate([g["top_ids"][j], g["subset_ids"]])
    ref = np.concatenate([g["top_logits"][j], g["subset"][j]]).astype(np.float64)
    ids, first = np.unique(ids, return_index=True)
    ref = ref[first]
    got = row[ids].astype(np.float64)
    cos = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref)))
    a = int(np.argmax(row))
    t0 = int(g["top_ids"][j, 0])
    margin = None
    if a != t0:  # the reference's logit of runq's choice, if in the golden's ids
        k = np.where(g["top_ids"][j] == a)[0]
        margin = float(g["top_logits"][j, 0] - g["top_logits"][j, k[0]]) if len(k) else float("inf")
    return cos, a == t0, margin, float(np.abs(got - ref).max())


def test_long(a, res, tmp, tags=None):
    tool = os.path.join(ROOT, "build", "long_ctx")
    gtool = os.path.join(ROOT, "build", "long_ctx-generic")
    th = int(os.environ.get("OMP_NUM_THREADS", "8"))
    for tag, rel in MODELS.items():
        if tags and tag not in tags:
            continue
        model = os.path.join(ROOT, rel) if not os.path.isabs(rel) else rel
        gp = os.path.join(GOLDEN, f"{tag}.npz")
        if not os.path.exists(model) or not os.path.exists(gp):
            print(f"   (long/{tag}: skipped, {'no golden ' + gp if os.path.exists(model) else rel + ' missing'})")
            continue
        g = np.load(gp)
        pre = f"long/{tag}"
        ids = corpus_ids(tool, model, tmp)[:int(g["n"])]
        ok = len(ids) == int(g["n"]) and ids_hash(ids) == str(g["ids_hash"])
        res.report(ok, f"{pre}/ids", f"{len(ids)} tokens of the corpus at {CORPUS_REV[:7]}, hash {ids_hash(ids)} "
                   f"(golden {g['ids_hash']})")
        if not ok:
            continue
        idf = os.path.join(tmp, f"{tag}.ids")
        ids.astype(np.int32).tofile(idf)
        pos = [int(p) for p in g["positions"]]
        out, log = os.path.join(tmp, f"{tag}.rows"), []
        vp = [p for p in pos if p >= 64]
        o = run_tool([tool, model, "-I", idf, "-n", str(pos[-1] + 1), "-P", ",".join(map(str, pos)), "-o", out, "-V"],
                     th, log)
        for line in o.splitlines():
            f = line.split()
            if f and f[0] == "verify" and int(f[1]) in vp:
                res.report(f[2] == "SAME", f"{pre}/verify_p{f[1]}",
                           f"4-row verify segment ({f[3]}), decode step ({f[4]}) == the last row of a 64-row prefill chunk")
        rp, rows = read_rows(out)
        cs, t1 = [], []
        for j, p in enumerate(pos):
            cos, top, margin, md = metrics(g, j, rows[j])
            cs.append(cos)
            t1.append(top or (margin is not None and margin < TH["top1_margin"]))
            res.report(cos >= TH["cos_pos"] and t1[-1], f"{pre}/ref_p{p}",
                       f"cos {cos:.6f}, top-1 {'same' if top else f'differs (reference margin {margin:.3f})'}, "
                       f"max |diff| {md:.4f} over {len(g['subset_ids']) + 64} ids")
        res.report(float(np.mean(cs)) >= TH["cos_pooled"], f"{pre}/cos_pooled",
                   f"mean {np.mean(cs):.6f}, min {np.min(cs):.6f} at {pos[int(np.argmin(cs))]} ({len(pos)} positions)")
        res.report(np.mean(t1) >= TH["top1"], f"{pre}/top1_pooled", f"{sum(t1)}/{len(t1)}")
        if os.path.exists(gtool):
            gpos = [p for p in pos if p <= GENERIC_MAX]
            gout = os.path.join(tmp, f"{tag}.grows")
            run_tool([gtool, model, "-I", idf, "-n", str(gpos[-1] + 1), "-P", ",".join(map(str, gpos)), "-o", gout],
                     th, log)
            _, grows = read_rows(gout)
            for j, p in enumerate(gpos):
                same = np.array_equal(grows[j].view(np.uint32), rows[j].view(np.uint32))
                res.report(same, f"{pre}/generic_p{p}", "generic build == native, bitwise" if same else
                           f"DIFFER (max |diff| {np.abs(grows[j] - rows[j]).max():.3g})")
        retrieval(tool, model, tmp, res, pre, min(RETR_N, int(g["n"]) - 4000), th, log)
        print(f"   ({pre}: {', '.join(log)})")


def retrieval(tool, model, tmp, res, pre, n, th, log):
    """the retrieval check on any qwen35 file (no golden): the greedy reply to the question after n corpus tokens"""
    r = retrieval_ids(tool, model, tmp, n)
    rf = os.path.join(tmp, "retrieval.ids")
    r.astype(np.int32).tofile(rf)
    o = run_tool([tool, model, "-I", rf, "-g", "24"], th, log)
    text = o.split("\ntext ", 1)[1] if "\ntext " in o else ""
    res.report("PLUM-7316-HERON" in text, f"{pre}/retrieval",
               f"{len(r)}-token prompt, the fact at token ~{FACT_AT}: greedy reply {text.strip()[:80]!r}")


def retrieval_ids(tool, model, tmp, n=RETR_N):
    """the chat prompt (Qwen3.5's template, thinking off): the corpus's first FACT_AT tokens, the fact, the corpus up to
    n tokens (RETR_N; a golden of fewer positions: its n - 4,000), the question"""
    ids = corpus_ids(tool, model, tmp)
    head = tokenize(tool, model, "<|im_start|>user\nHere is a long text.\n\n", tmp)
    fact = tokenize(tool, model, "\n\n" + FACT + "\n\n", tmp)
    tail = tokenize(tool, model, "\n\n" + QUESTION + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", tmp)
    return np.concatenate([head, ids[:FACT_AT], fact, ids[FACT_AT:n], tail]).astype(np.int32)


class _Res:
    def __init__(self):
        self.passed, self.failed = [], []

    def report(self, ok, name, detail=""):
        (self.passed if ok else self.failed).append(name)
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tags", default="", help="comma-separated model tags (default: every one with a golden)")
    ap.add_argument("--retrieval", default="", metavar="FILE", help="only the retrieval check, on this qwen35 file "
                    "(e.g. Qwen3.8-27B: no golden needed)")
    ap.add_argument("--retrieval-n", type=int, default=RETR_N, help=f"corpus tokens in its prompt (default {RETR_N})")
    a = ap.parse_args()
    res = _Res()
    t = time.time()
    with tempfile.TemporaryDirectory(prefix="gwenedeg-long-", dir=os.environ.get("TMPDIR", "/var/tmp")) as tmp:
        if a.retrieval:
            log = []
            retrieval(os.path.join(ROOT, "build", "long_ctx"), a.retrieval, tmp, res, "long/retrieval",
                      a.retrieval_n, int(os.environ.get("OMP_NUM_THREADS", "8")), log)
            print(f"   ({', '.join(log)})")
        else:
            test_long(a, res, tmp, set(a.tags.split(",")) if a.tags else None)
    print(f"\nSummary: {len(res.passed)} passed, {len(res.failed)} failed, {time.time() - t:.0f} s")
    return 1 if res.failed else 0


if __name__ == "__main__":
    sys.exit(main())
