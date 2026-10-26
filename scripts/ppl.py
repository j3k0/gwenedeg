#!/usr/bin/env python3
"""Perplexity of a text under HF fp32, runq (Q8_0) and mynah-slm (GGUF), same token ids.

Usage: .venv/bin/python scripts/ppl.py TEXT [--hf] [--runq] [--runq-model M ...] [--mynah GGUF ...] [-w 512] [-t 8]
  (no engine flag = all: --hf --runq and mynah-slm on the Q8_0 and Q4_K_M GGUFs if built)
  --runq-model: checkpoints for runq (.bin or GGUF; repeatable; default the Q8_0 .bin), implies --runq

Two protocols, both teacher-forced, raw text (no chat template, no BOS: Qwen3 has none):
  full   the whole text as one sequence, every token after the first scored (N-1 predictions).
         This is exactly what `mynah-slm ppl -f TEXT` does (one token at a time, f32 KV).
  win<W> non-overlapping windows of W token ids (the tail shorter than W is dropped), each from
         position 0, W-1 predictions per window; pooled NLL over all windows.
For runq and mynah-slm each window is passed as text (the HF decode of its ids) and the engine's
own tokenization is checked to give exactly the same ids (else the window is reported and
skipped). runq: `-m logits` (batched prefill, -b 512) on the whole sequence; mynah-slm:
`ppl -f` per window file. PPL = exp(mean NLL); nats per token.
Takes ~6-10 min with everything on a ~4,000-token text (mostly mynah-slm, token by token).
Temp files go to a fresh directory under $TMPDIR (removed at the end).
"""
import argparse, os, re, shutil, subprocess, sys, tempfile, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_DIR = os.path.join(ROOT, "models", "Qwen3-0.6B")
Q8BIN = os.path.join(ROOT, "models", "Qwen3-0.6B-q8.bin")
GGUF_DIR = os.path.join(ROOT, "models", "Qwen3-0.6B-GGUF")
MYNAH = os.path.join(ROOT, "third_party", "mynah-slm", "mynah-slm")


def nll_from_logits(logits, ids):
    """sum of -log p(ids[i+1] | ..i) over i, float64, logits (n, vocab) float32 (memmap ok)."""
    total = 0.0
    for i in range(len(ids) - 1):
        row = np.asarray(logits[i], dtype=np.float64)
        m = row.max()
        total -= row[ids[i + 1]] - m - np.log(np.exp(row - m).sum())
    return total


def hf_engine(threads):
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(threads)
    model = AutoModelForCausalLM.from_pretrained(HF_DIR, torch_dtype=torch.float32,
                                                 attn_implementation="eager", local_files_only=True)
    model.eval()

    def score(ids):
        with torch.no_grad():
            logits = model(torch.tensor([ids])).logits[0]
        lp = torch.log_softmax(logits.double(), dim=-1)
        t = torch.tensor(ids[1:])
        return float(-lp[torch.arange(len(ids) - 1), t].sum())
    return score


def runq_engine(tmp, threads, binary, model):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))

    def score(ids, text):
        out = os.path.join(tmp, "logits.bin")
        ctx = max(64, len(ids) + 16)
        subprocess.run([binary, model, "-c", str(ctx), "-m", "logits", "-i", text, "-o", out],
                       check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        n, vocab = (int(v) for v in np.fromfile(out, dtype=np.int32, count=2))
        got = np.fromfile(out, dtype=np.int32, count=n, offset=8)
        if list(got) != list(ids):
            os.remove(out)
            return None
        logits = np.memmap(out, dtype=np.float32, mode="r", offset=8 + 4 * n, shape=(n, vocab))
        s = nll_from_logits(logits, ids)
        del logits
        os.remove(out)
        return s
    return score


def mynah_engine(tmp, threads, gguf):
    def tokenize(path):
        with open(path, "rb") as f:
            r = subprocess.run([MYNAH, "tokenize", "-m", gguf], stdin=f, check=True,
                               capture_output=True, text=True)
        return [int(x) for x in r.stdout.split()]

    def score(ids, text):
        path = os.path.join(tmp, "win.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        if tokenize(path) != list(ids):
            return None
        r = subprocess.run([MYNAH, "ppl", "-m", gguf, "-f", path, "-t", str(threads)],
                           check=True, capture_output=True, text=True)
        m = re.search(r"tokens (\d+)\s+nll ([0-9.]+)", r.stdout)
        scored, nll = int(m.group(1)), float(m.group(2))
        assert scored == len(ids) - 1, (scored, len(ids))
        return nll * scored
    return score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text")
    ap.add_argument("--hf", action="store_true")
    ap.add_argument("--runq", action="store_true")
    ap.add_argument("--runq-model", action="append", metavar="M", help="runq checkpoint (.bin or GGUF), repeatable")
    ap.add_argument("--mynah", nargs="*", metavar="GGUF")
    ap.add_argument("-w", "--window", type=int, default=512)
    ap.add_argument("-t", "--threads", type=int, default=8)
    ap.add_argument("--runq-bin", default=os.path.join(ROOT, "runq"))
    ap.add_argument("--no-full", action="store_true", help="only the windowed protocol")
    a = ap.parse_args()
    if a.runq_model:
        a.runq = True
    if not (a.hf or a.runq or a.mynah is not None):
        a.hf = a.runq = True
        a.mynah = [] if os.access(MYNAH, os.X_OK) else None
    if a.mynah == []:
        a.mynah = [os.path.join(GGUF_DIR, f) for f in ("Qwen3-0.6B-Q8_0.gguf", "Qwen3-0.6B-Q4_K_M.gguf")]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF_DIR, local_files_only=True)
    text = open(a.text, encoding="utf-8").read()
    ids = tok.encode(text, add_special_tokens=False)
    W = a.window
    wins = [ids[i:i + W] for i in range(0, len(ids) - W + 1, W)]
    print(f"text {a.text}: {len(text.encode())} bytes, {len(ids)} tokens (HF); "
          f"full: {len(ids) - 1} predictions; win{W}: {len(wins)} windows, {len(wins) * (W - 1)} predictions")

    tmp = tempfile.mkdtemp(prefix="gwenedeg-ppl-")
    engines = []
    if a.hf:
        engines.append(("HF fp32", "hf", hf_engine(a.threads)))
    for m in (a.runq_model or [Q8BIN]) if a.runq else []:
        engines.append((f"runq {os.path.basename(m)}", "runq", runq_engine(tmp, a.threads, a.runq_bin, m)))
    for g in a.mynah or []:
        engines.append((f"mynah-slm {os.path.basename(g)}", "mynah", mynah_engine(tmp, a.threads, g)))
    try:
        print(f"{'engine':34s} {'protocol':8s} {'pred':>6s} {'nll':>8s} {'ppl':>8s} {'time':>7s}")
        for name, kind, score in engines:
            protos = ([] if a.no_full else [("full", [ids])]) + [(f"win{W}", wins)]
            for pname, seqs in protos:
                t0 = time.time(); tot = 0.0; npred = 0; skipped = 0
                for s in seqs:
                    r = score(s) if kind == "hf" else score(s, tok.decode(s))
                    if r is None:
                        skipped += 1
                        continue
                    tot += r; npred += len(s) - 1
                note = f"  ({skipped} windows skipped: tokenization differs)" if skipped else ""
                mean = tot / max(npred, 1)
                print(f"{name:34s} {pname:8s} {npred:6d} {mean:8.5f} {np.exp(mean):8.3f} "
                      f"{time.time() - t0:6.1f}s{note}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
