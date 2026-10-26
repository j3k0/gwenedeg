#!/usr/bin/env python3
"""Reference logits of a Qwen3 GGUF file from its dequantized weights (tests/golden/models/<tag>/), for the models
group of tests/check_models.py: the 4B / 8B models have no HF weights here (and their fp32 models, 16 / 33 GB,
would not fit in memory), so the reference is the file itself: fp32 activations on its weights dequantized with
ggml's expressions (tests/ref_forward.py, layer by layer: ~1 GB peak for 8B). runq on the file differs from it
by its int8 activations only (as the 0.6B Q4_K_M file vs golden/logits_q4).

Usage: .venv/bin/python tests/make_model_ref.py TAG FILE.gguf     (numpy + the HF tokenizer of models/Qwen3-0.6B)
Writes en / code / chat (tests/make_golden.py's prompts: every position) and long2 (check.py's 297-token prompt:
positions 248, 256, .., 296, past the first attention block): the golden format of make_golden.py with a
1,024-id vocab subset (files of ~20-60 KB) and `positions` for long2. ~1-4 min per model (8 threads).
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ggufmini as G  # noqa: E402
import ref_forward as R  # noqa: E402

PROMPTS = ["en", "code", "chat"]
SUBSET = 1024
TOPK = 64
LONG_POS = np.arange(248, 297, 8, dtype=np.int32)


def prompts():
    # the texts of tests/make_golden.py (LOGITS_PROMPTS, long_prompt), without importing torch
    src = open(os.path.join(HERE, "make_golden.py"), encoding="utf-8").read()
    start = src.index("LOGITS_PROMPTS = [")
    end = src.index("]\n", start) + 1
    ns = {}
    exec(src[start:end], ns)  # noqa: S102 (our own file: a list of (name, text) literals)
    texts = dict(ns["LOGITS_PROMPTS"])
    t = [x for _, x in ns["LOGITS_PROMPTS"]]
    all_text = " ".join(t)
    texts["long2"] = all_text + " " + all_text + " " + t[0] + " " + all_text + " " + t[1]
    return texts


def save(path, text, ids, logits, subset_ids, positions=None):
    top_ids = np.argsort(-logits, axis=1, kind="stable")[:, :TOPK].astype(np.int32)
    top_logits = np.take_along_axis(logits, top_ids, axis=1).astype(np.float32)
    extra = {"positions": positions} if positions is not None else {}
    np.savez_compressed(path, text=np.array(text), ids=np.array(ids, dtype=np.int32), subset_ids=subset_ids,
                        subset=logits[:, subset_ids].astype(np.float16), top_ids=top_ids, top_logits=top_logits, **extra)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    tag, path = sys.argv[1], sys.argv[2]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "models", "Qwen3-0.6B"), local_files_only=True)
    texts = prompts()
    names = PROMPTS + ["long2"]
    ids = [tok.encode(texts[n], add_special_tokens=False) for n in names]
    f = G.GgufFile(path)
    t0 = time.time()
    with np.errstate(over="ignore"):
        outs = R.forward(f, ids, verbose=True)
    vocab = outs[0].shape[1]
    subset_ids = np.sort(np.random.default_rng(0).choice(vocab, SUBSET, replace=False)).astype(np.int32)
    out = os.path.join(ROOT, "tests", "golden", "models", tag)
    os.makedirs(out, exist_ok=True)
    for n, i, lg in zip(names, ids, outs):
        p = os.path.join(out, n + ".npz")
        if n == "long2":
            save(p, texts[n], i, lg[LONG_POS], subset_ids, LONG_POS)
        else:
            save(p, texts[n], i, lg, subset_ids)
        print(f"wrote {p} ({len(i)} tokens, {os.path.getsize(p)} bytes)")
    print(f"{tag}: {time.time() - t0:.0f} s, theta {R.config(f)['theta']:g}, tied {R.config(f)['tied']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
