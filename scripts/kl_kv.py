#!/usr/bin/env python3
"""KL divergence of runq's next-token distributions from the fp32 model, per KV cache type: the measurement behind the
16-bit KV types (Step 7b: f16 +0.2%, bf16 +2.6%) and the bf16 dot-product attention (TODO G4 (b): -k bf16-dot /
f16-dot). Teacher-forced on pieces of real text, every position scored:
  KL(ref || runq) = sum_v p_ref(v) (log p_ref(v) - log p_runq(v)), float64, per position, averaged per piece and over
  all positions; also the top-1 agreement with the reference and the mean NLL of the text's next tokens.
The reference: the model's HF checkpoint in fp32, run by the repository's numpy forward passes (tests/ref_forward.py's
Qwen3 architecture, tests/ref_qwen35.py's Qwen3.5: one layer's weights at a time, so a 4B fits in ~4 GB where
transformers' fp32 model needs ~17 GB); --check-hf compares it with transformers on the first piece (the 0.6B / 0.8B).
The text: bench/ppl_beagle_ch1.txt (prose), then tests/check_long.py's corpus (this repository's files at a fixed commit:
docs, C, Python), cut at line ends into pieces of at most --tokens tokens, each its own sequence from position 0; a piece
tokenized by HF's tokenizer; runq gets the ids (build/long_ctx -A: every row's logits, streamed per 512-row chunk).

Usage: .venv/bin/python scripts/kl_kv.py --hf DIR --gguf FILE [--kv f32,f16,bf16,bf16-dot,f16-dot] [--pieces 13]
                                          [--tokens 3500] [-t 8] [--check-hf] [--out FILE.json]
  (make long-ctx; Qwen3.5 checkpoints: .venv-hf5, for the tokenizer and --check-hf's transformers 5.x; the numpy
  reference needs numpy only.) Files: the reference's logits of one piece and one runq run's at a time under $TMPDIR
  (float32: 3,500 x 248k = 3.5 GB each). Memory: the reference ~3.5 GB for the 4B, runq the model + a 512-row chunk.
Prints per piece and KV type the mean KL and the top-1 agreement, then the totals and each type's KL relative to the
first one's (and how many pieces are higher / lower: the noise of such a comparison).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import check_long as L  # noqa: E402
import ref_qwen35 as R35  # noqa: E402

EPS = 1e-6


# ------------------------------------------------------------------------------------------ the reference
class HfQwen3:
    """a Qwen3 HF checkpoint's tensors under the GGUF names (fp32, bf16 widened exactly; read with pread as
    ref_qwen35.HfWeights)"""
    _MAP = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
            "attn_output": "self_attn.o_proj", "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
            "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm", "ffn_gate": "mlp.gate_proj",
            "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}

    def __init__(self, d):
        cfg = json.load(open(os.path.join(d, "config.json")))
        self.c = {"layers": cfg["num_hidden_layers"], "dim": cfg["hidden_size"], "heads": cfg["num_attention_heads"],
                  "kv_heads": cfg["num_key_value_heads"], "hd": cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]),
                  "theta": float(cfg.get("rope_theta", 1e6)), "vocab": cfg["vocab_size"],
                  "tied": cfg.get("tie_word_embeddings", False)}
        self.h = R35.HfWeights.__new__(R35.HfWeights)  # (its safetensors reader only)
        self.h.heads = {}
        ix = os.path.join(d, "model.safetensors.index.json")
        idx = json.load(open(ix))["weight_map"] if os.path.exists(ix) else \
            {k: "model.safetensors" for k in self.h._header(os.path.join(d, "model.safetensors"))[1] if k != "__metadata__"}
        self.h.where = {k: os.path.join(d, v) for k, v in idx.items()}

    def get(self, name, r0=0, n=None):
        if name == "token_embd.weight":
            return self.h._get("model.embed_tokens.weight", r0, n)
        if name == "output.weight":
            return self.h._get("lm_head.weight", r0, n) if not self.c["tied"] else self.get("token_embd.weight", r0, n)
        if name == "output_norm.weight":
            return self.h._get("model.norm.weight")
        _, l, r = name.split(".", 2)
        return self.h._get(f"model.layers.{l}.{self._MAP[r.removesuffix('.weight')]}.weight")

    def take(self, name, idx):
        return self.h._get("model.embed_tokens.weight", idx=np.asarray(idx))


def _rmsnorm(x, w):
    ms = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
    return (x / np.sqrt(ms + np.float32(EPS))) * w


def _rope(x, pos, theta):  # (n, heads, hd), NeoX pairs (j, j + hd/2), as tests/ref_forward.py
    hd = x.shape[-1]
    inv = 1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float32) / hd))
    ang = pos[:, None].astype(np.float32) * inv[None, :]
    cos = np.cos(np.concatenate([ang, ang], -1))[:, None, :].astype(np.float32)
    sin = np.sin(np.concatenate([ang, ang], -1))[:, None, :].astype(np.float32)
    rot = np.concatenate([-x[..., hd // 2:], x[..., :hd // 2]], -1)
    return (x * cos + rot * sin).astype(np.float32)


def hidden_qwen3(w, ids, qc=512):
    """tests/ref_forward.py's forward (Qwen3: QK-norm, GQA, NeoX RoPE, SwiGLU; written from the architecture) on one
    sequence, in torch fp32 (multi-threaded elementwise ops; the weights one layer at a time), the attention per block
    of qc queries (exact softmax): the final-normed hidden states (n, dim)"""
    import torch
    c = w.c
    H, KVH, hd, n = c["heads"], c["kv_heads"], c["hd"], len(ids)
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32))

    def rms(x, g):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS) * g

    inv = 1.0 / (c["theta"] ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    ang = torch.arange(n, dtype=torch.float32)[:, None] * inv[None, :]
    cos, sin = torch.cat([ang, ang], -1).cos()[:, None, :], torch.cat([ang, ang], -1).sin()[:, None, :]
    rope = lambda x: x * cos + torch.cat([-x[..., hd // 2:], x[..., :hd // 2]], -1) * sin
    x = T(w.take("token_embd.weight", ids))
    sc, rep = 1 / np.sqrt(hd), H // KVH
    with torch.no_grad():
        for l in range(c["layers"]):
            W = {r: T(w.get(f"blk.{l}.{r}.weight")) for r in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate",
                                                                "ffn_up", "ffn_down", "attn_norm", "ffn_norm", "attn_q_norm",
                                                                "attn_k_norm")}
            h = rms(x, W["attn_norm"])
            q = rope(rms((h @ W["attn_q"].T).reshape(n, H, hd), W["attn_q_norm"]))
            k = rope(rms((h @ W["attn_k"].T).reshape(n, KVH, hd), W["attn_k_norm"]))
            v = (h @ W["attn_v"].T).reshape(n, KVH, hd)
            o = torch.empty(n, H, hd)
            for r0 in range(0, n, qc):
                r1 = min(n, r0 + qc)
                mask = torch.arange(r0, r1)[:, None] < torch.arange(r1)[None, :]
                for g in range(KVH):
                    s = torch.einsum("qhd,kd->hqk", q[r0:r1, g * rep:(g + 1) * rep], k[:r1, g]) * sc
                    s.masked_fill_(mask, float("-inf"))
                    o[r0:r1, g * rep:(g + 1) * rep] = torch.einsum("hqk,kd->qhd", torch.softmax(s, -1), v[:r1, g])
            x = x + o.reshape(n, H * hd) @ W["attn_output"].T
            h = rms(x, W["ffn_norm"])
            x = x + (torch.nn.functional.silu(h @ W["ffn_gate"].T) * (h @ W["ffn_up"].T)) @ W["ffn_down"].T
        return rms(x, T(w.get("output_norm.weight"))).numpy()


def hidden_q35(w, ids, rc=2048):
    """tests/ref_qwen35.py's forward_long on one sequence, every row: the final-normed hidden states (n, dim)"""
    c, n = w.c, len(ids)
    x = np.concatenate([w.take("token_embd.weight", ids[r0:r0 + rc]).astype(np.float32) for r0 in range(0, n, rc)])
    allpos = np.arange(n)
    for l in range(c["layers"]):
        lw = R35._LayerCache(w)
        an, fn = lw.vec(f"blk.{l}.attn_norm.weight"), lw.vec(f"blk.{l}.post_attention_norm.weight")
        ffn = {r: lw.mat(f"blk.{l}.{r}.weight") for r in ("ffn_gate", "ffn_up", "ffn_down")}
        if R35.is_linear(c, l):
            st = [None, None]
            for r0 in range(0, n, rc):
                x[r0:r0 + rc] += R35._lin_chunk(lw, c, l, R35._rmsnorm(x[r0:r0 + rc], an), st)
        else:
            K = np.empty((n, c["kv_heads"], c["hd"]), np.float32)
            V = np.empty_like(K)
            for r0 in range(0, n, rc):
                K[r0:r0 + rc], V[r0:r0 + rc] = R35._attn_kv(lw, c, l, R35._rmsnorm(x[r0:r0 + rc], an), allpos[r0:r0 + rc])
            for r0 in range(0, n, rc):
                xr = x[r0:r0 + rc]
                xr += R35._attn_q_chunked(lw, c, l, R35._rmsnorm(xr, an), allpos[r0:r0 + rc], K, V)
            del K, V
        for r0 in range(0, n, rc):
            xr = x[r0:r0 + rc]
            hh = R35._rmsnorm(xr, fn)
            xr += (R35._silu(hh @ ffn["ffn_gate"].T) * (hh @ ffn["ffn_up"].T)) @ ffn["ffn_down"].T
        del lw, ffn
    return R35._rmsnorm(x, w.vec("output_norm.weight")).astype(np.float32)


class Ref:
    """the fp32 reference of an HF checkpoint: hidden states of a piece, then its logits in row chunks"""

    def __init__(self, d):
        cfg = json.load(open(os.path.join(d, "config.json")))
        self.q35 = "text_config" in cfg or "linear_num_value_heads" in cfg
        self.w = R35.HfWeights(d) if self.q35 else HfQwen3(d)
        self.vocab = self.w.c["vocab"]

    def classifier(self):
        if self.q35:
            return self.w._get(self.w._BIG["token_embd.weight" if self.w.c["tied"] else "output.weight"])
        return self.w.get("output.weight")

    def logits_file(self, ids, path, chunk=256):
        h = hidden_q35(self.w, ids) if self.q35 else hidden_qwen3(self.w, np.asarray(ids))
        W = self.classifier()
        with open(path, "wb") as f:
            np.array([len(ids), W.shape[0]], np.int32).tofile(f)
            np.asarray(ids, np.int32).tofile(f)
            for r0 in range(0, len(ids), chunk):
                (h[r0:r0 + chunk] @ W.T).astype(np.float32).tofile(f)
        del W, h


# ------------------------------------------------------------------------------------------ the text
def pieces(tok, max_tokens, count):
    """texts of at most max_tokens tokens (HF's tokenizer), cut at line ends: Beagle ch. I, then the long corpus"""
    srcs = [open(os.path.join(ROOT, "bench", "ppl_beagle_ch1.txt"), encoding="utf-8").read(), L.corpus_text()]
    out = []
    for s in srcs:
        cur, n = [], 0
        for line in s.splitlines(keepends=True):
            k = len(tok(line))
            if cur and n + k > max_tokens:
                out.append("".join(cur))
                if len(out) == count:
                    return out
                cur, n = [], 0
            cur.append(line)
            n += k
        if cur and n > max_tokens // 2:
            out.append("".join(cur))
            if len(out) == count:
                return out
    return out


def read_logits(path):
    n, v = (int(x) for x in np.fromfile(path, dtype=np.int32, count=2))
    ids = np.fromfile(path, dtype=np.int32, count=2 + n)[2:]
    return ids, np.memmap(path, dtype=np.float32, mode="r", offset=4 * (2 + n), shape=(n, v))


def log_softmax(x):
    x = x.astype(np.float64)
    x -= x.max(-1, keepdims=True)
    x -= np.log(np.exp(x).sum(-1, keepdims=True))
    return x


def compare(ref, got, ids, chunk=256):
    """per position: KL(ref || got), top-1 equal, NLL of the next token under got (nan at the last position); float64,
    on torch's threads (numpy's elementwise exp / log run on one)"""
    import torch
    n = ref.shape[0]
    kl, t1, nll = np.empty(n), np.empty(n, bool), np.full(n, np.nan)
    for r0 in range(0, n, chunk):
        r1 = min(n, r0 + chunk)
        lr = torch.log_softmax(torch.from_numpy(np.array(ref[r0:r1])).double(), -1)
        lg = torch.log_softmax(torch.from_numpy(np.array(got[r0:r1])).double(), -1)
        kl[r0:r1] = (lr.exp() * (lr - lg)).sum(-1).numpy()
        t1[r0:r1] = (lr.argmax(-1) == lg.argmax(-1)).numpy()
        nxt = np.asarray(ids[r0 + 1:r1 + 1], np.int64)
        nll[r0:r0 + len(nxt)] = -lg[torch.arange(len(nxt)), torch.from_numpy(nxt)].numpy()
        del lr, lg
    return kl, t1, nll


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hf", required=True, help="the HF checkpoint directory (the fp32 reference, and the tokenizer)")
    ap.add_argument("--gguf", required=True, help="the GGUF file runq runs")
    ap.add_argument("--kv", default="f32,f16,bf16,bf16-dot,f16-dot", help="KV cache types (-k), the first the baseline")
    ap.add_argument("--pieces", type=int, default=13)
    ap.add_argument("--tokens", type=int, default=3500)
    ap.add_argument("-t", type=int, default=8, help="threads (runq; numpy's BLAS: OPENBLAS_NUM_THREADS)")
    ap.add_argument("--long-ctx", default=os.path.join(ROOT, "build", "long_ctx"))
    ap.add_argument("--check-hf", action="store_true", help="transformers' fp32 logits vs the numpy reference, piece 1")
    ap.add_argument("--out", default="", help="the per-piece results as JSON")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(a.hf, local_files_only=True)
    tok = lambda s: tk(s, add_special_tokens=False)["input_ids"]
    kvs = a.kv.split(",")
    import torch
    torch.set_num_threads(a.t)
    texts = pieces(tok, a.tokens, a.pieces)
    ref = Ref(a.hf)
    env = dict(os.environ, OMP_NUM_THREADS=str(a.t))
    res = {"model": a.gguf, "hf": a.hf, "kv": kvs, "pieces": []}
    print(f"# {os.path.basename(a.gguf)} vs {a.hf} (fp32, numpy): {len(texts)} pieces of <= {a.tokens} tokens, KV {', '.join(kvs)}",
          flush=True)
    with tempfile.TemporaryDirectory(prefix="kl_kv-", dir=os.environ.get("TMPDIR", "/var/tmp")) as tmp:
        rf, gf, idf = os.path.join(tmp, "ref.bin"), os.path.join(tmp, "runq.bin"), os.path.join(tmp, "piece.ids")
        for i, text in enumerate(texts):
            ids = tok(text)
            t0 = time.time()
            np.asarray(ids, np.int32).tofile(idf)
            ref.logits_file(ids, rf)
            if a.check_hf and i == 0:
                check_hf(a.hf, ids, rf)
            row = {"tokens": len(ids), "kv": {}}
            for kv in kvs:
                p = subprocess.run([a.long_ctx, a.gguf, "-I", idf, "-A", "-o", gf, "-k", kv], env=env,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                if p.returncode:
                    sys.exit(f"long_ctx failed: {p.stderr.decode(errors='replace')[-300:]}")
                gids, got = read_logits(gf)
                assert list(gids) == list(ids)
                _, rl = read_logits(rf)
                kl, t1, nll = compare(rl, got, ids)
                row["kv"][kv] = {"kl": float(kl.mean()), "kl_max": float(kl.max()), "top1": float(t1.mean()),
                                 "nll": float(np.nanmean(nll))}
                del got, rl
                os.remove(gf)
            os.remove(rf)
            res["pieces"].append(row)
            print(f"  piece {i:2d} ({len(ids)} tokens, {time.time() - t0:.0f} s): " +
                  "  ".join(f"{kv} KL {r['kl']:.6f} top1 {r['top1']:.4f}" for kv, r in row["kv"].items()), flush=True)
    P = res["pieces"]
    if not P:
        return 1
    tot = sum(p["tokens"] for p in P)
    base = kvs[0]
    print(f"\n# {tot} tokens in {len(P)} pieces; mean over positions (pieces weighted by length):")
    kb = sum(p["kv"][base]["kl"] * p["tokens"] for p in P) / tot
    for kv in kvs:
        k = sum(p["kv"][kv]["kl"] * p["tokens"] for p in P) / tot
        t = sum(p["kv"][kv]["top1"] * p["tokens"] for p in P) / tot
        nll = sum(p["kv"][kv]["nll"] * (p["tokens"] - 1) for p in P) / (tot - len(P))
        hi = sum(p["kv"][kv]["kl"] > p["kv"][base]["kl"] for p in P)
        lo = sum(p["kv"][kv]["kl"] < p["kv"][base]["kl"] for p in P)
        print(f"  {kv:9s} KL {k:.6f} ({(k / kb - 1) * 100:+.1f}% vs {base}: {hi} pieces higher, {lo} lower)  top-1 {t:.4f}"
              f"  NLL {nll:.5f}")
        res[kv] = {"kl": k, "top1": t, "nll": nll, "higher": hi, "lower": lo}
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0


def check_hf(d, ids, rf):
    """transformers' fp32 forward (eager attention) vs the numpy reference on one piece"""
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.float32, attn_implementation="eager", local_files_only=True)
    m.eval()
    with torch.no_grad():
        h = m.model(torch.tensor([ids])).last_hidden_state[0] if hasattr(m, "model") else None
        W = m.get_output_embeddings().weight
        _, rl = read_logits(rf)
        kl, md = [], 0.0
        for r0 in range(0, len(ids), 128):
            x = (h[r0:r0 + 128] @ W.T).numpy()
            lr, lx = log_softmax(rl[r0:r0 + 128]), log_softmax(x)
            kl.append((np.exp(lx) * (lx - lr)).sum(-1))
            md = max(md, float(np.abs(x - rl[r0:r0 + 128]).max()))
    print(f"  check-hf: transformers vs the reference, {len(ids)} positions: KL mean {np.concatenate(kl).mean():.3g}, "
          f"max |logit diff| {md:.3g}", flush=True)
    del m


if __name__ == "__main__":
    sys.exit(main())
