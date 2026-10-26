"""A numpy reference forward pass of a Qwen3 GGUF file (any of the family's shapes): fp32 activations on the
file's weights dequantized with ggml's expressions (tests/ggufmini.py), i.e. what runq computes minus its
int8 activations (and its attention / KV cache arithmetic). The same model as HF's Qwen3ForCausalLM (QK-norm,
GQA, NeoX RoPE with the file's theta, SwiGLU, RMSNorm eps 1e-6, tied or untied classifier), written from the
architecture, not from runq's code.

Memory: one layer's weights in fp32 at a time (8B: ~0.75 GB), the prompts' activations, the classifier in row
chunks; so 4B / 8B references fit where their fp32 models (16 / 33 GB) would not.

forward(f, prompts, rows=None) -> list of float32 (n_i, len(rows) or vocab) logits per prompt (all positions).
"""
import numpy as np

EPS = 1e-6


def _rmsnorm(x, w):
    x64 = x.astype(np.float32)
    ms = np.mean(x64 * x64, axis=-1, keepdims=True)
    return (x64 / np.sqrt(ms + EPS)) * w


def _rope(x, pos, theta):
    # x (n, heads, hd), NeoX pairs (j, j + hd/2), as HF's rotate_half
    hd = x.shape[-1]
    inv = 1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float32) / hd))
    ang = pos[:, None].astype(np.float32) * inv[None, :]  # (n, hd/2)
    cos = np.cos(np.concatenate([ang, ang], -1))[:, None, :].astype(np.float32)
    sin = np.sin(np.concatenate([ang, ang], -1))[:, None, :].astype(np.float32)
    rot = np.concatenate([-x[..., hd // 2:], x[..., :hd // 2]], -1)
    return x * cos + rot * sin


def config(f):
    kv = f.kv
    c = {k: int(kv[f"qwen3.{k}"]) for k in ("block_count", "embedding_length", "feed_forward_length")}
    c["heads"] = int(kv["qwen3.attention.head_count"])
    c["kv_heads"] = int(kv["qwen3.attention.head_count_kv"])
    c["hd"] = int(kv.get("qwen3.attention.key_length", c["embedding_length"] // c["heads"]))
    c["theta"] = float(kv["qwen3.rope.freq_base"])
    c["vocab"] = f.tensors["token_embd.weight"][0][1]
    c["tied"] = "output.weight" not in f.tensors
    return c


def forward(f, prompts, rows=None, chunk=16384, verbose=False):
    """prompts: lists of token ids; rows: classifier rows to compute (None: all)"""
    c = config(f)
    H, KVH, hd = c["heads"], c["kv_heads"], c["hd"]
    xs = [f.take_rows("token_embd.weight", np.asarray(p)) for p in prompts]
    pos = [np.arange(len(p)) for p in prompts]
    for l in range(c["block_count"]):
        w = {r: f.rows(f"blk.{l}.{r}.weight") for r in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up",
                                                         "ffn_down")}
        nrm = {r: f.rows(f"blk.{l}.{r}.weight")[0] for r in ("attn_norm", "ffn_norm", "attn_q_norm", "attn_k_norm")}
        for i, x in enumerate(xs):
            n = x.shape[0]
            h = _rmsnorm(x, nrm["attn_norm"])
            q = (h @ w["attn_q"].T).reshape(n, H, hd)
            k = (h @ w["attn_k"].T).reshape(n, KVH, hd)
            v = (h @ w["attn_v"].T).reshape(n, KVH, hd)
            q = _rope(_rmsnorm(q, nrm["attn_q_norm"]), pos[i], c["theta"])
            k = _rope(_rmsnorm(k, nrm["attn_k_norm"]), pos[i], c["theta"])
            rep = H // KVH
            k = np.repeat(k, rep, axis=1)  # head h uses kv head h // rep
            v = np.repeat(v, rep, axis=1)
            s = np.einsum("qhd,khd->hqk", q, k) / np.float32(np.sqrt(hd))
            s = s + np.triu(np.full((n, n), -np.inf, np.float32), 1)[None]
            s = np.exp(s - s.max(-1, keepdims=True))
            s = s / s.sum(-1, keepdims=True)
            o = np.einsum("hqk,khd->qhd", s, v).reshape(n, H * hd)
            x = x + o @ w["attn_output"].T
            h = _rmsnorm(x, nrm["ffn_norm"])
            g, u = h @ w["ffn_gate"].T, h @ w["ffn_up"].T
            with np.errstate(over="ignore"):  # (exp(-g) = inf for g < -88: silu = 0, as it should)
                a = (g / (1 + np.exp(-g))) * u
            x = x + a @ w["ffn_down"].T
            xs[i] = x.astype(np.float32)
        if verbose:
            print(f"  layer {l}", flush=True)
    out_norm = f.rows("output_norm.weight")[0]
    hs = [_rmsnorm(x, out_norm).astype(np.float32) for x in xs]
    name = "token_embd.weight" if c["tied"] else "output.weight"
    if rows is not None:
        W = f.take_rows(name, np.asarray(rows))
        return [h @ W.T for h in hs]
    V = c["vocab"]
    outs = [np.empty((h.shape[0], V), np.float32) for h in hs]
    for r0 in range(0, V, chunk):
        W = f.rows(name, r0, min(chunk, V - r0))
        for h, o in zip(hs, outs):
            o[:, r0:r0 + W.shape[0]] = h @ W.T
    return outs
