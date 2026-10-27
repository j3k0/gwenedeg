"""A numpy reference forward pass of a Qwen3.5 (GGUF architecture `qwen35`) text model: the spec for runq's hybrid
layers, written from the architecture (HF's modeling_qwen3_5.py, transformers 5.x), not from runq's code.

Qwen3.5 (0.8B: 24 layers, dim 1024, FFN 3584, vocab 248,320, tied classifier) stacks blocks of the usual
pre-norm shape (x += mixer(rmsnorm(x)); x += ffn(rmsnorm(x)), SwiGLU), where the mixer is, per layer:

  linear attention (Gated DeltaNet; layers with (l + 1) % full_attention_interval != 0, 18 of 24):
    qkv = conv(W_qkv h)       causal depthwise conv over the last 4 positions (kernel ssm_conv1d, no bias), SiLU;
                              channels [q: nk heads x 128 | k: nk x 128 | v: nv x 128] (0.8B: nk = nv = 16; 4B: 16 / 32)
    q, k = l2norm(q), l2norm(k) per head (x * 1 / sqrt(sum x^2 + 1e-6)); q *= 1 / sqrt(128)
    beta = sigmoid(W_beta h), g = ssm_a * softplus(W_alpha h + ssm_dt)    per value head (ssm_a = -exp(A_log))
    value head i uses the q / k of key head i // r in HF's order (r = nv / nk: grouped, repeat_interleave), of key
    head i % nk in the GGUF's (llama.cpp's converter reorders the value heads to a tiled order: GGUF value head j =
    HF value head (j % nk) * r + j // nk, in the v rows of W_qkv, W_z, W_alpha, W_beta, ssm_a, ssm_dt, the v channels
    of the conv and the input columns of W_out; not in q / k nor ssm_norm). Each set of weights is used in its own
    order here (HfWeights.vorder), so the two references check both mappings.
    per value head, the 128 x 128 state S (k index, v index), per token:
        S *= exp(g); delta = beta * (v - S^T k); S += k delta^T; o = S^T q
    o = rmsnorm(o per head, ssm_norm) * silu(W_z h); out = W_out o

  full attention (6 of 24): GQA 8 / 2 heads of 256 with an output gate
    W_q h = per head [q (256) | gate (256)]; q, k = rmsnorm(q / k per head, q_norm / k_norm)
    RoPE on the first 64 dims of each head only (partial_rotary_factor 0.25), NeoX pairs (j, j + 32), theta 1e7.
    (The model's RoPE is MRoPE with sections [11, 11, 10]; for text the three position ids are equal, which
    makes it this plain partial RoPE)
    o = softmax(q k^T / 16) v; out = W_o (o * sigmoid(gate))

Norm weights: HF's Qwen3_5RMSNorm computes x * (1 + w); llama.cpp's converter stores 1 + w in the GGUF (attn_norm,
post_attention_norm, output_norm, attn_q_norm, attn_k_norm), so all norms here are x * w on the stored values.
The gated norm (ssm_norm) has no + 1 in either. The MTP head and the vision tower are not part of the text model (a GGUF
MTP block, `nextn_predict_layers`, is skipped).

Weights: GgufWeights(GgufFile) (ggml's dequantization, tests/ggufmini.py) or HfWeights(model dir) (the
safetensors, converted to the GGUF names and values as llama.cpp's convert_hf_to_gguf.py does, except the value
head order (kept grouped); against transformers 5.21 fp32 this file is within 2-6e-5 on Qwen3.5-0.8B (golden/models/
3.5-0.8b-hf vs make_model_ref.py --hf) and 1e-4 on tiny random models with 2 / 3 value heads per key head, also on
their GGUF conversions by llama.cpp (docs/qwen35.md, design 8)).
Memory: forward() reads one layer's weights at a time (dequantized / bf16 -> fp32 once per layer for all prompts),
the embedding rows it needs and the classifier in chunks; HfWeights reads the safetensors with pread (numpy only,
no torch): 1.3 GB peak for 4B (make_model_ref.py, incl. the logits).

forward(w, prompts, rows=None) -> list of float32 (n_i, len(rows) or vocab) logits per prompt (all positions).
forward_long(w, ids, positions) -> float32 (len(positions), vocab): one long prompt (tens of thousands of tokens),
memory-frugal (rows in chunks, attention in query blocks with an exact softmax), the logits at some rows only
(tests/make_long_ref.py, TODO G2).
The recurrence runs token by token (the form of a decode step), in fp32 as runq does; HF's prefill uses the
chunked form, equal up to rounding.
"""
import json
import os

import numpy as np

EPS = 1e-6


class GgufWeights:
    vorder = "tiled"  # GGUF value head j uses key head j % nk

    def __init__(self, f):
        self.f = f
        kv, a = f.kv, "qwen35"
        assert kv.get("general.architecture") == a, "not a qwen35 GGUF"
        self.c = {  # (an MTP block, nextn_predict_layers: the last of block_count, not part of the main model)
            "layers": int(kv[f"{a}.block_count"]) - int(kv.get(f"{a}.nextn_predict_layers", 0)),
            "dim": int(kv[f"{a}.embedding_length"]),
            "heads": int(kv[f"{a}.attention.head_count"]), "kv_heads": int(kv[f"{a}.attention.head_count_kv"]),
            "hd": int(kv[f"{a}.attention.key_length"]), "theta": float(kv[f"{a}.rope.freq_base"]),
            "rot": int(kv[f"{a}.rope.dimension_count"]), "interval": int(kv.get(f"{a}.full_attention_interval", 4)),
            "conv": int(kv[f"{a}.ssm.conv_kernel"]), "dk": int(kv[f"{a}.ssm.state_size"]),
            "nk": int(kv[f"{a}.ssm.group_count"]), "nv": int(kv[f"{a}.ssm.time_step_rank"]),
            "inner": int(kv[f"{a}.ssm.inner_size"]), "vocab": f.tensors["token_embd.weight"][0][1],
            "tied": "output.weight" not in f.tensors,
        }
        self.c["dv"] = self.c["inner"] // self.c["nv"]

    def mat(self, name):  # (rows, cols) float32
        return self.f.rows(name)

    def vec(self, name):
        return self.f.rows(name).reshape(-1)

    def take(self, name, idx):
        return self.f.take_rows(name, np.asarray(idx))

    def chunk(self, name, r0, n):
        return self.f.rows(name, r0, n)


class HfWeights:
    """the HF checkpoint under the GGUF tensor names, with the converter's value transforms (not its value head
    reorder: HF's grouped order, value head i uses key head i // r)"""
    vorder = "grouped"

    def __init__(self, model_dir):
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        tc = cfg.get("text_config", cfg)
        rp = tc.get("rope_parameters") or {}
        hd = tc["head_dim"]
        self.c = {
            "layers": tc["num_hidden_layers"], "dim": tc["hidden_size"], "heads": tc["num_attention_heads"],
            "kv_heads": tc["num_key_value_heads"], "hd": hd, "theta": float(rp.get("rope_theta", tc.get("rope_theta"))),
            "rot": int(hd * rp.get("partial_rotary_factor", 1.0)), "interval": tc.get("full_attention_interval", 4),
            "conv": tc["linear_conv_kernel_dim"], "dk": tc["linear_key_head_dim"], "nk": tc["linear_num_key_heads"],
            "nv": tc["linear_num_value_heads"], "dv": tc["linear_value_head_dim"], "vocab": tc["vocab_size"],
            "tied": tc.get("tie_word_embeddings", cfg.get("tie_word_embeddings", True)),
        }
        assert self.c["nv"] % self.c["nk"] == 0, "num_v_heads is not a multiple of num_k_heads"
        self.heads = {}  # file -> (data offset, header)
        ix = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(ix):
            idx = json.load(open(ix))["weight_map"]
        else:  # one file (e.g. a small Qwen3_5ForCausalLM)
            one = os.path.join(model_dir, "model.safetensors")
            idx = {k: "model.safetensors" for k in self._header(one)[1] if k != "__metadata__"}
        self.where = {k: os.path.join(model_dir, v) for k, v in idx.items()}
        # the text model's tensors: model.language_model.* (Qwen3_5ForConditionalGeneration) or model.* (ForCausalLM)
        self.P = "model.language_model." if "model.language_model.embed_tokens.weight" in self.where else "model."
        self._BIG = {"token_embd.weight": self.P + "embed_tokens.weight", "output.weight": "lm_head.weight"}

    def _header(self, p):
        if p not in self.heads:
            with open(p, "rb") as fh:
                hn = int.from_bytes(fh.read(8), "little")
                self.heads[p] = (8 + hn, json.loads(fh.read(hn)))
        return self.heads[p]

    _DT = {"BF16": (np.uint16, 2), "F16": (np.float16, 2), "F32": (np.float32, 4)}

    def _get(self, key, r0=0, n=None, idx=None):
        """tensor key as float32 (bf16 widened exactly), read with pread (no mmap: its pages would count in the RSS
        of a 9 GB checkpoint): all of it, rows r0 .. r0 + n - 1, or the rows idx of a 2-D tensor"""
        p = self.where[key]
        off, head = self._header(p)
        e = head[key]
        dt, sz = self._DT[e["dtype"]]
        shape = e["shape"]
        a, b = e["data_offsets"]
        row = int(np.prod(shape[1:])) * sz if len(shape) > 1 else sz
        with open(p, "rb") as fh:
            if idx is not None:
                x = np.empty((len(idx), row // sz), dt)
                for j, i in enumerate(idx):
                    x[j] = np.frombuffer(os.pread(fh.fileno(), row, off + a + int(i) * row), dt)
                shape = [len(idx)] + list(shape[1:])
            else:
                n = shape[0] - r0 if n is None else n
                if len(shape) > 1 or n != shape[0]:
                    x = np.frombuffer(os.pread(fh.fileno(), n * row, off + a + r0 * row), dt)
                    shape = [n] + list(shape[1:])
                else:
                    x = np.frombuffer(os.pread(fh.fileno(), b - a, off + a), dt)
        if e["dtype"] == "BF16":
            x = (x.astype(np.uint32) << 16).view(np.float32)
        return x.astype(np.float32).reshape(shape)

    _MAP = {"attn_qkv": "linear_attn.in_proj_qkv", "attn_gate": "linear_attn.in_proj_z",
            "ssm_alpha": "linear_attn.in_proj_a", "ssm_beta": "linear_attn.in_proj_b", "ssm_out": "linear_attn.out_proj",
            "ssm_conv1d": "linear_attn.conv1d", "ssm_norm": "linear_attn.norm", "attn_q": "self_attn.q_proj",
            "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
            "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm", "attn_norm": "input_layernorm",
            "post_attention_norm": "post_attention_layernorm", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
            "ffn_down": "mlp.down_proj"}
    _PLUS1 = {"attn_norm", "post_attention_norm", "attn_q_norm", "attn_k_norm"}

    def _hf(self, name):
        P = self.P
        if name in self._BIG:
            return self._get(self._BIG[name])
        if name == "output_norm.weight":
            return 1.0 + self._get(P + "norm.weight")
        _, l, r = name.split(".", 2)
        r = r.removesuffix(".weight").removesuffix(".bias")
        if r == "ssm_a":
            return -np.exp(self._get(f"{P}layers.{l}.linear_attn.A_log"))
        if r == "ssm_dt":
            return self._get(f"{P}layers.{l}.linear_attn.dt_bias")
        x = self._get(f"{P}layers.{l}.{self._MAP[r]}.weight")
        if r == "ssm_conv1d":
            x = x.reshape(x.shape[0], -1)  # (channels, 1, kernel) -> (channels, kernel)
        if r in self._PLUS1:
            x = 1.0 + x
        return x.astype(np.float32)

    def mat(self, name):
        return self._hf(name)

    def vec(self, name):
        return self._hf(name).reshape(-1)

    def take(self, name, idx):
        return self._get(self._BIG[name], idx=np.asarray(idx)) if name in self._BIG else self._hf(name)[np.asarray(idx)]

    def chunk(self, name, r0, n):
        return self._get(self._BIG[name], r0, n) if name in self._BIG else self._hf(name)[r0:r0 + n]


def _rmsnorm(x, w):
    ms = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
    return (x / np.sqrt(ms + np.float32(EPS))) * w


def _silu(x):
    with np.errstate(over="ignore"):  # (exp(-x) = inf for x < -88: silu = 0)
        return x / (1 + np.exp(-x))


def _sigmoid(x):
    with np.errstate(over="ignore"):
        return 1 / (1 + np.exp(-x))


def _softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))  # torch's threshold 20


def _l2norm(x):
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True, dtype=np.float32) + np.float32(EPS))


def _rope_partial(x, pos, theta, rot):
    # x (n, heads, hd): NeoX pairs (j, j + rot/2) within the first rot dims; the rest passes through
    inv = 1.0 / (theta ** (np.arange(0, rot, 2, dtype=np.float32) / rot))
    ang = pos[:, None].astype(np.float32) * inv[None, :].astype(np.float32)  # (n, rot/2)
    cos = np.cos(np.concatenate([ang, ang], -1))[:, None, :].astype(np.float32)
    sin = np.sin(np.concatenate([ang, ang], -1))[:, None, :].astype(np.float32)
    xr, xp = x[..., :rot], x[..., rot:]
    h = rot // 2
    rh = np.concatenate([-xr[..., h:], xr[..., :h]], -1)
    return np.concatenate([xr * cos + rh * sin, xp], -1)


def is_linear(c, l):
    return (l + 1) % c["interval"] != 0


def _linear_attn(w, c, l, h):
    n = h.shape[0]
    nk, nv, dk, dv, K = c["nk"], c["nv"], c["dk"], c["dv"], c["conv"]
    p = f"blk.{l}."
    qkv = h @ w.mat(p + "attn_qkv.weight").T                       # (n, 2 nk dk + nv dv)
    z = (h @ w.mat(p + "attn_gate.weight").T).reshape(n, nv, dv)
    beta = _sigmoid(h @ w.mat(p + "ssm_beta.weight").T)            # (n, nv)
    a = h @ w.mat(p + "ssm_alpha.weight").T
    g = w.vec(p + "ssm_a") * _softplus(a + w.vec(p + "ssm_dt.bias"))  # (n, nv), < 0
    cw = w.mat(p + "ssm_conv1d.weight")                             # (channels, K): tap K-1 is the current token
    xp = np.concatenate([np.zeros((K - 1, qkv.shape[1]), np.float32), qkv], 0)
    conv = sum(xp[j:j + n] * cw[:, j] for j in range(K))
    conv = _silu(conv.astype(np.float32))
    q = conv[:, :nk * dk].reshape(n, nk, dk)
    k = conv[:, nk * dk:2 * nk * dk].reshape(n, nk, dk)
    v = conv[:, 2 * nk * dk:].reshape(n, nv, dv)
    q = _l2norm(q) * np.float32(1 / np.sqrt(dk))
    k = _l2norm(k)
    rep = nv // nk
    if w.vorder == "grouped":  # HF: value head i uses key head i // rep
        q, k = np.repeat(q, rep, axis=1), np.repeat(k, rep, axis=1)
    else:                      # GGUF (tiled): value head i uses key head i % nk
        q, k = np.tile(q, (1, rep, 1)), np.tile(k, (1, rep, 1))
    S = np.zeros((nv, dk, dv), np.float32)
    o = np.empty((n, nv, dv), np.float32)
    eg = np.exp(g).astype(np.float32)
    for t in range(n):
        S *= eg[t][:, None, None]
        kv_mem = np.einsum("hkv,hk->hv", S, k[t])
        delta = (v[t] - kv_mem) * beta[t][:, None]
        S += k[t][:, :, None] * delta[:, None, :]
        o[t] = np.einsum("hkv,hk->hv", S, q[t])
    o = _rmsnorm(o, w.vec(p + "ssm_norm.weight")) * _silu(z)
    return o.reshape(n, nv * dv) @ w.mat(p + "ssm_out.weight").T


def _full_attn(w, c, l, h, pos):
    n = h.shape[0]
    H, KVH, hd = c["heads"], c["kv_heads"], c["hd"]
    p = f"blk.{l}."
    qg = (h @ w.mat(p + "attn_q.weight").T).reshape(n, H, 2 * hd)
    q, gate = qg[..., :hd], qg[..., hd:].reshape(n, H * hd)
    k = (h @ w.mat(p + "attn_k.weight").T).reshape(n, KVH, hd)
    v = (h @ w.mat(p + "attn_v.weight").T).reshape(n, KVH, hd)
    q = _rope_partial(_rmsnorm(q, w.vec(p + "attn_q_norm.weight")), pos, c["theta"], c["rot"])
    k = _rope_partial(_rmsnorm(k, w.vec(p + "attn_k_norm.weight")), pos, c["theta"], c["rot"])
    rep = H // KVH
    k, v = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
    s = np.einsum("qhd,khd->hqk", q, k) / np.float32(np.sqrt(hd))
    s = s + np.triu(np.full((n, n), -np.inf, np.float32), 1)[None]
    s = np.exp(s - s.max(-1, keepdims=True))
    s = s / s.sum(-1, keepdims=True)
    o = np.einsum("hqk,khd->qhd", s, v).reshape(n, H * hd)
    return (o * _sigmoid(gate)) @ w.mat(p + "attn_output.weight").T


class _LayerCache:
    """one layer's weights, read (dequantized / converted to fp32) once for all the prompts"""

    def __init__(self, w):
        self.w, self.c, self.vorder, self.m = w, w.c, w.vorder, {}

    def mat(self, name):
        if name not in self.m:
            self.m[name] = self.w.mat(name)
        return self.m[name]

    def vec(self, name):
        return self.mat(name).reshape(-1)


def _lin_chunk(w, c, l, h, st):
    """_linear_attn on rows h of a longer sequence, carrying st = [S, the conv's last K - 1 inputs] (the same
    operations, row for row)"""
    n = h.shape[0]
    nk, nv, dk, dv, K = c["nk"], c["nv"], c["dk"], c["dv"], c["conv"]
    p = f"blk.{l}."
    qkv = h @ w.mat(p + "attn_qkv.weight").T
    z = (h @ w.mat(p + "attn_gate.weight").T).reshape(n, nv, dv)
    beta = _sigmoid(h @ w.mat(p + "ssm_beta.weight").T)
    a = h @ w.mat(p + "ssm_alpha.weight").T
    g = w.vec(p + "ssm_a") * _softplus(a + w.vec(p + "ssm_dt.bias"))
    cw = w.mat(p + "ssm_conv1d.weight")
    if st[1] is None:
        st[0] = np.zeros((nv, dk, dv), np.float32)
        st[1] = np.zeros((K - 1, qkv.shape[1]), np.float32)
    xp = np.concatenate([st[1], qkv], 0)
    st[1] = xp[-(K - 1):].copy()
    conv = _silu(sum(xp[j:j + n] * cw[:, j] for j in range(K)).astype(np.float32))
    q = conv[:, :nk * dk].reshape(n, nk, dk)
    k = conv[:, nk * dk:2 * nk * dk].reshape(n, nk, dk)
    v = conv[:, 2 * nk * dk:].reshape(n, nv, dv)
    q = _l2norm(q) * np.float32(1 / np.sqrt(dk))
    k = _l2norm(k)
    rep = nv // nk
    if w.vorder == "grouped":
        q, k = np.repeat(q, rep, axis=1), np.repeat(k, rep, axis=1)
    else:
        q, k = np.tile(q, (1, rep, 1)), np.tile(k, (1, rep, 1))
    S = st[0]
    o = np.empty((n, nv, dv), np.float32)
    eg = np.exp(g).astype(np.float32)[:, :, None, None]
    q4, k4, k3 = q[:, :, None, :], k[:, :, None, :], k[:, :, :, None]
    bv = beta[:, :, None]
    for t in range(n):  # (the recurrence of _linear_attn: matmul forms of its einsums)
        S *= eg[t]
        delta = (v[t] - np.matmul(k4[t], S)[:, 0]) * bv[t]
        S += k3[t] * delta[:, None, :]
        o[t] = np.matmul(q4[t], S)[:, 0]
    o = _rmsnorm(o, w.vec(p + "ssm_norm.weight")) * _silu(z)
    return o.reshape(n, nv * dv) @ w.mat(p + "ssm_out.weight").T


def _attn_kv(w, c, l, h, pos):
    """the roped, normed keys and the values of rows h at positions pos: (n, kv_heads, hd) each"""
    n, KVH, hd, p = h.shape[0], c["kv_heads"], c["hd"], f"blk.{l}."
    k = (h @ w.mat(p + "attn_k.weight").T).reshape(n, KVH, hd)
    v = (h @ w.mat(p + "attn_v.weight").T).reshape(n, KVH, hd)
    k = _rope_partial(_rmsnorm(k, w.vec(p + "attn_k_norm.weight")), pos, c["theta"], c["rot"])
    return k.astype(np.float32), v.astype(np.float32)


def _attn_q_chunked(w, c, l, h, pos, K, V, qc=128):
    """_full_attn for rows h at positions pos (any, ascending) over the cache K, V (positions 0..): exact softmax per
    query, causal, in blocks of qc queries x the keys up to the block's last position (memory: qc x heads/kv x n)"""
    n, H, KVH, hd, p = h.shape[0], c["heads"], c["kv_heads"], c["hd"], f"blk.{l}."
    qg = (h @ w.mat(p + "attn_q.weight").T).reshape(n, H, 2 * hd)
    q, gate = qg[..., :hd], qg[..., hd:].reshape(n, H * hd)
    q = _rope_partial(_rmsnorm(q, w.vec(p + "attn_q_norm.weight")), pos, c["theta"], c["rot"]).astype(np.float32)
    rep, o = H // KVH, np.empty((n, H, hd), np.float32)
    sc = np.float32(1 / np.sqrt(hd))
    for r0 in range(0, n, qc):
        r1 = min(n, r0 + qc)
        e = int(pos[r1 - 1]) + 1
        c0 = int(pos[r0]) + 1                                    # keys from c0 on: after some query of the block
        mask = pos[r0:r1, None] < np.arange(c0, e)[None, :]
        for g in range(KVH):
            qq = q[r0:r1, g * rep:(g + 1) * rep].transpose(1, 0, 2).reshape(rep * (r1 - r0), hd)
            s = qq @ K[:e, g].T                                   # (rep rows, e)
            s *= sc
            s = s.reshape(rep, r1 - r0, e)
            s[:, :, c0:][:, mask] = -np.inf
            s -= s.max(-1, keepdims=True)
            np.exp(s, out=s)
            s /= s.sum(-1, keepdims=True, dtype=np.float32)
            o[r0:r1, g * rep:(g + 1) * rep] = (s.reshape(rep * (r1 - r0), e) @ V[:e, g]).reshape(rep, r1 - r0, hd) \
                .transpose(1, 0, 2)
            del s
    return (o.reshape(n, H * hd) * _sigmoid(gate)) @ w.mat(p + "attn_output.weight").T


def forward_long(w, ids, positions, rc=2048, verbose=False, stats=None):
    """forward() on one long prompt, memory-frugal: the logits (full vocabulary) at the rows `positions` only. Rows in
    chunks of rc (the linear layers carry their state across chunks; full attention: the layer's K / V for all rows
    first, then the queries in blocks, _attn_q_chunked); the last layer computes the rows `positions` only if it is a
    full-attention one. Peak ~ 4 x n x dim floats + K / V + one layer's weights (0.8B at 98k: ~2 GB). stats (dict):
    per attention layer the max |K| / |V| (the f16 cache's range)"""
    c, n = w.c, len(ids)
    positions = np.asarray(positions)
    x = np.concatenate([w.take("token_embd.weight", ids[r0:r0 + rc]).astype(np.float32) for r0 in range(0, n, rc)])
    allpos = np.arange(n)
    t0 = __import__("time").time()
    for l in range(c["layers"]):
        lw = _LayerCache(w)
        an, fn = lw.vec(f"blk.{l}.attn_norm.weight"), lw.vec(f"blk.{l}.post_attention_norm.weight")
        ffn = {r: lw.mat(f"blk.{l}.{r}.weight") for r in ("ffn_gate", "ffn_up", "ffn_down")}
        last = l == c["layers"] - 1
        if is_linear(c, l):
            st = [None, None]
            for r0 in range(0, n, rc):
                x[r0:r0 + rc] += _lin_chunk(lw, c, l, _rmsnorm(x[r0:r0 + rc], an), st)
            rows = allpos
        else:
            K = np.empty((n, c["kv_heads"], c["hd"]), np.float32)
            V = np.empty_like(K)
            for r0 in range(0, n, rc):
                K[r0:r0 + rc], V[r0:r0 + rc] = _attn_kv(lw, c, l, _rmsnorm(x[r0:r0 + rc], an), allpos[r0:r0 + rc])
            if stats is not None:
                stats[f"k{l}"], stats[f"v{l}"] = float(np.abs(K).max()), float(np.abs(V).max())
            rows = positions if last else allpos
            if last:
                x = x[rows]
            for r0 in range(0, len(rows), rc):
                xr = x[r0:r0 + rc]
                xr += _attn_q_chunked(lw, c, l, _rmsnorm(xr, an), rows[r0:r0 + rc], K, V)
            del K, V
        for r0 in range(0, len(rows), rc):
            xr = x[r0:r0 + rc]
            h = _rmsnorm(xr, fn)
            xr += (_silu(h @ ffn["ffn_gate"].T) * (h @ ffn["ffn_up"].T)) @ ffn["ffn_down"].T
        del lw, ffn
        if verbose:
            print(f"  layer {l} ({'linear' if is_linear(c, l) else 'full'}): {__import__('time').time() - t0:.0f} s",
                  flush=True)
    if len(x) != len(positions):
        x = x[positions]
    return forward_rows(w, [x])[0]


def forward_rows(w, xs, rows=None, chunk=16384):
    """the final norm and the classifier on hidden states xs (list of (n, dim))"""
    c = w.c
    hs = [_rmsnorm(x, w.vec("output_norm.weight")).astype(np.float32) for x in xs]
    name = "token_embd.weight" if c["tied"] else "output.weight"
    if rows is not None:
        W = w.take(name, rows)
        return [h @ W.T for h in hs]
    V = c["vocab"]
    outs = [np.empty((h.shape[0], V), np.float32) for h in hs]
    for r0 in range(0, V, chunk):
        W = w.chunk(name, r0, min(chunk, V - r0))
        for h, o in zip(hs, outs):
            o[:, r0:r0 + W.shape[0]] = h @ W.T
    return outs


def forward(w, prompts, rows=None, chunk=16384, verbose=False):
    """w: GgufWeights / HfWeights; prompts: lists of token ids; rows: classifier rows to compute (None: all)"""
    c = w.c
    xs = [w.take("token_embd.weight", p).astype(np.float32) for p in prompts]
    for l in range(c["layers"]):
        lw = _LayerCache(w)
        an, fn = lw.vec(f"blk.{l}.attn_norm.weight"), lw.vec(f"blk.{l}.post_attention_norm.weight")
        ffn = {r: lw.mat(f"blk.{l}.{r}.weight") for r in ("ffn_gate", "ffn_up", "ffn_down")}
        for i, x in enumerate(xs):
            h = _rmsnorm(x, an)
            if is_linear(c, l):
                x = x + _linear_attn(lw, c, l, h)
            else:
                x = x + _full_attn(lw, c, l, h, np.arange(x.shape[0]))
            h = _rmsnorm(x, fn)
            x = x + (_silu(h @ ffn["ffn_gate"].T) * (h @ ffn["ffn_up"].T)) @ ffn["ffn_down"].T
            xs[i] = x.astype(np.float32)
        del lw, ffn
        if verbose:
            print(f"  layer {l} ({'linear' if is_linear(c, l) else 'full'})", flush=True)
    return forward_rows(w, xs, rows, chunk)
