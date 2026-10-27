"""Test groups for the other Qwen3 models (Step 11), used by tests/check.py:

synth   (default suite) synthetic GGUF models cut out of Qwen3-0.6B-Q4_K_M.gguf by tests/make_synth.py (cached in
        build/synth): the shapes the 0.6B tests do not cover (dim 768 not a power of two, GQA 4, q dim != dim, an
        untied Q4_K embedding with a Q6_K classifier, RoPE theta 5e6, a 262,144 context (default cap), the
        Instruct-2507 template); logits vs the numpy reference (tests/ref_forward.py: the file's dequantized
        weights, fp32), batched vs sequential, threads, AVX-512 vs generic (bitwise), the library API (batched
        decode rows == alone, digests), the chat template variants (`-m render` on tokenizer-only files with the
        three HF templates vs tests/golden/chat_render.json; the chat's KV cache ids)
q35     (default suite, if models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf is there: --q35) Qwen3.5-0.8B, the hybrid
        architecture (docs/qwen35.md): the Qwen3.5 tokenizer golden (golden/tokenizer_qwen35.json), CLI chats whose
        history diverges (-r 1: the cache goes back to a checkpoint, the logits must equal a single-shot run's), and
        the models group's checks on the file (Q35_TAG); and two synthetic models cut out of it (tests/make_synth_q35.py,
        cached in build/synth-q35: value heads = 2 / 3 x key heads, the shapes of Qwen3.5-4B / 9B / 27B's linear
        layers, ~87 MB each with the 248k-token embedding): logits vs tests/ref_qwen35.py, bitwise invariants, api_test;
        the 0.8B file and the r = 3 model in one process (api_test --two: interleaved == each alone)
models  (opt-in: tests/run_tests.sh --models, make test-models) every downloaded family model
        (scripts/download_models.py family; missing files are skipped): logits vs HF fp32 (Qwen3-1.7B) and vs
        the dequantized-weights reference (tests/golden/models, tests/make_model_ref.py), bitwise invariants
        (batched vs sequential, threads, generic, api_test: batched decode rows), greedy goldens, the chat
        template variant and its renderings, runq serve (greedy == the CLI, a constrained reply)
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GOLDEN = os.path.join(HERE, "golden")
sys.path.insert(0, HERE)
import ggufmini as G  # noqa: E402
import ref_forward as R  # noqa: E402

SYNTH_THREADS = 4         # the synth runs (in the background from the start, next to the rest of the suite)
SYNTH_GEN_THREADS = 3     # its generic-build runs
SYNTH_REF = ["en", "code", "long2"]   # logits vs the numpy reference (-b 64)
# vs the reference: runq's int8 activations on a model with odd statistics (cut-out weights; long2: 297 tokens)
# measured 1 - cos mean 0.7e-3 (en) .. 4.3e-3 (long2), position min 0.962, top-1 0.83..1; a RoPE theta of 1e6
# instead of 5e6 gives 0.98 / 0.61 (mean) and top-1 0.06 on long2, an embedding read as Q8_0 NaN
TH_SYNTH = dict(cos_mean=0.99, cos_pos=0.90, top1=0.6)
SYNTH_BATCHED = [("en", 7), ("en", 64), ("long2", 64)]   # vs -b 1, bitwise (long2: past one attention block)
SYNTH_KERNELS = [("en", 1), ("en", 7), ("en", 64)]                       # AVX-512 vs generic, bitwise


def run(cmd, threads=None, env=None, **kw):
    e = dict(os.environ, **(env or {}))
    if threads is not None:
        e["OMP_NUM_THREADS"] = str(threads)
    return subprocess.run(cmd, env=e, capture_output=True, **kw)


def read_logits(path):
    raw = np.fromfile(path, dtype=np.int32, count=2)
    n = int(raw[0])
    return np.fromfile(path, dtype=np.int32, count=2 + n)[2:], \
        np.fromfile(path, dtype=np.float32, offset=4 * (2 + n)).reshape(n, int(raw[1]))


def logits_run(runq, model, text, path, chunk, threads, extra=()):
    p = run([runq, model, "-c", "4096", "-m", "logits", "-b", str(chunk), "-i", text, "-o", path] + list(extra), threads)
    if p.returncode != 0:
        raise RuntimeError(f"runq -m logits failed: {p.stderr.decode(errors='replace')[-400:]}")
    return path


def same(pa, pb):
    with open(pa, "rb") as fa, open(pb, "rb") as fb:
        if fa.read() == fb.read():
            return True, "bit-identical"
    xa, xb = read_logits(pa)[1], read_logits(pb)[1]
    if xa.shape != xb.shape:
        return False, "DIFFER (shapes)"
    d = np.abs(xa.astype(np.float64) - xb)
    return False, f"DIFFER (max abs diff {d.max():.3g} at pos {int(d.max(axis=1).argmax())})"


def cos_rows(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


# ------------------------------------------------------------------------------------------ chat rendering

def render_cases(runq, model, variant, res, prefix, jobs=None):
    """runq -m render on model (tokenizer only, one process for all cases) for the golden cases of its variant"""
    with open(os.path.join(GOLDEN, "chat_render.json"), encoding="utf-8") as f:
        cases = [c for c in json.load(f)["cases"] if c["variant"] == variant]
    conv = [{"messages": c["messages"], "thinking": c["thinking"]} for c in cases]
    p = run([runq, model, "-m", "render", "-i", json.dumps(conv)], 1)
    lines = p.stdout.decode().splitlines() if p.returncode == 0 else []
    errs = p.stderr.decode(errors="replace").splitlines()
    if len(lines) != len(cases):
        res.report(False, f"{prefix}/run", f"exit status {p.returncode}, {len(lines)} renderings: {errs[-2:]}")
        return
    for c, line, err in zip(cases, lines, [e for e in errs if e.startswith("[render]")]):
        ids = [int(x) for x in line.split()]
        det = f"[{variant}] -r {c['thinking']}: {len(c['ids'])} tokens" + (", = HF's apply_chat_template" if c["hf"] else
              " (runq's documented form, not HF's)")
        ok = ids == c["ids"] and f"chat template {variant}," in err
        if not ok:
            det += f"  <- got {len(ids)} tokens, {err!r}"
        res.report(ok, f"{prefix}/r{c['thinking']}_{c['name']}", det)


# ------------------------------------------------------------------------------------------ synth

def synth_build(a):
    """build/synth (tests/make_synth.py) from the 0.6B Q4_K_M GGUF; None if that is missing"""
    if not a.q4:
        return None
    p = run([sys.executable, "-I", os.path.join(HERE, "make_synth.py"), "--src", a.q4])
    if p.returncode != 0:
        raise RuntimeError(f"make_synth.py failed: {p.stderr.decode(errors='replace')[-400:]}")
    return p.stdout.decode().strip()


def synth_start(a, tmp, ex, gx, texts):
    """start the synth group's runs (futures), after building the files"""
    d = synth_build(a)
    if d is None:
        return None
    m = os.path.join(d, "synth-q4.gguf")
    t = os.path.join(tmp, "synth")
    os.makedirs(t, exist_ok=True)
    nd = int(os.environ.get("OMP_NUM_THREADS", "8"))
    S = {"dir": d, "model": m, "tmp": t, "files": {}, "api": None, "api_gen": None}

    def lf(name, chunk, threads=SYNTH_THREADS, runq=None, gen=False):
        key = (name, chunk, threads, gen)
        if key not in S["files"]:
            path = os.path.join(t, f"{name}_b{chunk}_t{threads}{'_gen' if gen else ''}.bin")
            S["files"][key] = (gx if gen else ex).submit(logits_run, runq or a.runq, m, texts[name], path, chunk, threads)
        return S["files"][key]
    S["lf"] = lf
    if a.api_test:
        S["api"] = ex.submit(run, [a.api_test, m], SYNTH_THREADS)
        # (the generic build: the CLI runs below, SYNTH_KERNELS; its api digests cost ~1.7 core-s more)
        S["api_t"] = ex.submit(run, [a.api_test, m, "--digest-small"], 8 if nd != 8 else 2)
    for name in SYNTH_REF:
        lf(name, 64)
    for name, c in SYNTH_BATCHED:
        lf(name, 1)
        lf(name, c)
    lf("long2", 64, 3)
    if a.generic:
        for name, c in SYNTH_KERNELS:
            lf(name, c)
            lf(name, c, SYNTH_GEN_THREADS, a.generic, True)

    def chat():
        prefix = os.path.join(t, "chat")
        p = run([a.runq, m, "-t", "0", "-c", "1024", "-n", "1", "-r", "1", "-y", "Be brief.", "-i", "Hi there"], 2,
                env={"RUNQ_CHAT_DUMP": prefix})
        info = run([a.runq, m, "-t", "0", "-n", "1"], 2, stdin=subprocess.DEVNULL)
        msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi there"}]
        r = run([a.runq, m, "-m", "render", "-i", json.dumps(msgs)], 1)
        ids = open(prefix + ".ids").read().split() if p.returncode == 0 else []
        return p, info, r, ids
    S["chat"] = ex.submit(chat)
    return S


def test_synth(a, res, S, texts):
    if S is None:
        print("      skipped: no Qwen3-0.6B-Q4_K_M.gguf (--q4), the synthetic models are cut out of it")
        return
    lf = S["lf"]
    # load: shapes, the default context cap, the template variant
    p, info, r, ids = S["chat"].result()
    out = info.stdout.decode()
    ok = info.returncode == 0 and "hidden_size=768," in out and "num_kv_heads=2," in out and "ctx_length=40960," in out \
        and "shared_classifier=0," in out
    res.report(ok, "synth/load", "dim 768, 8 / 2 heads, untied Q4_K embedding, theta 5e6: the default context is "
               "capped at 40,960 of 262,144" + ("" if ok else f"  <- {out.strip()[:200]!r} {info.stderr.decode()[-200:]!r}"))
    # chat: Instruct-2507 template (no think block; -r 1 ignored with a note), the KV cache == -m render
    want = r.stdout.decode().split()
    err = p.stderr.decode(errors="replace")
    ok = p.returncode == 0 and ids[:-1] == want and "no thinking mode: -r 1 ignored" in err and "151667" not in want
    res.report(ok, "synth/chat_instruct_template", f"{len(want)} prompt tokens (no think block), -r 1 ignored"
               + ("" if ok else f"  <- rc {p.returncode}, {len(ids)} cached vs {len(want)} rendered, {err[-200:]!r}"))
    # logits vs the numpy reference (fp32 on the dequantized weights; a 4,096-row classifier subset)
    f = G.GgufFile(S["model"])
    sub = np.sort(np.random.default_rng(0).choice(f.tensors["output.weight"][0][1], 4096, replace=False))
    files = {n: read_logits(lf(n, 64).result()) for n in SYNTH_REF}
    with np.errstate(all="ignore"):
        refs = R.forward(f, [files[n][0].tolist() for n in SYNTH_REF], rows=sub)
    t1 = []
    for n, ref in zip(SYNTH_REF, refs):
        ids_, lg = files[n]
        if not np.isfinite(lg).all():
            res.report(False, f"synth/logits_ref/{n}", "non-finite logits")
            continue
        got = lg[:, sub]
        c = cos_rows(got, ref)
        top1 = got.argmax(1) == ref.argmax(1)
        t1.extend(top1.tolist())
        why = []
        if not c.mean() >= TH_SYNTH["cos_mean"]: why.append(f"cos mean < {TH_SYNTH['cos_mean']}")
        if not c.min() >= TH_SYNTH["cos_pos"]: why.append(f"cos min < {TH_SYNTH['cos_pos']} at pos {int(c.argmin())}")
        res.report(not why, f"synth/logits_ref/{n}", f"n={len(ids_)}: cos mean {c.mean():.5f} min {c.min():.5f}, top1 "
                   f"{int(top1.sum())}/{len(top1)} vs the numpy reference" + ("  <- " + "; ".join(why) if why else ""))
    if t1:
        res.report(bool(np.mean(t1) >= TH_SYNTH["top1"]), "synth/logits_ref/top1_pooled",
                   f"{int(np.sum(t1))}/{len(t1)} (min {TH_SYNTH['top1']})")
    # bitwise invariants
    for n, c in SYNTH_BATCHED:
        ok, det = same(lf(n, 1).result(), lf(n, c).result())
        res.report(ok, f"synth/batched/{n}_b{c}", f"-b {c} vs -b 1: {det}")
    ok, det = same(lf("long2", 64).result(), lf("long2", 64, 3).result())
    res.report(ok, "synth/threads/long2_b64", f"{SYNTH_THREADS} vs 3 threads: {det}")
    if a.generic:
        for n, c in SYNTH_KERNELS:
            ok, det = same(lf(n, c).result(), lf(n, c, SYNTH_GEN_THREADS, a.generic, True).result())
            res.report(ok, f"synth/kernels/{n}_b{c}", f"AVX-512 vs generic: {det}")
    # the library API: batched decode rows (dim 768, GQA 4) == alone, mixed passes, digests across threads / builds
    if S["api"] is not None:
        p = S["api"].result()
        lines = [l for l in p.stdout.decode(errors="replace").splitlines() if l.startswith(("PASS ", "FAIL "))]
        bad = [l[5:].split("  ")[0] for l in lines if l.startswith("FAIL ")]
        res.report(p.returncode == 0 and lines and not bad, "synth/api", f"api_test: {len(lines) - len(bad)}/{len(lines)} passed"
                   + (f"; failed: {', '.join(bad)}" if bad else "" if p.returncode == 0 else f"; exit status {p.returncode}"))
        ref = dict(l.split()[1:3] for l in p.stdout.decode().splitlines() if l.startswith("DIGEST "))
        for label, fut in (("threads", S.get("api_t")), ("generic", S.get("api_gen"))):
            if fut is None:
                continue
            q = fut.result()
            d = dict(l.split()[1:3] for l in q.stdout.decode().splitlines() if l.startswith("DIGEST "))
            common = [k for k in d if k in ref]
            diff = [k for k in common if d[k] != ref[k]]
            res.report(q.returncode == 0 and len(common) >= 2 and not diff, f"synth/api_digest_{label}",
                       f"{len(common)} batched passes == {SYNTH_THREADS} threads" + (f"; differ: {', '.join(diff)}" if diff else ""))
    # chat template variants: tokenizer-only files with the three HF templates
    for v in ("qwen3", "instruct-2507", "thinking-2507"):
        render_cases(a.runq, os.path.join(S["dir"], f"synth-tok-{v}.gguf"), v, res, f"synth/render_{v}")


# ------------------------------------------------------------------------------------------ models (slow)

def dequant_check(runq, model, golden, res, prefix):
    """runq -m dequant of the golden's rows (blocks and repacked layout) == gguf-py's float32, bitwise"""
    g = np.load(os.path.join(GOLDEN, "dequant", golden))
    for i, name in enumerate(g["names"]):
        name, rows, r0 = str(name), g[f"rows{i}"], int(g["row0"][i])
        d = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"dequant-{os.getpid()}-{i}.bin")
        p = run([runq, model, "-c", "64", "-m", "dequant", "-i", f"{name}:{r0}:{rows.shape[0]}", "-o", d], 2)
        what = f"{str(g[f'type{i}'])} rows {r0}..{r0 + rows.shape[0] - 1} x {rows.shape[1]}"
        if p.returncode != 0:
            res.report(False, f"{prefix}/{name}", f"runq -m dequant failed: {p.stderr.decode(errors='replace')[-300:]}")
            continue
        h = np.fromfile(d, dtype=np.int32, count=3)
        got = np.fromfile(d, dtype=np.float32, offset=12).reshape(1 + int(h[2]), int(h[0]), int(h[1]))
        os.remove(d)
        ok = got.shape[1:] == rows.shape and got.shape[0] == 2
        detail = []
        for j, src in enumerate(["blocks", "repacked"][:got.shape[0]]):
            same_ = ok and np.array_equal(got[j].view(np.uint32), rows.view(np.uint32))
            ok &= same_
            detail.append(f"{src} " + ("bit-identical" if same_ else "DIFFER"))
        res.report(ok, f"{prefix}/{name}", f"{what}: " + ", ".join(detail) + " (gguf-py)")


# tag, file, chat template variant, golden subdirectories (tests/golden/models/<sub>: hf = HF fp32, else the
# dequantized-weights reference of this file), thresholds per golden
FAMILY = [
    # Qwen3-0.6B Q5_K_M (Q5_K 168 tensors, Q6_K 29): the Q5_K kernels on a Qwen3 file (k = 1024 and 3072)
    ("0.6b-q5km", "Qwen3-0.6B-GGUF/Qwen3-0.6B-Q5_K_M.gguf", "qwen3", [("0.6b-q5km", "ref")]),
    ("1.7b-q8", "Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf", "qwen3", [("1.7b-hf", "q8_hf"), ("1.7b-q8", "ref")]),
    ("1.7b-q4km", "Qwen3-1.7B-GGUF/Qwen3-1.7B-Q4_K_M.gguf", "qwen3", [("1.7b-hf", "q4_hf"), ("1.7b-q4km", "ref")]),
    ("4b-q8", "Qwen3-4B-GGUF/Qwen3-4B-Q8_0.gguf", "qwen3", [("4b-q8", "ref")]),
    ("4b-q4km", "Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf", "qwen3", [("4b-q4km", "ref")]),
    ("4b-2507i-q4km", "Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q4_K_M.gguf", "instruct-2507", [("4b-2507i-q4km", "ref")]),
    ("4b-2507t-q4km", "Qwen3-4B-Thinking-2507-GGUF/Qwen3-4B-Thinking-2507-Q4_K_M.gguf", "thinking-2507", [("4b-2507t-q4km", "ref")]),
    ("8b-q4km", "Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf", "qwen3", [("8b-q4km", "ref")]),
    # Qwen3.5 (hybrid: Gated DeltaNet + gated attention; goldens made with transformers 5.21, docs/qwen35.md)
    ("3.5-0.8b-q8", "Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf", "qwen3.5", [("3.5-0.8b-hf", "q8_hf"), ("3.5-0.8b-q8", "ref")]),
    # Unsloth's Q4_K_M: attn_qkv / ssm_out of the linear layers Q5_K (Q4_K, Q6_K, Q8_0 for the rest)
    ("3.5-0.8b-q4km", "Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf", "qwen3.5", [("3.5-0.8b-hf", "q35q4_hf"), ("3.5-0.8b-q4km", "ref")]),
    # Qwen3.5-4B Q8_0: 32 value heads on 16 key heads (llama.cpp's tiled order: value head h uses key head h % 16). Its
    # HF golden is tests/ref_qwen35.py on the HF bf16 weights (make_model_ref.py --hf: transformers' fp32 forward needs
    # ~17 GB), which matches transformers on 0.8B and on tiny models with nv = 2 / 3 nk (docs/qwen35.md)
    ("3.5-4b-q8", "Qwen3.5-4B-GGUF/Qwen3.5-4B-Q8_0.gguf", "qwen3.5", [("3.5-4b-hf", "q8_hf"), ("3.5-4b-q8", "ref")]),
]
# dequantized rows vs gguf-py (tests/golden/dequant, tests/make_dequant_golden.py --q5), bitwise, per FAMILY tag:
# from the file's blocks and from the repacked layouts (`runq -m dequant`)
DEQUANT = {"3.5-0.8b-q4km": "q5_k.npz"}
# more kernel checks for the files with a new kernel type (Q5_K) or code path (4B: value heads != key heads): AVX-512 vs
# generic logits at -b 64 (the GEMM tiles of every Q5_K matrix: 15 rows = 6 + 6 + 3; the batched DeltaNet items), and
# the batched passes' api_test digests (--digest-small) of both builds
EXTRA_KERNELS = {"3.5-0.8b-q4km": [("en", 64)], "0.6b-q5km": [("en", 64)], "3.5-4b-q8": [("en", 64)]}
Q35_TAG = "3.5-0.8b-q8"
Q35_Q4_TAG = "3.5-0.8b-q4km"    # in the q35 group too, if its file is next to the Q8_0 one (~20 s)
# chats on Qwen3.5 (check.py's chat_case): -r 1, the history drops turn 1's reasoning (and its "<think>\n"): the cache
# goes back to the checkpoint at turn 1's assistant header; the generic build too; -r 0 with a system prompt (no
# rollback); context full: the oldest turns dropped (back to a checkpoint or 0)
Q35_CHAT_CASES = [
    ("q35_thinking", ["-r", "1", "-n", "300"], ["What is 2+2? Answer with the number only.", "And times 3?"],
     {"stripped": 1}),
    ("q35_multiturn", ["-n", "24", "-y", "Be brief."],
     ["My name is Ada and my favourite colour is green.", "What is my name and favourite colour?"],
     {"mentions": (1, ["ada", "green"]), "generic": True}),
    ("q35_context_full", ["-c", "160", "-n", "30", "-y", "You are a terse assistant."],
     ["My name is Ada.", "Name three fruits.", "Name three animals.", "Name three colours.", "What is my name?"],
     {"dropped": True}),
]
# thresholds (per prompt: cos mean / every position >= 1 / position 0, top-1 fraction; pooled cos), measured in
# Step 11 (docs/DEVELOPMENT.md, History): vs the dequantized-weights reference only runq's int8 activations differ (as Q8_0 vs HF
# for 0.6B); Q8_0 vs HF fp32 as 0.6B's Q8_0 thresholds; Q4_K_M vs HF fp32 is the 4-bit weights' own error,
# which the reference has too: the 1.7B Q4_K_M reference vs HF fp32 (golden vs golden) has cos means 0.890 (en:
# position 8, cos -0.137, the whole logit vector shifted, top-1 still right) / 0.997 / 0.992 / 0.977 (code, chat,
# long2), position 0 0.948-0.956, top-1 56 / 61, pooled 0.967; so no per-position bound there, means below the
# reference's own. (The Q8_0 reference vs HF: means 0.9985-0.9998, min 0.9955, pos0 0.994, pooled 0.99936)
TH = {
    # (cos_pos 0.97: 4B Q4_K_M's long2 position 248 is at 0.978 (0.983 with -k f32; every other position of the
    # 7 models >= 0.9966), a flat distribution where the int8 activations show; the batched / thread / kernel
    # invariants hold bitwise there)
    "ref": dict(cos_mean=0.995, cos_pos=0.97, cos_pos0=0.98, top1=0.75, cos_all=0.997),
    "q8_hf": dict(cos_mean=0.995, cos_pos=0.98, cos_pos0=0.95, top1=0.75, cos_all=0.997),
    "q4_hf": dict(cos_mean=0.85, cos_pos=-1.0, cos_pos0=0.90, top1=0.75, cos_all=0.95),
    # Qwen3.5-0.8B Unsloth Q4_K_M vs HF fp32 (Step 13): the 4-bit weights' own error is much smaller than the Qwen3
    # Q4_K_M files' (measured: cos means 0.9924 / 0.9959 / 0.9958 / 0.9925 (chat, code, en, long2), min 0.9677,
    # pos0 0.9855-0.9957, top-1 14/15, 25/25, 11/15, 7/7, pooled 0.9946); vs its own dequantized-weights reference
    # (the "ref" goldens) runq is at cos 0.9999, top-1 62/62. Bounds ~2x the error
    "q35q4_hf": dict(cos_mean=0.985, cos_pos=0.93, cos_pos0=0.95, top1=0.6, cos_all=0.985),
}
MODEL_THREADS = 8
MODEL_PROMPTS = ["en", "code", "chat"]
GREEDY = ["-m", "chat", "-t", "0", "-c", "2048", "-n", "24", "-i", "What is the capital of Australia? Answer in one sentence."]


def model_metrics(g, logits):
    """cos over U = subset + golden top-64, top-1 (as check.py's logit_metrics); rows = the golden's positions"""
    pos = g["positions"] if "positions" in g.files else np.arange(len(g["ids"]))
    cos, top1 = [], []
    for j, i in enumerate(pos):
        ids = np.concatenate([g["top_ids"][j], g["subset_ids"]])
        ref = np.concatenate([g["top_logits"][j], g["subset"][j].astype(np.float32)])
        ids, first = np.unique(ids, return_index=True)
        ref = ref[first].astype(np.float64)
        got = logits[i, ids].astype(np.float64)
        cos.append(float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref))))
        top1.append(int(ids[np.argmax(got)]) == int(g["top_ids"][j, 0]))
    return pos, np.array(cos), np.array(top1)


def server_smoke(runq, model, variant, greedy_text, port_file):
    """runq serve on the model: /v1/models, a greedy chat completion (== the CLI's text), a constrained reply"""
    proc = subprocess.Popen([runq, "serve", model, "--port", "0", "-c", "2048", "--parallel", "2", "--slots", "2",
                             "--threads", "4"], stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
    out = []
    try:
        port, log = None, ""
        t0 = time.time()
        while time.time() - t0 < 120:
            line = proc.stderr.readline()
            if not line:
                break
            log += line
            if "listening on http://" in line:
                port = int(line.rsplit(":", 1)[1])
                break
        if port is None:
            return [(False, "start", log[-300:])]
        base = f"http://127.0.0.1:{port}/v1"

        def post(body):
            rq = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(rq, timeout=300) as r:
                return json.loads(r.read())
        mid = json.loads(urllib.request.urlopen(base + "/models", timeout=30).read())["data"][0]["id"]
        out.append((mid == os.path.basename(model)[:-5], "models", f"id {mid!r}, chat template {variant} in the log: "
                    f"{('chat template ' + variant) in log}"))
        r = post({"messages": [{"role": "user", "content": GREEDY[-1]}], "temperature": 0, "max_tokens": 24})
        msg = r["choices"][0]["message"]
        text = msg["content"] if variant != "thinking-2507" else None
        if variant == "thinking-2507":  # (the CLI prints <think>\n, the reasoning, </think> and the answer)
            ok = "reasoning_content" in msg and msg["reasoning_content"] != ""
            out.append((ok, "greedy_thinking", f"reasoning {len(msg.get('reasoning_content', ''))} chars, content "
                        f"{len(msg['content'])} chars (24 tokens)"))
        else:
            ok = text.strip() == greedy_text.strip() if greedy_text is not None else bool(text)
            out.append((ok, "greedy_eq_cli", f"{text!r}" + ("" if ok else f" vs the CLI {greedy_text!r}")))
        schema = {"type": "object", "properties": {"tier": {"enum": ["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"]}},
                  "required": ["tier"]}
        r = post({"messages": [{"role": "user", "content": "Classify: what is 2+2?"}], "temperature": 0,
                  "response_format": {"type": "json_schema", "json_schema": {"name": "t", "schema": schema}}})
        c = r["choices"][0]["message"]["content"]
        ok = json.loads(c)["tier"] in schema["properties"]["tier"]["enum"]
        out.append((ok, "constrained", repr(c)))
    except Exception as e:  # noqa: BLE001
        out.append((False, "error", f"{type(e).__name__}: {e}"))
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
    return out


def greedy_text(stdout):
    """the reply of a -t 0 -i chat run (stdout without the trailing newline)"""
    return stdout.decode(errors="replace").rstrip("\n")


def test_models(a, res, tmp, texts, update=False, tags=None, prefix="models", path=None):
    """the checks of every downloaded FAMILY file (tags: only those; path: the file of the one tag)"""
    gdir = os.path.join(GOLDEN, "models")
    fam = [x for x in FAMILY if tags is None or x[0] in tags]
    where = lambda f: path or os.path.join(ROOT, "models", f)
    present = [(tag, where(f), v, gs) for tag, f, v, gs in fam if os.path.exists(where(f))]
    for tag, f, v, gs in fam:
        if not os.path.exists(where(f)):
            print(f"      {tag}: skipped (models/{f} not downloaded: scripts/download_models.py)")
    nd = MODEL_THREADS
    for tag, model, variant, goldens in present:
        t0 = time.time()
        d = os.path.join(tmp, "models", tag)
        os.makedirs(d, exist_ok=True)

        def lf(name, chunk, threads=nd, runq=a.runq, sfx=""):
            path = os.path.join(d, f"{name}_b{chunk}_t{threads}{sfx}.bin")
            if not os.path.exists(path):
                logits_run(runq, model, texts[name], path, chunk, threads)
            return path
        # template variant and renderings (tokenizer only)
        render_cases(a.runq, model, variant, res, f"{prefix}/{tag}/render")
        # dequantization vs gguf-py
        if tag in DEQUANT:
            dequant_check(a.runq, model, DEQUANT[tag], res, f"{prefix}/{tag}/dequant")
        # greedy golden
        gpath = os.path.join(gdir, "greedy", f"{tag}.txt")
        p = run([a.runq, model] + GREEDY, nd)
        txt = greedy_text(p.stdout)
        if update:
            os.makedirs(os.path.dirname(gpath), exist_ok=True)
            with open(gpath, "w", encoding="utf-8") as fh:
                fh.write(txt + "\n")
        want = open(gpath, encoding="utf-8").read().rstrip("\n") if os.path.exists(gpath) else None
        res.report(p.returncode == 0 and txt == want, f"{prefix}/{tag}/greedy", repr(txt[-120:]) +
                   ("" if txt == want else f"  <- golden {want!r}" if want is not None else "  <- no golden (--update-models)"))
        # logits vs the goldens
        for sub, kind in goldens:
            th = TH[kind]
            gd = os.path.join(gdir, sub)
            if not os.path.isdir(gd):
                res.report(False, f"{prefix}/{tag}/logits_{kind}", f"no goldens in tests/golden/models/{sub}")
                continue
            allc = []
            for fn in sorted(os.listdir(gd)):
                if not fn.endswith(".npz"):
                    continue
                g = np.load(os.path.join(gd, fn))
                name = fn[:-4]
                if str(g["text"]) != texts.get(name):
                    res.report(False, f"{prefix}/{tag}/logits_{kind}/{name}", "the golden's text is not check.py's prompt")
                    continue
                ids, lg = read_logits(lf(name, 64))
                if ids.tolist() != g["ids"].tolist() or not np.isfinite(lg).all():
                    res.report(False, f"{prefix}/{tag}/logits_{kind}/{name}", "token ids differ or non-finite logits")
                    continue
                pos, c, t1 = model_metrics(g, lg)
                rest = c[pos >= 1]
                allc.extend(rest.tolist())
                why = []
                if not rest.mean() >= th["cos_mean"]: why.append(f"cos mean < {th['cos_mean']}")
                if not rest.min() >= th["cos_pos"]: why.append(f"cos min < {th['cos_pos']}")
                if 0 in pos and not c[0] >= th["cos_pos0"]: why.append(f"cos pos0 < {th['cos_pos0']}")
                if not t1.mean() >= th["top1"]: why.append(f"top1 < {th['top1']}")
                res.report(not why, f"{prefix}/{tag}/logits_{kind}/{name}",
                           f"{len(pos)} positions: cos" + (f" pos0 {c[0]:.4f}," if 0 in pos else "") + f" mean {rest.mean():.5f} "
                           f"min {rest.min():.5f}, top1 {int(t1.sum())}/{len(t1)}" + ("  <- " + "; ".join(why) if why else ""))
            if allc:
                res.report(bool(np.mean(allc) >= th["cos_all"]), f"{prefix}/{tag}/logits_{kind}/cos_pooled",
                           f"{np.mean(allc):.5f} over {len(allc)} positions (min {th['cos_all']})")
        # bitwise invariants: batched vs sequential (the decode path), chunkings, threads, generic kernels
        for n, c in (("en", 7), ("en", 64), ("all", 64)):
            ok, det = same(lf(n, 1), lf(n, c))
            res.report(ok, f"{prefix}/{tag}/batched/{n}_b{c}", f"-b {c} vs -b 1: {det}")
        ok, det = same(lf("long2", 64), lf("long2", 512))
        res.report(ok, f"{prefix}/{tag}/batched/long2_b512", f"-b 512 vs -b 64 (297 tokens): {det}")
        ok, det = same(lf("all", 64), lf("all", 64, 3))
        res.report(ok, f"{prefix}/{tag}/threads/all_b64", f"{nd} vs 3 threads: {det}")
        if a.generic:
            for n, c in (("en", 1), ("en", 7)) + tuple(EXTRA_KERNELS.get(tag, ())):
                ok, det = same(lf(n, c), lf(n, c, 4, a.generic, "_gen"))
                res.report(ok, f"{prefix}/{tag}/kernels/{n}_b{c}", f"AVX-512 vs generic: {det}")
            gen_api = (a.api_test or "") + "-generic"
            if tag in EXTRA_KERNELS and a.api_test and os.path.exists(gen_api):
                dg = []
                for exe in (a.api_test, gen_api):
                    q = run([exe, model, "--digest-small"], 4)
                    dg.append(dict(l.split()[1:3] for l in q.stdout.decode(errors="replace").splitlines() if l.startswith("DIGEST ")))
                diff = [k for k in dg[0] if dg[1].get(k) != dg[0][k]]
                res.report(len(dg[0]) >= 1 and not diff and set(dg[0]) == set(dg[1]), f"{prefix}/{tag}/api_digest_generic",
                           f"{len(dg[0])} batched passes (batched decode rows, ...): AVX-512 == generic" + (f"; differ: {diff}" if diff else ""))
        if a.api_test:
            p = run([a.api_test, model], nd)
            lines = [l for l in p.stdout.decode(errors="replace").splitlines() if l.startswith(("PASS ", "FAIL "))]
            bad = [l[5:].split("  ")[0] for l in lines if l.startswith("FAIL ")]
            res.report(p.returncode == 0 and lines and not bad, f"{prefix}/{tag}/api",
                       f"api_test (batched decode rows == alone, mixed passes, ...): {len(lines) - len(bad)}/{len(lines)} passed"
                       + (f"; failed: {', '.join(bad)}" if bad else ""))
        for ok, name, det in server_smoke(a.runq, model, variant, want, None):
            res.report(ok, f"{prefix}/{tag}/server/{name}", det)
        for fn in os.listdir(d):
            os.remove(os.path.join(d, fn))
        print(f"      ({tag}: {time.time() - t0:.1f} s)")


# ------------------------------------------------------------------------------------------ q35

# the q35 group's synthetic models (tests/make_synth_q35.py, cut out of the 0.8B Q8_0 file): value heads = 2 / 3 x key
# heads, as Qwen3.5-4B / 9B and 27B. Logits vs tests/ref_qwen35.py on the file (value head h uses key head h % nk; a
# 4,096-row classifier subset), measured: cos mean 0.9996-0.9999, min 0.9971, top-1 0.87-1 (flat distributions);
# with the wrong head mapping (h // r) the cos means drop to 0.60-0.89. Bitwise: batched vs sequential, chunkings,
# threads, AVX-512 vs generic, api_test (batched decode rows == alone, checkpoints) and its digests across builds
Q35_SYNTH = {"synth-q35-r2.gguf": True, "synth-q35-r3.gguf": False}  # file: full api_test (else its digests only)
Q35_SYNTH_REF = ["en", "code", "long2"]
TH_Q35_SYNTH = dict(cos_mean=0.999, cos_pos=0.99, top1=0.8)
Q35_SYNTH_SAME = [(("en", 1), ("en", 7)), (("en", 1), ("en", 64)), (("all", 1), ("all", 64)), (("long2", 1), ("long2", 64)),
                  (("long2", 64), ("long2", 512)), (("long2", 64), ("long2", 64, 3)),
                  (("en", 1), ("en", 1, "gen")), (("en", 7), ("en", 7, "gen")), (("long2", 64), ("long2", 64, "gen"))]


def q35_synth_build(a):
    """build/synth-q35 (tests/make_synth_q35.py) from the 0.8B Q8_0 file: the directory"""
    p = run([sys.executable, "-I", os.path.join(HERE, "make_synth_q35.py"), "--src", a.q35])
    if p.returncode != 0:
        raise RuntimeError(f"make_synth_q35.py failed: {p.stderr.decode(errors='replace')[-400:]}")
    return p.stdout.decode().strip()


Q35_BAD = [("bad-q35-nv.gguf", "is not a multiple of ssm.group_count"),
           ("bad-q35-hadamard.gguf", "Hadamard-folded weights (prism.hadamard.* metadata) are not supported")]


def q35_bad_run(a, d):
    """the rejected files of make_synth_q35.py: [(ok, name, detail)]"""
    out = []
    for fn, msg in Q35_BAD:
        p = run([a.runq, os.path.join(d, fn), "-n", "1", "-i", "hi"], 1)
        err = p.stderr.decode(errors="replace").strip()
        out.append((p.returncode == 1 and msg in err, f"q35/synth/reject_{fn[8:-5]}", f"exit status {p.returncode}: {err[-160:]!r}"))
    return out


def q35_two_run(a, d):
    """api_test --two on the 0.8B Q8_0 file and the synthetic r = 3 model (two models in one process: interleaved
    prefills, decodes, batched passes and truncations to checkpoints == each alone; one freed): [(ok, name, detail)]"""
    p = run([a.api_test, "--two", a.q35, os.path.join(d, "synth-q35-r3.gguf")], SYNTH_THREADS)
    lines = [l for l in p.stdout.decode(errors="replace").splitlines() if l.startswith(("PASS ", "FAIL "))]
    out = [(l.startswith("PASS "), f"q35/two/{l[5:].partition('  ')[0].strip()}", l[5:].partition("  ")[2].strip()) for l in lines]
    if p.returncode not in (0, 1) or not lines:
        out.append((False, "q35/two/run", f"exit status {p.returncode}: {p.stderr.decode(errors='replace')[-300:]}"))
    return out


def q35_synth_run(a, tmp, model, texts, full_api):
    """the checks of one synthetic model, run in the background: [(ok, name, detail)]. The logits files (248k ids per
    position: long2 is 0.3 GB) are reduced to a sha256 and the reference's classifier rows, then removed"""
    import hashlib
    import ref_qwen35 as R35
    tag = os.path.basename(model)[:-5]
    pre = f"q35/{tag}"
    out = []
    f = G.GgufFile(model)
    sub = np.sort(np.random.default_rng(0).choice(f.tensors["token_embd.weight"][0][1], 4096, replace=False))
    runs = {}

    def lf(key):
        # key (prompt, -b[, 3 threads | "gen": the generic build, 3 threads]): (sha256, ids, logits[:, sub])
        if key not in runs:
            name, chunk = key[:2]
            gen = len(key) > 2 and key[2] == "gen"
            th = 3 if len(key) > 2 else SYNTH_THREADS
            path = os.path.join(tmp, f"{tag}_{name}_b{chunk}_{th}{'_gen' if gen else ''}.bin")
            logits_run(a.generic if gen else a.runq, model, texts[name], path, chunk, th)
            with open(path, "rb") as fh:
                h = hashlib.sha256(fh.read()).hexdigest()
            ids, lg = read_logits(path)
            runs[key] = (h, ids, lg[:, sub].copy(), bool(np.isfinite(lg).all()))
            del lg
            os.remove(path)
        return runs[key]
    # logits vs the numpy reference
    got = {n: lf((n, 64)) for n in Q35_SYNTH_REF}
    with np.errstate(all="ignore"):
        refs = R35.forward(R35.GgufWeights(f), [got[n][1].tolist() for n in Q35_SYNTH_REF], rows=sub)
    t1 = []
    for n, ref in zip(Q35_SYNTH_REF, refs):
        _, ids, lg, finite = got[n]
        if not finite:
            out.append((False, f"{pre}/logits_ref/{n}", "non-finite logits"))
            continue
        c = cos_rows(lg, ref)
        top1 = lg.argmax(1) == ref.argmax(1)
        t1.extend(top1.tolist())
        why = []
        if not c.mean() >= TH_Q35_SYNTH["cos_mean"]: why.append(f"cos mean < {TH_Q35_SYNTH['cos_mean']}")
        if not c.min() >= TH_Q35_SYNTH["cos_pos"]: why.append(f"cos min < {TH_Q35_SYNTH['cos_pos']} at pos {int(c.argmin())}")
        out.append((not why, f"{pre}/logits_ref/{n}", f"n={len(ids)}: cos mean {c.mean():.5f} min {c.min():.5f}, top1 "
                    f"{int(top1.sum())}/{len(top1)} vs the numpy reference" + ("  <- " + "; ".join(why) if why else "")))
    if t1:
        out.append((bool(np.mean(t1) >= TH_Q35_SYNTH["top1"]), f"{pre}/logits_ref/top1_pooled",
                    f"{int(np.sum(t1))}/{len(t1)} (min {TH_Q35_SYNTH['top1']})"))
    # bitwise invariants
    for x, y in Q35_SYNTH_SAME:
        if len(y) > 2 and y[2] == "gen" and not a.generic:
            continue
        kind = "kernels" if len(y) > 2 and y[2] == "gen" else "threads" if len(y) > 2 else "batched"
        what = "AVX-512 vs generic" if kind == "kernels" else f"{SYNTH_THREADS} vs 3 threads" if kind == "threads" else \
            f"-b {y[1]} vs -b {x[1]}"
        ok = lf(x)[0] == lf(y)[0]
        out.append((ok, f"{pre}/{kind}/{y[0]}_b{y[1]}", f"{what}: " + ("bit-identical" if ok else "DIFFER")))
    # the library API: batched decode rows == alone, checkpoints (full run), the batched passes' digests vs generic
    if a.api_test:
        if full_api:
            p = run([a.api_test, model], SYNTH_THREADS)
            lines = [l for l in p.stdout.decode(errors="replace").splitlines() if l.startswith(("PASS ", "FAIL "))]
            bad = [l[5:].split("  ")[0] for l in lines if l.startswith("FAIL ")]
            out.append((p.returncode == 0 and bool(lines) and not bad, f"{pre}/api", f"api_test (batched decode rows == alone, "
                        f"checkpoints, ...): {len(lines) - len(bad)}/{len(lines)} passed" + (f"; failed: {', '.join(bad)}" if bad else "")))
        gen_api = a.api_test + "-generic"
        if os.path.exists(gen_api):
            dg = []
            for exe, th in ((a.api_test, SYNTH_THREADS), (gen_api, 3)):
                q = run([exe, model, "--digest-small"], th)
                dg.append(dict(l.split()[1:3] for l in q.stdout.decode(errors="replace").splitlines() if l.startswith("DIGEST ")))
            diff = [k for k in dg[0] if dg[1].get(k) != dg[0][k]]
            out.append((len(dg[0]) >= 1 and not diff and set(dg[0]) == set(dg[1]), f"{pre}/api_digest_generic",
                        f"{len(dg[0])} batched passes (batched decode rows, ...): AVX-512 ({SYNTH_THREADS} threads) == generic (3)"
                        + (f"; differ: {diff}" if diff else "")))
    return out


class Recorded:
    """a Results stand-in that keeps the reports, for a group run in the background (replayed at its turn)"""
    def __init__(self):
        self.reports = []

    def report(self, ok, name, detail="", xfail=None):
        self.reports.append((ok, name, detail, xfail))


def q35_start(a, tmp, ex, mx, texts, tok_test, chat_case, load_vocab):
    """start the q35 group in the background: its chat cases (check.py's chat_case) on ex, the models group's
    checks of the file on mx (reports recorded); None if there is no Qwen3.5 file"""
    if not a.q35:
        return None
    a2 = argparse.Namespace(**{**vars(a), "model": a.q35, "vocab_model": a.q35})
    vocab = load_vocab(a.q35)
    d = os.path.join(GOLDEN, "greedy")
    rec, rec4 = Recorded(), Recorded()
    t = os.path.join(tmp, "q35")
    os.makedirs(os.path.join(t, "q4"), exist_ok=True)
    q4 = os.path.join(os.path.dirname(a.q35), "Qwen3.5-0.8B-Q4_K_M.gguf")
    q4 = q4 if os.path.exists(q4) else None
    try:
        sd, serr = q35_synth_build(a), None
    except RuntimeError as e:
        sd, serr = None, str(e)
    os.makedirs(os.path.join(t, "synth"), exist_ok=True)
    synth = [ex.submit(q35_synth_run, a, os.path.join(t, "synth"), os.path.join(sd, fn), texts, full)
             for fn, full in Q35_SYNTH.items()] + [ex.submit(q35_bad_run, a, sd)] + \
        ([ex.submit(q35_two_run, a, sd)] if a.api_test else []) if sd else []
    return {"a": a2, "tok_test": tok_test, "synth": synth, "synth_err": serr,
            "chats": [ex.submit(chat_case, a2, tmp, vocab, d, *c) for c in Q35_CHAT_CASES],
            "rec": rec, "models": [mx.submit(test_models, a, rec, t, texts, False, {Q35_TAG}, "q35", a.q35)] + (
                [mx.submit(test_models, a, rec4, os.path.join(t, "q4"), texts, False, {Q35_Q4_TAG}, "q35", q4)] if q4 else []),
            "rec4": rec4}


def test_q35(a, res, tmp, texts, S):
    if S is None:
        print("      skipped: no Qwen3.5-0.8B Q8_0 GGUF (--q35; scripts/download_models.py 3.5-0.8b-q8, 3.5-0.8b-q4km)")
        return
    S["tok_test"](S["a"], res, "q35/tokenizer", "tokenizer_qwen35.json")
    for f in S["chats"]:
        for ok, n, detail in f.result():
            res.report(ok, n.replace("chat/", "q35/chat/"), detail)
    if S["synth_err"]:
        res.report(False, "q35/synth/build", S["synth_err"])
    for f in S["synth"]:
        for ok, n, detail in f.result():
            res.report(ok, n, detail)
    for f in S["models"]:
        f.result()
    for ok, n, detail, xfail in S["rec"].reports + S["rec4"].reports:
        res.report(ok, n, detail, xfail)
