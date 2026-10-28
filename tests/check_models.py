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
        the models group's checks on the file (Q35_TAG); and synthetic models cut out of it (tests/make_synth_q35.py,
        cached in build/synth-q35: value heads = 2 / 3 x key heads, the shapes of Qwen3.5-4B / 9B / 27B's linear
        layers, ~87 MB each with the 248k-token embedding; from the Q4_K_M file next to it, Qwen3.8-27B's layout and
        types at a small scale, 330 MB): logits vs tests/ref_qwen35.py, bitwise invariants, api_test; the 0.8B file and
        the r = 3 model in one process (api_test --two: interleaved == each alone); files that must be refused (bad
        metadata); the 27B-shaped model with its MTP block (logits == without it, bitwise), with F32 alpha / beta and
        Qwen3.8's template (the checks above, a CLI chat, runq serve's prompts), Qwen3.8's renderings (tokenizer-only
        files with its two templates vs tests/golden/chat_render.json, incl. requests with tools); runq serve's tool
        calls on the 0.8B (tests/server_test.py --tools)
models  (opt-in: tests/run_tests.sh --models, make test-models) every downloaded family model
        (scripts/download_models.py family; missing files are skipped; Qwen3.8-27B (BIG) also when MemAvailable does
        not cover it): logits vs HF fp32 (Qwen3-1.7B) and vs
        the dequantized-weights reference (tests/golden/models, tests/make_model_ref.py), bitwise invariants
        (batched vs sequential, threads, generic, api_test: batched decode rows), greedy goldens, the chat
        template variant and its renderings, runq serve (greedy == the CLI, a constrained reply)
"""
import argparse
import concurrent.futures as cf
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
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
    conv = [{"request": c["request"]} if "request" in c else  # (tool calls: a chat completion body as runq serve renders it)
            dict({"messages": c["messages"], "thinking": c["thinking"]},
                 **({"reasoning_effort": c["reasoning_effort"]} if "reasoning_effort" in c else {})) for c in cases]
    p = run([runq, model, "-m", "render", "-i", json.dumps(conv)], 1)
    lines = p.stdout.decode().splitlines() if p.returncode == 0 else []
    errs = p.stderr.decode(errors="replace").splitlines()
    if len(lines) != len(cases):
        res.report(False, f"{prefix}/run", f"exit status {p.returncode}, {len(lines)} renderings: {errs[-2:]}")
        return
    for c, line, err in zip(cases, lines, [e for e in errs if e.startswith("[render]")]):
        ids = [int(x) for x in line.split()]
        det = f"[{variant}] -r {c['thinking']}" + (f" -e {c['reasoning_effort']}" if "reasoning_effort" in c else "") + \
            f": {len(c['ids'])} tokens" + (" (request with tools)" if "request" in c else "") + (", = HF's apply_chat_template" if c["hf"] else ", = HF's with reasoning_content"
                                          if c.get("hf_reasoning_content") else " (runq's documented form, not HF's)")
        ok = ids == c["ids"] and f"chat template {variant}," in err
        if not ok:
            det += f"  <- got {len(ids)} tokens, {err!r}"
        res.report(ok, f"{prefix}/r{c['thinking']}_{c['name']}", det)


def render_run(runq, model, variant, prefix):
    """render_cases in the background: [(ok, name, detail)]"""
    rec = Recorded()
    render_cases(runq, model, variant, rec, prefix)
    return [(ok, n, d) for ok, n, d, _ in rec.reports]


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
    # Unsloth's Q3_K_M (TODO G3): the Q3_K kernels on attn_q / attn_k / attn_gate / ffn_gate / ffn_up (78 tensors: the
    # roles of the 27B's Q3_K_M file but its token_embd), Q4_K / Q5_K for the rest, Q6_K token_embd, Q8_0 alpha / beta
    ("3.5-0.8b-q3km", "Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q3_K_M.gguf", "qwen3.5", [("3.5-0.8b-hf", "q35q3_hf"), ("3.5-0.8b-q3km", "ref")]),
    # Qwen3.5-4B Q8_0: 32 value heads on 16 key heads (llama.cpp's tiled order: value head h uses key head h % 16). Its
    # HF golden is tests/ref_qwen35.py on the HF bf16 weights (make_model_ref.py --hf: transformers' fp32 forward needs
    # ~17 GB), which matches transformers on 0.8B and on tiny models with nv = 2 / 3 nk (docs/qwen35.md)
    ("3.5-4b-q8", "Qwen3.5-4B-GGUF/Qwen3.5-4B-Q8_0.gguf", "qwen3.5", [("3.5-4b-hf", "q8_hf"), ("3.5-4b-q8", "ref")]),
    # Qwen3.8-27B Q4_K_M (docs/qwen38-speculative.md: 64 layers + an MTP block (validated, not loaded), r = 3, GQA 6,
    # untied, F32 ssm_alpha / ssm_beta, Qwen3.8's template: thinking on by default). ~16.3 GiB of RSS: run only when
    # MemAvailable allows it (BIG); its golden is tests/ref_qwen35.py on the file (make_model_ref.py: 214 s, ~2 GB)
    ("3.8-27b-q4km", "Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf", "qwen3.8", [("3.8-27b-q4km", "ref")]),
]
# files checked against MemAvailable first (runq's own estimate at the default context, api_test's, + BIG_MARGIN GiB
# for the logits files in TMPDIR and the server's slots): skipped with a note if it does not fit. Their runs go one
# at a time (the models group runs its files in turn); GWENEDEG_TEST_MEM (GiB) stands in for MemAvailable
BIG = {"3.8-27b-q4km"}
BIG_MARGIN = 1.5
# dequantized rows vs gguf-py (tests/golden/dequant, tests/make_dequant_golden.py --q5 / --q3), bitwise, per FAMILY tag:
# from the file's blocks and from the repacked layouts (`runq -m dequant`)
DEQUANT = {"3.5-0.8b-q4km": "q5_k.npz", "3.5-0.8b-q3km": "q3_k.npz"}
# more kernel checks for the files with a new kernel type (Q5_K, Q3_K) or code path (4B: value heads != key heads): AVX-512 vs
# generic logits at -b 64 (the GEMM tiles of every Q5_K matrix: 15 rows = 6 + 6 + 3; the batched DeltaNet items), and
# the batched passes' api_test digests (--digest-small) of both builds
EXTRA_KERNELS = {"3.5-0.8b-q4km": [("en", 64)], "0.6b-q5km": [("en", 64)], "3.5-4b-q8": [("en", 64)], "3.5-0.8b-q3km": [("en", 64)]}
Q35_TAG = "3.5-0.8b-q8"
Q35_Q4_TAG = "3.5-0.8b-q4km"    # in the q35 group too, if its file is next to the Q8_0 one (~20 s)
Q35_Q3_TAG = "3.5-0.8b-q3km"    # the same (Q3_K), lean: its weights' checks (render, greedy, api, server: --models)
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
    # Qwen3.5-0.8B Unsloth Q3_K_M vs HF fp32 (TODO G3): ~3.5x the Q4_K_M file's error (measured: cos means 0.9701 /
    # 0.9864 / 0.9815 / 0.9822 (chat, code, en, long2), min 0.8577 (chat), pos0 0.9379-0.9918, top-1 12/15, 24/25, 11/15,
    # 7/7, pooled 0.98089); vs its own dequantized-weights reference runq is at pooled cos 0.99984, top-1 62/62 (the
    # Q4_K_M file: 0.99986), i.e. all of it is the 3-bit weights' own error. Bounds ~2x the error
    "q35q3_hf": dict(cos_mean=0.94, cos_pos=0.72, cos_pos0=0.87, top1=0.6, cos_all=0.96),
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
        text = msg["content"] if variant not in ("thinking-2507", "qwen3.8") else None
        if variant in ("thinking-2507", "qwen3.8"):  # (the CLI prints <think>\n, the reasoning, </think> and the answer)
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


def mem_skip(runq, model):
    """None if the file fits in MemAvailable (runq's load estimate at the default context + BIG_MARGIN), else why not"""
    p = run([runq, model, "--mem", "0.01", "-n", "1", "-i", "x"], 1, stdin=subprocess.DEVNULL)
    err = p.stderr.decode(errors="replace")
    if "needs ~" not in err:
        return f"no memory estimate from runq: {err.strip()[-200:]!r}"
    need = float(err.split("needs ~", 1)[1].split()[0]) + BIG_MARGIN
    have = os.environ.get("GWENEDEG_TEST_MEM")
    if have is None:
        with open("/proc/meminfo") as f:
            have = next((int(l.split()[1]) / 1048576 for l in f if l.startswith("MemAvailable:")), None)
    if have is None or float(have) >= need:
        return None
    return f"needs ~{need:.1f} GiB (runq's estimate + {BIG_MARGIN}), MemAvailable {float(have):.1f} GiB"


def greedy_text(stdout):
    """the reply of a -t 0 -i chat run (stdout without the trailing newline)"""
    return stdout.decode(errors="replace").rstrip("\n")


def test_models(a, res, tmp, texts, update=False, tags=None, prefix="models", path=None, lean=False, extra=None):
    """the checks of every downloaded FAMILY file (tags: only those; path: the file of the one tag; lean: the weights'
    checks only, without render / greedy / api / server, whose code paths another file of the run covers; extra: more
    options of the logits runs, e.g. -k bf16-dot; default check.py's --models-kv, lean with it)"""
    if extra is None and getattr(a, "models_kv", ""):
        extra, lean = ["-k", a.models_kv], True
    extra = list(extra or [])
    gdir = os.path.join(GOLDEN, "models")
    fam = [x for x in FAMILY if tags is None or x[0] in tags]
    where = lambda f: path or os.path.join(ROOT, "models", f)
    present = [(tag, where(f), v, gs) for tag, f, v, gs in fam if os.path.exists(where(f))]
    for tag, f, v, gs in fam:
        if not os.path.exists(where(f)):
            print(f"      {tag}: skipped (models/{f} not downloaded: scripts/download_models.py)")
    nd = MODEL_THREADS
    for tag, model, variant, goldens in present:
        why = mem_skip(a.runq, model) if tag in BIG else None
        if why:
            print(f"      {tag}: skipped (not enough memory: {why})")
            continue
        t0 = time.time()
        d = os.path.join(tmp, "models", tag)
        os.makedirs(d, exist_ok=True)

        def lf(name, chunk, threads=nd, runq=a.runq, sfx=""):
            path = os.path.join(d, f"{name}_b{chunk}_t{threads}{sfx}.bin")
            if not os.path.exists(path):
                logits_run(runq, model, texts[name], path, chunk, threads, extra)
            return path
        # template variant and renderings (tokenizer only)
        if not lean:
            render_cases(a.runq, model, variant, res, f"{prefix}/{tag}/render")
        # dequantization vs gguf-py
        if tag in DEQUANT:
            dequant_check(a.runq, model, DEQUANT[tag], res, f"{prefix}/{tag}/dequant")
        # greedy golden
        gpath = os.path.join(gdir, "greedy", f"{tag}.txt")
        want = open(gpath, encoding="utf-8").read().rstrip("\n") if os.path.exists(gpath) else None
        if not lean:
            p = run([a.runq, model] + GREEDY, nd)
            txt = greedy_text(p.stdout)
            if update:
                os.makedirs(os.path.dirname(gpath), exist_ok=True)
                with open(gpath, "w", encoding="utf-8") as fh:
                    fh.write(txt + "\n")
                want = txt
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
                kvx = ["--kv-type", extra[extra.index("-k") + 1]] if "-k" in extra else []
                for exe in (a.api_test, gen_api):
                    q = run([exe, model, "--digest-small"] + kvx, 4)
                    dg.append(dict(l.split()[1:3] for l in q.stdout.decode(errors="replace").splitlines() if l.startswith("DIGEST ")))
                diff = [k for k in dg[0] if dg[1].get(k) != dg[0][k]]
                res.report(len(dg[0]) >= 1 and not diff and set(dg[0]) == set(dg[1]), f"{prefix}/{tag}/api_digest_generic",
                           f"{len(dg[0])} batched passes (batched decode rows, ...): AVX-512 == generic" + (f"; differ: {diff}" if diff else ""))
        if a.api_test and tag in BIG:
            # (its batched decode of up to 17 sequences, each with its recurrent state and checkpoints, would need
            # ~22.4 GiB of RSS on the 27B: run by hand, docs/qwen38-speculative.md)
            print(f"      {tag}: api_test skipped (memory)")
        elif a.api_test and not lean:
            p = run([a.api_test, model], nd)
            lines = [l for l in p.stdout.decode(errors="replace").splitlines() if l.startswith(("PASS ", "FAIL "))]
            bad = [l[5:].split("  ")[0] for l in lines if l.startswith("FAIL ")]
            res.report(p.returncode == 0 and lines and not bad, f"{prefix}/{tag}/api",
                       f"api_test (batched decode rows == alone, mixed passes, ...): {len(lines) - len(bad)}/{len(lines)} passed"
                       + (f"; failed: {', '.join(bad)}" if bad else ""))
        for ok, name, det in server_smoke(a.runq, model, variant, want, None) if not lean else ():
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
# synth-q35-27b.gguf (if the 0.8B Q4_K_M file is next to the Q8_0 one): the 27B's layout (GQA 6, r = 3, dim 768, untied
# Q4_K embedding + Q6_K classifier, Q4_K / Q5_K / Q6_K matrices as in its Q4_K_M file): measured cos mean 0.9984 / 0.9979 /
# 0.9966 (en, code, long2), min 0.9803, top-1 281/338 (runq's int8 activations on cut-out K-quant weights, as the synth
# group's 0.9993-0.9958); with the q heads on kv head h % 2 instead of h // 6 0.31-0.40, value heads grouped 0.25-0.38
# synth-q35-27b-f32.gguf: the same with F32 alpha / beta (fp32 activations there, f32_rows), the MTP block, Qwen3.8's
# template: measured cos mean 0.9984 / 0.9976 / 0.9966, min 0.9791, top-1 290/338
Q35_SYNTH = {"synth-q35-r2.gguf": True, "synth-q35-r3.gguf": False, "synth-q35-27b.gguf": True,
             "synth-q35-27b-f32.gguf": True}  # file: full api_test (else digests)
Q35_SYNTH_REF = ["en", "code", "long2"]
TH_Q35_SYNTH = dict(cos_mean=0.999, cos_pos=0.99, top1=0.8)
TH_Q35_SYNTH_KQ = {"synth-q35-27b": dict(cos_mean=0.993, cos_pos=0.95, top1=0.7),
                   "synth-q35-27b-f32": dict(cos_mean=0.993, cos_pos=0.95, top1=0.7)}  # (per model, else TH_Q35_SYNTH)
Q35_SYNTH_SAME = [(("en", 1), ("en", 7)), (("en", 1), ("en", 64)), (("all", 1), ("all", 64)), (("long2", 1), ("long2", 64)),
                  (("long2", 64), ("long2", 512)), (("long2", 64), ("long2", 64, 3)),
                  (("en", 1), ("en", 1, "gen")), (("en", 7), ("en", 7, "gen")), (("long2", 64), ("long2", 64, "gen"))]


def q35_synth_build(a):
    """build/synth-q35 (tests/make_synth_q35.py) from the 0.8B Q8_0 file: the directory"""
    p = run([sys.executable, "-I", os.path.join(HERE, "make_synth_q35.py"), "--src", a.q35,
             "--src-q4", os.path.join(os.path.dirname(a.q35), "Qwen3.5-0.8B-Q4_K_M.gguf")])
    if p.returncode != 0:
        raise RuntimeError(f"make_synth_q35.py failed: {p.stderr.decode(errors='replace')[-400:]}")
    return p.stdout.decode().strip()


# the rejected files (check name, message)
Q35_BAD = [("bad-q35-nv.gguf", "nv", "is not a multiple of ssm.group_count"),
           ("bad-q35-hadamard.gguf", "hadamard", "Hadamard-folded weights (prism.hadamard.* metadata) are not supported"),
           ("bad-q35-nextn.gguf", "nextn", "metadata qwen35.nextn_predict_layers out of range")]


def q35_bad_run(a, d):
    """the rejected files of make_synth_q35.py: [(ok, name, detail)]"""
    out = []
    for fn, name, msg in Q35_BAD:
        if not os.path.exists(os.path.join(d, fn)):
            continue
        p = run([a.runq, os.path.join(d, fn), "-n", "1", "-i", "hi"], 1)
        err = p.stderr.decode(errors="replace").strip()
        out.append((p.returncode == 1 and msg in err, f"q35/synth/reject_{name}", f"exit status {p.returncode}: {err[-200:]!r}"))
    return out


def q35_mtp_run(a, d, tmp, texts):
    """synth-q35-27b-mtp.gguf (the 27B's MTP block: blk.8 + nextn.*, block_count 9, nextn_predict_layers 1): the main
    model loads (the block validated, not used): its logits == synth-q35-27b.gguf's, bitwise (-b 64 and decode, both
    builds), its tokenizer alone too; and synth-q35-27b-f32.gguf (Qwen3.8's template, thinking on by default): the CLI
    chat's first prompt == -m render (the effort line merged with -y), and a second turn prefills only <|im_end|>\n +
    its user turn (Qwen3.8's history keeps the reply as generated): [(ok, name, detail)]"""
    import hashlib
    import re
    out = []
    mtp, base, f32 = (os.path.join(d, f) for f in ("synth-q35-27b-mtp.gguf", "synth-q35-27b.gguf", "synth-q35-27b-f32.gguf"))
    if not (os.path.exists(mtp) and os.path.exists(base)):
        return out
    for exe, b, th in [(a.runq, 64, SYNTH_THREADS), (a.runq, 1, SYNTH_THREADS)] + ([(a.generic, 7, 3)] if a.generic else []):
        hs = []
        for m in (base, mtp):
            path = os.path.join(tmp, f"mtp_{os.path.basename(m)}_{b}_{th}.bin")
            logits_run(exe, m, texts["en"], path, b, th)
            with open(path, "rb") as fh:
                hs.append(hashlib.sha256(fh.read()).hexdigest())
            os.remove(path)
        out.append((hs[0] == hs[1], f"q35/synth/mtp_logits/b{b}{'_gen' if exe == a.generic else ''}",
                    f"synth-q35-27b-mtp.gguf (MTP block) vs synth-q35-27b.gguf, -b {b}"
                    f"{' (generic build)' if exe == a.generic else ''}: " + ("bit-identical" if hs[0] == hs[1] else "DIFFER")))
    p = run([a.runq, mtp, "-m", "tokenize", "-i", "hello world"], 1)
    ids = p.stdout.decode(errors="replace").split()
    out.append((p.returncode == 0 and ids == ["14556", "1814"], "q35/synth/mtp_tokenizer_only",
                f"-m tokenize (no weights): exit status {p.returncode}, ids {ids}"))
    if not os.path.exists(f32):
        return out
    prefix = os.path.join(tmp, "chat38")
    p = run([a.runq, f32, "-t", "0", "-c", "1024", "-n", "6", "-e", "low", "-y", "Be brief."], 2, env={"RUNQ_CHAT_DUMP": prefix},
            input=b"Hi there\nAnd now?\n")
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi there"}]
    r = run([a.runq, f32, "-m", "render", "-e", "low", "-i", json.dumps(msgs)], 1)
    t = run([a.runq, f32, "-m", "tokenize", "-i", "<|im_end|>\n<|im_start|>user\nAnd now?<|im_end|>\n<|im_start|>assistant\n<think>\n"], 1)
    ids = open(prefix + ".ids").read().split() if p.returncode == 0 else []
    want, turn2 = r.stdout.decode().split(), t.stdout.decode().split()
    err = p.stderr.decode(errors="replace")
    np_ = [int(x) for x in re.findall(r"\[timing\] prompt: (\d+) tok", err)]
    so = p.stdout.decode(errors="replace")
    ok = p.returncode == 0 and len(want) > 40 and ids[:len(want)] == want and np_ == [len(want), len(turn2)] and \
        ids[-len(turn2) - 6:-6] == turn2 and so.count("<think>\n") == 2 and "chat template qwen3.8, thinking 1 (low)" in r.stderr.decode()
    out.append((ok, "q35/synth-q35-27b-f32/chat_qwen38", f"-e low -y: the first prompt ({len(want)} tokens) == -m render, the "
                f"second turn prefills {len(turn2)} tokens (<|im_end|>\\n + its user turn), the reply opens in <think>"
                + ("" if ok else f"  <- rc {p.returncode}, prompts {np_}, {len(ids)} cached, {err[-200:]!r}")))
    return out


def q35_server38(a, d):
    """runq serve on synth-q35-27b-f32.gguf (Qwen3.8's template): the prompt of each request == `-m render` of its
    messages with the request's thinking / effort (usage.prompt_tokens; and a greedy reply == the CLI's): thinking
    on by default (xhigh), reasoning_effort low / medium / none and chat_template_kwargs.reasoning_effort, system +
    developer messages merged, an assistant message's reasoning_content kept in the history; a system message after
    the first user one: 400 (the template raises). [(ok, name, detail)]"""
    model = os.path.join(d, "synth-q35-27b-f32.gguf")
    if not os.path.exists(model):
        return []
    S = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi there"}]
    hist = [{"role": "user", "content": "Hi!"}, {"role": "assistant", "content": "Hello! How can I help?",
                                                  "reasoning_content": "The user greets me."}, {"role": "user", "content": "And in French?"}]
    hist_r = [dict(m) for m in hist]
    hist_r[1] = {"role": "assistant", "content": "<think>\nThe user greets me.\n</think>\n\nHello! How can I help?"}
    two = [{"role": "system", "content": "You are terse."}, {"role": "developer", "content": "Answer in French."},
           {"role": "user", "content": "Hi!"}]
    # (request extras, the messages to render, -r, -e)
    cases = [("default", {}, S, 1, None), ("low", {"reasoning_effort": "low"}, S, 1, "low"),
             ("medium", {"reasoning_effort": "medium"}, S, 1, "medium"), ("none", {"reasoning_effort": "none"}, S, 0, None),
             ("kwargs_low", {"chat_template_kwargs": {"reasoning_effort": "low"}}, S, 1, "low"),
             ("enable_thinking_false", {"chat_template_kwargs": {"enable_thinking": False}}, S, 0, None),
             ("system_developer", {}, two, 1, None), ("reasoning_content", {}, hist, 1, None)]
    conv = [dict({"messages": r, "thinking": t}, **({"reasoning_effort": e} if e else {})) for _, _, m, t, e in cases
            for r in [hist_r if m is hist else m]]
    p = run([a.runq, model, "-m", "render", "-i", json.dumps(conv)], 1)
    want = [len(l.split()) for l in p.stdout.decode().splitlines()]
    cli = run([a.runq, model, "-t", "0", "-c", "2048", "-n", "6", "-y", "Be brief.", "-i", "Hi there"], 4)
    cli_text = cli.stdout.decode(errors="replace").split("<think>\n", 1)[-1].strip()
    proc = subprocess.Popen([a.runq, "serve", model, "--port", "0", "-c", "2048", "--parallel", "2", "--slots", "2",
                             "--threads", "4"], stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
    out = []
    try:
        port, log, t0 = None, "", time.time()
        while time.time() - t0 < 120:
            line = proc.stderr.readline()
            if not line:
                break
            log += line
            if "listening on http://" in line:
                port = int(line.rsplit(":", 1)[1])
                break
        if port is None or len(want) != len(cases):
            return [(False, "q35/synth-q35-27b-f32/server38/start", f"{log[-300:]!r} {p.stderr.decode()[-200:]!r}")]

        def post(body):
            rq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                        {"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(rq, timeout=300) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read() or b"{}")
        for (name, extra, msgs, think, eff), n in zip(cases, want):
            st, r = post(dict({"messages": msgs, "temperature": 0, "max_tokens": 6}, **extra))
            got = r.get("usage", {}).get("prompt_tokens") if st == 200 else None
            m = r["choices"][0]["message"] if st == 200 else {}
            ok = st == 200 and got == n and (("reasoning_content" in m) == bool(think))
            det = f"prompt {got} tokens == -m render -r {think}{' -e ' + eff if eff else ''} ({n})"
            if name == "default":
                ok &= m.get("reasoning_content", "").strip() == cli_text
                det += f", the greedy reply (reasoning) == the CLI's: {m.get('reasoning_content', '')!r}"
            out.append((ok, f"q35/synth-q35-27b-f32/server38/{name}", det + ("" if ok else f"  <- status {st}: {json.dumps(r)[:200]}")))
        st, r = post({"messages": [{"role": "user", "content": "Hi"}, {"role": "system", "content": "late"}], "max_tokens": 1})
        ok = st == 400 and "at the beginning" in r.get("error", {}).get("message", "")
        out.append((ok, "q35/synth-q35-27b-f32/server38/late_system", f"status {st}: {r.get('error', {}).get('message', '')!r}"))
    except Exception as e:  # noqa: BLE001
        out.append((False, "q35/synth-q35-27b-f32/server38/error", f"{type(e).__name__}: {e}"))
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
    return out


def q35_serve_slots(a):
    """runq serve's slot budget on the 0.8B (engine.c): a slot counts its KV cache, its recurrent state and the state's
    checkpoints (--checkpoints, default 4), as computed here from the file's metadata; --kv-mem 0.25 GiB at -c 2048
    holds 2 slots of ~121 MiB (the KV cache alone, 24 MiB, would allow 10): [(ok, name, detail)]"""
    import re
    f = G.GgufFile(a.q35)
    kv = lambda k: int(f.kv[f"qwen35.{k}"])
    L, iv = kv("block_count"), kv("full_attention_interval")
    n_attn = L // iv
    conv, dk, nk, nv, inner = (kv(f"ssm.{k}") for k in ("conv_kernel", "state_size", "group_count", "time_step_rank", "inner_size"))
    dv, ctx, budget = inner // nv, 2048, 0.25 * 2**30
    conv_dim = 2 * nk * dk + nv * dv
    pg = lambda n: -(-n // 4096) * 4096
    kv_b = 2 * pg(n_attn * ctx * kv("attention.head_count_kv") * kv("attention.key_length") * 2) + 4 * ctx  # (+ the token array)
    state = pg((L - n_attn) * (((conv - 1) * conv_dim + 15) // 16 * 16 + nv * dk * dv) * 4)
    per = kv_b + 5 * state
    want = int(budget // per)
    p = subprocess.Popen([a.runq, "serve", a.q35, "--port", "0", "-c", str(ctx), "--slots", "8", "--parallel", "2", "--kv-mem", "0.25",
                          "--threads", "2"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    lines, t0 = [], time.time()
    try:
        for line in p.stderr:
            lines.append(line)
            if "listening on" in line or time.time() - t0 > 60:
                break
    finally:
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()
    m = re.search(r"(\d+) slot\(s\) of 2048 positions \(per slot ([\d.]+) MiB: KV \d+ KB per position \+ recurrent state ([\d.]+) MiB "
                  r"x \(1 \+ 4 checkpoints\)", "".join(lines))
    got = (int(m.group(1)), float(m.group(2)), float(m.group(3))) if m else None
    ok = bool(m) and got[0] == want and abs(got[1] - per / 2**20) < 0.06 and abs(got[2] - state / 2**20) < 0.06 and want < budget // kv_b
    return [(ok, "q35/serve_slots", f"--kv-mem 0.25 -c 2048: {got[0] if m else '?'} slot(s) of {got[1] if m else '?'} MiB "
             f"(want {want} of {per / 2**20:.1f} MiB: KV {kv_b / 2**20:.1f} + 5 x state {state / 2**20:.1f}; KV alone: {int(budget // kv_b)})"
             + ("" if m else f"; no info line in {''.join(lines)[-300:]!r}"))]


def q35_server_tools(a):
    """tests/server_test.py --tools on the 0.8B (runq serve's tool calls: prompts == -m render, calls parsed and typed,
    streamed, with thinking, a round trip, errors): [(ok, name, detail)]"""
    p = run([sys.executable, "-I", os.path.join(HERE, "server_test.py"), "--runq", a.runq, "--model", a.q35, "--tools",
             "--threads", "2"], 2)
    lines = [l for l in p.stdout.decode(errors="replace").splitlines() if l.startswith(("PASS ", "FAIL "))]
    out = [(l.startswith("PASS "), f"q35/server/{l[5:].partition('  ')[0].strip()}", l[5:].partition("  ")[2].strip()) for l in lines]
    if p.returncode not in (0, 1) or not lines:
        out.append((False, "q35/server/tools_run", f"exit status {p.returncode}: {(p.stdout + p.stderr).decode(errors='replace')[-300:]}"))
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


def q35_api_dot(a):
    """api_test on the 0.8B file with the bf16 dot-product attention (--kv-type bf16-dot: every sequence of its
    tests), and its batched passes' digests on the generic build"""
    p = run([a.api_test, a.q35, "--kv-type", "bf16-dot"], SYNTH_THREADS)
    out = q35_api_lines(p, "q35_dot/api")
    gen = a.api_test + "-generic"
    if os.path.exists(gen):
        dg = [dict(l.split()[1:3] for l in q.stdout.decode(errors="replace").splitlines() if l.startswith("DIGEST "))
              for q in (p, run([gen, a.q35, "--digest-small", "--kv-type", "bf16-dot"], 3))]
        common = [k for k in dg[1] if k in dg[0]]
        diff = [k for k in common if dg[1][k] != dg[0][k]]
        out.append((len(common) >= 2 and not diff, "q35_dot/api_digest_generic",
                    f"{len(common)} batched passes: AVX-512 ({SYNTH_THREADS} threads) == generic (3)" + (f"; differ: {diff}" if diff else "")))
    return out


def q35_api_lines(p, name):
    lines = [l for l in p.stdout.decode(errors="replace").splitlines() if l.startswith(("PASS ", "FAIL "))]
    bad = [l[5:].split("  ")[0] for l in lines if l.startswith("FAIL ")]
    return [(p.returncode == 0 and bool(lines) and not bad, name, f"api_test --kv-type bf16-dot (every path of the API): "
             f"{len(lines) - len(bad)}/{len(lines)} passed" + (f"; failed: {', '.join(bad)}" if bad else ""))]


def q35_synth_run(a, tmp, model, texts, full_api, kv=""):
    """the checks of one synthetic model, run in the background: [(ok, name, detail)]. The logits files (248k ids per
    position: long2 is 0.3 GB) are reduced to a sha256 and the reference's classifier rows, then removed. kv: a KV cache
    type (-k; e.g. bf16-dot, the bf16 dot-product attention: the same checks, the same thresholds)"""
    import hashlib
    import ref_qwen35 as R35
    tag = os.path.basename(model)[:-5]
    pre = f"q35/{tag}" + (f"_{kv}" if kv else "")
    extra, kvx = (["-k", kv], ["--kv-type", kv]) if kv else ([], [])
    th = TH_Q35_SYNTH_KQ.get(tag, TH_Q35_SYNTH)
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
            path = os.path.join(tmp, f"{tag}{kv}_{name}_b{chunk}_{th}{'_gen' if gen else ''}.bin")
            logits_run(a.generic if gen else a.runq, model, texts[name], path, chunk, th, extra)
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
        if not c.mean() >= th["cos_mean"]: why.append(f"cos mean < {th['cos_mean']}")
        if not c.min() >= th["cos_pos"]: why.append(f"cos min < {th['cos_pos']} at pos {int(c.argmin())}")
        out.append((not why, f"{pre}/logits_ref/{n}", f"n={len(ids)}: cos mean {c.mean():.5f} min {c.min():.5f}, top1 "
                    f"{int(top1.sum())}/{len(top1)} vs the numpy reference" + ("  <- " + "; ".join(why) if why else "")))
    if t1:
        out.append((bool(np.mean(t1) >= th["top1"]), f"{pre}/logits_ref/top1_pooled",
                    f"{int(np.sum(t1))}/{len(t1)} (min {th['top1']})"))
    # bitwise invariants
    for x, y in Q35_SYNTH_SAME:
        if len(y) > 2 and y[2] == "gen" and not a.generic:
            continue
        kind = "kernels" if len(y) > 2 and y[2] == "gen" else "threads" if len(y) > 2 else "batched"
        what = "AVX-512 vs generic" if kind == "kernels" else f"{SYNTH_THREADS} vs 3 threads" if kind == "threads" else \
            f"-b {y[1]} vs -b {x[1]}"
        ok = lf(x)[0] == lf(y)[0]
        out.append((ok, f"{pre}/{kind}/{y[0]}_b{y[1]}", f"{what}: " + ("bit-identical" if ok else "DIFFER")))
    # the file's sampling defaults (TODO P5; general.sampling.temp / top_p / top_k = 1.0 / 0.95 / 20 as in the 27B's
    # file, written by make_synth_q35.py): applied to the options not given, silently. Seeded generate without -p / -K
    # == with -p 0.95 -K 20 (stdout, stderr without the numbers), != with the CLI's own defaults -p 0.9 -K 0
    if tag == "synth-q35-27b" and not kv:
        import re
        base = [a.runq, model, "-c", "256", "-t", "0.8", "-s", "7", "-n", "24", "-m", "generate", "-i", texts["en"][:60]]
        ps = [run(base + x, SYNTH_THREADS) for x in ([], ["-p", "0.95", "-K", "20"], ["-p", "0.9", "-K", "0"])]
        err = [re.sub(rb"[0-9.]+", b"N", p.stderr) for p in ps]
        ok = all(p.returncode == 0 for p in ps) and ps[0].stdout == ps[1].stdout and err[0] == err[1] and ps[0].stdout != ps[2].stdout
        out.append((ok, f"{pre}/sampling_defaults", "seeded generate: no -p / -K == -p 0.95 -K 20 (stdout, stderr), != -p 0.9 -K 0"
                    + ("" if ok else f": exit {[p.returncode for p in ps]}, same {ps[0].stdout == ps[1].stdout} / {err[0] == err[1]}, "
                       f"differs from the CLI defaults {ps[0].stdout != ps[2].stdout}")))
    # the library API: batched decode rows == alone, checkpoints (full run), the batched passes' digests vs generic
    if a.api_test:
        if full_api:
            p = run([a.api_test, model] + kvx, SYNTH_THREADS)
            lines = [l for l in p.stdout.decode(errors="replace").splitlines() if l.startswith(("PASS ", "FAIL "))]
            bad = [l[5:].split("  ")[0] for l in lines if l.startswith("FAIL ")]
            out.append((p.returncode == 0 and bool(lines) and not bad, f"{pre}/api", f"api_test (batched decode rows == alone, "
                        f"checkpoints, ...): {len(lines) - len(bad)}/{len(lines)} passed" + (f"; failed: {', '.join(bad)}" if bad else "")))
        gen_api = a.api_test + "-generic"
        if os.path.exists(gen_api):
            dg = []
            for exe, th in ((a.api_test, SYNTH_THREADS), (gen_api, 3)):
                q = run([exe, model, "--digest-small"] + kvx, th)
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
    rec, rec4, rec3, recd = Recorded(), Recorded(), Recorded(), Recorded()
    t = os.path.join(tmp, "q35")
    os.makedirs(os.path.join(t, "q4"), exist_ok=True)
    os.makedirs(os.path.join(t, "dot"), exist_ok=True)
    os.makedirs(os.path.join(t, "q3"), exist_ok=True)
    q4 = os.path.join(os.path.dirname(a.q35), "Qwen3.5-0.8B-Q4_K_M.gguf")
    q4 = q4 if os.path.exists(q4) else None
    q3 = os.path.join(os.path.dirname(a.q35), "Qwen3.5-0.8B-Q3_K_M.gguf")
    q3 = q3 if os.path.exists(q3) else None
    try:
        sd, serr = q35_synth_build(a), None
    except RuntimeError as e:
        sd, serr = None, str(e)
    os.makedirs(os.path.join(t, "synth"), exist_ok=True)
    synth = [ex.submit(q35_synth_run, a, os.path.join(t, "synth"), os.path.join(sd, fn), texts, full)
             for fn, full in Q35_SYNTH.items() if os.path.exists(os.path.join(sd, fn))] + [ex.submit(q35_bad_run, a, sd)] + \
        [ex.submit(q35_mtp_run, a, sd, os.path.join(t, "synth"), texts), ex.submit(q35_server38, a, sd)] + \
        [ex.submit(render_run, a.runq, os.path.join(sd, fn), "qwen3.8", f"q35/synth/render_{fn[10:-5]}")
         for fn in ("synth-tok-qwen3.8.gguf", "synth-tok-qwen3.8-hf.gguf")] + \
        ([ex.submit(q35_two_run, a, sd)] if a.api_test else []) if sd else []
    synth.append(ex.submit(q35_serve_slots, a))
    synth.append(ex.submit(q35_server_tools, a))
    # the bf16 dot-product attention (KV bf16-dot, TODO G4 (b)): synth-q35-27b (the 27B's attention shape) and every API
    # path on the 0.8B (the logits / bitwise checks of the 0.8B: test_models below, prefix q35_dot)
    s27 = os.path.join(sd, "synth-q35-27b.gguf") if sd else ""
    if s27 and os.path.exists(s27):
        os.makedirs(os.path.join(t, "synth_dot"), exist_ok=True)
        synth.append(ex.submit(q35_synth_run, a, os.path.join(t, "synth_dot"), s27, texts, True, "bf16-dot"))
    if a.api_test:
        synth.append(ex.submit(q35_api_dot, a))
    # the Q3_K_M file (lean) on a worker of its own from the start: on mx, after the other two, it made the suite's
    # tail (that chain) ~35 s longer; next to them the suite took as long as without it (185.4 / 186.5 s vs 181.0 /
    # 187.1 s, interleaved, load ~25)
    q3x = cf.ThreadPoolExecutor(max_workers=1) if q3 else None
    return {"a": a2, "tok_test": tok_test, "synth": synth, "synth_err": serr, "q3x": q3x,
            "chats": [ex.submit(chat_case, a2, tmp, vocab, d, *c) for c in Q35_CHAT_CASES],
            "rec": rec, "models": [mx.submit(test_models, a, rec, t, texts, False, {Q35_TAG}, "q35", a.q35),
                                   mx.submit(test_models, a, recd, os.path.join(t, "dot"), texts, False, {Q35_TAG}, "q35_dot",
                                             a.q35, True, ["-k", "bf16-dot"])] + (
                [mx.submit(test_models, a, rec4, os.path.join(t, "q4"), texts, False, {Q35_Q4_TAG}, "q35", q4)] if q4 else []) + (
                [q3x.submit(test_models, a, rec3, os.path.join(t, "q3"), texts, False, {Q35_Q3_TAG}, "q35", q3, True)] if q3 else []),
            "rec4": rec4, "rec3": rec3, "recd": recd}


def test_q35(a, res, tmp, texts, S):
    if S is None:
        print("      skipped: no Qwen3.5-0.8B Q8_0 GGUF (--q35; scripts/download_models.py 3.5-0.8b-q8, 3.5-0.8b-q4km, 3.5-0.8b-q3km)")
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
    if S["q3x"]:
        S["q3x"].shutdown()
    for ok, n, detail, xfail in S["rec"].reports + S["recd"].reports + S["rec4"].reports + S["rec3"].reports:
        res.report(ok, n, detail, xfail)
