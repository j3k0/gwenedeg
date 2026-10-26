#!/usr/bin/env python3
"""Gwenedeg regression checks (numpy only). Normally run via tests/run_tests.sh.

Tests (see tests/README.md):
  tokenizer  runq -m tokenize vs HF ids              (tests/golden/tokenizer.json)
             + NFC around invalid UTF-8 (token bytes vs unicodedata on the valid runs)
             + encode speed (~2,000 tokens, one 30,000-byte chunk), from runq's stderr timing
  logits     runq -m logits vs HF fp32 logits         (tests/golden/logits/*.npz)
             sequential path (-b 1) and batched prefill (-b 64), same thresholds
  batched    batched logits (-b 7, -b 64) vs sequential (-b 1): near-exact (bit-identical now)
  threads    logits bit-identical for 1 vs N OpenMP threads (sequential and batched), 3 vs N (long)
  kernels    logits bit-identical for the AVX-512 kernels (runq: VNNI GEMM/GEMV, attention) and the
             generic C ones (runq-generic, -DRUNQ_GENERIC): GEMV (-b 1) and GEMM (-b 7, -b 64) paths
  greedy     runq -t 0 output vs our own golden text  (tests/golden/greedy/*.txt), with the
             default batched prefill (and -b 5 once); seeded -t 0.8 chat equal for -b 1 and 5;
             seeded sampling goldens (chat -t 0.8 -p 0.9, generate -t 1.5 -p 0.95)
  chat       multi-turn chat from piped stdin (-t 0): transcript golden, memory across turns, and the
             KV cache conversation (RUNQ_CHAT_DUMP) == the chat-template rendering, its logits
             bit-identical to a fresh -m logits run on it; also -r 1 (reasoning dropped from the
             history) and context full (-c 160: oldest turns dropped, system prompt kept)
  gguf       the GGUF Q8_0 file (--gguf; tokenizer, templates and weights read from it): tokenizer
             goldens, logits vs HF (-b 1 and -b 64, same thresholds), batched vs sequential, threads
             and VNNI vs generic bitwise, and the 4 chat templates == the .bin side files (chat token
             ids of both models); its greedy goldens (gguf_*) run in the greedy group
  q4         the GGUF Q4_K_M file (--q4; Q4_K / Q6_K weights): dequantized rows vs gguf-py bitwise
             (tests/golden/dequant, from the blocks and from the repacked layouts), logits vs HF
             (loose Q4 thresholds) and vs the Q4 reference (golden/logits_q4: HF fp32 with the
             dequantized Q4_K_M weights; tight), batched vs sequential, threads and the AVX-512
             K-quant kernels (GEMV, GEMM over several chunks) vs the generic ones bitwise; its greedy
             goldens (q4_*) run in the greedy group
  server     runq serve over HTTP (tests/server_test.py, in the background from the start): endpoints,
             completions == runq -m chat, prefix reuse bitwise, streaming, json_schema constrained
             decoding, errors, queue / cancellation, fuzzing, shutdown
  kv         the KV cache types other than the default f16 (-k f32, -k bf16): logits vs HF (en, code,
             long2), batched vs sequential, threads and AVX-512 vs generic bitwise; their greedy goldens
             (f32_generate, bf16_generate) run in the greedy group
  synth      synthetic GGUF models cut out of the Q4_K_M file (tests/make_synth.py; the shapes of the bigger
             Qwen3 models: dim 768, GQA 4, untied Q4_K embedding, RoPE theta 5e6, the 2507 templates): logits vs a
             numpy reference, batched / threads / generic bitwise, api_test, chat template renderings
             (tests/check_models.py; in the background from the start)
  models     (not in the default set: run_tests.sh --models) every downloaded Qwen3 family model: logits vs
             HF / the dequantized-weights reference, bitwise invariants, greedy goldens, templates, runq serve
  api        the library API (tests/api_test.c per model, in the background from the start): sequences
             interleaved / batched / truncated == alone, batched decode of N sequences (1..17, 65) ==
             each alone incl. past 256 positions and mixed passes, rq_sample_batch, errors, chat
             rendering, detokenization; the batched passes' digests at 1 / 8 / 16 threads and on the
             generic build == the main runs

Logits metrics, per position, over U = (fixed 4096-id vocab subset) + (HF top-64 ids):
  cos   cosine similarity of runq vs HF logits on U
  top1  runq argmax over U == HF top-1
  top5  |runq top-5 over U  &  HF top-5| / 5
Position 0 is reported separately: Qwen3 puts a ~6500 "massive activation" in the residual
stream of the first token from layer ~3 on (HF fp32 too), and Q8_0 (64-wide groups, weights
and activations) loses most of the other values next to it, so pos-0 logits are much noisier.

Baseline, Q8_0 engine at the commit adding this suite (2026-10-24, OMP_NUM_THREADS=8):
  prompt  n   cos pos0  cos mean  cos min   top1    top5     (cos mean/min over pos >= 1)
  en      15  0.8316    0.99833   0.99607   13/15   0.893
  code    24  0.8323    0.99920   0.99838   21/24   0.942
  zh      15  0.9064    0.99958   0.99930   14/15   0.880
  chat    15  0.8967    0.99888   0.99663   14/15   0.920
  count   15  0.8653    0.99941   0.99901   14/15   0.907
  pooled: cos mean 0.99909 (79 positions), top1 76/84 = 0.905
Deliberate breakage (scratch builds, all fail): wrong RoPE theta 1e4 (the subtlest: pooled cos
0.9876, per-prompt cos mean 0.977..0.995), interleaved RoPE pairs (0.80..0.91), no QK-norm,
KV cache written at pos+1, V read from t-1, no final norm, attention scale 1/d, rmsnorm eps
1e-2, matmul dropping its last group (all <= 0.96). Rounding -> truncation in the activation
quantizer (a pure precision loss) fails logits/en (cos mean 0.99432) and greedy.
"""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import tempfile
import time
import unicodedata

# numpy's BLAS threads (OpenBLAS: OMP_NUM_THREADS otherwise, 8): its few matrix products here (the synth group's
# reference forward pass) are small, and idle BLAS threads spin next to the runq processes (~20 core-s per run)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_models  # noqa: E402  (groups synth and models: tests/check_models.py)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "tests", "golden")

# Thresholds: ~3x the baseline error (1 - cos) for the means, ~5x for single positions.
COS_MEAN_MIN = 0.995   # per prompt, mean over positions >= 1
COS_POS_MIN = 0.980    # every single position >= 1
COS_POS0_MIN = 0.75    # position 0 (massive-activation token, inherently noisy under Q8_0)
TOP1_MIN = 0.75        # per prompt, fraction of positions
TOP1_ALL_MIN = 0.85    # all prompts pooled
COS_ALL_MIN = 0.997    # all prompts pooled, mean over positions >= 1
TOP5_MIN = 0.75        # per prompt, mean overlap
TH_Q8 = dict(cos_mean=COS_MEAN_MIN, cos_pos=COS_POS_MIN, cos_pos0=COS_POS0_MIN, top1=TOP1_MIN,
             top1_all=TOP1_ALL_MIN, cos_all=COS_ALL_MIN, top5=TOP5_MIN)
# Q4_K_M (--q4) vs HF fp32: the 4-bit weights alone cost ~25x the Q8_0 error (an fp32 HF model with
# the dequantized Q4_K_M weights is as far from HF), so these are loose sanity bounds, ~2x the
# baseline error (tests/README.md). Position 0 is near useless here (cos ~0.36 in the baseline; a
# broken Q4_K kernel gave 0.44): bound only against NaN-like garbage
TH_Q4_HF = dict(cos_mean=0.94, cos_pos=0.85, cos_pos0=0.10, top1=0.50, top1_all=0.65, cos_all=0.97, top5=0.50)
# Q4_K_M vs the Q4 reference (golden/logits_q4: HF fp32 with the GGUF's dequantized weights): only
# runq's int8 activations differ, as for Q8_0 vs HF; ~3x the baseline error for means, ~5x per position
TH_Q4_REF = dict(cos_mean=0.995, cos_pos=0.98, cos_pos0=0.99, top1=0.75, top1_all=0.90, cos_all=0.998, top5=0.75)

# Batched vs sequential logits (same engine, same numerics): currently bit-identical. Deliberate
# breakage (scratch builds) gives max |diff| 17..32 and min cos <= 0.47: causal mask leaking one
# position, K RoPE at the chunk-local position, chunk K written one position early.
BATCH_CHUNKS = [7, 64]   # 7: several chunks with boundaries mid-prompt; 64: one chunk (two for "all")
BATCH_MAX_ABS = 1e-3     # logits are O(10)
BATCH_COS_MIN = 0.99999  # per row, full vocab
BATCH_LONG = "all"       # extra prompt: all logits prompts concatenated (86 tokens, 2 chunks at -b 64)
BATCH_LONG2 = "long"     # "all" twice + en: 187 tokens, 12 KV cache tiles, 3 chunks at -b 64
BATCH_LONG3 = "long2"    # long + all + code: 297 tokens, more than one attention block (ATT_P = 256
                         # positions, src/kernels_avx512.h): the partial results and their combination
LONG3_CHUNKS = [64, 512]  # 5 chunks / one chunk, both bitwise vs -b 1 (sequential decode: positions
                          # 256.. split over threads per (kv head, block))
LONG3_THREADS = [1, 3, 16]  # -b 64 vs OMP_NUM_THREADS (8): other attention / GEMM partitions
SEQ, BATCH = 1, 64       # -b of the sequential path and of the batched path (default -b is 512: the
                         # same code, one chunk for the short prompts either way)

CTX = "4096"
TOKENIZE_MAX_MS = 100.0  # encode time limit for the speed checks (measured ~1 ms and ~13 ms)
GREEDY_CASES = [
    # name, extra args (all run with -t 0 -c 4096 -s 1)
    ("chat", ["-m", "chat", "-n", "40", "-i", "What is the capital of Australia? Answer in one sentence."]),
    ("chat_thinking", ["-m", "chat", "-r", "1", "-n", "48", "-i", "What is 17 * 23?"]),
    ("chat_system", ["-m", "chat", "-n", "40", "-y", "You are a pirate. Always answer like a pirate.",
                     "-i", "How are you today?"]),
    ("generate", ["-m", "generate", "-n", "40", "-i", "Once upon a time, in a small village by the sea,"]),
    # the same golden with several prefill chunks (generate prints the prompt tokens itself)
    ("generate@b5", ["-m", "generate", "-n", "40", "-b", "5", "-i", "Once upon a time, in a small village by the sea,"]),
    # the GGUF file (name prefix gguf_: run on --gguf)
    ("gguf_chat", ["-m", "chat", "-n", "40", "-i", "What is the capital of Australia? Answer in one sentence."]),
    ("gguf_generate", ["-m", "generate", "-n", "40", "-i", "Once upon a time, in a small village by the sea,"]),
    # the Q4_K_M file (prefix q4_: run on --q4; 24 tokens: written in Step 6b-1 with the generic
    # kernels, which the AVX-512 ones match bit for bit)
    ("q4_chat", ["-m", "chat", "-n", "24", "-i", "What is the capital of Australia? Answer in one sentence."]),
    ("q4_generate", ["-m", "generate", "-n", "24", "-i", "Once upon a time, in a small village by the sea,"]),
    # the other KV cache types (the default is f16): f32_generate is the fp32 cache's text (= the
    # generate golden before Step 7b, which the fp32 cache still reproduces), bf16_generate the bf16's
    ("f32_generate", ["-m", "generate", "-n", "40", "-k", "f32", "-i", "Once upon a time, in a small village by the sea,"]),
    ("bf16_generate", ["-m", "generate", "-n", "40", "-k", "bf16", "-i", "Once upon a time, in a small village by the sea,"]),
]
# seeded sampling (-t 0.8 -s 42): output must not depend on -b (chat advances the RNG once per
# prompt token, as before batched prefill)
SEEDED_ARGS = ["-m", "chat", "-n", "40", "-t", "0.8", "-s", "42", "-c", CTX, "-i", "Tell me about cats."]
SEEDED_CHUNKS = [1, 5]  # token by token vs 2 chunks (-b 64 runs the same code)
# seeded sampling goldens (golden/greedy/<name>.txt): the -b 1 run of SEEDED_ARGS, and generate at a
# high temperature (~1-2k top-p candidates per token, nuclei of up to ~100 tokens)
SEEDED_GEN_ARGS = ["-m", "generate", "-n", "40", "-t", "1.5", "-p", "0.95", "-s", "7", "-c", CTX,
                   "-i", "Once upon a time, in a small village by the sea,"]
THREADS_PROMPT = "en"
# VNNI vs generic kernels: (prompt, -b). -b 1 on en: GEMV (decode path, incl. the classifier);
# -b 7 on en (15 tokens): GEMM tiles of 6 + 1 rows, last chunk 1 row; -b 64 on all (86 tokens,
# chunks 64 + 22): tiles of 6 + 4 rows, LOGITS_ALL classifier GEMM
# long2 on -b 64 (297 tokens, 5 chunks): attention over 19 KV tiles and past one attention block, causal row
# blocks in later chunks (generic: C attention). (long, 187 tokens, was a case too until Step 10b: long2 = long +
# all + code takes the same paths, so it covers it; the generic build's run of long was ~6 core-seconds)
KERNEL_CASES = [(THREADS_PROMPT, SEQ), (THREADS_PROMPT, BATCH_CHUNKS[0]), (BATCH_LONG, BATCH), (BATCH_LONG3, BATCH)]
THREADS_LONG = 3         # long -b 64: OMP_NUM_THREADS=3 vs the default (other work partitions)
# GGUF (group size 32): logits vs HF batched on all prompts and sequential on GGUF_SEQ; batched
# (prompt, -b) vs -b 1 bitwise (2-chunk GEMM tiles and the all-rows classifier: GGUF_KERNELS all_b64);
# threads 8 vs 3 (-b 7); VNNI vs generic (prompt, -b); the chat templates (-r / -y variants) vs the
# .bin side files
GGUF_SEQ = [THREADS_PROMPT, "code"]
GGUF_BATCHED = [(THREADS_PROMPT, BATCH_CHUNKS[0]), (THREADS_PROMPT, BATCH), ("code", BATCH_CHUNKS[0])]
GGUF_KERNELS = [(THREADS_PROMPT, SEQ), (THREADS_PROMPT, BATCH_CHUNKS[0]), (BATCH_LONG, BATCH)]
# Q4_K_M: logits vs HF and vs the Q4 reference batched on all prompts, sequential on Q4_SEQ; batched
# (prompt, -b) vs -b 1 bitwise; threads 8 vs 3 (Q4_THREADS: other GEMM chunk sizes and partitions);
# AVX-512 kernels vs the generic ones (Q4_KERNELS, bitwise): en -b 1 = the GEMV (decode, incl. the
# Q6_K classifier), en -b 7 = GEMM tiles of 6 + 1 rows (Q4_K) / 4 + 3 (Q6_K) and a 1-row chunk,
# all -b 7 = 13 chunks (last one 2 rows), all -b 64 = 2 chunks with tiles 6 + 4 / 4 + 2 and the
# all-rows classifier GEMM (Q6_K). ("all", not "long": the generic build takes ~1 s per 86 tokens
# with all-rows logits, mostly the classifier)
Q4_SEQ = [THREADS_PROMPT]
Q4_BATCHED = [(THREADS_PROMPT, BATCH_CHUNKS[0]), (BATCH_LONG, BATCH_CHUNKS[0]), (BATCH_LONG, BATCH)]
Q4_THREADS = [(BATCH_LONG, BATCH_CHUNKS[0]), (BATCH_LONG, BATCH)]
Q4_KERNELS = [(THREADS_PROMPT, SEQ), (THREADS_PROMPT, BATCH_CHUNKS[0]), (BATCH_LONG, BATCH_CHUNKS[0]), (BATCH_LONG, BATCH)]
# KV cache types other than the default (f16, Step 7b), on the .bin (group kv): logits vs HF (-b 64 on
# KV_LOGITS + pooled, the same thresholds; long2 past one attention block), batched vs sequential (en
# -b 7, long2 -b 64), threads (long2 -b 64, 8 vs 3) and AVX-512 vs generic (en -b 1, long2 -b 64), bitwise
KV_TYPES = ["f32", "bf16"]
KV_LOGITS = [THREADS_PROMPT, "code"]
KV_BATCHED = [(THREADS_PROMPT, BATCH_CHUNKS[0]), (BATCH_LONG3, BATCH)]
KV_KERNELS = [(THREADS_PROMPT, SEQ), (BATCH_LONG3, BATCH)]
GEN_THREADS = 4          # all generic-build runs (kernels/*, gguf_kernels/*, q4_kernels/*), in the
                         # background (results do not depend on the thread count)
SEQ_THREADS = 4          # the sequential (-b 1) runs unless a test names a thread count: token by token,
                         # 113 phases each (decode is memory-bound: 4 threads ~ 8 alone), the longest runs
                         # of the suite (long2: 297 tokens with every row's logits), 3 at a time next to the
                         # background work: at 8 threads ~8 s each, at 4 ~3 s (Step 10a)
PREFETCH_JOBS = 3        # runq -m logits runs at OMP_NUM_THREADS in parallel (prefetch_all), next to the
                         # background generic runs: the suite's ~60 logits runs, ~0.1 s of startup each
GGUF_TEMPLATES = [("plain", [], ".template"), ("thinking", ["-r", "1"], ".template.with-thinking"),
                  ("system", ["-y", "You are terse."], ".template.with-system"),
                  ("system_thinking", ["-r", "1", "-y", "You are terse."], ".template.with-system-and-thinking")]
# multi-turn chat (stdin, one line per turn; all -t 0): (name, args, user lines, checks). Each run
# dumps its KV cache ids and sampled logits (RUNQ_CHAT_DUMP), compared with a fresh -m logits run on
# the decoded conversation (bitwise), whose text must be the expected chat-template rendering
CHAT_CASES = [
    # memory across turns; transcript golden golden/greedy/chat_multiturn.txt
    ("multiturn", ["-n", "32"],
     ["My name is Ada and my favourite colour is green.", "What is my name and favourite colour?",
      "Spell my name backwards."], {"golden": True, "mentions": (1, ["ada", "green"])}),
    # -r 1: the reasoning of turn 1 is dropped from the history (re-prefill of its answer)
    ("thinking", ["-r", "1", "-n", "100"], ["Hi!", "And in French?"], {"stripped": 1}),
    # context full at -c 160: the oldest turns are dropped, the system prompt kept
    ("context_full", ["-c", "160", "-n", "30", "-y", "You are a terse assistant."],
     ["My name is Ada.", "Name three fruits.", "Name three animals.", "Name three colours.",
      "Name three cities.", "What is my name?"], {"dropped": True}),
    # a ~390-token prompt (system prompt: LONG_SYSTEM): the reply is decoded at positions past one
    # attention block (256), split over 16 threads per (kv head, block) and combined; its logits must
    # equal the batched prefill's (-m logits, 8 threads) bit for bit, and the generic build's
    ("long_context", ["-n", "12", "-y", "LONG_SYSTEM"], ["Summarize this in one sentence."],
     {"threads": 16, "generic": True}),
]
LONG_SYSTEM = ("The history of the printing press begins long before Gutenberg. Carved wooden blocks were used in "
               "China to print text and images on paper and silk, and movable type made of baked clay, later of wood "
               "and bronze, appeared there and in Korea centuries before it reached Europe. Around 1440, in Mainz, "
               "Johannes Gutenberg combined several existing technologies into a practical system: an adjustable hand "
               "mould for casting uniform metal type, an oil-based ink that adhered to metal, and a screw press adapted "
               "from those used to make wine and paper. The result was a dramatic reduction in the cost of books. Within "
               "fifty years, presses were operating in more than two hundred cities, and millions of volumes had been "
               "printed. Scholars could compare identical copies of a text, errors could be corrected in later editions, "
               "and ideas spread faster than any authority could suppress them. Questions for the reader: what changed "
               "first, and why did it matter?")
LONG_SYSTEM = LONG_SYSTEM + " " + LONG_SYSTEM  # 374 tokens


class Results:
    def __init__(self):
        self.passed, self.failed, self.xfailed, self.xpassed = [], [], [], []

    def report(self, ok, name, detail="", xfail=None):
        if xfail:
            if ok:
                self.xpassed.append(name)
                print(f"XPASS {name}  (marked xfail: {xfail}; remove the mark?)")
            else:
                self.xfailed.append(name)
                print(f"XFAIL {name}  ({xfail})")
        elif ok:
            self.passed.append(name)
            print(f"PASS  {name}" + (f"  {detail}" if detail else ""))
        else:
            self.failed.append(name)
            print(f"FAIL  {name}" + (f"  {detail}" if detail else ""))
        sys.stdout.flush()


def run(cmd, threads=None, **kw):
    env = dict(os.environ)
    if threads is not None:
        env["OMP_NUM_THREADS"] = str(threads)
    return subprocess.run(cmd, env=env, capture_output=True, **kw)


def runq_tokenize(a, text, timing=False):
    p = run([a.runq, a.model, "-c", "64", "-m", "tokenize", "-i", text], threads=1)
    if p.returncode != 0:
        raise RuntimeError(f"runq -m tokenize failed: {p.stderr.decode(errors='replace')[-500:]}")
    ids = [int(x) for x in p.stdout.split()]
    if not timing:
        return ids
    # stderr: "[tokenize] B bytes -> N tokens, encode X ms, tokenizer load Y ms"
    line = [l for l in p.stderr.decode().splitlines() if l.startswith("[tokenize]")][-1]
    f = line.replace(",", " ").split()
    return ids, float(f[f.index("encode") + 1]), float(f[f.index("load") + 1])


def runq_logits_run(a, text, path, threads, chunk):
    p = run([a.runq, a.model, "-c", CTX, "-m", "logits", "-b", str(chunk), "-i", text, "-o", path] + getattr(a, "extra", []),
            threads=threads)
    if p.returncode != 0:
        raise RuntimeError(f"runq -m logits failed: {p.stderr.decode(errors='replace')[-500:]}")


PENDING = {}  # logits file path -> future of the run writing it (prefetch_all)


def logits_file(a, tmp, name, text, chunk, threads=None):
    """runq -m logits for (prompt, chunk, threads), cached in tmp across tests (and prefetched)."""
    t = threads or (SEQ_THREADS if chunk == SEQ else int(os.environ["OMP_NUM_THREADS"]))
    path = os.path.join(tmp, f"logits_{name}_b{chunk}_t{t}.bin")
    fut = PENDING.get(path)
    if fut is not None:
        fut.result()
    elif not os.path.exists(path):
        runq_logits_run(a, text, path, t, chunk)
    return path


def prefetch_all(a, tmp, only, fg, bg):
    """Start every runq -m logits run the selected groups will read (the same files logits_file
    names): the generic build's at GEN_THREADS on bg, the others in group order on fg. The tests then
    run as before, waiting for their files (~60 short processes: their startup and serial parts
    overlap instead of adding up)"""
    texts = dict(batched_prompts())
    nd = int(os.environ["OMP_NUM_THREADS"])
    def sub(ex, var, d, name, chunk, threads=None):
        os.makedirs(d, exist_ok=True)
        t = threads or (SEQ_THREADS if chunk == SEQ else nd)
        path = os.path.join(d, f"logits_{name}_b{chunk}_t{t}.bin")
        if path not in PENDING and not os.path.exists(path):
            PENDING[path] = ex.submit(runq_logits_run, var, texts[name], path, t, chunk)
    gg = argparse.Namespace(**{**vars(a), "model": a.gguf}) if a.gguf else None
    q4 = argparse.Namespace(**{**vars(a), "model": a.q4}) if a.q4 else None
    if a.generic:  # (slowest first)
        for var, d, cases, grp in [(q4, os.path.join(tmp, "q4", "generic"), Q4_KERNELS, "q4"),
                                   (a, os.path.join(tmp, "generic"), KERNEL_CASES, "kernels"),
                                   (gg, os.path.join(tmp, "gguf", "generic"), GGUF_KERNELS, "gguf")]:
            if var is not None and grp in only:
                gen = argparse.Namespace(**{**vars(var), "runq": a.generic})
                for name, chunk in cases:
                    sub(bg, gen, d, name, chunk, GEN_THREADS)
    if a.generic and "kv" in only:
        for kv in KV_TYPES:
            gen = argparse.Namespace(**{**vars(a), "runq": a.generic, "extra": ["-k", kv]})
            for name, chunk in KV_KERNELS:
                sub(bg, gen, os.path.join(tmp, f"kv_{kv}", "generic"), name, chunk, GEN_THREADS)
    names = [n for n, _ in golden_prompts()]
    if "kv" in only:
        for kv in KV_TYPES:
            v, d = argparse.Namespace(**{**vars(a), "extra": ["-k", kv]}), os.path.join(tmp, f"kv_{kv}")
            sub(fg, v, d, BATCH_LONG3, SEQ)
            for n in KV_LOGITS:
                sub(fg, v, d, n, BATCH)
            for n, c in KV_BATCHED + KV_KERNELS:
                sub(fg, v, d, n, SEQ)
                sub(fg, v, d, n, c)
            sub(fg, v, d, BATCH_LONG3, BATCH, THREADS_LONG if nd != THREADS_LONG else 8)
    if "batched" in only:  # (the longest runs first: sequential, logits of every row)
        for n in (BATCH_LONG3, BATCH_LONG2):
            sub(fg, a, tmp, n, SEQ)
    if "logits" in only:
        for n in names:
            sub(fg, a, tmp, n, SEQ)
            sub(fg, a, tmp, n, BATCH)
        sub(fg, a, tmp, BATCH_LONG3, BATCH)
    if "batched" in only:
        for n, _ in batched_prompts():
            sub(fg, a, tmp, n, SEQ)
            for c in batched_chunks(n):
                sub(fg, a, tmp, n, c)
    if "threads" in only:
        n_b = 1 if nd != 1 else 8
        for c in (SEQ, BATCH_CHUNKS[0]):
            sub(fg, a, tmp, THREADS_PROMPT, c)
            sub(fg, a, tmp, THREADS_PROMPT, c, n_b)
        sub(fg, a, tmp, BATCH_LONG2, BATCH)
        sub(fg, a, tmp, BATCH_LONG2, BATCH, THREADS_LONG if nd != THREADS_LONG else 8)
        sub(fg, a, tmp, BATCH_LONG3, BATCH)
        for t in LONG3_THREADS:
            sub(fg, a, tmp, BATCH_LONG3, BATCH, t if t != nd else 8)
    if "kernels" in only and a.generic:
        for n, c in KERNEL_CASES:
            sub(fg, a, tmp, n, c)
    if "gguf" in only and gg:
        gt = os.path.join(tmp, "gguf")
        for n in GGUF_SEQ:
            sub(fg, gg, gt, n, SEQ)
        for n in names:
            sub(fg, gg, gt, n, BATCH)
        for n, c in GGUF_BATCHED + GGUF_KERNELS:
            sub(fg, gg, gt, n, SEQ)
            sub(fg, gg, gt, n, c)
        sub(fg, gg, gt, THREADS_PROMPT, BATCH_CHUNKS[0], THREADS_LONG if nd != THREADS_LONG else 8)
    if "q4" in only and q4:
        qt = os.path.join(tmp, "q4")
        for n in names:
            sub(fg, q4, qt, n, BATCH)
        for n, c in Q4_BATCHED + Q4_KERNELS:
            sub(fg, q4, qt, n, SEQ)
            sub(fg, q4, qt, n, c)
        for n, c in Q4_THREADS:
            sub(fg, q4, qt, n, c, THREADS_LONG if nd != THREADS_LONG else 8)


API_THREADS = {"bin": 3, "gguf": 4, "q4": 4}  # the API test's runs (in the background from the start)
# the batched passes again (api_test --digest: their logits hashes only) at other thread counts (Q4_K_M, else
# the .bin; 8 and 16 threads: the short passes) and on the generic build (--digest-small, 3 threads: the short
# passes of the .bin and Q4_K_M; the long ones' generic attention: chat/long_context_generic): the same DIGEST
# lines. 65 sequences in one pass (--n65): Q4_K_M
# only; a bf16 KV cache (--kv): the .bin
API_DIGEST_THREADS = (1, 8, 16)
API_GENERIC_THREADS = 3


def api_runs(a, ex, gx):
    """Start tests/api_test (the library API: src/runq.h) on each model, and its digest runs; futures by tag
    (digest runs: (tag, label) -> future)."""
    if not a.api_test:
        return {}, {}
    models = [("bin", a.model), ("gguf", a.gguf), ("q4", a.q4)]
    n65 = {"q4": ["--n65"]} if a.q4 else {"bin": ["--n65"]}
    main = {tag: ex.submit(run, [a.api_test, m] + n65.get(tag, []) + (["--kv"] if tag == "bin" else []), API_THREADS[tag])
            for tag, m in models if m}
    dg = {}
    gen = a.api_test + "-generic"
    if os.path.exists(gen):
        for tag, m in models:
            if m and tag != "gguf":
                dg[(tag, "generic")] = gx.submit(run, [gen, m, "--digest-small"], API_GENERIC_THREADS)
    tag, m = ("q4", a.q4) if a.q4 else ("bin", a.model)
    for t in API_DIGEST_THREADS:
        mode = ["--digest"] + n65.get(tag, []) if t == 1 else ["--digest-small"]
        dg[(tag, f"t{t}")] = ex.submit(run, [a.api_test, m] + mode, t)
    return main, dg


def digests(p):
    """DIGEST name hash lines of an api_test run -> dict"""
    return dict(l.split()[1:3] for l in p.stdout.decode(errors="replace").splitlines() if l.startswith("DIGEST "))


def test_api(a, res, runs):
    """tests/api_test.c per model: sequences interleaved / batched in one pass / truncated and
    re-prefilled give bit-identical logits, rq_generate == a hand-written loop, error codes, chat
    rendering, incremental detokenization, the sampler's token mask; batched decode rows == each
    sequence alone, mixed passes, batched sampling; the batched passes' digests equal at other thread
    counts and on the generic build"""
    futs, dfuts = runs
    if not futs:
        print("   (skipped: no --api-test binary)")
    ref = {}
    for tag, fut in futs.items():
        p = fut.result()
        ref[tag] = digests(p)
        lines = [l for l in p.stdout.decode(errors="replace").splitlines() if l.startswith(("PASS ", "FAIL "))]
        for l in lines:
            ok, rest = l.startswith("PASS "), l[5:]
            name, _, detail = rest.partition("  ")
            res.report(ok, f"api_{tag}/{name.strip()}", detail.strip())
        if p.returncode not in (0, 1) or not lines:
            res.report(False, f"api_{tag}/run", f"exit status {p.returncode}: {p.stderr.decode(errors='replace')[-300:]}")
    for (tag, label), fut in dfuts.items():
        p, r = fut.result(), ref.get(tag, {})
        d = digests(p)
        common = [n for n in d if n in r]
        bad = [n for n in common if d[n] != r[n]]
        ok = p.returncode == 0 and len(common) >= 2 and not bad
        what = "the generic build" if label == "generic" else f"{label[1:]} thread(s)"
        res.report(ok, f"api_{tag}/batch_digest_{label}",
                   f"{len(common)} batched passes on {what} == {API_THREADS[tag]} threads" + (f"; differ: {', '.join(bad)}" if bad
                   else "" if p.returncode == 0 else f"; exit status {p.returncode}"))


SERVER_THREADS = 2  # runq serve and its CLI reference runs (in the background from the start)


def server_run(a, ex):
    """Start tests/server_test.py (runq serve over HTTP) on the GGUF Q8_0 file (else the .bin)."""
    model = a.gguf or a.model
    return ex.submit(run, [sys.executable, "-I", os.path.join(ROOT, "tests", "server_test.py"), "--runq", a.runq,
                           "--model", model, "--threads", str(SERVER_THREADS)])


def test_server(a, res, fut):
    """tests/server_test.py: runq serve's endpoints, greedy / seeded completions == runq -m chat, prefix
    reuse == fresh prefill (bitwise logprobs), streaming, constrained json_schema outputs, errors, queue,
    cancellation, fuzzing, shutdown"""
    p = fut.result()
    lines = [l for l in p.stdout.decode(errors="replace").splitlines() if l.startswith(("PASS ", "FAIL "))]
    for l in lines:
        ok, rest = l.startswith("PASS "), l[5:]
        name, _, detail = rest.partition("  ")
        res.report(ok, f"server/{name.strip()}", detail.strip())
    if p.returncode not in (0, 1) or not lines:
        res.report(False, "server/run", f"exit status {p.returncode}: {(p.stdout + p.stderr).decode(errors='replace')[-500:]}")


def read_logits(path):
    raw = np.fromfile(path, dtype=np.int32, count=2)
    n = int(raw[0])
    return np.fromfile(path, dtype=np.int32, count=2 + n)[2:], \
        np.fromfile(path, dtype=np.float32, offset=4 * (2 + n)).reshape(n, int(raw[1]))


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


# ----------------------------------------------------------------------------- tokenizer

def test_tokenizer(a, res, prefix="tokenizer", golden="tokenizer.json"):
    with open(os.path.join(GOLDEN, golden), encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        outs = list(ex.map(lambda c: runq_tokenize(a, c["text"]), cases))
    for c, got in zip(cases, outs):
        want = c["ids"]
        ok = got == want
        detail = ""
        if not ok:
            i = first_diff(got, want)
            detail = (f"text={c['text']!r} first diff at index {i}\n"
                      f"      hf  : {want[max(0, i - 3):i + 6]}\n      runq: {got[max(0, i - 3):i + 6]}")
        res.report(ok, f"{prefix}/{c['name']}", detail, c.get("xfail"))
        if not ok and c.get("xfail") and a.verbose:
            print("      " + detail)

    # invalid UTF-8 (no HF reference: a Python str cannot hold it). runq's rule: each invalid byte
    # is kept as is and the NFC normalizer does not reorder or compose across it (it acts like
    # U+FFFD, a starter that composes with nothing); the concatenated token bytes must be the
    # valid runs NFC-normalized separately, with the invalid bytes in between
    def nfc_runs(b):
        out, cur = b"", ""
        for ch in b.decode("utf-8", "surrogateescape") + "\udc00":  # one escape per invalid byte
            if "\udc00" <= ch <= "\udcff":
                out += unicodedata.normalize("NFC", cur).encode() + bytes([ord(ch) - 0xdc00] if ch > "\udc00" else [])
                cur = ""
            else:
                cur += ch
        return out
    vocab = load_vocab(a.vocab_model)
    bad = []
    for b in [b"cafe\xff\xcc\x81 e\xcc\x81\xff", b"q\xcc\x87\xcc\xa3\xc0\xafa\xcc\x8a\x80\xcc\x81",
              b"\xed\xa0\x80\xcc\x81\xe2\x84\xab\xfe", b"\xe1\x84\x80\xff\xe1\x85\xa1 \xe1\x84\x80\xe1\x85\xa1\xe2\x82",
              b"\xf0\x9d\x85\xa0\xf8\xcc\xa3\xcc\x87 <|im_end|>\xcc\x81\xc3", b"\xe2\x82\xe0\xa4\x95\xe0\xa4\xbc\xf4\x90\x80\x80"]:
        got = b"".join(vocab[i] for i in runq_tokenize(a, b))
        if got != nfc_runs(b):
            bad.append(f"{b!r}: runq {got!r} want {nfc_runs(b)!r}")
    res.report(not bad, f"{prefix}/nfc_invalid_utf8", "\n      ".join(bad))

    # speed: ~2,000 tokens of varied text (the short golden cases, repeated), and one long
    # pre-tokenizer chunk (no spaces); the encode time is measured inside runq, without the load
    pool = [c for c in cases if len(c["text"]) < 1000 and not c.get("xfail")]
    parts, n = [], 0
    while n < 2000:
        for c in pool:
            parts.append(c["text"])
            n += len(c["ids"])
            if n >= 2000:
                break
    for name, text in [("speed_2k", "\n".join(parts)), ("speed_long_chunk", "Supercalifragilistic" * 1500)]:
        ids, ms, load_ms = runq_tokenize(a, text, timing=True)
        res.report(ms < TOKENIZE_MAX_MS, f"{prefix}/{name}",
                   f"{len(text.encode())} bytes -> {len(ids)} tokens: encode {ms:.2f} ms "
                   f"(max {TOKENIZE_MAX_MS:.0f}), tokenizer load {load_ms:.1f} ms")


# ----------------------------------------------------------------------------- logits

def logit_metrics(g, logits):
    """Per-position metrics of runq logits (n, vocab) vs the golden npz g."""
    sub_ids, sub, top_ids, top_logits = g["subset_ids"], g["subset"], g["top_ids"], g["top_logits"]
    cos, top1, top5 = [], [], []
    for i in range(len(g["ids"])):
        ids = np.concatenate([top_ids[i], sub_ids])
        ref = np.concatenate([top_logits[i], sub[i].astype(np.float32)])
        ids, first = np.unique(ids, return_index=True)  # keep the fp32 top-64 value for duplicates
        ref = ref[first].astype(np.float64)
        got = logits[i, ids].astype(np.float64)
        cos.append(float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref))))
        top1.append(int(ids[np.argmax(got)]) == int(top_ids[i, 0]))
        rq5 = set(ids[np.argsort(-got, kind="stable")[:5]].tolist())
        top5.append(len(rq5 & set(top_ids[i, :5].tolist())) / 5)
    return np.array(cos), np.array(top1), np.array(top5)


def golden_prompts(sub="logits"):
    d = os.path.join(GOLDEN, sub)
    order = ["en", "code", "zh", "chat", "count"]
    names = sorted((f[:-4] for f in os.listdir(d) if f.endswith(".npz")),
                   key=lambda n: (order.index(n) if n in order else len(order), n))
    return [(n, os.path.join(d, n + ".npz")) for n in names]


def test_logits(a, res, tmp, chunk=SEQ, prefix="logits", only=None, th=TH_Q8, sub="logits", pooled=True):
    all_top1, all_cos, rows = [], [], []
    for name, path in golden_prompts(sub):
        if only and name not in only:
            continue
        g = np.load(path)
        text = str(g["text"])
        ids, logits = read_logits(logits_file(a, tmp, name, text, chunk))
        tname = f"{prefix}/{name}"
        if ids.tolist() != g["ids"].tolist():
            i = first_diff(ids.tolist(), g["ids"].tolist())
            res.report(False, tname, f"token ids differ from HF at index {i} (tokenizer bug?), cannot compare\n"
                       f"      hf  : {g['ids'].tolist()}\n      runq: {ids.tolist()}")
            continue
        if not np.isfinite(logits).all():
            bad = np.where(~np.isfinite(logits).all(axis=1))[0].tolist()
            res.report(False, tname, f"non-finite logits at positions {bad}")
            continue
        cos, top1, top5 = logit_metrics(g, logits)
        all_top1.extend(top1.tolist())
        c0, cr = cos[0], cos[1:]
        all_cos.extend(cr.tolist())
        rows.append((name, len(ids), c0, cr.mean(), cr.min(), int(top1.sum()), top5.mean()))
        why = []  # written as `not (x >= t)` so that NaN fails
        if not cr.mean() >= th["cos_mean"]: why.append(f"cos mean < {th['cos_mean']}")
        if not cr.min() >= th["cos_pos"]: why.append(f"cos min < {th['cos_pos']} at pos {1 + int(cr.argmin())}")
        if not c0 >= th["cos_pos0"]: why.append(f"cos pos0 < {th['cos_pos0']}")
        if not top1.mean() >= th["top1"]: why.append(f"top1 < {th['top1']}")
        if not top5.mean() >= th["top5"]: why.append(f"top5 < {th['top5']}")
        detail = (f"n={len(ids)} cos pos0 {c0:.4f}, pos1+ mean {cr.mean():.5f} min {cr.min():.5f} | "
                  f"top1 {int(top1.sum())}/{len(top1)} | top5 {top5.mean():.3f}")
        res.report(not why, tname, detail + ("  <- " + "; ".join(why) if why else ""))
        if a.verbose:
            for i in range(len(cos)):
                print(f"      pos {i:2d} cos {cos[i]:.5f} top1 {int(top1[i])} top5 {top5[i]:.1f}")
    if all_cos and pooled:
        c = float(np.mean(all_cos))
        res.report(bool(c >= th["cos_all"]), f"{prefix}/cos_pooled",
                   f"mean cos over {len(all_cos)} positions (pos >= 1) = {c:.5f} (min {th['cos_all']})")
    if all_top1 and pooled:
        t = float(np.mean(all_top1))
        res.report(bool(t >= th["top1_all"]), f"{prefix}/top1_pooled",
                   f"{int(np.sum(all_top1))}/{len(all_top1)} = {t:.3f} (min {th['top1_all']})")
    if a.verbose and rows:
        print("      prompt  n   cos pos0  cos mean  cos min   top1    top5")
        for r in rows:
            print(f"      {r[0]:<7} {r[1]:<3} {r[2]:.4f}    {r[3]:.5f}   {r[4]:.5f}   {r[5]:>2}/{r[1]:<4} {r[6]:.3f}")


def test_logits_long(a, res, tmp, prefix="logits_long"):
    # positions past the first attention block (256): the block partial results and their
    # combination vs HF fp32 (golden/logits_long: every 4th position from 248 of long2), Q8_0 thresholds
    g = np.load(os.path.join(GOLDEN, "logits_long", f"{BATCH_LONG3}.npz"))
    text = dict(batched_prompts())[BATCH_LONG3]
    tname = f"{prefix}/{BATCH_LONG3}"
    if str(g["text"]) != text:
        res.report(False, tname, "the golden's text is not check.py's long2 prompt (make_golden.py --long)")
        return
    ids, logits = read_logits(logits_file(a, tmp, BATCH_LONG3, text, BATCH))
    if ids.tolist() != g["ids"].tolist():
        res.report(False, tname, f"token ids differ from HF at index {first_diff(ids.tolist(), g['ids'].tolist())}")
        return
    pos = g["positions"]
    cos, top1, top5 = logit_metrics({**{k: g[k] for k in g.files}, "ids": pos}, logits[pos])
    why = []
    if not cos.mean() >= COS_MEAN_MIN: why.append(f"cos mean < {COS_MEAN_MIN}")
    if not cos.min() >= COS_POS_MIN: why.append(f"cos min < {COS_POS_MIN} at pos {int(pos[cos.argmin()])}")
    if not top1.mean() >= TOP1_MIN: why.append(f"top1 < {TOP1_MIN}")
    if not top5.mean() >= TOP5_MIN: why.append(f"top5 < {TOP5_MIN}")
    res.report(not why, tname, f"n={len(ids)}, positions {int(pos[0])}..{int(pos[-1])} step 4 ({len(pos)}): cos mean "
               f"{cos.mean():.5f} min {cos.min():.5f} | top1 {int(top1.sum())}/{len(pos)} | top5 {top5.mean():.3f}"
               + ("  <- " + "; ".join(why) if why else ""))


def batched_prompts():
    texts = [(name, str(np.load(path)["text"])) for name, path in golden_prompts()]
    all_text = " ".join(t for _, t in texts)
    long_text = all_text + " " + all_text + " " + texts[0][1]
    return texts + [(BATCH_LONG, all_text), (BATCH_LONG2, long_text),
                    (BATCH_LONG3, long_text + " " + all_text + " " + texts[1][1])]


def batched_chunks(name):
    # -b values compared with -b 1: 7 and 64 on the short prompts, 64 on all / long (2 / 3 chunks),
    # 64 and 512 on long2 (more than one attention block)
    return LONG3_CHUNKS if name == BATCH_LONG3 else [BATCH] if name in (BATCH_LONG, BATCH_LONG2) else BATCH_CHUNKS


def test_batched(a, res, tmp):
    n_same = n_all = 0
    for name, text in batched_prompts():
        ids_s, seq = read_logits(logits_file(a, tmp, name, text, SEQ))
        for chunk in batched_chunks(name):
            n_all += 1
            tname = f"batched/{name}_b{chunk}"
            ids_b, bat = read_logits(logits_file(a, tmp, name, text, chunk))
            if ids_b.tolist() != ids_s.tolist() or bat.shape != seq.shape:
                res.report(False, tname, "token ids / shape differ from the sequential run")
                continue
            if not np.isfinite(bat).all():
                bad = np.where(~np.isfinite(bat).all(axis=1))[0].tolist()
                res.report(False, tname, f"non-finite logits at positions {bad}")
                continue
            same = np.array_equal(bat.view(np.uint32), seq.view(np.uint32))
            n_same += same
            ok = True
            if not same:  # (the float64 metrics take ~0.5 s on the long prompt: only if needed)
                x, y = seq.astype(np.float64), bat.astype(np.float64)
                d = np.abs(x - y).max(axis=1)
                cos = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
                ok = bool(d.max() <= BATCH_MAX_ABS and cos.min() >= BATCH_COS_MIN)
            n_chunks = -(-len(ids_s) // chunk)
            detail = f"n={len(ids_s)}, {n_chunks} chunk(s): " + ("bit-identical" if same else
                     f"max |diff| {d.max():.3g} (pos {int(d.argmax())}), min cos {cos.min():.7f}")
            if not ok:
                detail += f"  <- max {BATCH_MAX_ABS}, cos {BATCH_COS_MIN}"
            res.report(ok, tname, detail)
    print(f"      {n_same}/{n_all} batched runs bit-identical to the sequential path")


def test_threads(a, res, tmp):
    g = np.load(os.path.join(GOLDEN, "logits", THREADS_PROMPT + ".npz"))
    text = str(g["text"])
    n_a = int(os.environ.get("OMP_NUM_THREADS", "8"))
    n_b = 1 if n_a != 1 else 8
    for chunk, tname in [(SEQ, "threads/bitwise"), (BATCH_CHUNKS[0], f"threads/bitwise_b{BATCH_CHUNKS[0]}")]:
        pa = logits_file(a, tmp, THREADS_PROMPT, text, chunk, threads=n_a)
        pb = logits_file(a, tmp, THREADS_PROMPT, text, chunk, threads=n_b)
        with open(pa, "rb") as fa, open(pb, "rb") as fb:
            same = fa.read() == fb.read()
        detail = f"{THREADS_PROMPT} -b {chunk}: OMP_NUM_THREADS={n_a} vs {n_b} " + ("bit-identical" if same else "DIFFER")
        if not same:
            xa, xb = read_logits(pa)[1], read_logits(pb)[1]
            detail += f" (max abs diff {np.abs(xa - xb).max():.3g})"
        res.report(same, tname, detail)
    # long prompt, batched: attention work items (dynamic schedule) over other thread counts
    text = dict(batched_prompts())[BATCH_LONG2]
    n_c = THREADS_LONG if n_a != THREADS_LONG else 8
    pa = logits_file(a, tmp, BATCH_LONG2, text, BATCH, threads=n_a)
    pb = logits_file(a, tmp, BATCH_LONG2, text, BATCH, threads=n_c)
    with open(pa, "rb") as fa, open(pb, "rb") as fb:
        same = fa.read() == fb.read()
    res.report(same, f"threads/bitwise_{BATCH_LONG2}_b{BATCH}",
               f"{BATCH_LONG2} -b {BATCH}: OMP_NUM_THREADS={n_a} vs {n_c} " + ("bit-identical" if same else "DIFFER"))
    # more than one attention block (prefill: a vector's blocks in one item, then combined)
    text = dict(batched_prompts())[BATCH_LONG3]
    pa = logits_file(a, tmp, BATCH_LONG3, text, BATCH, threads=n_a)
    for t in LONG3_THREADS:
        t = t if t != n_a else 8
        same, detail = same_files(pa, logits_file(a, tmp, BATCH_LONG3, text, BATCH, threads=t))
        res.report(same, f"threads/bitwise_{BATCH_LONG3}_b{BATCH}_t{t}", f"{BATCH_LONG3} -b {BATCH}: OMP_NUM_THREADS={n_a} vs {t} {detail}")


# ----------------------------------------------------------------------------- kernels

def test_kernels(a, res, tmp):
    if not a.generic:
        print("      skipped: no generic build (--generic / RUNQ_GENERIC)")
        return
    texts = dict(batched_prompts())
    gen = argparse.Namespace(**{**vars(a), "runq": a.generic})
    gtmp = os.path.join(tmp, "generic")
    os.makedirs(gtmp, exist_ok=True)
    for name, chunk in KERNEL_CASES:
        pv = logits_file(a, tmp, name, texts[name], chunk)    # VNNI build (shared cache)
        pg = logits_file(gen, gtmp, name, texts[name], chunk, GEN_THREADS)
        with open(pv, "rb") as fv, open(pg, "rb") as fg:
            same = fv.read() == fg.read()
        n = len(read_logits(pv)[0])
        detail = f"{name} (n={n}) -b {chunk}: " + ("bit-identical" if same else "DIFFER")
        if not same:
            xv, xg = read_logits(pv)[1], read_logits(pg)[1]
            if xv.shape == xg.shape:
                d = np.abs(xv.astype(np.float64) - xg)
                detail += f" (max abs diff {d.max():.3g} at pos {int(d.max(axis=1).argmax())})"
        res.report(same, f"kernels/{name}_b{chunk}", detail)


# ----------------------------------------------------------------------------- greedy

def greedy_output(a, args, model=None):
    p = run([a.runq, model or a.model, "-t", "0", "-c", CTX, "-s", "1"] + args)
    if p.returncode != 0:
        raise RuntimeError(f"runq failed: {p.stderr.decode(errors='replace')[-500:]}")
    return p.stdout.decode("utf-8", errors="replace")


def greedy_start(a, ex):
    """the greedy / seeded runs, on ex (from the beginning of the suite; outputs do not depend on the thread
    count or timing): futures for test_greedy"""
    outs = {}
    for name, args in GREEDY_CASES:
        if (a.update_greedy and "@" in name) or (name.startswith("gguf_") and not a.gguf) or (name.startswith("q4_") and not a.q4):
            continue
        outs[name] = ex.submit(greedy_output, a, args, a.gguf if name.startswith("gguf_") else a.q4 if name.startswith("q4_") else None)
    seeded = {c: ex.submit(run, [a.runq, a.model, "-b", str(c)] + SEEDED_ARGS)
              for c in (SEEDED_CHUNKS[:1] if a.update_greedy else SEEDED_CHUNKS)}
    seeded_gen = ex.submit(run, [a.runq, a.model] + SEEDED_GEN_ARGS)
    return outs, seeded, seeded_gen


def test_greedy(a, res, futs):
    d = os.path.join(GOLDEN, "greedy")
    os.makedirs(d, exist_ok=True)
    outs, seeded, seeded_gen = futs
    for name, args in GREEDY_CASES:
        path = os.path.join(d, name.split("@")[0] + ".txt")  # name@variant: same golden
        if a.update_greedy and "@" in name:
            continue
        if name.startswith("gguf_") and not a.gguf:
            print(f"      skipped greedy/{name}: no GGUF file (--gguf)")
            continue
        if name.startswith("q4_") and not a.q4:
            print(f"      skipped greedy/{name}: no Q4_K_M file (--q4)")
            continue
        got = outs[name].result()
        if a.update_greedy:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(got)
            print(f"UPDATED greedy/{name} -> {os.path.relpath(path, ROOT)}")
            continue
        if not os.path.exists(path):
            res.report(False, f"greedy/{name}", "no golden; run tests/run_tests.sh --update-greedy")
            continue
        with open(path, encoding="utf-8", newline="") as f:
            want = f.read()
        if got == want:
            res.report(True, f"greedy/{name}")
        else:
            i = first_diff(got, want)
            res.report(False, f"greedy/{name}",
                       f"REGRESSION (greedy): output differs at char {i}: "
                       f"golden {want[i:i + 20]!r} vs runq {got[i:i + 20]!r}\n"
                       f"      --- golden ---\n{want}\n      --- runq ---\n{got}\n"
                       f"      (if the numerics changed on purpose and logits/* pass: "
                       f"tests/run_tests.sh --update-greedy)")
    outs = {}
    for chunk in SEEDED_CHUNKS[:1] if a.update_greedy else SEEDED_CHUNKS:
        p = seeded[chunk].result()
        if p.returncode != 0:
            raise RuntimeError(f"runq failed: {p.stderr.decode(errors='replace')[-500:]}")
        outs[chunk] = p.stdout
    ref = outs[SEEDED_CHUNKS[0]]
    if not a.update_greedy:
        bad = [c for c in SEEDED_CHUNKS if outs[c] != ref]
        res.report(not bad, "greedy/seeded_chunks",
                   f"chat -t 0.8 -s 42, -b {'/'.join(map(str, SEEDED_CHUNKS))}: " +
                   ("identical" if not bad else f"-b {bad} differ from -b {SEEDED_CHUNKS[0]}"))
    p = seeded_gen.result()
    if p.returncode != 0:
        raise RuntimeError(f"runq failed: {p.stderr.decode(errors='replace')[-500:]}")
    for name, out, what in [("seeded_chat", ref, "chat -t 0.8 -p 0.9 -s 42"),
                            ("seeded_generate", p.stdout, "generate -t 1.5 -p 0.95 -s 7")]:
        path = os.path.join(d, name + ".txt")
        got = out.decode("utf-8", errors="replace")
        if a.update_greedy:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(got)
            print(f"UPDATED greedy/{name} -> {os.path.relpath(path, ROOT)}")
            continue
        want = open(path, encoding="utf-8", newline="").read() if os.path.exists(path) else None
        if got == want:
            res.report(True, f"greedy/{name}", what)
        else:
            i = first_diff(got, want or "")
            res.report(False, f"greedy/{name}", f"{what}: " + ("no golden; run tests/run_tests.sh --update-greedy"
                       if want is None else f"REGRESSION (sampling): differs at char {i}: golden "
                       f"{want[i:i + 20]!r} vs runq {got[i:i + 20]!r}\n      --- runq ---\n{got}"))


# ----------------------------------------------------------------------------- chat (multi-turn)

def gguf_vocab(path):
    """token bytes from a GGUF's tokenizer metadata: normal tokens through GPT-2's byte-to-unicode map, control /
    user-defined (added) tokens as their UTF-8 text, unused ones empty"""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:] + [256 + i for i in range(256 - len(bs))]
    bs += [b for b in range(256) if b not in bs]
    dec = {chr(c): b for b, c in zip(bs, cs)}
    f = check_models.G.GgufFile(path)
    out = []
    for t, ty in zip(f.kv["tokenizer.ggml.tokens"], f.kv["tokenizer.ggml.token_type"]):
        s = t.decode("utf-8")
        out.append(bytes(dec[ch] for ch in s) if ty == 1 else s.encode() if ty in (3, 4) else b"")
    return out


def load_vocab(model):
    """token bytes from the .tokenizer file (header: 3 x u32; per token: f32 score, i32 len, bytes) of
    the .bin model (also for GGUF runs: its tokens decode to the same bytes); a .gguf path: its metadata"""
    if model.endswith(".gguf"):
        return gguf_vocab(model)
    with open(model + ".tokenizer", "rb") as f:
        data = f.read()
    vocab, off = [], 12
    while off + 8 <= len(data):
        n = int.from_bytes(data[off + 4:off + 8], "little", signed=True)
        vocab.append(data[off + 8:off + 8 + n])
        off += 8 + n
    return vocab


def chat_expected(system, users, replies, thinking):
    """the conversation as runq should have it in the KV cache (Qwen3 chat template, ChatML): the
    last reply as generated, earlier ones with -r 1 as the template renders them (no reasoning),
    with -r 0 after the empty think block of the generation prompt"""
    gen = "" if thinking else "<think>\n\n</think>\n\n"
    s = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    for i, (u, r) in enumerate(zip(users, replies)):
        if thinking and i < len(users) - 1 and "</think>" in r:
            r = r.split("</think>")[-1].lstrip("\n")
        s += f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{gen}{r}"
        if i < len(users) - 1:
            s += "<|im_end|>\n"
    return s


def split_transcript(out, users):
    """replies from a piped chat transcript: per turn '\\n> ' + user + '\\n' + reply + '\\n'"""
    replies, at = [], 0
    marks = []
    for u in users:
        m = out.index(f"\n> {u}\n", at)
        marks.append((m, m + len(u) + 4))
        at = m + len(u) + 4
    for i, (m, e) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else out.rindex("\n> ")
        replies.append(out[e:end - 1] if out[end - 1] == "\n" else out[e:end])
    return replies


def chat_case(a, tmp, vocab, d, name, args, users, chk):
    """one CHAT_CASES case; returns its reports [(ok, name, detail)] (cases run in parallel)"""
    reports = []
    rep = lambda ok, n, detail="": reports.append((ok, n, detail))
    args = [LONG_SYSTEM if x == "LONG_SYSTEM" else x for x in args]
    def chat_run(runq, prefix, threads):
        env = dict(os.environ, RUNQ_CHAT_DUMP=prefix)
        if threads:
            env["OMP_NUM_THREADS"] = str(threads)
        return subprocess.run([runq, a.model, "-t", "0", "-c", CTX, "-s", "1"] + args,
                              input=("\n".join(users) + "\n").encode(), env=env, capture_output=True)
    tname = f"chat/{name}"
    prefix = os.path.join(tmp, f"chat_{name}")
    gex = gfut = None
    if chk.get("generic") and a.generic and not a.update_greedy:
        gex = cf.ThreadPoolExecutor(max_workers=1)
        gfut = gex.submit(chat_run, a.generic, prefix + "_generic", GEN_THREADS)
    try:
        return chat_case_check(a, tmp, vocab, d, name, args, users, chk, tname, prefix, chat_run, gfut, reports, rep)
    finally:
        if gex:
            gex.shutdown(wait=True)


def chat_case_check(a, tmp, vocab, d, name, args, users, chk, tname, prefix, chat_run, gfut, reports, rep):
    p = chat_run(a.runq, prefix, chk.get("threads"))
    if p.returncode != 0:
        rep(False, tname, f"runq failed: {p.stderr.decode(errors='replace')[-500:]}")
        return reports
    out, err = p.stdout.decode("utf-8", errors="replace"), p.stderr.decode(errors="replace")
    if chk.get("golden"):
        path = os.path.join(d, f"chat_{name}.txt")
        if a.update_greedy:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(out)
            print(f"UPDATED {tname} -> {os.path.relpath(path, ROOT)}")
            return reports
        want = open(path, encoding="utf-8", newline="").read() if os.path.exists(path) else None
        if want is None:
            rep(False, f"{tname}_golden", "no golden; run tests/run_tests.sh --update-greedy")
        elif out != want:
            i = first_diff(out, want)
            rep(False, f"{tname}_golden", f"REGRESSION (greedy): transcript differs at char {i}: "
                       f"golden {want[i:i + 30]!r} vs runq {out[i:i + 30]!r}\n      --- runq ---\n{out}")
        else:
            rep(True, f"{tname}_golden", f"{len(users)} turns")
    elif a.update_greedy:
        return reports
    try:
        replies = split_transcript(out, users)
    except ValueError:
        rep(False, tname, f"cannot split the transcript into {len(users)} turns:\n{out}")
        return reports
    if "mentions" in chk:
        t, words = chk["mentions"]
        miss = [w for w in words if w not in replies[t].lower()]
        rep(not miss, f"{tname}_memory", f"turn {t + 1} reply {replies[t]!r}" +
                   (f"  <- missing {miss}" if miss else ""))
    # the KV cache: expected rendering, and logits == a fresh run on the whole conversation
    ids = [int(x) for x in open(prefix + ".ids").read().split()]
    text = b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
    system = args[args.index("-y") + 1] if "-y" in args else None
    n_kept = text.count("<|im_start|>user\n")
    why = []
    if chk.get("dropped"):
        if "[context] full" not in err or n_kept >= len(users):
            why.append("no turns dropped (context full not reached?)")
        c = int(args[args.index("-c") + 1])
        if len(ids) > c:
            why.append(f"{len(ids)} tokens in a {c}-token context")
    if "stripped" in chk and "</think>" not in replies[chk["stripped"] - 1]:
        why.append(f"turn {chk['stripped']} has no complete reasoning block (raise -n?)")
    expect = chat_expected(system, users[-n_kept:], replies[-n_kept:], "-r" in args and args[args.index("-r") + 1] == "1")
    if text != expect:
        i = first_diff(text, expect)
        why.append(f"KV cache conversation differs from the template rendering at char {i}: "
                   f"{text[max(0, i - 20):i + 30]!r} vs expected {expect[max(0, i - 20):i + 30]!r}")
    raw = np.fromfile(prefix + ".logits", dtype=np.int32, count=2)
    n_rec, nv = int(raw[0]), int(raw[1])
    pos = np.fromfile(prefix + ".logits", dtype=np.int32, count=2 + n_rec)[2:]
    rec = np.fromfile(prefix + ".logits", dtype=np.float32, offset=4 * (2 + n_rec)).reshape(n_rec, nv)
    lpath = os.path.join(tmp, f"chat_{name}_full.bin")
    p = run([a.runq, a.model, "-c", CTX, "-m", "logits", "-i", text, "-o", lpath])
    if p.returncode != 0:
        rep(False, tname, f"runq -m logits failed: {p.stderr.decode(errors='replace')[-300:]}")
        return reports
    fids, full = read_logits(lpath)
    detail = f"{len(users)} turns, {n_kept} in the cache, {len(ids)} tokens, {n_rec} sampled positions"
    if fids.tolist() != ids:
        i = first_diff(fids.tolist(), ids)
        why.append(f"token ids of the re-encoded conversation differ at {i}: {fids[i:i + 5].tolist()} vs {ids[i:i + 5]}")
    else:
        same = np.array_equal(full[pos].view(np.uint32), rec.view(np.uint32))
        if not same:
            dd = np.abs(full[pos].astype(np.float64) - rec).max(axis=1)
            why.append(f"logits differ from the single-shot run: max |diff| {dd.max():.3g} at pos {int(pos[dd.argmax()])}")
        else:
            detail += ": logits bit-identical to -m logits on the whole conversation"
    rep(not why, tname, detail + ("  <- " + "; ".join(why) if why else ""))
    if gfut is not None:
        # the generic build (C attention, generic kernels) on the same conversation: the same cache
        # ids and the same sampled logits, bit for bit
        gp = gfut.result()
        ok = gp.returncode == 0 and open(prefix + ".ids").read() == open(prefix + "_generic.ids").read()
        with open(prefix + ".logits", "rb") as fa, open(prefix + "_generic.logits", "rb") as fb:
            ok = ok and fa.read() == fb.read()
        rep(ok, f"{tname}_generic", f"generic build ({GEN_THREADS} threads): cache ids and sampled logits " +
            ("bit-identical" if ok else "DIFFER"))
    return reports


def chat_start(a, tmp, ex):
    """start the CHAT_CASES on ex (from the beginning of the suite: their runs only need the model)"""
    vocab = load_vocab(a.vocab_model)
    d = os.path.join(GOLDEN, "greedy")
    return [ex.submit(chat_case, a, tmp, vocab, d, *case) for case in CHAT_CASES]


def test_chat(a, res, futs):
    for f in futs:
        for ok, n, detail in f.result():
            res.report(ok, n, detail)


# ----------------------------------------------------------------------------- GGUF

def same_files(pa, pb):
    """(byte-identical?, detail) for two logits files"""
    with open(pa, "rb") as fa, open(pb, "rb") as fb:
        if fa.read() == fb.read():
            return True, "bit-identical"
    xa, xb = read_logits(pa)[1], read_logits(pb)[1]
    if xa.shape != xb.shape:
        return False, "DIFFER (shapes)"
    d = np.abs(xa.astype(np.float64) - xb)
    return False, f"DIFFER (max abs diff {d.max():.3g} at pos {int(d.max(axis=1).argmax())})"


def templates_start(a, tmp, ex):
    """chat templates: built into runq for GGUF, read from the .template* side files for the .bin. A 1-turn chat
    (-n 1) on both: the KV cache ids up to the reply (the rendered prompt) must be equal (same vocab, so the
    same bytes), and decode to the side file rendering (test_gguf). The runs, started on ex: futures by tag"""
    def chat_ids(model, tag, args):
        prefix = os.path.join(tmp, f"tmpl_{tag}")
        env = dict(os.environ, RUNQ_CHAT_DUMP=prefix, OMP_NUM_THREADS="2")
        p = subprocess.run([a.runq, model, "-t", "0", "-c", CTX, "-n", "1", "-i", "Hi there"] + args, env=env, capture_output=True)
        if p.returncode != 0:
            raise RuntimeError(f"runq chat failed: {p.stderr.decode(errors='replace')[-300:]}")
        return [int(x) for x in open(prefix + ".ids").read().split()][:-1]  # (minus the 1 reply token)
    return {f"{name}_{k}": ex.submit(chat_ids, m, f"{name}_{k}", args)
            for name, args, _ in GGUF_TEMPLATES for k, m in (("bin", a.model), ("gguf", a.gguf))}


def test_gguf(a, res, tmp, tmpl):
    if not a.gguf:
        print("      skipped: no GGUF file (--gguf)")
        return
    gg = argparse.Namespace(**{**vars(a), "model": a.gguf})
    gtmp = os.path.join(tmp, "gguf")
    os.makedirs(gtmp, exist_ok=True)
    test_tokenizer(gg, res, "gguf_tokenizer")
    test_logits(gg, res, gtmp, SEQ, "gguf_logits", only=GGUF_SEQ)
    test_logits(gg, res, gtmp, BATCH, "gguf_logits_batched")
    texts = dict(batched_prompts())
    for name, chunk in GGUF_BATCHED:
        same, detail = same_files(logits_file(gg, gtmp, name, texts[name], SEQ), logits_file(gg, gtmp, name, texts[name], chunk))
        res.report(same, f"gguf_batched/{name}_b{chunk}", f"-b {chunk} vs -b {SEQ}: {detail}")
    n_a = int(os.environ["OMP_NUM_THREADS"])
    n_b = THREADS_LONG if n_a != THREADS_LONG else 8
    chunk = BATCH_CHUNKS[0]
    same, detail = same_files(logits_file(gg, gtmp, THREADS_PROMPT, texts[THREADS_PROMPT], chunk, threads=n_a),
                              logits_file(gg, gtmp, THREADS_PROMPT, texts[THREADS_PROMPT], chunk, threads=n_b))
    res.report(same, f"gguf_threads/bitwise_b{chunk}", f"{THREADS_PROMPT} -b {chunk}: OMP_NUM_THREADS={n_a} vs {n_b} {detail}")
    if a.generic:
        gen = argparse.Namespace(**{**vars(gg), "runq": a.generic})
        gentmp = os.path.join(gtmp, "generic")
        os.makedirs(gentmp, exist_ok=True)
        for name, chunk in GGUF_KERNELS:
            same, detail = same_files(logits_file(gg, gtmp, name, texts[name], chunk),
                                      logits_file(gen, gentmp, name, texts[name], chunk, GEN_THREADS))
            res.report(same, f"gguf_kernels/{name}_b{chunk}", f"VNNI vs generic, -b {chunk}: {detail}")
    else:
        print("      gguf_kernels skipped: no generic build (--generic / RUNQ_GENERIC)")

    # chat templates (runs started early: templates_start)
    vocab = load_vocab(a.vocab_model)
    ids = {k: f.result() for k, f in tmpl.items()}
    for name, args, sfx in GGUF_TEMPLATES:
        with open(a.model + sfx, encoding="utf-8", newline="") as f:
            tmpl = f.read()
        want = tmpl.replace("%s", "You are terse.", 1).replace("%s", "Hi there", 1) if "-y" in args else tmpl.replace("%s", "Hi there")
        ib, ig = ids[f"{name}_bin"], ids[f"{name}_gguf"]
        text = b"".join(vocab[i] for i in ig).decode("utf-8", errors="replace")
        ok = ib == ig and text == want
        res.report(ok, f"gguf_template/{name}", f"{len(ig)} prompt tokens, " + (f"same ids as the .bin ({os.path.basename(a.model + sfx)})"
                   if ok else f"DIFFER: gguf {text!r} vs side file {want!r} (ids equal: {ib == ig})"))


# ----------------------------------------------------------------------------- Q4_K_M

def dequant_start(a, tmp, ex):
    """dequantization vs gguf-py, bitwise: rows from the file's blocks (embedding lookup) and from the repacked
    Q4P / Q6P layouts (what the matmul kernels read); the runq -m dequant runs, started on ex (test_q4 checks)"""
    qtmp = os.path.join(tmp, "q4")
    os.makedirs(qtmp, exist_ok=True)
    g = np.load(os.path.join(GOLDEN, "dequant", "q4_k_m.npz"))
    def dequant(i):
        name, rows = str(g["names"][i]), g[f"rows{i}"]
        path = os.path.join(qtmp, f"dequant{i}.bin")
        p = run([a.runq, a.q4, "-c", "64", "-m", "dequant", "-i", f"{name}:{int(g['row0'][i])}:{rows.shape[0]}", "-o", path], threads=2)
        if p.returncode != 0:
            return name, rows, None, p.stderr.decode(errors="replace")[-300:]
        h = np.fromfile(path, dtype=np.int32, count=3)
        return name, rows, np.fromfile(path, dtype=np.float32, offset=12).reshape(1 + int(h[2]), int(h[0]), int(h[1])), ""
    return [ex.submit(dequant, i) for i in range(len(g["names"]))]


def test_q4(a, res, tmp, deq):
    if not a.q4:
        print("      skipped: no Q4_K_M file (--q4)")
        return
    q4 = argparse.Namespace(**{**vars(a), "model": a.q4})
    qtmp = os.path.join(tmp, "q4")
    os.makedirs(qtmp, exist_ok=True)
    texts = dict(batched_prompts())
    # (the generic build's runs for q4_kernels are prefetched in the background: prefetch_all; the dequant
    # runs started early: dequant_start)
    g = np.load(os.path.join(GOLDEN, "dequant", "q4_k_m.npz"))
    outs = [f.result() for f in deq]
    for i, (name, rows, got, err) in enumerate(outs):
        what = f"{str(g[f'type{i}'])} rows {int(g['row0'][i])}..{int(g['row0'][i]) + rows.shape[0] - 1} x {rows.shape[1]}"
        if got is None:
            res.report(False, f"q4_dequant/{name}", f"runq -m dequant failed: {err}")
            continue
        ok = got.shape[1:] == rows.shape and got.shape[0] == 2
        detail = []
        for j, src in enumerate(["blocks", "repacked"][:got.shape[0]]):
            same = ok and np.array_equal(got[j].view(np.uint32), rows.view(np.uint32))
            ok &= same
            detail.append(f"{src} " + ("bit-identical" if same else
                          f"DIFFER (max |diff| {np.abs(got[j] - rows).max():.3g})" if got.shape[1:] == rows.shape else "DIFFER (shape)"))
        res.report(ok, f"q4_dequant/{name}", f"{what}: " + ", ".join(detail) + " (gguf-py)")

    # logits: vs HF fp32 (loose: the 4-bit weights' own error) and vs the Q4 reference (HF fp32 with
    # the dequantized Q4_K_M weights: runq's int8 activations only), batched on all prompts and
    # sequential on Q4_SEQ (the same files)
    test_logits(q4, res, qtmp, BATCH, "q4_logits_batched", th=TH_Q4_HF)
    test_logits(q4, res, qtmp, BATCH, "q4_ref_batched", th=TH_Q4_REF, sub="logits_q4")
    test_logits(q4, res, qtmp, SEQ, "q4_ref", only=Q4_SEQ, th=TH_Q4_REF, sub="logits_q4", pooled=False)

    for name, chunk in Q4_BATCHED:
        same, detail = same_files(logits_file(q4, qtmp, name, texts[name], SEQ), logits_file(q4, qtmp, name, texts[name], chunk))
        res.report(same, f"q4_batched/{name}_b{chunk}", f"-b {chunk} vs -b {SEQ}: {detail}")
    n_a = int(os.environ["OMP_NUM_THREADS"])
    n_b = THREADS_LONG if n_a != THREADS_LONG else 8
    for name, chunk in Q4_THREADS:
        same, detail = same_files(logits_file(q4, qtmp, name, texts[name], chunk, threads=n_a),
                                  logits_file(q4, qtmp, name, texts[name], chunk, threads=n_b))
        res.report(same, f"q4_threads/bitwise_{name}_b{chunk}", f"{name} -b {chunk}: OMP_NUM_THREADS={n_a} vs {n_b} {detail}")
    if a.generic:
        gen = argparse.Namespace(**{**vars(q4), "runq": a.generic})
        gentmp = os.path.join(qtmp, "generic")
        os.makedirs(gentmp, exist_ok=True)
        for name, chunk in Q4_KERNELS:
            pv = logits_file(q4, qtmp, name, texts[name], chunk)
            same, detail = same_files(pv, logits_file(gen, gentmp, name, texts[name], chunk, GEN_THREADS))
            res.report(same, f"q4_kernels/{name}_b{chunk}", f"SIMD (AVX-512 / NEON) kernels vs generic ({GEN_THREADS} threads), "
                       f"{name} (n={len(read_logits(pv)[0])}) -b {chunk}: {detail}")
    else:
        print("      q4_kernels skipped: no generic build (--generic / RUNQ_GENERIC)")


# ----------------------------------------------------------------------------- KV cache types

def test_kv(a, res, tmp):
    """the KV cache types other than the default (-k f32, -k bf16; Step 7b): logits vs HF with the
    Q8_0 thresholds, and the bitwise invariants (batched vs sequential, threads, AVX-512 vs generic)"""
    texts = dict(batched_prompts())
    n_a = int(os.environ["OMP_NUM_THREADS"])
    n_b = THREADS_LONG if n_a != THREADS_LONG else 8
    for kv in KV_TYPES:
        v, d = argparse.Namespace(**{**vars(a), "extra": ["-k", kv]}), os.path.join(tmp, f"kv_{kv}")
        os.makedirs(d, exist_ok=True)
        test_logits(v, res, d, BATCH, f"kv_{kv}_logits", only=KV_LOGITS)
        test_logits_long(v, res, d, f"kv_{kv}_logits_long")
        for name, chunk in KV_BATCHED:
            same, detail = same_files(logits_file(v, d, name, texts[name], SEQ), logits_file(v, d, name, texts[name], chunk))
            res.report(same, f"kv_{kv}_batched/{name}_b{chunk}", f"-k {kv} -b {chunk} vs -b {SEQ}: {detail}")
        same, detail = same_files(logits_file(v, d, BATCH_LONG3, texts[BATCH_LONG3], BATCH, threads=n_a),
                                  logits_file(v, d, BATCH_LONG3, texts[BATCH_LONG3], BATCH, threads=n_b))
        res.report(same, f"kv_{kv}_threads/bitwise_{BATCH_LONG3}_b{BATCH}", f"-k {kv}: OMP_NUM_THREADS={n_a} vs {n_b} {detail}")
        if a.generic:
            gen = argparse.Namespace(**{**vars(v), "runq": a.generic})
            gd = os.path.join(d, "generic")
            os.makedirs(gd, exist_ok=True)
            for name, chunk in KV_KERNELS:
                same, detail = same_files(logits_file(v, d, name, texts[name], chunk),
                                          logits_file(gen, gd, name, texts[name], chunk, GEN_THREADS))
                res.report(same, f"kv_{kv}_kernels/{name}_b{chunk}", f"-k {kv}, SIMD (AVX-512 / NEON) vs generic, -b {chunk}: {detail}")


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runq", default=os.path.join(ROOT, "runq"))
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "Qwen3-0.6B-q8.bin"))
    ap.add_argument("--generic", default="", help="generic-kernel build for the kernels test (empty: skip)")
    ap.add_argument("--gguf", default="", help="GGUF Q8_0 file for the gguf tests and greedy goldens (empty: skip)")
    ap.add_argument("--q4", default="", help="GGUF Q4_K_M file for the q4 tests and greedy goldens (empty: skip)")
    ap.add_argument("--api-test", default="", help="tests/api_test binary for the api group (empty: skip)")
    ap.add_argument("--q35", default="", help="Qwen3.5-0.8B Q8_0 GGUF for the q35 group (empty: skip)")
    ap.add_argument("--only", default="tokenizer,logits,batched,threads,kernels,greedy,chat,gguf,q4,kv,api,server,synth,q35",
                    help="comma-separated subset of tests to run (models: the bigger Qwen3 models, opt-in)")
    ap.add_argument("--update-models", action="store_true", help="write the models group's greedy goldens")
    ap.add_argument("--update-greedy", action="store_true", help="regenerate the greedy goldens")
    ap.add_argument("--jobs", type=int, default=8, help="parallel runq processes for the tokenizer test")
    ap.add_argument("-v", "--verbose", action="store_true", help="per-position logits metrics")
    a = ap.parse_args()
    a.vocab_model = a.model  # token bytes for decoding ids (.tokenizer file), also for GGUF runs
    os.environ.setdefault("OMP_NUM_THREADS", "8")

    t0 = time.time()
    res = Results()
    only = set(a.only.split(","))
    if a.update_greedy:
        only = {"greedy", "chat"}
    with tempfile.TemporaryDirectory(prefix="gwenedeg-test-") as tmp, \
            cf.ThreadPoolExecutor(max_workers=PREFETCH_JOBS) as fg, cf.ThreadPoolExecutor(max_workers=2) as bg, \
            cf.ThreadPoolExecutor(max_workers=2) as ax, cf.ThreadPoolExecutor(max_workers=1) as sx, \
            cf.ThreadPoolExecutor(max_workers=1) as gx, cf.ThreadPoolExecutor(max_workers=3) as cx, \
            cf.ThreadPoolExecutor(max_workers=2) as tx, cf.ThreadPoolExecutor(max_workers=2) as yx, \
            cf.ThreadPoolExecutor(max_workers=1) as qx:
        # started first, checked at their group's turn: the server and API tests, the greedy runs, the chat cases,
        # the GGUF template and Q4_K_M dequant runs (their processes then overlap the logits runs: the suite is
        # bound by CPU time, and in group order they ran with the CPUs half idle at the end, Step 10b)
        server = server_run(a, sx) if "server" in only and not a.update_greedy else None
        api = api_runs(a, ax, gx) if "api" in only else ({}, {})
        greedy = greedy_start(a, cx) if "greedy" in only else None
        chats = chat_start(a, tmp, cx) if "chat" in only else []
        tmpl = templates_start(a, tmp, tx) if "gguf" in only and a.gguf and not a.update_greedy else {}
        deq = dequant_start(a, tmp, tx) if "q4" in only and a.q4 and not a.update_greedy else []
        texts = dict(batched_prompts())
        synth = check_models.synth_start(a, tmp, yx, bg, texts) if "synth" in only and not a.update_greedy else None
        q35 = check_models.q35_start(a, tmp, cx, qx, dict(batched_prompts()), test_tokenizer, chat_case, load_vocab) \
            if "q35" in only and not a.update_greedy else None
        if not a.update_greedy:
            prefetch_all(a, tmp, only, fg, bg)
        for name, fn in [("tokenizer", lambda: test_tokenizer(a, res)),
                         ("logits", lambda: (test_logits(a, res, tmp),
                                             test_logits(a, res, tmp, BATCH, "logits_batched"),
                                             test_logits_long(a, res, tmp))),
                         ("batched", lambda: test_batched(a, res, tmp)),
                         ("threads", lambda: test_threads(a, res, tmp)),
                         ("kernels", lambda: test_kernels(a, res, tmp)),
                         ("greedy", lambda: test_greedy(a, res, greedy)),
                         ("chat", lambda: test_chat(a, res, chats)),
                         ("gguf", lambda: test_gguf(a, res, tmp, tmpl)),
                         ("q4", lambda: test_q4(a, res, tmp, deq)),
                         ("kv", lambda: test_kv(a, res, tmp)),
                         ("api", lambda: test_api(a, res, api)),
                         ("server", lambda: test_server(a, res, server)),
                         ("synth", lambda: check_models.test_synth(a, res, synth, texts)),
                         ("q35", lambda: check_models.test_q35(a, res, tmp, texts, q35)),
                         ("models", lambda: check_models.test_models(a, res, tmp, texts, a.update_models))]:
            if name in only:
                t = time.time()
                print(f"== {name}")
                try:
                    fn()
                except Exception as e:  # a crashing runq is a failure, not a traceback
                    res.report(False, f"{name}/error", f"{type(e).__name__}: {e}")
                print(f"   ({time.time() - t:.1f} s)")
        for f in PENDING.values():  # (runs nobody waited for, e.g. after a group that errored)
            f.cancel()
    dt = time.time() - t0
    print(f"\nSummary: {len(res.passed)} passed, {len(res.failed)} failed, "
          f"{len(res.xfailed)} xfail, {len(res.xpassed)} xpass, {dt:.1f} s "
          f"(OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']})")
    for n in res.failed:
        print(f"  FAILED: {n}")
    return 1 if res.failed else 0


if __name__ == "__main__":
    sys.exit(main())
