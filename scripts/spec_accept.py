#!/usr/bin/env python3
"""Draft acceptance for speculative decoding (docs/qwen38-speculative.md, "Draft acceptance"): the target's answers
from an OpenAI-compatible API, teacher-forced through a draft GGUF with runq, and the loop simulated on the result.

  gen      query the target model (temperature 0 or seeded sampling) for answers to the built-in prompts; saves the
           token ids (logprobs with return_tokens_as_token_ids), the text, the reasoning and the API's prompt length.
  measure  teacher-force each saved answer through a draft GGUF (`runq -m logits`) with the target's exact chat
           rendering (Qwen3.8: thinking on adds the reasoning-effort system line; the prompt length is checked against
           the API's); saves per answer token the target id, the draft's argmax, its top-1 probability and the
           probability it gave the target's token.
  report   the table from measure's files: top-1 match, tokens per verify pass at k = 1..8 (the loop simulated:
           draft k, accept the leading matches, + 1 target token), the cost-model tok/s per k and the best k; with
           --tau, adaptive draft length (stop drafting when the draft's top-1 probability < tau, at most k_max).

Usage:
  .venv/bin/python -I scripts/spec_accept.py gen OUT.json [--think] [--temp 0] [--max 400] [--set base|long]
      [--key-file hetzner.key] [--endpoint URL] [--model NAME] [-j 2]
  .venv/bin/python -I scripts/spec_accept.py measure GEN.json DRAFT.gguf [-o MEAS.json] [--think|--no-think]
      [--effort xhigh] [--template tmpl_38.jinja] [--threads 8] [--runq ./runq] [--tmp /var/tmp]
  .venv/bin/python -I scripts/spec_accept.py report MEAS.json ... [--step-ms 120] [--draft-ms 6]
      [--verify 1.03,...] [--by-kind] [--tau 0.3,0.5 --kmax 4,8]

The key is read from the key file (never from the command line). Python stdlib + numpy (jinja2 only for --template,
a cross-check of the built-in rendering against the model's own template). Logits files (~1 MB per position for a
248k vocabulary) go to --tmp, one at a time, deleted after use.

The cost model (report): tok/s = tokens per pass / pass time, pass time = step_ms x verify(n) + draft_ms x (n + 1)
for n drafted tokens (verify(0) = 1: a plain step); verify(n) is the cost of a pass of n + 1 rows of one sequence
relative to one decode step (--verify, k = 1..8; a guess until TODO S2 measures it). With --tau the teacher-forced
probabilities are exact up to the first wrong guess; after it they only decide how many wasted guesses are paid for.
"""
import argparse
import concurrent.futures as cf
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- prompts (written for this project) -----------------------------------------------------------------------------

CODE = '''def moving_average(xs, n):
    out = []
    for i in range(len(xs)):
        if i < n - 1:
            out.append(None)
        else:
            out.append(sum(xs[i-n+1:i+1]) / n)
    return out
'''
JSON_TXT = ("Order #4821 was placed by Maria Lopez (maria.lopez@example.com) on 2026-03-14. "
            "Items: 2x USB-C cable at 9.99 EUR, 1x laptop stand at 34.50 EUR, 3x notebook at 4.25 EUR. "
            "Shipping to Calle Mayor 12, 28013 Madrid, Spain. Paid by credit card. Express delivery requested.")
ARTICLE = ("The city council voted on Tuesday to approve a new bike lane network covering 42 kilometers of "
           "downtown streets. The plan, which will cost 18 million euros over four years, replaces roughly "
           "1,200 on-street parking spaces with protected lanes. Supporters argue it will cut traffic "
           "accidents and emissions, while shopkeepers worry about lost customers. The council also "
           "promised a 3 million euro fund to compensate affected businesses, and a review after two years. "
           "Construction begins in April on the northern avenues, with the riverside section last.")

PROMPTS = {
    "base": [  # 30 prompts in 6 kinds (the 2026-10-27 measurement)
        ("chat", "Explain why the sky is blue in a short paragraph."),
        ("chat", "What are the main differences between TCP and UDP?"),
        ("chat", "Give me five tips for learning a new language as an adult."),
        ("chat", "Why do cats purr? Answer briefly."),
        ("chat", "Describe how a bill becomes law in the United States."),
        ("chat", "What is the difference between weather and climate?"),
        ("chat", "Write a short, friendly email asking a colleague to review my pull request."),
        ("chat", "Explain what a hash table is to a beginner."),
        ("code", "Write a Python function that checks whether a string is a valid IPv4 address, with a few doctests."),
        ("code", "Write a C function that reverses a singly linked list in place."),
        ("code", "Write a JavaScript debounce function with a short usage example."),
        ("code", "Write a SQL query that returns the top 3 customers by total order value from tables customers(id, "
                 "name) and orders(id, customer_id, total)."),
        ("code", "Write a Bash script that counts lines of code per file extension in the current directory tree."),
        ("code", "Implement binary search in Rust for a sorted slice of i32, returning Option<usize>."),
        ("edit", "Rewrite this function to handle n larger than len(xs) and add type hints. Return the full "
                 "function.\n\n```python\n" + CODE + "```"),
        ("edit", "Convert this function to use itertools and a generator instead of a list. Return the full "
                 "code.\n\n```python\n" + CODE + "```"),
        ("edit", "Add docstrings and input validation to this code and show the complete result.\n\n```python\n"
                 + CODE + "```"),
        ("edit", "Translate this Python function to TypeScript, keeping the same behaviour.\n\n```python\n"
                 + CODE + "```"),
        ("math", "A train travels 180 km in 2.5 hours, then 120 km in 1.5 hours. What is its average speed for the "
                 "whole trip? Show the steps."),
        ("math", "Solve for x: 3x^2 - 12x + 9 = 0. Show the steps."),
        ("math", "If I invest 2,000 euros at 5% annual interest compounded yearly, how much do I have after 10 "
                 "years? Show the calculation."),
        ("math", "How many ways can 5 people sit in a row if two of them refuse to sit next to each other? "
                 "Explain."),
        ("extract", "Extract the order data below as JSON with keys order_id, customer, email, date, items (name, "
                    "qty, unit_price), total, address. Output only JSON.\n\n" + JSON_TXT),
        ("extract", "List every number that appears in the text below, one per line, then say which one is the "
                    "largest.\n\n" + ARTICLE),
        ("extract", "From the text below, output a JSON object with keys cost_eur, duration_years, km, "
                    "parking_spaces, compensation_eur.\n\n" + ARTICLE),
        ("extract", "Quote the sentence from the text below that mentions shopkeepers, then explain the concern "
                    "in one line.\n\n" + ARTICLE),
        ("summ", "Summarize the text below in three bullet points.\n\n" + ARTICLE),
        ("summ", "Summarize the text below in one sentence.\n\n" + ARTICLE),
        ("summ", "Write a headline and a two-sentence summary for the text below.\n\n" + ARTICLE),
        ("summ", "Rewrite the text below in simpler words for a 10-year-old.\n\n" + ARTICLE),
    ],
    "long": [  # prompts that make a thinking model reason at length (thinking on, ~1,500 tokens)
        ("math", "A bag has 4 red, 5 blue and 6 green marbles. You draw 4 without replacement. What is the "
                 "probability that you get at least one marble of each colour? Give the exact fraction."),
        ("code", "Design an LRU cache in C with O(1) get and put for int keys and values: the data structures, "
                 "the eviction, and the complete code with a small test."),
        ("logic", "Five houses in a row are painted red, green, blue, yellow and white. The green house is "
                  "immediately left of the white one. The red house is in the middle. The blue house is at one "
                  "end. The yellow house is not next to the blue one. List every arrangement that satisfies all "
                  "the clues and explain why there are no others."),
    ],
}

IM_END = 248046  # Qwen3.5 / 3.8 <|im_end|>
EFFORT_LINE = {  # Qwen3.8's chat template, thinking on: the system line per reasoning effort (medium: none)
    "xhigh": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
             "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final "
             "answer.",
    "medium": "",
    "low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
           "without unnecessary elaboration.",
}


def render(prompt, think, effort="xhigh"):
    """Qwen3.8's template for one user message with the generation prompt (no system message, no tools)."""
    line = EFFORT_LINE[effort] if think else ""
    s = f"<|im_start|>system\n{line}<|im_end|>\n" if line else ""
    s += "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
    return s + ("<think>\n" if think else "<think>\n\n</think>\n\n")


def render_jinja(path, prompt, think, effort):
    import jinja2.ext
    import jinja2.sandbox

    def fail(msg):
        raise ValueError(msg)
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                       extensions=[jinja2.ext.loopcontrols])
    env.globals["raise_exception"] = fail
    with open(path) as f:
        t = env.from_string(f.read())
    return t.render(messages=[{"role": "user", "content": prompt}], add_generation_prompt=True,
                    enable_thinking=think, reasoning_effort=effort)


# --- gen ------------------------------------------------------------------------------------------------------------

def cmd_gen(a):
    with open(a.key_file) as f:
        key = f.read().strip()
    prompts = PROMPTS[a.set]
    if a.only:
        prompts = [prompts[int(i)] for i in a.only.split(",")]

    def call(item):
        cat, prompt = item
        body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": a.max,
                "temperature": a.temp, "logprobs": True, "top_logprobs": a.top_logprobs,
                "return_tokens_as_token_ids": True, "chat_template_kwargs": {"enable_thinking": a.think}}
        if a.temp > 0:
            body.update(top_p=a.top_p, top_k=a.top_k, seed=a.seed)
        req = urllib.request.Request(a.endpoint, data=json.dumps(body).encode(),
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        d, err = None, ""
        for attempt in range(6):
            if attempt:
                time.sleep(5 * attempt)
            try:
                with urllib.request.urlopen(req, timeout=a.timeout) as r:
                    d = json.load(r)
                break
            except Exception as e:  # 429s and timeouts: retry with backoff
                err = str(e)[:100]
        rec = {"cat": cat, "prompt": prompt, "think": a.think, "temp": a.temp, "max_tokens": a.max}
        if d is None:
            return dict(rec, error=err)
        c = d["choices"][0]
        lp = c["logprobs"]["content"]
        m = c["message"]
        rec.update(ids=[int(t["token"].split(":")[1]) for t in lp], content=m.get("content"),
                   reasoning=m.get("reasoning") or m.get("reasoning_content"),
                   prompt_tokens=d["usage"]["prompt_tokens"], finish=c["finish_reason"])
        if a.top_logprobs:
            rec["top_logprobs"] = [[(int(x["token"].split(":")[1]), x["logprob"]) for x in t["top_logprobs"]]
                                   for t in lp]
        return rec

    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        res = list(ex.map(call, prompts))
    with open(a.out, "w") as f:
        json.dump(res, f)
    print(len(res), "done;", sum(1 for r in res if "error" in r), "errors;",
          sum(len(r.get("ids", [])) for r in res), "tokens")


# --- measure --------------------------------------------------------------------------------------------------------

def runq_env(threads):
    return dict(os.environ, OMP_NUM_THREADS=str(threads))


def tokenize(a, text):
    o = subprocess.run([a.runq, a.draft, "-m", "tokenize", "-i", text], capture_output=True, text=True,
                       env=runq_env(a.threads), check=True).stdout
    return [int(x) for x in o.split()]


def draft_rows(a, text, n_ctx, first):
    """The draft's logits at every position of text: (token ids, argmax, top-1 prob, prob of the next token) for
    positions first-1 .. n-2 (the predictions of tokens first .. n-1)."""
    import numpy as np
    with tempfile.TemporaryDirectory(dir=a.tmp, prefix="spec_accept-") as td:
        lf = os.path.join(td, "logits.bin")
        subprocess.run([a.runq, a.draft, "-m", "logits", "-c", str(n_ctx), "-i", text, "-o", lf],
                       capture_output=True, env=runq_env(a.threads), check=True)
        with open(lf, "rb") as f:
            n, v = struct.unpack("ii", f.read(8))
            ids = np.frombuffer(f.read(4 * n), dtype=np.int32).copy()
        lg = np.memmap(lf, dtype=np.float32, mode="r", offset=8 + 4 * n, shape=(n, v))
        am, p1, pt = [], [], []
        for s in range(first - 1, n - 1, 64):
            e = min(s + 64, n - 1)
            x = np.asarray(lg[s:e], dtype=np.float64)
            mx = x.max(axis=1, keepdims=True)
            z = np.exp(x - mx).sum(axis=1)
            am.append(x.argmax(axis=1))
            p1.append(1.0 / z)
            pt.append(np.exp(x[np.arange(e - s), ids[s + 1:e + 1]] - mx[:, 0]) / z)
        del lg
    return ids, np.concatenate(am), np.concatenate(p1), np.concatenate(pt)


def cmd_measure(a):
    with open(a.gen) as f:
        res = json.load(f)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.gen)), "meas_%s_%s" % (
        os.path.splitext(os.path.basename(a.draft))[0], os.path.basename(a.gen)))
    answers = []
    for r in res:
        if "ids" not in r:
            continue
        think = a.think if a.think is not None else r.get("think", bool(r.get("reasoning")))
        ids = [i for i in r["ids"] if i != IM_END]
        ptext = render(r["prompt"], think, a.effort)
        if a.template:
            jt = render_jinja(a.template, r["prompt"], think, a.effort)
            if jt != ptext:
                sys.exit(f"built-in rendering differs from {a.template}:\n{ptext!r}\n{jt!r}")
        if think:  # the reasoning, then (unless cut by max_tokens) the end of thinking and the answer
            body = (r.get("reasoning") or "") + ("</think>" + (r.get("content") or "")
                                                  if r.get("finish") != "length" else "")
        else:
            body = r.get("content") or ""
        full = ptext + body
        n_pr = len(tokenize(a, ptext))
        fids = tokenize(a, full)
        if n_pr != r["prompt_tokens"]:
            print(f"  prompt length {n_pr} != API {r['prompt_tokens']}", flush=True)
        # the target is the retokenized text (the API's ids where they agree: retok_same)
        same = fids[n_pr:] == ids
        if len(fids) - n_pr < 8:
            continue
        tid, am, p1, pt = draft_rows(a, full, len(fids) + 16, n_pr)
        tgt = tid[n_pr:]
        m = am == tgt
        answers.append({"cat": r["cat"], "think": think, "n_prompt": n_pr, "prompt_tokens_api": r["prompt_tokens"],
                        "retok_same": same, "finish": r.get("finish"), "target": tgt.tolist(),
                        "draft": am.tolist(), "p_top1": [round(float(x), 5) for x in p1],
                        "p_target": [round(float(x), 5) for x in pt]})
        print(f"{r['cat']:8s} n={len(tgt):4d} top1={m.mean():.2f} retok_same={same}", flush=True)
    meta = {"gen": os.path.abspath(a.gen), "draft": os.path.abspath(a.draft), "effort": a.effort,
            "threads": a.threads}
    with open(out, "w") as f:
        json.dump({"meta": meta, "answers": answers}, f)
    print("wrote", out)


# --- report ---------------------------------------------------------------------------------------------------------

def simulate(answers, k, verify, step_ms, draft_ms, tau=None):
    """The loop on teacher-forced matches: from the first answer token, draft up to k (with tau: stop at the first
    position whose draft top-1 probability < tau), accept the leading matches, + 1 target token.
    Returns (tokens per pass, drafted per pass, tok/s)."""
    toks = passes = drafted = 0
    ms = 0.0
    for ans in answers:
        m, p = ans["m"], ans["p_top1"]
        s, L = 0, len(m)
        while s < L:
            n = 0  # drafted
            while n < k and s + n < L and (tau is None or p[s + n] >= tau):
                n += 1
            acc = 0
            while acc < n and m[s + acc]:
                acc += 1
            toks += min(acc + 1, L - s)
            passes += 1
            drafted += n
            ms += step_ms * (verify[n] if n else 1.0) + draft_ms * (n + 1)
            s += acc + 1
    return toks / passes, drafted / passes, 1000.0 * toks / ms


def load_meas(path):
    with open(path) as f:
        d = json.load(f)
    for ans in d["answers"]:
        ans["m"] = [x == y for x, y in zip(ans["draft"], ans["target"])]
    return d


def cmd_report(a):
    verify = [1.0] + [float(x) for x in a.verify.split(",")]
    ks = list(range(1, len(verify)))
    taus = [float(x) for x in a.tau.split(",")] if a.tau else []
    kmaxs = [int(x) for x in a.kmax.split(",")]
    while len(verify) <= max(kmaxs):  # k_max past --verify: extrapolated with its last slope
        verify.append(2 * verify[-1] - verify[-2])
    print(f"cost model: step {a.step_ms} ms, draft step {a.draft_ms} ms, verify(k) = {a.verify}")
    for path in a.meas:
        d = load_meas(path)
        A = d["answers"]
        kinds = sorted({x["cat"] for x in A}, key=[x["cat"] for x in A].index)
        groups = ([(c, [x for x in A if x["cat"] == c]) for c in kinds] if a.by_kind else []) + [("ALL", A)]
        ntok = sum(len(x["m"]) for x in A)
        print(f"\n== {os.path.basename(path)}: draft {os.path.basename(d['meta']['draft'])}, {len(A)} answers, "
              f"{ntok} tokens, retokenization == API ids: {sum(x['retok_same'] for x in A)}/{len(A)}, "
              f"prompt length == API: {sum(x['n_prompt'] == x['prompt_tokens_api'] for x in A)}/{len(A)}")
        print(f"{'kind':8s} {'tokens':>6s} {'top1':>5s} | tokens per pass, k = " + " ".join(f"{k:5d}" for k in ks)
              + " | tok/s, k = " + " ".join(f"{k:5d}" for k in ks) + " | best")
        best_fixed = {}
        for name, g in groups:
            t = sum(len(x["m"]) for x in g)
            h = sum(sum(x["m"]) for x in g)
            sims = [simulate(g, k, verify, a.step_ms, a.draft_ms) for k in ks]
            bk = max(range(len(ks)), key=lambda i: sims[i][2])
            best_fixed[name] = (ks[bk], sims[bk])
            print(f"{name:8s} {t:6d} {h / t:5.0%} | {'':20s}" + " ".join(f"{s[0]:5.2f}" for s in sims)
                  + f" | {'':10s}" + " ".join(f"{s[2]:5.1f}" for s in sims) + f" | k={ks[bk]} {sims[bk][2]:.1f}")
        if not taus:
            continue
        print(f"adaptive (stop when the draft's top-1 p < tau): tokens per pass / drafted per pass / tok/s")
        print(f"{'kind':8s} {'k_max':>5s} " + " ".join(f"{'tau=' + str(t):>17s}" for t in taus)
              + " | best adaptive vs best fixed k")
        for name, g in groups:
            best = None
            for km in kmaxs:
                sims = [simulate(g, km, verify, a.step_ms, a.draft_ms, tau) for tau in taus]
                for tau, s in zip(taus, sims):
                    if best is None or s[2] > best[2][2]:
                        best = (tau, km, s)
                print(f"{name:8s} {km:5d} " + " ".join(f"{s[0]:5.2f}/{s[1]:4.2f}/{s[2]:5.1f}" for s in sims)
                      + (f" | tau={best[0]} k_max={best[1]} {best[2][2]:.1f} vs k={best_fixed[name][0]} "
                         f"{best_fixed[name][1][2]:.1f} ({best[2][2] / best_fixed[name][1][2] - 1:+.1%})"
                         if km == kmaxs[-1] else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="query the target model through the API")
    g.add_argument("out")
    g.add_argument("--think", action="store_true", help="chat_template_kwargs.enable_thinking (default off)")
    g.add_argument("--temp", type=float, default=0.0)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--top-k", type=int, default=20)
    g.add_argument("--seed", type=int, default=1234)
    g.add_argument("--max", type=int, default=400, help="max_tokens (2026-10-27: 400, 500 with thinking)")
    g.add_argument("--set", choices=sorted(PROMPTS), default="base")
    g.add_argument("--only", help="comma-separated prompt indices of the set")
    g.add_argument("--top-logprobs", type=int, default=0)
    g.add_argument("--endpoint", default="https://inference.hetzner.com/api/v1/chat/completions")
    g.add_argument("--model", default="Qwen3.8-27B")
    g.add_argument("--key-file", default=os.path.join(ROOT, "hetzner.key"))
    g.add_argument("-j", "--concurrency", type=int, default=2, help="parallel requests (HTTP 429 above ~4)")
    g.add_argument("--timeout", type=float, default=300)
    m = sub.add_parser("measure", help="teacher-force saved answers through a draft GGUF")
    m.add_argument("gen")
    m.add_argument("draft")
    m.add_argument("-o", "--out", help="default: meas_<draft>_<gen>.json next to GEN")
    m.add_argument("--think", dest="think", action="store_true", default=None)
    m.add_argument("--no-think", dest="think", action="store_false",
                   help="default: the record's think field, else whether it has a reasoning")
    m.add_argument("--effort", choices=sorted(EFFORT_LINE), default="xhigh")
    m.add_argument("--template", help="a chat template (jinja) to check the built-in rendering against")
    m.add_argument("--threads", type=int, default=8)
    m.add_argument("--runq", default=os.path.join(ROOT, "runq"))
    m.add_argument("--tmp", default="/var/tmp", help="directory for the logits files (not a tmpfs)")
    r = sub.add_parser("report", help="the acceptance table and the cost model")
    r.add_argument("meas", nargs="+")
    r.add_argument("--step-ms", type=float, default=120.0, help="one target decode step (27B Q4_K_M: ~120)")
    r.add_argument("--draft-ms", type=float, default=6.0, help="one draft decode step")
    r.add_argument("--verify", default="1.03,1.12,1.20,1.35,1.47,1.60,1.75,1.90",
                   help="verify pass of k + 1 rows / one decode step, for k = 1.. (the table's k range; a larger "
                        "--kmax extrapolates with the last slope)")
    r.add_argument("--by-kind", action="store_true")
    r.add_argument("--tau", help="comma-separated thresholds: adaptive draft length")
    r.add_argument("--kmax", default="4,8")
    a = ap.parse_args()
    {"gen": cmd_gen, "measure": cmd_measure, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
