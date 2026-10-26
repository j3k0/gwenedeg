#!/usr/bin/env python3
"""Judge-quality evaluation of runq serve as LiteLLM's complexity-router classifier (not part of make test).

Runs the labeled prompts of scripts/router_eval.jsonl (80: 20 per tier, labels following LiteLLM's tier
definitions of the legacy rubric; incl. hard prompts that look short and easy ones that look long)
through runq serve with the exact request LiteLLM's classifier sends: the system prompt from
classification_system_prompt(0, rubric), the user message from ComplexityRouter._build_classifier_user_payload,
response_format from type_to_response_format_param(TierClassification), temperature 0 (LiteLLM's own code
is imported from its venv, read-only; scripts/litellm_e2e.sh checks that it is what the proxy sends;
--check-log compares with a logged request too). Reports per model and rubric: accuracy, per-tier
recall, the confusion matrix, how often a COMPLEX / REASONING prompt is rated SIMPLE, mean probability
of the gold label, latency; and the per-label probabilities at the one decision step (the masked softmax
over the allowed tokens, from the logprobs: every allowed token is a prefix of one label).

Usage (with the Python of a venv that has LiteLLM installed, tested with 1.105.0; see scripts/litellm_e2e.sh):
  $LITELLM_VENV/bin/python -I scripts/eval_router.py [--models a.gguf,b.gguf] [--rubrics legacy,chat]
      [--threads 8] [--out /tmp/eval.json] [--check-log /tmp/gwenedeg-e2e/runq.log] [--show-probs]
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIERS = ["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"]
DEFAULT_MODELS = ["models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf", "models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf",
                  "models/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf", "models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf"]


def litellm_request(rubric, prompt):
    from litellm.llms.base_llm.base_utils import type_to_response_format_param
    from litellm.router_strategy.complexity_router import complexity_router as cr
    from litellm.router_strategy.complexity_router.config import ClassificationRubric
    rub = None if rubric == "legacy" else ClassificationRubric(rubric)
    return {"messages": [{"role": "system", "content": cr.classification_system_prompt(0, classification_rubric=rub)},
                         {"role": "user", "content": cr.ComplexityRouter._build_classifier_user_payload(prompt=prompt)}],
            "response_format": type_to_response_format_param(cr._tier_classification_model(tuple(TIERS))),
            "temperature": 0}


def start_server(runq, model, threads):
    log = tempfile.NamedTemporaryFile(prefix="eval-serve-", suffix=".log", delete=False)
    p = subprocess.Popen([runq, "serve", model, "--port", "0", "-c", "4096", "--slots", "2"], stderr=log,
                         stdout=subprocess.DEVNULL, env=dict(os.environ, OMP_NUM_THREADS=str(threads)))
    for _ in range(600):
        m = re.search(r"listening on http://127\.0\.0\.1:(\d+)", open(log.name).read())
        if m:
            return p, int(m.group(1)), log.name
        if p.poll() is not None:
            break
        time.sleep(0.05)
    raise RuntimeError(open(log.name).read()[-500:])


def classify(port, body):
    body = dict(body, logprobs=True, top_logprobs=20, model="judge")
    t0 = time.time()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode())
    with urllib.request.urlopen(r, timeout=600) as f:
        d = json.load(f)
    dt = time.time() - t0
    tier = json.loads(d["choices"][0]["message"]["content"])["tier"]
    # the decision step: the first token that was not forced; its alternatives are the allowed tokens
    lp = d["choices"][0]["logprobs"]["content"]
    dec = next(x for x in lp if len(x["top_logprobs"]) > 1 or x["logprob"] != 0)
    probs = {t: 0.0 for t in TIERS}
    canon = {t: 0.0 for t in TIERS}
    for alt in dec["top_logprobs"]:
        for t in TIERS:
            if (t + '"}').startswith(alt["token"]):
                probs[t] += math.exp(alt["logprob"])
                if alt["token"] == {"SIMPLE": "SIM", "MEDIUM": "M", "COMPLEX": "COM", "REASONING": "RE"}[t]:
                    canon[t] = math.exp(alt["logprob"])
    return tier, probs, canon, dt, d["usage"]


def report(name, rows, show_probs):
    n = len(rows)
    acc = sum(r["pred"] == r["gold"] for r in rows) / n
    adj = sum(abs(TIERS.index(r["pred"]) - TIERS.index(r["gold"])) <= 1 for r in rows) / n
    hard = [r for r in rows if r["gold"] in ("COMPLEX", "REASONING")]
    hard_simple = sum(r["pred"] == "SIMPLE" for r in hard)
    pgold = sum(r["probs"][r["gold"]] for r in rows) / n
    lat = sorted(r["dt"] for r in rows[1:])
    noncanon = max(sum(r["probs"].values()) - sum(r["canon"].values()) for r in rows)
    print(f"\n### {name}: accuracy {acc:.1%} ({sum(r['pred'] == r['gold'] for r in rows)}/{n}), within one tier {adj:.1%}, "
          f"hard prompts rated SIMPLE {hard_simple}/{len(hard)}, mean p(gold) {pgold:.2f}")
    print(f"latency: first {rows[0]['dt'] * 1000:.0f} ms (rubric not cached), then median {lat[len(lat) // 2] * 1000:.0f} ms, "
          f"max {lat[-1] * 1000:.0f} ms; non-canonical first tokens carry at most {noncanon:.1e} of the mass")
    print("gold \\ pred  " + " ".join(f"{t:>9s}" for t in TIERS) + "   recall")
    for g in TIERS:
        row = [sum(r["gold"] == g and r["pred"] == p for r in rows) for p in TIERS]
        print(f"{g:11s}  " + " ".join(f"{c:9d}" for c in row) + f"   {row[TIERS.index(g)] / max(1, sum(row)):.0%}")
    dist = {t: sum(r["pred"] == t for r in rows) for t in TIERS}
    print("predicted: " + ", ".join(f"{t} {dist[t]}" for t in TIERS))
    if show_probs:
        for r in rows:
            mark = "  " if r["pred"] == r["gold"] else "X "
            print(f"  {mark}{r['gold']:9s} -> {r['pred']:9s} " + " ".join(f"{r['probs'][t]:.2f}" for t in TIERS) + f"  {r['prompt'][:70]!r}")
    return {"accuracy": acc, "within_one": adj, "hard_rated_simple": hard_simple, "mean_p_gold": pgold,
            "latency_first": rows[0]["dt"], "latency_median": lat[len(lat) // 2]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runq", default=os.path.join(ROOT, "runq"))
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--rubrics", default="legacy,chat,agentic,business")
    ap.add_argument("--data", default=os.path.join(ROOT, "scripts", "router_eval.jsonl"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="")
    ap.add_argument("--check-log", default="", help="a runq serve --verbose log of LiteLLM's classifier calls (litellm_e2e.sh)")
    ap.add_argument("--show-probs", action="store_true", help="per prompt: the label probabilities")
    a = ap.parse_args()
    try:
        import litellm  # noqa: F401
    except ImportError:
        print("eval_router.py needs LiteLLM: run it with the Python of a venv that has it (tested with 1.105.0), "
              "e.g. $LITELLM_VENV/bin/python -I scripts/eval_router.py", file=sys.stderr)
        return 2
    data = [json.loads(l) for l in open(a.data) if l.strip()]
    if a.check_log:
        logged = [json.loads(m) for m in re.findall(r"^\[request\] \S+ /v1/chat/completions (.*)$", open(a.check_log).read(), re.M)]
        logged = [b for b in logged if b.get("response_format")]
        user = logged[0]["messages"][1]["content"]
        prompt = user.split("Classify this message:\n", 1)[1]
        mine = litellm_request("legacy", prompt)
        same = all(logged[0][k] == mine[k] for k in ("messages", "response_format", "temperature"))
        print(f"request built here == the one LiteLLM sent ({a.check_log}, prompt {prompt!r}): {same}")
        if not same:
            return 1
    summary = {}
    for model in [m for m in a.models.split(",") if m]:
        path = model if os.path.isabs(model) else os.path.join(ROOT, model)
        if not os.path.exists(path):
            print(f"\n(skipped: {model} not found)")
            continue
        srv, port, log = start_server(a.runq, path, a.threads)
        try:
            for rubric in a.rubrics.split(","):
                rows = []
                for x in data:
                    tier, probs, canon, dt, usage = classify(port, litellm_request(rubric, x["prompt"]))
                    rows.append(dict(x, gold=x["tier"], pred=tier, probs=probs, canon=canon, dt=dt, prompt_tokens=usage["prompt_tokens"],
                                     cached=usage["prompt_tokens_details"]["cached_tokens"]))
                name = f"{os.path.basename(model)}, rubric {rubric}"
                summary[name] = dict(report(name, rows, a.show_probs), rows=rows)
        finally:
            srv.terminate()
            srv.wait()
            os.unlink(log)
    print("\n| model | rubric | accuracy | within one tier | hard rated SIMPLE | mean p(gold) | latency first / median |")
    print("|---|---|---|---|---|---|---|")
    for name, s in summary.items():
        m, r = name.split(", rubric ")
        print(f"| {m} | {r} | {s['accuracy']:.0%} | {s['within_one']:.0%} | {s['hard_rated_simple']}/40 | {s['mean_p_gold']:.2f} | "
              f"{s['latency_first'] * 1000:.0f} / {s['latency_median'] * 1000:.0f} ms |")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
