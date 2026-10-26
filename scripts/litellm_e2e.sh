#!/usr/bin/env bash
# End-to-end check of runq serve as the LLM judge of LiteLLM's complexity router (not part of make test):
#   client (openai package) -> scratch LiteLLM proxy (auto_router/complexity_router, classifier_type llm)
#     -> classifier call -> runq serve (json_schema TierClassification, constrained decoding)
#     -> the chosen tier's model, also runq serve (each tier a model_list entry with its own model name)
# Starts its own runq serve and its own LiteLLM proxy on free localhost ports (no database, no telemetry,
# a clean environment: nothing of a running proxy or its config is used), sends a few requests, prints
# the classifier request LiteLLM sent (from runq serve --verbose), checks that its prompt is byte for
# byte what LiteLLM's own code builds, the per-request routing and the classification latencies (the
# first one cold, the others with the rubric prefix cached), then stops both.
# Usage: scripts/litellm_e2e.sh [MODEL]   (default the Q8_0 GGUF)
# Env:   LITELLM_VENV (default /opt/litellm/venv-1.105.0: used read-only), WORK (default /tmp/cw9/e2e),
#        THREADS (runq serve, default 8), RUBRIC (legacy | chat | agentic | business, default legacy),
#        CONCURRENT (requests at once in the second pass, default 16)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODEL="${1:-models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf}"
VENV="${LITELLM_VENV:-/opt/litellm/venv-1.105.0}"
WORK="${WORK:-/tmp/cw9/e2e}"
THREADS="${THREADS:-8}"
RUBRIC="${RUBRIC:-legacy}"
PY="$VENV/bin/python"
[[ -x "$PY" ]] || { echo "no LiteLLM venv at $VENV" >&2; exit 2; }
[[ -x ./runq ]] || make -s runq
mkdir -p "$WORK/home"
free_port() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
RQ_PORT=$(free_port)
LL_PORT=$(free_port)
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT

# 1. runq serve (--verbose: logs every request body)
OMP_NUM_THREADS=$THREADS ./runq serve "$MODEL" --port "$RQ_PORT" -c 8192 --verbose > "$WORK/runq.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 100); do grep -q "listening on" "$WORK/runq.log" && break; sleep 0.1; done
grep "runq serve:" "$WORK/runq.log"

# 2. the LiteLLM config: the judge and the four tier models all on runq serve (the model name tells
#    them apart in runq's log); temperature 0 for the judge; context window 0 (current message only)
RUBRIC_LINE=""
[[ "$RUBRIC" != legacy ]] && RUBRIC_LINE="          classification_rubric: $RUBRIC"
cat > "$WORK/config.yaml" <<EOF
model_list:
  - model_name: smart-router
    litellm_params:
      model: auto_router/complexity_router
      complexity_router_config:
        classifier_type: llm
        classifier_context_window_size: 0
        classifier_llm_config:
          model: runq-judge
          timeout_ms: 10000
$RUBRIC_LINE
        tiers:
          SIMPLE: tier-simple
          MEDIUM: tier-medium
          COMPLEX: tier-complex
          REASONING: tier-reasoning
  - model_name: runq-judge
    litellm_params:
      model: openai/judge
      api_base: http://127.0.0.1:$RQ_PORT/v1
      api_key: none
      temperature: 0
EOF
for t in simple medium complex reasoning; do
    cat >> "$WORK/config.yaml" <<EOF
  - model_name: tier-$t
    litellm_params:
      model: openai/tier-$t
      api_base: http://127.0.0.1:$RQ_PORT/v1
      api_key: none
      temperature: 0
      max_tokens: 24
EOF
done
cat >> "$WORK/config.yaml" <<EOF
litellm_settings:
  telemetry: false
general_settings:
  disable_spend_logs: true
EOF

# 3. the LiteLLM proxy: its own process, clean environment, no database, a random scratch master key
KEY="sk-$(python3 -c 'import secrets; print(secrets.token_hex(24))')"
env -i PATH=/usr/bin:/bin HOME="$WORK/home" LITELLM_TELEMETRY=False LITELLM_LOCAL_MODEL_COST_MAP=True LITELLM_MASTER_KEY="$KEY" \
    "$VENV/bin/litellm" --config "$WORK/config.yaml" --host 127.0.0.1 --port "$LL_PORT" --num_workers 1 > "$WORK/litellm.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 300); do
    curl -s -o /dev/null "http://127.0.0.1:$LL_PORT/health/liveliness" && break
    sleep 0.2
done
curl -s -o /dev/null "http://127.0.0.1:$LL_PORT/health/liveliness" || { tail -30 "$WORK/litellm.log"; exit 1; }
echo "LiteLLM $("$PY" -I -c 'import importlib.metadata as m; print(m.version("litellm"))') proxy on :$LL_PORT, runq serve on :$RQ_PORT"

# 4. requests through the router (openai package from the LiteLLM venv), then the checks
RQ_LOG="$WORK/runq.log" LL_PORT=$LL_PORT LL_KEY="$KEY" RUBRIC=$RUBRIC CONCURRENT="${CONCURRENT:-16}" "$PY" -I - <<'PYEOF'
import json, os, re, time
import openai
from litellm.router_strategy.complexity_router import complexity_router as cr
from litellm.router_strategy.complexity_router.config import ClassificationRubric

client = openai.OpenAI(base_url=f"http://127.0.0.1:{os.environ['LL_PORT']}/v1", api_key=os.environ["LL_KEY"], max_retries=0)
prompts = [
    "Hi there!",
    "What is the capital of Japan?",
    "Explain the difference between a list and a tuple in Python.",
    "Implement a lock-free multi-producer multi-consumer queue in C++ with correct memory ordering.",
    "Prove that there is no largest prime number.",
    "Should our startup build its own payment system or use Stripe, given strict EU data residency rules? Commit to one.",
    "Thanks!",
    "What is the capital of Japan?",
]
log = os.environ["RQ_LOG"]
def runq_log():
    """runq serve's chat requests in arrival order: (request body, response body or None, log line), paired by
    the client address (concurrent requests interleave in the log; one connection's requests are sequential)"""
    out, cur = [], {}
    for line in open(log).read().splitlines():
        if line.startswith("[request] ") and " /v1/chat/completions " in line:
            peer = line.split(" ", 3)[1]
            cur[peer] = [json.loads(line.split(" ", 3)[3]), None, None]
            out.append(cur[peer])
        elif line.startswith("[response] ") and line.split(" ", 2)[1] in cur:
            cur[line.split(" ", 2)[1]][1] = json.loads(line.split(" ", 2)[2])
        elif re.match(r"^\[[0-9-]+ [0-9:]+\] \S+ POST /v1/chat/completions ", line) and line.split(" ")[2] in cur:
            cur[line.split(" ")[2]][2] = line
    return out
print(f"\n{'prompt':58s} {'judge says':11s} {'routed to':15s} {'judge ms':>9s} {'cached':>7s} {'end-to-end ms':>14s}")
for p in prompts:
    n0 = len(runq_log())
    t0 = time.time()
    client.chat.completions.create(model="smart-router", messages=[{"role": "user", "content": p}])
    dt = time.time() - t0
    new = runq_log()[n0:]
    judge = [x for x in new if x[0].get("response_format")]
    routed = ",".join(x[0]["model"] for x in new if not x[0].get("response_format"))
    tier = jt = jc = "?"
    if judge and judge[0][1] and judge[0][2]:
        tier = json.loads(judge[0][1]["choices"][0]["message"]["content"])["tier"]
        m = re.search(r"cached=(\d+) .* total=([0-9.]+)s", judge[0][2])
        jc, jt = m.group(1), f"{float(m.group(2)) * 1000:.0f}"
    print(f"{p[:57]:58s} {tier:11s} {routed:15s} {jt:>9s} {jc:>7s} {dt * 1000:14.0f}")

seq_tier = {}
for x in runq_log():
    if x[0].get("response_format") and x[1]:
        seq_tier[x[0]["messages"][-1]["content"]] = json.loads(x[1]["choices"][0]["message"]["content"])["tier"]

# the same prompts again, CONCURRENT at once through the proxy (each twice): every request classified and
# routed, the same tier as in the sequential pass (temperature 0; a request's output does not depend on what
# runq batches it with), judge latency per request from runq's log
import concurrent.futures as cf
nconc = int(os.environ.get("CONCURRENT", "16"))
batch = [prompts[i % len(prompts)] for i in range(nconc)]
n0 = len(runq_log())
t0 = time.time()
def one(p):
    t = time.time()
    r = client.chat.completions.create(model="smart-router", messages=[{"role": "user", "content": p}])
    return time.time() - t, r.choices[0].message.content
with cf.ThreadPoolExecutor(max_workers=nconc) as ex:
    res = list(ex.map(one, batch))
wall = time.time() - t0
new = runq_log()[n0:]
judge = [x for x in new if x[0].get("response_format")]
tiers = [(x[0]["messages"][-1]["content"], json.loads(x[1]["choices"][0]["message"]["content"])["tier"]) for x in judge if x[1]]
same = all(seq_tier.get(u) == t for u, t in tiers)
jl = sorted(float(re.search(r"total=([0-9.]+)s", x[2]).group(1)) * 1000 for x in judge if x[2])
routed = sorted(x[0]["model"] for x in new if not x[0].get("response_format"))
print(f"\n{nconc} requests at once through the proxy: {wall * 1000:.0f} ms wall, end-to-end per request "
      f"{min(r[0] for r in res) * 1000:.0f}-{max(r[0] for r in res) * 1000:.0f} ms; {len(tiers)} classified, {len(routed)} routed "
      f"({', '.join(f'{m} x{routed.count(m)}' for m in sorted(set(routed)))}); every answer non-empty: {all(r[1] for r in res)}")
print(f"judge latency (runq log) p50 / max {jl[len(jl) // 2]:.0f} / {jl[-1]:.0f} ms; tiers == the sequential pass: {same}")

bodies = [x[0] for x in runq_log()]
judge_bodies = [b for b in bodies if b.get("response_format")]
print(f"\n{len(judge_bodies)} classifier calls, {len(bodies) - len(judge_bodies)} tier-model calls")
b = judge_bodies[0]
print("\nOne classifier request body as LiteLLM sent it (runq serve --verbose; message text shortened):")
show = json.loads(json.dumps(b))
for m in show["messages"]:
    if isinstance(m["content"], str) and len(m["content"]) > 160:
        m["content"] = m["content"][:120] + f"... [{len(m['content'])} chars]"
print(json.dumps(show, indent=1))
# the prompt must be what LiteLLM's own code builds, byte for byte
rub = os.environ["RUBRIC"]
want_sys = cr.classification_system_prompt(0, classification_rubric=None if rub == "legacy" else ClassificationRubric(rub))
want_user = cr.ComplexityRouter._build_classifier_user_payload(prompt=prompts[0])
ok = b["messages"] == [{"role": "system", "content": want_sys}, {"role": "user", "content": want_user}]
print(f"\nclassifier prompt == LiteLLM's classification_system_prompt + _build_classifier_user_payload: {ok}")
want_rf = json.loads(json.dumps(__import__("litellm").llms.base_llm.base_utils.type_to_response_format_param(
    cr._tier_classification_model(("SIMPLE", "MEDIUM", "COMPLEX", "REASONING")))))
print(f"response_format == type_to_response_format_param(TierClassification): {b['response_format'] == want_rf}")
PYEOF
echo
grep -E "POST /v1/chat/completions" "$WORK/runq.log" | grep -v "^\[request\]\|^\[response\]" | tail -20
