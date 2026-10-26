#!/usr/bin/env python3
"""Throughput / latency benchmark of runq serve under concurrent clients (Step 10b; not part of make test).
Python stdlib only; each client is a thread with its own HTTP connection.

Two loads, each against a fresh `runq serve` started here (on a free port, `-c 4096`):
  gen    N concurrent clients (--clients, default 1,2,4,8,16,32), each sending streamed chat completions
         with a short distinct prompt and a 128-token reply (--max-tokens; temperature 0): aggregate completion
         tok/s (all tokens / wall time), per-request latency p50 / p95, time to first token (the first content
         delta) p50 / p95, per-request decode speed after the first token. Each client sends --rounds requests
         (default 2 for N <= 4, else 1)
  judge  first a cold burst (max(--judge-clients) classifications at once on the fresh server: the rubric is
         prefilled once and copied / waited for, see --prefix-copy), then a LiteLLM-judge-like load: --judge-requests short classification requests (the legacy rubric as the
         system message, `\\nClassify this message:\\n` + a prompt of scripts/router_eval.jsonl, LiteLLM's
         TierClassification json_schema, temperature 0) from C concurrent clients (--judge-clients, default 1,4,8,16)
         after one warm-up request (the rubric cached): requests/s, latency p50 / p95
  mix    (--only mix) --mix-streams clients stream --mix-tokens-token replies while one client sends --mix-long
         requests with a ~2,000-token prompt: the long prompts' time to first token, the streams' inter-token gaps
         (p50 / p99 / max: how long a prefill chunk stalls them) and tok/s meanwhile; for tuning --chunk
Prints one table row per configuration (and --json FILE: all numbers).

Usage:
  scripts/bench_serve.py --runq ./runq --model models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf [--threads 12]
      [--only gen,judge] [--clients 1,2,4,8,16,32] [--judge-clients 1,4,8,16] [--serve-args "--parallel 16"]
      [--label after] [--json /tmp/x.json]
For a before / after comparison run it with the other binary (e.g. a build of an older commit in a scratch
worktree; --serve-args only for flags that binary knows).
"""
import argparse
import http.client
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
from server_test import RUBRIC, TIER_RF  # noqa: E402  (LiteLLM 1.105's rubric and response_format, verbatim)

TOPICS = ["a lighthouse keeper", "a lost umbrella", "the first snow", "a robot gardener", "an old map", "a night train",
          "a talking cat", "a desert well", "a broken clock", "a mountain village", "a paper boat", "a jazz pianist",
          "a winter market", "a quiet library", "a stubborn goat", "a hidden staircase", "a summer storm", "a glass bridge",
          "a retired pirate", "a moon garden", "a busy bakery", "a tiny dragon", "a forgotten song", "an island ferry",
          "a red bicycle", "a sleepy volcano", "a market in Lagos", "a chess champion", "a balloon race", "an ice hotel",
          "a beekeeper", "a museum guard", "a river otter", "a city of bridges", "a clockmaker", "a postcard from Mars"]


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, max(0, int(round(p / 100 * (len(v) - 1)))))] if v else float("nan")


def start_server(a, log):
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads))
    cmd = [a.runq, "serve", a.model, "--port", "0", "-c", str(a.ctx)] + shlex.split(a.serve_args)
    p = subprocess.Popen(cmd, stderr=log, stdout=subprocess.DEVNULL, env=env)
    for _ in range(600):
        m = re.search(r"listening on http://127\.0\.0\.1:(\d+)", open(log.name).read())
        if m:
            return p, int(m.group(1))
        if p.poll() is not None:
            break
        time.sleep(0.05)
    raise SystemExit("runq serve did not start:\n" + open(log.name).read()[-2000:])


def stream_req(port, body, times=None):
    """one streamed request on a new connection: (t_start, t_first_content or None, t_end, completion tokens);
    times: a list to append the arrival time of each content delta to"""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", json.dumps(dict(body, stream=True, stream_options={"include_usage": True})),
              {"Content-Type": "application/json"})
    r = c.getresponse()
    first, ntok, buf = None, 0, b""
    while True:
        line = r.fp.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data: ") or line == b"data: [DONE]":
            if line == b"data: [DONE]":
                break
            continue
        d = json.loads(line[6:])
        if "error" in d:
            raise RuntimeError(d["error"])
        if d.get("usage"):
            ntok = d["usage"]["completion_tokens"]
        for ch in d.get("choices", []):
            if ch["delta"].get("content"):
                now = time.time()
                first = first or now
                if times is not None:
                    times.append(now)
    c.close()
    return t0, first, time.time(), ntok


def plain_req(port, body):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    d = json.loads(r.read())
    c.close()
    if r.status != 200:
        raise RuntimeError(d)
    return t0, time.time(), d


def run_clients(n, work):
    """n threads, each running work(i); returns the list of their results (lists) concatenated, and the wall time"""
    out, lock = [], threading.Lock()
    def client(i):
        res = work(i)
        with lock:
            out.extend(res)
    ts = [threading.Thread(target=client, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out, time.time() - t0


def bench_gen(a, port, res):
    print(f"\n{a.label} gen: {os.path.basename(a.model)}, {a.threads} threads, max_tokens {a.max_tokens}")
    print(f"| clients | requests | tokens | wall s | aggregate tok/s | latency p50 / p95 s | TTFT p50 / p95 ms | per-request tok/s |")
    print("|---|---|---|---|---|---|---|---|")
    k = 0
    for n in a.clients:
        rounds = a.rounds or (2 if n <= 4 else 1)
        base = k
        k += n * rounds
        def work(i):
            rs = []
            for r in range(rounds):
                topic = TOPICS[(base + i * rounds + r) % len(TOPICS)]
                body = {"messages": [{"role": "user", "content": f"Write a long, detailed story about {topic}."}],
                        "max_tokens": a.max_tokens, "temperature": 0}
                rs.append(stream_req(port, body))
            return rs
        rs, wall = run_clients(n, work)
        toks = sum(x[3] for x in rs)
        lat = [x[2] - x[0] for x in rs]
        ttft = [(x[1] - x[0]) * 1000 for x in rs if x[1]]
        dec = [(x[3] - 1) / (x[2] - x[1]) for x in rs if x[1] and x[3] > 1 and x[2] > x[1]]
        row = dict(clients=n, requests=len(rs), tokens=toks, wall=wall, tps=toks / wall, lat50=pct(lat, 50), lat95=pct(lat, 95),
                   ttft50=pct(ttft, 50), ttft95=pct(ttft, 95), per_req=sum(dec) / len(dec) if dec else 0)
        res.setdefault("gen", []).append(row)
        print(f"| {n} | {len(rs)} | {toks} | {wall:.2f} | **{row['tps']:.0f}** | {row['lat50']:.2f} / {row['lat95']:.2f} | "
              f"{row['ttft50']:.0f} / {row['ttft95']:.0f} | {row['per_req']:.0f} |", flush=True)


def bench_judge(a, port, res):
    prompts = [json.loads(l)["prompt"] for l in open(os.path.join(ROOT, "scripts", "router_eval.jsonl"))]
    def body(p):
        return {"model": "judge", "messages": [{"role": "system", "content": RUBRIC},
                {"role": "user", "content": "\nClassify this message:\n" + p}], "response_format": TIER_RF, "temperature": 0}
    n = max(a.judge_clients)
    def cold(i):
        t0, t1, d = plain_req(port, body(prompts[(7 * i) % len(prompts)]))
        return [(t0, t1, d["usage"]["prompt_tokens_details"]["cached_tokens"])]
    rs, wall = run_clients(n, cold)
    lat = [(x[1] - x[0]) * 1000 for x in rs]
    res["judge_cold"] = dict(clients=n, wall=wall, lat50=pct(lat, 50), lat95=pct(lat, 95), cached=sorted(x[2] for x in rs))
    print(f"\n{a.label} judge, cold burst: {n} classifications at once on a fresh server: {wall * 1000:.0f} ms, latency p50 / p95 "
          f"{pct(lat, 50):.0f} / {pct(lat, 95):.0f} ms, cached tokens {sorted(x[2] for x in rs)}")
    plain_req(port, body("Hello there, warm-up."))
    print(f"\n{a.label} judge: {os.path.basename(a.model)}, {a.threads} threads, {a.judge_requests} classifications (rubric cached)")
    print("| clients | requests/s | latency p50 / p95 ms | cached tokens (median) |")
    print("|---|---|---|---|")
    for n in a.judge_clients:
        def work(i):
            rs = []
            for q in range(i, a.judge_requests, n):
                t0, t1, d = plain_req(port, body(prompts[q % len(prompts)]))
                json.loads(d["choices"][0]["message"]["content"])["tier"]
                rs.append((t0, t1, d["usage"]["prompt_tokens_details"]["cached_tokens"]))
            return rs
        rs, wall = run_clients(n, work)
        lat = [(x[1] - x[0]) * 1000 for x in rs]
        row = dict(clients=n, requests=len(rs), wall=wall, rps=len(rs) / wall, lat50=pct(lat, 50), lat95=pct(lat, 95),
                   cached=pct([x[2] for x in rs], 50))
        res.setdefault("judge", []).append(row)
        print(f"| {n} | **{row['rps']:.1f}** | {row['lat50']:.0f} / {row['lat95']:.0f} | {row['cached']} |", flush=True)


def bench_mix(a, port, res):
    """--mix-streams clients stream --mix-tokens-token replies while one client sends --mix-long requests with a ~2,000-token
    prompt each (16-token replies): the streams' inter-token gaps (the prefill stalls) and the long prompts' time to first token"""
    para = ("The committee met on a grey morning to review the harbour plans. Each proposal was read aloud, the costs were "
            "compared line by line, and the engineers answered questions about tides, foundations and the old sea wall. ")
    stop, times, firsts = threading.Event(), [[] for _ in range(a.mix_streams)], []
    def streamer(i):
        n = 0
        while not stop.is_set():
            body = {"messages": [{"role": "user", "content": f"Write a long, detailed story about {TOPICS[(i * 7 + n) % len(TOPICS)]}."}],
                    "max_tokens": a.mix_tokens, "temperature": 0}
            stream_req(port, body, times[i])
            n += 1
    ts = [threading.Thread(target=streamer, args=(i,)) for i in range(a.mix_streams)]
    for t in ts:
        t.start()
    time.sleep(1.0)
    t_long0 = time.time()
    for k in range(a.mix_long):
        body = {"messages": [{"role": "user", "content": f"Report {k}: " + para * 40 + " Summarize the report in one sentence."}],
                "max_tokens": 16, "temperature": 0}
        t0, first, t1, ntok = stream_req(port, body)
        firsts.append(((first or t1) - t0) * 1000)
        time.sleep(0.5)
    t_long1 = time.time()
    stop.set()
    for t in ts:
        t.join()
    gaps = [(b - a_) * 1000 for tl in times for a_, b in zip(tl, tl[1:]) if t_long0 <= a_ <= t_long1]
    ntok = sum(1 for tl in times for x in tl if t_long0 <= x <= t_long1)
    row = dict(streams=a.mix_streams, long=a.mix_long, ttft_long50=pct(firsts, 50), ttft_long_max=max(firsts), gap50=pct(gaps, 50),
               gap99=pct(gaps, 99), gapmax=max(gaps) if gaps else 0, stream_tps=ntok / (t_long1 - t_long0))
    res.setdefault("mix", []).append(row)
    print(f"\n{a.label} mix: {a.mix_streams} streams + {a.mix_long} prompts of ~2,000 tokens, {os.path.basename(a.model)}, {a.threads} threads, "
          f"serve args '{a.serve_args}'")
    print("| long prompt TTFT p50 / max ms | stream token gaps p50 / p99 / max ms | stream tok/s meanwhile |")
    print("|---|---|---|")
    print(f"| {row['ttft_long50']:.0f} / {row['ttft_long_max']:.0f} | {row['gap50']:.0f} / {row['gap99']:.0f} / {row['gapmax']:.0f} | "
          f"{row['stream_tps']:.0f} |", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runq", default=os.path.join(ROOT, "runq"))
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf"))
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--only", default="gen,judge")
    ap.add_argument("--clients", default="1,2,4,8,16,32")
    ap.add_argument("--rounds", type=int, default=0, help="requests per client (0: 2 for <= 4 clients, else 1)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--judge-clients", default="1,4,8,16")
    ap.add_argument("--judge-requests", type=int, default=64)
    ap.add_argument("--mix-streams", type=int, default=8)
    ap.add_argument("--mix-tokens", type=int, default=256)
    ap.add_argument("--mix-long", type=int, default=4)
    ap.add_argument("--serve-args", default="", help="extra runq serve flags")
    ap.add_argument("--label", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    a.clients = [int(x) for x in a.clients.split(",") if x]
    a.judge_clients = [int(x) for x in a.judge_clients.split(",") if x]
    a.label = a.label or os.path.basename(os.path.dirname(os.path.abspath(a.runq)))
    res = {"runq": a.runq, "model": a.model, "threads": a.threads, "serve_args": a.serve_args, "load": open("/proc/loadavg").read().split()[:3]}
    for what, fn in [("gen", bench_gen), ("judge", bench_judge), ("mix", bench_mix)]:
        if what not in a.only.split(","):
            continue
        log = tempfile.NamedTemporaryFile(prefix="bench-serve-", suffix=".log", delete=False)
        srv, port = start_server(a, log)
        try:
            fn(a, port, res)
        finally:
            srv.terminate()
            srv.wait()
            os.unlink(log.name)
    res["load_end"] = open("/proc/loadavg").read().split()[:3]
    print(f"(load {' '.join(res['load'])} -> {' '.join(res['load_end'])})")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
