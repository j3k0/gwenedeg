#!/usr/bin/env python3
"""The server benchmark of TODO S9c (PLAN-S9c-server.md S9c.G): one `runq serve` and the requests of one configuration.

  scripts/s9c_bench.py --model MODEL --serve-args "ARGS" [--short [--think 1,0]] [--long L --turns T] [--label X] [--json PATH]
  scripts/s9c_bench.py --compare A.json B.json ...      (the output hashes of the runs: equal per request, exit 0; else 1)

The server: `runq serve MODEL ARGS [--mem $S9_GATE_MEM] --host 127.0.0.1 --port 0 --timeout 0` (S9_GATE_MEM: exported by
scripts/s9_gate.sh's server mode; the server's threads, draft, pool and -v come with ARGS). Its startup lines, its
shutdown lines (-v: the [engine] summary, mtp: A / B / steps) and every second /health's mem and the server's VmRSS /
VmHWM (/proc/PID/status) are kept. Per request: prompt_n, prompt_ms, cache_n, predicted_n / predicted_ms (tok/s), the
speculative counters (passes per source: what auto chose; drafted / accepted; draft_rows; mtp_approx), lease_gib, the
wall time and a hash of the output (content + reasoning + finish_reason).

  --short   4 requests (bench/s9c): two LiveCodeBench-shaped ones (task-lcb1.txt, task-lcb2.txt; no system block: the
            nightly's load) and two agent-shaped ones (agent-block.txt as the system block + task-code.txt / task-tool.txt),
            max_tokens 512, greedy and seeded (T 1, top-p 0.95, top-k 20, seed 42): one client, then the 4 at once; per
            --think mode (1: thinking on, Qwen3.8's default; 0: chat_template_kwargs.enable_thinking false). The metric
            (S9c.G2): per mode the mean of the 8 one-client tok/s, and the 4 clients' aggregate tok/s (their tokens / the
            round's wall time)
  --long L  one prompt of ~L tokens (long-32k.txt cut at a line to ~L tokens by the model's tokenizer, runq -m tokenize,
            + task-long.txt; its cold prefill reported alone), then --turns T pi-shaped turns: the reply kept, ~2,500 new
            tokens each (the next slice of the same source: the rest of long-32k.txt, then long-more.txt) and a question;
            greedy, max_tokens 512, auto (the server's default); per turn the wall, prefill and decode times

Numbers go to --json (default /var/tmp/s9c/bench-LABEL-TIME.json) and its server log next to it (.log); a summary to stdout.
Python stdlib only.
"""
import argparse
import concurrent.futures as cf
import hashlib
import http.client
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "bench", "s9c")
MAX_TOKENS = 512
SEEDED = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 42}
GREEDY = {"temperature": 0}
TURN_TOKENS = 2500
TURN_ASK = "\n\nThat was the next part of the same file. What does it add to the request's path you described? Answer briefly."


def text(name):
    with open(os.path.join(BENCH, name), encoding="utf-8") as f:
        return f.read()


def short_prompts():
    """the --short requests' messages: (name, messages)"""
    agent = text("agent-block.txt").rstrip("\n")
    return [("lcb1", [{"role": "user", "content": text("task-lcb1.txt").strip()}]),
            ("lcb2", [{"role": "user", "content": text("task-lcb2.txt").strip()}]),
            ("code", [{"role": "system", "content": agent}, {"role": "user", "content": text("task-code.txt").strip()}]),
            ("tool", [{"role": "system", "content": agent}, {"role": "user", "content": text("task-tool.txt").strip()}])]


def ntokens(runq, model, s):
    """the token count of s by the model's tokenizer (runq -m tokenize: the header and the tokenizer only)"""
    p = subprocess.run([runq, model, "-m", "tokenize", "-i", s], capture_output=True)
    m = re.search(rb"\[tokenize\] \d+ bytes -> (\d+) tokens", p.stderr)
    if p.returncode != 0 or not m:
        sys.exit(f"s9c_bench: runq -m tokenize failed: {p.stderr.decode(errors='replace')[-300:]}")
    return int(m.group(1))


def cut(runq, model, s, want):
    """s cut at a line to ~want tokens of the model's tokenizer (within 2%, at most 4 tokenizer runs): (head, rest)"""
    b = s.encode()
    n = ntokens(runq, model, s)
    if want >= n:
        return s, ""
    pos, i = len(b) * want // n, 0
    for _ in range(4):
        i = b.rfind(b"\n", 0, min(pos, len(b))) + 1 or min(pos, len(b))
        k = ntokens(runq, model, b[:i].decode(errors="ignore"))
        if abs(k - want) <= want // 50 or k == 0:
            break
        pos = i * want // k
    return b[:i].decode(errors="ignore"), b[i:].decode(errors="ignore")


class Server:
    """runq serve MODEL ARGS on 127.0.0.1, an ephemeral port; its log in a file; a sampler of /health and /proc/PID/status"""

    def __init__(self, a, logpath):
        cmd = [a.runq, "serve", a.model] + shlex.split(a.serve_args)
        if os.environ.get("S9_GATE_MEM"):
            cmd += ["--mem", os.environ["S9_GATE_MEM"]]
        cmd += ["--host", "127.0.0.1", "--port", "0", "--timeout", "0"]
        self.cmd, self.logpath = cmd, logpath
        self.log = open(logpath, "w")
        self.t0 = time.time()
        self.p = subprocess.Popen(cmd, stderr=self.log, stdout=subprocess.DEVNULL)
        self.port, self.samples, self.stop_at = None, [], 0
        while time.time() - self.t0 < a.start_timeout:
            m = re.search(r"listening on http://127\.0\.0\.1:(\d+)", open(logpath).read())
            if m:
                self.port = int(m.group(1))
                break
            if self.p.poll() is not None:
                break
            time.sleep(0.1)
        self.load_s = time.time() - self.t0
        if not self.port:
            self.stop()
            sys.exit(f"s9c_bench: the server did not start ({' '.join(cmd)}): {open(logpath).read()[-600:]}")
        self.done = threading.Event()
        self.th = threading.Thread(target=self.sample, daemon=True)
        self.th.start()

    def status(self):
        st = {}
        try:
            with open(f"/proc/{self.p.pid}/status") as f:
                for line in f:
                    k, _, v = line.partition(":")
                    if k in ("VmRSS", "VmHWM", "VmSwap"):
                        st[k] = int(v.split()[0]) / 1048576  # (GiB)
        except OSError:
            pass
        return st

    def sample(self):
        while not self.done.is_set():
            try:
                _, h = self.get("/health")
            except Exception:  # noqa: BLE001  (busy or gone: no sample)
                h = {}
            s = {"t": round(time.time() - self.t0, 2), "running": h.get("running"), "queued": h.get("queued")}
            s.update({k: h["mem"][k] for k in ("budget", "live", "committed", "resident") if isinstance(h.get("mem"), dict) and k in h["mem"]})
            s.update({k: round(v, 4) for k, v in self.status().items()})
            self.samples.append(s)
            self.done.wait(1.0)

    def get(self, path):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("GET", path)
        r = c.getresponse()
        d = json.loads(r.read() or b"{}")
        c.close()
        return r.status, d

    def chat(self, body):
        """(status, reply, wall seconds)"""
        t = time.time()
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=7200)
        c.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        r = c.getresponse()
        raw = r.read()
        c.close()
        try:
            d = json.loads(raw)
        except ValueError:
            d = {"error": raw[:300].decode(errors="replace")}
        return r.status, d, time.time() - t

    def stop(self):
        """SIGTERM, the shutdown lines (everything the log gained after it)"""
        if getattr(self, "done", None):
            self.done.set()
        self.log.flush()
        self.stop_at = os.path.getsize(self.logpath)
        self.p.send_signal(signal.SIGTERM)
        try:
            self.rc = self.p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.rc = "killed"
        self.log.close()
        with open(self.logpath) as f:
            log = f.read()
        head = log.split("runq serve: listening", 1)[0].splitlines()
        return head, log[self.stop_at:].splitlines()


def record(key, st, d, wall, **kw):
    """one request's numbers"""
    ch = (d.get("choices") or [{}])[0] if st == 200 else {}
    msg = ch.get("message") or {}
    out = (msg.get("content") or "") + "\0" + (msg.get("reasoning_content") or "") + "\0" + str(ch.get("finish_reason"))
    tm = d.get("timings", {}) if st == 200 else {}
    sp = tm.get("speculative") or {}
    r = dict(key=key, status=st, wall_s=round(wall, 3), prompt_n=tm.get("prompt_n"), prompt_ms=tm.get("prompt_ms"),
             cache_n=tm.get("cache_n"), predicted_n=tm.get("predicted_n"), predicted_ms=tm.get("predicted_ms"),
             tok_s=round(tm["predicted_n"] / (tm["predicted_ms"] / 1e3), 3) if tm.get("predicted_ms") else None,
             lease_gib=tm.get("lease_gib"), finish=ch.get("finish_reason"),
             hash=hashlib.sha256(out.encode()).hexdigest()[:16], **kw)
    if sp:
        r.update(mode=sp.get("mode"), passes=sp.get("passes"), model=sp.get("model"), lookup=sp.get("lookup"),
                 plain_passes=sp.get("passes", 0) - sp.get("model", {}).get("passes", 0) - sp.get("lookup", {}).get("passes", 0),
                 draft_rows=sp.get("draft_rows"), mtp_approx=sp.get("mtp_approx"), refed=sp.get("refed"))
    if st != 200:
        r["error"] = str(d.get("error"))[:300]
    return r


def think_args(t):
    return {"chat_template_kwargs": {"enable_thinking": bool(t)}}


def run_short(srv, a, recs):
    """--short: per thinking mode, the 4 requests greedy then seeded at one client, then each sampling 4 at once"""
    summ = {}
    for t in a.think:
        one, agg = [], []
        for samp, sa in (("greedy", GREEDY), ("seeded", SEEDED)):
            for name, msgs in short_prompts():
                body = dict(messages=msgs, max_tokens=a.max_tokens, **sa, **think_args(t))
                r = record(f"short/think{t}/{name}/{samp}", *srv.chat(body), think=t, prompt=name, sampling=samp, clients=1)
                recs.append(r)
                one.append(r)
                print(f"  think {t} {samp:6s} {name:5s} 1 client: {r['tok_s']} tok/s, prompt {r['prompt_n']} ({r['prompt_ms']} ms), "
                      f"{r['predicted_n']} tokens, model {r.get('model')}, mtp_approx {r.get('mtp_approx')}", flush=True)
        for samp, sa in (("greedy", GREEDY), ("seeded", SEEDED)):
            bodies = [(name, dict(messages=msgs, max_tokens=a.max_tokens, **sa, **think_args(t))) for name, msgs in short_prompts()]
            t0 = time.time()
            with cf.ThreadPoolExecutor(max_workers=4) as ex:
                res = list(ex.map(lambda nb: (nb[0], srv.chat(nb[1])), bodies))
            wall = time.time() - t0
            rr = [record(f"short/think{t}/{name}/{samp}", *x, think=t, prompt=name, sampling=samp, clients=4) for name, x in res]
            recs.extend(rr)
            n = sum(r["predicted_n"] or 0 for r in rr)
            agg.append(dict(sampling=samp, tokens=n, wall_s=round(wall, 3), tok_s=round(n / wall, 3)))
            print(f"  think {t} {samp:6s} 4 clients: {n} tokens in {wall:.1f} s = {n / wall:.2f} tok/s", flush=True)
        ok = [r["tok_s"] for r in one if r["tok_s"]]
        summ[f"think{t}"] = dict(one_client_mean_tok_s=round(sum(ok) / len(ok), 3) if ok else None, n=len(ok),
                                 four_clients=agg, four_clients_mean_tok_s=round(sum(x["tok_s"] for x in agg) / len(agg), 3))
    return summ


def run_long(srv, a, recs):
    """--long L --turns T: the cold prompt, then T turns of ~2,500 new tokens each"""
    src, task, more = text("long-32k.txt"), text("task-long.txt"), text("long-more.txt")
    head, rest = cut(a.runq, a.model, src, a.long)
    rest += more
    t = a.think[0]
    msgs = [{"role": "user", "content": head + task}]
    turns = []
    for i in range(a.turns + 1):
        body = dict(messages=msgs, max_tokens=a.max_tokens, **GREEDY, **think_args(t))
        st, d, wall = srv.chat(body)
        r = record(f"long{a.long}/think{t}/turn{i}", st, d, wall, think=t, turn=i)
        recs.append(r)
        turns.append(r)
        print(f"  long {a.long} turn {i}: wall {r['wall_s']} s, prefill {r['prompt_n']} tokens {r['prompt_ms']} ms (cached {r['cache_n']}), "
              f"decode {r['predicted_n']} tokens {r['predicted_ms']} ms ({r['tok_s']} tok/s), model {r.get('model')}, "
              f"lookup {r.get('lookup')}, plain passes {r.get('plain_passes')}, mtp_approx {r.get('mtp_approx')}", flush=True)
        if r["status"] != 200 or i == a.turns:
            break
        piece, _ = cut(a.runq, a.model, rest[:16 * TURN_TOKENS], TURN_TOKENS)  # (the next ~2,500 tokens)
        rest = rest[len(piece):]
        if not piece:
            print("  (the source is used up)")
            break
        reply = (d["choices"][0]["message"].get("content") or "")  # (the reasoning is not part of the history: the template drops it)
        msgs = msgs + [{"role": "assistant", "content": reply}, {"role": "user", "content": "```c\n" + piece + "```" + TURN_ASK}]
    return dict(cold_prefill_ms=turns[0]["prompt_ms"] if turns else None, prompt_n=turns[0]["prompt_n"] if turns else None,
                turns=[{k: r.get(k) for k in ("turn", "wall_s", "prompt_n", "cache_n", "prompt_ms", "predicted_n", "predicted_ms", "tok_s")}
                       for r in turns])


def compare(paths):
    """the runs' output hashes per request key (every client count): equal, or the differences"""
    seen, bad, n = {}, [], 0
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        for r in d["requests"]:
            if r["status"] != 200:
                bad.append(f"{p}: {r['key']}: status {r['status']}")
                continue
            n += 1
            w = seen.setdefault(r["key"], (r["hash"], p))
            if w[0] != r["hash"]:
                bad.append(f"{r['key']}: {p} {r['hash']} vs {w[1]} {w[0]}")
    print(f"s9c_bench --compare: {n} replies of {len(paths)} runs, {len(seen)} requests: "
          + ("every hash equal" if not bad else f"{len(bad)} differ"))
    for b in bad:
        print("  " + b)
    return 1 if bad or not n else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model")
    ap.add_argument("--serve-args", default="", help="runq serve's arguments (threads, draft, pool, -v, ...)")
    ap.add_argument("--runq", default=os.path.join(ROOT, "runq"))
    ap.add_argument("--short", action="store_true")
    ap.add_argument("--think", default="1", help="1, 0 or 1,0 (thinking on / off)")
    ap.add_argument("--long", type=int, default=0, metavar="L")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--label", default="run")
    ap.add_argument("--json", help="default /var/tmp/s9c/bench-LABEL-TIME.json")
    ap.add_argument("--start-timeout", type=float, default=600)
    ap.add_argument("--compare", nargs="+", metavar="JSON")
    a = ap.parse_args()
    if a.compare:
        return compare(a.compare)
    if not a.model or not (a.short or a.long):
        ap.error("--model and --short and / or --long L are needed")
    a.think = [int(x) for x in a.think.split(",")]
    if any(x not in (0, 1) for x in a.think):
        ap.error("--think: 1, 0 or 1,0")
    a.json = a.json or f"/var/tmp/s9c/bench-{a.label}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    logpath = os.path.splitext(a.json)[0] + ".log"
    srv = Server(a, logpath)
    print(f"s9c_bench {a.label}: {' '.join(srv.cmd)} (loaded in {srv.load_s:.1f} s)", flush=True)
    recs, res = [], {}
    try:
        if a.short:
            res["short"] = run_short(srv, a, recs)
        if a.long:
            res["long"] = run_long(srv, a, recs)
    finally:
        head, tail = srv.stop()
    hwm = max((s.get("VmHWM", 0) for s in srv.samples), default=0)
    out = dict(label=a.label, model=a.model, serve_cmd=srv.cmd, s9_gate_mem=os.environ.get("S9_GATE_MEM"), load_s=round(srv.load_s, 2),
               time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), results=res, requests=recs, startup=head, shutdown=tail,
               server_rc=srv.rc, peak=dict(VmHWM=hwm, committed=max((s.get("committed", 0) for s in srv.samples), default=0)),
               samples=srv.samples)
    with open(a.json, "w") as f:
        json.dump(out, f, indent=1)
    for k, v in res.get("short", {}).items():
        print(f"s9c_bench {a.label} {k}: 1 client mean {v['one_client_mean_tok_s']} tok/s ({v['n']} requests); 4 clients "
              + ", ".join(f"{x['sampling']} {x['tok_s']}" for x in v["four_clients"]) + " tok/s")
    if "long" in res:
        print(f"s9c_bench {a.label} long: cold prefill {res['long']['prompt_n']} tokens {res['long']['cold_prefill_ms']} ms; turns "
              + "; ".join(f"{t['turn']}: wall {t['wall_s']} s (prefill {t['prompt_ms']} ms, decode {t['predicted_ms']} ms)"
                          for t in res["long"]["turns"]))
    print(f"s9c_bench {a.label}: peak VmHWM {hwm:.2f} GiB, committed {out['peak']['committed']} GiB; server exit {srv.rc}; "
          f"{sum(1 for r in recs if r['status'] != 200)} errors; {a.json}")
    for l in tail:
        if l.startswith("[engine] ") and any(w in l for w in ("speculation:", "mtp:", "steps")):
            print("  " + l)
    return 0 if srv.rc == 0 and all(r["status"] == 200 for r in recs) else 1


if __name__ == "__main__":
    sys.exit(main())
