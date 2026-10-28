#!/usr/bin/env python3
"""Replay benchmark of agentic sessions (TODO G0 (a), G5): pi's recorded sessions replayed against `runq serve`,
request by request, as pi would send them; per request the reused prefix, the tokens prefilled, rollbacks and the
timings. Local data only: the sessions (~/.pi/agent/sessions/*/*.jsonl) are read, never copied; the outputs (numbers
only, no content) go under --out (default /var/tmp/runq-replay; logs/ is git-ignored too).

  scripts/replay_sessions.py --model models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf --ctx 98304 --pick 3
  scripts/replay_sessions.py --model ... --ctx 32768 --session FILE [--session FILE ...] [--mode live]
  scripts/replay_sessions.py --report /var/tmp/runq-replay/RUN.jsonl      (the table of an earlier run)

How (scripts/replay_pi.mjs, run with node): pi's own code builds each request (pi-coding-agent's session context,
pi-ai's openai-completions provider: the same body, the same SSE parsing), with the server's context as pi's model
window (pi's compaction triggers there: its summary requests are replayed too). --mode force (default): every reply is
forced to the recorded one (runq serve's force_reply: the recorded message rendered as the template renders it, fed as
if generated; the next request carries the server's parse of it, through pi), so the cache sees what a model writing
in the template's format would have generated; --mode live: the model generates (up to --live-tokens) and its own
reply goes on in the history.

Per request (the JSON lines, the table): kind (turn after tool results / after a user message / after pi changed
its system prompt (a system message in the session: its sections or tools) / the first after a compaction; summary:
pi's compaction request), prompt tokens, cached (usage.prompt_tokens_details.cached_tokens),
prefilled (timings.prompt_n), new: the tokens a perfect cache would prefill (a turn: its prompt minus the previous
turn's prompt and reply; otherwise the whole prompt), extra = prefilled - new (what the cache missed; exact: extra <= 1;
beyond: tool turns whose extra is more than the previous reply and its header, i.e. not the reply sent back
differently from what was generated: the goal is none), whether the
prompt left the slot's tokens before their end (diverged: the history was sent differently from what was generated)
and whether the hybrid state rolled back further (rollback: the kept length is a checkpoint below the common prefix),
prefill and decode times, the reply tokens, and with speculation (--server-arg=--draft --server-arg=FILE, or
--server-arg=--lookup --server-arg=3; --mode live: forced replies do not speculate) the guesses accepted / drafted per
source (timings.speculative). --probe-every N (force mode): before every Nth reply after tool results, a live request
with the same context (kind probe, at most --probe-tokens tokens, its reply dropped; the forced request after it reuses
the prompt): the decode rate and the speculation at that position of the real session, the rest replayed as forced. Sessions are replayed one after the other on one server (a fresh one per run).
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pi_dir(arg):
    if arg:
        return arg
    rel = sorted(glob.glob(os.path.expanduser("~/.pi/agent/install/releases/*/node_modules/@earendil-works/pi-ai")))
    if not rel:
        sys.exit("pi's install not found (--pi DIR: the directory holding node_modules/@earendil-works)")
    return rel[-1].split("/node_modules/")[0]


def sessions_info(files):
    """per session file: assistant replies, thinking blocks, compactions, size (no content kept)"""
    out = []
    for f in files:
        na = nt = nc = 0
        try:
            for line in open(f, encoding="utf-8", errors="replace"):
                if '"type":"compaction"' in line:
                    nc += 1
                if '"role":"assistant"' in line:
                    na += 1
                    nt += '"type":"thinking"' in line
        except OSError:
            continue
        out.append({"file": f, "replies": na, "thinking": nt, "compactions": nc, "bytes": os.path.getsize(f)})
    return out


def pick(n, min_replies, max_bytes, need_thinking):
    files = glob.glob(os.path.expanduser("~/.pi/agent/sessions/*/*.jsonl"))
    now = time.time()
    files = [f for f in files if now - os.path.getmtime(f) > 3600]  # (not a session being written)
    info = [s for s in sessions_info(files) if s["replies"] >= min_replies and s["bytes"] <= max_bytes and
            (not need_thinking or s["thinking"] >= s["replies"] // 2)]
    info.sort(key=lambda s: (-s["replies"], s["file"]))
    return info[:n]


def start_server(a, log):
    cmd = [a.runq, "serve", a.model, "--port", "0", "-c", str(a.ctx), "--slots", str(a.slots), "--parallel", str(a.parallel)]
    cmd += a.server_arg
    if a.server_verbose:
        cmd.append("-v")
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads))
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(log, "w"), env=env)
    for _ in range(600):
        time.sleep(0.1)
        m = re.search(r"listening on http://127\.0\.0\.1:(\d+)", open(log, errors="replace").read())
        if m:
            return p, int(m.group(1))
        if p.poll() is not None:
            break
    sys.exit("runq serve did not start:\n" + open(log, errors="replace").read()[-2000:])


def table(recs, out=sys.stdout, per_request=False):
    """the per-session summary (and per request): what was prefilled against what a perfect cache would prefill"""
    sessions = {}
    for r in recs:
        sessions.setdefault(r["session"], []).append(r)
    tot = {}
    for sid, rs in sessions.items():
        prev = None
        rows = []
        for r in rs:
            if r["kind"] not in ("turn", "summary", "turn_prefix", "probe") or "usage" not in r:
                if r["kind"] == "compaction":
                    rows.append(("compaction", r))
                continue
            u, t = r["usage"], r.get("timings", {})
            n, cached, comp = u["prompt_tokens"], u["prompt_tokens_details"]["cached_tokens"], u["completion_tokens"]
            pre = t.get("prompt_n", n - cached)
            reply = None
            if r["kind"] == "turn":
                kind = "post-compaction" if r.get("after_compaction") else "user" if r.get("after") == "user" else \
                    "system-change" if r.get("after") == "system" else "tool"
                new = n - (prev["usage"]["prompt_tokens"] + prev["usage"]["completion_tokens"]) if prev and kind == "tool" else None
                reply = prev["usage"]["completion_tokens"] + 5 if prev else None  # (+ its header: <|im_start|>assistant\n<think>\n)
                prev = r
            else:
                kind, new = r["kind"], None
            if new is None or new < 1:
                new = None
            div = t.get("common_tokens", 0) < t.get("slot_tokens", 0) and t.get("common_tokens", 0) < n - 1
            rb = t.get("common_tokens", cached) - cached
            beyond = max(0, pre - new - reply) if new is not None and reply is not None else 0
            sp = t.get("speculative") or {}
            rows.append((kind, dict(r, n=n, cached=cached, comp=comp, pre=pre, new=new, div=div, rb=rb, beyond=beyond,
                                    pms=t.get("prompt_ms", 0), dms=t.get("predicted_ms", 0),
                                    sp=[sp.get(src, {}).get(k, 0) for src in ("model", "lookup") for k in ("drafted", "accepted")])))
        agg = {}
        for kind, x in rows:
            if kind == "compaction":
                continue
            g = agg.setdefault(kind, {"n": 0, "prompt": 0, "pre": 0, "new": 0, "extra": 0, "div": 0, "rb": 0, "pms": 0.0, "exact": 0,
                                      "beyond": 0, "comp": 0, "dms": 0.0, "md": 0, "ma": 0, "ld": 0, "la": 0})
            g["n"] += 1
            g["comp"] += x["comp"]
            g["dms"] += x["dms"]
            for k, v in zip(("md", "ma", "ld", "la"), x["sp"]):
                g[k] += v
            g["prompt"] += x["n"]
            g["pre"] += x["pre"]
            g["pms"] += x["pms"]
            g["div"] += x["div"]
            g["rb"] += x["rb"] > 0
            if x["new"] is not None:
                g["new"] += x["new"]
                g["extra"] += x["pre"] - x["new"]
                g["exact"] += x["pre"] - x["new"] <= 1
                g["beyond"] += x["beyond"] > 0
        name = rs[0].get("session_name", sid)
        print(f"\nsession {name}: {sum(1 for k, _ in rows if k == 'compaction')} compaction(s)", file=out)
        print(f"  {'kind':<16}{'reqs':>6}{'prompt tok':>12}{'prefilled':>11}{'new':>10}{'extra':>9}{'exact':>7}{'beyond':>7}{'diverged':>9}"
              f"{'rollback':>9}{'prefill s':>10}{'reply tok':>10}{'decode s':>9}  speculation: model, lookup accepted / drafted", file=out)
        for kind, g in sorted(agg.items()):
            tool = kind == "tool"
            newc = f"{g['new']:>10}" if tool else f"{'-':>10}"
            extra = f"{g['extra']:>9}" if tool else f"{'-':>9}"
            exact = f"{g['exact']:>7}" if tool else f"{'-':>7}"
            beyond = f"{g['beyond']:>7}" if tool else f"{'-':>7}"
            spec = f"  {g['ma']} / {g['md']}, {g['la']} / {g['ld']}" if g["md"] + g["ld"] else ""
            print(f"  {kind:<16}{g['n']:>6}{g['prompt']:>12}{g['pre']:>11}{newc}{extra}{exact}{beyond}{g['div']:>9}{g['rb']:>9}"
                  f"{g['pms'] / 1e3:>10.1f}{g['comp']:>10}{g['dms'] / 1e3:>9.1f}{spec}", file=out)
            t = tot.setdefault(kind, dict.fromkeys(g, 0))
            for k in g:
                t[k] += g[k]
        if per_request:
            print(f"  {'#':>4} {'kind':<16}{'prompt':>8}{'cached':>8}{'pre':>7}{'new':>7}{'reply':>7}{'slot':>5}{'slot tok':>9}"
                  f"{'common':>8}{'pre s':>8}{'dec s':>7}{'tok/s':>6}  flags", file=out)
            for kind, x in rows:
                if kind == "compaction":
                    print(f"  {x['seq']:>4} compaction ({x.get('reason')}, ~{x.get('tokens_before')} tokens before)", file=out)
                    continue
                t = x.get("timings", {})
                flags = ("diverged " if x["div"] else "") + (f"rollback {x['rb']} " if x["rb"] > 0 else "") + \
                        ("copied " if t.get("copied") else "") + (f"beyond the last reply {x['beyond']} " if x["beyond"] else "") + \
                        (f"spec {x['sp'][1]}/{x['sp'][0]} + {x['sp'][3]}/{x['sp'][2]}" if x["sp"][0] + x["sp"][2] else "")
                print(f"  {x['seq']:>4} {kind:<16}{x['n']:>8}{x['cached']:>8}{x['pre']:>7}{x['new'] if x['new'] is not None else '-':>7}"
                      f"{x['comp']:>7}{t.get('slot', -1):>5}{t.get('slot_tokens', 0):>9}{t.get('common_tokens', 0):>8}"
                      f"{x['pms'] / 1e3:>8.1f}{x['dms'] / 1e3:>7.1f}{x['comp'] / max(x['dms'] / 1e3, 1e-9) if x['dms'] else 0:>6.1f}  {flags}", file=out)
    if len(sessions) > 1:
        print("\nall sessions:", file=out)
        for kind, g in sorted(tot.items()):
            extra = f"new {g['new']}, extra {g['extra']}, exact {g['exact']}/{g['n']}, beyond the last reply {g['beyond']}, " if kind == "tool" else ""
            spec = f", speculation: model {g['ma']} / {g['md']}, lookup {g['la']} / {g['ld']} accepted" if g["md"] + g["ld"] else ""
            print(f"  {kind:<16}{g['n']:>5} requests, prompt {g['prompt']}, prefilled {g['pre']} ({extra}diverged {g['div']}, "
                  f"rollback {g['rb']}), prefill {g['pms'] / 1e3:.1f} s, {g['comp']} reply tokens in {g['dms'] / 1e3:.1f} s{spec}", file=out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runq", default=os.path.join(ROOT, "runq"))
    ap.add_argument("--model")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--server-arg", action="append", default=[], help="an extra `runq serve` argument (repeatable)")
    ap.add_argument("--server-verbose", action="store_true", help="run the server with -v; its [engine] lines (where a "
                    "prompt left the cached tokens, with the tokens around) are kept next to the results")
    ap.add_argument("--session", action="append", default=[], help="a session file (repeatable)")
    ap.add_argument("--pick", type=int, default=0, help="pick N sessions (the most replies; --min-replies, --max-kb, thinking)")
    ap.add_argument("--min-replies", type=int, default=20)
    ap.add_argument("--max-kb", type=int, default=4000)
    ap.add_argument("--any-thinking", action="store_true", help="also pick sessions recorded without reasoning")
    ap.add_argument("--mode", choices=["force", "live"], default="force")
    ap.add_argument("--live-tokens", type=int, default=512)
    ap.add_argument("--max-requests", type=int, default=0, help="per session (0: all)")
    ap.add_argument("--probe-every", type=int, default=0, help="--mode force: before every Nth reply after tool results a "
                    "live request with the same context (kind probe, its reply dropped): decode and speculation there")
    ap.add_argument("--probe-tokens", type=int, default=256, help="a probe's max_tokens")
    ap.add_argument("--reserve", type=int, help="pi's compaction.reserveTokens (default min(16384, ctx / 4))")
    ap.add_argument("--keep", type=int, help="pi's compaction.keepRecentTokens (default min(20000, ctx / 4))")
    ap.add_argument("--pi", help="pi's install directory (default: the newest ~/.pi/agent/install/releases/*)")
    ap.add_argument("--out", default="/var/tmp/runq-replay")
    ap.add_argument("--tag", default=time.strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--per-request", action="store_true", help="the table per request too")
    ap.add_argument("--report", help="print the table of an earlier run's JSON lines and exit")
    a = ap.parse_args()
    if a.report:
        table([json.loads(l) for l in open(a.report)], per_request=a.per_request)
        return 0
    if not a.model:
        ap.error("--model is required")
    if not shutil.which("node"):
        sys.exit("node is required (pi's code runs the requests)")
    files = list(a.session)
    if a.pick:
        files += [s["file"] for s in pick(a.pick, a.min_replies, a.max_kb * 1024, not a.any_thinking)]
    if not files:
        ap.error("no session: --session FILE or --pick N")
    os.makedirs(a.out, exist_ok=True)
    res = os.path.join(a.out, f"{a.tag}.jsonl")
    log = os.path.join(a.out, f"{a.tag}.server.log")
    srv, port = start_server(a, log)
    pdir = pi_dir(a.pi)
    recs = []
    try:
        with open(res, "w") as fo:
            for f in files:
                sid = hashlib.sha1(os.path.abspath(f).encode()).hexdigest()[:10]  # (no path in the results)
                info = sessions_info([f])[0]
                name = f"{sid} ({info['replies']} replies, {info['thinking']} with reasoning, {info['bytes'] // 1024} KB)"
                print(f"session {name} ...", file=sys.stderr, flush=True)
                cmd = ["node", os.path.join(ROOT, "scripts", "replay_pi.mjs"), "--pi", pdir, "--session", f,
                       "--url", f"http://127.0.0.1:{port}/v1", "--model", "runq", "--ctx", str(a.ctx), "--mode", a.mode,
                       "--live-tokens", str(a.live_tokens), "--max-requests", str(a.max_requests),
                       "--probe-every", str(a.probe_every), "--probe-tokens", str(a.probe_tokens)]
                if a.reserve is not None:
                    cmd += ["--reserve", str(a.reserve)]
                if a.keep is not None:
                    cmd += ["--keep", str(a.keep)]
                t0 = time.time()
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
                for line in p.stdout:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    r["session"], r["session_name"] = sid, name
                    recs.append(r)
                    fo.write(json.dumps(r) + "\n")
                    fo.flush()
                p.wait()
                print(f"  {sum(1 for r in recs if r['session'] == sid)} records, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    finally:
        srv.send_signal(signal.SIGTERM)
        try:
            srv.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv.kill()
    if a.server_verbose:  # (only the engine's lines: the request bodies are the sessions' content)
        with open(log, errors="replace") as fi, open(log + ".engine", "w") as fo:
            for line in fi:
                if line.startswith("[engine]") or line.startswith("runq serve:"):
                    fo.write(line)
        os.replace(log + ".engine", log)
    table(recs, per_request=a.per_request)
    print(f"\nresults: {res} (server log: {log})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
