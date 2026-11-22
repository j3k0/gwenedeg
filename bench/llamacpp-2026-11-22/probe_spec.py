#!/usr/bin/env python3
"""llama.cpp draft-length probe: per setting a fresh llama-server, the bench/s9c lcb1 prompt (thinking on), greedy,
256 new tokens; prints decode tok/s and the draft counters. usage: probe_spec.py OUT.md"""
import http.client, json, subprocess, sys, time

R = "/opt/projects/tools/gwenedeg-llama"
M = R + "/models/Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
L = "llama.cpp/build/bin/llama-server"
BASE = f"{L} -m {M} -c 98304 -np 1 -t 14 -fa on -ctk q8_0 -ctv q4_0 -ub 512 -b 2048 -lm none --no-repack --ctx-checkpoints 4 -cram 512 --jinja --host 127.0.0.1 --port 18733"
SETTINGS = [
    ("no MTP", ""),
    ("mtp n_max 1", "--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-p-min 0.5"),
    ("mtp n_max 2", "--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.5"),
    ("mtp n_max 3", "--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.5"),
    ("mtp n_max 3 p_min 0 (defaults)", "--spec-type draft-mtp"),
    ("mtp n_max 4", "--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.5"),
    ("mtp n_max 6", "--spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0.5"),
]
prompt = open(R + "/bench/s9c/task-lcb1.txt").read()
out = open(sys.argv[1], "w")
out.write(f"base: {BASE.replace(M, '{model}').replace(L, 'llama-server')}\n\n| setting | prompt tok/s | decode tok/s | drafted | accepted | 1-min load |\n|---|---|---|---|---|---|\n")
for name, extra in SETTINGS:
    log = open(f"runs/probe-{name.replace(' ', '_')}.log", "w")
    p = subprocess.Popen((BASE + " " + extra).split(), stderr=log, stdout=subprocess.DEVNULL)
    for _ in range(300):
        try:
            c = http.client.HTTPConnection("127.0.0.1", 18733, timeout=2); c.request("GET", "/health")
            if b"ok" in c.getresponse().read(): break
        except OSError:
            pass
        time.sleep(1)
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 256, "temperature": 0}
    c = http.client.HTTPConnection("127.0.0.1", 18733, timeout=3600)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    t = json.loads(c.getresponse().read())["timings"]
    load = open("/proc/loadavg").read().split()[0]
    row = f"| {name} | {t['prompt_per_second']:.1f} | {t['predicted_per_second']:.2f} | {t.get('draft_n', '-')} | {t.get('draft_n_accepted', '-')} | {load} |"
    print(row, flush=True); out.write(row + "\n"); out.flush()
    p.terminate(); p.wait()
