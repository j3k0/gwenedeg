#!/usr/bin/env python3
"""Speed and memory of runq per model file (Step 11's table): prefill of a ~450- and a ~1,500-token chat prompt,
decode after a short prompt and after the 1,500-token one (128 tokens, greedy), peak RSS of the 1,500-token run,
startup (`-n 1 -i hi`, wall time), per thread count; best of --reps rounds, the models interleaved within a
round (the box is shared: compare numbers of one session only).

Usage: scripts/bench_models.py [-t 8,12] [-r 2] [--runq ./runq] MODEL.gguf ...
Prompts: scripts/bench_prefill.sh -P 450 / -P 1500 (the fixed paragraph cut to length). Prints a markdown table.
"""
import argparse
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMING = re.compile(r"\[timing\] prompt: (\d+) tok in [\d.]+ s \(([\d.]+) tok/s\) \| generated: (\d+) tok in [\d.]+ s \(([\d.]+) tok/s\)")


def prompt(n):
    return subprocess.run([os.path.join(ROOT, "scripts", "bench_prefill.sh"), "-P", str(n)], capture_output=True,
                          text=True, check=True).stdout.rstrip("\n")


def run(runq, model, threads, text, n):
    """(prompt tokens, prefill tok/s, decode tok/s, peak RSS MB, wall s)"""
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    t0 = time.time()
    p = subprocess.Popen([runq, model, "-m", "chat", "-t", "0", "-c", "4096", "-n", str(n), "-i", text], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    err = p.stderr.read()
    _, status, ru = os.wait4(p.pid, 0)
    wall = time.time() - t0
    m = TIMING.search(err)
    if status != 0 or not m:
        raise RuntimeError(f"{model}: {err[-300:]}")
    return int(m.group(1)), float(m.group(2)), float(m.group(4)), ru.ru_maxrss / 1024, wall


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("models", nargs="+")
    ap.add_argument("-t", "--threads", default="8,12")
    ap.add_argument("-r", "--reps", type=int, default=2)
    ap.add_argument("--runq", default=os.path.join(ROOT, "runq"))
    a = ap.parse_args()
    threads = [int(x) for x in a.threads.split(",")]
    p450, p1500 = prompt(450), prompt(1500)
    short = "Write a short story about a lighthouse keeper."
    best = {}

    def keep(key, val, hi=True):
        old = best.get(key)
        best[key] = val if old is None else (max(old, val) if hi else min(old, val))
    print(f"load {os.getloadavg()[0]:.1f}", file=sys.stderr)
    for r in range(a.reps):
        for t in threads:
            for m in a.models:
                name = os.path.basename(m)
                n, pf, _, _, _ = run(a.runq, m, t, p450, 1)
                keep((name, t, "pf450"), pf)
                keep((name, t, "n450"), n)
                _, _, dec, _, _ = run(a.runq, m, t, short, 128)
                keep((name, t, "dec"), dec)
                n, pf, dec, rss, _ = run(a.runq, m, t, p1500, 128)
                keep((name, t, "pf1500"), pf)
                keep((name, t, "n1500"), n)
                keep((name, t, "dec1500"), dec)
                keep((name, t, "rss"), rss, hi=False)
                _, _, _, _, wall = run(a.runq, m, t, "hi", 1)
                keep((name, t, "start"), wall, hi=False)
                print(f"round {r + 1} {name} {t} threads done (load {os.getloadavg()[0]:.1f})", file=sys.stderr, flush=True)
    print(f"| file | threads | prefill {best[(os.path.basename(a.models[0]), threads[0], 'n450')]} / "
          f"{best[(os.path.basename(a.models[0]), threads[0], 'n1500')]} tok (tok/s) | decode short / after 1,500 (tok/s) "
          f"| peak RSS (1,500 + 128, -c 4096) | startup (`-n 1`) |")
    print("|---|---|---|---|---|---|")
    for m in a.models:
        name = os.path.basename(m)
        for t in threads:
            g = lambda k: best[(name, t, k)]  # noqa: E731
            print(f"| {name} | {t} | {g('pf450'):,.0f} / {g('pf1500'):,.0f} | {g('dec'):,.1f} / {g('dec1500'):,.1f} | "
                  f"{g('rss'):,.0f} MB | {g('start'):.2f} s |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
