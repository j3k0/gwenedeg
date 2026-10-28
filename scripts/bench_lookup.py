#!/usr/bin/env python3
"""tok/s of plain decoding, the model draft, the prompt lookup and both combined (TODO G6, docs/qwen38-speculative.md,
"Prompt lookup") on code-editing prompts (the reply rewrites code of the prompt with a change) and one chat prompt
(little to copy): `runq -m chat -t 0 -n N` per prompt and setting, the settings rotated within each rep, the generated
tok/s and the speculative counters parsed from stderr; every speculative run's stdout checked against plain's.

  .venv/bin/python -I scripts/bench_lookup.py TARGET.gguf [--draft DRAFT.gguf] [-n 384] [-r 2] [--threads 8]
      [--settings 'plain;-G 3;--lookup 3;...'] [--prompts edit_py,edit_c,chat]

Run it under the bench lock (flock /tmp/cwayn-bench.lock ...); settings with -G get -D DRAFT.
"""
import argparse
import os
import re
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PY = '''def summarize(records, key, n):
    """Group records by key and return, per group, the moving average of the
    last n values, the count and the maximum."""
    groups = {}
    for r in records:
        k = r.get(key)
        if k is None:
            continue
        groups.setdefault(k, []).append(r["value"])
    out = {}
    for k, values in groups.items():
        avg = []
        for i in range(len(values)):
            if i < n - 1:
                avg.append(None)
            else:
                avg.append(sum(values[i - n + 1:i + 1]) / n)
        out[k] = {"count": len(values), "max": max(values), "avg": avg}
    return out
'''
C = '''typedef struct {
    int *buf;
    int cap, head, len;
} Ring;

int ring_init(Ring *r, int cap) {
    r->buf = malloc((size_t)cap * sizeof(int));
    if (!r->buf) return -1;
    r->cap = cap;
    r->head = r->len = 0;
    return 0;
}

int ring_push(Ring *r, int x) {
    if (r->len == r->cap) return -1;
    r->buf[(r->head + r->len) % r->cap] = x;
    r->len++;
    return 0;
}

int ring_pop(Ring *r, int *x) {
    if (r->len == 0) return -1;
    *x = r->buf[r->head];
    r->head = (r->head + 1) % r->cap;
    r->len--;
    return 0;
}

void ring_free(Ring *r) {
    free(r->buf);
    r->buf = NULL;
    r->cap = r->head = r->len = 0;
}
'''
PROMPTS = {
    "edit_py": "Rewrite this function so that the parameter n is called window and missing averages are 0.0 instead "
               "of None. Return the full function, nothing else.\n\n```python\n" + PY + "```",
    "edit_c": "Rename the field cap to capacity everywhere in this C code and make ring_push overwrite the oldest "
              "element when the buffer is full. Return the complete code, nothing else.\n\n```c\n" + C + "```",
    "chat": "What are the main differences between TCP and UDP?",
}
SETTINGS = "plain;-G 3;--lookup 3;--lookup 8;-G 3 --lookup 8 --lookup-min 4"


def run(a, prompt, setting):
    args = [] if setting == "plain" else setting.split()
    cmd = [a.runq, a.target, "-m", "chat", "-t", "0", "-n", str(a.n), "-c", "4096", "-i", PROMPTS[prompt]] + args
    if "-G" in args:
        cmd += ["-D", a.draft]
    p = subprocess.run(cmd, capture_output=True, env=dict(os.environ, OMP_NUM_THREADS=str(a.threads)))
    if p.returncode:
        sys.exit(f"{' '.join(cmd[:4])} {setting}: exit status {p.returncode}: {p.stderr.decode()[-300:]}")
    err = p.stderr.decode()
    m = re.search(r"generated: (\d+) tok in ([\d.]+) s", err)
    spec = re.search(r"speculative: .*?: (\d+) passes, (\d+) drafted, (\d+) accepted.*?([\d.]+) tok/pass", err)
    lk = re.search(r"lookup: (\d+) passes, (\d+) drafted, (\d+) accepted", err)
    return p.stdout, int(m.group(1)) / float(m.group(2)), spec, lk


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("target")
    ap.add_argument("--draft")
    ap.add_argument("-n", type=int, default=384)
    ap.add_argument("-r", "--reps", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--settings", default=SETTINGS)
    ap.add_argument("--prompts", default="edit_py,edit_c,chat")
    ap.add_argument("--runq", default=os.path.join(ROOT, "runq"))
    a = ap.parse_args()
    settings = [s.strip() for s in a.settings.split(";")]
    if not a.draft:
        settings = [s for s in settings if "-G" not in s]
    prompts = a.prompts.split(",")
    res = {(p, s): [] for p in prompts for s in settings}
    info, ref = {}, {}
    for rep in range(a.reps):
        for p in prompts:
            for s in settings:
                out, tps, spec, lk = run(a, p, s)
                if s == "plain":
                    ref.setdefault(p, out)
                elif p in ref and out != ref[p]:
                    sys.exit(f"{p} / {s}: the output differs from plain decoding's")
                res[(p, s)].append(tps)
                if spec:
                    d, acc = int(spec.group(2)), int(spec.group(3))
                    info[(p, s)] = f"{100 * acc / max(d, 1):.0f}%, {float(spec.group(4)):.2f}" + (
                        f", L {lk.group(1)}/{spec.group(1)}" if lk and "-G" in s else "")
                print(f"  rep {rep + 1} {p:8s} {s:34s} {tps:6.1f} tok/s {info.get((p, s), '')}", file=sys.stderr,
                      flush=True)
    print(f"target {os.path.basename(a.target)}, draft {os.path.basename(a.draft) if a.draft else '-'}, -n {a.n}, "
          f"{a.threads} threads, {a.reps} reps (median [min-max] tok/s; accepted / guessed, tokens per pass, lookup "
          f"passes / passes); every output == plain's")
    print(f"| setting | " + " | ".join(prompts) + " | mean |")
    print("|---|" + "---|" * (len(prompts) + 1))
    for s in settings:
        cells, meds = [], []
        for p in prompts:
            v = res[(p, s)]
            med = statistics.median(v)
            meds.append(med)
            cells.append(f"{med:.1f} [{min(v):.0f}-{max(v):.0f}]" + (f" ({info[(p, s)]})" if (p, s) in info else ""))
        print(f"| `{s}` | " + " | ".join(cells) + f" | {statistics.mean(meds):.1f} |")


if __name__ == "__main__":
    main()
