#!/usr/bin/env python3
"""Compares two builds of runq on the CLI's MTP drafter (-D mtp; TODO S9c.1, CMP-MTP), which tests/compare_bins.sh does not
run: tests/check_spec.py's MTP_CASES x MTP_K (imported, not copied) on build/synth-q35's synth-q35-27b-mtp.gguf and
synth-q35-ista-mtp.gguf (built here as check_models does, cached) and the trained Qwen3.5-0.8B-mtp (if present), each run
with both binaries. Compared: the exit status, stdout and the RUNQ_CHAT_DUMP files (the cache's ids, every sampled logits
vector) byte for byte, stderr line by line with the [timing] lines' times masked: the speculative line keeps its counts
(passes, drafted, accepted, tok/pass, draft steps, "| mtp rows N (approx M)") but its seconds; with --spec auto (its
choices follow the costs calibrated at the start: they vary run to run) every number of it but the approximate rows;
the prompt and spec-auto costs lines without numbers. Not part of the suite (a refactoring's check, as compare_bins).

Usage: .venv/bin/python -I tests/compare_mtp.py --ref REF_RUNQ --new NEW_RUNQ [-j 3] [--threads 4] [--q35 Q8_0.gguf]
                                               [--mtp08 FILE]
Prints each difference and "N of N identical" (per file and in all); exit status 0 if every run is identical, 1 if
not, 2 on a setup error (a missing binary or source file).
"""
import argparse
import concurrent.futures as cf
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import check_models  # noqa: E402  (q35_synth_build)
import check_spec  # noqa: E402  (MTP_CASES, MTP_K, CLI_CASES, cli_run)

Q35 = os.path.join(ROOT, "models", "Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q8_0.gguf")
MTP08 = os.path.join(ROOT, "models", "Qwen3.5-0.8B-MTP-GGUF", "Qwen3.5-0.8B-mtp-Q8_0.gguf")


def masked(err, auto):
    """stderr's lines with the [timing] lines' run-dependent numbers masked (module doc)"""
    out = []
    for l in err.decode(errors="replace").split("\n"):
        if l.startswith("[timing] speculative"):
            if auto:
                ap = re.search(r"\(approx \d+\)", l)
                l = re.sub(r"\d+(\.\d+)?", "N", l) + (" " + ap.group(0) if ap else "")
            else:
                l = re.sub(r"\d+\.\d+ s\b", "N s", l)
        elif l.startswith("[timing]"):
            l = re.sub(r"\d+(\.\d+)?", "N", l)
        out.append(l)
    return out


def compare(ref, new, auto):
    """the reasons two runs differ (empty: identical)"""
    why = []
    if ref[0] != new[0]:
        why.append(f"exit status {new[0]} vs {ref[0]}")
    if ref[1] != new[1]:
        i = next((j for j, (x, y) in enumerate(zip(new[1], ref[1])) if x != y), min(len(new[1]), len(ref[1])))
        why.append(f"stdout differs at byte {i}: {new[1][i:i + 30]!r} vs {ref[1][i:i + 30]!r}")
    a, b = masked(ref[2], auto), masked(new[2], auto)
    if a != b:
        i = next((j for j, (x, y) in enumerate(zip(b, a)) if x != y), min(len(a), len(b)))
        why.append(f"stderr line {i}: {(b[i] if i < len(b) else '')[:160]!r} vs {(a[i] if i < len(a) else '')[:160]!r}")
    if ref[3] != new[3]:
        why.append("RUNQ_CHAT_DUMP " + ("ids" if ref[3] is None or new[3] is None or ref[3][0] != new[3][0] else "logits") + " differ")
    return why


def job(ref, new, model, case, k, tmp, tag, threads):
    name, args, users, _, _ = case
    r = check_spec.cli_run(ref, model, args, users, tmp, tag + "_ref", "mtp", k, threads)
    n = check_spec.cli_run(new, model, args, users, tmp, tag + "_new", "mtp", k, threads)
    return compare(r, n, "auto" in k)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", required=True, help="the reference runq")
    ap.add_argument("--new", required=True, help="the runq to compare")
    ap.add_argument("-j", type=int, default=3, help="jobs at once (each runs both binaries in turn)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--q35", default=Q35, help="the Qwen3.5-0.8B Q8_0 file (the synthetic files' source)")
    ap.add_argument("--mtp08", default=MTP08, help="the trained MTP file (skipped if missing)")
    a = ap.parse_args()
    for x in (a.ref, a.new):
        if not os.access(x, os.X_OK):
            print(f"missing {x}", file=sys.stderr)
            return 2
    if not os.path.exists(a.q35):
        print(f"missing {a.q35}", file=sys.stderr)
        return 2
    try:
        sd = check_models.q35_synth_build(argparse.Namespace(q35=a.q35))
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 2
    files = [os.path.join(sd, f) for f in ("synth-q35-27b-mtp.gguf", "synth-q35-ista-mtp.gguf")] + \
        ([a.mtp08] if os.path.exists(a.mtp08) else [])
    for f in files:
        if not os.path.exists(f):
            print(f"missing {f}", file=sys.stderr)
            return 2
    if not os.path.exists(a.mtp08):
        print(f"note: no {a.mtp08}: the trained block skipped")
    cases = [next(c for c in check_spec.CLI_CASES if c[0] == n) for n in check_spec.MTP_CASES]
    tot = same = 0
    with tempfile.TemporaryDirectory(prefix="cmpmtp.") as tmp, cf.ThreadPoolExecutor(max_workers=a.j) as ex:
        futs = []
        for fi, f in enumerate(files):
            for ci, c in enumerate(cases):
                for ki, k in enumerate(check_spec.MTP_K):
                    futs.append((f, c[0], k, ex.submit(job, a.ref, a.new, f, c, k, tmp, f"f{fi}c{ci}k{ki}", a.threads)))
        per = {}
        for f, name, k, fu in futs:
            why = fu.result()
            n = per.setdefault(f, [0, 0])
            n[0] += 1
            tot += 1
            if why:
                print(f"DIFF {os.path.basename(f)} {name} {' '.join(k)}: {'; '.join(why)}", flush=True)
            else:
                n[1] += 1
                same += 1
        for f in files:
            print(f"{os.path.basename(f)}: {per[f][1]} of {per[f][0]} identical")
    print(f"{same} of {tot} identical")
    return 0 if same == tot else 1


if __name__ == "__main__":
    sys.exit(main())
