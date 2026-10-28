"""Group spec of tests/check.py: speculative decoding (TODO S1, src/spec.c) == plain decoding.

  api        tests/api_test --spec on the Qwen3-0.6B pair (target Q8_0, draft Q4_K_M; every case and draft setting),
             the generic build's api_test --spec --quick, and Qwen3.5-0.8B Q8_0 + its Q4_K_M as the draft (--quick:
             the hybrid rollbacks, every one on its checkpoint); the vocabulary check refuses the other family
  cli        runq -D DRAFT -G k [--tau], --lookup N (the prompt lookup alone, no -D) and both combined vs the same
             command without them: stdout byte for byte, stderr but the [timing] lines, the exit status, and in chat
             mode the RUNQ_CHAT_DUMP files (the cache's ids and every sampled logits vector, bitwise): generate and
             chat, greedy and seeded (top-p, top-k), thinking off / on, -n inside a verify pass, -n 1 / 2, the
             context's end (generate, and chat turns dropped at -c 160), -b 1, piped multi-turn chats, a generation
             that copies its prompt (lookup: >= 3 tokens per pass); -G 0 with -D is plain decoding byte for byte
             (stderr included); a draft of another vocabulary, -G without -D and a bad --lookup / --lookup-min are
             refused; one generate and one chat case on the generic build
  server     runq serve with speculation (tests/server_test.py --spec: the Qwen3-0.6B pair, Qwen3.5-0.8B Q8_0 + Q4_K_M, the
             0.8B with the prompt lookup alone): every request with "speculative": false and with fixed settings / auto,
             outputs identical (content, reasoning, tool calls, finish, usage, logprobs; streamed and not; concurrent
             requests; a conversation's next turn reuses what plain decoding's would)
"""
import concurrent.futures as cf
import os
import subprocess
import sys

SPEC_THREADS = 4  # per runq / api_test process (the runs go 3 at a time, in the background from the start)
GEN_THREADS = 3

GEN_PROMPT = "def quicksort(arr):"
STORY = "Once upon a time, in a small village by the sea,"
CHAT_USERS = ["Name three colours.", "And three animals?", "Which of those is the largest?"]
# (name, args after the model, stdin lines or None, draft settings, flags) -- every case also runs without -D
# (the reference); draft settings: lists of extra args
# (-D DRAFT is passed with the settings that have -G; --lookup alone: no draft model)
LOOKUP = ["--lookup", "4"]
COMBINED = ["-G", "3", "--lookup", "8", "--lookup-min", "3"]
K_ALL = [["-G", "1"], ["-G", "3"], ["-G", "8"], ["-G", "8", "--tau", "0.5"], LOOKUP, ["--lookup", "30", "--lookup-min", "4"],
         COMBINED]
K_TWO = [["-G", "3"], ["-G", "8", "--tau", "0.5"], LOOKUP, COMBINED]
COPY_PROMPT = ("def moving_average(xs, n):\n    out = []\n    for i in range(len(xs)):\n        if i < n - 1:\n"
               "            out.append(None)\n        else:\n            out.append(sum(xs[i-n+1:i+1]) / n)\n    return out\n\n\n"
               "# the same function again, copied verbatim:\n\ndef moving_average(xs, n):\n")
CLI_CASES = [
    ("gen_greedy", ["-m", "generate", "-t", "0", "-n", "60", "-i", GEN_PROMPT], None, K_ALL, {}),
    ("gen_seeded_topk", ["-m", "generate", "-t", "1", "-p", "0.95", "-K", "20", "-s", "7", "-n", "50", "-i", STORY], None, K_TWO, {}),
    ("gen_seeded_n2", ["-m", "generate", "-t", "0.8", "-s", "3", "-n", "2", "-i", STORY], None, K_TWO, {}),
    ("gen_n1", ["-m", "generate", "-t", "0", "-n", "1", "-i", STORY], None, K_TWO, {}),
    ("gen_context", ["-m", "generate", "-t", "0", "-c", "37", "-i", GEN_PROMPT], None, K_TWO, {}),
    ("gen_seeded_context_b1", ["-m", "generate", "-t", "1", "-K", "20", "-s", "9", "-c", "41", "-b", "1", "-i", STORY], None, K_TWO, {}),
    ("chat_greedy", ["-m", "chat", "-t", "0", "-n", "64"], CHAT_USERS, K_ALL, {"dump": True}),
    ("chat_seeded_think", ["-m", "chat", "-r", "1", "-t", "1", "-p", "0.95", "-K", "20", "-s", "5", "-n", "70"],
     CHAT_USERS[:2], K_TWO, {"dump": True}),
    ("chat_seeded_system_n13", ["-m", "chat", "-t", "0.7", "-s", "11", "-n", "13", "-y", "Be brief."], CHAT_USERS, K_TWO, {"dump": True}),
    ("chat_context_full", ["-m", "chat", "-t", "0", "-c", "160", "-n", "30", "-y", "You are a terse assistant."],
     ["My name is Ada.", "Name three fruits.", "Name three animals.", "Name three colours.", "Name three cities.",
      "What is my name?"], K_TWO, {"dump": True, "dropped": True}),
    ("chat_reply_cut", ["-m", "chat", "-t", "1", "-s", "8", "-c", "100", "-i", "Write a long story about a lighthouse."], None,
     K_TWO, {"dump": True, "cut": True}),
    ("chat_one_turn", ["-m", "chat", "-t", "0", "-r", "1", "-n", "48", "-i", "What is 17 * 23?"], None, K_TWO, {"dump": True}),
    ("gen_copy", ["-m", "generate", "-t", "0", "-n", "48", "-i", COPY_PROMPT], None, [["--lookup", "8"], COMBINED], {"copy": True}),
]
# on the generic build (fewer: its kernels are slower)
GENERIC_CASES = [("gen_greedy", ["-G", "3"]), ("chat_greedy", ["-G", "8", "--tau", "0.5"]), ("gen_copy", COMBINED)]
# Qwen3.5-0.8B Q8_0 + its Q4_K_M: a thinking chat whose second turn rolls the recurrent state back to a checkpoint
Q35_CASES = [("q35_chat_think", ["-m", "chat", "-r", "1", "-t", "0", "-n", "48"], CHAT_USERS[:2], K_TWO, {"dump": True}),
             ("q35_gen_seeded", ["-m", "generate", "-t", "1", "-K", "20", "-s", "4", "-n", "40", "-i", STORY], None, K_TWO, {})]


def run(cmd, threads, stdin=None, env=None):
    e = dict(os.environ, **(env or {}))
    e["OMP_NUM_THREADS"] = str(threads)
    return subprocess.run(cmd, env=e, capture_output=True, input=stdin)


def no_timing(err):
    return "\n".join(l for l in err.decode(errors="replace").split("\n") if not l.startswith("[timing]"))


def cli_run(runq, model, args, users, tmp, tag, draft=None, k=(), threads=SPEC_THREADS):
    """one runq run: (exit status, stdout, stderr, dump ids + logits bytes or None, spec stats line)"""
    prefix = os.path.join(tmp, f"spec_{tag}")
    stdin = ("\n".join(users) + "\n").encode() if users is not None else None
    cmd = [runq, model] + ([] if "-c" in args else ["-c", "4096"]) + list(args) + (["-D", draft] if draft else []) + list(k)
    p = run(cmd, threads, stdin, {"RUNQ_CHAT_DUMP": prefix})
    dump = None
    if os.path.exists(prefix + ".ids"):
        with open(prefix + ".ids", "rb") as f, open(prefix + ".logits", "rb") as g:
            dump = (f.read(), g.read())
        os.remove(prefix + ".ids")
        os.remove(prefix + ".logits")
    stats = [l for l in p.stderr.decode(errors="replace").splitlines() if l.startswith("[timing] speculative")]
    return p.returncode, p.stdout, p.stderr, dump, stats


def compare(ref, got, what, timing_ok=True):
    """(ok, detail) of a run against the reference: exit status, stdout, stderr but [timing] lines, dump files"""
    why = []
    if got[0] != ref[0]:
        why.append(f"exit status {got[0]} vs {ref[0]}: {got[2].decode(errors='replace')[-200:]!r}")
    if got[1] != ref[1]:
        i = next((j for j, (x, y) in enumerate(zip(got[1], ref[1])) if x != y), min(len(got[1]), len(ref[1])))
        why.append(f"stdout differs at byte {i}: {got[1][i:i + 30]!r} vs {ref[1][i:i + 30]!r}")
    if (no_timing(got[2]) if timing_ok else got[2]) != (no_timing(ref[2]) if timing_ok else ref[2]):
        why.append("stderr differs" + (" (without the [timing] lines)" if timing_ok else ""))
    if got[3] != ref[3]:
        why.append("RUNQ_CHAT_DUMP " + ("ids" if got[3] is None or ref[3] is None or got[3][0] != ref[3][0] else "logits") + " differ")
    return not why, "; ".join(why)


def acc(stats):
    """accepted / drafted over a run's speculative lines (and tokens per pass of the last)"""
    a = d = 0
    for l in stats:
        f = l.replace(",", "").split()
        d += int(f[f.index("drafted") - 1])
        a += int(f[f.index("accepted") - 1])
    return f"{a}/{d}"


def tok_per_pass(stats):
    f = stats[-1].replace(",", "").split() if stats else []
    return float(f[f.index("tok/pass") - 1]) if "tok/pass" in f else 0.0


def cli_case(runq, model, draft, case, tmp, tag, ks=None, suffix=""):
    name, args, users, kset, chk = case
    name += suffix
    ref = cli_run(runq, model, args, users, tmp, f"{tag}_ref")
    out = []
    if ref[0] != 0:
        return [(False, name, f"reference run failed: {ref[2].decode(errors='replace')[-300:]}")]
    detail, ok = [], True
    for k in (ks or kset):
        got = cli_run(runq, model, args, users, tmp, f"{tag}_{'_'.join(k)}", draft if "-G" in k else None, k)
        o, why = compare(ref, got, " ".join(k))
        o &= len(got[4]) == (len(users) if users is not None and "-i" not in args else 1)
        if chk.get("copy") and o and tok_per_pass(got[4]) < 3.0:
            o, why = False, f"a verbatim copy at {tok_per_pass(got[4])} tokens per pass (< 3)"
        ok &= o
        detail.append(f"{' '.join(k)}: {acc(got[4])}" + (f" ({tok_per_pass(got[4]):.2f} tok/pass)" if chk.get("copy") else "")
                      + ("" if o else f" DIFFER ({why or 'no speculative line'})"))
    if chk.get("dropped") and b"[context] full" not in ref[2]:
        ok = False
        detail.append("context full not reached")
    if chk.get("cut") and b"reply cut" not in ref[2]:
        ok = False
        detail.append("the reply did not reach the end of the context")
    what = "stdout" + (" + chat dumps (cache ids, sampled logits)" if chk.get("dump") else "") + " == plain"
    out.append((ok, name, f"{what}; accepted/drafted " + ", ".join(detail)))
    return out


def spec_start(a, tmp, ex):
    """start the group's runs on ex (api_test --spec, the CLI cases); futures"""
    futs = []
    if not (a.gguf and a.q4):
        return futs
    d = os.path.join(tmp, "spec")
    os.makedirs(d, exist_ok=True)
    q35d = os.path.join(os.path.dirname(a.q35), "Qwen3.5-0.8B-Q4_K_M.gguf") if a.q35 else ""
    if a.api_test:
        futs.append(("api", "spec_api/qwen3", ex.submit(run, [a.api_test, "--spec", a.gguf, a.q4] + (["--other", a.q35] if a.q35 else []),
                                                        SPEC_THREADS)))
        gen = a.api_test + "-generic"
        if os.path.exists(gen):
            futs.append(("api", "spec_api/qwen3_generic", ex.submit(run, [gen, "--spec", a.gguf, a.q4, "--quick"], GEN_THREADS)))
        if a.q35 and os.path.exists(q35d):
            futs.append(("api", "spec_api/qwen35", ex.submit(run, [a.api_test, "--spec", a.q35, q35d, "--quick", "--other", a.gguf],
                                                             SPEC_THREADS)))
    # runq serve with speculation (server_test.py --spec): the Qwen3 pair, the Qwen3.5 pair, the lookup alone
    srv = [sys.executable, "-I", os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_test.py"), "--runq", a.runq,
           "--threads", str(SPEC_THREADS)]
    futs.append(("api", "spec_server/qwen3", ex.submit(run, srv + ["--model", a.gguf, "--spec", a.q4], SPEC_THREADS)))
    if a.q35 and os.path.exists(q35d):
        futs.append(("api", "spec_server/qwen35", ex.submit(run, srv + ["--model", a.q35, "--spec", q35d], SPEC_THREADS)))
    if a.q35:
        futs.append(("api", "spec_server/lookup", ex.submit(run, srv + ["--model", a.q35, "--spec", "none"], SPEC_THREADS)))
    for i, case in enumerate(CLI_CASES):
        futs.append(("cli", "spec_cli", ex.submit(cli_case, a.runq, a.gguf, a.q4, case, d, f"c{i}")))
    if a.generic:
        for name, k in GENERIC_CASES:
            case = next(c for c in CLI_CASES if c[0] == name)
            futs.append(("cli", "spec_cli", ex.submit(cli_case, a.generic, a.gguf, a.q4, case, d, f"g{name}", [k], "_generic")))
    if a.q35 and os.path.exists(q35d):
        for i, case in enumerate(Q35_CASES):
            futs.append(("cli", "spec_cli", ex.submit(cli_case, a.runq, a.q35, q35d, case, d, f"q{i}")))
    # -G 0 with -D: plain decoding, byte for byte (stderr included: nothing loaded, nothing printed)
    def g0():
        args = ["-m", "generate", "-t", "1", "-s", "2", "-n", "20", "-i", STORY]
        ref = cli_run(a.runq, a.gguf, args, None, d, "g0_ref")
        got = cli_run(a.runq, a.gguf, args, None, d, "g0", a.q4, ["-G", "0"])
        ok, why = compare(ref, got, "-G 0", timing_ok=True)
        # (the timing lines' numbers vary run to run: compared without their numbers)
        strip = lambda e: b"".join(c for c in [bytes([x]) for x in e] if not c.isdigit())
        ok &= strip(got[2]) == strip(ref[2])
        return [(ok, "g0", "-G 0 -D DRAFT == no -D: stdout, stderr (the [timing] line's numbers aside)" + (f"; {why}" if why else ""))]
    futs.append(("cli", "spec_cli", ex.submit(g0)))
    # refusals: a draft of another vocabulary (Qwen3.5 for Qwen3); -G without -D
    def refused():
        out = []
        if a.q35:
            p = run([a.runq, a.gguf, "-m", "generate", "-n", "4", "-i", "hi", "-D", a.q35], 2)
            e = p.stderr.decode(errors="replace")
            out.append((p.returncode == 1 and not p.stdout and "not usable as a draft" in e, "refused_vocab",
                        f"-D Qwen3.5 for Qwen3: exit status {p.returncode}, {e.strip()[-150:]!r}"))
        p = run([a.runq, a.gguf, "-m", "generate", "-n", "4", "-i", "hi", "-G", "3"], 2)
        e = p.stderr.decode(errors="replace")
        out.append((p.returncode == 1 and "needs a draft model" in e, "refused_no_draft", f"-G 3 without -D: exit status {p.returncode}"))
        p = run([a.runq, a.gguf, "-m", "generate", "-n", "4", "-i", "hi", "-D", a.q4, "--tau", "2"], 2)
        out.append((p.returncode == 1 and b"Usage:" in p.stderr, "bad_tau", f"--tau 2: exit status {p.returncode}"))
        bad = [["--lookup", "31"], ["--lookup", "-1"], ["--lookup", "4", "--lookup-min", "1"], ["--lookup", "x"]]
        codes = [run([a.runq, a.gguf, "-m", "generate", "-n", "4", "-i", "hi"] + b, 2) for b in bad]
        out.append((all(p.returncode == 1 and b"Usage:" in p.stderr for p in codes), "bad_lookup",
                    "; ".join(f"{' '.join(b)}: exit status {p.returncode}" for b, p in zip(bad, codes))))
        p = run([a.runq, a.gguf, "-m", "generate", "-n", "4", "-i", "hi", "--lookup", "4", "-G", "3"], 2)
        e = p.stderr.decode(errors="replace")
        out.append((p.returncode == 1 and "needs a draft model" in e, "refused_lookup_g", f"--lookup 4 -G 3 without -D: exit status {p.returncode}"))
        return out
    futs.append(("cli", "spec_cli", ex.submit(refused)))
    return futs


def test_spec(a, res, futs):
    if not futs:
        print("   (skipped: needs the GGUF Q8_0 and Q4_K_M files)")
        return
    for kind, name, f in futs:
        if kind == "api":
            p = f.result()
            lines = [l for l in p.stdout.decode(errors="replace").splitlines() if l.startswith(("PASS ", "FAIL "))]
            for l in lines:
                ok, rest = l.startswith("PASS "), l[5:]
                n, _, detail = rest.partition("  ")
                res.report(ok, f"{name}/{n.strip()}", detail.strip())
            if p.returncode not in (0, 1) or not lines:
                res.report(False, f"{name}/run", f"exit status {p.returncode}: {p.stderr.decode(errors='replace')[-300:]}")
        else:
            for ok, n, detail in f.result():
                res.report(ok, f"{name}/{n}", detail)


if __name__ == "__main__":
    sys.exit("run through tests/check.py --only spec")
