#!/usr/bin/env python3
"""Prompt-lookup draft acceptance on agentic transcripts (TODO G0 (c), G6; docs/qwen38-speculative.md, "Prompt
lookup"): teacher-forced, would an n-gram lookup in the context (the prompt + the reply so far) have guessed the
next tokens, alone and combined with a draft model?

  sessions  pi's recorded sessions (~/.pi/agent/sessions/*/*.jsonl; local data: the output file stays local, only
            aggregate numbers are reported) rendered as Qwen3.8's template renders them for runq serve (system + the
            tools' instructions, user / tool turns, every assistant message with its reasoning, tool calls in the XML
            form: string arguments as they are, others as JSON, e.g. edit's `edits` array) and tokenized with the
            Qwen3.5 tokenizer (models/Qwen3.5-0.8B/tokenizer.json, = runq's); each session's main branch is one token
            stream (a compaction or a changed system prompt starts another), each assistant message a reply span whose
            tokens are labeled by kind: think (reasoning), prose (text), syntax (the template's and the XML's tokens),
            args (tool-call arguments), old / new (an edit's oldText / newText), write (write's content).
  draft     a draft GGUF teacher-forced on the streams (build/teacher_force: its argmax and top-1 probability at
            every reply position, the whole context before it); --max-pos / --budget keep it cheap.
  report    the loop simulated on the teacher-forced tokens: tokens per target pass, guesses per pass, accepted /
            guessed and the cost model's tok/s (spec_accept.py's), per kind, for the lookup rules (n-gram length,
            longest-first, k), the model draft (k, tau) and both combined (the lookup when it finds a match, else the
            model). --answers: also the saved 27B answers of spec_accept.py (GEN.json:MEAS.json pairs: their prompts
            re-rendered and tokenized; the model draft from MEAS.json).

The lookup (as src/spec.c's): at a pass whose last sampled token is t (the sequence S = context + reply so far + t),
for n = n_max down to n_min: the most recent earlier occurrence of S's last n tokens (ending before t), and the tokens
that followed it in S, at most k (and no further than t) are the guesses; none found: no guess (lookup alone: a plain
step; combined: the model's guesses).

Usage:
  .venv/bin/python -I scripts/lookup_accept.py sessions /var/tmp/streams.pkl [--glob '~/.pi/agent/sessions/*/*.jsonl']
      [--check-template logs/specmeas-2026-10-27/tmpl_38.jinja]
  .venv/bin/python -I scripts/lookup_accept.py draft /var/tmp/streams.pkl DRAFT.gguf -o /var/tmp/draft.npz
      [--max-pos 24576] [--budget 600000] [--threads 8]
  .venv/bin/python -I scripts/lookup_accept.py report [/var/tmp/streams.pkl] [--draft /var/tmp/draft.npz]
      [--answers GEN.json:MEAS.json ...] [--cost short|long] [--by-kind]

Python stdlib + numpy + tokenizers (+ jinja2 for --check-template).
"""
import argparse
import glob
import json
import os
import pickle
import struct
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import spec_accept  # noqa: E402  (render, EFFORT_LINE, IM_END)

KINDS = ["think", "prose", "syntax", "args", "old", "new", "write"]
K_ID = {k: i for i, k in enumerate(KINDS)}

# --- sessions: Qwen3.8's rendering, piece by piece ----------------------------------------------------------------

XML_INSTR = ('\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n'
             '<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n'
             '<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\n'
             'multiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST '
             'follow the specified format: an inner <function=...></function> block must be nested within '
             '<tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional '
             'reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there '
             'is no function call available, answer the question like normal with your current knowledge and do not '
             'tell the user about function calls\n</IMPORTANT>')


def _fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": req}}}


# pi's core tools, approximately (the sessions do not record the schemas; only the system prompt's size depends on it)
TOOLS = [
    _fn("read", "Read the contents of a file. Supports text files and images. Use offset/limit for large files.",
        {"path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
         "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
         "limit": {"type": "number", "description": "Maximum number of lines to read"}}, ["path"]),
    _fn("bash", "Execute a bash command in the current working directory. Returns stdout and stderr.",
        {"command": {"type": "string", "description": "Bash command to execute"},
         "timeout": {"type": "number", "description": "Timeout in seconds (optional)"}}, ["command"]),
    _fn("edit", "Edit a single file using exact text replacement. Every edits[].oldText must match a unique region "
        "of the original file.",
        {"path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
         "edits": {"type": "array", "description": "One or more targeted replacements.", "items": {
             "type": "object", "properties": {"oldText": {"type": "string"}, "newText": {"type": "string"}},
             "required": ["oldText", "newText"]}}}, ["path", "edits"]),
    _fn("write", "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
        {"path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
         "content": {"type": "string", "description": "Content to write to the file"}}, ["path", "content"]),
]


def tojson(x):
    return json.dumps(x, ensure_ascii=False)  # (HF's tojson for chat templates)


def system_text(sys_text):
    s = "<|im_start|>system\n" + spec_accept.EFFORT_LINE["xhigh"] + "\n\n"
    s += "# Tools\n\nYou have access to the following functions:\n\n<tools>"
    for t in TOOLS:
        s += "\n" + tojson(t)
    s += "\n</tools>" + XML_INSTR
    if sys_text.strip():
        s += "\n\n" + sys_text.strip()
    return s + "<|im_end|>\n"


def json_pieces(v, kind_of_key, key=None):
    """json.dumps(v, ensure_ascii=False) as (text, kind) pieces: strings under an oldText / newText key labeled"""
    if isinstance(v, dict):
        out = [("{", "args")]
        for i, (k, x) in enumerate(v.items()):
            out.append(((", " if i else "") + tojson(k) + ": ", "args"))
            out += json_pieces(x, kind_of_key, k)
        return out + [("}", "args")]
    if isinstance(v, list):
        out = [("[", "args")]
        for i, x in enumerate(v):
            if i:
                out.append((", ", "args"))
            out += json_pieces(x, kind_of_key, key)
        return out + [("]", "args")]
    return [(tojson(v), kind_of_key(key) if isinstance(v, str) else "args")]


def arg_kind(tool, name):
    if name in ("oldText", "old_string", "old_text"):
        return "old"
    if name in ("newText", "new_string", "new_text"):
        return "new"
    if tool == "write" and name == "content":
        return "write"
    return "args"


def reply_pieces(reasoning, content, calls):
    """an assistant message as Qwen3.8 renders it after '<|im_start|>assistant\\n<think>\\n', up to <|im_end|>"""
    c = content.strip()
    P = [(reasoning.strip(), "think"), ("\n</think>\n\n", "syntax"), (c, "prose")]
    for i, (name, args) in enumerate(calls):
        pre = ("\n\n" if c else "") if i == 0 else "\n"
        P.append((pre + "<tool_call>\n<function=" + name + ">\n", "syntax"))
        if isinstance(args, dict):
            for an, av in args.items():
                P.append(("<parameter=" + an + ">\n", "syntax"))
                if isinstance(av, str):
                    P.append((av, arg_kind(name, an)))
                else:
                    P += json_pieces(av, lambda k, name=name: arg_kind(name, k or ""))
                P.append(("\n</parameter>\n", "syntax"))
        P.append(("</function>\n</tool_call>", "syntax"))
    P.append(("<|im_end|>", "syntax"))
    return [(t, k) for t, k in P if t]


def text_parts(content):
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content or [] if isinstance(p, dict) and p.get("type") == "text")


def call_args(x):
    if isinstance(x, str):  # (a JSON string: as runq serve parses it)
        try:
            x = json.loads(x)
        except ValueError:
            x = None
    return x if isinstance(x, dict) else {}


def session_messages(path):
    """the main branch (the path from the last entry to the root) as a list of events: ('system', text),
    ('restart', messages), ('user', text), ('assistant', reasoning, content, calls), ('tool', text)"""
    entries, order = {}, []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if "id" in d:
                entries[d["id"]] = d
                order.append(d["id"])
    if not order:
        return []
    chain, cur = [], order[-1]
    while cur in entries:
        chain.append(entries[cur])
        cur = entries[cur].get("parentId")
    chain.reverse()
    msgs, pos = [], {}  # msgs: (entry id, message tuple) of the current context
    events = []
    for d in chain:
        t = d.get("type")
        if t == "compaction":
            first = pos.get(d.get("firstKeptEntryId"), 1 << 30)
            kept = [(i, m) for i, m in msgs if pos.get(i, -1) >= first]
            summary = ("The conversation history before this point was compacted into the following summary:\n\n"
                       "<summary>\n" + d.get("summary", "") + "\n</summary>")
            pos[d["id"]] = len(pos)
            msgs = [(d["id"], ("user", summary))] + kept
            events.append(("restart", [m for _, m in msgs]))
            continue
        m = None
        if t == "custom_message" and d.get("display", True):
            txt = text_parts(d.get("content"))
            if txt.strip():
                m = ("user", txt)
        elif t == "message":
            x = d["message"]
            r = x.get("role")
            if r == "system":
                txt = x.get("content") or ""
                if not txt and isinstance(x.get("sections"), dict):
                    txt = "".join(v for v in x["sections"].values() if isinstance(v, str))
                events.append(("system", txt))
                continue
            if r == "user":
                m = ("user", text_parts(x.get("content")))
            elif r == "bashExecution":
                if not x.get("excludeFromContext"):
                    m = ("user", "Ran `" + x.get("command", "") + "`\n```\n" + (x.get("output") or "") + "\n```")
            elif r == "toolResult":
                m = ("tool", text_parts(x.get("content")))
            elif r == "assistant":
                parts = x.get("content") or []
                reasoning = "\n".join(p.get("thinking", "") for p in parts if p.get("type") == "thinking")
                content = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
                calls = [(p.get("name", "tool"), call_args(p.get("arguments"))) for p in parts if p.get("type") == "toolCall"]
                if not (reasoning.strip() or content.strip() or calls):
                    continue
                m = ("assistant", reasoning, content, calls)
        if m is not None:
            pos[d["id"]] = len(pos)
            msgs.append((d["id"], m))
            events.append(m)
    return events


class Stream:
    """one token stream: pieces tokenized one message at a time (the template's special tokens separate them)"""

    def __init__(self, tok, sys_text):
        self.tok, self.ids, self.kind, self.spans, self.models = tok, [], [], [], []
        self.text = []  # (for --check-template)
        self.prev = None  # the previous message's role
        self.add(system_text(sys_text))

    def add(self, s, kinds=None):
        """append text s (context: kind -1; kinds: (text, kind) pieces of s, labels per token by its first char)"""
        if not s:
            return
        self.text.append(s)
        enc = self.tok.encode(s, add_special_tokens=False)
        self.ids += enc.ids
        if kinds is None:
            self.kind += [-1] * len(enc.ids)
            return
        bounds, kk, o = [], [], 0
        for t, k in kinds:
            o += len(t)
            bounds.append(o)
            kk.append(K_ID[k])
        b = np.searchsorted(np.array(bounds), np.array([a for a, _ in enc.offsets]), side="right")
        self.kind += [kk[min(i, len(kk) - 1)] for i in b]

    def close_tools(self):
        if self.prev == "tool":
            self.add("<|im_end|>\n")

    def message(self, m, span=True):
        """append a message; span False: an assistant message as context only (a restart's replay)"""
        if m[0] == "tool":
            self.add(("" if self.prev == "tool" else "<|im_start|>user") + "\n<tool_response>\n" + m[1].strip()
                     + "\n</tool_response>")
        else:
            self.close_tools()
            if m[0] == "user":
                self.add("<|im_start|>user\n" + m[1].strip() + "<|im_end|>\n")
            else:  # assistant: the generation prompt, the reply (a span), then the newline after <|im_end|>
                self.add("<|im_start|>assistant\n<think>\n")
                P = reply_pieces(m[1], m[2], m[3])
                a = len(self.ids)
                self.add("".join(t for t, _ in P), P if span else None)
                if span:
                    self.spans.append((a, len(self.ids)))
                self.add("\n")
        self.prev = m[0]


def cmd_sessions(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    files = sorted(glob.glob(os.path.expanduser(a.glob)))
    streams, nrep = [], 0
    checked = 0
    for fi, path in enumerate(files):
        events = session_messages(path)
        st, sys_text, msgs = None, "", []

        def flush():
            if st is not None and st.spans:
                streams.append({"ids": np.array(st.ids, dtype=np.int32), "kind": np.array(st.kind, dtype=np.int8),
                                "spans": np.array(st.spans, dtype=np.int64), "file": fi})

        for ev in events:
            if ev[0] == "system":
                if st is None or ev[1] != sys_text:
                    flush()
                    sys_text = ev[1]
                    st = Stream(tok, sys_text)
                    for m in msgs:
                        st.message(m, False)
                continue
            if ev[0] == "restart":
                flush()
                msgs = list(ev[1])
                st = Stream(tok, sys_text)
                for m in msgs:
                    st.message(m, False)
                continue
            if st is None:
                st = Stream(tok, sys_text)
            msgs.append(ev)
            st.message(ev)
        if a.check_template and st is not None and st.spans and checked < a.check_n:
            check_template(a.check_template, sys_text, msgs, "".join(st.text))
            checked += 1
        flush()
        nrep = sum(len(s["spans"]) for s in streams)
        if (fi + 1) % 20 == 0:
            print(f"  {fi + 1}/{len(files)} sessions, {len(streams)} streams, {nrep} replies", flush=True)
    with open(a.out, "wb") as f:
        pickle.dump(streams, f)
    ntok = sum(len(s["ids"]) for s in streams)
    nr = sum(int((s["kind"] >= 0).sum()) for s in streams)
    print(f"{len(files)} sessions -> {len(streams)} streams, {ntok} tokens, {nrep} replies ({nr} reply tokens); "
          f"wrote {a.out} (local data: do not commit)")
    if a.check_template:
        print(f"template check: {checked} sessions rendered == {a.check_template}")


def check_template(path, sys_text, msgs, mine):
    """the rendering (no generation prompt) vs the jinja template's on OpenAI-style messages"""
    import jinja2.ext
    import jinja2.sandbox
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                       extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = lambda x, **kw: tojson(x)

    def fail(msg):
        raise ValueError(msg)
    env.globals["raise_exception"] = fail
    with open(path) as f:
        t = env.from_string(f.read())
    om = [{"role": "system", "content": sys_text}] if sys_text.strip() else []
    for m in msgs:
        if m[0] == "assistant":
            om.append({"role": "assistant", "content": m[2], "reasoning_content": m[1],
                       "tool_calls": [{"type": "function", "function": {"name": n, "arguments": x}} for n, x in m[3]]})
        else:
            om.append({"role": m[0], "content": m[1]})
    ref = t.render(messages=om, tools=TOOLS, add_generation_prompt=False)
    # mine: each assistant message is followed by its newline; a trailing tool turn is not closed yet
    if mine.endswith("</tool_response>"):
        mine += "<|im_end|>\n"
    if ref != mine:
        i = next((j for j, (x, y) in enumerate(zip(ref, mine)) if x != y), min(len(ref), len(mine)))
        sys.exit(f"rendering differs from {path} at char {i}: {ref[i - 60:i + 60]!r} vs {mine[i - 60:i + 60]!r}")


# --- draft: the model teacher-forced ---------------------------------------------------------------------------------

def run_tool(tool, args, streams, sel, tmp, dtype, threads=1):
    """build/teacher_force on the streams' spans sel [(stream index, [(start, end), ...]), ...]: its output records"""
    with tempfile.TemporaryDirectory(dir=tmp, prefix="lookup_accept-") as td:
        fin, fout = os.path.join(td, "in.bin"), os.path.join(td, "out.bin")
        with open(fin, "wb") as f:
            f.write(struct.pack("<i", len(sel)))
            for si, sp in sel:
                ids = streams[si]["ids"][:sp[-1][1]]
                f.write(struct.pack("<i", len(ids)))
                f.write(ids.astype("<i4").tobytes())
                f.write(struct.pack("<i", len(sp)))
                f.write(np.array(sp, dtype="<i4").tobytes())
        subprocess.run([tool] + args + [fin, fout], env=dict(os.environ, OMP_NUM_THREADS=str(threads)), check=True)
        return np.fromfile(fout, dtype=dtype)


def cmd_draft(a):
    with open(a.streams, "rb") as f:
        streams = pickle.load(f)
    sel, total = [], 0
    for si in np.random.RandomState(a.seed).permutation(len(streams)):  # streams in a random order: each one's
        s = streams[si]  # replies up to --max-pos, while they fit in the budget
        sp = [(int(x), int(y)) for x, y in s["spans"] if y <= a.max_pos]
        if not sp or total + sp[-1][1] > a.budget:
            continue
        sel.append((int(si), sp))
        total += sp[-1][1]
    sel.sort()
    print(f"{len(sel)} streams, {total} tokens to feed, {sum(y - x for _, sp in sel for x, y in sp)} reply positions")
    raw = run_tool(a.tool, [a.model], streams, sel, a.tmp, [("am", "<i4"), ("p1", "<f4")], a.threads)
    am, p1, o = {}, {}, 0
    for si, sp in sel:
        n = sum(y - x for x, y in sp)
        am[si], p1[si] = raw["am"][o:o + n], raw["p1"][o:o + n]
        o += n
    assert o == len(raw)
    np.savez(a.out, model=os.path.basename(a.model), max_pos=a.max_pos,
             **{f"am{si}": am[si] for si, _ in sel}, **{f"p1{si}": p1[si] for si, _ in sel},
             **{f"sp{si}": np.array(sp) for si, sp in sel})
    print("wrote", a.out)


# --- report: the loop simulated ------------------------------------------------------------------------------------

CAP = 64  # the longest lookup acceptance counted (k <= CAP)
NMAX = 16  # src/spec.c's RQ_LOOKUP_LMAX: matches counted up to 16 tokens


def lookup_tables(ids, spans, nmax):
    """per reply position p (in span order) and n = 1..nmax: avail (tokens the lookup can guess: those after the most
    recent earlier occurrence of the n-gram ending at p - 1, up to p - 1; 0: no occurrence) and lcp (how many of them
    equal the reply's next tokens, within the span)"""
    T = ids.astype(np.uint64)
    N = len(T)
    pos = np.concatenate([np.arange(x, y) for x, y in spans])
    end = np.concatenate([np.full(y - x, y) for x, y in spans])
    avail = np.zeros((nmax + 1, len(pos)), dtype=np.int32)
    lcp = np.zeros((nmax + 1, len(pos)), dtype=np.int32)
    h = np.zeros(N, dtype=np.uint64)
    M = np.uint64(0x9E3779B97F4A7C15)
    idx = np.arange(N)
    with np.errstate(over="ignore"):
        for n in range(1, nmax + 1):
            hn = np.empty(N, dtype=np.uint64)
            hn[0] = 0
            hn[1:] = h[:-1] * M
            hn += T + np.uint64(1)
            h = hn.copy()
            hv = hn.copy()
            hv[:n - 1] = np.uint64(0xFFFFFFFFFFFFFFFF) - idx[:n - 1].astype(np.uint64)  # no n-gram there: unique
            order = np.lexsort((idx, hv))
            hs = hv[order]
            prev = np.full(N, -1, dtype=np.int64)
            same = hs[1:] == hs[:-1]
            prev[order[1:][same]] = order[:-1][same]
            e = prev[pos - 1]
            ok = e >= 0
            av = np.where(ok, pos - 1 - e, 0)
            avail[n] = np.minimum(av, CAP)
            # lcp: T[e + 1 + c] == T[pos + c] while c < min(avail, CAP, end - pos)
            lim = np.minimum(np.minimum(av, CAP), end - pos)
            run = ok.copy()
            cnt = np.zeros(len(pos), dtype=np.int32)
            ee = np.where(ok, e, 0)
            for c in range(CAP):
                run &= c < lim
                if not run.any():
                    break
                a_ = np.minimum(ee + 1 + c, N - 1)
                b_ = np.minimum(pos + c, N - 1)
                run &= T[a_] == T[b_]
                cnt += run
            lcp[n] = cnt
    return avail, lcp


def rule_arrays(avail, lcp, nmin, nmax):
    """longest-first from nmax down to nmin: (avail, lcp, n used) per position"""
    av = np.zeros(avail.shape[1], dtype=np.int32)
    lc = np.zeros_like(av)
    nu = np.zeros_like(av)
    for n in range(nmin, nmax + 1):  # the longer n overrides where it found an occurrence
        m = avail[n] > 0
        av[m], lc[m], nu[m] = avail[n][m], lcp[n][m], n
    return av, lc, nu


class Data:
    """replies as flat arrays: target ids, kind, span offsets; lookup tables; the model's argmax / p_top1 (or None)"""

    def __init__(self, name):
        self.name, self.parts = name, []

    def add(self, ids, spans, kind, nmax, am=None, p1=None):
        avail, lcp = lookup_tables(ids, spans, nmax)
        tgt = np.concatenate([ids[x:y] for x, y in spans])
        lens = np.array([y - x for x, y in spans])
        self.parts.append(dict(tgt=tgt, kind=kind, lens=lens, avail=avail, lcp=lcp,
                               m=None if am is None else (am == tgt), p1=p1))

    def cat(self):
        P = self.parts
        self.tgt = np.concatenate([p["tgt"] for p in P])
        self.kind = np.concatenate([p["kind"] for p in P])
        self.lens = np.concatenate([p["lens"] for p in P])
        self.avail = np.concatenate([p["avail"] for p in P], axis=1)
        self.lcp = np.concatenate([p["lcp"] for p in P], axis=1)
        self.has_model = all(p["m"] is not None for p in P)
        if self.has_model:
            self.m = np.concatenate([p["m"] for p in P])
            self.p1 = np.concatenate([p["p1"] for p in P])
        del self.parts
        return self


def simulate(D, cost, lk=None, mk=None, tau=None, kinds_of=None):
    """the loop on the teacher-forced tokens. lk: (avail, lcp, k) of a lookup rule; mk: the model's k (with tau).
    Returns {kind: [tokens, passes, guesses, accepted, lookup passes, ms]} (a pass counted under its first token's
    kind) and 'ALL'."""
    kind = D.kind.tolist() if kinds_of is None else kinds_of
    lens = D.lens.tolist()
    if lk is not None:
        lav, llc, k_l = lk[0].tolist(), lk[1].tolist(), lk[2]
    if mk is not None:
        mm, pp = D.m.tolist(), D.p1.tolist()
    step, ver, dms = cost["step"], cost["verify"], cost["draft"]
    acc = {}
    o = 0
    for L in lens:
        s = 0
        while s < L:
            i = o + s
            nl = 0
            if lk is not None and lav[i] > 0:
                n = min(k_l, lav[i])
                a_ = min(llc[i], n)
                src, md = 1, 0
            elif mk is not None:
                n = 0
                while n < mk and s + n < L and (tau is None or pp[i + n] >= tau):
                    n += 1
                a_ = 0
                while a_ < n and mm[i + a_]:
                    a_ += 1
                src, md = 0, n + 1
            else:
                n = a_ = src = md = 0
            nl = min(a_ + 1, L - s)
            r = acc.setdefault(kind[i], [0, 0, 0, 0, 0, 0.0])
            r[0] += nl
            r[1] += 1
            r[2] += n
            r[3] += a_
            r[4] += src
            r[5] += step * (ver[n] if n else 1.0) + dms * md
            s += a_ + 1
        o += L
    tot = [0, 0, 0, 0, 0, 0.0]
    for r in acc.values():
        for j in range(6):
            tot[j] += r[j]
    acc["ALL"] = tot
    return acc


COSTS = {  # the 27B Q4_K_M + the 0.8B Q4_K_M draft (docs/qwen38-speculative.md): a short context and ~98k positions
    "short": {"step": 120.0, "draft": 5.1, "verify": [1.0, 1.14, 1.16, 1.25, 1.41, 1.62, 1.83, 2.15, 2.52]},
    "long": {"step": 207.0, "draft": 19.0, "verify": [1.0, 1.41, 1.785, 2.16, 2.63, 3.10, 3.57, 4.04, 4.51]},
}


def cost_of(name):
    c = dict(COSTS[name])
    v = list(c["verify"])
    while len(v) <= CAP:  # past k = 8: the last slope
        v.append(2 * v[-1] - v[-2])
    c["verify"] = v
    return c


def fmt(r):
    tok, pas, gu, ac, lp, ms = r
    return (f"{tok / pas:5.2f} {gu / pas:5.2f} {100 * ac / max(gu, 1):4.0f}% {100 * lp / pas:4.0f}% "
            f"{1000 * tok / ms:5.1f}")


HEAD = "tok/pass guess/pass acc% lookup% tok/s"


def table(D, cost, rows, by_kind, kind_names):
    kinds = ["ALL"] + ([k for k in range(len(kind_names)) if (D.kind == k).any()] if by_kind else [])
    names = {"ALL": "ALL", **{k: kind_names[k] for k in range(len(kind_names))}}
    cnt = {k: int((D.kind == k).sum()) for k in kinds if k != "ALL"}
    cnt["ALL"] = len(D.kind)
    print(f"  {'setting':34s} " + " | ".join(f"{names[k]} ({cnt[k]})".center(33) for k in kinds))
    print(f"  {'':34s} " + " | ".join(f"{'tok/pass gu/pass acc lookup tok/s':>33s}" for _ in kinds))
    for label, kw in rows:
        r = simulate(D, cost, **kw)
        print(f"  {label:34s} " + " | ".join(f"{fmt(r[k]) if k in r else '':>33s}" for k in kinds), flush=True)


def report(D, a, kind_names):
    cost = cost_of(a.cost)
    print(f"\n== {D.name}: {len(D.lens)} replies, {len(D.tgt)} tokens; cost model '{a.cost}': step {cost['step']} ms, "
          f"draft step {cost['draft']} ms, verify(1..8) {cost['verify'][1:9]}")
    print("   per kind: tokens per target pass, guesses per pass, accepted / guessed, passes drafted by the lookup, "
          "tok/s (the cost model; a lookup guess costs no draft step)")
    rows = [("plain", {})]
    nmax = D.avail.shape[0] - 1
    for n in [1, 2, 3, 4, 8]:  # one n-gram length (its most recent occurrence)
        av, lc, _ = rule_arrays(D.avail, D.lcp, n, n)
        rows.append((f"lookup n={n} only, k=4", {"lk": (av, lc, 4)}))
    R = {m: rule_arrays(D.avail, D.lcp, m, nmax) for m in (2, 3, 4, 6, 8)}  # src/spec.c's: the longest match (<= 16)
    for k in [1, 2, 3, 4, 6, 8, 16]:
        rows.append((f"lookup >=2 k={k}", {"lk": (R[2][0], R[2][1], k)}))
    for m in (3, 4, 6, 8):
        for k in (3, 8):
            rows.append((f"lookup >={m} k={k}", {"lk": (R[m][0], R[m][1], k)}))
    if D.has_model:
        for k in [2, 3, 4]:
            rows.append((f"model k={k}", {"mk": k}))
        rows.append(("model k=6 tau=0.4", {"mk": 6, "tau": 0.4}))
        for m in (2, 3, 4, 6, 8):
            for kl in (3, 8):
                rows.append((f"comb lookup >={m} k={kl} + model k=3", {"lk": (R[m][0], R[m][1], kl), "mk": 3}))
        for m in (4, 8):
            rows.append((f"comb lookup >={m} k=8 + k6 tau.4", {"lk": (R[m][0], R[m][1], 8), "mk": 6, "tau": 0.4}))
    table(D, cost, rows, a.by_kind, kind_names)


def load_answers(spec):
    """GEN.json:MEAS.json -> Data (the prompts re-rendered as spec_accept measure did, tokenized with HF's tokenizer)"""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(ROOT, "models/Qwen3.5-0.8B/tokenizer.json"))
    gen_path, meas_path = spec.split(":")
    with open(gen_path) as f:
        gen = [r for r in json.load(f) if "ids" in r]
    with open(meas_path) as f:
        meas = json.load(f)["answers"]
    cats = []
    D = Data(f"{os.path.basename(meas_path)} (the 27B's answers, model draft {os.path.basename(meas_path)})")
    gi = 0
    for ans in meas:
        while gi < len(gen):  # (measure skipped the short ones)
            r = gen[gi]
            gi += 1
            pr = tok.encode(spec_accept.render(r["prompt"], ans["think"]), add_special_tokens=False).ids
            if len(pr) == ans["n_prompt"] and r["cat"] == ans["cat"]:
                break
        else:
            sys.exit(f"{meas_path}: no prompt of {ans['n_prompt']} tokens for an answer")
        ids = np.array(pr + ans["target"], dtype=np.int32)
        sp = [(len(pr), len(ids))]
        if ans["cat"] not in cats:
            cats.append(ans["cat"])
        kind = np.full(len(ans["target"]), cats.index(ans["cat"]), dtype=np.int8)
        D.add(ids, sp, kind, NMAX, np.array(ans["draft"]), np.array(ans["p_top1"]))
    return D.cat(), cats


def check_engine(a, D, streams, used):
    """the simulation's rule (longest match first, n = 16 .. 2) vs src/spec.c's lookup run on the same positions"""
    raw = run_tool(a.tool, ["-l"], streams, used, a.tmp, [("len", "<i4"), ("n", "<i4"), ("acc", "<i4")])
    av, lc, nu = rule_arrays(D.avail, D.lcp, 2, D.avail.shape[0] - 1)
    same_len = raw["len"] == nu
    same = same_len & (raw["n"] == av) & (raw["acc"] == lc)
    print(f"engine check: src/spec.c's lookup (rq_lookup_draft, build/teacher_force -l) at {len(raw)} positions: "
          f"{same.mean():.4%} identical to the simulation's (match length, guesses, accepted); match length "
          f"{same_len.mean():.4%}; positions with a match {np.mean(raw['len'] > 0):.2%} vs {np.mean(nu > 0):.2%}")


def cmd_report(a):
    if a.streams:
        with open(a.streams, "rb") as f:
            streams = pickle.load(f)
        dr = np.load(a.draft) if a.draft else None
        sel = sorted(int(k[2:]) for k in dr.files if k.startswith("am")) if dr is not None else range(len(streams))
        D = Data(f"pi sessions ({len(sel)} streams, replies ending before position {a.max_pos}" + (
            f", model draft {dr['model']} up to {int(dr['max_pos'])} positions)" if dr is not None else ")"))
        used = []
        for si in sel:
            s = streams[si]
            sp = [(int(x), int(y)) for x, y in (dr[f"sp{si}"] if dr is not None else s["spans"]) if y <= a.max_pos]
            if not sp:
                continue
            used.append((si, sp))
            kind = np.concatenate([s["kind"][x:y] for x, y in sp])
            D.add(s["ids"], sp, kind, NMAX, dr[f"am{si}"] if dr is not None else None,
                  dr[f"p1{si}"] if dr is not None else None)
        D.cat()
        if a.engine:
            check_engine(a, D, streams, used)
        report(D, a, KINDS)
    for spec in a.answers or []:
        D, cats = load_answers(spec)
        report(D, a, cats)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sessions", help="render and tokenize pi's sessions into token streams")
    s.add_argument("out")
    s.add_argument("--glob", default="~/.pi/agent/sessions/*/*.jsonl")
    s.add_argument("--tokenizer", default=os.path.join(ROOT, "models/Qwen3.5-0.8B/tokenizer.json"))
    s.add_argument("--check-template", help="Qwen3.8's chat template (jinja): the rendering checked against it")
    s.add_argument("--check-n", type=int, default=20, help="sessions checked (the first ones)")
    d = sub.add_parser("draft", help="teacher-force a draft GGUF on the streams' replies")
    d.add_argument("streams")
    d.add_argument("model")
    d.add_argument("-o", "--out", required=True)
    d.add_argument("--max-pos", type=int, default=24576, help="replies ending before this position only")
    d.add_argument("--budget", type=int, default=600000, help="positions fed in all (streams in order)")
    d.add_argument("--seed", type=int, default=1, help="the streams' order (random)")
    d.add_argument("--threads", type=int, default=8)
    d.add_argument("--tool", default=os.path.join(ROOT, "build/teacher_force"))
    d.add_argument("--tmp", default="/var/tmp")
    r = sub.add_parser("report", help="the loop simulated: lookup, model draft, combined")
    r.add_argument("streams", nargs="?")
    r.add_argument("--draft", help="draft's output: only the streams it covers, the model and combined rows")
    r.add_argument("--answers", nargs="*", help="GEN.json:MEAS.json pairs of spec_accept.py (the 27B's answers)")
    r.add_argument("--max-pos", type=int, default=98304, help="replies ending before this position only (default: "
                   "the 27B's 96k window)")
    r.add_argument("--cost", choices=sorted(COSTS), default="short")
    r.add_argument("--by-kind", action="store_true")
    r.add_argument("--engine", action="store_true", help="check the simulated lookup against src/spec.c's "
                   "(build/teacher_force -l)")
    r.add_argument("--tool", default=os.path.join(ROOT, "build/teacher_force"))
    r.add_argument("--tmp", default="/var/tmp")
    a = ap.parse_args()
    {"sessions": cmd_sessions, "draft": cmd_draft, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
