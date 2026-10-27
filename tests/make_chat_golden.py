#!/usr/bin/env python3
"""Golden chat-template renderings for the Qwen3 template variants, Qwen3.5's and Qwen3.8's (tests/golden/chat_render.json).

Usage: .venv/bin/python tests/make_chat_golden.py      (transformers' jinja; no network, no model weights)

Templates (tests/golden/templates, the chat_template of each HF repo's tokenizer_config.json, Apache-2.0):
  qwen3.jinja          Qwen/Qwen3-0.6B .. 14B (identical in all of them; e.g. Qwen/Qwen3-4B @ 1cfa9a7)
  instruct-2507.jinja  Qwen/Qwen3-4B-Instruct-2507 @ cdbee75
  thinking-2507.jinja  Qwen/Qwen3-4B-Thinking-2507 @ 768f209
  qwen3.5.jinja        Qwen/Qwen3.5-0.8B @ 2fc0636 (chat_template.jinja)
  qwen3.8.jinja        unsloth/Qwen3.8-27B-GGUF @ 4a03c64: Qwen3.8-27B-Q4_K_M.gguf's tokenizer.chat_template (Unsloth's
                       rewrite of Qwen's: developer role, merged system messages, 'high' = 'xhigh', tool calls)
  qwen3.8-hf.jinja     Qwen/Qwen3.8-27B @ 1d4bf0f (chat_template.jinja; checked too: "hf_qwen")
Tokenizer: models/Qwen3-0.6B (the Qwen3 family shares tokenizer.json); models/Qwen3.5-0.8B for qwen3.5 and qwen3.8
(the 27B GGUF's tokens and merges are the 0.8B's).

For each variant, thinking mode (0 / 1) and conversation: `text` / `ids` = what runq's rq_chat_render must give
(`runq -m render`, checked by tests/check_models.py on tokenizer-only GGUFs carrying each template), built here
from runq's documented canonical form (src/runq.h, src/chat.c); `hf` = whether that equals HF's
apply_chat_template(add_generation_prompt=True, enable_thinking=thinking) (asserted for the cases marked so).
The known differences, by design: the hybrid template with thinking off keeps the empty think block in every
user turn and earlier replies verbatim (the CLI keeps the conversation in the KV cache as generated; HF drops
both from earlier turns); Thinking-2507 with thinking off is not in its template (runq serve's constrained
replies only). Qwen3.5 renders as Thinking-2507 with thinking on and as the hybrid with it off (the same known
difference), and trims every message text (|trim).
Qwen3.8, per conversation thinking off and on with each reasoning effort (xhigh: the default, no reasoning_effort
passed; medium; low): equal to HF's rendering (reasoning_effort, enable_thinking) everywhere, except one intended
difference: an assistant text with "</think>" is runq's reasoning + answer (the CLI's replies are so), which HF's
template renders as its reasoning_content + content; given the message's content alone it would put an empty think
block in front of the whole text ("hf_reasoning_content": runq's form == HF's with the message split so). Qwen's own
template (qwen3.8-hf.jinja) gives the same as Unsloth's except where Unsloth's adds features (developer role,
several system messages: "hf_qwen" false / null for an error).
Tool calls (runq serve with tools, src/tools.c; Qwen3.5 and Qwen3.8): the cases with a "request" (an OpenAI chat completion
body: tools, tool messages, assistant tool_calls with their arguments as a JSON string, reasoning_content, content parts)
rendered by `runq -m render` as runq serve renders it; `text` = HF's apply_chat_template of the same conversation, the
arguments parsed into objects first (as vLLM does), thinking per chat_template_kwargs.enable_thinking (Qwen3.8 also its
reasoning_effort): runq's rendering must equal HF's.
"""
import json
import os
import re
import sys

import transformers
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES = os.path.join(ROOT, "tests", "golden", "templates")
OUT = os.path.join(ROOT, "tests", "golden", "chat_render.json")
VARIANTS = ["qwen3", "instruct-2507", "thinking-2507", "qwen3.5", "qwen3.8"]
TOKENIZERS = {"qwen3.5": "Qwen3.5-0.8B", "qwen3.8": "Qwen3.5-0.8B"}  # (else Qwen3-0.6B)
G = "<think>\n\n</think>\n\n"

CONVERSATIONS = [
    ("one_turn", [("user", "Hi there")]),
    ("system", [("system", "You are terse."), ("user", "What is 2+2?")]),
    ("multi_turn", [("user", "My name is Ada."), ("assistant", "Hello Ada!"), ("user", "What is my name?")]),
    ("reasoning_history", [("system", "Be brief."), ("user", "Hi!"),
                           ("assistant", "<think>\nThe user greets me.\n</think>\n\nHello! How can I help?"),
                           ("user", "And in French?")]),
    ("unicode", [("user", "Écris un haïku\nsur la mer 🌊, s'il te plaît.")]),
    # (surrounding whitespace: Qwen3.5's template trims it, Python's str.strip(): U+3000 too; the others keep it)
    ("spaces", [("system", "  Be brief.\n"), ("user", "\u3000Hi! \n"), ("assistant", " Hello.\n"), ("user", "\tAnd now? ")]),
]
# (Qwen3.8 also: the developer role, system messages merged, an empty one; Unsloth's template only)
CONVERSATIONS_38 = CONVERSATIONS + [
    ("developer", [("developer", "Answer in French."), ("user", "Hi!")]),
    ("two_systems", [("system", "You are terse. "), ("developer", "\nAnswer in French."), ("user", "Hi!")]),
    ("empty_system", [("system", " \n"), ("user", "Hi!")]),
]
EFFORTS_38 = [(0, None), (1, None), (1, "medium"), (1, "low"), (1, "high")]  # (thinking, reasoning_effort): None = default


def runq_text_38(thinking, effort, msgs):
    """Qwen3.8's form in runq (src/chat.c): the effort instruction + the merged leading system / developer messages,
    user turns as Qwen3.5's, every assistant text as "<think>\n" R "\n</think>\n\n" C (split at its last </think>)"""
    msgs = [(r, t.strip()) for r, t in msgs]
    ins = "" if not thinking or effort == "medium" else LOW_38 if effort == "low" else XHIGH_38
    i, sys_ = 0, []
    while i < len(msgs) and msgs[i][0] in ("system", "developer"):
        if msgs[i][1]:
            sys_.append(msgs[i][1])
        i += 1
    sys_ = "\n".join(sys_)
    out = f"<|im_start|>system\n{ins + chr(10) * 2 if ins and sys_ else ins}{sys_}<|im_end|>\n" if ins or sys_ else ""
    for j, (role, text) in enumerate(msgs[i:], i):
        if role == "user":
            answered = j + 1 < len(msgs) and msgs[j + 1][0] == "assistant"
            out += f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n" + ("" if answered else "<think>\n" if thinking else G)
        else:
            r, c = split_38(text)
            out += f"<think>\n{r}\n</think>\n\n{c}<|im_end|>\n"
    return out


def split_38(text):
    """an assistant text -> (reasoning, answer): at its last </think>, the reasoning after a <think>, both stripped"""
    if "</think>" not in text:
        return "", text.strip()
    r, c = text.rsplit("</think>", 1)
    return r.split("<think>")[-1].strip(), c.strip()


XHIGH_38 = ("Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider "
            "plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.")
LOW_38 = ("Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without "
          "unnecessary elaboration.")


def hf_38(tok, tmpl, conv, thinking, effort, split):
    """HF's rendering of a Qwen3.8 case (split: assistant texts as reasoning_content + content); None: the template
    raised an exception"""
    msgs = []
    for r, c in conv:
        if r == "assistant" and split and "</think>" in c:
            rc, c = split_38(c)
            msgs.append({"role": r, "content": c, "reasoning_content": rc})
        else:
            msgs.append({"role": r, "content": c})
    kw = {"enable_thinking": bool(thinking)}
    if effort:
        kw["reasoning_effort"] = effort
    try:
        return tok.apply_chat_template(msgs, chat_template=tmpl, tokenize=False, add_generation_prompt=True, **kw)
    except Exception:  # (jinja2's TemplateError from raise_exception: e.g. Qwen's own template has no developer role)
        return None


def runq_text(variant, thinking, msgs):
    """runq's canonical rendering (src/chat.c): [system] + per user message P_i (+ the variant's generation
    suffix; in an answered turn only the empty think block stays, Thinking-2507's "<think>\\n" does not), per
    assistant message its text (after the last </think>, leading newlines stripped, when the mode thinks) +
    <|im_end|>\\n"""
    think = 0 if variant == "instruct-2507" else thinking
    suffix = {"qwen3": "" if think else G, "instruct-2507": "", "thinking-2507": "<think>\n" if think else G,
              "qwen3.5": "<think>\n" if think else G}[variant]
    if variant == "qwen3.5":
        msgs = [(r, t.strip()) for r, t in msgs]
    out, i = "", 0
    if msgs[0][0] == "system":
        out += f"<|im_start|>system\n{msgs[0][1]}<|im_end|>\n"
        i = 1
    for j, (role, text) in enumerate(msgs[i:], i):
        if role == "user":
            answered = j + 1 < len(msgs) and msgs[j + 1][0] == "assistant"
            out += f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n{'' if answered and suffix != G else suffix}"
        else:
            if think and "</think>" in text:
                text = text.split("</think>")[-1].lstrip("\n")
            out += f"{text}<|im_end|>\n"
    return out


# tool calls: tools (unicode, controls, numbers in Python's float forms) and conversations (OpenAI messages)
TOOLS = [
    {"type": "function", "function": {
        "name": "read", "description": "Read a file. Paths are relative to the cwd.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path to the file"},
            "offset": {"type": "integer", "description": "Line to start from (1-indexed)", "minimum": 1},
            "limit": {"type": "number", "minimum": 0.5, "maximum": 1E20, "multipleOf": 1.0, "default": -0, "x": 1e-7}},
            "required": ["path"], "additionalProperties": False}, "strict": False}},
    {"type": "function", "function": {
        "name": "bash", "description": "Run a command \u2014 \u00ablike this\u00bb \u2713 \U0001f41a, tab\there \"quoted\" \\ \u0001 \u007f \u2028.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}, "timeout": {"type": ["integer", "null"]}, "env": {"type": "object"},
            "flags": {"type": "array", "items": {"type": "string"}}, "dry": {"type": "boolean"}, "ratio": {"type": "number"}},
            "required": ["command"]}}},
]
CALL_READ = {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": json.dumps({"path": "notes/a.txt", "offset": 3})}}
CALL_BASH = {"id": "call_2", "type": "function", "function": {"name": "bash", "arguments":
             '{"command": "ls -la\\nwc -l *.c", "timeout": null, "env": {"A": "1", "B": [1, 2.50, true]}, "flags": ["-x", "y"], '
             '"dry": true, "ratio": 1.50, "big": 1E5, "neg": -0, "tiny": 1e-7, "dup": 1, "dup": "second", "uni": "caf\\u00e9 \\ud83d\\udc1a \\u0007"}'}}
CALL_EMPTY = {"id": "call_3", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
U, S = {"role": "user", "content": "What is in notes/a.txt?"}, {"role": "system", "content": "You are a coding agent. "}
TOOL_CASES = [  # (name, tools, messages)
    ("tools_one_turn", TOOLS, [S, U]),
    ("tools_no_system", TOOLS, [U]),
    ("tools_empty_system", TOOLS, [{"role": "system", "content": " "}, U]),
    ("tool_round_trip", TOOLS, [S, U, {"role": "assistant", "content": None, "tool_calls": [CALL_READ]},
                                {"role": "tool", "tool_call_id": "call_1", "content": "hello\nworld\n"}]),
    ("tool_calls_multi", TOOLS, [S, U, {"role": "assistant", "content": "Let me check. ", "tool_calls": [CALL_BASH, CALL_READ, CALL_EMPTY]},
                                 {"role": "tool", "tool_call_id": "call_2", "content": "a.c b.c"},
                                 {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "part 1, "},
                                                                                         {"type": "text", "text": "part 2"}]},
                                 {"role": "tool", "tool_call_id": "call_3", "content": ""},
                                 {"role": "assistant", "content": "It has two lines.", "reasoning_content": " The file says hello. "},
                                 {"role": "user", "content": "Thanks! Now count the C files."}]),
    ("tool_reasoning", TOOLS, [U, {"role": "assistant", "content": "", "reasoning_content": "I should read it.\n",
                                   "tool_calls": [CALL_READ]},
                               {"role": "tool", "tool_call_id": "call_1", "content": "hello"},
                               {"role": "assistant", "content": "Reading more.", "reasoning_content": "", "tool_calls": [CALL_READ]},
                               {"role": "tool", "tool_call_id": "call_1", "content": "world"}]),
    ("tool_history_no_tools", [], [U, {"role": "assistant", "content": None, "tool_calls": [CALL_READ]},
                                   {"role": "tool", "tool_call_id": "call_1", "content": "hello"}]),
    ("tool_response_user", TOOLS, [U, {"role": "assistant", "content": "", "tool_calls": [CALL_READ]},
                                   {"role": "user", "content": "<tool_response>\nhello\n</tool_response>"}]),
]
TOOL_CASES_35 = [  # (Qwen3.5 only: its split of an assistant text at </think>; runq's Qwen3.8 form differs, src/chat.c)
    ("tool_think_text", TOOLS, [U, {"role": "assistant", "content": "<think>\nLet me see.\n</think>\n\nI will read it.",
                                    "tool_calls": [CALL_READ]}, {"role": "tool", "tool_call_id": "call_1", "content": "hi"}]),
]
TOOL_CASES_38 = [  # (Qwen3.8 only: Unsloth's merged system / developer messages)
    ("tools_developer", TOOLS, [S, {"role": "developer", "content": "Prefer bash."}, U]),
]


def hf_tools(tok, tmpl, tools, msgs, kw):
    """HF's rendering of an OpenAI request's conversation: tool call arguments parsed into objects (vLLM's form); None
    if the template raised"""
    conv = []
    for m in msgs:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [dict(c, function=dict(c["function"], arguments=json.loads(c["function"]["arguments"])))
                               for c in m["tool_calls"]]
        conv.append(m)
    try:
        return tok.apply_chat_template(conv, tools=tools or None, chat_template=tmpl, tokenize=False, add_generation_prompt=True, **kw)
    except Exception:  # (jinja2's TemplateError from raise_exception)
        return None


def tool_cases(v, tok, tmpl, tmpl_qwen):
    """the tool call cases of a variant (Qwen3.5: thinking off / on; Qwen3.8: off, on (xhigh), low)"""
    out = []
    modes = [(0, None), (1, None)] if v == "qwen3.5" else [(0, None), (1, None), (1, "low")]
    for thinking, effort in modes:
        for name, tools, msgs in TOOL_CASES + (TOOL_CASES_35 if v == "qwen3.5" else TOOL_CASES_38):
            kw = {"enable_thinking": bool(thinking)}
            body = {"messages": msgs, "chat_template_kwargs": dict(kw)}
            if tools:
                body["tools"] = tools
            if effort:
                kw["reasoning_effort"] = body["reasoning_effort"] = effort
            text = hf_tools(tok, tmpl, tools, msgs, kw)
            if text is None:
                sys.exit(f"{v} {name}: HF's template raised")
            c = {"variant": v, "thinking": thinking, "name": name if effort is None else f"{name}_{effort}", "request": body,
                 "text": text, "ids": tok.encode(text, add_special_tokens=False), "hf": True}
            if effort:
                c["reasoning_effort"] = effort
            if tmpl_qwen:
                q = hf_tools(tok, tmpl_qwen, tools, msgs, kw)
                c["hf_qwen"] = None if q is None else q == text
            out.append(c)
    return out


def main():
    toks = {d: AutoTokenizer.from_pretrained(os.path.join(ROOT, "models", d), local_files_only=True)
            for d in {"Qwen3-0.6B", *TOKENIZERS.values()}}
    cases = []
    for v in VARIANTS:
        tok = toks[TOKENIZERS.get(v, "Qwen3-0.6B")]
        with open(os.path.join(TEMPLATES, v + ".jinja"), encoding="utf-8") as f:
            tmpl = f.read()
        if v == "qwen3.8":
            with open(os.path.join(TEMPLATES, "qwen3.8-hf.jinja"), encoding="utf-8") as f:
                tmpl_qwen = f.read()
            for thinking, effort in EFFORTS_38:
                for name, conv in CONVERSATIONS_38:
                    text = runq_text_38(thinking, effort, conv)
                    hf, hf_rc, hf_q = (hf_38(tok, t, conv, thinking, effort, sp)
                                       for t, sp in ((tmpl, False), (tmpl, True), (tmpl_qwen, True)))
                    split = any(r == "assistant" and "</think>" in c for r, c in conv)
                    if text != hf_rc or (text != hf) != split:
                        sys.exit(f"{v} thinking {thinking} {effort} {name}: runq's form != HF's\n  runq {text!r}\n  hf   {hf!r}"
                                 f"\n  hf (reasoning_content) {hf_rc!r}")
                    c = {"variant": v, "thinking": thinking, "name": name if effort is None else f"{name}_{effort}",
                         "messages": [{"role": r, "content": t} for r, t in conv], "text": text,
                         "ids": tok.encode(text, add_special_tokens=False), "hf": text == hf}
                    if effort:
                        c["reasoning_effort"] = effort
                    if text != hf:
                        c["hf_text"] = hf
                        c["hf_reasoning_content"] = text == hf_rc
                    c["hf_qwen"] = None if hf_q is None else text == hf_q
                    cases.append(c)
            cases += tool_cases(v, tok, tmpl, tmpl_qwen)
            continue
        for thinking in (0, 1):
            for name, conv in CONVERSATIONS:
                msgs = [{"role": r, "content": c} for r, c in conv]
                hf = tok.apply_chat_template(msgs, chat_template=tmpl, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=bool(thinking))
                text = runq_text(v, thinking, conv)
                expect_hf = not (v in ("qwen3", "qwen3.5") and not thinking and any(r == "assistant" for r, _ in conv)) and \
                    not (v == "thinking-2507" and not thinking)
                if expect_hf and text != hf:
                    sys.exit(f"{v} thinking {thinking} {name}: runq's form != HF's\n  runq {text!r}\n  hf   {hf!r}")
                c = {"variant": v, "thinking": thinking, "name": name, "messages": msgs, "text": text,
                     "ids": tok.encode(text, add_special_tokens=False), "hf": text == hf}
                if text != hf:
                    c["hf_text"] = hf
                cases.append(c)
        if v == "qwen3.5":
            cases += tool_cases(v, tok, tmpl, None)
    meta = {"transformers": transformers.__version__, "tokenizer": "models/Qwen3-0.6B",
            "tokenizers": {v: f"models/{d}" for v, d in TOKENIZERS.items()},
            "templates": {v: f"tests/golden/templates/{v}.jinja" for v in VARIANTS}}
    text = json.dumps({"meta": meta, "cases": cases}, ensure_ascii=False, indent=1)
    # (token id lists on one line each)
    text = re.sub(r"\[\n\s*(-?\d+(?:,\n\s*-?\d+)*)\n\s*\]", lambda m: "[" + re.sub(r",\n\s*", ", ", m.group(1)) + "]", text)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    n38 = [c for c in cases if c["variant"] == "qwen3.8"]
    print(f"wrote {OUT}: {len(cases)} cases, {sum(c['hf'] for c in cases)} equal to HF's rendering; qwen3.8: {len(n38)} cases, "
          f"{sum(c['hf'] for c in n38)} equal to HF's, {sum(c.get('hf_reasoning_content', False) for c in n38)} to HF's with "
          f"reasoning_content, {sum(c['hf_qwen'] is True for c in n38)} to Qwen's own template's; "
          f"{sum('request' in c for c in cases)} tool call cases (requests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
