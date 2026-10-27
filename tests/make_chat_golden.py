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
"""
import json
import os
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
    meta = {"transformers": transformers.__version__, "tokenizer": "models/Qwen3-0.6B",
            "tokenizers": {v: f"models/{d}" for v, d in TOKENIZERS.items()},
            "templates": {v: f"tests/golden/templates/{v}.jinja" for v in VARIANTS}}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "cases": cases}, f, ensure_ascii=False, indent=1)
        f.write("\n")
    n38 = [c for c in cases if c["variant"] == "qwen3.8"]
    print(f"wrote {OUT}: {len(cases)} cases, {sum(c['hf'] for c in cases)} equal to HF's rendering; qwen3.8: {len(n38)} cases, "
          f"{sum(c['hf'] for c in n38)} equal to HF's, {sum(c.get('hf_reasoning_content', False) for c in n38)} to HF's with "
          f"reasoning_content, {sum(c['hf_qwen'] is True for c in n38)} to Qwen's own template's")
    return 0


if __name__ == "__main__":
    sys.exit(main())
