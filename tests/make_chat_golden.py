#!/usr/bin/env python3
"""Golden chat-template renderings for the Qwen3 template variants and Qwen3.5's (tests/golden/chat_render.json).

Usage: .venv/bin/python tests/make_chat_golden.py      (transformers' jinja; no network, no model weights)

Templates (tests/golden/templates, the chat_template of each HF repo's tokenizer_config.json, Apache-2.0):
  qwen3.jinja          Qwen/Qwen3-0.6B .. 14B (identical in all of them; e.g. Qwen/Qwen3-4B @ 1cfa9a7)
  instruct-2507.jinja  Qwen/Qwen3-4B-Instruct-2507 @ cdbee75
  thinking-2507.jinja  Qwen/Qwen3-4B-Thinking-2507 @ 768f209
  qwen3.5.jinja        Qwen/Qwen3.5-0.8B @ 2fc0636 (chat_template.jinja)
Tokenizer: models/Qwen3-0.6B (the Qwen3 family shares tokenizer.json); models/Qwen3.5-0.8B for qwen3.5.

For each variant, thinking mode (0 / 1) and conversation: `text` / `ids` = what runq's rq_chat_render must give
(`runq -m render`, checked by tests/check_models.py on tokenizer-only GGUFs carrying each template), built here
from runq's documented canonical form (src/runq.h, src/chat.c); `hf` = whether that equals HF's
apply_chat_template(add_generation_prompt=True, enable_thinking=thinking) (asserted for the cases marked so).
The known differences, by design: the hybrid template with thinking off keeps the empty think block in every
user turn and earlier replies verbatim (the CLI keeps the conversation in the KV cache as generated; HF drops
both from earlier turns); Thinking-2507 with thinking off is not in its template (runq serve's constrained
replies only). Qwen3.5 renders as Thinking-2507 with thinking on and as the hybrid with it off (the same known
difference), and trims every message text (|trim).
"""
import json
import os
import sys

import transformers
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES = os.path.join(ROOT, "tests", "golden", "templates")
OUT = os.path.join(ROOT, "tests", "golden", "chat_render.json")
VARIANTS = ["qwen3", "instruct-2507", "thinking-2507", "qwen3.5"]
TOKENIZERS = {"qwen3.5": "Qwen3.5-0.8B"}  # (else Qwen3-0.6B)
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
    print(f"wrote {OUT}: {len(cases)} cases, {sum(c['hf'] for c in cases)} equal to HF's rendering")
    return 0


if __name__ == "__main__":
    sys.exit(main())
