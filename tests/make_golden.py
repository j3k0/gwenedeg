#!/usr/bin/env python3
"""Generate the HF golden references for the cwayn test suite (run once, commit the output).

Usage: .venv/bin/python tests/make_golden.py [--tokenizer-only]
       .venv/bin/python tests/make_golden.py --q4ref [--gguf-py DIR]
Needs torch + transformers (.venv); reads models/Qwen3-0.6B only (local_files_only, no network).
--tokenizer-only rewrites tests/golden/tokenizer.json only (no model load, ~2 s).
--q4ref writes only tests/golden/logits_q4/*.npz (~20 s): the same prompts and format, from the
same HF model with every weight replaced by the Q4_K_M GGUF's tensors dequantized by gguf-py
(gguf.quants.dequantize; norms F32 as stored), i.e. Q4_K_M weights with fp32 activations. runq on
the Q4_K_M file should differ from it by its int8 activations only. Needs the `gguf` package
(pip install gguf, 0.19.0 used; pure Python): importable, or DIR = a directory containing it
(e.g. a scratch venv's site-packages; the .venv is not modified).
The greedy goldens (our own engine) are made by `tests/run_tests.sh --update-greedy` instead.

Output format
-------------
tests/golden/tokenizer.json
    {"meta": {...versions...},
     "cases": [{"name": str, "text": str, "ids": [int, ...]}, ...]}
    ids = tokenizer.encode(text, add_special_tokens=False) with the HF tokenizer.
    Optional "xfail": str marks a known engine bug (reported, does not fail the suite);
    the marks come from TOKENIZER_XFAIL below.

tests/golden/logits/<name>.npz   (np.savez_compressed, no pickles), one per prompt
tests/golden/logits_q4/<name>.npz (--q4ref; the reference: HF with the Q4_K_M weights):
    text        0-d unicode array, the raw prompt (no chat template applied)
    ids         int32 (n,)        HF token ids, the model input
    subset_ids  int32 (S,)        fixed sorted vocab subset, S = 4096,
                                  np.random.default_rng(0).choice(vocab, S, replace=False)
    subset      float16 (n, S)    HF logits at subset_ids, row i = logits after token i
    top_ids     int32 (n, 64)     HF top-64 token ids per position, descending by logit
    top_logits  float32 (n, 64)   their logits
    The HF reference is Qwen3ForCausalLM in float32 on CPU (eager attention), one forward
    pass over the full sequence, so row i only sees tokens 0..i (causal).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen3-0.6B")
Q4_GGUF = os.path.join(ROOT, "models", "Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf")
# llama.cpp tensor roles (blk.N.<role>.weight) -> HF Qwen3 module names (no Q/K permutation for qwen3)
GGUF_ROLES = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
              "attn_output": "self_attn.o_proj", "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
              "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm", "ffn_gate": "mlp.gate_proj",
              "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
OUT_DIR = os.path.join(ROOT, "tests", "golden")

SUBSET_SIZE = 4096
TOPK = 64

TOKENIZER_CASES = [
    ("plain", "The quick brown fox jumps over the lazy dog."),
    ("hello", "Hello, world!"),
    ("leading_spaces", "   leading spaces"),
    ("trailing_spaces", "trailing spaces   "),
    ("multi_spaces", "multiple    spaces   between  words"),
    ("newlines", "line one\nline two\n\nline four\n"),
    ("tabs", "a\tb\t\tc\n\tindented"),
    ("digits", "1234567890"),
    ("numbers", "In 2024 there were 365 days; pi is 3.14159 and e is 2.71828."),
    ("punctuation", "Wait... what?! (yes) [no] {maybe} -- #1 @user $5.00 100% a&b"),
    ("contractions", "I'm sure they don't know, but we'll see. It's fine, you've done it."),
    ("code_c", "int main(void) {\n    printf(\"%d\\n\", 42);\n    return 0;\n}\n"),
    ("code_python", "def f(x):\n    if x > 0:\n        return x * 2\n    return None\n"),
    ("french", "Café crème brûlée à la française, où est-ce ?"),
    ("german", "Größe, Übermäßig, Straße, Äpfel"),
    ("chinese", "你好，世界！今天天气很好。"),
    ("japanese", "日本語のテキストです。カタカナとひらがな"),
    ("emoji", "I love pizza 🍕🎉 and 👍🏽"),
    ("mixed_unicode", "Mixed: English, français, 中文, 日本語, emoji 🚀."),
    ("long_word", "Pneumonoultramicroscopicsilicovolcanoconiosis"),
    ("url", "See https://example.com/path?q=1&r=two for details"),
    ("special_chat", "<|im_start|>user\nhi<|im_end|>"),
    ("special_think", "<think>\nLet me think.\n</think>"),
    ("angle_brackets", "if (x<y && y>z) return <not_a_token>;"),
    ("single_space", " "),
    ("single_char", "a"),
    ("single_newline", "\n"),
    # added in Step 1b (pre-tokenizer regex, byte fallback, added tokens)
    ("ws_before_word", "a b  c   d    e     f      g\t\th\t \ti"),
    ("ws_before_punct", "a ,b  ,c   (d)    [e]\t\t!f"),
    ("ws_before_newline", "a \nb  \nc   \n\nd \t \n  e\n   \n"),
    ("ws_before_digit", "x 1  2   3    45"),
    ("ws_only", "  \t \n \n\n   \t"),
    ("trailing_ws", "end with tab\t"),
    ("trailing_ws_mixed", "end with mix \t \n \n  "),
    ("crlf", "line one\r\nline two\r\n\r\nline four\r\n  indented\r\n"),
    ("cr_only", "a\rb\r\rc \r d"),
    ("indent_mixed", "def f():\n\t    x = 1\n  \tif x:\n\t\treturn 2\n    \n\t\n"),
    ("indent_deep", "{\n        {\n                deep();\n        }\n}\n"),
    ("contractions_upper", "DON'T stop. She'S here. I'M ok, WE'LL go, THEY'RE in, YOU'VE won, HE'D try."),
    ("contractions_quirks", "'sam 'Tis rock 'n' roll ''s x'ſx x'ﬆx 'LL 'lL O'Neil"),
    ("digits_letters", "abc123def x9y 0x1F v2.0.1-rc3"),
    ("digits_long", "3.14159 1,000,000 12345678901234567890 -42 +7e-10"),
    ("digits_unicode", "٣٤٥ १२३ ①② Ⅻ ½ x² 〇"),
    ("punct_runs", "!!! ... ?! ??? !!!\n\n --- *** ### ``` ~~~ // /* */ ->"),
    ("punct_newlines", "end.\n\nNext!\r\n:)\n"),
    ("nbsp", "100\u00a0km, a\u00a0\u00a0b \u00a0c\u00a0"),
    ("ideographic_space", "日本\u3000語\u3000\u3000です \u3000x"),
    ("unicode_spaces", "a\u2009b\u202fc\u205fd\u2028e\u2029f\u0085g\u1680h\u2002\u200ai"),
    ("not_spaces", "x\x1c\x1cy\x1fz a\u200bb\ufeffc\u180ed\u2060e"),
    ("control_chars", "a\x01b\x07\x1b[0m\x7f"),
    ("emoji_zwj", "family 👨‍👩‍👧‍👦 and 👩🏽‍💻 and 🏳️‍🌈"),
    ("emoji_flags", "flags 🇺🇸🇯🇵🇩🇪 🇫🇷!"),
    ("emoji_skin", "👋🏻👋🏼👋🏽👋🏾👋🏿 ✌️ 1️⃣ ❤️"),
    ("cjk_rare", "𠀀𠀁𪚥𰻞𮯰 龘靐齉 鬱"),
    ("arabic", "مرحبا بالعالم، كيف حالك؟ ١٢٣"),
    ("hindi", "नमस्ते दुनिया, आप कैसे हैं?"),
    ("korean", "안녕하세요 세계! 반갑습니다."),
    ("russian", "Привет, мир! Как дела? ЁЛКА ёлка"),
    ("math", "∀x ∈ ℝ: x² ≥ 0, ∑ᵢ aᵢ ≤ ∞, √2 ≈ 1.414, ½ + ¼ = ¾, ∫₀¹ f(x) dx"),
    ("mixed_scripts", "Ελληνικά עברית ไทย ქართული Հայերեն 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 ｆｕｌｌｗｉｄｔｈ"),
    ("long_no_spaces", "".join("abcdefghijklmnopqrstuvwxyz0123456789.,;-_"[(i * 7 + i // 13) % 41]
                                   for i in range(3000))),
    ("long_letters", "Supercalifragilistic" * 150),
    ("special_adjacent", "hi<|im_end|>there<|im_start|>x"),
    ("special_back_to_back", "<|im_start|><|im_end|><think></think><|endoftext|>"),
    ("special_others", "a<tool_call>{}</tool_call>b<|fim_prefix|>p<|fim_suffix|>s<|fim_middle|>"),
    ("special_partial", "<<|im_start|>> <|im_start <|im_end| <thinking> </ think>"),
    ("special_in_ws", "  <|im_start|>  user \n<think>\n\n</think>\n\n"),
    ("nfc_decomposed", "cafe\u0301 n\u0303"),
    ("nfc_singleton", "a\u2000b \u212b \u2126 \uf900"),  # NFC maps these to U+2002, U+00C5, U+03A9, U+8C48
    # added in Step 5 (NFC normalizer)
    ("nfc_latin", "e\u0301 A\u030a E\u0301cole A\u030angstro\u0308m vs \u00c5ngstr\u00f6m, u\u0308\u0304 n\u0303o\u0302"),
    ("nfc_mark_order", "q\u0307\u0323 q\u0323\u0307 s\u0307\u0323 d\u0323\u0307 a\u0301\u0316 x\u0300\u0315\u0301 "
                       "\u1ea1\u0302 o\u031b\u0301\u0323 a\u0305\u0301 e\u0301\u0301 \u1100\u0301\u1161"),  # blocked: same ccc, starter
    ("nfc_hangul_jamo", "\u1112\u1161\u11ab\u1100\u116e\u11a8 \u1100\u1161 \uac00\u11a8 \u1100\u1100\u1161 "
                        "\u1161\u11a8 \u1100\u1161\u11a8\u11a8 \uac00\u11a7 \ud55c\u1100\u1161"),
    ("nfc_vietnamese", "Tie\u0302\u0301ng Vie\u0323\u0302t co\u0301 da\u0302\u0301u: Nguye\u0302\u0303n, "
                       "ngu\u031bo\u031b\u0300i, \u0111u\u031bo\u031b\u0323c, pho\u031b\u0309, a\u0306\u0323"),
    ("nfc_singletons", "\u212b \u2126 \u212a \u2000\u2001x \uf900\uf901\ufa10 \U0002f800 a\u0340\u0341 "
                       "\u0374\u037e\u0387 \u1fef \u1f71 \u0344"),
    ("nfc_exclusions", "\u0958\u0959 \u0915\u093c\u0916\u093c \u095f \ufb1d \ufb2c \u2adc \U0001d15e\U0001d160 "
                       "\u0f73\u0f71\u0f72"),
    ("nfc_greek", "\u0391\u0301\u03b8\u03b7\u0342\u03bd\u03b1\u03b9 \u03b9\u0308\u0301 \u03c5\u0308\u0301 "
                  "\u03b1\u0313\u0301\u0345 \u03bb\u03bf\u0301\u03b3\u03bf\u03c2 \u03ca\u0301 \u03b5\u0301"),
    ("nfc_long_marks", "a" + "\u0301\u0316\u0334" * 200 + " e" + "\u0307" * 40 + "\u0323" * 40 + " z\u0301\u0300"
                       + " o" + "\u0301\u0300\u0316\u0317\u0334" * 20),  # stable order within one ccc
    ("nfc_expansion", "\U0001d160" * 50 + " \u0958" * 20),  # more tokens than input bytes
    ("nfc_special_adjacent", "<|im_start|>\u0301e\u0301<|im_end|>\u0301 cafe<|im_end|>\u0301 \u1100<think>\u1161"
                             "</think>\u11a8 a<|endoftext|>\u030a"),
    ("nfc_mixed", "Plain ASCII text first, then a decomposed word: re\u0301sume\u0301, and more plain text "
                  "after it; \u4e2d\u6587 \uac00\u11a8 \u00e9\u0301 \u212b."),
    ("nfc_hf_compat", "a\u1df6\u0301 a\u0301\u1df6 \U00011935\U00011930 x\u0d3b\u0323"),  # HF's Unicode 9.0 data
]

# Known runq tokenizer limitations (TODO.md): reported as XFAIL, do not fail the suite.
# (none since Step 5: the two NFC cases above pass)
TOKENIZER_XFAIL = {}

LOGITS_PROMPTS = [
    ("en", "The quick brown fox jumps over the lazy dog. The capital of France is"),
    # (No multi-space indentation here: written when runq's tokenizer still merged whole
    # whitespace runs; kept unchanged so the goldens stay byte-identical.)
    ("code", "int add(int a, int b) { return a + b; }\nint main() { return add(2,"),
    ("zh", "中国的首都是北京。法国的首都是巴黎。日本的首都是"),
    ("chat", "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"),
    ("count", "Counting: one, two, three, four, five, six,"),
]


def main():
    ap = argparse.ArgumentParser(description="Generate the HF goldens for the cwayn tests.")
    ap.add_argument("--tokenizer-only", action="store_true", help="only rewrite golden/tokenizer.json")
    ap.add_argument("--q4ref", action="store_true", help="only write golden/logits_q4 (HF with the Q4_K_M weights)")
    ap.add_argument("--gguf-py", default="", help="directory containing the gguf package (--q4ref)")
    a = ap.parse_args()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    meta = {"model": "Qwen/Qwen3-0.6B", "torch": torch.__version__,
            "transformers": transformers.__version__, "numpy": np.__version__}
    if a.q4ref:
        return write_logits(tok, os.path.join(OUT_DIR, "logits_q4"), q4_weights=Q4_GGUF, gguf_py=a.gguf_py)

    tok_path = os.path.join(OUT_DIR, "tokenizer.json")
    cases = []
    for name, text in TOKENIZER_CASES:
        c = {"name": name, "text": text, "ids": tok.encode(text, add_special_tokens=False)}
        if name in TOKENIZER_XFAIL:
            c["xfail"] = TOKENIZER_XFAIL[name]
        cases.append(c)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(tok_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "cases": cases}, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"wrote {tok_path} ({len(cases)} cases)")
    if a.tokenizer_only:
        return 0

    return write_logits(tok, os.path.join(OUT_DIR, "logits"))


def load_q4_weights(model, path, gguf_py):
    """replace every weight of the HF model with the GGUF tensor of the same role, dequantized"""
    if gguf_py:
        sys.path.insert(0, gguf_py)
    import gguf
    from gguf.quants import dequantize
    sd = model.state_dict()
    for t in gguf.GGUFReader(path).tensors:
        if t.name == "token_embd.weight":
            key = "model.embed_tokens.weight"  # (lm_head is tied to it)
        elif t.name == "output_norm.weight":
            key = "model.norm.weight"
        else:
            _, layer, role, _ = t.name.split(".")
            key = f"model.layers.{layer}.{GGUF_ROLES[role]}.weight"
        w = dequantize(t.data, t.tensor_type).astype(np.float32).reshape(tuple(sd[key].shape))
        with torch.no_grad():
            sd[key].copy_(torch.from_numpy(w))
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    print(f"weights from {os.path.relpath(path, ROOT)} (gguf-py at {os.path.dirname(gguf.__file__)})")


def write_logits(tok, log_dir, q4_weights=None, gguf_py=""):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager")
    model.eval()
    if q4_weights:
        load_q4_weights(model, q4_weights, gguf_py)
    vocab = model.config.vocab_size
    subset_ids = np.sort(np.random.default_rng(0).choice(vocab, SUBSET_SIZE, replace=False)).astype(np.int32)

    os.makedirs(log_dir, exist_ok=True)
    for name, text in LOGITS_PROMPTS:
        ids = tok.encode(text, add_special_tokens=False)
        with torch.no_grad():
            logits = model(torch.tensor([ids])).logits[0].float().numpy()  # (n, vocab)
        assert logits.shape == (len(ids), vocab)
        top_ids = np.argsort(-logits, axis=1, kind="stable")[:, :TOPK].astype(np.int32)
        top_logits = np.take_along_axis(logits, top_ids, axis=1).astype(np.float32)
        path = os.path.join(log_dir, f"{name}.npz")
        np.savez_compressed(path, text=np.array(text), ids=np.array(ids, dtype=np.int32),
                            subset_ids=subset_ids, subset=logits[:, subset_ids].astype(np.float16),
                            top_ids=top_ids, top_logits=top_logits)
        print(f"wrote {path} ({len(ids)} tokens, {os.path.getsize(path)} bytes, "
              f"top-1 after last token: {tok.decode([int(top_ids[-1, 0])])!r})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
