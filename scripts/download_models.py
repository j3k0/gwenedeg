#!/usr/bin/env python3
"""Download pinned Qwen3 model files from Hugging Face and verify them.

Uses only the Python standard library. Revisions are pinned to fixed commits.
Every file is checked: LFS files by sha256, small files by their git blob sha1.
Interrupted downloads resume from the partial .part file.

Usage:
    scripts/download_models.py                 # default sets: hf q4km q8gguf
    scripts/download_models.py q8gguf          # pick sets explicitly (runq loads this GGUF directly)
    scripts/download_models.py family          # a group: the bigger Qwen3 models of Step 11 (~21 GB)
    scripts/download_models.py --list

Files go to models/<set-dir>/ (override with --dest or $GWENEDEG_MODELS).
Set $HF_TOKEN if you need authenticated access (not needed for these repos).
"""

import argparse
import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# name: (repo, revision, dest subdir, [(path, size, kind, digest)])
#   kind "sha256" -> LFS sha256 of the content
#   kind "blob"   -> git blob sha1 ("blob <size>\0" + content)
SETS = {
    # Official bf16 weights + tokenizer: input for qwen3.c's Q8_0 export.
    "hf": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", "Qwen3-0.6B", [
        ("LICENSE", 11343, "blob", "6634c8cc3133b3848ec74b9f275acaaa1ea618ab"),
        ("README.md", 13965, "blob", "a50b19e76f5274f9ec99f5a5d99873dca5bff25e"),
        ("config.json", 726, "blob", "f5c3703b78ae2a478ae15b247e9f855e0ce2107b"),
        ("generation_config.json", 239, "blob", "20a8a9156fc8c3f25295ca067f61fdf120d517c5"),
        ("merges.txt", 1671853, "blob", "31349551d90c7606f325fe0f11bbb8bd5fa0d7c7"),
        ("tokenizer_config.json", 9732, "blob", "417d038a63fa3de29cfde265caedae14d1a58d92"),
        ("vocab.json", 2776833, "blob", "4783fe10ac3adce15ac8f358ef5462739852c569"),
        ("tokenizer.json", 11422654, "sha256", "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
        ("model.safetensors", 1503300328, "sha256", "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"),
    ]),
    # Q4_K_M GGUF (imatrix-calibrated by Unsloth); the official Qwen GGUF repo only has Q8_0.
    "q4km": ("unsloth/Qwen3-0.6B-GGUF", "50968a4468ef4233ed78cd7c3de230dd1d61a56b", "Qwen3-0.6B-GGUF", [
        ("Qwen3-0.6B-Q4_K_M.gguf", 396705472, "sha256", "ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a"),
    ]),
    # Q8_0 GGUF from the same repo: runq loads it directly (no export step); also used by the tests
    # and for comparisons with GGUF-based engines.
    "q8gguf": ("unsloth/Qwen3-0.6B-GGUF", "50968a4468ef4233ed78cd7c3de230dd1d61a56b", "Qwen3-0.6B-GGUF", [
        ("Qwen3-0.6B-Q8_0.gguf", 639447744, "sha256", "e150ed544dfe6016930c026a93913a5e3184181ebfe6ab2223ae01dd0491784c"),
    ]),

    # Step 11: bigger models of the same architecture (Qwen3 dense: QK-norm, GQA 8 kv heads, head_dim
    # 128, NeoX RoPE, SwiGLU, vocab 151,936). GGUFs from Unsloth (imatrix Q4_K_M, as for 0.6B).
    # Qwen3-1.7B: 28 layers, dim 2048, 16 / 8 heads, ffn 6144, tied, RoPE theta 1e6.
    "1.7b-hf": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", "Qwen3-1.7B", [
        ("LICENSE", 11343, "blob", "6634c8cc3133b3848ec74b9f275acaaa1ea618ab"),
        ("README.md", 13963, "blob", "88aaef81cfe31bcdfe47203649a69ff12042826a"),
        ("config.json", 726, "blob", "044a86ecf7cb32238f3fae4184e55d354787edec"),
        ("generation_config.json", 239, "blob", "20a8a9156fc8c3f25295ca067f61fdf120d517c5"),
        ("merges.txt", 1671853, "blob", "31349551d90c7606f325fe0f11bbb8bd5fa0d7c7"),
        ("tokenizer_config.json", 9732, "blob", "417d038a63fa3de29cfde265caedae14d1a58d92"),
        ("vocab.json", 2776833, "blob", "4783fe10ac3adce15ac8f358ef5462739852c569"),
        ("tokenizer.json", 11422654, "sha256", "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
        ("model.safetensors.index.json", 25605, "blob", "986d7db875b47d21f68530f6baac038f1b297b39"),
        ("model-00001-of-00002.safetensors", 3441185608, "sha256", "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5"),
        ("model-00002-of-00002.safetensors", 622329984, "sha256", "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9"),
    ]),
    "1.7b-q8": ("unsloth/Qwen3-1.7B-GGUF", "d7f544eead698dbd1f15126ef60b45a1e1933222", "Qwen3-1.7B-GGUF", [
        ("Qwen3-1.7B-Q8_0.gguf", 1834426944, "sha256", "0becaa825564295d82e9af4d008bca5f8b7f5f73bf1c6a0b58f7c53ef26b47fd"),
    ]),
    "1.7b-q4km": ("unsloth/Qwen3-1.7B-GGUF", "d7f544eead698dbd1f15126ef60b45a1e1933222", "Qwen3-1.7B-GGUF", [
        ("Qwen3-1.7B-Q4_K_M.gguf", 1107409472, "sha256", "b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897"),
    ]),
    # Qwen3-4B: 36 layers, dim 2560, 32 / 8 heads (q dim 4096 != dim), ffn 9728, tied, RoPE theta 1e6.
    "4b-q4km": ("unsloth/Qwen3-4B-GGUF", "22c9fc8a8c7700b76a1789366280a6a5a1ad1120", "Qwen3-4B-GGUF", [
        ("Qwen3-4B-Q4_K_M.gguf", 2497281312, "sha256", "f6f851777709861056efcdad3af01da38b31223a3ba26e61a4f8bf3a2195813a"),
    ]),
    "4b-q8": ("unsloth/Qwen3-4B-GGUF", "22c9fc8a8c7700b76a1789366280a6a5a1ad1120", "Qwen3-4B-GGUF", [
        ("Qwen3-4B-Q8_0.gguf", 4280405792, "sha256", "eed555233267a33c7e8ee31682762cc7751b3f6d224039086e0e846f05fffa5d"),
    ]),
    # Qwen3-4B-Instruct-2507: the 4B shape, RoPE theta 5e6, 262,144 context, no thinking (own template).
    "4b-2507i-q4km": ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "a06e946bb6b655725eafa393f4a9745d460374c9",
                      "Qwen3-4B-Instruct-2507-GGUF", [
        ("Qwen3-4B-Instruct-2507-Q4_K_M.gguf", 2497281120, "sha256", "3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597"),
    ]),
    # Qwen3-4B-Thinking-2507: as Instruct-2507, always thinking (template opens <think>). Not in "family".
    "4b-2507t-q4km": ("unsloth/Qwen3-4B-Thinking-2507-GGUF", "f40adb104d4d44aee52f398b60597c5866a973a3",
                      "Qwen3-4B-Thinking-2507-GGUF", [
        ("Qwen3-4B-Thinking-2507-Q4_K_M.gguf", 2497281152, "sha256", "ddd52e18200baab281c5c46f70d544ce4d4fe4846eab1608f2fff48a64554212"),
    ]),
    # Qwen3-8B: 36 layers, dim 4096, 32 / 8 heads, ffn 12288, untied classifier (output.weight).
    "8b-q4km": ("unsloth/Qwen3-8B-GGUF", "a6adef130ffb23ddaf1a62fec9dced968c9bc482", "Qwen3-8B-GGUF", [
        ("Qwen3-8B-Q4_K_M.gguf", 5027784512, "sha256", "120307ba529eb2439d6c430d94104dabd578497bc7bfe7e322b5d9933b449bd4"),
    ]),

    # Qwen3.5-0.8B: the hybrid architecture (GGUF qwen35: Gated DeltaNet + gated attention, partial RoPE), not
    # supported by runq yet: docs/qwen35.md. 24 layers, dim 1024, vocab 248,320, tied. The HF checkpoint
    # (bf16, with the vision tower and the MTP head) is the input of the goldens (.venv-hf5: transformers 5.x).
    "3.5-0.8b-hf": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17", "Qwen3.5-0.8B", [
        ("LICENSE", 11544, "blob", "f938136e3adacfd92be087f6e113b5d6d97f678f"),
        ("README.md", 61705, "blob", "5824f1761b2b3a55a2141a9a1172a7f92c7c2ad9"),
        ("config.json", 2907, "blob", "715f0448b9d38103211f0ad88bbb4d6e4f4be8c9"),
        ("chat_template.jinja", 7755, "blob", "0ef09f214eaa6d9bca297988afc1454b5827b2c7"),
        ("merges.txt", 3353259, "blob", "a494e019ca1502219fd0128658b979e5f05ae8e8"),
        ("tokenizer_config.json", 16709, "blob", "fae3ce993e07c092ad024dde45e592379fde91bb"),
        ("vocab.json", 6722759, "blob", "0aa0ce0658d60ac4a5d609f4eadb0e8e43514176"),
        ("tokenizer.json", 12807982, "sha256", "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"),
        ("model.safetensors.index.json", 50900, "blob", "f691cefdb79d73270895ebd6d9594ddcecfc1838"),
        ("model.safetensors-00001-of-00001.safetensors", 1746942600, "sha256", "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"),
    ]),
    "3.5-0.8b-q8": ("unsloth/Qwen3.5-0.8B-GGUF", "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0", "Qwen3.5-0.8B-GGUF", [
        ("Qwen3.5-0.8B-Q8_0.gguf", 811843840, "sha256", "0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c"),
    ]),
    # (Q4_K / Q6_K, plus Q5_K for attn_qkv and ssm_out: needs Q5_K support)
    "3.5-0.8b-q4km": ("unsloth/Qwen3.5-0.8B-GGUF", "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0", "Qwen3.5-0.8B-GGUF", [
        ("Qwen3.5-0.8B-Q4_K_M.gguf", 532517120, "sha256", "bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517"),
    ]),
}
DEFAULT_SETS = ["hf", "q4km", "q8gguf"]
# names that stand for several sets
GROUPS = {
    "family": ["1.7b-q8", "1.7b-q4km", "1.7b-hf", "4b-q4km", "4b-q8", "4b-2507i-q4km", "8b-q4km"],
    "qwen35": ["3.5-0.8b-hf", "3.5-0.8b-q8", "3.5-0.8b-q4km"],
}
CHUNK = 1 << 20


def digest_of(path, kind, size):
    h = hashlib.sha256() if kind == "sha256" else hashlib.sha1()
    if kind == "blob":
        h.update(b"blob %d\0" % size)
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, part, size):
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink()
        have = 0
    if have == size:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "gwenedeg-download/1"})
    if os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", "Bearer " + os.environ["HF_TOKEN"])
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:  # server ignored Range: start over
            have = 0
        mode = "ab" if have else "wb"
        done, t0, last = have, time.time(), 0.0
        with open(part, mode) as out:
            while chunk := r.read(CHUNK):
                out.write(chunk)
                done += len(chunk)
                now = time.time()
                if sys.stderr.isatty() and now - last > 0.5:
                    rate = (done - have) / max(now - t0, 1e-6) / 1e6
                    print(f"\r    {done / 1e6:8.1f} / {size / 1e6:.1f} MB  {rate:6.1f} MB/s",
                          end="", file=sys.stderr, flush=True)
                    last = now
        if sys.stderr.isatty():
            print(file=sys.stderr)


def get_file(repo, rev, dest, path, size, kind, want, retries=3):
    target = dest / path
    if target.exists() and target.stat().st_size == size:
        if digest_of(target, kind, size) == want:
            print(f"  ok      {path} (already present)")
            return
        print(f"  BAD     {path}: checksum mismatch, re-downloading")
        target.unlink()
    part = target.with_name(target.name + ".part")
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{path}"
    for attempt in range(1, retries + 1):
        try:
            print(f"  fetch   {path} ({size / 1e6:.1f} MB)")
            fetch(url, part, size)
            if part.stat().st_size != size:
                raise IOError(f"size {part.stat().st_size} != expected {size}")
            got = digest_of(part, kind, size)
            if got != want:
                part.unlink()
                raise IOError(f"{kind} mismatch: got {got}, want {want}")
            part.rename(target)
            print(f"  ok      {path}")
            return
        except Exception as e:  # network errors, bad checksum
            print(f"  error   {path} (attempt {attempt}/{retries}): {e}", file=sys.stderr)
            if attempt == retries:
                raise
            time.sleep(2 * attempt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sets", nargs="*", help=f"sets to fetch (default: {' '.join(DEFAULT_SETS)})")
    ap.add_argument("--dest", default=os.environ.get("GWENEDEG_MODELS", str(ROOT / "models")))
    ap.add_argument("--list", action="store_true", help="list available sets and exit")
    args = ap.parse_args()

    if args.list:
        for name, (repo, rev, sub, files) in SETS.items():
            total = sum(f[1] for f in files) / 1e6
            print(f"{name:14s} {repo}@{rev[:10]} -> {sub}/  ({total:.0f} MB)")
        for name, members in GROUPS.items():
            total = sum(f[1] for m in members for f in SETS[m][3]) / 1e6
            print(f"{name:14s} group: {' '.join(members)}  ({total:.0f} MB)")
        return
    names = [m for n in (args.sets or DEFAULT_SETS) for m in GROUPS.get(n, [n])]
    unknown = [n for n in names if n not in SETS]
    if unknown:
        ap.error(f"unknown set(s): {', '.join(unknown)} (see --list)")

    for name in names:
        repo, rev, sub, files = SETS[name]
        dest = Path(args.dest) / sub
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[{name}] {repo} @ {rev} -> {dest}")
        for path, size, kind, want in files:
            get_file(repo, rev, dest, path, size, kind, want)
    print("done")


if __name__ == "__main__":
    main()
