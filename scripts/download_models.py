#!/usr/bin/env python3
"""Download pinned Qwen3-0.6B model files from Hugging Face and verify them.

Uses only the Python standard library. Revisions are pinned to fixed commits.
Every file is checked: LFS files by sha256, small files by their git blob sha1.
Interrupted downloads resume from the partial .part file.

Usage:
    scripts/download_models.py                 # default sets: hf q4km
    scripts/download_models.py hf q4km q8gguf  # pick sets explicitly
    scripts/download_models.py --list

Files go to models/<set-dir>/ (override with --dest or $CWAYN_MODELS).
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
    # Q8_0 GGUF from the same repo, for quality/speed comparisons with GGUF-based engines.
    "q8gguf": ("unsloth/Qwen3-0.6B-GGUF", "50968a4468ef4233ed78cd7c3de230dd1d61a56b", "Qwen3-0.6B-GGUF", [
        ("Qwen3-0.6B-Q8_0.gguf", 639447744, "sha256", "e150ed544dfe6016930c026a93913a5e3184181ebfe6ab2223ae01dd0491784c"),
    ]),
}
DEFAULT_SETS = ["hf", "q4km"]
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
    req = urllib.request.Request(url, headers={"User-Agent": "cwayn-download/1"})
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
    ap.add_argument("--dest", default=os.environ.get("CWAYN_MODELS", str(ROOT / "models")))
    ap.add_argument("--list", action="store_true", help="list available sets and exit")
    args = ap.parse_args()

    if args.list:
        for name, (repo, rev, sub, files) in SETS.items():
            total = sum(f[1] for f in files) / 1e6
            print(f"{name:8s} {repo}@{rev[:10]} -> {sub}/  ({total:.0f} MB)")
        return
    names = args.sets or DEFAULT_SETS
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
