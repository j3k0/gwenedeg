# Upstream

- Repo: https://github.com/adriancable/qwen3.c
- Commit: `55643ffbe4ba39fdb4530bb7828102d0dc91fb40` (2025-07-05)
- License: MIT (see `LICENSE`)

This directory is kept pristine (identical to upstream, apart from this file) so that
our changes can be diffed against it:

    diff -u third_party/qwen3.c/runq.c src/runq.c

- `export.py` and `model.py` are used as-is by `scripts/export_q8.sh` (HF weights to Q8_0 `.bin`).
- The engine we build and modify lives in `src/runq.c` (built by the top-level `Makefile`).
  Nothing here is compiled.
