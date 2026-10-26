# Upstream

- Repo: https://github.com/adriancable/qwen3.c
- Commit: `55643ffbe4ba39fdb4530bb7828102d0dc91fb40` (2025-07-05)
- License: MIT (see `LICENSE`)

This directory is kept pristine (identical to upstream, apart from this file) so that
our changes can be diffed against it. Gwenedeg's engine started as a copy of `runq.c` (`src/runq.c`,
until Step 8 of TODO.md split it into the modules of `src/`; the single file is in git history):

    git show 87e08aa:src/runq.c | diff -u third_party/qwen3.c/runq.c -   # upstream + the -n / [timing] patch

- `export.py` and `model.py` are used as-is by `scripts/export_q8.sh` (HF weights to Q8_0 `.bin`).
- The engine we build and modify lives in `src/` (built by the top-level `Makefile`).
  Nothing here is compiled.
