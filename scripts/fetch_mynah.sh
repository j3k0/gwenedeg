#!/usr/bin/env bash
# Fetch mynah-slm (reference engine, benchmarks only) at a pinned commit and build it.
# Not vendored: lives in third_party/mynah-slm/ (gitignored).
# Usage: scripts/fetch_mynah.sh        (idempotent; needs libopenblas-dev to build)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
URL="https://github.com/mynah-org/mynah-slm"
SHA="cde15993051e02f290fca60a3febc7143716983b"
DIR="$ROOT/third_party/mynah-slm"

if [ -d "$DIR/.git" ]; then
  head=$(git -C "$DIR" rev-parse HEAD 2>/dev/null || echo none)
  if [ "$head" = "$SHA" ]; then
    echo "mynah-slm: present at pinned commit ${SHA:0:7}"
  else
    echo "WARNING: $DIR is at $head, expected $SHA (left as-is)" >&2
  fi
elif [ -e "$DIR" ]; then
  echo "ERROR: $DIR exists but is not a git checkout; remove it and re-run" >&2
  exit 1
else
  echo "mynah-slm: fetching $URL @ ${SHA:0:7}"
  mkdir -p "$DIR"
  git -C "$DIR" init -q
  git -C "$DIR" remote add origin "$URL"
  git -C "$DIR" fetch -q --depth 1 origin "$SHA"
  git -C "$DIR" checkout -q FETCH_HEAD
fi

make -C "$DIR" -j"$(nproc)"
