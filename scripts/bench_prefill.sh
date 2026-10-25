#!/usr/bin/env bash
# Prefill benchmark: prompt tok/s vs prompt length, prefill chunk size (-b) and threads.
# Usage: scripts/bench_prefill.sh [-l "450 1500"] [-b "1 64 256 512"] [-t "8"] [-r REPS] [-m model.bin]
#        scripts/bench_prefill.sh -P 450    # only print the prompt of that length
#   -l  target prompt lengths in tokens (incl. the chat template), approximate (+-10)
#   -b  chunk sizes; 1 = token-by-token forward() (the old path)
#   -t  OMP_NUM_THREADS values
#   -r  runs per cell, best kept (default 3)
# Chat mode, greedy, -n 16, -c 4096. The prompt is a fixed paragraph repeated and cut to length,
# so runs are reproducible. Column "out" checks that each run's generated text equals that of
# the first chunk size in the list (batched prefill must not change the output).
# Defaults take ~2.5 min, mostly the -b 1 runs at 1,500 tokens; -r 1 -b "1 512" ~30 s.
# Env: RUNQ=binary (default ./runq); ALL=1 adds a column with every run's prefill/decode t/s
# (run-to-run spread). Check /proc/loadavg first: the machine may be shared.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNQ="${RUNQ:-$ROOT/runq}"
MODEL="$ROOT/models/Qwen3-0.6B-q8.bin"
LENS="450 1500"; CHUNKS="1 64 256 512"; THREADS="8"; REPS=3; PRINT=""
while getopts "l:b:t:r:m:P:h" o; do
  case $o in
    l) LENS=$OPTARG ;; b) CHUNKS=$OPTARG ;; t) THREADS=$OPTARG ;; r) REPS=$OPTARG ;; m) MODEL=$OPTARG ;; P) PRINT=$OPTARG ;;
    *) sed -n '2,13p' "$0"; exit 1 ;;
  esac
done
[ -x "$RUNQ" ] || make -C "$ROOT" -s runq
ERR=$(mktemp); trap 'rm -f "$ERR"' EXIT

PARA="The history of the printing press begins long before Gutenberg. Carved wooden blocks were \
used in China to print text and images on paper and silk, and movable type made of baked clay, \
later of wood and bronze, appeared there and in Korea centuries before it reached Europe. Around \
1440, in Mainz, Johannes Gutenberg combined several existing technologies into a practical \
system: an adjustable hand mould for casting uniform metal type, an oil-based ink that adhered \
to metal, and a screw press adapted from those used to make wine and paper. The result was a \
dramatic reduction in the cost of books. Within fifty years, presses were operating in more \
than two hundred cities, and millions of volumes had been printed. Scholars could compare \
identical copies of a text, errors could be corrected in later editions, and ideas spread \
faster than any authority could suppress them. Questions for the reader: what changed first, \
and why did it matter?"
read -r -a WORDS <<< "$PARA"
NW=${#WORDS[@]}

ntok() {  # tokens of the chat-rendered prompt = raw tokens + template (measured once)
  local n
  n=$("$RUNQ" "$MODEL" -c 64 -m tokenize -i "$1" 2>/dev/null | wc -w)
  echo $((n + TEMPLATE))
}
words() { local out="" i; for ((i = 0; i < $1; i++)); do out+="${WORDS[i % NW]} "; done; echo "${out% }"; }
# template overhead: [timing] prompt count of a 1-word chat prompt minus its raw token count
t_line=$(OMP_NUM_THREADS=1 "$RUNQ" "$MODEL" -c 64 -t 0 -n 1 -b 1 -i "Hello" 2>&1 >/dev/null | grep '\[timing\]')
TEMPLATE=$(( $(sed -E 's/.*prompt: ([0-9]+) tok.*/\1/' <<<"$t_line") - 1 ))

make_prompt() {  # shortest word prefix with >= target tokens (binary search on the word count)
  local target=$1 lo=1 hi=1
  while [ "$(ntok "$(words $hi)")" -lt "$target" ]; do lo=$hi; hi=$((hi * 2)); done
  while [ $lo -lt $hi ]; do
    local mid=$(((lo + hi) / 2))
    if [ "$(ntok "$(words $mid)")" -lt "$target" ]; then lo=$((mid + 1)); else hi=$mid; fi
  done
  words $lo
}

if [ -n "$PRINT" ]; then make_prompt "$PRINT"; exit 0; fi
cat "$MODEL" > /dev/null   # warm the page cache

printf "%-7s %-7s %-6s %12s %11s %5s\n" prompt threads chunk "prefill t/s" "decode t/s" out
for len in $LENS; do
  prompt=$(make_prompt "$len")
  for t in $THREADS; do
    ref=""
    for b in $CHUNKS; do
      best_p=0; best_d=0; same=yes; n=0; all=""
      for _ in $(seq "$REPS"); do
        out=$(OMP_NUM_THREADS=$t "$RUNQ" "$MODEL" -c 4096 -t 0 -n 16 -b "$b" -i "$prompt" 2>"$ERR") || { cat "$ERR" >&2; exit 1; }
        line=$(grep '\[timing\]' "$ERR")
        n=$(sed -E 's/.*prompt: ([0-9]+) tok.*/\1/' <<<"$line")
        p=$(sed -E 's/.*prompt: [0-9]+ tok in [0-9.]+ s \(([0-9.]+) tok\/s\).*/\1/' <<<"$line")
        d=$(sed -E 's/.*generated: [0-9]+ tok in [0-9.]+ s \(([0-9.]+) tok\/s\).*/\1/' <<<"$line")
        best_p=$(awk -v a="$best_p" -v b="$p" 'BEGIN{print (b>a)?b:a}')
        best_d=$(awk -v a="$best_d" -v b="$d" 'BEGIN{print (b>a)?b:a}')
        all+=" $p/$d"
        [ -z "$ref" ] && ref=$out
        [ "$out" == "$ref" ] || same=NO
      done
      printf "%-7s %-7s %-6s %12s %11s %5s" "$n" "$t" "$b" "$best_p" "$best_d" "$same"
      if [ -n "${ALL:-}" ]; then echo "  all:$all"; else echo; fi
    done
  done
done
