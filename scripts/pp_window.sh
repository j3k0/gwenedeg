#!/usr/bin/env bash
# The Qwen3.8-27B measurements of PP-RESULTS.md (the i-quants' tiled prefill GEMM): needs a measurement window (the 27B
# takes ~12.5 GB of RAM with --repack; stop the live server first). Builds everything it runs in a temporary directory
# (the base commit and this tree, plain, -DGWENEDEG_PROFILE and -DRQ_BATCH_MAX=2048), then:
#   1. bit-identity on the 27B: -m logits of a ~300-token prompt, base -b 512 == new -b 512 == new -b 1 == new -b 77
#      (3 threads) == new --repack -b 512 (sha256 of the logits files)
#   2. prefill tok/s, base vs new, a ~2,500-token prompt, -b 512, --repack, 8 / 14 / 16 threads (best of REPS); new
#      without --repack at 14 threads
#   3. the phase split (-DGWENEDEG_PROFILE), base vs new, 14 threads, --repack
#   4. -b 256 / 512 / 1024 / 2048 (the RQ_BATCH_MAX=2048 build), ~4,100-token prompt, 14 threads, --repack
# Usage: scripts/pp_window.sh [MODEL [BASE_COMMIT]]   (env REPS, default 2; JOBS: make -j, default 8;
#        TH: step 2's thread counts, default "8 14 16"; TP: the other steps' threads, default 14; SKIP="1 4": skip those steps)
# Duration ~20-25 min on the 16-vCPU Genoa box; peak RSS ~13 GB (one gwenedeg at a time). Output: bench/pp-window-DATE/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="${1:-/opt/projects/tools/gwenedeg/models/Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf}"
BASE="${2:-$(git -C "$ROOT" merge-base HEAD main)}"  # (the branch point: base and new differ by this branch only)
REPS="${REPS:-2}"
JOBS="${JOBS:-8}"
TH="${TH:-8 14 16}"   # thread counts of step 2
TP="${TP:-14}"        # threads of steps 1, 3, 4 (production: 14)
SKIP=" ${SKIP:-} "
OUT="$ROOT/bench/pp-window-$(date +%F)"
mkdir -p "$OUT"
W="$(mktemp -d /var/tmp/ppw.XXXXXX)"
trap 'rm -rf "$W"' EXIT
LOG="$OUT/log.txt"
say() { echo "$*" | tee -a "$LOG"; }
say "== $(date -Is) model $M, base $BASE, new $(git -C "$ROOT" rev-parse --short HEAD)$(git -C "$ROOT" diff --quiet || echo +dirty)"

# builds (-j$JOBS, default 8: the window is ours)
mkdir -p "$W/base" "$W/new"
git -C "$ROOT" archive "$BASE" | tar -x -C "$W/base"
(cd "$ROOT" && git ls-files -z | xargs -0 tar -c) | tar -x -C "$W/new"
for t in base new; do
    # (a base from before 2026-11-22: the target was runq, the macro RUNQ_PROFILE)
    T=gwenedeg; P=GWENEDEG_PROFILE; grep -q '^gwenedeg:' "$W/$t/Makefile" || { T=runq; P=RUNQ_PROFILE; }
    make -C "$W/$t" -s -j$JOBS $T >/dev/null && mv "$W/$t/$T" "$W/$t/gwenedeg-plain"
    make -C "$W/$t" -s -j$JOBS $T B=build/prof CFLAGS="-Ofast -fopenmp -march=native -D$P" >/dev/null && mv "$W/$t/$T" "$W/$t/gwenedeg-prof"
done
make -C "$W/new" -s -j$JOBS gwenedeg B=build/b2048 CFLAGS="-Ofast -fopenmp -march=native -DRQ_BATCH_MAX=2048" >/dev/null && mv "$W/new/gwenedeg" "$W/new/gwenedeg-b2048"
B0="$W/base/gwenedeg-plain" B1="$W/new/gwenedeg-plain"

# prompts: the bench paragraph (scripts/bench_prefill.sh) cut to a word count (~1.19 tokens per word + the template)
PARA=$(sed -n '/^PARA="/,/^read /p' "$ROOT/scripts/bench_prefill.sh" | sed '1s/^PARA="//; $d' | sed '$s/"$//' | tr -d '\\' | tr '\n' ' ')
read -r -a WORDS <<< "$PARA"
words() { local out="" i; for ((i = 0; i < $1; i++)); do out+="${WORDS[i % ${#WORDS[@]}]} "; done; echo "${out% }"; }
words 250 > "$W/p300.txt"; words 2100 > "$W/p2500.txt"; words 3450 > "$W/p4100.txt"
load() { cut -d' ' -f1-3 /proc/loadavg; }

if [[ "$SKIP" != *" 1 "* ]]; then
    say "== 1. bit-identity (-m logits, ~300 tokens; load $(load))"
    run_l() { local th=$1; shift; OMP_NUM_THREADS=$th "$@" -c 2048 -m logits -i "$(cat "$W/p300.txt")" -o "$W/l.bin" 2>>"$OUT/stderr.txt" >/dev/null; sha256sum < "$W/l.bin" | cut -c1-16; }
    say "base -b 512          $(run_l $TP "$B0" "$M" -b 512)"
    say "new  -b 512          $(run_l $TP "$B1" "$M" -b 512)"
    say "new  -b 1            $(run_l $TP "$B1" "$M" -b 1)"
    say "new  -b 77, 3 thr    $(run_l 3 "$B1" "$M" -b 77)"
    say "new  --repack -b 512 $(run_l $TP "$B1" "$M" --repack -b 512)"
    rm -f "$W/l.bin"
fi

pf() { # threads bin args...: prompt tok/s of one run (the [timing] line)
    local th=$1 bin=$2; shift 2
    OMP_NUM_THREADS=$th "$bin" "$M" -c 8192 -t 0 -n 1 "$@" 2>"$W/err.txt" >/dev/null || { cat "$W/err.txt" >&2; return 1; }
    cat "$W/err.txt" >> "$OUT/stderr.txt"
    grep '\[timing\]' "$W/err.txt" | sed -E 's/.*prompt: ([0-9]+) tok in ([0-9.]+) s \(([0-9.]+) tok\/s\).*/\1 tok \2 s \3 tok\/s/'
}
if [[ "$SKIP" != *" 2 "* ]]; then
    say "== 2. prefill, ~2,500 tokens, -b 512, best of $REPS (load $(load))"
    for th in $TH; do
        for r in $(seq "$REPS"); do
            say "  $th thr base --repack: $(pf "$th" "$B0" --repack -b 512 -i "$(cat "$W/p2500.txt")")"
            say "  $th thr new  --repack: $(pf "$th" "$B1" --repack -b 512 -i "$(cat "$W/p2500.txt")")"
        done
    done
    say "  $TP thr new, no --repack: $(pf $TP "$B1" -b 512 -i "$(cat "$W/p2500.txt")")"
    say "  $TP thr base, no --repack: $(pf $TP "$B0" -b 512 -i "$(cat "$W/p2500.txt")")"
    say "  (load $(load))"
fi

if [[ "$SKIP" != *" 3 "* ]]; then
    say "== 3. phase split, $TP threads, --repack, ~2,500 tokens (load $(load))"
    for t in base new; do
        OMP_NUM_THREADS=$TP "$W/$t/gwenedeg-prof" "$M" -c 8192 -t 0 -n 1 --repack -b 512 -i "$(cat "$W/p2500.txt")" 2>"$W/err.txt" >/dev/null
        say "-- $t"; grep -E '\[timing\]|\[profile\]' "$W/err.txt" | tee -a "$LOG"
    done
fi

if [[ "$SKIP" != *" 4 "* ]]; then
    say "== 4. -b, ~4,100 tokens, $TP threads, --repack, the RQ_BATCH_MAX=2048 build (load $(load))"
    for b in 256 512 1024 2048; do
        say "  -b $b: $(pf $TP "$W/new/gwenedeg-b2048" --repack -b "$b" -i "$(cat "$W/p4100.txt")")"
    done
fi
say "== done $(date -Is) (load $(load)); $OUT"
