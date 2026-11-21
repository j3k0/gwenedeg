#!/usr/bin/env bash
# A/B prefill tok/s of several runq binaries on one model, runs alternating, best of REPS (PP-RESULTS.md). Env PROMPT
# (a prompt file, default /var/tmp/pp-prompt1500.txt: scripts/bench_prefill.sh -m MODEL -P 1500 > it), CTX, T (threads, 8).
# usage: prefill_ab.sh MODEL REPS "bin1 bin2 ..." [extra runq args]; prints best prefill tok/s per binary (alternating runs)
M=$1; R=$2; BINS=$3; shift 3
P="$(cat ${PROMPT:-/var/tmp/pp-prompt1500.txt})"
declare -A best
echo "load before: $(cut -d' ' -f1-3 /proc/loadavg)"
for r in $(seq $R); do for b in $BINS; do
  l=$(OMP_NUM_THREADS=${T:-8} nice -n 10 $b $M -c ${CTX:-4096} -t 0 -n 1 "$@" -i "$P" 2>&1 >/dev/null | grep '\[timing\]')
  p=$(sed -E 's/.*\(([0-9.]+) tok\/s\).*generated.*/\1/' <<<"$l")
  best[$b]=$(awk -v a="${best[$b]:-0}" -v b="$p" 'BEGIN{print (b>a)?b:a}')
done; done
for b in $BINS; do echo "$b ${best[$b]} tok/s"; done
echo "load after: $(cut -d' ' -f1-3 /proc/loadavg)"
