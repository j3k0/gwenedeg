#!/bin/bash
# usage: run.sh LABEL CMD...  — flock the bench lock, run CMD in a 22G/no-swap user scope, log load before/after
label=$1; shift
cd runs || exit 1
{
  echo "== $label start $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg) avail $(awk '/MemAvailable/{printf "%.1fG", $2/1048576}' /proc/meminfo)"
  echo "cmd: $*"
} | tee -a runs.log
flock /tmp/cwayn-bench.lock systemd-run --user --scope -q -p MemoryMax=22G -p MemorySwapMax=0 "$@"
rc=$?
echo "== $label end $(date -u +%FT%TZ) rc $rc load $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a runs.log
exit $rc
