#!/bin/bash
# The 27B gate (TODO S9d, PLAN-S9-mtp.md 4.4): runs one memory-heavy runq / spec_sweep job only when it fits next to
# what else may need the RAM, and caps it so that it cannot push anything else out.
#
#   scripts/s9_gate.sh [--dry] [--until TIME] [options] -- COMMAND ARGS...      (COMMAND: runq or build/spec_sweep;
#                                                                                 without --mem: the gate gives it)
# 1. The bench lock (flock /tmp/cwayn-bench.lock, held for the whole run; --lock-wait S: wait up to S seconds).
# 2. MemAvailable, SwapFree (/proc/meminfo).
# 3. The servers: every running `runq serve` (its --host / --port) and the endpoints of --health (default the
#    launch-server ports, 8418 saluki and 8417): GET /health must answer with a mem object (--admit bound / chunk) and
#    running 0; the margin is, per server, max(0, mem.budget - mem.committed) (what it may still commit) + --server-floor
#    (default 1 GiB) + its swapped-out memory (VmSwap: comes back into RAM when it runs). A server process whose
#    /health does not answer (still loading) closes the gate.
# 4. The servers to come: the timers of --timer (default runq-day.timer, which restores saluki in the morning, and
#    lcb-bench.timer, the nightly benchmark that starts servers): one whose service runs closes the gate (unless it is
#    a runq serve that answered in 3: counted there; a oneshot job is "activating" while it runs); one with a
#    next start closes it unless --until (a date(1) time) ends the run at least --guard minutes (default 5) before
#    that start. The job is killed at --until (the watchdog), so it never overlaps the start.
# 5. The scope: a probe `systemd-run --user --scope -p MemorySwapMax=0` must read memory.swap.max 0 in its cgroup.
# 6. The estimate, from the files' metadata only (no tensor data read; each probe in a scope of 1 GiB): the job's
#    own command with --mem 0.01 refuses with the model's estimate ("needs ~X GiB (weights .., runtime R, a sequence
#    of N positions: ...recurrent state S MiB ..."); + the MTP view if the job uses it (-D mtp, spec_sweep's mtp
#    settings): its block's bytes in the file (tests/ggufmini.py) x 1.10 (repacked) + its norms + R (the view's
#    context: not more than the model's runtime) + its KV cache of N positions; + the speculative loop's
#    checkpoints, S x (kmax + 2); + a draft model's estimate (runq DRAFT --mem 0.01) and its S x (kmax + 2).
#    spec_sweep --list gives its settings' kmax / mtp / draft; for runq: -c, -k, -D, -G, --lookup.
#    cap = the estimate + 1 GiB; budget = MemAvailable - the servers' margin - --reserve (default 3 GiB).
# 7. Closed (exit 3, the numbers printed) if budget <= 0, budget < cap or SwapFree < --swap-min (default 4 GiB), or
#    by any check above. --dry: the numbers and the decision only (exit 0 open, 3 closed).
# 8. Open: COMMAND ARGS --mem <budget> (always > 0) under
#      systemd-run --user --scope --unit=s9gate-PID -p MemoryMax=<cap> -p MemorySwapMax=0 choom -n 1000 nice -n 10
#    (the job checks its scope's memory.max / memory.swap.max before it starts), and a watchdog (every second) that
#    kills the scope when MemAvailable < --watch (default 3 GiB; exit 4) or at --until (exit 5). Else the job's
#    exit status. At the end: the scope's memory.peak (its last read) against the cap and the estimate.
# Options: --dry, --until TIME (= --deadline), --reserve G, --server-floor G, --swap-min G, --watch G, --guard MIN, --lock-wait S,
# --health URL (repeatable: replaces the default endpoints), --timer UNIT (repeatable: replaces the default timers),
# --nice N (default 10). Numbers in GiB. Needs systemd (user manager with the memory controller), curl, choom, flock,
# python3 with numpy (.venv/bin/python if present).
#
# The server mode (TODO S9c.5): --server MODEL --server-ctx N --server-kv TYPE --server-kv-mem G [--server-draft mtp|FILE]
# -- COMMAND ARGS... (any command, e.g. tests/server_test.py or scripts/s9c_bench.py, that starts `runq serve MODEL --ctx N
# -k TYPE --admit bound --kv-mem G [--draft mtp|FILE]` with --mem $S9_GATE_MEM; one server at a time). The estimate: the
# target's probe (runq MODEL -c N -k TYPE --mem 0.01) without its one sequence (KV cache and recurrent state: the pool's)
# + G (the pool: the leases' KV and states, the draft's sequences included) + the drafter: the MTP view's block bytes x
# 1.10 + norms + R (its KV is in the pool), or a draft model's probe without its sequence. No checkpoint term (in the
# pool). The command gets no --mem: the gate exports S9_GATE_MEM=<budget> to it instead (the driver and its server run
# in the capped scope). Fixtures for testing the gate itself (not for a 27B job): --meminfo FILE (MemAvailable and
# SwapFree read from FILE, a /proc/meminfo copy, for the decision; the watchdog reads /proc/meminfo), --no-procs (the
# runq serve processes not looked for: only the --health endpoints count).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=$ROOT/.venv/bin/python
[ -x "$PY" ] || PY=python3
DRY=0 UNTIL="" RESERVE=3 FLOOR=1 SWAP_MIN=4 WATCH=3 GUARD=5 LOCK=/tmp/cwayn-bench.lock LOCK_WAIT=0 NICE=10
HEALTH=() TIMERS=()
SRV="" SRV_CTX="" SRV_KV="" SRV_KVMEM="" SRV_DRAFT="" MEMINFO=/proc/meminfo PROCS=1
die() { echo "s9_gate: $*" >&2; exit 2; }
while [ $# -gt 0 ]; do
    case "$1" in
        --dry) DRY=1; shift ;;
        --until|--deadline) UNTIL=$2; shift 2 ;;
        --reserve) RESERVE=$2; shift 2 ;;
        --server-floor) FLOOR=$2; shift 2 ;;
        --swap-min) SWAP_MIN=$2; shift 2 ;;
        --watch) WATCH=$2; shift 2 ;;
        --guard) GUARD=$2; shift 2 ;;
        --lock-wait) LOCK_WAIT=$2; shift 2 ;;
        --health) HEALTH+=("$2"); shift 2 ;;
        --timer) TIMERS+=("$2"); shift 2 ;;
        --nice) NICE=$2; shift 2 ;;
        --server) SRV=$2; shift 2 ;;
        --server-ctx) SRV_CTX=$2; shift 2 ;;
        --server-kv) SRV_KV=$2; shift 2 ;;
        --server-kv-mem) SRV_KVMEM=$2; shift 2 ;;
        --server-draft) SRV_DRAFT=$2; shift 2 ;;
        --meminfo) MEMINFO=$2; shift 2 ;;
        --no-procs) PROCS=0; shift ;;
        --) shift; break ;;
        *) die "unknown option $1 (usage: s9_gate.sh [--dry] [--until TIME] [options] -- COMMAND ARGS...)" ;;
    esac
done
[ -n "$SRV" ] || [ $# -ge 2 ] || die "no command (s9_gate.sh [options] -- runq|spec_sweep MODEL ...)"
CMD=("$@")
for a in "${CMD[@]}"; do [ "$a" = --mem ] && die "the command must not have --mem: the gate computes it"; done
if [ -n "$SRV" ]; then
    [ -f "$SRV" ] || die "--server: no file $SRV"
    [[ "$SRV_CTX" =~ ^[0-9]+$ ]] || die "--server needs --server-ctx N"
    [ -n "$SRV_KV" ] || die "--server needs --server-kv TYPE"
    [[ "$SRV_KVMEM" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "--server needs --server-kv-mem G (the command's --kv-mem)"
    [ -z "$SRV_DRAFT" ] || [ "$SRV_DRAFT" = mtp ] || [ -f "$SRV_DRAFT" ] || die "--server-draft: no file $SRV_DRAFT"
elif [ -n "$SRV_CTX$SRV_KV$SRV_KVMEM$SRV_DRAFT" ]; then
    die "--server-ctx / --server-kv / --server-kv-mem / --server-draft need --server MODEL"
fi
[ -r "$MEMINFO" ] || die "--meminfo: cannot read $MEMINFO"
[ ${#HEALTH[@]} -gt 0 ] || HEALTH=(http://100.67.84.55:8418/health http://100.67.84.55:8417/health)
[ ${#TIMERS[@]} -gt 0 ] || TIMERS=(runq-day.timer lcb-bench.timer)
for v in "$RESERVE" "$FLOOR" "$SWAP_MIN" "$WATCH" "$GUARD" "$LOCK_WAIT"; do
    [[ "$v" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "not a number: $v"
done

CLOSED=()  # the reasons
close() { CLOSED+=("$*"); }
say() { echo "[gate] $*" >&2; }
fge() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a + 0 >= b + 0) }'; }  # a >= b (floats)
calc() { awk "BEGIN { printf \"%.2f\", $* }"; }
NOW=$(date +%s)
UNTIL_S=""
if [ -n "$UNTIL" ]; then
    UNTIL_S=$(date -d "$UNTIL" +%s 2>/dev/null) || die "--until: not a time: $UNTIL"
    [ "$UNTIL_S" -gt $((NOW + 60)) ] || close "--until $(date -u -d @"$UNTIL_S" +%FT%TZ) is less than a minute away"
fi

# 1. the bench lock (fd 9; held until the gate exits)
exec 9>>"$LOCK" || die "cannot open $LOCK"
if flock -w "$LOCK_WAIT" 9; then LOCKED="held by the gate"
else LOCKED="busy"; close "the bench lock $LOCK is held by another job"; fi

# 2. memory
mem_gib() { awk -v k="$1:" '$1 == k { printf "%.2f", $2 / 1048576 }' "$MEMINFO"; }
AVAIL=$(mem_gib MemAvailable)
SWAPFREE=$(mem_gib SwapFree)

# 3. the servers: runq serve processes and the --health endpoints
declare -A URLS=() SWAPS=() PIDS=()
ANSWERED=" "  # (the pids of the runq serve processes whose /health answered with a mem object)
for pid in $([ "$PROCS" = 1 ] && { pgrep -x runq; pgrep -x runq-generic; }); do
    mapfile -d '' -t av < "/proc/$pid/cmdline" 2>/dev/null || continue
    [ "${av[1]:-}" = serve ] || continue
    host=127.0.0.1 port=8080
    for ((i = 2; i + 1 < ${#av[@]}; i++)); do
        [ "${av[i]}" = --host ] && host=${av[i + 1]}
        [ "${av[i]}" = --port ] && port=${av[i + 1]}
    done
    u="http://$host:$port/health"
    URLS[$u]="pid $pid"
    PIDS[$u]=$pid
    SWAPS[$u]=$(awk '$1 == "VmSwap:" { printf "%.2f", $2 / 1048576 }' "/proc/$pid/status" 2>/dev/null)
done
for u in "${HEALTH[@]}"; do [ -n "${URLS[$u]+x}" ] || URLS[$u]=""; done
MARGIN=0 SERVERS=()
for u in "${!URLS[@]}"; do
    body=$(curl -s -m 3 "$u")
    if [ -z "$body" ]; then
        if [ -n "${URLS[$u]}" ]; then close "runq serve (${URLS[$u]}): $u does not answer (still loading?)"; fi
        continue
    fi
    r=$("$PY" -I -c '
import json, sys
try:
    d = json.loads(sys.argv[1])
except ValueError:
    print("bad"); sys.exit()
m = d.get("mem")
if not isinstance(m, dict) or "budget" not in m or "committed" not in m:
    print("nomem", d.get("running", -1)); sys.exit()
print("ok", d.get("running", -1), max(0.0, float(m["budget"]) - float(m["committed"])), m["budget"], m["committed"])' "$body")
    read -r st run free bud com <<< "$r"
    sw=${SWAPS[$u]:-0}
    sw=${sw:-0}
    case "$st" in
        ok)
            m=$(calc "$free + $FLOOR + $sw")
            MARGIN=$(calc "$MARGIN + $m")
            [ -n "${PIDS[$u]:-}" ] && ANSWERED+="${PIDS[$u]} "
            SERVERS+=("$u${URLS[$u]:+ (${URLS[$u]})}: running $run, budget $bud - committed $com = $free + floor $FLOOR + swapped $sw = $m GiB")
            [ "$run" = 0 ] || close "the server at $u is running $run request(s)" ;;
        nomem) close "the server at $u has no mem object in /health (not --admit bound / chunk): its growth is unknown"
               SERVERS+=("$u: running $run, no mem object") ;;
        *) close "the server at $u: /health is not JSON" ;;
    esac
done

# 4. the servers to come: timers that start one
TJSON=$(systemctl --user list-timers --all --output=json 2>/dev/null)
TIMES=()
for t in "${TIMERS[@]}"; do
    systemctl --user cat "$t" >/dev/null 2>&1 || continue  # (not loaded: nothing scheduled)
    svc=$(systemctl --user show "$t" -p Triggers --value 2>/dev/null)
    svc=${svc%% *}
    if [ -n "$svc" ]; then
        # its service running: a server up and counted above (a runq serve of its cgroup answered /health) is fine; else
        # (starting, a oneshot job such as the nightly benchmark: "activating", a server not answering) closed
        sa=$(systemctl --user is-active "$svc" 2>/dev/null)
        case "$sa" in
            active|activating|reloading)
                ok=0
                cg=$(systemctl --user show "$svc" -p ControlGroup --value 2>/dev/null)
                if [ "$sa" = active ] && [ -n "$cg" ]; then
                    for p in $(cat "/sys/fs/cgroup$cg/cgroup.procs" 2>/dev/null); do
                        [[ "$ANSWERED" == *" $p "* ]] && ok=1
                    done
                fi
                if [ $ok = 1 ]; then TIMES+=("$svc (started by $t) is running: its server is counted above")
                else close "$svc (started by $t) is $sa: a server may start at any time"; fi ;;
        esac
    fi
    next=$("$PY" -I -c '
import json, sys
try:
    ts = json.loads(sys.argv[1])
except ValueError:
    ts = []
print(next((int(x["next"] or 0) // 1000000 for x in ts if x.get("unit") == sys.argv[2]), 0))' "$TJSON" "$t")
    [ "${next:-0}" -gt 0 ] || continue
    TIMES+=("$t -> ${svc:-?} at $(date -u -d @"$next" +%FT%TZ)")
    if [ -z "$UNTIL_S" ]; then
        close "$t starts ${svc:-a server} at $(date -u -d @"$next" +%FT%TZ): give --until before $(date -u -d @$((next - GUARD * 60)) +%FT%TZ)"
    elif [ "$UNTIL_S" -gt $((next - GUARD * 60)) ]; then
        close "--until $(date -u -d @"$UNTIL_S" +%FT%TZ) is less than $GUARD min before $t starts ${svc:-a server} at $(date -u -d @"$next" +%FT%TZ)"
    fi
done

# 5. the scope: MemorySwapMax=0 must reach the cgroup
SWAPMAX=$(systemd-run --user --scope -q -p MemoryMax=64M -p MemorySwapMax=0 -- sh -c \
    'cat "/sys/fs/cgroup$(sed -n "s/^0:://p" /proc/self/cgroup)/memory.swap.max"' 2>&1)
[ "$SWAPMAX" = 0 ] || close "systemd-run --user --scope -p MemorySwapMax=0: memory.swap.max reads '$SWAPMAX', not 0"

# 6. the estimate
probe() { # a command that refuses before reading tensor data (--mem 0.01), in a scope it cannot grow out of
    timeout 300 systemd-run --user --scope -q -p MemoryMax=1G -p MemorySwapMax=0 -- "$@" 2>&1
}
parse_needs() { # "needs ~X GiB (weights W, runtime R, a sequence of N positions: KV cache K + recurrent state S MiB" -> X R N S
    sed -n 's/.*needs ~\([0-9.]*\) GiB (weights [0-9.]*, runtime \([0-9.]*\), a sequence of \([0-9]*\) positions: KV cache [0-9.]*\(.*\)/\1 \2 \3 \4/p' |
        head -1 | awk '{ s = 0; if (match($0, /recurrent state [0-9.]+ MiB/)) { split(substr($0, RSTART, RLENGTH), a, " "); s = a[3] }
                         printf "%s %s %s %s\n", $1, $2, $3, s }'
}
seqless() { # a probe's output -> "E R N": its estimate without its one sequence (the KV cache, the recurrent state x (1 +
            # its checkpoints)), its runtime, the positions
    "$PY" -I -c '
import re, sys
m = re.search(r"needs ~([0-9.]+) GiB \(weights [0-9.]+, runtime ([0-9.]+), a sequence of (\d+) positions: KV cache ([0-9.]+)"
              r"(?: \+ recurrent state ([0-9.]+) MiB x \(1 \+ (\d+) checkpoints\))?", sys.stdin.read())
if m:
    st = float(m.group(5) or 0) * (1 + int(m.group(6) or 0)) / 1024
    print("%.2f %s %s" % (float(m.group(1)) - float(m.group(4)) - st, m.group(2), m.group(3)))'
}
mtp_est() { # MODEL NPOS KVB RUNTIME: the MTP view's block bytes in the file x 1.10 (repacked) + its norms + its KV cache of NPOS
            # positions at KVB bytes per value (0: none) + RUNTIME (its context: not more than the model's) in GiB
    "$PY" -I -c '
import sys
sys.path.insert(0, sys.argv[1] + "/tests")
import ggufmini as g
f = g.GgufFile(sys.argv[2])
a = f.kv["general.architecture"]
L = int(f.kv[a + ".block_count"]) - int(f.kv.get(a + ".nextn_predict_layers", 0))
mats = norms = 0
for name, (ne, ty, off) in f.tensors.items():
    if name.startswith("blk.%d." % L):
        rows = 1
        for d in ne[1:]:
            rows *= int(d)
        b = g.row_bytes(ty, int(ne[0])) * rows
        if len(ne) > 1:
            mats += b
        else:
            norms += b
kvh = f.kv[a + ".attention.head_count_kv"]
kvh = int(max(kvh)) if hasattr(kvh, "__len__") else int(kvh)
hd = int(f.kv.get(a + ".attention.key_length", int(f.kv[a + ".embedding_length"]) // int(f.kv[a + ".attention.head_count"])))
kv = 2 * kvh * hd * float(sys.argv[4]) * (int(sys.argv[3]) + 256)
print("%.3f" % ((mats * 1.10 + norms + kv) / 2**30 + float(sys.argv[5])))' "$ROOT" "$@" 2>&1
}
KMAX=0 USE_MTP=0 DRAFT="" KV=f16 T_STATE=0 MTP_EST=0 CK_EST=0 D_EST=0 D_CK=0
if [ -n "$SRV" ]; then
    # the server mode: the target without its sequence + the pool + the drafter (the probe and the header only)
    out=$(probe "$ROOT/runq" "$SRV" -c "$SRV_CTX" -k "$SRV_KV" -m generate -i x --mem 0.01)
    read -r T_EST T_RT NPOS <<< "$(echo "$out" | seqless)"
    if [ -z "${T_EST:-}" ]; then
        close "the model's estimate: no 'needs ~X GiB' in the probe's output"
        echo "$out" | tail -3 | sed 's/^/[gate]   probe: /' >&2
        T_EST=0 T_RT=0 NPOS=0
    fi
    DWHAT="none"
    if [ "$SRV_DRAFT" = mtp ]; then
        D_EST=$(mtp_est "$SRV" 0 0 "$T_RT")
        [[ "$D_EST" =~ ^[0-9.]+$ ]] || { close "the MTP view's estimate: $D_EST"; D_EST=0; }
        DWHAT="the MTP view (block x 1.10 + norms + runtime $T_RT)"
    elif [ -n "$SRV_DRAFT" ]; then
        dout=$(probe "$ROOT/runq" "$SRV_DRAFT" -c "$SRV_CTX" -k "$SRV_KV" -m generate -i x --mem 0.01)
        read -r D_EST _ _ <<< "$(echo "$dout" | seqless)"
        if [ -z "${D_EST:-}" ]; then close "the draft's estimate: no 'needs ~X GiB' from $ROOT/runq $SRV_DRAFT"; D_EST=0; fi
        DWHAT="$(basename "$SRV_DRAFT") without its sequence"
    fi
    EST=$(calc "$T_EST + $SRV_KVMEM + $D_EST")
else
    BIN=$(basename "${CMD[0]}")
    MODEL=${CMD[1]}
    case "$BIN" in
        spec_sweep*)
            plan=$(probe "${CMD[@]}" --list | sed -n 's/^plan //p' | head -1)
            [ -n "$plan" ] || close "spec_sweep --list gave no plan"
            read -r _ CTXA _ KV _ KMAX _ USE_MTP _ _ _ DRAFT _ _ <<< "$plan"
            [ "$DRAFT" = - ] && DRAFT="" ;;
        runq*)
            G=-1 D="" mode=chat lk=0
            for ((i = 2; i + 1 < ${#CMD[@]}; i++)); do
                case "${CMD[i]}" in
                    -k) KV=${CMD[i + 1]} ;; -D) D=${CMD[i + 1]} ;; -G) G=${CMD[i + 1]} ;; -m) mode=${CMD[i + 1]} ;;
                    --lookup) lk=${CMD[i + 1]} ;;
                esac
            done
            if [ "$mode" = generate ] || [ "$mode" = chat ]; then
                [ -n "$D" ] && [ "$G" -lt 0 ] && G=3  # (runq's default -G with -D)
                if [ -n "$D" ] && [ "$G" -gt 0 ]; then
                    KMAX=$G
                    if [ "$D" = mtp ]; then USE_MTP=1; else DRAFT=$D; fi
                fi
                [ "$lk" -gt "$KMAX" ] && KMAX=$lk
            fi ;;
        *) close "unknown command $BIN (runq or spec_sweep)" ;;
    esac
    out=""
    [[ "$BIN" == runq* || "$BIN" == spec_sweep* ]] && out=$(probe "${CMD[@]}" --mem 0.01) # (nothing else is run)
    read -r T_EST T_RT NPOS T_STATE <<< "$(echo "$out" | parse_needs)"
    if [ -z "${T_EST:-}" ]; then
        close "the model's estimate: no 'needs ~X GiB' in the probe's output"
        echo "$out" | tail -3 | sed 's/^/[gate]   probe: /' >&2
        T_EST=0 T_RT=0 NPOS=0 T_STATE=0
    fi
    case "$KV" in f32) KVB=4 ;; bf16*|f16*) KVB=2 ;; q8*) KVB=1.1 ;; q4k4*) KVB=0.55 ;; q4*) KVB=0.85 ;; *) KVB=2 ;; esac
    if [ "$USE_MTP" = 1 ]; then
        MTP_EST=$(mtp_est "$MODEL" "$NPOS" "$KVB" "$T_RT")
        [[ "$MTP_EST" =~ ^[0-9.]+$ ]] || { close "the MTP view's estimate: $MTP_EST"; MTP_EST=0; }
    fi
    [ "$KMAX" -gt 0 ] && CK_EST=$(calc "${T_STATE:-0} * ($KMAX + 2) / 1024")
    if [ -n "$DRAFT" ]; then
        dout=$(probe "$ROOT/runq" "$DRAFT" -c "$NPOS" -k "$KV" -m generate -i x --mem 0.01)
        read -r D_EST _ _ D_STATE <<< "$(echo "$dout" | parse_needs)"
        if [ -z "${D_EST:-}" ]; then close "the draft's estimate: no 'needs ~X GiB' from $ROOT/runq $DRAFT"; D_EST=0 D_STATE=0; fi
        D_CK=$(calc "${D_STATE:-0} * ($KMAX + 2) / 1024")
    fi
    EST=$(calc "$T_EST + $MTP_EST + $CK_EST + $D_EST + $D_CK")
fi
CAP=$(calc "$EST + 1")
CAP_MIB=$(awk -v c="$CAP" 'BEGIN { printf "%d", c * 1024 + 0.999 }')
BUDGET=$(calc "$AVAIL - $MARGIN - $RESERVE")

# 7. the decision
fge 0 "$BUDGET" && close "budget $BUDGET GiB <= 0"
fge "$BUDGET" "$CAP" || close "budget $BUDGET GiB < cap $CAP GiB"
fge "$SWAPFREE" "$SWAP_MIN" || close "SwapFree $SWAPFREE GiB < $SWAP_MIN GiB"
fge 0 "$EST" && close "no estimate"
say "$(date -u +%FT%TZ): $(printf '%q ' "${CMD[@]}")"
say "MemAvailable $AVAIL GiB, SwapFree $SWAPFREE GiB; bench lock $LOCK: $LOCKED"
if [ ${#SERVERS[@]} -eq 0 ]; then say "servers: none up (checked ${!URLS[*]})"; fi
for s in "${SERVERS[@]}"; do say "server $s"; done
for s in "${TIMES[@]}"; do say "scheduled: $s"; done
[ -n "$UNTIL_S" ] && say "--until $(date -u -d @"$UNTIL_S" +%FT%TZ) (the job is killed then)"
say "scope probe: memory.swap.max $SWAPMAX"
if [ -n "$SRV" ]; then
    say "server mode: runq serve $SRV --ctx $SRV_CTX -k $SRV_KV --kv-mem $SRV_KVMEM${SRV_DRAFT:+ --draft $SRV_DRAFT}"
    say "estimate $EST GiB = model $T_EST without its sequence (runtime $T_RT, $NPOS positions, KV $SRV_KV) + the pool" \
        "$SRV_KVMEM + the drafter $D_EST ($DWHAT)"
else
    say "estimate $EST GiB = model $T_EST (runtime $T_RT, $NPOS positions, KV $KV) + MTP view $MTP_EST + checkpoints $CK_EST" \
        "(state $T_STATE MiB x (kmax $KMAX + 2)) + draft $D_EST + its checkpoints $D_CK"
fi
say "cap $CAP GiB (MemoryMax ${CAP_MIB}M); budget $BUDGET GiB = MemAvailable $AVAIL - servers' margin $MARGIN - reserve $RESERVE"
if [ ${#CLOSED[@]} -gt 0 ]; then
    for r in "${CLOSED[@]}"; do say "closed: $r"; done
    say "decision: CLOSED"
    exit 3
fi
MEMHOW="--mem $BUDGET"
[ -n "$SRV" ] && MEMHOW="S9_GATE_MEM=$BUDGET"
say "decision: OPEN ($MEMHOW, MemoryMax ${CAP_MIB}M, MemorySwapMax 0, watchdog MemAvailable < $WATCH GiB${UNTIL_S:+, deadline $(date -u -d @"$UNTIL_S" +%FT%TZ)})"
[ "$DRY" = 1 ] && exit 0

# 8. the run: the scope, the job (its scope's limits checked first), the watchdog
UNIT=s9gate-$$
FLAG=$(mktemp "${TMPDIR:-/var/tmp}/s9gate-$$.XXXXXX")
WRAP='cg=$(sed -n "s/^0:://p" /proc/self/cgroup); d=/sys/fs/cgroup$cg
mx=$(cat "$d/memory.max"); sw=$(cat "$d/memory.swap.max")
echo "[gate] scope ${cg##*/}: memory.max $mx, memory.swap.max $sw, oom_score_adj $(cat /proc/self/oom_score_adj)" >&2
if [ "$sw" != 0 ] || [ "$mx" != "$S9_GATE_MAX" ]; then echo "[gate] the scope is not as set: not run" >&2; exit 3; fi
exec "$@"'
MEMARG=(--mem "$BUDGET")
if [ -n "$SRV" ]; then MEMARG=(); export S9_GATE_MEM=$BUDGET; fi  # (the server mode: the command passes it to runq serve)
S9_GATE_MAX=$((CAP_MIB * 1048576)) systemd-run --user --scope -q --unit="$UNIT" -p MemoryMax="${CAP_MIB}M" -p MemorySwapMax=0 \
    -- choom -n 1000 -- nice -n "$NICE" bash -c "$WRAP" s9gate "${CMD[@]}" "${MEMARG[@]}" &
JOB=$!
kill_job() { systemctl --user kill --signal=KILL "$UNIT.scope" 2>/dev/null || kill -KILL "$JOB" 2>/dev/null; }
trap 'kill_job; exit 130' INT TERM
(
    cg=""
    while kill -0 "$JOB" 2>/dev/null; do
        # (the scope's peak so far, for the end's line: its cgroup is gone once the job ends)
        [ -n "$cg" ] || cg=$(systemctl --user show "$UNIT.scope" -p ControlGroup --value 2>/dev/null)
        [ -n "$cg" ] && cat "/sys/fs/cgroup$cg/memory.peak" > "$FLAG.peak" 2>/dev/null
        a=$(awk '$1 == "MemAvailable:" { printf "%.2f", $2 / 1048576 }' /proc/meminfo)
        if ! fge "$a" "$WATCH"; then echo "memory MemAvailable $a GiB < $WATCH GiB" > "$FLAG"; kill_job; break; fi
        if [ -n "$UNTIL_S" ] && [ "$(date +%s)" -ge "$UNTIL_S" ]; then echo "deadline --until $UNTIL" > "$FLAG"; kill_job; break; fi
        sleep 1
    done
) &
WD=$!
{ wait "$JOB"; } 2>/dev/null # (no shell "Killed" line: the reason follows)
RC=$?
kill "$WD" 2>/dev/null
wait "$WD" 2>/dev/null
WHY=$(cat "$FLAG" 2>/dev/null)
PEAK=$(awk '{ printf "%.2f", $1 / 1073741824 }' "$FLAG.peak" 2>/dev/null)
rm -f "$FLAG" "$FLAG.peak"
[ -n "$PEAK" ] && say "the scope's memory.peak: >= $PEAK GiB (the last read; cap $CAP, estimate $EST)"
if [ -n "$WHY" ]; then
    say "$(date -u +%FT%TZ): killed by the watchdog: $WHY (the job's status $RC)"
    case "$WHY" in memory*) exit 4 ;; *) exit 5 ;; esac
fi
say "$(date -u +%FT%TZ): the job exited with status $RC"
exit "$RC"
