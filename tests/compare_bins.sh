#!/usr/bin/env bash
# Compares two builds of runq on everything that must not change in a refactoring: the outputs of
# the same command lines must be byte-identical.
#   -m logits files (3 models x both builds x KV f16 / bf16 / f32 x -b 1 / 7 / 64 / 512 x a short
#   prompt and one of > 256 tokens, i.e. past one attention block); greedy generate (short / long
#   prompt, -b 1 / 7 / 512, default -c, a prompt longer than -c); seeded sampling (-t 0.8 -s 7,
#   generate and chat); multi-turn chat on stdin (-r 0 with a system prompt, -r 1, context full at
#   -c 160, -b 1 / 7) incl. the RUNQ_CHAT_DUMP files; -m tokenize; -m dequant; error cases (prompt
#   too long, bad flags, missing file): stdout, stderr (numbers of the [timing] / [tokenize] lines
#   masked) and the exit status.
# Usage: tests/compare_bins.sh [-j JOBS] [-t THREADS] [--quick] REF_DIR NEW_DIR
#   REF_DIR / NEW_DIR: directories holding runq and runq-generic (e.g. a copy of the binaries of the
#   commit before a change, and the repo root). --quick: only the AVX-512 (native) builds, -b 1 / 64.
#   Env: CMP_CACHE=DIR keeps the REF results there (keyed by the binaries' and this job list's
#   sha256), so repeated comparisons against the same reference only run the new binaries.
# Exit status 0 if every job gave the same results. Takes ~3-6 min (-j 3, 4 threads per job).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOBS=3 THREADS=4 QUICK=0
while [[ $# -gt 2 ]]; do
    case "$1" in
        -j) JOBS="$2"; shift 2 ;;
        -t) THREADS="$2"; shift 2 ;;
        --quick) QUICK=1; shift ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
[[ $# -eq 2 ]] || { sed -n '2,19p' "$0" >&2; exit 2; }
REF="$(cd "$1" && pwd)" NEW="$(cd "$2" && pwd)"
cd "$ROOT"
BIN=models/Qwen3-0.6B-q8.bin
Q8=models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf
Q4=models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf
for f in "$BIN" "$Q8" "$Q4"; do [[ -f "$f" ]] || { echo "missing $f" >&2; exit 2; }; done
BUILDS="runq runq-generic"
[[ $QUICK = 1 ]] && BUILDS="runq"
for d in "$REF" "$NEW"; do for b in $BUILDS; do [[ -x "$d/$b" ]] || { echo "missing $d/$b" >&2; exit 2; }; done; done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/cmpbins.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

SHORT="The capital of France is"
LONG="The history of the lighthouse begins with fires lit on hilltops to guide ships home at night. \
Over the centuries these fires became towers of stone, and the towers were fitted with lamps that \
burned oil, then gas, and finally electricity. A keeper lived at each station, trimming the wicks, \
polishing the great lenses and writing the weather into a logbook every few hours. The lenses \
themselves were a marvel: rings of glass prisms, designed by Augustin Fresnel in 1822, that bent \
the light of a single flame into a beam visible for twenty miles or more. Each lighthouse had its \
own pattern of flashes, so that a sailor could tell from the rhythm alone which coast lay ahead. \
Today most lights are automatic. Sensors switch them on at dusk, batteries charged by solar panels \
keep them running through storms, and satellite navigation has made them less essential than they \
once were. Yet many coastal towns still treasure their towers, and volunteers keep the old \
keepers' cottages open as small museums, where visitors can climb the spiral stairs, look out over \
the water and imagine the long winter nights when a single person kept the light burning until \
morning. Questions to consider: why did flash patterns matter, what replaced the keepers, and how \
did the arrival of radio beacons in the twentieth century change the way captains planned a voyage \
along a dangerous and rocky shore?"
printf 'My name is Ada and my favourite colour is green.\nWhat is my name and favourite colour?\nWrite a haiku about the sea.\n' > "$WORK/turns3.txt"
printf 'What is 12 + 30?\nAnd times two?\n' > "$WORK/turns2.txt"
printf 'Name a fruit.\nName a colour.\nName an animal.\nName a country.\nName a planet.\nName a metal.\n' > "$WORK/turns6.txt"
: > "$WORK/empty.txt"

# job list, one line per job: name, stdin file, env flags (DUMP=1: RUNQ_CHAT_DUMP), then the build and
# its arguments; fields separated by \x1e, each argument terminated by \x1f (empty ones survive)
JL="$WORK/jobs.txt"
job() { # name stdin env build args...
    local name="$1" in="$2" env="$3"; shift 3
    { printf '%s\x1e%s\x1e%s\x1e' "$name" "$in" "$env"; printf '%s\x1f' "$@"; printf '\n'; } >> "$JL"
}
: > "$JL"
BS="1 7 64 512"
[[ $QUICK = 1 ]] && BS="1 64"
for b in $BUILDS; do
    for m in bin q8 q4; do
        case $m in bin) M=$BIN ;; q8) M=$Q8 ;; q4) M=$Q4 ;; esac
        for k in f16 bf16 f32; do
            for c in $BS; do
                job "$b/$m/logits/k$k/b$c/short" empty.txt "" "$b" "$M" -c 1024 -k $k -m logits -b $c -i "$SHORT" -o OUT
                job "$b/$m/logits/k$k/b$c/long" empty.txt "" "$b" "$M" -c 1024 -k $k -m logits -b $c -i "$LONG" -o OUT
            done
            job "$b/$m/gen/k$k/short" empty.txt "" "$b" "$M" -c 1024 -k $k -t 0 -n 40 -m generate -i "$SHORT"
            job "$b/$m/gen/k$k/long" empty.txt "" "$b" "$M" -c 1024 -k $k -t 0 -n 40 -m generate -i "$LONG"
            job "$b/$m/chat3/k$k" turns3.txt "DUMP=1" "$b" "$M" -c 1024 -k $k -t 0 -n 24 -y "You are a helpful assistant."
        done
        for c in 1 7; do
            job "$b/$m/gen/b$c/long" empty.txt "" "$b" "$M" -c 1024 -b $c -t 0 -n 24 -m generate -i "$LONG"
            job "$b/$m/chat3/b$c" turns3.txt "DUMP=1" "$b" "$M" -c 1024 -b $c -t 0 -n 16 -y "You are a helpful assistant."
        done
        job "$b/$m/gen/defctx" empty.txt "" "$b" "$M" -t 0 -n 16 -m generate -i "$SHORT"
        job "$b/$m/gen/overflow" empty.txt "" "$b" "$M" -c 64 -t 0 -n 16 -m generate -i "$LONG"
        job "$b/$m/gen/overflow_b1" empty.txt "" "$b" "$M" -c 64 -b 1 -t 0 -n 16 -m generate -i "$LONG"
        job "$b/$m/gen/n1" empty.txt "" "$b" "$M" -c 256 -t 0 -n 1 -m generate -i "$SHORT"
        job "$b/$m/seeded/gen" empty.txt "" "$b" "$M" -c 1024 -t 0.8 -s 7 -n 40 -m generate -i "$SHORT"
        job "$b/$m/seeded/gen_p" empty.txt "" "$b" "$M" -c 1024 -t 1.5 -p 0.95 -s 7 -n 32 -m generate -i "$SHORT"
        job "$b/$m/seeded/chat" empty.txt "" "$b" "$M" -c 1024 -t 0.8 -s 7 -n 40 -i "Tell me about cats."
        job "$b/$m/seeded/chat_b1" empty.txt "" "$b" "$M" -c 1024 -b 1 -t 0.8 -s 7 -n 40 -i "Tell me about cats."
        job "$b/$m/seeded/chat3" turns3.txt "" "$b" "$M" -c 1024 -t 0.8 -s 7 -n 24
        job "$b/$m/chat/think" turns2.txt "DUMP=1" "$b" "$M" -c 1024 -t 0 -n 64 -r 1
        job "$b/$m/chat/think_sys" turns2.txt "" "$b" "$M" -c 1024 -t 0 -n 48 -r 1 -y "Answer briefly."
        job "$b/$m/chat/ctxfull" turns6.txt "DUMP=1" "$b" "$M" -c 160 -t 0 -n 30 -y "You are terse."
        job "$b/$m/chat/ctxfull_b7" turns6.txt "" "$b" "$M" -c 160 -b 7 -t 0 -n 30 -r 1 -y "You are terse."
        job "$b/$m/chat/toolong" empty.txt "" "$b" "$M" -c 64 -t 0 -n 8 -i "$LONG"
        job "$b/$m/chat/cut" turns3.txt "" "$b" "$M" -c 48 -t 0 -n 0
        job "$b/$m/chat/i" empty.txt "" "$b" "$M" -c 1024 -t 0 -n 24 -i "What is the capital of Australia?"
        job "$b/$m/tokenize" empty.txt "" "$b" "$M" -m tokenize -i "$LONG"
        job "$b/$m/err/logits_long" empty.txt "" "$b" "$M" -c 64 -m logits -i "$LONG" -o OUT
        job "$b/$m/err/gen_empty" empty.txt "" "$b" "$M" -c 64 -m generate -i ""
        job "$b/$m/err/badk" empty.txt "" "$b" "$M" -k f8 -i hi
        job "$b/$m/err/mode" empty.txt "" "$b" "$M" -c 64 -m nosuch -i hi
    done
    job "$b/q4/dequant" empty.txt "" "$b" "$Q4" -c 64 -m dequant -i blk.3.ffn_down.weight:14:4 -o OUT
    job "$b/q8/dequant" empty.txt "" "$b" "$Q8" -c 64 -m dequant -i token_embd.weight:5:2 -o OUT
    job "$b/bin/err/dequant" empty.txt "" "$b" "$BIN" -c 64 -m dequant -i token_embd.weight -o OUT
    job "$b/err/usage" empty.txt "" "$b"
    job "$b/err/missing" empty.txt "" "$b" models/nonexistent.gguf -i hi
    job "$b/err/odd" empty.txt "" "$b" "$BIN" -t
done

run_one() { # DIR line -> "name <hashes>"
    local dir="$1" line="$2" name in env args
    IFS=$'\x1e' read -r name in env args <<< "$line"
    local t; t="$(mktemp -d "$WORK/run.XXXXXX")"
    local -a a
    readarray -d $'\x1f' -t a < <(printf '%s' "$args")
    local exe="$dir/${a[0]}"
    a=("${a[@]:1}")
    for i in "${!a[@]}"; do [[ "${a[$i]}" == OUT ]] && a[$i]="$t/out"; done
    local -a e=(OMP_NUM_THREADS="$THREADS")
    [[ "$env" == *DUMP=1* ]] && e+=(RUNQ_CHAT_DUMP="$t/dump")
    local rc=0
    env "${e[@]}" "$exe" "${a[@]}" < "$WORK/$in" > "$t/stdout" 2> "$t/stderr" || rc=$?
    sed -E '/^\[(timing|tokenize|profile)\]/ s/[0-9]+(\.[0-9]+)?/N/g' "$t/stderr" > "$t/stderr.n"
    local h="rc=$rc out=$(sha256sum < "$t/stdout" | cut -c1-16) err=$(sha256sum < "$t/stderr.n" | cut -c1-16)"
    for f in out dump.ids dump.logits; do
        [[ -f "$t/$f" ]] && h="$h $f=$(sha256sum < "$t/$f" | cut -c1-16)"
    done
    printf '%s %s\n' "$name" "$h"
    rm -rf "$t"
}
export -f run_one
export WORK THREADS

run_all() { # DIR OUTFILE
    local dir="$1" out="$2"
    tr '\n' '\0' < "$JL" | xargs -0 -n 1 -P "$JOBS" bash -c 'run_one "$0" "$1"' "$dir" | sort > "$out"
}

key() { # cache key of a build dir + job list
    { for b in $BUILDS; do sha256sum < "$1/$b"; done; sha256sum < "$JL"; } | sha256sum | cut -c1-24
}

t0=$(date +%s)
REFOUT="$WORK/ref.txt"
if [[ -n "${CMP_CACHE:-}" ]]; then
    mkdir -p "$CMP_CACHE"
    C="$CMP_CACHE/ref-$(key "$REF").txt"
    if [[ -s "$C" ]]; then cp "$C" "$REFOUT"; echo "reference results from the cache ($C)"
    else run_all "$REF" "$REFOUT"; cp "$REFOUT" "$C"; fi
else
    run_all "$REF" "$REFOUT"
fi
run_all "$NEW" "$WORK/new.txt"
n=$(wc -l < "$JL")
if diff "$REFOUT" "$WORK/new.txt" > "$WORK/diff.txt"; then
    echo "compare_bins: all $n jobs identical ($(( $(date +%s) - t0 )) s)"
    exit 0
fi
echo "compare_bins: DIFFERENCES ($n jobs, $(( $(date +%s) - t0 )) s):"
cat "$WORK/diff.txt"
exit 1
