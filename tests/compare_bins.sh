#!/usr/bin/env bash
# Compares two builds of runq on everything that must not change in a refactoring: the outputs of
# the same command lines must be byte-identical, on every model family.
# Qwen3 (families bin / q8 / q4: Qwen3-0.6B .bin, Q8_0, Q4_K_M):
#   -m logits files (both builds x KV f16 / bf16 / f32 x -b 1 / 7 / 64 / 512 x a short prompt and one of
#   > 256 tokens, i.e. past one attention block); greedy generate (short / long prompt, -b 1 / 7 / 512,
#   default -c, a prompt longer than -c); seeded sampling (-t 0.8 -s 7, generate and chat); multi-turn
#   chat on stdin (-r 0 with a system prompt, -r 1, context full at -c 160, -b 1 / 7) incl. the
#   RUNQ_CHAT_DUMP files; -m tokenize; -m dequant; error cases (prompt too long, bad flags, missing file).
# Qwen3.5 (families q35-q8 / q35-q4: Qwen3.5-0.8B Q8_0, Q4_K_M; synth-r2 / synth-r3: the synthetic models
#   with 2 / 3 value heads per key head of tests/make_synth_q35.py, built from ROOT's script into
#   ROOT/build/synth-q35: data, the same files for both builds; q35-4b: Qwen3.5-4B Q8_0, with --q35-4b):
#   the -m logits grid (both builds x KV f16 / bf16 x 3 / 8 threads x -b 1 / 7 / 64 / 512 x short / long
#   prompt); greedy and seeded generate; -r 1 chats whose second turn rolls the recurrent state back to a
#   checkpoint (-b 7, bf16 KV, seeded, context full), with the RUNQ_CHAT_DUMP files; -m tokenize.
# api_test --digest (the batched passes' logits hashes, tests/api_test.c) on the Qwen3.5 families, and
#   --digest-small with api_test-generic, if both REF_DIR and NEW_DIR have them (DIR/api_test or
#   DIR/build/api_test, and -generic; `make api-test` builds them).
# Compared: stdout, stderr (numbers of the [timing] / [tokenize] lines masked), the exit status and the
# output / dump files. A family whose model file is missing is skipped (noted in the summary).
# Usage: tests/compare_bins.sh [-j JOBS] [-t THREADS] [--quick] [--q35-4b] [--no-api] REF_DIR NEW_DIR
#   REF_DIR / NEW_DIR: directories holding runq and runq-generic (e.g. a copy of the binaries of the
#   commit before a change, and the repo root). --quick: only the AVX-512 (native) builds, -b 1 / 64,
#   a few jobs per Qwen3.5 family. --q35-4b: also a few jobs on Qwen3.5-4B Q8_0 (4.5 GB: run one at a
#   time, 8 threads). --no-api: no api_test jobs. Env: CMP_CACHE=DIR keeps the REF results there (keyed
#   by the binaries', the synthetic models' and this job list's sha256), so repeated comparisons against
#   the same reference only run the new binaries. PYTHON: the interpreter for make_synth_q35.py (numpy).
# Exit status 0 if every job gave the same results. 792 jobs, ~25 min (-j 3, 4 threads per job; ~7 with the
# reference from CMP_CACHE); --quick: 172 jobs, ~1.5 min.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOBS=3 THREADS=4 QUICK=0 Q4B=0 API=1
while [[ $# -gt 2 ]]; do
    case "$1" in
        -j) JOBS="$2"; shift 2 ;;
        -t) THREADS="$2"; shift 2 ;;
        --quick) QUICK=1; shift ;;
        --q35-4b) Q4B=1; shift ;;
        --no-api) API=0; shift ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
[[ $# -eq 2 ]] || { sed -n '2,29p' "$0" >&2; exit 2; }
REF="$(cd "$1" && pwd)" NEW="$(cd "$2" && pwd)"
cd "$ROOT"
BIN=models/Qwen3-0.6B-q8.bin
Q8=models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf
Q4=models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf
Q35Q8=models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf
Q35Q4=models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf
Q35B4=models/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q8_0.gguf
SYN=build/synth-q35
BUILDS="runq runq-generic"
[[ $QUICK = 1 ]] && BUILDS="runq"
for d in "$REF" "$NEW"; do for b in $BUILDS; do [[ -x "$d/$b" ]] || { echo "missing $d/$b" >&2; exit 2; }; done; done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/cmpbins.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# the executables of each side: $WORK/ref, $WORK/new (symlinks: runq, runq-generic, api_test[-generic])
NOTES=()
api_find() { # DIR NAME -> path or nothing
    local p; for p in "$1/$2" "$1/build/$2"; do [[ -x "$p" ]] && { echo "$p"; return; }; done; true
}
for side in ref new; do
    mkdir -p "$WORK/$side"
    d="$REF"; [[ $side = new ]] && d="$NEW"
    for b in $BUILDS; do ln -s "$d/$b" "$WORK/$side/$b"; done
done
APIS=""
if [[ $API = 1 ]]; then
    for a in api_test api_test-generic; do
        [[ $a = api_test-generic && $QUICK = 1 ]] && continue
        r="$(api_find "$REF" $a)" n="$(api_find "$NEW" $a)"
        if [[ -n "$r" && -n "$n" ]]; then
            ln -s "$r" "$WORK/ref/$a"; ln -s "$n" "$WORK/new/$a"; APIS="$APIS $a"
        else
            NOTES+=("no $a jobs: not found in $( [[ -z "$r" ]] && echo "$REF" || echo "$NEW" ) (DIR/$a or DIR/build/$a; make api-test)")
        fi
    done
fi

# the families: name -> model file ("" and a note if missing)
declare -A FM=([bin]=$BIN [q8]=$Q8 [q4]=$Q4 [q35-q8]=$Q35Q8 [q35-q4]=$Q35Q4)
FAMS="bin q8 q4 q35-q8 q35-q4 synth-r2 synth-r3"
if [[ -f "$Q35Q8" ]]; then
    if "${PYTHON:-python3}" -I tests/make_synth_q35.py --src "$Q35Q8" --out "$SYN" > "$WORK/synth.log" 2>&1; then
        FM[synth-r2]=$SYN/synth-q35-r2.gguf FM[synth-r3]=$SYN/synth-q35-r3.gguf
    else
        NOTES+=("make_synth_q35.py failed: $(tail -c 300 "$WORK/synth.log")"); FM[synth-r2]="" FM[synth-r3]=""
    fi
else
    FM[synth-r2]=$Q35Q8 FM[synth-r3]=$Q35Q8 # (missing: skipped below)
fi
if [[ $Q4B = 1 ]]; then FAMS="$FAMS q35-4b"; FM[q35-4b]=$Q35B4; fi
declare -A SKIP=()
for f in $FAMS; do
    [[ -n "${FM[$f]}" && -f "${FM[$f]}" ]] || { SKIP[$f]="skipped: missing ${FM[$f]:-the synthetic model}"; }
done
has() { [[ -z "${SKIP[$1]:-}" ]]; }

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

# job list, one line per job: name (build/family/...), stdin file, env flags (DUMP=1: RUNQ_CHAT_DUMP;
# T=n: n threads instead of -t; HEAVY=1: run one at a time), then the executable and its arguments;
# fields separated by \x1e, each argument terminated by \x1f (empty ones survive)
JL="$WORK/jobs.txt"
job() { # name stdin env exe args...
    local name="$1" in="$2" env="$3"; shift 3
    { printf '%s\x1e%s\x1e%s\x1e' "$name" "$in" "$env"; printf '%s\x1f' "$@"; printf '\n'; } >> "$JL"
}
: > "$JL"
BS="1 7 64 512"
[[ $QUICK = 1 ]] && BS="1 64"
for b in $BUILDS; do
    for m in bin q8 q4; do
        has $m || continue
        M=${FM[$m]}
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
    if has q4; then job "$b/q4/dequant" empty.txt "" "$b" "$Q4" -c 64 -m dequant -i blk.3.ffn_down.weight:14:4 -o OUT; fi
    if has q8; then job "$b/q8/dequant" empty.txt "" "$b" "$Q8" -c 64 -m dequant -i token_embd.weight:5:2 -o OUT; fi
    if has bin; then job "$b/bin/err/dequant" empty.txt "" "$b" "$BIN" -c 64 -m dequant -i token_embd.weight -o OUT; fi
    job "$b/misc/err/usage" empty.txt "" "$b"
    job "$b/misc/err/missing" empty.txt "" "$b" models/nonexistent.gguf -i hi
    job "$b/misc/err/odd" empty.txt "" "$b" "$BIN" -t
done

# Qwen3.5: the recurrent state (DeltaNet) of the linear-attention layers, its checkpoints and rollback.
# A -r 1 chat's second turn diverges right after turn 1's assistant header (the history drops the reasoning
# block), so the cache goes back to the checkpoint taken there and the rest is prefilled again; the
# RUNQ_CHAT_DUMP files hold every sampled logits vector and the final cache's ids
q35_jobs() { # family model
    local f="$1" M="$2" b k t c
    for b in $BUILDS; do
        if [[ $QUICK = 1 ]]; then
            job "$b/$f/logits/kf16/b1/long" empty.txt "" "$b" "$M" -c 1024 -m logits -b 1 -i "$LONG" -o OUT
            job "$b/$f/logits/kf16/b64/long" empty.txt "" "$b" "$M" -c 1024 -m logits -b 64 -i "$LONG" -o OUT
            job "$b/$f/logits/kbf16/b64/short" empty.txt "" "$b" "$M" -c 1024 -k bf16 -m logits -b 64 -i "$SHORT" -o OUT
            job "$b/$f/gen/long" empty.txt "" "$b" "$M" -c 1024 -t 0 -n 24 -m generate -i "$LONG"
            job "$b/$f/seeded/gen" empty.txt "" "$b" "$M" -c 1024 -t 0.8 -s 7 -n 32 -m generate -i "$SHORT"
            job "$b/$f/chat/think" turns2.txt "DUMP=1" "$b" "$M" -c 1024 -t 0 -n 48 -r 1
            continue
        fi
        for k in f16 bf16; do
            for t in 3 8; do
                for c in $BS; do
                    job "$b/$f/logits/k$k/t$t/b$c/short" empty.txt "T=$t" "$b" "$M" -c 1024 -k $k -m logits -b $c -i "$SHORT" -o OUT
                    job "$b/$f/logits/k$k/t$t/b$c/long" empty.txt "T=$t" "$b" "$M" -c 1024 -k $k -m logits -b $c -i "$LONG" -o OUT
                done
            done
        done
        job "$b/$f/gen/short" empty.txt "" "$b" "$M" -c 1024 -t 0 -n 40 -m generate -i "$SHORT"
        job "$b/$f/gen/long" empty.txt "" "$b" "$M" -c 1024 -t 0 -n 40 -m generate -i "$LONG"
        job "$b/$f/gen/kbf16/long" empty.txt "" "$b" "$M" -c 1024 -k bf16 -t 0 -n 24 -m generate -i "$LONG"
        for c in 1 7; do
            job "$b/$f/gen/b$c/long" empty.txt "" "$b" "$M" -c 1024 -b $c -t 0 -n 24 -m generate -i "$LONG"
        done
        job "$b/$f/gen/overflow" empty.txt "" "$b" "$M" -c 64 -t 0 -n 16 -m generate -i "$LONG"
        job "$b/$f/gen/n1" empty.txt "" "$b" "$M" -c 256 -t 0 -n 1 -m generate -i "$SHORT"
        job "$b/$f/seeded/gen" empty.txt "" "$b" "$M" -c 1024 -t 0.8 -s 7 -n 40 -m generate -i "$SHORT"
        job "$b/$f/seeded/gen_p" empty.txt "" "$b" "$M" -c 1024 -t 1.5 -p 0.95 -s 7 -n 32 -m generate -i "$SHORT"
        job "$b/$f/seeded/chat" empty.txt "" "$b" "$M" -c 1024 -t 0.8 -s 7 -n 40 -i "Tell me about cats."
        job "$b/$f/seeded/chat_b1" empty.txt "" "$b" "$M" -c 1024 -b 1 -t 0.8 -s 7 -n 40 -i "Tell me about cats."
        job "$b/$f/chat/think" turns2.txt "DUMP=1" "$b" "$M" -c 1024 -t 0 -n 64 -r 1
        job "$b/$f/chat/think_b7" turns2.txt "DUMP=1" "$b" "$M" -c 1024 -b 7 -t 0 -n 48 -r 1
        job "$b/$f/chat/think_kbf16" turns2.txt "DUMP=1" "$b" "$M" -c 1024 -k bf16 -t 0 -n 48 -r 1
        job "$b/$f/chat/think_seeded" turns3.txt "DUMP=1" "$b" "$M" -c 1024 -t 0.8 -s 7 -n 40 -r 1 -y "Answer briefly."
        job "$b/$f/chat/sys3" turns3.txt "DUMP=1" "$b" "$M" -c 1024 -t 0 -n 24 -y "You are a helpful assistant."
        job "$b/$f/chat/ctxfull" turns6.txt "DUMP=1" "$b" "$M" -c 160 -t 0 -n 30 -y "You are terse."
        job "$b/$f/chat/ctxfull_r1" turns6.txt "DUMP=1" "$b" "$M" -c 160 -b 7 -t 0 -n 30 -r 1 -y "You are terse."
        job "$b/$f/chat/cut" turns3.txt "" "$b" "$M" -c 48 -t 0 -n 0
        job "$b/$f/tokenize" empty.txt "" "$b" "$M" -m tokenize -i "$LONG"
        job "$b/$f/err/logits_long" empty.txt "" "$b" "$M" -c 64 -m logits -i "$LONG" -o OUT
    done
    for a in $APIS; do # the batched passes of api_test (section 9) as hashes; the generic build: the short ones
        if [[ $a = api_test && $QUICK = 0 ]]; then job "$a/$f/digest" empty.txt "" $a "$M" --digest
        else job "$a/$f/digest_small" empty.txt "" $a "$M" --digest-small; fi
    done
}
for f in q35-q8 q35-q4 synth-r2 synth-r3; do if has $f; then q35_jobs $f "${FM[$f]}"; fi; done
if [[ $Q4B = 1 ]] && has q35-4b; then # one at a time (4.5 GB each), 8 threads
    M=$Q35B4 E="HEAVY=1 T=8"
    job "runq/q35-4b/logits/kf16/b64/short" empty.txt "$E" runq "$M" -c 1024 -m logits -b 64 -i "$SHORT" -o OUT
    job "runq/q35-4b/logits/kf16/b64/long" empty.txt "$E" runq "$M" -c 1024 -m logits -b 64 -i "$LONG" -o OUT
    if [[ $QUICK = 0 ]]; then job "runq/q35-4b/logits/kbf16/b1/long" empty.txt "$E" runq "$M" -c 1024 -k bf16 -m logits -b 1 -i "$LONG" -o OUT; fi
    job "runq/q35-4b/gen/long" empty.txt "$E" runq "$M" -c 1024 -t 0 -n 24 -m generate -i "$LONG"
    if [[ $QUICK = 0 ]]; then job "runq/q35-4b/seeded/gen" empty.txt "$E" runq "$M" -c 1024 -t 0.8 -s 7 -n 32 -m generate -i "$SHORT"; fi
    job "runq/q35-4b/chat/think" turns2.txt "DUMP=1 $E" runq "$M" -c 1024 -t 0 -n 32 -r 1
    if [[ " $APIS " = *" api_test "* && $QUICK = 0 ]]; then
        job "api_test/q35-4b/digest_small" empty.txt "$E" api_test "$M" --digest-small
    fi
fi

run_one() { # DIR line -> "name <hashes>"
    local dir="$1" line="$2" name in env args
    IFS=$'\x1e' read -r name in env args <<< "$line"
    local t; t="$(mktemp -d "$WORK/run.XXXXXX")"
    local -a a
    readarray -d $'\x1f' -t a < <(printf '%s' "$args")
    local exe="$dir/${a[0]}"
    a=("${a[@]:1}")
    for i in "${!a[@]}"; do [[ "${a[$i]}" == OUT ]] && a[$i]="$t/out"; done
    local th="$THREADS"
    [[ "$env" =~ (^|\ )T=([0-9]+) ]] && th="${BASH_REMATCH[2]}"
    local -a e=(OMP_NUM_THREADS="$th")
    [[ "$env" == *DUMP=1* ]] && e+=(RUNQ_CHAT_DUMP="$t/dump")
    local rc=0
    env "${e[@]}" "$exe" "${a[@]}" < "$WORK/$in" > "$t/stdout" 2> "$t/stderr" || rc=$?
    sed -E '/^\[(timing|tokenize|profile)\]/ s/[0-9]+(\.[0-9]+)?/N/g' "$t/stderr" > "$t/stderr.n"
    local h="rc=$rc out=$(sha256sum < "$t/stdout" | cut -c1-16) err=$(sha256sum < "$t/stderr.n" | cut -c1-16)"
    for f in out dump.ids dump.logits; do # (out: the -o file)
        [[ -f "$t/$f" ]] && h="$h ${f/#out/file}=$(sha256sum < "$t/$f" | cut -c1-16)"
    done
    printf '%s %s\n' "$name" "$h"
    rm -rf "$t"
}
export -f run_one
export WORK THREADS

run_all() { # DIR OUTFILE: the jobs (the HEAVY ones one at a time, after the others)
    local dir="$1" out="$2"
    {
        awk -F'\x1e' '$3 !~ /HEAVY=1/' "$JL" | tr '\n' '\0' | xargs -0 -r -n 1 -P "$JOBS" bash -c 'run_one "$0" "$1"' "$dir"
        awk -F'\x1e' '$3 ~ /HEAVY=1/' "$JL" | tr '\n' '\0' | xargs -0 -r -n 1 -P 1 bash -c 'run_one "$0" "$1"' "$dir"
    } | sort > "$out"
}

key() { # cache key of a side's executables + the synthetic models + the job list
    {
        for b in $BUILDS $APIS; do sha256sum < "$1/$b"; done
        for f in synth-r2 synth-r3; do if has $f; then sha256sum < "${FM[$f]}"; fi; done
        sha256sum < "$JL"
    } | sha256sum | cut -c1-24
}

t0=$(date +%s)
REFOUT="$WORK/ref.txt"
if [[ -n "${CMP_CACHE:-}" ]]; then
    mkdir -p "$CMP_CACHE"
    C="$CMP_CACHE/ref-$(key "$WORK/ref").txt"
    if [[ -s "$C" ]]; then cp "$C" "$REFOUT"; echo "reference results from the cache ($C)"
    else run_all "$WORK/ref" "$REFOUT"; cp "$REFOUT" "$C"; fi
else
    run_all "$WORK/ref" "$REFOUT"
fi
run_all "$WORK/new" "$WORK/new.txt"
n=$(wc -l < "$JL")

# per-family summary (family: the second part of a job name)
echo "compare_bins: $REF vs $NEW ($BUILDS${APIS:+,$APIS}$( [[ $QUICK = 1 ]] && echo ", --quick"))"
awk -v fams="$FAMS misc" 'NR == FNR { r[$1] = $0; next }
    { split($1, p, "/"); f = p[2]; n[f]++; if (r[$1] != $0) d[f]++; delete r[$1] }
    END { for (k in r) { split(k, p, "/"); n[p[2]]++; d[p[2]]++ }
          m = split(fams, F, " ")
          for (i = 1; i <= m; i++) if (n[F[i]]) printf "  %-9s %4d jobs  %s\n", F[i], n[F[i]], d[F[i]] ? d[F[i]] " DIFFER" : "identical" }' \
    "$REFOUT" "$WORK/new.txt"
for f in $FAMS; do if [[ -n "${SKIP[$f]:-}" ]]; then printf '  %-9s %s\n' "$f" "${SKIP[$f]}"; fi; done
for x in "${NOTES[@]}"; do echo "  note: $x"; done
if diff "$REFOUT" "$WORK/new.txt" > "$WORK/diff.txt"; then
    echo "compare_bins: all $n jobs identical ($(( $(date +%s) - t0 )) s)"
    exit 0
fi
echo "compare_bins: DIFFERENCES ($n jobs, $(( $(date +%s) - t0 )) s):"
cat "$WORK/diff.txt"
exit 1
