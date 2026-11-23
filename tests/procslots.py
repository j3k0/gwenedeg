"""The test suite's processes (tests/check.py and its groups' run helpers): how many run at once, and their waits for free
memory.

GWENEDEG_TEST_PROCS=N: at most N of the suite's subprocess.run calls at a time (each a gwenedeg / api_test / server_test.py
process, most loading a model: ~0.3-0.9 GB each), across all of check.py's thread pools; 0: no cap. Unset: as many as
MemAvailable holds at the start, PROC_GIB each beyond RESERVE_GIB, and no cap from ~19 GiB on (the suite starts ~25 at
once, ~12-16 GB; a 10-12 GB memory cap OOM-kills that within seconds). The long-lived `gwenedeg serve` processes of
check_models.py (Popen) are not counted. The same checks either way; only the wall time grows.

A failed run whose output has the memory check's refusal for lack of free memory (REFUSED: "... more than MemAvailable
(X GiB): not loaded", src/model.c, not a --mem budget; in gwenedeg's stderr, api_test's or a test driver's stdout)
runs again every few seconds, out of its slot, for up to GWENEDEG_TEST_MEM_WAIT seconds (default 120; 10 once a run has
waited that long in vain): free memory dips while other processes run. Then run() raises RuntimeError with the refusal,
and check.py reports the check, or the rest of its group, as SKIP (not enough free memory), not as a failure.
"""
import os
import random
import re
import subprocess
import sys
import threading
import time

REFUSED = re.compile(r"more than MemAvailable \([0-9.]+ GiB\): not loaded")  # (the loader's and the MTP view's refusal)
PROC_GIB, RESERVE_GIB = 0.7, 2.0
WAIT = float(os.environ.get("GWENEDEG_TEST_MEM_WAIT", "") or 120)


def mem_available():
    """MemAvailable in GiB (Linux), None where unknown"""
    try:
        with open("/proc/meminfo") as f:
            return next((int(l.split()[1]) / 1048576 for l in f if l.startswith("MemAvailable:")), None)
    except OSError:
        return None


def _cap():
    v = os.environ.get("GWENEDEG_TEST_PROCS", "")
    if v:
        return int(v), "GWENEDEG_TEST_PROCS"
    m = mem_available()
    if m is None:
        return 0, "MemAvailable unknown"
    n = int((m - RESERVE_GIB) / PROC_GIB)
    return (0 if n >= 25 else max(n, 1)), f"MemAvailable {m:.1f} GiB"


N, WHY = _cap()
_SLOTS = threading.BoundedSemaphore(N) if N > 0 else None
_LOCK = threading.Lock()
_STATE = {"noted": False, "starved": False}


def describe():
    """one line for the suite's start: the cap and why"""
    how = WHY if WHY == "GWENEDEG_TEST_PROCS" else f"{WHY}; GWENEDEG_TEST_PROCS=N sets it"
    return f"test processes: {f'at most {N} at a time' if N > 0 else 'no cap'} ({how})"


def refusal(p):
    """the memory check's refusal for lack of free memory in a failed run's output, or None"""
    if p.returncode == 0:
        return None
    for out in (p.stderr, p.stdout):
        s = out.decode(errors="replace") if isinstance(out, bytes) else out if isinstance(out, str) else ""
        m = REFUSED.search(s)
        if m:
            return s[s.rfind("\n", 0, m.start()) + 1:m.end()]
    return None


def run(*args, **kw):
    """subprocess.run, in one of the N slots; again while the memory check refuses a load for lack of free memory"""
    t0 = time.time()
    while True:
        if _SLOTS is None:
            p = subprocess.run(*args, **kw)
        else:
            with _SLOTS:
                p = subprocess.run(*args, **kw)
        why = refusal(p)
        if why is None:
            return p
        with _LOCK:
            wait = min(10, WAIT) if _STATE["starved"] else WAIT
            if not _STATE["noted"]:
                _STATE["noted"] = True
                print(f"note: a load was refused for lack of free memory; such runs wait for it, up to {wait:.0f} s each "
                      f"(GWENEDEG_TEST_MEM_WAIT): {why[-160:]}", file=sys.stderr, flush=True)
        if time.time() - t0 >= wait:
            with _LOCK:
                _STATE["starved"] = True
            raise RuntimeError(f"not enough free memory after {time.time() - t0:.0f} s: {why[-300:]}")
        time.sleep(random.uniform(1.0, 3.0))
