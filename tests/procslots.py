"""A cap on the test suite's processes running at once (tests/check.py and its groups' run helpers).

GWENEDEG_TEST_PROCS=N: at most N of the suite's subprocess.run calls at a time (each a runq / api_test / server_test.py
process, most loading a model: ~0.3-0.8 GB each), across all of check.py's thread pools; unset or 0: no cap (the
default: ~25 at once at the start, ~12-16 GB, which a 10-12 GB memory cap OOM-kills within seconds). The long-lived
`runq serve` processes of check_models.py (Popen) are not counted. The same checks either way; only the wall time grows.
"""
import os
import subprocess
import threading

N = int(os.environ.get("GWENEDEG_TEST_PROCS", "0") or 0)
_SLOTS = threading.BoundedSemaphore(N) if N > 0 else None


def run(*args, **kw):
    """subprocess.run, in one of the N slots"""
    if _SLOTS is None:
        return subprocess.run(*args, **kw)
    with _SLOTS:
        return subprocess.run(*args, **kw)
