#!/usr/bin/env python3
"""Verify that a timed-out shell command leaves no surviving children.

This is the mitigation section 7 recommends, so it is worth testing rather than
asserting. Three agent-authored `python3 reproduce_issue.py` processes survived
the original harness for up to five days at 99.9% CPU; the cause was
subprocess.run(timeout=...) killing only the shell, not its process group.

CPU-only, no model, runs in about a minute.

    python3 test_orphan_reaping.py
"""

import os
import subprocess
import sys
import tarfile
import tempfile
import time
import types

sys.modules.setdefault("requests", types.ModuleType("requests"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness_v2 import Workspace, BASH_TIMEOUT_S  # noqa: E402

MARKER = "orphan_reaping_probe"


def _empty_snapshot():
    """Minimal tgz so Workspace has something to extract."""
    d = tempfile.mkdtemp()
    p = os.path.join(d, "empty.tgz")
    inner = os.path.join(d, "placeholder")
    os.makedirs(inner, exist_ok=True)
    with open(os.path.join(inner, "README"), "w") as f:
        f.write("probe\n")
    with tarfile.open(p, "w:gz") as tf:
        tf.add(inner, arcname="placeholder")
    return p


def _survivors():
    out = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True, text=True)
    return [l for l in out.stdout.splitlines()
            if MARKER in l and "ps -eo" not in l and "grep" not in l]


def main():
    ws = Workspace(_empty_snapshot())
    before = _survivors()
    print(f"probe processes before: {len(before)}")

    # A shell that backgrounds a spinning grandchild and then itself blocks.
    # Under the old implementation the grandchild survived the timeout.
    cmd = (f"nohup python3 -c \"import time\n"
           f"# {MARKER}\n"
           f"while True: time.sleep(0.01)\" >/dev/null 2>&1 &\n"
           f"sleep {BASH_TIMEOUT_S + 20}")

    print(f"running a command that spawns a child and blocks past the "
          f"{BASH_TIMEOUT_S}s timeout ...")
    t0 = time.time()
    result, _ = ws.run_bash_command(cmd)
    print(f"returned after {time.time() - t0:.1f}s: {result.splitlines()[0][:70]}")

    time.sleep(2)
    after_timeout = _survivors()
    print(f"probe processes after timeout: {len(after_timeout)}")

    ws.close()
    time.sleep(2)
    after_close = _survivors()
    print(f"probe processes after workspace close: {len(after_close)}")

    ok = len(after_close) <= len(before)
    print()
    if ok:
        print("PASS: no probe process outlived the workspace.")
    else:
        print("FAIL: survivors remain ->")
        for l in after_close:
            print("   ", l.strip())
        print("Kill them with: pkill -f", MARKER)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
