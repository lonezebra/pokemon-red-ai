"""
Check the yield-the-top-tier policy: when workers should defer the best
cores to the user, and that the decision actually reaches spawned workers.

    .venv/bin/python3 tools/test_yield_super.py

What is testable here is the policy and the plumbing; the scheduling
effect itself is macOS's, exercised only on a real Mac. Split accordingly:

  - decide_yield is pure logic: automatic yes below the core count,
    automatic no at or above it, and the env override wins in both
    directions. All assertable anywhere.
  - The decision must actually reach workers. It travels via the
    environment because spawn re-imports from scratch; a worker spawned
    after mark_decision_for_workers must see the flag. Asserted with a
    real spawn-context process.
  - apply_worker_qos on Linux must be a clean no-op returning False --
    never an exception -- because the same code runs in the development
    container. The darwin branch can only be verified on the Mac, and the
    manual verification is documented in the module: Activity Monitor's
    per-core view during a partial-worker run, supers mostly idle.
"""

import multiprocessing as mp
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from core.scheduling import (
    ENV_FLAG,
    apply_worker_qos,
    decide_yield,
    mark_decision_for_workers,
)

failures = []


def check(label, ok, fail_detail="", ok_detail=""):
    detail = ok_detail if ok else fail_detail
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def _spawned_child(path):
    # Runs in a fresh interpreter under spawn: what does it see?
    pathlib.Path(path).write_text(os.environ.get(ENV_FLAG, "MISSING"))


def main():
    print("\nPolicy (pure logic)\n")
    check("13 of 18 cores yields", decide_yield(13, total_cores=18, env={}),
          fail_detail="partial run should leave the top tier to the user")
    check("18 of 18 does not", not decide_yield(18, total_cores=18, env={}),
          fail_detail="full run has no cores to spare, nothing to yield")
    check("20 of 18 (oversubscribed) does not",
          not decide_yield(20, total_cores=18, env={}))
    check("override 0 beats automatic yes",
          not decide_yield(13, total_cores=18, env={ENV_FLAG: "0"}))
    check("override 1 beats automatic no",
          decide_yield(18, total_cores=18, env={ENV_FLAG: "1"}))
    check("zero workers never yields", not decide_yield(0, total_cores=18, env={}))

    print("\nPlumbing (decision reaches a spawned worker)\n")
    mark_decision_for_workers(True)
    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d) / "seen"
        ctx = mp.get_context("spawn")
        proc = ctx.Process(target=_spawned_child, args=(str(out),))
        proc.start()
        proc.join()
        seen = out.read_text() if out.exists() else "NO OUTPUT"
    check("spawned worker sees the yield flag", seen == "1",
          fail_detail=f"child saw {seen!r}",
          ok_detail="flag crossed the spawn boundary via the environment")

    print("\nSyscall guard (this platform)\n")
    mark_decision_for_workers(True)
    try:
        result = apply_worker_qos()
        threw = False
    except Exception as exc:
        result, threw = None, True
    if sys.platform == "darwin":
        check("QoS call succeeds on macOS", result is True and not threw,
              fail_detail=f"returned {result}, threw={threw}")
    else:
        check("clean no-op off macOS", result is False and not threw,
              fail_detail=f"returned {result}, threw={threw}",
              ok_detail="returned False without touching anything")

    mark_decision_for_workers(False)
    check("disabled flag means no attempt anywhere", apply_worker_qos() is False)

    print()
    if failures:
        print(f"{len(failures)} failure(s): {failures}")
        return 1
    print("Partial worker counts now defer the best cores to the user; the")
    print("scheduler-level effect verifies on the Mac via Activity Monitor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
