"""
Check that Ctrl-C during a training round still saves that round, and that
relaunching resumes from it with a different worker count.

    .venv/bin/python3 tools/test_graceful_stop.py

This is the workflow of sharing a machine with something else: train with
a few workers, interrupt when the cores are needed, pick up later with
more. Interrupting used to discard the whole in-flight round, because the
merge and save_progress only run after every worker finishes -- and rounds
here run from forty minutes to a few hours, so that was a real cost rather
than a rounding error.

Four things have to hold:

  - SIGINT during the worker phase still produces a saved round. If the
    state file is missing or still on the previous round, the interrupt
    threw the work away.
  - The saved round records fewer episodes than the full budget, proving
    the stop actually cut the round short rather than quietly running it
    to completion.
  - Training exits rather than starting another round.
  - Relaunching resumes from the round after the saved one, with a
    different worker count, and keeps the accumulated totals.

Runs training as a real subprocess and sends a real signal, because the
thing under test *is* signal delivery across a process group: workers
inherit the terminal's SIGINT, so they have to ignore it and let the
driver coordinate the drain. Testing that in-process would not exercise
it at all.
"""

import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import textwrap
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
PYTHON = REPO / ".venv" / "bin" / "python3"

# A stub env slow enough that the round is guaranteed still running when
# the signal arrives, but quick enough per episode that draining takes a
# moment rather than a minute.
DRIVER = textwrap.dedent(
    """
    import pathlib, sys, time
    sys.path.insert(0, {src!r})
    import train_navigation_parallel as t
    t.save_gif = lambda *a, **k: None

    class Img:
        def copy(self): return self
    class Screen:
        image = Img()
    class PyBoy:
        screen = Screen()

    class SlowEnv:
        def __init__(self, max_steps=5):
            self.max_steps = max_steps
            self.visited_positions = set()
            self.pyboy = PyBoy()
        def reset(self):
            self.visited_positions = {{(0, 0)}}
            return {{"map_id": 1, "x": 0, "y": 0}}
        def step(self, action):
            time.sleep(0.05)
            return ({{"map_id":1,"x":1,"y":1}}, 1.0, True,
                    {{"reached_goal": True, "step_count": 1}})
        def close(self):
            pass

    def main():
        d = pathlib.Path({workdir!r})
        t.WORKER_DIR = d / "workers"
        t.train(
            env_class=SlowEnv,
            model_path=d / "q.json",
            state_path=d / "state.json",
            gif_prefix="t",
            max_steps=2,
            num_workers={workers},
            episodes_per_round={budget},
            max_rounds=50,
        )

    if __name__ == "__main__":
        main()
    """
)

BUDGET = 600
failures = []


def run(workdir, workers, script_path, interrupt_after=None):
    script_path.write_text(
        DRIVER.format(
            src=str(REPO / "src"), workdir=str(workdir), workers=workers, budget=BUDGET
        )
    )
    process = subprocess.Popen(
        [str(PYTHON), str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        # Own process group, so the signal can be sent to the group exactly
        # the way a terminal's Ctrl-C would deliver it.
        start_new_session=True,
    )
    if interrupt_after is not None:
        time.sleep(interrupt_after)
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
    out, _ = process.communicate(timeout=300)
    return process.returncode, out


def check(label, ok, fail_detail="", ok_detail=""):
    # Per-outcome detail: a single shared string prints things like
    # "PASS -- no stop message in output", which reads as a contradiction
    # and makes the whole report untrustworthy.
    detail = ok_detail if ok else fail_detail
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def main():
    workdir = pathlib.Path(tempfile.mkdtemp())
    script = workdir / "driver.py"
    state_path = workdir / "state.json"

    print(f"\nInterrupting a {BUDGET}-episode round at 4 workers\n")
    code, out = run(workdir, workers=4, script_path=script, interrupt_after=6.0)

    check("training exited cleanly", code == 0, fail_detail=f"exit code {code}")
    check(
        "announced the graceful stop",
        "letting workers finish" in out,
        fail_detail="no stop message in output",
    )
    check("state file written", state_path.exists(), fail_detail="no state.json")
    if not state_path.exists():
        print("\n--- output ---\n" + out)
        return 1

    saved = json.loads(state_path.read_text())
    check(
        "saved the interrupted round",
        saved["round"] == 1,
        fail_detail=f"round={saved['round']}, expected 1",
        ok_detail="round 1",
    )
    check(
        "round was genuinely cut short",
        0 < saved["total_episodes"] < BUDGET,
        fail_detail=f"{saved['total_episodes']} episodes against a {BUDGET} budget -- "
                    f"not actually interrupted mid-round",
        ok_detail=f"{saved['total_episodes']} of {BUDGET} episodes",
    )
    check(
        "stopped instead of starting another round",
        "Round   2" not in out,
        fail_detail="a second round began after the interrupt",
    )
    first_episodes = saved["total_episodes"]
    print(f"        saved round {saved['round']}, {first_episodes} episodes, "
          f"epsilon {saved['epsilon']:.4f}")

    print(f"\nResuming at 18 workers\n")
    code, out = run(workdir, workers=18, script_path=script, interrupt_after=6.0)

    check("resumed run exited cleanly", code == 0, fail_detail=f"exit code {code}")
    # initial_state() reports the round it loaded and then continues from
    # the one after it, so the resume line names round 1 while the first
    # round actually run is 2.
    check(
        "picked up the saved round",
        "Resuming from round 1" in out,
        fail_detail="did not report resuming from the saved round",
        ok_detail="reported resuming from round 1",
    )
    check(
        "continued at the next round rather than redoing round 1",
        "Round   2" in out and "Round   1" not in out,
        fail_detail="re-ran round 1 instead of continuing",
        ok_detail="first round run was 2",
    )
    resumed = json.loads(state_path.read_text())
    check(
        "kept the earlier episodes",
        resumed["total_episodes"] > first_episodes,
        fail_detail=f"totals did not accumulate: {first_episodes} -> {resumed['total_episodes']}",
        ok_detail=f"{first_episodes} -> {resumed['total_episodes']}",
    )
    check(
        "accepted the new worker count",
        "18 workers" in out,
        fail_detail="worker count not reflected at launch",
        ok_detail="launched with 18 workers",
    )
    print(f"        now at round {resumed['round']}, "
          f"{resumed['total_episodes']} episodes total")

    print()
    if failures:
        print(f"{len(failures)} failure(s): {failures}")
        print("\n--- output of last run ---\n" + out)
        return 1
    print("Interrupt-and-resume works: a round survives Ctrl-C, and the")
    print("worker count can change between launches without losing progress.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
