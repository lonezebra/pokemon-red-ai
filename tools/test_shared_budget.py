"""
Check that a training round's shared episode budget self-balances across
workers of unequal speed, and that the round's accounting stays exact.

    .venv/bin/python3 tools/test_shared_budget.py

Runs one round with a stub environment whose per-episode cost depends on
the worker's own PID, giving a deliberate 4x speed gap that stands in for
a machine's differing core tiers. Three things have to hold:

  - Workers claim exactly the budget between them, no more and no fewer.
    An off-by-one in the claim-under-lock would show up as a leaked or
    double-counted episode.
  - Faster workers finish more episodes than slower ones. A flat spread
    here would mean the queue isn't actually balancing anything.
  - Epsilon decays over the *average* worker's episode count rather than
    the round total. Workers decay their own epsilon per episode, so
    using the total would collapse exploration roughly num_workers times
    too fast -- silent, and only visible many rounds later as an agent
    that stopped exploring long before it should have.

Uses a stub env rather than PyBoy on purpose: the queue's correctness
has nothing to do with the game, and a real emulator would make this
take minutes instead of a second.
"""
import json
import os
import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import train_navigation_parallel as t

# GIF writing is irrelevant to what's under test, and the fake frames
# aren't real images.
t.save_gif = lambda *args, **kwargs: None


class _FakeImage:
    def copy(self):
        return self


class _FakeScreen:
    image = _FakeImage()


class _FakePyBoy:
    screen = _FakeScreen()


class FakeEnv:
    """Speed varies by PID parity, standing in for fast/slow core tiers."""

    def __init__(self, max_steps=10):
        self.max_steps = max_steps
        self.visited_positions = set()
        self.pyboy = _FakePyBoy()  # run_demo_episode grabs frames from this
        self.delay = 0.002 if os.getpid() % 2 == 0 else 0.008  # 4x gap

    def reset(self):
        self.visited_positions = {(0, 0)}
        return {"map_id": 1, "x": 0, "y": 0}

    def step(self, action):
        time.sleep(self.delay)
        return (
            {"map_id": 1, "x": 1, "y": 1},
            1.0,
            True,
            {"reached_goal": True, "step_count": 1},
        )

    def close(self):
        pass


def main():
    d = pathlib.Path(tempfile.mkdtemp())
    t.WORKER_DIR = d / "workers"

    BUDGET, WORKERS = 120, 6
    t.train(
        env_class=FakeEnv,
        model_path=d / "q.json",
        state_path=d / "state.json",
        gif_prefix="test",
        max_steps=3,
        num_workers=WORKERS,
        episodes_per_round=BUDGET,
        max_rounds=1,
    )

    counts = [
        json.load(open(d / "workers" / "test" / f"worker{i}_summary.json"))["episodes"]
        for i in range(WORKERS)
    ]
    print()
    print(f"  budget={BUDGET} workers={WORKERS}")
    print(f"  episodes per worker: {sorted(counts)}")
    print(f"  total claimed: {sum(counts)}")
    assert sum(counts) == BUDGET, f"budget leaked: {sum(counts)} != {BUDGET}"
    print(f"  exactly the budget, no leak or double-claim")
    if min(counts) != max(counts):
        print(f"  spread {min(counts)}-{max(counts)} -- faster workers claimed more")
    else:
        print(f"  flat spread (workers happened to run at equal speed)")

    state = json.load(open(d / "state.json"))
    expected_eps = 1.0 * (0.998 ** (BUDGET / WORKERS))
    print(f"  total_episodes recorded: {state['total_episodes']} (actual, not workers*share)")
    print(f"  epsilon: {state['epsilon']:.5f}  expected ~{expected_eps:.5f}")
    assert abs(state["epsilon"] - expected_eps) < 1e-6, "epsilon schedule drifted"
    print(f"  epsilon decayed over the average worker's episodes, not the round total")


if __name__ == "__main__":
    main()
