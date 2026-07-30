"""
Check that a demo episode escapes a policy whose best action is blocked.

    .venv/bin/python3 tools/test_demo_loop_breaker.py

A purely greedy demo deadlocks by construction. The observation is exactly
(map_id, x, y), so if the Q-table's argmax at some tile points into a wall
or a trainer, the step doesn't move, the next observation is identical, the
argmax is identical, and the episode burns its whole step budget bumping
one tile. Unlike training, a demo performs no updates, so the offending
action's value never drops and nothing breaks the cycle. That is what
"stuck on a trainer" was, and why demos reported steps=2000 with a handful
of tiles visited while the training success rate was climbing.

Two cases, using a stub env with one deliberately impassable direction and
an agent that always prefers exactly that direction -- the worst case:

  - Without a loop breaker the episode must consume every step and end up
    nowhere, confirming the deadlock is real rather than hypothetical.
  - With one it must escape, reach the goal, and report a non-zero nudge
    count, since the nudges are what carried it.

The nudge count is the point of the second assertion. A loop breaker that
silently rescued a bad policy would make every demo look better than the
agent is; reporting nudges keeps that visible, so "reached the goal" and
"solved the maze" stay distinguishable.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import train_navigation_parallel as t

BLOCKED_DIRECTION = 0  # whichever action index the stub refuses to move on
GOAL_X = 4


class _FakeImage:
    def copy(self):
        return self


class _FakeScreen:
    image = _FakeImage()


class _FakePyBoy:
    screen = _FakeScreen()


class WallEnv:
    """
    A corridor where action BLOCKED_DIRECTION never moves the player and
    every other action advances toward the goal. Standing still returns an
    identical observation, which is the precondition for the deadlock.
    """

    def __init__(self, max_steps=200):
        self.max_steps = max_steps
        self.pyboy = _FakePyBoy()
        self.visited_positions = set()
        self.x = 0
        self.steps = 0

    def reset(self):
        self.x = 0
        self.steps = 0
        self.visited_positions = {(0,)}
        return {"map_id": 1, "x": 0, "y": 0}

    def step(self, action):
        self.steps += 1
        if action != BLOCKED_DIRECTION:
            self.x += 1
        self.visited_positions.add((self.x,))
        reached = self.x >= GOAL_X
        done = reached or self.steps >= self.max_steps
        return (
            {"map_id": 1, "x": self.x, "y": 0},
            1.0 if reached else -0.01,
            done,
            {"reached_goal": reached, "step_count": self.steps, "moved": action != BLOCKED_DIRECTION},
        )

    def close(self):
        pass


class StubbornAgent:
    """Always picks the one action that cannot move. No learning."""

    def choose_action(self, obs, greedy=False):
        return BLOCKED_DIRECTION


failures = []


def check(label, ok, fail_detail="", ok_detail=""):
    detail = ok_detail if ok else fail_detail
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def main():
    max_steps = 200

    print("\nWithout a loop breaker (the old behavior)\n")
    original = t.DEMO_STUCK_LIMIT
    t.DEMO_STUCK_LIMIT = 10**9  # effectively never nudge
    try:
        demo = t.run_demo_episode(WallEnv(max_steps=max_steps), StubbornAgent(), max_steps)
    finally:
        t.DEMO_STUCK_LIMIT = original

    check(
        "deadlocks for the whole budget",
        demo["steps"] == max_steps and not demo["reached_goal"],
        fail_detail=f"steps={demo['steps']} reached_goal={demo['reached_goal']}",
        ok_detail=f"steps={demo['steps']}, reached_goal=False, "
                  f"tiles_visited={demo['tiles_visited']}",
    )
    check(
        "never nudged",
        demo["nudges"] == 0,
        fail_detail=f"nudges={demo['nudges']}",
        ok_detail="nudges=0",
    )

    print(f"\nWith the loop breaker (DEMO_STUCK_LIMIT={original})\n")
    demo = t.run_demo_episode(WallEnv(max_steps=max_steps), StubbornAgent(), max_steps)

    check(
        "escapes and reaches the goal",
        demo["reached_goal"],
        fail_detail=f"still stuck: steps={demo['steps']} tiles={demo['tiles_visited']}",
        ok_detail=f"reached_goal in {demo['steps']} steps",
    )
    check(
        "reports the nudges that carried it",
        demo["nudges"] > 0,
        fail_detail="nudges=0, so the escape is unexplained",
        ok_detail=f"nudges={demo['nudges']} -- visibly not the policy's own doing",
    )
    check(
        "does not silently rescue a bad policy",
        demo["nudges"] >= GOAL_X,
        fail_detail=f"nudges={demo['nudges']} for {GOAL_X} required moves",
        ok_detail=f"{demo['nudges']} nudges for {GOAL_X} forward moves -- the count "
                  f"makes the policy's uselessness legible",
    )

    print()
    if failures:
        print(f"{len(failures)} failure(s): {failures}")
        return 1
    print("A blocked argmax no longer costs the whole episode, and the nudge")
    print("count keeps a rescued demo from reading as a competent one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
