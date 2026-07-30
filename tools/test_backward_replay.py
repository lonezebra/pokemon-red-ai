"""
Check that an episode's terminal outcome reaches the episode's start in
one episode, not one tile per episode.

    .venv/bin/python3 tools/test_backward_replay.py

The forest's greedy frontier sat pinned at distance ~93 while training
successes climbed, because one-step Q-learning applied forward moves a
terminal outcome exactly one state per episode: the update at step t uses
the successor's value as it was *before* the successor's own update.
Replaying the same updates backward at episode end lets each update see
its successor's fresh value, so a single episode carries its own outcome
along its entire length.

The check runs a worker (the real run_worker, spawned exactly as training
spawns it) on a 20-tile corridor whose only reward is +100 at the end,
with a single episode's budget. Under forward updating, one episode
leaves the start tile's value at 0 (the reward has propagated one tile).
Under backward replay, the start tile's best value must already be
positive -- gamma^19 * alpha-scaled, small but strictly nonzero -- and
monotone along the corridor.

Also asserts the visit-count sidecar still matches what was replayed,
since the weighted merge depends on those counts staying truthful.
"""

import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import train_navigation_parallel as t

CORRIDOR = 20


class _Img:
    def copy(self):
        return self


class _Screen:
    image = _Img()


class _PyBoy:
    screen = _Screen()


class CorridorEnv:
    """
    Tiles x=0..CORRIDOR-1; action 3 ("right") advances, everything else
    stays put. Reward only on reaching the last tile. No shaping, so any
    value at the start tile can only have arrived via propagation.
    """

    def __init__(self, max_steps=CORRIDOR + 5):
        self.max_steps = max_steps
        self.pyboy = _PyBoy()
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
        if action == 3:
            self.x += 1
        self.visited_positions.add((self.x,))
        reached = self.x >= CORRIDOR - 1
        done = reached or self.steps >= self.max_steps
        reward = 100.0 if reached else 0.0
        return (
            {"map_id": 1, "x": self.x, "y": 0},
            reward,
            done,
            {"reached_goal": reached, "step_count": self.steps},
        )

    def close(self):
        pass


class AlwaysRight:
    """Deterministic scripted policy; epsilon plumbing is irrelevant here."""
    epsilon = 0.0

    def choose_action(self, obs, greedy=False):
        return 3


failures = []


def check(label, ok, fail_detail="", ok_detail=""):
    detail = ok_detail if ok else fail_detail
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def main():
    d = pathlib.Path(tempfile.mkdtemp())
    table_path = d / "worker0.json"
    summary_path = d / "worker0_summary.json"

    # Drive run_worker directly, in-process: one episode's budget. The
    # scripted policy needs to survive QLearningAgent's interface, so
    # monkeypatch the agent class it constructs.
    class ScriptedAgent(t.QLearningAgent):
        def choose_action(self, obs, greedy=False):
            return 3

    original = t.QLearningAgent
    t.QLearningAgent = ScriptedAgent
    try:
        remaining = t._SPAWN_CTX.Value("i", 1)
        t.run_worker(
            CorridorEnv, remaining, 0.0, d / "none.json",
            table_path, summary_path, CORRIDOR + 5,
        )
    finally:
        t.QLearningAgent = original

    table = json.loads(table_path.read_text())
    start_value = max(table.get("1,0,0", [0, 0, 0, 0]))
    mid_value = max(table.get(f"1,{CORRIDOR // 2},0", [0, 0, 0, 0]))
    end_value = max(table.get(f"1,{CORRIDOR - 2},0", [0, 0, 0, 0]))

    check(
        "terminal reward reaches the start tile in one episode",
        start_value > 0,
        fail_detail=f"start value {start_value} -- forward updating would leave "
                    f"this at 0 until ~{CORRIDOR} episodes had run",
        ok_detail=f"start={start_value:.2e} > 0 after a single episode "
                  f"(alpha*gamma compounds per hop, so one pass leaves a tiny "
                  f"but strictly nonzero trace the whole way back)",
    )
    check(
        "value is monotone along the corridor",
        end_value > mid_value > start_value > 0,
        fail_detail=f"start={start_value:.2e} mid={mid_value:.2e} end={end_value:.2e}",
        ok_detail=f"start={start_value:.2e} < mid={mid_value:.2e} < end={end_value:.2e}",
    )

    counts = json.loads((pathlib.Path(str(table_path) + ".counts")).read_text())
    check(
        "visit counts match the replayed transitions",
        counts.get("1,0,0|3") == 1 and sum(counts.values()) == CORRIDOR - 1,
        fail_detail=f"counts off: {dict(list(counts.items())[:3])}... "
                    f"total {sum(counts.values())}, expected {CORRIDOR - 1}",
        ok_detail=f"{sum(counts.values())} updates counted, one per transition",
    )

    print()
    if failures:
        print(f"{len(failures)} failure(s): {failures}")
        return 1
    print("One episode now teaches its whole trajectory, not just its last tile.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
