"""
Clamp a Q-table poisoned by the goal-strip reward seesaw back into the
reward function's legitimate range.

    .venv/bin/python3 tools/repair_forest_q_table.py [path]

(defaults to models/forest_q_table.json, rewritten in place, atomically)

What happened: the warp strip at (1,0) -- the tile the game walks the
player through mid-exit -- was missing from the distance map, so the two
halves of every successful exit scored roughly -147 then +248 instead of
a clean +101. Q-values near the goal inflated to 190 against a
theoretical ceiling of ~101, the inflation propagated backward through
merges, and once mid-corridor action orderings scrambled, training
success collapsed 27% -> 3% in one round.

The reward bug is fixed in rewards/forest_rewards.py; this repairs the
already-poisoned table so the run resumes from its 16k episodes of real
learning instead of from scratch. Clamping to the achievable value range
is deliberately the *whole* repair: inside the corridor the relative
orderings are mostly sound (they got the agent 27% success), and the
bootstrap will re-descend the clamped ceiling values to their true
levels far faster than retraining would rebuild everything.

Bounds, derived from the reward function rather than picked:
  upper: reaching the goal next step pays at most -0.01 + 1 + 100, and no
         state can be worth more than that undiscounted best case ~101.
  lower: the worst terminal is a faint near the goal, forfeiting the full
         potential (~ -148 - 20); anything below that is seesaw damage,
         but values that low are also legitimate enough to keep -- the
         lower clamp mainly documents the floor. -170 covers it.
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from core.atomic_io import write_json_atomic
from core.config import PROJECT_ROOT

UPPER = 101.0
LOWER = -170.0


def main():
    path = pathlib.Path(
        sys.argv[1] if len(sys.argv) > 1
        else PROJECT_ROOT / "models" / "forest_q_table.json"
    )
    table = json.loads(path.read_text())

    clamped = 0
    worst = 0.0
    for key, row in table.items():
        for i, value in enumerate(row):
            if value > UPPER:
                worst = max(worst, value)
                row[i] = UPPER
                clamped += 1
            elif value < LOWER:
                row[i] = LOWER
                clamped += 1

    write_json_atomic(path, table)
    print(f"{path}: clamped {clamped} of {sum(len(r) for r in table.values())} "
          f"values into [{LOWER}, {UPPER}] (worst offender {worst:.1f})")
    if clamped == 0:
        print("table was already within range -- nothing to repair")
    return 0


if __name__ == "__main__":
    sys.exit(main())
