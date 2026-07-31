import os

# Must precede any core import -- see train_forest_agent.py for why a
# real SDL window per worker is actively harmful here.
os.environ.setdefault("POKEMON_AI_WINDOW_MODE", "null")

import json  # noqa: E402
import pathlib  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))

from core.config import PROJECT_ROOT  # noqa: E402
from envs.forest_curriculum_env import (  # noqa: E402
    STAGE_VAR,
    CurriculumForestEnv,
    stage_start_states,
)
from policy_accuracy import is_correct, load_edges  # noqa: E402
from rewards.forest_rewards import _DISTANCES  # noqa: E402
from train_navigation_parallel import load_progress, train  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "forest_curriculum_q_table.json"
STATE_PATH = PROJECT_ROOT / "models" / "forest_curriculum_parallel_state.json"
SEED_TABLE_PATH = PROJECT_ROOT / "models" / "forest_q_table.json"
STAGE_LOG_PATH = PROJECT_ROOT / "models" / "forest_curriculum_stages.json"

# Stages advance along the captured checkpoints, which are every 5 hops.
STAGE_STEP = 5
FIRST_STAGE = 10
FINAL_STAGE = 125

# A stage is mastered when the greedy policy is right on this fraction of
# the tiles it covers. Not 100%: a handful of tiles sit on genuinely
# equivalent alternative routes, and the distance-reducing test scores
# only one of them as correct, so demanding perfection would stall
# forever on tiles that are not actually wrong.
MASTERY_ACCURACY = 0.97

# Give up on a stage rather than grinding it forever. Hitting this is
# information, not just a timeout: it means visiting those tiles
# constantly still is not fixing them, which would point at the reward or
# the state representation rather than the visit distribution.
MAX_ROUNDS_PER_STAGE = 12

# Episodes start at most `stage` hops out, so they need nowhere near the
# 2000 steps entrance-start training uses. Generous multiple of the
# stage distance, floored so the earliest stages still have room for a
# trainer fight and some wandering.
STEP_BUDGET_MULTIPLIER = 12
MIN_MAX_STEPS = 250


def stage_max_steps(stage):
    return max(MIN_MAX_STEPS, stage * STEP_BUDGET_MULTIPLIER)


def covered_tiles(stage):
    """Tiles the current stage can actually reach: everything at most
    `stage` hops from the goal. Scoring accuracy over the whole map would
    average in regions this stage never visits and mask its progress."""
    return [t for t, d in _DISTANCES.items() if 0 < d <= stage]


def _accuracy_over(table, edges, tiles):
    correct = seen = 0
    for tile in tiles:
        verdict = is_correct(table, edges, tile)
        if verdict is None:
            continue
        seen += 1
        correct += bool(verdict)
    return (correct / seen if seen else 0.0), correct, seen


def stage_accuracy(stage, edges):
    """Fraction of this stage's tiles whose greedy action moves strictly
    closer to the goal, plus the counts behind it."""
    if not MODEL_PATH.exists():
        return 0.0, 0, 0
    table = json.loads(MODEL_PATH.read_text())
    return _accuracy_over(table, edges, covered_tiles(stage))


def whole_map_accuracy(edges):
    """
    Accuracy across every tile with a known distance, not just this
    stage's. Purely a visibility check, never a mastery gate: the
    uniform-over-included-states sampling in stage_start_states already
    keeps earlier tiles under traffic as the stage widens, so they are
    not expected to regress -- this is what would notice it if that
    assumption turned out to be wrong.
    """
    if not MODEL_PATH.exists():
        return 0.0, 0, 0
    table = json.loads(MODEL_PATH.read_text())
    return _accuracy_over(table, edges, [t for t, d in _DISTANCES.items() if d > 0])


def seed_table_if_missing():
    """
    Start from the entrance-trained table rather than from scratch.

    It is already ~92% accurate overall; the curriculum's job is the
    stubborn remainder, and rebuilding the 92% would waste the 18000
    episodes that produced it. Copied rather than shared so the running
    entrance-start job -- if it is still going -- cannot be corrupted by
    this one, and so a failed experiment is thrown away by deleting one
    file.
    """
    if MODEL_PATH.exists():
        return
    if SEED_TABLE_PATH.exists():
        shutil.copy2(SEED_TABLE_PATH, MODEL_PATH)
        print(f"Seeded {MODEL_PATH.name} from {SEED_TABLE_PATH.name}")
    else:
        print(f"No {SEED_TABLE_PATH.name} to seed from; starting from scratch")


def log_stage(entry):
    log = []
    if STAGE_LOG_PATH.exists():
        try:
            log = json.loads(STAGE_LOG_PATH.read_text())
        except json.JSONDecodeError:
            log = []
    log.append(entry)
    STAGE_LOG_PATH.write_text(json.dumps(log, indent=2))


def main():
    edges = load_edges()
    seed_table_if_missing()

    stage = FIRST_STAGE
    while stage <= FINAL_STAGE:
        states = stage_start_states(stage)
        max_steps = stage_max_steps(stage)
        os.environ[STAGE_VAR] = str(stage)

        accuracy, correct, seen = stage_accuracy(stage, edges)
        print(
            f"\n=== stage d<={stage}: {len(states)} start states, "
            f"max_steps={max_steps}, accuracy {accuracy:.1%} ({correct}/{seen}) ==="
        )

        if accuracy >= MASTERY_ACCURACY:
            print(f"already at or above {MASTERY_ACCURACY:.0%}; advancing")
            log_stage({"stage": stage, "rounds": 0, "accuracy": accuracy,
                       "correct": correct, "seen": seen, "mastered": True})
            stage += STAGE_STEP
            continue

        rounds_done = 0
        while rounds_done < MAX_ROUNDS_PER_STAGE:
            progress = load_progress(STATE_PATH)
            next_round = (progress["round"] + 1) if progress else 1

            # One round per call: train() runs range(start_round,
            # max_rounds + 1), so this returns after exactly one, letting
            # mastery be checked between rounds instead of after a fixed
            # block of them.
            train(
                env_class=CurriculumForestEnv,
                model_path=MODEL_PATH,
                state_path=STATE_PATH,
                gif_prefix=f"forest_curriculum_d{stage:03d}",
                max_steps=max_steps,
                max_rounds=next_round,
            )
            rounds_done += 1

            accuracy, correct, seen = stage_accuracy(stage, edges)
            overall_acc, overall_correct, overall_seen = whole_map_accuracy(edges)
            print(f"  stage d<={stage} round {rounds_done}: "
                  f"accuracy {accuracy:.1%} ({correct}/{seen}), "
                  f"whole-map {overall_acc:.1%} ({overall_correct}/{overall_seen})")

            if accuracy >= MASTERY_ACCURACY:
                break

        mastered = accuracy >= MASTERY_ACCURACY
        log_stage({"stage": stage, "rounds": rounds_done, "accuracy": accuracy,
                   "correct": correct, "seen": seen, "mastered": mastered,
                   "whole_map_accuracy": overall_acc})

        if not mastered:
            print(
                f"\nstage d<={stage} did not reach {MASTERY_ACCURACY:.0%} in "
                f"{MAX_ROUNDS_PER_STAGE} rounds (stopped at {accuracy:.1%}).\n"
                f"Stopping here rather than widening the start line over an "
                f"unlearned segment -- every later stage has to walk through it."
            )
            return 1

        print(f"stage d<={stage} mastered in {rounds_done} round(s)")
        stage += STAGE_STEP

    print("\nall stages mastered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
