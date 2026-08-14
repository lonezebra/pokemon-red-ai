import os

# Must precede any core import -- see train_forest_agent.py for why a
# real SDL window per worker is actively harmful here.
os.environ.setdefault("POKEMON_AI_WINDOW_MODE", "null")

import json  # noqa: E402
import pathlib  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))

from actions import num_actions  # noqa: E402
from agents.q_learning_agent import QLearningAgent  # noqa: E402
from core.config import PROJECT_ROOT  # noqa: E402
from envs.forest_curriculum_env import (  # noqa: E402
    STAGE_VAR,
    CurriculumForestEnv,
    stage_start_states,
)
from envs.forest_env import PokemonRedForestEnv  # noqa: E402
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


def is_mastered(correct, seen):
    """
    Wrong-tile budget rather than a bare percentage. At small stages the
    percentage collapses into perfection -- 97% of stage 15's 30 tiles is
    29.1, i.e. 30/30 -- and at least one tile there has a corrupted
    answer key (the survey snapshotted it mid trainer-sighting, recording
    edges from a frozen cutscene), so perfection is not merely hard, it
    can be structurally unreachable. One wrong tile is always allowed;
    larger stages keep the 3% budget the percentage always gave them.

    NOTE: since the behavioral gate below was added, this scores
    telemetry only. It stopped being the gate the day it blocked stage
    d<=25 on tiles where the agent was measurably right and the answer
    key wrong: the shortest-path route those tiles were graded against
    runs straight through an unbeaten trainer, and the agent's preferred
    detour -- persistently valued higher across thousands of episodes --
    was it correctly pricing in a forced battle the trainer-blind
    distance map cannot see.
    """
    if seen == 0:
        return False
    allowed = max(1, int(seen * (1.0 - MASTERY_ACCURACY)))
    return (seen - correct) <= allowed


def stage_demo(stage, max_steps):
    """
    The behavioral mastery gate: a purely greedy walk from the stage's
    deepest start state. Mastered means it reaches the exit with zero
    nudges -- the policy is genuinely solving this stretch of forest,
    not being carried by the demo's loop-breaker.

    This is the criterion the per-tile accuracy score was standing in
    for, and the reason it can't stand in: accuracy grades each tile
    against "step toward the goal by shortest graph path", but the graph
    is trainer-blind. Where the shortest path crosses an unbeaten
    trainer, the truly optimal policy detours, and accuracy counts every
    tile of that correct detour as an error. Reaching the exit greedily
    is the thing itself rather than a proxy for it.

    Up to three attempts, with attempt N taking N random steps before
    going greedy. One attempt turned out to be a trap: the emulator is
    deterministic from a restored save under a fixed action sequence, so
    a single greedy rollout is the *same* rollout every round -- stage
    d<=120 replayed one unlucky low-HP wild-battle death identically
    twelve times and failed its whole budget while whole-map accuracy
    sat at an all-time high. The random prefix de-correlates the
    attempts; requiring the greedy remainder to reach the exit keeps the
    gate honest, since a couple of prefix steps can't carry a walk that
    the policy isn't doing itself.
    """
    states = stage_start_states(stage)
    env = PokemonRedForestEnv(max_steps=max_steps, start_states=[states[-1]])
    agent = QLearningAgent(num_actions=num_actions())
    agent.load(MODEL_PATH)
    rng = random.Random(stage)
    try:
        for attempt in range(3):
            obs = env.reset()
            info = {}
            for _ in range(attempt):
                obs, _, done, info = env.step(rng.randrange(num_actions()))
                if done:
                    break
            else:
                for _ in range(max_steps):
                    action = agent.choose_action(obs, greedy=True)
                    obs, _, done, info = env.step(action)
                    if done:
                        break
            if info.get("reached_goal"):
                return {
                    "reached_goal": True,
                    "nudges": 0,
                    "steps": info.get("step_count", 0),
                    "start": states[-1].name,
                    "attempt": attempt,
                }
    finally:
        env.close()
    return {
        "reached_goal": False,
        "nudges": 0,
        "steps": info.get("step_count", 0),
        "start": states[-1].name,
        "attempt": 2,
    }

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
    """
    Score every tile passed in, counting one the table has never seen as
    wrong rather than skipping it.

    is_correct returns None both for "no Q-row yet" and for "not a
    scorable tile", but every caller here pre-filters to tiles with a
    real distance to the goal, so None can only mean the agent has never
    been there -- which is not a correct policy, it is no policy.

    Skipping them silently broke the gate the first time it mattered:
    clearing two poisoned rows so they would relearn from neutral made
    them invisible instead, and stage d<=15 declared mastery at round 0
    on the 28 tiles it could still see -- advancing past exactly the two
    tiles the clearing was meant to fix.
    """
    correct = seen = 0
    for tile in tiles:
        seen += 1
        correct += bool(is_correct(table, edges, tile))
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
        demo = stage_demo(stage, max_steps)
        print(
            f"\n=== stage d<={stage}: {len(states)} start states, "
            f"max_steps={max_steps}, accuracy {accuracy:.1%} ({correct}/{seen}), "
            f"greedy from {demo['start']}: reached={demo['reached_goal']} "
            f"nudges={demo['nudges']} ==="
        )

        if demo["reached_goal"] and demo["nudges"] == 0:
            print("greedy already solves this stage nudge-free; advancing")
            log_stage({"stage": stage, "rounds": 0, "accuracy": accuracy,
                       "correct": correct, "seen": seen, "mastered": True,
                       "demo": demo})
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

            # A Ctrl-C lands inside train(), which abandons the round and
            # returns normally -- so without this the stage loop would
            # relaunch immediately, making the driver impossible to exit
            # (each ^C also burned a phantom stage round: rounds_done
            # advanced with zero episodes trained). An abandoned round is
            # detectable as the state file not advancing; treat it as the
            # interrupt it is and stop the whole driver.
            after = load_progress(STATE_PATH)
            if (after["round"] if after else 0) < next_round:
                print("round was abandoned (Ctrl-C); stopping the curriculum "
                      "driver -- relaunch to resume this stage")
                return 130
            rounds_done += 1

            accuracy, correct, seen = stage_accuracy(stage, edges)
            overall_acc, overall_correct, overall_seen = whole_map_accuracy(edges)
            demo = stage_demo(stage, max_steps)
            print(f"  stage d<={stage} round {rounds_done}: "
                  f"accuracy {accuracy:.1%} ({correct}/{seen}), "
                  f"whole-map {overall_acc:.1%} ({overall_correct}/{overall_seen}), "
                  f"greedy reached={demo['reached_goal']} nudges={demo['nudges']}")

            if demo["reached_goal"] and demo["nudges"] == 0:
                break

        mastered = demo["reached_goal"] and demo["nudges"] == 0
        log_stage({"stage": stage, "rounds": rounds_done, "accuracy": accuracy,
                   "correct": correct, "seen": seen, "mastered": mastered,
                   "whole_map_accuracy": overall_acc, "demo": demo})

        if not mastered:
            print(
                f"\nstage d<={stage}: greedy still can't reach the exit "
                f"nudge-free from {demo['start']} after "
                f"{MAX_ROUNDS_PER_STAGE} rounds (accuracy {accuracy:.1%}).\n"
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
