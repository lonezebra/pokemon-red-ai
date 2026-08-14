import os
import pathlib

from core.config import PROJECT_ROOT
from envs.forest_env import PokemonRedForestEnv

CURRICULUM_DIR = PROJECT_ROOT / "saves" / "forest_curriculum"

# Which stage the workers should train. Passed through the environment
# rather than as a constructor argument because train_navigation_parallel
# instantiates the env class as env_class(max_steps=...) inside spawned
# worker processes -- there is no channel for extra arguments, and a
# closure over the stage would not survive spawn's pickling. Children
# inherit os.environ, so the driver setting this before spawning is
# enough. Defaults to the whole forest, which makes an unset variable
# behave like ordinary entrance-to-exit training rather than silently
# training some arbitrary fragment.
STAGE_VAR = "POKEMON_RED_FOREST_STAGE"
DEFAULT_STAGE = 999


def stage_start_states(stage):
    """
    Every captured checkpoint at most `stage` hops from the goal.

    Returning the whole prefix rather than only the checkpoint at exactly
    `stage` is what keeps earlier stages alive: an agent that trains
    solely at its current start line stops visiting the segment it
    already learned, and tabular Q-values there decay as the merge
    averages in workers that never touched them. Sampling uniformly over
    every stage up to the current one means each advance widens the
    distribution instead of moving it.
    """
    states = []
    for path in sorted(CURRICULUM_DIR.glob("d*.state")):
        try:
            distance = int(path.name[1:4])
        except ValueError:
            continue
        if distance <= stage:
            states.append(path)
    return states


def current_stage():
    try:
        return int(os.environ.get(STAGE_VAR, DEFAULT_STAGE))
    except ValueError:
        return DEFAULT_STAGE


class CurriculumForestEnv(PokemonRedForestEnv):
    """
    The forest env, started from a save state near the goal instead of at
    the entrance.

    The problem this addresses, measured rather than assumed: across
    rounds 52-67 of entrance-start training, overall policy accuracy sat
    flat at 91-93% (55-63 wrong tiles of 712) with no trend, and 33 tiles
    were wrong in every round sampled -- roughly 2500 episodes without
    fixing them. Reported depth swung 33 -> 92 -> 36 -> 68 over the same
    span, but that was a threshold on near-tie Q-values rather than
    policy change; one flipped tile accounted for a 32-hop swing.

    A tile only improves if it is visited, and from the entrance the
    agent reaches goal-side tiles rarely enough that they collect almost
    no updates. Starting 5 or 10 hops out puts exactly those tiles under
    constant traffic. Nothing else changes -- same reward function, same
    action space, same trainer handling -- so what this tests is
    specifically the visit distribution, not a new learning rule.
    """

    def __init__(self, max_steps=1000):
        stage = current_stage()
        states = stage_start_states(stage)
        if not states:
            raise FileNotFoundError(
                f"No curriculum states at or below distance {stage} in "
                f"{CURRICULUM_DIR}. Run tools/build_curriculum_states.py first."
            )
        super().__init__(max_steps=max_steps, start_states=states)
