import os

# Headless unless explicitly asked otherwise -- must precede any core
# import, since core/config.py reads this at import time. See
# train_forest_agent.py for why a real SDL window is actively harmful
# for a script that just wants to run the emulator as fast as possible.
os.environ.setdefault("POKEMON_AI_WINDOW_MODE", "null")

import random  # noqa: E402

from stable_baselines3 import DQN  # noqa: E402

from core.battle_runner import fight_current_battle  # noqa: E402
from core.config import PROJECT_ROOT  # noqa: E402
from core.controls import wait_for_free_movement, walk_tile  # noqa: E402
from core.emulator import create_emulator, run_frames  # noqa: E402
from core.memory import (  # noqa: E402
    get_battle_type,
    get_party_hp,
    get_party_hp_fraction,
    get_party_level,
    get_party_max_hp,
    get_player_position,
    is_in_battle,
)
from core.pathfind import walk_to_map  # noqa: E402
from core.state import load_state, save_state  # noqa: E402
from survey_route3 import PEWTER_CITY_MAP_ID, ROUTE_3_MAP_ID, heal_at_pewter_center  # noqa: E402

# Levels the Route 3 party up before pushing further east, the same
# playbook as create_leveled_state.py's forest grind: measured, not
# guessed. A live replay of the trainer blocking the survey's heal-trip
# route showed a deterministic 72% of max HP lost on a clean win (8/8
# trials, identical result each time) -- a single fight, let alone the
# two-in-a-row a heal trip can hit, is not survivable at Lv13-14's
# ~41-43 max HP. That is a level gap, not a policy weakness or bad luck,
# exactly the forest's original "always Tackle wins 63%" lesson: no
# amount of reward tuning or threshold adjustment fixes an unreachable
# target.
#
# Grinding happens on Route 2, not Route 3 itself: a live probe of the
# already-surveyed 88-tile Route 3 pocket found zero wild encounters in
# 300 random steps -- it's pure entrance walkway with no grass at all,
# so there is nothing there to grind against. Route 2 borders Pewter
# City directly (confirmed by walk_to_map), already has a real
# wild-encounter rate (confirmed: one within 48 random steps), and is
# fully cleared territory from earlier chapters of this project, so
# there is no unbeaten-trainer risk in walking it freely. Route 2 ->
# Viridian City failed to path (almost certainly the one-way ledge this
# project's own history already documents for that route), which would
# have been the way to reach Route 1 instead -- but Route 1 was never
# actually required, any already-cleared outdoor route with grass does
# the job, and Route 2 already is one.
WILD_MODEL_PATH = PROJECT_ROOT / "models" / "wild_battle_dqn.zip"
TRAINER_MODEL_PATH = PROJECT_ROOT / "models" / "trainer_battle_dqn.zip"
START_STATE_PATH = PROJECT_ROOT / "saves" / "route3_entry.state"
LEVELED_STATE_PATH = PROJECT_ROOT / "saves" / "route3_leveled.state"

GRIND_MAP_ID = 13  # Route 2

# Roughly proportionate to the forest's own Lv6->Lv10 jump (a level gap
# clearing an unwinnable fight), scaled up since Route 3's opposition
# levels (9-11) sit higher than the forest's (the current party is
# already Lv13-14, started higher, and the fight cost was measured at
# 72% even from there).
TARGET_LEVEL = 20
HEAL_BELOW_FRACTION = 0.5
MAX_WALK_STEPS = 8000
DIRECTIONS = ("up", "down", "left", "right")


def grind(pyboy, wild_model, trainer_model, target_level=TARGET_LEVEL, seed=0):
    rng = random.Random(seed)
    battles = 0

    for _ in range(MAX_WALK_STEPS):
        if get_party_level(pyboy) >= target_level:
            return battles

        if get_party_hp_fraction(pyboy) < HEAL_BELOW_FRACTION:
            print(f"  HP {get_party_hp(pyboy)}/{get_party_max_hp(pyboy)} -- healing")
            if not heal_at_pewter_center(pyboy):
                print("  heal failed -- stopping")
                return battles
            walk_to_map(pyboy, GRIND_MAP_ID, max_tiles=3000)
            continue

        if get_player_position(pyboy)["map_id"] != GRIND_MAP_ID:
            if not walk_to_map(pyboy, GRIND_MAP_ID, max_tiles=3000):
                print("  could not get back to Route 2 -- stopping")
                return battles
            continue

        walk_tile(pyboy, rng.choice(DIRECTIONS), verbose=False)
        run_frames(pyboy, 5)

        if not is_in_battle(pyboy):
            continue

        # A trainer encountered incidentally gets the trainer policy, a
        # wild encounter the wild one -- the forest grind's own version
        # of this loop always used one model regardless, a harmless
        # simplification there (Route 1 has no trainers) but a real
        # mismatch here, where Route 2 does have some and using the
        # wrong policy against one would waste HP for nothing.
        is_trainer = get_battle_type(pyboy) == 2
        model = trainer_model if is_trainer else wild_model
        fight_current_battle(pyboy, model)
        wait_for_free_movement(pyboy)
        battles += 1

        if battles % 3 == 0:
            print(f"  {battles} battles: Lv{get_party_level(pyboy)} "
                  f"HP{get_party_hp(pyboy)}/{get_party_max_hp(pyboy)}")

    return battles


def main():
    wild_model = DQN.load(str(WILD_MODEL_PATH))
    trainer_model = DQN.load(str(TRAINER_MODEL_PATH))

    pyboy = create_emulator()
    load_state(pyboy, START_STATE_PATH)
    run_frames(pyboy, 30)

    print(f"Starting: Lv{get_party_level(pyboy)} "
          f"HP{get_party_hp(pyboy)}/{get_party_max_hp(pyboy)}, "
          f"target Lv{TARGET_LEVEL}")

    # One-time move to the grinding ground and back at the end, rather
    # than re-deriving the route on every heal trip -- heal_at_pewter_
    # center and the per-step check inside grind() already handle
    # getting back to GRIND_MAP_ID after any interruption.
    walk_to_map(pyboy, PEWTER_CITY_MAP_ID, max_tiles=2000)
    walk_to_map(pyboy, GRIND_MAP_ID, max_tiles=2000)

    battles = grind(pyboy, wild_model, trainer_model)

    reached = get_party_level(pyboy) >= TARGET_LEVEL
    print(f"\n{battles} battles fought. Final: Lv{get_party_level(pyboy)} "
          f"HP{get_party_hp(pyboy)}/{get_party_max_hp(pyboy)} "
          f"(target {'reached' if reached else 'NOT reached'})")

    walk_to_map(pyboy, PEWTER_CITY_MAP_ID, max_tiles=2000)
    walk_to_map(pyboy, ROUTE_3_MAP_ID, max_tiles=2000)
    save_state(pyboy, LEVELED_STATE_PATH)
    print(f"Saved {LEVELED_STATE_PATH}")
    pyboy.stop()

    return 0 if reached else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
