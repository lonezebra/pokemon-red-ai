"""
Walk up to Brock, capture the pre-battle state, then fight him once.

    cd src && ../.venv/bin/python3 ../tools/capture_brock_battle_state.py

Two outputs, in the order create_trainer_battle_states.py established:

  saves/brock_battle.state  -- standing at Brock's platform, battle NOT yet
      started. Captured before the first A press, so the eventual Brock
      battle environment can reset here and own the whole fight, exactly
      as the forest trainer states do.

  The fight itself, run once with the trainer DQN, purely as scoping
      data: the gym survey beat the wandering Jr Trainer everywhere it met
      him but never engaged Brock at all, so whether the current Lv12
      Squirtle (Bubble hits his Rock/Ground line at 4x) actually wins is
      still an assumption until a real fight settles it. Win or lose, the
      survey's snapshot discipline doesn't apply here -- a loss blacks out
      to the Pokemon Center -- which is fine, since this script owns its
      own emulator and saves the battle state first.

Also answers why the survey never engaged him: this walks the same
corridor the survey mapped and reports what actually blocks progress in
front of the platform, step by step.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from stable_baselines3 import DQN

from core.emulator import create_emulator, run_frames
from core.state import load_state, save_state
from core.config import PROJECT_ROOT, SCREENSHOT_DIR
from core.pathfind import walk_to, _try_engage_trainer
from core.controls import wait_for_free_movement
from core.battle_runner import fight_current_battle
from core.memory import (
    get_player_position,
    get_party_hp,
    get_party_max_hp,
    get_party_level,
    is_in_battle,
    get_enemy_mon_species,
    get_enemy_mon_level,
)

# Brock's tile, read off the survey meta rather than eyeballed: row y=1
# is walkable at every x except 4, so (4,1) is his body, approached from
# (4,2) facing up. The first attempt used "y <= 4" as the predicate and
# walk_to stopped at the top-LEFT corner -- the first matching tile its
# BFS happened to reach -- and pressed A into a rock.
BROCK_TILE = (4, 1)
APPROACH_TILE = (4, 2)
# Brock's pre-battle speech runs several boxes longer than a forest Bug
# Catcher's one-liner; the default 12 presses can run out mid-speech.
ENGAGE_PRESSES = 30
CHECKPOINT = PROJECT_ROOT / "saves" / "brock_battle.state"
MODEL_PATH = PROJECT_ROOT / "models" / "trainer_battle_dqn.zip"


def main():
    pyboy = create_emulator()
    load_state(pyboy, PROJECT_ROOT / "saves" / "pewter_gym_entry.state")
    run_frames(pyboy, 30)

    print(f"Walking to {APPROACH_TILE}, directly south of Brock at {BROCK_TILE}...")
    if not walk_to(pyboy, lambda p: (p["x"], p["y"]) == APPROACH_TILE, max_tiles=200):
        print("Could not reach the approach tile.")
        pyboy.stop()
        return 1
    position = get_player_position(pyboy)
    print(f"Standing at: {position}")

    pyboy.screen.image.convert("RGB").save(SCREENSHOT_DIR / "brock_approach.png")
    print("Saved screenshots/brock_approach.png")

    # Save BEFORE any A press, so the battle env can own the whole fight.
    save_state(pyboy, CHECKPOINT)
    print(f"Saved {CHECKPOINT} (pre-battle)")

    print("Engaging (pressing A into whatever blocks the way up)...")
    started = _try_engage_trainer(pyboy, max_presses=ENGAGE_PRESSES)
    if not started:
        print("No trainer battle started from here -- Brock's tile may not "
              "be adjacent; see the screenshot for what is actually ahead.")
        pyboy.stop()
        return 1

    # Enemy identity read after engage; if this still shows nonsense the
    # diagnostic settle bug reproduces here too.
    print(f"Battle started: enemy species {get_enemy_mon_species(pyboy)} "
          f"Lv{get_enemy_mon_level(pyboy)}")
    print(f"Our side: Lv{get_party_level(pyboy)} "
          f"{get_party_hp(pyboy)}/{get_party_max_hp(pyboy)} HP")

    model = DQN.load(str(MODEL_PATH))
    fight_current_battle(pyboy, model)
    wait_for_free_movement(pyboy)

    after = get_player_position(pyboy)
    hp = get_party_hp(pyboy)
    print(f"After the fight: position {after}, HP {hp}/{get_party_max_hp(pyboy)}")
    if after["map_id"] == 54 and hp > 0:
        print("WON: still standing in the Gym.")
        pyboy.screen.image.convert("RGB").save(SCREENSHOT_DIR / "brock_beaten.png")
        print("Saved screenshots/brock_beaten.png")
    else:
        print("LOST: blacked out (or left the Gym) -- the Brock env needs "
              "more than the current party/policy.")

    pyboy.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
