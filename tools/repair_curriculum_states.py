"""
Verify every curriculum start state restores to a controllable player,
and re-capture the ones that don't.

    .venv/bin/python3 tools/repair_curriculum_states.py

How a start state goes bad: build_curriculum_states.py saves on arrival
at each checkpoint tile, and if arriving at that tile crossed a
trainer's line of sight, the save can land on the exact frames of the
sighting cutscene -- exclamation bubble up, player control locked until
the trainer walks over. The walk itself recovers (its next _step
resolves the battle and moves on), but every future restore of that
file re-enters the frozen cutscene, and an env stepping direction
buttons there gets moved=False forever: episodes that start paralyzed
and time out. Found live as stage d<=15 refusing to master while its
start tile's Q-row sat flat at the all-failures value.

Verification is behavioral, not heuristic: restore the state, try all
four directions, and require at least one to actually move (probed from
a snapshot that is restored afterward, so probing can't corrupt what
gets saved). A frozen state fails all four by definition of the freeze.

Repair is a re-walk, not an in-place nudge: restore the nearest deeper
checkpoint that verified clean, walk the survey graph's shortest path
to the broken tile with the trainer-battle DQN handling any fight
triggered on the way -- the sighting then resolves properly mid-walk,
exactly as it did during the original capture -- and save on arrival,
this time with the trainer already beaten so the freeze cannot recur.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from stable_baselines3 import DQN

from build_curriculum_states import load_forward_graph, next_move_toward_goal
from core.battle_runner import fight_current_battle
from core.config import PROJECT_ROOT
from core.controls import wait_for_free_movement
from core.emulator import create_emulator, run_frames
from core.memory import get_player_position
from core.pathfind import DIRECTIONS, _restore, _snapshot, _step
from core.state import load_state, save_state

CURRICULUM_DIR = PROJECT_ROOT / "saves" / "forest_curriculum"
TRAINER_MODEL_PATH = PROJECT_ROOT / "models" / "trainer_battle_dqn.zip"
MAX_REPAIR_MOVES = 200


def state_files():
    """(distance, tile, path) for every curriculum state, nearest first."""
    out = []
    for path in sorted(CURRICULUM_DIR.glob("d*.state")):
        stem = path.stem  # dNNN_x_y
        parts = stem.split("_")
        out.append((int(parts[0][1:]), (int(parts[1]), int(parts[2])), path))
    return out


def is_mobile(pyboy):
    """
    Can the restored player move WITHOUT a battle being fought for them?
    Probed with handle_battle=None deliberately: a sighting-frozen state
    can technically recover if a handler fights the spotting trainer, but
    that recovery re-fights the battle at the start of every single
    episode -- ~15 wasted no-move steps whose self-loop updates flatten
    the tile's whole Q-row toward the all-failures value, which is
    exactly the damage this tool exists to prevent. A handler-equipped
    probe therefore reports the broken state as fine. Handler-free, the
    freeze fails all four directions and is caught.

    Probed from a snapshot restored afterward, so probing (including any
    battle a probe wedges into) never contaminates what gets saved.
    """
    snapshot = _snapshot(pyboy)
    mobile = False
    for direction in DIRECTIONS:
        _restore(pyboy, snapshot)
        if _step(pyboy, direction):
            mobile = True
            break
    _restore(pyboy, snapshot)
    return mobile


def repair(pyboy, donor_path, target_tile, handle_battle, graph):
    """Walk from the donor checkpoint to the broken tile and return True
    on arrival with the player verified mobile."""
    load_state(pyboy, donor_path)
    run_frames(pyboy, 30)

    for _ in range(MAX_REPAIR_MOVES):
        position = get_player_position(pyboy)
        tile = (position["x"], position["y"])
        if tile == target_tile:
            if is_mobile(pyboy):
                return True
            # Frozen on arrival: the very sighting that poisoned the
            # original capture is pending right now -- walking onto this
            # tile is what triggers it, so no en-route battle ever
            # resolved it. A handler-equipped step runs the engage probe
            # (A presses advance the sighting dialogue into the battle,
            # the handler wins it), after which the player may have been
            # moved; the loop then walks back and re-checks, this time
            # with the trainer beaten and nothing left to freeze on.
            _step(pyboy, "up", handle_battle=handle_battle)
            continue
        direction = next_move_toward_goal(graph, tile, target=target_tile)
        if direction is None:
            return False
        _step(pyboy, direction, handle_battle=handle_battle)
    return False


def main():
    states = state_files()
    if not states:
        print(f"no states in {CURRICULUM_DIR}")
        return 1

    graph = load_forward_graph()
    model = DQN.load(str(TRAINER_MODEL_PATH))

    def handle_battle(emulator):
        fight_current_battle(emulator, model)
        wait_for_free_movement(emulator)

    pyboy = create_emulator()

    verified = {}   # distance -> path, for states that restore mobile
    broken = []
    for distance, tile, path in states:
        load_state(pyboy, path)
        run_frames(pyboy, 60)
        if is_mobile(pyboy):
            verified[distance] = path
            print(f"  ok      d={distance:3d} {path.name}")
        else:
            broken.append((distance, tile, path))
            print(f"  FROZEN  d={distance:3d} {path.name}")

    repaired = failed = 0
    for distance, tile, path in broken:
        donors = sorted(d for d in verified if d > distance)
        if not donors:
            print(f"  cannot repair {path.name}: no clean deeper checkpoint "
                  f"-- re-run build_curriculum_states.py for it")
            failed += 1
            continue
        donor = verified[donors[0]]
        print(f"  repairing {path.name} by re-walk from {donor.name} ...")
        if repair(pyboy, donor, tile, handle_battle, graph):
            save_state(pyboy, path)
            verified[distance] = path
            repaired += 1
            print(f"  repaired d={distance:3d} {path.name}")
        else:
            failed += 1
            print(f"  FAILED to repair {path.name}")

    pyboy.stop()
    print(f"\n{len(states)} states: {len(states) - len(broken)} clean, "
          f"{repaired} repaired, {failed} unrepairable")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
