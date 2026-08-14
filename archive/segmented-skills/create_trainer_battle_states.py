import io
from collections import deque

from core.emulator import create_emulator, run_frames
from core.state import load_state, save_state, TRAINER_BATTLE_STATE_DIR
from core.config import PROJECT_ROOT
from core.controls import walk_tile, press_button, attempt_run_from_wild_battle, advance_battle_dialogue
from core.memory import (
    get_player_position,
    is_in_battle,
    get_battle_type,
    is_battle_menu_open,
    get_enemy_mon_species,
    get_enemy_mon_level,
)

# Captures one save state per Viridian Forest trainer, at the first
# FIGHT/PKMN/ITEM/RUN menu of that battle -- the training checkpoints for
# envs/trainer_battle_env.py.
#
# Why these trainers matter: a survey of Viridian Forest found its only
# reachable exit leads back the way we came. The way onward is guarded by
# Bug Catchers, and since a trainer physically occupies its tile,
# pathfinding correctly reads one as a wall. Getting past means winning,
# because Gen 1 does not allow fleeing a trainer.
#
# Finding them is a flood-fill of the forest that, at every tile it
# cannot walk out of, presses A to see what is there. Two details were
# learned the hard way and are the reason this works:
#
#   - A trainer states their line *before* the battle starts. Checking
#     for a battle immediately after one A press found nothing at all;
#     the press has to be repeated until the battle actually begins.
#   - Standing in a trainer's line of sight leaves the game mid-trigger,
#     so all four directions from that tile look blocked and respond
#     identically. Without dedup that captured the same Bug Catcher eight
#     times (one tile pair, four directions each). Capturing at most one
#     battle per tile, and only from tiles well away from an
#     already-captured one, gets distinct trainers instead.

# Prefers the levelled checkpoint when it exists. Captured at Lv6 these
# battles were measured unwinnable at the project's usual bar -- an
# essentially optimal "always attack" policy topped out near 67%,
# because a Lv6 Squirtle's only damaging move is Tackle and multi-Pokemon
# Bug Catcher parties win on poison chip damage alone. create_leveled_
# state.py exists to fix that, so capture from its result if available.
LEVELED_STATE_PATH = PROJECT_ROOT / "saves" / "leveled.state"
FOREST_ENTRY_STATE_PATH = PROJECT_ROOT / "saves" / "viridian_forest_entry.state"


def _start_state_path():
    return LEVELED_STATE_PATH if LEVELED_STATE_PATH.exists() else FOREST_ENTRY_STATE_PATH
VIRIDIAN_FOREST_MAP_ID = 51

MAX_TILES = 2500
MIN_SEPARATION = 4       # Manhattan distance between distinct trainers
MAX_TALK_PRESSES = 12    # enough to get through a trainer's pre-battle line


def _snapshot(pyboy):
    buf = io.BytesIO()
    pyboy.save_state(buf)
    return buf.getvalue()


def _restore(pyboy, data):
    buf = io.BytesIO(data)
    buf.seek(0)
    pyboy.load_state(buf)
    run_frames(pyboy, 2)


def _try_start_trainer_battle(pyboy):
    """Talk to whatever is being faced; True if a trainer battle begins."""

    for _ in range(MAX_TALK_PRESSES):
        press_button(pyboy, "a", hold_frames=12, release_frames=26)
        run_frames(pyboy, 20)
        if is_in_battle(pyboy):
            break
    else:
        return False

    if get_battle_type(pyboy) != 2:
        attempt_run_from_wild_battle(pyboy)
        return False

    advance_battle_dialogue(pyboy)
    return is_battle_menu_open(pyboy)


def capture_trainer_battles():
    pyboy = create_emulator()
    load_state(pyboy, _start_state_path())
    run_frames(pyboy, 30)

    start = get_player_position(pyboy)
    start_key = (start["x"], start["y"])
    states = {start_key: _snapshot(pyboy)}
    tiles = {start_key}
    queue = deque([start_key])
    found = []

    def far_from_known(tile):
        return all(
            abs(tile[0] - known[0]) + abs(tile[1] - known[1]) >= MIN_SEPARATION
            for known, _, _ in found
        )

    while queue and len(tiles) < MAX_TILES:
        key = queue.popleft()
        captured_here = False

        for direction in ("up", "down", "left", "right"):
            if captured_here:
                break

            _restore(pyboy, states[key])
            moved = walk_tile(pyboy, direction, verbose=False)
            run_frames(pyboy, 6)

            if is_in_battle(pyboy):
                attempt_run_from_wild_battle(pyboy)
                continue

            position = get_player_position(pyboy)
            if position["map_id"] != VIRIDIAN_FOREST_MAP_ID:
                continue

            if moved:
                next_key = (position["x"], position["y"])
                if next_key not in tiles:
                    tiles.add(next_key)
                    states[next_key] = _snapshot(pyboy)
                    queue.append(next_key)
                continue

            if not far_from_known(key):
                continue
            if not _try_start_trainer_battle(pyboy):
                continue

            species = get_enemy_mon_species(pyboy)
            level = get_enemy_mon_level(pyboy)
            path = TRAINER_BATTLE_STATE_DIR / f"trainer_{key[0]}_{key[1]}.state"
            save_state(pyboy, path)
            found.append((key, species, level))
            captured_here = True
            print(f"Trainer at {key}: enemy species {species} Lv{level} -> {path.name}")

    pyboy.stop()
    return found


def main():
    TRAINER_BATTLE_STATE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Capturing from {_start_state_path().name}")
    for stale in TRAINER_BATTLE_STATE_DIR.glob("trainer_*.state"):
        stale.unlink()
    found = capture_trainer_battles()

    print()
    print(f"Captured {len(found)} distinct trainer battles:")
    for tile, species, level in found:
        print(f"  {tile}  enemy species {species}  Lv{level}")


if __name__ == "__main__":
    main()
