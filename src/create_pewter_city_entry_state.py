import functools

from core.config import PROJECT_ROOT
from core.emulator import create_emulator, run_frames
from core.state import save_state
from core.memory import get_player_position
from core.pathfind import _restore, _step
from core.parallel_survey import parallel_survey_map
from survey_viridian_forest import build_worker_handle_battle, build_worker_heal_if_needed

# Rebuilds the checkpoint chain from the levelled Lv10 party through
# Viridian Forest, a small connector room, a previously-unmapped
# northern stretch of Route 2, and into Pewter City itself (confirmed
# visually -- the panorama shows labelled GYM and MART buildings, not
# just an unverified map ID).
#
# This exists because the checkpoints from the session that originally
# found this route did not survive a container restart, while the code
# and the trained model did. Re-deriving them from scratch is cheap now
# that core/parallel_survey.py exists: the original single-process walk
# to this same point took hours; the parallel version gets there in
# minutes, using this machine's full core count to fight the same
# trainers and explore the same maze simultaneously across processes
# instead of one battle at a time.
#
# At each leg: parallel-survey the current map exhaustively (not just a
# blind rush toward the known exit -- this also produces real, current-
# code-verified panorama/exit data as a side effect), pick out the exit
# that leads toward Pewter City specifically, restore that exact tile's
# already-known snapshot and step across, and save a checkpoint before
# moving to the next leg. That checkpointing is deliberate: another
# interruption partway through only costs whichever leg was in progress,
# not the whole chain.

CHAIN = [
    # (state_name to start from, its map ID, the next map to find an exit
    #  to, the checkpoint filename to save once there)
    ("leveled", 51, 47, "forest_entry47"),
    ("forest_entry47", 47, 13, "map47_entry13"),
    ("map47_entry13", 13, 2, "pewter_city_entry"),
]

MAX_TILES = 8000


def find_exit_to(exits, target_map):
    for (tile, direction), dest in exits.items():
        if dest[0] == target_map:
            return tile, direction, dest
    return None


def cross(state_name, start_map, target_map, checkpoint_name):
    print(f"=== Surveying map {start_map} (from {state_name}.state), "
          f"looking for an exit to map {target_map} ===")

    state_path = PROJECT_ROOT / "saves" / f"{state_name}.state"
    tiles, exits, complete, _frames, map_id, states = parallel_survey_map(
        state_path, max_tiles=MAX_TILES,
        build_handle_battle=build_worker_handle_battle,
        build_heal_if_needed=functools.partial(build_worker_heal_if_needed, target_map=start_map),
        capture_frames=False, num_workers=4,
    )
    print(f"map {map_id}: {len(tiles)} tiles, complete={complete}, {len(exits)} exit tiles")

    found = find_exit_to(exits, target_map)
    if found is None:
        print(f"No exit to map {target_map} found from map {start_map} -- stopping.")
        return None

    tile, direction, dest = found
    print(f"Using exit {tile} {direction} -> map {dest[0]} {dest[1:]}")

    pyboy = create_emulator()
    _restore(pyboy, states[tile])
    _step(pyboy, direction, handle_battle=build_worker_handle_battle())
    run_frames(pyboy, 10)

    after = get_player_position(pyboy)
    print(f"crossed to: {after}")

    checkpoint_path = PROJECT_ROOT / "saves" / f"{checkpoint_name}.state"
    save_state(pyboy, checkpoint_path)
    pyboy.stop()
    print(f"Saved {checkpoint_path}")

    return checkpoint_name, after["map_id"]


def main():
    for state_name, start_map, target_map, checkpoint_name in CHAIN:
        # Skip a leg whose checkpoint already exists -- lets this be
        # re-run to resume after an interruption without redoing
        # already-completed legs.
        checkpoint_path = PROJECT_ROOT / "saves" / f"{checkpoint_name}.state"
        if checkpoint_path.exists():
            print(f"{checkpoint_path} already exists, skipping this leg.")
            continue

        result = cross(state_name, start_map, target_map, checkpoint_name)
        if result is None:
            print("Chain stopped early.")
            return

    print()
    print("Reached Pewter City. Checkpoint saved as pewter_city_entry.state")


if __name__ == "__main__":
    main()
