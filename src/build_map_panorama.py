import json
import sys

from core.emulator import create_emulator, run_frames
from core.state import load_state
from core.config import PROJECT_ROOT, SCREENSHOT_DIR
from core.pathfind import survey_map
from core.parallel_survey import parallel_survey_map, NUM_WORKERS
from core.panorama import stitch_panorama
from core.memory import get_player_position

# Builds a panorama of whatever map a save state starts on, by walking
# every reachable tile and capturing the screen at each one.
#
# This supersedes build_route1_map.py's approach for new maps. That one
# walks a biased random route and stitches whatever it happens to cross,
# which is fine for a corridor like Route 1 but leaves holes anywhere
# mazier -- and Viridian Forest is a real maze, filling only 45% of its
# bounding box. Driving the capture from pathfind.survey_map instead
# means the flood-fill decides where to go, so coverage is exactly the
# reachable set: complete by construction, and the same pass reports the
# map's exits for free.
#
# Usage:
#   python src/build_map_panorama.py <save_state_name> <output_prefix>
# e.g.
#   python src/build_map_panorama.py viridian_forest_entry forest

MAX_TILES = 5000  # bumped from 2500 -- the maps past Viridian Forest are
# turning out bigger than anything measured before it


def build(state_name, output_prefix, build_handle_battle=None, build_heal_if_needed=None,
          parallel=True, num_workers=NUM_WORKERS, max_tiles=MAX_TILES):
    """
    `build_handle_battle`/`build_heal_if_needed`, if given, are zero-arg
    top-level factory functions -- `build_handle_battle()` returns a
    handle_battle callable, `build_heal_if_needed(handle_battle)` returns
    a heal_if_needed callable. Needed for any map (Viridian Forest and
    beyond) where a trainer occupies a tile the survey would otherwise
    read as a permanent wall, and where fighting through more than one or
    two of them means managing HP between fights too. They have to be
    factories rather than already-built instances so the parallel path
    can hand each of its worker processes one built independently (see
    core/parallel_survey.py for why); the sequential path just calls them
    once itself.

    `parallel=True` (the default) uses core.parallel_survey to split the
    flood fill across `num_workers` processes instead of walking it one
    tile at a time in this one -- this machine's full core count should
    be the default for any survey, the same as it already is for
    training, unless there's a specific reason to watch one process at a
    time (debugging a new handle_battle/heal_if_needed, for instance).
    """
    state_path = PROJECT_ROOT / "saves" / f"{state_name}.state"

    if parallel:
        tiles, exits, complete, frames, map_id, _states, edges = parallel_survey_map(
            state_path, max_tiles=max_tiles,
            build_handle_battle=build_handle_battle,
            build_heal_if_needed=build_heal_if_needed,
            capture_frames=True, num_workers=num_workers,
        )
    else:
        pyboy = create_emulator()
        load_state(pyboy, state_path)
        run_frames(pyboy, 30)

        start = get_player_position(pyboy)
        map_id = start["map_id"]
        print(f"Surveying map {map_id} from {(start['x'], start['y'])}...")

        frames = []

        def capture(emulator, x, y):
            frames.append((x, y, emulator.screen.image.convert("RGB")))

        handle_battle = build_handle_battle() if build_handle_battle else None
        heal_if_needed = build_heal_if_needed(handle_battle) if build_heal_if_needed else None

        tiles, exits, complete = survey_map(
            pyboy, max_tiles=max_tiles, on_visit=capture,
            handle_battle=handle_battle, heal_if_needed=heal_if_needed,
        )
        pyboy.stop()
        edges = None  # survey_map doesn't track the adjacency graph the parallel path does

    print(f"Captured {len(frames)} tiles (survey complete={complete})")
    if not complete:
        print("Warning: hit the tile cap, so the panorama may be missing regions.")

    image, meta = stitch_panorama(frames)
    meta["map_id"] = map_id
    # The flood fill already knows every walkable tile, so record it
    # rather than throw it away -- a navigation env for this map gets its
    # reachable set (and so a sanity check on any recorded rollout)
    # without paying for a second survey.
    meta["tiles"] = sorted([x, y] for x, y in tiles)
    meta["exits"] = [
        {"from": list(tile), "direction": direction, "to_map": dest[0],
         "to": [dest[1], dest[2]]}
        for (tile, direction), dest in sorted(exits.items())
    ]
    if edges is not None:
        # The real walkable-adjacency graph, not tiles alone -- lets a
        # navigation env's reward shaping compute an exact shortest-path
        # distance to a goal tile instead of guessing from raw geometry,
        # which would get a one-way ledge wrong in one direction.
        meta["edges"] = [
            {"from": list(tile), "direction": direction, "to": list(dest)}
            for (tile, direction), dest in sorted(edges.items())
        ]

    SCREENSHOT_DIR.mkdir(exist_ok=True)
    image_path = SCREENSHOT_DIR / f"{output_prefix}_map.png"
    meta_path = SCREENSHOT_DIR / f"{output_prefix}_map_meta.json"
    image.save(image_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    print(f"Saved {image_path} ({image.size[0]}x{image.size[1]})")
    print(f"Saved {meta_path}")
    for exit_info in meta["exits"]:
        print(f"  exit {tuple(exit_info['from'])} {exit_info['direction']:5s} "
              f"-> map {exit_info['to_map']}")


def main():
    if len(sys.argv) != 3:
        print(__doc__ or "usage: build_map_panorama.py <save_state_name> <output_prefix>")
        print("example: build_map_panorama.py viridian_forest_entry forest")
        return
    build(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
