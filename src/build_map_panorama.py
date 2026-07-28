import json
import sys

from core.emulator import create_emulator, run_frames
from core.state import load_state
from core.config import PROJECT_ROOT, SCREENSHOT_DIR
from core.pathfind import survey_map
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

MAX_TILES = 2500


def build(state_name, output_prefix):
    pyboy = create_emulator()
    load_state(pyboy, PROJECT_ROOT / "saves" / f"{state_name}.state")
    run_frames(pyboy, 30)

    start = get_player_position(pyboy)
    print(f"Surveying map {start['map_id']} from {(start['x'], start['y'])}...")

    frames = []

    def capture(emulator, x, y):
        frames.append((x, y, emulator.screen.image.convert("RGB")))

    tiles, exits, complete = survey_map(pyboy, max_tiles=MAX_TILES, on_visit=capture)
    pyboy.stop()

    print(f"Captured {len(frames)} tiles (survey complete={complete})")
    if not complete:
        print("Warning: hit the tile cap, so the panorama may be missing regions.")

    image, meta = stitch_panorama(frames)
    meta["map_id"] = start["map_id"]
    meta["exits"] = [
        {"from": list(tile), "direction": direction, "to_map": dest[0],
         "to": [dest[1], dest[2]]}
        for (tile, direction), dest in sorted(exits.items())
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
