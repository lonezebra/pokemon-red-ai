import json
import random

import numpy as np
from PIL import Image

from core.emulator import create_emulator, run_frames
from core.state import load_state, ROUTE_1_ENTRY_STATE_PATH
from core.controls import walk_tile, attempt_run_from_wild_battle
from core.memory import get_player_position, is_in_battle
from core.config import SCREENSHOT_DIR
from rewards.route1_rewards import VIRIDIAN_CITY_MAP_ID

# One-off cartography tool: walks the actual route once, capturing the
# real screen at each tile, then stitches those captures into a single
# panorama image of Route 1 -- purely so later scripts (the run mashup)
# have a real map to draw agent positions on top of. This is scripted
# scaffolding for a visualization, not the learned navigation task
# itself (see README's "scripts are only allowed as scaffolding" rule).
#
# How the stitching works: the player's on-screen position always maps
# to the same world tile in a consistent way frame to frame (Gen 1's
# camera scrolls in fixed 16px/tile steps), so placing each captured
# frame at canvas pixel (x, y) * TILE (relative to some fixed origin)
# lines every frame up correctly, without needing to know exactly where
# on screen the camera centers the player. Taking the per-pixel median
# across all frames that overlap a given canvas pixel then cancels out
# the one truly moving thing between frames -- the player's own sprite
# -- leaving just the static terrain. Confirmed by inspection: a first
# attempt with plain overwrite-pasting left player-sprite ghosts
# scattered across the map; the median version doesn't.

TILE = 16
PAD = 24
STRIP_HEIGHT = 200  # bounds memory during the median stack, regardless of map size

MAP_IMAGE_PATH = SCREENSHOT_DIR / "route1_map.png"
MAP_META_PATH = SCREENSHOT_DIR / "route1_map_meta.json"

# Weighted toward "up" (the direction that makes progress toward
# Viridian City), with enough randomness in every direction that the
# walk doesn't get stuck oscillating in a 2-move cycle against a single
# obstacle the way a strict fixed-priority walker did during development.
DIRECTION_WEIGHTS = ["up"] * 5 + ["left"] * 2 + ["right"] * 2 + ["down"]

STUCK_WINDOW = 100
ESCAPE_BURST = 40


def scout_route1(max_steps=4000, seed=2):
    """
    Walk from the Route 1 entry state to Viridian City, capturing
    (x, y, screen image) every other step along the way.
    """

    pyboy = create_emulator()
    load_state(pyboy, ROUTE_1_ENTRY_STATE_PATH)
    run_frames(pyboy, 30)

    rng = random.Random(seed)
    frames = []

    pos = get_player_position(pyboy)
    frames.append((pos["x"], pos["y"], pyboy.screen.image.convert("RGB")))

    recent_positions = []

    for step in range(max_steps):
        walk_tile(pyboy, rng.choice(DIRECTION_WEIGHTS), verbose=False)
        run_frames(pyboy, 5)
        if is_in_battle(pyboy):
            attempt_run_from_wild_battle(pyboy)

        pos = get_player_position(pyboy)

        if step % 2 == 0:
            frames.append((pos["x"], pos["y"], pyboy.screen.image.convert("RGB")))

        if pos["map_id"] == VIRIDIAN_CITY_MAP_ID:
            print(f"Reached Viridian City at step {step}: {pos}")
            break

        recent_positions.append((pos["x"], pos["y"]))
        if len(recent_positions) > STUCK_WINDOW:
            recent_positions.pop(0)
            xs = [p[0] for p in recent_positions]
            ys = [p[1] for p in recent_positions]
            if max(xs) - min(xs) <= 1 and max(ys) - min(ys) <= 1:
                for _ in range(ESCAPE_BURST):
                    walk_tile(pyboy, rng.choice(["up", "down", "left", "right"]), verbose=False)
                    run_frames(pyboy, 3)
                recent_positions = []
    else:
        print(f"Warning: hit max_steps ({max_steps}) without reaching Viridian City.")

    pyboy.stop()

    # Drop the final Viridian City frame -- it's on a different map with
    # its own unrelated local x/y coordinates, which would corrupt the
    # Route 1 panorama's alignment.
    return frames[:-1]


def stitch_panorama(frames):
    frame_h, frame_w = np.array(frames[0][2]).shape[:2]

    xs = [f[0] for f in frames]
    ys = [f[1] for f in frames]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    canvas_w = (max_x - min_x) * TILE + frame_w + PAD * 2
    canvas_h = (max_y - min_y) * TILE + frame_h + PAD * 2

    result = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)

    for strip_top in range(0, canvas_h, STRIP_HEIGHT):
        strip_bottom = min(strip_top + STRIP_HEIGHT, canvas_h)
        strip_h = strip_bottom - strip_top

        relevant = []
        for x, y, img in frames:
            oy = PAD + (y - min_y) * TILE
            if oy < strip_bottom and oy + frame_h > strip_top:
                relevant.append((x, y, np.array(img)))

        if not relevant:
            continue

        stack = np.full((len(relevant), strip_h, canvas_w, 3), np.nan, dtype=np.float32)
        for i, (x, y, arr) in enumerate(relevant):
            ox = PAD + (x - min_x) * TILE
            oy = PAD + (y - min_y) * TILE
            src_top = max(0, strip_top - oy)
            src_bottom = min(frame_h, strip_bottom - oy)
            dst_top = oy + src_top - strip_top
            dst_bottom = oy + src_bottom - strip_top
            stack[i, dst_top:dst_bottom, ox:ox + frame_w, :] = arr[src_top:src_bottom, :, :]

        with np.errstate(all="ignore"):
            median = np.nanmedian(stack, axis=0)
        mask = ~np.isnan(median)
        result[strip_top:strip_bottom, :, :] = np.where(mask, median, 30.0).astype(np.uint8)

    image = Image.fromarray(result, mode="RGB")
    meta = {
        "tile": TILE,
        "pad": PAD,
        "frame_width": frame_w,
        "frame_height": frame_h,
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "width": image.size[0],
        "height": image.size[1],
    }
    return image, meta


def main():
    print("Scouting Route 1...")
    frames = scout_route1()
    print(f"Captured {len(frames)} frames, stitching panorama...")

    image, meta = stitch_panorama(frames)

    SCREENSHOT_DIR.mkdir(exist_ok=True)
    image.save(MAP_IMAGE_PATH)
    with open(MAP_META_PATH, "w") as f:
        json.dump(meta, f)

    print(f"Saved {MAP_IMAGE_PATH} ({image.size[0]}x{image.size[1]})")
    print(f"Saved {MAP_META_PATH}: {meta}")


if __name__ == "__main__":
    main()
