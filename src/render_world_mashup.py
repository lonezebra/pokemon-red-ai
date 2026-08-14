import argparse
import json

from PIL import Image

from core.config import PROJECT_ROOT, SCREENSHOT_DIR
from core.screen import save_gif
from render_route1_mashup import (
    directions_for_run,
    load_tinted_sprites,
)
from build_player_sprite import PLAYER_SPRITE_BOUNDS

"""
The animated version of build_world_atlas.py's stitch: replay whole-game
rollouts across the Route 1 <-> Viridian City boundary as one continuous
mashup, on the shared canvas that script already built and verified.

Reuses render_route1_mashup's sprite loading and direction logic rather than
duplicating it -- the only genuinely new thing here is that a step's
direction and pixel position are computed in WORLD space (each map's local
(x, y) plus its verified offset from world_atlas_meta.json) instead of one
map's local space, which is what lets a run's sprite walk smoothly across
the map seam instead of jumping.

    cd src && ../.venv/bin/python3 render_world_mashup.py
"""

ROLLOUT_DIR = PROJECT_ROOT / "models" / "whole_game_rollouts"


def load_world_meta():
    return json.loads((SCREENSHOT_DIR / "world_atlas_meta.json").read_text())


def world_segments(path, map_ids):
    """Contiguous runs of a [map_id, x, y] path while map_id stays inside
    the stitched cluster -- crossing out and back later is a new segment,
    same reasoning as render_whole_game_runs.segments_by_map."""
    segments = []
    current = []
    for map_id, x, y in path:
        if map_id in map_ids:
            current.append((map_id, x, y))
        elif current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", default=None,
                        help="rollout JSON (default: newest)")
    parser.add_argument("--duration-ms", type=int, default=40)
    parser.add_argument("--gif-name", default="world_mashup.gif")
    parser.add_argument("--stride", type=int, default=1,
                        help="keep every Nth animation frame -- the full "
                             "per-step render is ~13MB, too large to embed "
                             "anywhere; this is the lever for that, not "
                             "image quality")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="downscale each finished frame by this factor "
                             "(e.g. 2 = half width/height) -- independent "
                             "of --stride, which drops frames instead")
    args = parser.parse_args()

    if args.rollouts:
        rollout_path = PROJECT_ROOT / args.rollouts
    else:
        candidates = sorted(ROLLOUT_DIR.glob("*_rollouts.json"),
                            key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No rollouts in {ROLLOUT_DIR}.")
        rollout_path = candidates[-1]

    data = json.loads(rollout_path.read_text())
    run_label = rollout_path.stem
    world = load_world_meta()
    map_ids = {int(m) for m in world["maps"]}

    tile, pad = world["tile"], world["pad"]
    sprite_center_x = (PLAYER_SPRITE_BOUNDS[0] + PLAYER_SPRITE_BOUNDS[2]) // 2
    sprite_center_y = (PLAYER_SPRITE_BOUNDS[1] + PLAYER_SPRITE_BOUNDS[3]) // 2

    def world_tile(map_id, x, y):
        m = world["maps"][str(map_id)]
        ox, oy = m["offset"]
        return ox + x, oy + y

    def to_pixel(map_id, x, y):
        wx, wy = world_tile(map_id, x, y)
        px = pad + (wx - world["world_min_x"]) * tile + sprite_center_x
        py = pad + (wy - world["world_min_y"]) * tile + sprite_center_y
        return px, py

    runs = []
    for episode in data["episodes"]:
        segments = world_segments(episode.get("path", []), map_ids)
        for index, segment in enumerate(segments):
            if len(segment) <= 1:
                continue
            world_positions = [world_tile(m, x, y) for m, x, y in segment]
            runs.append({
                "positions": [(m, x, y) for m, x, y in segment],
                "world_positions": world_positions,
                # Same convention render_whole_game_runs.py uses: no literal
                # goal tile exists in a whole-game run, so "reached the
                # goal" is reused for "left the cluster under its own
                # power" -- true for every segment but the last one in its
                # episode, since world_segments only starts a new segment
                # when the map leaves {Route 1, Viridian}.
                "reached_goal": index < len(segments) - 1,
            })

    if not runs:
        print("No runs crossed the stitched cluster in this rollout file.")
        return

    print(f"{len(runs)} run-segments touch the stitched world "
          f"(Route 1 / Viridian City)")

    base = Image.open(SCREENSHOT_DIR / "world_atlas.png").convert("RGB")
    sprites, sprite_w, sprite_h = load_tinted_sprites()

    for run in runs:
        run["directions"] = directions_for_run(run["world_positions"])

    max_len = max(len(run["positions"]) for run in runs)
    frame_times = range(0, max_len, args.stride)
    scaled_size = None
    if args.scale != 1.0:
        scaled_size = (round(base.width / args.scale), round(base.height / args.scale))
    frames = []

    for t in frame_times:
        frame = base.copy()

        for run in runs:
            positions = run["positions"]
            idx = min(t, len(positions) - 1)
            map_id, x, y = positions[idx]
            direction = run["directions"][idx]
            outcome = "in_progress" if t < len(positions) - 1 else (
                "success" if run["reached_goal"] else "unfinished"
            )
            sprite = sprites[direction][outcome]

            center_x, center_y = to_pixel(map_id, x, y)
            frame.paste(
                sprite,
                (center_x - sprite_w // 2, center_y - sprite_h // 2),
                sprite,
            )

        if scaled_size:
            # NEAREST, not a smoothing filter: this is flat-color pixel
            # art, and any resampling that blends pixels (LANCZOS, BILINEAR)
            # turns sharp tile edges into gradients -- which multiplies the
            # number of distinct colors per frame and badly hurts GIF
            # compression. A first attempt at this used LANCZOS and produced
            # a 122MB file at a SMALLER pixel size than the unscaled 13MB
            # original. build_player_sprite.py's own upscaling uses the same
            # nearest-neighbor choice, for the same reason.
            frame = frame.resize(scaled_size, Image.NEAREST)

        frames.append(frame)
        if t % 50 == 0:
            print(f"Rendered frame {t}/{max_len}")

    successes = sum(1 for run in runs if run["reached_goal"])
    print(f"{successes}/{len(runs)} run-segments moved on under their own power")

    # Each kept frame now stands in for `stride` real steps, so its display
    # time scales up too -- otherwise dropping frames for file size would
    # also speed the whole animation up, which isn't the goal.
    save_gif(frames, f"mashups/{run_label}/{args.gif_name}",
              duration_ms=args.duration_ms * args.stride)


if __name__ == "__main__":
    main()
