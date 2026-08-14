import os

os.environ.setdefault("POKEMON_AI_WINDOW_MODE", "null")

import argparse  # noqa: E402
import json  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from build_player_sprite import PLAYER_SPRITE_BOUNDS  # noqa: E402
from core.atomic_io import write_json_atomic  # noqa: E402
from core.config import PROJECT_ROOT, SCREENSHOT_DIR  # noqa: E402
from render_route1_mashup import (  # noqa: E402
    main as render_mashup,
    map_image_path,
    map_meta_path,
)

"""
Turn whole-game rollouts into the mashup GIFs and exploration heatmaps this
project already renders for its single-route agents.

The renderer itself is not rewritten here. render_route1_mashup.main() is
already generic despite its name -- render_forest_mashup.py reuses it
verbatim for a completely different map -- so this script's real job is
translation: a whole-game rollout wanders across many maps, and every
existing visualisation in this project is built around exactly one.

That mismatch is structural, not an oversight. Each panorama has its own
local origin and there is no world-space layout anywhere in the repo (the
survey `exits` are a warp graph -- they say map 51 connects to map 47, not
where 47 sits relative to 51). So rather than invent a global atlas, this
splits each run into per-map segments and renders each map separately, which
reuses the existing pipeline exactly as-is and stays honest about what is
actually known.

    cd src && ../.venv/bin/python3 render_whole_game_runs.py
    cd src && ../.venv/bin/python3 render_whole_game_runs.py --heatmap-only
"""

ROLLOUT_DIR = PROJECT_ROOT / "models" / "whole_game_rollouts"
MASHUP_DIR = SCREENSHOT_DIR / "mashups"

# Panorama prefixes for the maps this project has actually surveyed. Anything
# else the agent reaches is reported rather than silently dropped -- a
# whole-game agent walking somewhere with no map imagery is expected, and
# knowing which map to survey next is useful output in itself.
MAP_PREFIXES = {
    12: "route1",
    51: "forest",
    2: "map2_badged",
    54: "map54",
    14: "route3",
    # The opening maps, surveyed once the first whole-game rollouts showed
    # where the agent actually spends its time -- Oak's Lab alone took 17,076
    # of the steps in one five-episode batch, and none of it was drawable.
    38: "map38",  # player's bedroom, where every run starts
    0: "map0",    # Pallet Town
    40: "map40",  # Oak's Lab
    37: "map37",  # house, downstairs
    1: "map1",    # Viridian City -- confirmed by its own exits (south to
                   # Route 1, west to Route 22), not just its ID number
}

# Visit counts colour-mapped by hand, in numpy and PIL. matplotlib is not a
# dependency of this project and every image it produces is built this way
# (see core/panorama.py); adding a plotting library to draw one heatmap would
# be a real change to requirements.txt for very little.
HEATMAP_COLD = np.array([40, 60, 140], dtype=np.float32)
HEATMAP_HOT = np.array([250, 220, 60], dtype=np.float32)
HEATMAP_ALPHA = 0.65


def segments_by_map(path):
    """
    Split one run's [map_id, x, y] path into contiguous per-map segments.

    Contiguous rather than grouped: walking into Viridian Forest, back out to
    Route 2, and in again is three separate visits, and flattening them into
    one list would draw a sprite teleporting across the map when the agent
    actually left and returned.
    """
    segments = []
    current_map = None
    current = []

    for map_id, x, y in path:
        if map_id != current_map:
            if current:
                segments.append((current_map, current))
            current_map = map_id
            current = []
        current.append([x, y])

    if current:
        segments.append((current_map, current))

    return segments


def write_map_rollouts(run_label, map_id, runs, max_steps):
    """Write one map's segments in the schema render_route1_mashup expects:
    {"max_steps": N, "runs": [{"positions": [[x, y], ...],
    "reached_goal": bool}]}."""
    out_dir = MASHUP_DIR / run_label
    out_dir.mkdir(parents=True, exist_ok=True)

    name = f"whole_game_map{map_id}_rollouts.json"
    write_json_atomic(out_dir / name, {"max_steps": max_steps, "runs": runs})
    return name


def render_heatmap(map_id, prefix, visit_counts, run_label):
    """
    Paint visit frequency over the map's panorama.

    The mashup GIF shows where a run went; this shows where the policy
    *spends its time*, which is the view that exposes the failure this
    project has hit repeatedly -- a policy that looks busy while circling a
    handful of tiles. Route 1's revisit loop was exactly that: 777 of 801
    steps re-treading 24 tiles.
    """
    image_path = map_image_path(prefix)
    meta_path = map_meta_path(prefix)
    if not image_path.exists() or not meta_path.exists():
        return None

    meta = json.loads(meta_path.read_text())
    base = Image.open(image_path).convert("RGB")
    overlay = np.array(base, dtype=np.float32)

    tile, pad = meta["tile"], meta["pad"]
    min_x, min_y = meta["min_x"], meta["min_y"]

    busiest = max(visit_counts.values())

    # World tile (x, y) does NOT land at the corner of its pasted frame --
    # it lands wherever the player's own sprite renders inside that frame,
    # because the camera holds the player at a fixed screen position. This is
    # the same offset render_route1_mashup.build_canvas applies, and the long
    # comment there explains why it is not tile//2.
    #
    # Worth stating plainly because the first version of this heatmap
    # skipped the offset entirely and the result was visibly wrong: every
    # patch sat up and to the left of the room the agent actually walked in.
    sprite_center_x = (PLAYER_SPRITE_BOUNDS[0] + PLAYER_SPRITE_BOUNDS[2]) // 2
    sprite_center_y = (PLAYER_SPRITE_BOUNDS[1] + PLAYER_SPRITE_BOUNDS[3]) // 2

    for (x, y), count in visit_counts.items():
        # Centre a tile-sized patch on the tile, rather than hanging it off
        # the corner.
        px = pad + (x - min_x) * tile + sprite_center_x - tile // 2
        py = pad + (y - min_y) * tile + sprite_center_y - tile // 2
        if not (0 <= px < overlay.shape[1] - tile
                and 0 <= py < overlay.shape[0] - tile):
            continue

        # Square-rooted so that one heavily-farmed tile doesn't flatten every
        # other visited tile to the same cold colour.
        weight = (count / busiest) ** 0.5
        colour = HEATMAP_COLD + (HEATMAP_HOT - HEATMAP_COLD) * weight

        patch = overlay[py:py + tile, px:px + tile]
        overlay[py:py + tile, px:px + tile] = (
            patch * (1.0 - HEATMAP_ALPHA) + colour * HEATMAP_ALPHA
        )

    out_dir = MASHUP_DIR / run_label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"whole_game_map{map_id}_heatmap.png"
    Image.fromarray(overlay.astype(np.uint8)).save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", default=None,
                        help="rollout JSON (default: newest)")
    parser.add_argument("--heatmap-only", action="store_true")
    parser.add_argument("--duration-ms", type=int, default=60)
    args = parser.parse_args()

    if args.rollouts:
        rollout_path = PROJECT_ROOT / args.rollouts
    else:
        candidates = sorted(ROLLOUT_DIR.glob("*_rollouts.json"),
                            key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(
                f"No rollouts in {ROLLOUT_DIR}. Run watch_whole_game.py first."
            )
        rollout_path = candidates[-1]

    data = json.loads(rollout_path.read_text())
    run_label = rollout_path.stem
    print(f"Rendering {rollout_path.name} ({len(data['episodes'])} episodes)")

    per_map_runs = defaultdict(list)
    per_map_counts = defaultdict(lambda: defaultdict(int))

    for episode in data["episodes"]:
        segments = segments_by_map(episode.get("path", []))

        for index, (map_id, positions) in enumerate(segments):
            # A segment that never moves is one frame of a stationary sprite.
            # It still counts as time spent (so it lands in the heatmap) but
            # animating it adds nothing.
            if len(positions) > 1:
                per_map_runs[map_id].append({
                    "positions": positions,
                    # No goal tile exists in a whole-game run -- there is
                    # nothing to "reach" -- so the renderer's success flag is
                    # reused for the closest real equivalent: did the agent
                    # leave this map under its own power, or was it still
                    # here when the episode ran out? Green for left, red for
                    # stuck, which is exactly the distinction worth seeing.
                    #
                    # Any segment but the last one ended by moving on, since
                    # segments_by_map only starts a new one when the map
                    # changes.
                    "reached_goal": index < len(segments) - 1,
                })
            for x, y in positions:
                per_map_counts[map_id][(x, y)] += 1

    if not per_map_runs:
        print("No movement recorded in these rollouts -- nothing to draw.")
        return

    print()
    unmapped = []
    for map_id in sorted(per_map_counts, key=lambda m: -len(per_map_counts[m])):
        tiles = len(per_map_counts[map_id])
        steps = sum(per_map_counts[map_id].values())
        prefix = MAP_PREFIXES.get(map_id)

        if prefix is None or not map_image_path(prefix).exists():
            unmapped.append((map_id, tiles, steps))
            continue

        print(f"map {map_id:>3} ({prefix}): {tiles} tiles, {steps} steps")

        heatmap = render_heatmap(
            map_id, prefix, per_map_counts[map_id], run_label
        )
        if heatmap:
            print(f"  heatmap -> {heatmap.relative_to(PROJECT_ROOT)}")

        if args.heatmap_only or map_id not in per_map_runs:
            continue

        rollouts_name = write_map_rollouts(
            run_label, map_id, per_map_runs[map_id], data.get("max_steps", 0)
        )
        render_mashup(
            run_label=run_label,
            duration_ms=args.duration_ms,
            map_prefix=prefix,
            rollouts_name=rollouts_name,
            gif_name=f"whole_game_map{map_id}_mashup.gif",
        )

    if unmapped:
        print()
        print("Visited, but no panorama exists to draw them on:")
        for map_id, tiles, steps in unmapped:
            print(f"  map {map_id:>3}: {tiles} tiles, {steps} steps")
        print("  Build one with:  python src/build_map_panorama.py "
              "<save_state_name> <output_prefix>")
        print("  then add it to MAP_PREFIXES in this file.")


if __name__ == "__main__":
    main()
