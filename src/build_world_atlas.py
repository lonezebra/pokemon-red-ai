import json
import sys

from core.atomic_io import write_json_atomic
from core.config import PROJECT_ROOT, SCREENSHOT_DIR

from PIL import Image

"""
Stitch two or more per-map panoramas into one shared world-space image, for
the specific case where that's actually geometrically valid: outdoor routes
that border each other on a continuous tile grid.

It is NOT valid in general. Checked by hand before this was written: a
building's stairs (map37/map38, the player's house) record FOUR different
departure tiles that all funnel to the SAME single arrival tile below --
computing an offset from each gives four different, contradictory answers,
because a staircase is a discrete many-to-one warp, not a rigid grid edge.
Route 1 <-> Viridian City is the opposite case, and this was checked the
same way before trusting it: Viridian's two adjacent doorway tiles, (20,35)
and (21,35), land on Route 1's (10,0) and (11,0) -- adjacent in, adjacent
out, and both give the identical offset. That agreement is the proof a
rigid placement is even valid here; it would not be for a door.

So this script only ever stitches maps connected by an *outdoor* exit,
verified this way, not anything reachable via `exits` alone.

    cd src && ../.venv/bin/python3 build_world_atlas.py
"""

DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

# Each entry: (map_id, panorama_prefix). The first is the anchor (world
# offset (0, 0)); every other map's offset is computed from an exit
# connecting it back to a map already placed. Add a map here only after
# checking its connecting exits agree with each other the way the module
# docstring describes -- an unchecked addition is exactly the mistake this
# script exists to avoid.
WORLD_MAPS = [
    (1, "map1"),    # Viridian City -- anchor
    (12, "route1"), # Route 1, via the verified doorway pair above
]


def load_meta(prefix):
    path = SCREENSHOT_DIR / f"{prefix}_map_meta.json"
    return json.loads(path.read_text())


def offset_from_exit(known_offset, from_xy, direction, to_xy):
    """World-space offset for the tile on the far side of one exit, given
    the near side's already-known offset. See the module docstring for why
    this is only trustworthy when multiple exits between the same two maps
    agree -- callers are expected to have checked that already."""
    dx, dy = DELTA[direction]
    ox, oy = known_offset
    return (
        ox + from_xy[0] + dx - to_xy[0],
        oy + from_xy[1] + dy - to_xy[1],
    )


def find_offset(metas, target_map_id, placed):
    """Search every placed map's exits for one leading to target_map_id."""
    for map_id, offset in list(placed.items()):
        meta = metas[map_id]
        for exit in meta.get("exits", []):
            if exit["to_map"] == target_map_id:
                return offset_from_exit(
                    offset, exit["from"], exit["direction"], exit["to"]
                )
    return None


def main():
    metas = {}
    prefixes = {}
    for map_id, prefix in WORLD_MAPS:
        metas[map_id] = load_meta(prefix)
        prefixes[map_id] = prefix

    anchor_id = WORLD_MAPS[0][0]
    placed = {anchor_id: (0, 0)}

    for map_id, _ in WORLD_MAPS[1:]:
        offset = find_offset(metas, map_id, placed)
        if offset is None:
            raise SystemExit(
                f"No exit found connecting map {map_id} to an already-"
                f"placed map -- add it to WORLD_MAPS only after the map "
                f"it borders is placed, and only after checking its exits "
                f"agree the way route1<->viridian's do."
            )
        placed[map_id] = offset
        print(f"map {map_id} ({prefixes[map_id]}): offset {offset}")

    # World-space bounding box across every placed map, in tiles, so the
    # canvas fits everything with no cropping.
    world_min_x = min(placed[m][0] + metas[m]["min_x"] for m in placed)
    world_min_y = min(placed[m][1] + metas[m]["min_y"] for m in placed)
    world_max_x = max(placed[m][0] + metas[m]["max_x"] for m in placed)
    world_max_y = max(placed[m][1] + metas[m]["max_y"] for m in placed)

    tile = metas[anchor_id]["tile"]
    pad = metas[anchor_id]["pad"]
    frame_w = metas[anchor_id]["frame_width"]
    frame_h = metas[anchor_id]["frame_height"]

    canvas_w = (world_max_x - world_min_x) * tile + frame_w + pad * 2
    canvas_h = (world_max_y - world_min_y) * tile + frame_h + pad * 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))

    placements = {}
    for map_id, prefix in WORLD_MAPS:
        meta = metas[map_id]
        panorama = Image.open(SCREENSHOT_DIR / f"{prefix}_map.png").convert("RGB")

        # Same "paste at (meta.min - world.min) * tile" placement math
        # render_route1_mashup.build_canvas uses for a single map against
        # its own rollout-expanded bounds -- just generalised to more than
        # one panorama sharing the same canvas.
        px = pad + (placed[map_id][0] + meta["min_x"] - world_min_x) * tile
        py = pad + (placed[map_id][1] + meta["min_y"] - world_min_y) * tile
        canvas.paste(panorama, (px, py))
        placements[map_id] = {"px": px, "py": py, "offset": placed[map_id]}
        print(f"  pasted at pixel ({px}, {py})")

    out_path = SCREENSHOT_DIR / "world_atlas.png"
    canvas.save(out_path)
    print(f"Saved {out_path} ({canvas_w}x{canvas_h})")

    write_json_atomic(SCREENSHOT_DIR / "world_atlas_meta.json", {
        "tile": tile,
        "pad": pad,
        "frame_width": frame_w,
        "frame_height": frame_h,
        "world_min_x": world_min_x,
        "world_min_y": world_min_y,
        "maps": {
            str(map_id): {
                "prefix": prefixes[map_id],
                "offset": list(placed[map_id]),
                "min_x": metas[map_id]["min_x"],
                "min_y": metas[map_id]["min_y"],
            }
            for map_id, _ in WORLD_MAPS
        },
    })
    print(f"Saved {SCREENSHOT_DIR / 'world_atlas_meta.json'}")


if __name__ == "__main__":
    sys.exit(main())
