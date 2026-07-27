import numpy as np
from PIL import Image

from core.emulator import create_emulator, run_frames
from core.state import load_state
from core.controls import walk_tile
from core.config import PROJECT_ROOT, SCREENSHOT_DIR

# One-off asset-generation tool: captures a small reusable RGBA cutout of
# the player's own overworld sprite, for anything that wants to draw the
# actual character rather than an abstract dot (see render_route1_mashup.py).
#
# The overworld sprite is always drawn at this fixed 16x16 pixel box on
# the 160x144 screen (the camera keeps the player centered) -- found
# empirically by cropping a candidate region from a still screenshot and
# adjusting until it framed the sprite exactly, not derived from any
# documented constant. Confirmed to hold generally (not just at the one
# spot it was first found): cross-correlating two frames one tile apart
# showed the background shifts by exactly 16px, matching a fixed camera
# offset, in an entirely different location than where this box was
# first measured.
PLAYER_SPRITE_BOUNDS = (63, 58, 79, 74)
PLAYER_SPRITE_PATH = SCREENSHOT_DIR / "player_sprite.png"

# Where the sprite gets captured from matters. Two real problems ruled
# out Route 1's own entry point (the first location tried):
#   - Route 1 is tall grass, which visually overlaps the sprite's lower
#     half -- fine for a small dot, obviously wrong once the actual
#     character is on screen.
#   - Right outside the player's house, the camera turned out not to
#     center the player the usual way (confirmed by comparing the
#     sprite's position there against elsewhere -- it was visibly
#     shifted), almost certainly map-edge clamping the same way Route
#     1's own narrow width clamps horizontally (see route1_rewards.py's
#     history). A few tiles further from the house, away from that edge,
#     the sprite sits exactly where the fixed bounds above expect.
#
# This walk (from saves/outside_house.state) reaches a spot on Pallet
# Town's plain path terrain, clear of both problems.
REFERENCE_STATE_PATH = PROJECT_ROOT / "saves" / "outside_house.state"
REFERENCE_WALK = ["down", "down", "down", "left", "left"]

TILE = 16
DIFF_THRESHOLD = 30


def largest_connected_component(mask):
    """
    Keeps only the largest 8-connected blob of True pixels in a 2D
    boolean array, zeroing out everything else. Plain BFS rather than
    reaching for scipy.ndimage.label -- the sprite mask is 16x16, so
    there's no real cost to a from-scratch implementation here.

    8-connected (not 4-connected) deliberately -- a first attempt at
    4-connectivity trimmed the sprite's feet off, since they're only
    diagonally connected to the body at this resolution.
    """

    visited = np.zeros_like(mask, dtype=bool)
    best_component = None

    for start_row in range(mask.shape[0]):
        for start_col in range(mask.shape[1]):
            if not mask[start_row, start_col] or visited[start_row, start_col]:
                continue

            component = []
            queue = [(start_row, start_col)]
            visited[start_row, start_col] = True

            while queue:
                row, col = queue.pop()
                component.append((row, col))
                for dr, dc in (
                    (-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1),
                ):
                    nr, nc = row + dr, col + dc
                    if (
                        0 <= nr < mask.shape[0]
                        and 0 <= nc < mask.shape[1]
                        and mask[nr, nc]
                        and not visited[nr, nc]
                    ):
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            if best_component is None or len(component) > len(best_component):
                best_component = component

    result = np.zeros_like(mask, dtype=bool)
    if best_component:
        for row, col in best_component:
            result[row, col] = True
    return result


def extract_player_sprite():
    """
    Isolates the sprite from its background without needing a whole
    player-free reference image (e.g. a stitched panorama): capture one
    frame standing still, then one more frame after moving exactly one
    tile. The camera's fixed 16px/tile shift means whatever was hidden
    behind the sprite in the first frame is now visible, unobscured, at
    a known offset in the second -- diffing the two isolates the sprite
    pixels directly.
    """

    pyboy = create_emulator()
    load_state(pyboy, REFERENCE_STATE_PATH)
    run_frames(pyboy, 30)

    for direction in REFERENCE_WALK:
        walk_tile(pyboy, direction, verbose=False)
        run_frames(pyboy, 10)

    live_frame = np.array(pyboy.screen.image.convert("RGB"))

    walk_tile(pyboy, "right", verbose=False)
    run_frames(pyboy, 10)
    shifted_frame = np.array(pyboy.screen.image.convert("RGB"))

    pyboy.stop()

    sx0, sy0, sx1, sy1 = PLAYER_SPRITE_BOUNDS
    live = live_frame[sy0:sy1, sx0:sx1]
    # moved right, so the background shifts left by one tile
    background = shifted_frame[sy0:sy1, sx0 - TILE:sx1 - TILE]

    diff = np.abs(live.astype(int) - background.astype(int)).sum(axis=2)
    raw_mask = diff > DIFF_THRESHOLD
    mask = largest_connected_component(raw_mask)
    alpha = np.where(mask, 255, 0).astype(np.uint8)

    rgba = np.dstack([live, alpha])
    return Image.fromarray(rgba, mode="RGBA")


def main():
    sprite = extract_player_sprite()
    PLAYER_SPRITE_PATH.parent.mkdir(exist_ok=True)
    sprite.save(PLAYER_SPRITE_PATH)
    print(f"Saved {PLAYER_SPRITE_PATH}")


if __name__ == "__main__":
    main()
