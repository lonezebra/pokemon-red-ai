import numpy as np
from PIL import Image

from core.emulator import create_emulator, run_frames
from core.state import load_state
from core.controls import walk_tile
from core.config import PROJECT_ROOT, SCREENSHOT_DIR

# One-off asset-generation tool: captures small reusable RGBA cutouts of
# the player's own overworld sprite, one per facing direction, for
# anything that wants to draw the actual character rather than an
# abstract dot (see render_route1_mashup.py).
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


def sprite_path(direction):
    return SCREENSHOT_DIR / f"player_sprite_{direction}.png"


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

# A single press doesn't just turn to face a new direction the way some
# later-gen Pokemon games do -- tested directly (a 4-frame press toward
# an unblocked direction still moved a full tile) -- so capturing each
# facing direction means actually stepping that way, then stepping back
# to the same base spot afterward (both to keep all four captures at one
# consistent position, and to reveal the background hidden behind the
# sprite: the camera's fixed 16px/tile shift means whatever was behind
# the sprite in the "facing" frame is visible, unobscured, in the
# "stepped back" frame, at a known pixel offset).
OPPOSITE_DIRECTION = {"up": "down", "down": "up", "left": "right", "right": "left"}

# How much each axis shifts (dx, dy) in the *background* when the player
# steps in the opposite direction (i.e. steps back to the base spot) --
# sign convention confirmed empirically while building the Route 1
# panorama (stepping up shifted the background down by exactly 16px).
BACKGROUND_SHIFT_FOR_STEP_BACK = {
    "up": (0, 16),
    "down": (0, -16),
    "left": (16, 0),
    "right": (-16, 0),
}


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


def build_sprite(live_frame, shifted_frame, shift_dx, shift_dy):
    sx0, sy0, sx1, sy1 = PLAYER_SPRITE_BOUNDS
    live = live_frame[sy0:sy1, sx0:sx1]
    background = shifted_frame[sy0 + shift_dy:sy1 + shift_dy, sx0 + shift_dx:sx1 + shift_dx]

    diff = np.abs(live.astype(int) - background.astype(int)).sum(axis=2)
    mask = largest_connected_component(diff > DIFF_THRESHOLD)
    alpha = np.where(mask, 255, 0).astype(np.uint8)

    return Image.fromarray(np.dstack([live, alpha]), mode="RGBA")


def extract_player_sprites():
    """
    One sprite per facing direction: walk to the reference spot, then
    for each direction, step that way (now facing it), capture, step
    back to the same base spot (revealing the background), capture
    again, and diff.
    """

    pyboy = create_emulator()
    load_state(pyboy, REFERENCE_STATE_PATH)
    run_frames(pyboy, 30)

    for direction in REFERENCE_WALK:
        walk_tile(pyboy, direction, verbose=False)
        run_frames(pyboy, 10)

    sprites = {}

    for direction in ["up", "down", "left", "right"]:
        walk_tile(pyboy, direction, verbose=False)
        run_frames(pyboy, 10)
        live_frame = np.array(pyboy.screen.image.convert("RGB"))

        step_back = OPPOSITE_DIRECTION[direction]
        walk_tile(pyboy, step_back, verbose=False)
        run_frames(pyboy, 10)
        shifted_frame = np.array(pyboy.screen.image.convert("RGB"))

        shift_dx, shift_dy = BACKGROUND_SHIFT_FOR_STEP_BACK[step_back]
        sprites[direction] = build_sprite(live_frame, shifted_frame, shift_dx, shift_dy)

    pyboy.stop()
    return sprites


def main():
    sprites = extract_player_sprites()
    SCREENSHOT_DIR.mkdir(exist_ok=True)
    for direction, sprite in sprites.items():
        path = sprite_path(direction)
        sprite.save(path)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
