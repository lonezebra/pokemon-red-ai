import json

import numpy as np
from PIL import Image

from core.config import SCREENSHOT_DIR
from core.screen import save_gif

MAP_IMAGE_PATH = SCREENSHOT_DIR / "route1_map.png"
MAP_META_PATH = SCREENSHOT_DIR / "route1_map_meta.json"
PLAYER_SPRITE_PATH = SCREENSHOT_DIR / "route1_player_sprite.png"
MASHUP_DIR = SCREENSHOT_DIR / "mashups"

IN_PROGRESS_COLOR = (240, 220, 60)
SUCCESS_COLOR = (60, 220, 90)
UNFINISHED_COLOR = (230, 90, 60)
BACKGROUND_COLOR = (30, 30, 30)


def latest_run_label():
    # Picks up the run generate_route1_mashup_rollouts.py just produced
    # without needing any state passed explicitly between the two
    # scripts -- convenient for the common case of running them back to
    # back, one training milestone at a time.
    run_dirs = [d for d in MASHUP_DIR.iterdir() if d.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run folders found under {MASHUP_DIR}")

    return max(run_dirs, key=lambda d: d.stat().st_mtime).name


def tint_sprite(sprite, color):
    """
    Recolors the (grayscale, since the Game Boy has no color) player
    sprite by scaling each channel of the target color by the sprite's
    own per-pixel brightness -- keeps the original shading (the cap's
    outline stays dark, highlights stay bright) while giving each outcome
    category a distinct, recognizable color. Pixels the sprite's alpha
    mask marks as background stay fully transparent.
    """

    arr = np.array(sprite).astype(float)
    rgb, alpha = arr[:, :, :3], arr[:, :, 3]

    intensity = rgb.mean(axis=2, keepdims=True) / 255.0
    tinted_rgb = intensity * np.array(color, dtype=float)

    tinted = np.dstack([tinted_rgb, alpha]).astype(np.uint8)
    return Image.fromarray(tinted, mode="RGBA")


def build_canvas(meta, runs):
    """
    The panorama only covers the tiles the mapping scout actually
    walked through -- a still-training agent can easily wander outside
    that range. Rather than clip or skip those runs, expand the canvas
    to fit every recorded position too, pasting the real panorama in at
    the right offset and leaving the rest as plain background.
    """

    tile = meta["tile"]
    pad = meta["pad"]

    all_xs = [meta["min_x"], meta["max_x"]]
    all_ys = [meta["min_y"], meta["max_y"]]
    for run in runs:
        for x, y in run["positions"]:
            all_xs.append(x)
            all_ys.append(y)

    min_x, max_x = min(all_xs), max(all_xs)
    min_y, max_y = min(all_ys), max(all_ys)

    panorama = Image.open(MAP_IMAGE_PATH).convert("RGB")

    canvas_w = (max_x - min_x) * tile + meta["frame_width"] + pad * 2
    canvas_h = (max_y - min_y) * tile + meta["frame_height"] + pad * 2

    base = Image.new("RGB", (canvas_w, canvas_h), BACKGROUND_COLOR)
    paste_x = pad + (meta["min_x"] - min_x) * tile - pad
    paste_y = pad + (meta["min_y"] - min_y) * tile - pad
    base.paste(panorama, (paste_x, paste_y))

    def to_pixel(x, y):
        px = paste_x + pad + (x - meta["min_x"]) * tile + tile // 2
        py = paste_y + pad + (y - meta["min_y"]) * tile + tile // 2
        return px, py

    return base, to_pixel


def main(run_label=None, duration_ms=60):
    run_label = run_label or latest_run_label()
    run_dir = MASHUP_DIR / run_label

    with open(MAP_META_PATH) as f:
        meta = json.load(f)
    with open(run_dir / "route1_mashup_rollouts.json") as f:
        data = json.load(f)

    runs = data["runs"]
    base_canvas, to_pixel = build_canvas(meta, runs)

    base_sprite = Image.open(PLAYER_SPRITE_PATH).convert("RGBA")
    sprites = {
        "in_progress": tint_sprite(base_sprite, IN_PROGRESS_COLOR),
        "success": tint_sprite(base_sprite, SUCCESS_COLOR),
        "unfinished": tint_sprite(base_sprite, UNFINISHED_COLOR),
    }
    sprite_w, sprite_h = base_sprite.size

    max_len = max(len(run["positions"]) for run in runs)
    frames = []

    for t in range(max_len):
        frame = base_canvas.copy()

        for run in runs:
            positions = run["positions"]
            idx = min(t, len(positions) - 1)
            x, y = positions[idx]

            # Color by outcome only once a run has actually stopped
            # moving -- while still walking, every sprite is the same
            # in-progress color regardless of how it'll end.
            if t < len(positions) - 1:
                sprite = sprites["in_progress"]
            else:
                sprite = sprites["success"] if run["reached_goal"] else sprites["unfinished"]

            center_x, center_y = to_pixel(x, y)
            paste_x = center_x - sprite_w // 2
            paste_y = center_y - sprite_h // 2
            frame.paste(sprite, (paste_x, paste_y), sprite)

        frames.append(frame)

        if t % 50 == 0:
            print(f"Rendered frame {t}/{max_len}")

    successes = sum(1 for run in runs if run["reached_goal"])
    print(f"{successes}/{len(runs)} runs reached Viridian City")

    save_gif(frames, f"mashups/{run_label}/route1_mashup.gif", duration_ms=duration_ms)


if __name__ == "__main__":
    main()
