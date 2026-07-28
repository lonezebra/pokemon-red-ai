import numpy as np
from PIL import Image

# Stitches per-tile screen captures into one image of a whole map.
#
# The trick that makes this work at all: the camera keeps the player at a
# fixed position on screen and scrolls the world in exact 16-pixel steps
# (confirmed by cross-correlating two frames one tile apart). So placing
# each captured frame at (x, y) * TILE, relative to a common origin,
# lines every frame up with every other one.
#
# Taking the per-pixel *median* across all frames overlapping a given
# canvas pixel then removes the one thing that moves between frames --
# the player's own sprite -- leaving only terrain. A first attempt that
# simply pasted frames left player-shaped ghosts scattered across the
# map; the median version does not.

TILE = 16
PAD = 24
STRIP_HEIGHT = 200  # bounds peak memory during the median, whatever the map size
BACKGROUND = 30


def stitch_panorama(frames, tile=TILE, pad=PAD):
    """
    `frames` is an iterable of (x, y, PIL.Image) captured while standing
    on tile (x, y). Returns (image, meta), where meta records the offsets
    a caller needs to map a world tile back onto a canvas pixel.
    """

    frames = list(frames)
    if not frames:
        raise ValueError("no frames to stitch")

    frame_h, frame_w = np.array(frames[0][2]).shape[:2]

    xs = [f[0] for f in frames]
    ys = [f[1] for f in frames]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    canvas_w = (max_x - min_x) * tile + frame_w + pad * 2
    canvas_h = (max_y - min_y) * tile + frame_h + pad * 2
    result = np.full((canvas_h, canvas_w, 3), BACKGROUND, dtype=np.uint8)

    for strip_top in range(0, canvas_h, STRIP_HEIGHT):
        strip_bottom = min(strip_top + STRIP_HEIGHT, canvas_h)
        strip_h = strip_bottom - strip_top

        relevant = []
        for x, y, image in frames:
            origin_y = pad + (y - min_y) * tile
            if origin_y < strip_bottom and origin_y + frame_h > strip_top:
                relevant.append((x, y, np.array(image)))

        if not relevant:
            continue

        stack = np.full((len(relevant), strip_h, canvas_w, 3), np.nan, dtype=np.float32)
        for i, (x, y, arr) in enumerate(relevant):
            origin_x = pad + (x - min_x) * tile
            origin_y = pad + (y - min_y) * tile
            src_top = max(0, strip_top - origin_y)
            src_bottom = min(frame_h, strip_bottom - origin_y)
            dst_top = origin_y + src_top - strip_top
            dst_bottom = origin_y + src_bottom - strip_top
            stack[i, dst_top:dst_bottom, origin_x:origin_x + frame_w, :] = (
                arr[src_top:src_bottom, :, :]
            )

        with np.errstate(all="ignore"):
            median = np.nanmedian(stack, axis=0)
        mask = ~np.isnan(median)
        result[strip_top:strip_bottom, :, :] = np.where(
            mask, median, float(BACKGROUND)
        ).astype(np.uint8)

    image = Image.fromarray(result, mode="RGB")
    meta = {
        "tile": tile,
        "pad": pad,
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
