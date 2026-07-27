from datetime import datetime

from core.config import SCREENSHOT_DIR


def save_screenshot(pyboy, filename=None):
    """
    Save a screenshot of the current game screen.

    If no filename is provided, we create one using the current date/time.
    """

    SCREENSHOT_DIR.mkdir(exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"

    screenshot_path = SCREENSHOT_DIR / filename

    image = pyboy.screen.image
    image.save(screenshot_path)

    print(f"Saved screenshot: {screenshot_path}")

    return screenshot_path


def save_gif(frames, filename, duration_ms=80):
    """
    Stitch a list of PIL Images (e.g. captured via pyboy.screen.image.copy()
    once per step) into an animated GIF.
    """

    if not frames:
        return None

    SCREENSHOT_DIR.mkdir(exist_ok=True)
    gif_path = SCREENSHOT_DIR / filename

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )

    print(f"Saved GIF: {gif_path} ({len(frames)} frames)")

    return gif_path