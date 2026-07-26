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