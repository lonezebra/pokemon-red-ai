from emulator import run_frames


VALID_BUTTONS = {
    "up",
    "down",
    "left",
    "right",
    "a",
    "b",
    "start",
    "select",
}


def press_button(pyboy, button, hold_frames=20, release_frames=20):
    """
    Press and release a Game Boy button.

    Args:
        pyboy: The running PyBoy emulator.
        button: The button to press, such as "a", "start", or "up".
        hold_frames: How long to hold the button.
        release_frames: How long to wait after releasing the button.

    Why hold and release frames matter:
        Games do not respond to an abstract "button click".
        They respond to whether a button is currently down during a frame.
        So we press the button, tick the emulator forward, release the button,
        then tick forward again.
    """

    if button not in VALID_BUTTONS:
        raise ValueError(f"Invalid button: {button}. Valid buttons are: {VALID_BUTTONS}")

    print(f"Pressing {button}")

    pyboy.button_press(button)
    run_frames(pyboy, hold_frames)

    pyboy.button_release(button)
    run_frames(pyboy, release_frames)


def press_sequence(pyboy, buttons, hold_frames=20, release_frames=20):
    """
    Press a list of buttons in order.

    Example:
        press_sequence(pyboy, ["start", "a", "a"])
    """

    for button in buttons:
        press_button(
            pyboy,
            button,
            hold_frames=hold_frames,
            release_frames=release_frames,
        )