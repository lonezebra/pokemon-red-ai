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
    """

    for button in buttons:
        press_button(
            pyboy,
            button,
            hold_frames=hold_frames,
            release_frames=release_frames,
        )


def walk_tile(pyboy, direction):
    """
    Attempt to walk one tile in a direction.

    Pokemon movement is tile-based. A short tap may only turn the character.
    Holding the direction longer usually moves one full tile.
    """

    if direction not in {"up", "down", "left", "right"}:
        raise ValueError(f"Invalid walking direction: {direction}")

    press_button(pyboy, direction, hold_frames=25, release_frames=25)


def walk_tiles(pyboy, direction, count):
    """
    Walk multiple tiles in one direction.
    """

    for _ in range(count):
        walk_tile(pyboy, direction)