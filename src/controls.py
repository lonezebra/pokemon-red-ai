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


DIRECTION_BUTTONS = {
    "up",
    "down",
    "left",
    "right",
}


def press_button(pyboy, button, hold_frames=20, release_frames=20):
    """
    Press and release a Game Boy button.
    """

    if button not in VALID_BUTTONS:
        raise ValueError(f"Invalid button: {button}. Valid buttons are: {VALID_BUTTONS}")

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


def position_changed(before, after):
    """
    Return True if map/x/y changed.
    """

    return (
        before["map_id"] != after["map_id"]
        or before["x"] != after["x"]
        or before["y"] != after["y"]
    )


def walk_tile(pyboy, direction, max_hold_frames=60, settle_frames=20):
    """
    Walk one tile by holding a direction until memory says the player moved.

    This is better than using a fixed hold time because movement timing can vary.
    """

    if direction not in DIRECTION_BUTTONS:
        raise ValueError(f"Invalid walking direction: {direction}")

    from memory import get_player_position

    before = get_player_position(pyboy)

    print(f"Walking {direction} from {before}")

    pyboy.button_press(direction)

    moved = False
    after = before

    for held_frames in range(1, max_hold_frames + 1):
        pyboy.tick()
        after = get_player_position(pyboy)

        if position_changed(before, after):
            moved = True
            print(f"  Position changed after {held_frames} held frame(s): {after}")
            break

    pyboy.button_release(direction)

    # Let the walking animation finish before the next command.
    run_frames(pyboy, settle_frames)

    if not moved:
        print(f"  No movement detected after holding {direction} for {max_hold_frames} frame(s).")
        return False

    return True


def walk_until_position_changes(pyboy, direction, max_attempts=3):
    """
    Try to walk in a direction.

    Kept for compatibility with navigation.py.
    """

    for attempt in range(1, max_attempts + 1):
        print(f"Attempt {attempt} to walk {direction}")
        moved = walk_tile(pyboy, direction)

        if moved:
            return True

    return False


def walk_tiles(pyboy, direction, count):
    """
    Walk multiple tiles in one direction.
    """

    for _ in range(count):
        moved = walk_tile(pyboy, direction)

        if not moved:
            return False

    return True


def advance_battle_dialogue(pyboy, max_presses=60, hold_frames=10, release_frames=15):
    """
    Press A repeatedly until the battle is ready for the next player
    decision (the FIGHT/ITEM/RUN menu reopens) or the battle has ended.

    Different moves/messages take different numbers of text boxes to
    clear, so this checks the actual game state after each press instead
    of using a fixed press count -- the same idea as walk_tile() checking
    position instead of trusting a fixed hold time.
    """

    from memory import is_battle_menu_open, is_in_battle

    for _ in range(max_presses):
        if is_battle_menu_open(pyboy) or not is_in_battle(pyboy):
            return True

        press_button(pyboy, "a", hold_frames=hold_frames, release_frames=release_frames)

    return is_battle_menu_open(pyboy) or not is_in_battle(pyboy)