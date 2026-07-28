from core.emulator import run_frames


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


def walk_tile(pyboy, direction, max_hold_frames=60, settle_frames=20, verbose=True):
    """
    Walk one tile by holding a direction until memory says the player moved.

    This is better than using a fixed hold time because movement timing can vary.

    verbose=False silences the per-tile prints below -- useful for the small
    scripted route scripts, but a real RL training loop calls this thousands
    of times and the prints would just bury everything else in noise.
    """

    if direction not in DIRECTION_BUTTONS:
        raise ValueError(f"Invalid walking direction: {direction}")

    from core.memory import get_player_position

    before = get_player_position(pyboy)

    if verbose:
        print(f"Walking {direction} from {before}")

    pyboy.button_press(direction)

    moved = False
    after = before

    for held_frames in range(1, max_hold_frames + 1):
        pyboy.tick()
        after = get_player_position(pyboy)

        if position_changed(before, after):
            moved = True
            if verbose:
                print(f"  Position changed after {held_frames} held frame(s): {after}")
            break

    pyboy.button_release(direction)

    # Let the walking animation finish before the next command.
    run_frames(pyboy, settle_frames)

    if not moved:
        if verbose:
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


def wait_for_position_to_settle(pyboy, check_interval=10, max_checks=30, stable_checks_required=3):
    """
    Wait out an automatic, no-input-required movement sequence (e.g. the
    game auto-walking the player forward out of a doorway right after a
    building-exit warp) by polling position instead of guessing how many
    frames it takes.

    Discovered while chaining the leave-house Q-agent into the
    controller: the moment map_id becomes 0 (Pallet Town) is *not* the
    final resting position -- the game keeps walking the player a couple
    more tiles on its own afterward. Trying to act immediately (as the
    controller originally did) collided with that in-progress movement
    and produced a nonsensical multi-tile position jump. Waiting for
    position to stop changing on its own, instead of either a fixed
    frame count or acting immediately, works regardless of exactly how
    long that automatic walk takes.
    """

    from core.memory import get_player_position

    stable_count = 0
    last_position = get_player_position(pyboy)

    for _ in range(max_checks):
        run_frames(pyboy, check_interval)
        position = get_player_position(pyboy)

        if position == last_position:
            stable_count += 1
            if stable_count >= stable_checks_required:
                return position
        else:
            stable_count = 0

        last_position = position

    return last_position


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

    from core.memory import is_battle_menu_open, is_in_battle

    for _ in range(max_presses):
        if is_battle_menu_open(pyboy) or not is_in_battle(pyboy):
            return True

        press_button(pyboy, "a", hold_frames=hold_frames, release_frames=release_frames)

    return is_battle_menu_open(pyboy) or not is_in_battle(pyboy)


def attempt_run_from_wild_battle(pyboy, max_attempts=10):
    """
    Try to flee a wild battle by selecting RUN from the FIGHT/PKMN/ITEM/RUN
    menu, retrying if a flee attempt fails (Gen 1 run success depends on
    a speed comparison against the wild Pokemon, so it isn't guaranteed
    on the first try).

    The main battle menu is a 2x2 grid with the cursor starting on FIGHT
    (top-left) every time it opens: FIGHT/PKMN on top, ITEM/RUN on the
    bottom. Pressing down then right moves the cursor from FIGHT to RUN,
    confirmed by reading the window tilemap's cursor-arrow position
    before and after each press rather than assuming the layout. Returns
    True once the battle has actually ended (is_in_battle is False), or
    False if it's still ongoing after max_attempts.

    Unused for the rival battle (running from a Gen 1 trainer battle is
    not possible), but wild encounters -- e.g. while navigating Route 1
    -- are exactly the case this project's design notes call out as
    "the AI needs to learn when to run."
    """

    from core.memory import is_in_battle

    for _ in range(max_attempts):
        advance_battle_dialogue(pyboy)
        if not is_in_battle(pyboy):
            return True

        press_button(pyboy, "down", hold_frames=10, release_frames=15)
        press_button(pyboy, "right", hold_frames=10, release_frames=15)
        press_button(pyboy, "a", hold_frames=10, release_frames=15)
        run_frames(pyboy, 20)

    return not is_in_battle(pyboy)

def wait_for_free_movement(pyboy, max_presses=60):
    """
    Press A until the player can actually walk again, then report
    whether control came back.

    Needed because several things hand control back only after a text
    box that nothing else clears -- most importantly a blackout, where
    fainting teleports the player to a Pokemon Center and leaves the
    "you scurried back" message on screen. Pathfinding into that state
    reads every direction as blocked and concludes, quite reasonably,
    that there is nowhere to go.

    Direction-agnostic on purpose, unlike
    create_starter_obtained_state.wait_for_control_and_walk, which
    tests one specific next move in a known route. Here the caller just
    needs *any* movement to prove control is back, and it undoes the
    probe step so the player finishes where they started.
    """

    from core.memory import get_player_position

    for _ in range(max_presses):
        for direction in ("up", "down", "left", "right"):
            before = get_player_position(pyboy)
            if walk_tile(pyboy, direction, max_hold_frames=20, settle_frames=5, verbose=False):
                after = get_player_position(pyboy)
                if after["map_id"] == before["map_id"]:
                    walk_tile(pyboy, OPPOSITE[direction], max_hold_frames=20,
                              settle_frames=5, verbose=False)
                return True

        press_button(pyboy, "a", hold_frames=12, release_frames=24)

    return False


OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
