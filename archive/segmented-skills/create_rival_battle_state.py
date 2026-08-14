from core.emulator import create_emulator, run_frames
from core.state import load_state, save_state, STARTER_OBTAINED_STATE_PATH, RIVAL_BATTLE_STATE_PATH
from core.controls import press_button, walk_tile
from core.memory import get_player_position, print_player_position
from core.screen import save_screenshot
from create_starter_obtained_state import wait_for_control_and_walk


# Findings from probing this by hand, starting at starter_obtained.state
# (Oak's Lab, map 40, x=7, y=4):
#
#   - The rival (Blue) does not wait outside. He stops the player from
#     inside Oak's Lab itself, at map 40, x=5, y=6, while walking toward
#     the exit.
#   - After choosing a starter, there is a lengthy scripted dialogue
#     sequence (Blue picking his own Pokemon, taunting, etc.) that has to
#     be advanced with A before the player regains movement control.
#     60 presses was not enough; 120 reliably clears it.
#   - Once the "Wait RED!" text box appears (automatically, just from
#     reaching x=5, y=6 -- no extra input needed) and is advanced, the
#     battle begins on its own.
#   - From the "Wait RED!" trigger, 31 more A-presses reaches the first
#     FIGHT/ITEM/RUN menu: the very first turn, before either Pokemon has
#     made a move. That is the save point used here.

TRIGGER_MAP_ID = 40
TRIGGER_X = 5
TRIGGER_Y = 6

# Route from the starter-selection spot (7,4) to the tile where the
# rival stops the player (5,6).
ROUTE_TO_RIVAL_TRIGGER = ["down", "left", "left", "down"]

DIALOGUE_PRESSES_AFTER_STARTER = 120
PRESSES_TO_FIRST_BATTLE_MENU = 31


def advance_dialogue(pyboy, presses, hold_frames=15, release_frames=25):
    for _ in range(presses):
        press_button(pyboy, "a", hold_frames=hold_frames, release_frames=release_frames)


def walk_to_rival_trigger_and_battle(pyboy):
    """
    From a live session where the player already has free movement in
    Oak's Lab right after choosing a starter (e.g. straight out of
    create_starter_obtained_state.choose_starter(), continuing the same
    pyboy instance rather than reloading a save file), walk to the
    rival's trigger tile and through the dialogue into the first battle
    menu.

    Unlike main()'s fixed DIALOGUE_PRESSES_AFTER_STARTER count -- tuned
    for the specific entry point of loading starter_obtained.state fresh
    -- this uses the same robust "press A, then test the real next move"
    pattern as create_starter_obtained_state.py, since chaining live from
    a different exact entry point is exactly the kind of frame-timing
    drift that made fixed counts unreliable elsewhere in this project.
    """

    if not wait_for_control_and_walk(pyboy, ROUTE_TO_RIVAL_TRIGGER[0]):
        print("Warning: never regained control to start walking to the rival trigger.")
        return False

    for direction in ROUTE_TO_RIVAL_TRIGGER[1:]:
        walk_tile(pyboy, direction, verbose=False)
        run_frames(pyboy, 10)

    pos = get_player_position(pyboy)
    print_player_position(pyboy, "Position at rival trigger")

    if pos["map_id"] != TRIGGER_MAP_ID or pos["x"] != TRIGGER_X or pos["y"] != TRIGGER_Y:
        print(
            f"Warning: expected trigger tile (map {TRIGGER_MAP_ID}, "
            f"x={TRIGGER_X}, y={TRIGGER_Y}), got {pos}."
        )
        return False

    advance_dialogue(pyboy, PRESSES_TO_FIRST_BATTLE_MENU)
    return True


def main():
    pyboy = create_emulator()

    print("Loading starter_obtained state...")
    load_state(pyboy, STARTER_OBTAINED_STATE_PATH)
    run_frames(pyboy, 300)

    print("Clearing the 'Blue picks his starter' dialogue...")
    advance_dialogue(pyboy, DIALOGUE_PRESSES_AFTER_STARTER)

    print_player_position(pyboy, "Position after regaining control")

    print()
    print("Walking toward the lab exit...")
    for direction in ROUTE_TO_RIVAL_TRIGGER:
        walk_tile(pyboy, direction)
        run_frames(pyboy, 10)

    pos = get_player_position(pyboy)
    print_player_position(pyboy, "Position at rival trigger")
    save_screenshot(pyboy, "rival_trigger_reached.png")

    if pos["map_id"] != TRIGGER_MAP_ID or pos["x"] != TRIGGER_X or pos["y"] != TRIGGER_Y:
        print(
            f"Warning: expected trigger tile (map {TRIGGER_MAP_ID}, "
            f"x={TRIGGER_X}, y={TRIGGER_Y}), got {pos}."
        )

    print()
    print("Advancing through the rival's dialogue and into battle...")
    advance_dialogue(pyboy, PRESSES_TO_FIRST_BATTLE_MENU)

    save_screenshot(pyboy, "rival_battle_first_menu.png")

    print()
    print(f"Saving battle-start state to {RIVAL_BATTLE_STATE_PATH}...")
    save_state(pyboy, RIVAL_BATTLE_STATE_PATH)

    print("Done. Check screenshots/rival_battle_first_menu.png to confirm")
    print("it shows the FIGHT/ITEM/RUN menu with both Pokemon at full HP.")

    pyboy.stop()


if __name__ == "__main__":
    main()
