from core.emulator import create_emulator, run_frames
from core.state import load_state, save_state
from core.config import SAVE_DIR
from core.controls import walk_tile, press_button
from core.memory import get_player_position, print_player_position
from core.screen import save_screenshot


# Findings from probing this by hand, starting at outside_house.state
# (Pallet Town, map 0, x=5, y=6):
#
#   - The route to Oak's trigger tile (map 0, x=10, y=1) is a straight
#     line: 5 right, then 5 up.
#   - Reaching it starts a long, mostly-automatic sequence: Oak's "Hey!
#     Wait!", the game itself walking the player through the door and
#     across the lab to a fixed arrival tile (map 40, x=5, y=3), then more
#     dialogue explaining the choice.
#   - Two things made this sequence unreliable with fixed press counts
#     (the same class of bug as the battle dialogue elsewhere in this
#     project): the exact number of A-presses needed drifts slightly
#     between runs (small frame-timing variance carried over from the
#     walk_tile calls earlier), and once true player control returns,
#     *continuing* to press A immediately re-triggers the same prompt
#     again -- so a fixed count can just as easily overshoot as fall
#     short. The fix used here is two-phase: press A until memory
#     confirms we've actually landed on the known arrival tile, then
#     press A one at a time while probing for real movement after each
#     press, stopping the instant movement actually succeeds.
#   - wPartyCount (0xD163) reliably reads 0 before a starter is chosen
#     and 1 immediately after -- confirmed against outside_house.state,
#     oak_lab.state, starter_obtained.state, and rival_battle.state
#     before being trusted here.

ROUTE_TO_OAK_TRIGGER = ["right", "right", "right", "right", "right", "up", "up", "up", "up", "up"]

LAB_MAP_ID = 40
LAB_ARRIVAL_X = 5
LAB_ARRIVAL_Y = 3

PARTY_COUNT_ADDR = 0xD163

# Starter Pokemon ball positions in Oak's Lab (map 40), and the route to
# each from the lab arrival tile (5, 3). Only Squirtle is used by
# default: the trained rival-battle DQN was verified specifically
# against a Squirtle-vs-Bulbasaur matchup, so switching starters here
# would need retraining that model too.
STARTERS = {
    "charmander": {"x": 6, "y": 4, "route": ["down", "right"]},
    "squirtle": {"x": 7, "y": 4, "route": ["down", "right", "right"]},
    "bulbasaur": {"x": 8, "y": 4, "route": ["down", "right", "right", "right"]},
}


def walk_to_oak_trigger(pyboy):
    for direction in ROUTE_TO_OAK_TRIGGER:
        walk_tile(pyboy, direction, verbose=False)
        run_frames(pyboy, 10)


def wait_for_lab_arrival(pyboy, max_presses=150):
    for _ in range(max_presses):
        press_button(pyboy, "a", hold_frames=10, release_frames=15)
        pos = get_player_position(pyboy)
        if pos["map_id"] == LAB_MAP_ID and pos["x"] == LAB_ARRIVAL_X and pos["y"] == LAB_ARRIVAL_Y:
            return True

    return False


def wait_for_control_and_walk(pyboy, direction, max_presses=200):
    """
    Press A while repeatedly probing for real movement in `direction`,
    stopping the instant a press actually moves the player -- see the
    module docstring for why a fixed press count doesn't work here.
    """

    for _ in range(max_presses):
        press_button(pyboy, "a", hold_frames=15, release_frames=25)

        if walk_tile(pyboy, direction, max_hold_frames=20, settle_frames=5, verbose=False):
            return True

    return False


def choose_starter(pyboy, starter_name, nickname_presses=60):
    starter = STARTERS[starter_name]

    if not wait_for_control_and_walk(pyboy, starter["route"][0]):
        return False

    for direction in starter["route"][1:]:
        walk_tile(pyboy, direction, verbose=False)
        run_frames(pyboy, 10)

    # Face the ball before interacting with it.
    walk_tile(pyboy, "up", verbose=False)
    run_frames(pyboy, 10)

    press_button(pyboy, "a", hold_frames=15, release_frames=25)

    # Push through "do you want this Pokemon?", Oak's comment, and the
    # nickname keyboard. Blindly mashing A on the nickname screen just
    # repeatedly picks the first letter and confirms -- matching the
    # STARTER_OBTAINED_STATE_PATH state this project already had, whose
    # Squirtle ended up nicknamed "AAAAAAAA" the same way.
    for _ in range(nickname_presses):
        press_button(pyboy, "a", hold_frames=15, release_frames=25)

    return pyboy.memory[PARTY_COUNT_ADDR] == 1


def main(starter_name="squirtle"):
    pyboy = create_emulator()

    print("Loading outside_house state...")
    load_state(pyboy, SAVE_DIR / "outside_house.state")
    run_frames(pyboy, 60)

    print("Walking to Oak's trigger...")
    walk_to_oak_trigger(pyboy)
    print_player_position(pyboy, "At Oak's trigger")
    save_screenshot(pyboy, "starter_route_at_trigger.png")

    print("Waiting for the automatic walk-in to the lab...")
    if not wait_for_lab_arrival(pyboy):
        print("Warning: did not reach the lab arrival tile as expected.")
        pyboy.stop()
        return

    print(f"Choosing {starter_name}...")
    obtained = choose_starter(pyboy, starter_name)

    print_player_position(pyboy, "Final position")
    save_screenshot(pyboy, "starter_route_final.png")

    if not obtained:
        print("Warning: wPartyCount never reached 1 -- starter may not have been obtained.")
        pyboy.stop()
        return

    print("Starter obtained. Saving state...")
    save_state(pyboy, SAVE_DIR / "starter_obtained.state")
    print(f"Saved to {SAVE_DIR / 'starter_obtained.state'}")

    pyboy.stop()


if __name__ == "__main__":
    main()
