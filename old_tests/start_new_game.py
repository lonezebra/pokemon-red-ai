from emulator import create_emulator, run_frames
from controls import press_button
from screen import save_screenshot


def tap_a(pyboy, times=1, pause_frames=45):
    """
    Press A one or more times.

    Pokemon Red uses A to advance most text boxes and confirm menu choices.
    """
    for _ in range(times):
        press_button(pyboy, "a", hold_frames=20, release_frames=pause_frames)


def tap_start(pyboy, times=1, pause_frames=45):
    """
    Press Start one or more times.
    """
    for _ in range(times):
        press_button(pyboy, "start", hold_frames=20, release_frames=pause_frames)


def tap_down(pyboy, times=1, pause_frames=45):
    """
    Press Down one or more times.
    """
    for _ in range(times):
        press_button(pyboy, "down", hold_frames=20, release_frames=pause_frames)


def get_to_oak_intro(pyboy):
    """
    Move from boot/title screen into the New Game intro.

    This assumes a fresh boot of Pokemon Red.
    """
    print("Waiting for boot/title sequence...")
    run_frames(pyboy, 600)

    save_screenshot(pyboy, "01_after_boot_wait.png")

    print("Pressing Start at title screen...")
    tap_start(pyboy, times=1, pause_frames=120)

    save_screenshot(pyboy, "02_after_start.png")

    print("Selecting New Game...")
    # Usually New Game is already selected.
    # A confirms it.
    tap_a(pyboy, times=1, pause_frames=180)

    save_screenshot(pyboy, "03_after_new_game.png")


def advance_oak_intro(pyboy):
    """
    Advance through Professor Oak's opening dialogue.

    This is intentionally brute-force for now.
    We press A many times slowly enough for text boxes to advance.
    """
    print("Advancing through Oak intro text...")

    tap_a(pyboy, times=20, pause_frames=90)

    save_screenshot(pyboy, "04_after_oak_intro_text.png")


def choose_player_name(pyboy):
    """
    Choose the default player name.

    In Pokemon Red, when the naming screen appears, we can usually confirm
    the default name by pressing Start or A depending on the screen state.

    This may need adjustment based on your ROM/version.
    """
    print("Trying to accept default player name...")

    # Try pressing Start, then A.
    # If the game is on the naming screen, Start usually confirms the name.
    tap_start(pyboy, times=1, pause_frames=120)
    tap_a(pyboy, times=3, pause_frames=90)

    save_screenshot(pyboy, "05_after_player_name.png")


def choose_rival_name(pyboy):
    """
    Choose the default rival name.

    Same idea as player name: use default name to avoid typing letters.
    """
    print("Trying to accept default rival name...")

    tap_a(pyboy, times=8, pause_frames=90)
    tap_start(pyboy, times=1, pause_frames=120)
    tap_a(pyboy, times=5, pause_frames=90)

    save_screenshot(pyboy, "06_after_rival_name.png")


def finish_intro_to_bedroom(pyboy):
    """
    Continue pressing A until we reach the bedroom.

    This is brute force. Later we will replace this with smarter screen/memory checks.
    """
    print("Finishing intro and waiting for bedroom...")

    tap_a(pyboy, times=30, pause_frames=90)

    # Give the game time for the shrinking animation / transition.
    run_frames(pyboy, 300)

    save_screenshot(pyboy, "07_bedroom_attempt.png")


def main():
    pyboy = create_emulator()

    print("Starting new game automation...")

    get_to_oak_intro(pyboy)
    advance_oak_intro(pyboy)
    choose_player_name(pyboy)
    choose_rival_name(pyboy)
    finish_intro_to_bedroom(pyboy)

    print("Automation finished.")
    print("Look at screenshots/07_bedroom_attempt.png")
    print("The emulator will stay open so you can inspect where it ended.")

    while pyboy.tick():
        pass

    pyboy.stop()


if __name__ == "__main__":
    main()