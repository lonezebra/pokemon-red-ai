"""
Interactive session starting right at the trainer whose battle keeps
resolving back to (19,4) with no further progress in every automated
test. Opens a real window -- use the arrow keys to walk, Z for A, X for
B (PyBoy's default keyboard bindings).

    ../.venv/bin/python3 play_route3_wall.py

Deliberately does NOT set POKEMON_AI_WINDOW_MODE=null -- this is the one
script in the project that's supposed to open a window, so you can
drive it yourself and show us the real path around this trainer that
every automated survey and battle-handler test has failed to find.

Close the window (or Ctrl-C) when done. Nothing is saved automatically;
if you find the way through, let us know the button sequence and we'll
capture a fresh state from here.
"""

from core.emulator import create_emulator
from core.state import load_state
from core.config import PROJECT_ROOT

STATE_PATH = PROJECT_ROOT / "saves" / "route3_before_the_wall.state"


def main():
    pyboy = create_emulator()
    load_state(pyboy, STATE_PATH)

    print("Playing from right before the wall at (19,4).")
    print("Arrow keys to walk, Z = A, X = B. Close the window when done.")

    while pyboy.tick():
        pass

    pyboy.stop()


if __name__ == "__main__":
    main()
