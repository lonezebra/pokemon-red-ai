from emulator import create_emulator, run_frames
from state import load_bedroom_state
from controls import walk_tile
from memory import print_player_position, get_player_position
from screen import save_screenshot


DOWNSTAIRS_MAP_ID = 37


def main():
    pyboy = create_emulator()

    print("Loading bedroom state...")
    load_bedroom_state(pyboy)
    run_frames(pyboy, 60)

    print_player_position(pyboy, "Starting position")
    save_screenshot(pyboy, "replay_route_01_start.png")

    route = [
        "right",
        "up",
        "up",
        "up",
        "right",
    ]

    for direction in route:
        print()
        print(f"Walking {direction}")
        walk_tile(pyboy, direction)
        run_frames(pyboy, 240)
        print_player_position(pyboy, f"After {direction}")

        pos = get_player_position(pyboy)

        if pos["map_id"] == DOWNSTAIRS_MAP_ID:
            print("Success: reached downstairs.")
            break

    save_screenshot(pyboy, "replay_route_02_final.png")
    print_player_position(pyboy, "Final position")

    print()
    print("The emulator will stay open so you can inspect the result.")

    while pyboy.tick():
        pass

    pyboy.stop()


if __name__ == "__main__":
    main()