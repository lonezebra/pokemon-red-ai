from memory import get_player_position, print_player_position
from controls import walk_until_position_changes
from emulator import run_frames


def walk_to(pyboy, target_x, target_y, max_steps=50):
    """
    Walk toward a target X/Y coordinate.

    This is a simple greedy navigator.

    It tries to fix X first:
      - If current X is too small, walk right.
      - If current X is too large, walk left.

    Then it fixes Y:
      - If current Y is too small, walk down.
      - If current Y is too large, walk up.

    This version does not understand walls yet.
    """

    print()
    print(f"Walking to target X={target_x}, Y={target_y}")

    for step in range(1, max_steps + 1):
        pos = get_player_position(pyboy)
        current_x = pos["x"]
        current_y = pos["y"]

        print()
        print(f"Navigation step {step}")
        print(f"Current position: map={pos['map_id']}, x={current_x}, y={current_y}")

        if current_x == target_x and current_y == target_y:
            print("Reached target.")
            return True

        if current_x < target_x:
            direction = "right"
        elif current_x > target_x:
            direction = "left"
        elif current_y < target_y:
            direction = "down"
        elif current_y > target_y:
            direction = "up"
        else:
            print("Reached target.")
            return True

        print(f"Chosen direction: {direction}")

        moved = walk_until_position_changes(pyboy, direction)

        if not moved:
            print(f"Blocked while trying to walk {direction}.")
            print_player_position(pyboy, "Blocked position")
            return False

        run_frames(pyboy, 10)

    print("Failed: exceeded max_steps.")
    print_player_position(pyboy, "Final position after failure")
    return False