MOVEMENT_ACTIONS = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
}


def get_action_name(action_id):
    """
    Convert an integer action into a Game Boy movement direction.

    For now, our agent only moves.
    Later we will add:
      A
      B
      Start
      Select
    """

    if action_id not in MOVEMENT_ACTIONS:
        raise ValueError(f"Invalid action_id: {action_id}")

    return MOVEMENT_ACTIONS[action_id]


def num_actions():
    """
    Return the number of available actions.
    """

    return len(MOVEMENT_ACTIONS)