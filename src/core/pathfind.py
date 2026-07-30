import io
from collections import deque

from core.emulator import run_frames
from core.controls import walk_tile, press_button, attempt_run_from_wild_battle, wait_for_position_to_settle
from core.memory import get_player_position, is_in_battle, get_battle_type

# General-purpose overworld pathfinding scaffolding.
#
# Every route in this project up to now was found by biased random walks
# with stuck-escape heuristics (see build_route1_map.py,
# create_route22_entry_state.py). That works, but it is slow, seed-
# dependent, and it failed outright on anything maze-like -- several
# thousand scripted steps never found Route 2's exit, and the same
# approach wandered into Viridian's buildings as often as its actual
# exits.
#
# This replaces that with a breadth-first search over real game states:
# from the current position, try all four directions, snapshot every
# newly reached tile, and keep expanding until a tile satisfies the
# caller's predicate. Two properties make it reliable:
#
#   - It is exhaustive within a map, so "no path exists" is a real
#     answer rather than "the random walk got unlucky". A BFS like this
#     is what established that Viridian City has no northern exit at
#     all (500 reachable tiles, exits only to Route 1, Route 22, and
#     four buildings).
#   - On success it *loads the snapshot it found* rather than replaying
#     the moves that got there. Replaying could diverge -- Route 1's
#     grass can interrupt any step with a wild encounter, and fleeing
#     consumes a different number of turns each time -- but restoring a
#     state that was already verified to be at the target cannot.
#
# Wild encounters during the search are fled automatically, the same way
# envs/route1_env.py does, so the search only ever deals with overworld
# positions.

DIRECTIONS = ("up", "down", "left", "right")

# Bounds worst-case memory: each visited tile holds a full save state
# (~170KB), so ~1200 tiles is roughly 200MB. Every overworld map this
# project has measured is well inside that (Viridian City is 500).
DEFAULT_MAX_TILES = 1200


def _snapshot(pyboy):
    buf = io.BytesIO()
    pyboy.save_state(buf)
    return buf.getvalue()


def _restore(pyboy, data):
    buf = io.BytesIO(data)
    buf.seek(0)
    pyboy.load_state(buf)
    run_frames(pyboy, 2)


def _try_engage_trainer(pyboy, max_presses=12):
    """
    Press A repeatedly to see whether a blocked tile is a trainer, rather
    than a genuine wall.

    Confirmed empirically while building create_trainer_battle_states.py:
    unlike the wild encounters this project has handled everywhere else,
    walking toward a trainer's tile does not start their battle by
    itself -- it just blocks the move, the same as any other impassable
    tile. Only talking to them does, and their pre-battle line comes
    first, so the press has to repeat until the battle actually begins
    rather than checking once.
    """
    for _ in range(max_presses):
        press_button(pyboy, "a", hold_frames=12, release_frames=26)
        run_frames(pyboy, 20)
        if is_in_battle(pyboy):
            return get_battle_type(pyboy) == 2
    return False


WALL_RETRIES = 3


def _step(pyboy, direction, handle_battle=None, should_engage_trainer=None):
    """
    One tile move, fleeing any wild encounter it triggers.

    `handle_battle(pyboy)`, if given, is tried when a blocked move turns
    out to be a trainer rather than a wall (unlike a wild Pokemon, Gen 1
    does not allow fleeing one at all -- without a handler, a trainer is
    simply unreachable, which is correct for any map whose trainers can't
    yet be beaten). Probing for one is only attempted when a handler is
    given: pressing A into whatever is blocking the way is harmless for a
    trainer, but would have unwanted side effects elsewhere (opening a
    sign's text, talking to an unrelated NPC) on any map that doesn't
    have trainers to find this way. If the handler clears the battle, the
    step is retried once, since the block was the trainer's presence, not
    the tile itself.

    `should_engage_trainer(before_position, direction)`, if given, gates
    the probe itself. `_try_engage_trainer` spends up to 12 button presses
    of ~58 ticked frames each, which measures at roughly 0.2s of wall time
    (headless PyBoy runs about fifty times real-time, so its ~11s of
    *emulated* time is not 11s of anything a person waits for -- see
    tools/test_trainer_probe_cost.py, written after an earlier estimate
    here confused the two). That is negligible for a bounded one-shot BFS,
    where each tile's walls are only ever bumped once, but it adds up in RL
    training, where a near-random policy re-bumps the same walls tens of
    thousands of times per round.

    It receives the direction as well as the position because whether a
    probe is worth paying for depends on which way the blocked move went:
    a tile can be adjacent to a trainer on one side and plain wall on the
    others. Default (None) preserves the old always-probe behavior for
    callers like the survey that rely on it to find trainers with no prior
    knowledge of where they are.

    A move that still looks blocked after that gets a few more plain
    retries before it's accepted as a real wall. Every route and the
    forest surveyed so far only ever had *stationary* trainers blocking
    a tile, so one failed attempt was always conclusive there -- but
    Viridian City has ordinary pedestrian NPCs that wander on their own
    timer, and one can transiently stand in the way of a tile that is
    otherwise perfectly walkable. Caught directly: a heal trip's return
    path surveyed only 26 tiles out of Viridian City's north entrance,
    with every exit leading right back the way it came -- looking exactly
    like Route 22's real dead end, except a manual walk through the same
    spot moments later got blocked once and then succeeded on an
    immediate, otherwise-identical retry. One failure can't tell a real
    wall from an NPC that will have stepped aside a moment later.
    """
    before = get_player_position(pyboy)

    moved = walk_tile(pyboy, direction, verbose=False)
    run_frames(pyboy, 6)

    if is_in_battle(pyboy):
        attempt_run_from_wild_battle(pyboy)
    elif not moved and handle_battle is not None:
        if (
            should_engage_trainer is None
            or should_engage_trainer(before, direction)
        ) and _try_engage_trainer(pyboy):
            handle_battle(pyboy)
            moved = walk_tile(pyboy, direction, verbose=False)
            run_frames(pyboy, 6)

    if not moved and not is_in_battle(pyboy):
        for _ in range(WALL_RETRIES):
            run_frames(pyboy, 15)
            moved = walk_tile(pyboy, direction, verbose=False)
            run_frames(pyboy, 6)
            if moved:
                break

    # Crossing a map boundary (a door, or a route edge) hands control
    # back only after the game finishes auto-walking the player clear of
    # it -- the same settling problem controller.py hit at the house and
    # lab exits. Snapshotting mid-animation would store a position that
    # is about to change on its own.
    if get_player_position(pyboy)["map_id"] != before["map_id"]:
        wait_for_position_to_settle(pyboy)

    return moved


def walk_to(pyboy, predicate, max_tiles=DEFAULT_MAX_TILES, stay_on_map=True, handle_battle=None,
            heal_if_needed=None):
    """
    Search outward from the player's current position until reaching a
    tile where `predicate(position_dict)` is true, then leave the
    emulator standing on that tile.

    `stay_on_map=True` (the default) treats a map change as a wall: the
    search notes it and backs off rather than expanding through it. That
    keeps a search bounded to the current map, so looking for "the tile
    that exits to Route 1" doesn't wander off into Route 1 and start
    exploring that too. Pass False to search across map boundaries.

    `handle_battle`, if given, is passed through to `_step` -- see there
    for what it's for.

    `heal_if_needed(pyboy, (x, y))`, if given, is called once per tile
    dequeued for exploration, the same as `survey_map`'s parameter of the
    same name -- necessary for any search deep enough to need HP managed
    along the way, e.g. reaching a single faraway coordinate in Viridian
    Forest by refighting every trainer along whichever path the BFS takes
    there, the same as the main survey itself needs.

    Returns True and leaves the player at the target, or returns False
    and leaves the player where they started.
    """

    origin = _snapshot(pyboy)
    start = get_player_position(pyboy)

    if predicate(start):
        return True

    start_map = start["map_id"]
    start_key = (start["map_id"], start["x"], start["y"])

    states = {start_key: origin}
    seen = {start_key}
    queue = deque([start_key])

    while queue and len(seen) < max_tiles:
        key = queue.popleft()

        if heal_if_needed is not None:
            _restore(pyboy, states[key])
            if heal_if_needed(pyboy, key[1:]):
                states[key] = _snapshot(pyboy)

        for direction in DIRECTIONS:
            _restore(pyboy, states[key])
            moved = _step(pyboy, direction, handle_battle=handle_battle)
            position = get_player_position(pyboy)

            changed_map = position["map_id"] != start_map
            if not moved and not changed_map:
                continue
            if changed_map and stay_on_map:
                if predicate(position):
                    return True
                continue

            next_key = (position["map_id"], position["x"], position["y"])
            if next_key in seen:
                continue

            snapshot = _snapshot(pyboy)
            if predicate(position):
                return True

            seen.add(next_key)
            states[next_key] = snapshot
            queue.append(next_key)

    _restore(pyboy, origin)
    return False


def survey_map(pyboy, max_tiles=DEFAULT_MAX_TILES, on_visit=None, handle_battle=None,
               heal_if_needed=None):
    """
    Exhaustively flood-fill the map the player is standing on, without
    stepping off it, and report what is actually there:

        tiles -- every reachable (x, y) on this map
        exits -- {((x, y), direction): (map_id, x, y)} for each tile that
                 leads somewhere else

    `on_visit(pyboy, x, y)`, if given, is called once per newly reached
    tile while the emulator is actually standing on it -- which is what
    lets the map-panorama builder grab a screenshot of every tile rather
    than only the ones a random walk happened to cross.

    `handle_battle(pyboy)`, if given, is passed through to `_step` so a
    trainer occupying a tile can be fought and beaten rather than simply
    read as a wall -- see `_step` for why that only applies to trainer
    battles, never wild ones.

    `heal_if_needed(pyboy, key)`, if given, is called once per tile
    dequeued for exploration, before any of its four directions are
    tried, and should return True if it took the party away to heal (in
    which case its snapshot is refreshed to the now-healed state before
    continuing). Needed for any map with more than one or two trainers to
    fight through: nothing else here manages HP between battles, and a
    policy measured only at full HP can lose fights it would otherwise
    win once several have chipped away at it in a row.

    If the search finishes before hitting `max_tiles`, the result is
    complete: anything absent from `exits` genuinely is not reachable
    from here. That is the whole point -- it is what turned "we never
    found Viridian's north exit" into "Viridian has no north exit", and
    the same distinction would have caught the Route 22 mix-up
    immediately (a route with no forward exit is obvious in one survey,
    but invisible across 1500 training episodes).

    Leaves the player where they started.
    """

    origin = _snapshot(pyboy)
    start = get_player_position(pyboy)
    start_map = start["map_id"]
    start_key = (start["x"], start["y"])

    states = {start_key: origin}
    tiles = {start_key}
    exits = {}
    queue = deque([start_key])

    if on_visit is not None:
        on_visit(pyboy, start["x"], start["y"])

    while queue and len(tiles) < max_tiles:
        key = queue.popleft()

        if heal_if_needed is not None:
            _restore(pyboy, states[key])
            if heal_if_needed(pyboy, key):
                states[key] = _snapshot(pyboy)

        for direction in DIRECTIONS:
            _restore(pyboy, states[key])
            moved = _step(pyboy, direction, handle_battle=handle_battle)
            position = get_player_position(pyboy)

            if position["map_id"] != start_map:
                exits[(key, direction)] = (
                    position["map_id"],
                    position["x"],
                    position["y"],
                )
                continue
            if not moved:
                continue

            next_key = (position["x"], position["y"])
            if next_key not in tiles:
                tiles.add(next_key)
                states[next_key] = _snapshot(pyboy)
                queue.append(next_key)
                if on_visit is not None:
                    on_visit(pyboy, position["x"], position["y"])

    complete = not queue
    _restore(pyboy, origin)
    return tiles, exits, complete


def walk_to_tile(pyboy, x, y, **kwargs):
    return walk_to(pyboy, lambda p: p["x"] == x and p["y"] == y, **kwargs)


def walk_to_map(pyboy, map_id, **kwargs):
    """
    Cross into `map_id`, e.g. stepping through a building door or a route
    boundary.

    Keeps the default stay_on_map=True on purpose: the search should
    hunt for the doorway across the *current* map and step through it,
    not treat every other neighbouring map as more territory to explore.
    That means the target map has to border the current one -- callers
    chain one leg at a time rather than asking for a destination several
    maps away.
    """
    return walk_to(pyboy, lambda p: p["map_id"] == map_id, **kwargs)
