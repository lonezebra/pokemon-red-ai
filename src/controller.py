"""
Hand-written controller: chains individually trained skills together
through one uniform interface, rather than each skill needing its own
special-cased driving code.

What this actually does right now is run the two skills that exist so
far, each through skill.choose_action(observation) -> action, each
starting from its own save state:

    saves/bedroom.state      -> [leave-house Q-agent] -> Pallet Town
    saves/rival_battle.state -> [rival-battle DQN]     -> win/loss

These two segments are NOT chained into one continuous run yet. The
missing piece is the middle of the game's opening: walking to Professor
Oak, choosing a starter, and walking back to where the rival stops the
player. That was done by hand on the project owner's machine (see the
original handoff notes) but hasn't been ported into this repo as a
scripted route yet.

Once that route exists, this is the natural place to wire it in as a
third segment, so a single run can go:

    bedroom.state -> Q-agent -> Pallet Town
                  -> scripted route -> Oak -> starter -> rival trigger
                  -> DQN -> win

with the hand-off condition at each step being something already proven
in this project (a map_id check, or the battle-flag check from
memory.is_in_battle) -- script the hand-offs, learn the behavior in
between.
"""

from envs.simple_env import PokemonRedLeaveHouseEnv
from envs.battle_env import PokemonRedRivalBattleEnv
from agents.skills import LeaveHouseSkill, RivalBattleSkill
from core.config import PROJECT_ROOT


def run_leave_house_segment(max_steps=200):
    print()
    print("Segment 1: bedroom.state -> leave-house Q-agent -> Pallet Town")
    print("-" * 62)

    env = PokemonRedLeaveHouseEnv(max_steps=max_steps)
    skill = LeaveHouseSkill(PROJECT_ROOT / "models" / "leave_house_q_table.json")

    obs = env.reset()
    info = {}

    for _ in range(max_steps):
        action = skill.choose_action(obs)
        obs, reward, done, info = env.step(action)

        if done:
            break

    env.close()

    reached_goal = info.get("reached_goal", False)

    if reached_goal:
        print(f"Reached Pallet Town in {info['step_count']} steps.")
    else:
        print("Did not reach Pallet Town within the step limit.")

    return reached_goal


def run_rival_battle_segment(max_steps=30):
    print()
    print("Segment 2: rival_battle.state -> rival-battle DQN -> win/loss")
    print("-" * 62)

    env = PokemonRedRivalBattleEnv(max_steps=max_steps)
    skill = RivalBattleSkill(PROJECT_ROOT / "models" / "rival_battle_dqn.zip")

    obs, info = env.reset()

    for _ in range(max_steps):
        action = skill.choose_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    env.close()

    won = info["after"]["enemy_mon_hp"] == 0

    print("Won the rival battle." if won else "Did not win the rival battle.")

    return won


def main():
    print("Pokemon Red AI -- controller")
    print()
    print("Running each currently-trained skill through the same")
    print("choose_action(observation) -> action interface. See this")
    print("file's module docstring for why these are two separate")
    print("segments rather than one continuous run, for now.")

    leave_house_ok = run_leave_house_segment()
    battle_ok = run_rival_battle_segment()

    print()
    print("Summary")
    print("-" * 62)
    print(f"Leave-house segment:  {'OK' if leave_house_ok else 'FAILED'}")
    print(f"Rival-battle segment: {'OK' if battle_ok else 'FAILED'}")


if __name__ == "__main__":
    main()
