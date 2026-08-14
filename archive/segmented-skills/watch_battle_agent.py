from stable_baselines3 import DQN

from envs.battle_env import PokemonRedRivalBattleEnv
from core.config import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "rival_battle_dqn.zip"

# Success bar from the project's design notes: a policy is considered
# "good enough to move on" at >=90% wins over 100 greedy (no exploration)
# evaluation episodes.
TARGET_WIN_RATE = 0.90
NUM_EVAL_EPISODES = 100


def evaluate(num_episodes=NUM_EVAL_EPISODES, verbose=True):
    env = PokemonRedRivalBattleEnv(max_steps=30)
    model = DQN.load(str(MODEL_PATH))

    wins = 0
    losses = 0
    truncated_count = 0

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        result = "truncated"

        for _ in range(30):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))

            if terminated:
                if info["after"]["enemy_mon_hp"] == 0:
                    result = "win"
                elif info["after"]["battle_mon_hp"] == 0:
                    result = "loss"
                break

            if truncated:
                break

        if result == "win":
            wins += 1
        elif result == "loss":
            losses += 1
        else:
            truncated_count += 1

        if verbose:
            print(f"Episode {episode:3d}: {result}")

    env.close()

    win_rate = wins / num_episodes
    print()
    print(f"Wins: {wins}/{num_episodes}  Losses: {losses}  Truncated: {truncated_count}")
    print(f"Win rate: {win_rate:.1%} (target: {TARGET_WIN_RATE:.0%})")

    return win_rate


if __name__ == "__main__":
    evaluate()
