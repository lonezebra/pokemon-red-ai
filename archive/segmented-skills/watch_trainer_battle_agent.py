from stable_baselines3 import DQN

from envs.trainer_battle_env import PokemonRedTrainerBattleEnv
from core.config import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "trainer_battle_dqn.zip"

# Unlike wild encounters, there is no fleeing here -- an episode ends in
# a win or a loss. The bar is the project's usual >=90% for a battle
# policy, the same one the rival battle DQN was held to.
TARGET_WIN_RATE = 0.90
NUM_EVAL_EPISODES = 100
MAX_STEPS = 60


def evaluate(num_episodes=NUM_EVAL_EPISODES, verbose=True):
    env = PokemonRedTrainerBattleEnv(max_steps=MAX_STEPS)
    model = DQN.load(str(MODEL_PATH))

    wins = losses = truncated_count = 0

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        result = "truncated"

        for _ in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            if terminated:
                result = "win" if info["won"] else "loss"
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
