from stable_baselines3 import DQN

from envs.wild_battle_env import PokemonRedWildBattleEnv
from core.config import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "wild_battle_dqn.zip"

# Route 1's actual wild encounters (Pidgey/Rattata, both level 2-3) are
# trivially weak next to a level 6 Squirtle with real moves -- there's no
# good reason to ever need to flee one. So unlike the rival battle (where
# winning is the only good outcome, since running isn't legal there), the
# bar here is on *not losing* (win or successful flee both count), not on
# winning specifically -- fleeing well only actually matters once this
# environment faces a wild Pokemon actually worth avoiding, which Route 1
# alone doesn't provide.
TARGET_NOT_LOST_RATE = 0.95
NUM_EVAL_EPISODES = 100


def evaluate(num_episodes=NUM_EVAL_EPISODES, verbose=True):
    env = PokemonRedWildBattleEnv(max_steps=30)
    model = DQN.load(str(MODEL_PATH))

    wins = 0
    losses = 0
    fled = 0
    truncated_count = 0

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        result = "truncated"

        for _ in range(30):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))

            if terminated:
                if info["won"]:
                    result = "win"
                elif info["lost"]:
                    result = "loss"
                else:
                    result = "fled"
                break

            if truncated:
                break

        if result == "win":
            wins += 1
        elif result == "loss":
            losses += 1
        elif result == "fled":
            fled += 1
        else:
            truncated_count += 1

        if verbose:
            print(f"Episode {episode:3d}: {result}")

    env.close()

    not_lost_rate = (wins + fled) / num_episodes
    print()
    print(f"Wins: {wins}/{num_episodes}  Losses: {losses}  Fled: {fled}  Truncated: {truncated_count}")
    print(f"Not-lost rate: {not_lost_rate:.1%} (target: {TARGET_NOT_LOST_RATE:.0%})")

    return not_lost_rate


if __name__ == "__main__":
    evaluate()
