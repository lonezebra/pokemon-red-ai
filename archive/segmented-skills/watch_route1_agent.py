from envs.route1_env import PokemonRedRoute1Env
from agents.q_learning_agent import QLearningAgent
from actions import num_actions
from core.config import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "route1_q_table.json"


def main(num_episodes=30, max_steps=150):
    env = PokemonRedRoute1Env(max_steps=max_steps)
    agent = QLearningAgent(num_actions=num_actions())
    agent.load(MODEL_PATH)

    successes = 0

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        info = {}

        for _ in range(max_steps):
            action = agent.choose_action(obs, greedy=True)
            obs, reward, done, info = env.step(action)

            if done:
                break

        result = "reached Viridian City" if info.get("reached_goal") else "did not finish"
        print(
            f"Episode {episode:2d}: {result} in {info['step_count']} steps "
            f"({info['encounters']} wild encounter(s))"
        )

        if info.get("reached_goal"):
            successes += 1

    print()
    print(f"Wins: {successes}/{num_episodes}")

    env.close()


if __name__ == "__main__":
    main()
