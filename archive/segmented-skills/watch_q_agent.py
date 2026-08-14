from envs.simple_env import PokemonRedLeaveHouseEnv
from agents.q_learning_agent import QLearningAgent
from actions import num_actions, get_action_name
from core.config import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "leave_house_q_table.json"


def main(num_episodes=10, max_steps=200):
    env = PokemonRedLeaveHouseEnv(max_steps=max_steps)
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

        result = "reached Pallet Town" if info.get("reached_goal") else "did not finish"
        print(f"Episode {episode:2d}: {result} in {info['step_count']} steps")

        if info.get("reached_goal"):
            successes += 1

    print()
    print(f"Wins: {successes}/{num_episodes}")

    env.close()


if __name__ == "__main__":
    main()
