from envs.route2_env import PokemonRedRoute2Env
from agents.q_learning_agent import QLearningAgent
from actions import num_actions
from core.config import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "route2_q_table.json"

NUM_EVAL_EPISODES = 50
MAX_STEPS = 600


def evaluate(num_episodes=NUM_EVAL_EPISODES, verbose=True):
    env = PokemonRedRoute2Env(max_steps=MAX_STEPS)
    agent = QLearningAgent(num_actions=num_actions())
    agent.load(MODEL_PATH)

    successes = 0
    step_counts = []

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        info = {}

        for _ in range(MAX_STEPS):
            action = agent.choose_action(obs, greedy=True)
            obs, _, done, info = env.step(action)
            if done:
                break

        reached = info.get("reached_goal", False)
        if reached:
            successes += 1
            step_counts.append(info["step_count"])

        if verbose:
            result = "reached the Viridian Forest gate" if reached else "did not finish"
            print(
                f"Episode {episode:2d}: {result} in {info['step_count']} steps "
                f"({info['encounters']} wild encounter(s))"
            )

    env.close()

    print()
    print(f"Wins: {successes}/{num_episodes}")
    if step_counts:
        print(
            f"Steps on successful runs: best {min(step_counts)}, "
            f"worst {max(step_counts)}, mean {sum(step_counts)/len(step_counts):.1f}"
        )

    return successes / num_episodes


if __name__ == "__main__":
    evaluate()
