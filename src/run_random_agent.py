import random

from simple_env import PokemonRedLeaveHouseEnv
from actions import num_actions


def run_episode(max_steps=200):
    """
    Run one episode with random actions.

    This is not intelligent yet.
    It is just a test that our environment works.
    """

    env = PokemonRedLeaveHouseEnv(max_steps=max_steps)

    try:
        observation = env.reset()

        print("Starting random-agent episode.")
        print(f"Initial observation: {observation}")
        print()

        total_reward = 0.0

        for step in range(max_steps):
            action_id = random.randrange(num_actions())

            observation, reward, done, info = env.step(action_id)

            total_reward += reward

            print(
                f"Step {step + 1:03d} | "
                f"Action: {info['direction']:>5} | "
                f"Position: map={observation['map_id']}, "
                f"x={observation['x']}, y={observation['y']} | "
                f"Reward: {reward:.2f} | "
                f"Total: {total_reward:.2f}"
            )

            if done:
                print()
                print("Episode finished.")
                print(f"Reached goal: {info['reached_goal']}")
                print(f"Total reward: {total_reward:.2f}")
                print(f"Steps: {info['step_count']}")
                break

    finally:
        env.close()


if __name__ == "__main__":
    run_episode(max_steps=200)