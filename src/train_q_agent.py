from envs.simple_env import PokemonRedLeaveHouseEnv
from agents.q_learning_agent import QLearningAgent
from actions import num_actions
from core.config import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "leave_house_q_table.json"


def main(num_episodes=2000, max_steps=200):
    # 500 episodes was tried first and looked like it worked (successes
    # climbing during training) but the trained policy actually failed
    # every single evaluation episode, stuck repeating a wall-bump action
    # at one downstairs tile. Q-learning only updates a state when the
    # agent actually passes through it, and reaching downstairs was itself
    # rare early on, so everything past it barely got trained -- its
    # Q-values were nearly identical across all four actions (undecided
    # noise, not a real preference). 2000 episodes, combined with a slower
    # epsilon decay (see QLearningAgent) so exploration keeps reaching
    # those deeper states for longer, converges to a policy that wins
    # reliably instead of just sometimes during training.
    env = PokemonRedLeaveHouseEnv(max_steps=max_steps)
    agent = QLearningAgent(num_actions=num_actions())

    successes = 0
    best_steps = None

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        total_reward = 0.0
        info = {}

        for _ in range(max_steps):
            action = agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action)

            agent.update(obs, action, reward, next_obs, done)

            obs = next_obs
            total_reward += reward

            if done:
                break

        agent.decay_epsilon()

        if info.get("reached_goal"):
            successes += 1
            if best_steps is None or info["step_count"] < best_steps:
                best_steps = info["step_count"]

        if episode % 50 == 0:
            print(
                f"Episode {episode:4d}/{num_episodes}  "
                f"epsilon={agent.epsilon:.3f}  "
                f"successes so far: {successes}/{episode}  "
                f"last reward: {total_reward:.2f}"
            )

    print()
    print(f"Total successes: {successes}/{num_episodes}")
    if best_steps is not None:
        print(f"Best successful episode steps: {best_steps}")

    agent.save(MODEL_PATH)
    print(f"Saved Q-table to {MODEL_PATH}")

    env.close()


if __name__ == "__main__":
    main()
