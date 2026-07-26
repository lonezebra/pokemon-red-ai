from envs.route1_env import PokemonRedRoute1Env
from agents.q_learning_agent import QLearningAgent
from actions import num_actions
from core.config import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "route1_q_table.json"


def main(num_episodes=4000, max_steps=150):
    # Route 1 is a much longer corridor than the leave-house task (roughly
    # 35 tiles from the Pallet Town entrance to Viridian City, versus a
    # ~19-step optimal path out of the bedroom), so it needs a larger step
    # budget per episode and more episodes to give the deeper/further-away
    # tiles enough visits to actually be learned -- the same lesson the
    # leave-house agent's under-training bug taught: a state only gets
    # updated when the agent actually passes through it.
    env = PokemonRedRoute1Env(max_steps=max_steps)
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
