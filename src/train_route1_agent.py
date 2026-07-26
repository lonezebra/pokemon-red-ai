from envs.route1_env import PokemonRedRoute1Env
from agents.q_learning_agent import QLearningAgent
from actions import num_actions
from core.config import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "route1_q_table.json"


def main(num_episodes=1500, max_steps=800):
    # A first attempt at this used max_steps=150 (matching leave-house's
    # cap) and ran for 1000+ episodes without a single success, even once
    # epsilon had decayed low enough that the agent was mostly exploiting
    # rather than exploring -- a sign the goal was simply never reached,
    # not that it wasn't learned yet. Scouting the route by hand (an
    # up-biased random walk) confirmed why: reaching Viridian City takes
    # around 670 steps, more than 3x that budget. Since Q-learning can
    # only learn from the goal reward if an episode actually reaches it,
    # a too-small step cap means the +100 reward never enters the table
    # at all, no matter how many episodes run. 800 gives real headroom
    # above that 670-step reference; episode count was reduced from an
    # initial 4000 to keep total training time from ballooning now that
    # each episode can run much longer.
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
