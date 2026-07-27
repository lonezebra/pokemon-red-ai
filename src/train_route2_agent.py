import json
import math

from envs.route2_env import PokemonRedRoute2Env
from agents.q_learning_agent import QLearningAgent
from actions import num_actions
from core.config import PROJECT_ROOT
from core.screen import save_gif

MODEL_PATH = PROJECT_ROOT / "models" / "route2_q_table.json"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "route2_checkpoint.json"

DEMO_MAX_FRAMES = 300
WARM_START_EPSILON = 0.3


def save_checkpoint(episode, agent, successes, best_steps, best_demo_key):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(
            {
                "episode": episode,
                "epsilon": agent.epsilon,
                "successes": successes,
                "best_steps": best_steps,
                "best_demo_key": list(best_demo_key) if best_demo_key is not None else None,
            },
            f,
        )


def load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        return None

    with open(CHECKPOINT_PATH) as f:
        data = json.load(f)

    if data["best_demo_key"] is not None:
        data["best_demo_key"] = tuple(data["best_demo_key"])

    return data


def run_demo_episode(env, agent, max_steps):
    obs = env.reset()
    frames = [env.pyboy.screen.image.copy()]
    info = {}

    for _ in range(max_steps):
        action = agent.choose_action(obs, greedy=True)
        obs, reward, done, info = env.step(action)
        frames.append(env.pyboy.screen.image.copy())

        if done:
            break

    if len(frames) > DEMO_MAX_FRAMES:
        stride = math.ceil(len(frames) / DEMO_MAX_FRAMES)
        frames = frames[::stride]

    return {
        "frames": frames,
        "reached_goal": info.get("reached_goal", False),
        "tiles_visited": len(env.visited_positions),
        "steps": info.get("step_count", 0),
    }


def main(num_episodes=1500, max_steps=1000):
    # Route 2's actual scale isn't as well understood as Route 1's was
    # before training started (see rewards/route2_rewards.py) -- 1000 is
    # a generous-but-uncertain guess, not a verified reference distance
    # the way Route 1's 800 (vs. a scouted ~670-step path) was. Worth
    # revisiting based on what training itself shows.
    env = PokemonRedRoute2Env(max_steps=max_steps)
    agent = QLearningAgent(num_actions=num_actions())

    successes = 0
    best_steps = None
    best_demo_key = None
    start_episode = 1

    checkpoint = load_checkpoint()
    if checkpoint is not None and MODEL_PATH.exists():
        agent.load(MODEL_PATH)
        agent.epsilon = checkpoint["epsilon"]
        successes = checkpoint["successes"]
        best_steps = checkpoint["best_steps"]
        best_demo_key = checkpoint["best_demo_key"]
        start_episode = checkpoint["episode"] + 1
        print(
            f"Resuming from checkpoint: episode {checkpoint['episode']}, "
            f"epsilon={agent.epsilon:.3f}, successes so far: {successes}"
        )
    elif MODEL_PATH.exists():
        agent.load(MODEL_PATH)
        agent.epsilon = WARM_START_EPSILON
        print(f"Warm-starting from existing Q-table at {MODEL_PATH}, epsilon={agent.epsilon:.3f}")

    for episode in range(start_episode, num_episodes + 1):
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

        if episode % 100 == 0:
            demo = run_demo_episode(env, agent, max_steps)
            save_gif(demo["frames"], f"route2_progress_ep{episode:04d}.gif")

            demo_key = (demo["reached_goal"], demo["tiles_visited"], -demo["steps"])
            print(
                f"  [demo] reached_goal={demo['reached_goal']} "
                f"tiles_visited={demo['tiles_visited']} steps={demo['steps']}"
            )

            if best_demo_key is None or demo_key > best_demo_key:
                best_demo_key = demo_key
                save_gif(demo["frames"], "route2_best_so_far.gif")
                print(f"  [demo] new best so far (episode {episode})")

            agent.save(MODEL_PATH)
            save_checkpoint(episode, agent, successes, best_steps, best_demo_key)

    print()
    print(f"Total successes: {successes}/{num_episodes}")
    if best_steps is not None:
        print(f"Best successful episode steps: {best_steps}")

    agent.save(MODEL_PATH)
    print(f"Saved Q-table to {MODEL_PATH}")

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    env.close()


if __name__ == "__main__":
    main()
