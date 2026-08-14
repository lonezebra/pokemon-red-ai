import json

from core.atomic_io import write_json_atomic
import random


class QLearningAgent:
    """
    A tabular Q-learning agent.

    The Q-table maps a (state, action) pair to the expected future reward
    of taking that action from that state. It starts empty -- every state
    is worth 0 until the agent actually visits it -- and gets filled in
    gradually from trial and error using the standard Q-learning update:

        Q(s, a) += learning_rate * (reward + discount * max(Q(s', .)) - Q(s, a))

    State here is just (map_id, x, y), matching the observation returned
    by envs/simple_env.py.
    """

    def __init__(
        self,
        num_actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.998,  # see train_q_agent.py for why this isn't faster
    ):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = {}

    def _state_key(self, observation):
        return (observation["map_id"], observation["x"], observation["y"])

    def _action_values(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.num_actions

        return self.q_table[state_key]

    def choose_action(self, observation, greedy=False):
        """
        Pick an action for the given observation.

        With probability epsilon, pick a random action instead of the
        best-known one -- this is what lets the agent discover things it
        hasn't tried yet. Pass greedy=True (no randomness) to see what the
        agent has actually learned, e.g. when watching a trained agent.
        """

        action_values = self._action_values(self._state_key(observation))

        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        best_value = max(action_values)
        best_actions = [a for a, value in enumerate(action_values) if value == best_value]
        return random.choice(best_actions)

    def update(self, observation, action, reward, next_observation, done):
        action_values = self._action_values(self._state_key(observation))
        next_action_values = self._action_values(self._state_key(next_observation))

        best_next_value = 0.0 if done else max(next_action_values)
        target = reward + self.discount_factor * best_next_value

        action_values[action] += self.learning_rate * (target - action_values[action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        # JSON object keys must be strings, so (map_id, x, y) becomes
        # "map_id,x,y" and is parsed back apart in load().
        serializable = {f"{k[0]},{k[1]},{k[2]}": v for k, v in self.q_table.items()}

        path.parent.mkdir(exist_ok=True)
        # Atomic: tools/checkpoint_artifacts.sh commits this file while
        # training is still running, so a plain write leaves a window where
        # the committed checkpoint is truncated and can't be resumed from.
        write_json_atomic(path, serializable, indent=2)

    def load(self, path):
        with open(path) as f:
            serializable = json.load(f)

        self.q_table = {}
        for key_str, values in serializable.items():
            map_id, x, y = (int(part) for part in key_str.split(","))
            self.q_table[(map_id, x, y)] = values
