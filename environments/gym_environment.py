import gym
import numpy as np
from advantage.environments.environment import Environment

class GymEnvironment(Environment):
    """ Integrates the OpenAI Gym Environments
    """
    def __init__(self, environment_name):
        self._gym = gym.make(environment_name)

        state = self._gym.reset()
        state_col_dim = state.shape[0] if isinstance(state, np.ndarray) else 1
        action = self._gym.action_space.sample()
        action_col_dim = action.shape[0] if isinstance(action, np.ndarray) else 1
        _, reward, __, ___ = self._gym.step(action)
        reward_col_dim = reward.shape[0] if isinstance(reward, np.ndarray) else 1

        self._dims = {"state_col_dim": state_col_dim, "action_col_dim": action_col_dim,\
            "reward_col_dim": reward_col_dim, "next_state_col_dim": state_col_dim} # TODO state/next_state is redundant

    def step(self, action):
        next_state, reward, done, _ = self._gym.step(action)
        return (next_state, reward, done)

    def reset(self):
        return self._gym.reset()

    @property
    def action_space(self):
        return self._gym.action_space

    @property
    def dims(self):
        return self._dims
