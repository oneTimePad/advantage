import gym
from .environment import Environment

class GymEnvironment(Environment):
    """ Integrates the OpenAI Gym Environments
    """
    def __init__(self, environment_name):
        self._gym = gym.make(environment_name)

    def step(self, action):
        next_state, reward, done, _ = self._gym.step(action)
        return (next_state, reward, done)

    def reset(self):
        return self._gym.reset()
