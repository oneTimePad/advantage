from .environment import Environment

class MockEnvironment(Environment):
    """ Mocks a Random MDP with a deterministic one
    """
    def __init__(self, initial_state, state_step, constant_reward=1.0):
        self._state = initial_state.copy()
        self._initial_state = initial_state
        self._state_step = state_step
        self._constant_reward = constant_reward

    def step(self, action):
        self._state += self._state_step
        return self._state, self._constant_reward, 0

    def reset(self):
        self._state = self._initial_state
        return self._state
