import random
import numpy as np

def epsilon_greedy(fn, action_space, epsilon):
    """ Decorator for acting epsilong greedily
            Args:
                fn: action sampling method
                action_space: the action_space to randomly sample from
                epsilon: the sampling threshold

            Returns:
                wrapped function

            Raises:
                ValueError: for invalid epsilon
    """
    if epsilon > 1.0 or epsilon < 0:
        raise ValueError("Epsilon must be in [0, 1.0]")

    def act(conditional_policy, training):
        action = fn(conditional_policy)
        if training:
            prob = random.random()
            return action if prob > epsilon else np.random.sample(action_space, 1)
        else:
            return action

    return act
