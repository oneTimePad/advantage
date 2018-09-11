import random
import numpy as np

def epsilon_greedy(fn):
    """ Decorator for acting epsilong greedily
            Args:
                fn: action sampling method

            Returns:
                wrapped function

    """

    def act(self, conditional_policy, training):


        if not hasattr(self, "num_of_actions"):
            raise ValueError("decorator expects object to have attribute num_of_actions")
        
        if not hasattr(self, "epsilon"):
            raise ValueError("decorator expects object to have attribute epsilon")

        action = fn(self, conditional_policy, training)

        if self.epsilon > 1.0 or self.epsilon < 0:
            raise ValueError("Epsilon must be in [0, 1.0]")

        if training:
            num_actions = self.num_of_actions
            prob = random.random()
            return action if prob > self.epsilon else np.random.sample(num_actions, 1)
        else:
            return action

    return act
