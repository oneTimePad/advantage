import numpy as np
from functools import partial
from utils.policy_tools import epsilon_greedy

class DeepQAgent(ActionValueAgent, DiscreteActionSpaceAgent):
    """ Implements the DeepQNetworks Agent. The DQN Agent
    utilizes two networks to stabilize Q-Learning for Deep RL approximators.
    A ExperienceReplayBuffer is utilize to allow for the I.I.D necessary condition
    for Neural Networks.
    """

    def __init__(self, session, actor_q_network, target_q_network, environment):
        maximum_fn = partial(np.amax, axis=1)
        self._session = session

        ActionValueAgent.__init__(self, actor_q_network, environment, maximum_fn)
        DiscreteActionSpaceAgent.__init__(self, actor_q_network, environment)

    def set_up(self):
        pass

    def evaluate_policy(self, state):
        pass

    def improve_policy(self, sarsa_samples):
        pass

    def improve_target(self, sarsa_samples, targets):
        """ Trains the Target Q-Network on I.I.D samples from the Replay Buffer
                Args:
                    sarsa_samples: collection of sarsa samples
                    targets: the regression targets
        """
        pass
