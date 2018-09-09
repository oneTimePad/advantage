import numpy as np
from functools import partial
from utils.policy_tools import epsilon_greedy
from .base_agents import LearningAgent, ActionValueAgent, DiscreteActionSpaceAgent

class ApproximateAgent(LearningAgent):
    """Approximate Learning Agent"""
    def __init__(self, policy, environment, graph, session, **kwargs):
        self._graph = graph
        self._session = session
        super(ApproximateAgent, self).__init__(policy=policy, environment=environment, **kwargs)

    @property
    def graph(self):
        return self._graph

    @property
    def session(self):
        return self._session

class DeepQAgent(ApproximateAgent, DiscreteActionSpaceAgent, ActionValueAgent):
    """ Implements the DeepQNetworks Agent. The DQN Agent
    utilizes two networks to stabilize Q-Learning for Deep RL approximators.
    A ExperienceReplayBuffer is utilize to allow for the I.I.D necessary condition
    for Neural Networks.
    """

    def __init__(self, graph, session, policy_q_network, target_q_network, environment):
        maximum_fn = partial(np.amax, axis=1)

        super(DeepQAgent, self).__init__(policy=policy_q_network,
                                            environment=environment,
                                            graph=graph,
                                            session=session,
                                            maximum_function=maximum_fn)


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
