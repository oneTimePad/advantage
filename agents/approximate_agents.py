import numpy as np
from functools import partial
import tensorflow as tf
from utils.policy_tools import epsilon_greedy
from .base_agents import LearningAgent, ActionValueAgent, DiscreteActionSpaceAgent
from utils.values import apply_bellman_operator
class ApproximateAgent(LearningAgent):
    """Approximate Learning Agent"""
    def __init__(self, policy, environment, discount_factor, graph, session, **kwargs):
        self._graph = graph
        self._session = session

        super().__init__(policy=policy,
                        environment=environment,
                        discount_factor=discount_factor,
                        **kwargs)

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

    def __init__(self, graph,
                session,
                environment,
                discount_factor,
                policy_q_network,
                target_q_network,
                epsilon):

        maximum_fn = partial(np.argmax, axis=1)

        self._target = target_q_network
        self._epsilon = epsilon

        super().__init__(policy=policy_q_network,
                        environment=environment,
                        graph=graph,
                        session=session,
                        discount_factor=discount_factor,
                        maximum_function=maximum_fn)

    @property
    def epsilon(self):
        return self._epsilon

    def set_up(self):

        target_net = self._target.network

        with self._graph.as_default():
            # the bellman operator targets for training target network
            target_plh = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="target_plh")
            self._target.add_target_placeholder(target_plh)

            # the actions taken by the policy leading to the bellman transition
            action_taken_plh = tf.placeholder(shape=[None, 1], dtype=tf.int32, name="action_taken_plh")
            self._target.add_target_placeholder(action_taken_plh)

            # the mean square error between target and network
            loss = tf.reduce_mean(tf.square(target_plh - tf.reduce_sum(tf.one_hot(indices=action_taken_plh, depth=self.num_of_actions) * target_net, axis=1)))

        gradients = self._target.gradients(loss)
        self._target.apply_gradients(gradients)

        self._policy.initialize(self._session)
        self._target.initialize(self._session)

    def evaluate_policy(self, state):
        return self._policy.inference(self._session, {"policy_state_plh" : [state]})

    def improve_policy(self, sarsa_samples):
        """Policy is improved by copying target params to policy network"""
        target_params = self._target.trainable_parameters_dict\

        #with self._session.as_default():
        target_params_runtime = self._session.run(target_params)

        target_params_runtime = {self._policy.strip_and_replace_scope(k) : v for k,v in target_params_runtime.items()}
        self._policy.copy(self._session, target_params_runtime)

    def improve_target(self, sarsa_samples):
        """ Trains the Target Q-Network on I.I.D samples batch from the Replay Buffer
                Args:
                    sarsa_samples: list of Sarsa samples
        """
        states, actions_taken, targets = apply_bellman_operator(self._session, self._policy, sarsa_samples, self._discount_factor, "policy_state_plh")

        feed_dict_in = {"tgt_state_plh" : states}

        feed_dict_target = {"target_plh" : targets, "action_taken_plh": actions_taken}

        self._target.update(self._session, feed_dict_in, feed_dict_target)


    @epsilon_greedy
    def sample_action(self, conditional_policy, training):
        """Adds Epsilon-Greedy acting during training"""
        return super(DeepQAgent, self).sample_action(conditional_policy, training)
