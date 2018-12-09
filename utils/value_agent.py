import tensorflow as tf
import numpy as np
import attr
from advantage.elements import Sarsa
from advantage.utils.tf_utils import get_or_create_improve_step
import advantage.loggers as loggers

""" Computes various RL special values
"""



def bellman_operator(session, policy, sarsa, gamma, state_plh_name):
    """ Computes the target part of Q-Learning based Advantage function
            Args:
                session: session
                policy: tgt network
                sarsa: Sarsa object
                gamme: one-step discount factor
                state_plh_name: the name of the placeholder of the state input to network

            Returns:
                list of states before bellman transition
                list of actions leading to the bellman transition
                a numpy array of targets

            Raises:
                ValueError: for bad arguments
    """

    if not isinstance(session, tf.Session):
        raise ValueError("Must pass in a tf.Session object")

    states, actions, rewards, dones, next_states, _ = sarsa.unzip_to_tuple()

    q_values = policy.inference(session, {state_plh_name: next_states})

    max_qs = np.expand_dims(np.amax(q_values, axis=1), axis=1)

    return states, actions, rewards + gamma * np.invert(dones) * max_qs

@attr.s
class Epsilon:
    """ Holds attrs for defining
    decaying epsilon
    """
    init = attr.ib(kw_only=True)
    min = attr.ib(kw_only=True)
    decay_steps = attr.ib(kw_only=True)
    decay_rate = attr.ib(kw_only=True)

    @classmethod
    def from_config(cls, config):
        """ Creates from configuration

                Args:
                    config: protobuf config

                Returns:
                    Epsilon
        """
        return cls(init=config.initial_value,
                   min=config.min_value,
                   decay_steps=config.decay_steps,
                   decay_rate=config.decay_rate)


def decayed_epsilon(agent,
                    epsilon):
    """ Constructs a callable that returns
    the decayed epsilon

        Args:
            agent: agent who has session
            scope: scope to make vars/ops in
            epsilon: epsilon config

        Returns:
            callable that returns epsilon value
    """
    improve_step = get_or_create_improve_step(agent.agent_scope)
    with agent.agent_scope():
        eps = tf.train.exponential_decay(epsilon.init,
                                         improve_step,
                                         epsilon.decay_steps,
                                         epsilon.decay_rate,
                                         staircase=True,
                                         name="decayed_epsilon")
    min_epsilon = epsilon.min

    @loggers.value("Agent current epsilon is %.2f",
                   loggers.LogVarType.RETURNED_VALUE,
                   "epsilon",
                   tensorboard=True)
    def fetch_eps():
        nonlocal agent, min_epsilon

        eps_runtime = agent.session.run(eps)

        return eps_runtime if eps_runtime > min_epsilon else min_epsilon

    return fetch_eps
