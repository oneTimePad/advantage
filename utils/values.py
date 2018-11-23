import tensorflow as tf
import numpy as np
from advantage.elements import Sarsa

""" Computes various RL special values """



def apply_bellman_operator(session, policy, sarsa, gamma, state_plh_name):
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

    if not isinstance(sarsa, Sarsa):
        raise ValueError("sarsa_split must be a tuple of np.ndarrays")

    states, actions, rewards, dones, next_states, next_actions = sarsa.unzip_to_tuple()

    if state_plh_name not in policy.feed_dict_keys:
        raise ValueError("%s is not in network feed_dict_keys" % state_plh_name)

    q_values = policy.inference(session, {state_plh_name: next_states})

    max_qs = np.expand_dims(np.amax(q_values, axis=1), axis=1)

    return states, actions, rewards + gamma * np.invert(dones) * max_qs
