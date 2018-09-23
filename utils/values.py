import tensorflow as tf
import numpy as np


""" Computes various RL special values """



def apply_bellman_operator(session, policy, sarsas_split, gamma, state_plh_name):
    """ Computes the target part of Q-Learning based Advantage function
            Args:
                session: session
                policy: tgt network
                sarsas_split:  tuple of sarsa components in np.ndarrays (i.e. (np.ndarray(states), np.ndarray(actions), ...))
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

    if not isinstance(sarsas_split, tuple):
        raise ValueError("sarsa_split must be a tuple of np.ndarrays")

    states, actions, rewards, dones, next_states, next_actions = sarsas_split

    if state_plh_name not in policy.feed_dict_keys:
        raise ValueError("%s is not in network feed_dict_keys" % state_plh_name)

    q_values = policy.inference(session, {state_plh_name: next_states})

    max_qs = np.expand_dims(np.amax(q_values, axis=1), axis=1)

    dones = np.array(dones, dtype=np.bool)

    return states, actions, np.array(rewards) + gamma * np.invert(dones) * max_qs
