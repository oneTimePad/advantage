import tensorflow as tf
import numpy as np

""" Computes various RL special values """



def compute_q_part_advantage(session, policy, sarsas, gamma, state_plh_name):
    """ Computes the target part of Q-Learning based Advantage function
            Args:
                session: session
                policy: tgt network
                sarsa:  namedtuple.Sarsa from a buffer
                gamme: one-step discount factor
                state_plh_name: the name of the placeholder of the state input to network

            Returns:
                a numpy array of targets

            Raises:
                ValueError: for bad arguments
    """
    if not isinstance(session, tf.Session):
        raise ValueError("Must pass in a tf.Session object")

    next_states = [ sarsa.next_state for sarsa in sarsas]
    rewards = [sarsa.reward for sarsa in sarsas]
    dones = [sarsa.done for sarsa in sarsas]

    if state_plh_name not in policy.feed_dict_keys:
        raise ValueError("%s is not in network feed_dict_keys" % state_plh_name)

    q_values = policy.inference(session, {state_plh_name: next_states})

    max_qs = np.amax(q_values, axis=1)

    dones = np.array(dones, dtype=np.bool)

    return np.array(rewards) + gamma * np.invert(dones) * max_qs
