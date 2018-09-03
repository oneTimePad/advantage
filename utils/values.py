import tensorflow as tf
import numpy as np

""" Computes various RL special values """



def compute_q_part_advantage(session, target, state_placeholder, sarsas, gamma):
    """ Computes the target part of Q-Learning based Advantage function
            Args:
                session: session
                target: tgt network
                state_placeholder: tf.placeholder for feeding input network
                sarsa:  namedtuple.Sarsa from a buffer
                gamme: one-step discount factor

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
    with session.as_default():
        max_qs = np.amax(session.run(target, feed_dict={state_placeholder: next_states}),
                        axis=1)

    dones = np.array(dones, dtype=np.bool)

    return np.array(rewards) + gamma * np.invert(dones) * max_qs
