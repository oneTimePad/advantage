import tensorflow as tf
import gym
import threading
import time
import numpy as np
import random

""" Asynchronous One-Step Q-Learning """

tf.reset_default_graph()

dqn_graph = tf.Graph()
with dqn_graph.as_default() as g:

    state = tf.placeholder(shape=(None, 4), dtype=tf.float32, name="state")
    action = tf.placeholder(shape=(None, 1), dtype=tf.float32, name="action")
    state_action = tf.concat([state,action], axis=1)

    """ The DQN for acting """
    one_step_dqn_scope = "one_step_dqn"

    with tf.variable_scope(one_step_dqn_scope):

        hidden = tf.layers.dense(state_action, 80, activation=tf.nn.relu6)
        hidden = tf.layers.dense(hidden, 30, activation=tf.nn.relu6)
        q = tf.layers.dense(hidden, 1, activation=None)

        target = tf.placeholder(shape=(None, 1), dtype=tf.float32, name="target")

        # fetch the network params so we can compute gradients
        dqn_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=one_step_dqn_scope)
        dqn_gradients = {v.name: tf.clip_by_value(tf.gradients(tf.square(target - q), v), -5, 5) for v in dqn_params }

    """ The DQN for computing the target """
    one_step_dqn_neg_scope = "one_step_dqn_neg"

    with tf.variable_scope(one_step_dqn_neg_scope):

        hidden = tf.layers.dense(state_action, 80, activation=tf.nn.relu6)
        hidden = tf.layers.dense(hidden, 30, activation=tf.nn.relu6)
        q_neg = tf.layers.dense(hidden, 1, activation=None)

        dqn_neg_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=one_step_dqn_neg_scope)
        dqn_param_copy = [tf.assign(v, tgt) for v, tgt in zip(dqn_params, dqn_neg_params) ]


def apply_gradients(session, accum_dq):
    """ Apply the accumulated gradients to the dqn

        Args:
            session: tf session
            accum_dq: accumulated gradients for async update to dqn
    """
    for param in dqn_params:
        session.run(tf.assign_add(param, accum_dq[param.name]))

def compute_target(session, q, next_state, next_action, reward, gamma, end=False):
    """ Computes the Q value target

        Args:
            session: tf session
            q: dqn tensor (q_neg)
            next_state: the next state agent goes into
            next_action: the next action the agent chooses
            reward: the reward received before preceeding to next state
            gamma: the discount factor
            end: whether we are at the end of a episode

        Returns:
            the Q value target for gradient update
    """

    if end:
        return [[reward]]
    else:
        return reward + gamma * session.run(q, feed_dict={"state:0": [next_state], "action:0": [[next_action]]})

def act(session, q, state, actions, eps):
    """ act epsilon greedily

        Args:
            session: tf session
            q: deep-Q network
            state: state list
            actions: list of possible actions
            eps: epsilon param for epsilon-greedy
        Returns:
            an action from actions
    """
    q_values = []
    for a in actions:
        q_s_a = sess.run(q, feed_dict={"state:0": [state], "action:0": [[a]]})
        q_values.append(q_s_a[0])

    return actions[np.argmax(q_values)] if np.random.uniform(0, 1) > eps else actions[random.sample(range(len(actions)), 1)[0]]

GAMMA = 0.9


def run(session, eps, accum_gradients, num_episodes, actions):
    """ main function for each thread
        Runs the action RL algorithm for each actor following a separate policy
    """

    global q, q_neg, state, action, target, dqn_gradients, dqn_param_copy, GAMMA

    env = gym.make('CartPole-v0')
    env.reset()
    session.graph.finalize()
    for episode in range(1, num_episodes):
        done = False
        G, reward = 0, 0
        s = env.reset()
        session.graph.finalize()
        a = act(session, q, s, actions, eps)
        steps = 0
        while done != True:
            prev_s = s
            s, reward, done, info = env.step(a)

            prev_a = a
            session.graph.finalize()
            a = act(session, q, s, actions, eps)
            session.graph.finalize()
            y = compute_target(session, q_neg, s, a, reward, GAMMA, end=done)

            for grad in accum_gradients.keys():
                session.graph.finalize()
                accum_gradients[grad] += session.run(dqn_gradients[grad], feed_dict={target: y, state: [prev_s], action: [[prev_a]]})
            print(accum_gradients)


EPSILONS = [ 0.1, 0.2, 0.3, 0.4]
ACCUM_GRADIENTS = [{v.name: 0.0 for v in dqn_params} for _ in range(len(EPSILONS))]
NUM_EPISODES = 1001
ACTIONS = [0, 1]
with dqn_graph.as_default():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        threads = [threading.Thread(target=run, args=(sess, eps, accum_gradients, NUM_EPISODES, ACTIONS)) for eps, accum_gradients in zip(EPSILONS, ACCUM_GRADIENTS)]
        for t in threads:
            t.start()
        while True:
            time.sleep(1)
