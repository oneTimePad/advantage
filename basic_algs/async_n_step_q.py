import tensorflow as tf
import gym
import threading
import time
import numpy as np
import random
import math

""" Asynchronous n-Step Q-Learning for CartPole """

tf.reset_default_graph()

ALPHA = 0.9
learning_rate = 1e-2

RMS_PROP_EPS = 1e-8
EPSILONS = [.1, .1, .1, .1]
dqn_graph = tf.Graph()
with dqn_graph.as_default() as g:

    state = tf.placeholder(shape=(None, 4), dtype=tf.float32, name="state")
    action = tf.placeholder(shape=(None, 1), dtype=tf.float32, name="action")
    state_action = tf.concat([state,action], axis=1)

    """ The DQN for acting """
    n_step_dqn_scope = "n_step_dqn"
    initializer = tf.contrib.layers.variance_scaling_initializer()
    with tf.variable_scope(n_step_dqn_scope):
        hidden = tf.layers.dense(state_action, 4, activation=tf.nn.relu6, kernel_initializer=initializer)
        q = tf.layers.dense(hidden, 1, activation=None, kernel_initializer=initializer)

        target = tf.placeholder(shape=(None, 1), dtype=tf.float32, name="target")

        # fetch the network params so we can compute gradients
        dqn_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=n_step_dqn_scope)

        dqn_apply_gradients = {}
        for param in dqn_params:
            place = tf.placeholder(shape=param.get_shape(), dtype=tf.float32)
            dqn_apply_gradients[param.name] = (place, tf.assign_add(param, place))

    """ The DQN for computing the target """
    n_step_dqn_neg_scope = "n_step_dqn_neg"

    with tf.variable_scope(n_step_dqn_neg_scope):
        hidden = tf.layers.dense(state_action, 4, activation=tf.nn.relu6, kernel_initializer=initializer)
        q_neg = tf.layers.dense(hidden, 1, activation=None, kernel_initializer=initializer)

        dqn_neg_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=n_step_dqn_neg_scope)
        dqn_param_copy = [tf.assign(v, tgt) for v, tgt in zip(dqn_neg_params, dqn_params) ]

    """ The DQN for each thread """
    thread_graphs = []
    thread_dqn_gradients = []
    thread_dqn_sync = []
    n_step_dqn_thread_scope = "n_step_dqn_thread_%d"
    for i in range(len(EPSILONS)):
        with tf.variable_scope(n_step_dqn_thread_scope % i):
            hidden = tf.layers.dense(state_action, 4, activation=tf.nn.relu6, kernel_initializer=initializer)
            q_thread = tf.layers.dense(hidden, 1, activation=None, kernel_initializer=initializer)
        thread_graphs.append(q_thread)
        dqn_params_thread = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=n_step_dqn_thread_scope % i)
        dqn_gradients = {t.name: tf.clip_by_value(tf.gradients(tf.square(target - q_thread), v), -50, 50) for t, v in zip(dqn_params, dqn_params_thread) }
        thread_dqn_gradients.append(dqn_gradients)
        thread_dqn_sync.append([tf.assign(v, tgt) for v, tgt in zip(dqn_params_thread, dqn_params)])



def apply_gradients(session, accum_dq, g_rms_prop, steps_since_async_update):
    """ Apply the accumulated gradients to the dqn

        Args:
            session: tf session
            accum_dq: accumulated gradients for async update to dqn
    """
    global dqn_apply_gradients, learning_rate
    for param in dqn_apply_gradients.keys():

        dq = accum_dq[param][0]/steps_since_async_update

        g_rms_prop[param] = ALPHA * g_rms_prop[param] + (1 - ALPHA) * (dq ** 2)

        dq = -learning_rate * dq / np.sqrt(g_rms_prop[param] + RMS_PROP_EPS)

        session.run(dqn_apply_gradients[param][1], feed_dict={dqn_apply_gradients[param][0]: dq} )

def compute_target(session, q, next_state, actions, reward, gamma, end=False):
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
        return 0
    else:
        return max([ session.run(q, feed_dict={"state:0": [next_state], "action:0": [[a]]}) for a in actions])[0][0]

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

GAMMA = 0.7
ASYNC_UPDATE = 50
GLOBAL_UPDATE = 50
LEARNING_DECAY = 1
EPS_DECAY = .9
EPS_DECAY_TIME = 100
STEP_TIME = 50
T = 0

def run(tid, lock, session, q_thread, q_thread_grad, q_thread_sync, eps, accum_gradients, g_rms_prop, num_episodes, actions):
    """ main function for each thread
        Runs the action RL algorithm for each actor following a separate policy
    """

    global EPS_DECAY_TIME, EPS_DECAY, learning_rate, T, ALPHA, LEARNING_DECAY, ASYNC_UPDATE, GLOBAL_UPDATE, q, q_neg, state, action, target, dqn_gradients, dqn_param_copy, GAMMA
    t = 0
    env = gym.make('CartPole-v0')
    env.reset()

    T_MAX = 5
    session.graph.finalize()
    session.run(q_thread_sync)
    for episode in range(1, num_episodes):
        done = False
        reward = 0
        s = env.reset()
        steps = 0
        t_start = t
        rewards = []
        action_list = []
        states = []
        session.graph.finalize()
        session.run(q_thread_sync)
        while done != True and t - t_start < T_MAX:
            states.append(s)
            session.graph.finalize()
            a = act(session, q_thread, s, actions, eps)
            action_list.append(a)
            s, reward, done, info = env.step(a)
            rewards.append(reward)

            t += 1
            with lock:
                T += 1
            steps += 1

        session.graph.finalize()
        R = compute_target(session, q_neg, s, ACTIONS, reward, GAMMA, end=done)

        for i in range(0, t - t_start):
            R = rewards[i] + GAMMA * R
            for grad in accum_gradients.keys():
                session.graph.finalize()
                accum_gradients[grad] += session.run(q_thread_grad[grad], feed_dict={target: [[R]], state: [states[i]], action: [[action_list[i]]]})

        with lock:
            session.graph.finalize()
            apply_gradients(session, accum_gradients, g_rms_prop, t - t_start)
            accum_gradients = {v.name: 0.0 for v in dqn_params}
            if T % GLOBAL_UPDATE == 0:
                session.graph.finalize()
                session.run(dqn_param_copy)
                learning_rate = LEARNING_DECAY * learning_rate

        if episode % EPS_DECAY_TIME == 0:
            eps = EPS_DECAY * eps
        if episode % STEP_TIME == 0:
            T_MAX  += 10
        print("TID %d STEP %d EPISODE %d EPS %f T_MAX %d" %(tid, steps, episode, eps, T_MAX))

lock = threading.Lock()


ACCUM_GRADIENTS = [{v.name: 0.0 for v in dqn_params} for _ in range(len(EPSILONS))]
G_RMS_PROP = [ {v.name: 0.0 for v in dqn_params} for _ in range(len(EPSILONS))]
NUM_EPISODES = 500
NUM_EPISODES_TEST = 500
ACTIONS = [0, 1]
with dqn_graph.as_default():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(dqn_param_copy)

        threads = [threading.Thread(target=run, args=(tid, lock, sess, q_thread, q_thread_grads, q_thread_sync, eps, accum_gradients, g_rms_prop, NUM_EPISODES, ACTIONS)) for tid, eps, q_thread, q_thread_grads, q_thread_sync, accum_gradients, g_rms_prop in zip(range(len(EPSILONS)), EPSILONS, thread_graphs, thread_dqn_gradients, thread_dqn_sync, ACCUM_GRADIENTS, G_RMS_PROP)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        sess.run(dqn_param_copy)

        env = gym.make('CartPole-v0')
        for episode in range(NUM_EPISODES_TEST):
            done = False
            s = env.reset()
            steps = 0
            while done != True:
                a = act(sess, q, s, ACTIONS, eps=0)
                env.render()
                s, reward, done, info = env.step(a)
                steps += 1
            print("STEP %d EPISODE %d" %(steps, episode))
