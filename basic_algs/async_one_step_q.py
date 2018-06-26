import tensorflow as tf
import gym
import threading
import time
import numpy as np
import random
import math

""" Asynchronous One-Step Q-Learning """

tf.reset_default_graph()

ALPHA = 0.9
learning_rate = 1e-2

RMS_PROP_EPS = 1e-8

dqn_graph = tf.Graph()
with dqn_graph.as_default() as g:

    state = tf.placeholder(shape=(None, 4), dtype=tf.float32, name="state")
    action = tf.placeholder(shape=(None, 1), dtype=tf.float32, name="action")
    state_action = tf.concat([state,action], axis=1)

    """ The DQN for acting """
    one_step_dqn_scope = "one_step_dqn"
    initializer = tf.contrib.layers.variance_scaling_initializer()
    with tf.variable_scope(one_step_dqn_scope):
        hidden = tf.layers.dense(state_action, 4, activation=tf.nn.relu6, kernel_initializer=initializer)
        #hidden = tf.layers.dense(hidden, 256, activation=tf.nn.relu6, kernel_initializer=initializer)
        #hidden = tf.layers.dense(hidden, 128, activation=tf.nn.relu6, kernel_initializer=initializer)
        #hidden = tf.layers.dense(hidden, 64, activation=tf.nn.relu6, kernel_initializer=initializer)
        #hidden = tf.layers.dense(hidden, 32, activation=tf.nn.relu6, kernel_initializer=initializer)
        #hidden = tf.layers.dense(hidden, 16, activation=tf.nn.relu6, kernel_initializer=initializer)
        q = tf.layers.dense(hidden, 1, activation=None, kernel_initializer=initializer)

        target = tf.placeholder(shape=(None, 1), dtype=tf.float32, name="target")

        # fetch the network params so we can compute gradients
        dqn_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=one_step_dqn_scope)
        dqn_gradients = {v.name: tf.clip_by_value(tf.gradients(tf.square(target - q), v), -50, 50) for v in dqn_params }

        dqn_apply_gradients = {}
        for param in dqn_params:
            place = tf.placeholder(shape=param.get_shape(), dtype=tf.float32)
            dqn_apply_gradients[param.name] = (place, tf.assign_add(param, place))

    """ The DQN for computing the target """
    one_step_dqn_neg_scope = "one_step_dqn_neg"

    with tf.variable_scope(one_step_dqn_neg_scope):
        hidden = tf.layers.dense(state_action, 4, activation=tf.nn.relu6, kernel_initializer=initializer)
        #hidden = tf.layers.dense(hidden, 256, activation=tf.nn.relu6, kernel_initializer=initializer)
        #hidden = tf.layers.dense(hidden, 128, activation=tf.nn.relu6, kernel_initializer=initializer)
        #hidden = tf.layers.dense(hidden, 64, activation=tf.nn.relu6, kernel_initializer=initializer)
        #hidden = tf.layers.dense(hidden, 32, activation=tf.nn.relu6, kernel_initializer=initializer)
        #hidden = tf.layers.dense(hidden, 16, activation=tf.nn.relu6, kernel_initializer=initializer)
        q_neg = tf.layers.dense(hidden, 1, activation=None, kernel_initializer=initializer)

        dqn_neg_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=one_step_dqn_neg_scope)
        dqn_param_copy = [tf.assign(v, tgt) for v, tgt in zip(dqn_neg_params, dqn_params) ]




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
        return [[reward]]
    else:
        return reward + gamma * max([ session.run(q, feed_dict={"state:0": [next_state], "action:0": [[a]]}) for a in actions])

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
    #print(q_values)
    return actions[np.argmax(q_values)] if np.random.uniform(0, 1) > eps else actions[random.sample(range(len(actions)), 1)[0]]

GAMMA = 0.95
ASYNC_UPDATE = 50
GLOBAL_UPDATE = 50
LEARNING_DECAY = 1#0.1
EPS_DECAY = .9
EPS_DECAY_TIME = 50
T = 0

def run(tid, lock, session, eps, accum_gradients, g_rms_prop, num_episodes, actions):
    """ main function for each thread
        Runs the action RL algorithm for each actor following a separate policy
    """

    global EPS_DECAY_TIME, EPS_DECAY, learning_rate, T, ALPHA, LEARNING_DECAY, ASYNC_UPDATE, GLOBAL_UPDATE, q, q_neg, state, action, target, dqn_gradients, dqn_param_copy, GAMMA
    t = 0
    env = gym.make('CartPole-v0')
    env.reset()
    session.graph.finalize()
    steps_since_async_update = 0
    for episode in range(1, num_episodes):
        done = False
        G, reward = 0, 0
        s = env.reset()
        session.graph.finalize()
        #a = act(session, q, s, actions, eps)
        steps = 0
        #steps_since_async_update = 0
        #t = 0
        while done != True:
            prev_s = s
            session.graph.finalize()
            a = act(session, q, s, actions, eps)
            s, reward, done, info = env.step(a)
            #prev_a = a
            session.graph.finalize()
            #session.graph.finalize()
            y = compute_target(session, q_neg, s, ACTIONS, reward, GAMMA, end=done)

            for grad in accum_gradients.keys():
                session.graph.finalize()
                accum_gradients[grad] += session.run(dqn_gradients[grad], feed_dict={target: y, state: [prev_s], action: [[a]]})
            #t += 1
            steps += 1
            with lock:
                T += 1
                if T % GLOBAL_UPDATE == 0:
                    #print("GLOBAL_UPDATE")
                    session.graph.finalize()
                    session.run(dqn_param_copy)
                    learning_rate = LEARNING_DECAY * learning_rate

            steps_since_async_update += 1
            t += 1
            if t % ASYNC_UPDATE == 0 or done:
                with lock:
                    session.graph.finalize()
                    apply_gradients(session, accum_gradients, g_rms_prop, steps_since_async_update)
                    steps_since_async_update = 0
                    #print(accum_gradients)
                    accum_gradients = {v.name: 0.0 for v in dqn_params}

                    #g_rms_prop =  {v.name: 0.0 for v in dqn_params}
                    #print(accum_gradients)
                    #print(session.run(dqn_params[1]))
                    #print("ASYNC_UPDATE")

                    #print(session.run(dqn_neg_params))
        if episode % EPS_DECAY_TIME == 0:
            eps = EPS_DECAY * eps
        #print(t, T)
        print("TID %d STEP %d EPISODE %d EPS %f" %(tid, steps, episode, eps))

    #apply_gradients(session, accum_gradients, g_rms_prop, steps_since_async_update)

lock = threading.Lock()
#EPSILONS =[0.05, 0.1, 0.15, 0.2]
EPSILONS = [.1, .1, .1, .1]#[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.1, 0.2, 0.5, 0.9]#[0.05, 0.1, 0.15, 0.2, 0.25]#, 0.3, 0.35, 0.4]
#EPSILONS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.15, 0.1]
ACCUM_GRADIENTS = [{v.name: 0.0 for v in dqn_params} for _ in range(len(EPSILONS))]
G_RMS_PROP = [ {v.name: 0.0 for v in dqn_params} for _ in range(len(EPSILONS))]
NUM_EPISODES = 500
NUM_EPISODES_TEST = 500
ACTIONS = [0, 1]
with dqn_graph.as_default():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(dqn_param_copy)

        threads = [threading.Thread(target=run, args=(tid, lock, sess, eps, accum_gradients, g_rms_prop, NUM_EPISODES, ACTIONS)) for tid, eps, accum_gradients, g_rms_prop in zip(range(len(EPSILONS)), EPSILONS, ACCUM_GRADIENTS, G_RMS_PROP)]
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
                #print(a)
                env.render()
                s, reward, done, info = env.step(a)
                steps += 1
            print("STEP %d EPISODE %d" %(steps, episode))
