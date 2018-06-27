import tensorflow as tf
import gym
import threading
import time
import numpy as np
import random
import math

""" Asynchronous Advantage Actor-Critic for CartPole """

tf.reset_default_graph()

ALPHA = 0.9

BETA = .01
RMS_PROP_EPS = 1e-1
NUM_AGENTS = 12
ACTIONS = [0, 1]

actor_critic_graph = tf.Graph()
with actor_critic_graph.as_default() as g:

    state = tf.placeholder(shape=(None, 4), dtype=tf.float32, name="state")
    #action = tf.placeholder(shape=(None, 1), dtype=tf.int32, name="action")

    """ Actor-Critic Network """
    actor_critic_network_scope = "actor_critic_network"
    initializer = tf.contrib.layers.variance_scaling_initializer()
    with tf.variable_scope(actor_critic_network_scope):
        hidden = tf.layers.dense(state, 256, activation=tf.nn.elu, kernel_initializer=initializer)
        hidden = tf.layers.dense(hidden, 128, activation=tf.nn.elu, kernel_initializer=initializer)
        hidden = tf.layers.dense(hidden, 64, activation=tf.nn.elu, kernel_initializer=initializer)
        hidden = tf.layers.dense(hidden, 32, activation=tf.nn.elu, kernel_initializer=initializer)
        hidden = tf.layers.dense(hidden, 16, activation=tf.nn.elu, kernel_initializer=initializer)
    hidden_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=actor_critic_network_scope)

    actor_network_scope = "actor_network"
    with tf.variable_scope(actor_network_scope):
        policy = tf.layers.dense(hidden, len(ACTIONS), activation=None, kernel_initializer=initializer)
        policy = tf.nn.softmax(policy, axis=1)
    policy_params = hidden_params + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=actor_network_scope)

    policy_async_update = {}
    for param in policy_params:
        place = tf.placeholder(shape=param.get_shape(), dtype=tf.float32)
        policy_async_update[param.name] = (place, tf.assign_add(param, place))

    critic_network_scope = "critic_network"
    with tf.variable_scope(critic_network_scope):
        value = tf.layers.dense(hidden, 1, activation=None, kernel_initializer=initializer)

    value_params = hidden_params + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=critic_network_scope)

    value_async_update = {}
    for param in value_params:
        place = tf.placeholder(shape=param.get_shape(), dtype=tf.float32)
        value_async_update[param.name] = (place, tf.assign_add(param, place))

    target = tf.placeholder(shape=(None, 1), dtype=tf.float32, name="target")

    """ Thread Actor-Critic Networks """
    thread_policies = []
    thread_values = []
    thread_policy_grads = []
    thread_value_grads = []
    thread_sync_policies = []
    thread_sync_values = []
    thread_hidden_network_scope = "thread_hidden_network_%d"
    thread_actor_network_scope = "thread_actor_network_%d"
    thread_critic_network_scope = "thread_critic_network_%d"

    for i in range(NUM_AGENTS):
        with tf.variable_scope(thread_hidden_network_scope%i):
            thread_hidden = tf.layers.dense(state, 256, activation=tf.nn.elu, kernel_initializer=initializer)
            thread_hidden = tf.layers.dense(thread_hidden, 128, activation=tf.nn.elu, kernel_initializer=initializer)
            thread_hidden = tf.layers.dense(thread_hidden, 64, activation=tf.nn.elu, kernel_initializer=initializer)
            thread_hidden = tf.layers.dense(thread_hidden, 32, activation=tf.nn.elu, kernel_initializer=initializer)
            thread_hidden = tf.layers.dense(thread_hidden, 16, activation=tf.nn.elu, kernel_initializer=initializer)
        thread_hidden_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=thread_hidden_network_scope%i)


        with tf.variable_scope(thread_critic_network_scope%i):
            thread_value = tf.layers.dense(thread_hidden, 1, activation=None, kernel_initializer=initializer)
        thread_values.append(thread_value)

        thread_value_params = thread_hidden_params + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=thread_critic_network_scope%i)

        thread_value_grad = {t.name: tf.clip_by_value(tf.gradients(0.5 * tf.square(target - thread_value), v), -40, 40) for t, v in zip(value_params, thread_value_params)}
        thread_value_grads.append(thread_value_grad)

        thread_sync_value = [ tf.assign(v, tgt) for v, tgt in zip(thread_value_params, value_params)]
        thread_sync_values.append(thread_sync_value)


        with tf.variable_scope(thread_actor_network_scope%i):
            thread_policy = tf.layers.dense(thread_hidden, len(ACTIONS), activation=None, kernel_initializer=initializer)
            thread_policy = tf.nn.softmax(thread_policy, axis=1)
        thread_policies.append(thread_policy)

        thread_policy_params = thread_hidden_params + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=thread_actor_network_scope%i)

        thread_policy_grad = {t.name: tf.clip_by_value((tf.gradients(-tf.log(tf.reduce_max(thread_policy, axis=1)), v) * (target - thread_value)) + (BETA * tf.reduce_sum(tf.gradients(thread_policy * tf.log(thread_policy), v), axis=1)), -40, 40) for t, v in zip(policy_params, thread_policy_params)}
        thread_policy_grads.append(thread_policy_grad)

        thread_sync_policy = [ tf.assign(v, tgt) for v, tgt in zip(thread_policy_params, policy_params)]
        thread_sync_policies.append(thread_sync_policy)







def gradient_update(session, gradient_updates, accum_grads, g_rms_prop, steps_since_async_update, learning_rate):
    """ Apply the accumulated gradients to the network

        Args:
            session: tf session
            gradient_updates: gradient update ops
            accum_grads: accumulated gradients
            g_rms_prop: g values for RMSProp
            steps_since_async_update: number of gradient updates made
    """

    for param in gradient_updates.keys():

        dtheta = accum_grads[param]/steps_since_async_update

        g_rms_prop[param] = ALPHA * g_rms_prop[param] + (1 - ALPHA) * (dtheta ** 2)

        dtheta = -learning_rate * dtheta / np.sqrt(g_rms_prop[param] + RMS_PROP_EPS)

        session.run(gradient_updates[param][1], feed_dict={gradient_updates[param][0]: dtheta[0]} )


def compute_target(session, t_value, next_state, reward, end=False):
    """ Computes the Q value target

        Args:
            session: tf session
            value : thread value network
            next_state: the next state agent goes into
            end: whether we are at the end of a episode

        Returns:
            the Value target for gradient update
    """

    if end:
        return reward
    else:
        return session.run(t_value, feed_dict={"state:0": [next_state]})[0][0]


def act(session, t_policy, next_state):
    """ act following policy

        Args:
            session: tf session
            policy: stochastic policy
            state: current state

        Return an action from actions
    """
    global ACTIONS

    return ACTIONS[np.argmax(session.run(t_policy, feed_dict={"state:0": [next_state]}))]


NUM_EPISODES =  250
T_REDUCE = NUM_EPISODES/10
GAMMA_REDUCE = 50
b = threading.Barrier(NUM_AGENTS)
T_MAXS = [30] * 12
GAMMAS = [.65] * 12
LEARNING_DECAY = .99
DECAY_TIME = 50
def run(tid, lock,
        session,
        thread_policy,
        thread_value,
        thread_policy_grad,
        thread_value_grad,
        thread_sync_policy,
        thread_sync_value,
        accum_policy_gradients,
        accum_value_gradients,
        g_rms_prop_policy,
        g_rms_prop_value,
        num_episodes):
    """ main function for each thread
        Runs the action RL algorithm for each actor following a separate policy
    """

    t = 0
    env = gym.make('CartPole-v0')
    env.reset()
    learning_rate = 7e-4
    session.graph.finalize()

    for episode in range(1, num_episodes):
        done = False
        reward = 0
        s = env.reset()
        steps = 0
        t_start = t
        rewards = []
        states = []
        b.wait()
        session.graph.finalize()
        session.run(thread_sync_policy)
        session.graph.finalize()
        session.run(thread_sync_value)
        b.wait()
        while done != True and t - t_start < T_MAXS[tid]:
            states.append(s)
            session.graph.finalize()
            a = act(session, thread_policy, s)
            s, reward, done, info = env.step(a)
            rewards.append(reward)
            t += 1
            steps += 1
        if episode % T_REDUCE == 0 and T_MAXS[tid] > 20:
            T_MAXS[tid] -= 1


        if episode % GAMMA_REDUCE == 0 and T_MAXS[tid] > 20:
            GAMMAS[tid] += .01

        session.graph.finalize()
        R = compute_target(session, thread_value, s, 0, end=done)

        for i in reversed(range(0, t - t_start - 1)):

            R = rewards[i] + GAMMAS[tid] * R
            for grad in accum_policy_gradients.keys():
                session.graph.finalize()
                accum_policy_gradients[grad] += session.run(thread_policy_grad[grad], feed_dict={target: [[R]], state: [states[i]]})

            for grad in accum_value_gradients.keys():
                session.graph.finalize()
                accum_value_gradients[grad] += session.run(thread_value_grad[grad], feed_dict={target: [[R]], state: [states[i]]})
        with lock:
            session.graph.finalize()
            gradient_update(session, policy_async_update, accum_policy_gradients, g_rms_prop_policy, t - t_start - 1, learning_rate)
            gradient_update(session, value_async_update, accum_value_gradients, g_rms_prop_value, t - t_start - 1, learning_rate)
            accum_policy_gradients = {v.name: 0.0 for v in policy_params}
            accum_value_gradients = {v.name: 0.0 for v in value_params}

        if episode % DECAY_TIME == 0:
            learning_rate = LEARNING_DECAY * learning_rate

        print("TID %d STEP %d EPISODE %d  T_MAX %d GAMMA %f" %(tid, steps, episode, T_MAXS[tid], GAMMAS[tid]))

lock = threading.Lock()


ACCUM_GRADIENTS_POLICY = [{v.name: 0.0 for v in policy_params} for _ in range(NUM_AGENTS)]
ACCUM_GRADIENTS_VALUE = [{v.name: 0.0 for v in value_params} for _ in range(NUM_AGENTS)]
G_RMS_PROP_POLICY = [ {v.name: 0.0 for v in policy_params} for _ in range(NUM_AGENTS)]
G_RMS_PROP_VALUE = [ {v.name: 0.0 for v in value_params} for _ in range(NUM_AGENTS)]
NUM_EPISODES_TEST = 500

with actor_critic_graph.as_default():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


        threads = [threading.Thread(target=run, args=(tid, lock,
                                                      sess,
                                                      thread_policy,
                                                      thread_value,
                                                      thread_policy_grad,
                                                      thread_value_grad,
                                                      thread_sync_policy,
                                                      thread_sync_value,
                                                      accum_gradients_policy,
                                                      accum_gradients_value,
                                                      g_rms_prop_policy,
                                                      g_rms_prop_value,
                                                      NUM_EPISODES)) for tid, thread_policy, thread_value, thread_policy_grad, thread_value_grad, thread_sync_policy, thread_sync_value, accum_gradients_policy, accum_gradients_value, g_rms_prop_policy, g_rms_prop_value in zip(range(NUM_AGENTS), thread_policies, thread_values, thread_policy_grads, thread_value_grads, thread_sync_policies, thread_sync_values, ACCUM_GRADIENTS_POLICY, ACCUM_GRADIENTS_VALUE, G_RMS_PROP_POLICY, G_RMS_PROP_VALUE)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        env = gym.make('CartPole-v0')
        for episode in range(NUM_EPISODES_TEST):
            done = False
            s = env.reset()
            steps = 0
            while done != True:
                a = act(sess, policy, s)
                env.render()
                s, reward, done, info = env.step(a)
                steps += 1
            print("STEP %d EPISODE %d" %(steps, episode))
