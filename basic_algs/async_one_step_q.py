import tensorflow as tf
import gym
import threading
import time
import numpy as np
import random

""" Asynchronous One-Step Q-Learning """

tf.reset_default_graph()

ALPHA = 0.05

dqn_graph = tf.Graph()
with dqn_graph.as_default() as g:

    state = tf.placeholder(shape=(None, 4), dtype=tf.float32, name="state")
    action = tf.placeholder(shape=(None, 1), dtype=tf.float32, name="action")
    state_action = tf.concat([state,action], axis=1)

    """ The DQN for acting """
    one_step_dqn_scope = "one_step_dqn"
    initializer = tf.contrib.layers.variance_scaling_initializer()
    with tf.variable_scope(one_step_dqn_scope):

        hidden = tf.layers.dense(state_action, 4, activation=tf.nn.elu, kernel_initializer=initializer)
        #hidden = tf.layers.dense(hidden, 30, activation=tf.nn.relu6)
        q = tf.layers.dense(hidden, 1, activation=None, kernel_initializer=initializer)

        target = tf.placeholder(shape=(None, 1), dtype=tf.float32, name="target")

        # fetch the network params so we can compute gradients
        dqn_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=one_step_dqn_scope)
        dqn_gradients = {v.name: tf.clip_by_value(tf.gradients(tf.square(target - q), v), -5, 5) for v in dqn_params }

        dqn_apply_gradients = {}
        for param in dqn_params:
            place = tf.placeholder(shape=param.get_shape(), dtype=tf.float32)
            dqn_apply_gradients[param.name] = (place, tf.assign_add(param, place))

    """ The DQN for computing the target """
    one_step_dqn_neg_scope = "one_step_dqn_neg"

    with tf.variable_scope(one_step_dqn_neg_scope):

        hidden = tf.layers.dense(state_action, 4, activation=tf.nn.elu, kernel_initializer=initializer)
        #hidden = tf.layers.dense(hidden, 30, activation=tf.nn.relu6)
        q_neg = tf.layers.dense(hidden, 1, activation=None, kernel_initializer=initializer)

        dqn_neg_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=one_step_dqn_neg_scope)
        dqn_param_copy = [tf.assign(v, tgt) for v, tgt in zip(dqn_params, dqn_neg_params) ]




def apply_gradients(session, accum_dq, steps_since_async_update):
    """ Apply the accumulated gradients to the dqn

        Args:
            session: tf session
            accum_dq: accumulated gradients for async update to dqn
    """
    global dqn_apply_gradients
    for param in dqn_apply_gradients.keys():
        session.run(dqn_apply_gradients[param][1], feed_dict={dqn_apply_gradients[param][0]: -ALPHA*accum_dq[param][0]/steps_since_async_update} )

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
    print(q_values)
    return actions[np.argmax(q_values)] if np.random.uniform(0, 1) > eps else actions[random.sample(range(len(actions)), 1)[0]]

GAMMA = 0.95
ASYNC_UPDATE = 20
GLOBAL_UPDATE = 500
LEARNING_DECAY = 0.1
T = 0

def run(tid, lock, session, eps, accum_gradients, num_episodes, actions):
    """ main function for each thread
        Runs the action RL algorithm for each actor following a separate policy
    """

    global T, ALPHA, LEARNING_DECAY, ASYNC_UPDATE, GLOBAL_UPDATE, q, q_neg, state, action, target, dqn_gradients, dqn_param_copy, GAMMA
    t = 0
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
        steps_since_async_update = 0
        #t = 0
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
            t += 1
            steps += 1

            steps_since_async_update += 1
            if t % ASYNC_UPDATE == 0:
                with lock:
                    session.graph.finalize()
                    apply_gradients(session, accum_gradients, steps_since_async_update)
                    steps_since_async_update = 0
                    #print(accum_gradients)
                    accum_gradients = {v.name: 0.0 for v in dqn_params}
                    #print(accum_gradients)
                    #print(session.run(dqn_params[1]))
                    print("ASYNC_UPDATE")
            with lock:
                T += 1
                if T % GLOBAL_UPDATE == 0:
                    ALPHA = LEARNING_DECAY * ALPHA
                    print("GLOBAL_UPDATE")
                    session.graph.finalize()
                    session.run(dqn_param_copy)
                    #print(session.run(dqn_neg_params))
        #print(t, T)
        print("TID %d STEP %d EPISODE %d" %(tid, steps, episode))


lock = threading.Lock()
EPSILONS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
ACCUM_GRADIENTS = [{v.name: 0.0 for v in dqn_params} for _ in range(len(EPSILONS))]
NUM_EPISODES = 250
NUM_EPISODES_TEST = 100
ACTIONS = [0, 1]
with dqn_graph.as_default():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(dqn_param_copy)
        threads = [threading.Thread(target=run, args=(tid, lock, sess, eps, accum_gradients, NUM_EPISODES, ACTIONS)) for tid, eps, accum_gradients in zip(range(len(EPSILONS)), EPSILONS, ACCUM_GRADIENTS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        env = gym.make('CartPole-v0')
        for episode in range(NUM_EPISODES_TEST):
            done = False
            s = env.reset()
            while done != True:
                a = act(sess, q_neg, s, ACTIONS, eps=0)
                print(a)
                env.render()
                s, reward, done, info = env.step(a)
