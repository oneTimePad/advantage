
import collections
import numpy as np
import os
import multiprocessing
#import multiprocessing
import threading
import random
import pickle
import time
#import operator
#from functools import reduce
import math
import gym
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

""" Evolved Policy Gradients for CartPole"""

BUFFER_SIZE = 1024 # N
MEMORY_SIZE = 32
SAMPLE_SIZE = 64 # M
NUM_ACTIONS = 17
STATE_SPACE = 376
BATCH_SIZE = 32
L2_BETA = 1e-3
policy_lr_init = 1e-3
memory_lr_init = 1e-3
loss_es_lr_init = 1e-2
value_lr_init = 1e-3
PPO_KL_COEFF = 1e-3
PPO_CLIP_EPS = 0.2


def build_graph(tf):
    epg_graph = tf.Graph()

    with epg_graph.as_default() as g:

        """ The ES memory takes as input a vector of ones.
        Acts as a vector of biases.
        """
        memory_scope = "memory"
        with tf.variable_scope(memory_scope):
            # memory units
            memory = tf.layers.dense(tf.ones([1, 2 * MEMORY_SIZE]), MEMORY_SIZE, activation=tf.nn.tanh)

        # replicate memory to concate with M (BUFFER_SIZE) samples
        memory_tile = tf.tile(memory, [BUFFER_SIZE, 1])

        # replicate memory to concate with BATCH_SIZE samples
        memory_tile_batch = tf.tile(memory, [BATCH_SIZE, 1])

        memory_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=memory_scope)

        #memory_assign_plhs = {}
        #memory_assign_ops = []
        #for param in memory_params:
        #    param_value_plh = tf.placeholder(shape=param.get_shape(), dtype=tf.float32)s
        #    assign_op = tf.assign(param, param_value_plh)
        #    memory_assign_plhs[param.name] = param_value_plh
        #    memory_assign_ops.append(assign_op)

        # require params be assigned before evaluating
        #with tf.control_dependencies(memory_assign_ops):
        #    memory_tile = tf.identity(memory_tile)

        # {state, termination_signal, reward}
        state_plh = tf.placeholder(shape=[BUFFER_SIZE, STATE_SPACE], dtype=tf.float32, name="state_plh")
        terminate_plh = tf.placeholder(shape=[BUFFER_SIZE, 1], dtype=tf.float32, name="terminate_plh")
        reward_plh = tf.placeholder(shape=[BUFFER_SIZE, 1], dtype=tf.float32, name="reward_plh")
        action_plh = tf.placeholder(shape=[BUFFER_SIZE, NUM_ACTIONS], dtype=tf.float32, name="action_plh")

        """ Policy Network operates on state
        operates on the state samples from the buffer N for context computation
        """
        initializer = tf.contrib.layers.variance_scaling_initializer()
        policy_scope = "policy"
        with tf.variable_scope(policy_scope) as scope:
            hidden = tf.layers.dense(state_plh, 64, activation=tf.nn.tanh, kernel_initializer=initializer)
            hidden = tf.layers.dense(hidden, 64, activation=tf.nn.tanh, kernel_initializer=initializer)
            #hidden = tf.layers.dense(hidden, 64, activation=tf.nn.tanh, kernel_initializer=initializer)
            #hidden = tf.layers.dense(hidden, 64, activation=tf.nn.tanh, kernel_initializer=initializer)
            policy = tf.layers.dense(hidden, NUM_ACTIONS, activation=None, kernel_initializer=initializer)
            #policy = tf.nn.softmax(policy, axis=1)

            #policy = tf.nn.tanh(policy)
            sigma = tf.layers.dense(tf.ones([1, NUM_ACTIONS]), NUM_ACTIONS, activation=None, use_bias=False, kernel_initializer=initializer)
            #sigma = tf.nn.relu(sigma)
            sigma_tile = tf.tile(sigma, [BUFFER_SIZE, 1])
        """ Policy Network operates on state
        operates on a single state sample for computing action
        """
        state_sample_plh = tf.placeholder(shape=[1, STATE_SPACE], dtype=tf.float32, name="state_sample_plh")
        with tf.variable_scope(scope, reuse=True):
            hidden = tf.layers.dense(state_sample_plh, 64, activation=tf.nn.tanh, kernel_initializer=initializer)
            hidden = tf.layers.dense(hidden, 64, activation=tf.nn.tanh, kernel_initializer=initializer)
            #hidden = tf.layers.dense(hidden, 64, activation=tf.nn.tanh, kernel_initializer=initializer)
            #hidden = tf.layers.dense(hidden, 64, activation=tf.nn.tanh, kernel_initializer=initializer)
            policy_sample = tf.layers.dense(hidden, NUM_ACTIONS, activation=None, kernel_initializer=initializer)
            sigma_sample = tf.layers.dense(tf.ones([1, NUM_ACTIONS]), NUM_ACTIONS, activation=None, use_bias=False, kernel_initializer=initializer)
            #policy_sample = tf.nn.softmax(policy_sample, axis=1)
            #policy_sample = tf.nn.tanh(policy_sample)
            # sample action
        policy_sample = tf.identity(policy_sample, name="policy_sample")#tf.argmax(policy_sample, axis=1, name="policy_sample")
        sigma_sample = tf.identity(sigma_sample, name="sigma_sample")
        """ Policy Network operates on state
        operates on a batch of samples for computing loss and gradints of policy and mem
        """
        state_batch_plh = tf.placeholder(shape=[BATCH_SIZE, STATE_SPACE], dtype=tf.float32, name="state_batch_plh")
        terminate_batch_plh = tf.placeholder(shape=[BATCH_SIZE, 1], dtype=tf.float32, name="terminate_batch_plh")
        reward_batch_plh = tf.placeholder(shape=[BATCH_SIZE, 1], dtype=tf.float32, name="reward_batch_plh")
        action_batch_plh = tf.placeholder(shape=[BATCH_SIZE, NUM_ACTIONS], dtype=tf.float32, name="action_batch_plh")
        with tf.variable_scope(scope, reuse=True):
            hidden = tf.layers.dense(state_batch_plh, 64, activation=tf.nn.tanh, kernel_initializer=initializer)
            hidden = tf.layers.dense(hidden, 64, activation=tf.nn.tanh, kernel_initializer=initializer)
            #hidden = tf.layers.dense(hidden, 64, activation=tf.nn.tanh, kernel_initializer=initializer)
            #hidden = tf.layers.dense(hidden, 64, activation=tf.nn.tanh, kernel_initializer=initializer)
            policy_batch = tf.layers.dense(hidden, NUM_ACTIONS, activation=None, kernel_initializer=initializer)
            sigma_batch = tf.layers.dense(tf.ones([1, NUM_ACTIONS]), NUM_ACTIONS, activation=None, use_bias=False, kernel_initializer=initializer)

            sigma_tile_batch = tf.tile(sigma_batch, [BATCH_SIZE, 1])
            #policy_batch = tf.nn.softmax(policy_batch, axis=1)
            #policy_batch = tf.nn.tanh(policy_batch)

        policy_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=policy_scope)


        """ Old policy for computing policy improvement ratio for weighted average advantage for PPO """
        policy_old_scope = "policy_old"
        with tf.variable_scope(policy_old_scope):
            hidden = tf.layers.dense(state_batch_plh, 64, activation=tf.nn.tanh, kernel_initializer=initializer)
            hidden = tf.layers.dense(hidden, 64, activation=tf.nn.tanh, kernel_initializer=initializer)
            #hidden = tf.layers.dense(hidden, 64, activation=tf.nn.tanh, kernel_initializer=initializer)
            #hidden = tf.layers.dense(hidden, 64, activation=tf.nn.tanh, kernel_initializer=initializer)
            policy_old = tf.layers.dense(hidden, NUM_ACTIONS, activation=None, kernel_initializer=initializer)
            sigma_old = tf.layers.dense(tf.ones([1, NUM_ACTIONS]), NUM_ACTIONS, activation=None, use_bias=False, kernel_initializer=initializer)
            sigma_tile_old = tf.tile(sigma_old, [BATCH_SIZE, 1])

        policy_old_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=policy_old_scope)

        policy_old_copy = [tf.assign(p_old, p) for p_old, p in zip(policy_old_params, policy_params)]


        """ Value function for PPO. we minimize the square advantage """
        value_ppo_scope = "value_ppo"
        q_values_plh = tf.placeholder(shape=[BATCH_SIZE, 1], dtype=tf.float32, name="q_values_plh")
        with tf.variable_scope(value_ppo_scope) as scope:
            hidden = tf.layers.dense(state_batch_plh, 64, activation=tf.nn.leaky_relu, kernel_initializer=initializer)
            hidden = tf.layers.dense(hidden, 64, activation=tf.nn.leaky_relu, kernel_initializer=initializer)
            #hidden = tf.layers.dense(hidden, 64, activation=tf.nn.leaky_relu, kernel_initializer=initializer)

            value_ppo = tf.layers.dense(hidden, 1, activation=None, kernel_initializer=initializer)
        value_optimizer = tf.train.AdamOptimizer(value_lr_init)
        # minimize the expected square advantage
        value_gradients = value_optimizer.minimize(tf.reduce_mean( (q_values_plh - value_ppo) ** 2))

        state_sample_size_plh = tf.placeholder(shape=[SAMPLE_SIZE, STATE_SPACE], dtype=tf.float32, name="state_sample_size_plh")
        with tf.variable_scope(scope, reuse=True):
            hidden = tf.layers.dense(state_sample_size_plh, 64, activation=tf.nn.leaky_relu, kernel_initializer=initializer)
            hidden = tf.layers.dense(hidden, 64, activation=tf.nn.leaky_relu, kernel_initializer=initializer)
            #hidden = tf.layers.dense(hidden, 64, activation=tf.nn.leaky_relu, kernel_initializer=initializer)
            value_ppo_sample = tf.layers.dense(hidden, 1, activation=None, kernel_initializer=initializer)

        value_ppo_sample = tf.identity(value_ppo_sample, name="value_ppo_sample")




        # assign parameters to policy network, to allow for changing of params on each evaluations
        #policy_assign_plhs = {} # the placeholder for the params to assign
        #policy_assign_ops = [] # the actual assign ops
        #for param in policy_params:
        #    param_value_plh = tf.placeholder(shape=param.get_shape(), dtype=tf.float32)
        #    assign_op = tf.assign(param, param_value_plh)
        #    policy_assign_plhs[param.name] = param_value_plh
        #    policy_assign_ops.append(assign_op)

        # require parameters be assign before evaluating
        #with tf.control_dependencies(policy_assign_ops):
        #    policy = tf.identity(policy)


        """ context vector network, operates on N buffer samples
        these samples include N {state, termination_signal pairs}
        TF computes the rest of the required components: mem, action, policy distribution
        In this case we don't add the reward; the reward is optional since it can be inferred.
        """
        # ES update
        context_scope = "context"
        with tf.variable_scope(context_scope):
            # tf.cast(tf.expand_dims(tf.argmax(policy, axis=1), axis=1), tf.float32)
            context_input = tf.concat([state_plh, terminate_plh,  action_plh, memory_tile, policy, sigma_tile], axis=1)# tf.square(action_plh - policy) / (2 * tf.square(tf.exp(sigma_tile)))], axis=1)
            context_input = tf.expand_dims(context_input, axis=0)
            hidden = tf.layers.conv1d(context_input, 16, 6, strides=5, activation=tf.nn.leaky_relu, padding="same")
            hidden = tf.layers.conv1d(hidden, 32, 4, strides=2, activation=tf.nn.leaky_relu, padding="same")
            context = tf.layers.conv1d(hidden, 32, int(hidden.get_shape()[1]), strides=1, activation=tf.nn.leaky_relu, padding="valid")


        context = tf.squeeze(context, axis=0)

        #compute context batch
        context = tf.tile(context, [BATCH_SIZE, 1])

        #context_input = tf.squeeze(context_input, axis=0)
        #samples = tf.concat([context_input[:SAMPLE_SIZE,4:], context], axis=1) # compute batch of M samples

        # compute the memory and context
        #samples = tf.concat([memory_tile[:BATCH_SIZE], context], axis=1)


        #samples_plh = tf.placeholder(shape=[BATCH_SIZE, bar.get_shape()[1]], dtype=tf.float32, name="samples_plh")
        context_plh = tf.placeholder(shape=context.get_shape(), dtype=tf.float32, name="context_plh")
        # tf.cast(tf.expand_dims(tf.argmax(policy_batch, axis=1), axis=1), tf.float32)
        loss_input = tf.concat([state_batch_plh, terminate_batch_plh,  action_batch_plh,  memory_tile_batch, policy_batch, sigma_tile_batch], axis=1)#tf.square(action_batch_plh - policy_batch) / (2 * tf.square(tf.exp(sigma_tile_batch)))], axis=1)

        """ loss network operates on BATCH_SIZE buffer samples
        these samples include M {state, termination_signal pairs}
        TF computes the rest of the required components : f_context, mem, action, policy distribution
        """
        loss_scope = "loss"
        with tf.variable_scope(loss_scope):

            hidden = tf.layers.dense(loss_input, 16, activation=tf.nn.leaky_relu, kernel_initializer=initializer)
            loss  = tf.layers.dense(hidden, 1, activation=None, kernel_initializer=initializer, name="loss")

        loss_es_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=context_scope) + \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=loss_scope)

        # same as memory as policy assign
        loss_es_assign_plhs = {}
        #loss_es_assign_ops = []
        for param in loss_es_params:
            param_value_plh = tf.placeholder(shape=param.get_shape(), dtype=tf.float32)
            assign_op = tf.assign(param, param_value_plh)
            loss_es_assign_plhs[param.name] = (param_value_plh, assign_op)

        loss_es_grad_plhs = {}
        loss_es_optimizer = tf.train.AdamOptimizer(loss_es_lr_init, beta1=0.0)
        l2_loss = sum([ tf.reduce_mean(tf.nn.l2_loss(param)) for param in loss_es_params ])

        l2_grads_and_vars = loss_es_optimizer.compute_gradients(l2_loss, loss_es_params)
        total_grads_and_vars = []
        for l2_grad_and_var in l2_grads_and_vars:
            l2_gradient, param = l2_grad_and_var
            param_grad_plh = tf.placeholder(shape=param.get_shape(), dtype=tf.float32)
            loss_es_grad_plhs[param.name] = param_grad_plh
            total_grads_and_vars.append((param_grad_plh - L2_BETA * l2_gradient, param))
        loss_es_gradients = loss_es_optimizer.apply_gradients(total_grads_and_vars)







            #loss_es_assign_ops.append(assign_op)

        # require params be assigned before evaluating
        #with tf.control_dependencies(loss_es_assign_ops):
        #    loss = tf.identity(loss)

        # allows workers to access the loss params
        loss_es_params_dict = {}
        for param in loss_es_params:
            loss_es_params_dict[param.name] = param

        policy_optimizer = tf.train.AdamOptimizer(policy_lr_init)

        # policy gradient update given M (SAMPLE_SIZE) loss samples
        """
        policy_gradients = {}
        learning_rate_policy_plh = tf.placeholder(shape=[1], dtype=tf.float32, name="learning_rate_policy_plh")
        for param in policy_params:
            policy_gradients[param.name] = tf.assign_add(param, -learning_rate_policy_plh * tf.clip_by_value(tf.reduce_mean(tf.gradients(loss, param), axis=0), -50, 50))
        """

        # use log policy for PPO loss and KL-divergence
        log_policy = - (BATCH_SIZE / 2) * tf.log(2 * np.pi) - ((BATCH_SIZE / 2) * tf.reduce_sum(sigma_tile_batch ** 2)) - 0.5 * tf.reduce_sum(((action_batch_plh - policy_batch) / sigma_tile_batch) ** 2)
        log_policy_old = - (BATCH_SIZE / 2) * tf.log(2 * np.pi) - ((BATCH_SIZE / 2) * tf.reduce_sum(sigma_tile_old ** 2)) - 0.5 * tf.reduce_sum(((action_batch_plh - policy_old) / sigma_tile_old) ** 2)
        #lambda_plh = tf.placeholder(shape=(), dtype=tf.float32, name="lambda_plh")
        alpha_plh = tf.placeholder(shape=(), dtype=tf.float32, name="alpha_plh")
        adv_plh = tf.placeholder(shape=[BATCH_SIZE, 1], name="adv_plh", dtype=tf.float32)
        kl_divergence = tf.identity(tf.reduce_sum((tf.exp(log_policy_old) * log_policy) - tf.exp(log_policy_old) * log_policy_old), name="kl_divergence")
        policy_ratio = tf.exp(log_policy - log_policy_old)

        ppo_surr_loss = tf.reduce_sum(policy_ratio * adv_plh)
        ppo_surr_constraint_clip = tf.clip_by_value(policy_ratio, 1 - PPO_CLIP_EPS, 1 + PPO_CLIP_EPS) * adv_plh
        ppo_loss = - tf.reduce_mean(tf.minimum(ppo_surr_loss, ppo_surr_constraint_clip)) + \
            PPO_KL_COEFF * kl_divergence

        policy_gradients = policy_optimizer.minimize( (1 - alpha_plh) * tf.reduce_mean(loss, axis=0) + \
            alpha_plh * (ppo_loss), var_list=policy_params)
        # memory gradient update  given M (SAMPLE_SIZE) loss samples
        """
        memory_gradients = {}
        learning_rate_memory_plh = tf.placeholder(shape=[1], dtype=tf.float32, name="learning_rate_memory_plh")
        for param in memory_params:
            memory_gradients[param.name] = tf.assign_add(param, -learning_rate_memory_plh * tf.clip_by_value(tf.reduce_mean(tf.gradients(loss, param), axis=0), -50, 50))
        """

        memory_optimizer = tf.train.AdamOptimizer(memory_lr_init)

        memory_gradients = memory_optimizer.minimize(tf.reduce_mean(loss), var_list=memory_params)
        # used to access param values
        #policy_params_dict = {}
        #for param in policy_params:
        #    policy_params_dict[param.name] = param

        # same as above
        #memory_params_dict = {}
        #for param in memory_params:
        #    memory_params_dict[param.name] = param
    return epg_graph,  loss_es_assign_plhs, loss_es_params_dict, loss_es_grad_plhs, loss_es_gradients,  policy_gradients, memory_gradients, loss, context, policy_old_copy, value_gradients


# Hyperparameters
NUM_STEPS = 128 * SAMPLE_SIZE # U
NUM_WORKERS = 256
TRAJ_SAMPLES = 7
ES_GAMMA = 0.95
PPO_GAMMA = 0.99
V = 64
SIGMA = 0.05
NUM_EPOCHS = 5000
LEARNING_RATE_LOSS_ES = 1e-2
LEARNING_DECAY = 0.99
SIGMA_DECAY = 1.0
NUM_PROCS = 8

PPO_LAMBDA = 0.95
#PPO_BETA_HIGH = 0.2
#PPO_BETA_LOW = 0.1
#PPO_KL_TARGET= 1e-3
#PPO_ALPHA = 1.2

#PPO_HORIZON = 1

ALPHA_DECAY = 0.99
ENV = "Humanoid-v1"

def run_inner_loop(gpu_lock, thread_lock, gym, tf, tid, barrier, loss_params, average_returns, run_sim=False, num_samples=None, num_epochs=None):
    """ Inner Loop run for each worker
        Samples from MDP and follows a randomly initialized policy
        updates both memory and policy every M steps

        Args:
            lock: thread lock
            barrier: thread barrier
            loss_params: the loss params for this worker
            average_returns: average returns list
    """




    #env = gym.make('BipedalWalker-v2')
    global ENV, PPO_LAMBDA, PPO_GAMMA, ES_GAMMA#, PPO_BETA_HIGH, PPO_BETA_LOW, PPO_KL_TARGET, PPO_ALPHA, PPO_HORIZON

    ALPHA = 1.0

    env = gym.make(ENV)
    #tf.reset_default_graph()
    epg_graph,  loss_es_assign_plhs, loss_es_params_dict, loss_es_grad_plhs, loss_es_gradients,  policy_gradients, memory_gradients, loss, context, policy_old_copy, value_gradients = build_graph(tf)

    with epg_graph.as_default() as g:
        policy_sample = g.get_tensor_by_name("policy_sample:0")
        sigma_sample = g.get_tensor_by_name("sigma_sample:0")
        value_ppo_sample = g.get_tensor_by_name("value_ppo_sample:0")
        kl_divergence = g.get_tensor_by_name("kl_divergence:0")

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True

        with tf.Session(graph=g, config=config) as sess:
            for _ in range(NUM_EPOCHS if not num_epochs else num_epochs):
                # buffers for state, term_signal, reward
                state_buffer = collections.deque([], maxlen=BUFFER_SIZE)
                term_buffer = collections.deque([], maxlen=BUFFER_SIZE)
                reward_buffer = collections.deque([], maxlen=BUFFER_SIZE)
                action_buffer = collections.deque([], maxlen=BUFFER_SIZE)

                # this can either be the n-step q_value or the full expected return depending on the HORIZON
                #q_values_buffer = collections.deque([], maxlen=SAMPLE_SIZE)
                # we collect rewards until there are HORIZON amount, then we compute the advantage
                #current_rewards = collections.deque([], maxlen=PPO_HORIZON)
                # whenever an advantage is computed, we need to keep track of the next state to compute the adv for
                #ppo_next_state_buffer = collections.deque([], maxlen=PPO_HORIZON)

                state_running_average  = np.array([])
                reward_running_average = None
                action_running_average = np.array([])

                state_running_stddev = np.array([])
                reward_running_stddev = 0
                action_running_stddev = np.array([])

                num_rewards = 0
                num_states = 0
                num_actions = 0
                gc.collect()
                if thread_lock:
                    while True:
                        with thread_lock:
                            if loss_params[tid % int(NUM_WORKERS/NUM_PROCS)] is not None:
                                break
                        time.sleep(0.1)

                sess.run(tf.global_variables_initializer())
                # assign perturb phi (loss params)
                if thread_lock and gpu_lock:
                    with thread_lock:
                        with gpu_lock:
                            for param in loss_params[tid % int(NUM_WORKERS/NUM_PROCS)].keys():
                                sess.run(loss_es_assign_plhs[param][1], feed_dict={loss_es_assign_plhs[param][0]: loss_params[tid % int(NUM_WORKERS/NUM_PROCS)][param]})
                            sess.run(policy_old_copy)
                else:
                    for param in loss_params[tid % int(NUM_WORKERS/NUM_PROCS)].keys():
                        sess.run(loss_es_assign_plhs[param][1], feed_dict={loss_es_assign_plhs[param][0]: loss_params[tid % int(NUM_WORKERS/NUM_PROCS)][param]})
                    sess.run(policy_old_copy)

                if barrier is not None:
                    barrier.wait()
                t = 0
                while t < NUM_STEPS:
                    s = env.reset()
                    start_state_ppo = s
                    done = False
                    rewards = 0
                    steps = 0

                    #state_running_average  = np.array([])
                    #reward_running_average = None


                    #state_running_stddev = np.array([])
                    #reward_running_stddev = 0


                    #num_rewards = 0
                    #num_states = 0

                    while done != True:
                        steps += 1

                        num_states += 1
                        old_mean_state = state_running_average
                        state_running_average = state_running_average +  (s - state_running_average)/num_states if state_running_average.any() else s

                        state_running_stddev = ((num_states - 1) * (state_running_stddev) + (s - old_mean_state) * (s - state_running_average)) / num_states if old_mean_state.any() else np.array([0] * num_states)


                        if sum(state_running_stddev > 0) != 0:
                            s = (s - state_running_average) / np.sqrt(state_running_stddev + 1e-5)
                        s[np.isnan(s)] = 0.0
                        if thread_lock and gpu_lock:
                            with thread_lock:
                                with gpu_lock:
                                    mean, sigma = sess.run((policy_sample, sigma_sample), feed_dict={"state_sample_plh:0": np.array([s])})
                        else:
                            mean, sigma = sess.run((policy_sample, sigma_sample), feed_dict={"state_sample_plh:0": np.array([s])})
                        a = np.random.multivariate_normal(mean=mean[0], cov=np.diag(np.exp(sigma[0])))


                        next_step, reward, done, info = env.step(a)

                        rewards += reward

                        num_rewards += 1
                        old_mean_reward = reward_running_average
                        reward_running_average = reward_running_average +  (reward - reward_running_average)/num_rewards if reward_running_average else reward
                        reward_running_stddev = ((num_rewards - 1) * reward_running_stddev + (reward - old_mean_reward) * (reward - reward_running_average)) / num_rewards if old_mean_reward else 0

                        if reward_running_stddev != 0 :
                            reward = (reward - reward_running_average) / np.sqrt(reward_running_stddev + 1e-5)

                        num_actions += 1
                        old_mean_action = action_running_average
                        action_running_average = action_running_average +  (a - action_running_average)/num_actions if action_running_average.any() else a
                        action_running_stddev = ((num_actions - 1) * action_running_stddev + (a - old_mean_action) * (a - action_running_average)) / num_actions if old_mean_action.any() else np.array([0] * NUM_ACTIONS)

                        if sum(action_running_stddev > 0) != 0:
                            a = (a - action_running_average) / np.sqrt(action_running_stddev + 1e-5)

                        state_buffer.append(s)
                        term_buffer.append([done])
                        reward_buffer.append([reward])
                        action_buffer.append(a)


                        t += 1

                        """
                        amount_of_rewards = len(current_rewards)

                        ppo_next_state_buffer.append(s)
                        if amount_of_rewards % PPO_HORIZON == 0 and not done and t > 0:
                            current_rewards.append(sess.run(value_ppo_sample, feed_dict={"state_sample_plh:0": [next_step]})[0])
                        else:
                            current_rewards.append(reward)

                        if (amount_of_rewards % PPO_HORIZON == 0 or done) and t > 0:
                            R = 0.0
                            for i in reversed(range(len(current_rewards))):
                                R = current_rewards[i] + GAMMA * R
                            q_values_buffer.append((R, start_state_ppo))

                            start_state_ppo = next_step
                        """

                        s = next_step

                        if t >= NUM_STEPS:
                            break
                        if done:
                            print("REWARDS %d STEPS %d REMAINING %d" %(rewards, steps, t))

                        # once we have enoguh samples perform policy and mem param update
                        if  t == BUFFER_SIZE or (t >= BUFFER_SIZE and t % SAMPLE_SIZE == 0):
                            # context is computed over the whole buffer, it is replicated BATCH_SIZE times
                            context_values = sess.run(context, feed_dict={"state_plh:0": list(state_buffer), "terminate_plh:0": list(term_buffer), "reward_plh:0": list(reward_buffer), "action_plh:0": list(action_buffer)})
                            state_sample = list(state_buffer)[-SAMPLE_SIZE:]
                            term_sample = list(term_buffer)[-SAMPLE_SIZE:]
                            reward_sample = list(reward_buffer)[-SAMPLE_SIZE:]
                            action_sample = list(action_buffer)[-SAMPLE_SIZE:]

                            # see original PPO papers for truncated advantage summation (similar to n-step TD)
                            value_sample = sess.run(value_ppo_sample, feed_dict={"state_sample_size_plh:0": state_sample})
                            td_deltas = reward_sample + PPO_GAMMA * (1 - np.array(term_sample)) * np.vstack([value_sample[1:], value_sample[-1]]) - value_sample
                            td_delta_coeff = PPO_GAMMA * PPO_LAMBDA * (1 - np.array(term_sample))
                            #print(td_deltas.shape, td_delta_coeff.shape, value_sample.shape, value_sample[-1].shape)
                            advantages = collections.deque([])
                            for m in reversed(range(len(td_deltas))):
                                advantages.appendleft(td_deltas[m] + td_delta_coeff[m] * (0 if m == len(td_deltas) - 1 else advantages[0]))
                            advantages = np.array(list(advantages))
                            q_sample = advantages + value_sample
                            adv_sample = (advantages - np.mean(advantages)) / np.std(advantages)
                            # we randomly sample batches for param update
                            joint = list(zip(state_sample, term_sample, reward_sample, action_sample, adv_sample, q_sample))
                            random.shuffle(joint)
                            random.shuffle(joint)
                            random.shuffle(joint)
                            random.shuffle(joint)

                            state_batch, term_batch, reward_batch, action_batch, adv_batch, q_batch = zip(*joint)

                            num_batches = int(SAMPLE_SIZE / BATCH_SIZE)
                            for i in range(num_batches):

                                state_mb = state_batch[BATCH_SIZE * i:BATCH_SIZE * (i + 1)]
                                term_mb = term_batch[BATCH_SIZE * i:BATCH_SIZE * (i + 1)]
                                reward_mb = reward_batch[BATCH_SIZE * i:BATCH_SIZE * (i + 1)]
                                action_mb = action_batch[BATCH_SIZE * i:BATCH_SIZE * (i + 1)]
                                adv_mb = adv_batch[BATCH_SIZE * i: BATCH_SIZE * (i + 1)]
                                q_mb = q_batch[BATCH_SIZE * i: BATCH_SIZE * (i + 1)]

                                """
                                q_values = list(q_values_buffer)
                                advantages = []
                                states_ppo = []
                                for q in q_values[BATCH_SIZE * i: BATCH_SIZE * (i+1)]:
                                    value, state = q
                                    advantages.append(value - sess.run(value_ppo_sample, feed_dict={"state_sample_plh:0": [state]})[0])
                                    states_ppo.append(state)

                                advantages = np.array(advantages)
                                advantages = (advantages - np.mean(advantages)) / np.std(advantages)
                                """

                                # policy and mem gradient update
                                #if run_sim:
                                if thread_lock and gpu_lock:
                                    with thread_lock:
                                        with gpu_lock:
                                            sess.run(memory_gradients, feed_dict={"context_plh:0": context_values, "state_batch_plh:0": state_mb,
                                                "terminate_batch_plh:0": term_mb, "reward_batch_plh:0": reward_mb, "action_batch_plh:0": action_mb})

                                            sess.run(policy_gradients, feed_dict={"context_plh:0": context_values, "state_batch_plh:0": state_mb,
                                                "terminate_batch_plh:0": term_mb, "reward_batch_plh:0": reward_mb, "action_batch_plh:0": action_mb,
                                                "alpha_plh:0": ALPHA, "adv_plh:0": adv_mb})

                                            #q_values_batch, state_q_mb = map(list, zip(*q_values[BATCH_SIZE * i: BATCH_SIZE * (i+1)]))
                                            #if len(list(np.array(q_values_batch).shape)) == 1:
                                            #    q_values_batch =  np.expand_dims(np.array(q_values_batch), axis=1)
                                            sess.run(value_gradients, feed_dict={"q_values_plh:0": q_mb, "state_batch_plh:0": state_mb})

                                            #kl_div = sess.run(kl_divergence, feed_dict={"state_batch_plh:0": states_ppo, "action_batch_plh:0": action_mb})
                                            sess.run(policy_old_copy)

                                else:
                                            sess.run(memory_gradients, feed_dict={"context_plh:0": context_values, "state_batch_plh:0": state_mb,
                                                "terminate_batch_plh:0": term_mb, "reward_batch_plh:0": reward_mb, "action_batch_plh:0": action_mb})
                                            sess.run(policy_gradients, feed_dict={"context_plh:0": context_values, "state_batch_plh:0": state_mb,
                                                "terminate_batch_plh:0": term_mb, "reward_batch_plh:0": reward_mb, "action_batch_plh:0": action_mb,
                                                "alpha_plh:0": ALPHA, "adv_plh:0": adv_mb})

                                            #q_values_batch, state_q_mb = map(list, zip(*q_values[BATCH_SIZE * i: BATCH_SIZE * (i+1)]))
                                            #if len(list(np.array(q_values_batch).shape)) == 1:
                                            #    q_values_batch =  np.expand_dims(np.array(q_values_batch), axis=1)
                                            sess.run(value_gradients, feed_dict={"q_values_plh:0": q_mb, "state_batch_plh:0": state_mb})


                                            #kl_div = sess.run(kl_divergence, feed_dict={"state_batch_plh:0": states_ppo, "action_batch_plh:0": action_mb})

                                            sess.run(policy_old_copy)
                                """
                                if kl_div > PPO_BETA_HIGH * PPO_KL_TARGET:
                                    PPO_LAMBDA *= PPO_ALPHA
                                elif kl_div < PPO_BETA_LOW * PPO_KL_TARGET:
                                    PPO_LAMBDA *= 1 / PPO_ALPHA
                                """


                                #learning_rate_policy *= LEARNING_DECAY
                                #learning_rate_memory *= LEARNING_DECAY


                """ Now we use learned policy to sample some trajectories,
                and compute the average return from all trajectories
                """

                returns = []
                for tr in range(TRAJ_SAMPLES if not num_samples else num_samples):
                    s = env.reset()
                    done = False
                    R = 0
                    rewards = []
                    steps = 0
                    reward_total = 0
                    #num_rewards = 0
                    #reward_running_stddev = 0
                    #reward_running_average = None

                    #state_running_average  = np.array([])


                    #state_running_stddev = np.array([])


                    #num_states = 0

                    while done != True:
                        num_states += 1
                        old_mean_state = state_running_average
                        state_running_average = state_running_average +  (s - state_running_average)/num_states if state_running_average.any() else s

                        state_running_stddev = ((num_states - 1) * (state_running_stddev) + (s - old_mean_state) * (s - state_running_average)) / num_states if old_mean_state.any() else np.array([0] * num_states)

                        if sum(state_running_stddev > 0) != 0:
                            s = (s - state_running_average) / np.sqrt(state_running_stddev + 1e-5)
                        s[np.isnan(s)] = 0.0

                        if thread_lock and gpu_lock:
                            with thread_lock:
                                with gpu_lock:
                                    mean, sigma = sess.run((policy_sample, sigma_sample), feed_dict={"state_sample_plh:0": np.array([s])})
                        else:
                            mean, sigma = sess.run((policy_sample, sigma_sample), feed_dict={"state_sample_plh:0": np.array([s])})
                        #a = mean[0]
                        a = np.random.multivariate_normal(mean=mean[0], cov=np.diag(np.exp(sigma[0])))
                        #a[a > 1.0] = 1.0
                        #a[a < -1.0] = -1.0
                        s, reward, done, info = env.step(a)
                        """
                        if steps > 500:
                            reward = -500
                            done = True
                        """
                        """
                        num_rewards += 1
                        old_mean_reward = reward_running_average
                        reward_running_average = reward_running_average +  (reward - reward_running_average)/num_rewards if reward_running_average else reward
                        reward_running_stddev = ((num_rewards - 1) * reward_running_stddev + (reward - old_mean_reward) * (reward - reward_running_average)) / num_rewards if old_mean_reward else 0

                        if reward_running_stddev != 0 :
                            reward = (reward - reward_running_average) / np.sqrt(reward_running_stddev)
                        """
                        reward_total += reward
                        if run_sim:
                            env.render()
                        rewards.append(reward)
                        steps += 1


                    print("TRAJ REWARDS %d STEPS %d REMAINING %d" %(reward_total, steps, tr))
                    for i in reversed(range(len(rewards))):
                        R = rewards[i] + ES_GAMMA * R
                    returns.append(R)
                if average_returns is not None:
                    average_returns[tid % int(NUM_WORKERS/NUM_PROCS)] = (tid, sum(returns) / TRAJ_SAMPLES)
                    with thread_lock:
                        loss_params[tid % int(NUM_WORKERS/NUM_PROCS)] = None
                    ALPHA *= ALPHA_DECAY
    epg_graph = None
    g = None
    loss_es_assign_plhs = None
    loss_es_params_dict = None
    policy_gradients = None
    memory_gradients = None
    loss = None
    context = None
    gc.collect()

def relative_ranks(x):
    def ranks(x):
        ranks = np.zeros(len(x), dtype=int)
        ranks[x.argsort()] = np.arange(len(x))
        return ranks

    y = ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    return y / (x.size - 1.) - 0.5

def run_outer_loop():
    """ Runs the outer loop of the algorithm
    Spawns the workers and performs updates to phi (the ES loss params)
    """
    global LEARNING_RATE_LOSS_ES, LEARNING_DECAY, SIGMA
    queue = multiprocessing.Queue()
    def spawn_threads(event, gpu_lock, vectors_queue, rewards_queue, tid_start):
        import gym
        import tensorflow as tf
        import threading
        thread_lock = threading.Lock()
        barrier = threading.Barrier(int(NUM_WORKERS/NUM_PROCS))
        threads = []
        while not event.is_set():
            time.sleep(.1)
        returns = [None] * int(NUM_WORKERS/NUM_PROCS)
        num_workers_per_proc = int(NUM_WORKERS/NUM_PROCS)
        loss_params = [None] * int(NUM_WORKERS/NUM_PROCS)

        for tid in range(tid_start, tid_start + num_workers_per_proc):
            threads.append(threading.Thread(target=run_inner_loop, args=(gpu_lock, thread_lock,gym, tf, tid, barrier, loss_params, returns)))
        for thread in threads:
            thread.start()

        while event.is_set():
            vectors = vectors_queue.get(block=True, timeout=None)

            if not event.is_set():
                break
            num_vecs = len(vectors)

            #print(tid_start, num_workers_per_proc, num_vecs)
            with thread_lock:

                for i in range(num_workers_per_proc):
                    loss_params[i] = vectors[math.floor(( (i) * (num_vecs)/(num_workers_per_proc)))]

            while True:
                with thread_lock:
                    j = 0
                    for loss in loss_params:
                        if loss is None:
                            j += 1
                if j == len(loss_params):
                    break
                else:
                    time.sleep(10)
            queue.put(returns)
            gc.collect()
        for thread in threads:
            thread.join()
        exit(0)
    events = [multiprocessing.Event()] * NUM_PROCS
    for event in events:
        event.set()
    vector_queues = [multiprocessing.Queue(maxsize=1)] * NUM_PROCS
    gpu_lock = multiprocessing.Lock()
    processes = [ multiprocessing.Process(target=spawn_threads, args=(events[i], gpu_lock, vector_queues[i], queue, i * int(NUM_WORKERS/NUM_PROCS))) for i in range(NUM_PROCS)]

    for process in processes:
        process.start()

    import tensorflow as tf
    epg_graph, _, loss_es_params_dict, loss_es_grad_plhs, loss_es_gradients, _, _, _, _, _, _ = build_graph(tf)
    with epg_graph.as_default():
        # initialize phi (the loss ES params)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())#tf.variables_initializer(loss_es_params_dict.values()))
            loss_es_params_values = sess.run(loss_es_params_dict)
            with open(ENV + "-epg_loss_params.pkl", "wb") as f:
                pickle.dump(loss_es_params_values, f, pickle.HIGHEST_PROTOCOL)
                #loss_es_params_values = pickle.load(f)
            #return loss_es_params_values
            for e in range(NUM_EPOCHS):
                # construct perturbed phi (loss) params
                epsilon_vectors = []
                normal_vectors = []

                for i in range(V):
                    param_pertubed = {}
                    normal_vectors_dict = {}
                    for param in loss_es_params_values.keys():
                        param_shape = loss_es_params_values[param].shape#map(lambda x: int(x), loss_es_params_dict[param].get_shape())
                        #num_params = reduce(operator.mul, param_shape)

                        normal = tf.random_normal(shape=param_shape)
                        normal_vectors_dict[param] = sess.run(normal)
                        param_pertubed[param] = loss_es_params_values[param] + SIGMA * normal_vectors_dict[param]#np.reshape(np.random.multivariate_normal([0] * num_params, np.eye(num_params, num_params)), param_shape)
                    epsilon_vectors.append(param_pertubed)
                    normal_vectors.append(normal_vectors_dict)

                for i, vector_queue in enumerate(vector_queues):
                    vector_queue.put(epsilon_vectors[i * int(V/NUM_PROCS): (i+1) * int(V/NUM_PROCS)])


                average_returns = [None] * NUM_WORKERS
                workers = 0
                while workers != NUM_WORKERS:
                    returns = queue.get()
                    for r in returns:
                        average_returns[r[0]] = r[1]
                    workers += int(NUM_WORKERS/NUM_PROCS)

                loss_es_grad_feed_dict = {}
                average_returns = np.array(average_returns)
                average_returns = list(relative_ranks(average_returns))
                #print(average_returns)
                #average_returns = (average_returns - np.mean(average_returns)) / np.std(average_returns)
                #average_returns[np.sign(average_returns) == -1] = 0.0
                #average_returns = list(average_returns)
                # compute ES gradients and update
                for param in loss_es_params_values.keys():
                    F = []
                    num_workers_per_set = int(NUM_WORKERS / V) # guarantee divis
                    for i in range(V):
                        F.append((sum(average_returns[num_workers_per_set *i: num_workers_per_set*(i+1)])/(num_workers_per_set)) * normal_vectors[i][param])
                    grad = sum(F)/(V)
                    loss_es_grad_feed_dict[loss_es_grad_plhs[param]] = grad
                sess.run(loss_es_gradients, feed_dict=loss_es_grad_feed_dict)
                #LEARNING_RATE_LOSS_ES *= LEARNING_DECAY
                #SIGMA *= SIGMA_DECAY

                #for param in loss_es_params_values.keys():
                #    sess.run(loss_es_assign_plhs[param][1], feed_dict={loss_es_assign_plhs[param][0]: loss_es_params_values[param]})


                print("EPOCH %d " % e)#, average_returns)
                #print("AVERAGE %f" % (sum(average_returns)/ NUM_WORKERS))
                loss_es_params_values = sess.run(loss_es_params_dict)
                with open(ENV + "-epg_loss_params.pkl", "wb") as f:
                    pickle.dump(loss_es_params_values, f, pickle.HIGHEST_PROTOCOL)
                run_inner_loop(None, None, gym, tf, 0,  None, [loss_es_params_values], None, run_sim=True, num_samples=16, num_epochs=1)

                gc.collect()
            for event in events:
                event.clear()
            for vector_queue in vector_queues:
                vector_queue.put([])
            for process in processes:
                process.join()
        return loss_es_params_values




loss_params = run_outer_loop()
print("DONE")
import tensorflow as tf
run_inner_loop(None, None, gym, tf, 0,  None, loss_params, None, run_sim=True)
