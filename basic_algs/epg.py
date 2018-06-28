import tensorflow as tf
import gym



""" Evolved Policy Gradients """




BUFFER_SIZE = 512
MEMORY_SIZE = 32
SAMPLE_SIZE = 32
NUM_ACTIONS = 2


tf.reset_default_graph()

epg_graph = tf.Graph()

with epg_graph.as_default() as g:

    memory_scope = "memory"
    with tf.variable_scope(memory_scope):
        # memory units
        memory = tf.layers.dense(tf.ones([1, MEMORY_SIZE]), MEMORY_SIZE, activation=tf.nn.tanh, use_bias=False)

    state_plh = tf.placeholder(shape=[BUFFER_SIZE, 1])
    terminate_plh = tf.placeholder(shape=[BUFFER_SIZE, 1])

    initializer = tf.contrib.layers.variance_scaling_initializer()
    policy_scope = "policy"
    with tf.variable_scope(policy_scope):
        hidden = tf.layers.dense(state_plh, 64, activation=tf.nn.tanh, kernel_initializer=initializer)
        hidden = tf.layers.dense(hidden, 64, activation=tf.nn.tanh, kernel_initializer=initializer)
        policy = tf.layers.dense(hidden, NUM_ACTIONS, activation=None, kernel_initializer=initializer)
        policy = tf.nn.softmax(policy, axis=1)


    context_scope = "context"
    with tf.variable_scope(context_scope):
        memory_tile = tf.tile(memory, [BUFFER_SIZE])
        context_input = tf.concat([state_plh, terminate_plh, tf.argmax(policy, axis=1), memory, policy], axis=1)

        hidden = tf.layers.conv1d(context_input, 10, 8, strides=7, activation=tf.nn.elu, padding="same")
        hidden = tf.layers.conv1d(hidden, 10, 4, strides=2, activation=tf.nn.elu, padding="same")
        context = tf.layers.conv1d(hidden, 32, hidden.get_shape()[1], strides=1, activation=tf.nn.elu, padding="same")


    loss_scope = "loss"
    with tf.variable_scope(loss_scope):
        context = tf.tile(context, [SAMPLE_SIZE])

        samples = tf.concat([context_input[:SAMPLE_SIZE], context], axis=1)

        hidden = tf.layers.dense(samples, 16, activation=tf.nn.elu, kernel_initializer=initializer)
        hidden = tf.layers.dense(hidden, SAMPLE_SIZE, activation=None, kernel_initializer=initializer)
