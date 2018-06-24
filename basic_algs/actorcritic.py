import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
import numpy as np

"""
Actor-Critic for Cartpole
"""


env = gym.make('CartPole-v0')
env.reset()



#estimate value function
state = tf.placeholder(shape=(None,4),dtype=tf.float32,name='state')
with tf.variable_scope('critic'):
    h1 = slim.fully_connected(state,4,activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.variance_scaling_initializer())
    #h2 = slim.fully_connected(h1,30,activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.variance_scaling_initializer())
    V = slim.fully_connected(h1,1,activation_fn=None,weights_initializer=tf.contrib.layers.variance_scaling_initializer())

#policy
with tf.variable_scope('actor'):
    h1 = slim.fully_connected(state,4,activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.variance_scaling_initializer())
    #h2 = slim.fully_connected(h1,30,activation_fn=tf.nn.sigmoid,weights_initializer=tf.contrib.layers.variance_scaling_initializer())
    logits = slim.fully_connected(h1,1,activation_fn=None,weights_initializer=tf.contrib.layers.variance_scaling_initializer())
    action_fn = tf.nn.sigmoid(logits)

#network params
critic_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='critic')
actor_param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='actor')

alpha = .001
beta = .01
gamma = .7

delta_ph = tf.placeholder(shape=(None,1),dtype=tf.float32,name='delta_ph')



init = tf.global_variables_initializer()

#gradients
actor_gradient = { v.name: tf.scalar_mul(alpha,tf.clip_by_value(tf.gradients(tf.log(tf.squeeze(action_fn)),v)[0]*tf.squeeze(delta_ph),-5,5)) for v in actor_param}
critic_gradient = { v.name:tf.scalar_mul(beta,tf.clip_by_value(tf.squeeze(delta_ph)*tf.gradients(tf.squeeze(V),v)[0],-5,5)) for v in critic_param}

def get_action_n(sess,state):
    #print(state)
    p = sess.run(action_fn,feed_dict={'state:0':[state]})
    #print(state,p)
    return 1  if p <.5 else 0

def update(d,g):
    for key in d.keys():
        d[key]+=g[key]

actor_grads = {v.name: 0.0 for v in actor_param}
critic_grads = {v.name: 0.0 for v in critic_param}
with tf.Session() as sess:
    init.run()
    for episode in range(1,1001):
        done = False,
        G,reward = 0,0
        state = env.reset()
        action = get_action_n(sess,state)
        steps = 0
        while done!=True:
            prev_state = state
            state, reward,done,info = env.step(action)
            import pdb;pdb.set_trace()
            env.render()
            prev_action = action
            action = get_action_n(sess,state)
            #print(action)
            #TD-delta computation
            V_next = sess.run(V,feed_dict={'state:0':[state]})
            V_prev = sess.run(V,feed_dict={'state:0':[prev_state]})
            #used to approximate the advantage function
            delta = (reward+gamma*V_next - V_prev)

            a = sess.run(actor_gradient,feed_dict={'state:0':[prev_state],'delta_ph:0':delta})
            a = a if action else {v: -1*a[v] for v in a.keys()}
            update(actor_grads,a)
            update(critic_grads,sess.run(critic_gradient,feed_dict={'delta_ph:0':delta,'state:0':[prev_state]}))
            steps+=1
        #if episode % 1 ==0:
            #print('update')
    #if episode % 2 ==0:
        for k in actor_grads.keys():
            sess.run(tf.assign_add(tf.get_default_graph().get_tensor_by_name(k),actor_grads[k]))
        actor_grads = {v.name: 0.0 for v in actor_param}
        for k in critic_grads.keys():
            sess.run(tf.assign_add(tf.get_default_graph().get_tensor_by_name(k),critic_grads[k]))
        critic_grads = {v.name: 0.0 for v in critic_param}
            #print(sess.run(actor_param[0]))
            #print(sess.run(critic_param[0]))
        print('FAILED %d'%steps)
