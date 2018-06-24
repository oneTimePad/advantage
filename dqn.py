import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
import numpy as np
import os
from collections import deque
env = gym.make('CartPole-v0')
env.reset()


state = tf.placeholder(shape=(None,4),dtype=tf.float32,name='state')

with tf.variable_scope('estimator_q_network'):
    h1 = slim.fully_connected(state,4,activation_fn=tf.nn.sigmoid,weights_initializer=tf.contrib.layers.variance_scaling_initializer())
    #h2 = slim.fully_connected(h1,80,activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.variance_scaling_initializer())
    est_q  = slim.fully_connected(h1,2,activation_fn=None,weights_initializer=tf.contrib.layers.variance_scaling_initializer())

est_q_params = {v.name: v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='estimator_q_network')}

with tf.variable_scope('actor_q_network'):
    h1 = slim.fully_connected(state,4,activation_fn=tf.nn.sigmoid,weights_initializer=tf.contrib.layers.variance_scaling_initializer())
    #h2 = slim.fully_connected(h1,80,activation_fn=tf.nn.relu,weights_initializer=tf.contrib.layers.variance_scaling_initializer())
    act_q  = slim.fully_connected(h1,2,activation_fn=None,weights_initializer=tf.contrib.layers.variance_scaling_initializer())

k = 0
def get_action(sess,state,eps=.3):
    global k
    r = np.random.uniform()
    if r <eps:
        k+=1
        return int((np.random.uniform() > .5))
    return np.argmax(sess.run(act_q,feed_dict={'state:0':[state]}))

act_q_params = {v.name: v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='actor_q_network')}

copy_ops = [tf.assign(act_q_params[os.path.join('actor_q_network',*k.split('/')[1:])],est_q_params[k]) for k in est_q_params.keys()]

#batch of sarsa memories
one_step_return = tf.placeholder(shape=(None,1),dtype=tf.float32,name='one_step_return')
action_chosen   = tf.placeholder(shape=(None),dtype=tf.int64)

estimator_q_value = tf.reduce_sum(tf.one_hot(action_chosen,2)*est_q,axis=1,keep_dims=True)

loss = tf.reduce_mean(tf.square(one_step_return-estimator_q_value))

optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

memories = np.empty((0,4))
gamma = 5
BATCH_SIZE = 200
init = tf.global_variables_initializer()
total_steps = 0
with tf.Session() as sess:
    init.run()
    for episode in range(1,1000000):
        done = False
        state = env.reset()
        action = get_action(sess,state)
        steps = 0
        while done !=True:
            env.render()
            next_state,reward,done,info = env.step(action)
            next_action = get_action(sess,next_state)
            memories = np.append(memories,np.array([[state,action,reward,next_state]]),axis=0)
            state = next_state
            action = next_action
            steps+=1
            total_steps+=1
            print(len(memories))
            if len(memories) >1000 and total_steps %150 == 0:
                print('learn')
                choices = np.random.randint(0,len(memories),BATCH_SIZE)
                #import pdb;pdb.set_trace()
                batch = (np.array(memories)[choices,:])
                next_states =batch[:,3]

                next_q_values = np.amax(sess.run(act_q,feed_dict={'state:0':np.vstack(next_states)}),axis=1).reshape(BATCH_SIZE,1)
                rewards = np.array(batch[:,2]).reshape(BATCH_SIZE,1)
                targets = rewards+gamma*next_q_values
                sess.run(train_op,feed_dict={one_step_return:targets,action_chosen:np.array(batch[:,1]),'state:0':np.vstack(batch[:,0])})
                sess.run(copy_ops)
                memories =np.delete(memories,choices,axis=0)
        print('FAILED %d' % steps)
        #print([sess.run(act_q_params[k]) for k in act_q_params])
