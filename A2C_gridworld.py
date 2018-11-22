#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
import gym
import collections
import itertools
import matplotlib.pyplot as plt
from gridworld import Gridworld
# In[3]:


env = gym.make('Gridworld-v0')

# In[ ]:


class Actor():
    def __init__(self, lr=0.001):
        self.state = tf.placeholder(tf.float32, [1, int(env.observation_space.n)], "State")
        self.action = tf.placeholder(tf.int32, None, "Action")
        self.target = tf.placeholder(tf.float32, None, "Target")

        with tf.variable_scope('Actor'):
            layer1 = tf.layers.dense(
                inputs=self.state,
                units=20,  # hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.1),  # Initializing weights
                bias_initializer=tf.constant_initializer(0.1),  # Adding bias
                name="Layer_1"
            )

            # Second layer with inputs from Layer 1 and outputs action probs
            self.output_layer = tf.layers.dense(
                inputs=layer1,
                units=env.action_space.n,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.1),  # Initializing weights
                bias_initializer=tf.constant_initializer(0.1),  # Adding bias
                name="Action_Prob_Layer"
            )

        with tf.variable_scope('Loss'):
            self.action_probs = tf.squeeze(tf.nn.softmax(self.output_layer))
            self.picked_action_prob = tf.gather(self.action_probs, self.action)

            # Loss and train op
            self.loss = -tf.log(self.picked_action_prob) * self.target

        with tf.variable_scope('train_actor'):
            self.train_operation = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        state_ten = one_hot(state)
        feed_dict = {self.state: state_ten, self.target: target, self.action: action}
        _, loss = sess.run([self.train_operation, self.loss], feed_dict)
        return loss

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        state_ten = one_hot(state)
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, {self.state: state_ten})


class Critic():
    def __init__(self, lr=0.01):
        self.state = tf.placeholder(tf.float32, [1, int(env.observation_space.n)], "State")
        self.target = tf.placeholder(tf.float32, None, "Target")

        with tf.variable_scope('Critic'):
            layer1 = tf.layers.dense(
                inputs=self.state,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='Layer1'
            )

            self.output_layer = tf.layers.dense(
                inputs=layer1,
                units=1,  # output unitslearning_rate
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='Value_Estimate_Layer'
            )

        with tf.variable_scope('Loss'):
            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)
        with tf.variable_scope('train_critic'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        state_ten = one_hot(state)
        feed_dict = {self.state: state_ten, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        state_ten = one_hot(state)
        return sess.run(self.value_estimate, {self.state: state_ten})

def one_hot(state):
    state_n = np.zeros((1,env.observation_space.n))
    state_n[0][state] = 1
    return state_n

def actor_critic(env, actor, critic, iterations=300, gamma=1.0):
    reward_list, epi_len_list = [], []
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    for i_episode in range(iterations):
        # Reset the environment and pick the first action
        state = env.reset()
        transitions = []
        episode_reward, episode_len = 0, 0

        for t in itertools.count():

            # Take a step
            action_probs = actor.predict(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            # Keep track of the transition
            transitions.append(Transition(
                state=state, action=action, reward=reward, next_state=next_state, done=done))

            # Update statistics
            episode_reward += reward
            episode_len = t

            # Calculate TD Target
            value_next = critic.predict(next_state)
            td_target = reward + gamma * value_next
            td_error = td_target - critic.predict(state)

            # Update the value estimator
            critic.update(state, td_target)

            # Update the policy estimator
            # using the td error as our advantage estimate
            actor.update(state, td_error, action)

            # Print out which step we're on, useful for debugging.
            print("\rStep {} @ Episode {}/{} ({})".format(
                t, i_episode + 1, iterations, episode_reward), end="")

            if done:
                break

            state = next_state
        reward_list.append(episode_reward)
        epi_len_list.append(iterations)

    return reward_list, epi_len_list


tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = Actor()
value_estimator = Critic()

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Note, due to randomness in the policy the number of episodes you need to learn a good
    # policy may vary. ~300 seemed to work well for me.
    reward_list, epi_len_list = actor_critic(env, policy_estimator, value_estimator)
writer.close()

#Plot Rewards per episode
plt.plot(np.arange(len(reward_list)), reward_list)
plt.savefig('./Plots/Plot.png')