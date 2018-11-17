#!/usr/bin/env python
# coding: utf-8

# In[1]:



import gym
import itertools
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf
import collections

if "../" not in sys.path:
  sys.path.append("../") 

# In[2]:


env = gym.envs.make("MountainCar-v0")


# In[3]:


class PolicyEstimator():
    """
    Policy Function approximator. 
    """
    
    def __init__(self, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [1, int(env.observation_space.shape[0])], "state")
            self.action = tf.placeholder(dtype=tf.int32, name="action")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=self.state,
                num_outputs=env.action_space.n,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.action_probs = tf.squeeze(tf.nn.softmax(self.output_layer))
            self.picked_action_prob = tf.gather(self.action_probs, self.action)

            # Loss and train op
            self.loss = -tf.log(self.picked_action_prob) * self.target

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())
    
    def predict(self, state, sess=None):
        state_ten = state.reshape((1, int(env.observation_space.shape[0])))
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, { self.state: state_ten })

    def update(self, state, target, action, sess=None):
        state_ten = state.reshape((1, int(env.observation_space.shape[0])))
        sess = sess or tf.get_default_session()
        feed_dict = { self.state: state_ten, self.target: target, self.action: action  }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


# In[4]:


class ValueEstimator():
    """
    Value Function approximator. 
    """
    
    def __init__(self, learning_rate=0.1, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [1, int(env.observation_space.shape[0])], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just table lookup estimator
#             state_one_hot = tf.one_hot(self.state, int(env.observation_space.shape[0]))
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=self.state,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())        
    
    def predict(self, state, sess=None):
        state_ten = state.reshape((1, int(env.observation_space.shape[0])))
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, { self.state: state_ten })

    def update(self, state, target, sess=None):
        state_ten = state.reshape((1, int(env.observation_space.shape[0])))
        sess = sess or tf.get_default_session()
        feed_dict = { self.state: state_ten, self.target: target }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


# In[9]:


def actor_critic(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0):
    """
    Actor Critic Algorithm. Optimizes the policy 
    function approximator using policy gradient.
    
    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized 
        estimator_value: Value function approximator, used as a critic
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    episode_reward, episode_len = 0,0
    
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    
    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action
        state = env.reset()
        episode = []
        
        # One step in the environment
        for t in itertools.count():
            
            img = env.render(mode='rgb_array')  
            print(type(img))
            # Take a step
            action_probs = estimator_policy.predict(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            
            # Keep track of the transition
            episode.append(Transition(
              state=state, action=action, reward=reward, next_state=next_state, done=done))
            
            # Update statistics
            episode_reward += reward
            episode_len = t
            
            # Calculate TD Target
            value_next = estimator_value.predict(next_state)
            td_target = reward + discount_factor * value_next
            td_error = td_target - estimator_value.predict(state)
            
            # Update the value estimator
            estimator_value.update(state, td_target)
            
            # Update the policy estimator
            # using the td error as our advantage estimate
            estimator_policy.update(state, td_error, action)
            
            # Print out which step we're on, useful for debugging.
            print("\rStep {} @ Episode {}/{} ({})".format(
                    t, i_episode + 1, num_episodes, episode_reward), end="")

            if done:
                break
                
            state = next_state
    
    return episode_reward, episode_len


# In[10]:


tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator()
value_estimator = ValueEstimator()

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Note, due to randomness in the policy the number of episodes you need to learn a good
    # policy may vary. ~300 seemed to work well for me.
    episode_reward, episode_len = actor_critic(env, policy_estimator, value_estimator, 300)
writer.close()


# In[ ]:




