#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
import numpy as np
import gym


# In[3]:


env = gym.make('CartPole-v0')


# In[ ]:


class Actor():
    def __init__(self, n_features, n_actions, lr = 0.001):
        self.state = tf.placeholder(tf.float32, [1, n_features], "state")
        self.action = tf.placeholder(tf.int32, None, "action")
        self.target = tf.placeholder(tf.float32, None, "TD_Error")
        
        with tf.variable_scope('Actor'):
            layer1 = tf.layers.dense(
                inputs = self.state,
                units = 20, #hidden units
                activation = tf.nn.relu,
                kernel_initializer = tf.random_normal_initializer(0, 0.1), #Initializing weights
                bias_initializer = tf.constant_initializer(0.1), #Adding bias
                name = "Layer_1" 
            )
            
            #Second layer with inputs from Layer 1 and outputs action probs
            self.action_probs = tf.layers.dense(
                inputs = layer1,
                units = n_actions, 
                activation = tf.nn.relu,
                kernel_initializer = tf.random_normal_initializer(0, 0.1), #Initializing weights
                bias_initializer = tf.constant_initializer(0.1), #Adding bias
                name = "Action_Prob Layer" 
            )
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




