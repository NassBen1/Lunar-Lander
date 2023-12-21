import gymnasium as gym
import numpy as np

import time
from collections import deque, namedtuple # for replay memory
import random
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE

#Valeur à changer pour le nombre d'itération
tf.random.set_seed(42)

# Hyperparameters
memorysize = 1000000
gamma = 0.99
alpha = 0.0001
numbers_steps_for_update = 4


env = gym.make("LunarLander-v2", render_mode="human")

state_size = env.observation_space.shape
action_size = env.action_space.n

current_state = env.reset()

print("State size:", state_size)
print("Action size:", action_size)

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()