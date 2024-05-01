import pickle

import gymnasium as gym
import numpy as np


env = gym.make("LunarLander-v2", render_mode="human")


model = pickle.load(open("model-reinforce.pickle", "rb"))
weight = model["network.0.weight"]
bias = model["network.0.bias"]

def softmax(x):
    return(np.exp(x) / np.exp(x).sum())

episodes = 10
total_reward = 0.0
for i in range(episodes):
    observation, _ = env.reset()
    while True:
        actions = softmax(np.matmul(observation, weight.T) + bias)
        action = np.random.choice(np.array(range(4)), 1, p=actions)[0]
        if observation[6] == 1. and observation[7] == 1.:
            action = 0
        observation, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward

        if terminated or truncated:
            break
env.close()
print("Total reward", total_reward / episodes)
