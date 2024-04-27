import pickle

import gymnasium as gym
import numpy as np

env = gym.make("LunarLander-v2", render_mode="human")


model = pickle.load(open("model.pickle", "rb"))
weight = model["network.weight"].numpy()
bias = model["network.bias"].numpy()

episodes = 10
total_reward = 0.0
for i in range(episodes):
    observation, _ = env.reset()
    while True:
        action = np.argmax(np.matmul(observation, weight.T) + bias)
        observation, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward

        if terminated or truncated:
            break
env.close()
print("Total reward", total_reward / episodes)
