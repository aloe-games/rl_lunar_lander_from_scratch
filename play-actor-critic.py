import pickle

import gymnasium as gym
import numpy as np


env = gym.make("LunarLander-v2", render_mode="human")


model = pickle.load(open("model-actor-critic.pickle", "rb"))

affine_weight = model["affine1.weight"]
affine_bias = model["affine1.bias"]
action_weight = model["action_head.weight"]
action_bias = model["action_head.bias"]


def softmax(x):
    return np.exp(x) / np.exp(x).sum()


def relu(x):
    return np.maximum(0, x)


episodes = 10
total_reward = 0.0
for i in range(episodes):
    observation, _ = env.reset()
    while True:
        affine = relu(np.matmul(observation, affine_weight.T) + affine_bias)
        actions = softmax(np.matmul(affine, action_weight.T) + action_bias)
        action = np.random.choice(np.array(range(4)), 1, p=actions)[0]
        observation, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward

        if terminated or truncated:
            break
env.close()
print("Total reward", total_reward / episodes)
