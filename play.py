import gymnasium as gym
import torch
import numpy as np

env = gym.make("LunarLander-v2", render_mode="human")


agent = torch.load("model")
weight = agent.state_dict()['network.weight'].numpy()
bias = agent.state_dict()['network.bias'].numpy()

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
