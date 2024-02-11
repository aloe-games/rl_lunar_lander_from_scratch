import gymnasium as gym
import torch

from q_network import QNetwork

env = gym.make("LunarLander-v2", render_mode="human")


agent = QNetwork()
agent.load_state_dict(torch.load("model_new"))

episodes = 10
total_reward = 0.0
for i in range(episodes):
    observation, _ = env.reset()
    while True:
        actions = agent(torch.Tensor(observation))
        action = actions.argmax(dim=0).item()
        observation, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward

        if terminated or truncated:
            break
env.close()
print("Total reward", total_reward / episodes)
