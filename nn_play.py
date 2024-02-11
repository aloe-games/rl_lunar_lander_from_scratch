import gymnasium as gym
import torch
from torch import nn

env = gym.make("LunarLander-v2", render_mode="human")


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        return self.network(x)


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
