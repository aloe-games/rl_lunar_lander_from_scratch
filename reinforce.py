import os

import gymnasium as gym
import numpy as np
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 4), nn.Softmax()
        )

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        return self.network(x)


gamma = 0.99
policy = Policy()
if os.path.exists("model-reinforce"):
    policy.load_state_dict(torch.load("model-reinforce"))
optimizer = optim.Adam(policy.parameters(), lr=5e-4)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    returns = deque()
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    torch.nn.utils.clip_grad_value_(policy.parameters(), 100)
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


env = gym.make("LunarLander-v2")
running_reward = 10
best_reward = -np.inf
for i_episode in count(1):
    state, _ = env.reset()
    ep_reward = 0
    while True:
        action = select_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        policy.rewards.append(reward)
        ep_reward += reward
        if terminated or truncated:
            break

    running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
    finish_episode()
    if i_episode % 10 == 0:
        print("Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}".format(i_episode, ep_reward, running_reward))
    if running_reward >= best_reward:
        best_reward = running_reward
        print("New best model! Running reward is now {}".format(running_reward))
        torch.save(policy.state_dict(), "model-reinforce")
