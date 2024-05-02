import os
import pickle

import gymnasium as gym
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.network = nn.Sequential(nn.Linear(8, 4), nn.Softmax())

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        return self.network(x)


gamma = 0.99
policy = Policy()
if os.path.exists("model-reinforce"):
    policy.load_state_dict(torch.load("model-reinforce"))
optimizer = optim.Adam(policy.parameters())


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
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


env = gym.make("LunarLander-v2")
running_reward = 10
best_reward = -300
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
        print(
            "Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}\tBest reward: {:.2f}".format(
                i_episode, ep_reward, running_reward, best_reward
            )
        )
    if running_reward >= best_reward:
        best_reward = running_reward
        torch.save(policy.state_dict(), "model-reinforce")
        pickle.dump({k: v.numpy() for k, v in policy.state_dict().items()}, open("model-reinforce.pickle", "wb"))
