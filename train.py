import random
import numpy
import torch
from torch import nn
from collections import deque
import gymnasium as gym
from q_network import QNetwork

observation_space = 8
action_space = 4

eps_start = 1.0
eps_end = 0.01
eps_decay = 0.995
eps = eps_start

batch_size = 64
gamma = 0.99
lr = 5e-4
tau = 0.001

agent = QNetwork(observation_space, action_space)
target = QNetwork(observation_space, action_space)
target.load_state_dict(agent.state_dict())
loss_fn = nn.functional.mse_loss
optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

env = gym.make("LunarLander-v2")

episodes = 10000

memory_size = 10000
observations = torch.zeros(memory_size, observation_space, dtype=torch.float)
actions = torch.zeros(memory_size, 1, dtype=torch.long)
rewards = torch.zeros(memory_size, 1, dtype=torch.float)
next_observations = torch.zeros(memory_size, observation_space, dtype=torch.float)
dones = torch.zeros(memory_size, 1, dtype=torch.uint8)

step = 0
last_rewards = deque(maxlen=100)
best_last_rewards = float("-inf")
for episode in range(episodes):
    episode_reward = 0.0
    observation, _ = env.reset()
    while True:
        memory_index = step % memory_size
        observations[memory_index] = torch.tensor(observation, dtype=torch.float)
        with torch.no_grad():
            if random.random() > eps:
                action = agent(torch.Tensor(observation)).argmax(dim=0).item()
            else:
                action = env.action_space.sample()
        actions[memory_index] = torch.tensor(action, dtype=torch.long)
        observation, reward, terminated, truncated, _ = env.step(action)
        rewards[memory_index] = torch.tensor(reward, dtype=torch.float)
        next_observations[memory_index] = torch.tensor(observation, dtype=torch.float)
        dones[memory_index] = torch.tensor(terminated or truncated, dtype=torch.uint8)

        batch_size = 64
        if step >= batch_size and step % 4 == 0:
            rnd = torch.randint(0, min(step, memory_size), (batch_size,))
            future_rewards = target(next_observations.index_select(0, rnd)).detach().max(1)[0].unsqueeze(1)
            current_rewards = rewards.index_select(0, rnd) + gamma * future_rewards * (1 - dones.index_select(0, rnd))
            pred = agent(observations.index_select(0, rnd)).gather(1, actions.index_select(0, rnd))
            loss = loss_fn(pred, current_rewards)
            loss.backward()
            torch.nn.utils.clip_grad_value_(agent.parameters(), 100)
            optimizer.step()
            optimizer.zero_grad()

            target_state_dict = target.state_dict()
            agent_state_dict = agent.state_dict()
            for key in agent_state_dict:
                target_state_dict[key] = agent_state_dict[key] * tau + target_state_dict[key] * (1 - tau)
            target.load_state_dict(target_state_dict)

        episode_reward += reward

        step += 1
        if terminated or truncated:
            last_rewards.append(episode_reward)
            eps = max(eps_end, eps_decay * eps)
            break
    if episode >= 100:
        avg = numpy.mean(last_rewards)
        if episode % 10 == 0:
            print(episode, avg)
        if avg >= 230.0:
            print("Solved in {} episodes".format(episode))
            torch.save(agent.state_dict(), "model")
            break
