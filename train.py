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
observations = deque(maxlen=memory_size)
actions = deque(maxlen=memory_size)
rewards = deque(maxlen=memory_size)
next_observations = deque(maxlen=memory_size)
dones = deque(maxlen=memory_size)

eps_start = 1.0
eps_end = 0.01
eps_decay = 0.995
eps = eps_start

step = 0
last_rewards = deque(maxlen=100)
best_last_rewards = float("-inf")
for episode in range(episodes):
    episode_reward = 0.0
    observation, _ = env.reset()
    while True:
        observations.append(observation)
        with torch.no_grad():
            if random.random() > eps:
                action = agent(torch.Tensor(observation)).argmax(dim=0).item()
            else:
                action = env.action_space.sample()
        actions.append(action)
        observation, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        next_observations.append(observation)
        dones.append(int(terminated or truncated))

        batch_size = 64
        if len(observations) >= batch_size and step % 4 == 0:
            batch_observations = []
            batch_actions = []
            batch_rewards = []
            batch_next_observations = []
            batch_dones = []
            n = len(observations)
            for _ in range(batch_size):
                rnd = random.randrange(n)
                batch_observations.append(observations[rnd])
                batch_actions.append(actions[rnd])
                batch_rewards.append(rewards[rnd])
                batch_next_observations.append(next_observations[rnd])
                batch_dones.append(dones[rnd])

            with torch.no_grad():
                future_rewards = target(torch.Tensor(numpy.stack(batch_next_observations, axis=0))).detach()
                current_rewards = agent(torch.Tensor(numpy.stack(batch_observations, axis=0))).detach()
            for i in range(batch_size):
                current_rewards[i][batch_actions[i]] = batch_rewards[i] + 0.99 * future_rewards[i].max().item() * (1 - batch_dones[i])

            pred = agent(torch.Tensor(numpy.stack(batch_observations, axis=0)))
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
        if avg >= 200.0:
            print("Solved in {} episodes".format(episode))
            torch.save(agent.state_dict(), "model")
            break
