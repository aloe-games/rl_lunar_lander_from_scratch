import random
import torch
from torch import nn
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = QNetwork(observation_space, action_space).to(device)
target = QNetwork(observation_space, action_space).to(device)
target.load_state_dict(agent.state_dict())
loss_fn = nn.functional.mse_loss
optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

env = gym.make("LunarLander-v2")

episodes = 10000

memory_size = 10000
observations = torch.zeros(memory_size, observation_space, dtype=torch.float, device=device)
actions = torch.zeros(memory_size, 1, dtype=torch.long, device=device)
rewards = torch.zeros(memory_size, 1, dtype=torch.float, device=device)
next_observations = torch.zeros(memory_size, observation_space, dtype=torch.float, device=device)
dones = torch.zeros(memory_size, 1, dtype=torch.uint8, device=device)


def evaluate(agent, episodes):
    total_reward = 0
    for i in range(episodes):
        observation, _ = env.reset()
        while True:
            with torch.no_grad():
                actions = agent(torch.tensor(observation, device=device, dtype=torch.float))
            action = actions.argmax(dim=0).item()
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
    return total_reward / episodes


step = 0
best_model = float("-inf")
for episode in range(episodes):
    episode_reward = 0.0
    observation, _ = env.reset()
    while True:
        memory_index = step % memory_size
        observations[memory_index] = torch.tensor(observation, dtype=torch.float)
        with torch.no_grad():
            if random.random() > eps:
                action = agent(observations[memory_index]).argmax(dim=0).item()
            else:
                action = env.action_space.sample()
        actions[memory_index] = torch.tensor(action, dtype=torch.long)
        observation, reward, terminated, truncated, _ = env.step(action)
        rewards[memory_index] = torch.tensor(reward, dtype=torch.float)
        next_observations[memory_index] = torch.tensor(observation, dtype=torch.float)
        dones[memory_index] = torch.tensor(terminated or truncated, dtype=torch.uint8)

        batch_size = 64
        if step >= batch_size and step % 4 == 0:
            rnd = torch.randint(0, min(step, memory_size), (batch_size,)).to(device)
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
            eps = max(eps_end, eps_decay * eps)
            break
    if episode_reward > best_model:
        print(episode, episode_reward, best_model)
        if evaluate(agent, 3) > best_model and evaluate(agent, 7) > best_model and evaluate(agent, 20) > best_model:
            current_model = evaluate(agent, 70)
            if current_model > best_model:
                best_model = current_model
                torch.save(agent.state_dict(), "model", _use_new_zipfile_serialization=False)
