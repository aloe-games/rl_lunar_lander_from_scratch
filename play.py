import gymnasium as gym
import numpy as np

env = gym.make("LunarLander-v2", render_mode="human")

weights = np.array([[-0.4, -0.3, 1.1, 10.0, 1.1, 0.0, 2.4, 2.7], [5.0, 1.1, 3.4, 3.3, -7.5, -3.3, -2.2, -3.3],
                    [0.4, -2.0, -0.4, -7.1, -0.8, -1.9, 0.8, -0.3], [-5.1, 1.7, -4.2, 1.8, 8.2, 4.9, -4.5, -1.3], ])
bias = np.array([-0.7, 0.9, -0.3, 0.1])

episodes = 10
total_reward = 0.0
for i in range(episodes):
    observation, _ = env.reset()
    while True:
        action = np.argmax(np.matmul(observation, weights.T) + bias)
        observation, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward

        if terminated or truncated:
            break
env.close()
print("Total reward", total_reward / episodes)
