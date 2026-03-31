import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def q_learning(env, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000):
    q_tab = np.zeros((env.grid_size, env.grid_size, 4))  # Q-values for each state-action pair
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            x, y = state
            action = np.random.choice(4) if np.random.rand() < epsilon else np.argmax(q_tab[x, y]) # Epsilon-greedy action selection
            next_state, reward, done = env.step(action)
            q_tab[x, y, action] += alpha * (reward + gamma * np.max(q_tab[next_state[0], next_state[1]]) - q_tab[x, y, action]) # Q-learning update
            state = next_state
    return q_tab


def run_policy(env, q_table):
    state = env.reset()
    done = False
    path = [state]
    while not done:
        x, y = state
        action = np.argmax(q_table[x, y])   
        state, _, done = env.step(action)
        path.append(state)
        if len(path) > env.grid_size * 2:  # Prevent infinite loops
            break

    return path

