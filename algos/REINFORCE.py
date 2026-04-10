import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1) # output action probabilities
        )

    def forward(self, x):
        return self.net(x) # output dimension: batch_size * action_dim, each row is a probability distribution over actions

def reinforce(env, episodes=1000, gamma=0.99, lr=1e-2):
    # Policy network: a simple MLP with one hidden layer
    policy_net = PolicyNet(env.state_dim, env.action_dim).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    episode_rewards = []
    for ep in range(episodes):
        state = env.reset()
        done = False
        rewards = []
        log_probs = []

        # Generate a trajectory
        while not done:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device) # add batch dimension
            action = torch.multinomial(policy_net(state), num_samples=1).item() # sample action from policy and convert it to an integer
            log_prob = torch.log(policy_net(state)[0, action]) # compute log probability of selected action
            log_probs.append(log_prob)
            next_state, reward, terminated, truncated = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            state = next_state
        episode_rewards.append(sum(rewards))

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G) # insert at the beginning to maintain correct order

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)   # Optional: normalize returns
        # returns = returns - returns.mean()

        # Update policy
        policy_loss = 0
        for k in range(len(rewards)):
            policy_loss += -returns[k] * log_probs[k] # REINFORCE loss with discountingq
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step() # DO NOT UPDATE MULTIPLE STEPS IN A ROW WITHOUT INTERACTION WITH THE ENVIRONMENT, OTHERWISE THE POLICY GRADIENT UPDATE WILL NOT BE CORRECT (BIASED)!

        if ep % 10 == 0:
            print(f"Episode {ep}, Reward: {episode_rewards[-1]}")
    
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("REINFORCE on LunarLander")
    plt.show()

    return policy_net


def run_reinforce_lunarlander(env, model, render=False):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        if render:
            env.render()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action = torch.argmax(model(state_tensor), dim=-1).item() # greedy action selection
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state

    print(f"Total reward: {total_reward}")
    
     
                


            

