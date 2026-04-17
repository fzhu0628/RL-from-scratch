import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from torch.distributions import Categorical
import gymnasium as gym
from algos.GAE import gae

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(f"Using device: {device}")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        # Small dedicated heads
        self.actor_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.critic_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU())

        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        shared = self.shared(x)
        logits = self.actor(self.actor_head(shared))
        value = self.critic(self.critic_head(shared))
        return logits, value
    

def ppo_multi_env(envs, rounds=20000, n_steps=10, gamma=0.99, lr=1e-4, epsilon=0.2, m_train=4):
    # ✅ get dims correctly
    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.n

    model = ActorCritic(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    N = envs.num_envs
    episode_rewards = np.zeros(N)

    state, _ = envs.reset()

    reward_history = []
    

    for ep in range(rounds):

        log_probs_list = []
        values_list = []
        rewards_list = []
        dones_list = []
        state_list = []
        actions_list = []

        # 🔹 collect rollout
        for _ in range(n_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device) # [N, state_dim]

            with torch.no_grad():
                logits, value = model(state_tensor)      # [N, A], [N,1]
                value = value.squeeze(-1)                # [N]
                dist = Categorical(logits=logits)
                action = dist.sample()                  # [N]
                log_probs = dist.log_prob(action)       # [N]

            next_state, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
            done = terminated | truncated

            log_probs_list.append(log_probs)
            values_list.append(value)
            rewards_list.append(torch.tensor(reward, dtype=torch.float32).to(device))
            dones_list.append(torch.tensor(done, dtype=torch.float32).to(device))
            state_list.append(state_tensor)
            actions_list.append(action)

            episode_rewards += reward

            # reset finished envs
            for i, d in enumerate(done):
                if d:
                    reward_history.append(episode_rewards[i])
                    episode_rewards[i] = 0

            state = next_state

        # 🔹 bootstrap
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            _, next_value = model(state_tensor) # [N,1]
            next_value = next_value.squeeze(-1)      # [N]
        next_value = next_value * (1 - np.float32(done))

        # 🔹 stack → [T, N]
        values = torch.stack(values_list)
        log_probs = torch.stack(log_probs_list)
        rewards = torch.stack(rewards_list)
        dones = torch.stack(dones_list)
        states = torch.stack(state_list)
        actions = torch.stack(actions_list)

        # 🔹 GAE (must support [T, N])
        advantages = gae(rewards, values, next_value, gamma, lam=0.95)
        returns = advantages + values

        # 🔹 flatten
        T, N = rewards.shape
        advantages = advantages.reshape(T * N)
        returns = returns.reshape(T * N)
        log_probs_old = log_probs.reshape(T * N)
        states = states.reshape(T * N, state_dim)
        actions = actions.reshape(T * N)

        # 🔹 normalize advantages (actor only)
        advantages_actor = advantages.detach()
        advantages_actor = (advantages_actor - advantages_actor.mean()) / (advantages_actor.std() + 1e-8)

        # 🔹 losses
        for _ in range(m_train): # PPO typically uses multiple epochs over the same data
            logits, values_new = model(states) # [T*N, A], [T*N,1] Can further use minibatches here if desired
            dist = Categorical(logits=logits)
            log_probs_new = dist.log_prob(actions) # [T*N]
            entropies = dist.entropy() # [T*N]

            ratio = torch.exp(log_probs_new - log_probs_old)
            clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
            actor_loss = -torch.min(ratio * advantages_actor, clipped_ratio * advantages_actor).mean()

            critic_loss = nn.functional.huber_loss(values_new.squeeze(-1), returns.detach())
            entropy_loss = entropies.mean()

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        if ep % 10 == 0 and len(reward_history) > 0:
            print(f"Round {ep}, Avg Reward: {np.mean(reward_history[-50:]):.2f}")

    # 🔹 plot
    plt.plot(reward_history)
    plt.title("PPO Multi-Env")
    plt.xlabel("Rounds")
    plt.ylabel("Reward")
    plt.ylim(-500, 300)
    plt.grid()
    plt.show()

    return model, reward_history
    
     
                


            

