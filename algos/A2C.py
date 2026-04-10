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

def a2c(env, episodes=2000, n_steps=5, gamma=0.99, lr=3e-4):
    model = ActorCritic(env.state_dim, env.action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    episode_rewards = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:

            log_probs = []
            values = []
            rewards = []
            entropies = []

            # 🔹 collect n steps
            for _ in range(n_steps):
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

                logits, value = model(state_tensor)
                dist = Categorical(logits=logits)

                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()

                next_state, reward, terminated, truncated = env.step(action.item())
                done = terminated or truncated

                # reward = np.clip(reward, -1, 1)

                log_probs.append(log_prob)
                values.append(value.squeeze())
                rewards.append(torch.tensor(reward, dtype=torch.float32).to(device))
                entropies.append(entropy)

                total_reward += reward
                state = next_state

                if done:
                    break

            # 🔹 bootstrap next value
            if terminated:
                next_value = torch.tensor(0.0).to(device)
            else:
                with torch.no_grad():
                    next_state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    _, next_value = model(next_state_tensor)
                    next_value = next_value.squeeze()

            # 🔹 stack
            values = torch.stack(values)
            log_probs = torch.stack(log_probs)
            rewards = torch.stack(rewards)
            entropies = torch.stack(entropies)

            # # 🔥 🔑 Compute returns
            # returns = []
            # G = next_value.detach()
            # for r in reversed(rewards):
            #     G = r + gamma * G
            #     returns.insert(0, G)
            # returns = torch.stack(returns)

            # advantages = returns - values              # raw (for critic)
            # advantages_actor = advantages              # copy

            # # normalize ONLY for actor
            # std = advantages_actor.std()
            # if std > 1e-8:
            #     advantages_actor = (advantages_actor - advantages_actor.mean()) / (std + 1e-8)
            # else:
            #     advantages_actor = advantages_actor - advantages_actor.mean()
            advantages = gae(rewards, values, next_value, gamma, lam=0.95)
            advantages_actor = advantages.detach().clone()
            # normalize ONLY for actor
            std = advantages_actor.std()
            if std > 1e-8:
                advantages_actor = (advantages_actor - advantages_actor.mean()) / (std + 1e-8)
            else:
                advantages_actor = advantages_actor - advantages_actor.mean()


            # 🔹 losses
            actor_loss = -(log_probs * advantages_actor).mean()
            # critic_loss = advantages.pow(2).mean()
            returns = advantages + values

            critic_loss = nn.functional.huber_loss(values, returns.detach())
            entropy_loss = entropies.mean()

            loss = actor_loss + 0.5 * critic_loss - 0.03 * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        episode_rewards.append(total_reward)
        if ep % 10 == 0:
            print(f"Episode {ep}, Reward: {total_reward}")
        if ep % 100 == 0:
            total_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
            print(f"Grad norm: {total_norm:.4f}")

    plt.plot(episode_rewards)
    plt.title("A2C on LunarLander")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.ylim(-500, 300)
    plt.show()

    return model


def run_a2c_lunarlander(env, model, render=False):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        if render:
            env.render()

        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = model(state_tensor)

        # 🔑 deterministic action (greedy)
        dist = Categorical(logits=logits)
        action = dist.probs.argmax().item()

        state, reward, terminated, truncated = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Total reward: {total_reward}")

def a2c_multi_env(envs, rounds=20000, n_steps=10, gamma=0.99, lr=1e-4):
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
        entropies_list = []

        # 🔹 collect rollout
        for _ in range(n_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device) # [N, state_dim]

            logits, value = model(state_tensor)      # [N, A], [N,1]
            value = value.squeeze(-1)                # [N]

            dist = Categorical(logits=logits)

            actions = dist.sample()                  # [N]
            log_probs = dist.log_prob(actions)       # [N]
            entropy = dist.entropy()                 # [N]

            next_state, reward, terminated, truncated, _ = envs.step(actions.cpu().numpy())
            done = terminated | truncated

            log_probs_list.append(log_probs)
            values_list.append(value)
            rewards_list.append(torch.tensor(reward, dtype=torch.float32).to(device))
            dones_list.append(torch.tensor(done, dtype=torch.float32).to(device))
            entropies_list.append(entropy)

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
        entropies = torch.stack(entropies_list)

        # 🔹 GAE (must support [T, N])
        advantages = gae(rewards, values, next_value, gamma, lam=0.95)
        returns = advantages + values

        # 🔹 flatten
        T, N = rewards.shape
        advantages = advantages.reshape(T * N)
        returns = returns.reshape(T * N)
        values = values.reshape(T * N)
        log_probs = log_probs.reshape(T * N)
        entropies = entropies.reshape(T * N)

        # 🔹 normalize advantages (actor only)
        advantages_actor = advantages.detach()
        advantages_actor = (advantages_actor - advantages_actor.mean()) / (advantages_actor.std() + 1e-8)

        # 🔹 losses
        actor_loss = -(log_probs * advantages_actor).mean()
        critic_loss = nn.functional.huber_loss(values, returns.detach())
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
    plt.title("A2C Multi-Env")
    plt.xlabel("Rounds")
    plt.ylabel("Reward")
    plt.ylim(-500, 300)
    plt.grid()
    plt.show()

    return model
    
     
                


            

