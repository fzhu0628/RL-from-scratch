import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt

# GPU device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)
    


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = zip(*batch)

        return (
            np.array(s),
            np.array(a),
            np.array(r),
            np.array(s_next),
            np.array(done)
        )

    def __len__(self):
        return len(self.buffer)

def dqn_separate_buffer_and_training(env, episodes=1000, batch_size=32, gamma=0.9, lr=1e-4, training_steps=10):

    q_net = QNet(env.state_dim, env.action_dim).to(device)
    target_net = QNet(env.state_dim, env.action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer()

    epsilon = 1.0
    window = 50
    episode_rewards = []

    for ep in range(episodes):
        total_reward = 0
        state = env.reset()
        done = False

        while not done:
            # s = np.array(state) / (env.grid_size - 1) # gridworld specific
            s = np.array(state, dtype=np.float32)

            # ε-greedy
            if np.random.rand() < epsilon:
                action = np.random.choice(env.action_dim)
            else:
                with torch.no_grad():
                    q_values = q_net(torch.tensor(s, dtype=torch.float32).to(device))
                    action = torch.argmax(q_values).item() # .item() to get the scalar value from tensor

            next_state, reward, done = env.step(action)
            total_reward += reward

            # s_next = np.array(next_state) / (env.grid_size - 1) # gridworld specific
            s_next = np.array(next_state, dtype=np.float32)

            buffer.push(s, action, reward, s_next, done)

            state = next_state

        episode_rewards.append(total_reward)

        
        if len(buffer) >= 1000:
            # training step
            for _ in range(training_steps):
                # preprocess the batch
                s_b, a_b, r_b, s_next_b, done_b = buffer.sample(batch_size)

                s_b = torch.tensor(s_b, dtype=torch.float32).to(device)
                a_b = torch.tensor(a_b, dtype=torch.long).to(device)
                r_b = torch.tensor(r_b, dtype=torch.float32).to(device)
                s_next_b = torch.tensor(s_next_b, dtype=torch.float32).to(device)
                done_b = torch.tensor(done_b, dtype=torch.float32).to(device)

                # Q(s,a)
                q_values = q_net(s_b) # forward pass (with grad), dimension: batch_size * 4
                q_sa = q_values.gather(1, a_b.unsqueeze(1)).squeeze() # .gather(dim, index) gathers along dim using index. outputs dimension batch_size

                # target (no grad!)
                with torch.no_grad():
                    q_next = target_net(s_next_b).max(1)[0] # batch_size * 4 -> max along dim 1 -> values, indices -> values
                    target = r_b + gamma * q_next * (1 - done_b) # element wise product, since they're of the same dim.

                loss = nn.MSELoss()(q_sa, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if ep % 20 == 0:
                target_net.load_state_dict(q_net.state_dict())

            # decay epsilon
            epsilon = max(0.05, epsilon * 0.995)
        
        # stopping check
        # recent = episode_rewards[-window:]

        # if len(recent) == window and np.std(recent) < 5:
        #     print("Converged (plateau)")
        #     break
        
        # if total_reward > 450:
        #     print("Solved!")
        #     break

        
    
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN on CartPole")
    plt.show()

    return q_net


def dqn_interleave_buffer_and_training(
    env,
    episodes=1000,
    batch_size=32,
    gamma=0.99,
    lr=1e-3,
    replay_start_size=500,
    target_update_steps=1000,
    epsilon_decay=0.99,
    epsilon_min=0.01,
):

    q_net = QNet(env.state_dim, env.action_dim).to(device)
    target_net = QNet(env.state_dim, env.action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict()) 

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer()

    epsilon = 1.0
    window = 50 # stopping criterion
    episode_rewards = []

    global_step = 0

    for ep in range(episodes):
        total_reward = 0
        state = env.reset()
        done = False

        while not done:
            # normalize state
            # s = np.array(state) / (env.grid_size - 1)
            s = np.array(state, dtype=np.float32)

            # ε-greedy
            if np.random.rand() < epsilon:
                action = np.random.choice(env.action_dim)
            else:
                with torch.no_grad():
                    q_values = q_net(torch.tensor(s, dtype=torch.float32).to(device))
                    action = torch.argmax(q_values).item() # .item() to get the scalar value from tensor

            next_state, reward, done = env.step(action)
            total_reward += reward

            # s_next = np.array(next_state) / (env.grid_size - 1)
            s_next = np.array(next_state, dtype=np.float32)
            

            buffer.push(s, action, reward, s_next, done)
            global_step += 1

            state = next_state

            # training step
            if len(buffer) >= replay_start_size:
                # preprocess the batch
                s_b, a_b, r_b, s_next_b, done_b = buffer.sample(batch_size)

                s_b = torch.tensor(s_b, dtype=torch.float32).to(device)
                a_b = torch.tensor(a_b, dtype=torch.long).to(device)
                r_b = torch.tensor(r_b, dtype=torch.float32).to(device)
                s_next_b = torch.tensor(s_next_b, dtype=torch.float32).to(device)
                done_b = torch.tensor(done_b, dtype=torch.float32).to(device)

                # Q(s,a)
                q_values = q_net(s_b) # forward pass (with grad), dimension: batch_size * 4
                q_sa = q_values.gather(1, a_b.unsqueeze(1)).squeeze() # .gather(dim, index) gathers along dim using index. outputs dimension batch_size

                # target (no grad!)
                with torch.no_grad():
                    q_next = target_net(s_next_b).max(1)[0] # batch_size * 4 -> max along dim 1 -> values, indices -> values
                    target = r_b + gamma * q_next * (1 - done_b) # element wise product, since they're of the same dim.

                loss = nn.MSELoss()(q_sa, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network on a fixed environment-step schedule
            if global_step % target_update_steps == 0:
                target_net.load_state_dict(q_net.state_dict())

        # decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        episode_rewards.append(total_reward)

        # stopping check
        recent = episode_rewards[-window:]

        if len(recent) == window and np.std(recent) < 1:
            print("Converged (plateau)")
            break

    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN on CartPole")
    plt.show()

    return q_net


def run_dqn_policy_gridworld(env, model):
    state = env.reset()
    path = [state]
    visited = set([state])

    while True:
        s = torch.tensor(np.array(state)/(env.grid_size-1), dtype=torch.float32)

        with torch.no_grad():
            action = torch.argmax(model(s)).item()

        next_state, _, done = env.step(action)
        path.append(next_state)

        if done or next_state in visited:
            break

        visited.add(next_state)
        state = next_state

    return path

def run_dqn_policy_cartpole(env, model, render=False):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        s = torch.tensor(state, dtype=torch.float32).to(device)

        with torch.no_grad():
            action = torch.argmax(model(s)).item()

        next_state, reward, done = env.step(action)

        if render:
            env.render()

        total_reward += reward
        state = next_state

    return total_reward
