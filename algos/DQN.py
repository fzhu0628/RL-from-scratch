import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
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
    
def dqn_interleave_buffer_and_training(env, episodes=100, batch_size=32, gamma=0.9, lr=1e-3):

    q_net = QNet()
    target_net = QNet()
    target_net.load_state_dict(q_net.state_dict()) 

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer()

    epsilon = 1.0

    for ep in range(episodes):
        state = env.reset()
        done = False

        while not done:
            # normalize state
            s = np.array(state) / (env.grid_size - 1)

            # ε-greedy
            if np.random.rand() < epsilon:
                action = np.random.choice(4)
            else:
                with torch.no_grad():
                    q_values = q_net(torch.tensor(s, dtype=torch.float32))
                    action = torch.argmax(q_values).item() # .item() to get the scalar value from tensor

            next_state, reward, done = env.step(action)

            s_next = np.array(next_state) / (env.grid_size - 1)

            buffer.push(s, action, reward, s_next, done)

            state = next_state

            # training step
            if len(buffer) > batch_size:
                # preprocess the batch
                s_b, a_b, r_b, s_next_b, done_b = buffer.sample(batch_size)

                s_b = torch.tensor(s_b, dtype=torch.float32)
                a_b = torch.tensor(a_b, dtype=torch.long)
                r_b = torch.tensor(r_b, dtype=torch.float32)
                s_next_b = torch.tensor(s_next_b, dtype=torch.float32)
                done_b = torch.tensor(done_b, dtype=torch.float32)

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
        if ep % 2 == 0:
            target_net.load_state_dict(q_net.state_dict())

        # decay epsilon
        epsilon = max(0.05, epsilon * 0.995)

    return q_net

def dqn_separate_buffer_and_training(env, episodes=100, batch_size=32, gamma=0.9, lr=1e-3, training_steps=100):

    q_net = QNet()
    target_net = QNet()
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer()

    epsilon = 1.0

    for ep in range(episodes):
        counter = 0
        while counter < 100:
            state = env.reset()
            done = False

            while not done and counter < 100:
                s = np.array(state) / (env.grid_size - 1)

                # ε-greedy
                if np.random.rand() < epsilon:
                    action = np.random.choice(4)
                else:
                    with torch.no_grad():
                        q_values = q_net(torch.tensor(s, dtype=torch.float32))
                        action = torch.argmax(q_values).item() # .item() to get the scalar value from tensor

                next_state, reward, done = env.step(action)

                s_next = np.array(next_state) / (env.grid_size - 1)

                buffer.push(s, action, reward, s_next, done)
                counter += 1

                state = next_state
        
        # training step
        for _ in range(training_steps):
            # preprocess the batch
            s_b, a_b, r_b, s_next_b, done_b = buffer.sample(batch_size)

            s_b = torch.tensor(s_b, dtype=torch.float32)
            a_b = torch.tensor(a_b, dtype=torch.long)
            r_b = torch.tensor(r_b, dtype=torch.float32)
            s_next_b = torch.tensor(s_next_b, dtype=torch.float32)
            done_b = torch.tensor(done_b, dtype=torch.float32)

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
        target_net.load_state_dict(q_net.state_dict())

        # decay epsilon
        epsilon = max(0.05, epsilon * 0.995)

    return q_net


def run_dqn_policy(env, model):
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