import numpy as np
class GridWorld:
    def __init__(self, grid_size=100, goal_state=(99, 99)):
        self.grid_size = grid_size
        self.goal_state = goal_state
        self.state = (0, 0)  # Start at the top-left corner

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:  # Up
            x = max(x - 1, 0)
        elif action == 1:  # Down
            x = min(x + 1, self.grid_size - 1)
        elif action == 2:  # Left
            y = max(y - 1, 0)
        elif action == 3:  # Right
            y = min(y + 1, self.grid_size - 1)

        self.state = (x, y)
        reward = -1 if self.state != self.goal_state else 0
        done = self.state == self.goal_state
        return self.state, reward, done
    
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

    return path




'''
if __name__ == "__main__":
    env = GridWorld()
    q_table = q_learning(env)
    print("Learned Q-table:")
    print(q_table)
    path = run_policy(env, q_table)
    print("Path taken by the learned policy:")
    print(path)
    if len(path) == env.grid_size * 2 - 1:
        print("Optimal path found!")
    else:
        print("Suboptimal path found.")
'''