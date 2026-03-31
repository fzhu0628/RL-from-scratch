import numpy as np

def q_learning_LFA(env, alpha=0.05, gamma=0.9, epsilon=0.1, episodes=40000):
    weights = np.zeros(15) # Weights for linear function approximation
    # def features_mapping(state, action):
    #     x, y = state
    #     x = x / (env.grid_size - 1)
    #     y = y / (env.grid_size - 1)
    #     actions = np.zeros(4)
    #     actions[action] = 1
    #     return np.array([x, y, *actions]) # Simple feature vector (can be more complex)
    def features_mapping(state, action):
        x, y = state

        # normalize
        x = x / (env.grid_size - 1)
        y = y / (env.grid_size - 1)

        # action one-hot
        a_vec = np.zeros(4)
        a_vec[action] = 1

        # interaction: state × action
        interaction = np.outer([x, y], a_vec).flatten()

        # bias term (VERY important)
        bias = np.array([1.0])

        return np.concatenate([
            [x, y],       # state
            a_vec,        # action
            interaction,  # interaction
            bias          # bias
        ])
    features_all = np.zeros((env.grid_size, env.grid_size, 4, 15)) # Precompute features for all state-action pairs
    for x in range(env.grid_size):
        for y in range(env.grid_size):
            for a in range(4):
                features_all[x, y, a] = features_mapping((x, y), a)
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            x, y = state
            # get all features for current state (shape: 4 x d)
            phis = features_all[x, y]
            # compute Q-values (vectorized)
            Q_values = phis @ weights
            # ε-greedy
            if np.random.rand() < epsilon:
                action = np.random.choice(4)
            else:
                action = np.argmax(Q_values)
            next_state, reward, done = env.step(action)
            # current Q
            phi_sa = phis[action]
            Q_sa = Q_values[action]
            # next state Q
            if done:
                Q_next = 0
            else:
                nx, ny = next_state
                next_phis = features_all[nx, ny]
                Q_next = np.max(next_phis @ weights)
            # TD update
            delta = reward + gamma * Q_next - Q_sa
            weights += alpha * delta * phi_sa
            state = next_state
    q_tab = np.zeros((env.grid_size, env.grid_size, 4))  # Q-values for each state-action pair
    for x in range(env.grid_size):
        for y in range(env.grid_size):
            for a in range(4):
                q_tab[x, y, a] = weights @ features_mapping((x, y), a) # Compute Q-values from weights
    return q_tab