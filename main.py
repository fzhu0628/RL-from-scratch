import numpy as np
from envs.env_gridworld import GridWorld, run_policy
from algos.q_learning import q_learning

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
