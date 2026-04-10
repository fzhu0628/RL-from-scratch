import numpy as np
import torch
from envs.env_gridworld import GridWorld
from algos.Vanilla_Q import q_learning, q_learning_LFA, run_policy
from algos.DQN import dqn_interleave_buffer_and_training, dqn_separate_buffer_and_training, run_dqn_policy_gridworld
from algos.Q_LFA import q_learning_LFA



if __name__ == "__main__":
    env = GridWorld()
    # q_table = q_learning_LFA(env)
    # print("Learned Q-table:")
    # print(q_table)
    # path = run_policy(env, q_table)
    path = run_dqn_policy_gridworld(env, dqn_separate_buffer_and_training(env))
    # path = run_dqn_policy(env, dqn_interleave_buffer_and_training(env))
    print("Path taken by the learned policy:")
    print(path)
    if len(path) == env.grid_size * 2 - 1:
        print("Optimal path found!")
    else:
        print("Suboptimal path found.")
