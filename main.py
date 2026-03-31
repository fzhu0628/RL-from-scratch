import numpy as np
from envs.env_gridworld import GridWorld
from algos.q_learning import q_learning, q_learning_LFA, run_policy
from algos.DQN import dqn_interleave_buffer_and_training, run_dqn_policy, dqn_separate_buffer_and_training
from algos.q_learning_LFA import q_learning_LFA

if __name__ == "__main__":
    env = GridWorld()
    # q_table = q_learning_LFA(env)
    # print("Learned Q-table:")
    # print(q_table)
    # path = run_policy(env, q_table)
    path = run_dqn_policy(env, dqn_separate_buffer_and_training(env))
    # path = run_dqn_policy(env, dqn_interleave_buffer_and_training(env))
    print("Path taken by the learned policy:")
    print(path)
    if len(path) == env.grid_size * 2 - 1:
        print("Optimal path found!")
    else:
        print("Suboptimal path found.")
