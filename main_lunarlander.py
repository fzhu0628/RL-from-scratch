import gymnasium as gym
from algos.REINFORCE import reinforce, run_reinforce_lunarlander
from envs.env_lunarlander import LunarLander

env = LunarLander(render=False)
model = reinforce(env)

run_reinforce_lunarlander(env, model)
