import torch

from gomoku_rl.env import GomokuEnv
from gomoku_rl.utils.policy import uniform_policy

env=GomokuEnv(num_envs=128,board_size=10,device="cuda")

t0,t1=env.rollout(max_steps=100,player_0=uniform_policy,player_1=uniform_policy)
print(t0)
