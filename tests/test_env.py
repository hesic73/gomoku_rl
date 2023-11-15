from gomoku_rl.utils.policy import uniform_policy
from gomoku_rl.env import GomokuEnv
from tqdm import tqdm
import torch

env = GomokuEnv(num_envs=2048, board_size=8, device="cuda:0")

bar = tqdm(range(100))
for i in bar:
    transitions_black, transitions_white = env.rollout(
        max_steps=40, player_black=uniform_policy, player_white=uniform_policy
    )
    bar.set_postfix(
        {
            "rollout_fps": env._fps,
        }
    )
