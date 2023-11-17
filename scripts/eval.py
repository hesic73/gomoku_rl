from gomoku_rl.utils.eval import eval_win_rate
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import datetime
from gomoku_rl import CONFIG_PATH
import torch


from gomoku_rl.env import GomokuEnv

from gomoku_rl.utils.wandb import init_wandb
from gomoku_rl.policy import get_policy
import logging
from tqdm import tqdm
import numpy as np
from typing import Callable, Any, Dict


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="eval")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    env = GomokuEnv(
        num_envs=cfg.num_envs,
        board_size=cfg.board_size,
        device=cfg.device,
    )

    seed = cfg.get("seed", 12345)
    torch.manual_seed(seed)
    np.random.seed(seed)

    player_0 = get_policy(
        name=cfg.algo.name,
        cfg=cfg.algo,
        action_spec=env.action_spec,
        observation_spec=env.observation_spec,
        device=env.device,
    )

    player_1 = get_policy(
        name=cfg.algo.name,
        cfg=cfg.algo,
        action_spec=env.action_spec,
        observation_spec=env.observation_spec,
        device=env.device,
    )

    player_0.load_state_dict(torch.load(cfg.black_checkpoint))
    player_1.load_state_dict(torch.load(cfg.white_checkpoint))

    print(eval_win_rate(env=env, player_black=player_0, player_white=player_1))


if __name__ == "__main__":
    main()
