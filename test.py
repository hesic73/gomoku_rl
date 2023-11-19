import hydra
from omegaconf import DictConfig, OmegaConf
import os
import datetime
from gomoku_rl import CONFIG_PATH
import torch


from gomoku_rl.env import GomokuEnv
from gomoku_rl.utils.eval import get_payoff_matrix

from gomoku_rl.policy import get_policy
import logging
from tqdm import tqdm
import numpy as np
from typing import Callable, Any, Dict


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train_InRL")
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

    def from_checkpoint(path: str):
        player = get_policy(
            name=cfg.algo.name,
            cfg=cfg.algo,
            action_spec=env.action_spec,
            observation_spec=env.observation_spec,
            device=env.device,
        )
        player.load_state_dict(torch.load(path))
        return player

    checkpoints = [
        "pretrained_models/10_10/baseline.pt",
        "wandb/latest-run/files/player_black.pt",
        "wandb/latest-run/files/player_white.pt",
    ]
    polices = [from_checkpoint(p) for p in checkpoints]
    payoff = get_payoff_matrix(env, polices)
    print(payoff)


if __name__ == "__main__":
    main()
