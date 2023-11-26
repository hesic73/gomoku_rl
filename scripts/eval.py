from gomoku_rl.utils.eval import get_payoff_matrix
from gomoku_rl.utils.visual import annotate_heatmap, heatmap
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import datetime
from gomoku_rl import CONFIG_PATH
import torch

import matplotlib.pyplot as plt

from gomoku_rl.env import GomokuEnv

from gomoku_rl.utils.wandb import init_wandb
from gomoku_rl.utils.psro import calculate_jpc
from gomoku_rl.policy import get_pretrained_policy
import logging
from tqdm import tqdm
import numpy as np
from typing import Callable, Any, Dict
import functools


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

    make_player = functools.partial(
        get_pretrained_policy,
        name=cfg.algo.name,
        cfg=cfg.algo,
        action_spec=env.action_spec,
        observation_spec=env.observation_spec,
        device=env.device,
    )

    checkpoints = cfg.checkpoints
    players = [make_player(checkpoint_path=p) for p in checkpoints]

    payoff = get_payoff_matrix(env=env, row_policies=players, col_policies=players, n=5)
    print(payoff)
    print(f"JPC:{calculate_jpc(payoff.numpy())}")
    im, _ = heatmap(
        payoff * 100,
        row_labels=[os.path.split(p)[1] for p in checkpoints],
        col_labels=[os.path.split(p)[1] for p in checkpoints],
    )
    annotate_heatmap(im, valfmt="{x:.2f}%")
    plt.tight_layout()
    plt.savefig("payoff.png")


if __name__ == "__main__":
    main()
