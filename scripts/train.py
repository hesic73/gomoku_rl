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
from tensordict.nn import TensorDictModule
from typing import Callable, Any
from tensordict import TensorDict


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    run = init_wandb(cfg=cfg)

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

    epochs: int = cfg.get("epochs")
    episode_len: int = cfg.get("episode_len")

    pbar = tqdm(range(epochs))

    for i in pbar:
        data_0, data_1, info = env.rollout(
            episode_len=episode_len, player_black=player_0, player_white=player_1
        )

        info.update(player_0.learn(data_0))
        info.update(player_1.learn(data_1))

        pbar.set_postfix(
            {
                "fps": env._fps,
            }
        )


if __name__ == "__main__":
    main()
