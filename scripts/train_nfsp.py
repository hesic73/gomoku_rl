import hydra
from omegaconf import DictConfig, OmegaConf
import os
import datetime
from gomoku_rl import CONFIG_PATH
import torch
from torchrl.data import TensorSpec

from gomoku_rl.env import GomokuEnv

from gomoku_rl.utils.wandb import init_wandb

from gomoku_rl.algo.nfsp import make_nfsp_agent

from tqdm import tqdm
import numpy as np


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train_nfsp")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    run = init_wandb(cfg=cfg)
    device = cfg.get("device", "cuda")
    board_size = cfg.board_size

    seed = cfg.get("seed", 12345)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = GomokuEnv(num_envs=cfg.num_envs, board_size=board_size, device=device)

    player_black = make_nfsp_agent(
        cfg=cfg.algo.actor,
        action_spec=env.action_spec,
        observation_spec=env.observation_spec,
        device=device,
    )

    player_white = make_nfsp_agent(
        cfg=cfg.algo.actor,
        action_spec=env.action_spec,
        observation_spec=env.observation_spec,
        device=device,
    )

    bar = tqdm(range(100))
    for i in bar:
        player_black._sample_episode_policy()
        player_white._sample_episode_policy()
        print(player_black.mode, player_white.mode)
        data0, data1 = env.rollout(
            max_steps=100, player_black=player_black, player_white=player_white
        )
        loss_rl_0 = player_black.train_rl(data0)
        loss_rl_1 = player_white.train_rl(data1)

        loss_sl_0 = player_black.train_sl()

        # loss_sl_1 = player_white.train_sl()

        # dict_to_log = {
        #     "loss_rl": (loss_rl_0 + loss_rl_1) / 2,
        # }
        # if loss_sl_0 is not None and loss_sl_1 is not None:
        #     dict_to_log.update(
        #         {
        #             "loss_sl": (loss_sl_0 + loss_sl_1) / 2,
        #         }
        #     )
        # run.log(dict_to_log)

        bar.set_postfix(
            {
                "rollout_fps": env._fps,
            }
        )


if __name__ == "__main__":
    main()
