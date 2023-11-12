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




@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    run = init_wandb(cfg=cfg)
    device = cfg.get("device", "cuda")
    board_size = cfg.board_size

    env = GomokuEnv(num_envs=cfg.num_envs, board_size=board_size, device=device)

    player_0 = make_nfsp_agent(
        cfg=cfg.algo.actor,
        action_spec=env.action_spec,
        device=device,
    )

    player_1 = make_nfsp_agent(
        cfg=cfg.algo.actor,
        action_spec=env.action_spec,
        device=device,
    )

    env.rollout(max_steps=1000, player_0=player_0, player_1=player_1)
    print(len(player_0.reservoir_buffer))


if __name__ == "__main__":
    main()
