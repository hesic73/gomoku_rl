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


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train_nfsp")
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
        observation_spec=env.observation_spec,
        device=device,
    )

    player_1 = make_nfsp_agent(
        cfg=cfg.algo.actor,
        action_spec=env.action_spec,
        observation_spec=env.observation_spec,
        device=device,
    )

    for i in tqdm(range(100)):
        player_0._sample_episode_policy()
        player_1._sample_episode_policy()
        data0, data1 = env.rollout(max_steps=100, player_0=player_0, player_1=player_1)
        loss_rl_0 = player_0.train_rl(data0)
        loss_rl_1 = player_1.train_rl(data1)        
        
        loss_sl_0 = player_0.train_sl()

        loss_sl_1 = player_1.train_sl()



if __name__ == "__main__":
    main()
