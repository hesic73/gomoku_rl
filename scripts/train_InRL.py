import hydra
from omegaconf import DictConfig, OmegaConf
import os
import datetime
from gomoku_rl import CONFIG_PATH
import torch


from gomoku_rl.env import GomokuEnv
from gomoku_rl.utils.eval import eval_win_rate
from gomoku_rl.utils.wandb import init_wandb
from gomoku_rl.utils.policy import uniform_policy
from gomoku_rl.policy import get_policy
import logging
from tqdm import tqdm
import numpy as np
from typing import Callable, Any, Dict


def add_prefix(d: Dict, prefix: str):
    return {prefix + k: v for k, v in d.items()}


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

    if black_checkpoint := cfg.get("black_checkpoint", None):
        player_0.load_state_dict(torch.load(black_checkpoint))
    if white_checkpoint := cfg.get("white_checkpoint", None):
        player_1.load_state_dict(torch.load(white_checkpoint))

    epochs: int = cfg.get("epochs")
    episode_len: int = cfg.get("episode_len")

    pbar = tqdm(range(epochs))

    for i in pbar:
        data_0, data_1, info = env.rollout(
            episode_len=episode_len,
            player_black=player_0,
            player_white=player_1,
            augment=cfg.get("augment", False),
        )

        info.update(add_prefix(player_0.learn(data_0), "train/player_black_"))
        info.update(add_prefix(player_1.learn(data_1), "train/player_white_"))

        info.update(
            {
                "eval/black_against_random": eval_win_rate(
                    env, player_black=player_0, player_white=uniform_policy
                ),
                "eval/white_against_random": 1
                - eval_win_rate(
                    env, player_white=player_1, player_black=uniform_policy
                ),
            }
        )

        run.log(info)

        pbar.set_postfix(
            {
                "fps": env._fps,
            }
        )

    run.log(
        {
            "eval/black_win_final": eval_win_rate(
                env, player_black=player_0, player_white=player_1
            ),
        }
    )

    torch.save(player_0.state_dict(), os.path.join(run.dir, "player_black.pt"))
    torch.save(player_1.state_dict(), os.path.join(run.dir, "player_white.pt"))


if __name__ == "__main__":
    main()
