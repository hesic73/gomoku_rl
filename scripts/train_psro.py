import hydra
from omegaconf import DictConfig, OmegaConf
import os
import datetime
from gomoku_rl import CONFIG_PATH
import torch


from gomoku_rl.env import GomokuEnv
from gomoku_rl.utils.misc import add_prefix, set_seed
from gomoku_rl.utils.eval import eval_win_rate
from gomoku_rl.utils.wandb import init_wandb
from gomoku_rl.utils.policy import uniform_policy
from gomoku_rl.policy import get_policy
import logging
from tqdm import tqdm
import numpy as np
from typing import Callable, Any, Dict


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train_psro")
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
    set_seed(seed)

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

    baseline_path = os.path.join(
        "pretrained_models", f"{cfg.board_size}_{cfg.board_size}", "baseline.pt"
    )
    if os.path.isfile(baseline_path):
        baseline = get_policy(
            name=cfg.algo.name,
            cfg=cfg.algo,
            action_spec=env.action_spec,
            observation_spec=env.observation_spec,
            device=env.device,
        )
        baseline.load_state_dict(torch.load(baseline_path))

        logging.info(f"Baseline: {baseline_path}.")
    else:
        baseline = uniform_policy
        logging.info("Baseline: random.")

    if black_checkpoint := cfg.get("black_checkpoint", None):
        player_0.load_state_dict(torch.load(black_checkpoint))
    if white_checkpoint := cfg.get("white_checkpoint", None):
        player_1.load_state_dict(torch.load(white_checkpoint))

    epochs: int = cfg.get("epochs")
    rounds: int = cfg.get("rounds")

    pbar = tqdm(range(epochs))

    for i in pbar:
        data_0, data_1, info = env.rollout(
            rounds=rounds,
            player_black=player_0,
            player_white=player_1,
            augment=cfg.get("augment", False),
        )

        info.update(add_prefix(player_0.learn(data_0), "player_black/"))
        info.update(add_prefix(player_1.learn(data_1), "player_white/"))

        info.update(
            {
                "eval/black_vs_white": eval_win_rate(
                    env, player_black=player_0, player_white=player_1
                ),
                "eval/black_vs_baseline": eval_win_rate(
                    env, player_black=player_0, player_white=baseline
                ),
                "eval/white_vs_baseline": 1
                - eval_win_rate(env, player_white=player_1, player_black=baseline),
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
