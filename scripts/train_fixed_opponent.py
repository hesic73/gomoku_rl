import hydra
from omegaconf import DictConfig, OmegaConf
import os
import datetime
from gomoku_rl import CONFIG_PATH
import torch


from gomoku_rl.env import GomokuEnv
from gomoku_rl.utils.misc import add_prefix
from gomoku_rl.utils.eval import eval_win_rate
from gomoku_rl.utils.wandb import init_wandb
from gomoku_rl.utils.policy import uniform_policy
from gomoku_rl.policy import get_policy
import logging
from tqdm import tqdm
import numpy as np
from typing import Callable, Any, Dict


@hydra.main(
    version_base=None, config_path=CONFIG_PATH, config_name="train_fixed_opponent"
)
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

    player = get_policy(
        name=cfg.algo.name,
        cfg=cfg.algo,
        action_spec=env.action_spec,
        observation_spec=env.observation_spec,
        device=env.device,
    )

    if opponent_checkpoint_path := cfg.get("opponent_checkpoint", None):
        opponent = get_policy(
            name=cfg.algo.name,
            cfg=cfg.algo,
            action_spec=env.action_spec,
            observation_spec=env.observation_spec,
            device=env.device,
        )
        opponent.load_state_dict(torch.load(opponent_checkpoint_path))

        logging.info(f"Opponent: {opponent_checkpoint_path}.")
    else:
        opponent = uniform_policy
        logging.info("Opponent: random.")

    if checkpoint := cfg.get("checkpoint", None):
        player.load_state_dict(torch.load(checkpoint))

    epochs: int = cfg.get("epochs")
    rounds: int = cfg.get("rounds")
    log_interval: int = cfg.get("log_interval")
    color: str = cfg.get("color").lower()
    assert color in ("white", "black")

    if color == "white":
        player_black = opponent
        player_white = player
    else:
        player_black = player
        player_white = opponent

    pbar = tqdm(range(epochs))

    for i in pbar:
        data_0, data_1, info = env.rollout(
            rounds=rounds,
            player_black=player_black,
            player_white=player_white,
            augment=cfg.get("augment", False),
            return_black_transitions=color == "black",
            return_white_transitions=color == "white",
        )
        data = data_0 if color == "black" else data_1

        info.update(add_prefix(player.learn(data), f"player_{color}/"))

        info.update(
            {
                "eval/black_vs_white": eval_win_rate(
                    env, player_black=player_black, player_white=player_white
                ),
            }
        )
        if i % log_interval == 0:
            print(
                "Black vs White:{:.2f}%".format(
                    info["eval/black_vs_white"] * 100,
                )
            )

        run.log(info)

        pbar.set_postfix(
            {
                "fps": env._fps,
            }
        )

    torch.save(player.state_dict(), os.path.join(run.dir, f"player_{color}.pt"))


if __name__ == "__main__":
    main()
