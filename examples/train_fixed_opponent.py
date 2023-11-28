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
    set_seed(seed)

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
        opponent.eval()
        logging.info(f"Opponent: {opponent_checkpoint_path}.")
    else:
        opponent = uniform_policy
        logging.info("Opponent: random.")

    baseline_path = os.path.join(
        "pretrained_models", f"{cfg.board_size}_{cfg.board_size}", "0.pt"
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
        baseline.eval()
        logging.info(f"Baseline: {baseline_path}.")
    else:
        baseline = uniform_policy
        logging.info("Baseline: random.")

    if checkpoint := cfg.get("checkpoint", None):
        player.load_state_dict(torch.load(checkpoint))

    epochs: int = cfg.get("epochs")
    rounds: int = cfg.get("rounds")
    log_interval: int = cfg.get("log_interval")
    save_interval: int = cfg.get("save_interval", -1)
    color: str = cfg.get("color").lower()
    assert color in ("white", "black")
    run_dir = cfg.get("run_dir", None)
    if run_dir is None:
        run_dir = run.dir
    os.makedirs(run_dir, exist_ok=True)
    logging.info(f"run_dir:{run_dir}")

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
            buffer_batch_size=cfg.get("buffer_batch_size", cfg.num_envs),
            augment=cfg.get("augment", False),
            return_black_transitions=color == "black",
            return_white_transitions=color == "white",
            buffer_device=cfg.get("buffer_device", "cpu"),
        )
        data = data_0 if color == "black" else data_1

        info.update(add_prefix(player.learn(data), f"player_{color}/"))

        info.update(
            {
                "eval/black_vs_white": eval_win_rate(
                    env, player_black=player_black, player_white=player_white
                ),
                "eval/player_vs_baseline": eval_win_rate(
                    env, player_black=player, player_white=baseline
                ),
                "eval/baseline_vs_player": eval_win_rate(
                    env, player_black=baseline, player_white=player
                ),
            }
        )
        if i % log_interval == 0:
            print(
                "Black vs White:{:.2f}%\tPlayer vs Baseline:{:.2f}%,\tBaseline vs Player:{:.2f}%".format(
                    info["eval/black_vs_white"] * 100,
                    info["eval/player_vs_baseline"] * 100,
                    info["eval/baseline_vs_player"] * 100,
                )
            )

        if i % save_interval == 0 and save_interval > 0:
            torch.save(
                player.state_dict(), os.path.join(run_dir, f"{color}_{i:4d}.pt")
            )
        run.log(info)

        pbar.set_postfix(
            {
                "fps": env._fps,
            }
        )

    torch.save(player.state_dict(), os.path.join(run_dir, f"{color}_final.pt"))


if __name__ == "__main__":
    main()
