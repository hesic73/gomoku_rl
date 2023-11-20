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
import copy


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train_self_play")
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

    if checkpoint := cfg.get("checkpoint", None):
        player.load_state_dict(torch.load(checkpoint))

    epochs: int = cfg.get("epochs")
    steps: int = cfg.get("steps")
    save_interval: int = cfg.get("save_interval")
    log_interval: int = cfg.get("log_interval")

    pbar = tqdm(range(epochs))

    for i in pbar:
        data, info = env.self_play_rollout(
            steps=steps,
            player=player,
            augment=cfg.get("augment", False),
        )

        info.update(add_prefix(player.learn(data), "player/"))

        info.update(
            {
                "eval/black_vs_white": eval_win_rate(
                    env, player_black=player, player_white=player
                ),
                "eval/black_vs_baseline": eval_win_rate(
                    env, player_black=player, player_white=baseline
                ),
                "eval/white_vs_baseline": 1
                - eval_win_rate(env, player_white=player, player_black=baseline),
            }
        )

        if i % log_interval == 0:
            print(
                "Black vs baseline:{:.2f}%\tWhite vs baseline:{:.2f}%".format(
                    info["eval/black_vs_baseline"] * 100,
                    info["eval/white_vs_baseline"] * 100,
                )
            )

        run.log(info)

        pbar.set_postfix(
            {
                "fps": env._fps,
            }
        )

        if i % save_interval == 0 and i != 0:
            path = os.path.join(run.dir, f"player_{i}.pt")
            torch.save(player.state_dict(), path)
            logging.info(f"Save checkpoint to {path}.")
            if (
                info["eval/black_vs_baseline"] + info["eval/white_vs_baseline"]
            ) / 2 > 0.7:
                logging.info(f"New baseline: {path}")
                baseline = copy.deepcopy(player)

    run.log(
        {
            "eval/black_win_final": eval_win_rate(
                env, player_black=player, player_white=player
            ),
        }
    )

    torch.save(player.state_dict(), os.path.join(run.dir, "player_final.pt"))


if __name__ == "__main__":
    main()
