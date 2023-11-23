import hydra
from omegaconf import DictConfig, OmegaConf
import os
import datetime
from gomoku_rl import CONFIG_PATH
import torch


from gomoku_rl.env import GomokuEnv
from gomoku_rl.utils.misc import add_prefix, set_seed
from gomoku_rl.utils.eval import eval_win_rate, get_payoff_matrix
from gomoku_rl.utils.visual import annotate_heatmap, heatmap
import matplotlib.pyplot as plt
from PIL import Image
import io
import wandb
from gomoku_rl.utils.wandb import init_wandb
from gomoku_rl.utils.policy import uniform_policy, _policy_t
from gomoku_rl.policy import get_policy, get_pretrained_policy
import logging
from tqdm import tqdm
import numpy as np
from typing import Callable, Any, Dict
import functools


def payoff_headmap(
    env: GomokuEnv,
    row_policies: list[_policy_t],
    col_policies: list[_policy_t],
    row_labels: list[str],
    col_labels: list[str],
):
    payoff = get_payoff_matrix(
        env=env, row_policies=row_policies, col_policies=col_policies, n=1
    )
    print(payoff)
    im, _ = heatmap(
        payoff * 100,
        row_labels=row_labels,
        col_labels=col_labels,
    )
    annotate_heatmap(im, valfmt="{x:.2f}%")
    plt.tight_layout()
    # https://stackoverflow.com/questions/8598673/how-to-save-a-pylab-figure-into-in-memory-file-which-can-be-read-into-pil-image/8598881
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    im = Image.open(buf)
    im = wandb.Image(im)
    return im


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train_InRL")
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
    save_interval: int = cfg.get("save_interval")
    log_interval: int = cfg.get("log_interval")
    run_dir = cfg.get("run_dir", None)
    if run_dir is None:
        run_dir = run.dir
    os.makedirs(run_dir, exist_ok=True)
    logging.info(f"run_dir:{run_dir}")

    history_paths_black: list[str] = []
    history_paths_white: list[str] = []

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
            path_black = os.path.join(run_dir, f"player_black_{i}.pt")
            history_paths_black.append(path_black)
            torch.save(player_0.state_dict(), path_black)
            path_white = os.path.join(run_dir, f"player_white_{i}.pt")
            history_paths_white.append(path_white)
            torch.save(player_1.state_dict(), path_white)

    run.log(
        {
            "eval/black_win_final": eval_win_rate(
                env, player_black=player_0, player_white=player_1
            ),
        }
    )

    torch.save(player_0.state_dict(), os.path.join(run_dir, "player_black_final.pt"))
    torch.save(player_1.state_dict(), os.path.join(run_dir, "player_white_final.pt"))

    make_player = functools.partial(
        get_pretrained_policy,
        name=cfg.algo.name,
        cfg=cfg.algo,
        action_spec=env.action_spec,
        observation_spec=env.observation_spec,
        device=env.device,
    )
    num_players = 5
    history_paths_black = history_paths_black[-num_players:]
    history_paths_white = history_paths_white[-num_players:]
    assert len(history_paths_white) == len(history_paths_black)
    num_players = len(history_paths_black)

    players_black = [make_player(checkpoint_path=p) for p in history_paths_black]
    labels_black = [os.path.split(p)[1] for p in history_paths_black]
    players_white = [make_player(checkpoint_path=p) for p in history_paths_white]
    labels_white = [os.path.split(p)[1] for p in history_paths_white]

    run.log(
        {
            "payoff": payoff_headmap(
                env, players_black, players_white, labels_black, labels_white
            )
        }
    )


if __name__ == "__main__":
    main()
