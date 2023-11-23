import hydra
from omegaconf import DictConfig, OmegaConf
import os
import datetime
from gomoku_rl import CONFIG_PATH
import torch


from gomoku_rl.env import GomokuEnv
from gomoku_rl.utils.policy import _policy_t
from gomoku_rl.utils.eval import get_payoff_matrix
from gomoku_rl.utils.visual import annotate_heatmap, heatmap
from gomoku_rl.utils.misc import add_prefix, set_seed
from gomoku_rl.utils.eval import eval_win_rate
from gomoku_rl.utils.wandb import init_wandb
from gomoku_rl.utils.policy import uniform_policy
from gomoku_rl.policy import get_policy, get_pretrained_policy
import logging
from tqdm import tqdm
import numpy as np
import copy
import functools
import matplotlib.pyplot as plt
import io
from PIL import Image
import wandb


def payoff_headmap(env: GomokuEnv, policies: list[_policy_t], labels: list[str]):
    payoff = get_payoff_matrix(env=env, policies=policies, n=1)
    print(payoff)
    im, _ = heatmap(
        payoff * 100,
        row_labels=labels,
        col_labels=labels,
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
    set_seed(seed)

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
    run_dir = cfg.get("run_dir", None)
    if run_dir is None:
        run_dir = run.dir
    os.makedirs(run_dir, exist_ok=True)
    logging.info(f"run_dir:{run_dir}")

    history_paths: list[str] = []

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
            history_paths.append(path)
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

    make_player = functools.partial(
        get_pretrained_policy,
        name=cfg.algo.name,
        cfg=cfg.algo,
        action_spec=env.action_spec,
        observation_spec=env.observation_spec,
        device=env.device,
    )
    num_players = 5
    history_paths = history_paths[-num_players:]
    num_players = len(history_paths)

    players = [make_player(checkpoint_path=p) for p in history_paths]
    labels = [os.path.split(p)[1] for p in history_paths]

    run.log({"payoff": payoff_headmap(env, players, labels)})


if __name__ == "__main__":
    main()
