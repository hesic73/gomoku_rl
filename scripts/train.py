import hydra
from omegaconf import DictConfig, OmegaConf
import os
import datetime
from gomoku_rl import CONFIG_PATH
import torch


from gomoku_rl.env import GomokuEnv
from gomoku_rl.utils.misc import add_prefix, set_seed
from gomoku_rl.utils.psro import (
    ConvergedIndicator,
    PSROPolicyWrapper,
    get_new_payoffs,
    solve_nash,
    payoffs_append_col,
    payoffs_append_row,
    calculate_jpc,
)
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
import copy


def payoff_headmap(
    env: GomokuEnv,
    row_policies: list[_policy_t],
    col_policies: list[_policy_t],
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
):
    payoff = get_payoff_matrix(
        env=env, row_policies=row_policies, col_policies=col_policies, n=1
    )
    print(payoff)
    if row_labels is None:
        row_labels = [i for i in range(len(row_policies))]
    if col_labels is None:
        col_labels = [i for i in range(len(col_policies))]

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


def get_baseline(
    cfg: DictConfig,
    action_spec,
    observation_spec,
    device,
    baseline_path: str | None = None,
):
    if baseline_path is None:
        baseline_path = os.path.join(
            "pretrained_models", f"{cfg.board_size}_{cfg.board_size}", "baseline.pt"
        )
    if os.path.isfile(baseline_path):
        baseline = get_policy(
            name=cfg.algo.name,
            cfg=cfg.algo,
            action_spec=action_spec,
            observation_spec=observation_spec,
            device=device,
        )
        baseline.load_state_dict(torch.load(baseline_path))

        logging.info(f"Baseline: {baseline_path}.")
    else:
        baseline = uniform_policy
        logging.info("Baseline: random.")

    return baseline


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.board_size = int(cfg.board_size)
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

    if black_checkpoint := cfg.get("black_checkpoint", None):
        player_0.load_state_dict(torch.load(black_checkpoint))
        logging.info(f"black_checkpoint:{black_checkpoint}")
    if white_checkpoint := cfg.get("white_checkpoint", None):
        player_1.load_state_dict(torch.load(white_checkpoint))
        logging.info(f"white_checkpoint:{white_checkpoint}")

    baseline = get_baseline(
        cfg=cfg,
        action_spec=env.action_spec,
        observation_spec=env.observation_spec,
        device=cfg.device,
    )

    epochs: int = cfg.get("epochs")
    rounds: int = cfg.get("rounds")
    save_interval: int = cfg.get("save_interval")
    log_interval: int = cfg.get("log_interval")
    run_dir = cfg.get("run_dir", None)
    if run_dir is None:
        run_dir = run.dir
    os.makedirs(run_dir, exist_ok=True)
    logging.info(f"run_dir:{run_dir}")

    learning_player_id = 0
    converged_indicator = ConvergedIndicator(
        mean_threshold=cfg.get("mean_threshold", 0.99),
        std_threshold=cfg.get("std_threshold", 0.005),
    )

    player_0 = PSROPolicyWrapper(
        player_0, dir=os.path.join(run_dir, "population_0"), device=cfg.device
    )
    player_1 = PSROPolicyWrapper(
        player_1, dir=os.path.join(run_dir, "population_1"), device=cfg.device
    )
    player_0.set_oracle_mode(True)
    player_1.set_oracle_mode(False)

    payoffs = get_new_payoffs(
        env=env,
        population_0=player_0.population,
        population_1=player_1.population,
        old_payoffs=None,
    )

    pbar = tqdm(range(epochs))

    for i in pbar:
        if learning_player_id == 0:
            player_1.sample()
        else:
            player_0.sample()
        data_0, data_1, info = env.rollout(
            rounds=rounds,
            player_black=player_0,
            player_white=player_1,
            buffer_batch_size=cfg.get("buffer_batch_size", cfg.num_envs),
            augment=cfg.get("augment", False),
            return_black_transitions=learning_player_id == 0,
            return_white_transitions=learning_player_id != 0,
            buffer_device=cfg.get("buffer_device", "cpu"),
        )
        if learning_player_id == 0:
            info.update(add_prefix(player_0.policy.learn(data_0), "player_black/"))
        else:
            info.update(add_prefix(player_1.policy.learn(data_1), "player_white/"))

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
                "Black vs White:{:.2f}%\tBlack vs baseline:{:.2f}%\tWhite vs baseline:{:.2f}%".format(
                    info["eval/black_vs_white"] * 100,
                    info["eval/black_vs_baseline"] * 100,
                    info["eval/white_vs_baseline"] * 100,
                )
            )

        run.log(info)

        converged_indicator.update(
            info["eval/black_vs_white"]
            if learning_player_id == 0
            else (1 - info["eval/black_vs_white"])
        )
        if converged_indicator.converged():
            converged_indicator.reset()
            if learning_player_id == 0:
                player_0.add_current_policy()
                player_0.set_oracle_mode(False)
                player_1.set_oracle_mode(True)
            else:
                player_1.add_current_policy()
                player_1.set_oracle_mode(False)
                player_0.set_oracle_mode(True)

            if learning_player_id % 2 == 0:
                payoffs = payoffs_append_row(
                    env,
                    population_0=player_0.population,
                    population_1=player_1.population,
                    old_payoffs=payoffs,
                )
            else:
                payoffs = payoffs_append_col(
                    env,
                    population_0=player_0.population,
                    population_1=player_1.population,
                    old_payoffs=payoffs,
                )

            meta_policy_0, meta_policy_1 = solve_nash(payoffs=payoffs)
            logging.info(f"Meta Policy: Black {meta_policy_0}, White {meta_policy_1}")
            player_0.set_meta_policy(meta_policy=meta_policy_0)
            player_1.set_meta_policy(meta_policy=meta_policy_1)

            learning_player_id = (learning_player_id + 1) % 2
            logging.info(f"learning_player_id:{learning_player_id}")
            if learning_player_id % 2 == 0:
                logging.info(f"JPC:{calculate_jpc(payoffs+1)/2}")

        if i % save_interval == 0 and i != 0:
            torch.save(
                player_0.policy.state_dict(), os.path.join(run_dir, f"black_{i}.pt")
            )
            torch.save(
                player_1.policy.state_dict(), os.path.join(run_dir, f"white_{i}.pt")
            )

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

    torch.save(
        player_0.policy.state_dict(), os.path.join(run_dir, "player_black_final.pt")
    )
    torch.save(
        player_1.policy.state_dict(), os.path.join(run_dir, "player_white_final.pt")
    )

    run.log(
        {
            "payoff": payoff_headmap(
                env,
                player_0.population.policy_sets,
                player_1.population.policy_sets,
            )
        }
    )


if __name__ == "__main__":
    main()
