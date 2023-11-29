import logging
from typing import Any
from omegaconf import DictConfig

from gomoku_rl.utils.policy import _policy_t
from .base import Runner
from gomoku_rl.utils.misc import get_kwargs, add_prefix
from gomoku_rl.utils.visual import payoff_headmap
import os
from gomoku_rl.utils.psro import (
    ConvergedIndicator,
    PSROPolicyWrapper,
    get_new_payoffs,
    get_meta_solver,
    calculate_jpc,
)
from gomoku_rl.utils.eval import eval_win_rate
import wandb
import torch
from gomoku_rl.policy import get_policy


class PSRORunner(Runner):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        ci_kwargs = get_kwargs(
            cfg,
            "mean_threshold",
            "std_threshold",
            "min_iter_steps",
            "max_iter_steps",
        )
        self.converged_indicator = ConvergedIndicator(**ci_kwargs)

        self.learning_player_id = cfg.get("first_id", 0)

        self.player_0 = PSROPolicyWrapper(
            self.policy_black,
            dir=os.path.join(self.run_dir, "population_0"),
            device=cfg.device,
        )
        self.player_1 = PSROPolicyWrapper(
            self.policy_white,
            dir=os.path.join(self.run_dir, "population_1"),
            device=cfg.device,
        )
        self.player_0.set_oracle_mode(self.learning_player_id == 0)
        self.player_1.set_oracle_mode(self.learning_player_id != 0)

        self.payoffs = get_new_payoffs(
            env=self.env,
            population_0=self.player_0.population,
            population_1=self.player_1.population,
            old_payoffs=None,
        )
        self.meta_solver = get_meta_solver(cfg.get("meta_solver", "uniform"))

    def _get_baseline(self) -> _policy_t:
        pretrained_dir = os.path.join(
            "pretrained_models",
            f"{self.cfg.board_size}_{self.cfg.board_size}",
        )
        ckpts = [
            p
            for f in os.listdir(pretrained_dir)
            if os.path.isfile(p := os.path.join(pretrained_dir, f))
            and p.endswith(".pt")
        ]

        if ckpts:
            baseline = get_policy(
                name=self.cfg.algo.name,
                cfg=self.cfg.algo,
                action_spec=self.env.action_spec,
                observation_spec=self.env.observation_spec,
                device=self.env.device,
            )
            logging.info(f"Baseline:{ckpts[0]}")
            baseline.load_state_dict(torch.load(ckpts[0]))
            baseline.eval()
            return baseline
        else:
            return super()._get_baseline()

    def _epoch(self, epoch: int) -> dict[str, Any]:
        data_0, data_1, info = self.env.rollout(
            rounds=self.rounds,
            player_black=self.player_0,
            player_white=self.player_1,
            buffer_batch_size=self.cfg.get("buffer_batch_size", self.cfg.num_envs),
            augment=self.cfg.get("augment", False),
            return_black_transitions=self.learning_player_id == 0,
            return_white_transitions=self.learning_player_id != 0,
            buffer_device=self.cfg.get("buffer_device", "cpu"),
        )
        if self.learning_player_id == 0:
            info.update(add_prefix(self.player_0.policy.learn(data_0), "black/"))
            del data_0
        else:
            info.update(add_prefix(self.player_1.policy.learn(data_1), "white/"))
            del data_1

        info.update(
            {
                "eval/black_vs_white": eval_win_rate(
                    self.env,
                    player_black=self.policy_black,
                    player_white=self.policy_white,
                ),
                "eval/black_vs_baseline": eval_win_rate(
                    self.env, player_black=self.player_0, player_white=self.baseline
                ),
                "eval/white_vs_baseline": 1
                - eval_win_rate(
                    self.env, player_black=self.baseline, player_white=self.player_1
                ),
            }
        )

        if epoch % 5:
            print(
                "Black vs White:{:.2f}%\tBlack vs baseline:{:.2f}%\tWhite vs baseline:{:.2f}%".format(
                    info["eval/black_vs_white"] * 100,
                    info["eval/black_vs_baseline"] * 100,
                    info["eval/white_vs_baseline"] * 100,
                )
            )

        self.converged_indicator.update(
            info["eval/black_vs_white"]
            if self.learning_player_id == 0
            else (1 - info["eval/black_vs_white"])
        )
        if self.converged_indicator.converged():
            self.converged_indicator.reset()
            if self.learning_player_id == 0:
                self.player_0.set_oracle_mode(False)
                self.player_1.set_oracle_mode(True)
            else:
                self.player_1.set_oracle_mode(False)
                self.player_0.set_oracle_mode(True)

            self.learning_player_id = (self.learning_player_id + 1) % 2
            logging.info(f"learning_player_id:{self.learning_player_id}")
            if self.learning_player_id == self.cfg.get("first_id", 0):
                self.player_0.add_current_policy()
                self.player_1.add_current_policy()
                payoffs = get_new_payoffs(
                    env=self.env,
                    population_0=self.player_0.population,
                    population_1=self.player_1.population,
                    old_payoffs=payoffs,
                )
                meta_policy_0, meta_policy_1 = self.meta_solver(payoffs=payoffs)
                logging.info(
                    f"Meta Policy: Black {meta_policy_0}, White {meta_policy_1}"
                )
                self.player_0.set_meta_policy(meta_policy=meta_policy_0)
                self.player_1.set_meta_policy(meta_policy=meta_policy_1)

                logging.info(f"JPC:{calculate_jpc(payoffs+1)/2}")

        return info

    def _post_run(self):
        wandb.log(
            {
                "payoff": payoff_headmap(
                    (self.payoffs + 1) / 2,
                )
            }
        )
