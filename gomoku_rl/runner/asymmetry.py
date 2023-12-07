import logging
from typing import Any
from omegaconf import DictConfig

from gomoku_rl.utils.policy import _policy_t, uniform_policy
from .base import SPRunner
from gomoku_rl.utils.misc import get_kwargs, add_prefix
import os
import copy
from gomoku_rl.utils.psro import (
    ConvergedIndicator,
    Population,
    get_meta_solver,
    PayoffType,
    get_new_payoffs_sp,
)
from gomoku_rl.utils.eval import eval_win_rate
import torch
from gomoku_rl.policy import get_policy
import numpy as np


class SimpleRunner(SPRunner):
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
        if self.cfg.get("checkpoint", None):
            _policy = copy.deepcopy(self.policy)
            _policy.eval()
        else:
            _policy = uniform_policy
        self.population = Population(
            initial_policy=_policy,
            dir=os.path.join(self.run_dir, "population"),
            device=cfg.device,
        )
        self.payoffs = get_new_payoffs_sp(
            env=self.env,
            population=self.population,
            old_payoffs=None,
            type=PayoffType.black_vs_white,
        )  # black vs white
        print(repr(self.payoffs))
        self.meta_solver = get_meta_solver(cfg.get("meta_solver", "uniform"))
        self.meta_policy: np.ndarray | None = None

    def _get_baseline(self) -> _policy_t:
        pretrained_dir = os.path.join(
            "pretrained_models",
            f"{self.cfg.board_size}_{self.cfg.board_size}",
            f"{self.cfg.algo.name}",
        )

        if os.path.isdir(pretrained_dir) and (
            ckpts := [
                p
                for f in os.listdir(pretrained_dir)
                if os.path.isfile(p := os.path.join(pretrained_dir, f))
                and p.endswith(".pt")
            ]
        ):
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
        self.population.sample(self.meta_policy)
        data, info = self.env.rollout_player_white(
            rounds=self.rounds,
            player=self.policy,
            opponent=self.population,
            augment=self.cfg.get("augment", False),
            out_device=self.cfg.get("out_device", None),
        )
        info.update(add_prefix(self.policy.learn(data.to_tensordict()), "policy/"))
        del data

        info.update(
            {
                "eval/opponent_vs_player": eval_win_rate(
                    self.eval_env,
                    player_black=self.population,
                    player_white=self.policy,
                ),
                "eval/player_vs_baseline": eval_win_rate(
                    self.eval_env, player_black=self.policy, player_white=self.baseline
                ),
                "eval/baseline_vs_player": eval_win_rate(
                    self.eval_env, player_black=self.baseline, player_white=self.policy
                ),
            }
        )

        wr = 1 - info["eval/opponent_vs_player"]
        info.update({"win_rate": wr})

        self.converged_indicator.update(wr)
        if self.converged_indicator.converged():
            self.converged_indicator.reset()
            _policy = copy.deepcopy(self.policy)
            _policy.eval()
            self.population.add(_policy)
            self.payoffs = get_new_payoffs_sp(
                env=self.env,
                population=self.population,
                old_payoffs=self.payoffs,
                type=PayoffType.black_vs_white,
            )
            print(repr(self.payoffs))
            self.meta_policy, _ = self.meta_solver(payoffs=self.payoffs)
            logging.info(f"Meta Policy: {self.meta_policy}")

        if epoch % 50 == 0 and epoch != 0:
            torch.cuda.empty_cache()

        return info

    def _post_run(self):
        pass

    def _log(self, info: dict[str, Any], epoch: int):
        if epoch % 5 == 0:
            print(
                "Opponent vs Player:{:.2f}%\tPlayer vs Baseline:{:.2f}%\tBaseline vs Player:{:.2f}%".format(
                    info["eval/opponent_vs_player"] * 100,
                    info["eval/player_vs_baseline"] * 100,
                    info["eval/baseline_vs_player"] * 100,
                )
            )
        return super()._log(info, epoch)
