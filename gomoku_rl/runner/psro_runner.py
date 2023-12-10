import logging
from typing import Any
from omegaconf import DictConfig

from gomoku_rl.utils.policy import _policy_t, uniform_policy
from .base import Runner, SPRunner
from gomoku_rl.utils.misc import get_kwargs, add_prefix
from gomoku_rl.utils.visual import payoff_headmap
import os
import copy
from gomoku_rl.utils.psro import (
    ConvergedIndicator,
    Population,
    PSROPolicyWrapper,
    get_new_payoffs,
    get_new_payoffs_sp,
    init_payoffs_sp,
    get_meta_solver,
    calculate_jpc,
    PayoffType,
)
from gomoku_rl.utils.eval import eval_win_rate
import wandb
import torch


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

        if self.cfg.get("black_checkpoint", None):
            _policy = copy.deepcopy(self.policy_black)
            _policy.eval()
        else:
            _policy = uniform_policy
        self.player_0 = PSROPolicyWrapper(
            self.policy_black,
            population=Population(
                initial_policy=_policy,
                dir=os.path.join(self.run_dir, "population_0"),
                device=cfg.device,
            ),
        )
        if self.cfg.get("white_checkpoint", None):
            _policy = copy.deepcopy(self.policy_white)
            _policy.eval()
        else:
            _policy = uniform_policy
        self.player_1 = PSROPolicyWrapper(
            self.policy_white,
            population=Population(
                initial_policy=_policy,
                dir=os.path.join(self.run_dir, "population_1"),
                device=cfg.device,
            ),
        )
        self.player_0.set_oracle_mode(self.learning_player_id == 0)
        self.player_1.set_oracle_mode(self.learning_player_id != 0)

        self.payoffs = get_new_payoffs(
            env=self.eval_env,
            population_0=self.player_0.population,
            population_1=self.player_1.population,
            old_payoffs=None,
        )
        self.meta_solver = get_meta_solver(cfg.get("meta_solver", "uniform"))

    def _epoch(self, epoch: int) -> dict[str, Any]:
        if self.learning_player_id == 0:
            self.player_1.sample()
        else:
            self.player_0.sample()
        data_0, data_1, info = self.env.rollout(
            rounds=self.rounds,
            player_black=self.player_0,
            player_white=self.player_1,
            augment=self.cfg.get("augment", False),
            return_black_transitions=self.learning_player_id == 0,
            return_white_transitions=self.learning_player_id != 0,
        )

        info.update(
            {
                "pure_strategy_black": self.player_0.population._idx
                if self.learning_player_id == 1
                else -1,
                "pure_strategy_white": self.player_1.population._idx
                if self.learning_player_id == 0
                else -1,
            }
        )

        if self.learning_player_id == 0:
            info.update(add_prefix(self.policy_black.learn(data_0), "black/"))
            del data_0
        else:
            info.update(add_prefix(self.policy_white.learn(data_1), "white/"))
            del data_1

        info.update(
            {
                "eval/black_vs_white": eval_win_rate(
                    self.eval_env,
                    player_black=self.player_0,
                    player_white=self.player_1,
                ),
                "eval/black_vs_baseline": eval_win_rate(
                    self.eval_env,
                    player_black=self.policy_black,
                    player_white=self.baseline,
                ),
                "eval/white_vs_baseline": 1
                - eval_win_rate(
                    self.eval_env,
                    player_black=self.baseline,
                    player_white=self.policy_white,
                ),
            }
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
                self.payoffs = get_new_payoffs(
                    env=self.eval_env,
                    population_0=self.player_0.population,
                    population_1=self.player_1.population,
                    old_payoffs=self.payoffs,
                )
                print(repr(self.payoffs))
                meta_policy_0, meta_policy_1 = self.meta_solver(payoffs=self.payoffs)
                logging.info(
                    f"Meta Policy: Black {meta_policy_0}, White {meta_policy_1}"
                )
                self.player_0.set_meta_policy(meta_policy=meta_policy_0)
                self.player_1.set_meta_policy(meta_policy=meta_policy_1)

                logging.info(f"JPC:{calculate_jpc(self.payoffs+1)/2}")

        return info

    def _post_run(self):
        wandb.log(
            {
                "payoff": payoff_headmap(
                    (self.payoffs[:-5, :-5] + 1) / 2 * 100,
                )
            }
        )

    def _log(self, info: dict[str, Any], epoch: int):
        if epoch % 5 == 0:
            print(
                "Black vs White:{:.2f}%\tBlack vs baseline:{:.2f}%\tWhite vs baseline:{:.2f}%".format(
                    info["eval/black_vs_white"] * 100,
                    info["eval/black_vs_baseline"] * 100,
                    info["eval/white_vs_baseline"] * 100,
                )
            )
        return super()._log(info, epoch)


class PSROSPRunner(SPRunner):
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
        if (population_dir := cfg.get("population_dir", None)) and os.path.isdir(
            population_dir
        ):
            _policy = []
            for p in os.listdir(population_dir):
                tmp = copy.deepcopy(self.policy)
                tmp.load_state_dict(torch.load(os.path.join(population_dir, p)))
                tmp.eval()
                _policy.append(tmp)
        elif self.cfg.get("checkpoint", None):
            _policy = copy.deepcopy(self.policy)
            _policy.eval()
        else:
            _policy = uniform_policy
        self.population = Population(
            initial_policy=_policy,
            dir=os.path.join(self.run_dir, "population"),
            device=cfg.device,
        )

        self.payoffs = init_payoffs_sp(
            env=self.eval_env,
            population=self.population,
            type=PayoffType.black_vs_white,
        )
        print(repr(self.payoffs))
        self.meta_solver = get_meta_solver(cfg.get("meta_solver", "uniform"))
        if len(self.population) > 1:
            self.meta_policy_black, self.meta_policy_white = self.meta_solver(
                payoffs=self.payoffs
            )
            logging.info(
                f"Meta Policy: {self.meta_policy_black}, {self.meta_policy_white}"
            )
        else:
            self.meta_policy_black, self.meta_policy_white = None, None

    def _epoch(self, epoch: int) -> dict[str, Any]:
        info = {}
        self.population.sample(self.meta_policy_white)
        info.update({"pure_strategy_white": self.population._idx})
        data1, info1 = self.env.rollout_player_black(
            rounds=self.rounds,
            player=self.policy,
            opponent=self.population,
            augment=self.cfg.get("augment", False),
            out_device=self.cfg.get("out_device", None),
        )
        info.update(info1)
        self.population.sample(self.meta_policy_black)
        info.update({"pure_strategy_black": self.population._idx})
        data2, info2 = self.env.rollout_player_white(
            rounds=self.rounds,
            player=self.policy,
            opponent=self.population,
            augment=self.cfg.get("augment", False),
            out_device=self.cfg.get("out_device", None),
        )
        info.update(info2)
        data = torch.cat([data1, data2], dim=-1)
        info.update(add_prefix(self.policy.learn(data.to_tensordict()), "policy/"))
        del data

        info.update(
            {
                "eval/player_vs_opponent": eval_win_rate(
                    self.eval_env,
                    player_black=self.policy,
                    player_white=self.population,
                ),
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

        alpha = 0.5
        weighted_wr = alpha * info["eval/player_vs_opponent"] + (1 - alpha) * (
            1 - info["eval/opponent_vs_player"]
        )
        info.update({"weighted_win_rate": weighted_wr})

        self.converged_indicator.update(weighted_wr)
        if self.converged_indicator.converged():
            self.converged_indicator.reset()
            _policy = copy.deepcopy(self.policy)
            _policy.eval()
            self.population.add(_policy)

            self.payoffs = get_new_payoffs_sp(
                env=self.eval_env,
                population=self.population,
                old_payoffs=self.payoffs,
                type=PayoffType.black_vs_white,
            )
            print(repr(self.payoffs))
            self.meta_policy_black, self.meta_policy_white = self.meta_solver(
                payoffs=self.payoffs
            )
            logging.info(
                f"Meta Policy: {self.meta_policy_black}, {self.meta_policy_white}"
            )

        return info

    def _post_run(self):
        wandb.log(
            {
                "payoff": payoff_headmap(
                    (self.payoffs[-5:, -5:] + 1) / 2 * 100,
                )
            }
        )

    def _log(self, info: dict[str, Any], epoch: int):
        if epoch % 5 == 0:
            print(
                "Player vs Opponent:{:.2f}%\tOpponent vs Player:{:.2f}%\tPlayer vs Baseline:{:.2f}%\tBaseline vs Player:{:.2f}%".format(
                    info["eval/player_vs_opponent"] * 100,
                    info["eval/opponent_vs_player"] * 100,
                    info["eval/player_vs_baseline"] * 100,
                    info["eval/baseline_vs_player"] * 100,
                )
            )
        return super()._log(info, epoch)
