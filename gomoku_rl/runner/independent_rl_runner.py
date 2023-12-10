from typing import Any
from omegaconf import DictConfig
from .base import SPRunner, Runner
from gomoku_rl.utils.misc import add_prefix
from gomoku_rl.utils.eval import eval_win_rate
import torch


class IndependentRLRunner(Runner):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def _epoch(self, epoch: int) -> dict[str, Any]:
        data_black, data_white, info = self.env.rollout(
            rounds=self.rounds,
            player_black=self.policy_black,
            player_white=self.policy_white,
            augment=self.cfg.get("augment", False),
            out_device=self.cfg.get("out_device", None),
        )
        info.update(
            add_prefix(
                self.policy_black.learn(data_black.to_tensordict()), "policy_black/"
            )
        )
        del data_black
        info.update(
            add_prefix(
                self.policy_white.learn(data_white.to_tensordict()), "policy_white/"
            )
        )
        del data_white

        info.update(
            {
                "eval/black_vs_white": eval_win_rate(
                    self.eval_env,
                    player_black=self.policy_black,
                    player_white=self.policy_white,
                ),
                "eval/black_vs_baseline": eval_win_rate(
                    self.eval_env,
                    player_black=self.policy_black,
                    player_white=self.baseline,
                ),
                "eval/baseline_vs_white": eval_win_rate(
                    self.eval_env,
                    player_black=self.baseline,
                    player_white=self.policy_white,
                ),
            }
        )

        if epoch % 50 == 0 and epoch != 0:
            torch.cuda.empty_cache()

        return info

    def _post_run(self):
        pass

    def _log(self, info: dict[str, Any], epoch: int):
        if epoch % 5 == 0:
            print(
                "Black vs White:{:.2f}%\tBlack vs Baseline:{:.2f}%\tBaseline vs White:{:.2f}%".format(
                    info["eval/black_vs_white"] * 100,
                    info["eval/black_vs_baseline"] * 100,
                    info["eval/baseline_vs_white"] * 100,
                )
            )
        return super()._log(info, epoch)


class IndependentRLSPRunner(SPRunner):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def _epoch(self, epoch: int) -> dict[str, Any]:
        data, info = self.env.rollout_self_play(
            steps=self.steps,
            player=self.policy,
            augment=self.cfg.get("augment", False),
            out_device=self.cfg.get("out_device", None),
        )
        info.update(add_prefix(self.policy.learn(data.to_tensordict()), "policy/"))
        del data

        info.update(
            {
                "eval/player_vs_player": eval_win_rate(
                    self.eval_env,
                    player_black=self.policy,
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

        if epoch % 50 == 0 and epoch != 0:
            torch.cuda.empty_cache()

        return info

    def _post_run(self):
        pass

    def _log(self, info: dict[str, Any], epoch: int):
        if epoch % 5 == 0:
            print(
                "Player vs Player:{:.2f}%\tPlayer vs Baseline:{:.2f}%\tBaseline vs Player:{:.2f}%".format(
                    info["eval/player_vs_player"] * 100,
                    info["eval/player_vs_baseline"] * 100,
                    info["eval/baseline_vs_player"] * 100,
                )
            )
        return super()._log(info, epoch)
