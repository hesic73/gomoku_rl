import abc
from omegaconf import DictConfig
from gomoku_rl.env import GomokuEnv
from gomoku_rl.utils.misc import set_seed
from gomoku_rl.utils.policy import _policy_t, uniform_policy
from gomoku_rl.policy import get_policy
from tqdm import tqdm
import torch
import logging
import os
import wandb
from typing import Any


class Runner(abc.ABC):
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        self.env = GomokuEnv(
            num_envs=cfg.num_envs,
            board_size=cfg.board_size,
            device=cfg.device,
        )
        self.eval_env = GomokuEnv(
            num_envs=512,
            board_size=cfg.board_size,
            device=cfg.device,
        )
        seed = cfg.get("seed", None)
        set_seed(seed)

        self.epochs: int = cfg.get("epochs")
        self.steps = cfg.steps
        self.save_interval: int = cfg.get("save_interval", -1)

        self.policy_black = get_policy(
            name=cfg.algo.name,
            cfg=cfg.algo,
            action_spec=self.env.action_spec,
            observation_spec=self.env.observation_spec,
            device=self.env.device,
        )
        self.policy_white = get_policy(
            name=cfg.algo.name,
            cfg=cfg.algo,
            action_spec=self.env.action_spec,
            observation_spec=self.env.observation_spec,
            device=self.env.device,
        )

        if black_checkpoint := cfg.get("black_checkpoint", None):
            self.policy_black.load_state_dict(
                torch.load(black_checkpoint, map_location=self.cfg.device)
            )
            logging.info(f"black_checkpoint:{black_checkpoint}")
        if white_checkpoint := cfg.get("white_checkpoint", None):
            self.policy_white.load_state_dict(
                torch.load(white_checkpoint, map_location=self.cfg.device)
            )
            logging.info(f"white_checkpoint:{white_checkpoint}")

        self.baseline = self._get_baseline()

        run_dir = cfg.get("run_dir", None)
        if run_dir is None:
            run_dir = wandb.run.dir
        os.makedirs(run_dir, exist_ok=True)
        logging.info(f"run_dir:{run_dir}")
        self.run_dir = run_dir

    def _get_baseline(self) -> _policy_t:
        pretrained_dir = os.path.join(
            "pretrained_models",
            f"{self.cfg.board_size}_{self.cfg.board_size}",
            f"{self.cfg.baseline.name}",
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
                name=self.cfg.baseline.name,
                cfg=self.cfg.baseline,
                action_spec=self.env.action_spec,
                observation_spec=self.env.observation_spec,
                device=self.env.device,
            )
            ckpts.sort()
            logging.info(f"Baseline:{ckpts[0]}")
            baseline.load_state_dict(torch.load(
                ckpts[0], map_location=self.cfg.device))
            baseline.eval()
            return baseline
        else:
            return uniform_policy

    @abc.abstractmethod
    def _epoch(self, epoch: int) -> dict[str, Any]:
        ...

    # @abc.abstractmethod
    def _post_run(self):
        pass

    def _log(self, info: dict[str, Any], epoch: int):
        if wandb.run is not None:
            wandb.run.log(info)

    def run(self, disable_tqdm: bool = False):
        pbar = tqdm(range(self.epochs), disable=disable_tqdm)
        for i in pbar:
            info = {}
            info.update(self._epoch(epoch=i))
            self._log(info=info, epoch=i)

            if i % self.save_interval == 0 and self.save_interval > 0:
                torch.save(
                    self.policy_black.state_dict(),
                    os.path.join(self.run_dir, f"black_{i:04d}.pt"),
                )
                torch.save(
                    self.policy_white.state_dict(),
                    os.path.join(self.run_dir, f"white_{i:04d}.pt"),
                )

            pbar.set_postfix(
                {
                    "fps": info['fps'],
                }
            )

        torch.save(
            self.policy_black.state_dict(),
            os.path.join(self.run_dir, f"black_final.pt"),
        )
        torch.save(
            self.policy_white.state_dict(),
            os.path.join(self.run_dir, f"white_final.pt"),
        )

        self._post_run()


class SPRunner(abc.ABC):
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        self.env = GomokuEnv(
            num_envs=cfg.num_envs,
            board_size=cfg.board_size,
            device=cfg.device,
        )
        self.eval_env = GomokuEnv(
            num_envs=512,
            board_size=cfg.board_size,
            device=cfg.device,
        )
        seed = cfg.get("seed", None)
        set_seed(seed)

        self.epochs: int = cfg.get("epochs")
        self.steps: int = cfg.steps
        self.save_interval: int = cfg.get("save_interval", -1)

        self.policy = get_policy(
            name=cfg.algo.name,
            cfg=cfg.algo,
            action_spec=self.env.action_spec,
            observation_spec=self.env.observation_spec,
            device=self.env.device,
        )

        if checkpoint := cfg.get("checkpoint", None):
            self.policy.load_state_dict(
                torch.load(checkpoint, map_location=self.cfg.device)
            )
            logging.info(f"load from {checkpoint}")

        self.baseline = self._get_baseline()

        run_dir = cfg.get("run_dir", None)
        if run_dir is None:
            run_dir = wandb.run.dir
        os.makedirs(run_dir, exist_ok=True)
        logging.info(f"run_dir:{run_dir}")
        self.run_dir = run_dir

    def _get_baseline(self) -> _policy_t:
        pretrained_dir = os.path.join(
            "pretrained_models",
            f"{self.cfg.board_size}_{self.cfg.board_size}",
            f"{self.cfg.baseline.name}",
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
                name=self.cfg.baseline.name,
                cfg=self.cfg.baseline,
                action_spec=self.env.action_spec,
                observation_spec=self.env.observation_spec,
                device=self.env.device,
            )
            ckpts.sort()
            logging.info(f"Baseline:{ckpts[0]}")
            baseline.load_state_dict(torch.load(
                ckpts[0], map_location=self.cfg.device))
            baseline.eval()
            return baseline
        else:
            return uniform_policy

    @abc.abstractmethod
    def _epoch(self, epoch: int) -> dict[str, Any]:
        ...

    # @abc.abstractmethod
    def _post_run(self):
        pass

    def _log(self, info: dict[str, Any], epoch: int):
        if wandb.run is not None:
            wandb.run.log(info)

    def run(self, disable_tqdm: bool = False):
        pbar = tqdm(range(self.epochs), disable=disable_tqdm)
        for i in pbar:
            info = {}
            info.update(self._epoch(epoch=i))
            self._log(info=info, epoch=i)

            if i % self.save_interval == 0 and self.save_interval > 0:
                torch.save(
                    self.policy.state_dict(),
                    os.path.join(self.run_dir, f"{i:04d}.pt"),
                )

            pbar.set_postfix(
                {
                    "fps": info['fps'],
                }
            )

        torch.save(
            self.policy.state_dict(),
            os.path.join(self.run_dir, f"final.pt"),
        )

        self._post_run()
