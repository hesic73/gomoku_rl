import hydra
from omegaconf import DictConfig, OmegaConf
import os
import datetime
from gomoku_rl import CONFIG_PATH
import torch


from gomoku_rl.env import GomokuEnvWithOpponent
from gomoku_rl.utils.transforms import LogOnEpisode
from torchrl.envs.transforms import (
    TransformedEnv,
    InitTracker,
    Compose,
)
from torchrl.collectors import SyncDataCollector
from gomoku_rl.utils.wandb import init_wandb
from gomoku_rl.utils.misc import ActorBank
from gomoku_rl.algo import get_policy
import logging
from tqdm import tqdm
import numpy as np
from tensordict.nn import TensorDictModule
from typing import Callable, Any
from tensordict import TensorDict


@torch.no_grad()
def eval_win_rate(env: TransformedEnv, policy, max_episode_length: int = 180):
    rates = []
    for _ in range(10):
        rates.append(_eval_win_rate(env, policy, max_episode_length))
    return sum(rates) / len(rates)


@torch.no_grad()
def _eval_win_rate(env: TransformedEnv, policy, max_step: int):
    env.eval()
    tensordict = env.reset()
    n = tensordict.batch_size[0]

    episode_done = torch.zeros(n, device=tensordict.device, dtype=torch.bool)  # (E,)

    interested_tensordict = []

    tensordict_ = tensordict
    for _ in range(max_step):
        tensordict_ = policy(tensordict_)
        tensordict, tensordict_ = env.step_and_maybe_reset(tensordict_)
        done = tensordict.get(("next", "done"))
        truncated = tensordict.get(
            ("next", "truncated"),
            default=torch.zeros((), device=done.device, dtype=torch.bool),
        )
        done = done | truncated

        _reset: torch.Tensor = done.squeeze(-1)  # (E,)
        index: torch.Tensor = _reset & ~episode_done
        episode_done: torch.Tensor = episode_done | _reset

        interested_tensordict.extend(tensordict["next", "stats"][index].unbind(0))

        if episode_done.all().item():
            break

    env.reset()
    env.train()

    interested_tensordict = torch.stack(interested_tensordict, dim=0)
    return interested_tensordict["win"].float().mean().item()


def calculate_win_rate_matrix(
    env: TransformedEnv,
    actor_paths: list[str],
    create_actor_func: Callable[
        [
            str,
        ],
        TensorDictModule,
    ],
    max_episode_length: int = 180,
):
    n = len(actor_paths)
    m = np.zeros(n, n)
    for i in range(n):
        for j in range(n):
            actor_0 = create_actor_func(actor_paths[i])
            actor_1 = create_actor_func(actor_paths[j])
            env.base_env.set_opponent_policy(actor_1)
            m[i, j] = _eval_win_rate(
                env, actor_0, max_episode_length=max_episode_length
            )

    return m


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    run = init_wandb(cfg=cfg)

    root_dir: str = cfg.get("root_dir", "sp_root")
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)
        logging.info(f"Create directory {root_dir}.")

    actorBank = ActorBank(
        dir=os.path.join(root_dir, datetime.datetime.now().strftime("%m-%d_%H-%M")),
        latest_n=1,
    )

    initial_policy = cfg.get("initial_policy", None)
    if initial_policy is not None:
        initial_policy = torch.load(initial_policy)
        logging.info(f"Initial Policy: {initial_policy}")

    base_env = GomokuEnvWithOpponent(
        num_envs=cfg.num_envs,
        board_size=cfg.board_size,
        device=cfg.device,
        initial_policy=initial_policy,
    )

    stats_keys = base_env.stats_keys

    def _stone_win_rate_process_func(x: torch.Tensor):
        x = x.view(-1, 2)
        mask = x[:, 1]
        x = x[mask]
        a = torch.sum(x[..., 0].float())
        b = torch.sum(x[..., 1].float())
        tmp = (a / b).item()
        return tmp

    logger = LogOnEpisode(
        cfg.num_envs,
        in_keys=stats_keys,
        log_keys=stats_keys,
        logger_func=lambda x: run.log(x),
        process_func={
            ("stats", "black_win_rate"): _stone_win_rate_process_func,
            ("stats", "white_win_rate"): _stone_win_rate_process_func,
        },
    )
    transforms = [InitTracker(), logger]
    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    policy = get_policy(
        name=cfg.algo.name,
        cfg=cfg.algo,
        action_spec=env.action_spec,
        observation_spec=env.observation_spec,
        device=env.device,
    )

    frames_per_batch = cfg.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    update_interval = int(cfg.get("update_interval", 10))
    assert update_interval > 0
    update_threshold = cfg.get("update_threshold")
    assert 0.7 < update_threshold < 1.0

    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.device,
        storing_device=cfg.get("storing_device", "cpu"),
        return_same_td=True,
    )

    pbar = tqdm(collector, total=total_frames // frames_per_batch)
    env.train()
    for i, data in enumerate(pbar):
        # data (E,train_every)
        info = {"env_frames": collector._frames}
        info.update(policy.train_op(data.to_tensordict()))

        if i != 0 and i % update_interval == 0:
            actor_paths = actorBank.get_actor_paths()
            if len(actor_paths) == 0:
                wr = eval_win_rate(
                    env=env,
                    policy=policy,
                    max_episode_length=int((cfg.board_size**2) // 2),
                )
            else:
                _wrs = []
                for actor_path in actorBank.get_actor_paths():
                    opponent = policy.from_checkpoint(
                        torch.load(actor_path),
                        cfg=cfg.algo.actor,
                        action_spec=env.action_spec,
                        device=cfg.device,
                    )
                    base_env.set_opponent_policy(opponent)
                    _wr = eval_win_rate(
                        env=env,
                        policy=policy,
                        max_episode_length=int((cfg.board_size**2) // 2),
                    )
                    _wrs.append(_wr)
                    print(actor_path, _wr)

                wr = sum(_wrs) / len(_wrs)

            logging.info(f"Win Rate: {wr*100:.2f}%")

            if wr > update_threshold:
                ckpt_path = actorBank.save(policy.get_actor().state_dict())
                logging.info(f"Save checkpoint to {ckpt_path}")

        run.log(info)
        pbar.set_postfix(
            {
                "frames": collector._frames,
            }
        )

        if len(actorBank) > 0:
            opponent = policy.from_checkpoint(
                actorBank.get_random(),
                cfg=cfg.algo.actor,
                action_spec=env.action_spec,
                device=cfg.device,
            )
            base_env.set_opponent_policy(opponent)
            collector.reset()


if __name__ == "__main__":
    main()
