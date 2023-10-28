from typing import Callable, Dict, List, Any
from tensordict import TensorDictBase, TensorDict
from tensordict.utils import NestedKey
import torch
import torch.nn as nn

from tensordict.nn import TensorDictModule
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.modules import DuelingCnnDQNet, EGreedyWrapper, QValueActor
from torchrl.trainers import (
    LogReward,
    Recorder,
    ReplayBufferTrainer,
    Trainer,
    UpdateWeights,
)
from torchrl.data import TensorSpec, DiscreteTensorSpec
from torch.cuda import _device_t
from torchrl.envs.utils import ExplorationType
from torchrl.envs import TransformedEnv
import datetime
import tempfile
from torchrl.record.loggers.csv import CSVLogger

from .common import get_replay_buffer, env_next_to_agent_next
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.collectors import SyncDataCollector
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm


def print_gpu_usage():
    gpu_memory_allocated = torch.cuda.memory_allocated()
    gpu_memory_cached = torch.cuda.memory_cached()

    print(f"GPU memory allocated: {gpu_memory_allocated / (1024**3):.2f} GB")
    print(f"GPU memory cached: {gpu_memory_cached / (1024**3):.2f} GB")


def make_actor(
    cfg: DictConfig,
    action_spec: TensorSpec,
    observation_key: NestedKey,
    action_space_size: int,
    device: _device_t = None,
):
    cnn_kwargs = OmegaConf.to_container(cfg.cnn_kwargs)
    cnn_kwargs.update(
        {"activation_class": getattr(nn, cnn_kwargs.get("activation_class", "ReLU"))}
    )
    mlp_kwargs = OmegaConf.to_container(cfg.mlp_kwargs)
    mlp_kwargs.update(
        {"activation_class": getattr(nn, mlp_kwargs.get("activation_class", "ReLU"))}
    )

    net = DuelingCnnDQNet(action_space_size, 1, cnn_kwargs, mlp_kwargs).to(device)
    actor = QValueActor(
        net,
        in_keys=[
            observation_key,
        ],
        spec=action_spec,
    ).to(device)
    return actor


def make_actor_explore(
    actor: TensorDictModule, annealing_num_steps: int, eps_init: float, eps_end: float
):
    actor_explore = EGreedyWrapper(
        actor,
        annealing_num_steps=annealing_num_steps,
        eps_init=eps_init,
        eps_end=eps_end,
    )
    return actor_explore


def get_loss_module(actor: TensorDictModule, gamma: float):
    loss_module = DQNLoss(actor, delay_value=True)
    loss_module.make_value_estimator(gamma=gamma)
    target_updater = SoftUpdate(loss_module, eps=0.995)
    return loss_module, target_updater


def train(
    cfg: DictConfig, env: TransformedEnv, total_frames: int, frames_per_batch: int
):
    device = env.device

    batch_size: int = cfg.batch_size
    buffer_size: int = min(cfg.buffer_size, 100000)

    # the learning rate of the optimizer
    lr = 2e-3
    # weight decay
    wd = 1e-5
    # the beta parameters of Adam
    betas = (0.9, 0.999)

    # Optimization steps per batch collected (aka UPD or updates per data)
    n_optim: int = cfg.n_optim

    gamma = cfg.gamma

    action_spec: DiscreteTensorSpec = env.action_spec
    assert isinstance(action_spec, DiscreteTensorSpec)

    actor_cfg = cfg.actor
    actor = make_actor(
        actor_cfg,
        env.action_spec,
        "observation",
        action_space_size=action_spec.space.n,
        device=device,
    )
    with torch.no_grad():
        actor(env.fake_tensordict())

    actor_explore = make_actor_explore(
        actor=actor,
        annealing_num_steps=cfg.annealing_num_steps,
        eps_init=cfg.eps_init,
        eps_end=cfg.eps_end,
    )

    loss_module, target_net_updater = get_loss_module(actor, gamma)

    collector = SyncDataCollector(
        lambda: env,
        policy=actor_explore,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        exploration_type=ExplorationType.RANDOM,
        device=device,
        postproc=env_next_to_agent_next,
    )

    optimizer = torch.optim.Adam(
        loss_module.parameters(), lr=lr, weight_decay=wd, betas=betas
    )

    replay_buffer = get_replay_buffer(
        buffer_size=buffer_size,
        batch_size=batch_size,
        sampler=RandomSampler(),
        device=cfg.buffer_device,
    )

    for data in tqdm(collector, total=total_frames // frames_per_batch, disable=False):
        replay_buffer.extend(data.reshape(-1))
        if len(replay_buffer) < buffer_size:
            # print(f"Buffer:{len(replay_buffer)}/{buffer_size}")
            continue

        for gradient_step in range(n_optim):
            transition = replay_buffer.sample().to(env.device)
            loss: torch.Tensor = loss_module(transition)
            optimizer.zero_grad()
            loss["loss"].backward()
            optimizer.step()

            target_net_updater.step()
