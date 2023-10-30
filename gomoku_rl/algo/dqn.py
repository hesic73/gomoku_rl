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
):
    cnn_kwargs = OmegaConf.to_container(cfg.cnn_kwargs)
    cnn_kwargs.update(
        {"activation_class": getattr(nn, cnn_kwargs.get("activation_class", "ReLU"))}
    )
    mlp_kwargs = OmegaConf.to_container(cfg.mlp_kwargs)
    mlp_kwargs.update(
        {"activation_class": getattr(nn, mlp_kwargs.get("activation_class", "ReLU"))}
    )

    net = DuelingCnnDQNet(action_space_size, 1, cnn_kwargs, mlp_kwargs)
    actor = QValueActor(
        net,
        in_keys=[
            observation_key,
        ],
        spec=action_spec,
    )
    return actor


def load_actor(actor_cfg: DictConfig, action_spec: TensorSpec, ckpt_path: str):
    actor = make_actor(
        cfg=actor_cfg,
        action_spec=action_spec,
        observation_key="observation",
        action_space_size=action_spec.space.n,
    )
    actor.load_state_dict(torch.load(ckpt_path))

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
    cfg: DictConfig,
    env: TransformedEnv,
    total_frames: int,
    frames_per_batch: int,
    run=None,
):
    device = env.device

    batch_size: int = cfg.batch_size
    buffer_size: int = min(cfg.buffer_size, 100000)

    # Optimization steps per batch collected (aka UPD or updates per data)
    n_optim: int = cfg.n_optim

    action_spec: DiscreteTensorSpec = env.action_spec
    assert isinstance(action_spec, DiscreteTensorSpec)

    actor_cfg = cfg.actor
    actor = make_actor(
        actor_cfg,
        env.action_spec,
        "observation",
        action_space_size=action_spec.space.n,
    ).to(device)
    with torch.no_grad():
        actor(env.fake_tensordict())

    annealing_num_steps = cfg.annealing_num_steps
    actor_explore = make_actor_explore(
        actor=actor,
        annealing_num_steps=annealing_num_steps,
        eps_init=cfg.eps_init,
        eps_end=cfg.eps_end,
    )
    
    def policy(tensordict:TensorDictModule)->TensorDictBase:
        tensordict=actor_explore(tensordict)
        actor_explore.step()
        return tensordict

    loss_module, target_net_updater = get_loss_module(actor, cfg.gamma)

    collector = SyncDataCollector(
        lambda: env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        exploration_type=ExplorationType.RANDOM,
        device=device,
    )

    optimizer = torch.optim.Adam(
        loss_module.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
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

        losses = []
        grad_norms = []

        for gradient_step in range(1, n_optim + 1):
            transition = replay_buffer.sample().to(env.device)
            loss: torch.Tensor = loss_module(transition)["loss"]
            losses.append(loss.clone().detach())
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
            grad_norms.append(grad_norm)

            optimizer.step()

            if gradient_step % cfg.target_update_interval:
                target_net_updater.step()

        avg_loss = torch.stack(losses).mean()
        avg_grad_norm = torch.stack(grad_norms).mean()
        if run is not None:
            run.log({"loss": avg_loss.item(), "grad_norm": avg_grad_norm.item()})
        # print(f"Loss:{avg_loss.item():.4f}\tGrad Norm:{avg_grad_norm:.4f}")

    torch.save(actor.state_dict(), "dqn_actor.pt")
