# based on https://pytorch.org/rl/tutorials/coding_dqn.html

from typing import Callable, Dict, List, Any
from tensordict import TensorDictBase
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
from torchrl.envs import EnvBase
import datetime
import tempfile
from torchrl.record.loggers.csv import CSVLogger
import warnings

from .common import get_collector, get_replay_buffer


def make_actor(
    action_spec: TensorSpec,
    observation_key: NestedKey,
    action_space_size: int,
    device: _device_t = None,
):
    cnn_kwargs = {
        "num_cells": [32, 64, 128, 4],
        "kernel_sizes": [3, 3, 3, 1],
        "strides": [1, 1, 1, 1],
        "paddings": [1, 1, 1, 0],
        "activation_class": nn.ReLU,
        # This can be used to reduce the size of the last layer of the CNN
        # "squeeze_output": True,
        # "aggregator_class": nn.AdaptiveAvgPool2d,
        # "aggregator_kwargs": {"output_size": (1, 1)},
    }
    mlp_kwargs = {
        "depth": 2,
        "num_cells": [
            64,
            64,
        ],
        "activation_class": nn.ReLU,
    }
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


def train(env: EnvBase):
    device = env.device

    # the learning rate of the optimizer
    lr = 2e-3
    # weight decay
    wd = 1e-5
    # the beta parameters of Adam
    betas = (0.9, 0.999)
    # Optimization steps per batch collected (aka UPD or updates per data)
    n_optim = 8

    gamma = 0.99

    tau = 0.02

    total_frames = 5_000  # 500000
    frames_per_batch = 32  # 128
    total_frames = total_frames // frames_per_batch * frames_per_batch

    batch_size = 32  # 256

    buffer_size = min(total_frames, 100000)

    action_spec: DiscreteTensorSpec = env.action_spec
    assert isinstance(action_spec, DiscreteTensorSpec)

    actor = make_actor(
        env.action_spec,
        "observation",
        action_space_size=action_spec.space.n,
        device=device,
    )
    actor(env.fake_tensordict())
    actor_explore = make_actor_explore(
        actor=actor, annealing_num_steps=total_frames, eps_init=1.0, eps_end=0.05
    )

    loss_module, target_net_updater = get_loss_module(actor, gamma)

    collector = get_collector(
        env_func=lambda: env,
        policy=actor_explore,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
    )

    optimizer = torch.optim.Adam(
        loss_module.parameters(), lr=lr, weight_decay=wd, betas=betas
    )

    exp_name = "dqn/{}".format(datetime.datetime.now().strftime("%m-%d_%H-%M"))
    tmpdir = tempfile.TemporaryDirectory()
    logger = CSVLogger(exp_name=exp_name, log_dir=tmpdir.name)
    warnings.warn(f"log dir: {logger.experiment.log_dir}")

    log_interval = 500

    trainer = Trainer(
        collector=collector,
        total_frames=total_frames,
        frame_skip=1,
        loss_module=loss_module,
        optimizer=optimizer,
        logger=logger,
        optim_steps_per_batch=n_optim,
        log_interval=log_interval,
    )

    buffer_hook = ReplayBufferTrainer(
        get_replay_buffer(buffer_size, batch_size=batch_size),
    )
    buffer_hook.register(trainer)
    weight_updater = UpdateWeights(collector, update_weights_interval=1)
    weight_updater.register(trainer)
    recorder = Recorder(
        record_interval=100,  # log every 100 optimization steps
        record_frames=1000,  # maximum number of frames in the record
        frame_skip=1,
        policy_exploration=actor_explore,
        environment=env,
        exploration_type=ExplorationType.MODE,
        log_keys=[("next", "reward")],
        out_keys={("next", "reward"): "rewards"},
        log_pbar=True,
    )
    recorder.register(trainer)

    trainer.register_op("post_optim", target_net_updater.step)

    log_reward = LogReward(log_pbar=True)
    log_reward.register(trainer)

    trainer.train()
