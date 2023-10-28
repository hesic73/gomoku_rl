from tensordict import TensorDictBase
import torch
from torchrl.envs import EnvBase
from typing import Callable
from tqdm import tqdm

from torchrl.data import ReplayBuffer, TensorDictReplayBuffer
from torchrl.data import LazyMemmapStorage, LazyTensorStorage, ListStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement, Sampler

from torchrl.envs import EnvBase
from torchrl.collectors import SyncDataCollector
from torch.cuda import _device_t
from torchrl.envs.utils import ExplorationType


def get_replay_buffer(
    buffer_size: int,
    batch_size: int,
    sampler: Sampler | None = None,
    device: _device_t = None,
):
    storage = LazyTensorStorage(max_size=buffer_size, device=device)
    buffer = TensorDictReplayBuffer(
        storage=storage, batch_size=batch_size, sampler=sampler
    )
    return buffer


def get_collector(
    env_func: Callable[[], EnvBase],
    policy: Callable[[TensorDictBase], TensorDictBase],
    frames_per_batch: int,
    total_frames: int,
    device: _device_t,
):
    data_collector = SyncDataCollector(
        env_func,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        # this is the default behaviour: the collector runs in ``"random"`` (or explorative) mode
        exploration_type=ExplorationType.RANDOM,
        # We set the all the devices to be identical. Below is an example of
        # heterogeneous devices
        device=device,
        storing_device=device,
        split_trajs=False,
    )
    return data_collector
