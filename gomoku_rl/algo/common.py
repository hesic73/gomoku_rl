from tensordict import TensorDictBase
import torch
from torchrl.envs import EnvBase
from typing import Callable
from tqdm import tqdm

from torchrl.data import ReplayBuffer, TensorDictReplayBuffer
from torchrl.data import LazyMemmapStorage, LazyTensorStorage, ListStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement, Sampler

from torch.cuda import _device_t


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
