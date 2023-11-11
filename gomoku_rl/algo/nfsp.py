"""An implementation of neural fictitious self-play (NFSP)
    References:
    https://arxiv.org/pdf/1603.01121.pdf
    https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/nfsp.py
"""

from omegaconf import DictConfig
from tensordict.tensordict import TensorDictBase
import torch
from torch.cuda import _device_t
from torchrl.data.tensor_specs import DiscreteTensorSpec, TensorSpec
from .common import uniform_policy,_policy_t
from typing import Any, List, Sequence

from torchrl.data import TensorDictReplayBuffer,Writer
import numpy as np
import torch


from .policy import Policy

class NFSPPolicy(Policy):
  def __init__(self, cfg: DictConfig, action_spec: DiscreteTensorSpec, observation_spec: TensorSpec, device: _device_t = "cuda") -> None:
    super().__init__(cfg, action_spec, observation_spec, device)


class ReservoirWriter(Writer):
  def __init__(self) -> None:
    super().__init__()
    self._add_calls=0
    
  def extend(self, data: Sequence) -> torch.Tensor:
    raise NotImplementedError
  
  def add(self, data: Any) -> int:
    
    self._add_calls += 1  
    if len(self._storage)<self._storage.max_size:  
      ret = self._cursor
      self._storage[self._cursor] = data
      self._cursor = (self._cursor + 1) % self._storage.max_size
      return ret
    else:
      idx = np.random.randint(0, self._add_calls)
      if idx < self._storage.max_size:
        self._storage[idx] = data
      return idx
  
  def _empty(self):
    self._add_calls=0
    super()._empty()
    

class ReservoirBuffer(TensorDictReplayBuffer):
  def __init__(self, *, priority_key: str = "td_error", **kw) -> None:
    super().__init__(priority_key=priority_key,writer=ReservoirWriter(), **kw)