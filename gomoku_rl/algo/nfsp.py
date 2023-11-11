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
from gomoku_rl.utils.policy import uniform_policy,_policy_t
from typing import Any, List, Sequence

from torchrl.data import TensorDictReplayBuffer,Writer
import numpy as np
import torch
import enum


from .policy import Policy

class MODE(enum.Enum):
  best_response=enum.auto()
  average_policy=enum.auto()


class NFSPAgentWrapper(object):
  def __init__(self,behavioural_strategy:_policy_t,anticipatory_param:float) -> None:
    self.behavioural_strategy=behavioural_strategy
    self._anticipatory_param=anticipatory_param
    
    self._sample_episode_policy()
    
  def _sample_episode_policy(self):
    if np.random.rand() < self._anticipatory_param:
      self._mode = MODE.best_response
    else:
      self._mode = MODE.average_policy


class NFSPPolicy(Policy):
  def __init__(self, cfg: DictConfig, action_spec: DiscreteTensorSpec, observation_spec: TensorSpec, device: _device_t = "cuda") -> None:
    super().__init__(cfg, action_spec, observation_spec, device)
    
    self.batch_size: int = cfg.batch_size
    self.buffer_size: int = min(cfg.buffer_size, 100000)

    # Optimization steps per batch collected (aka UPD or updates per data)
    self.n_optim: int = cfg.n_optim


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