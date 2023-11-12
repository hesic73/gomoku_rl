"""An implementation of neural fictitious self-play (NFSP)
    References:
    https://arxiv.org/pdf/1603.01121.pdf
    https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/nfsp.py
"""

from omegaconf import DictConfig
from tensordict import TensorDict
import torch
from torch.cuda import _device_t
from torchrl.data.tensor_specs import DiscreteTensorSpec, TensorSpec
from gomoku_rl.utils.policy import uniform_policy, _policy_t
from typing import Any, List, Sequence

from torchrl.data import TensorDictReplayBuffer, Writer, LazyMemmapStorage
import numpy as np
import torch
import enum


from .policy import Policy

from .common import make_egreedy_actor, make_ppo_actor

def make_nfsp_agent(cfg: DictConfig, action_spec: TensorSpec, device):
    actor_explore = make_egreedy_actor(cfg=cfg, action_spec=action_spec).to(device)
    average_policy_network = make_ppo_actor(
        cfg=cfg, action_spec=action_spec, device=device
    )
    player = NFSPAgent(
        behavioural_strategy=actor_explore,
        average_policy_network=average_policy_network,
    )
    return player


class MODE(enum.Enum):
    best_response = enum.auto()
    average_policy = enum.auto()


class NFSPAgent(object):
    def __init__(
        self,
        behavioural_strategy: _policy_t,
        average_policy_network: _policy_t,
        anticipatory_param: float = 0.1,
        reservoir_buffer_capacity: int = 100_000,
        batch_size: int = 2048,
    ) -> None:
        self.behavioural_strategy = behavioural_strategy
        self.average_policy_network = average_policy_network
        self._anticipatory_param = anticipatory_param

        self.reservoir_buffer = ReservoirBuffer(
            storage=LazyMemmapStorage(max_size=reservoir_buffer_capacity),
            batch_size=batch_size,
        )

        self._sample_episode_policy()

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        if self._mode == MODE.best_response:
            tensordict = self.behavioural_strategy(tensordict)
            self.reservoir_buffer.add(
                tensordict.select("observation", "action_mask", "action")
            )
            return tensordict
        else:
            return self.average_policy_network(tensordict)

    def _sample_episode_policy(self):
        if np.random.rand() < self._anticipatory_param:
            self._mode = MODE.best_response
        else:
            self._mode = MODE.average_policy

    @property
    def mode(self):
        return self._mode


class ReservoirWriter(Writer):
    def __init__(self) -> None:
        super().__init__()
        self._cursor = 0
        self._add_calls = 0

    def extend(self, data: Sequence) -> torch.Tensor:
        raise NotImplementedError

    def add(self, data: Any) -> int:
        self._add_calls += 1
        if len(self._storage) < self._storage.max_size:
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
        self._add_calls = 0
        super()._empty()


class ReservoirBuffer(TensorDictReplayBuffer):
    def __init__(self, *, priority_key: str = "td_error", **kw) -> None:
        super().__init__(priority_key=priority_key, writer=ReservoirWriter(), **kw)
