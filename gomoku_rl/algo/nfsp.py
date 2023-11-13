"""An implementation of neural fictitious self-play (NFSP)
    References:
    https://arxiv.org/pdf/1603.01121.pdf
    https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/nfsp.py
"""

from gomoku_rl.utils.module import init_params

from omegaconf import DictConfig
from tensordict import TensorDict
import torch
from torch.cuda import _device_t
from torchrl.data.tensor_specs import DiscreteTensorSpec, TensorSpec
from tensordict.nn import TensorDictModule
from typing import Any, List, Sequence

from torchrl.data import TensorDictReplayBuffer, Writer, LazyMemmapStorage
import numpy as np
import torch
import enum
from torch.optim import Adam
import contextlib

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
        device=device,
    )
    return player


class MODE(enum.Enum):
    best_response = enum.auto()
    average_policy = enum.auto()


class NFSPAgent(object):
    def __init__(
        self,
        behavioural_strategy: TensorDictModule,
        average_policy_network: TensorDictModule,
        anticipatory_param: float = 0.1,
        buffer_capacity: int = 200_000, 
        batch_size: int = 2048,
        sl_rl: float = 5e-4,
        device: _device_t = "cuda",
    ) -> None:
        self.behavioural_strategy = behavioural_strategy.to(device)
        self.average_policy_network = average_policy_network.to(device)

        self.behavioural_strategy.apply(init_params)
        self.average_policy_network.apply(init_params)

        self._anticipatory_param = anticipatory_param

        # reservoir buffer的并行性很差
        self.reservoir_buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(max_size=buffer_capacity),
            batch_size=batch_size,
        )

        self.opt_sl = Adam(params=self.average_policy_network.parameters(), lr=sl_rl)

        self._step_counter = 0

        self._sample_episode_policy()

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        if self._mode == MODE.best_response:
            tensordict = self.behavioural_strategy(tensordict)
            self.reservoir_buffer.extend(tensordict.select("observation", "action_mask", "action"))
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

    @mode.setter
    def mode(self, value: MODE):
        assert isinstance(value, MODE)
        self._mode = value

    @contextlib.contextmanager
    def temp_mode_as(self, mode: MODE):
        """Context manager to temporarily overwrite the mode."""
        assert isinstance(mode, MODE)
        previous_mode = self._mode
        self._mode = mode
        yield
        self._mode = previous_mode

    def train_sl(self):
        if len(self.reservoir_buffer)<self.reservoir_buffer._storage.max_size:
            return None
        
        transitions=self.reservoir_buffer.sample()
        
        self.opt_sl.zero_grad()
        print(self.average_policy_network)
        self.average_policy_network.train()


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
