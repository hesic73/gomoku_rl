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

from torchrl.data import TensorDictReplayBuffer, Writer, LazyMemmapStorage, ReplayBuffer
import numpy as np
import torch
import enum
from torch.optim import Adam
import contextlib
from torchrl.objectives import DQNLoss, SoftUpdate

from .policy import Policy

from .common import (
    make_egreedy_actor,
    make_dqn_actor,
    make_ppo_actor,
    make_dataset_naive,
)


def make_nfsp_agent(
    cfg: DictConfig, action_spec: TensorSpec, observation_spec: TensorSpec, device
):
    actor_explore = make_egreedy_actor(cfg=cfg, action_spec=action_spec).to(device)
    average_policy_network = make_ppo_actor(
        cfg=cfg, action_spec=action_spec, device=device
    )
    fake_tensordict: TensorDict = observation_spec.zero()
    fake_tensordict.set("action_mask", torch.ones_like(fake_tensordict["action_mask"]))
    with torch.no_grad():
        actor_explore(fake_tensordict)
        average_policy_network(fake_tensordict)
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
        sl_lr: float = 5e-4,
        rl_lr: float = 5e-4,
        gamma: float = 0.95,
        device: _device_t = "cuda",
    ) -> None:
        self.device = device
        self.behavioural_strategy = behavioural_strategy.to(device)
        self.average_policy_network = average_policy_network.to(device)

        self.behavioural_strategy.apply(init_params)
        self.average_policy_network.apply(init_params)

        self._anticipatory_param = anticipatory_param

        self.batch_size = batch_size
        # reservoir buffer的并行性很差
        self.reservoir_buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(max_size=buffer_capacity),
            batch_size=batch_size,
        )

        self.loss_module = DQNLoss(self.behavioural_strategy, delay_value=True)
        self.loss_module.make_value_estimator(gamma=gamma)
        self.target_updater = SoftUpdate(self.loss_module, eps=0.995)

        self.opt_sl = Adam(params=self.average_policy_network.parameters(), lr=sl_lr)
        self.opt_rl = Adam(params=self.loss_module.parameters(), lr=rl_lr)

        self._sample_episode_policy()

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        if self._mode == MODE.best_response:
            tensordict = self.behavioural_strategy(tensordict)
            self.reservoir_buffer.extend(
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
        if len(self.reservoir_buffer) < self.reservoir_buffer._storage.max_size:
            return None

        transitions = self.reservoir_buffer.sample().to(self.device)

        self.opt_sl.zero_grad()
        self.average_policy_network.train()

        action = transitions.get("action")
        eval_probs = torch.nn.functional.one_hot(
            action, num_classes=transitions.get("action_mask").shape[-1]
        )

        input = transitions.select("observation", "action_mask")
        output = self.average_policy_network(input)
        forcast_probs: torch.Tensor = output["probs"]
        log_forcast_probs = torch.log(forcast_probs)

        loss = -(eval_probs * log_forcast_probs).sum(dim=-1).mean()
        loss.backward()
        self.opt_sl.step()

        self.average_policy_network.eval()

        return loss.item()

    def train_rl(self, data: ReplayBuffer):
        self.behavioural_strategy.train()

        losses = []
        data._batch_size = self.batch_size

        for minibatch in data:
            minibatch: TensorDict = minibatch.to(self.device)
            minibatch["next", "done"] = minibatch["next", "done"].unsqueeze(-1)
            self.opt_rl.zero_grad()
            loss: torch.Tensor = self.loss_module(minibatch)["loss"]
            loss.backward()
            self.opt_rl.step()

            losses.append(loss.item())

        self.behavioural_strategy.eval()

        return sum(losses) / len(losses)


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
