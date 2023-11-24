from typing import Any
import numpy as np
from .policy import _policy_t, uniform_policy
from tensordict.nn import TensorDictModule, set_interaction_type, InteractionType
from tensordict import TensorDict
from torch.cuda import _device_t


class ConvergedIndicator:
    def __init__(
        self,
        max_size: int = 10,
        mean_threshold: float = 0.99,
        std_threshold: float = 0.005,
        min_iter_steps: int = 10,
        max_iter_steps: int = 100,
    ) -> None:
        self.win_rates = []
        self.max_size = max_size
        self.mean_threshold = mean_threshold
        self.std_threshold = std_threshold
        self.min_iter_steps = min_iter_steps
        self.max_iter_steps = max_iter_steps
        self._step_cnt = 0

    def update(self, value: float):
        self.win_rates.append(value)
        self._step_cnt += 1
        if len(self.win_rates) > self.max_size:
            self.win_rates.pop(0)

    def reset(self):
        self.win_rates = []
        self._step_cnt = 0

    def converged(self) -> bool:
        if len(self.win_rates) < self.max_size:
            return False

        if self._step_cnt < self.min_iter_steps:
            return False
        if self._step_cnt > self.max_iter_steps:
            return True

        mean = np.mean(self.win_rates)
        std = np.std(self.win_rates)
        # print(f"Recent Win Rate: {mean:.2f}/{std:.4f}")
        return mean >= self.mean_threshold and std <= self.std_threshold


class Population:
    def __init__(
        self,
        initial_policy: _policy_t = uniform_policy,
        device: _device_t = "cuda",
    ):
        self.device = device
        self.policy_sets: list[_policy_t] = [initial_policy]
        self._idx = 0

    def add(self, policy: TensorDictModule):
        self.policy_sets.append(policy)

    def sample(self, meta_policy: np.ndarray | None = None):
        self._idx = np.random.choice(len(self.policy_sets), p=meta_policy)

    @set_interaction_type(type=InteractionType.MEAN)
    def __call__(self, tensordict: TensorDict) -> TensorDict:
        tensordict = tensordict.to(self.device)
        return self.policy_sets[self._idx](tensordict)
