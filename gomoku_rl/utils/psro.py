from typing import Any
import numpy as np
from .policy import _policy_t, uniform_policy
from .eval import eval_win_rate
from tensordict.nn import TensorDictModule, set_interaction_type, InteractionType
from tensordict import TensorDict
from torch.cuda import _device_t
from gomoku_rl.policy import Policy
import copy
import contextlib


class ConvergedIndicator:
    def __init__(
        self,
        max_size: int = 10,
        mean_threshold: float = 0.99,
        std_threshold: float = 0.005,
        min_iter_steps: int = 10,
        max_iter_steps: int = 150,
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


# to do: save to cpu/disk
class Population:
    def __init__(
        self,
        initial_policy: _policy_t = uniform_policy,
        device: _device_t = "cuda",
    ):
        self.device = device
        self.policy_sets: list[_policy_t] = [initial_policy]
        self._idx = 0

    def __len__(self) -> int:
        return len(self.policy_sets)

    def add(self, policy: TensorDictModule):
        self.policy_sets.append(policy)

    def sample(self, meta_policy: np.ndarray | None = None):
        self._idx = np.random.choice(len(self.policy_sets), p=meta_policy)

    # @set_interaction_type(type=InteractionType.MODE)
    def __call__(self, tensordict: TensorDict) -> TensorDict:
        tensordict = tensordict.to(self.device)
        return self.policy_sets[self._idx](tensordict)

    @contextlib.contextmanager
    def pure_strategy(self, index: int):
        _idx = self._idx
        yield
        self._idx = _idx


class PSROPolicyWrapper:
    def __init__(self, policy: Policy, device: _device_t):
        self.policy = policy
        actor = copy.deepcopy(policy.actor)
        actor.eval()
        self.population = Population(initial_policy=actor, device=device)
        # self.meta_policy = np.ones(shape=1)
        self._oracle_mode = True
        self._cnt = 0

    def set_oracle_mode(self, value: bool = True):
        self._oracle_mode = value

    def sample(self):
        assert not self._oracle_mode
        self.population.sample()

    def add_current_policy(self):
        actor = copy.deepcopy(self.policy.actor)
        actor.eval()
        self.population.add(actor)
        self._cnt += 1

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        if self._oracle_mode:
            return self.policy(tensordict)
        else:
            return self.population(tensordict)


def get_new_payoffs(
    env,
    population_0: Population,
    population_1: Population,
    old_payoffs: np.ndarray | None,
):
    assert len(population_0) == len(population_1)
    n = len(population_0)
    if old_payoffs is not None:
        assert (
            len(old_payoffs.shape) == 2
            and old_payoffs.shape[0] == old_payoffs.shape[1]
            and old_payoffs.shape[0] == n
        )
    new_payoffs = np.zeros(shape=(n, n))
    if old_payoffs is not None:
        new_payoffs[:-1, :-1] = old_payoffs
    for i in range(n):
        with population_0.pure_strategy(index=n - 1):
            with population_1.pure_strategy(index=i):
                wr = eval_win_rate(
                    env=env, player_black=population_0, player_white=population_1
                )
        new_payoffs[-1, i] = 2 * wr - 1

    for i in range(n - 1):
        with population_0.pure_strategy(index=i):
            with population_1.pure_strategy(index=n - 1):
                wr = eval_win_rate(
                    env=env, player_black=population_0, player_white=population_1
                )
        new_payoffs[i, -1] = 2 * wr - 1

    print(new_payoffs)
    return new_payoffs
