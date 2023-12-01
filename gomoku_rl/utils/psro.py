import numpy as np
from .policy import _policy_t, uniform_policy
from .eval import eval_win_rate
from tensordict.nn import TensorDictModule, set_interaction_type, InteractionType
from tensordict import TensorDict
from torch.cuda import _device_t
from gomoku_rl.policy import Policy
import copy
import contextlib
import nashpy
import os
import torch
import logging


class ConvergedIndicator:
    def __init__(
        self,
        max_size: int = 15,
        mean_threshold: float = 0.99,
        std_threshold: float = 0.005,
        min_iter_steps: int = 20,
        max_iter_steps: int = 300,
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
        dir: str,
        initial_policy: _policy_t = uniform_policy,
        device: _device_t = "cuda",
    ):
        self.dir = dir
        os.makedirs(self.dir, exist_ok=True)
        self._module_cnt = 0
        self._module = None  # assume all modules are homogeneous
        # used in `make_behavioural_strategy` to avoid calling 'copy.deepcopy' multiple times
        self._module_backup = None
        self._idx = -1
        self.device = device
        # this should be deterministic, as PSRO requires pure strategies. But it seems it easily overfits
        self._interaction_type = InteractionType.MODE
        # self._interaction_type = InteractionType.RANDOM

        self.policy_sets: list[_policy_t | int] = []
        # if it's a module, we save it on disk
        if isinstance(initial_policy, (TensorDictModule, Policy)):
            self.add(initial_policy)
        else:
            self.policy_sets.append(initial_policy)

        self._func = None

        self.sample()

    def __len__(self) -> int:
        return len(self.policy_sets)

    def add(self, policy: TensorDictModule):
        if self._module is None:
            self._module = policy  # assume `policy` is a copy
        torch.save(
            policy.state_dict(),
            os.path.join(self.dir, f"{self._module_cnt}.pt"),
        )
        self.policy_sets.append(self._module_cnt)
        self._module_cnt += 1

    def _set_policy(self, index: int):
        if self._idx == index:
            return

        self._idx = index
        if not isinstance(self.policy_sets[index], int):
            self._func = self.policy_sets[index]
        else:
            assert self._module is not None
            self._module.load_state_dict(
                torch.load(os.path.join(self.dir, f"{self.policy_sets[index]}.pt"))
            )
            self._module.eval()
            self._func = self._module

    def sample(self, meta_policy: np.ndarray | None = None):
        self._set_policy(np.random.choice(len(self.policy_sets), p=meta_policy))

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        tensordict = tensordict.to(self.device)
        with set_interaction_type(type=self._interaction_type):
            return self._func(tensordict)

    @contextlib.contextmanager
    def fixed_behavioural_strategy(self, index: int):
        _idx = self._idx
        self._idx = index
        self._set_policy(self._idx)
        _interaction_type = self._interaction_type
        self._interaction_type = InteractionType.RANDOM
        yield
        self._idx = _idx
        self._set_policy(self._idx)
        self._interaction_type = _interaction_type

    def make_behavioural_strategy(self, index: int) -> _policy_t:
        """
        **share _module_backup!!!**
        ```
        s1=population.make_behavioural_strategy(0)
        s2=population.make_behavioural_strategy(1)
        then s1 and s2 are the same strategy!!!
        ```
        """
        if not isinstance(self.policy_sets[index], int):
            return self.policy_sets[index]

        if self._module_backup is None:
            self._module_backup = copy.deepcopy(self._module)

        self._module_backup.load_state_dict(
            torch.load(os.path.join(self.dir, f"{self.policy_sets[index]}.pt"))
        )
        self._module_backup.eval()

        def _strategy(tensordict: TensorDict) -> TensorDict:
            tensordict = tensordict.to(self.device)
            with set_interaction_type(type=InteractionType.RANDOM):
                return self._module_backup(tensordict)

        return _strategy


class PSROPolicyWrapper:
    def __init__(self, policy: Policy, dir: str, device: _device_t):
        self.policy = policy
        actor = copy.deepcopy(policy)
        actor.eval()
        self.population = Population(initial_policy=actor, dir=dir, device=device)
        self.meta_policy = None
        self._oracle_mode = True
        self._cnt = 0

    def set_meta_policy(self, meta_policy: np.ndarray):
        assert len(meta_policy) == len(self.population)
        self.meta_policy = meta_policy

    def set_oracle_mode(self, value: bool = True):
        self._oracle_mode = value

    def sample(self):
        assert not self._oracle_mode
        self.population.sample(meta_policy=self.meta_policy)

    def add_current_policy(self):
        actor = copy.deepcopy(self.policy)
        actor.eval()
        self.population.add(actor)
        self._cnt += 1

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        if self._oracle_mode:
            return self.policy(tensordict)
        else:
            return self.population(tensordict)

    def eval(self):
        if self._oracle_mode:
            # strategies in the policy set are always in eval mode
            self.policy.eval()

    def train(self):
        if self._oracle_mode:
            self.policy.train()


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
            and old_payoffs.shape[0] + 1 == n
        )
    new_payoffs = np.zeros(shape=(n, n))
    if old_payoffs is not None:
        new_payoffs[:-1, :-1] = old_payoffs

    for i in range(n):
        with population_0.fixed_behavioural_strategy(index=n - 1):
            with population_1.fixed_behavioural_strategy(index=i):
                wr = eval_win_rate(
                    env=env,
                    player_black=population_0,
                    player_white=population_1,
                    n=2,
                )
        new_payoffs[-1, i] = 2 * wr - 1

    for i in range(n - 1):
        with population_0.fixed_behavioural_strategy(index=i):
            with population_1.fixed_behavioural_strategy(index=n - 1):
                wr = eval_win_rate(
                    env=env,
                    player_black=population_0,
                    player_white=population_1,
                    n=2,
                )
        new_payoffs[i, -1] = 2 * wr - 1
    return new_payoffs


def get_new_payoffs_sp(
    env,
    population: Population,
    old_payoffs: np.ndarray | None,
):
    n = len(population)
    if old_payoffs is not None:
        assert (
            len(old_payoffs.shape) == 2
            and old_payoffs.shape[0] == old_payoffs.shape[1]
            and old_payoffs.shape[0] + 1 == n
        )
    new_payoffs = np.zeros(shape=(n, n))
    if old_payoffs is not None:
        new_payoffs[:-1, :-1] = old_payoffs

    for i in range(n - 1):
        with population.fixed_behavioural_strategy(index=n - 1):
            player_i = population.make_behavioural_strategy(index=i)
            wr_1 = eval_win_rate(
                env=env,
                player_black=player_i,
                player_white=population,
                n=2,
            )
            wr_2 = 1 - eval_win_rate(
                env=env,
                player_black=population,
                player_white=player_i,
                n=2,
            )

        # the policy has 50% chance to play black and 50% chance to play white
        # so the utility for it is 0.5*(win_rate_black+ win_rate_white)
        # we transform it so that it's zero-sum
        new_payoffs[i, -1] = wr_1 + wr_2 - 1
        new_payoffs[-1, i] = -new_payoffs[i, -1]

    # for i in range(n):
    #     with population.fixed_behavioural_strategy(index=n - 1):
    #         player_i = population.make_behavioural_strategy(index=i)
    #         wr = eval_win_rate(
    #             env=env,
    #             player_black=player_i,
    #             player_white=population,
    #             n=2,
    #         )
    #     new_payoffs[i, -1] = 2 * wr - 1

    # for i in range(n - 1):
    #     with population.fixed_behavioural_strategy(index=n - 1):
    #         player_i = population.make_behavioural_strategy(index=i)
    #         wr = eval_win_rate(
    #             env=env,
    #             player_black=population,
    #             player_white=player_i,
    #             n=2,
    #         )
    #     new_payoffs[-1, i] = 2 * wr - 1

    return new_payoffs


def print_payoffs(payoffs: np.ndarray):
    print(
        "payoffs:\n"
        + "\n".join(["\t".join([f"{item:+.3f}" for item in line]) for line in payoffs])
    )


def solve_nash(payoffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    game = nashpy.Game(payoffs)
    # print(game)
    eqs = game.support_enumeration()
    try:
        return list(eqs)[0]
    except IndexError:
        logging.error("solve_nash failed.")


def solve_uniform(payoffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.ones(shape=payoffs.shape[0]) / payoffs.shape[0],
        np.ones(shape=payoffs.shape[1]) / payoffs.shape[1],
    )


def get_meta_solver(name: str):
    tmp = {
        "uniform": solve_uniform,
        "nash": solve_nash,
    }
    name = name.lower()
    assert name in tmp
    return tmp[name]


def calculate_jpc(payoffs: np.ndarray):
    assert len(payoffs.shape) == 2 and payoffs.shape[0] == payoffs.shape[1]
    n = payoffs.shape[0]
    assert n > 1
    d = np.trace(payoffs) / n
    o = (np.sum(payoffs) - n * d) / (n * (n - 1))
    r = (d - o) / d
    return r
