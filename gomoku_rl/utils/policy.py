import torch
from torch.distributions import Categorical
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from typing import Callable

_policy_t = Callable[
    [
        TensorDict,
    ],
    TensorDict,
]


def _uniform_policy_with_mask(action_mask: torch.Tensor):
    probs = torch.zeros_like(action_mask, dtype=torch.float)
    probs = torch.where(action_mask == 1, probs, -float('inf'))
    probs = torch.special.softmax(probs, dim=-1)
    dist = Categorical(probs=probs)
    return dist.sample()


def uniform_policy(tensordict: TensorDict) -> TensorDict:
    action_mask = tensordict.get("action_mask", None)
    if action_mask is None:
        # not tested
        obs: torch.Tensor = tensordict.get("observation")
        assert len(obs.shape) == 4 and obs.shape[-2] == obs.shape[-1]
        num_envs = obs.shape[0]
        board_size = obs.shape[-1]
        action = torch.randint(
            low=0, high=board_size * board_size, size=(num_envs,), device=obs.device
        )
    else:
        action = _uniform_policy_with_mask(action_mask=action_mask)
    tensordict.update({"action": action})
    return tensordict
