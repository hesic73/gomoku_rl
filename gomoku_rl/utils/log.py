from typing import Callable, Dict, Any
from tensordict import TensorDict
import torch
from collections import defaultdict


class Mean:
    def __init__(self) -> None:
        self._cnt = 0
        self._acc = 0.0

    def update(self, t: torch.Tensor) -> None:
        self._acc += t.sum().item()
        self._cnt += t.numel()

    @property
    def value(self) -> float:
        return self._acc / self._cnt


def get_log_func(info: Dict):
    meters = defaultdict(lambda: Mean())

    def _fun(tensordict: TensorDict) -> None:
        done: torch.Tensor = tensordict.get("done")
        if not done.any():
            return

        stats_td: TensorDict = tensordict.get("stats")[done]
        for key in stats_td.keys():
            meters[key].update(stats_td[key])
            info[key] = meters[key].value

    return _fun
