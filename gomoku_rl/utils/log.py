from typing import Callable, Dict, Any
from tensordict import TensorDict
import torch

class Mean:
    def __init__(self) -> None:
        self._cnt = 0
        self._acc = 0.0

    def update(self, val: float) -> None:
        self._acc += val
        self._cnt += 1

    @property
    def value(self) -> float:
        return self._acc / self._cnt


def get_log_func(info: Dict):
    def _fun(tensordict: TensorDict) -> None:
        done:torch.Tensor = tensordict.get("done")
        if not done.any():
            return
        
        
        stats_td=tensordict.get('stats')[done]
        print(done.sum())

    return _fun
