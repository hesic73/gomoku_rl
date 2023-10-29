from collections import defaultdict
from typing import Any, Dict, Optional, Sequence, Union, Tuple,Callable

from tensordict.utils import NestedKey

import torch
from tensordict.tensordict import TensorDictBase, TensorDict
from torchrl.data.tensor_specs import TensorSpec
from torchrl.envs.common import EnvBase
from torchrl.envs.transforms import (
    TransformedEnv,
    Transform,
    Compose,
    FlattenObservation,
    CatTensors,
)

from dataclasses import replace



import numpy as np
import os
import logging



class LogOnEpisode(Transform):
    def __init__(
        self,
        n_episodes: int,
        in_keys: Sequence[str] = None,
        log_keys: Sequence[str] = None,
        logger_func: Callable = None,
        process_func: Dict[str, Callable] = None,
    ):
        super().__init__(in_keys=in_keys)
        if not len(in_keys) == len(log_keys):
            raise ValueError
        self.in_keys = in_keys
        self.log_keys = log_keys

        self.n_episodes = n_episodes
        self.logger_func = logger_func
        self.process_func = defaultdict(lambda: lambda x: torch.mean(x.float()).item())
        if process_func is not None:
            self.process_func.update(process_func)

        self.stats = []
        self._frames = 0

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict

    def _step(self, tensordict: TensorDictBase,next_tensordict: TensorDictBase) -> TensorDictBase:
        _reset = next_tensordict.get("done", None)
        if _reset is None:
            _reset = torch.zeros(
                tensordict.batch_size, dtype=torch.bool, device=tensordict.device
            )
        if _reset.any():
            _reset = _reset.all(-1).cpu()
            rst_tensordict = next_tensordict.select(*self.in_keys).cpu()
            self.stats.extend(rst_tensordict[_reset].unbind(0))
            if len(self.stats) >= self.n_episodes:
                stats: TensorDictBase = torch.stack(self.stats)
                dict_to_log = {}
                for in_key, log_key in zip(self.in_keys, self.log_keys):
                    try:
                        process_func = self.process_func[log_key]
                        if isinstance(log_key, tuple):
                            log_key = ".".join(log_key)
                        dict_to_log[log_key] = process_func(stats[in_key])
                    except:
                        pass

                # skip None
                if self.training:
                    dict_to_log = {
                        f"train/{k}": v for k, v in dict_to_log.items() if v is not None
                    }
                else:
                    dict_to_log = {
                        f"eval/{k}": v for k, v in dict_to_log.items() if v is not None
                    }

                if self.logger_func is not None:
                    dict_to_log["env_frames"] = self._frames
                    self.logger_func(dict_to_log)
                self.stats.clear()

        if self.training:
            self._frames += tensordict.numel()
        return next_tensordict


