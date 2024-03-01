import os
import logging
import torch
from typing import Optional, Dict
import random
import numpy as np
from tensordict import TensorDict
from omegaconf import DictConfig


def add_prefix(d: Dict, prefix: str):
    return {prefix + k: v for k, v in d.items()}


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_kwargs(cfg: DictConfig, *names):
    kwargs = {}
    for name in names:
        if param := cfg.get(name, None):
            kwargs.update(
                {
                    name: param,
                }
            )

    return kwargs
