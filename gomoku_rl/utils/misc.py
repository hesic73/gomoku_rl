import os
import logging
import torch
from typing import Optional, Dict
import random
import numpy as np
from tensordict import TensorDict


def add_prefix(d: Dict, prefix: str):
    return {prefix + k: v for k, v in d.items()}


def no_nan_in_tensordict(tensordict: TensorDict):
    for n, t in tensordict.items(include_nested=True, leaves_only=True):
        if not isinstance(t, torch.Tensor):
            continue
        if torch.isnan(t).any():
            return False
    return True


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
