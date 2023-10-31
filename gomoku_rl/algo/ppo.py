from typing import Callable, Dict, List, Any
from tensordict import TensorDictBase, TensorDict
from tensordict.utils import NestedKey
import torch
import torch.nn as nn

from tensordict.nn import TensorDictModule
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.modules import DuelingCnnDQNet

from torchrl.data import TensorSpec, DiscreteTensorSpec
from torch.cuda import _device_t
from torchrl.envs.utils import ExplorationType
from torchrl.envs import TransformedEnv
import datetime
import tempfile

from torchrl.collectors import SyncDataCollector
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm

from gomoku_rl.utils.misc import get_checkpoint_dir
import os
from math import isqrt
import logging

from torchrl.modules import ProbabilisticActor

from torch.distributions.categorical import Categorical

from torchrl.modules.models import ConvNet,MLP


class Obs2Logits(nn.Module):
    def __init__(self,device:_device_t,n_action,cnn_kwargs:dict,mlp_kwargs:dict) -> None:
        super().__init__(device=device)
        self.features=ConvNet(device=device,**cnn_kwargs)
        self.advantage=MLP(out_features=n_action, device=device, **mlp_kwargs)
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = self.features(x)
        advantage = self.advantage(x)
        logits=torch.special.logit(advantage)
        return logits


def make_actor(
    cfg: DictConfig,
    action_spec: TensorSpec,
    observation_key: NestedKey,
    action_space_size: int,
    device:_device_t,
):
    cnn_kwargs = OmegaConf.to_container(cfg.cnn_kwargs)
    cnn_kwargs.update(
        {"activation_class": getattr(nn, cnn_kwargs.get("activation_class", "ReLU"))}
    )
    mlp_kwargs = OmegaConf.to_container(cfg.mlp_kwargs)
    mlp_kwargs.update(
        {"activation_class": getattr(nn, mlp_kwargs.get("activation_class", "ReLU"))}
    )
    
    features=ConvNet(device=device,**cnn_kwargs)

    net = DuelingCnnDQNet(action_space_size, 1, cnn_kwargs, mlp_kwargs)
    
    policy_module=TensorDictModule(module=nn.Sequential(net,))
    
    return actor
    