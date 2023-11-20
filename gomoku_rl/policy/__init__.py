from .base import Policy
from .ppo import PPOPolicy
from .random import RandomPolicy

from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.tensor_specs import DiscreteTensorSpec, TensorSpec
from omegaconf import DictConfig
from torch.cuda import _device_t
import torch


def get_policy(
    name: str,
    cfg: DictConfig,
    action_spec: DiscreteTensorSpec,
    observation_spec: TensorSpec,
    device: _device_t = "cuda",
) -> Policy:
    policies = {
        "ppo": PPOPolicy,
        "random": RandomPolicy,
    }
    assert name.lower() in policies
    cls = policies[name.lower()]
    return cls(
        cfg=cfg,
        action_spec=action_spec,
        observation_spec=observation_spec,
        device=device,
    )


def get_pretrained_policy(
    name: str,
    cfg: DictConfig,
    action_spec: DiscreteTensorSpec,
    observation_spec: TensorSpec,
    checkpoint_path: str,
    device: _device_t = "cuda",
) -> Policy:
    policy = get_policy(
        name=name,
        cfg=cfg,
        action_spec=action_spec,
        observation_spec=observation_spec,
        device=device,
    )
    policy.load_state_dict(torch.load(checkpoint_path))
    return policy
