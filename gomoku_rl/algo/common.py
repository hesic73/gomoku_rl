import torch
import torch.nn as nn
from torch.cuda import _device_t
from torchrl.modules import ProbabilisticActor
from torch.distributions.categorical import Categorical
from torchrl.modules.models import ConvNet, MLP
from torchrl.modules import ValueOperator
from torchrl.data import TensorSpec
from torchrl.modules import DuelingCnnDQNet, EGreedyWrapper, QValueActor

from tensordict.nn import TensorDictModule
from tensordict import TensorDict


from omegaconf import DictConfig, OmegaConf
from typing import Callable


class _Actor(nn.Module):
    def __init__(
        self, device: _device_t, n_action, cnn_kwargs: dict, mlp_kwargs: dict
    ) -> None:
        super().__init__()
        self.features = ConvNet(device=device, **cnn_kwargs)
        self.advantage = MLP(out_features=n_action, device=device, **mlp_kwargs)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.features(x)
        advantage: torch.Tensor = self.advantage(x)
        if mask is not None:
            advantage = torch.where(mask == 0, advantage, -1e10)
        probs = torch.special.softmax(advantage, dim=-1)  # (E, board_size^2)
        return probs


def make_dqn_actor(
    cfg: DictConfig,
    action_spec: TensorSpec,
    action_space_size: int,
):
    cnn_kwargs = OmegaConf.to_container(cfg.cnn_kwargs)
    cnn_kwargs.update(
        {"activation_class": getattr(nn, cnn_kwargs.get("activation_class", "ReLU"))}
    )
    mlp_kwargs = OmegaConf.to_container(cfg.mlp_kwargs)
    mlp_kwargs.update(
        {"activation_class": getattr(nn, mlp_kwargs.get("activation_class", "ReLU"))}
    )

    net = DuelingCnnDQNet(action_space_size, 1, cnn_kwargs, mlp_kwargs)
    actor = QValueActor(
        net,
        spec=action_spec,
        action_mask_key="action_mask",
    )
    return actor


def make_ppo_actor(
    cfg: DictConfig,
    action_spec: TensorSpec,
    device: _device_t,
):
    cnn_kwargs = OmegaConf.to_container(cfg.cnn_kwargs)
    cnn_kwargs.update(
        {"activation_class": getattr(nn, cnn_kwargs.get("activation_class", "ReLU"))}
    )
    mlp_kwargs = OmegaConf.to_container(cfg.mlp_kwargs)
    mlp_kwargs.update(
        {"activation_class": getattr(nn, mlp_kwargs.get("activation_class", "ReLU"))}
    )

    actor_net = _Actor(
        device=device,
        n_action=action_spec.space.n,
        cnn_kwargs=cnn_kwargs,
        mlp_kwargs=mlp_kwargs,
    )

    policy_module = TensorDictModule(
        module=actor_net, in_keys=["observation", "action_mask"], out_keys=["probs"]
    )

    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys=["probs"],
        distribution_class=Categorical,
        return_log_prob=True,
    )

    return policy_module


def make_critic(
    cfg: DictConfig,
    device: _device_t,
):
    cnn_kwargs = OmegaConf.to_container(cfg.cnn_kwargs)
    cnn_kwargs.update(
        {"activation_class": getattr(nn, cnn_kwargs.get("activation_class", "ReLU"))}
    )
    mlp_kwargs = OmegaConf.to_container(cfg.mlp_kwargs)
    mlp_kwargs.update(
        {"activation_class": getattr(nn, mlp_kwargs.get("activation_class", "ReLU"))}
    )

    value_net = nn.Sequential(
        ConvNet(device=device, **cnn_kwargs),
        MLP(out_features=1, device=device, **mlp_kwargs),
    )

    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )

    return value_module


