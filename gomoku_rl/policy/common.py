import torch
import torch.nn as nn
from torch.cuda import _device_t
from torchrl.modules import ProbabilisticActor
from torch.distributions.categorical import Categorical
from torchrl.modules.models import ConvNet, MLP
from torchrl.modules import ValueOperator
from torchrl.data import TensorSpec
from torchrl.modules import DuelingCnnDQNet, EGreedyModule, QValueActor

from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict import TensorDict


from omegaconf import DictConfig, OmegaConf
from typing import Callable

from gomoku_rl.utils.module import ValueNet, ActorNet


def _get_cnn_mlp_kwargs(cfg: DictConfig):
    cnn_kwargs = OmegaConf.to_container(cfg.cnn_kwargs)
    cnn_kwargs.update(
        {"activation_class": getattr(nn, cnn_kwargs.get("activation_class", "ReLU"))}
    )
    if (norm_class := cnn_kwargs.get("norm_class", None)) is not None:
        cnn_kwargs.update({"norm_class": getattr(nn, norm_class)})

    mlp_kwargs = OmegaConf.to_container(cfg.mlp_kwargs)
    mlp_kwargs.update(
        {"activation_class": getattr(nn, mlp_kwargs.get("activation_class", "ReLU"))}
    )
    if (norm_class := mlp_kwargs.get("norm_class", None)) is not None:
        mlp_kwargs.update({"norm_class": getattr(nn, norm_class)})

    return cnn_kwargs, mlp_kwargs


def make_dqn_actor(
    cfg: DictConfig,
    action_spec: TensorSpec,
    device: _device_t,
):
    cnn_kwargs, mlp_kwargs = _get_cnn_mlp_kwargs(cfg)

    net = DuelingCnnDQNet(action_spec.space.n, 1, cnn_kwargs, mlp_kwargs, device=device)
    actor = QValueActor(
        net,
        spec=action_spec,
        action_mask_key="action_mask",
    ).to(device)
    return actor


def make_egreedy_actor(
    cfg: DictConfig,
    action_spec: TensorSpec,
    eps_init: float = 1.0,
    eps_end: float = 0.10,
    annealing_num_steps: int = 1000,
    device: _device_t = "cuda",
):
    actor = make_dqn_actor(cfg=cfg, action_spec=action_spec, device=device)

    explorative_policy = TensorDictSequential(
        actor,
        EGreedyModule(
            spec=action_spec,
            eps_init=eps_init,
            eps_end=eps_end,
            annealing_num_steps=annealing_num_steps,
            action_mask_key="action_mask",
        ),
    )
    return explorative_policy


def make_ppo_actor(
    cfg: DictConfig,
    action_spec: TensorSpec,
    device: _device_t,
):
    actor_net = ActorNet(
        in_channels=3,
        out_features=action_spec.space.n,
        num_channels=cfg.num_channels,
        num_residual_blocks=cfg.num_residual_blocks,
    ).to(device)

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
    value_net = ValueNet(
        in_channels=3,
        num_channels=cfg.num_channels,
        num_residual_blocks=cfg.num_residual_blocks,
    ).to(device)

    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )

    return value_module


def make_dataset_naive(tensordict: TensorDict, num_minibatches: int = 8):
    tensordict = tensordict.reshape(-1)
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]
