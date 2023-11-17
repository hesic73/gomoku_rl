from typing import Callable, Dict, List, Any, Union, Iterable
from tensordict import TensorDict
from tensordict.utils import NestedKey
import torch
import torch.nn as nn

from tensordict.nn import TensorDictModule, make_functional

from torchrl.data import TensorSpec, DiscreteTensorSpec
from torch.cuda import _device_t


from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm

import logging

from torchrl.modules import ProbabilisticActor

from torch.distributions.categorical import Categorical

from torchrl.modules.models import ConvNet, MLP
from torchrl.modules import ValueOperator

from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from torchrl.data.replay_buffers import ReplayBuffer

from .base import Policy
from .common import make_ppo_actor, make_critic


class PPOPolicy(Policy):
    def __init__(
        self,
        cfg: DictConfig,
        action_spec: DiscreteTensorSpec,
        observation_spec: TensorSpec,
        device: _device_t = "cuda",
    ) -> None:
        super().__init__(cfg, action_spec, observation_spec, device)
        self.cfg: DictConfig = cfg
        self.device: _device_t = device

        self.clip_param: float = cfg.clip_param
        self.ppo_epoch: int = int(cfg.ppo_epochs)

        self.entropy_coef: float = cfg.entropy_coef
        self.gae_gamma: float = cfg.gamma
        self.gae_lambda: float = cfg.gae_lambda

        self.max_grad_norm: float = cfg.max_grad_norm

        self.actor = make_ppo_actor(
            cfg=cfg.actor, action_spec=action_spec, device=self.device
        )
        self.critic = make_critic(cfg=cfg.critic, device=self.device)

        fake_input = observation_spec.zero()
        fake_input["action_mask"] = ~fake_input["action_mask"]
        self.actor(fake_input)
        self.critic(fake_input)

        self.advantage_module = GAE(
            gamma=self.gae_gamma,
            lmbda=self.gae_lambda,
            value_network=self.critic,
            average_gae=True,
        )

        self.loss_module = ClipPPOLoss(
            actor=self.actor,
            critic=self.critic,
            clip_epsilon=self.clip_param,
            entropy_bonus=bool(self.entropy_coef),
            entropy_coef=self.entropy_coef,
            # these keys match by default but we set this for completeness
            value_target_key=self.advantage_module.value_target_key,
            loss_critic_type="smooth_l1",
        )

        self.optim = torch.optim.Adam(self.loss_module.parameters(), cfg.lr)

    def __call__(self, tensordict: TensorDict):
        actor_input = tensordict.select("observation", "action_mask", strict=False)
        actor_output: TensorDict = self.actor(actor_input)
        actor_output = actor_output.exclude("probs")
        tensordict.update(actor_output)

        # critic_input = tensordict.select("observation")
        # critic_output = self.critic(critic_input)
        # tensordict.update(critic_output)

        return tensordict

    def learn(self, data: ReplayBuffer):
        self.actor.train()
        self.critic.train()
        losses = []
        for _ in range(self.ppo_epoch):
            for minibatch in data:
                minibatch = minibatch.to(self.device)
                with torch.no_grad():
                    self.advantage_module(minibatch)
                loss_vals = self.loss_module(minibatch)
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                losses.append(loss_value.clone().detach())
                # Optimization: backward, grad clipping and optim step
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(
                    self.loss_module.parameters(), self.max_grad_norm
                )
                self.optim.step()
                self.optim.zero_grad()

        self.critic.eval()
        self.actor.eval()
        return {
            "loss": torch.stack(losses).mean().item(),
        }

    def state_dict(self) -> Dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.advantage_module = GAE(
            gamma=self.gae_gamma,
            lmbda=self.gae_lambda,
            value_network=self.critic,
            average_gae=True,
        )

        self.loss_module = ClipPPOLoss(
            actor=self.actor,
            critic=self.critic,
            clip_epsilon=self.clip_param,
            entropy_bonus=bool(self.entropy_coef),
            entropy_coef=self.entropy_coef,
            # these keys match by default but we set this for completeness
            value_target_key=self.advantage_module.value_target_key,
            loss_critic_type="smooth_l1",
        )

        self.optim = torch.optim.Adam(self.loss_module.parameters(), self.cfg.lr)

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()
