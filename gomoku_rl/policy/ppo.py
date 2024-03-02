from typing import Callable, Dict, List, Any, Union, Iterable
from tensordict import TensorDict
import torch
from torchrl.data import TensorSpec, DiscreteTensorSpec
from torch.cuda import _device_t
from omegaconf import DictConfig, OmegaConf
import logging
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.functional import vec_generalized_advantage_estimate
from .base import Policy
from .common import (
    make_dataset_naive,
    make_ppo_ac,
    get_optimizer,
    make_critic,
    make_ppo_actor,
)
from gomoku_rl.utils.module import (
    count_parameters,
)


class PPO(Policy):
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
        self.average_gae: float = cfg.average_gae
        self.batch_size: int = int(cfg.batch_size)

        self.max_grad_norm: float = cfg.max_grad_norm
        if self.cfg.get("share_network"):
            actor_value_operator = make_ppo_ac(
                cfg, action_spec=action_spec, device=self.device
            )
            self.actor = actor_value_operator.get_policy_operator()
            self.critic = actor_value_operator.get_value_head()
        else:
            self.actor = make_ppo_actor(
                cfg=cfg, action_spec=action_spec, device=self.device
            )
            self.critic = make_critic(cfg=cfg, device=self.device)

        fake_input = observation_spec.zero()
        fake_input["action_mask"] = ~fake_input["action_mask"]
        with torch.no_grad():
            self.actor(fake_input)
            self.critic(fake_input)
        # print(f"actor params:{count_parameters(self.actor)}")
        # print(f"critic params:{count_parameters(self.critic)}")

        self.loss_module = ClipPPOLoss(
            actor=self.actor,
            critic=self.critic,
            clip_epsilon=self.clip_param,
            entropy_bonus=bool(self.entropy_coef),
            entropy_coef=self.entropy_coef,
            normalize_advantage=self.cfg.get("normalize_advantage", True),
            loss_critic_type="smooth_l1",
        )

        self.optim = get_optimizer(self.cfg.optimizer, self.loss_module.parameters())

    def __call__(self, tensordict: TensorDict):
        tensordict=tensordict.to(self.device)
        actor_input = tensordict.select("observation", "action_mask", strict=False)
        actor_output: TensorDict = self.actor(actor_input)
        actor_output = actor_output.exclude("probs")
        tensordict.update(actor_output)

        # share_network=True, use `hidden` as input
        # share_network=False, use `observation` as input
        critic_input = tensordict.select("hidden", "observation", strict=False)
        critic_output = self.critic(critic_input)
        tensordict.update(critic_output)

        return tensordict

    def learn(self, data: TensorDict):
        # to do: compute the gae for each batch
        value = data["state_value"].to(self.device)
        next_value = data["next", "state_value"].to(self.device)
        done = data["next", "done"].unsqueeze(-1).to(self.device)
        reward = data["next", "reward"].to(self.device)
        with torch.no_grad():
            adv, value_target = vec_generalized_advantage_estimate(
                self.gae_gamma,
                self.gae_lambda,
                value,
                next_value,
                reward,
                done=done,
                terminated=done,
                time_dim=data.ndim - 1,
            )
            loc = adv.mean()
            scale = adv.std().clamp_min(1e-4)
            if self.average_gae:
                adv = adv - loc
                adv = adv / scale

            data.set("advantage", adv)
            data.set("value_target", value_target)

        # filter out invalid white transitions
        invalid = data.get("invalid", None)
        if invalid is not None:
            data = data[~invalid]

        data=data.reshape(-1)

        self.train()
        loss_objectives = []
        loss_critics = []
        loss_entropies = []
        losses = []
        grad_norms = []
        for _ in range(self.ppo_epoch):
            for minibatch in make_dataset_naive(
                data, batch_size=self.batch_size
            ):
                minibatch: TensorDict = minibatch.to(self.device)
                loss_vals = self.loss_module(minibatch)
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                loss_objectives.append(loss_vals["loss_objective"].clone().detach())
                loss_critics.append(loss_vals["loss_critic"].clone().detach())
                loss_entropies.append(loss_vals["loss_entropy"].clone().detach())
                losses.append(loss_value.clone().detach())
                # Optimization: backward, grad clipping and optim step
                loss_value.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.loss_module.parameters(), self.max_grad_norm
                )
                grad_norms.append(grad_norm.clone().detach())
                self.optim.step()
                self.optim.zero_grad()

        self.eval()
        return {
            "advantage_meam": loc.item(),
            "advantage_std": scale.item(),
            "grad_norm": torch.stack(grad_norms).mean().item(),
            "loss": torch.stack(losses).mean().item(),
            "loss_objective": torch.stack(loss_objectives).mean().item(),
            "loss_critic": torch.stack(loss_critics).mean().item(),
            "loss_entropy": torch.stack(loss_entropies).mean().item(),
        }

    def state_dict(self) -> Dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict):
        self.critic.load_state_dict(state_dict["critic"], strict=False)
        self.actor.load_state_dict(state_dict["actor"])

        self.loss_module = ClipPPOLoss(
            actor=self.actor,
            critic=self.critic,
            clip_epsilon=self.clip_param,
            entropy_bonus=bool(self.entropy_coef),
            entropy_coef=self.entropy_coef,
            normalize_advantage=self.cfg.get("normalize_advantage", True),
            loss_critic_type="smooth_l1",
        )

        self.optim = get_optimizer(self.cfg.optimizer, self.loss_module.parameters())

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()
