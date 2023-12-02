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
from .common import make_ppo_actor, make_critic, make_dataset_naive
from gomoku_rl.utils.module import count_parameters


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
        self.average_gae: float = cfg.average_gae

        self.max_grad_norm: float = cfg.max_grad_norm

        self.actor = make_ppo_actor(
            cfg=cfg.actor, action_spec=action_spec, device=self.device
        )
        self.critic = make_critic(cfg=cfg.critic, device=self.device)

        fake_input = observation_spec.zero()
        fake_input["action_mask"] = ~fake_input["action_mask"]
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

        self.optim = torch.optim.Adam(self.loss_module.parameters(), cfg.lr)

    def __call__(self, tensordict: TensorDict):
        actor_input = tensordict.select("observation", "action_mask", strict=False)
        actor_output: TensorDict = self.actor(actor_input)
        actor_output = actor_output.exclude("probs")
        tensordict.update(actor_output)

        critic_input = tensordict.select("observation")
        critic_output = self.critic(critic_input)
        tensordict.update(critic_output)

        return tensordict

    def learn(self, data: TensorDict):
        
        # to do: compute the gae for each batch
        value = data["state_value"]
        next_value = data["next", "state_value"]
        done = data["next", "done"].unsqueeze(-1)
        reward = data["next", "reward"]
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
            if self.average_gae:
                loc = adv.mean()
                scale = adv.std().clamp_min(1e-4)
                adv = adv - loc
                adv = adv / scale

            data.set("advantage", adv)
            data.set("value_target", value_target)

        self.train()
        loss_objectives = []
        loss_critics = []
        loss_entropies = []
        losses = []
        for _ in range(self.ppo_epoch):
            for minibatch in make_dataset_naive(data, num_minibatches=8):
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
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(
                    self.loss_module.parameters(), self.max_grad_norm
                )
                self.optim.step()
                self.optim.zero_grad()

        self.eval()
        return {
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
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])

        self.loss_module = ClipPPOLoss(
            actor=self.actor,
            critic=self.critic,
            clip_epsilon=self.clip_param,
            entropy_bonus=bool(self.entropy_coef),
            entropy_coef=self.entropy_coef,
            normalize_advantage=self.cfg.get("normalize_advantage", True),
            loss_critic_type="smooth_l1",
        )

        self.optim = torch.optim.Adam(self.loss_module.parameters(), self.cfg.lr)

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()
