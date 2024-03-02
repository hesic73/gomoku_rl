from omegaconf import DictConfig
from torch.cuda import _device_t
from torchrl.data.tensor_specs import DiscreteTensorSpec, TensorSpec
from .base import Policy

from .common import make_dqn_actor, get_optimizer, make_egreedy_actor

from typing import Callable, Dict, List, Any
from tensordict import TensorDictBase, TensorDict
import torch
import torch.nn as nn

from tensordict.nn import TensorDictModule
from torchrl.objectives import DQNLoss, SoftUpdate

from torchrl.data import TensorSpec, DiscreteTensorSpec
from torch.cuda import _device_t


from torchrl.data.replay_buffers.samplers import RandomSampler, Sampler
from omegaconf import DictConfig

from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage


def get_replay_buffer(
    buffer_size: int,
    batch_size: int,
    sampler: Sampler | None = None,
    device: _device_t = None,
):
    storage = LazyTensorStorage(max_size=buffer_size, device=device)
    buffer = TensorDictReplayBuffer(
        storage=storage, batch_size=batch_size, sampler=sampler
    )
    return buffer


class DQN(Policy):
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

        self.actor = make_dqn_actor(cfg, action_spec, device)
        fake_input = observation_spec.zero()
        fake_input["action_mask"] = ~fake_input["action_mask"]
        with torch.no_grad():
            self.actor(fake_input)

        self.actor_explore = make_egreedy_actor(
            actor=self.actor,
            action_spec=action_spec,
            annealing_num_steps=cfg.annealing_num_steps,
            eps_init=cfg.eps_init,
            eps_end=cfg.eps_end,
        )

        self.max_grad_norm: float = cfg.max_grad_norm
        self.gamma = cfg.gamma
        self.batch_size: int = cfg.batch_size
        self.buffer_size: int = cfg.buffer_size
        # Optimization steps per batch collected (aka UPD or updates per data)
        self.n_optim: int = cfg.n_optim
        self.target_update_interval: int = cfg.target_update_interval

        self.loss_module = DQNLoss(self.actor, delay_value=True)
        self.loss_module.make_value_estimator(gamma=self.gamma)
        self.target_updater = SoftUpdate(self.loss_module, eps=0.995)

        self.optimizer = get_optimizer(
            self.cfg.optimizer, self.loss_module.parameters()
        )

        self.replay_buffer = get_replay_buffer(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            sampler=RandomSampler(),
            device=cfg.buffer_device,
        )

        self._eval = False

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        if self._eval:
            return self.actor(tensordict)
        else:
            tensordict = self.actor_explore(tensordict)
            self.actor_explore.module[-1].step()
            # print(self.actor_explore.module[-1].eps)
            return tensordict

    def learn(self, data: TensorDict) -> Dict:
        invalid = data.get("invalid", None)
        if invalid is not None:
            data = data[~invalid]
        data["next", "done"] = data["next", "done"].unsqueeze(-1)
        self.replay_buffer.extend(data.reshape(-1))
        if len(self.replay_buffer) < self.buffer_size:
            return {}

        losses = []
        grad_norms = []

        for gradient_step in range(1, self.n_optim + 1):
            transition = self.replay_buffer.sample().to(self.device)
            loss: torch.Tensor = self.loss_module(transition)["loss"]
            losses.append(loss.clone().detach())
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.loss_module.parameters(), self.max_grad_norm
            )
            grad_norms.append(grad_norm)

            self.optimizer.step()

            if gradient_step % self.target_update_interval:
                self.target_updater.step()

        avg_loss = torch.stack(losses).mean().item()
        avg_grad_norm = torch.stack(grad_norms).mean().item()

        return {
            "loss": avg_loss,
            "grad_norm": avg_grad_norm,
        }

    def eval(self):
        self.actor.eval()
        self._eval = True

    def train(self):
        self.actor.train()
        self._eval = False

    def state_dict(self) -> Dict:
        # Resuming training will become impossible if the parameters of the loss_module are discarded.
        return {
            "actor": self.actor.state_dict(),
            "loss": self.loss_module.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict):
        self.actor.load_state_dict(state_dict["actor"])
        self.loss_module.load_state_dict(state_dict["loss"])
        self.loss_module.make_value_estimator(gamma=self.gamma)
        self.target_updater = SoftUpdate(self.loss_module, eps=0.995)
