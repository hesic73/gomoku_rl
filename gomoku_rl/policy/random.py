from typing import Dict
from omegaconf import DictConfig
from tensordict import TensorDict
from torch.cuda import _device_t
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.tensor_specs import DiscreteTensorSpec, TensorSpec
from .base import Policy
from gomoku_rl.utils.policy import uniform_policy


class RandomPolicy(Policy):
    def __init__(
        self,
        cfg: DictConfig,
        action_spec: DiscreteTensorSpec,
        observation_spec: TensorSpec,
        device: _device_t = "cuda",
    ) -> None:
        pass

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        return uniform_policy(tensordict)

    def state_dict(self) -> Dict:
        return {}

    def load_state_dict(self, state_dict: Dict):
        pass

    def learn(self, data: ReplayBuffer) -> Dict:
        return {}

    def train(self):
        pass
    
    def eval(self):
        pass