import abc
from typing import Dict
from tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.tensor_specs import DiscreteTensorSpec, TensorSpec
from omegaconf import DictConfig
from torch.cuda import _device_t


class Policy(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        cfg: DictConfig,
        action_spec: DiscreteTensorSpec,
        observation_spec: TensorSpec,
        device: _device_t = "cuda",
    ) -> None:
        ...

    @abc.abstractmethod
    def __call__(self, tensordict: TensorDict) -> TensorDict:
        ...

    @abc.abstractmethod
    def learn(self, data: ReplayBuffer) -> Dict:
        ...

    @abc.abstractmethod
    def state_dict(self) -> Dict:
        ...

    @abc.abstractmethod
    def load_state_dict(self, state_dict: Dict):
        ...
        
    @abc.abstractmethod
    def train(self):
        ...
        
    @abc.abstractmethod
    def eval(self):
        ...
        
    
