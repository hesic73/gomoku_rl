from abc import ABC,abstractmethod
from typing import Optional,Dict
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch.cuda import _device_t
from torchrl.data.tensor_specs import DiscreteTensorSpec,TensorSpec
from omegaconf import DictConfig

class Policy(ABC):
    @abstractmethod
    def __init__(self, cfg:DictConfig,action_spec:DiscreteTensorSpec,observation_spec:TensorSpec, device:_device_t="cuda") -> None:
        ...
    @abstractmethod
    def __call__(self, tensordict: TensorDict)->TensorDict:
        ...
    @abstractmethod
    def train_op(self, tensordict: TensorDict)->Dict:
        ...
        
    @abstractmethod
    def get_actor(self)->TensorDictModule:
        ...
        
    @staticmethod
    @abstractmethod
    def from_checkpoint(checkpoint:Dict,*args, **kwargs)->TensorDictModule:
        ...