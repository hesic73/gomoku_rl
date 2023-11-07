from .policy import Policy
from .ppo import PPOPolicy
from torch.cuda import _device_t
from torchrl.data.tensor_specs import DiscreteTensorSpec,TensorSpec
from omegaconf import DictConfig

def get_policy(name:str, cfg:DictConfig,action_spec:DiscreteTensorSpec,observation_spec:TensorSpec, device:_device_t="cuda")->Policy:
    if name.lower()=="ppo":
        return PPOPolicy(cfg=cfg,action_spec=action_spec,observation_spec=observation_spec,device=device)
    else:
        raise NotImplementedError(f"{name} not implemented.")