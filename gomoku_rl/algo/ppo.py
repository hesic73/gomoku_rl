from typing import Callable, Dict, List, Any,Union,Iterable
from tensordict import TensorDictBase, TensorDict
from tensordict.utils import NestedKey
import torch
import torch.nn as nn

from tensordict.nn import TensorDictModule
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.modules import DuelingCnnDQNet

from torchrl.data import TensorSpec, DiscreteTensorSpec
from torch.cuda import _device_t
from torchrl.envs.utils import ExplorationType
from torchrl.envs import TransformedEnv
import datetime
import tempfile


from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm

import logging

from torchrl.modules import ProbabilisticActor

from torch.distributions.categorical import Categorical

from torchrl.modules.models import ConvNet,MLP
from torchrl.modules import ValueOperator

from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from .policy import Policy


class _Actor(nn.Module):
    def __init__(self,device:_device_t,n_action,cnn_kwargs:dict,mlp_kwargs:dict) -> None:
        super().__init__()
        self.features=ConvNet(device=device,**cnn_kwargs)
        self.advantage=MLP(out_features=n_action, device=device, **mlp_kwargs)
        
    def forward(self,x:torch.Tensor,mask:torch.Tensor|None=None)->torch.Tensor:
        x = self.features(x)
        advantage:torch.Tensor = self.advantage(x)
        if mask is not None:
            advantage=torch.where(mask==0,advantage,-999999)
        probs=torch.special.softmax(advantage,dim=-1) # (E, board_size^2)
        return probs


def make_actor(
    cfg: DictConfig,
    action_spec: TensorSpec,
    device:_device_t,
):
    cnn_kwargs = OmegaConf.to_container(cfg.cnn_kwargs)
    cnn_kwargs.update(
        {"activation_class": getattr(nn, cnn_kwargs.get("activation_class", "ReLU"))}
    )
    mlp_kwargs = OmegaConf.to_container(cfg.mlp_kwargs)
    mlp_kwargs.update(
        {"activation_class": getattr(nn, mlp_kwargs.get("activation_class", "ReLU"))}
    )

    actor_net=_Actor(device=device,n_action=action_spec.space.n,cnn_kwargs=cnn_kwargs,mlp_kwargs=mlp_kwargs)
    
    policy_module=TensorDictModule(module=actor_net,in_keys=['observation',"action_mask"],out_keys=[
        'probs'
    ])
    
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
    device:_device_t,
):
    cnn_kwargs = OmegaConf.to_container(cfg.cnn_kwargs)
    cnn_kwargs.update(
        {"activation_class": getattr(nn, cnn_kwargs.get("activation_class", "ReLU"))}
    )
    mlp_kwargs = OmegaConf.to_container(cfg.mlp_kwargs)
    mlp_kwargs.update(
        {"activation_class": getattr(nn, mlp_kwargs.get("activation_class", "ReLU"))}
    )

    value_net=nn.Sequential(ConvNet(device=device,**cnn_kwargs),MLP(out_features=1,device=device,**mlp_kwargs))
    
    value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
)
    
    return value_module


class PPOPolicy(Policy):
    def __init__(self, cfg:DictConfig,action_spec:DiscreteTensorSpec,observation_spec:TensorSpec, device:_device_t="cuda") -> None:
        super().__init__(cfg,action_spec,observation_spec,device)
        self.cfg:DictConfig=cfg
        self.device:_device_t = device
          
        self.clip_param:float = cfg.clip_param
        self.ppo_epoch:int = int(cfg.ppo_epochs)

        self.entropy_coef:float = cfg.entropy_coef
        self.gae_gamma :float= cfg.gamma
        self.gae_lambda :float= cfg.gae_lambda
        
        self.max_grad_norm:float=cfg.max_grad_norm
        
        self.actor=make_actor(cfg=cfg.actor,action_spec=action_spec,device=self.device)
        self.critic=make_critic(cfg=cfg.critic,device=self.device)
        
        fake_input = observation_spec.zero()
        self.actor(fake_input)
        self.critic(fake_input)
        
        
        self.advantage_module = GAE(
        gamma=self.gae_gamma, lmbda=self.gae_lambda, value_network=self.critic, average_gae=True
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

        self.optim = torch.optim.Adam(self.loss_module.parameters(),cfg.lr)
        
            
    def __call__(self, tensordict: TensorDict):
        actor_input=tensordict.select("observation","action_mask",strict=False)
        actor_output:TensorDict = self.actor(actor_input)
        actor_output=actor_output.exclude("probs")
        tensordict.update(actor_output)
        
        
        critic_input=tensordict.select("observation")
        critic_output = self.critic(critic_input)
        tensordict.update(critic_output)
        
        return tensordict


    def train_op(self, tensordict: TensorDict):      
        losses=[]
        for _ in range(self.ppo_epoch):
            dataset = make_dataset_naive(
                tensordict,
                int(self.cfg.num_minibatches))
            
            for minibatch in dataset:
                minibatch=minibatch.to(self.device)
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
                torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), self.max_grad_norm)
                self.optim.step()
                self.optim.zero_grad()

        return {
            "loss":torch.stack(losses).mean().item(),
        }

    
    def get_actor(self):
        return self.actor

    
    @staticmethod
    def from_checkpoint(checkpoint:Dict,cfg:DictConfig,action_spec:DiscreteTensorSpec ,device:_device_t,deterministic:bool=False)->TensorDictModule:
        actor=make_actor(cfg,action_spec,device)
        actor.load_state_dict(checkpoint)
        if deterministic:
            def _policy(tensordict:TensorDict):
                tensordict=actor(tensordict)
                probs:torch.Tensor=tensordict.get("probs")
                action_mask:torch.Tensor=tensordict.get("action_mask",None)
                if action_mask is not None:
                    probs=probs*(action_mask==0).float()
                action=probs.argmax(dim=-1)
                tensordict.update({"action":action})
                return tensordict
            
            _policy.device=actor.device
            return _policy
        else:
            return actor
    
def make_dataset_naive(
    tensordict: TensorDict, num_minibatches: int = 4
):

    tensordict = tensordict.reshape(-1)
    perm = torch.randperm(
            (tensordict.shape[0] // num_minibatches) * num_minibatches,
            device=tensordict.device,
        ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]
