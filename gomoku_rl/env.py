from typing import Optional, Union, Callable
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.tensordict import TensorDictBase
import torch
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs import EnvBase
from torchrl.data.tensor_specs import (
    CompositeSpec,
    DiscreteTensorSpec,
    BinaryDiscreteTensorSpec,
    BoundedTensorSpec,
    UnboundedContinuousTensorSpec,
)
from torch.distributions import Categorical

from .core import Gomoku

_policy_t=TensorDictModule| Callable[
            [
                TensorDictBase,
            ],
            TensorDictBase,
        ]


def random_policy_with_mask(action_mask:torch.Tensor):
    probs=torch.zeros_like(action_mask,dtype=torch.float)
    probs=torch.where(action_mask==0,probs,-999)
    probs=torch.special.softmax(probs,dim=-1)
    dist=Categorical(probs=probs)
    return dist.sample()

class GomokuEnvWithOpponent(EnvBase):
    def __init__(
        self,
        num_envs: int,
        board_size: int = 19,
        initial_policy: _policy_t
        | None = None,
        device: DEVICE_TYPING = None,
    ):
        super().__init__(device, batch_size=[num_envs])
        self.gomoku = Gomoku(num_envs=num_envs, board_size=board_size, device=device)
        
        
        if initial_policy is None:
            def initial_policy(tensordict:TensorDict):
                action_mask=tensordict.get("action_mask",None)
                if action_mask is None:
                    return self.rand_action(tensordict)
                action=random_policy_with_mask(action_mask=action_mask)
                tensordict.update({"action":action})
                return tensordict
                
        self.opponent_policy = initial_policy
        
        
        self.observation_spec = CompositeSpec(
            {
                "observation": UnboundedContinuousTensorSpec(
                    device=self.device,
                    shape=[num_envs, 3, board_size, board_size],
                ),
            },
            shape=[
                num_envs,
            ],
            device=self.device,
        )
        self.action_spec = DiscreteTensorSpec(
            board_size * board_size,
            shape=[
                num_envs,
            ],
            device=self.device,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=[num_envs, 1],
            device=self.device,
        )
        
        self.stats_keys=[
            "episode_len",
            "illegal",
            "win",
            "opponent_win",
            "opponent_illegal",
            "game_win",
            "black",
        ]
        self.stats_keys=[("stats",k) for k in self.stats_keys]
        
        self.black=torch.zeros(num_envs,dtype=torch.bool,device=device) # (E,)

    def set_opponent_policy(
        self,
        policy: _policy_t,
    ):
        self.opponent_policy = policy
        self.reset()

    def to(self, device: DEVICE_TYPING) -> EnvBase:
        self.gomoku.to(device)
        if isinstance(self.opponent_policy, TensorDictModule):
            self.opponent_policy.to(device)
        return super().to(device)

    def _set_seed(self, seed: int | None):
        torch.manual_seed(seed)

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        if tensordict is not None:
            env_mask = tensordict.get("_reset").reshape(self.gomoku.num_envs)
        else:
            env_mask = torch.ones(self.gomoku.num_envs, dtype=bool, device=self.device)

        env_ids = env_mask.cpu().nonzero().squeeze(-1).to(self.device)

        self.gomoku.reset(env_ids=env_ids)
        
        opponent_tensordict = TensorDict(
            {"observation": self.gomoku.get_encoded_board(),
             "action_mask":self.gomoku.get_action_mask(),},
            self.batch_size,
            device=self.device,
        )
        with torch.no_grad():
            opponent_tensordict = self.opponent_policy(opponent_tensordict)
        opponent_action = opponent_tensordict.get("action")
        env_mask=(torch.rand_like(env_mask,dtype=torch.float)>0.5)&env_mask
        self.gomoku.step(action=opponent_action,env_indices=env_mask)
        self.black[env_ids]=env_mask[env_ids]
        
        
        tensordict = TensorDict(
            {"observation": self.gomoku.get_encoded_board(),
             "action_mask":self.gomoku.get_action_mask(),},
            self.batch_size,
            device=self.device,
        )
        return tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action: torch.Tensor = tensordict.get("action")
        episode_len=self.gomoku.move_count.clone() # (E,)
        
        win, illegal = self.gomoku.step(action=action)
        reset_envs_ids = (win | illegal).cpu().nonzero().squeeze(-1).to(self.device)
        env_indices=~(win|illegal)
        self.gomoku.reset(env_ids=reset_envs_ids)
        episode_len=torch.where(self.gomoku.move_count==0,episode_len,self.gomoku.move_count) # (E,)


        opponent_tensordict = TensorDict(
            {"observation": self.gomoku.get_encoded_board(),
             "action_mask":self.gomoku.get_action_mask(),},
            self.batch_size,
            device=self.device,
        )
        with torch.no_grad():
            opponent_tensordict = self.opponent_policy(opponent_tensordict)
        opponent_action = opponent_tensordict.get("action")
        
        # if the environment has been reset, the opponent can never win or make an illegal move at the first step
        # so opponent_win/illegal is nonzero only if win/illegal is nonzero
        
        # UPDATE: if the game is over, the opponent will not make a move
        opponent_win, opponent_illegal = self.gomoku.step(action=opponent_action,env_indices=env_indices)
        opponent_win=opponent_win&env_indices
        opponent_illegal=opponent_illegal&env_indices
        reset_envs_ids = (opponent_win | opponent_illegal).cpu().nonzero().squeeze(-1).to(self.device)
        self.gomoku.reset(env_ids=reset_envs_ids)
        episode_len=torch.where(self.gomoku.move_count==0,episode_len,self.gomoku.move_count) # (E,)
        
        
        game_win=win|opponent_illegal
        done = win | illegal | opponent_win | opponent_illegal
        
        # 对手下错是对手太菜了，而不是我厉害
        reward = (
            win.float()
            - illegal.float()
            - opponent_win.float()
        #    + opponent_illegal.float()
        )

        tensordict = TensorDict({}, self.batch_size, device=self.device)
        tensordict.update(
            {
                "done": done,
                "observation": self.gomoku.get_encoded_board(),
                "action_mask":self.gomoku.get_action_mask(),
                "reward": reward,         
                "stats":{  
                    "episode_len":episode_len, 
                    'illegal':illegal,
                    "win":win,
                    "opponent_win":opponent_win,
                    "opponent_illegal":opponent_illegal,
                    "game_win":game_win,
                    "black":self.black.clone(),
                }
            }
        )
        return tensordict
