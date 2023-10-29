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


from .core import Gomoku


class GomokuEnv(EnvBase):
    def __init__(
        self, num_envs: int, board_size: int = 19, device: DEVICE_TYPING = None
    ):
        super().__init__(device, batch_size=[num_envs])
        self.gomoku = Gomoku(num_envs=num_envs, board_size=board_size, device=device)
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

    def to(self, device: DEVICE_TYPING) -> EnvBase:
        self.gomoku.to(device)
        return super().to(device)

    def _set_seed(self, seed: int | None):
        torch.manual_seed(seed)

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        if tensordict is not None:
            env_mask = tensordict.get("_reset").reshape(self.gomoku.num_envs)
        else:
            env_mask = torch.ones(self.gomoku.num_envs, dtype=bool, device=self.device)
        tensordict = TensorDict({}, batch_size=self.batch_size, device=self.device)

        env_ids = env_mask.cpu().nonzero().squeeze(-1).to(self.device)

        self.gomoku.reset(env_ids=env_ids)
        tensordict.set("observation", self.gomoku.get_encoded_board())
        return tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action: torch.Tensor = tensordict.get("action")
        done, illegal = self.gomoku.step(action=action)
        tensordict = TensorDict({}, self.batch_size)
        tensordict.update(
            {
                "done": done | illegal,
                "observation": self.gomoku.get_encoded_board(),
                "reward": done.float() - illegal.float(),
                "stats": {"episode_len": self.gomoku.move_count},
            }
        )

        return tensordict


class GomokuEnvWithOpponent(EnvBase):
    def __init__(
        self,
        num_envs: int,
        board_size: int = 19,
        initial_policy: TensorDictModule
        | Callable[
            [
                TensorDictBase,
            ],
            TensorDictBase,
        ]
        | None = None,
        device: DEVICE_TYPING = None,
    ):
        super().__init__(device, batch_size=[num_envs])
        self.gomoku = Gomoku(num_envs=num_envs, board_size=board_size, device=device)
        if initial_policy is None:
            initial_policy = lambda tensordict: self.rand_action(tensordict)
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

    def set_opponent_policy(
        self,
        policy: TensorDictModule
        | Callable[
            [
                TensorDictBase,
            ],
            TensorDictBase,
        ],
    ):
        self.opponent_policy = policy

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
        tensordict = TensorDict({}, batch_size=self.batch_size, device=self.device)

        env_ids = env_mask.cpu().nonzero().squeeze(-1).to(self.device)

        self.gomoku.reset(env_ids=env_ids)
        # TO DO: opponent play a move here
        tensordict.set("observation", self.gomoku.get_encoded_board())
        return tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action: torch.Tensor = tensordict.get("action")
        win, illegal = self.gomoku.step(action=action)

        reset_envs = (win | illegal).cpu().nonzero().squeeze(-1).to(self.device)
        self.gomoku.reset(env_ids=reset_envs)

        opponent_tensordict = TensorDict(
            {"observation": self.gomoku.get_encoded_board()},
            self.batch_size,
            device=self.device,
        )
        with torch.no_grad():
            opponent_tensordict = self.opponent_policy(opponent_tensordict)
        opponent_action = opponent_tensordict.get("action")
        
        # if the environment has been reset, the opponent can never win or make an illegal move
        # so opponent_win/illegal is nonzero only if win/illegal is nonzero
        opponent_win, opponent_illegal = self.gomoku.step(action=opponent_action)

        done = win | illegal | opponent_win | opponent_illegal
        reward = (
            done.float()
            - illegal.float()
            - opponent_win.float()
            + opponent_illegal.float()
        )

        tensordict = TensorDict({}, self.batch_size, device=self.device)
        tensordict.update(
            {
                "done": done,
                "observation": self.gomoku.get_encoded_board(),
                "reward": reward,
            }
        )
        return tensordict
