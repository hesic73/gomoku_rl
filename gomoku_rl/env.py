from typing import Union, Callable
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, set_interaction_type, InteractionType
import torch

from torchrl.data.tensor_specs import (
    CompositeSpec,
    DiscreteTensorSpec,
    BinaryDiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
import time
from gomoku_rl.utils.policy import _policy_t
from gomoku_rl.utils.misc import add_prefix
from .core import Gomoku


from gomoku_rl.utils.log import get_log_func
from collections import defaultdict


class GomokuEnv:
    def __init__(
        self,
        num_envs: int,
        board_size: int,
        device=None,
    ):
        """Initializes a parallel Gomoku environment.

        Args:
            num_envs (int): The number of independent game environments to run in parallel. Each environment represents a separate instance of a Gomoku game.
            board_size (int): The size of the square Gomoku game board.
            device: The computational device (e.g., CPU, GPU) on which the game simulations will run. If `None`, the default device is used.
        """
        self.gomoku = Gomoku(
            num_envs=num_envs, board_size=board_size, device=device)

        self.observation_spec = CompositeSpec(
            {
                "observation": UnboundedContinuousTensorSpec(
                    device=self.device,
                    shape=[num_envs, 3, board_size, board_size],
                ),
                "action_mask": BinaryDiscreteTensorSpec(
                    n=board_size * board_size,
                    device=self.device,
                    shape=[num_envs, board_size * board_size],
                    dtype=torch.bool,
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

        self._post_step: Callable[
            [
                TensorDict,
            ],
            None,
        ] | None = None

    @property
    def batch_size(self):
        return torch.Size((self.num_envs,))

    @property
    def board_size(self):
        return self.gomoku.board_size

    @property
    def device(self):
        return self.gomoku.device

    @property
    def num_envs(self):
        return self.gomoku.num_envs

    def reset(self, env_indices: torch.Tensor | None = None) -> TensorDict:
        """Resets the specified game environments to their initial states, or all environments if none are specified.

        Args:
            env_indices (torch.Tensor | None, optional): Indices of environments to reset. Resets all if None. Defaults to None.

        Returns:
            TensorDict: A tensor dictionary containing the initial observations and action masks for all environments.
        """
        self.gomoku.reset(env_indices=env_indices)
        tensordict = TensorDict(
            {
                "observation": self.gomoku.get_encoded_board(),
                "action_mask": self.gomoku.get_action_mask(),
            },
            self.batch_size,
            device=self.device,
        )
        return tensordict

    def step(
        self,
        tensordict: TensorDict,
    ) -> TensorDict:
        """Advances the state of the environments by one timestep based on the actions provided in the `tensordict`.

        Args:
            tensordict (TensorDict): A dictionary containing tensors with the actions to be taken in each environment. May also include optional environment masks to specify which environments should be updated.

        Returns:
            TensorDict: output tensor dictionary containing the updated observations, action masks, and other information for all environments.
        """
        action: torch.Tensor = tensordict.get("action")
        env_mask: torch.Tensor = tensordict.get("env_mask", None)
        episode_len = self.gomoku.move_count + 1  # (E,)
        win, illegal = self.gomoku.step(action=action, env_mask=env_mask)

        assert not illegal.any()

        done = win
        black_win = win & (episode_len % 2 == 1)
        white_win = win & (episode_len % 2 == 0)
        tensordict = TensorDict({}, self.batch_size, device=self.device)
        tensordict.update(
            {
                "observation": self.gomoku.get_encoded_board(),
                "action_mask": self.gomoku.get_action_mask(),
                "done": done,
                "win": win,
                # reward is calculated later
                "stats": {
                    "episode_len": episode_len,
                    "black_win": black_win,
                    "white_win": white_win,
                },
            }
        )
        if self._post_step:
            self._post_step(tensordict)
        return tensordict

    def step_and_maybe_reset(
        self,
        tensordict: TensorDict,
        env_mask: torch.Tensor | None = None,
    ) -> TensorDict:
        """Simulates a single step of the game environment and resets the environment if the game ends.

        Args:
            tensordict (TensorDict): A dictionary containing tensors with the current observations, action masks, and actions for each environment.
            env_mask (torch.Tensor | None, optional): A 1D tensor specifying which environments should be updated. If `None`, all environments are updated.

        Returns:
            TensorDict: A dictionary containing tensors with the updated observations, action masks, and other relevant information for each environment.
            For environments that have concluded their game and are reset, the 'observation' key will reflect the new initial state,
            but **the 'done' flag remains set to True** to indicate the end of the previous game within this timestep.
        """

        if env_mask is not None:
            tensordict.set("env_mask", env_mask)
        next_tensordict = self.step(tensordict=tensordict)
        tensordict.exclude("env_mask", inplace=True)

        done: torch.Tensor = next_tensordict.get("done")  # (E,)
        env_ids = done.nonzero().squeeze(0)
        reset_td = self.reset(env_indices=env_ids)
        next_tensordict.update(reset_td)  # no impact on training
        return next_tensordict

    def set_post_step(self, post_step: Callable[[TensorDict], None] | None = None):
        """Sets a function to be called after each step in the environment.

        Args:
            post_step (Callable[[TensorDict], None] | None, optional): A function that takes a tensor dictionary as input and performs some action. Defaults to None.
        """
        self._post_step = post_step
