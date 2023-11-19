from typing import Optional, Union, Callable
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, set_interaction_type, InteractionType
from tensordict.tensordict import TensorDictBase
import torch
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.envs import EnvBase
from torchrl.data.tensor_specs import (
    CompositeSpec,
    DiscreteTensorSpec,
    BinaryDiscreteTensorSpec,
    BoundedTensorSpec,
    UnboundedContinuousTensorSpec,
)
import time
from gomoku_rl.utils.policy import _policy_t
from gomoku_rl.utils.augment import augment_transition
from .core import Gomoku


from gomoku_rl.utils.log import get_log_func
from collections import defaultdict


def make_transition(
    tensordict_t_minus_1, tensordict_t, tensordict_t_plus_1
) -> TensorDict:
    # if a player wins at time t, its opponent cannot win immediately after reset
    reward: torch.Tensor = (
        tensordict_t.get("done").float() - tensordict_t_plus_1.get("done").float()
    ).unsqueeze(-1)
    transition: TensorDict = tensordict_t_minus_1.select(
        "observation",
        "action_mask",
        "action",
        "sample_log_prob",
        strict=False,
    )
    transition.set(
        "next",
        tensordict_t_plus_1.select("observation", "action_mask"),
    )
    transition.set(("next", "reward"), reward)
    done_white = tensordict_t_plus_1["done"] | tensordict_t["done"]
    transition.set(("next", "done"), done_white)
    return transition


class GomokuEnv:
    def __init__(
        self,
        num_envs: int,
        board_size: int,
        device=None,
    ):
        self.gomoku = Gomoku(num_envs=num_envs, board_size=board_size, device=device)

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

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> TensorDict:
        self.gomoku.reset(env_ids=env_ids)
        tensordict = TensorDict(
            {
                "observation": self.gomoku.get_encoded_board(),
                "action_mask": self.gomoku.get_action_mask(),
            },
            self.batch_size,
            device=self.device,
        )
        return tensordict

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """_summary_

        Args:
            tensordict (TensorDict): (action,env_indices|None)

        Returns:
            TensorDict: (observation,action_mask,done,stats)
        """
        action: torch.Tensor = tensordict.get("action")
        env_indices: torch.Tensor = tensordict.get("env_indices", None)
        episode_len = self.gomoku.move_count + 1  # (E,)
        win, illegal = self.gomoku.step(action=action, env_indices=env_indices)

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

    def _step_and_maybe_reset(
        self, tensordict: TensorDict, env_indices: Optional[torch.Tensor] = None
    ) -> TensorDict:
        """_summary_

        Args:
            tensordict (TensorDict): (observation,action_mask,action)
            env_indices (Optional[torch.Tensor], optional): indices of the envs that will take a step. Defaults to None.

        Returns:
            TensorDict: (observation,action_mask,done,stats) the finished envs are reset, but their done flags are kept.
        """
        if env_indices is not None:
            tensordict.set("env_indices", env_indices)
        next_tensordict = self._step(tensordict=tensordict)
        tensordict.exclude("env_indices", inplace=True)

        done: torch.Tensor = next_tensordict.get("done")  # (E,)
        reset_td = self.reset(done)
        next_tensordict.update(reset_td)  # no impact on training
        return next_tensordict

    def _round(
        self,
        tensordict_t_minus_1: TensorDict,
        tensordict_t: TensorDict,
        player_black: _policy_t,
        player_white: _policy_t,
        return_black_transitions: bool = True,
        return_white_transitions: bool = True,
    ) -> tuple[TensorDict | None, TensorDict | None, TensorDict, TensorDict]:
        with set_interaction_type(type=InteractionType.RANDOM):
            tensordict_t = player_black(tensordict_t)
        tensordict_t_plus_1 = self._step_and_maybe_reset(tensordict=tensordict_t)

        if return_white_transitions:
            transition_white = make_transition(
                tensordict_t_minus_1, tensordict_t, tensordict_t_plus_1
            )
            # for player_white, if the env is reset at t-1, he won't make a move at t
            # this is different from player_black
            transition_white = transition_white[~tensordict_t_minus_1["done"]]
        else:
            transition_white = None

        with set_interaction_type(type=InteractionType.RANDOM):
            tensordict_t_plus_1 = player_white(tensordict_t_plus_1)
        # if player_black wins at t (and the env is reset), the env doesn't take a step at t+1
        # this makes no difference to player_black's transition from t to t+2
        # but player_white's transition from t-1 to t+1 is invalid where tensordict_t_minus_1['done']==True
        tensordict_t_plus_2 = self._step_and_maybe_reset(
            tensordict_t_plus_1, env_indices=~tensordict_t_plus_1.get("done")
        )

        if return_black_transitions:
            transition_black = make_transition(
                tensordict_t, tensordict_t_plus_1, tensordict_t_plus_2
            )
        else:
            transition_black = None

        return (
            transition_black,
            transition_white,
            tensordict_t_plus_1,
            tensordict_t_plus_2,
        )

    @torch.no_grad
    def _rollout(
        self,
        rounds: int,
        player_black: _policy_t,
        player_white: _policy_t,
        augment: bool = False,
        return_black_transitions: bool = True,
        return_white_transitions: bool = True,
    ):
        tensordict_t_minus_1 = self.reset()
        tensordict_t = self.reset()

        tensordict_t_minus_1.update(
            {
                "done": torch.ones(self.num_envs, dtype=torch.bool, device=self.device),
                "action": self.action_spec.zero(),
            }
        )
        tensordict_t.update(
            {"done": torch.ones(self.num_envs, dtype=torch.bool, device=self.device)}
        )

        buffer_size = rounds * self.num_envs
        if augment:
            buffer_size *= 8
        if return_black_transitions:
            buffer_black = TensorDictReplayBuffer(
                storage=LazyTensorStorage(max_size=buffer_size),
                sampler=SamplerWithoutReplacement(drop_last=True),
                batch_size=self.num_envs,
            )
        else:
            buffer_black = None
        if return_white_transitions:
            buffer_white = TensorDictReplayBuffer(
                storage=LazyTensorStorage(max_size=buffer_size),
                sampler=SamplerWithoutReplacement(drop_last=True),
                batch_size=self.num_envs,
            )
        else:
            buffer_white = None

        for i in range(rounds):
            (
                transition_black,
                transition_white,
                tensordict_t_minus_1,
                tensordict_t,
            ) = self._round(
                tensordict_t_minus_1,
                tensordict_t,
                player_black,
                player_white,
                return_black_transitions,
                return_white_transitions,
            )

            if augment:
                transition_black = (
                    augment_transition(transition_black)
                    if return_black_transitions
                    else None
                )
                transition_white = (
                    augment_transition(transition_white)
                    if return_white_transitions
                    else None
                )
            if return_black_transitions:
                buffer_black.extend(transition_black)
            if return_white_transitions and len(transition_white) > 0:
                buffer_white.extend(transition_white)

        return buffer_black, buffer_white

    def rollout(
        self,
        rounds: int,
        player_black: _policy_t,
        player_white: _policy_t,
        augment: bool = False,
        return_black_transitions: bool = True,
        return_white_transitions: bool = True,
    ):
        info: defaultdict[str, float] = defaultdict(float)
        self._post_step = get_log_func(info)

        start = time.perf_counter()
        r = self._rollout(
            rounds=rounds,
            player_black=player_black,
            player_white=player_white,
            augment=augment,
            return_black_transitions=return_black_transitions,
            return_white_transitions=return_white_transitions,
        )
        end = time.perf_counter()
        self._fps = (rounds * 2 * self.num_envs) / (end - start)

        self._post_step = None
        return *r, info

    @torch.no_grad
    def _flexible_rollout(
        self,
        rounds: int,
        player_0: _policy_t,
        player_1: _policy_t,
        augment: bool = False,
        return_0_transitions: bool = True,
        return_1_transitions: bool = True,
    ):
        raise NotImplementedError

    @torch.no_grad
    def flexible_rollout(
        self,
        rounds: int,
        player_0: _policy_t,
        player_1: _policy_t,
        augment: bool = False,
        return_0_transitions: bool = True,
        return_1_transitions: bool = True,
    ):
        info: defaultdict[str, float] = defaultdict(float)
        self._post_step = get_log_func(info)

        start = time.perf_counter()
        r = self._flexible_rollout(
            rounds=rounds,
            player_0=player_0,
            player_1=player_1,
            augment=augment,
            return_0_transitions=return_0_transitions,
            return_1_transitions=return_1_transitions,
        )
        end = time.perf_counter()
        self._fps = (rounds * 2 * self.num_envs) / (end - start)

        self._post_step = None
        return *r, info
