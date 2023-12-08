from typing import Optional, Union, Callable
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, set_interaction_type, InteractionType
import torch

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
from gomoku_rl.utils.misc import add_prefix
from .core import Gomoku


from gomoku_rl.utils.log import get_log_func
from collections import defaultdict


def make_transition(
    tensordict_t_minus_1: TensorDict,
    tensordict_t: TensorDict,
    tensordict_t_plus_1: TensorDict,
) -> TensorDict:
    # if a player wins at time t, its opponent cannot win immediately after reset
    reward: torch.Tensor = (
        tensordict_t.get("win").float() - tensordict_t_plus_1.get("win").float()
    ).unsqueeze(-1)
    transition: TensorDict = tensordict_t_minus_1.select(
        "observation",
        "action_mask",
        "action",
        "sample_log_prob",
        "state_value",
        strict=False,
    )
    transition.set(
        "next",
        tensordict_t_plus_1.select(
            "observation", "action_mask", "state_value", strict=False
        ),
    )
    transition.set(("next", "reward"), reward)
    done = tensordict_t_plus_1["done"] | tensordict_t["done"]
    transition.set(("next", "done"), done)
    # assert no_nan_in_tensordict(transition)
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

    def _step(
        self,
        tensordict: TensorDict,
        is_last: bool,
    ) -> TensorDict:
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

        done = win | is_last
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

    def _step_and_maybe_reset(
        self,
        tensordict: TensorDict,
        env_indices: Optional[torch.Tensor] = None,
        is_last: bool = False,
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
        next_tensordict = self._step(tensordict=tensordict, is_last=is_last)
        tensordict.exclude("env_indices", inplace=True)

        done: torch.Tensor = next_tensordict.get("done")  # (E,)
        env_ids = done.nonzero().squeeze(0)
        reset_td = self.reset(env_ids=env_ids)
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
        is_last: bool = False,
    ) -> tuple[TensorDict | None, TensorDict | None, TensorDict, TensorDict]:
        tensordict_t_plus_1 = self._step_and_maybe_reset(tensordict=tensordict_t)

        with set_interaction_type(type=InteractionType.RANDOM):
            tensordict_t_plus_1 = player_white(tensordict_t_plus_1)

        if return_white_transitions:
            transition_white = make_transition(
                tensordict_t_minus_1, tensordict_t, tensordict_t_plus_1
            )
            # for player_white, if the env is reset at t-1, he won't make a move at t
            # this is different from player_black

            # transition_white = transition_white[~tensordict_t_minus_1["done"]]
            # the trick is that we set done=True for computing gae
            # after that we just discard these invalid transition
            invalid: torch.Tensor = tensordict_t_minus_1["done"]
            # transition_white["observation"] = (
            #     -invalid.float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            #     + (1 - invalid.float()).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            #     * transition_white["observation"]
            # )
            transition_white["next", "done"] = (
                invalid | transition_white["next", "done"]
            )
            transition_white.set("invalid", invalid)

        else:
            transition_white = None

        # if player_black wins at t (and the env is reset), the env doesn't take a step at t+1
        # this makes no difference to player_black's transition from t to t+2
        # but player_white's transition from t-1 to t+1 is invalid where tensordict_t_minus_1['done']==True
        tensordict_t_plus_2 = self._step_and_maybe_reset(
            tensordict_t_plus_1,
            env_indices=~tensordict_t_plus_1.get("done"),
            is_last=is_last,
        )

        with set_interaction_type(type=InteractionType.RANDOM):
            tensordict_t_plus_2 = player_black(tensordict_t_plus_2)

        if return_black_transitions:
            transition_black = make_transition(
                tensordict_t, tensordict_t_plus_1, tensordict_t_plus_2
            )
            transition_black.set(
                "invalid",
                torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
            )
        else:
            transition_black = None

        return (
            transition_black,
            transition_white,
            tensordict_t_plus_1,
            tensordict_t_plus_2,
        )

    @torch.no_grad()
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
                "win": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            }
        )  # here we set it to True
        with set_interaction_type(type=InteractionType.RANDOM):
            tensordict_t = player_black(tensordict_t)

        tensordict_t.update(
            {
                "done": torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                ),
                "win": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            }
        )

        blacks: list[TensorDict] = []
        whites: list[TensorDict] = []

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
                is_last=i == rounds - 1,
            )

            if augment:
                transition_black = (
                    augment_transition(transition_black)
                    if return_black_transitions
                    else transition_black
                )
                transition_white = (
                    augment_transition(transition_white)
                    if return_white_transitions
                    else transition_white
                )
            if return_black_transitions:
                blacks.append(transition_black)
            if return_white_transitions and i != 0:
                whites.append(transition_white)

        return blacks, whites

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
        info_buffer = defaultdict(float)
        self._post_step = get_log_func(info_buffer)

        start = time.perf_counter()
        blacks, whites = self._rollout(
            rounds=rounds,
            player_black=player_black,
            player_white=player_white,
            augment=augment,
            return_black_transitions=return_black_transitions,
            return_white_transitions=return_white_transitions,
        )
        end = time.perf_counter()
        self._fps = (rounds * 2 * self.num_envs) / (end - start)
        info.update(add_prefix(info_buffer, "train/"))
        self._post_step = None

        blacks = torch.stack(blacks, dim=-1) if blacks else None
        whites = torch.stack(whites, dim=-1) if whites else None
        return blacks, whites, info

    @torch.no_grad()
    def _rollout_fixed_opponent(
        self,
        rounds: int,
        player_black: _policy_t,
        player_white: _policy_t,
        return_black_transitions: bool,
        out_device,
        augment: bool = False,
    ) -> list[TensorDict]:
        if out_device is None:
            out_device = self.device
        tensordicts = []
        tensordict_t_minus_1 = self.reset()
        tensordict_t = self.reset()

        tensordict_t_minus_1.update(
            {
                "done": torch.ones(self.num_envs, dtype=torch.bool, device=self.device),
                "win": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                "action": torch.zeros(
                    self.num_envs, dtype=torch.long, device=self.device
                ),  # placeholder
            }
        )  # here we set it to True

        with set_interaction_type(type=InteractionType.RANDOM):
            tensordict_t = player_black(tensordict_t)

        tensordict_t.update(
            {
                "done": torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                ),
                "win": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            }
        )

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
                not return_black_transitions,
                is_last=i == rounds - 1,
            )
            if augment:
                if return_black_transitions:
                    transition_black = augment_transition(transition_black)
                elif len(transition_white) > 0:
                    transition_white = augment_transition(transition_white)
            if return_black_transitions:
                tensordicts.append(transition_black.to(out_device))
            elif len(transition_white) > 0 and i != 0:
                tensordicts.append(transition_white.to(out_device))

        return tensordicts

    def rollout_fixed_opponent(
        self,
        rounds: int,
        player: _policy_t,
        opponent: _policy_t,
        out_device=None,
        augment: bool = False,
    ) -> tuple[TensorDict, dict[str, float]]:
        info: defaultdict[str, float] = defaultdict(float)

        tensordicts: list[TensorDict] = []

        start = time.perf_counter()
        info_buffer = defaultdict(float)
        self._post_step = get_log_func(info_buffer)
        tensordicts.extend(
            self._rollout_fixed_opponent(
                rounds=rounds,
                player_black=player,
                player_white=opponent,
                return_black_transitions=True,
                out_device=out_device,
                augment=augment,
            )
        )

        info.update(
            {
                "train/player_black_win": info_buffer["black_win"],
                "train/player_black_episode_len": info_buffer["episode_len"],
            }
        )
        info_buffer.clear()

        self._post_step = get_log_func(info_buffer)
        tensordicts.extend(
            self._rollout_fixed_opponent(
                rounds=rounds,
                player_black=opponent,
                player_white=player,
                return_black_transitions=False,
                out_device=out_device,
                augment=augment,
            )
        )
        end = time.perf_counter()
        self._fps = (2 * rounds * 2 * self.num_envs) / (end - start)
        info.update(
            {
                "train/player_white_win": info_buffer["white_win"],
                "train/player_white_episode_len": info_buffer["episode_len"],
            }
        )
        self._post_step = None

        out = torch.stack(tensordicts, dim=-1)

        return out, info

    def rollout_player_white(
        self,
        rounds: int,
        player: _policy_t,
        opponent: _policy_t,
        out_device=None,
        augment: bool = False,
    ) -> tuple[TensorDict, dict[str, float]]:
        info: defaultdict[str, float] = defaultdict(float)

        tensordicts: list[TensorDict] = []

        start = time.perf_counter()
        info_buffer = defaultdict(float)

        self._post_step = get_log_func(info_buffer)
        tensordicts.extend(
            self._rollout_fixed_opponent(
                rounds=rounds,
                player_black=opponent,
                player_white=player,
                return_black_transitions=False,
                out_device=out_device,
                augment=augment,
            )
        )
        end = time.perf_counter()
        self._fps = (rounds * 2 * self.num_envs) / (end - start)
        info.update(
            {
                "train/player_white_win": info_buffer["white_win"],
                "train/player_white_episode_len": info_buffer["episode_len"],
            }
        )
        self._post_step = None

        out = torch.stack(tensordicts, dim=-1)

        return out, info

    def rollout_player_black(
        self,
        rounds: int,
        player: _policy_t,
        opponent: _policy_t,
        out_device=None,
        augment: bool = False,
    ) -> tuple[TensorDict, dict[str, float]]:
        info: defaultdict[str, float] = defaultdict(float)

        tensordicts: list[TensorDict] = []

        start = time.perf_counter()
        info_buffer = defaultdict(float)

        self._post_step = get_log_func(info_buffer)
        tensordicts.extend(
            self._rollout_fixed_opponent(
                rounds=rounds,
                player_black=player,
                player_white=opponent,
                return_black_transitions=True,
                out_device=out_device,
                augment=augment,
            )
        )
        end = time.perf_counter()
        self._fps = (rounds * 2 * self.num_envs) / (end - start)
        info.update(
            {
                "train/player_black_win": info_buffer["black_win"],
                "train/player_black_episode_len": info_buffer["episode_len"],
            }
        )
        self._post_step = None

        out = torch.stack(tensordicts, dim=-1)

        return out, info
