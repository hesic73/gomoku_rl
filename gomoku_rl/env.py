from typing import Optional, Union, Callable
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.tensordict import TensorDictBase
import torch
from torchrl.data.utils import DEVICE_TYPING
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
from gomoku_rl.utils.policy import _policy_t, uniform_policy

from .core import Gomoku


class GomokuEnvWithOpponent(EnvBase):
    def __init__(
        self,
        num_envs: int,
        board_size: int = 19,
        initial_policy: Optional[_policy_t] = None,
        device: DEVICE_TYPING = None,
    ):
        super().__init__(device, batch_size=[num_envs])
        self.gomoku = Gomoku(num_envs=num_envs, board_size=board_size, device=device)

        self.opponent_policy = initial_policy or uniform_policy

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

        self.stats_keys = [
            "episode_len",
            "win",
            "black_win_rate",
            "white_win_rate",
        ]
        self.stats_keys = [("stats", k) for k in self.stats_keys]

        self.black = torch.zeros(num_envs, dtype=torch.bool, device=device)  # (E,)

    def set_opponent_policy(
        self,
        policy: _policy_t,
    ):
        self.opponent_policy = policy

    def to(self, device: DEVICE_TYPING) -> EnvBase:
        self.gomoku.to(device)
        if isinstance(self.opponent_policy, TensorDictModule):
            self.opponent_policy.to(device)
        return super().to(device)

    def _set_seed(self, seed: Optional[int]):
        torch.manual_seed(seed)

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        if tensordict is not None:
            env_mask = tensordict.get("_reset").reshape(self.gomoku.num_envs)
        else:
            env_mask = torch.ones(self.gomoku.num_envs, dtype=bool, device=self.device)

        self.gomoku.reset(env_ids=env_mask)

        opponent_tensordict = TensorDict(
            {
                "observation": self.gomoku.get_encoded_board(),
                "action_mask": self.gomoku.get_action_mask(),
            },
            self.batch_size,
            device=self.device,
        )
        with torch.no_grad():
            opponent_tensordict = self.opponent_policy(opponent_tensordict)
        opponent_action = opponent_tensordict.get("action")
        _opponent_first_mask = torch.rand_like(env_mask, dtype=torch.float) > 0.5
        self.gomoku.step(
            action=opponent_action, env_indices=env_mask & _opponent_first_mask
        )
        self.black = torch.where(env_mask, ~_opponent_first_mask, self.black)

        tensordict = TensorDict(
            {
                "observation": self.gomoku.get_encoded_board(),
                "action_mask": self.gomoku.get_action_mask(),
            },
            self.batch_size,
            device=self.device,
        )
        return tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action: torch.Tensor = tensordict.get("action")
        episode_len = self.gomoku.move_count.clone()  # (E,)

        # obs1: torch.Tensor = tensordict.get("observation").bool()
        # obs2 = self.gomoku.get_encoded_board().bool()
        # eq = (obs1 == obs2).flatten(start_dim=1).all(dim=-1)
        # try:
        #     assert eq.all()
        # except AssertionError:
        #     print("GG")
        #     exit()

        win, illegal = self.gomoku.step(action=action)

        assert not illegal.any()

        self.gomoku.reset(env_ids=win)
        episode_len = torch.where(
            self.gomoku.move_count == 0, episode_len, self.gomoku.move_count
        )  # (E,)

        opponent_tensordict = TensorDict(
            {
                "observation": self.gomoku.get_encoded_board(),
                "action_mask": self.gomoku.get_action_mask(),
            },
            self.batch_size,
            device=self.device,
        )
        with torch.no_grad():
            opponent_tensordict = self.opponent_policy(opponent_tensordict)
        opponent_action = opponent_tensordict.get("action")

        # if the environment has been reset, the opponent can never win or make an illegal move at the first step
        # so opponent_win/illegal is nonzero only if win/illegal is nonzero

        # UPDATE: if the game is over, the opponent will not make a move

        env_indices = ~win
        opponent_win, opponent_illegal = self.gomoku.step(
            action=opponent_action, env_indices=env_indices
        )

        assert not opponent_illegal.any()

        self.gomoku.reset(env_ids=opponent_win)
        episode_len = torch.where(
            self.gomoku.move_count == 0, episode_len, self.gomoku.move_count
        )  # (E,)

        done = win | opponent_win

        black_win_rate = torch.zeros(
            self.gomoku.num_envs, 2, device=self.device, dtype=torch.bool
        )
        black_win_rate[:, 0] = win
        black_win_rate[:, 1] = self.black
        white_win_rate = torch.zeros(
            self.gomoku.num_envs, 2, device=self.device, dtype=torch.bool
        )
        white_win_rate[:, 0] = win
        white_win_rate[:, 1] = ~self.black

        reward = win.float() - opponent_win.float()

        tensordict = TensorDict({}, self.batch_size, device=self.device)
        tensordict.update(
            {
                "done": done,
                "observation": self.gomoku.get_encoded_board(),
                "action_mask": self.gomoku.get_action_mask(),
                "reward": reward,
                "stats": {
                    "episode_len": episode_len,
                    "win": win,
                    "black_win_rate": black_win_rate,
                    "white_win_rate": white_win_rate,
                },
            }
        )
        return tensordict


def _action_to_xy(action: torch.Tensor, board_size: int):
    action = action.item()
    y = action % board_size
    x = action // board_size
    return f"({x:02d},{y:02d})"


class GomokuEnv:
    def __init__(
        self,
        num_envs: int,
        board_size: int,
        device: DEVICE_TYPING = None,
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
        try:
            assert not illegal.any()
        except AssertionError as e:
            illegal_ids = illegal.nonzero()[0].tolist()
            illegal_id = illegal_ids[0]

            print(illegal_ids)
            self.gomoku._debug_info(illegal_id)
            obs: torch.Tensor = tensordict.get("observation")[illegal_id].long()
            action_mask: torch.Tensor = tensordict.get("action_mask")[illegal_id].long()
            action_mask = action_mask.view(obs.shape[1:])
            print(obs)
            print(action_mask)
            print(_action_to_xy(action[illegal_id], self.board_size))
            print(tensordict["probs"][illegal_id])
            raise e
        done = win
        tensordict = TensorDict({}, self.batch_size, device=self.device)
        tensordict.update(
            {
                "observation": self.gomoku.get_encoded_board(),
                "action_mask": self.gomoku.get_action_mask(),
                "done": done,
                # reward is calculated later
                "stats": {
                    "episode_len": episode_len,
                    "win": win,
                },
            }
        )
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
    ) -> tuple[TensorDict, TensorDict]:
        with torch.no_grad():
            tensordict_t = player_black(tensordict_t)
        tensordict_t_plus_1 = self._step_and_maybe_reset(tensordict=tensordict_t)

        # if a player wins at time t, its opponent cannot win immediately after reset
        reward_white: torch.Tensor = (
            tensordict_t.get("done").float() - tensordict_t_plus_1.get("done").float()
        ).unsqueeze(-1)

        transition_white: TensorDict = tensordict_t_minus_1.select(
            "observation",
            "action_mask",
            "action",
            "sample_log_prob",
            strict=False,
        )
        transition_white.set(
            "next",
            tensordict_t_plus_1.select("observation", "action_mask"),
        )
        transition_white.set(("next", "reward"), reward_white)
        done_white = tensordict_t_plus_1["done"] | tensordict_t["done"]
        transition_white.set(("next", "done"), done_white)
        transition_white = transition_white[~tensordict_t_minus_1["done"]]

        with torch.no_grad():
            tensordict_t_plus_1 = player_white(tensordict_t_plus_1)
        # if player_black wins at t (and the env is reset), the env doesn't take a step at t+1
        # this makes no difference to player_black's transition from t to t+2
        # but player_white's transition from t-1 to t+1 is invalid where tensordict_t_minus_1['done']==True
        tensordict_t_plus_2 = self._step_and_maybe_reset(
            tensordict_t_plus_1, env_indices=~tensordict_t_plus_1.get("done")
        )

        reward_black: torch.Tensor = (
            tensordict_t_plus_1.get("done").float()
            - tensordict_t_plus_2.get("done").float()
        ).unsqueeze(-1)

        transition_black: TensorDict = tensordict_t.select(
            "observation",
            "action_mask",
            "action",
            "sample_log_prob",
            strict=False,
        )
        transition_black.set(
            "next", tensordict_t_plus_2.select("observation", "action_mask")
        )
        transition_black.set(("next", "reward"), reward_black)
        done_black = tensordict_t_plus_1["done"] | tensordict_t_plus_2["done"]
        transition_black.set(("next", "done"), done_black)

        return (
            transition_black,
            transition_white,
            tensordict_t_plus_1,
            tensordict_t_plus_2,
        )

    @torch.no_grad
    def _rollout(
        self,
        max_steps: int,
        player_black: _policy_t,
        player_white: _policy_t,
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

        buffer_black = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=max_steps * self.num_envs),
            sampler=SamplerWithoutReplacement(drop_last=False),
            batch_size=self.num_envs,
        )
        buffer_white = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=max_steps * self.num_envs),
            sampler=SamplerWithoutReplacement(drop_last=False),
            batch_size=self.num_envs,
        )

        info = {}

        for i in range(max_steps):
            (
                transition_black,
                transition_white,
                tensordict_t_minus_1,
                tensordict_t,
            ) = self._round(
                tensordict_t_minus_1, tensordict_t, player_black, player_white
            )

            buffer_black.extend(transition_black)
            if len(transition_white) > 0:
                buffer_white.extend(transition_white)

        return buffer_black, buffer_white, info

    def rollout(
        self,
        episode_len: int,
        player_black: _policy_t,
        player_white: _policy_t,
    ):
        start = time.perf_counter()
        r = self._rollout(
            max_steps=episode_len, player_black=player_black, player_white=player_white
        )
        end = time.perf_counter()
        self._fps = (episode_len * 2 * self.num_envs) / (end - start)
        return r
