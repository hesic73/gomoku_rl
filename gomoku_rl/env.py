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
from gomoku_rl.utils.policy import _policy_t, uniform_policy

from .core import Gomoku


class GomokuEnvWithOpponent(EnvBase):
    def __init__(
        self,
        num_envs: int,
        board_size: int = 19,
        initial_policy: _policy_t | None = None,
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
            "opponent_win",
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

    def _set_seed(self, seed: int | None):
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
        self.black = torch.where(env_mask, _opponent_first_mask, self.black)

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
                    "opponent_win": opponent_win,
                },
            }
        )
        return tensordict


class GomokuEnv:
    """This environment is incompatible with `torchrl.envs.EnvBase`
    I refered to the implementation in `rlcard.envs.Env`
    """

    def __init__(
        self,
        num_envs: int,
        board_size: int = 19,
        device: DEVICE_TYPING = None,
    ):
        self.batch_size = torch.Size((num_envs,))
        self.device = device
        self.gomoku = Gomoku(num_envs=num_envs, board_size=board_size, device=device)

        self.board_size = board_size

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

    def reset(self, env_ids: torch.Tensor | None = None) -> TensorDict:
        self.gomoku.reset(env_ids=env_ids)
        episode_len = self.gomoku.move_count + 1  # (E,)
        tensordict = TensorDict(
            {
                "observation": self.gomoku.get_encoded_board(),
                "action_mask": self.gomoku.get_action_mask(),
                "stats": {
                    "episode_len": episode_len,
                    "win": torch.zeros(
                        self.batch_size, device=self.device, dtype=torch.bool
                    ),
                    "illegal": torch.zeros(
                        self.batch_size, device=self.device, dtype=torch.bool
                    ),
                },
            },
            self.batch_size,
            device=self.device,
        )
        return tensordict

    def step(self, tensordict: TensorDict) -> TensorDict:
        action: torch.Tensor = tensordict.get("action")
        env_indices: torch.Tensor = tensordict.get("env_indices", None)
        episode_len = self.gomoku.move_count + 1  # (E,)
        win, illegal = self.gomoku.step(action=action, env_indices=env_indices)
        done = win | illegal
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
                    "illegal": illegal,  # with action masking, it should be zero
                },
            }
        )
        return tensordict

    def _round(
        self,
        tensordict_t_minus_1: TensorDict | None,
        tensordict_t: TensorDict,
        player_0: _policy_t,
        player_1: _policy_t,
    ):
        # player_0 下黑棋 player_1 下白旗

        with torch.no_grad():
            tensordict_t = player_0(tensordict_t)
        tensordict_t_plus_1 = self.step(tensordict=tensordict_t)

        done_t_plus_1: torch.Tensor = tensordict_t_plus_1.get("done")  # (E,)
        reset_td = self.reset(done_t_plus_1)
        tensordict_t_plus_1.update(reset_td)  # done的時候DQN實際沒用到observation，所以沒有影響

        # 如果在t时刻是win/illegal会被reset，t+1时刻不可能win/illegal
        reward_t_plus_1: torch.Tensor = (
            tensordict_t.get(("stats", "win")).float()
            - tensordict_t.get(("stats", "illegal")).float()
            - tensordict_t_plus_1.get(("stats", "win")).float()
            + tensordict_t_plus_1.get(("stats", "illegal")).float()
        )

        if tensordict_t_minus_1 is not None:
            transition_player_1: TensorDict = tensordict_t_minus_1.select(
                "observation", "action_mask", "action"
            )
            transition_player_1.set(
                "next", tensordict_t_plus_1.select("observation", "action_mask")
            )
            transition_player_1.set(("next", "reward"), reward_t_plus_1)
        else:
            transition_player_1 = None

        with torch.no_grad():
            tensordict_t_plus_1 = player_1(tensordict_t_plus_1)
        # 如果黑棋下完游戏就结束了，白棋也不下。
        # 但這裡不走對transition是沒有影響的
        tensordict_t_plus_1.set("env_indices", ~done_t_plus_1)
        tensordict_t_plus_2 = self.step(tensordict=tensordict_t_plus_1)
        tensordict_t_plus_1.exclude("env_indices", inplace=True)

        done_t_plus_2 = tensordict_t_plus_2.get("done")
        reset_td = self.reset(done_t_plus_2)
        tensordict_t_plus_2.update(reset_td)

        reward_t_plus_2: torch.Tensor = (
            tensordict_t_plus_1.get(("stats", "win")).float()
            - tensordict_t_plus_1.get(("stats", "illegal")).float()
            - tensordict_t_plus_2.get(("stats", "win")).float()
            + tensordict_t_plus_2.get(("stats", "illegal")).float()
        )

        transition_player_0: TensorDict = tensordict_t.select(
            "observation", "action_mask", "action"
        )
        transition_player_0.set(
            "next", tensordict_t_plus_2.select("observation", "action_mask")
        )
        transition_player_0.set(("next", "reward"), reward_t_plus_2)

        return (
            transition_player_0,
            transition_player_1,
            tensordict_t_plus_1,
            tensordict_t_plus_2,
        )

    @torch.no_grad
    def rollout(
        self,
        max_steps: int,
        player_0: _policy_t,
        player_1: _policy_t,
    ):
        tensordict_t_minus_1 = None
        tensordict_t = self.reset()

        transitions_player_0 = []
        transitions_player_1 = []

        for i in range(max_steps):
            (
                transition_player_0,
                transition_player_1,
                tensordict_t_minus_1,
                tensordict_t,
            ) = self._round(tensordict_t_minus_1, tensordict_t, player_0, player_1)

            transitions_player_0.append(transition_player_0.cpu())
            if transition_player_1 is not None:
                transitions_player_1.append(transition_player_1.cpu())

        transitions_player_0 = torch.stack(transitions_player_0, dim=1)
        transitions_player_1 = torch.stack(transitions_player_1, dim=1)

        return transitions_player_0, transitions_player_1
