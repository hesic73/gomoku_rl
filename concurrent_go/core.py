import torch
from typing import Optional
from torch.cuda import _device_t


class GoGame:
    def __init__(self, n: int, board_size: int = 19, device: _device_t = None) -> None:
        assert n > 0
        assert board_size > 0

        self.n: int = n
        self.board_size: int = board_size
        self.device: _device_t = device
        # board 0 empty 1 black -1 white
        self.board: torch.Tensor = torch.zeros(
            n, self.board_size, self.board_size, device=self.device, dtype=torch.long
        )  # (E,B,B)
        self.done: torch.Tensor = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.turn: torch.Tensor = torch.zeros(n, dtype=torch.long, device=self.device)

        self.neighbors_delta = torch.tensor(
            [[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=torch.long, device=self.device
        )  # (4,2)

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """_summary_

        Args:
            env_ids (torch.Tensor): (E,)
        """
        if env_ids is None:
            self.board.zero_()
            self.done.zero_()
            self.turn.zero_()
        else:
            self.board[env_ids] = 0
            self.done[env_ids] = False
            self.turn[env_ids] = 0

    def step(self, action: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            action (torch.Tensor): (E,3) x,y,black/white/pass 1/-1/0

        Returns:
            torch.Tensor: (E,) done? invalid?
        """

        values_on_board = self.board[
            torch.arange(self.n, device=self.device), action[:, 0], action[:, 1]
        ]  # (E,)

        invalid_not_empty = (values_on_board != 0) & (action[:, 2] != 0)  # (E,)
        self.board[
            torch.arange(self.n, device=self.device), action[:, 0], action[:, 1]
        ] = action[:, 2]
