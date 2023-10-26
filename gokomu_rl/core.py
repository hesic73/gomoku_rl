import torch
from typing import Optional
from torch.cuda import _device_t
import torch.nn.functional as F


def compute_done(board: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        board (torch.Tensor): (E,board_size,board_size)
        The value can only be either 0 or 1, indicating the presence of a stone on the board.
        It must be precisely 1 !!!

    Returns:
        torch.Tensor: done (E,)
    """
    assert len(board.shape) == 3 and board.shape[1] == board.shape[2]
    board = board.unsqueeze(1)  # (E,1,B,B)

    kernel_horizontal = (
        torch.tensor([1, 1, 1, 1, 1], device=board.device, dtype=board.dtype)
        .unsqueeze(-1)
        .unsqueeze(0)
        .unsqueeze(0)
    )  # (1,1,5,1)

    kernel_vertical = (
        torch.tensor([1, 1, 1, 1, 1], device=board.device, dtype=board.dtype)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
    )  # (1,1,1,5)

    kernel_diagonal = torch.tensor(
        [
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            [
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
            ],
        ],
        device=board.device,
        dtype=board.dtype,
    ).unsqueeze(
        1
    )  # (2,1,5,5)

    output_horizontal = F.conv2d(input=board, weight=kernel_horizontal)  # (E,1,B-4,B)

    done_horizontal = (output_horizontal.flatten(start_dim=1) > 4.5).any(dim=-1)  # (E,)

    output_vertical = F.conv2d(input=board, weight=kernel_vertical)  # (E,1,B,B-4)

    done_vertical = (output_vertical.flatten(start_dim=1) > 4.5).any(dim=-1)  # (E,)

    output_diagonal = F.conv2d(input=board, weight=kernel_diagonal)  # (E,2,B-4,B-4)

    done_diagonal = (output_diagonal.flatten(start_dim=1) > 4.5).any(dim=-1)  # (E,)

    done = done_horizontal | done_vertical | done_diagonal

    return done


def turn_to_piece(turn: torch.Tensor) -> torch.Tensor:
    piece = torch.where(turn == 0, 1, -1)
    return piece


class Gokomu:
    def __init__(
        self, num_envs: int, board_size: int = 19, device: _device_t = None
    ) -> None:
        assert num_envs > 0
        assert board_size >= 5

        self.num_envs: int = num_envs
        self.board_size: int = board_size
        self.device: _device_t = device
        # board 0 empty 1 black -1 white
        self.board: torch.Tensor = torch.zeros(
            num_envs,
            self.board_size,
            self.board_size,
            device=self.device,
            dtype=torch.long,
        )  # (E,B,B)
        self.done: torch.Tensor = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device
        )
        self.turn: torch.Tensor = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )

        self.move_count: torch.Tensor = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )

    def to(self, device: _device_t):
        self.board.to(device=device)
        self.done.to(device=device)
        self.move_count.to(device=device)
        return self

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """_summary_

        Args:
            env_ids (torch.Tensor): (E,)
        """
        if env_ids is None:
            self.board.zero_()
            self.done.zero_()
            self.turn.zero_()
            self.move_count.zero_()
        else:
            self.board[env_ids] = 0
            self.done[env_ids] = False
            self.turn[env_ids] = 0
            self.move_count[env_ids] = 0

    def _update(self, valid_move: torch.Tensor):
        """_summary_

        Args:
            valid_move (torch.Tensor): (E,)
            turn (torch.Tensor): (E,)
        """
        self.move_count = self.move_count + valid_move.long()
        piece = turn_to_piece(self.turn)
        # F.conv2d doesn't support LongTensor on CUDA. So we use float.
        board_one_side = (self.board == piece.unsqueeze(-1).unsqueeze(-1)).float()
        self.done = compute_done(board_one_side)
        self.turn = (self.turn + valid_move.long()) % 2

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            action (torch.Tensor): (E,1) x_i*board_size+y_i

        Returns:
            tuple[torch.Tensor, torch.Tensor]: done (E,), invalid (E,)

        Warnings:
            No check on `action`'s value. If `action` is invalid, `Gokomu` doesn't specify its behavior, and it's the user's duty to ensure it.
        """
        x = action[:, 0] // self.board_size
        y = action[:, 0] % self.board_size

        values_on_board = self.board[
            torch.arange(self.num_envs, device=self.device), x, y
        ]  # (E,)

        not_empty = values_on_board != 0  # (E,)

        piece = turn_to_piece(self.turn)
        self.board[torch.arange(self.num_envs, device=self.device), x, y] = torch.where(
            not_empty, values_on_board, piece
        )

        self._update(torch.logical_not(not_empty))

        return self.done, not_empty

    def get_board_state(self):
        return self.board.flatten(start_dim=1)

    def get_turn(self):
        return torch.where(self.turn == 0, 1.0, -1.0).unsqueeze(-1)


Wuziqi = Gokomu
