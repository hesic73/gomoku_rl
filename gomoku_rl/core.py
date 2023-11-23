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


class Gomoku:
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

        self.last_move: torch.Tensor = -torch.ones(
            num_envs, dtype=torch.long, device=self.device
        )

    def to(self, device: _device_t):
        self.board.to(device=device)
        self.done.to(device=device)
        self.turn.to(device=device)
        self.move_count.to(device=device)
        self.last_move.to(device=device)
        return self

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """_summary_

        Args:
            env_ids (torch.Tensor): (#,)
        """
        if env_ids is None:
            self.board.zero_()
            self.done.zero_()
            self.turn.zero_()
            self.move_count.zero_()
            self.last_move.fill_(-1)
        else:
            self.board[env_ids] = 0
            self.done[env_ids] = False
            self.turn[env_ids] = 0
            self.move_count[env_ids] = 0
            self.last_move[env_ids] = -1

    def step(
        self, action: torch.Tensor, env_indices: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            action (torch.Tensor): (E,) x_i*board_size+y_i
            env_indices (Optional[torch.Tensor]): (E,)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: done (E,), illegal (E,)

            If `env_indices` is not None, the elements of `done` and `illegal` are 0 where `env_indices`==0

        """

        if env_indices is None:
            env_indices = torch.ones_like(action, dtype=torch.bool)

        # if action isn't in [0,{board_size}^2), the indexing will crash
        x = action // self.board_size
        y = action % self.board_size

        values_on_board = self.board[
            torch.arange(self.num_envs, device=self.device), x, y
        ]  # (E,)

        nop = (values_on_board != 0) | (~env_indices)  # (E,)

        piece = turn_to_piece(self.turn)
        self.board[torch.arange(self.num_envs, device=self.device), x, y] = torch.where(
            nop, values_on_board, piece
        )
        self.move_count = self.move_count + torch.logical_not(nop).long()

        # F.conv2d doesn't support LongTensor on CUDA. So we use float.
        board_one_side = (self.board == piece.unsqueeze(-1).unsqueeze(-1)).float()
        self.done = compute_done(board_one_side) | (
            self.move_count == self.board_size * self.board_size
        )

        self.turn = (self.turn + torch.logical_not(nop).long()) % 2
        self.last_move = torch.where(nop, self.last_move, action)

        return self.done & env_indices, nop & env_indices

    def get_encoded_board(self):
        piece = turn_to_piece(self.turn).unsqueeze(-1).unsqueeze(-1)  # (E,1,1)

        layer1 = (self.board == piece).float()
        layer2 = (self.board == -piece).float()

        last_x = self.last_move // self.board_size  # (E,)
        last_y = self.last_move % self.board_size  # (E,)

        # (1,B)==(E,1)-> (E,B)-> (E,B,1)
        # (1,B)==(E,1)-> (E,B)-> (E,1,B)
        layer3 = (
            (
                torch.arange(self.board_size, device=self.device).unsqueeze(0)
                == last_x.unsqueeze(-1)
            ).unsqueeze(-1)
        ) & (
            (
                torch.arange(self.board_size, device=self.device).unsqueeze(0)
                == last_y.unsqueeze(-1)
            ).unsqueeze(1)
        )  # (E,B,B)
        layer3 = layer3.float()

        # layer4 = (self.turn == 0).float().unsqueeze(-1).unsqueeze(-1)  # (E,1,1)
        # layer4 = layer4.expand(-1, self.board_size, self.board_size)

        output = torch.stack(
            [
                layer1,
                layer2,
                layer3,
                # layer4,
            ],
            dim=1,
        )  # (E,*,B,B)
        return output

    def get_action_mask(self):
        return (self.board == 0).flatten(start_dim=1)

    def is_valid(self, action: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            action (torch.Tensor): (E,)

        Returns:
            torch.Tensor: (E,)
        """
        out_of_range = action < 0 | (action >= self.board_size * self.board_size)
        x = action // self.board_size
        y = action % self.board_size

        values_on_board = self.board[
            torch.arange(self.num_envs, device=self.device), x, y
        ]  # (E,)

        not_empty = values_on_board != 0  # (E,)

        invalid = out_of_range | not_empty

        return ~invalid
