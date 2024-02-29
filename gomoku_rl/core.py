import torch
import torch.nn.functional as F


def compute_done(
    board: torch.Tensor,
    kernel_horizontal: torch.Tensor,
    kernel_vertical: torch.Tensor,
    kernel_diagonal: torch.Tensor,
) -> torch.Tensor:
    """Determines if any game has been won in a batch of Gomoku boards.

    Checks for a winning sequence of stones horizontally, vertically, and diagonally.

    Args:
        board (torch.Tensor): The game boards, shaped (E, B, B), with E being the number of environments, 
                              and B being the board size. Values are 0 (empty), 1 (black stone), or -1 (white stone).
        kernel_horizontal (torch.Tensor): Horizontal detection kernel, shaped (1, 1, 5, 1).
        kernel_vertical (torch.Tensor): Vertical detection kernel, shaped (1, 1, 1, 5).
        kernel_diagonal (torch.Tensor): Diagonal detection kernels, shaped (2, 1, 5, 5), for both diagonals.

    Returns:
        torch.Tensor: Boolean tensor shaped (E,), indicating if the game is won (True) in each environment.
    """

    board = board.unsqueeze(1)  # (E,1,B,B)

    output_horizontal = F.conv2d(
        input=board, weight=kernel_horizontal)  # (E,1,B-4,B)

    done_horizontal = (output_horizontal.flatten(
        start_dim=1) > 4.5).any(dim=-1)  # (E,)

    output_vertical = F.conv2d(
        input=board, weight=kernel_vertical)  # (E,1,B,B-4)

    done_vertical = (output_vertical.flatten(
        start_dim=1) > 4.5).any(dim=-1)  # (E,)

    output_diagonal = F.conv2d(
        input=board, weight=kernel_diagonal)  # (E,2,B-4,B-4)

    done_diagonal = (output_diagonal.flatten(
        start_dim=1) > 4.5).any(dim=-1)  # (E,)

    done = done_horizontal | done_vertical | done_diagonal

    return done


class Gomoku:
    def __init__(
        self, num_envs: int, board_size: int = 15, device=None
    ):
        """Initializes a batch of parallel Gomoku game environments.

        Args:
            num_envs (int): Number of parallel game environments.
            board_size (int, optional): Side length of the square game board. Defaults to 15.
            device: Torch device on which the tensors are allocated. Defaults to None (CPU).
        """
        assert num_envs > 0
        assert board_size >= 5

        self.num_envs: int = num_envs
        self.board_size: int = board_size
        self.device = device
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

        self.kernel_horizontal = (
            torch.tensor([1, 1, 1, 1, 1], device=self.device,
                         dtype=torch.float)
            .unsqueeze(-1)
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1,1,5,1)

        self.kernel_vertical = (
            torch.tensor([1, 1, 1, 1, 1], device=self.device,
                         dtype=torch.float)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1,1,1,5)

        self.kernel_diagonal = torch.tensor(
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
            device=self.device,
            dtype=torch.float,
        ).unsqueeze(
            1
        )  # (2,1,5,5)

    def to(self, device):
        """Transfers all internal tensors to the specified device.

        Args:
            device: The target device.

        Returns:
            self: The instance with its tensors moved to the new device.
        """
        self.board.to(device=device)
        self.done.to(device=device)
        self.turn.to(device=device)
        self.move_count.to(device=device)
        self.last_move.to(device=device)
        return self

    def reset(self, env_indices: torch.Tensor | None = None):
        """Resets specified game environments to their initial state.

        Args:
            env_indices (torch.Tensor | None, optional): Indices of environments to reset. Resets all if None. Defaults to None.
        """
        if env_indices is None:
            self.board.zero_()
            self.done.zero_()
            self.turn.zero_()
            self.move_count.zero_()
            self.last_move.fill_(-1)
        else:
            self.board[env_indices] = 0
            self.done[env_indices] = False
            self.turn[env_indices] = 0
            self.move_count[env_indices] = 0
            self.last_move[env_indices] = -1

    def step(
        self, action: torch.Tensor, env_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs actions in specified environments and updates their states based on the provided action tensor. If an environment mask is provided, only the environments corresponding to `True` values in the mask are updated; otherwise, all environments are updated.

        Args:
            action (torch.Tensor): 1D positions to place a stone, one per environment. Shape: (E,)
            env_indices (torch.Tensor | None, optional): Boolean mask to select environments for updating. If `None`, updates all. Shape should match environments.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
                - done_statuses: Boolean tensor with `True` where games ended.
                - invalid_actions: Boolean tensor with `True` for invalid actions in environments.
        """

        if env_mask is None:
            env_mask = torch.ones_like(action, dtype=torch.bool)

        board_1d_view = self.board.view(self.num_envs, -1)

        values_on_board = board_1d_view[
            torch.arange(self.num_envs, device=self.device),
            action,
        ]  # (E,)

        nop = (values_on_board != 0) | (~env_mask)  # (E,)
        inc = torch.logical_not(nop).long()  # (E,)
        piece = torch.where(self.turn == 0, 1, -1)
        board_1d_view[
            torch.arange(self.num_envs, device=self.device), action
        ] = torch.where(nop, values_on_board, piece)
        self.move_count = self.move_count + inc

        # F.conv2d doesn't support LongTensor on CUDA. So we use float.
        board_one_side = (
            self.board == piece.unsqueeze(-1).unsqueeze(-1)).float()
        self.done = compute_done(
            board_one_side,
            self.kernel_horizontal,
            self.kernel_vertical,
            self.kernel_diagonal,
        ) | (self.move_count == self.board_size * self.board_size)

        self.turn = (self.turn + inc) % 2
        self.last_move = torch.where(nop, self.last_move, action)

        return self.done & env_mask, nop & env_mask

    def get_encoded_board(self) -> torch.Tensor:
        """Encodes the current board state into a tensor format suitable for neural network input.

        Returns:
            torch.Tensor: Encoded board state, shaped (E, 3, B, B), with separate channels for the current player's stones, the opponent's stones, and the last move.
        """
        piece = torch.where(self.turn == 0, 1, -
                            1).unsqueeze(-1).unsqueeze(-1)  # (E,1,1)

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

    def get_action_mask(self) -> torch.Tensor:
        """Generates a mask indicating valid actions for each environment.

        Returns:
            torch.Tensor: Action mask tensor, shaped (E, B*B), with 1s for valid actions and 0s otherwise.
        """
        return (self.board == 0).flatten(start_dim=1)

    def is_valid(self, action: torch.Tensor) -> torch.Tensor:
        """Checks the validity of the specified actions in each environment.

        Args:
            action (torch.Tensor): Actions to be checked, linearly indexed.

        Returns:
            torch.Tensor: Boolean tensor, shaped (E,), indicating the validity of each action.
        """
        out_of_range = action < 0 | (
            action >= self.board_size * self.board_size)
        x = action // self.board_size
        y = action % self.board_size

        values_on_board = self.board[
            torch.arange(self.num_envs, device=self.device), x, y
        ]  # (E,)

        not_empty = values_on_board != 0  # (E,)

        invalid = out_of_range | not_empty

        return ~invalid
