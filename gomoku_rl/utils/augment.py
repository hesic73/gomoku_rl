import torch
import abc
from tensordict import TensorDict


class Transform(abc.ABC):
    @abc.abstractmethod
    def map_board(self, board: torch.Tensor) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def inverse_map_board(self, board: torch.Tensor) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def map_index(self, index: torch.Tensor, board_size: int) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def inverse_map_index(self, index: torch.Tensor, board_size: int) -> torch.Tensor:
        ...


class Identity(Transform):
    def map_board(self, board: torch.Tensor) -> torch.Tensor:
        return board

    def inverse_map_board(self, board: torch.Tensor) -> torch.Tensor:
        return board

    def map_index(self, index: torch.Tensor, board_size: int) -> torch.Tensor:
        return index

    def inverse_map_index(self, index: torch.Tensor, board_size: int) -> torch.Tensor:
        return index


class Rotation(Transform):
    def __init__(self, k: int = 1) -> None:
        self.k = k % 4
        assert self.k in (1, 2, 3)

    def map_board(self, board: torch.Tensor):
        return torch.rot90(board, k=self.k, dims=(-2, -1))

    def inverse_map_board(self, board: torch.Tensor):
        return torch.rot90(board, k=4 - self.k, dims=(-2, -1))

    def map_index(self, index: torch.Tensor, board_size: int) -> torch.Tensor:
        x = index // board_size
        y = index % board_size
        if self.k == 1:
            index = y * board_size + (board_size - x - 1)
        elif self.k == 2:
            index = (board_size - x - 1) * board_size + (board_size - y - 1)
        elif self.k == 3:
            index = (board_size - y - 1) * board_size + x
        else:
            raise RuntimeError
        return index

    def inverse_map_index(self, index: torch.Tensor, board_size: int) -> torch.Tensor:
        x = index // board_size
        y = index % board_size
        if self.k == 1:
            index = (board_size - y - 1) * board_size + x
        elif self.k == 2:
            index = (board_size - x - 1) * board_size + (board_size - y - 1)
        elif self.k == 3:
            index = y * board_size + (board_size - x - 1)
        else:
            raise RuntimeError
        return index


class HorizontalFlip(Transform):
    def __init__(self) -> None:
        pass

    def map_board(self, board: torch.Tensor):
        return torch.flip(board, dims=[-1])

    def inverse_map_board(self, board: torch.Tensor):
        return torch.flip(board, dims=[-1])

    def map_index(self, index: torch.Tensor, board_size: int) -> torch.Tensor:
        x = index // board_size
        y = index % board_size
        index = x * board_size + (board_size - y - 1)
        return index

    def inverse_map_index(self, index: torch.Tensor, board_size: int) -> torch.Tensor:
        x = index // board_size
        y = index % board_size
        index = x * board_size + (board_size - y - 1)
        return index


class VerticalFlip(Transform):
    def __init__(self) -> None:
        pass

    def map_board(self, board: torch.Tensor):
        return torch.flip(board, dims=[-2])

    def inverse_map_board(self, board: torch.Tensor):
        return torch.flip(board, dims=[-2])

    def map_index(self, index: torch.Tensor, board_size: int) -> torch.Tensor:
        x = index // board_size
        y = index % board_size
        index = (board_size - x - 1) * board_size + y
        return index

    def inverse_map_index(self, index: torch.Tensor, board_size: int) -> torch.Tensor:
        x = index // board_size
        y = index % board_size
        index = (board_size - x - 1) * board_size + y
        return index


class DiagonalFlip(Transform):
    def __init__(self) -> None:
        pass

    def map_board(self, board: torch.Tensor):
        return torch.transpose(board, dim0=-2, dim1=-1)

    def inverse_map_board(self, board: torch.Tensor):
        return torch.transpose(board, dim0=-2, dim1=-1)

    def map_index(self, index: torch.Tensor, board_size: int) -> torch.Tensor:
        x = index // board_size
        y = index % board_size
        index = y * board_size + x
        return index

    def inverse_map_index(self, index: torch.Tensor, board_size: int) -> torch.Tensor:
        x = index // board_size
        y = index % board_size
        index = y * board_size + x
        return index


class AntiDiagonalFlip(Transform):
    def __init__(self) -> None:
        pass

    def map_board(self, board: torch.Tensor):
        board = torch.transpose(board, dim0=-2, dim1=-1)
        return torch.rot90(board, k=2, dims=(-2, -1))

    def inverse_map_board(self, board: torch.Tensor):
        board = torch.transpose(board, dim0=-2, dim1=-1)
        return torch.rot90(board, k=2, dims=(-2, -1))

    def map_index(self, index: torch.Tensor, board_size: int) -> torch.Tensor:
        x = index // board_size
        y = index % board_size
        index = (board_size - y - 1) * board_size + (board_size - x - 1)
        return index

    def inverse_map_index(self, index: torch.Tensor, board_size: int) -> torch.Tensor:
        x = index // board_size
        y = index % board_size
        index = (board_size - y - 1) * board_size + (board_size - x - 1)
        return index


_TRANSFORMS = [
    Identity(),
    Rotation(1),
    Rotation(2),
    Rotation(3),
    HorizontalFlip(),
    VerticalFlip(),
    DiagonalFlip(),
    AntiDiagonalFlip(),
]


def _augment(transition: TensorDict, transform: Transform):
    pass


def augment_transition(transition: TensorDict) -> TensorDict:
    print(transition)
    exit()
