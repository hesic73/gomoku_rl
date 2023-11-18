import torch
import abc
from tensordict import TensorDict


class Augmentation(abc.ABC):
    @abc.abstractmethod
    def call(self, board: torch.Tensor) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def inverse(self, board: torch.Tensor) -> torch.Tensor:
        ...


class RotationAugmentation(Augmentation):
    def __init__(self, k: int = 1) -> None:
        self.k = k % 4

    def call(self, board: torch.Tensor):
        return torch.rot90(board, k=self.k, dims=(-2, -1))

    def inverse(self, board: torch.Tensor):
        return torch.rot90(board, k=4 - self.k, dims=(-2, -1))


class HorizontalFlipAugmentation(Augmentation):
    def __init__(self) -> None:
        pass

    def call(self, board: torch.Tensor):
        return torch.flip(board, dims=[-1])

    def inverse(self, board: torch.Tensor):
        return torch.flip(board, dims=[-1])


class VerticalFlipAugmentation(Augmentation):
    def __init__(self) -> None:
        pass

    def call(self, board: torch.Tensor):
        return torch.flip(board, dims=[-2])

    def inverse(self, board: torch.Tensor):
        return torch.flip(board, dims=[-2])


_AUGMENTATIONS=[
    RotationAugmentation(1),
    RotationAugmentation(2),
    RotationAugmentation(3),
]

def _augment_transition(transition:TensorDict,augmentation:Augmentation):
    pass


def augment_transition(transition: TensorDict) -> TensorDict:
    print(transition)
    exit()
