import torch
import enum
from tensordict import TensorDict


class Type(enum.Enum):
    black = enum.auto()
    white = enum.auto()
    mixed = enum.auto()


EPS: float = 1e-8


def assert_tensor_1d_all(t: torch.Tensor):
    try:
        assert t.all()
    except AssertionError as e:
        idxs = (~t).nonzero().squeeze(0)
        print(idxs, len(idxs) / len(t))
        raise e


def assert_observation(observation: torch.Tensor, type: Type):
    assert not ((observation[:, 0] > EPS) & (observation[:, 1] > EPS)).any()
    tmp = (observation[:, 0] > EPS).long().sum(-1).sum(-1) - (
        observation[:, 1] > EPS
    ).long().sum(-1).sum(-1)
    if type == Type.black:
        assert_tensor_1d_all(tmp == 0)
    elif type == Type.white:
        assert_tensor_1d_all(tmp == -1)
    elif type == Type.mixed:
        assert_tensor_1d_all((tmp == 0) | (tmp == -1))


def assert_layer_transition(
    layer: torch.Tensor, next_layer: torch.Tensor, done: torch.Tensor
):
    same = (layer > EPS) & (next_layer > EPS)
    assert_tensor_1d_all((same == (layer > EPS)).all(-1).all(-1) | done)
    assert_tensor_1d_all(
        (layer.long().sum(-1).sum(-1) + 1 ==
         next_layer.long().sum(-1).sum(-1)) | done
    )


def no_nan_in_tensordict(tensordict: TensorDict):
    for n, t in tensordict.items(include_nested=True, leaves_only=True):
        if not isinstance(t, torch.Tensor):
            continue
        if torch.isnan(t).any():
            return False
    return True


def assert_transition(tensordict: TensorDict, type: Type):
    assert no_nan_in_tensordict(tensordict)
    done: torch.Tensor = tensordict["next", "done"]
    action: torch.Tensor = tensordict["action"]
    observation: torch.Tensor = tensordict["observation"]  # (E,3,B,B)
    next_observation: torch.Tensor = tensordict["next", "observation"]

    assert_observation(observation[~done], type=type)
    assert_observation(next_observation[~done], type=type)

    assert_layer_transition(observation[:, 0], next_observation[:, 0], done)
    assert_layer_transition(observation[:, 1], next_observation[:, 1], done)

    board_size: int = observation.shape[-1]
    num_envs: int = observation.shape[0]
    device = observation.device
    layer1 = observation[:, 0]
    x = action // board_size
    y = action % board_size
    assert_tensor_1d_all(
        (layer1[torch.arange(num_envs, device=device), x, y] < EPS))
    layer1 = layer1.clone()
    layer1[torch.arange(num_envs, device=device), x, y] = 1.0
    assert_tensor_1d_all(
        torch.isclose(layer1, next_observation[:, 0]).all(-1).all(-1) | done
    )
