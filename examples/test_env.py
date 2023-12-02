from gomoku_rl.env import GomokuEnv
from gomoku_rl.utils.policy import uniform_policy
from gomoku_rl.utils.misc import no_nan_in_tensordict, set_seed
from tensordict import TensorDict
import torch
import random
import numpy as np
from tqdm import tqdm
import enum


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
        (layer.long().sum(-1).sum(-1) + 1 == next_layer.long().sum(-1).sum(-1)) | done
    )


def assert_transition(tensordict: TensorDict, type: Type):
    assert no_nan_in_tensordict(tensordict)
    done: torch.Tensor = tensordict["next", "done"]
    action: torch.Tensor = tensordict["action"]
    observation: torch.Tensor = tensordict["observation"]  # (E,3,B,B)
    next_observation: torch.Tensor = tensordict["next", "observation"]

    assert_observation(observation, type=type)
    assert_observation(next_observation[~done], type=type)

    assert_layer_transition(observation[:, 0], next_observation[:, 0], done)
    assert_layer_transition(observation[:, 1], next_observation[:, 1], done)

    board_size: int = observation.shape[-1]
    num_envs: int = observation.shape[0]
    device = observation.device
    layer1 = observation[:, 0]
    x = action // board_size
    y = action % board_size
    assert_tensor_1d_all((layer1[torch.arange(num_envs, device=device), x, y] < EPS))
    layer1 = layer1.clone()
    layer1[torch.arange(num_envs, device=device), x, y] = 1.0
    assert_tensor_1d_all(
        torch.isclose(layer1, next_observation[:, 0]).all(-1).all(-1) | done
    )


def main():
    device = "cuda:0"
    num_envs = 2048
    board_size = 15
    seed = 1234
    set_seed(seed)
    # torch.use_deterministic_algorithms(True)
    env = GomokuEnv(num_envs=num_envs, board_size=board_size, device=device)
    transitions_black, transitions_white, info = env.rollout(
        100,
        player_black=uniform_policy,
        player_white=uniform_policy,
        batch_size=num_envs,
        augment=False,
    )
    print(f"FPS:{env._fps:.2e}")
    for transition in tqdm(transitions_black, total=len(transitions_black) // num_envs):
        assert_transition(transition, type=Type.black)

    for transition in tqdm(transitions_white, total=len(transitions_white) // num_envs):
        assert_transition(transition, type=Type.white)

    transitions, info = env.rollout_fixed_opponent(
        100,
        player=uniform_policy,
        opponent=uniform_policy,
        batch_size=num_envs,
    )
    for transition in tqdm(transitions, total=len(transitions) // num_envs):
        assert_transition(transition, type=Type.mixed)


if __name__ == "__main__":
    main()
