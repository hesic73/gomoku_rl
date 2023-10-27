import torch
import torch.nn as nn
from torchrl.data.tensor_specs import TensorSpec, DiscreteTensorSpec
from torchrl.modules import DuelingCnnDQNet


if __name__ == "__main__":
    board_size = 9
    num_envs = 128
    device = "cuda"
    cnn_kwargs = {
        "num_cells": [32, 64, 128, 4],
        "kernel_sizes": [3, 3, 3, 1],
        "strides": [1, 1, 1, 1],
        "paddings": [1, 1, 1, 0],
        "activation_class": nn.ReLU,
        # This can be used to reduce the size of the last layer of the CNN
        # "squeeze_output": True,
        # "aggregator_class": nn.AdaptiveAvgPool2d,
        # "aggregator_kwargs": {"output_size": (1, 1)},
    }
    mlp_kwargs = {
        "depth": 2,
        "num_cells": [
            64,
            64,
        ],
        "activation_class": nn.ReLU,
    }
    net = DuelingCnnDQNet(board_size * board_size, 1, cnn_kwargs, mlp_kwargs).to(device)
    # net = nn.Sequential(
    #     nn.Flatten(start_dim=1),
    #     nn.LazyLinear(board_size * board_size).to(device),
    # )
    observation = torch.randn(num_envs, 4, board_size, board_size, device=device)
    q = net(observation)
    print(q.shape)
