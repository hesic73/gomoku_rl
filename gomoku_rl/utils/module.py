import torch
import torch.nn as nn


def init_params(m: nn.Module):
    for p in m.parameters():
        if len(p.data.shape) > 1:
            nn.init.xavier_uniform_(p)


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())


# The architecture described in https://arxiv.org/pdf/1809.10595.pdf
# https://github.com/PolyKen/15_by_15_AlphaGomoku/blob/demo/AlphaGomoku/network/network.py
# It seems in the paper the residual tower has 2 residual blocks
# but in the repository the residual tower has 3 residual blocks
# Besides, action masking is added


class ResidualBlock(nn.Module):
    def __init__(self, num_channels: int, track_running_stats: bool = True) -> None:
        super().__init__()
        self.cnn_0 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
        )
        self.bn_0 = nn.LazyBatchNorm2d(track_running_stats=track_running_stats)
        self.cnn_1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
        )
        self.bn_1 = nn.LazyBatchNorm2d(track_running_stats=track_running_stats)

    def forward(self, x: torch.Tensor):
        x_shortcut = x
        x = self.cnn_0(x)
        x = self.bn_0(x)
        x = nn.functional.relu(x)
        x = self.cnn_1(x)
        x = self.bn_1(x)
        x = nn.functional.relu(x)
        x = x + x_shortcut
        x = nn.functional.relu(x)
        return x


class ResidualTower(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_channels: int,
        num_residual_blocks: int,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.cnn = nn.Conv2d(
            in_channels=in_channels, out_channels=num_channels, kernel_size=3, padding=1
        )
        self.bn = nn.LazyBatchNorm2d(track_running_stats=track_running_stats)

        tmp = [
            ResidualBlock(
                num_channels=num_channels, track_running_stats=track_running_stats
            )
            for _ in range(num_residual_blocks)
        ]
        self.layers = nn.Sequential(*tmp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])

        x = self.cnn(x)
        x = self.bn(x)
        x = nn.functional.relu(x)
        x = self.layers(x)

        x = x.reshape(*batch_shape, *x.shape[1:])
        return x


class _PolicyHead(nn.Module):
    def __init__(self, out_features: int, num_channels: int) -> None:
        super().__init__()
        self.cnn = nn.Conv2d(in_channels=num_channels, out_channels=2, kernel_size=1)
        self.bn = nn.LazyBatchNorm2d()
        self.linear = nn.LazyLinear(out_features=out_features)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_shape = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        if mask is not None:
            mask = mask.reshape(-1, *mask.shape[-1:])

        x = self.cnn(x)
        x = self.bn(x)
        x = nn.functional.relu(x)
        x = torch.flatten(x, start_dim=-3)
        x = self.linear(x)
        if mask is not None:
            x = torch.where(mask == 0, -float("inf"), x)

        x = x.reshape(*batch_shape, *x.shape[1:])
        return x


class PolicyHead(_PolicyHead):
    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = super().forward(x, mask)
        x = nn.functional.softmax(x, dim=-1)
        return x


class ValueHead(nn.Module):
    def __init__(
        self,
        num_channels: int,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.cnn = nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=1)
        self.bn = nn.LazyBatchNorm1d(track_running_stats=track_running_stats)
        self.linear_0 = nn.LazyLinear(out_features=32)
        self.linear_1 = nn.LazyLinear(out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])

        # Note: the order is different from the original implementation(bn,relu,flatten)
        # it seems when using vmap and nn.Conv.out_channels=1,
        # it will throw a RuntimeError
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=-3)
        x = self.bn(x)
        x = nn.functional.relu(x)
        x = self.linear_0(x)
        x = nn.functional.relu(x)
        x = self.linear_1(x)

        x = x.reshape(*batch_shape, *x.shape[1:])
        return x


class ActorNet(nn.Module):
    def __init__(
        self,
        residual_tower: ResidualTower,
        out_features: int,
        num_channels: int = 32,
    ) -> None:
        super().__init__()
        self.residual_tower: nn.Module = residual_tower
        self.policy_head: nn.Module = PolicyHead(
            out_features=out_features,
            num_channels=num_channels,
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        embedding = self.residual_tower(x)
        probs: torch.Tensor = self.policy_head(embedding, mask)
        return probs


class ValueNet(nn.Module):
    def __init__(
        self,
        residual_tower: ResidualTower,
        num_channels: int = 32,
    ) -> None:
        super().__init__()

        self.residual_tower = residual_tower
        self.value_head = ValueHead(num_channels=num_channels)

    def forward(self, x: torch.Tensor):
        x = self.residual_tower(x)
        x = self.value_head(x)
        return x


class MyDuelingCnnDQNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_features: int,
        num_residual_blocks: int = 3,
        num_channels: int = 32,
    ) -> None:
        super().__init__()
        self.features = ResidualTower(
            in_channels=in_channels,
            num_channels=num_channels,
            num_residual_blocks=num_residual_blocks,
        )
        self.advantage = _PolicyHead(
            out_features=out_features, num_channels=num_channels
        )
        self.value = _PolicyHead(out_features=out_features, num_channels=num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)
