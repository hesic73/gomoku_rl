import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """policy-value network module"""

    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(
            4 * board_width * board_height, board_width * board_height
        )
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  # (E,C,B,B)

        # action policy layers
        x_act = F.relu(self.act_conv1(x))  # （E,4,8,8)
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)  # (E,4*B*B)
        print(f"x_act:{x_act.shape}")
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)  # (E,B*B)
        print(f"x_act:{x_act.shape}")
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))  # (E,1)
        return x_act, x_val


def make_mlp(n_channels: list[int]):
    assert len(n_channels) >= 2
    tmp = []

    for i in range(len(n_channels) - 1):
        tmp.append(
            nn.Linear(
                in_features=n_channels[i],
                out_features=n_channels[i + 1],
            )
        )
        tmp.append(nn.BatchNorm1d(n_channels[i + 1]))
        tmp.append(nn.ReLU())

    return nn.Sequential(*tmp)


def make_conv_layers(n_channels: list[int]):
    assert len(n_channels) >= 2
    tmp = []

    for i in range(len(n_channels) - 1):
        tmp.append(
            nn.Conv2d(
                in_channels=n_channels[i],
                out_channels=n_channels[i + 1],
                kernel_size=3,
                padding=1,
            )
        )
        tmp.append(nn.BatchNorm2d(n_channels[i + 1]))
        tmp.append(nn.ReLU())

    return nn.Sequential(*tmp)


