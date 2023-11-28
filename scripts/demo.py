import hydra
from omegaconf import DictConfig, OmegaConf
from gomoku_rl import CONFIG_PATH
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
import logging
from gomoku_rl.policy import get_policy
from torchrl.data.tensor_specs import (
    DiscreteTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    BinaryDiscreteTensorSpec,
)
import torch
from PyQt5.QtWidgets import (
    QWidget,
)
from PyQt5.QtGui import (
    QPainter,
    QColor,
    QBrush,
    QPaintEvent,
    QFont,
    QMouseEvent,
)
from PyQt5.QtCore import Qt
import logging

import enum

from gomoku_rl.core import Gomoku
import torch
import random

from typing import Callable
from tensordict import TensorDict


class Piece(enum.Enum):
    EMPTY = enum.auto()
    BLACK = enum.auto()
    WHITE = enum.auto()


def _is_action_valid(x: int, y: int, board_size: int):
    return 0 <= x < board_size and 0 <= y < board_size


def rand_valid_action(
    board_size: int,
    is_valid_action: Callable[
        [
            int,
        ],
        bool,
    ],
):
    while True:
        action = random.randint(0, board_size * board_size - 1)
        if not is_valid_action(action):
            logging.warning(f"Invalid action {action}. Retry.")
            continue
        break

    x = action // board_size
    y = action % board_size

    return [x, y]


class GomokuBoard(QWidget):
    def __init__(
        self,
        board_size: int = 19,
        human_color: Piece | None = Piece.BLACK,
        model: Callable[
            [
                TensorDict,
            ],
            TensorDict,
        ]
        | None = None,
    ):
        super().__init__()
        self.board_size = board_size
        self.grid_size = 56
        self.piece_radius = 24
        assert 5 <= self.board_size < 20

        self.board: list[list[Piece]] = [
            [Piece.EMPTY] * board_size for _ in range(board_size)
        ]  # 0 represents an empty intersection

        self.human_color = human_color

        self._env = Gomoku(num_envs=1, board_size=board_size, device="cpu")
        self._env.reset()

        self.model = model

        if self.human_color == Piece.WHITE:
            self._AI_step()

        self.setStyleSheet("background-color: rgba(255,212,101,255);")

    def reset(self):
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                self.board[i][j] = Piece.EMPTY

        self._env.reset()

        if self.human_color == Piece.WHITE:
            self._AI_step()

        self.update()

    @property
    def current_player(self):
        turn = self._env.turn.item()
        if turn == 0:
            return Piece.BLACK
        else:
            return Piece.WHITE

    @property
    def done(self):
        return self._env.done.item()

    def _is_action_valid(self, action: int):
        return self._env.is_valid(torch.tensor([action])).item()

    def _AI_step(self):
        if self.done:
            logging.warning(f"_AI_step:Game already done!!!")
            return

        if self.model is None:
            x, y = rand_valid_action(self.board_size, self._is_action_valid)
        else:
            tensordict = TensorDict(
                {
                    "observation": self._env.get_encoded_board(),
                    "action_mask": self._env.get_action_mask(),
                },
                batch_size=1,
            ).to(self.model.device)
            with torch.no_grad():
                tensordict = self.model(tensordict).cpu()
            action: int = tensordict["action"].item()
            x = action // self.board_size
            y = action % self.board_size
            if not self._is_action_valid(action):
                # 设计上有点小缺陷，state全储存在_env（除了board），而GomokuEnv目前的设置是收到非法操作
                # state不变。所以如果在这结束不太好处理，就随机选一个action吧（好像也比较合理）。
                logging.warning(
                    f"AI:{self.current_player} ({x},{y}) Invalid! Use random policy instead."
                )
                x, y = rand_valid_action(self.board_size, self._is_action_valid)

        logging.info(f"AI:{self.current_player} ({x},{y})")
        self.step([x, y])

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter()
        painter.begin(self)
        self.drawBoard(painter)
        painter.end()

    def drawBoard(self, painter: QPainter):
        # Define the size of the board and calculate intersection size
        w, h = self.width(), self.height()
        assert w < h
        self.total_board_size = (self.board_size - 1) * self.grid_size
        assert self.total_board_size < h
        self.margin_size_x = int((w - self.total_board_size) / 2)
        self.margin_size_y = min(self.margin_size_x, 80)

        # Draw board grid
        painter.setPen(QColor(0, 0, 0))
        painter.setFont(QFont("Arial", 16))
        for i in range(self.board_size):
            painter.drawLine(
                self.margin_size_x,
                self.margin_size_y + i * self.grid_size,
                self.margin_size_x + self.total_board_size,
                self.margin_size_y + i * self.grid_size,
            )
            painter.drawLine(
                self.margin_size_x + i * self.grid_size,
                self.margin_size_y,
                self.margin_size_x + i * self.grid_size,
                self.margin_size_y + self.total_board_size,
            )

            painter.drawText(
                self.margin_size_x - 35,
                self.margin_size_y + i * self.grid_size + 10,
                f"{i:>2d}",
            )

            painter.drawText(
                self.margin_size_x + i * self.grid_size - 10,
                self.margin_size_y - 15,
                f"{i:2d}",
            )

        # Draw stones
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == Piece.BLACK:
                    painter.setBrush(QBrush(QColor(0, 0, 0)))
                    painter.drawEllipse(
                        self.margin_size_x - self.piece_radius + i * self.grid_size,
                        self.margin_size_y - self.piece_radius + j * self.grid_size,
                        self.piece_radius * 2,
                        self.piece_radius * 2,
                    )
                elif self.board[i][j] == Piece.WHITE:
                    painter.setBrush(QBrush(QColor(255, 255, 255)))
                    painter.drawEllipse(
                        self.margin_size_x - self.piece_radius + i * self.grid_size,
                        self.margin_size_y - self.piece_radius + j * self.grid_size,
                        self.piece_radius * 2,
                        self.piece_radius * 2,
                    )

    def step(self, action: list[int]):
        assert 2 == len(action)
        x = action[0]
        y = action[1]

        valid = self._is_action_valid(x * self.board_size + y)

        if not valid:
            logging.warning(f"Invalid Action: ({x},{y})")
            return

        self.board[x][y] = self.current_player

        self._env.step(torch.tensor([x * self.board_size + y]))
        self.update()  # Redraw the board

        if self.done:
            if self.current_player == Piece.BLACK:
                color = "WHITE"
            else:
                color = "BLACK"
            print("{} wins.".format(color))

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            # Calculate the clicked intersection
            x = int(
                (event.x() - self.margin_size_x + self.piece_radius) / self.grid_size
            )
            y = int(
                (event.y() - self.margin_size_y + self.piece_radius) / self.grid_size
            )

            if self.done:
                return

            if not self._is_action_valid(x * self.board_size + y):
                return

            human_turn = (
                self.human_color is not None and self.current_player == self.human_color
            )

            if human_turn and _is_action_valid(x, y, self.board_size):
                logging.info(f"Human:{self.current_player} ({x},{y})")
                self.step([x, y])
                if not self.done:
                    self._AI_step()


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="demo")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    # pprint(OmegaConf.to_container(cfg))

    # 设计的有点问题，耦合了
    # 有空再改

    if cfg.get("human_black", True):
        human_color = Piece.BLACK
    else:
        human_color = Piece.WHITE

    model_ckpt_path = cfg.get("checkpoint", None)
    if model_ckpt_path is not None:
        board_size = cfg.board_size
        device = cfg.device
        action_spec = DiscreteTensorSpec(
            board_size * board_size,
            shape=[
                1,
            ],
            device=device,
        )
        # when using PPO, setting num_envs=1 will cause an error in critic
        observation_spec = CompositeSpec(
            {
                "observation": UnboundedContinuousTensorSpec(
                    device=cfg.device,
                    shape=[2, 3, board_size, board_size],
                ),
                "action_mask": BinaryDiscreteTensorSpec(
                    n=board_size * board_size,
                    device=device,
                    shape=[2, board_size * board_size],
                    dtype=torch.bool,
                ),
            },
            shape=[
                2,
            ],
            device=device,
        )
        model = get_policy(
            name=cfg.algo.name,
            cfg=cfg.algo,
            action_spec=action_spec,
            observation_spec=observation_spec,
            device=cfg.device,
        )
        model.load_state_dict(torch.load(model_ckpt_path, map_location = cfg.device))
        model.eval()
    else:
        model = None

    app = QApplication(sys.argv)

    board = GomokuBoard(
        board_size=cfg.get("board_size", 19),
        human_color=human_color,
        model=model,
    )
    window = QMainWindow()
    window.setFixedSize(1200, 1600)
    window.setCentralWidget(board)
    window.setWindowTitle("demo")
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
