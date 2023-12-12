import hydra
from omegaconf import DictConfig, OmegaConf
from gomoku_rl import CONFIG_PATH
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
import logging
from gomoku_rl.policy import get_policy, Policy
from gomoku_rl.utils.policy import uniform_policy, _policy_t
from torchrl.data.tensor_specs import (
    DiscreteTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    BinaryDiscreteTensorSpec,
)
import torch
from PyQt5.QtWidgets import (
    QWidget,
    QAction,
    QActionGroup,
    QFileDialog,
    QVBoxLayout,
    QLabel,
    QSizePolicy,
)
from PyQt5.QtGui import (
    QPainter,
    QColor,
    QBrush,
    QPen,
    QPaintEvent,
    QFont,
    QMouseEvent,
    QKeySequence,
)
from PyQt5.QtCore import Qt, QSize
import logging

import enum

from gomoku_rl.core import Gomoku
import torch
from tensordict import TensorDict


class Piece(enum.Enum):
    EMPTY = enum.auto()
    BLACK = enum.auto()
    WHITE = enum.auto()


def _is_action_valid(x: int, y: int, board_size: int):
    return 0 <= x < board_size and 0 <= y < board_size


def make_model(cfg: DictConfig):
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
    return model


class GomokuBoard(QWidget):
    def __init__(
        self,
        grid_size: int,
        piece_radius: int,
        board_size: int = 19,
        human_color: Piece | None = Piece.BLACK,
        model: _policy_t | None = None,
    ):
        super().__init__()
        self.board_size = board_size
        self.grid_size = grid_size
        self.piece_radius = piece_radius
        assert 5 <= self.board_size < 20

        self.board: list[list[Piece]] = [
            [Piece.EMPTY] * board_size for _ in range(board_size)
        ]  # 0 represents an empty intersection

        self.latest_move: tuple[int, int] | None = None

        self.human_color = human_color
        self.model = model or uniform_policy

        self._env = Gomoku(num_envs=1, board_size=board_size, device="cpu")
        self._env.reset()

        if self.human_color == Piece.WHITE:
            self._AI_step()

        self.setStyleSheet("background-color: rgba(255,212,101,255);")
        tmp = (self.board_size - 1) * self.grid_size + 100
        self.setMinimumHeight(tmp)
        self.setMinimumWidth(tmp)

    def reset(self):
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                self.board[i][j] = Piece.EMPTY

        self._env.reset()
        self.latest_move = None

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

        tensordict = TensorDict(
            {
                "observation": self._env.get_encoded_board(),
                "action_mask": self._env.get_action_mask(),
            },
            batch_size=1,
        )
        with torch.no_grad():
            tensordict = self.model(tensordict).cpu()
        action: int = tensordict["action"].item()
        x = action // self.board_size
        y = action % self.board_size

        assert self._is_action_valid(action)

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
        # assert w < h
        self.total_board_size = (self.board_size - 1) * self.grid_size
        assert self.total_board_size < h
        self.margin_size_x = int((w - self.total_board_size) / 2)
        self.margin_size_y = min(int((h - self.total_board_size) / 2), 80)

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

        if self.latest_move is not None:
            x, y = self.latest_move
            if self.board[x][y] == Piece.BLACK:
                painter.setPen(QPen(QColor(255, 255, 255), 2))
            elif self.board[x][y] == Piece.WHITE:
                painter.setPen(QPen(QColor(0, 0, 0), 2))

            painter.drawLine(
                self.margin_size_x + x * self.grid_size,
                self.margin_size_y + y * self.grid_size - int(self.piece_radius * 0.75),
                self.margin_size_x + x * self.grid_size,
                self.margin_size_y + y * self.grid_size + int(self.piece_radius * 0.75),
            )
            painter.drawLine(
                self.margin_size_x + x * self.grid_size - int(self.piece_radius * 0.75),
                self.margin_size_y + y * self.grid_size,
                self.margin_size_x + x * self.grid_size + int(self.piece_radius * 0.75),
                self.margin_size_y + y * self.grid_size,
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
        self.latest_move = (x, y)

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

    def sizeHint(self) -> QSize:
        tmp = (self.board_size - 1) * self.grid_size + 100
        return QSize(tmp, tmp)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="demo")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    # 设计的有点问题，耦合了
    # 有空再改

    if cfg.get("human_black", True):
        human_color = Piece.BLACK
    else:
        human_color = Piece.WHITE

    if model_ckpt_path := cfg.get("checkpoint", None):
        model = make_model(cfg)
        model.load_state_dict(torch.load(model_ckpt_path, map_location=cfg.device))
        model.eval()
    else:
        model = uniform_policy

    app = QApplication(sys.argv)

    board = GomokuBoard(
        grid_size=cfg.get("grid_size", 28),
        piece_radius=cfg.get("piece_radius", 12),
        board_size=cfg.get("board_size", 15),
        human_color=human_color,
        model=model,
    )

    window = QMainWindow()
    window.setMinimumSize(board.sizeHint())
    window.setWindowTitle("demo")

    central_widget = QWidget()
    label = QLabel()
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setFont(QFont("Arial", 24))
    # TO DO
    label.setText("")
    label.setMinimumHeight(32)
    layout = QVBoxLayout(central_widget)
    layout.addWidget(label, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignVCenter)
    layout.addWidget(board, Qt.AlignmentFlag.AlignCenter)
    window.setCentralWidget(central_widget)

    menu = window.menuBar().addMenu("&Menu")
    status_bar = window.statusBar()
    open_action = QAction("&Open")

    def open_file():
        path = QFileDialog.getOpenFileName(
            parent=window, caption="Open", filter="*.pt"
        )[0]
        if path:
            if not isinstance(board.model, Policy):
                board.model = make_model(cfg)

            try:
                board.model.load_state_dict(torch.load(path, map_location=cfg.device))
                board.model.eval()
                status_bar.showMessage(f"Checkpoint path:{path}")
            except Exception:
                logging.warning(
                    f"Failed to load checkpoint {path}. Use the random policy"
                )
                board.model = uniform_policy

    open_action.triggered.connect(open_file)
    open_action.setShortcut(QKeySequence.Open)
    open_action.setToolTip("Load a pretrained checkpoint.")
    menu.addAction(open_action)

    reset_action = QAction("&Reset")
    reset_action.triggered.connect(board.reset)
    reset_action.setShortcut("Ctrl+R")
    reset_action.setToolTip("Reset the board.")
    menu.addAction(reset_action)

    human_color_action_group = QActionGroup(window)

    black_color_action = QAction("&Black")
    black_color_action.setCheckable(True)
    human_color_action_group.addAction(black_color_action)

    def _black():
        board.human_color = Piece.BLACK
        board.reset()

    black_color_action.triggered.connect(_black)

    white_color_action = QAction("&White")
    white_color_action.setCheckable(True)
    human_color_action_group.addAction(white_color_action)

    def _white():
        board.human_color = Piece.WHITE
        board.reset()

    white_color_action.triggered.connect(_white)

    black_color_action.setChecked(board.human_color == Piece.BLACK)
    white_color_action.setChecked(board.human_color == Piece.WHITE)

    menu.addSeparator().setText("Human Color")
    menu.addAction(black_color_action)
    menu.addAction(white_color_action)

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
