import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
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

from enum import Enum

from .core import Gokomu
import torch
import random


class Piece(Enum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2


def board_to_tensor(board: list[list[Piece]]) -> torch.Tensor:
    t = torch.zeros(len(board), len(board[0]), dtype=torch.long)

    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == Piece.EMPTY:
                t[i][j] = 0
            elif board[i][j] == Piece.BLACK:
                t[i][j] = 1
            elif board[i][j] == Piece.WHITE:
                t[i][j] = -1

    return t.unsqueeze(0)  # (1,B,B)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(600, 800)
        board = GoBoard()
        self.setCentralWidget(board)
        self.setWindowTitle("GUI")


class GoBoard(QWidget):
    def __init__(self, board_size: int = 19, human_color: Piece | None = Piece.WHITE):
        super().__init__()
        self.board_size = board_size
        self.grid_size = 28
        self.piece_radius = 12
        assert 5 <= self.board_size < 20

        self.board: list[list[Piece]] = [
            [Piece.EMPTY] * board_size for _ in range(board_size)
        ]  # 0 represents an empty intersection
        self.history = []
        self.done = False

        self.player = Piece.BLACK
        self.human_color = human_color

        self._env = Gokomu(num_envs=1, board_size=board_size, device="cpu")
        self._env.reset()

        if self.human_color == Piece.WHITE:
            self._AI_step()

        self.show()

    def _AI_step(self):
        if self.done:
            logging.warning(f"_AI_step:Game already done!!!")
            return

        while True:
            action = random.randint(0, self.board_size * self.board_size - 1)
            x = action // self.board_size
            y = action % self.board_size
            done, invalid = self.step([x, y])
            if invalid:
                print(f"AI generated an invalid action.Retry.")
                continue
            logging.info(f"AI action:({x},{y})")
            print(f"AI action:({x},{y})")
            break

    def reset(self):
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                self.board[i][j] = Piece.EMPTY

        self.player = Piece.BLACK
        self.history.clear()
        self.done = False

        self._env.reset()

        if self.human_color == Piece.WHITE:
            self._AI_step()

        self.update()

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

    def step(self, action: list[int]) -> tuple[bool, bool]:
        invalid_action: bool = False
        assert 2 <= len(action) <= 3
        x = action[0]
        y = action[1]
        invalid_action = not (0 <= x < self.board_size and 0 <= y < self.board_size)

        if len(action) == 3:
            invalid_action = invalid_action or Piece(action[2]) != self.player

        invalid_action = invalid_action or self.board[x][y] != Piece.EMPTY

        if not invalid_action:
            self.board[x][y] = self.player

            if self.player == Piece.BLACK:
                self.player = Piece.WHITE
            else:
                self.player = Piece.BLACK

            done, invalid = self._env.step(
                torch.tensor([x * self.board_size + y]).unsqueeze(0)
            )
            self.done = done.item()
            self.update()  # Redraw the board
        else:
            logging.warning(f"Invalid Action: ({x},{y})")

        if self.done:
            print("Done!!!")

        return (
            self.done,
            invalid_action,
        )

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            # Calculate the clicked intersection
            x = int(
                (event.x() - self.margin_size_x + self.piece_radius) / self.grid_size
            )
            y = int(
                (event.y() - self.margin_size_y + self.piece_radius) / self.grid_size
            )
            human_turn = (
                self.human_color is not None and self.player == self.human_color
            )
            if human_turn and not self.done:
                self.step([x, y])
                if not self.done:
                    self._AI_step()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
