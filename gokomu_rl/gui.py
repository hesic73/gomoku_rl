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


class Piece(Enum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(600, 800)
        board = GoBoard()
        self.setCentralWidget(board)
        self.setWindowTitle("GUI")


class GoBoard(QWidget):
    def __init__(self, board_size: int = 19):
        super().__init__()
        self.board_size = board_size
        self.grid_size = 28
        self.piece_radius = 12
        assert 1 < self.board_size < 20

        self.board: list[list[Piece]] = [
            [Piece.EMPTY] * board_size for _ in range(board_size)
        ]  # 0 represents an empty intersection
        self.player = Piece.BLACK
        self.history = []
        self.done = False

        self.show()

    def reset(self):
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                self.board[i][j] = Piece.EMPTY

        self.player = Piece.BLACK
        self.history.clear()
        self.done = False

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

    def _update_done(self):
        pass

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

            self._update_done()

            self.update()  # Redraw the board
        else:
            logging.warning(f"Invalid Action")

        return invalid_action, self.done

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            # Calculate the clicked intersection
            x = int(
                (event.x() - self.margin_size_x + self.piece_radius) / self.grid_size
            )
            y = int(
                (event.y() - self.margin_size_y + self.piece_radius) / self.grid_size
            )

            self.step([x, y])


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
