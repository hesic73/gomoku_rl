#ifndef BOARD_H
#define BOARD_H

#include <QWidget>
#include <QGuiApplication>
#include <QScreen>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QPainter>
#include <QBrush>
#include <QPen>
#include <Qt>

#include <cstdint>
#include <iostream>
#include <string>
#include <filesystem>
#include "core.hpp"
#include "model.hpp"

class Board : public QWidget
{
    Q_OBJECT
public:
    Board(QWidget *parent = nullptr, std::uint32_t board_size = 15, Cell human = Cell::Black) : QWidget(parent), gomoku(board_size), human(human)
    {
        setStyleSheet("background-color: rgba(255,212,101,255);");

        auto screen_geometry = QGuiApplication::primaryScreen()->geometry();
        auto h = screen_geometry.height();
        auto w = screen_geometry.width();

        auto L = std::min(h, w) * 0.75;
        grid_size = static_cast<int>(L / (board_size - 1));
        piece_radius = static_cast<int>(grid_size * 0.45);
        auto tmp = static_cast<int>(board_size - 1) * grid_size + 100;
        setFixedSize(tmp, tmp);
        margin_size_x = 50;
        margin_size_y = 50;
        reset();
    }
    ~Board() {}
    Cell get_human() const { return human; }

public slots:
    void load(QString str)
    {
        auto bytes = str.toUtf8();
        auto path = std::filesystem::u8path(bytes.constData());
        model.load(path.string());
    }
    void reset()
    {
        gomoku.reset();
        if (human == Cell::White)
        {
            AI_step();
        }
        update();
    }
    void undo()
    {
        if (gomoku.get_move_count() < 2)
        {
            return;
        }
        gomoku.unstep();
        gomoku.unstep();
        update();
    }
    void set_human(Cell color)
    {
        human = color;
    }

protected:
    virtual void mousePressEvent(QMouseEvent *event) override
    {
        if (event->button() != Qt::MouseButton::LeftButton || gomoku.is_done())
        {
            return;
        }
        auto x = (event->x() - margin_size_x + piece_radius) / grid_size;
        auto y = (event->y() - margin_size_y + piece_radius) / grid_size;
        if (!gomoku.action_valid(x * gomoku.board_size + y))
        {
            return;
        }
        if (human != gomoku.get_turn())
        {
            return;
        }

        std::cout << "Human:" << x << "," << y << std::endl;
        gomoku.step(x * gomoku.board_size + y);
        update();
        if (gomoku.is_done())
        {
            std::cout << "Done" << std::endl;
        }
        if (!gomoku.is_done())
        {
            AI_step();
        }
    }

    virtual void paintEvent(QPaintEvent *event) override
    {
        auto painter = QPainter();
        painter.begin(this);

        painter.setPen(QColor(0, 0, 0));
        painter.setFont(QFont("Arial", 16));

        auto total_board_size = (gomoku.board_size - 1) * grid_size;

        auto board = gomoku.get_board_view();

        for (int i = 0; i < static_cast<int>(gomoku.board_size); i++)
        {
            painter.drawLine(
                margin_size_x,
                margin_size_y + i * grid_size,
                margin_size_x + total_board_size,
                margin_size_y + i * grid_size);
            painter.drawLine(
                margin_size_x + i * grid_size,
                margin_size_y,
                margin_size_x + i * grid_size,
                margin_size_y + total_board_size);
            painter.drawText(margin_size_x - 35, margin_size_y + i * grid_size + 10, QString("%1").arg(i));
            painter.drawText(margin_size_x + i * grid_size - 15, margin_size_y - 15, QString("%1").arg(i));
        }

        for (int i = 0; i < static_cast<int>(gomoku.board_size); i++)
        {
            for (int j = 0; j < static_cast<int>(gomoku.board_size); j++)
            {
                switch (static_cast<Cell>(board[i * gomoku.board_size + j]))
                {
                case Cell::Black:
                    painter.setBrush(QBrush(QColor(0, 0, 0)));
                    painter.drawEllipse(
                        margin_size_x - piece_radius + i * grid_size,
                        margin_size_y - piece_radius + j * grid_size,
                        piece_radius * 2,
                        piece_radius * 2);
                    break;
                case Cell::White:
                    painter.setBrush(QBrush(QColor(255, 255, 255)));
                    painter.drawEllipse(
                        margin_size_x - piece_radius + i * grid_size,
                        margin_size_y - piece_radius + j * grid_size,
                        piece_radius * 2,
                        piece_radius * 2);
                    break;

                default:
                    break;
                }
            }
        }

        if (gomoku.last_action() != std::numeric_limits<std::uint32_t>::max())
        {
            auto x = gomoku.last_action() / gomoku.board_size;
            auto y = gomoku.last_action() % gomoku.board_size;
            switch (board[x * gomoku.board_size + y])
            {
            case static_cast<int>(Cell::Black):
                painter.setPen(QPen(QColor(255, 255, 255), 2));
                break;
            case static_cast<int>(Cell::White):
                painter.setPen(QPen(QColor(0, 0, 0), 2));
                break;
            default:
                break;
            }

            painter.drawLine(
                margin_size_x + x * grid_size,
                margin_size_y + y * grid_size - static_cast<int>(piece_radius * 0.75),
                margin_size_x + x * grid_size,
                margin_size_y + y * grid_size + static_cast<int>(piece_radius * 0.75));
            painter.drawLine(
                margin_size_x + x * grid_size - static_cast<int>(piece_radius * 0.75),
                margin_size_y + y * grid_size,
                margin_size_x + x * grid_size + static_cast<int>(piece_radius * 0.75),
                margin_size_y + y * grid_size);
        }

        painter.end();
    }

private:
    Gomoku gomoku;
    Cell human;
    int grid_size;
    int piece_radius;
    int margin_size_x;
    int margin_size_y;
    Model model;

    void AI_step()
    {
        std::uint32_t action;
        if (model.is_loaded())
            action = model(gomoku.get_board_view(), gomoku.board_size, gomoku.get_turn(), gomoku.last_action());
        else
        {
            action = gomoku.get_random_valid_action();
        }

        std::cout << "AI:" << action / gomoku.board_size << "," << action % gomoku.board_size << std::endl;
        gomoku.step(action);
        if (gomoku.is_done())
        {
            std::cout << "Done" << std::endl;
        }
        update();
    }
};

#endif // BOARD_H
