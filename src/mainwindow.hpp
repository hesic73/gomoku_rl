#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QString>
#include <QWidget>
#include <QMenu>
#include <QMenuBar>
#include <QAction>
#include <QStatusBar>
#include <QFileDialog>
#include <QString>
#include "board.hpp"

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    MainWindow(QWidget *parent = nullptr) : QMainWindow(parent)
    {
        board = new Board(this);
        setWindowTitle("gomoku-gui");
        setCentralWidget(board);

        auto menu_bar = menuBar();
        auto file_menu = menu_bar->addMenu("&File");
        auto open_action = new QAction("&Open", this);

        connect(open_action, &QAction::triggered, [this]()
                {
                    auto path = QFileDialog::getOpenFileName(this, "Open", QString(), "*.pt");
                    this->board->load(path); });
        file_menu->addAction(open_action);
    }

    ~MainWindow() {}

private:
    Board *board;
};
#endif // MAINWINDOW_H
