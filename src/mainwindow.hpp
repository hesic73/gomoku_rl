#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QString>
#include <QWidget>
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
    }
    ~MainWindow() {}
    

private:
    Board *board;
};
#endif // MAINWINDOW_H
