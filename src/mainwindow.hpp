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
#include <QKeySequence>
#include <QActionGroup>
#include "board.hpp"
#include "buttonpanel.hpp"

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    MainWindow(QWidget *parent = nullptr) : QMainWindow(parent)
    {
        board = new Board(this);
        button_panel = new ButtonPanel(this);
        button_panel->connect_restart([this]()
                                      { this->board->reset(); });
        button_panel->connect_undo([this]()
                                   { this->board->undo(); });
        button_panel->connect_color_index_changed([this](int index)
                                                  {
            Cell color;
            if(index==0){
                color=Cell::Black;
            }else{
                color=Cell::White;
            }

            if(this->board->get_human()==color){
                        return;
                    }
            this->board->set_human(color);
            this->board->reset(); });

        auto w = new QWidget(this);
        auto layout = new QVBoxLayout(w);
        layout->addWidget(board);
        layout->addWidget(button_panel);
        w->setLayout(layout);

        setWindowTitle("gomoku-gui");
        setCentralWidget(w);

        auto menu_bar = menuBar();
        auto file_menu = menu_bar->addMenu("&File");

        auto open_action = new QAction("&Open", this);
        open_action->setShortcut(QKeySequence::Open);
        connect(open_action, &QAction::triggered, [this]()
                {
                    auto path = QFileDialog::getOpenFileName(this, "Open", QString(), "*.pt");
                    this->board->load(path); });
        file_menu->addAction(open_action);

        auto reset_action = new QAction("&Reset", this);
        reset_action->setShortcut(QKeySequence("Ctrl+R"));
        connect(reset_action, &QAction::triggered, board, &Board::reset);

        file_menu->addAction(reset_action);

        auto color_action_group = new QActionGroup(this);

        auto black_action = new QAction("&Black");
        black_action->setCheckable(true);
        black_action->setChecked(board->get_human() == Cell::Black);
        color_action_group->addAction(black_action);
        connect(black_action, &QAction::triggered, [this]()
                {
                    if(this->board->get_human()==Cell::Black){
                        return;
                    }
            this->board->set_human(Cell::Black);
            this->board->reset(); });

        auto white_action = new QAction("&White");
        white_action->setCheckable(true);
        white_action->setChecked(board->get_human() == Cell::White);
        color_action_group->addAction(white_action);
        connect(white_action, &QAction::triggered, [this]()
                {
                    if(this->board->get_human()==Cell::White){
                        return;
                    }
            this->board->set_human(Cell::White);
            this->board->reset(); });

        file_menu->addSeparator()->setText("Human");
        file_menu->addAction(black_action);
        file_menu->addAction(white_action);
    }

    ~MainWindow() {}

private:
    Board *board;
    ButtonPanel *button_panel;
};
#endif // MAINWINDOW_H
