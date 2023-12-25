#ifndef BUTTONPANEL_H
#define BUTTONPANEL_H

#include <QWidget>
#include <QHBoxLayout>
#include <QPushButton>
#include <QComboBox>
#include <functional>
#include "labeledcombobox.hpp"

class ButtonPanel : public QWidget
{
    Q_OBJECT

public:
    ButtonPanel(QWidget *parent = nullptr) : QWidget(parent)
    {
        human_color = new LabeledComboBox;
        restart_button = new QPushButton;
        restart_button->setText("Restart");
        undo_button = new QPushButton;
        undo_button->setText("Undo");

        auto layout = new QHBoxLayout(this);
        layout->addWidget(human_color);
        layout->addWidget(restart_button);
        layout->addWidget(undo_button);
        setLayout(layout);
    }
    ~ButtonPanel() {}

    void connect_color_index_changed(std::function<void(int)> callable)
    {
        human_color->connect_index_changed(callable);
    }

    void connect_restart(std::function<void()> callable)
    {
        connect(restart_button,&QPushButton::clicked,callable);
    }
    void connect_undo(std::function<void()> callable)
    {
        connect(undo_button,&QPushButton::clicked,callable);
    }

private:
    LabeledComboBox *human_color;
    QPushButton *restart_button;
    QPushButton *undo_button;
};
#endif // BUTTONPANEL_H