#ifndef LABELEDCOMBOBOX_H
#define LABELEDCOMBOBOX_H
#include <QWidget>
#include <QComboBox>
#include <QHBoxLayout>
#include <QLabel>
#include <functional>

class LabeledComboBox : public QWidget
{
    Q_OBJECT

public:
    LabeledComboBox(QWidget *parent = nullptr) : QWidget(parent)
    {
        auto layout = new QHBoxLayout(this);
        layout->setSpacing(0);
        layout->setContentsMargins(0, 0, 0, 0);

        auto label = new QLabel("Player:");
        layout->addWidget(label, 0, Qt::AlignRight);

        comboBox = new QComboBox;
        comboBox->addItem("black");
        comboBox->addItem("white");

        layout->addWidget(comboBox, 0, Qt::AlignLeft);
        setLayout(layout);
    }

    void connect_index_changed(std::function<void(int)> callable)
    {
        connect(comboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
                callable);
    }

private:
    QComboBox *comboBox;
};

#endif // LABELEDCOMBOBOX_H