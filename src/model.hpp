#ifndef MODEL_H
#define MODEL_H

#include "core.hpp"

// https://github.com/pytorch/pytorch/issues/24308
// 我们C++真是太有意思辣
#undef slots
#include <torch/script.h>
#define slots Q_SLOTS
#include <string>
#include <memory>
#include <cstdint>
#include <vector>

class Model
{
public:
    bool load(std::string path)
    {
        try
        {
            m = std::make_unique<torch::jit::Module>(torch::jit::load(path));
        }
        catch (const c10::Error &e)
        {
            std::cerr << e.msg() << std::endl;
            return false;
        }
        m->eval();
        m->to(torch::kCPU);
        return true;
    }
    void release()
    {
        m.reset();
    }
    bool is_loaded() const
    {
        return m.get() != nullptr;
    }

    std::uint32_t operator()(const int *board, std::uint32_t board_size, Cell turn, std::uint32_t last_move)
    {
        auto opponent = turn == Cell::Black ? Cell::White : Cell::Black;

        torch::NoGradGuard no_grad;
        auto input = std::vector<torch::jit::IValue>{};
        auto options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);
        auto observation = torch::zeros({3, board_size * board_size}, options);
        for (std::uint32_t i = 0; i < board_size * board_size; i++)
        {
            observation[0][i] = static_cast<float>(board[i] == turn);
            observation[1][i] = static_cast<float>(board[i] == opponent);
            observation[2][i] = static_cast<float>(i == last_move);
        }
        observation = observation.reshape({3, board_size, board_size}).unsqueeze(0);

        auto action_mask = torch::ones({board_size * board_size}, options);
        for (std::uint32_t i = 0; i < board_size * board_size; i++)
        {
            action_mask[i] = static_cast<float>(board[i] == Cell::Empty);
        }
        action_mask = action_mask.unsqueeze(0);

        input.push_back(observation);
        input.push_back(action_mask);
        auto output = m->forward(input).toTuple();

        auto action = output->elements()[2].toTensor();

        return static_cast<std::uint32_t>(action.item().toInt());
    }

private:
    std::unique_ptr<torch::jit::Module> m;
};

#endif // MODEL_H