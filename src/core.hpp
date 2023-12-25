#ifndef CORE_H
#define CORE_H
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_set>
#include <numeric>
#include <random>

enum Cell
{
    Empty = 0,
    Black,
    White,
};
static constexpr void set_cell(int *board, std::uint32_t size, std::uint32_t x, std::uint32_t y, Cell c)
{
    board[x * size + y] = static_cast<int>(c);
}
static constexpr int get_cell(const int *board, std::uint32_t size, std::uint32_t x, std::uint32_t y)
{
    return board[x * size + y];
}

static bool compute_done(const int *board, std::uint32_t size, int color)
{
    for (std::uint32_t i = 0; i < size; i++)
    {
        int cnt = 0;
        for (std::uint32_t j = 0; j < size; j++)
        {
            cnt += get_cell(board, size, i, j) == color;
            if (j >= 5)
                cnt -= get_cell(board, size, i, j - 5) == color;
            if (cnt == 5)
                return true;
        }
    }

    for (std::uint32_t i = 0; i < size; i++)
    {
        int cnt = 0;
        for (std::uint32_t j = 0; j < size; j++)
        {
            cnt += get_cell(board, size, j, i) == color;
            if (j >= 5)
                cnt -= get_cell(board, size, j - 5, i) == color;
            if (cnt == 5)
                return true;
        }
    }

    // diagonal

    for (std::uint32_t i = 5; i < size; i++)
    {
        int cnt = 0;
        for (std::uint32_t j = 0; j <= i; j++)
        {
            cnt += get_cell(board, size, size - i + j, j) == color;
            if (j >= 5)
                cnt -= get_cell(board, size, size - i + j - 5, j - 5) == color;
            if (cnt == 5)
                return true;
        }
    }

    {
        int cnt = 0;
        for (std::uint32_t i = 0; i < size; i++)
        {
            cnt += get_cell(board, size, i, i) == color;
            if (i >= 5)
                cnt -= get_cell(board, size, i - 5, i - 5) == color;
            if (cnt == 5)
                return true;
        }
    }

    for (std::uint32_t i = 5; i < size; i++)
    {
        int cnt = 0;
        for (std::uint32_t j = 0; j <= i; j++)
        {
            cnt += get_cell(board, size, j, size - i + j) == color;
            if (j >= 5)
                cnt -= get_cell(board, size, j - 5, size - i + j - 5) == color;
            if (cnt == 5)
                return true;
        }
    }

    // antidiagonal

    for (std::uint32_t i = 5; i < size; i++)
    {
        int cnt = 0;
        for (std::uint32_t j = 0; j <= i; j++)
        {
            cnt += get_cell(board, size, i - j, j) == color;
            if (j >= 5)
                cnt -= get_cell(board, size, i - (j - 5), j - 5) == color;
            if (cnt == 5)
                return true;
        }
    }

    {
        int cnt = 0;
        for (std::uint32_t i = 0; i < size; i++)
        {
            cnt += get_cell(board, size, size - 1 - i, i) == color;
            if (i >= 5)
                cnt -= get_cell(board, size, size - 1 - (i - 5), i - 5) == color;
            if (cnt == 5)
                return true;
        }
    }

    for (std::uint32_t i = 5; i < size; i++)
    {
        int cnt = 0;
        for (std::uint32_t j = 0; j <= i; j++)
        {
            cnt += get_cell(board, size, size - 1 - j, size - i + j) == color;
            if (j >= 5)
                cnt -= get_cell(board, size, size - 1 - (j - 5), size - i + j - 5) == color;
            if (cnt == 5)
                return true;
        }
    }

    return false;
}

class Gomoku
{
private:
    int *board;
    Cell turn;
    std::vector<std::uint32_t> history;
    std::unordered_set<std::uint32_t> available_actions;
    bool done;

public:
    Gomoku(std::uint32_t board_size) : board_size(board_size), history({}), done(false), available_actions({})
    {
        board = new int[board_size * board_size];
        reset();
    }
    ~Gomoku() { delete[] board; }

    void reset()
    {
        std::memset(board, 0, sizeof(int) * board_size * board_size);
        turn = Cell::Black;
        history.clear();
        done = false;
        for (std::uint32_t i = 0; i < board_size * board_size; i++)
        {
            available_actions.insert(i);
        }
    }
    bool action_valid(std::uint32_t action) const
    {
        return action >= 0 && action < board_size * board_size && board[action] == static_cast<int>(Cell::Empty);
    }

    bool step(std::uint32_t action)
    {
        if (done || !action_valid(action))
        {
            return false;
        }

        board[action] = static_cast<int>(turn);
        done = compute_done(board, board_size, static_cast<int>(turn));

        if (turn == Cell::Black)
        {
            turn = Cell::White;
        }
        else
        {
            turn = Cell::Black;
        }
        history.push_back(action);
        available_actions.erase(action);
        return true;
    }
    bool unstep()
    {
        if (history.empty())
        {
            return false;
        }

        auto action = history.back();
        history.pop_back();
        available_actions.insert(action);
        board[action] = static_cast<int>(Cell::Empty);
        done = false;
        if (turn == Cell::Black)
        {
            turn = Cell::White;
        }
        else
        {
            turn = Cell::Black;
        }
        return true;
    }
    bool is_done() const { return done; }
    Cell get_turn() const { return turn; }
    int get_move_count() const
    {
        return static_cast<int>(history.size());
    }

    std::uint32_t get_random_valid_action() const
    {
        if (available_actions.empty())
        {
            return std::numeric_limits<std::uint32_t>::max();
        }

        std::random_device rand_dev;
        std::mt19937 generator(rand_dev());
        std::uniform_int_distribution<int> distr(0, static_cast<int>(available_actions.size()) - 1);

        auto it = std::next(std::begin(available_actions), distr(generator));
        return *it;
    }
    std::uint32_t last_action() const
    {
        if (history.empty())
        {
            return std::numeric_limits<std::uint32_t>::max();
        }
        return history.back();
    }
    const int *get_board_view() const { return board; }

    std::string string_repr()
    {
        std::string s;
        s.reserve(board_size * (board_size + 1));
        for (std::uint32_t i = 0; i < board_size; i++)
        {
            for (std::uint32_t j = 0; j < board_size; j++)
            {
                char c;

                switch (get_cell(board, board_size, j, i))
                {
                case Cell::Empty:
                    c = ' ';
                    break;
                case Cell::Black:
                    c = 'B';
                    break;
                case Cell::White:
                    c = 'W';
                    break;
                default:
                    c = '?';
                }
                s.push_back(c);
            }
            s.push_back('\n');
        }
        return s;
    }
    const std::uint32_t board_size;
};

#endif // CORE_H