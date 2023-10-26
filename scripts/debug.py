from gokomu_rl import Gokomu
import torch


def main():
    device = "cuda"
    game = Gokomu(num_envs=2, board_size=8, device=device)
    game.board[:, 0, 0] = 1
    game.board[:, 1, 1] = 1
    game.board[:, 2, 2] = 1
    game.board[:, 3, 3] = 1
    game.board[:, 4, 4] = 1
    action = torch.tensor([[1, 2, 1], [1, 2, 1]], device=device)
    game.step(action=action)
    # action = torch.tensor([[1, 2, -1], [3, 4, -1]])
    # game.step(action=action)
    # print(game.board[0])
    # print(game.board[1])


if __name__ == "__main__":
    main()
