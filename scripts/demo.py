from concurrent_go import GoGame
import torch


def main():
    n = 2
    game = GoGame(n=n)

    action = torch.zeros(n, 3, dtype=torch.long)

    game.step(action=action)


if __name__ == "__main__":
    main()
