from gokomu_rl.learning.modules import Encoder, Net
import torch


def main():
    device = "cuda"
    encoder = Encoder()
    encoder = Net(board_width=8, board_height=8)
    n = 96
    c = 4
    board_size = 8
    board_state = torch.randn(n, c, board_size, board_size)

    features = encoder(board_state)


if __name__ == "__main__":
    main()
