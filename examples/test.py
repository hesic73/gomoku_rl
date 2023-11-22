from gomoku_rl.env import GomokuEnv
from gomoku_rl.utils.policy import uniform_policy
from tensordict import TensorDict
import torch


def assert_transition_valid(tensordict: TensorDict):
    done: torch.Tensor = tensordict["next", "done"]
    action: torch.Tensor = tensordict["action"]
    observation: torch.Tensor = tensordict["observation"].bool()  # (E,3,B,B)
    board = observation[:, 0] | observation[:, 1]  # (E,B,B)

    next_observation: torch.Tensor = tensordict["next", "observation"].bool()
    next_board = next_observation[:, 0] | next_observation[:, 1]

    diff_board = torch.logical_xor(next_board, board)

    tmp = diff_board.long().sum(-1).sum(-1)
    tmp = tmp[~done]
    idxs = (tmp != 2).nonzero().squeeze()
    if len(idxs) > 0:
        idx = idxs[0]
        print(board.long()[idx])
        print(next_board.long()[idx])
        print(board.long()[idx].sum().item())
        print(next_board.long()[idx].sum().item())
        print(action[idx])
        exit()

    assert (tmp == 2).all()


def main():
    device = "cuda:1"
    num_envs = 128
    board_size = 15
    env = GomokuEnv(num_envs=num_envs, board_size=board_size, device=device)

    transitions_black, transitions_white, info = env.rollout(
        100, player_black=uniform_policy, player_white=uniform_policy, augment=False
    )
    print(f"FPS:{env._fps:.2e}")
    for transition in transitions_black:
        assert_transition_valid(transition)


if __name__ == "__main__":
    main()
