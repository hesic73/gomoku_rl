from gokomu_rl import GokomuEnv
import torch

if __name__ == "__main__":
    import sys
    sys.setrecursionlimit(30)
    num_envs = 4
    board_size = 9
    device = "cuda"
    env = GokomuEnv(num_envs=4, board_size=board_size, device=device)
    action_pos = torch.randint(
        low=0, high=board_size * board_size, size=(num_envs,), device=device
    )
    action_piece = torch.ones((num_envs,), device=device)

    action = torch.stack([action_pos, action_piece], dim=-1).to(torch.long)

    env.rollout(max_steps=100)
