from gokomu_rl import GokomuEnv
import torch

if __name__ == "__main__":
    num_envs = 4
    board_size = 9
    device = "cuda"
    env = GokomuEnv(num_envs=4, board_size=board_size, device=device)

    tensordict = env.rollout(100, break_when_any_done=False)
    print(tensordict[0][80][("observation", "board")])
    print(tensordict[0][79]['next'][("observation", "board")])
