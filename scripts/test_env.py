from gokomu_rl import GokomuEnv
import torch
from torchrl.envs.utils import check_env_specs

if __name__ == "__main__":
    num_envs = 4
    board_size = 9
    device = "cuda"
    env = GokomuEnv(num_envs=4, board_size=board_size, device=device)
    print(env.action_spec)
    check_env_specs(env)
    tensordict = env.rollout(100, break_when_any_done=False)

