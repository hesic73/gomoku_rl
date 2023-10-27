from gokomu_rl.learning.dqn import train
from gokomu_rl.env import GokomuEnv
import torch


def main():
    env = GokomuEnv(num_envs=128, board_size=9, device="cuda")
    train(env=env)


if __name__ == "__main__":
    main()
