from gomoku_rl import GomokuEnv
from gomoku_rl.utils.policy import uniform_policy
from pprint import pprint


def main():
    env = GomokuEnv(num_envs=128, board_size=10, device="cuda")

    data, info = env.rollout_fixed_opponent(
        rounds=50,
        player=uniform_policy,
        opponent=uniform_policy,
        augment=False,
    )
    print(data)

    pprint(dict(info))


if __name__ == "__main__":
    main()
