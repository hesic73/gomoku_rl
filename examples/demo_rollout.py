from gomoku_rl import GomokuEnv
from gomoku_rl.utils.policy import uniform_policy
from pprint import pprint


def main():
    env = GomokuEnv(num_envs=128, board_size=10, device="cuda")

    transitions_black, transitions_white, info = env.rollout(
        rounds=50,
        player_black=uniform_policy,
        player_white=uniform_policy,
        augment=False,
    )

    print(transitions_black)

    pprint(dict(info))


if __name__ == "__main__":
    main()
