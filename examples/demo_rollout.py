from gomoku_rl import GomokuEnv
from gomoku_rl.collector import VersusPlayCollector
from gomoku_rl.utils.policy import uniform_policy
from pprint import pprint


def main():
    env = GomokuEnv(num_envs=128, board_size=10, device="cuda")
    collector = VersusPlayCollector(env, uniform_policy, uniform_policy)
    transitions_black, transitions_white, info = collector.rollout(
        steps=100,
    )

    print(transitions_black)
    print(transitions_white)

    pprint(info)


if __name__ == "__main__":
    main()
