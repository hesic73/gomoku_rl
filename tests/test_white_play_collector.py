from gomoku_rl.policy.common import make_dataset_naive
from gomoku_rl.env import GomokuEnv
from gomoku_rl.utils.policy import uniform_policy
from gomoku_rl.utils.misc import set_seed
from gomoku_rl.collector import WhitePlayCollector

from gomoku_rl.utils.test import assert_transition, Type


def test_white():
    device = "cuda:0"
    num_envs = 256
    board_size = 10
    seed = 1234
    set_seed(seed)
    env = GomokuEnv(num_envs=num_envs, board_size=board_size, device=device)
    collector = WhitePlayCollector(env, uniform_policy, uniform_policy)

    black_transitions, info = collector.rollout(50)

    for transition in make_dataset_naive(black_transitions, batch_size=1024):
        assert_transition(transition, type=Type.white)


if __name__ == "__main__":
    test_white()
