from gomoku_rl.policy.common import make_dataset_naive
from gomoku_rl.env import GomokuEnv
from gomoku_rl.utils.policy import uniform_policy
from gomoku_rl.utils.misc import set_seed
from gomoku_rl.collector import SelfPlayCollector

from gomoku_rl.utils.test import assert_transition, Type


def test_self_play_collector():
    device = "cuda:0"
    num_envs = 256
    board_size = 10
    seed = 1234
    set_seed(seed)
    env = GomokuEnv(num_envs=num_envs, board_size=board_size, device=device)
    collector = SelfPlayCollector(env, uniform_policy)

    transitions, info = collector.rollout(50)

    for transition in make_dataset_naive(transitions, batch_size=1024):
        assert_transition(transition, type=Type.mixed)


if __name__ == "__main__":
    test_self_play_collector()
