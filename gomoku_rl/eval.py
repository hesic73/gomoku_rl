import torch
import random
from .env import GomokuEnv
from tensordict.nn import TensorDictModule


def evaluate(
    actor_0: TensorDictModule,
    actor_1: TensorDictModule,
    env: GomokuEnv,
    n: int = 100,
) -> tuple[float, float]:
    assert n % 2 == 0
    win_0 = 0
    win_1 = 0
    for i in range(n // 2):
        x, y = _evaluate(actor_0, actor_1, env)
        win_0 += x
        win_1 += y
        y, x = _evaluate(actor_1, actor_0, env)
        win_0 += x
        win_1 += y

    n: int = win_0 + win_1
    win_0 = win_0 / n
    win_1 = win_1 / n

    print(win_0, win_1)
    return win_0, win_1


@torch.no_grad()
def _evaluate(actor_0: TensorDictModule, actor_1: TensorDictModule, env: GomokuEnv):
    # 目前一切从简，用最简单的方式来计算胜率
    # （连reset都没有……）
    tensordict = env.reset()
    win_0: int = 0
    win_1: int = 0
    n = env.gomoku.num_envs

    board_size = env.gomoku.board_size

    first_episode_done = torch.zeros(n).bool()  # (E,)
    episode_len = torch.zeros(n).long()

    for i in range(board_size * board_size):
        if first_episode_done.all():
            break

        if i % 2 == 0:
            actor_0(tensordict)
        else:
            actor_1(tensordict)
        tensordict = env.step(tensordict)

        done = tensordict[("next", "done")].squeeze(-1).cpu()  # (E,)

        new_done = done & (~first_episode_done)  # (E,)
        episode_len[new_done] = i + 1
        first_episode_done = first_episode_done | done
        new_done_cnt = new_done.long().sum().item()

        if i % 2 == 0:
            win_0 += new_done_cnt
        else:
            win_1 += new_done_cnt

    return win_0, win_1
