from tensordict.nn import set_interaction_type, InteractionType
from gomoku_rl.env import GomokuEnv
from gomoku_rl.utils.policy import _policy_t
import torch


def eval_win_rate(
    env: GomokuEnv, player_black: _policy_t, player_white: _policy_t, n: int = 1
):
    tmp = [_eval_win_rate(env, player_black, player_white) for _ in range(n)]
    return sum(tmp) / len(tmp)


@set_interaction_type(type=InteractionType.RANDOM)
@torch.no_grad()
def _eval_win_rate(env: GomokuEnv, player_black: _policy_t, player_white: _policy_t):
    board_size = env.board_size

    tensordict = env.reset()
    if hasattr(player_black, "eval"):
        player_black.eval()
    if hasattr(player_white, "eval"):
        player_white.eval()

    episode_done = torch.zeros(
        env.num_envs, device=tensordict.device, dtype=torch.bool
    )  # (E,)

    interested_tensordict = []
    for i in range(board_size * board_size + 1):
        if i % 2 == 0:
            tensordict = player_black(tensordict)
        else:
            tensordict = player_white(tensordict)
        # In new episodes the players may play the opposite color, but these episodes are not used for evaluation
        tensordict = env.step_and_maybe_reset(tensordict)

        done = tensordict.get("done")

        index: torch.Tensor = done & ~episode_done
        episode_done: torch.Tensor = episode_done | done

        interested_tensordict.extend(tensordict["stats"][index].unbind(0))

        if episode_done.all().item():
            break

    env.reset()

    if hasattr(player_black, "train"):
        player_black.train()
    if hasattr(player_white, "train"):
        player_white.train()

    interested_tensordict = torch.stack(interested_tensordict, dim=0)
    return interested_tensordict["black_win"].float().mean().item()


def get_payoff_matrix(
    env: GomokuEnv,
    row_policies: list[_policy_t],
    col_policies: list[_policy_t],
    n: int = 1,
):
    payoff = torch.zeros(len(row_policies), len(col_policies))
    for i, row_p in enumerate(row_policies):
        for j, col_p in enumerate(col_policies):
            payoff[i, j] = eval_win_rate(
                env, player_black=row_p, player_white=col_p, n=n
            )

    return payoff
