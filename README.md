# Gomoku RL

**Note: The project is currently under development and I am addressing issues with the implementation of PPO.**

**Note: although AI has learned complex strategies, it is still difficult to defeat humans at present.  In the first screenshot, humans defeated AI, while in the second screenshot, humans lost to AI.**

![](/images/lose1.gif)
![](/images/win1.gif)

## Introduction

*gomoku_rl* is an open-sourced project that trains agents to play the game of Gomoku through deep reinforcement learning. Previous works often rely on variants of AlphaGo/AlphaZero and inefficiently use GPU resources. Notably, many existing projects are limited to small boards, with only a few exceptions. [[1]](#refer-anchor-1) incorporates curriculum learning and other enhancements;  [[2]](#refer-anchor-2)  and  [[3]](#refer-anchor-3)  collect transitions from multiple environments and also parallelize MCTS execution. In contrast, *gomoku_rl* features GPU-parallelized simulation and leverages recent advancements in **MARL**. Starting from random play, a model can ~~achieve human-level performance~~ on a $15\times15$ board.

## Installation

Install *gomoku_rl* with the following command:

```bash
conda create -n gomoku python=3.11.5
conda activate gomoku
pip install -e .
```

I use python 3.11.5, torch 2.1.0 and **torchrl 0.2.1**. Lower versions of python and torch 1.x should be compatible as well. 

## Getting Started

*gomoku_rl* uses `hydra` to configure training hyperparameters. You can modify the settings in `cfg/train.yaml` or override them via the command line:

```bash
# override default settings in cfg/train.yaml
python scripts/train.py board_size=15 num_env=1024 device=cuda algo=ppo epochs=100 wandb.mode=online
# or simply:
python scripts/train.py
```

The default location for saving checkpoints is `wandb/*/files` or `tempfile.gettempdir()` if `wandb.mode=='disabled'`. Modify the output directory by specifying the `run_dir` parameter.

After training, play Gomoku with your model using the `scripts/demo.py` script:

```bash
# Install PyQt5
pip install PyQt5
python scripts/demo.py human_black=True board_size=15 checkpoint=/path/to/your/model
# default checkpoint (only for board_size=15)
python scripts/demo.py
```

Pretrained models for a $15\times15$ board are available under  `pretrained_models/15_15/`. Be aware that using the wrong model  for the board size will lead to loading errors due to mismatches in AI architectures.

## API Usage

```python
from gomoku_rl import GomokuEnv
from gomoku_rl.utils.policy import uniform_policy
from pprint import pprint


def main():
    env = GomokuEnv(num_envs=128, board_size=10, device="cuda")

    transitions_black, transitions_white, info = env.rollout(
        rounds=50,
        player_black=uniform_policy,
        player_white=uniform_policy,
        batch_size=256,
        augment=False,
    )
    for data in transitions_black:
        print(data)
        break

    pprint(dict(info))


if __name__ == "__main__":
    main()

```

## Details

Free-style Gomoku is a two-player zero-sum extensive-form game. Two players alternatively place black and white stones on a board and the first who forms an unbroken line of five or more stones of his color wins. In the context of Multi-Agent Reinforcement Learning (MARL), two agents learn in the environment competitively. During each agent's turn, its observation is the (encoded) current board state, and its action is the selection of a position on the board to place a stone. We use action masking to prevent illegal moves. Winning rewards the agent with +1, while losing incurs a penalty of -1. 

## Limitations

- Constrained to Free-style Gomoku support only.
- The GUI is very rudimentary and cannot adapt to different resolutions.
- The internal use of `vmap` in `torchrl.objectives.value.GAE` clashes with `torch.nn.BatchNorm2d(track_running_stats=True)`. Consequently, for batch normalization modules in the critic, `track_running_stats` is set to False, rendering it unusable in evaluation mode.

## References

<div id="refer-anchor-1"></div>

- [1] [https://arxiv.org/pdf/1809.10595](https://arxiv.org/pdf/1809.10595)

<div id="refer-anchor-2"></div>

- [2] [https://github.com/initial-h/AlphaZero_Gomoku_MPI](https://github.com/initial-h/AlphaZero_Gomoku_MPI)

<div id="refer-anchor-3"></div>

- [3] [https://github.com/hijkzzz/alpha-zero-gomoku](https://github.com/hijkzzz/alpha-zero-gomoku)

<div id="refer-anchor-4"></div>

- [4] [A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning](https://arxiv.org/pdf/1711.00832.pdf)

<div id="refer-anchor-5"></div>

- [5] [What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study](https://arxiv.org/pdf/2006.05990.pdf)