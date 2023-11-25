# Gomoku RL

## Introduction

*gomoku_rl* is an open-sourced project that trains agents to play the game of Gomoku through deep reinforcement learning. Previous works often rely on variants of AlphaGo/AlphaZero and inefficiently use GPU resources. Notably, many existing projects are limited to small boards, with only a few exceptions. [[1]](#refer-anchor-1) incorporates curriculum learning and other enhancements;  [[2]](#refer-anchor-2)  and  [[3]](#refer-anchor-3)  collect transitions from multiple environments and also parallelize MCTS execution. In contrast, *gomoku_rl* features GPU-parallelized simulation and leverages recent advancements in **MARL**. Starting from random play, an model with fewer than 1M parameters can achieve human-level performance on a $15\times15$ board after a few hours of training on a 3090.

## Getting Started

To get started, install *gomoku_rl* with the following command:

```bash
conda create -n gomoku python=3.11.5
conda activate gomoku
pip install -e .
```

I use `python 3.11.5`, `torch 2.1.0` and `torchrl 0.2.1`, but lower versions of python and torch 1.x should be compatible as well. 

*gomoku_rl* uses `hydra` to configure training hyperparameters. You can modify the settings in `cfg/train_*.yaml` or override them via the command line:

```bash
# override default settings in cfg/train_InRL.yaml
python scripts/train_InRL.py board_size=15 num_env=1024 device=cuda algo=ppo epochs=100 wandb.mode=online
# or simply:
python scripts/train_InRL.py
```

The default location for saving checkpoints is `wandb/*/files` or `tempfile.gettempdir()` if `wandb.mode` is disabled. Modify the output directory by specifying the `run_dir` parameter.

After training, play Gomoku with your model using the `scripts/demo.py` script:

```bash
# Install PyQt5
pip install PyQt5
python scripts/demo.py human_black=True board_size=15 checkpoint=/path/to/your/model
# default checkpoint (only for board_size=15)
python scripts/demo.py human_black=True
```

A pretrained model for a $15\times15$ board is available at `pretrained_models/15_15/baseline.pt`, serving as the default checkpoint. Be aware that using the wrong model  for the board size will lead to loading errors due to mismatches in AI architectures.

## API Usage



## Details

### General

Free-style Gomoku is a two-player zero-sum extensive-form game. Two players alternatively place black and white stones on a board and the first who forms an unbroken line of five or more stones of his color wins. In the context of Multi-Agent Reinforcement Learning (**MARL**), two agents learn in the environment competitively. During each agent's turn, its observation is the (encoded) current board state, and its action is the selection of a position on the board to place a stone. We use action masking to prevent illegal moves. Winning rewards the agent with +1, while losing incurs a penalty of -1. 

The simplest form is *independent reinforcement learning* (InRL), where each agen treats its opponent as part of the environment. In `scripts/train_InRL.py`, black and white players are controlled by distinct neural networks. In `scripts/train_self_play.py`,  a single model is used to handle both black and white roles.

### Neural Networks





## Limitations

- Limited to Free-style Gomoku support only.
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