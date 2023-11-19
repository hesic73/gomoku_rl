# Gomoku RL

## Introduction

*gomoku_rl* is an open-sourced project that trains Gomoku agents through self-play. Previous works often rely on variants of AlphaGo/AlphaZero and inefficiently use GPU resources. Notably, many existing projects are limited to small boards, with only a few exceptions. [[1]](#refer-anchor-1) incorporates curriculum learning and other enhancements, and [[2]](#refer-anchor-2) and [[3]](#refer-anchor-3) parallelize MCTS execution. In contrast, *gomoku_rl* features GPU-parallelized simulation and leverages recent advancements in **MARL**. This enables AI training without exploiting game rules through MCTS, except the use of action masking to prevent illegal moves.

## Algorithms

TO DO

## Getting Started

To get started, install *gomoku_rl* with the following command:

```bash
conda create -n gomoku python=3.11.5
conda activate gomoku
pip install -e .
```

I use python 3.11.5, and the following versions of packages:

- `torchrl==0.2.1`
- `torch==2.1.0`

PS: lower versions of python and torch 1.x should be compatible as well. Note that torchrl is currently in development, so I can't assure the code's functionality with other versions.

`PyQt5` is needed to run `scripts/demo.py`:

```bash
pip install PyQt5
```

*gomoku_rl* uses `hydra` to configure training hyperparameters. You can modify the settings in `cfg/train.yaml` or override them via the command line:

```bash
python scripts/train.py board_size=10
```

Once you've trained an AI, you can play Gomoku with it using the `scripts/demo.py` script:

```bash
python scripts/demo.py human_black=True board_size=10 checkpoint=/path/to/your/model
```

## Details

- The network architecture remains consistent with [[1]](#refer-anchor-1), with modifications detailed in `gomoku_rl.utils.module`.

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