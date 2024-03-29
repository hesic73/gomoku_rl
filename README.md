# Gomoku RL

Documentation: https://hesic73.github.io/gomoku_rl/

![](/assets//images/screenshot_0.gif)

[TOC]

## Introduction

*gomoku_rl* is an open-sourced project that trains agents to play the game of Gomoku through deep reinforcement learning. Previous works often rely on variants of AlphaGo/AlphaZero and inefficiently use GPU resources. Notably, many existing projects are limited to small boards, with only a few exceptions. [[1]](#refer-anchor-1) incorporates curriculum learning and other enhancements;  [[2]](#refer-anchor-2)  and  [[3]](#refer-anchor-3)  collect transitions from multiple environments and also parallelize MCTS execution. In contrast, *gomoku_rl* features GPU-parallelized simulation and leverages recent advancements in **MARL**. Starting from random play, a model can achieve human-level performance on a $15\times15$ board within hours of training on a 3090.

## Installation

Install *gomoku_rl* with the following command:

```bash
git clone git@github.com:hesic73/gomoku_rl.git
cd gomoku_rl
conda create -n gomoku python=3.11.5
conda activate gomoku
pip install -e .
```

I use python 3.11.5, torch 2.1.0 and **torchrl 0.2.1**. Lower versions of python and torch 1.x should be compatible as well. 

## Getting Started

*gomoku_rl* uses `hydra` to configure training hyperparameters. You can modify the settings in `cfg/train_InRL.yaml` or override them via the command line:

```bash
# override default settings in cfg/train_InRL.yaml
python scripts/train_InRL.py num_env=1024 device=cuda epochs=500 wandb.mode=online
# or simply:
python scripts/train_InRL.py.py
```

The default location for saving checkpoints is `wandb/*/files` or `tempfile.gettempdir()` if `wandb.mode=='disabled'`. Modify the output directory by specifying the `run_dir` parameter.

After training, play Gomoku with your model using the `scripts/demo.py` script:

```bash
# Install PyQt5
pip install PyQt5
python scripts/demo.py device=cpu grid_size=56 piece_radius=24 checkpoint=/model/path
# default checkpoint (only for board_size=15)
python scripts/demo.py
```

Pretrained models for a $15\times15$ board are available under  `pretrained_models/15_15/`. Be aware that using the wrong model for the board size will lead to loading errors due to mismatches in AI architectures. In PPO, when `share_network=True`, the actor and the critic could utilize a shared encoding module. At present, a `PPO` object with a shared encoder cannot load from a checkpoint without sharing.

## GUI

**Note:  for deployment, we opt for `torch.jit.ScriptModule` instead of `torch.nn.Module`.** The `*.pt` files used in `scripts/train_*.py` are state dicts of a `torch.nn.Module` and cannot be directly utilized in this context.


In addition to `scripts/demo.py`, there is a standalone C++ GUI application. To compile the source code, make sure to have Qt, Libtorch and cmake installed. Refer to [https://pytorch.org/cppdocs/installing.html](https://pytorch.org/cppdocs/installing.html) for instructions on how to install C++ distributions of Pytorch.

Here are the commands to build the executable:

```bash
# Make a directory
mkdir build; cd build

# Generate the build system
# If torch is not installed on your computer, specify the absolute path to Libtorch
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ../src

# Alternatively, if torch is installed, use the following command
cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ../src

# Build the executable
cmake --build . --config Release

```

**PS**: If CMake cannot find Torch, try `set(Torch_DIR /absolute/path/to/libtorch/share/cmake/torch)`.


## Algorithms

Presently, the framework incorporates PPO and DQN algorithms, with a designed flexibility for incorporating additional RL methods. In the realm of multi-agent training, it supports Independent RL and PSRO.

Notably, Independent RL has demonstrated superior efficacy over PSRO. As mentioned in [[1]](#refer-anchor-1), due to Gomoku's asymmetry, it's hard to train a network to play both black and white.

(Maybe I need to tune hyperparameters for PSRO.)

## Details

Free-style Gomoku is a two-player zero-sum extensive-form game. Two players alternatively place black and white stones on a board and the first who forms an unbroken line of five or more stones of his color wins. In the context of Multi-Agent Reinforcement Learning (MARL), two agents learn in the environment competitively. During each agent's turn, its observation is the (encoded) current board state, and its action is the selection of a position on the board to place a stone. We use action masking to prevent illegal moves. Winning rewards the agent with +1, while losing incurs a penalty of -1. 

## TO DO

- [x] Restructure the code to decouple rollout functionality from `GomokuEnv`.
- [ ] Enhance documentaion.
- [ ] Further improvement

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



## Citation
Please use this bibtex if you want to cite this repository:
```
@misc{He2023gomoku_rl,
  author = {He, Sicheng},
  title = {gomoku_rl},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/hesic73/gomoku_rl}},
}
```