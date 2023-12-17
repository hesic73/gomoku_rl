# Gomoku RL
[![](https://tokei.rs/b1/github/hesic73/gomoku_rl)](https://github.com/hesic73/gomoku_rl).

**Empirically, Independent RL is enough (and in fact much better than PSRO).** As mentioned in [[1]](#refer-anchor-1), due to Gomoku's asymmetry, it's hard to train a network to play both black and white.

![](/images/screenshot_0.gif)

## Introduction

*gomoku_rl* is an open-sourced project that trains agents to play the game of Gomoku through deep reinforcement learning. Previous works often rely on variants of AlphaGo/AlphaZero and inefficiently use GPU resources. Notably, many existing projects are limited to small boards, with only a few exceptions. [[1]](#refer-anchor-1) incorporates curriculum learning and other enhancements;  [[2]](#refer-anchor-2)  and  [[3]](#refer-anchor-3)  collect transitions from multiple environments and also parallelize MCTS execution. In contrast, *gomoku_rl* features GPU-parallelized simulation and leverages recent advancements in **MARL**. Starting from random play, a model can achieve human-level performance on a $15\times15$ board within hours of training on a 3090.

## Installation

Install *gomoku_rl* with the following command:

```bash
conda create -n gomoku python=3.11.5
conda activate gomoku
pip install -e .
```

I use python 3.11.5, torch 2.1.0 and **torchrl 0.2.1**. Lower versions of python and torch 1.x should be compatible as well. 

## Getting Started

*gomoku_rl* uses `hydra` to configure training hyperparameters. You can modify the settings in `cfg/train_InRL.yaml` or override them via the command line:

```bash
# override default settings in cfg/train_InRL.yaml
python scripts/train_InRL.py num_env=1024 device=cuda epochs=3000 wandb.mode=online
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

Pretrained models for a $15\times15$ board are available under  `pretrained_models/15_15/`. Be aware that using the wrong model for the board size will lead to loading errors due to mismatches in AI architectures. In PPO, when `share_network=True`, the actor and the critic could utilize a shared encoding module. At present, a `PPOPolicy` object with a shared encoder cannot load from a checkpoint without sharing.


## Supported Algorithms

- PPO
- DQN


## Details

Free-style Gomoku is a two-player zero-sum extensive-form game. Two players alternatively place black and white stones on a board and the first who forms an unbroken line of five or more stones of his color wins. In the context of Multi-Agent Reinforcement Learning (MARL), two agents learn in the environment competitively. During each agent's turn, its observation is the (encoded) current board state, and its action is the selection of a position on the board to place a stone. We use action masking to prevent illegal moves. Winning rewards the agent with +1, while losing incurs a penalty of -1. 


## API

### Policy
In general, policies are expected to be of the type `Callable[[Tensordict,], Tensordict]`. Here, `Tensordict` is a class with dictionary-like properties inherited from tensors (refer to [https://github.com/pytorch/tensordict](https://github.com/pytorch/tensordict)). The input `Tensordict` contains `observation` and `action_mask`. As described in [1], `observation` has a shape of [\*, 3, B, B], where $B$ denotes the board size. Similarly, `action_mask` has a shape of [\*, B^2]. The expected output `action` should have a shape of [\*,] within the range of [0, B^2).

Like in [tianshou](https://github.com/thu-ml/tianshou), a `Policy` interface is defined. See `gomoku_rl/policy/base.py` for more information.


### GomokuEnv

`GomokuEnv` comprises `num_envs` independent Gomoku environments, compatible with both CPU and CUDA. It effortlessly attains $10^5$ fps, enabling the concurrent execution of thousands of environments. It provides high level APIs such as `GomokuEnv.rollout` and `GomokuEnv.rollout_self_play`.

Example:

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
        augment=False,
    )

    print(transitions_black)

    pprint(dict(info))


if __name__ == "__main__":
    main()

```

## Limitations

- Constrained to Free-style Gomoku support only.

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