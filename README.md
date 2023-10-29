# Gomoku(五子棋) Environment for Reinforcement Learning

## Introduction

There are repositories out there that utilize reinforcement learning algorithms to teach AI how to play Gomoku. However, these repositories are based on legacy approaches and do not employ modern RL algorithms. Besides, they fail to fully leverage the potential of GPU resources.

## Features

- **Parallel Environment Execution**: Execute multiple Gomoku environments in parallel on GPU hardware, enabling the efficient training of RL agents at a larger scale.

- **19×19 Board**: Train AI models to play Gomoku on a standard 19x19 board.

## Installation

```bash
python setup.py install
```

## Dependency

- `torchrl`==0.2.1. When I was working on the project, it was still in development, and it seemed to be plagued by several lingering bugs. Switching to other versions of `torchrl` could pose potential issues. Regarding other packages, using different versions should be acceptable.

- `python==3.11.5`
- `torch==2.1.0`
- `PyQt5==5.15.10`
- `omegaconf==2.3.0`
- `hydra-core==1.3.2`
- `gymnasium==0.29.1`

## Usage

```python
python scripts/train.py
```

## Notes

- Gomoku is a game for two players, but `GomokuEnv` is designed as a single-agent environment. As a result, the next state for a single agent corresponds to the second following state in the environment. This might seem a bit unconventional and challenge the assumptions of the `torchrl` framework (to my knowledge).

- `action` is **unchecked**, and if it's invalid, the environment's response becomes undefined. It's the user's duty to verify input correctness.

## References

- [https://github.com/junxiaosong/AlphaZero_Gomoku/](https://github.com/junxiaosong/AlphaZero_Gomoku/)

- [https://arxiv.org/pdf/1809.10595](https://arxiv.org/pdf/1809.10595,)
