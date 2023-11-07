# Gomoku(五子棋) Environment for Reinforcement Learning

## Introduction

*gomoku_rl* is an open-sourced project focused on training Gomoku AI through reinforcement learning. While there are repositories using RL algorithms to teach AI to play Gomoku, they are based on legacy approaches and do not harness the full potential of GPU resources. *gomoku_rl* features GPU-parallelized simulation and pure PPO self-play training.

## Features

- **GPU-parallelized Environment**

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


- `action` is **unchecked**, and if it's invalid, the environment's response becomes undefined. It's the user's duty to verify input correctness.

## References

- [https://github.com/junxiaosong/AlphaZero_Gomoku/](https://github.com/junxiaosong/AlphaZero_Gomoku/)

- [https://arxiv.org/pdf/1809.10595](https://arxiv.org/pdf/1809.10595,)
