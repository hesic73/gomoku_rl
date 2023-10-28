# Gomoku(五子棋) Environment for Reinforcement Learning

## Introduction

There are repositories out there that utilize reinforcement learning algorithms to teach AI how to play Gomoku. However, these repositories are based on legacy approaches and do not employ modern RL algorithms. Besides, they fail to fully leverage the potential of GPU resources. 

## Features

- **Parallel Environment Execution**: Execute multiple Gomoku environments in parallel on GPU hardware, enabling the efficient training of RL agents at a larger scale.

- **19×19 Board**: Train AI models to play Gomoku on a standard 19x19 board.


## installation

```bash
python setup.py install
```


## Usage

```python
python scripts/train.py
```

## Notes

`action` is **unchecked**, and if it's invalid, the environment's response becomes undefined. It's the user's duty to verify input correctness.


## References

- [https://github.com/junxiaosong/AlphaZero_Gomoku/](https://github.com/junxiaosong/AlphaZero_Gomoku/)

- [https://arxiv.org/pdf/1809.10595](https://arxiv.org/pdf/1809.10595,)
