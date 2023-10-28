# Gomoku(五子棋) Environment for Reinforcement Learning

Many existing repositories are outdated and don't make the most of GPU resources.

## Features

- **Parallel Environment Execution**: Run n Gomoku environments in parallel on GPU, making it suitable for training RL agents at scale.


## Usage

```python
python scripts/train.py
```

## Notes

`action` is **unchecked**, and if it's invalid, the environment's response becomes undefined. It's the user's duty to verify input correctness.


## References

- [https://github.com/junxiaosong/AlphaZero_Gomoku/](https://github.com/junxiaosong/AlphaZero_Gomoku/)

- [https://arxiv.org/pdf/1809.10595](https://arxiv.org/pdf/1809.10595,)
