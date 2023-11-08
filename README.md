# Reinforcement Learning in Gomoku(五子棋)

## Introduction

*gomoku_rl* is an open-sourced project focused on training Gomoku AI through reinforcement learning. While there are repositories using RL algorithms to teach AI to play Gomoku, they are based on legacy approaches and do not harness the full potential of GPU resources. *gomoku_rl* features GPU-parallelized simulation and modern self-play algorithms.

## Installation

To get started, install gomoku_rl with the following command:
```bash
pip install -e .
```

## Dependency

- `torchrl==0.2.1`
- `python==3.11.5`
- `torch==2.1.0`
- `PyQt5==5.15.10`
- `omegaconf==2.3.0`
- `hydra-core==1.3.2`

## Usage
*gomoku_rl* uses `hydra` to configure training hyperparameters. You can modify the settings in `cfg/train.yaml` or override them via the command line:

```bash
python scripts/train.py board_size=10
```

Once you've trained an AI, you can play Gomoku with it using the `scripts/demo.py` script:

```bash
python scripts/demo.py human_first=True board_size=10 model_ckpt_path=/path/to/your/model
```


## References

- [https://github.com/junxiaosong/AlphaZero_Gomoku/](https://github.com/junxiaosong/AlphaZero_Gomoku/)

- [https://arxiv.org/pdf/1809.10595](https://arxiv.org/pdf/1809.10595)
