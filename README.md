# Gokomu(五子棋) Environment for Reinforcement Learning

## Features

- **Parallel Environment Execution**: Run n Gokomu environments in parallel on GPU, making it suitable for training RL agents at scale.

## Usage

```python
from gokomu_rl import Gokomu
```

## Notes

`action` is **unchecked**, and if it's invalid, the environment's response becomes undefined. It's the user's duty to verify input correctness.

