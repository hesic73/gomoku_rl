.. gomoku_rl documentation master file, created by
   sphinx-quickstart on Thu Feb 29 12:40:14 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to gomoku_rl's documentation!
=====================================

Introduction
------------

*gomoku_rl* is an open-sourced project that trains agents to play the game of Gomoku through deep reinforcement learning.
Previous works often rely on variants of AlphaGo/AlphaZero and inefficiently use GPU resources.
*gomoku_rl* features GPU-parallelized simulation and leverages recent advancements in **MARL**.
Starting from random play, a model can achieve human-level performance on a :math:`15\times15` board within hours of training on a 3090.


Installation
------------

Install *gomoku_rl* with the following command:

.. code-block:: bash

   git clone git@github.com:hesic73/gomoku_rl.git
   cd gomoku_rl
   conda create -n gomoku_rl python=3.11.5
   conda activate gomoku_rl
   pip install -e .

I use python 3.11.5, torch 2.1.0 and **torchrl 0.2.1**. Lower versions of python and torch 1.x should be compatible as well. 

Usage
-----

*gomoku_rl* uses `hydra` to configure training hyperparameters. You can modify the settings in `cfg/train_InRL.yaml` or override them via the command line:

.. code-block:: bash

   # override default settings in cfg/train_InRL.yaml
   python scripts/train_InRL.py num_env=1024 device=cuda epochs=3000 wandb.mode=online
   # or simply:
   python scripts/train_InRL.py.py


The default location for saving checkpoints is `wandb/*/files` or `tempfile.gettempdir()` if `wandb.mode=='disabled'`. Modify the output directory by specifying the `run_dir` parameter.

After training, play Gomoku with your model using the `scripts/demo.py` script:

.. code-block:: bash

   # Install PyQt5
   pip install PyQt5
   python scripts/demo.py device=cpu grid_size=56 piece_radius=24 checkpoint=/model/path
   # default checkpoint (only for board_size=15)
   python scripts/demo.py


Pretrained models for a :math:`15\times15` board are available under  `pretrained_models/15_15/`. Be aware that using the wrong model for the board size will lead to loading errors due to mismatches in AI architectures. In PPO, when `share_network=True`, the actor and the critic could utilize a shared encoding module. At present, a `PPO` object with a shared encoder cannot load from a checkpoint without sharing.

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   gomoku_rl


