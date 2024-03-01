python scripts/train_InRL_sp.py \
    num_envs=256 \
    board_size=8 \
    steps=64 \
    epochs=100 \
    algo=ppo \
    algo.batch_size=512 \
    wandb.mode=disabled \