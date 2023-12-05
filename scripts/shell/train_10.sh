python scripts/train_sp.py \
    num_envs=256 \
    board_size=10 \
    rounds=50 \
    epochs=200 \
    algo=ppo \
    augment=true \
    min_iter_steps=20 \
    device="cuda:1"
