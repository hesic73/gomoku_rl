import hydra
from omegaconf import DictConfig, OmegaConf

from gomoku_rl import CONFIG_PATH
from pprint import pprint

from gomoku_rl.algo.dqn import train
from gomoku_rl.env import GomokuEnv
from torchrl.envs import TransformedEnv


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    # pprint(OmegaConf.to_container(cfg))
    base_env = GomokuEnv(
        num_envs=cfg.num_envs, board_size=cfg.board_size, device=cfg.device
    )
    env = TransformedEnv(base_env)

    total_frames: int = cfg.total_frames
    frames_per_batch: int = cfg.frames_per_batch
    total_frames: int = total_frames // frames_per_batch * frames_per_batch

    train(
        cfg=cfg.algo,
        env=env,
        total_frames=total_frames,
        frames_per_batch=frames_per_batch,
    )


if __name__ == "__main__":
    main()
