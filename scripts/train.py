import hydra
from omegaconf import DictConfig, OmegaConf

from gomoku_rl import CONFIG_PATH
from pprint import pprint


from gomoku_rl.env import GomokuEnvWithOpponent,GomokuEnv
from gomoku_rl.utils.transforms import LogOnEpisode
from torchrl.envs.transforms import (
    TransformedEnv,
    InitTracker,
    Compose,
)
from gomoku_rl.utils.wandb import init_wandb
import logging


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)  
    run = init_wandb(cfg=cfg)
    

    initial_policy=None
        
    base_env = GomokuEnvWithOpponent(
        num_envs=cfg.num_envs,
        board_size=cfg.board_size, 
        device=cfg.device,
        initial_policy=initial_policy
    )
    
    stats_keys=["episode_len","reward",'illegal_move',"win",'lose']
    stats_keys=[('stats',k) for k in stats_keys]
    logger = LogOnEpisode(
        cfg.num_envs,
        in_keys=stats_keys,
        log_keys=stats_keys,
        logger_func=lambda x:run.log(x),
    )
    transforms = [InitTracker(), logger]
    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    total_frames: int = cfg.total_frames
    frames_per_batch: int = cfg.frames_per_batch
    total_frames: int = total_frames // frames_per_batch * frames_per_batch

    


if __name__ == "__main__":
    main()
