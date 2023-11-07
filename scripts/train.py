import hydra
from omegaconf import DictConfig, OmegaConf
import os
import datetime
from gomoku_rl import CONFIG_PATH
import torch


from gomoku_rl.env import GomokuEnvWithOpponent
from gomoku_rl.utils.transforms import LogOnEpisode
from torchrl.envs.transforms import (
    TransformedEnv,
    InitTracker,
    Compose,
)
from gomoku_rl.utils.wandb import init_wandb
from gomoku_rl.utils.misc import ActorBank
import logging


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)  
    run = init_wandb(cfg=cfg)
    
    root_dir: str = cfg.get("root_dir", "sp_root")
    if not os.path.isdir(root_dir):
        logging.info(f"Create directory {root_dir}.")
        os.makedirs(root_dir)

    actorBank = ActorBank(
        dir=os.path.join(root_dir, datetime.datetime.now().strftime("%m-%d_%H-%M")),
        latest_n=5,
    )
    

    initial_policy=cfg.get("initial_policy",None)
    if initial_policy is not None:
        initial_policy=torch.load(initial_policy)
        logging.info(f"Initial Policy: {initial_policy}")
        
    base_env = GomokuEnvWithOpponent(
        num_envs=cfg.num_envs,
        board_size=cfg.board_size, 
        device=cfg.device,
        initial_policy=initial_policy
    )
    
    stats_keys=base_env.stats_keys
    logger = LogOnEpisode(
        cfg.num_envs,
        in_keys=stats_keys,
        log_keys=stats_keys,
        logger_func=lambda x:run.log(x),
    )
    transforms = [InitTracker(), logger]
    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    
    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    update_interval = int(cfg.get("update_interval", 10))
    assert update_interval > 0

    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )



    


if __name__ == "__main__":
    main()
