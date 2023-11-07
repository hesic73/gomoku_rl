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
from torchrl.collectors import SyncDataCollector
from gomoku_rl.utils.wandb import init_wandb
from gomoku_rl.utils.misc import ActorBank
from gomoku_rl.algo import get_policy
import logging
from tqdm import tqdm

from copy import deepcopy

@torch.no_grad()
def eval_win_rate(env: TransformedEnv, policy,max_episode_length:int=180):
    rates=[]
    for i in range(100):
        rates.append(_eval_win_rate(env,policy,max_episode_length))
    return sum(rates)/len(rates)

def _eval_win_rate(env: TransformedEnv, policy,max_episode_length:int=180):
    env.reset()
    env.eval()
    td = env.rollout(
        max_steps=max_episode_length,
        policy=policy,
        auto_reset=True,
        break_when_any_done=False,
        return_contiguous=False,
    )
    env.reset()
    env.train()
    done: torch.Tensor = td["next", "done"].squeeze(-1)  # (E,max_episode_len,)
    wins: torch.Tensor = td["next", "stats","game_win"].squeeze(
        -1
    )  # (E,max_episode_len,)
    
    episode_done = torch.zeros_like(done[:, 0])  # (E,)

    rates = []

    for i in range(max_episode_length):
        _reset = done[:, i]  # (E,)
        index = _reset & ~episode_done
        episode_done = episode_done | _reset
        rates.extend(wins[index, i].cpu().unbind(0))

        if episode_done.all().item():
            break
    
    rate = torch.stack(rates).float()
    return rate.mean().item()




@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)  
    run = init_wandb(cfg=cfg)
    
    root_dir: str = cfg.get("root_dir", "sp_root")
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)
        logging.info(f"Create directory {root_dir}.")

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
    
    policy=get_policy(name=cfg.algo.name,cfg=cfg.algo,action_spec=env.action_spec,observation_spec=env.observation_spec,device=env.device)

    
    frames_per_batch = cfg.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    update_interval = int(cfg.get("update_interval", 10))
    assert update_interval > 0

    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.device,
        storing_device=cfg.get("storing_device","cpu"),
        return_same_td=True,
    )
    
    
    pbar = tqdm(collector, total=total_frames // frames_per_batch)
    env.train()
    for i, data in enumerate(pbar):
        # data (E,train_every)
        info = {"env_frames": collector._frames}
        info.update(policy.train_op(data.to_tensordict().to(cfg.device)))
        
        if i != 0 and i % update_interval == 0:
            ckpt_path=actorBank.save(policy.get_actor().state_dict())
            logging.info(f"Save checkpoint to {ckpt_path}")
            wr = eval_win_rate(env=env, policy=policy,max_episode_length=int((cfg.board_size**2)//2))
            
            if wr > 0.8:
                logging.info(f"Win Rate: {wr*100:.2f}%. Updating opponent's policy.")
                base_env.set_opponent_policy(deepcopy(policy.get_actor()))
            else:
                logging.info(f"Win Rate: {wr*100:.2f}%")

        run.log(info)
        pbar.set_postfix(
            {
                "frames": collector._frames,
            }
        )


    


if __name__ == "__main__":
    main()
