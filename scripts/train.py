import hydra
from omegaconf import DictConfig, OmegaConf

from gomoku_rl import CONFIG_PATH
from pprint import pprint

from gomoku_rl.algo.dqn import train
from gomoku_rl.env import GomokuEnvWithOpponent, GomokuEnv
from torchrl.envs import TransformedEnv

from gomoku_rl.utils.wandb import init_wandb
from torchrl.data.tensor_specs import DiscreteTensorSpec
from gomoku_rl.algo.dqn import load_actor,make_actor_explore
import logging


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    # pprint(OmegaConf.to_container(cfg))
    action_spec = DiscreteTensorSpec(
        cfg.board_size * cfg.board_size,
        shape=[
            cfg.num_envs,
        ],
        device="cpu",
    )

    model_ckpt_path = cfg.get("model_ckpt_path", None)
    if model_ckpt_path is not None:
        logging.info(f"Loading model from {model_ckpt_path}")
        model = load_actor(
            actor_cfg=cfg.algo.actor,
            action_spec=action_spec,
            ckpt_path=model_ckpt_path,
        ).to(cfg.get("device", "cpu"))
        model=make_actor_explore(model,annealing_num_steps=100,eps_init=0.5,eps_end=0.5)
    else:
        model=None
        
    # model=None
    base_env = GomokuEnvWithOpponent(
        num_envs=cfg.num_envs, board_size=cfg.board_size, device=cfg.device,initial_policy=model
    )
    env = TransformedEnv(base_env)

    total_frames: int = cfg.total_frames
    frames_per_batch: int = cfg.frames_per_batch
    total_frames: int = total_frames // frames_per_batch * frames_per_batch

    run = init_wandb(cfg=cfg)
    train(
        cfg=cfg.algo,
        env=env,
        total_frames=total_frames,
        frames_per_batch=frames_per_batch,
        run=run,
    )


if __name__ == "__main__":
    main()
