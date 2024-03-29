import hydra
from omegaconf import DictConfig, OmegaConf
from gomoku_rl import CONFIG_PATH
import torch
from gomoku_rl.env import GomokuEnv
from gomoku_rl.policy import get_pretrained_policy
import numpy as np
import functools


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="eval")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    env = GomokuEnv(
        num_envs=2,
        board_size=cfg.board_size,
        device=cfg.device,
    )

    seed = cfg.get("seed", 12345)
    torch.manual_seed(seed)
    np.random.seed(seed)

    make_player = functools.partial(
        get_pretrained_policy,
        name=cfg.algo.name,
        cfg=cfg.algo,
        action_spec=env.action_spec,
        observation_spec=env.observation_spec,
        device=env.device,
    )

    checkpoint = cfg.checkpoints[0]
    player = make_player(checkpoint_path=checkpoint)
    actor=player.actor.eval().cpu()

    tensordict = env.reset()[[0]].cpu()
    s = torch.jit.trace(actor, (tensordict["observation"], tensordict["action_mask"]))
    print(s)
    torch.jit.save(s, "tsmodule.pt")


if __name__ == "__main__":
    main()
