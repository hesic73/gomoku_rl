import hydra
from omegaconf import DictConfig, OmegaConf

from gomoku_rl import CONFIG_PATH
from pprint import pprint

from gomoku_rl.algo.dqn import make_actor
from gomoku_rl.eval import evaluate
from gomoku_rl.env import GomokuEnv
from torchrl.envs import TransformedEnv
import torch


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="eval")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    # pprint(OmegaConf.to_container(cfg))
    base_env = GomokuEnv(
        num_envs=1, board_size=cfg.board_size, device=cfg.device
    )
    env = TransformedEnv(base_env)
    actor_0 = make_actor(
        cfg=cfg.algo.actor,
        action_spec=env.action_spec,
        observation_key="observation",
        action_space_size=env.action_spec.space.n,
        device=env.device,
    )
    actor_0.load_state_dict(torch.load("dqn_actor.pt"))

    actor_1 = lambda *args, **kwargs: env.action_spec.rand()
    evaluate(actor_0, actor_1, env)


if __name__ == "__main__":
    main()
