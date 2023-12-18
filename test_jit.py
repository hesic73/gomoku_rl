from gomoku_rl.utils.eval import get_payoff_matrix
from gomoku_rl.utils.visual import annotate_heatmap, heatmap
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from gomoku_rl import CONFIG_PATH
import torch

import matplotlib.pyplot as plt

from gomoku_rl.env import GomokuEnv
from gomoku_rl.utils.policy import uniform_policy
from gomoku_rl.utils.module import ActorNet
from gomoku_rl.policy import get_pretrained_policy
import numpy as np
import functools
import copy


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
    # encoder = player.actor.module[0].module
    # policy_head = player.actor.module[1].module

    # actor = ActorNet(
    #     encoder,
    #     out_features=env.action_spec.space.n,
    #     num_channels=cfg.algo.num_channels,
    # )
    # actor.policy_head = policy_head
    # actor.eval().cpu()
    actor=player.actor.eval().cpu()

    tensordict = env.reset()[[0]].cpu()
    # s = torch.jit.script(
    #     actor, example_inputs=[tensordict["observation"], tensordict["action_mask"]]
    # )
    s = torch.jit.trace(actor, (tensordict["observation"], tensordict["action_mask"]))
    print(s)
    torch.jit.save(s, "tsmodule.pt")


if __name__ == "__main__":
    main()
