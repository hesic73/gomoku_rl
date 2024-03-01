import torch
from tensordict import TensorDict
import os

from omegaconf import DictConfig, OmegaConf
from gomoku_rl import CONFIG_PATH
from gomoku_rl.env import GomokuEnv
from gomoku_rl.utils.policy import uniform_policy
from gomoku_rl.utils.misc import set_seed
from gomoku_rl.collector import WhitePlayCollector
from gomoku_rl.policy import get_policy


def main():
    device = "cuda:0"
    num_envs = 256
    steps = 50
    board_size = 10
    seed = 1234
    set_seed(seed)
    env = GomokuEnv(num_envs=num_envs, board_size=board_size, device=device)

    ppo_config = OmegaConf.load(os.path.join(CONFIG_PATH, "algo", "ppo.yaml"))
    agent = get_policy(ppo_config.name, ppo_config,
                       env.action_spec, env.observation_spec, device)

    collector = WhitePlayCollector(env, uniform_policy, agent)

    epochs = 30

    for epoch in range(epochs):
        white_transitions, info = collector.rollout(steps)
        info.update(agent.learn(white_transitions))

        print(f"Epoch:{epoch:02d}")
        print(OmegaConf.to_yaml(info))


if __name__ == "__main__":
    main()
