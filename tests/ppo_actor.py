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
import logging

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import SamplerWithoutReplacement,ReplayBuffer,LazyTensorStorage

from gomoku_rl.algo.ppo import make_actor,make_critic

import torch


from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)  
    
    
    lr = 3e-4
    max_grad_norm = 1.0
    
    sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
    num_epochs = 10  # optimisation steps per batch of data collected
    clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 1e-4
    

    initial_policy=None
        
    base_env = GomokuEnvWithOpponent(
        num_envs=cfg.num_envs,
        board_size=cfg.board_size, 
        device=cfg.device,
        initial_policy=initial_policy
    )
    
    transforms = []
    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    total_frames: int = cfg.total_frames
    frames_per_batch: int = cfg.frames_per_batch
    total_frames: int = total_frames // frames_per_batch * frames_per_batch
    
    actor=make_actor(cfg.algo.actor,action_spec=env.action_spec,device=cfg.device)
    actor(env.fake_tensordict())
    
    critic=make_critic(cfg.algo.critic,device=cfg.device)
    critic(env.fake_tensordict())
    
    
    collector = SyncDataCollector(
    env,
    actor,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=cfg.device,
)

    replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(frames_per_batch,device=cfg.device),
    sampler=SamplerWithoutReplacement(),
)
    advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=critic, average_gae=True
)

    loss_module = ClipPPOLoss(
    actor=actor,
    critic=critic,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # these keys match by default but we set this for completeness
    value_target_key=advantage_module.value_target_key,
    critic_coef=1.0,
    gamma=0.99,
    loss_critic_type="smooth_l1",
)

    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)

    
    pbar = tqdm(total=total_frames)

# We iterate over the collector until it reaches the total number of frames it was
# designed to collect:
    for i, tensordict_data in enumerate(collector):
    # we now have a batch of data to work with. Let's learn something from it.
        for _ in range(num_epochs):
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
            with torch.no_grad():
                advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view)
            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata.to(cfg.device))
                loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # Optimization: backward, grad clipping and optim step
                loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        pbar.update(tensordict_data.numel())
    
    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()


    


if __name__ == "__main__":
    main()
