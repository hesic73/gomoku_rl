import torch
from torchrl.data.tensor_specs import TensorSpec, DiscreteTensorSpec
from gokomu_rl.algo.dqn import make_actor, make_actor_explore
from tensordict import TensorDict

if __name__ == "__main__":
    board_size = 9
    num_envs = 128
    device = "cuda"
    action_spec = DiscreteTensorSpec(
        board_size * board_size,
        shape=[
            num_envs,
        ],
        device=device,
    )
    actor = make_actor(
        action_spec=action_spec,
        observation_key="observation",
        action_space_size=board_size * board_size,
        device="cuda",
    )

    actor_explore = make_actor_explore(
        actor=actor, annealing_num_steps=10000, eps_init=1.0, eps_end=0.05
    )

    td = TensorDict({}, batch_size=num_envs, device=device)
    observation = torch.randn(num_envs, 4, board_size, board_size, device=device)
    td.update({"observation": observation})

    td = actor_explore(td)
    print(td)
