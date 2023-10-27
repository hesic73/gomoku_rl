import torch
from torchrl.data.tensor_specs import TensorSpec, DiscreteTensorSpec
from gokomu_rl.learning.dqn import make_actor
from tensordict import TensorDict

if __name__ == "__main__":
    board_size = 9
    num_envs = 128
    device = "cuda"
    action_spec = DiscreteTensorSpec(
        board_size * board_size,
        shape=[num_envs, 1],
        device=device,
    )
    actor = make_actor(
        action_spec=action_spec,
        observation_key="observation",
        action_space_size=board_size * board_size,
        device="cuda",
    )

    td = TensorDict({}, batch_size=num_envs, device=device)
    observation = torch.randn(num_envs, 4, board_size, board_size, device=device)
    td.update({"observation": observation})

    td = actor(td)
    print(td)
