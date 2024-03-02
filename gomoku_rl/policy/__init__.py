from .base import Policy
from .ppo import PPO
from .dqn import DQN

from torchrl.data.tensor_specs import DiscreteTensorSpec, TensorSpec
from omegaconf import DictConfig
import torch


def get_policy(
    name: str,
    cfg: DictConfig,
    action_spec: DiscreteTensorSpec,
    observation_spec: TensorSpec,
    device="cuda",
) -> Policy:
    """
    Retrieves a policy object based on the specified policy name, configuration, action and observation specifications, and device.

    Args:
        name (str): The name of the policy to retrieve, which should match a key in the Policy registry.
        cfg (DictConfig): Configuration settings for the policy, typically containing hyperparameters and other policy-specific settings.
        action_spec (DiscreteTensorSpec): The specification of the action space, defining the shape, type, and bounds of actions the policy can take.
        observation_spec (TensorSpec): The specification of the observation space, defining the shape and type of observations the policy will receive from the environment.
        device: The computing device ('cuda' or 'cpu') where the policy computations will be performed. Defaults to "cuda".

    Returns:
        Policy: An instance of the requested policy class, initialized with the provided configurations, action and observation specifications, and device.
    """

    cls = Policy.REGISTRY[name.lower()]
    return cls(
        cfg=cfg,
        action_spec=action_spec,
        observation_spec=observation_spec,
        device=device,
    )


def get_pretrained_policy(
    name: str,
    cfg: DictConfig,
    action_spec: DiscreteTensorSpec,
    observation_spec: TensorSpec,
    checkpoint_path: str,
    device="cuda",
) -> Policy:
    """
    Initializes and returns a pretrained policy object based on the specified policy name, configuration, action and observation specifications, checkpoint path, and device.

    Args:
        name (str): The name of the policy to be loaded, corresponding to a key in the Policy registry.
        cfg (DictConfig): Configuration settings for the policy, typically containing hyperparameters and other policy-specific settings.
        action_spec (DiscreteTensorSpec): The specification of the action space, detailing the shape, type, and bounds of actions the policy can execute.
        observation_spec (TensorSpec): The specification of the observation space, detailing the shape and type of observations the policy will receive from the environment.
        checkpoint_path (str): The file path to the saved model checkpoint from which the policy's state should be loaded.
        device: The computing device ('cuda' or 'cpu') on which the policy computations will be executed. Defaults to "cuda".

    Returns:
        Policy: An instance of the specified policy class, initialized with the provided configurations, action and observation specifications, and pretrained weights loaded from the given checkpoint path.
    """
    policy = get_policy(
        name=name,
        cfg=cfg,
        action_spec=action_spec,
        observation_spec=observation_spec,
        device=device,
    )
    policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return policy
