import abc
from typing import Dict, Type
from tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.tensor_specs import DiscreteTensorSpec, TensorSpec
from omegaconf import DictConfig


class Policy(abc.ABC):

    REGISTRY: dict[str, Type["Policy"]] = {}

    @classmethod
    def __init_subclass__(cls, **kwargs):
        if cls.__name__ in Policy.REGISTRY:
            raise ValueError
        super().__init_subclass__(**kwargs)
        Policy.REGISTRY[cls.__name__] = cls
        Policy.REGISTRY[cls.__name__.lower()] = cls

    @abc.abstractmethod
    def __init__(
        self,
        cfg: DictConfig,
        action_spec: DiscreteTensorSpec,
        observation_spec: TensorSpec,
        device="cuda",
    ):
        """Initializes the policy.

        Args:
            cfg (DictConfig): Configuration object containing policy-specific settings.
            action_spec (DiscreteTensorSpec): Specification of the action space.
            observation_spec (TensorSpec): Specification of the observation space.
            device: The device (e.g., 'cuda' or 'cpu') where the policy's tensors will be allocated. Defaults to 'cuda'.
        """
        ...

    @abc.abstractmethod
    def __call__(self, tensordict: TensorDict) -> TensorDict:
        """Defines the computation performed at every call of the policy.

        Args:
            tensordict (TensorDict): Input tensor dictionary containing at least the observation data.

        Returns:
            TensorDict: Output tensor dictionary containing at least the actions to be taken in the environment.
        """
        ...

    @abc.abstractmethod
    def learn(self, data: TensorDict) -> Dict:
        """Updates the policy based on a batch of data.

        Args:
            data (TensorDict): A batch of data typically including observations, actions, rewards, and next observations.

        Returns:
            Dict: A dictionary containing information about the learning step, such as loss values.
        """
        ...

    @abc.abstractmethod
    def state_dict(self) -> Dict:
        """Returns the state of the policy as a dictionary.

        Returns:
            Dict: the state of the policy.
        """
        ...

    @abc.abstractmethod
    def load_state_dict(self, state_dict: Dict):
        """Loads the policy state from a dictionary.

        Args:
            state_dict (Dict): the state of the policy.
        """
        ...

    @abc.abstractmethod
    def train(self):
        """Sets the policy to training mode.
        """
        ...

    @abc.abstractmethod
    def eval(self):
        """Sets the policy to evaluation mode.
        """
        ...
