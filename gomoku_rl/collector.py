import abc
from collections import defaultdict
import torch
from tensordict import TensorDict

import time

from tensordict.nn import TensorDictModule, set_interaction_type, InteractionType


from gomoku_rl.utils.policy import _policy_t
from gomoku_rl.utils.log import get_log_func
from gomoku_rl.utils.augment import augment_transition

from .env import GomokuEnv


def make_transition(
    tensordict_t_minus_1: TensorDict,
    tensordict_t: TensorDict,
    tensordict_t_plus_1: TensorDict,
) -> TensorDict:
    """
    Constructs a transition tensor dictionary for a two-player game by integrating the game state and actions from three consecutive time steps (t-1, t, and t+1).

    Args:
        tensordict_t_minus_1 (TensorDict): A tensor dictionary containing the game state and associated information at time t-1.
        tensordict_t (TensorDict): A tensor dictionary containing the game state and associated information at time t.
        tensordict_t_plus_1 (TensorDict): A tensor dictionary containing the game state and associated information at time t+1.

    Returns:
        TensorDict: A new tensor dictionary representing the transition from time t-1 to t+1.

    The function calculates rewards based on the win status at times t and t+1, and flags the transition as done if the game ends at either time t or t+1. The resulting tensor dictionary is structured to facilitate learning from this transition in reinforcement learning algorithms.
    """
    # if a player wins at time t, its opponent cannot win immediately after reset
    reward: torch.Tensor = (
        tensordict_t.get("win").float() -
        tensordict_t_plus_1.get("win").float()
    ).unsqueeze(-1)
    transition: TensorDict = tensordict_t_minus_1.select(
        "observation",
        "action_mask",
        "action",
        "sample_log_prob",
        "state_value",
        strict=False,
    )
    transition.set(
        "next",
        tensordict_t_plus_1.select(
            "observation", "action_mask", "state_value", strict=False
        ),
    )
    transition.set(("next", "reward"), reward)
    done = tensordict_t_plus_1["done"] | tensordict_t["done"]
    transition.set(("next", "done"), done)

    return transition


def round(env: GomokuEnv,
          policy_black: _policy_t,
          policy_white: _policy_t,
          tensordict_t_minus_1: TensorDict,
          tensordict_t: TensorDict,
          return_black_transitions: bool = True,
          return_white_transitions: bool = True,):
    """Executes two sequential steps in the Gomoku environment, applying black and white policies alternately.

    Args:
        env (GomokuEnv): The Gomoku game environment instance.
        policy_black (_policy_t): The policy function for the black player, which determines the action based on the current game state.
        policy_white (_policy_t): The policy function for the white player, similar to policy_black but for the white player.
        tensordict_t_minus_1 (TensorDict): The game state tensor dictionary at time t-1, before the white player's action.
        tensordict_t (TensorDict): The game state tensor dictionary at time t, before the black player's action.
        return_black_transitions (bool, optional): If True, returns transition data for the black player. Defaults to True.
        return_white_transitions (bool, optional): If True, returns transition data for the white player. Defaults to True.

    Returns:
        tuple: Contains transition data for the black player (if requested), transition data for the white player (if requested), the game state after the white player's action (t+1), and the game state after the black player's action (t+2).\

    Note:
        - If the environment is reset at time t-1, the white player won't make a move at time t. This is different from the black player's behavior.
        - If the black player wins at time t and the environment is reset, the environment does not take a step at t+1. This affects the validity of the white player's transition from t-1 to t+1, which is marked invalid where `tensordict_t_minus_1['done']` is True.

    """

    tensordict_t_plus_1 = env.step_and_maybe_reset(
        tensordict=tensordict_t)

    with set_interaction_type(type=InteractionType.RANDOM):
        tensordict_t_plus_1 = policy_white(tensordict_t_plus_1)

    if return_white_transitions:
        transition_white = make_transition(
            tensordict_t_minus_1, tensordict_t, tensordict_t_plus_1
        )
        # for player_white, if the env is reset at t-1, he won't make a move at t
        # this is different from player_black

        # transition_white = transition_white[~tensordict_t_minus_1["done"]]
        # the trick is that we set done=True for computing gae
        # after that we just discard these invalid transition
        invalid: torch.Tensor = tensordict_t_minus_1["done"]
        # transition_white["observation"] = (
        #     -invalid.float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        #     + (1 - invalid.float()).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        #     * transition_white["observation"]
        # )
        transition_white["next", "done"] = (
            invalid | transition_white["next", "done"]
        )
        transition_white.set("invalid", invalid)

    else:
        transition_white = None

    # if player_black wins at t (and the env is reset), the env doesn't take a step at t+1
    # this makes no difference to player_black's transition from t to t+2
    # but player_white's transition from t-1 to t+1 is invalid where tensordict_t_minus_1['done']==True
    tensordict_t_plus_2 = env.step_and_maybe_reset(
        tensordict_t_plus_1,
        env_mask=~tensordict_t_plus_1.get("done"),
    )

    with set_interaction_type(type=InteractionType.RANDOM):
        tensordict_t_plus_2 = policy_black(tensordict_t_plus_2)

    if return_black_transitions:
        transition_black = make_transition(
            tensordict_t, tensordict_t_plus_1, tensordict_t_plus_2
        )
        transition_black.set(
            "invalid",
            torch.zeros(env.num_envs, device=env.device,
                        dtype=torch.bool),
        )
    else:
        transition_black = None

    return (
        transition_black,
        transition_white,
        tensordict_t_plus_1,
        tensordict_t_plus_2,
    )


def self_play_step(
    env: GomokuEnv,
    policy: _policy_t,
    tensordict_t_minus_1: TensorDict,
    tensordict_t: TensorDict,
):
    """Executes a single step of self-play in a Gomoku environment using a specified policy.

    Args:
        env (GomokuEnv): The Gomoku game environment instance where the self-play step is executed.
        policy (_policy_t): The policy function to determine the next action based on the current game state.
        tensordict_t_minus_1 (TensorDict):  The game state tensor dictionary at time t-1.
        tensordict_t (TensorDict): The game state tensor dictionary at time t.

    Returns:
        tuple: Contains the transition information resulting from the action taken in this step, the game state tensor dictionary at time t (unchanged from the input), and the game state tensor dictionary at time t+1 after the action and potential reset.

        - The transition information includes the states at times t-1 and t, the action taken at time t, and the resulting state at time t+1.
        - The unchanged game state tensor dictionary at time t is returned to facilitate chaining of self-play steps or integration with other functions.
        - The updated game state tensor dictionary at time t+1 reflects the new state of the environment after applying the action and potentially resetting the environment if the game concluded in this step.

    """
    tensordict_t_plus_1 = env.step_and_maybe_reset(
        tensordict=tensordict_t
    )
    with set_interaction_type(type=InteractionType.RANDOM):
        tensordict_t_plus_1 = policy(tensordict_t_plus_1)
    transition = make_transition(
        tensordict_t_minus_1, tensordict_t, tensordict_t_plus_1
    )
    return (
        transition,
        tensordict_t,
        tensordict_t_plus_1,
    )


class Collector(abc.ABC):
    @abc.abstractmethod
    def rollout(self, steps: int) -> tuple[TensorDict, dict]:
        ...

    @abc.abstractmethod
    def reset(self):
        """
        Resets the collector's internal state.

        """
        ...


class SelfPlayCollector(Collector):
    def __init__(self, env: GomokuEnv, policy: _policy_t, out_device=None, augment: bool = False):
        """Initializes a collector for self-play data in a Gomoku environment.

        This collector facilitates the collection of game transitions generated through self-play, where both players use the same policy.

        Args:
            env (GomokuEnv): The Gomoku game environment where self-play will be conducted.
            policy (_policy_t): The policy function to be used for both players during self-play.
            out_device: The device on which collected data will be stored. If None, uses the device specified by the environment.
            augment (bool, optional): If True, applies data augmentation to the collected transitions. Defaults to False.

        """
        self._env = env
        self._policy = policy
        self._out_device = out_device or self._env.device
        self._augment = augment

        self._t = None
        self._t_minus_1 = None

    def reset(self):
        self._env.reset()
        self._t = None
        self._t_minus_1 = None

    @torch.no_grad()
    def rollout(self, steps: int) -> tuple[TensorDict, dict]:
        """
        Executes a rollout in the environment, collecting data for a specified number of steps.

        Args:
            steps (int): The number of steps to execute in the environment for this rollout.

        Returns:
            tuple[TensorDict, dict]: A tuple containing two elements:
                - A TensorDict holding the collected transitions from the rollout. Each transition includes the game state before the action, the action taken, and the resulting state.
                - A dictionary with additional information about the rollout.
        """
        info: defaultdict[str, float] = defaultdict(float)
        self._env.set_post_step(get_log_func(info))

        tensordicts = []

        start = time.perf_counter()

        if self._t_minus_1 is None and self._t is None:
            self._t_minus_1 = self._env.reset()
            with set_interaction_type(type=InteractionType.RANDOM):
                self._t_minus_1 = self._policy(self._t_minus_1)
            self._t = self._env.step(self._t_minus_1)
            with set_interaction_type(type=InteractionType.RANDOM):
                self._t = self._policy(self._t)

        for i in range(steps - 1):
            (
                transition,
                self._t_minus_1,
                self._t,
            ) = self_play_step(self._env, self._policy, self._t_minus_1, self._t)

            # truncate the last transition
            if i == steps-2:
                transition["next", "done"] = torch.ones(
                    transition["next", "done"].shape, dtype=torch.bool, device=transition.device)

            if self._augment:
                transition = augment_transition(transition)

            tensordicts.append(transition.to(self._out_device))

        end = time.perf_counter()
        fps = (steps * self._env.num_envs) / (end - start)

        self._env.set_post_step(None)

        tensordicts = torch.stack(tensordicts, dim=-1)

        info.update({"fps": fps})

        return tensordicts, dict(info)


class VersusPlayCollector(Collector):
    def __init__(self, env: GomokuEnv, policy_black: _policy_t, policy_white: _policy_t, out_device=None, augment: bool = False):
        """Initializes a collector for versus play data in a Gomoku environment, facilitating the collection of game transitions where two players, each using a distinct policy, compete against each other.

        Args:
            env (GomokuEnv): The Gomoku game environment where the two-player game will be conducted.
            policy_black (_policy_t): The policy function to be used for the black player.
            policy_white (_policy_t): The policy function to be used for the white player.
            out_device: The device on which collected data will be stored. If None, uses the device specified by the environment.
            augment (bool, optional): If True, applies data augmentation to the collected transitions. Defaults to False.

        """
        self._env = env
        self._policy_black = policy_black
        self._policy_white = policy_white
        self._out_device = out_device or self._env.device
        self._augment = augment

        self._t_minus_1 = None
        self._t = None

    def reset(self):
        self._env.reset()
        self._t_minus_1 = None
        self._t = None

    @torch.no_grad()
    def rollout(self, steps: int) -> tuple[TensorDict, TensorDict, dict]:
        """Executes a rollout in the environment, collecting data for a specified number of steps, alternating between the black and white policies.

        Args:
            steps (int): The number of steps to execute in the environment for this rollout. It is adjusted to be an even number to ensure an equal number of actions for both players.

        Returns:
            tuple: A tuple containing three elements:
                - A TensorDict of transitions collected for the black player, with each transition representing a game state before the black player's action, the action taken, and the resulting state.
                - A TensorDict of transitions collected for the white player, structured similarly to the black player's transitions. Note that for the first step, the white player does not take an action, so their collection starts from the second step.
                - A dictionary containing additional information about the rollout.

        """

        steps = (steps//2)*2

        info: defaultdict[str, float] = defaultdict(float)
        self._env.set_post_step(get_log_func(info))

        blacks = []
        whites = []

        start = time.perf_counter()

        if self._t_minus_1 is None and self._t is None:
            self._t_minus_1 = self._env.reset()
            self._t = self._env.reset()

            self._t_minus_1.update(
                {
                    "done": torch.ones(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "win": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                }
            )  # here we set it to True
            with set_interaction_type(type=InteractionType.RANDOM):
                self._t = self._policy_black(self._t)

            self._t .update(
                {
                    "done": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "win": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                }
            )

        for i in range(steps//2):
            (
                transition_black,
                transition_white,
                self._t_minus_1,
                self._t,
            ) = round(self._env, self._policy_black, self._policy_white, self._t_minus_1, self._t)

            # truncate the last transition
            if i == steps//2-1:
                transition_black["next", "done"] = torch.ones(
                    transition_black["next", "done"].shape, dtype=torch.bool, device=transition_black.device)
                transition_white["next", "done"] = torch.ones(
                    transition_white["next", "done"].shape, dtype=torch.bool, device=transition_white.device)

            if self._augment:
                transition_black = augment_transition(transition_black)
                if i != 0:
                    transition_white = augment_transition(transition_white)

            blacks.append(transition_black.to(self._out_device))
            if i != 0:
                whites.append(transition_white.to(self._out_device))

        blacks = torch.stack(blacks, dim=-1) if blacks else None
        whites = torch.stack(whites, dim=-1) if whites else None

        end = time.perf_counter()
        fps = (steps * self._env.num_envs) / (end - start)

        self._env.set_post_step(None)

        info.update({"fps": fps})

        return blacks, whites, dict(info)


class BlackPlayCollector(Collector):
    def __init__(self, env: GomokuEnv, policy_black: _policy_t, policy_white: _policy_t, out_device=None, augment: bool = False):
        """
        Initializes a collector for capturing game transitions where the black player is controlled by a trainable policy against a white player using a fixed policy.

        Args:
            env (GomokuEnv): The game environment where the collection takes place.
            policy_black (_policy_t): The trainable policy used for the black player.
            policy_white (_policy_t): The fixed policy for the white player, simulating a consistent opponent.
            out_device: The device (e.g., CPU, GPU) where the collected data will be stored. Defaults to the environment's device if not specified.
            augment (bool, optional): Whether to apply data augmentation to the collected transitions, enhancing the dataset's diversity. Defaults to False.
        """
        self._env = env
        self._policy_black = policy_black
        self._policy_white = policy_white
        self._out_device = out_device or self._env.device
        self._augment = augment

        self._t_minus_1 = None
        self._t = None

    def reset(self):
        self._env.reset()
        self._t_minus_1 = None
        self._t = None

    @torch.no_grad()
    def rollout(self, steps: int) -> tuple[TensorDict, dict]:
        """
        Executes a data collection session over a specified number of game steps, focusing on transitions involving the black player.

        Args:
            steps (int): The total number of steps to collect data for. This will be adjusted to ensure an even number of steps for symmetry in turn-taking.

        Returns:
            tuple[TensorDict, dict]: A tuple containing the collected transitions for the black player and a dictionary with additional information such as the frames per second (fps) achieved during the collection.
        """

        steps = (steps//2)*2

        info: defaultdict[str, float] = defaultdict(float)
        self._env.set_post_step(get_log_func(info))

        blacks = []

        start = time.perf_counter()

        if self._t_minus_1 is None and self._t is None:
            self._t_minus_1 = self._env.reset()
            self._t = self._env.reset()

            self._t_minus_1.update(
                {
                    "done": torch.ones(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "win": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                }
            )  # here we set it to True
            with set_interaction_type(type=InteractionType.RANDOM):
                self._t = self._policy_black(self._t)

            self._t .update(
                {
                    "done": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "win": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                }
            )

        for i in range(steps//2):
            (
                transition_black,
                transition_white,
                self._t_minus_1,
                self._t,
            ) = round(self._env, self._policy_black, self._policy_white, self._t_minus_1, self._t, return_black_transitions=True, return_white_transitions=False)

            # truncate the last transition
            if i == steps//2-1:
                transition_black["next", "done"] = torch.ones(
                    transition_black["next", "done"].shape, dtype=torch.bool, device=transition_black.device)

            if self._augment:
                transition_black = augment_transition(transition_black)

            blacks.append(transition_black.to(self._out_device))

        blacks = torch.stack(blacks, dim=-1) if blacks else None

        end = time.perf_counter()
        fps = (steps * self._env.num_envs) / (end - start)

        self._env.set_post_step(None)

        info.update({"fps": fps})

        return blacks, dict(info)


class WhitePlayCollector(Collector):
    def __init__(self, env: GomokuEnv, policy_black: _policy_t, policy_white: _policy_t, out_device=None, augment: bool = False):
        """
        Initializes a collector focused on capturing game transitions from the perspective of the white player, who is controlled by a trainable policy, against a black player using a fixed policy.

        Args:
            env (GomokuEnv): The game environment where the collection takes place.
            policy_black (_policy_t): The fixed policy for the black player, providing a consistent challenge.
            policy_white (_policy_t): The trainable policy used for the white player.
            out_device: The device for storing collected data, defaulting to the environment's device if not specified.
            augment (bool, optional): Indicates whether to augment the collected transitions to enhance the dataset. Defaults to False.
        """
        self._env = env
        self._policy_black = policy_black
        self._policy_white = policy_white
        self._out_device = out_device or self._env.device
        self._augment = augment

        self._t_minus_1 = None
        self._t = None

    def reset(self):
        self._env.reset()
        self._t_minus_1 = None
        self._t = None

    @torch.no_grad()
    def rollout(self, steps: int) -> tuple[TensorDict, dict]:
        """
        Performs a data collection session, focusing on the game transitions where the white player is active, over a specified number of steps.

        Args:
            steps (int): The number of steps for which data will be collected, adjusted to be even for fairness in gameplay.

        Returns:
            tuple[TensorDict, dict]: A tuple containing the collected transitions for the white player and additional session information, such as collection performance (fps).
        """
        steps = (steps//2)*2

        info: defaultdict[str, float] = defaultdict(float)
        self._env.set_post_step(get_log_func(info))

        whites = []

        start = time.perf_counter()

        if self._t_minus_1 is None and self._t is None:
            self._t_minus_1 = self._env.reset()
            self._t = self._env.reset()

            self._t_minus_1.update(
                {
                    "done": torch.ones(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "win": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ), "action": -torch.ones(
                        self._env.num_envs, dtype=torch.long, device=self._env.device
                    ),  # placeholder or the action key will not appear in the stacked tensordict.
                }
            )  # here we set it to True
            with set_interaction_type(type=InteractionType.RANDOM):
                self._t = self._policy_black(self._t)

            self._t .update(
                {
                    "done": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "win": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                }
            )

        for i in range(steps//2):
            (
                transition_black,
                transition_white,
                self._t_minus_1,
                self._t,
            ) = round(self._env, self._policy_black, self._policy_white, self._t_minus_1, self._t, return_black_transitions=False, return_white_transitions=True)

            # truncate the last transition
            if i == steps//2-1:
                transition_white["next", "done"] = torch.ones(
                    transition_white["next", "done"].shape, dtype=torch.bool, device=transition_white.device)

            if self._augment:
                if i != 0 and len(transition_white) > 0:
                    transition_white = augment_transition(transition_white)

            if i != 0:
                whites.append(transition_white.to(self._out_device))

        whites = torch.stack(whites, dim=-1) if whites else None

        end = time.perf_counter()
        fps = (steps * self._env.num_envs) / (end - start)

        self._env.set_post_step(None)

        info.update({"fps": fps})

        return whites, dict(info)
