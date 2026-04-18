import os
import pickle
from typing import Dict

import gym
import numpy as np
import ray
import ray.tune.callback
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import get_trainable_cls
from soccer_twos import AgentInterface, EnvType
from utils import create_rllib_env, soccer_twos

ALGORITHM = "PPO"
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "./ceia_params.pkl",
)
WEIGHT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "./ray_selfplay_weights3.pkl",
)
POLICY_NAME = "default"  # this may be useful when training with selfplay

class RayAgent(AgentInterface):
    """
    RayAgent is an agent that uses ray to train a model.
    """

    def __init__(self, env: gym.Env):
        """Initialize the RayAgent.
        Args:
            env: the competition environment.
        """
        super().__init__()
        ray.init(ignore_reinit_error=True)
        with open(CHECKPOINT_PATH, "rb") as f:
            config = pickle.load(f)

        # no need for parallelism on evaluation
        config["num_workers"] = 0
        config["num_gpus"] = 0
        config["model"] = {
                            "vf_share_layers": False,
                        "fcnet_hiddens": [256, 256],
                        "fcnet_activation": "relu",
        }
        temp_env = create_rllib_env(
            {"variation": EnvType.multiagent_team, "expanded_obs":True}
        )
        obs_space = temp_env.observation_space
        act_space = temp_env.action_space
        temp_env.close()
        config["multiagent"]["policies"]["default"] = (None, obs_space, act_space, {})

        tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())
        config["env"] = "DummyEnv"

        # create the Trainer from config
        cls = get_trainable_cls(ALGORITHM)
        agent = cls(env=config["env"], config=config)
        # load state from checkpoint
        # get policy for evaluation
        self.policy = agent.get_policy(POLICY_NAME)

        with open(WEIGHT_PATH, "rb") as f:
            loaded_weights = pickle.load(f)

        self.policy.set_weights(loaded_weights)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """The act method is called when the agent is asked to act.
        Args:
            observation: a dictionary where keys are team member ids and
                values are their corresponding observations of the environment,
                as numpy arrays.
        Returns:
            action: a dictionary where keys are team member ids and values
                are their corresponding actions, as np.arrays.
        """
        computed_actions = self.policy.compute_single_action(np.concatenate((observation[0]["base"], observation[1]["base"], observation[0]["expanded"])))[0]
        action_0, action_1 =np.split(computed_actions,2)
        # for player_id in observation:
        #     # compute_single_action returns a tuple of (action, action_info, ...)
        #     # as we only need the action, we discard the other elements
        #     actions[player_id], *_ = self.policy.compute_single_action(
        #         observation[player_id]
        #     )
        return {
            0:action_0,
            1:action_1,
        }
