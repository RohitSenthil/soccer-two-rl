import os
import pickle
from typing import Dict

import gym
import numpy as np
import ray
import ray.rllib.agents.ppo.ppo_torch_policy
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.tune.registry import get_trainable_cls
from soccer_twos import AgentInterface, EnvType
from utils import create_rllib_env

ALGORITHM = "PPO"
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "./ray_results/PPO_selfplay_rec_expandedobs_fixed2_2/PPO_Soccer_c90f3_00000_0_2026-04-15_14-06-11/checkpoint_000633/checkpoint-633",
)
POLICY_NAME = "main"
ray.init(ignore_reinit_error=True)

# Load configuration from checkpoint file.
config_path = ""
if CHECKPOINT_PATH:
    config_dir = os.path.dirname(CHECKPOINT_PATH)
    config_path = os.path.join(config_dir, "params.pkl")
    # Try parent directory.
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")

# Load the config from pickled.
if os.path.exists(config_path):
    with open(config_path, "rb") as f:
        config = pickle.load(f)
else:
    # If no config in given checkpoint -> Error.
    raise ValueError(
        "Could not find params.pkl in either the checkpoint dir or "
        "its parent directory!"
    )

# no need for parallelism on evaluation

config["num_workers"] = 0
config["num_gpus"] = 0
config["disable_env_checking"] = True

class FakeEnv(BaseEnv):
    @property
    def observation_space(self) -> gym.Space:
        return gym.spaces.Discrete(1)
    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Discrete(1)

tune.registry.register_env("DummyEnv", lambda *_: FakeEnv())
config["env"] = "DummyEnv"

cls = get_trainable_cls(ALGORITHM)
agent = cls(env=config["env"], config=config)

agent.restore(CHECKPOINT_PATH)
policy = agent.get_policy(POLICY_NAME)

with open("./new_pol_weights.pkl", "wb") as f:
    pickle.dump(policy.get_weights(), f)