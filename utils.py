from pprint import pprint
from random import uniform as randfloat

import gym
import numpy as np
import soccer_twos
from ray.rllib import MultiAgentEnv


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    pass
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # pprint(info)
        team_1_info = np.array([
            np.concatenate((player["player_info"]["position"],player["player_info"]["velocity"],[(player["player_info"]["rotation_y"]+270)%360])) for team in info.values() for player in team.values()
        ], dtype=np.float32).flatten()
        team_1_info = np.concatenate((team_1_info,info[0][0]["ball_info"]["position"],info[0][0]["ball_info"]["velocity"] ))
        team_2_info = np.array([
            np.concatenate((player["player_info"]["position"]*-1,player["player_info"]["velocity"]*-1,[(player["player_info"]["rotation_y"]+90)%360])) for team in reversed(info.values()) for player in team.values()
        ], dtype=np.float32).flatten()
        team_2_info = np.concatenate((team_2_info,info[0][0]["ball_info"]["position"]*-1,info[0][0]["ball_info"]["velocity"] *-1))
        # pprint(team_2_info.shape)
        # pprint(team_2_info)
        # pprint(info)
        # pprint(team_2_info.dtype)
        # pprint(info)
        # pprint(len(obs))
        # pprint(obs[0].dtype)
        # print(obs.shape)
        return obs, reward, done, info
        


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    # env = TransitionRecorderWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s
