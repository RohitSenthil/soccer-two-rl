from pprint import pprint
from random import uniform as randfloat

import gym
import numpy as np
import soccer_twos
from ray.rllib import MultiAgentEnv
from soccer_twos import EnvType, MultiagentTeamWrapper, MultiAgentUnityWrapper


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    def reset(self):
        obs = self.env.reset()
        if self.env.expanded_obs and type(self.env.env)==MultiagentTeamWrapper:
            action = np.array([0, 0, 0, 0, 0, 0], dtype=np.int64)
            obs, reward, done, info = self.env.step({0:action,1:action})
            team_1_info = np.array([
                np.concatenate((player["player_info"]["position"],player["player_info"]["velocity"],[(player["player_info"]["rotation_y"]+270)%360])) for team in info.values() for player in team.values()
            ], dtype=np.float32).flatten()
            team_1_info = np.concatenate((team_1_info,info[0][0]["ball_info"]["position"],info[0][0]["ball_info"]["velocity"] ))
            obs[0] = {"base" :obs[0] , "expanded": np.concatenate((obs[0], team_1_info))}
            team_2_info = np.array([
                np.concatenate((player["player_info"]["position"]*-1,player["player_info"]["velocity"]*-1,[(player["player_info"]["rotation_y"]+90)%360])) for team in reversed(info.values()) for player in team.values()
            ], dtype=np.float32).flatten()
            team_2_info = np.concatenate((team_2_info,info[0][0]["ball_info"]["position"]*-1,info[0][0]["ball_info"]["velocity"] *-1))
            obs[1] = {"base" :obs[1] , "expanded": np.concatenate((obs[1], team_2_info))}
        elif self.env.expanded_obs and type(self.env.env)==MultiAgentUnityWrapper:
            action = np.array([0, 0, 0, 0, 0, 0], dtype=np.int64)
            obs, reward, done, info = self.env.step({0:action,1:action})
            info = {
                0 : {0:info[0], 1:info[1]},
                1 : {0:info[2], 1:info[3]}
            }
            team_1_info = np.array([
                np.concatenate((player["player_info"]["position"],player["player_info"]["velocity"],[(player["player_info"]["rotation_y"]+270)%360])) for team in info.values() for player in team.values()
            ], dtype=np.float32).flatten()
            team_1_info = np.concatenate((team_1_info,info[0][0]["ball_info"]["position"],info[0][0]["ball_info"]["velocity"] ))
            team_2_info = np.array([
                np.concatenate((player["player_info"]["position"]*-1,player["player_info"]["velocity"]*-1,[(player["player_info"]["rotation_y"]+90)%360])) for team in reversed(info.values()) for player in team.values()
            ], dtype=np.float32).flatten()
            team_2_info = np.concatenate((team_2_info,info[0][0]["ball_info"]["position"]*-1,info[0][0]["ball_info"]["velocity"] *-1))
            obs[0] = {"base" :obs[0] , "expanded": team_1_info}
            obs[1] = {"base" :obs[1] , "expanded": team_1_info}
            obs[2] = {"base" :obs[2] , "expanded": team_2_info}
            obs[3] = {"base" :obs[3] , "expanded": team_2_info}
            # obs[1] = np.concatenate((obs[1], team_2_info))
            # print(team_1_info)
            # print(team_2_info)
            # team_1_info = np.load("team_1_start.npy")
            # team_2_info = np.load("team_2_start.npy")
            # obs[0] = np.concatenate((obs[0], team_1_info))
            # obs[1] = np.concatenate((obs[1], team_2_info))
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # pprint(info)
        if self.env.expanded_obs and type(self.env.env)==MultiagentTeamWrapper:
            team_1_info = np.array([
                np.concatenate((player["player_info"]["position"],player["player_info"]["velocity"],[(player["player_info"]["rotation_y"]+270)%360])) for team in info.values() for player in team.values()
            ], dtype=np.float32).flatten()
            team_1_info = np.concatenate((team_1_info,info[0][0]["ball_info"]["position"],info[0][0]["ball_info"]["velocity"] ))
            obs[0] = {"base" :obs[0] , "expanded": np.concatenate((obs[0], team_1_info))}
            team_2_info = np.array([
                np.concatenate((player["player_info"]["position"]*-1,player["player_info"]["velocity"]*-1,[(player["player_info"]["rotation_y"]+90)%360])) for team in reversed(info.values()) for player in team.values()
            ], dtype=np.float32).flatten()
            team_2_info = np.concatenate((team_2_info,info[0][0]["ball_info"]["position"]*-1,info[0][0]["ball_info"]["velocity"] *-1))
            obs[1] = {"base" :obs[1] , "expanded": np.concatenate((obs[1], team_2_info))}
        elif self.env.expanded_obs and type(self.env.env)==MultiAgentUnityWrapper:
            info = {
                0 : {0:info[0], 1:info[1]},
                1 : {0:info[2], 1:info[3]}
            }
            team_1_info = np.array([
                np.concatenate((player["player_info"]["position"],player["player_info"]["velocity"],[(player["player_info"]["rotation_y"]+270)%360])) for team in info.values() for player in team.values()
            ], dtype=np.float32).flatten()
            team_1_info = np.concatenate((team_1_info,info[0][0]["ball_info"]["position"],info[0][0]["ball_info"]["velocity"] ))
            team_2_info = np.array([
                np.concatenate((player["player_info"]["position"]*-1,player["player_info"]["velocity"]*-1,[(player["player_info"]["rotation_y"]+90)%360])) for team in reversed(info.values()) for player in team.values()
            ], dtype=np.float32).flatten()
            team_2_info = np.concatenate((team_2_info,info[0][0]["ball_info"]["position"]*-1,info[0][0]["ball_info"]["velocity"] *-1))
            obs[0] = {"base" :obs[0] , "expanded": team_1_info}
            obs[1] = {"base" :obs[1] , "expanded": team_1_info}
            obs[2] = {"base" :obs[2] , "expanded": team_2_info}
            obs[3] = {"base" :obs[3] , "expanded": team_2_info}
            # print(info, team_1_info, team_2_info)
            # print(team_1_info)
            # print(team_2_info)
            # np.save("team_1_start", team_1_info)
            # np.save("team_2_start", team_2_info)
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
    env.expanded_obs = env_config.get("expanded_obs", False)
    if env.expanded_obs:
        env.observation_space = gym.spaces.Box(
                -np.inf,
                np.inf, dtype=np.float32, shape=(672 + 24,)
        )
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
