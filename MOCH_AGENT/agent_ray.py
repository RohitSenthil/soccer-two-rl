import os
import pickle
import threading
from typing import Dict, Optional, Tuple

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.tune.registry import get_trainable_cls
from soccer_twos import AgentInterface

ALGORITHM = "PPO"
POLICY_NAME = "default"
_DUMMY_ENV_NAME = "MOCH_AGENT_DummyEnv"

BASE_PLAYER_OBS_DIM = 336
EXTRA_TEAM_FEATURE_DIM = 24
POLICY_OBS_DIM = BASE_PLAYER_OBS_DIM * 2 + EXTRA_TEAM_FEATURE_DIM  # 696
TEAM_ACTION_DIM = 6
PLAYER_ACTION_DIM = 3

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "./ceia_params.pkl",
)
WEIGHT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "./ray_selfplay_weights4.pkl",
)


def _to_1d_float_array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _fit_dim(array: np.ndarray, expected_dim: int) -> np.ndarray:
    out = np.zeros(expected_dim, dtype=np.float32)
    if array.size == 0:
        return out
    copy_dim = min(expected_dim, array.size)
    out[:copy_dim] = array[:copy_dim]
    return out


_ENV_INFO_CACHE_PATCHED = False
_EXTRA_CACHE = threading.local()


def _get_extra_cache():
    by_obs_pair = getattr(_EXTRA_CACHE, "by_obs_pair", None)
    by_obs = getattr(_EXTRA_CACHE, "by_obs", None)
    if by_obs_pair is None or by_obs is None:
        by_obs_pair = {}
        by_obs = {}
        _EXTRA_CACHE.by_obs_pair = by_obs_pair
        _EXTRA_CACHE.by_obs = by_obs
    return by_obs_pair, by_obs


def _obs_fingerprint(player_obs):
    if isinstance(player_obs, dict):
        player_obs = player_obs.get("base", [])
    arr = _to_1d_float_array(player_obs)
    if arr.size == 0:
        return None
    return (arr.size, arr.tobytes())


def _normalise_info(info):
    if not isinstance(info, dict) or not info:
        return None

    first_value = next(iter(info.values()))
    if isinstance(first_value, dict) and "player_info" in first_value:
        if not all(player_id in info for player_id in range(4)):
            return None
        return {
            0: {0: info[0], 1: info[1]},
            1: {0: info[2], 1: info[3]},
        }

    if not all(team_id in info for team_id in (0, 1)):
        return None
    return info


def _player_state_features(player_info, sign: float, rotation_offset: float) -> np.ndarray:
    player = player_info.get("player_info", {})
    position = _fit_dim(_to_1d_float_array(player.get("position", [])), 2) * sign
    velocity = _fit_dim(_to_1d_float_array(player.get("velocity", [])), 2) * sign
    rotation = _fit_dim(_to_1d_float_array([player.get("rotation_y", 0.0)]), 1)
    rotation = (rotation + rotation_offset) % 360
    return np.concatenate((position, velocity, rotation)).astype(np.float32, copy=False)


def _ball_state_features(player_info, sign: float) -> np.ndarray:
    ball = player_info.get("ball_info", {})
    position = _fit_dim(_to_1d_float_array(ball.get("position", [])), 2) * sign
    velocity = _fit_dim(_to_1d_float_array(ball.get("velocity", [])), 2) * sign
    return np.concatenate((position, velocity)).astype(np.float32, copy=False)


def _team_extra_features(info, team_id: int) -> Optional[np.ndarray]:
    # Matches the expanded observation ordering used in MOCH_AGENT2/utils.py.
    info = _normalise_info(info)
    if info is None:
        return None

    try:
        if team_id == 0:
            players = [
                _player_state_features(player, 1.0, 270.0)
                for team in info.values()
                for player in team.values()
            ]
            ball = _ball_state_features(info[0][0], 1.0)
        else:
            players = [
                _player_state_features(player, -1.0, 90.0)
                for team in reversed(list(info.values()))
                for player in team.values()
            ]
            ball = _ball_state_features(info[0][0], -1.0)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None

    return _fit_dim(np.concatenate((*players, ball)), EXTRA_TEAM_FEATURE_DIM)


def _remember_extra_for_obs(obs, player_ids, extra):
    fingerprints = [_obs_fingerprint(obs.get(player_id)) for player_id in player_ids]
    fingerprints = [fp for fp in fingerprints if fp is not None]
    if not fingerprints:
        return

    by_obs_pair, by_obs = _get_extra_cache()
    extra = _fit_dim(_to_1d_float_array(extra), EXTRA_TEAM_FEATURE_DIM)
    for fp in fingerprints:
        by_obs[fp] = extra
    if len(fingerprints) >= 2:
        by_obs_pair[tuple(fingerprints[:2])] = extra
        by_obs_pair[tuple(reversed(fingerprints[:2]))] = extra


def _cache_expanded_info(obs, info):
    team0_extra = _team_extra_features(info, 0)
    team1_extra = _team_extra_features(info, 1)
    if team0_extra is None or team1_extra is None or not isinstance(obs, dict):
        return

    by_obs_pair, by_obs = _get_extra_cache()
    by_obs_pair.clear()
    by_obs.clear()

    if all(player_id in obs for player_id in range(4)):
        _remember_extra_for_obs(obs, (0, 1), team0_extra)
        _remember_extra_for_obs(obs, (2, 3), team1_extra)
    elif all(team_id in obs for team_id in (0, 1)):
        _remember_extra_for_obs(obs, (0,), team0_extra)
        _remember_extra_for_obs(obs, (1,), team1_extra)


def _lookup_cached_extra(player0_obs, player1_obs) -> Optional[np.ndarray]:
    by_obs_pair, by_obs = _get_extra_cache()
    fp0 = _obs_fingerprint(player0_obs)
    fp1 = _obs_fingerprint(player1_obs)
    if fp0 is not None and fp1 is not None:
        extra = by_obs_pair.get((fp0, fp1))
        if extra is not None:
            return extra
    if fp0 is not None:
        extra = by_obs.get(fp0)
        if extra is not None:
            return extra
    if fp1 is not None:
        return by_obs.get(fp1)
    return None


def _install_env_info_cache_patch() -> None:
    global _ENV_INFO_CACHE_PATCHED
    if _ENV_INFO_CACHE_PATCHED:
        return

    try:
        from soccer_twos.wrappers import MultiAgentUnityWrapper
    except Exception:
        return

    if getattr(MultiAgentUnityWrapper, "_moch_agent_info_cache_patched", False):
        _ENV_INFO_CACHE_PATCHED = True
        return

    original_reset = MultiAgentUnityWrapper.reset
    original_step = MultiAgentUnityWrapper.step
    original_single_step = MultiAgentUnityWrapper._single_step

    def patched_single_step(self, info):
        result = original_single_step(self, info)
        records = getattr(self, "_moch_agent_single_step_records", None)
        if records is not None:
            observations, _, _, step_info = result
            records.append((observations, step_info))
        return result

    def patched_reset(self):
        self._moch_agent_single_step_records = []
        try:
            obs = original_reset(self)
            records = getattr(self, "_moch_agent_single_step_records", [])
        finally:
            self._moch_agent_single_step_records = None

        info = {}
        for group_id, (group_obs, group_info) in enumerate(
            records[-getattr(self, "num_groups", 0) :]
        ):
            group_size = len(group_obs)
            for member_id, member_info in group_info.items():
                info[group_id * group_size + member_id] = member_info
        _cache_expanded_info(obs, info)
        return obs

    def patched_step(self, action):
        result = original_step(self, action)
        obs, _, _, info = result
        _cache_expanded_info(obs, info)
        return result

    MultiAgentUnityWrapper._single_step = patched_single_step
    MultiAgentUnityWrapper.reset = patched_reset
    MultiAgentUnityWrapper.step = patched_step
    MultiAgentUnityWrapper._moch_agent_info_cache_patched = True
    _ENV_INFO_CACHE_PATCHED = True


class RayAgent(AgentInterface):
    """
    Ray policy agent for team-level actions with expanded observations.
    """

    def __init__(self, env: gym.Env):
        super().__init__()
        self.name = "MOCH_AGENT"
        _install_env_info_cache_patch()

        if not os.path.exists(CHECKPOINT_PATH):
            raise ValueError(f"Checkpoint config not found: {CHECKPOINT_PATH}")
        if not os.path.exists(WEIGHT_PATH):
            raise ValueError(f"Policy weights not found: {WEIGHT_PATH}")

        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            dashboard_host="127.0.0.1",
            _node_ip_address="127.0.0.1",
        )

        with open(CHECKPOINT_PATH, "rb") as f:
            config = pickle.load(f)

        # Inference-only settings.
        config["num_workers"] = 0
        config["num_gpus"] = 0
        config["model"] = {
            "vf_share_layers": False,
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        }

        obs_space = gym.spaces.Box(
            -np.inf, np.inf, dtype=np.float32, shape=(POLICY_OBS_DIM,)
        )
        act_space = gym.spaces.MultiDiscrete(np.repeat(3, TEAM_ACTION_DIM))
        config["multiagent"]["policies"]["default"] = (None, obs_space, act_space, {})

        try:
            tune.registry.register_env(_DUMMY_ENV_NAME, lambda *_: BaseEnv())
        except Exception:
            # Already registered in this interpreter/session.
            pass
        config["env"] = _DUMMY_ENV_NAME

        trainer_cls = get_trainable_cls(ALGORITHM)
        trainer = trainer_cls(env=config["env"], config=config)
        self._trainer = trainer

        self.policy = trainer.get_policy(POLICY_NAME)
        if self.policy is None:
            policy_map = trainer.workers.local_worker().policy_map
            if "default_policy" in policy_map:
                self.policy = trainer.get_policy("default_policy")
            else:
                self.policy = trainer.get_policy(next(iter(policy_map.keys())))

        with open(WEIGHT_PATH, "rb") as f:
            loaded_weights = pickle.load(f)
        self.policy.set_weights(loaded_weights)

    def _extract_player_obs(
        self, player_obs
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns:
          base_obs: player's base observation (possibly empty).
          extra_features: extra team features if available (possibly None).
          full_vector: full 696-dim policy vector if already provided (possibly None).
        """
        if isinstance(player_obs, dict):
            base = _to_1d_float_array(player_obs.get("base", []))
            expanded = _to_1d_float_array(player_obs.get("expanded", []))

            if expanded.size == POLICY_OBS_DIM:
                return np.zeros(0, dtype=np.float32), None, expanded
            if base.size == POLICY_OBS_DIM:
                return np.zeros(0, dtype=np.float32), None, base

            extra = None
            if expanded.size > 0:
                # Supports both "extra only" and "base+extra" payloads.
                if expanded.size >= EXTRA_TEAM_FEATURE_DIM:
                    extra = expanded[-EXTRA_TEAM_FEATURE_DIM:]
                else:
                    extra = _fit_dim(expanded, EXTRA_TEAM_FEATURE_DIM)
            return base, extra, None

        arr = _to_1d_float_array(player_obs)
        if arr.size == POLICY_OBS_DIM:
            return np.zeros(0, dtype=np.float32), None, arr
        return arr, None, None

    def _build_policy_observation(self, observation: Dict[int, np.ndarray]) -> np.ndarray:
        player0_obs = observation.get(0)
        player1_obs = observation.get(1)

        if player0_obs is None:
            player0_obs = next(iter(observation.values()))
        if player1_obs is None:
            player1_obs = player0_obs

        base0, extra0, full0 = self._extract_player_obs(player0_obs)
        base1, extra1, full1 = self._extract_player_obs(player1_obs)

        # If caller already gives the full vector, use it directly.
        if full0 is not None:
            return _fit_dim(full0, POLICY_OBS_DIM)
        if full1 is not None:
            return _fit_dim(full1, POLICY_OBS_DIM)

        base0 = _fit_dim(base0, BASE_PLAYER_OBS_DIM)
        base1 = _fit_dim(base1, BASE_PLAYER_OBS_DIM)
        extra = extra0 if extra0 is not None else extra1
        if extra is None:
            extra = _lookup_cached_extra(player0_obs, player1_obs)
        if extra is None:
            raise ValueError(
                "MOCH_AGENT requires the 24 expanded team/ball observation "
                "features, but none were provided or cached."
            )
        extra = _fit_dim(extra, EXTRA_TEAM_FEATURE_DIM)

        return np.concatenate((base0, base1, extra), axis=0)

    def _decode_discrete_team_action(self, flat_action: int) -> np.ndarray:
        action = np.zeros(TEAM_ACTION_DIM, dtype=np.int64)
        remainder = int(flat_action)
        for idx in range(TEAM_ACTION_DIM - 1, -1, -1):
            action[idx] = remainder % 3
            remainder //= 3
        return action

    def _format_team_action(self, raw_action) -> np.ndarray:
        action = np.asarray(raw_action)
        if action.ndim == 0:
            return self._decode_discrete_team_action(int(action.item()))

        action = action.reshape(-1).astype(np.int64, copy=False)
        if action.size >= TEAM_ACTION_DIM:
            return action[:TEAM_ACTION_DIM]
        if action.size == PLAYER_ACTION_DIM:
            # Fallback for per-player policies: mirror action to teammate.
            return np.concatenate((action, action), axis=0)

        padded = np.zeros(TEAM_ACTION_DIM, dtype=np.int64)
        padded[: action.size] = action
        return padded

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        policy_obs = self._build_policy_observation(observation)
        raw_action, *_ = self.policy.compute_single_action(policy_obs)
        team_action = self._format_team_action(raw_action)

        action_0 = team_action[:PLAYER_ACTION_DIM]
        action_1 = team_action[PLAYER_ACTION_DIM:]
        return {0: action_0, 1: action_1}
