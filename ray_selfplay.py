# Takes about 15 iterations for a win rate of 0.27
import numpy as np
import ray
from random_policy import RandomPolicy
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune import CLIReporter
from soccer_twos import EnvType
from utils import create_rllib_env

NUM_ENVS_PER_WORKER = 3


# def policy_mapping_fn(agent_id, episode, worker, **kwargs):
#     return "main" if episode.episode_id % 2 == agent_id else "random"
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if episode.episode_id % 2 == agent_id:
        return "main"
    policies = [p for p in worker.policy_map.keys() if p.startswith("main_v")]
    return "random" if not policies else np.random.choice(policies)


class SelfPlayCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        # 0=RandomPolicy, 1=1st main policy snapshot,
        # 2=2nd main policy snapshot, etc..
        self.current_opponent = 0

        self.win_rate_threshold = 0.9

    def on_episode_end(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        # Determine the matchup results at the end of the episode securely
        agent_ids = list(episode.agent_rewards.keys())
        # print("Episode End:" + str(agent_ids))
        if len(agent_ids) == 2:
            p1_pol = agent_ids[0][1]
            p2_pol = agent_ids[1][1]
            r1 = episode.agent_rewards[agent_ids[0]]
            r2 = episode.agent_rewards[agent_ids[1]]
            if p2_pol == "main":
                p1_pol, p2_pol = p2_pol, p1_pol
                r1, r2 = r2, r1
            res = 1.0 if r1 > r2 else (0.5 if r1 == r2 else 0.0)

            key = "main_wins"
            if key not in episode.hist_data:
                episode.hist_data[key] = []
            episode.hist_data[key].append(res)

    def on_train_result(self, *, trainer, result, **kwargs):
        # Get the win rate for the train batch.
        # Note that normally, one should set up a proper evaluation config,
        # such that evaluation always happens on the already updated policy,
        # instead of on the already used train_batch.
        # main_rew = result["hist_stats"].pop("policy_main_reward")
        # win_rate = sum(1 for rew in main_rew if rew>0) / len(main_rew)
        main_records = result["hist_stats"]["main_wins"]
        print(result["hist_stats"])
        win_rate = sum(main_records) / len(main_records)
        result["win_rate"] = win_rate

        print(f"Iter={trainer.iteration} win-rate={win_rate} -> ", end="")
        # If win rate is good -> Snapshot current policy and play against
        # it next, keeping the snapshot fixed and only improving the "main"
        # policy.
        if win_rate > self.win_rate_threshold:
            policies = trainer.workers.local_worker().policy_dict.keys()
            self.current_opponent = len(policies) - 2
            self.current_opponent += 1
            new_pol_id = f"main_v{self.current_opponent}"
            print(f"adding new opponent to the mix ({new_pol_id}).")

            new_policy = trainer.add_policy(
                policy_id=new_pol_id,
                policy_cls=type(trainer.get_policy("main")),
            )

            main_state = trainer.get_policy("main").get_weights()
            new_policy.set_weights(main_state)
            trainer.workers.sync_weights()
        else:
            print("not good enough; will keep learning ...")

        result["league_size"] = self.current_opponent + 2


if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env(
        {"variation": EnvType.multiagent_team, "expanded_obs": True}
    )
    # temp_env = create_rllib_env({"variation": EnvType.multiagent_team})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name="PPO_selfplay_rec_expandedobs_fixed2_2",
        config={
            # system settings
            "num_gpus": 0,
            "num_workers": 12,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": SelfPlayCallback,
            # RL setup
            "multiagent": {
                "policies": {
                    "main": (None, obs_space, act_space, {}),
                    "random": (RandomPolicy, obs_space, act_space, {}),
                },
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["main"],
            },
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.multiagent_team,
                "base_port": 8000,
                "expanded_obs": True,
            },
            "model": {
                "vf_share_layers": False,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "rollout_fragment_length": 1000,
            "batch_mode": "complete_episodes",
            "train_batch_size": 36000,
            "sgd_minibatch_size": 4000,
        },
        stop={
            "timesteps_total": 20000000,
            "time_total_s": 64200,
        },
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        progress_reporter=CLIReporter(
            metric_columns={
                "training_iteration": "iter",
                "time_total_s": "time_total_s",
                "timesteps_total": "ts",
                "episodes_this_iter": "train_episodes",
                "policy_reward_mean/main": "reward",
                "win_rate": "win_rate",
                "league_size": "league_size",
            },
            sort_by_metric=True,
        ),
        restore="./ray_results/PPO_selfplay_rec_expandedobs_fixed2/PPO_Soccer_f3f48_00000_0_2026-04-14_20-27-58/checkpoint_000515/checkpoint-515",
    )
    # Gets best trial based on max accuracy across all training iterations.
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    # Gets best checkpoint for trial based on accuracy.
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
