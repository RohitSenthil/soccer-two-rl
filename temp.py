import numpy as np
import soccer_twos
from soccer_twos import EnvType
from soccer_twos.side_channels import EnvConfigurationChannel
from utils import RLLibWrapper

env_channel = EnvConfigurationChannel()

env = soccer_twos.make(
    watch=True,
    # base_port=args.base_port,
    env_channel=env_channel,
    variation= EnvType.multiagent_team,
)
env.reset()
# env_channel.set_parameters(
#     ball_state={
#         "position": [0, 0],
#         "velocity": [0,0],
#     },
#     players_states={
#         # 2: {
#         #     # "position": [-5,-5],
#         #     # "rotation_y": 45,
#         #     # "velocity": [5, 5],
#         # },
#     }
# )
env = RLLibWrapper(env)
# action = np.array([0, 0, 1, 0, 0, 0], dtype=np.int64)
action = np.array([0, 0, 0, 0, 0, 0], dtype=np.int64)
# action = env.action_space.sample()
# print(env.observation_space.shape)
# print(env.observation_space.sample())
# print(env.action_space.shape)
# print(env.action_space.sample())
# print(type(env.action_space.sample()))
for _ in range(12):
    # env.step({0:action,1:action})
    env.step({0:action,1:action})
# input()