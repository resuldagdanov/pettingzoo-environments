"""
Prison v3

Environment Properties:
    Actions:                Either
    Agents:	                8
    Parallel API:	        Yes
    Manual Control:	        Yes
    Action Shape:	        (1,)
    Action Values:	        [0, 2]
    Observation Shape:	    (100, 300, 3) or (1,)
    Observation Values:	    (0, 255) or (-300, 300)
    Agents Agents:          ['prisoner_0', 'prisoner_1', ..., 'prisoner_7']
    State Shape:            (650, 750, 3)
    State Values:	        (0, 255)
    Average Total Reward:   2.77
"""

from pettingzoo.butterfly import prison_v3
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
import supersuit as ss


env = prison_v3.env(vector_observation=False, continuous=False, synchronized_start=False,
                    identical_aliens=False, max_cycles=150, num_floors=4, random_aliens=False)

env = ss.color_reduction_v0(env, mode="B")
env = ss.resize_v0(env, x_size=84, y_size=84)
