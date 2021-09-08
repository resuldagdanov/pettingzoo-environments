"""
Cooperative Pong v3

Environment Properties:
    Actions:                Discrete
    Agents:	                2
    Parallel API:	        Yes
    Manual Control:	        Yes
    Action Shape:	        Discrete(3)
    Action Values:	        [0, 1]
    Observation Shape:	    (280, 480, 3)
    Observation Values:	    [0, 255]
    Agents Agents:          ['paddle_0', 'paddle_1']
    State Shape:            (560, 960, 3)
    State Values:	        (0, 255)
    Average Total Reward:   -92.9
"""

from pettingzoo.butterfly import cooperative_pong_v3
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
import supersuit as ss

env = cooperative_pong_v3.env(ball_speed=9, left_paddle_speed=12,
                            right_paddle_speed=12, cake_paddle=True,
                            max_cycles=900, bounce_randomness=False)

# convert environment to only black and white color and resize to 84x84
env = ss.color_reduction_v0(env, mode="B")
env = ss.resize_v0(env, x_size=84, y_size=84)

# stack past 3 frames together to see the ball's acceleration clearly
env = ss.frame_stack_v1(env, 3)

# make policy parameter sharing for multi-agent compatibility
env = ss.pettingzoo_env_to_vec_env_v0(env)

# make environment multiple versions at the same time (8 different environments in parallel)
# env = ss.concat_vec_envs_v0(env, 8, num_cpus=4, base_class="stable_baselines3")

# define model
model = PPO(CnnPolicy, env, verbose=3, gamma=0.95, n_steps=256, ent_coef=0.0905168,
            learning_rate=0.00062211, vf_coef=0.042202, max_grad_norm=0.9, gae_lambda=0.99,
            n_epochs=5, clip_range=0.3, batch_size=256)