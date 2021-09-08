import supersuit
from pettingzoo.atari import space_invaders_v1
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO

env = space_invaders_v1.env()

# as per openai baseline's MaxAndSKip wrapper, maxes over the last 2 frames
# to deal with frame flickering
env = supersuit.max_observation_v0(env, 2)

# repeat_action_probability is set to 0.25 to introduce non-determinism to the system
env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25)

# skip frames for faster processing and less control
# to be compatible with gym, use frame_skip(env, (2,5))
env = supersuit.frame_skip_v0(env, 4)

# downscale observation for faster processing
env = supersuit.resize_v0(env, 84, 84)

# allow agent to see everything on the screen despite Atari's flickering screen problem
env = supersuit.frame_stack_v1(env, 4)

# define model
model = PPO(CnnPolicy, env, verbose=3, gamma=0.95, n_steps=256, ent_coef=0.0905168,
            learning_rate=0.00062211, vf_coef=0.042202, max_grad_norm=0.9, gae_lambda=0.99,
            n_epochs=5, clip_range=0.3, batch_size=256)