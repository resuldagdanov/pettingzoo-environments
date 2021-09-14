"""
Pistonball v4

Environment Properties:
    Actions:                Either
    Agents:	                20
    Parallel API:	        Yes
    Manual Control:	        Yes
    Action Shape:	        (1,)
    Action Values:	        [-1, 1]
    Observation Shape:	    (457, 120, 3)
    Observation Values:	    [0, 255]
    State Shape:            (560, 880, 3)
    State Values:	        (0, 255)
    Average Total Reward:   -91.2
"""

from pettingzoo.butterfly import pistonball_v4
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
import supersuit as ss


# False: training ; True: evaluation
is_evaluation = True


def train():
    env = pistonball_v4.parallel_env(n_pistons=20, local_ratio=0, time_penalty=-0.1, continuous=True,
                                    random_drop=True, random_rotate=True, ball_mass=0.75, ball_friction=0.3,
                                    ball_elasticity=1.5, max_cycles=125)

    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v0(env, x_size=84, y_size=84)

    # stack past 3 frames together to see the ball's acceleration clearly
    env = ss.frame_stack_v1(env, 3)

    # make policy parameter sharing for multi-agent compatibility
    env = ss.pettingzoo_env_to_vec_env_v0(env)

    # make environment multiple versions at the same time (8 different environments in parallel)
    env = ss.concat_vec_envs_v0(env, 8, num_cpus=4, base_class="stable_baselines3")

    # define model
    model = PPO(CnnPolicy, env, verbose=3, gamma=0.95, n_steps=256, ent_coef=0.0905168,
                learning_rate=0.00062211, vf_coef=0.042202, max_grad_norm=0.9, gae_lambda=0.99,
                n_epochs=5, clip_range=0.3, batch_size=256)

    # TRAINING and SAVING
    print("\nTraining is Starting ...\n")

    model.learn(total_timesteps=200000)
    model.save("policy")


def evaluate():
    print("\nEvaluation is Starting ...\n")

    env = pistonball_v4.env()
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v0(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)

    # load save policy model
    model = PPO.load("policy")

    # run the model according to the policy with render open mode
    env.reset()
    for agent in env.agent_iter():
        obs, reward, done, info = env.last()
        act = model.predict(obs, deterministic=True)[0] if not done else None
        env.step(act)
        env.render()


if __name__ == "__main__":

    if is_evaluation:
        evaluate()
    else:
        train()
