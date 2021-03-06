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

from ray.rllib.agents import ppo
from ray.rllib.agents.registry import get_trainer_class
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
from pettingzoo.butterfly import pistonball_v4
from custom_model import MLPModelV2
from copy import deepcopy
import supersuit
import ray
import os
import PIL


def create_env(args, is_parallel):
    if is_parallel:
        env = pistonball_v4.parallel_env(n_pistons=20, local_ratio=0, time_penalty=-0.1, continuous=True,
                                        random_drop=True, random_rotate=True, ball_mass=0.75, 
                                        ball_friction=0.3, ball_elasticity=1.5, max_cycles=125)
    else:
        env = pistonball_v4.env(n_pistons=20, local_ratio=0, time_penalty=-0.1, continuous=True,
                                random_drop=True, random_rotate=True, ball_mass=0.75, 
                                ball_friction=0.3, ball_elasticity=1.5, max_cycles=125)

    # convert environment to only black and white color and resize to 84x84
    env = supersuit.color_reduction_v0(env, mode='B')
    env = supersuit.dtype_v0(env, 'float32')
    env = supersuit.resize_v0(env, x_size=84, y_size=84)

    # normalization of the observation space vector elements
    env = supersuit.normalize_obs_v0(env, env_min=0, env_max=1)

    # stack past 3 frames together to see the ball's acceleration clearly
    env = supersuit.frame_stack_v1(env, 3)
    return env


def gen_policy(env, i):
    obs_space = env.observation_space
    act_space = env.action_space 

    config = {
        "model": {
          "custom_model": "MLPModelV2",
        },
        "gamma": 0.99,
    }
    return (None, obs_space, act_space, config)


if __name__ == "__main__":
    is_evaluation = True

    algo_name = "PPO"
    env_name = "pistonball_v4"
    checkpoint_path = "/home/resul/Research/PettingZoo/pettingzoo-environments/Butterfly_Envs/pistonball/parameters/checkpoint_000001/checkpoint-1"

    # register custom CNN model into the catalog
    ModelCatalog.register_custom_model("MLPModelV2", MLPModelV2)

    # default configurations for PPO
    # configs = deepcopy(get_trainer_class(algo_name)._default_config)

    # manual change of default configurations
    # configs["env_config"] = {"local_ratio": 0.5}
    
    configs={
        "env": env_name,
        "num_workers": 2,
        "num_envs_per_worker": 1,
        "framework": "torch",
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        'use_critic': True,
        'use_gae': True,
        "lambda": 0.95,
        "gamma": 0.99,
        "kl_coeff": 0.5,
        "clip_rewards": True,  
        "clip_param": 0.1,
        'grad_clip': None,
        "entropy_coeff": 0.001,
        'lr': 5e-05,
        "train_batch_size": 200,
        "sgd_minibatch_size": 32,
        'vf_loss_coeff': 1.0,
        'clip_actions': True,
        'no_done_at_end': False,
        'rollout_fragment_length': 100,
        #'model': {
        #  'custom_model': 'MLPModelV2',
        #},
        #'multiagent': { # for multiagent setup with policy sharing
        #    'policies': policies,
        #    'policy_mapping_fn': lambda agent_id, episode, **kwargs: "policy_0", # mapping all agents to the same policy
        #}
    }

    # create a parallel environment creator
    register_env(env_name, lambda config: ParallelPettingZooEnv(create_env(args=config, is_parallel=True)))

    # reset and shutdown current ray process
    ray.shutdown()
    ray.init()

    # train or evaluate
    if is_evaluation:

        # define trainer algorithm
        PPOagent = ppo.PPOTrainer(env=env_name, config=configs)
        
        # restore saved checkpoint
        PPOagent.restore(checkpoint_path)

        # do not run environments in parallel for evaluations
        env = create_env(args=configs, is_parallel=False)

        reward_sum = 0
        frame_list = []
        i = 0
        env.reset()

        # loop through each agent to get MDP transitions separately
        for agent in env.agent_iter():

            observation, reward, done, info = env.last()

            if done:
                action = None
            else:
                # define PPO policy
                policy = PPOagent.get_policy()
                action, _, _ = policy.compute_single_action(observation)

            print("agent #" + str(agent), "action :", action, "reward :", reward, "done :", done, "info :", info)

            # step inside the environment
            env.step(action)
            env.render()

            i += 1
            if i % (len(env.possible_agents)+1) == 0:
                frame_list.append(PIL.Image.fromarray(env.render(mode='rgb_array')))
                
        env.close()

        print("Total Obtained Reward : ", reward_sum)

        # save the output as gif file with 3 frame duration
        frame_list[0].save("evaluation_output.gif", save_all=True, append_images=frame_list[1:], duration=3, loop=0)

    # training
    else:

        # make parallel environments during training
        env = ParallelPettingZooEnv(create_env(args={}, is_parallel=True))

        # define policies with index (id)
        policies = {
            "policy_0": gen_policy(env, 0)
            }
        policy_ids = list(policies.keys())

        # initialize trainer object
        trainer = get_trainer_class(algo_name)(env=env_name, config=configs)

        for _ in range(100):
            # train once
            trainer.train()
            trainer.save()
