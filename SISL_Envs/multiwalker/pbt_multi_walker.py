"""
Multiwalker v7

Environment Properties:
    Actions:                Continuous
    Agents:	                3
    Parallel API:	        Yes
    Manual Control:	        No
    Action Shape:	        (4,)
    Action Values:	        (-1, 1)
    Observation Shape:	    (31,)
    Observation Values:	    [-inf, inf]
    Agents Agents:          ['walker_0', 'walker_1', 'walker_2']
    Average Total Reward:   -300.86
"""

from pettingzoo.sisl import multiwalker_v7
from ray import tune
from ray.tune.registry import register_env
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.models import ModelCatalog
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.examples.models.centralized_critic_models import TorchCentralizedCriticModel
import numpy as np
import ray
import random
import os

# number of agents (default is 3)
N_AGENTS = 3
ALGORITHM = "PPO"


def env_creator(args):
    # parallel_env will execute synchronized actions
    env = multiwalker_v7.parallel_env(n_walkers=N_AGENTS, position_noise=1e-3, angle_noise=1e-3,
                                    local_ratio=1.0, forward_reward=1.0, terminate_reward=-100.0, fall_reward=-10.0,
                                    terminate_on_fall=True, remove_on_fall=True, max_cycles=500)
    return env


def gen_policy(i):
    obs_space = env.observation_space
    act_space = env.action_space 

    # default centralized critic model is used
    config = {
        "model": {
          "custom_model": "TorchCentralizedCriticModel",
        },
        "gamma": 0.99,
    }
    return (None, obs_space, act_space, config)


def policy_mapping_fn(agent_id, episode, **kwargs):
    pol_id = random.choice(policy_ids)
    return pol_id


if __name__ == "__main__":
    # use 1 GB of memory
    ray.init(_memory=10**9)

    # run environments in parallel
    env = ParallelPettingZooEnv(env_creator(None))

    ModelCatalog.register_custom_model("TorchCentralizedCriticModel", TorchCentralizedCriticModel)

    register_env("multi_agent_multi_walker", lambda _: env)

    # define PBT scheduler configs
    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=5,
        hyperparam_mutations={
            # distribution for resampling
            "lr": lambda: np.random.uniform(0.0001, 1)
        })

    # different agents will have different policies
    policies = {
        "policy_{}".format(i): gen_policy(i) for i in range(N_AGENTS)
    }
    policy_ids = list(policies.keys())

    # general configuration parameters
    config = {
        "env": "multi_agent_multi_walker",
        "env_config": {
            "num_agents": N_AGENTS,
        },
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "1")),
        "num_sgd_iter": 10,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "framework": "torch",
        "lr": 0.0001
    }

    # stop when 10M frames (steps) is reached
    stop = {
        "timesteps_total": 10000000,
    }

    # train the model
    analysis = tune.run(ALGORITHM, stop=stop, config=config, scheduler=scheduler, metric="episode_reward_mean", mode="max", checkpoint_freq=20, verbose=3, num_samples=4)

    print("Best hyperparameters found were: ", analysis.best_config)
