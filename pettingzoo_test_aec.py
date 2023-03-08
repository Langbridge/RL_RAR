from pettingzoo.test import parallel_api_test  # noqa: E402
from aec_env import AsyncMapEnv

import random
from pprint import pprint

import gym, ray
from ray.rllib.algorithms import ppo
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv
from ray.tune.registry import register_env

from pettingzoo.classic import rps_v2

from ray import tune, air

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('num_iters', type=int, help='Number of iterations to train for.')
parser.add_argument('-c', '--constant', dest='constant', action='store_true', help='If true, set pollution level and height of all edges to the same value for deterministic routing.')
parser.add_argument('-n', '--num_agents', dest='num_agents', type=int, default=10, help='Number of agents to initialise the environment with.')
parser.add_argument('-r', '--reinit_agents', action='store_true')
args = parser.parse_args()

env_config = {
    'num_agents': args.num_agents,
    'map_size': 4,
    'num_iters': 500,
    'reinit_agents': args.reinit_agents,
    # 'render_mode': 'human'
}

if __name__ == "__main__":
    ray.init()

    if args.constant:
        env_config['const_graph'] = True

    env_creator = lambda config: AsyncMapEnv(**config)
    register_env('simple', lambda config: PettingZooEnv(env_creator(config)))

    stop = {'training_iteration': args.num_iters}
    results = tune.Tuner(
            "PPO",
            run_config=air.RunConfig(
                stop=stop,
                checkpoint_config=air.CheckpointConfig(checkpoint_frequency=50, checkpoint_at_end=True)
                ),
            param_space=(
                ppo.PPOConfig()
                .environment(env='simple', env_config=env_config)
                .rollouts()
                .framework('torch')
            ).to_dict(),
        ).fit()
    
    # Get the best result based on a particular metric.
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    best_checkpoint = best_result.checkpoint
    # print(f"Best avg. reward {best_result['episode_reward_mean']}; avg. len {best_result['episode_len_mean']}")
    print(best_checkpoint)
    print(best_result)
    # /Users/abilangbridge/ray_results/PPO/PPO_simple_0673d_00000_0_2023-03-02_15-58-08/checkpoint_000005

    # manual, untuned training for 500 iters
    # algo = ppo.PPOConfig().environment(env='simple', env_config=env_config).framework(framework='torch').build()
    # for i in range(500):
    #     results = algo.train()
    #     print(f"Iter: {i}; avg. reward {results['episode_reward_mean']}; avg. len {results['episode_len_mean']}")

    #     if (i+1) % 10 == 0:
    #         print("saving..")
    #         print(algo.save("/tmp/rllib_checkpoint"))