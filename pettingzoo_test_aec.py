from aec_env import AsyncMapEnv

from pprint import pprint

import ray
from ray.rllib.algorithms import ppo
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv
from ray.tune.registry import register_env

import numpy as np
from ray import tune, air

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('num_iters', type=int, help='Number of iterations to train for.')
parser.add_argument('-c', '--constant', dest='constant', action='store_true', help='If true, set pollution level and height of all edges to the same value for deterministic routing.')
parser.add_argument('-h', '--hills', dest='hills', action='store_true', help='If true, set pollution level and height to simulate a hilly, relatively low pollution region.')
parser.add_argument('-n', '--num_agents', dest='num_agents', type=int, default=10, help='Number of agents to initialise the environment with.')
parser.add_argument('-m', '--map_size', dest='map_size', type=int, default=4, help='Map size to test on (note this is the sqrt of number of nodes in the graph).')
parser.add_argument('-r', '--reinit_agents', action='store_true')
args = parser.parse_args()

env_config = {
    'num_agents': args.num_agents,
    'map_size': args.map_size,
    'num_iters': args.num_agents * np.hypot(args.map_size, args.map_size),
    'reinit_agents': args.reinit_agents,
}


if __name__ == "__main__":
    ray.init()

    if args.constant:
        env_config['const_graph'] = True
    elif args.hills:
        env_config['hill_attrs'] =  [
                        [[5,2], 4, 2],
                        [[3,6], 7, 3],
                    ]
        env_config['poll_attrs'] = [
                        [[0,2], 7, 2],
                        [[0,7], 5, 2],
                        [[7,6], 6, 2],
                    ]

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