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
parser.add_argument('-c', '--constant', dest='constant', action='store_true', help='If true, set pollution level and height of all edges to the same value for deterministic routing.')
args = parser.parse_args()

env_config = {
    'num_agents': 10,
    'map_size': 4,
    'num_iters': 500,
    # 'render_mode': 'human'
}

if __name__ == "__main__":
    ray.init()

    if args.constant:
        env_config['const_graph'] = True

    env_creator = lambda config: AsyncMapEnv(**config)
    register_env('simple', lambda config: PettingZooEnv(env_creator(config)))

    algo = ppo.PPOConfig().environment(env='simple', env_config=env_config).framework(framework='torch').build()
    for i in range(500):
        results = algo.train()
        print(f"Iter: {i}; avg. reward {results['episode_reward_mean']}; avg. len {results['episode_len_mean']}")

        if (i+1) % 10 == 0:
            print("saving..")
            print(algo.save("/tmp/rllib_checkpoint"))