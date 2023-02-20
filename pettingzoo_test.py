from pettingzoo.test import parallel_api_test  # noqa: E402
from pettingzoo_env import CustomEnvironment

import random
from pprint import pprint

import gym, ray
from ray.rllib.algorithms import ppo
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv
from ray.tune.registry import register_env

from pettingzoo.classic import rps_v2

from ray import tune, air

env_config = {
    'num_agents': 10,
    'map_size': 10,
    'num_iters': 1_000_000,
    # 'render_mode': 'human'
}

if __name__ == "__main__":
    ray.init()

    env_creator = lambda config: CustomEnvironment(**config)
    register_env('simple', lambda config: ParallelPettingZooEnv(env_creator(config)))

    algo = ppo.PPOConfig().environment(env='simple', env_config=env_config).framework(framework='torch').build()
    for i in range(100):
        results = algo.train()
        print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")

        if i % 2 == 0:
            print("saving..")
            print(algo.save("/tmp/rllib_checkpoint"))