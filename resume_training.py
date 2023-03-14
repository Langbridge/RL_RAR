from aec_env import AsyncMapEnv
import ray
from ray.rllib.env import PettingZooEnv
from ray import tune, air
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms import ppo
from ray.tune.registry import register_env


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=int)
parser.add_argument('--path', dest='path', type=str, default='/tmp/rllib_checkpoint/')
parser.add_argument('-n', '--num_agents', dest='num_agents', type=int, default=1)
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

    env_creator = lambda config: AsyncMapEnv(**config)
    register_env('simple', lambda config: PettingZooEnv(env_creator(config)))

    algo = Algorithm.from_checkpoint(f'{args.path}checkpoint_{str(args.checkpoint).zfill(6)}')

    print("Training...")
    for x in range(5000):
        results = algo.train()
        if (x+1) % 50 == 0:
            print(f"Iter: {x+args.checkpoint}; avg. reward {results['episode_reward_mean']}; avg. len {results['episode_len_mean']}")
            print(algo.save(f'{args.path}checkpoint_{str(x).zfill(6)}'))
