from pettingzoo.test import parallel_api_test  # noqa: E402
from pettingzoo_env import CustomEnvironment

import random
from pprint import pprint

import gym, ray
from ray.rllib.algorithms import ppo
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv
from ray.tune.registry import register_env

from pettingzoo.classic import rps_v2

config = {
    'num_agents': 2,
    'map_size': 3,
    'num_iters': 1_000_000,
    # 'render_mode': 'human'
}

if __name__ == "__main__":

    # # cyclist creation test
    # env = CustomEnvironment(**config)
    # env.reset()
    # pprint(env.agent_name_mapping)

    # # parallel API test script
    # parallel_api_test(CustomEnvironment(**config), num_cycles=1_000_000)

    # # test valid movesets [only works with single agent]
    # for i in range(5):
    #     print(f"Valid moves: {[y for (x, y) in env.G.edges(env.positions[env.agents[0]])]}")
    #     action = random.choice([y for (x, y) in env.G.edges(env.positions[env.agents[0]])])

    #     res = env.step({env.agents[0]: action})
    #     print(f"Moving to {action}, cumulative reward: {res[1]}")

    # test PPO training
    ray.init()

    env_creator = lambda config: CustomEnvironment(**config)
    register_env('simple', lambda config: ParallelPettingZooEnv(env_creator(config)))

    # env_creator = lambda config: rps_v2.parallel_env()
    # register_env('simple', lambda config: ParallelPettingZooEnv(env_creator(config)))

    algo = ppo.PPOConfig().environment(env='simple', env_config=config).framework(framework='torch').build()
    for i in range(5):
        results = algo.train()
        print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")