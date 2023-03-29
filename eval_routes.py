import numpy as np
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv
from ray.rllib.utils.spaces import space_utils
from aec_env import AsyncMapEnv, AsyncMapEnv_NoVel

from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.algorithm import Algorithm
import imageio.v2 as imageio
import networkx as nx

from collections import defaultdict
from copy import deepcopy

from brute_search import SearchTree
from pprint import pprint

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=int)
parser.add_argument('--path', dest='path', type=str, default='/tmp/rllib_checkpoint/')
parser.add_argument('-n', '--num_agents', dest='num_agents', type=int, default=1)
parser.add_argument('-m', '--map_size', dest='map_size', type=int, default=4)
parser.add_argument('-v', '--const_vel', dest='const_vel', action='store_true')
parser.add_argument('-r', '--reinit_agents', action='store_true')
args = parser.parse_args()

env_config = {
    'num_agents': args.num_agents,
    'map_size': args.map_size,
    'num_iters': args.num_agents * args.map_size * args.map_size,
    'reinit_agents': args.reinit_agents,
    'congestion': True,
    'corners': True,
    'fit_split': 2,
    'hill_attrs': [
                    # [[5,2], 4, 2],
                    # [[3,6], 7, 3],
                    [[3, 3], 50, 2],
                  ],
    'poll_attrs': [
                [[0,2], 7, 2],
                [[0,7], 5, 2],
                [[7,6], 6, 2],
              ],
    # 'render_mode': 'human',
    'figpath': 'figures/img',
}
if args.const_vel:
    raw_env = AsyncMapEnv_NoVel(**env_config)
else:
    raw_env = AsyncMapEnv(**env_config)
env = PettingZooEnv(raw_env)

if args.checkpoint:
    chkpt = f'{args.path}checkpoint_{str(args.checkpoint).zfill(6)}'
    restored_policy = Policy.from_checkpoint(chkpt)

    poll_optimality = defaultdict(list)
    obs, infos = env.reset()
    truncations = {
        agent: False for agent in env.env.possible_agents
    }
    env.env.positions = {
            agent: 0 for agent in env.env.agents
        }
    env.env.goals = {
            agent: env.env.num_nodes-1 for agent in env.env.agents
        }
    print(env.env.positions, env.env.goals)

    # # brute force optimal (non-congested) route
    # search = SearchTree(env.env)
    # search.build_tree()
    # optimal_polls = search.pollutions
    # optimal_routes = search.routes

    policy_paths = defaultdict(list)

    while not any(truncations.values()): # until truncation
        curr_agent = env.env.agent_selection
        batch_obs = space_utils.flatten_to_single_ndarray(obs[curr_agent])
        action = [restored_policy['default_policy'].compute_single_action(batch_obs)][0][0]
        policy_paths[curr_agent].append(tuple(action.values()))

        obs, rewards, terminations, truncations, infos = env.step({curr_agent: action})

    # for agent in policy_paths:
    #     poll_optimality[agent].append(search.pollutions[agent] / env.env.pollution[agent])

    # pprint(optimal_routes)
    print(policy_paths)
    print(env.env.pollution)
    # pprint(poll_optimality)
    # print(np.mean([poll_optimality[i] for i in poll_optimality.keys()]))

else:
    raw_env.positions = {
            agent: 0 for agent in raw_env.agents
        }
    raw_env.goals = {
            agent: raw_env.num_nodes-1 for agent in raw_env.agents
        }
    print(raw_env.positions, raw_env.goals)
    pprint(raw_env.agent_name_mapping)
    search = SearchTree(raw_env)

    # calculate shortest path
    short_path = nx.shortest_path(raw_env.G, source=0, target=raw_env.num_nodes-1, weight='l')
    print(short_path)
    ptrs = {
            agent: 1 for agent in raw_env.agents
        }
    while len([i for i in ptrs.values() if i < len(short_path)]) > 0:
        curr_agent = env.env.agent_selection
        env.step({curr_agent: {'destination': short_path[ptrs[curr_agent]]}})
        print({curr_agent: {'destination': short_path[ptrs[curr_agent]]}})
        ptrs[curr_agent] += 1
    print(env.env.pollution)

    # brute force optimal (non-congested) route
    if args.const_vel:
        search.build_tree(verbose=1, velocities=25)
    else:
        search.build_tree(verbose=1)
    optimal_polls = search.pollutions
    optimal_routes = search.routes

    print(optimal_routes)
    print(optimal_polls)