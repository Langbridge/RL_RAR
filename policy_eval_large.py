import numpy as np
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv
from ray.rllib.utils.spaces import space_utils
from aec_env import AsyncMapEnv, AsyncMapEnv_NoVel

from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.algorithm import Algorithm
import imageio.v2 as imageio
import os
import glob

from collections import defaultdict
from pprint import pprint
import csv

from brute_search import SearchTree

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=int)
parser.add_argument('-s', '--start', dest='start', type=int, default=10)
parser.add_argument('--step', dest='step', type=int, default=10)
parser.add_argument('--path', dest='path', type=str, default='/tmp/rllib_checkpoint/')
parser.add_argument('--hills', dest='hills', action='store_true', help='If true, set pollution level and height to simulate a hilly, relatively low pollution region.')
parser.add_argument('-n', '--num_agents', dest='num_agents', type=int, default=1)
parser.add_argument('-m', '--map_size', dest='map_size', type=int, default=4)
parser.add_argument('-r', '--reinit_agents', action='store_true')
parser.add_argument('-v', '--velocity', action='store_true')
args = parser.parse_args()

env_config = {
    'num_agents': args.num_agents,
    'map_size': args.map_size,
    'num_iters': args.num_agents * args.map_size * args.map_size,
    'reinit_agents': args.reinit_agents,
    'fit_split': 2,
    'corners': True,
    # 'render_mode': 'human'
}

if args.hills:
    env_config['hill_attrs'] =  [
                    # [[5,2], 4, 2],
                    # [[3,6], 7, 3],
                    [[3, 3], 50, 2],
                ]
    env_config['poll_attrs'] = [
                    [[0,2], 7, 2],
                    [[0,7], 5, 2],
                    [[7,6], 6, 2],
                ]
if args.velocity:
    raw_env = AsyncMapEnv(**env_config)
else:
    raw_env = AsyncMapEnv_NoVel(**env_config)
env = PettingZooEnv(raw_env)

for c in range(args.start, args.checkpoint+1, args.step):
    chkpt = f'{args.path}checkpoint_{str(c).zfill(6)}'
    print(chkpt)
    restored_policy = Policy.from_checkpoint(chkpt)
    env.env.global_iters = c*3

    length = defaultdict(list)
    tot_reward = defaultdict(list)
    polls = defaultdict(list)
    durs = defaultdict(list)
    for i in range(int(np.ceil(800/args.num_agents))): # 80 runs for 10 agent, 800 for 1 agent
        obs, infos = env.reset()
        truncations = {
            agent: False for agent in env.env.possible_agents
        }

        reward_store = defaultdict(int)
        episode_lengths = defaultdict(int)
        while not any(truncations.values()): # until truncation
            batch_obs = space_utils.flatten_to_single_ndarray(obs[env.env.agent_selection])

            action = [restored_policy['default_policy'].compute_single_action(batch_obs)][0][0]
            obs, rewards, terminations, truncations, infos = env.step({env.env.agent_selection: action})

            for agent, reward in rewards.items():
                reward_store[agent] += reward
                episode_lengths[agent] += 1

        for agent in reward_store:
            length[agent].append(episode_lengths[agent])
            tot_reward[agent].append(reward_store[agent])
            polls[agent].append(env.env.pollution[agent])
            durs[agent].append(env.env.duration[agent])

    mean_len = np.mean([np.mean(length[agent]) for agent in length.keys()])
    mean_tot_reward = np.mean([np.mean(tot_reward[agent]) for agent in tot_reward.keys()])
    mean_polls = [np.mean(polls[agent]) for agent in polls.keys()]
    mean_durs = [np.mean(durs[agent]) for agent in durs.keys()]

    print(f"Checkpoint {c}: \t{mean_len}\t{mean_tot_reward}\t{np.mean(mean_polls)}\t{np.mean(mean_durs)}")


    fitness_polls = defaultdict(list)
    fitness_durs = defaultdict(list)
    for agent in polls.keys():
        fitness_polls[env.env.agent_name_mapping[agent].fitness] += polls[agent]
        fitness_durs[env.env.agent_name_mapping[agent].fitness] += durs[agent]
    fitness_polls = {fit: np.mean(fitness_polls[fit]) for fit in fitness_polls.keys()}
    fitness_durs = {fit: np.mean(fitness_durs[fit]) for fit in fitness_durs.keys()}
    for x in range(3):
        try:
            fitness_polls[x]
        except KeyError:
            fitness_polls[x] = np.nan
            fitness_durs[x] = np.nan
    pprint(fitness_polls)
    pprint(fitness_durs)

    with open('pollution_eval.csv', 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([args.path, c, args.num_agents, mean_len, mean_tot_reward, np.mean(mean_polls), fitness_polls[0], fitness_polls[1], fitness_polls[2], fitness_durs[0], fitness_durs[1], fitness_durs[2]])