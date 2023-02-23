import numpy as np
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv
from ray.rllib.utils.spaces import space_utils
from pettingzoo_env import CustomEnvironment
from map_env import CustomEnvironment

from ray.rllib.policy.policy import Policy
import imageio
import os
import glob

from collections import defaultdict

env_config = {
    'num_agents': 2,
    'map_size': 4,
    'num_iters': 100,
    # 'render_mode': 'human'
}

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=int)
parser.add_argument('-p', '--prefix', dest='prefix', default='PPO')
parser.add_argument('-g', '--gif', dest='gif', action='store_true')
parser.add_argument('-e', '--eval', dest='eval', action='store_true')
args = parser.parse_args()

chkpt = f'/tmp/rllib_checkpoint/checkpoint_{str(args.checkpoint).zfill(6)}'
restored_policy = Policy.from_checkpoint(chkpt)

if args.gif:
    print("creating GIF")
    env_config['render_mode'] = 'human'
    env_config['figpath'] = 'figures/img'
    env = ParallelPettingZooEnv(CustomEnvironment(**env_config))
    obs, infos = env.reset()

    fname = f'{args.prefix}_{env_config["num_agents"]}agent_{env_config["map_size"]}x{env_config["map_size"]}_{args.checkpoint}'
    with imageio.get_writer(f'figures/{fname}.gif', mode='I', duration=0.3) as writer:
        for x in range(100):
            batch_obs = {
                agent: space_utils.flatten_to_single_ndarray(obs[agent])
                for agent in env.par_env.agents
                }
            action = {
                agent: restored_policy['default_policy'].compute_single_action(batch_obs[agent])[0]
                for agent in env.par_env.agents
                }
            obs, rewards, terminations, truncations, infos = env.step(action)
            # print(reward)

            try:
                writer.append_data(imageio.imread(f'{env_config["figpath"]}/img_{x}.png'))

            finally:
                # check if all agents have finished
                if any([truncations[a] for a in truncations]):
                    print("Truncating...")
                    break
                continue
    print(f"{env.par_env.pollution}")
    print(f"{terminations}")
    # tidy up img dir
    files = glob.glob(f'{env_config["figpath"]}/*')
    for f in files:
        os.remove(f)

if args.eval:
    print("Evaluating performance")
    env_config['render_mode'] = None # remove rendering to accelerate
    env = ParallelPettingZooEnv(CustomEnvironment(**env_config))

    length = defaultdict(list)
    tot_reward = defaultdict(list)
    
    for i in range(50): # 20 runs
        obs, infos = env.reset()
        truncations = {
            agent: False for agent in env.par_env.possible_agents
        }

        reward_store = defaultdict(int)
        episode_lengths = defaultdict(int)
        while not any(truncations.values()): # until truncation
            batch_obs = {
                agent: space_utils.flatten_to_single_ndarray(obs[agent])
                for agent in env.par_env.agents
                }
            action = {
                agent: restored_policy['default_policy'].compute_single_action(batch_obs[agent])[0]
                for agent in env.par_env.agents
                }
            obs, rewards, terminations, truncations, infos = env.step(action)

            for agent, reward in rewards.items():
                reward_store[agent] += reward
                episode_lengths[agent] += 1
        for agent in reward_store:
            length[agent].append(episode_lengths[agent])
            tot_reward[agent].append(reward_store[agent])

    # print(length)
    # print(tot_reward)
    print(f"Checkpoint {args.checkpoint}: \t{np.mean(list(length.values()))}\t{np.mean(list(tot_reward.values()))}")

    pass