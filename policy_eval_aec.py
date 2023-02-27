import numpy as np
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv
from ray.rllib.utils.spaces import space_utils
from pettingzoo_env import CustomEnvironment
from aec_env import AsyncMapEnv

from ray.rllib.policy.policy import Policy
import imageio.v2 as imageio
import os
import glob

from collections import defaultdict

env_config = {
    'num_agents': 10,
    'map_size': 4,
    'num_iters': 500,
    # 'render_mode': 'human'
}

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=int)
parser.add_argument('-p', '--prefix', dest='prefix', default='PPO')
parser.add_argument('-g', '--gif', dest='gif', action='store_true')
parser.add_argument('-e', '--eval', dest='eval', action='store_true')
args = parser.parse_args()

if args.gif:
    print("creating GIF")
    env_config['render_mode'] = 'human'
    env_config['figpath'] = 'figures/img'

    chkpt = f'/tmp/rllib_checkpoint/checkpoint_{str(args.checkpoint).zfill(6)}'
    restored_policy = Policy.from_checkpoint(chkpt)

    env = PettingZooEnv(AsyncMapEnv(**env_config))
    obs, infos = env.reset()

    fname = f'{args.prefix}_{env_config["num_agents"]}agent_{env_config["map_size"]}x{env_config["map_size"]}_{args.checkpoint}'
    with imageio.get_writer(f'figures/{fname}.gif', mode='I', duration=0.3) as writer:
        x = 0
        while env.env.agents:
            if x > 100:
                break

            curr_agent = env.env.agent_selection
            batch_obs = space_utils.flatten_to_single_ndarray(obs[curr_agent])
            action = [restored_policy['default_policy'].compute_single_action(batch_obs)][0][0]
            obs, rewards, terminations, truncations, infos = env.env.step(action)
            # print({curr_agent: action})
            # print(rewards['curr_agent'])

            try:
                writer.append_data(imageio.imread(f'{env_config["figpath"]}/img_{x}.png'))
            except FileNotFoundError:
                print(f"File {x} not found...")
                break
            finally:
                # check if all agents have finished
                if any([truncations[a] for a in truncations]):
                    print("Truncating...")
                    break

            x += 1
        # repeat final frame 5 times
        for i in range(4):
            writer.append_data(imageio.imread(f'{env_config["figpath"]}/img_{x-1}.png'))

    print(f"{env.env.pollution}")
    # tidy up img dir
    files = glob.glob(f'{env_config["figpath"]}/*')
    for f in files:
        os.remove(f)

if args.eval:
    print("Evaluating performance")
    env_config['render_mode'] = None # remove rendering to accelerate
    env = PettingZooEnv(AsyncMapEnv(**env_config))

    for c in range(10, args.checkpoint+1, 10):
        chkpt = f'/tmp/rllib_checkpoint/checkpoint_{str(c).zfill(6)}'
        restored_policy = Policy.from_checkpoint(chkpt)

        length = defaultdict(list)
        tot_reward = defaultdict(list)
        for i in range(500): # 500 runs
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

        mean_len = np.mean([np.mean(length[agent]) for agent in length.keys()])
        mean_tot_reward = np.mean([np.mean(tot_reward[agent]) for agent in tot_reward.keys()])
        print(f"Checkpoint {c}: \t{mean_len}\t{mean_tot_reward}")