import numpy as np
from pettingzoo_env import CustomEnvironment
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv

from ray.rllib.policy.policy import Policy
import ray.rllib.algorithms.ppo
import imageio

from pettingzoo_test import env_config
env_config['render_mode'] = 'human'
env_config['figpath'] = 'figures/img'

chkpt = '/tmp/rllib_checkpoint/checkpoint_000010'
restored_policy = Policy.from_checkpoint(chkpt)

env = ParallelPettingZooEnv(CustomEnvironment(**env_config))
obs, infos = env.reset()

batch_obs = {
    agent: obs[agent]['pollution']
    for agent in env.par_env.agents
    }

fname = f'PPO_{env_config["num_agents"]}agent_{env_config["map_size"]}x{env_config["map_size"]}_{chkpt[-2:]}'
with imageio.get_writer(f'figures/{fname}.gif', mode='I', duration=0.3) as writer:
    for x in range(25):
        action = {
            agent: restored_policy['default_policy'].compute_single_action(batch_obs[agent])[0]
            for agent in env.par_env.agents
            }
        obs, rewards, terminations, truncations, infos = env.step(action)
        print(rewards)
        try:
            writer.append_data(imageio.imread(f'{env_config["figpath"]}/img_{x}.png'))
        finally:
            continue
