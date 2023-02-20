import numpy as np
from pettingzoo_env import CustomEnvironment
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv

from ray.rllib.policy.policy import Policy
import ray.rllib.algorithms.ppo


env_config = {
    'num_agents': 10,
    'map_size': 10,
    'num_iters': 1_000_000,
    'render_mode': 'human'
}

restored_policy = Policy.from_checkpoint("/tmp/rllib_checkpoint/checkpoint_000003")

env = ParallelPettingZooEnv(CustomEnvironment(**env_config))
obs, infos = env.reset()

batch_obs = {
    agent: np.concatenate([obs[agent]['pollution'], obs[agent]['action_mask']])
    for agent in env.par_env.agents
    }

for x in range(10):
    action = {
        agent: restored_policy['default_policy'].compute_single_action(batch_obs[agent])[0]
        for agent in env.par_env.agents
        }
    obs, rewards, terminations, truncations, infos = env.step(action)

