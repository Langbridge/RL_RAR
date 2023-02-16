from ray.tune.registry import register_env
from ray.rllib.env import ParallelPettingZooEnv
from pettingzoo_env import CustomEnvironment

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

# define how to make the environment. This way takes an optional environment config, num_floors
env_creator = lambda config: CustomEnvironment(map_size=config.get("map_size", 5), num_agents=config.get("num_agents", 2))
register_env('custom', lambda config: ParallelPettingZooEnv(env_creator(config)))
# you can pass arguments to the environment creator with the env_config option in the config
# config['env_config'] = {"num_floors": 5}

config = (
    PPOConfig()
        .environment(env='custom')
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=0)
        # .env_config(num_agents=2)
)

tune.run(
    "PPO",
    name="PPO",
    stop={"timesteps_total": 5000000},
    checkpoint_freq=10,
    local_dir="~/ray_results/custom",
    config=config.to_dict(),
)