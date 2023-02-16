import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Tuple, Dict
import logging
import random

import networkx as nx
import numpy as np

from ray.rllib.env import BaseEnv, MultiAgentEnv, ExternalEnv

logger = logging.getLogger(__name__)
MAX_STEPS = 100

# agent is tasked with finding the pollution-optimal path from A -> B
# observation space:    [curr_node, pollution levels (Nx1 Box)]     :: congestion (Nx1 Box)
# action space:         [move to accessible node (masked Nx1 Box)]  :: velocity Discrete(2)

class SimpleGridEnv(gym.Env):
    def __init__(self, env_config):
        super(SimpleGridEnv, self).__init__()

        # create graph
        # self.G = nx.grid_graph(dim=(env_config['size'], env_config['size']))
        self.G = nx.grid_graph(dim=(5, 5))
        self.G = nx.convert_node_labels_to_integers(self.G)
        self.G = nx.DiGraph(self.G)
        self.num_nodes = len(self.G.nodes())
        self.num_edges = len(self.G.edges())

        # create task
        self.start_pos, self.end_pos = random.sample(self.G.nodes(), 2)
        logger.info("Start pos {} end pos {}".format(self.start_pos, self.end_pos))
        print(f"Starting from {self.start_pos} and travelling to {self.end_pos}")
        self.reset()

        # set up observation and action spaces
        self.observation_space = Tuple(
            [
                Discrete(self.num_nodes),                       # current node idx
                Box(low=0, high=10, shape=(self.num_edges,)),   # pollution level at all edges
                Box(low=0, high=1, shape=(self.num_nodes,))     # action mask
            ]
        )
        self.action_space = Discrete(self.num_nodes)  # node to move to

    def reset(self, *, seed=None, options=None):
        self.pos = self.start_pos
        self.journey_len = 0
        self.pollution = 0
        # <obs>, <info: dict>
        return [self.pos, self._get_pollution(), self._get_action_mask(self.pos)], {}

    def step(self, action):
        if self._get_action_mask(self.pos)[action] == 0:
            raise ValueError(
                f"Invalid action sent to env!"
            )

        # update penalties for length and pollution
        self.journey_len += 1
        self.pollution += self._get_pollution((self.pos, action))
        logger.info("Cyclist moved from {} to {}".format(self.pos, action))
        print(f"Moving to {self.pos}")

        # update position and check for end conditions
        self.pos = action
        at_goal = self.pos == self.end_pos
        truncated = self.journey_len >= MAX_STEPS
        done = at_goal or truncated

        # <obs>, <reward: float>, <done: bool>, <info: dict>
        return (
            [self.pos, self._get_pollution(), self._get_action_mask(self.pos)],
            self._reward(at_goal),
            done,
            {},
        )
    
    def _get_action_mask(self, node):
        valid_actions = np.zeros((self.num_nodes,))
        idxs = [y for (x, y) in self.G.edges(node)]
        valid_actions[idxs] = 1
        return valid_actions
    
    def _get_pollution(self, edge=None):
        if edge:
            # return a single value for pollution corresponding to an action (ie edge traversal)
            return 0.25
        else:
            # return the full array of pollution values
            return np.array([0.25]*self.num_edges)
        
    def _reward(self, at_goal):
        return -0.1*self.journey_len + -1*self.pollution + 100*int(at_goal)
    
MAP_DATA = """
#########
#S      #
####### #
      # #
      # #
####### #
#F      #
#########"""

class WindyMazeEnv(gym.Env):

    config = {
        "disable_env_checking":True
    }

    def __init__(self, env_config):
        self.map = [m for m in MAP_DATA.split("\n") if m]
        self.x_dim = len(self.map)
        self.y_dim = len(self.map[0])
        logger.info("Loaded map {} {}".format(self.x_dim, self.y_dim))
        for x in range(self.x_dim):
            for y in range(self.y_dim):
                if self.map[x][y] == "S":
                    self.start_pos = (x, y)
                elif self.map[x][y] == "F":
                    self.end_pos = (x, y)
        logger.info("Start pos {} end pos {}".format(self.start_pos, self.end_pos))
        self.observation_space = Tuple(
            [
                Box(0, 100, shape=(2,)),  # (x, y)
                Discrete(4),  # wind direction (N, E, S, W)
            ]
        )
        self.action_space = Discrete(2)  # whether to move or not

    def reset(self, *, seed=None, options=None):
        self.wind_direction = random.choice([0, 1, 2, 3])
        self.pos = self.start_pos
        self.num_steps = 0
        return [[self.pos[0], self.pos[1]], self.wind_direction], {}

    def step(self, action):
        if action == 1:
            self.pos = self._get_new_pos(self.pos, self.wind_direction)
        self.num_steps += 1
        self.wind_direction = random.choice([0, 1, 2, 3])
        at_goal = self.pos == self.end_pos
        truncated = self.num_steps >= 200
        done = at_goal or truncated
        return (
            [[self.pos[0], self.pos[1]], self.wind_direction],
            100 * int(at_goal),
            done,
            truncated,
            {},
        )

    def _get_new_pos(self, pos, direction):
        if direction == 0:
            new_pos = (pos[0] - 1, pos[1])
        elif direction == 1:
            new_pos = (pos[0], pos[1] + 1)
        elif direction == 2:
            new_pos = (pos[0] + 1, pos[1])
        elif direction == 3:
            new_pos = (pos[0], pos[1] - 1)
        if (
            new_pos[0] >= 0
            and new_pos[0] < self.x_dim
            and new_pos[1] >= 0
            and new_pos[1] < self.y_dim
            and self.map[new_pos[0]][new_pos[1]] != "#"
        ):
            return new_pos
        else:
            return pos  # did not move