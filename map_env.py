import functools
import random

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from gymnasium.spaces import Discrete, MultiDiscrete, Box, Tuple, Dict
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils import agent_selector

from cyclist import Cyclist, random_cyclists


class CustomEnvironment(ParallelEnv):
    ambient_pm = 10
    velocity = 20
    large_const = 1e3
    params = {
        'pollution': -50,
        'neighbourhood': -0.05,
        'goal': 100,
    }

    def __init__(self, map_size=5, num_agents=2, num_iters=None, const_graph=True, render_mode=None, figpath='figures'):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces
        These attributes should not be changed after initialization.
        """
        super().__init__()

        self.G = nx.grid_graph(dim=(map_size, map_size))
        self.G = nx.convert_node_labels_to_integers(self.G)
        self.G = nx.DiGraph(self.G)
        self.num_nodes = len(self.G.nodes())
        self.num_edges = len(self.G.edges())
        if const_graph:
            node_attrs = {
                i: {'h': 0} for i in self.G.nodes()
            }
            edge_attrs = {
                i: {'pollution': 5} for i in self.G.edges()
            }
        else:
            random.seed(0) # ensure graph is same every time
            node_attrs = {
                i: {'h': random.normalvariate(10, 2)} for i in self.G.nodes()
            }
            edge_attrs = {
                i: {'pollution': max(0, random.normalvariate(5, 2.5))} for i in self.G.edges()
            }
            random.seed() # reset seed for selecting O-D pairs
        
        nx.set_node_attributes(self.G, node_attrs)
        nx.set_edge_attributes(self.G, edge_attrs)

        if num_iters:
            self.num_iters = num_iters
        else:
            self.num_iters = 100

        self.timestep = None
        self.possible_agents = ["cyclist_" + str(r) for r in range(num_agents)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, random_cyclists(num_agents, mass_mean=85, mass_std=15, hr_0_mean=70, hr_0_std=10,
                                                      rise_time_mean=30, rise_time_std=5, hr_max_mean=180, hr_max_std=20,
                                                      kf_mean=3e-5, kf_std=1e-5, c_mean=0.3, c_std=0.05))
        )

        self.render_mode = render_mode
        if self.render_mode == "human":
            self.pos = nx.spring_layout(self.G, iterations=1_500)
            self.figpath = figpath

        print(f"Initialised env with {num_agents} agents on a {map_size}x{map_size} graph.")

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Dict({
            "position": Box(low=0, high=1, shape=(self.num_nodes,)),        # one hot position
            "goal": Box(low=0, high=1, shape=(self.num_nodes,)),            # one hot goal
            "hr": Box(low=0, high=300, shape=(1,)),                         # current heart rate
            "map": Box(low=0, high=np.inf, shape=(self.num_nodes, self.num_nodes)),  # pollution-weighted adjacency matrix
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(self.num_nodes)

    def reset(self, seed=None, return_info=False, options=None):
        if seed:
            random.seed(seed)

        self.agents = self.possible_agents[:]

        self.timestep = 0
        tasks = {
            agent: random.sample(self.G.nodes(), 2) for agent in self.agents
        }
        self.positions = {
            agent: tasks[agent][0] for agent in self.agents
        }
        self.goals = {
            agent: tasks[agent][1] for agent in self.agents
        }

        self.observations = {
            agent: {
                'position': self._one_hot(tasks[agent][0]),
                'goal': self._one_hot(tasks[agent][1]),
                'hr': np.array([self.agent_name_mapping[agent].hr]),
                'map': self._pollution_map(),
            } for agent in self.agents
        }

        self.pollution = {
            agent: 0 for agent in self.agents
        }

        heuristic = {agent: nx.shortest_path_length(self.G, target=self.goals[agent])
                     for agent in self.agents}
        node_attrs = {
            i:
            {f'heur_{agent}': heuristic[agent][i] for agent in self.agents}
            for i in range(self.num_nodes)
        }
        nx.set_node_attributes(self.G, node_attrs)

        if self.render_mode:
            self.render()

        if not return_info:
            return self.observations
        else:
            infos = {agent: {} for agent in self.agents}
            return self.observations, infos

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # print("starting step")
        # rewards for all agents are placed in the rewards dictionary to be returned
        rewards = {}
        terminations = {}
        for agent in self.agents:
            if self._get_action_mask(self.positions[agent])[actions[agent]]:
                pollution = self._get_pollution(agent, actions[agent])
                at_goal = (actions[agent] == self.goals[agent])
                heuristic = nx.get_node_attributes(self.G, name=f'heur_{agent}')[actions[agent]]
                rewards[agent] = self._get_reward(pollution, heuristic, at_goal)
                terminations[agent] = at_goal

                self.pollution[agent] += pollution
                self.positions[agent] = actions[agent]
            else:
                # if invalid move, don't move agent
                pollution = self.large_const
                rewards[agent] = self._get_reward(pollution, 0, False) # no heurstic as no movement
                self.pollution[agent] += pollution

        self.timestep += 1
        env_truncation = self.timestep >= self.num_iters
        truncations = {agent: env_truncation for agent in self.agents}

        self.observations = {
            agent: {
                'position': self._one_hot(self.positions[agent]),
                'goal': self._one_hot(self.goals[agent]),
                'hr': np.array([self.agent_name_mapping[agent].hr]),
                'map': self._pollution_map(),
            } for agent in self.agents
        }

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if self.render_mode:
            self.render(0, rewards)

        if env_truncation:
            self.agents = []
        else:
            for agent, status in terminations.items():
                if status:
                    self.agents.remove(agent)

        return self.observations, rewards, terminations, truncations, infos
    
    def observe(self, agent):
        return np.array(self.observations[agent])

    def _get_action_mask(self, node):
        valid_actions = np.zeros((self.num_nodes,))
        idxs = [y for (x, y) in self.G.edges(node)]
        valid_actions[idxs] = 1
        return valid_actions
    
    def _get_pollution(self, agent, action=None):
        heights = nx.get_node_attributes(self.G, 'h')

        if action is not None and self._get_action_mask(self.positions[agent])[action]:
            # return a single value for pollution corresponding to an action (ie edge traversal)
            dh = heights[self.positions[agent]] - heights[action]
            dl = 100
            p_req = self.agent_name_mapping[agent].get_segment_power(dh, dl, self.velocity/3.6)
            rdd = self.agent_name_mapping[agent].eval_segment(self.ambient_pm, p_req, dl*3.6/self.velocity)
            return rdd
        elif action is not None:
            return self.large_const
        
        else:
            # return the full array of pollution values for traversal from curr_node
            observation = np.array([self.large_const]*self.num_nodes)
            for (x, y) in self.G.edges(self.positions[agent]):
                observation[y] = self._get_pollution(agent, action=y)
            return observation
        
    def _pollution_map(self):
        return nx.attr_matrix(self.G, edge_attr='pollution')[0]
        
    def _get_reward(self, pollution, neighbourhood, at_goal):
        return self.params['pollution']*pollution + self.params['neighbourhood']*neighbourhood + self.params['goal']*int(at_goal)
    
    def _one_hot(self, idx):
        arr = np.zeros(shape=(self.num_nodes,))
        arr[idx] = 1
        return arr

    def render(self, agent_idx=0, reward=None):
        if self.render_mode == "ascii":
            print("Step ", self.timestep, self.positions, self.pollution)

        elif self.render_mode == "human":
            options = {
                "font_size": 14,
                "node_size": 2000,
                "edgecolors": "black",
                "linewidths": 5,
                "width": 5,
            }
            if reward == None:
                reward = {agent: 0 for agent in self.possible_agents}

            try:
                plt.figure(figsize=(15,15))
                plt.title(f'Step {self.timestep}, {self.possible_agents[agent_idx]} reward {reward[self.possible_agents[agent_idx]]}')
                nx.draw_networkx(self.G, pos=self.pos, node_color='white', **options)
                nx.draw_networkx(self.G, nodelist=[self.positions[self.possible_agents[agent_idx]]], pos=self.pos, node_color='blue', alpha=0.4, label='Cyclist', **options)
                nx.draw_networkx(self.G, nodelist=[self.goals[self.possible_agents[agent_idx]]], pos=self.pos, node_color='green', alpha=0.4, label='Goal', **options)
                plt.savefig(f'{self.figpath}/img_{self.timestep}')
                plt.close()
            except KeyError:
                pass
