import functools
import random

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import heapdict

from gymnasium.spaces import Discrete, MultiDiscrete, Box, Tuple, Dict
from pettingzoo.utils.env import ParallelEnv, AECEnv
from pettingzoo.test import api_test
from pettingzoo.utils import agent_selector, parallel_to_aec

from cyclist import Cyclist, random_cyclists

class AsyncMapEnv(AECEnv):
    velocity = 20
    large_const = 1e3
    params = {
        'pollution': -50,
        'neighbourhood': -0.05,
        'goal': 100,
    }
    metadata = {}

    def __init__(self, map_size=5, num_agents=2, num_iters=None, const_graph=False, render_mode=None, figpath='figures'):
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
                i: {
                    'pollution': max(0, random.normalvariate(5, 2.5)), 'l': max(0, random.normalvariate(200, 50))
                } for i in self.G.edges()
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
        self.agent_queue = heapdict.heapdict() # agent: length of journey (s) to next node

        # self.action_spaces = {
        #     agent: self.action_space(agent) for agent in self.possible_agents
        # }
        # self.observation_spaces = {
        #     agent: self.observation_space(agent) for agent in self.possible_agents
        # }

        self.render_mode = render_mode
        if self.render_mode == "human":
            self.pos = nx.spring_layout(self.G, iterations=1_500)
            self.figpath = figpath
            self.colourmap = cm.hsv(np.linspace(0, 1, num_agents+1))
        
        print(f"Initialised env with {num_agents} agents on a {map_size}x{map_size} graph.")

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Dict({
            "position": Box(low=0, high=1, shape=(self.num_nodes,), dtype=int),         # one hot position
            "goal": Box(low=0, high=1, shape=(self.num_nodes,), dtype=int),             # one hot goal
            "hr": Box(low=0, high=300, shape=(1,)),                                     # current heart rate
            "map": Box(low=0, high=300, shape=(self.num_nodes, self.num_nodes)),        # pollution-weighted adjacency matrix
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Dict({
            "destination": Discrete(self.num_nodes),        # destination node
            "velocity": Discrete(6),                        # movement velocity encoded as 0: 5 through 6: 35
        })

    def agent_selector(self):
        agent, duration = self.agent_queue.peekitem()
        return agent
    
    def reset(self, seed=None, return_info=False, options=None):
        if seed:
            random.seed(seed)

        self.agents = self.possible_agents[:]
        for agent in self.agents:
            self.agent_queue[agent] = 0
            self.agent_name_mapping[agent].reset()
        self.agent_selection = self.agent_selector()

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
        self.rewards = {
            agent: 0 for agent in self.agents
        }
        self._cumulative_rewards = {
            agent: 0 for agent in self.agents
        }
        self.truncations = {
            agent: False for agent in self.agents
        }
        self.terminations = {
            agent: False for agent in self.agents
        }

        heuristic = {agent: nx.shortest_path_length(self.G, target=self.goals[agent])
                     for agent in self.agents}
        node_attrs = {
            i:
            {f'heur_{agent}': heuristic[agent][i] for agent in self.agents}
            for i in range(self.num_nodes)
        }
        nx.set_node_attributes(self.G, node_attrs)

        self.state = {
            agent: None for agent in self.agents
        }

        if self.render_mode:
            self.render()

        self.infos = {agent: {} for agent in self.agents}

        # return self.observations, self.infos

    def step(self, action):
        if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            # self._was_dead_step(action)
            return

        # action = {'destination': node_idx, 'velocity': float}
        agent = self.agent_selection
        self.rewards = {
            agent: 0 for agent in self.agents
        }
        if self._get_action_mask(self.positions[agent])[action['destination']]: # if valid move
            self.state[agent] = ((self.positions[agent], action['destination']), action['velocity'])
            for a, state in self.state.items():
                if state:
                    edge, vel = state
                    if (edge == self.state[agent][0]) and (a != agent):
                        action['velocity'] = min(action['velocity'], vel)

            pollution, duration = self._get_pollution(agent, action['destination'], (action['velocity']+1)*5)
            at_goal = (action['destination'] == self.goals[agent])
            heuristic = nx.get_node_attributes(self.G, name=f'heur_{agent}')[action['destination']]
            self.rewards[agent] = self._get_reward(pollution, heuristic, at_goal)
            self.terminations[agent] = at_goal

            self.pollution[agent] += pollution
            self.positions[agent] = action['destination']

        else: # if invalid move, don't move agent
            self.state[agent] = None
            pollution, duration = self._get_pollution(agent, action['destination'], (action['velocity']+1)*5)
            heuristic = nx.get_node_attributes(self.G, name=f'heur_{agent}')[self.positions[agent]]
            self.rewards[agent] = self._get_reward(pollution, 0, False) # no heurstic as no movement
            # self.pollution[agent] += pollution

        if self.render_mode:
            self.render()

        self._accumulate_rewards()

        self.timestep += 1
        env_truncation = self.timestep >= self.num_iters
        self.truncations = {agent: env_truncation for agent in self.agents}


        self.observations = {
            agent: {
                'position': self._one_hot(self.positions[agent]),
                'goal': self._one_hot(self.goals[agent]),
                'hr': np.array([self.agent_name_mapping[agent].hr]),
                'map': self._pollution_map(),
            } for agent in self.agents
        }

        if env_truncation:
            self.agents = []
        else:
            if self.terminations[self.agent_selection]:
                self.agents.remove(self.agent_selection)
                self.agent_queue[agent] = 0
                self.agent_queue.popitem()
            else:
                self.agent_queue[self.agent_selection] += duration

            if len(self.agents) > 0:
                self.agent_selection = self.agent_selector()

        self.infos = {agent: {} for agent in self.possible_agents}

        return self.observations, self.rewards, self.terminations, self.truncations, self.infos
    
    def observe(self, agent):
        return np.array(self.observations[agent]).item()

    def _get_action_mask(self, node):
        valid_actions = np.zeros((self.num_nodes,))
        idxs = [y for (x, y) in self.G.edges(node)]
        valid_actions[idxs] = 1
        return valid_actions
    
    def _get_pollution(self, agent, action, velocity):
        heights = nx.get_node_attributes(self.G, 'h')
        lens = nx.get_edge_attributes(self.G, 'l')
        polls = nx.get_edge_attributes(self.G, 'pollution')

        if (action is not None) and (self._get_action_mask(self.positions[agent])[action]):
            # return a single value for pollution corresponding to an action (ie edge traversal)
            dh = heights[self.positions[agent]] - heights[action]
            dl = lens[(self.positions[agent], action)]
            p_req = self.agent_name_mapping[agent].get_segment_power(dh, dl, velocity/3.6)
            rdd = self.agent_name_mapping[agent].eval_segment(polls[(self.positions[agent], action)], p_req, dl*3.6/velocity)
            return rdd, dl*3.6/velocity
        elif action is not None:
            return self.large_const, 0
        # else:
        #     # return the full array of pollution values for traversal from curr_node
        #     observation = np.array([self.large_const]*self.num_nodes)
        #     for (x, y) in self.G.edges(self.positions[agent]):
        #         observation[y] = self._get_pollution(agent, action=y)
        #     return observation
        
    def _pollution_map(self):
        return nx.attr_matrix(self.G, edge_attr='pollution')[0]
        
    def _get_reward(self, pollution, neighbourhood, at_goal):
        return self.params['pollution']*pollution + self.params['neighbourhood']*neighbourhood + self.params['goal']*int(at_goal)
    
    def _one_hot(self, idx):
        arr = np.zeros(shape=(self.num_nodes,), dtype=int)
        arr[idx] = 1
        return arr

    def render(self):
        if self.render_mode == "ascii":
            print("Step ", self.timestep, self.positions, self.pollution)

        elif self.render_mode == "human":
            options = {
                "node_size": 2000,
                "linewidths": 5,
            }

            try:
                plt.figure(figsize=(15,15))
                plt.title(f'Step {self.timestep}')
                nx.draw_networkx(self.G, pos=self.pos, node_color='white', edgecolors='black', font_size=14, width=5, **options) # draw empty graph
                
                for agent in self.agents:
                    c = [self.colourmap[int(agent.split('_')[-1])]]
                    nx.draw_networkx_nodes(self.G, nodelist=[self.goals[agent]], pos=self.pos, edgecolors=c, node_color='white', label='Goal', **options)
                    if self.state[agent]:
                        nx.draw_networkx_edges(self.G, edgelist=[self.state[agent][0]], pos=self.pos, edge_color=c, width=options['linewidths'], label='Cyclist')
                    else:
                        nx.draw_networkx_nodes(self.G, nodelist=[self.positions[agent]], pos=self.pos, node_color=c, alpha=0.4, edgecolors='black', label='Cyclist', **options)
                
                if self.figpath:
                    plt.savefig(f'{self.figpath}/img_{self.timestep}')
                else:
                    plt.show()

            except KeyError:
                pass
            finally:
                plt.close()

if __name__ == "__main__":
    env_config = {
        'num_agents': 2,
        'map_size': 2,
        'num_iters': 1_000,
        # 'render_mode': 'human',
        # 'figpath': None,
    }

    env = AsyncMapEnv(**env_config)
    env.reset()
    print(env.goals)

    ctr = 0
    while len(env.agents) > 0:
        ctr += 1
        destination = random.choice([y for (x, y) in env.G.edges(env.positions[env.agent_selection])])
        vel = random.randint(0,6)
        # print(env.agent_selection, destination, (vel+1)*5)
        env.step({'destination': destination, 'velocity': vel})
    print(f"Took {ctr} iterations")