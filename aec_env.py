import functools
import random

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import heapdict
from pprint import pprint

from gymnasium.spaces import Discrete, MultiDiscrete, Box, Tuple, Dict
from pettingzoo.utils.env import ParallelEnv, AECEnv
from pettingzoo.test import api_test
from pettingzoo.utils import agent_selector, parallel_to_aec

from cyclist import Cyclist, random_cyclists, cyclist_sample

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    return np.exp(-fac / 2) / N

def hill_create(map_size, height=1, width=1, centre=np.array([0., 0.]), skewness=np.array([[ 1., 0], [0,  1.]])):
    X = np.array([i for i in range(map_size)])
    Y = np.array([i for i in range(map_size)])
    X, Y = np.meshgrid(X, Y)

    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    
    hill = multivariate_gaussian(pos, centre, width*skewness)
    return height*hill

def make_landscape(map_size, hill_params):
    Z = np.zeros((map_size, map_size))
    for centre, height, width in hill_params:
        Z += hill_create(map_size, height, width, np.array(centre))
    return Z


class AsyncMapEnv(AECEnv):
    velocity = 25
    large_const = 1e3
    params = {
        'pollution': -50,
        'neighbourhood': -0.05,
        'goal': 1,
    }
    vel_reference = { # low and high velocities in kph for three fitness levels
        0: {0: 10, 1: 20},
        1: {0: 15, 1: 25},
        2: {0: 25, 1: 35},
    }
    metadata = {}

    def __init__(self, map_size=5, num_agents=2, reinit_agents=False, num_iters=None, const_graph=False, congestion=True, hill_attrs=[], poll_attrs=[], corners=False, fit_split=3, render_mode=None, figpath='figures'):
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
                i: {'h': 0, 'pollution': 5} for i in self.G.nodes()
            }
            nx.set_node_attributes(self.G, node_attrs)

        elif len(hill_attrs) > 0:
            np.random.seed(0)
            random.seed(0)
            map_ax = [i for i in range(map_size)]
            X, Y = np.meshgrid(map_ax, map_ax)
            heightmap = np.random.random_sample(size=(map_size, map_size)) + make_landscape(map_size, hill_attrs)*10
            pollmap = (np.random.random_sample(size=(map_size, map_size))*0.1 + make_landscape(map_size, poll_attrs))*25
            heightmap = heightmap.reshape(map_size**2,)
            pollmap = pollmap.reshape(map_size**2,)
            node_attrs = {
                i: {'h': heightmap[i], 'pollution': pollmap[i]} for i in self.G.nodes()
            }
            nx.set_node_attributes(self.G, node_attrs)

        else:
            random.seed(0) # ensure graph is same every time
            node_attrs = {
                i: {'h': random.normalvariate(10, 2), 'pollution': random.normalvariate(5, 2.5)} for i in self.G.nodes()
            }
            nx.set_node_attributes(self.G, node_attrs)

        edge_attrs = {
            i: {
                'pollution': max(0.1, np.mean([self.G.nodes[i[1]]['pollution'], self.G.nodes[i[0]]['pollution']])),
                'l': max(0.1, random.normalvariate(500, 20)),
                'dh': self.G.nodes[i[1]]['h'] - self.G.nodes[i[0]]['h'],
            } for i in self.G.edges()
        }
        nx.set_edge_attributes(self.G, edge_attrs)
        random.seed() # reset seed for selecting O-D pairs
        np.random.seed()

        if num_iters:
            self.num_iters = num_iters
        else:
            self.num_iters = 1_000

        self.timestep = None
        self.possible_agents = ["cyclist_" + str(r) for r in range(num_agents)]

        # implement specification of cyclist fitnesses
        if fit_split == 2:
            cyclists = cyclist_sample(num_agents//2, 0, num_agents//2 + num_agents%2)
        else:
            cyclists = cyclist_sample(num_agents//3, num_agents//3, num_agents//3 + num_agents%3)
        self.agent_name_mapping = dict(
            zip(self.possible_agents, cyclists)
        )
        self.reinit_agents = reinit_agents
        self.agent_queue = heapdict.heapdict() # agent: length of journey (s) to next node

        self.render_mode = render_mode
        if self.render_mode == "human":
            self.pos = nx.spring_layout(self.G, iterations=1_500)
            self.figpath = figpath
            self.colourmap = cm.hsv(np.linspace(0, 1, num_agents+1))

        self.congestion = congestion
        self.fit_split = fit_split
        self.corners = corners
        if self.corners:
            self.corner_list = [0, map_size-1, map_size**2-map_size, map_size**2-1]
        
        print(f"Initialised env with {num_agents} agents on a {map_size}x{map_size} graph.")

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Dict({
            "position": Box(low=0, high=1, shape=(self.num_nodes,), dtype=int),         # one hot position
            "goal": Box(low=0, high=1, shape=(self.num_nodes,), dtype=int),             # one hot goal
            "cyclist": Box(low=0, high=300, shape=(8,)),                                # cyclist params
            "pollution": Box(low=0, high=300, shape=(self.num_nodes, self.num_nodes)),  # pollution-weighted adjacency matrix
            "steepness": Box(low=-5, high=5, shape=(self.num_nodes, self.num_nodes)),   # gradient-weighted adjacency matrix
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Dict({
            "destination": Discrete(self.num_nodes),        # destination node
            "velocity": Discrete(2),                        # movement velocity encoded as 0: low, 1: high
        })

    def agent_selector(self):
        agent, duration = self.agent_queue.peekitem()
        return agent
    
    def reset(self, seed=None, return_info=False, options=None):
        if seed:
            random.seed(seed)
        if options == None:
            options = {'test': False}

        self.agents = self.possible_agents[:]
        if self.reinit_agents:
            if self.fit_split == 2:
                self.agent_name_mapping = dict(
                    zip(self.possible_agents, cyclist_sample(len(self.agents)//2, 0, len(self.agents)//2 + len(self.agents)%2))
                )
            else:
                self.agent_name_mapping = dict(
                    zip(self.possible_agents, cyclist_sample(len(self.agents)//3, len(self.agents)//3, len(self.agents)//3 + len(self.agents)%3))
                )
        for agent in self.agents:
            self.agent_queue[agent] = 0
            self.agent_name_mapping[agent].reset()
        self.agent_selection = self.agent_selector()

        self.timestep = 0
        if options['test']:
            tasks = {
                agent: [0, self.num_nodes-1] for agent in self.agents
            }
        elif self.corners:
            tasks = {
                agent: random.sample(self.corner_list, 2) for agent in self.agents
            }
        else:
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
                'cyclist': self.agent_name_mapping[agent]._param_list(),
                'pollution': self._pollution_map(),
                'steepness': self._power_map(),
            } for agent in self.agents
        }

        self.pollution = {
            agent: 0 for agent in self.agents
        }
        self.duration = {
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
            {
                f'heur_{agent}': heuristic[agent][i] for agent in self.agents
            } for i in range(self.num_nodes)
            
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
        action['velocity'] = self._act_to_vel(action['velocity'])
        if self._get_action_mask(self.positions[agent])[action['destination']]: # if valid move
            if self.congestion:
                for a, state in self.state.items():
                    if state:
                        edge, vel = state
                        if (edge == (self.positions[agent], action['destination'])) and (a != agent):
                            # print(f"Congestion on {edge} between {agent} and {a}: {action['velocity']}, {vel} = {min(action['velocity'], vel)}")
                            action['velocity'] = min(action['velocity'], vel)
                self.state[agent] = ((self.positions[agent], action['destination']), action['velocity'])

            pollution, duration = self._get_pollution(agent, action['destination'], action['velocity'])
            at_goal = (action['destination'] == self.goals[agent])
            heuristic = nx.get_node_attributes(self.G, name=f'heur_{agent}')[action['destination']]
            self.rewards[agent] = self._get_reward(pollution, heuristic, at_goal)
            self.terminations[agent] = at_goal

            self.pollution[agent] += pollution
            self.duration[agent] += duration
            self.positions[agent] = action['destination']

        else: # if invalid move, don't move agent
            self.state[agent] = None
            pollution, duration = self._get_pollution(agent, action['destination'], action['velocity'])
            self.rewards[agent] = self._get_reward(pollution, 0, False)
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
                'cyclist': self.agent_name_mapping[agent]._param_list(),
                'pollution': self._pollution_map(),
                'steepness': self._power_map(),
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
        heights = nx.get_edge_attributes(self.G, 'dh')
        lens = nx.get_edge_attributes(self.G, 'l')
        polls = nx.get_edge_attributes(self.G, 'pollution')

        if (action is not None) and (self._get_action_mask(self.positions[agent])[action]):
            # return a single value for pollution corresponding to an action (ie edge traversal)
            # dh = heights[self.positions[agent]] - heights[action]
            dh = heights[(self.positions[agent], action)]
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
        
    def _power_map(self):
        dh = nx.attr_matrix(self.G, edge_attr='dh')[0]
        l = nx.attr_matrix(self.G, edge_attr='l')[0]
        res = np.zeros_like(dh)
        np.divide(dh, l, out=res, where=(l > 0))
        return np.nan_to_num(res)
        # return nx.attr_matrix(self.G, edge_attr='dh')[0]

    def _pollution_map(self):
        return nx.attr_matrix(self.G, edge_attr='pollution')[0]
        
    def _get_reward(self, pollution, neighbourhood, at_goal):
        # return self.params['pollution']*pollution + self.params['neighbourhood']*neighbourhood + self.params['goal']*int(at_goal)
        return self.params['pollution']*pollution + self.params['goal']*int(at_goal)
    
    def _act_to_vel(self, action, fitness=None):
        if fitness is not None:
            return self.vel_reference[fitness][action]
        else:
            return self.vel_reference[self.agent_name_mapping[self.agent_selection].fitness][action]

    def _one_hot(self, idx):
        arr = np.zeros(shape=(self.num_nodes,), dtype=int)
        arr[idx] = 1
        return arr

    def render(self):
        if self.render_mode == "ascii":
            print("Step ", self.timestep, self.positions, self.pollution)

        elif self.render_mode == "human":
            options = {
                "linewidths": 5,
            }
            if self.num_nodes <= 25:
                options["node_size"] = 2000
            else:
                options["node_size"] = 500

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

class AsyncMapEnv_NoVel(AECEnv):
    velocity = 25
    large_const = 1e3
    params = {
        'pollution': -50,
        'neighbourhood': -0.01,
        'goal': 10,
    }
    vel_reference = { # low and high velocities in kph for three fitness levels
        0: {0: 10, 1: 20},
        1: {0: 15, 1: 25},
        2: {0: 25, 1: 35},
    }
    metadata = {}

    def __init__(self, map_size=5, num_agents=2, reinit_agents=False, num_iters=None, const_graph=False, congestion=True, hill_attrs=[], poll_attrs=[], corners=False, fit_split=3, goal_scheduling=False, render_mode=None, figpath='figures'):
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
                i: {'h': 0, 'pollution': 5} for i in self.G.nodes()
            }
            nx.set_node_attributes(self.G, node_attrs)

        elif len(hill_attrs) > 0:
            np.random.seed(0)
            random.seed(0)
            map_ax = [i for i in range(map_size)]
            X, Y = np.meshgrid(map_ax, map_ax)
            heightmap = np.random.random_sample(size=(map_size, map_size)) + make_landscape(map_size, hill_attrs)*10
            pollmap = (np.random.random_sample(size=(map_size, map_size))*0.1 + make_landscape(map_size, poll_attrs))*25
            heightmap = heightmap.reshape(map_size**2,)
            pollmap = pollmap.reshape(map_size**2,)
            node_attrs = {
                i: {'h': heightmap[i], 'pollution': pollmap[i]} for i in self.G.nodes()
            }
            nx.set_node_attributes(self.G, node_attrs)

        else:
            random.seed(0) # ensure graph is same every time
            node_attrs = {
                i: {'h': random.normalvariate(10, 2), 'pollution': random.normalvariate(5, 2.5)} for i in self.G.nodes()
            }
            nx.set_node_attributes(self.G, node_attrs)

        edge_attrs = {
            i: {
                'pollution': max(0.1, np.mean([self.G.nodes[i[1]]['pollution'], self.G.nodes[i[0]]['pollution']])),
                'l': max(0.1, random.normalvariate(500, 20)),
                'dh': self.G.nodes[i[1]]['h'] - self.G.nodes[i[0]]['h'],
            } for i in self.G.edges()
        }
        nx.set_edge_attributes(self.G, edge_attrs)
        random.seed() # reset seed for selecting O-D pairs
        np.random.seed()

        if num_iters:
            self.num_iters = num_iters
        else:
            self.num_iters = 1_000

        self.timestep = None
        self.possible_agents = ["cyclist_" + str(r) for r in range(num_agents)]

        self.goal_scheduling = goal_scheduling
        self.global_iters = -5 # deal with Ray setup

        # implement specification of cyclist fitnesses
        if fit_split == 2:
            cyclists = cyclist_sample(num_agents//2, 0, num_agents//2 + num_agents%2)
        else:
            cyclists = cyclist_sample(num_agents//3, num_agents//3, num_agents//3 + num_agents%3)
        self.agent_name_mapping = dict(
            zip(self.possible_agents, cyclists)
        )
        self.reinit_agents = reinit_agents
        self.agent_queue = heapdict.heapdict() # agent: length of journey (s) to next node

        self.render_mode = render_mode
        if self.render_mode == "human":
            self.pos = nx.spring_layout(self.G, iterations=1_500)
            self.figpath = figpath
            self.colourmap = cm.hsv(np.linspace(0, 1, num_agents+1))

        self.congestion = congestion
        self.fit_split = fit_split
        self.corners = corners
        if self.corners:
            self.corner_list = [0, map_size-1, map_size**2-map_size, map_size**2-1]
        
        print(f"Initialised env with {num_agents} agents on a {map_size}x{map_size} graph.")

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Dict({
            "position": Box(low=0, high=1, shape=(self.num_nodes,), dtype=int),         # one hot position
            "goal": Box(low=0, high=1, shape=(self.num_nodes,), dtype=int),             # one hot goal
            "cyclist": Box(low=0, high=300, shape=(8,)),                                # cyclist params
            "pollution": Box(low=0, high=300, shape=(self.num_nodes, self.num_nodes)),  # pollution-weighted adjacency matrix
            "steepness": Box(low=-5, high=5, shape=(self.num_nodes, self.num_nodes)),   # gradient-weighted adjacency matrix
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Dict({
            "destination": Discrete(self.num_nodes),        # destination node
            # "velocity": Discrete(2),                        # movement velocity encoded as 0: low, 1: high
        })

    def agent_selector(self):
        agent, duration = self.agent_queue.peekitem()
        return agent
    
    def reset(self, seed=None, return_info=False, options=None):
        if seed:
            random.seed(seed)
        if options == None:
            options = {'test': False}

        self.agents = self.possible_agents[:]
        if self.reinit_agents:
            if self.fit_split == 2:
                self.agent_name_mapping = dict(
                    zip(self.possible_agents, cyclist_sample(len(self.agents)//2, 0, len(self.agents)//2 + len(self.agents)%2))
                )
            else:
                self.agent_name_mapping = dict(
                    zip(self.possible_agents, cyclist_sample(len(self.agents)//3, len(self.agents)//3, len(self.agents)//3 + len(self.agents)%3))
                )
        for agent in self.agents:
            self.agent_queue[agent] = 0
            self.agent_name_mapping[agent].reset()
        self.agent_selection = self.agent_selector()

        self.timestep = 0
        if options['test']:
            tasks = {
                agent: [0, self.num_nodes-1] for agent in self.agents
            }
        elif self.corners:
            tasks = {
                agent: random.sample(self.corner_list, 2) for agent in self.agents
            }
        else:
            if self.goal_scheduling:
                k = max(1, int(2+(self.global_iters//3-500)/500))
                tasks = {
                    agent: random.sample(self.G.nodes(), 2) for agent in self.agents
                }
                for agent in self.agents:
                    nodes = nx.ego_graph(self.G, tasks[agent][0], radius=k).nodes()
                    tasks[agent][1] = tasks[agent][0]
                    while tasks[agent][0] == tasks[agent][1]:
                        tasks[agent][1] = random.sample(nodes, 1)[0]
            else:
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
                'cyclist': self.agent_name_mapping[agent]._param_list(),
                'pollution': self._pollution_map(),
                'steepness': self._power_map(),
            } for agent in self.agents
        }

        self.pollution = {
            agent: 0 for agent in self.agents
        }
        self.duration = {
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

        heuristic = {agent: nx.shortest_path_length(self.G, target=self.goals[agent]) for agent in self.agents}

        node_attrs = {
            i:
            {
                f'heur_{agent}': heuristic[agent][i] for agent in self.agents
            } for i in range(self.num_nodes)
            
        }
        nx.set_node_attributes(self.G, node_attrs)

        self.state = {
            agent: None for agent in self.agents
        }

        if self.render_mode:
            self.render()

        self.infos = {agent: {} for agent in self.agents}
        self.global_iters += 1

        # return self.observations, self.infos

    def step(self, action):
        if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            # self._was_dead_step(action)
            return

        agent = self.agent_selection
        self.rewards = {
            agent: 0 for agent in self.agents
        }

        if self._get_action_mask(self.positions[agent])[action['destination']]: # if valid move
            self.state[agent] = ((self.positions[agent], action['destination']), self.velocity)
            pollution, duration = self._get_pollution(agent, action['destination'], self.velocity)
            at_goal = (action['destination'] == self.goals[agent])
            heuristic = nx.get_node_attributes(self.G, name=f'heur_{agent}')[action['destination']]
            self.rewards[agent] = self._get_reward(pollution, heuristic, at_goal)
            self.terminations[agent] = at_goal

            self.pollution[agent] += pollution
            self.duration[agent] += duration
            self.positions[agent] = action['destination']

        else: # if invalid move, don't move agent
            self.state[agent] = None
            pollution, duration = self._get_pollution(agent, action['destination'], self.velocity)
            heuristic = nx.get_node_attributes(self.G, name=f'heur_{agent}')[action['destination']]
            self.rewards[agent] = self._get_reward(pollution, heuristic, False)
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
                'cyclist': self.agent_name_mapping[agent]._param_list(),
                'pollution': self._pollution_map(),
                'steepness': self._power_map(),
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
        heights = nx.get_edge_attributes(self.G, 'dh')
        lens = nx.get_edge_attributes(self.G, 'l')
        polls = nx.get_edge_attributes(self.G, 'pollution')

        if (action is not None) and (self._get_action_mask(self.positions[agent])[action]):
            # return a single value for pollution corresponding to an action (ie edge traversal)
            # dh = heights[self.positions[agent]] - heights[action]
            dh = heights[(self.positions[agent], action)]
            dl = lens[(self.positions[agent], action)]
            p_req = self.agent_name_mapping[agent].get_segment_power(dh, dl, velocity/3.6)
            rdd = self.agent_name_mapping[agent].eval_segment(polls[(self.positions[agent], action)], p_req, dl*3.6/velocity)
            return rdd, dl*3.6/velocity
        elif action is not None:
            return self.large_const, 0
        
    def _power_map(self):
        dh = nx.attr_matrix(self.G, edge_attr='dh')[0]
        l = nx.attr_matrix(self.G, edge_attr='l')[0]
        res = np.zeros_like(dh)
        np.divide(dh, l, out=res, where=(l > 0))
        return np.nan_to_num(res)
        # return nx.attr_matrix(self.G, edge_attr='dh')[0]

    def _pollution_map(self):
        return nx.attr_matrix(self.G, edge_attr='pollution')[0]
        
    def _get_reward(self, pollution, neighbourhood, at_goal):
        return self.params['pollution']*pollution + self.params['goal']*int(at_goal)
        # return self.params['pollution']*pollution + self.params['neighbourhood']*neighbourhood + self.params['goal']*int(at_goal)

    def _one_hot(self, idx):
        arr = np.zeros(shape=(self.num_nodes,), dtype=int)
        arr[idx] = 1
        return arr

    def render(self):
        if self.render_mode == "ascii":
            print("Step ", self.timestep, self.positions, self.pollution)

        elif self.render_mode == "human":
            options = {
                "linewidths": 5,
            }
            if self.num_nodes <= 25:
                options["node_size"] = 2000
            else:
                options["node_size"] = 500

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
        'num_agents': 4,
        'map_size': 10,
        'num_iters': 1_000,
        'fit_split': 3,
        'corners': False,
        'hill_attrs': [
                        [[10,5], 10, 4],
                        [[7,12], 20, 10],
                        [[15,13], 15, 6],
                      ],
        'goal_scheduling': True,
        # 'render_mode': 'human',
        # 'figpath': None,
    }

    # --- ENV TESTING
    env = AsyncMapEnv_NoVel(**env_config)
    env.reset()
    # pprint(env.agent_name_mapping)
    print(env.positions)
    print(env.goals)
    # print(env.observations[env.agent_selection])

    # ctr = 0
    # while len(env.agents) > 0:
    #     ctr += 1
    #     destination = random.choice([y for (x, y) in env.G.edges(env.positions[env.agent_selection])])
    #     vel = random.randint(0,1)
    #     print(env.agent_selection, destination, vel)
    #     env.step({'destination': destination, 'velocity': vel})
    # print(f"Took {ctr} iterations")

    # ---- HILL TESTING
    # map_size = 8
    # map_ax = [i for i in range(map_size)]
    # X, Y = np.meshgrid(map_ax, map_ax)
    # np.random.seed(0)

    # base_map = np.random.random_sample(size=(map_size, map_size))*0.1
    # # hills = [x, y], height, width
    # hill_attrs =  [
    #                 [[5,2], 4, 2],
    #                 [[3,6], 7, 3],
    #                 # [[8,7], 7.5, 3],
    #               ]
    # heightmap = base_map + make_landscape(map_size, hill_attrs)
    # heightmap *= 10

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, heightmap, cmap='viridis', edgecolor='none')
    # plt.show()