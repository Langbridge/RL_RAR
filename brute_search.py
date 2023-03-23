import numpy as np
import networkx as nx
import random

from aec_env import AsyncMapEnv
from cyclist import Cyclist

from copy import deepcopy
import heapq
import itertools

from pprint import pprint

env_config = {
    'num_agents': 5,
    'map_size': 4,
    'num_iters': 500,
    'congestion': True,
}

class Node:
    def __init__(self, id, pollution, cyclist, vel=0):
        self.id = id
        self.children = []
        self.history = []
        self.pollution = pollution
        self.cyclist = cyclist
        self.vel = vel

class SearchTree:
    def __init__(self, env):
        self.routes = {}
        self.env = env

        self.pollutions = {agent: 0 for agent in self.env.possible_agents}

    def build_tree(self, velocities=[0,1], verbose=0):
        if type(velocities) != list:
            self.default_vel = velocities
            velocities = [-1]
        for agent in self.env.possible_agents:
            start = self.env.positions[agent]
            goal = self.env.goals[agent]
            if verbose: print(f"Building tree for agent {agent} from {start} to {goal}")
            cyclist = self.env.agent_name_mapping[agent]

            # add initial node
            root = Node(start, 0, cyclist, None)

            # build priority queue
            pq = []
            pos = root
            while pos.id != goal:
                for (x, y) in self.env.G.edges(pos.id):
                    if y in [k[0] for k in pos.history]:
                        continue
                    for act_v in velocities:
                        child = self.insert(pos, y, act_v)
                        heapq.heappush(pq, [child.pollution, child.cyclist.hr, child.id, act_v, np.random.random(), child])
                
                pol, hr, id, act_v, _, pos = heapq.heappop(pq)
                # print("Traversed to child", id, "from", pos.history[-1], "at", self.env._act_to_vel(act_v, pos.cyclist.fitness), "with", pol)

            path = deepcopy(pos.history)
            path.append((pos.id, pos.vel))
            self.routes[agent] = [path, 1]
            self.pollutions[agent] = pol
            # print(f"{agent} optimal route {path} with pollution {pol}")

    def insert(self, parent: Node, child_id, vel):
        child_cyclist = deepcopy(parent.cyclist)

        dh = nx.get_edge_attributes(self.env.G, 'dh')[(parent.id, child_id)]
        length = nx.get_edge_attributes(self.env.G, 'l')[(parent.id, child_id)]
        amb_pol = nx.get_edge_attributes(self.env.G, 'pollution')[(parent.id, child_id)]

        if vel in {0,1}:
            power = child_cyclist.get_segment_power(d_height=dh, l=length, v=self.env._act_to_vel(vel, child_cyclist.fitness)/3.6)
            pollution = child_cyclist.eval_segment(amb_pol, power, length*3.6/self.env._act_to_vel(vel, child_cyclist.fitness))
        else:
            power = child_cyclist.get_segment_power(d_height=dh, l=length, v=self.default_vel/3.6)
            pollution = child_cyclist.eval_segment(amb_pol, power, length*3.6/self.default_vel)

        child = Node(child_id, parent.pollution+pollution, child_cyclist, vel)
        parent.children.append(child)
        child.history = deepcopy(parent.history)
        child.history.append((parent.id, parent.vel))

        # print("Child ", child.id, vel, child.history, child.pollution, child.cyclist.hr)
        return child


if __name__ == "__main__":
    env = AsyncMapEnv(**env_config)
    env.reset()
    pprint(env.agent_name_mapping)

    # for agent in env.possible_agents:
    #     env.positions[agent] = 0
    #     env.goals[agent] = 15

    # # create hill in centre of map & fix edge lengths
    # diag_edges = [(0,5), (5,10), (10,15), (3,6), (6,9), (9,12)]
    # diag_attr = {
    #         edge: {'pollution': max(0.1, random.normalvariate(5, 2.5)), 'l': max(0.1, random.normalvariate(200, 50))}
    #         for edge in diag_edges
    #     }
    # env.G.add_edges_from([(e[0], e[1], diag_attr[(e[0], e[1])]) for e in diag_edges]) # add diagonal edges
    # hill_nodes = [5,6,9,10]
    # hill_edges = list(itertools.combinations(hill_nodes, 2))
    # hill_edges += hill_edges[::-1]
    # nx.set_node_attributes(env.G, {n: nx.get_node_attributes(env.G, 'h')[n]+10 for n in hill_nodes}, 'h')
    # nx.set_edge_attributes(env.G, {e: 200 for e in env.G.edges()}, 'l')
    # nx.set_edge_attributes(env.G, {e: nx.get_edge_attributes(env.G, 'pollution')[e]+10 for e in hill_edges}, 'pollution')
    # nx.set_edge_attributes(env.G, {e: env.G.nodes[e[1]]['h'] - env.G.nodes[e[0]]['h'] for e in env.G.edges()}, 'dh')

    search = SearchTree(env)
    search.build_tree()
    pprint(search.routes)

    while env.agents:
        curr_agent = env.agent_selection
        curr_cyclist = deepcopy(env.agent_name_mapping[curr_agent])
        dest, vel = search.routes[curr_agent][0][search.routes[curr_agent][1]]
        # dh = nx.get_edge_attributes(env.G, 'dh')[(env.positions[curr_agent], dest)]
        # length = nx.get_edge_attributes(env.G, 'l')[(env.positions[curr_agent], dest)]
        # amb_pol = nx.get_edge_attributes(env.G, 'pollution')[(env.positions[curr_agent], dest)]
        # power = curr_cyclist.get_segment_power(d_height=(dh), l=length, v=env._act_to_vel(vel, curr_cyclist.fitness)/3.6)
        # pollution = curr_cyclist.eval_segment(amb_pol, power, length*3.6/env._act_to_vel(vel, curr_cyclist.fitness))

        # print({'destination': dest, 'velocity': env._act_to_vel(vel, curr_cyclist.fitness)}, pollution, curr_cyclist.hr)
        env.step({'destination': dest, 'velocity': vel})
        # print(f"{curr_agent}\t{env.agent_name_mapping[curr_agent].hr:.2f}\t{pollution:.4f}\t{env.pollution[curr_agent]:.4f}")

        search.routes[curr_agent][1] += 1

    pprint(search.pollutions)
    pprint(env.pollution)
