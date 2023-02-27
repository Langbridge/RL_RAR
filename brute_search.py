import numpy as np
import networkx as nx

from aec_env import AsyncMapEnv
from copy import deepcopy
import heapq

env_config = {
    'num_agents': 1,
    'map_size': 4,
    'num_iters': 500,
}
env = AsyncMapEnv(**env_config)
env.reset()

class Node:
    def __init__(self, id, pollution, cyclist):
        self.id = id
        self.children = []
        self.history = []
        self.pollution = pollution
        self.cyclist = cyclist
        self.height = nx.get_node_attributes(env.G, 'h')[self.id]

def insert(parent: Node, child_id, vel):
    child_cyclist = deepcopy(parent.cyclist)

    child_height = nx.get_node_attributes(env.G, 'h')[child_id]
    length = nx.get_edge_attributes(env.G, 'l')[(parent.id, child_id)]
    amb_pol = nx.get_edge_attributes(env.G, 'pollution')[(parent.id, child_id)]

    power = child_cyclist.get_segment_power(d_height=(parent.height-child_height), l=length, v=vel)
    pollution = child_cyclist.eval_segment(amb_pol, power, length/vel)

    child = Node(child_id, parent.pollution+pollution, child_cyclist)
    parent.children.append(child)
    child.history = deepcopy(parent.history)
    child.history.append(parent.id)

    print("Child ", child.id, child.history, child.pollution, child.cyclist.hr, child.cyclist.power_history[0])
    return child

for agent in env.possible_agents:
    start = env.positions[agent]
    goal = env.goals[agent]
    cyclist = env.agent_name_mapping[agent]
    print("Start", start, "goal", goal)

    # add initial node
    root = Node(start, 0, cyclist)
    print("Root ", root.id, root.history, root.pollution, root.cyclist.hr, "\n")

    # build priority queue
    pq = []
    pos = root
    while pos.id != goal:
        for (x, y) in env.G.edges(pos.id):
            if y in pos.history:
                print("Node", y, "already visited")
                continue
            child = insert(pos, y, 20/3.6)
            heapq.heappush(pq, [child.pollution, child.cyclist.hr, child.id, child])
        
        pol, hr, id, pos = heapq.heappop(pq)
        print("\nTraversed to child", id, "from", pos.history[-1])

    path = deepcopy(pos.history)
    path.append(pos.id)
    print("Optimal path:", path, pos.pollution)

    # test with env
    for step in path[1:]:
        env.step({'destination': step, 'velocity': 3})
    print(env._cumulative_rewards, env.pollution)
