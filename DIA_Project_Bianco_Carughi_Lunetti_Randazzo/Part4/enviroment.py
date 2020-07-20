from graph import generate_graph, weight_nodes, weight_edges, get_probabilities
import numpy as np

class Environment:
    def __init__(self, graph):
        self.graph = graph
        self.probabilities = get_probabilities(graph)

    def round(self, pulled_arm):
        rewards = dict.fromkeys(pulled_arm, 0)
        for (u, v) in rewards:
            rewards[(u, v)] = np.random.binomial(1, self.graph[u][v]['prob'])
        return rewards



