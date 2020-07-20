from graph import generate_graph, weight_nodes, weight_edges, get_probabilities
from greedy import greedy_celf
from enviroment import *
from information_cascade import *
import numpy as np
import time
import networkx as nx
import matplotlib.pyplot as plt
import copy


class TSLearner:
    def __init__(self, Graph, budget):
        self.graph = copy.deepcopy(Graph)
        self.alpha = dict.fromkeys(self.graph.edges, 1)
        self.beta = dict.fromkeys(self.graph.edges, 1)
        self.t = 0
        self.budget = budget

        for node1, node2 in self.graph.edges():
            prob_ts = round(np.random.beta(self.alpha[(node1, node2)], self.beta[(node1, node2)]), 3)
            self.graph[node1][node2]['prob'] = prob_ts

    def pull_superarm(self):
        '''
        get all edges from seeds as superarm
        '''

        superarm = set()
        seeds = []
        seeds = greedy_celf(self.graph, self.budget)[1]
        print('\nFor t = {} pulled arms from these seeds {}'.format(self.t, seeds))

        for seed in seeds:
            for u, v in self.graph.edges():
                if (u == seed): superarm.add((u, v))
                if (v == seed) and (u, v) not in superarm: superarm.add((u, v))
        #print('superarm: {}'.format(superarm))
        return superarm

    def update(self, reward):

        self.t += 1
        for (u, v) in reward.keys():
            if reward[(u, v)] == 1:
                self.alpha[(u, v)] += 1
            else: self.beta[(u, v)] += 1


        # Update the probs of the graph with the updated prob estimes
        for u, v in self.graph.edges:
            self.graph[u][v]['prob'] = round(np.random.beta(self.alpha[(u, v)], self.beta[(u, v)]), 3)
        return

    def get_estimated_probabilities(self):
        estimated_prob = dict.fromkeys(self.graph.edges, 1)
        for node1, node2 in self.graph.edges():
            prob_ts = round(np.random.beta(self.alpha[(node1, node2)], self.beta[(node1, node2)]), 3)
            estimated_prob[(node1, node2)] = prob_ts
        return estimated_prob
