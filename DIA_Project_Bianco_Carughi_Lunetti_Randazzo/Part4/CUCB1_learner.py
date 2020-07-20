from graph import generate_graph, weight_nodes, weight_edges, get_probabilities
from greedy import greedy_celf
from enviroment import *
from information_cascade import *
import numpy as np
import time
import networkx as nx
import matplotlib.pyplot as plt
import copy


class UCBLearner:
    def __init__(self, Graph, budget):
        self.graph = copy.deepcopy(Graph)
        self.empirical_mean = dict.fromkeys(self.graph.edges, 1)
        self.empirical_mean_no_bound = dict.fromkeys(self.graph.edges, 1)
        self.cumulative_reward = dict.fromkeys(self.graph.edges, 0)
        self.T = dict.fromkeys(self.graph.edges, 0)
        self.t = 0
        self.budget = budget

        for node1, node2 in self.graph.edges():
            self.graph[node1][node2]['prob'] = 1

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

        # print('reward: {}'.format(reward))
        # print('update round at t = {}'.format(self.t))
        # print('edges tested: {}'.format(reward.keys()))
        # print('T : {}'.format(self.T))

        # for u,v in reward.keys():
        #     print('for u = {} and v = {} '.format(u, v))
        #     print('T is {}'.format(self.T[(u, v)]))
        #     print('reward is {}'.format(reward[(u, v)]))

        for (u, v) in reward.keys():
            self.T[(u, v)] = self.T[(u, v)] + 1
            self.cumulative_reward[(u, v)] += reward[(u, v)]
            bound = np.sqrt((3 * np.log(self.t)) / (2 * self.T[(u, v)]))
            self.empirical_mean_no_bound[(u, v)] = min(self.cumulative_reward[(u, v)] / self.T[(u, v)], 1)
            self.empirical_mean[(u, v)] = min(self.cumulative_reward[(u, v)] / self.T[(u, v)] + bound, 1)

        # Update the probs of the graph with the updated prob estimes
        for u, v in self.graph.edges:
            self.graph[u][v]['prob'] = self.empirical_mean[(u, v)]
        return

    def get_estimated_probabilities(self):
        return self.empirical_mean_no_bound
