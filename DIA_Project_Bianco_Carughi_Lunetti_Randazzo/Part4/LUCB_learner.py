from greedy import greedy_celf
import numpy as np
import copy


class LUCBLearner:
    def __init__(self, Graph, budget, n_features, c):
        self.graph = copy.deepcopy(Graph)
        self.n_features = n_features
        self.M = np.identity(self.n_features)
        self.b = np.zeros(self.n_features)
        self.b = self.b.reshape(4, 1)
        self.c = c
        self.budget = budget
        self.t = 0

        self.theta = []
        self.theta = np.dot(np.linalg.inv(self.M), self.b)

        for node1, node2 in self.graph.edges():
            self.graph[node1][node2]['prob'] = np.dot(self.theta.T, self.graph[node1][node2]['features']).item()


    def pull_superarm(self):
        inv_M = np.linalg.inv(self.M)
        # print('M shape: {}'.format(inv_M.shape))
        # print('B shape: {}'.format(self.b.shape))
        self.theta = np.dot(inv_M, self.b)
        # print('Theta :{}'.format(self.theta.reshape(1, 4)))

        for edge in self.graph.edges:
            feature = self.graph[edge[0]][edge[1]]['features']
            ucb = np.clip(np.dot(self.theta.T, feature) + self.c * np.sqrt(
                np.dot(feature.T, np.dot(np.linalg.inv(self.M), feature))),
                          0, 1)
            self.graph[edge[0]][edge[1]]['prob'] = ucb

        superarm = set()
        seeds = []
        seeds = greedy_celf(self.graph, self.budget)[1]
        print('\nFor t = {} pulled arms from these seeds {}'.format(self.t, seeds))

        for seed in seeds:
            for u, v in self.graph.edges():
                if (u == seed): superarm.add((u, v))
                if (v == seed) and (u, v) not in superarm: superarm.add((u, v))
        return superarm

    def update(self, reward):
        self.t += 1

        for (u, v) in reward.keys():
            features = self.graph[u][v]['features']
            self.M = self.M + np.dot(features, features.T)
            # print('New M shape: {} '.format(self.M.shape))
            # print(self.M)
            if reward[(u, v)] == 1:
                self.b = self.b + features
        return


    def get_estimated_probabilities(self):
        estimated_prob = dict.fromkeys(self.graph.edges, 1)
        for node1, node2 in self.graph.edges():
            prob_linucb = np.dot(self.theta.T, self.graph[node1][node2]['features'])
            estimated_prob[(node1, node2)] = prob_linucb
        return estimated_prob
