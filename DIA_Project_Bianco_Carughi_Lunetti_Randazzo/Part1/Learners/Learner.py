import numpy as np


class Learner:

    def __init__(self, n_arms,prices):
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = x = [[] for i in range(n_arms)]
        self.collected_rewards = np.array([])
        self.pulled_arms = np.array([])
        self.prices = prices


    def update_observations(self, pulled_arm, reward):

        self.rewards_per_arm[pulled_arm].append(reward)
        self.pulled_arms = np.append(self.pulled_arms, float(pulled_arm))
        self.collected_rewards = np.append(self.collected_rewards, reward*self.prices[pulled_arm])
