from Learner import *

class Greedy_Learner(Learner):

    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.expected_rewards = np.zeros(n_arms)

    def pull_arm(self):
        # To ensure that each arm is pulled once so we add these two lines
        if (self.t < self.n_arms):
            return self.t
        # we select the index of the arm with the maximum expected reward
        # we use argwhere instead of argmax because there could be more max values
        idxs = np.argwhere(self.expected_rewards == self.expected_rewards.max()).reshape(-1)
        # since idxs could contain more than one index we select one randomly
        pulled_arm = np.random.choice(idxs)
        return pulled_arm

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        # update the reward making a simple average incrementally
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm]*(self.t-1) + reward) / self.t