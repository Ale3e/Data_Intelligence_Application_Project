import numpy as np


class LinearEnviroment:

    def __init__(self, n_arms, dim):                                    #dim of features vector
        self.theta = np.random.dirichlet(np.ones(dim), size=1)          #dim = n. of probabilities that sum up to 1
        self.arms_features = np.random.binomial(1, 0.5, size=(n_arms, dim))
        self.p = np.zeros(n_arms)
        for armIDX in range(0, n_arms):                                 #assign the prob of each arm as
            self.p[armIDX] = np.dot(self.theta, self.arms_features[i])

    def round(self, pulled_arm):
        return 1 if np.random.random() < self.p[pulled_arm] else 0

    def opt(self):
        return np.max(self.p)







arms_feats = np.random.binomial(1, 0.5, size=(2, 4))
print(arms_feats)
dirich = np.random.dirichlet(np.ones(4), size=1)
print(dirich)
probs = np.zeros(2)
for i in range(0, 2):
   probs[i] = np.dot(dirich, arms_feats[i])

print(probs)