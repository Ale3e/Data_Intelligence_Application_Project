from Learners.UCB1_Learner import *

class SWUCB1_Learner(UCB1_Learner):
    def __init__(self, n_arms, window_size, prices):
        super().__init__(n_arms, prices)
        self.window_size = window_size
        self.selections_windowed = [0.0] * n_arms

    def pull_arm(self):

        pulled_arm = 0
        max_upper_bound = 0
        total_counts = 0
        bound_length = 0

        for arm in range(0, self.n_arms):
            # if the arm arm has been already pulled once

            if self.selections_windowed[arm] > 0:
                total_counts = self.t
                # if total_counts > 20:
                #    total_counts = 20

                # after I reach the window size, I won't do more than 20 selections, so I can fix this number.
                bound_length = math.sqrt(0.01*math.log(self.t) / float(self.selections_windowed[arm]))
                upper_bound = self.empirical_mean[arm] + bound_length

            else:
                upper_bound = 1e100  # this happens just when arms haven't been pulled before, so
                # we give them an very large (inf) upper bounds

            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                pulled_arm = arm

        return pulled_arm

    def update(self, pulled_arm, reward):

        self.t += 1
        self.update_observations(pulled_arm, reward)

        self.numbers_of_selections[pulled_arm] = self.numbers_of_selections[pulled_arm] + 1

        self.pulled_arms = self.pulled_arms.astype(int)
        temp = np.bincount(self.pulled_arms[-self.window_size:], minlength=self.n_arms)

        self.selections_windowed = temp

        num_selections_pulled_arm = self.selections_windowed[pulled_arm]

        # overall is the sum of the last 'num_selections_pulled_arm' rewards
        # where 'num_selections_pulled_arm' is the number of times the arm has been pulled in the last window-size pulls
        overall = np.sum(self.rewards_per_arm[pulled_arm][-num_selections_pulled_arm:])
        size = len(self.rewards_per_arm[pulled_arm][-num_selections_pulled_arm:])

        #update the empirical mean taking into account only the last N rewards where N = self.window_size
        self.empirical_mean[pulled_arm] = overall/size

