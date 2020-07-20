import matplotlib.pyplot as plt
from Environments.Environment import *
from Learners.TS_Learner import TS_Learner
from Learners.UCB1_Learner import UCB1_Learner

np.random.seed(10)

n_arms = 5
p = np.array([0.293, 0.383, 0.229, 0.161,0.090])
prices = np.array([325, 350, 375,400,425])
opt = p[1]*prices[1]
T = 365
n_experiments = 10

ts_rewards_per_experiment = []
ucb_rewards_per_experiment = []

for e in range(0, n_experiments):
    print(e*100/10000)
    env = Environment(n_arms=n_arms, probabilities=p)
    ts_learner = TS_Learner(n_arms=n_arms, prices=prices)
    ucb_learner = UCB1_Learner(n_arms=n_arms, prices=prices)

    for i in range(0, T):

            #Thomposon Sampling Learner
            pulled_arm = ts_learner.pull_arm()
            reward = env.round(pulled_arm)
            ts_learner.update(pulled_arm, reward)

            #UCB1 Learner
            pulled_arm = ucb_learner.pull_arm()
            reward = env.round(pulled_arm)
            ucb_learner.update(pulled_arm, reward)
    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    ucb_rewards_per_experiment.append(ucb_learner.collected_rewards)

plt.figure(0)
plt.xlabel('T')
plt.ylabel('Regret')
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)),'r')
plt.plot(np.cumsum(np.mean(opt - ucb_rewards_per_experiment, axis=0)),'b')
plt.legend(['TS', 'UCB-1'])
plt.show()


print(np.mean(ts_rewards_per_experiment))
print(np.mean(ucb_rewards_per_experiment))
plt.figure(0)
plt.xlabel('t')
plt.ylabel('Reward')
plt.plot(np.mean(ts_rewards_per_experiment, axis=0), 'r')
plt.plot(np.mean(ucb_rewards_per_experiment, axis=0), 'b')
plt.plot(T * [opt], '--k')
plt.legend(['TS', 'UCB-1', 'optimum'])
plt.show()
