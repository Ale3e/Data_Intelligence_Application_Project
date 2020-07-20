import numpy as np
import matplotlib.pyplot as plt
from Environments.Non_Stationary_Environment import *
from Learners.SWUCB1_Learner import SWUCB1_Learner
from Learners.UCB1_Learner import UCB1_Learner

np.random.seed(10)
n_arms = 5

p = np.array([[0.312, 0.224, 0.140, 0.068, 0.016], [0.312, 0.224, 0.140, 0.068, 0.016], [0.245, 0.157, 0.112, 0.050, 0.007], [0.262, 0.166, 0.123, 0.055, 0.011]])


T = 365
n_experiments = 10

swucb_reward_per_experiment = []
ucb_reward_per_experiment = []
window_size = int(np.sqrt(n_experiments)*n_arms)
prices = np.array([325, 350, 375, 400, 425])


for e in range(0, n_experiments):
    print(e)
    ucb_env = Non_Stationary_Environment(n_arms=n_arms, probabilities=p, horizon=T)
    ucb_learner = UCB1_Learner(n_arms=n_arms, prices=prices)

    swucb_env = Non_Stationary_Environment(n_arms=n_arms, probabilities=p, horizon=T)
    swucb_learner = SWUCB1_Learner(n_arms=n_arms, window_size=window_size, prices=prices)

    for t in range(0,T):
        pulled_arm = ucb_learner.pull_arm()
        reward = ucb_env.round(pulled_arm)
        ucb_learner.update(pulled_arm, reward)

        pulled_arm = swucb_learner.pull_arm()
        reward = swucb_env.round(pulled_arm)
        swucb_learner.update(pulled_arm, reward)

    ucb_reward_per_experiment.append(ucb_learner.collected_rewards)
    swucb_reward_per_experiment.append(swucb_learner.collected_rewards)

ucb_instantaneus_regret = np.zeros(T)
swucb_instantaneus_regret = np.zeros(T)

n_phases = len(p)
print(n_phases)
phases_len = int(T/n_phases)
opt_per_phases = p.max(axis=1)*prices[0]
print(opt_per_phases)
opt_per_round = np.zeros(T)

for i in range(0, n_phases):
    opt_per_round[i*phases_len: (i+1)*phases_len+1] = opt_per_phases[i]
    ucb_instantaneus_regret[i*phases_len: (i+1)*phases_len] = opt_per_phases[i] - np.mean(ucb_reward_per_experiment, axis=0)[i*phases_len : (i+1)*phases_len]
    swucb_instantaneus_regret[i*phases_len: (i+1)*phases_len] = opt_per_phases[i] - np.mean(swucb_reward_per_experiment, axis=0)[i*phases_len : (i+1)*phases_len]

#In the first figure we show the reward
plt.figure(0)
plt.xlabel('t')
plt.ylabel('Reward')
plt.plot(np.mean(ucb_reward_per_experiment, axis=0), 'r')
plt.plot(np.mean(swucb_reward_per_experiment, axis=0), 'b')
plt.plot(opt_per_round, '--k')
plt.legend(['UCB1', 'SW-UCB1', 'Optimum'])
plt.show()

#In the second plot we show the regret
plt.figure(1)
plt.xlabel('t')
plt.ylabel('Regret')
plt.plot(np.cumsum(ucb_instantaneus_regret), 'r')
plt.plot(np.cumsum(swucb_instantaneus_regret), 'b')
plt.legend(['UCB1', 'SW-UCB1'])
plt.show()
