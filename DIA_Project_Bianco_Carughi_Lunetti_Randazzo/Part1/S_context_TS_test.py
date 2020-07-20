import numpy as np
import matplotlib.pyplot as plt
from Environments.Environment import *
from Learners.TS_Learner import TS_Learner
import pandas as pd
import math


np.random.seed(10)
n_arms = 5
#p = np.array([0.15, 0.1, 0.1, 0.35])
#p = np.array([0.73, 0.61, 0.5, 0.4])






p = np.array([[0.298, 0.210, 0.135, 0.063, 0.014],[0.251, 0.151, 0.118, 0.054, 0.009],[0.183, 0.102, 0.090, 0.053, 0.004]])
prices = np.array([325, 350, 375,400,425])
opt = np.max(np.mean(p,axis=0)*prices)


T = 365

n_experiments = 10
ts_rewards_per_experiment = []
gr_rewards_per_experiment = []
z = pd.DataFrame(columns=['U30','Retired','Arm','Reward'])

arms = np.array([0,1,2,3,4])

def split_on_retired(z,delta=0.1):

    G = compute_profit(z,delta)

    z_l = z.loc[z['Retired'] == 0]
    z_r = z.loc[z['Retired'] == 1]
    G_l = compute_profit(z_l,delta= delta/4)
    G_r = compute_profit(z_r,delta= delta/4)

    return (G_r + G_l - G > 0)

def split_on_u30(z,delta=0.05):
    z_no = z.loc[z['Retired'] == 0]

    G = compute_profit(z_no,delta)

    z_u = z_no.loc[z['U30'] == 0]
    z_o = z_no.loc[z['U30'] == 1]
    G_l = compute_profit(z_u, delta=delta / 4)
    G_r = compute_profit(z_o, delta=delta / 4)
    return (G_r + G_l - G > 0)

def compute_profit(z,delta):
    profit_per_arm = []
    for arm in arms:
        n = z.shape[0]
        z1 = z.loc[z['Arm'] == arm]

        if z1.shape[0] == 0:
            profit_per_arm.append(-1e400)
            continue
        p = z.shape[0] / n
        p_confidence_bound = math.sqrt(-(math.log(delta/2)/(2*n)))
        p_lower_bound = p - p_confidence_bound

        arm_reward = z1.sum(axis=0)['Reward']
        rew_confidence_bound = math.sqrt(-(math.log(delta/2)/(2*n)))

        rew_lower_bound = (arm_reward / z1.shape[0]) - rew_confidence_bound
        G = p_lower_bound * rew_lower_bound
        profit_per_arm.append(G)

    #print(np.argmax(profit_per_arm))
    return np.max(profit_per_arm)

def get_learner(retired,u30,learners,split_retired=False,split_u30=False):
    if (split_u30):
        if(u30 == 0):
            return learners[3]
        else: return learners[4]
    if (split_retired):
        if(retired == 0):
            return learners[1]
        else:return learners[2]
    else:return learners[0]


def generate_reward(retired,u30,probabilities,pulled_arm):
    if (retired == 0 and u30 == 1):
        reward = np.random.binomial(1, probabilities[0][pulled_arm])
        return reward
    if(retired == 0 and u30 == 0):
        reward = np.random.binomial(1, probabilities[1][pulled_arm])
        return reward
    else:
        reward = np.random.binomial(1, probabilities[2][pulled_arm])
        return reward




for e in range(0,n_experiments):
    learners = []
    z = pd.DataFrame(columns=['U30', 'Retired', 'Arm', 'Reward'])
    print('Experiment {}'.format(e))
    env = Environment(n_arms=n_arms, probabilities=p)
    ts_learner = TS_Learner(n_arms=n_arms,prices=prices)
    gr_learner = TS_Learner(n_arms=n_arms,prices=prices)
    learners.append(ts_learner)
    check_split_on_retired = True
    check_split_on_u30 = False
    split_retired = False
    split_u30 = False
    for i in range(0,T):
        u30 = np.random.binomial(1, 0.5)
        retired = np.random.binomial(1, 0.5)
        ts_learner = get_learner(retired,u30,learners,split_retired,split_u30)
        if ((i % 7) == 0 and i != 0):
            if (check_split_on_retired):
                split_retired = split_on_retired(z)
                if (split_retired):
                    learner_yes = ts_learner
                    learner_no = ts_learner
                    learners.append(learner_yes)
                    learners.append(learner_no)
                    check_split_on_retired = False
                    check_split_on_u30 = True


            if(check_split_on_u30):
                split_u30 = split_on_u30(z)
                if (split_u30):
                    learner_under_30 = ts_learner
                    learner_over_30 = ts_learner
                    learners.append(learner_under_30)
                    learners.append(learner_over_30)
                    check_split_on_u30 = False
        #Thomposon Sampling Learner
        pulled_arm = ts_learner.pull_arm()
        #reward = env.round(pulled_arm)
        reward = generate_reward(retired,u30,p,pulled_arm)
        ts_learner.update(pulled_arm,reward)

        z = z.append({'U30' : u30, 'Retired': retired,'Arm':pulled_arm,'Reward':reward},ignore_index=True)

        #Greedy Learner
        pulled_arm = gr_learner.pull_arm()
        #reward = env.round(pulled_arm)
        reward = generate_reward(retired,u30,p,pulled_arm)
        gr_learner.update(pulled_arm,reward)

    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    gr_rewards_per_experiment.append(gr_learner.collected_rewards)


#G = generate_context(z)
#print(G)

plt.figure(0)
plt.xlabel('t')
plt.ylabel('Reward')
plt.plot(np.mean(ts_rewards_per_experiment, axis=0), 'r')
plt.plot(np.mean(gr_rewards_per_experiment, axis=0), 'b')
plt.plot(T * [opt], '--k')
#plt.plot(opt_per_round, '--k')
plt.legend(['Contextual-UCB1','UCB1','Optimum'])
plt.show()


plt.figure(0)
plt.xlabel('T')
plt.ylabel('Regret')
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment,axis=0)),'r')
plt.plot(np.cumsum(np.mean(opt - gr_rewards_per_experiment,axis=0)),'b')
plt.legend(['Contextual-UCB1','UCB1'])
plt.show()
