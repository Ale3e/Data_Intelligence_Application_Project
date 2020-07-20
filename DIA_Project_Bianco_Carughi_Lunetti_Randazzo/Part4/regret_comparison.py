from TS_learner import *
from LUCB_example import *
from CUCB1_example import *

if __name__ == "__main__":
    n_features = 4

    graph = generate_graph(100, 5, 0.1, 1234)
    graph = weight_edges(graph, n_features)
    graph = weight_nodes(graph)

    budget = 7.5
    delta = 0.5


    # optimal with greedy_celf#
    greedy_N_simulations = 1000

    start_time = time.time()
    greedy = []
    greedy = greedy_celf(graph, budget, delta)
    opt_seeds = sorted(greedy[1])
    spread_cumulative = []

    for n in range(greedy_N_simulations):
        IC = information_cascade(graph, opt_seeds)[0]
        spread_cumulative.append(IC)

    opt_spread = np.mean(spread_cumulative)

    print('Time for optimal greedy simulation: {} '.format(time.time() - start_time))
    print('Seeds: {}'.format(sorted(opt_seeds)))
    print('Optimal spread: {} \n'.format(round(float(opt_spread), 3)))

    # LinearUCB_Learner

    spreads_LUCB = []
    cumulative_spreads_LUCB = []
    true_probs = get_probabilities(graph)
    env = Environment(graph)

    coffiecient_c = 2
    lucb_learner = LUCBLearner(graph, budget, n_features, coffiecient_c)

    N_mc_simulations = 100
    T = 500

    for t in tqdm.tqdm(range(T)):
        start_time = time.time()
        super_arm = lucb_learner.pull_superarm()
        reward = env.round(super_arm)
        lucb_learner.update(reward)

        estimated_seeds = greedy_celf(lucb_learner.graph, budget)[1]

        for n in range(N_mc_simulations):
            IC = information_cascade(graph, estimated_seeds)[0]
            cumulative_spreads_LUCB.append(IC)
        means_spread = np.mean(cumulative_spreads_LUCB)
        means_spread = round(means_spread, 3)
        spreads_LUCB.append(means_spread)
        print('Spread: {}'.format(means_spread))
        print('Time for iteration {} : {}'.format(t, time.time() - start_time))

    print('Opt-spread: {}'.format(opt_spread))
    print('Spreads: {}'.format(spreads_LUCB))
    regret_LUCB = np.abs(opt_spread - spreads_LUCB)

    # TS_Learner

    spreads_TS = []
    cumulative_spreads_TS = []
    ts_learner = TSLearner(graph, budget)


    for t in tqdm.tqdm(range(T)):
        start_time = time.time()
        super_arm = ts_learner.pull_superarm()
        reward = env.round(super_arm)
        ts_learner.update(reward)

        estimated_seeds = greedy_celf(ts_learner.graph, budget)[1]

        for n in range(N_mc_simulations):
            IC = information_cascade(graph, estimated_seeds)[0]
            cumulative_spreads_TS.append(IC)
        means_spread = np.mean(cumulative_spreads_TS)
        means_spread = round(means_spread, 3)
        spreads_TS.append(means_spread)
        print('Spread: {}'.format(means_spread))
        print('Time for iteration {} : {}'.format(t, time.time() - start_time))

    print('Opt-spread: {}'.format(opt_spread))
    print('Spreads: {}'.format(spreads_TS))
    regret_TS = np.abs(opt_spread - spreads_TS)

    # CUCB1_Learner

    spreads_CUCB1 = []
    cumulative_spreads_CUCB1 = []
    ucb_learner = UCBLearner(graph, budget)

    for t in tqdm.tqdm(range(T)):
        start_time = time.time()
        super_arm = ucb_learner.pull_superarm()
        reward = env.round(super_arm)
        ucb_learner.update(reward)

        estimated_seeds = greedy_celf(ucb_learner.graph, budget)[1]

        for n in range(N_mc_simulations):
            IC = information_cascade(graph, estimated_seeds)[0]
            cumulative_spreads_CUCB1.append(IC)
        means_spread = np.mean(cumulative_spreads_CUCB1)
        means_spread = round(means_spread, 3)
        spreads_CUCB1.append(means_spread)
        print('Spread: {}'.format(means_spread))
        print('Time for iteration {} : {}'.format(t, time.time() - start_time))

    print('Opt-spread: {}'.format(opt_spread))
    print('Spreads: {}'.format(spreads_CUCB1))
    regret_CUCB1 = np.abs(opt_spread - spreads_CUCB1)

    ### PLOT ###

    plt.style.use('seaborn')  # pretty matplotlib plots
    plt.rcParams['figure.figsize'] = (12, 8)
    opt_spreads = []
    for t in range(T): opt_spreads.append(opt_spread)

    plt.plot(spreads_CUCB1, color='blue', label='CUCB1')
    plt.plot(spreads_TS, color='green', label='TS')
    plt.plot(spreads_LUCB, color='orange', label='LinUCB')
    plt.plot(opt_spreads, color='red', label='opt')
    plt.xlabel('t')
    plt.ylabel('Spread')
    plt.title('Reward comparison')
    plt.legend()
    plt.show()

    plt.plot(np.cumsum(regret_CUCB1), color='blue', label='CUCB1')
    plt.plot(np.cumsum(regret_TS), color='green', label='TS')
    plt.plot(np.cumsum(regret_LUCB), color='orange', label='LinUCB')
    plt.xlabel('t')
    plt.ylabel('Regret')
    plt.title('Regret Comparison')
    plt.legend()
    plt.show()

    print('True probabilities: {}'.format(true_probs))
    print('Estimated probabilities: {}'.format(list(ucb_learner.get_estimated_probabilities().values())))
    print('Estimated probabilities: {}'.format(list(lucb_learner.get_estimated_probabilities().values())))
    print('Estimated probabilities: {}'.format(list(ts_learner.get_estimated_probabilities().values())))
