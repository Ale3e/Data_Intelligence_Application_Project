from graph import generate_graph, weight_nodes, weight_edges
from information_cascade import information_cascade
import matplotlib.pyplot as plt
import numpy as np
import time

'''
Problemi:
1. levare dai seeds l'ultimo nodo che viene aggiunto quando non potremmo me pare na stronzata
2. forse la differenza tra i nodi del grafo è troppo piccola e per questo non da differenze significative
3. nella formula per il calcolo delle simulazioni non è detto che solo il delta sia da modificare
'''


def greedy_celf(graph, budget, delta=0.95):
    """
    Cost efficient lazy forward algorithm, by Leskovec et al. (2007)
    Input:  graph object, number of seed nodes
    Output: a[0] remaining budget, a[1] optimal seed set
    """

    seeds = []
    remaining_budget = budget
    epsilon = 0.1
    delta = delta

    # evaluate each node in the graph for marginal increase in greedy algorithm
    marginal_gain = dict.fromkeys(graph.nodes, 0)
    nodes_left_to_evaluate = set(marginal_gain.keys())

    # all_node_weight = sum(set([graph.nodes[g]['cost'] for g in graph.nodes]))

    tot_weight = 0.0
    for node_idx in range(graph.number_of_nodes()):
        weight = []
        # print("\nNode : {} with cost {}".format(graph.nodes[node_idx], graph.nodes[node_idx]['cost']))
        tot_weight += graph.nodes[node_idx]['cost']

    # print('Total graph cost= {}'.format(tot_weight))

    if budget >= tot_weight:
        print('Error: budget too high, you can buy all nodes')
        return 0

    while remaining_budget > 0 and nodes_left_to_evaluate and min(
            set([graph.nodes[g]['cost'] for g in graph.nodes])) <= remaining_budget:

        for n in nodes_left_to_evaluate:  # aggiorno marginal_gain per ogni nodo [n]

            if remaining_budget >= graph.nodes[n]['cost']:  # calcola solo lo spread dei nodi che posso permettermi

                cost = graph.nodes[n]['cost']
                n_simulations = int((1 / (epsilon ** 2)) * np.log(len(seeds + [n]) + 1) * np.log(1 / delta))
                IC_cumulative = []

                for simulation in range(n_simulations):
                    # [0]returns the spread of the influenced nodes in the cascade
                    IC_result = information_cascade(graph, seeds + [n])[0]
                    IC_cumulative.append(IC_result)

                spread_node = round(np.mean(IC_cumulative), 3)
                marginal_gain[n] = spread_node
            else:  # metti spread a zero per i nodi che non possono permettermi
                marginal_gain[n] = 0.0

        gain_max = max(marginal_gain.values())

        # prendi best nodo da aggiungere ai seeds
        index_max = list(marginal_gain.keys())[list(marginal_gain.values()).index(gain_max)]
        remaining_budget -= graph.nodes[index_max]['cost']
        remaining_budget = round(remaining_budget, 3)
        seeds.append(index_max)

        # togliere nodo aggiunto ai seed dalla lista @node_left_to_evaluate
        nodes_left_to_evaluate.remove(index_max)
        marginal_gain.pop(index_max)

    # cost = graph.nodes[seeds[len(seeds) - 1]]['cost']
    # remaining_budget += cost
    # remaining_budget = round(remaining_budget, 3)
    # seeds.pop()

    return remaining_budget, seeds


if __name__ == "__main__":

    # features = [0.1, 0.08, 0.05, 0.02]
    n_features = 4

    graph = generate_graph(1000, 5, 0.1, 1234)
    graph = weight_edges(graph, n_features)
    graph = weight_nodes(graph)

    budget = 5
    delta = [0.95, 0.8, 0.4, 0.2]
    N_simulations = 1000
    spreads = []

    all_node_weight = sum(set([graph.nodes[g]['cost'] for g in graph.nodes]))
    print('Weight of all nodes in the graph: {}'.format(round(all_node_weight, 3)))

    for d in delta:

        start_time = time.time()
        greedy = []
        greedy = greedy_celf(graph, budget, delta=d)
        remaining_budget = greedy[0]
        seeds = sorted(greedy[1])
        spread_cumulative = []
        print('Simulation with delta : {}'.format(d))

        for n in range(N_simulations):
            IC = information_cascade(graph, seeds)[0]
            spread_cumulative.append(IC)

        spread = np.mean(spread_cumulative)
        spreads.append(spread)
        print('Seeds: {}'.format(sorted(seeds)))
        print('Remaining budget: {}'.format(round(remaining_budget, 3)))
        print('Spread: {}'.format(round(float(spread), 3)))
        print('Time for simulation: {} \n'.format(time.time() - start_time))

    plt.plot(delta, spreads)
    plt.show()
