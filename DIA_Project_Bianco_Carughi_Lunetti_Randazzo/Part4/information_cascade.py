import networkx as nx
import numpy as np
import tqdm
from graph import generate_graph, weight_nodes, weight_edges

#struttura nodi grafo
#graph.nodes() = {0: {'id': 0, 'cost': 0.6, 'status': 'susceptible'}, 1: {'id': 1, 'cost': 0.38, 'status': 'susceptible'}, 2: {'id'...

def information_cascade(graph, seed_set):
    """
    Simulates an information cascade in a graph
    :param graph: input graph of susceptible nodes
    :param seed_set: seed set from which propagates the information cascade
    :return: returns a list with a[0]: the sum of the costs of the nodes influenced during the IC
                            and a[1]: list of activated nodes during the IC
    """

    t = 0
    weighted_spread = 0.0
    triggered_nodes = []
    todo_nodes = []

    #todo_nodes.append(seed_set)
    for seed in seed_set:
        todo_nodes.append(seed)

    # activate seed nodes
    for i in range(len(todo_nodes)):
        graph.nodes[todo_nodes[i]]['status'] = 'active'

    # IC
    # for node in todo_nodes:
    while len(todo_nodes) > 0:
        node = todo_nodes[0]
        # print("At time: ")
        # print(t)
        # print(" from node ")
        # print(graph.nodes[node]['id'])

        triggered_nodes.append(graph.nodes[node]['id'])

        #print("information propagates to node: ")

        for adj_node in graph.neighbors(node):

            if graph.nodes[adj_node]['status'] == 'susceptible':

                if np.random.rand() <= graph[node][adj_node]['prob']:
                    #print(graph.nodes[adj_node]['id'])

                    graph.nodes[adj_node]['status'] = 'active'

                    todo_nodes.append(graph.nodes[adj_node]['id'])

        weighted_spread += graph.nodes[node]['cost']
        weighted_spread = round(weighted_spread, 2)
        graph.nodes[node]['status'] = 'inactive'
        todo_nodes.remove(node)
        t += 1

    # reset status of nodes of the graph to susceptible
    for n in graph.nodes():
        graph.nodes[n]['status'] = 'susceptible'

    a = [weighted_spread, triggered_nodes]
    return a


if __name__ == "__main__":

    features = [0.10, 0.08, 0.05, 0.02]

    graph = generate_graph(100, 5, 0.05, 123)
    graph = weight_edges(graph, features)
    graph = weight_nodes(graph)

    N_simulations = 100
    seed_set = [4, 8, 15, 16, 23, 42]
    spread_simulation = []

    # for u, v in graph.edges():
    #     print(u,v)
    #     for j,k in graph.edges():
    #         if (u,v) == (k,j):
    #             print("Oh no")
    #             break

    # a = []
    # a = information_cascade(graph, seed_set)
    #
    # weighted_spread = a[0]
    # triggered_nodes = a[1]
    #
    # print(" Weighted spread is : {}".format(weighted_spread))
    # print("Triggered nodes are: ")
    # for n in range(len(triggered_nodes)):
    #     node = triggered_nodes[n]
    #     cost = graph.nodes[node]['cost']
    #     print('Node: {} costs {}'.format(str(node), cost))


    means = []
    for j in tqdm.tqdm(range(10)):
        for n in range(N_simulations):
            spread_simulation.append(information_cascade(graph, seed_set)[0])
        mean_spread = np.mean(spread_simulation)
        mean_spread = round(mean_spread, 3)
        means.append(mean_spread)
    print(means)



    # seed_set_1 = []
    # for x in range(0, 10):
    #     seed_set_1.append(x)
    #     b = []
    #     b = information_cascade(graph, seed_set_1)
    #     print('With {} nodes the spread in {}'.format(x, b[0]))















