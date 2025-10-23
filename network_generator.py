import sys
import os
import math
from random import randint
from matplotlib import pyplot as plt
import networkx as nx
# import matplotlib.pyplot as plt
import random
from networkx.readwrite import json_graph

#  0 ≤ β ≤ 1 0\leq \beta \leq 1 and N ≫ K ≫ ln ⁡ N ≫ 1 {\displaystyle N\gg K\gg \ln N\gg 1}

# num_nodes = 1024
# k = 50
# num_nodes = 512
# k = 28
# num_nodes = 256
# k = 20
# num_nodes = 128
# k = 17
num_nodes = 128
# k = 15

watts_strogatz_prob = 0.03

erdos_renyi_prob = 0.5
internet_graph_seed = None  # optional


def add_edge_weights(graph):
    for e in graph.edges:
        w = randint(1, 20)
        graph.add_edge(e[0], e[1], weight=w)


def get_diameter(graph, weighted = True):
    if weighted:
        paths_for_diameter = nx.shortest_path_length(graph, weight='weight')
        ecc = nx.eccentricity(graph, sp=dict(paths_for_diameter))
        diameter = nx.diameter(graph, e=ecc, weight='weight')
    else:
        paths_for_diameter = nx.shortest_path_length(graph)
        ecc = nx.eccentricity(graph, sp=dict(paths_for_diameter))
        diameter = nx.diameter(graph, e=ecc, weight='weight')
    return diameter


# write to a file
def write_to_a_file(graph, param):
    # diameter = get_diameter(graph)
    diameter = nx.diameter(graph, weight='weight')
    print("Diameter of the graph yoo:", diameter)
    # exit()
    graph_name = './graphs_new/' + str(num_nodes) + str(param) + '_diameter' + str(diameter) + 'test.edgelist'
    nx.write_graphml(graph, graph_name)
    return graph_name


def build_random_graph():
    random_graph = nx.gnp_random_graph(num_nodes, erdos_renyi_prob)
    print(nx.is_connected(random_graph))
    # nx.draw(random_graph, with_labels=True)
    # plt.show()

    if nx.is_connected(random_graph) is False:
        # print("NOT CONNECTED, NEED TO ADD EDGES")
        # exit()
        # at first connect nodes with 0 neighbors
        for i in range(0, num_nodes):
            print("HERE")
            node = None
            neighbors = []
            for a in [n for n in random_graph.neighbors(i)]:
                neighbors.append(a)
            if len(neighbors) == 0:
                node = i

                print("NODE ", node, " HAS 0 neighbours. CONNECTING IT TO THE LONGEST COMPONENT")
                longest_connected_comp = sorted(nx.connected_components(random_graph), key=len, reverse=True)[0]
                # a node in longest_connected_comp
                comp_node = next(iter(longest_connected_comp))
                random_graph.add_edge(node, comp_node, weight=randint(1, 20))

        #     attach smaller components to the longest component
        for i in list(reversed(range(1, len(sorted(nx.connected_components(random_graph), key=len, reverse=True))))):
            print("HERE1")
            source_node = next(iter(sorted(nx.connected_components(random_graph), key=len, reverse=True)[i]))
            dest_node = next(iter(sorted(nx.connected_components(random_graph), key=len, reverse=True)[0]))
            random_graph.add_edge(int(source_node), int(dest_node), weight=randint(1, 20))

    assert nx.is_connected(random_graph)
    add_edge_weights(random_graph)

    # print(nx.is_connected(random_graph))
    # nx.draw(random_graph, with_labels=True)
    # plt.show()

    return random_graph


def draw(graph):
    plt.axis('off')

    # some properties
    print("node degree")
    for v in nx.nodes(graph):
        print('%s %d' % (v, nx.degree(graph, v)))


    # print the adjacency list to terminal
    try:
        nx.write_adjlist(graph, sys.stdout)
    except TypeError:
        nx.write_adjlist(graph, sys.stdout.buffer)

    node_pos = nx.spring_layout(graph)

    edge_weight = nx.get_edge_attributes(graph, 'weight')
    # Draw the nodes
    nx.draw_networkx(graph, node_pos, node_color='grey', node_size=100)

    # Draw the edges
    nx.draw_networkx_edges(graph, node_pos, edge_color='black')

    # Draw the edge labels
    nx.draw_networkx_edge_labels(graph, node_pos, edge_labels=edge_weight)
    print(edge_weight)
    print(graph)
    print(graph.nodes())
    print("THE GRAPH")
    # plt.show()


def build_graphs():
    random_graph = build_random_graph()
    graph_name = write_to_a_file(random_graph, "random")
    # draw(random_graph)
    # nx.draw(random_graph, with_labels=True)
    # plt.show()
    return graph_name
      


if __name__ == '__main__':
    build_graphs()
