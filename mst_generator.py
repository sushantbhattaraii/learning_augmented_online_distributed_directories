import networkx as nx
import matplotlib.pyplot as plt
import sys
import os

def load_graph(network_file_name):
    graphml_file = os.path.join('graphs_new', str(network_file_name))
    G_example = nx.read_graphml(graphml_file)
    G_example = nx.relabel_nodes(G_example, lambda x: int(x))
    return G_example

def write_to_a_file(graph, param, num_nodes, diameter):
    graph_name = './graphs_new/mst/' + str(num_nodes) + str(param) + '_diameter' + str(diameter) + 'test.edgelist'
    nx.write_graphml(graph, graph_name)
    return graph_name


if __name__ == '__main__':
    network_file_name = "1024random_diameter4test.edgelist"
    num_nodes = 1024
    # network_file_name = "256random_diameter5test.edgelist"
    # network_file_name = "512random_diameter4test.edgelist"
    # network_file_name = "1024random_diameter4test.edgelist"
    G_example = load_graph(network_file_name)

    # Contrcut MST_g of Graph G_example for Arrow protocol
    mst_g = nx.minimum_spanning_tree(G_example, weight='weight')
    # see_graph(mst_g)
    diameter_of_mst_g = nx.diameter(mst_g, weight='weight')

    graph_name = write_to_a_file(mst_g, "random", num_nodes, diameter_of_mst_g)