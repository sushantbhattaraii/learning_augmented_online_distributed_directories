import networkx as nx
import numpy as np
import os

# 1. Generate a random graph (Erdos-Renyi model)
# n=32 nodes, p=0.2 probability of edge creation
# G = nx.erdos_renyi_graph(n=32, p=0.2, seed=42)

def load_graph(network_file_name):
    graphml_file = os.path.join('graphs_new', str(network_file_name))
    G_example = nx.read_graphml(graphml_file)
    G_example = nx.relabel_nodes(G_example, lambda x: int(x))
    return G_example


G = load_graph("32random_diameter20test.edgelist")

# 2. Convert the graph to a NumPy adjacency matrix
adj_matrix = nx.to_numpy_array(G, dtype=int)

# 3. Print
np.set_printoptions(linewidth=200, edgeitems=32)
print("NetworkX Generated Adjacency Matrix:\n")
print(adj_matrix)