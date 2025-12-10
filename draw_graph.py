import networkx as nx
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import math
import pandas as pd

# Load the graph from a GraphML file
# graphml_file = '\graphs_new\16random_diameter21test.edgelist'
# graphml_file = os.path.join("graphs_new", "16random_diameter21test.edgelist")
# G = nx.read_graphml(graphml_file)

def see_graph(G):
    # Position nodes for better visualization
    pos = nx.spring_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos)

    # Draw edges
    nx.draw_networkx_edges(G, pos)

    # Draw edge labels (weights)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # # Draw node labels
    nx.draw_networkx_labels(G, pos)

    # Show the plot

    # for u, v, w in G.edges(data=True):
    #     print(f"{u} -- {v} : weight = {w['weight']}")
        
    plt.show()