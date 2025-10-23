# #importing the networkx library
# # import networkx as nx

# # #importing the matplotlib library for plotting the graph
# # import matplotlib.pyplot as plt

# # G = nx.erdos_renyi_graph(10,0.1)
# # print(nx.is_connected(G))
# # nx.draw(G, with_labels=True)
# # plt.show()


# import matplotlib.pyplot as plt

# # Example data for x and y axes
# # x = [1, 2, 3, 4, 5]
# # y = [2, 4, 1, 8, 7]

# # plt.figure()
# # plt.plot(x, y, marker='o', linestyle='-')
# # plt.xlabel('X-axis label')
# # plt.ylabel('Y-axis label')
# # plt.title('Sample 2D Plot')
# # plt.grid(True)
# # plt.show()



# # import networkx as nx

# # G = nx.read_graphml(".\\graphs\\256random_diameter71test.edgelist")

# # # Gather all data‐keys present on edges
# # edge_keys = {
# #     k
# #     for u, v, data in G.edges(data=True)
# #     for k in data.keys()
# # }

# # if "weight" in edge_keys:
# #     print("✓ Found an edge attribute named 'weight'")
# # else:
# #     print("✗ No 'weight' attribute on any edge")
    
# # # (you can also print edge_keys to see other attributes)
# # print("All edge attributes:", edge_keys)


# import matplotlib.pyplot as plt

# # Create the plot
# fig, ax = plt.subplots()

# # Set y-ticks
# y_ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
# for i in range(1, 41):
#     y_ticks.append(1 + i * 0.1)

# ax.set_yticks(y_ticks)
# ax.set_ylim(0.0, 4.0)  # Set y-axis limits to match ticks

# # Add a dummy plot to prevent matplotlib from automatically setting the y-ticks
# ax.plot([0], [0], color='white')

# # Show the plot
# plt.show()


# from draw_graph import see_graph
# import networkx as nx
# import matplotlib.pyplot as plt
# import os

# network_file_name = "16random_diameter35test.edgelist"

# graphml_file = os.path.join('graphs_new', str(network_file_name))
# G_example = nx.read_graphml(graphml_file)
# G_example = nx.relabel_nodes(G_example, lambda x: int(x))
# see_graph(G_example)


import networkx as nx
from itertools import combinations
from draw_graph import see_graph

# 1) Build the original weighted, undirected graph G
G = nx.Graph()
edges = [
    (1, 2, 10),
    (1, 9, 1),
    (2, 3, 8),
    (2, 6, 0.5),
    (3, 4, 9),   # 3 - 4 = 9
    (3, 5, 2),
    (4, 5, 2),
    (5, 6, 1),
    (5, 9, 1),
    (6, 7, 1),
    (7, 8, 0.5),
    (8, 9, 0.5),
]
G.add_weighted_edges_from(edges)

# 2) Terminals / Steiner points
S = [1, 2, 3, 4]
S = set(S)

# 3) Build the complete graph G1 on S using shortest-path distances in G
#    (a.k.a. the metric closure restricted to S)
#    NetworkX provides a handy metric_closure that also stores the realizing path.
MC = nx.algorithms.approximation.steinertree.metric_closure(G)

G1 = nx.Graph()
G1.add_nodes_from(S)

for u, v in combinations(S, 2):
    dist = MC[u][v]["distance"]    # shortest-path distance in G
    path = MC[u][v]["path"]        # realizing path in G
    G1.add_edge(u, v, weight=dist, path=path)

see_graph(G1)
T1 = nx.minimum_spanning_tree(G1, weight="weight", algorithm="kruskal")

see_graph(T1)

# Build G_s by replacing each mst_g1 edge with its shortest path in G
G_s = nx.Graph()

for u, v in T1.edges():
    # Prefer the realizing path we stored in G1; fall back to recomputing if absent
    # path = G1[u][v].get("path") or nx.shortest_path(G, u, v, weight="weight")
    path = G1[u][v].get("path")

    # Add every consecutive edge on this path with weights from G
    for a, b in zip(path[:-1], path[1:]):
        w = G[a][b]["weight"]
        if not G_s.has_edge(a, b):
            G_s.add_edge(a, b, weight=w)

see_graph(G_s)

# Build mst of G_s and name it as T_s
T_s = nx.minimum_spanning_tree(G_s, weight="weight", algorithm="kruskal")
see_graph(T_s)

# Construct final steiner tree T_H by pruning leaves not in S
# Repeatedly delete leaves not in `terminals` until every leaf is a terminal.

T_H = T_s.copy()
leaves = [n for n in T_H.nodes if T_H.degree(n) == 1 and n not in S]
print("Leaves to prune:", leaves)
exit()
while leaves:
        T_H.remove_nodes_from(leaves)
        leaves = [n for n in T_H.nodes if T_H.degree(n) == 1 and n not in S]

see_graph(T_H)



