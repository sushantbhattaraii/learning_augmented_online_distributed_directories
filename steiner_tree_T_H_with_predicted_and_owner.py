import networkx as nx
from matplotlib import pyplot as plt
import random

def steiner_tree(G, steiner_vertices):
    """
    Constructs a Steiner Tree T_H for the undirected, weighted graph G
    and the set of Steiner vertices S (steiner_vertices).

    Parameters:
    -----------
    G : networkx.Graph
        Undirected, weighted graph. Each edge must have a 'weight' attribute.
    steiner_vertices : iterable
        The set (or list) of Steiner vertices in G.

    Returns:
    --------
    T_H : networkx.Graph
        A subgraph of G that is the Steiner tree connecting all vertices in S.
    """
    # Convert steiner_vertices to a set for quick membership checks
    S = set(steiner_vertices)

    # Step 1: Construct the complete graph G1 on the Steiner vertices, using Dijkstra's algorithm
    #         with edge weights given by shortest path distances in G.
    #         We'll use all-pairs shortest paths restricted to S.
    #         dist[u][v] = shortest distance from u to v in G.
    dist = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    G1 = nx.Graph()
    for u in S:
        G1.add_node(u)
    # Add edges to make G1 complete on S, with weights = shortest path distances
    for u in S:
        for v in S:
            if u < v:
                G1.add_edge(u, v, weight=dist[u][v])

    # Step 2: Find a Minimum Spanning Tree (T1) of G1 using Kruskal's algo.
    T1 = nx.minimum_spanning_tree(G1, weight='weight')

    # Step 3: Construct G_s by replacing each edge (u, v) in T1 with
    #         the corresponding shortest path in the original graph G.
    G_s = nx.Graph()
    # We'll need actual shortest paths, not just distances:
    paths = dict(nx.all_pairs_dijkstra_path(G, weight='weight'))

    for (u, v) in T1.edges():
        shortest_path_uv = paths[u][v]
        # Add edges along this shortest path to G_s
        for i in range(len(shortest_path_uv) - 1):
            a, b = shortest_path_uv[i], shortest_path_uv[i + 1]
            w = G[a][b]['weight']
            G_s.add_edge(a, b, weight=w)

    # Step 4: Find a Minimum Spanning Tree (T_s) of G_s using Kruskal.
    T_s = nx.minimum_spanning_tree(G_s, weight='weight')

    # Step 5: Prune leaves in T_s that are not Steiner vertices.
    #         i.e., repeatedly remove any leaf node not in S.
    leaves = [n for n in T_s.nodes() if T_s.degree(n) == 1 and n not in S]
    while leaves:
        for leaf in leaves:
            T_s.remove_node(leaf)
        leaves = [n for n in T_s.nodes() if T_s.degree(n) == 1 and n not in S]

    # T_s is now the final Steiner tree T_H
    pos = nx.spring_layout(T_s)
    edge_weight = nx.get_edge_attributes(T_s, 'weight')
    nx.draw(T_s, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    nx.draw_networkx_edge_labels(T_s, pos, edge_labels=edge_weight)
    plt.title("Steiner Tree Visualization")
    plt.show()
    return T_s


def choose_steiner_set(G):
    """
    Randomly choose Vp (1/4 of all nodes) as predicted nodes,
    and then choose one additional 'owner' node not in Vp.
    Return the set S = Vp ∪ {owner}, along with Vp and owner.
    """
    G = nx.relabel_nodes(G, lambda x: int(x))
    nodes = list(G.nodes())
    total_nodes = len(nodes)
    vp_size = total_nodes // 4  # using integer division for 1/3 of nodes
    Vp = set(random.sample(nodes, vp_size))
    
    # Choose an owner node that is not in Vp
    remaining = set(nodes) - Vp
    owner = random.choice(list(remaining))
    
    S = Vp.union({owner})
    return S, Vp, owner


if __name__ == "__main__":
    # Example usage:
    # Create a simple weighted graph
    # G_example = nx.Graph()
    # edges = [
    #     (1, 2, 10),
    #     (1, 9, 1),
    #     (2, 6, 1),
    #     (2, 3, 8),
    #     (3, 5, 2),
    #     (3, 4, 9),
    #     (4, 5, 2),
    #     (5, 6, 1),
    #     (5, 9, 1),
    #     (6, 7, 1),
    #     (7, 8, 0.5),
    #     (8, 9, 0.5)
    # ]
    # G_example.add_weighted_edges_from(edges)

    # Or, Load the graph from a GraphML file
    graphml_file = '.\\graphs\\'+'10random_diameter6test.edgelist'
    G_example = nx.read_graphml(graphml_file)

    pos = nx.spring_layout(G_example)
    edge_weight = nx.get_edge_attributes(G_example, 'weight')
    nx.draw(G_example, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    nx.draw_networkx_edge_labels(G_example, pos, edge_labels=edge_weight)
    plt.title("GraphML Graph Visualization")
    plt.show()

    print("Nodes in G_example:", list(G_example.nodes()))
    G_example = nx.relabel_nodes(G_example, lambda x: int(x))
    # for node in G_example.nodes:
    #     print(f"Node: {node}, Data Type: {type(node)}")

    # Suppose Steiner vertices S are {1, 3, 5, 4}
    # S_example = {1, 2, 3, 4}

    # Or, take a random fraction of total nodes (say) 1/4th of the total nodes
    
    # Choose the Steiner set S = Vp ∪ {owner}
    S_example, Vp, owner = choose_steiner_set(G_example)
    print("Randomly chosen Predicted Vertices (Vp):", Vp)
    print("Owner node:", owner)
    print("Steiner set S:", S_example)

    # Compute Steiner tree
    T_H_example = steiner_tree(G_example, S_example)

    # Print edges of the resulting Steiner tree
    print("Steiner Tree edges:")
    for (u, v, data) in T_H_example.edges(data=True):
        print(f"{u} - {v}, weight = {v['weight'] if isinstance(v, dict) else v}")
