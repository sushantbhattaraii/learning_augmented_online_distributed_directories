import networkx as nx
from typing import Hashable, Iterable, Tuple, Optional
from draw_graph import see_graph

def augment_tree_with_remaining_nodes(G,T,*,weight):
    """
    Augments a tree T (a subgraph of G) by attaching every node in G \ V(T)
    to the current tree with its lightest edge to a node already in the tree.
    
    This preserves the tree property: each step adds exactly one node and one edge.

    Parameters
    ----------
    G : nx.Graph
        Undirected, weighted graph. Edges should carry an attribute `weight`.
    T : nx.Graph
        A connected tree subgraph of G with p ≤ n nodes. T will NOT be modified in place.
    weight : str, optional
        Edge attribute name for weights (default "weight").
    process_order : iterable of nodes, optional
        If provided, nodes will be considered in this order (must be a subset/permutation
        of the remaining nodes). If None, arbitrary order is used.

    Returns
    -------
    T_final : nx.Graph
        The augmented tree containing original nodes of T plus any attachable nodes.
    unattached : set
        Nodes that could not be attached (i.e., no edge from that node to any node in the tree;
        happens if G is disconnected relative to T).

    Notes
    -----
    - If a remaining node has multiple equally light edges into T, an arbitrary lightest neighbor is chosen.
    - If a remaining node has *no* neighbors in T (disconnected component), it is left unattached.
    - If `process_order` includes nodes already in T, they are skipped harmlessly.
    """
    if not nx.is_tree(T):
        raise ValueError("T must be a tree (connected and acyclic).")

    # Work on a copy to avoid mutating inputs
    T_final = T.copy()
    tree_nodes = set(T_final.nodes)
    remaining = set(G.nodes) - tree_nodes

    
    order = list(remaining)

    unattached = set()

    for u in order:
        if u in tree_nodes:
            continue  # already in T

        # Find the lightest edge from u to any node currently in the tree
        best_neighbor = None
        best_w = None
        for v in G.neighbors(u):
            if v in tree_nodes:
                w = G[u][v].get(weight, 1.0)
                if best_w is None or w < best_w or (w == best_w and (best_neighbor is None or v < best_neighbor)):
                    best_w = w
                    best_neighbor = v

        if best_neighbor is None:
            # No connection into T right now (disconnected component)
            unattached.add(u)
            continue

        # Attach u to the tree via the lightest edge (u, best_neighbor)
        T_final.add_node(u, **G.nodes[u])
        T_final.add_edge(u, best_neighbor, **G[u][best_neighbor])

        # Maintaining tree property:
        # - We added exactly one new node and one new edge → no cycle introduced.
        tree_nodes.add(u)

    return T_final, unattached


# Build a sample graph
# G = nx.Graph()
# G.add_weighted_edges_from([
#     ("a","b",1), ("b","c",2), ("c","d",3), ("d","e",1), ("b","e",4), ("a","f",5), ("f","g",1), ("g","h",2), ("h","e",3), ("c","f",2), ("d","g",4), ("a","h",10), ("b","g",6), ("c","e",5)
# ])

# # see_graph(G)
# # Suppose T is the tree over nodes {"a","b","c"} with edges a-b, b-c
# T = nx.Graph()
# T.add_node("a"); T.add_node("b"); T.add_node("c")
# T.add_edge("a","b",weight=1)
# T.add_edge("b","c",weight=2)
# T.add_edge("g","h",weight=2)
# T.add_edge("b","g",weight=6)
# # see_graph(T)

# # Augment
# T_aug, unattached = augment_tree_with_remaining_nodes(G, T, weight="weight")

# # see_graph(T_aug)
# print("Augmented nodes:", T_aug.nodes())
# print("Augmented edges:", list(T_aug.edges(data=True)))
# print("Unattached:", unattached)

