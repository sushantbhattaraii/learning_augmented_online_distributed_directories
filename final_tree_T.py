import networkx as nx
from matplotlib import pyplot as plt
import random
import math
import heapq
from plot_graph import show_graph
from run import count_duplicates
from random_no_consecutive_numbers_generator import random_no_consecutive
from draw_graph import see_graph
from itertools import combinations


# def steiner_tree(G, steiner_vertices):
#     """
#     Constructs a Steiner Tree T_H for the undirected, weighted graph G
#     and the set of Steiner vertices S (steiner_vertices).

#     Parameters:
#     -----------
#     G : networkx.Graph
#         Undirected, weighted graph. Each edge must have a 'weight' attribute.
#     steiner_vertices : iterable
#         The set (or list) of Steiner vertices in G.

#     Returns:
#     --------
#     T_H : networkx.Graph
#         A subgraph of G that is the Steiner tree connecting all vertices in S.
#     """
#     # Convert steiner_vertices to a set for quick membership checks
#     S = set(steiner_vertices)

#     print("\n Inside steiner_tree function \n")
#     print("Steiner vertices in S:", S)

#     # Step 1: Construct the complete graph G1 on the Steiner vertices, using Dijkstra's algorithm
#     #         with edge weights given by shortest path distances in G.
#     #         We'll use all-pairs shortest paths restricted to S.
#     #         dist[u][v] = shortest distance from u to v in G.
#     dist = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
#     G1 = nx.Graph()
#     for u in S:
#         G1.add_node(u)
#     # Add edges to make G1 complete on S, with weights = shortest path distances
#     for u in S:
#         for v in S:
#             if u < v:
#                 G1.add_edge(u, v, weight=dist[u][v])

#     # Step 2: Find a Minimum Spanning Tree (T1) of G1 using Kruskal's algo.
#     T1 = nx.minimum_spanning_tree(G1, weight='weight')
#     see_graph(T1)

#     # Step 3: Construct G_s by replacing each edge (u, v) in T1 with
#     #         the corresponding shortest path in the original graph G.
#     G_s = nx.Graph()
#     # We'll need actual shortest paths, not just distances:
#     paths = dict(nx.all_pairs_dijkstra_path(G, weight='weight'))

#     for (u, v) in T1.edges():
#         shortest_path_uv = paths[u][v]
#         # Add edges along this shortest path to G_s
#         for i in range(len(shortest_path_uv) - 1):
#             a, b = shortest_path_uv[i], shortest_path_uv[i + 1]
#             w = G[a][b]['weight']
#             G_s.add_edge(a, b, weight=w)

#     # Step 4: Find a Minimum Spanning Tree (T_s) of G_s using Kruskal.
#     T_s = nx.minimum_spanning_tree(G_s, weight='weight')
#     see_graph(T_s)

#     # Step 5: Prune leaves in T_s that are not Steiner vertices.
#     #         i.e., repeatedly remove any leaf node not in S.
#     leaves = [n for n in T_s.nodes() if T_s.degree(n) == 1 and n not in S]
#     while leaves:
#         for leaf in leaves:
#             T_s.remove_node(leaf)
#         leaves = [n for n in T_s.nodes() if T_s.degree(n) == 1 and n not in S]

#     # T_s is now the final Steiner tree T_H
#     # pos = nx.spring_layout(T_s)
#     # edge_weight = nx.get_edge_attributes(T_s, 'weight')
#     # nx.draw(T_s, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
#     # nx.draw_networkx_edge_labels(T_s, pos, edge_labels=edge_weight)
#     # plt.title("Steiner Tree Visualization")
#     # plt.show()
#     return T_s


def steiner_tree(G, S):
    S = set(S)

    # Handle trivial case where S has only one node
    if(len(S) == 1):
        T_H = nx.Graph()
        only_node = next(iter(S))
        T_H.add_node(only_node)
        return T_H

    # Construct a complete undirected graph from G and S
    MC = nx.algorithms.approximation.steinertree.metric_closure(G)

    G1 = nx.Graph()
    G1.add_nodes_from(S)

    for u, v in combinations(S, 2):
        dist = MC[u][v]["distance"]    # shortest-path distance in G
        path = MC[u][v]["path"]        # realizing path in G
        G1.add_edge(u, v, weight=dist, path=path)

    # see_graph(G1)

    # Construct a mst from this graph G1, which we will say is: T1
    T1 = nx.minimum_spanning_tree(G1, weight="weight", algorithm="kruskal")

    # see_graph(T1)

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

    # Build mst of G_s and name it as T_s
    T_s = nx.minimum_spanning_tree(G_s, weight="weight", algorithm="kruskal")

    # see_graph(T_s)

    # Construct final steiner tree T_H by pruning leaves not in S
    # Repeatedly delete leaves not in `terminals` until every leaf is a terminal.

    T_H = T_s.copy()
    leaves = [n for n in T_H.nodes if T_H.degree(n) == 1 and n not in S]
    while leaves:
        T_H.remove_nodes_from(leaves)
        leaves = [n for n in T_H.nodes if T_H.degree(n) == 1 and n not in S]

    return T_H


# def construct_original_Vp(
#     G,
#     vp_size,
#     diameter_of_G=None,
#     weight="weight",
#     seed=None,
#     max_attempts=50000):
#     """
#     Build original_Vp of length vp_size:
#       • First pick TWO random nodes whose weighted distance <= diameter/3.
#       • Then repeatedly pick ONE random node whose distance to the LAST chosen node <= diameter/3.
#       • Finally, shuffle into an order with NO consecutive duplicates (values may repeat, just not adjacently).

#     Returns: list of nodes (original_Vp).
#     Raises:  ValueError if construction is impossible within max_attempts.
#     """
#     if vp_size < 2:
#         raise ValueError("vp_size must be at least 2 to pick an initial pair as requested.")

#     rng = random.Random(seed)
#     nodes = list(G.nodes())
#     if not nodes:
#         raise ValueError("Graph has no nodes.")
#     if diameter_of_G is None:
#         # Compute weighted diameter (on the largest connected component).
#         # For weighted graphs, use all_pairs_dijkstra_path_length & take max finite distance.
#         # If graph is disconnected, use the largest CC subgraph.
#         if isinstance(G, (nx.Graph, nx.DiGraph)):
#             if not nx.is_connected(G.to_undirected()):
#                 # restrict to the largest connected component
#                 cc = max(nx.connected_components(G.to_undirected()), key=len)
#                 H = G.subgraph(cc).copy()
#             else:
#                 H = G
#         else:
#             H = G
#         # Dijkstra APSP
#         maxd = 0.0
#         for s, dist_map in nx.all_pairs_dijkstra_path_length(H, weight=weight):
#             if dist_map:
#                 md = max(dist_map.values())
#                 if md > maxd:
#                     maxd = md
#         diameter_of_G = maxd if maxd > 0 else 0.0

#     threshold = diameter_of_G / 3.0

#     # helper: weighted shortest-path distance (inf if no path)
#     def wdist(u, v):
#         try:
#             return nx.shortest_path_length(G, u, v, weight=weight, method="dijkstra")
#         except nx.NetworkXNoPath:
#             return math.inf

#     # 1) Pick initial pair
#     attempts = 0
#     while attempts < max_attempts:
#         u = rng.choice(nodes)
#         v = rng.choice(nodes)
#         if u == v:
#             d = 0.0
#         else:
#             d = wdist(u, v)
#         if d <= threshold:
#             original_Vp = [u, v]
#             break
#         attempts += 1
#     else:
#         raise ValueError("Could not find an initial pair within threshold; increase threshold or check graph.")

#     # 2) Grow to vp_size: choose nodes close to the last element
#     while len(original_Vp) < vp_size:
#         last = original_Vp[-1]
#         found = False
#         for _ in range(1000):  # inner retries before giving up (tune as needed)
#             x = rng.choice(nodes)
#             d = 0.0 if x == last else wdist(last, x)
#             if d <= threshold:
#                 original_Vp.append(x)
#                 found = True
#                 break
#         if not found:
#             # Global retry loop to avoid dead-ends
#             attempts += 1
#             if attempts >= max_attempts:
#                 raise ValueError(
#                     f"Failed to extend sequence to vp_size={vp_size} under threshold={threshold}. "
#                     "Consider increasing the threshold or ensuring the graph is well-connected."
#                 )

#     # 3) Shuffle so there are NO consecutive duplicates (multiset preserved)
#     #    Use a max-heap rearrangement (like 'reorganize string' problem).
#     def shuffle_no_consecutive_dupes(seq):
#         from collections import Counter
#         cnt = Counter(seq)
#         # build max-heap of (-count, value)
#         heap = [(-c, val) for val, c in cnt.items()]
#         heapq.heapify(heap)
#         result = []
#         prev = None  # (neg_count, val) kept out one turn

#         while heap or prev:
#             if not heap:
#                 # Only prev remains: if its count > 0, we cannot place it without adjacency
#                 nc, val = prev
#                 if -nc > 0:
#                     # This means impossible to avoid adjacency with this multiset
#                     return None
#                 break

#             nc, val = heapq.heappop(heap)
#             # place val
#             result.append(val)
#             nc += 1  # since nc is negative, adding 1 moves it toward 0

#             # push back the prev (if any) because we used a different value now
#             if prev:
#                 heapq.heappush(heap, prev)
#                 prev = None

#             # hold current if it still has remaining count
#             if nc < 0:
#                 prev = (nc, val)

#         return result

#     perm = shuffle_no_consecutive_dupes(original_Vp)
#     if perm is None:
#         # Fall back: try a few random shuffles to break ties (rare; typically when one node dominates)
#         for _ in range(200):
#             rng.shuffle(original_Vp)
#             ok = all(original_Vp[i] != original_Vp[i+1] for i in range(len(original_Vp)-1))
#             if ok:
#                 perm = original_Vp[:]
#                 break
#         if perm is None:
#             # If still impossible, return the built list without the final constraint (or raise).
#             raise ValueError(
#                 "Could not produce a permutation without consecutive duplicates. "
#                 "Your list is too dominated by a single node. Consider relaxing constraints or diversifying picks."
#             )

#     return perm

def build_original_Vp(G, vp_size, nodes, diameter_of_G=None, seed=None, max_attempts_per_slot=5000):
    """
    Construct original_Vp of length vp_size.
    Rule: each chosen node must be within (diameter_of_G)/3 of every node already in original_Vp,
          and it cannot be the same as the immediately previous node (no consecutive duplicates).

    Parameters
    ----------
    G : networkx.Graph
        Assumed connected (diameter must exist).
    vp_size : int
        Desired length of the output list.
    diameter_of_G : float | int | None
        If None, computed via nx.diameter(G).
    seed : int | None
        Random seed for reproducibility.
    max_attempts_per_slot : int
        Safety cap on random draws per position before giving up.

    Returns
    -------
    list
        original_Vp of length vp_size satisfying the constraints.

    Raises
    ------
    ValueError
        If constraints are impossible to satisfy for some slot.
    """
    if vp_size <= 0:
        return []

    rng = random.Random(seed)

    # Compute diameter if not provided
    if diameter_of_G is None:
        diameter_of_G = nx.diameter(G, weight='weight')

    # Distance threshold
    thresh = diameter_of_G/3.0

    # Precompute all-pairs shortest path lengths (works well for small/medium graphs).
    # For very large graphs, replace with an on-demand cache using single-source distances.
    all_dists = dict(nx.all_pairs_shortest_path_length(G))

    # nodes = list(G.nodes())
    if not nodes:
        raise ValueError("Graph has no nodes.")

    original_Vp = []

    # Helper to check if a candidate is valid
    def is_valid_candidate(cand):
        # No consecutive duplicates
        if original_Vp and cand == original_Vp[-1]:
            return False
        # Must be within thresh of ALL existing picks
        for v in original_Vp:
            # If disconnected, distance won't exist; treat as invalid
            try:
                d = all_dists[cand][v]
            except KeyError:
                return False
            if d >= thresh:
                return False
        return True

    # Fill positions 0 .. vp_size-1
    for pos in range(vp_size):
        # Special note: when original_Vp is empty, the "within thresh of all elements"
        # condition is vacuously true, so any node can start.
        tries = 0
        found = False

        while tries < max_attempts_per_slot and not found:
            cand = rng.choice(nodes)
            if is_valid_candidate(cand):
                original_Vp.append(cand)
                found = True
            else:
                tries += 1

        if not found:
            # Optional: you could try a deterministic scan instead of random after the cap
            # to avoid unlucky randomness:
            for cand in rng.sample(nodes, k=len(nodes)):
                if is_valid_candidate(cand):
                    original_Vp.append(cand)
                    found = True
                    break

        if not found:
            raise ValueError(
                f"Failed to place a node at position {pos} under the given constraints. "
                f"Consider increasing the threshold (diameter), ensuring the graph is connected, "
                f"or reducing vp_size."
            )

    return original_Vp

def choose_steiner_set(G, fraction, diameter_of_G, myNodeCount):
    """
    Randomly choose Vp ('fraction' of all nodes) as predicted nodes,
    and then choose one additional 'owner' node not in Vp.
    Return the set S = Vp ∪ {owner}, along with Vp and owner.
    """
    # G = nx.relabel_nodes(G, lambda x: int(x))
    # diameter_of_G = nx.diameter(G, weight='weight')
    nodes = list(range(myNodeCount))
    random.shuffle(nodes)  # Shuffle the nodes to ensure randomness

    vp_size = int(fraction) # Fraction of nodes to be chosen as Vp
    # vp_size = int(total_nodes * fraction) # Fraction of nodes to be chosen as Vp
    # original_Vp = list(random.choices(nodes, k=vp_size))

    # original_Vp = construct_original_Vp(
    #     G,
    #     vp_size,
    #     diameter_of_G=diameter_of_G,
    #     weight="weight",
    #     seed=42
    # )
    original_Vp = build_original_Vp(G, vp_size, nodes, diameter_of_G=diameter_of_G, seed=random.randint(0, 100), max_attempts_per_slot=50000)

    # This function 'random_no_consecutive' generates a list of 'vp_size' random numbers between 0 and total_nodes-1 with guaranteed no consecutive duplicates 
    # original_Vp = random_no_consecutive(n=vp_size, a=0, b=total_nodes-1, rng=random.Random())
    
    # random.shuffle(original_Vp)  # Shuffle Vp to ensure randomness

    # print("Predicted Vertices (original_Vp):", original_Vp, " and its length: ", len(original_Vp))

    dup_counts = count_duplicates(original_Vp)
    # print("Length of Original Vp: ",len(original_Vp))
    # extra dups = sum of (count - 1) for each duplicated element
    extra_dups = sum(cnt for cnt in dup_counts.values())

    # if dup_counts:
    #     print("Duplicate elements in original_Vp and their counts:")
    #     for element, count in dup_counts.items():
    #         print(f"{element}: {count}")
    # else:
    #     print("No duplicate elements found.")

    reduced_Vp = set(original_Vp)  # Convert to a set for uniqueness

    # Vp = random.shuffle(Vp)  # Convert to a set for uniqueness
    # print("Set (reduced_Vp):", reduced_Vp)
    # print("Length of Set reduced_Vp: ",len(reduced_Vp))
    # print("Type of Set reduced_Vp:", type(reduced_Vp))

    reduced_Vp = list(reduced_Vp)  # Convert back to a list for indexing
    random.shuffle(reduced_Vp)  # Shuffle Vp to ensure randomness
    # print("List (reduced_Vp):", reduced_Vp)
    # print("Length of List reduced_Vp: ",len(reduced_Vp))
    # print("Type of list reduced_Vp:", type(reduced_Vp))

    # Choose an owner node that is not in Vp
    # remaining = set(nodes) - set(reduced_Vp)
    owner = random.choice(list(nodes))

    print ("Owner node:", owner)
    # print("Type of owner:", type(owner))

    # print("Again (original_Vp):", original_Vp, " and its length: ", len(original_Vp))

    # Insert owner to reduced_Vp list at a random position
    insert_position = random.randint(0, len(reduced_Vp))
    reduced_Vp.insert(insert_position, owner)
    S = reduced_Vp.copy()
    S = set(S)  # Convert to a set for uniqueness
    # print("List after inserting owner in reduced Vp:", reduced_Vp)
    # print("Length of List after inserting owner: ",len(reduced_Vp))
    # print("\nSet after inserting owner in S:", S)
    # print("Length of Set S after inserting owner: ",len(S))
    # print("Type of list after inserting owner:", type(reduced_Vp))
    # print("Type of original vp:", type(original_Vp))

    # Returning original_Vp as list, reduced_Vp as list, and owner as integer

    return S, original_Vp, owner


def augment_steiner_tree_with_remaining_vertices(G, T_H, myNodeCount):
    """
    Augments a given Steiner tree T_H by adding the remaining vertices of G,
    connecting each vertex (from V \ V(T_H)) to the current tree T1 via the shortest path.
    
    Parameters:
    -----------
    G : networkx.Graph
        The original weighted, undirected graph. Each edge must have a 'weight' attribute.
    T_H : networkx.Graph
        The Steiner tree (subgraph of G) computed from the previous algorithm.
    
    Returns:
    --------
    T_final : networkx.Graph
        A tree that spans all vertices of G.
    """
    # Start with a copy of the Steiner tree.
    T_final = T_H.copy()
    
    # Set of vertices already in the tree.
    current_nodes = set(T_final.nodes())
    
    # Set of vertices not yet added.
    remaining_nodes = set(list(range(myNodeCount))) - current_nodes
    
    # Continue until all vertices from G are in the tree.
    while remaining_nodes:
        # Use multi-source Dijkstra to compute shortest distances from all nodes in T_final.
        distances, paths = nx.multi_source_dijkstra(G, current_nodes, weight='weight')
        
        # Among the remaining nodes, find one with the minimum distance to T_final.
        candidate = min(remaining_nodes, key=lambda v: distances.get(v, float('inf')))
        
        # Retrieve the shortest path from T_final (one of its nodes) to the candidate.
        path = paths[candidate]
        
        # Add all vertices and edges along the path to T_final.
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if not T_final.has_node(u):
                T_final.add_node(u)
            if not T_final.has_node(v):
                T_final.add_node(v)
            # Add the edge if it doesn't already exist.
            if not T_final.has_edge(u, v):
                T_final.add_edge(u, v, weight=G[u][v]['weight'])
        
        # Update the current tree's vertex set.
        current_nodes.update(path)
        # Update the remaining nodes.
        remaining_nodes = set(G.nodes()) - current_nodes

    return T_final


def augment_modified_mst_with_remaining_vertices(G, T_H, myNodeCount):
    """
    Augments a given Steiner tree T_H by adding the remaining vertices of G,
    connecting each vertex (from V \ V(T_H)) to the current tree T1 via the shortest path.
    
    Parameters:
    -----------
    G : networkx.Graph
        The original weighted, undirected graph. Each edge must have a 'weight' attribute.
    T_H : networkx.Graph
        The Steiner tree (subgraph of G) computed from the previous algorithm.
    
    Returns:
    --------
    T_final : networkx.Graph
        A tree that spans all vertices of G.
    """
    # Start with a copy of the Steiner tree.
    T_final = T_H.copy()
    
    # Set of vertices already in the tree.
    current_nodes = set(T_final.nodes())
    
    # Set of vertices not yet added.
    remaining_nodes = set(list(range(myNodeCount))) - current_nodes
    
    # Continue until all vertices from G are in the tree.
    while remaining_nodes:
        # Use multi-source Dijkstra to compute shortest distances from all nodes in T_final.
        distances, paths = nx.multi_source_dijkstra(G, current_nodes, weight='weight')
        
        # Among the remaining nodes, find one with the minimum distance to T_final.
        candidate = min(remaining_nodes, key=lambda v: distances.get(v, float('inf')))
        
        # Retrieve the shortest path from T_final (one of its nodes) to the candidate.
        path = paths[candidate]
        
        # Add all vertices and edges along the path to T_final.
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if not T_final.has_node(u):
                T_final.add_node(u)
            if not T_final.has_node(v):
                T_final.add_node(v)
            # Add the edge if it doesn't already exist.
            if not T_final.has_edge(u, v):
                T_final.add_edge(u, v, weight=G[u][v]['weight'])
        
        # Update the current tree's vertex set.
        current_nodes.update(path)
        # Update the remaining nodes.
        remaining_nodes = set(G.nodes()) - current_nodes

    return T_final

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

    # pos = nx.spring_layout(G_example)
    # edge_weight = nx.get_edge_attributes(G_example, 'weight')
    # nx.draw(G_example, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    # nx.draw_networkx_edge_labels(G_example, pos, edge_labels=edge_weight)
    # plt.title("GraphML Graph Visualization")
    # plt.show()

    print("Nodes in G_example:", list(G_example.nodes()))
    G_example = nx.relabel_nodes(G_example, lambda x: int(x))
    # for node in G_example.nodes:
    #     print(f"Node: {node}, Data Type: {type(node)}")

    # Suppose Steiner vertices S are {1, 3, 5, 4}
    # S_example = {1, 2, 3, 4}

    # Or, take a random fraction of total nodes (say) 1/4th of the total nodes
    
    # Choose the Steiner set S = Vp ∪ {owner}
    S_example, Vp, owner = choose_steiner_set(G_example, 0.25)
    print("Randomly chosen Predicted Vertices (Vp):", Vp)
    print("Owner node:", owner)
    print("Steiner set S:", S_example)

    # Compute Steiner tree
    T_H = steiner_tree(G_example, S_example)

    # Print edges of the resulting Final tree
    print("Final Tree edges:")
    for (u, v, data) in T_H.edges(data=True):
        print(f"{u} - {v}, weight = {v['weight'] if isinstance(v, dict) else v}")

    # Compute Final tree T
    T, root = augment_steiner_tree_with_remaining_vertices(G_example, T_H)
    show_graph(T)

    # Print edges of the resulting Final tree
    print("Final Tree edges:")
    for (u, v, data) in T.edges(data=True):
        print(f"{u} - {v}, weight = {v['weight'] if isinstance(v, dict) else v}")
