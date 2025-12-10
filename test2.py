import networkx as nx
import heapq
from final_tree_T import steiner_tree
from draw_graph import see_graph

def construct_augmented_spanning_tree(G, S, T_H):
    """
    Constructs a spanning tree T covering all vertices V of G.
    It starts with Steiner Tree T_H and adds remaining vertices 
    based on shortest distance to S.

    Parameters:
    - G: The original graph (networkx.Graph)
    - S: The subset of vertices (list or set) used as terminals
    - T_H: The existing Steiner Tree (networkx.Graph)
    
    Returns:
    - T: The final spanning tree (networkx.Graph)
    """
    
    # 1. Initialize the final tree T as a copy of the Steiner Tree
    T = T_H.copy()
    
    # Set of vertices already in the tree (to avoid cycles or redundancy)
    # T_H might contain Steiner points (nodes not in S), so we track all nodes in T_H
    nodes_in_tree = set(T.nodes())
    
    # Check if we are already done (if T_H already spans G)
    if len(nodes_in_tree) == len(G.nodes()):
        return T

    # 2. Multi-Source Dijkstra Initialization
    # Priority Queue stores tuples: (current_distance, current_node, parent_node)
    pq = []
    
    # Dictionary to store the shortest distance found to any node in S
    # Initialize with infinity
    shortest_dist = {v: float('inf') for v in G.nodes()}
    
    # Initialize the sources (S)
    for s in S:
        shortest_dist[s] = 0
        # We push (distance 0, source node s, parent is None)
        heapq.heappush(pq, (0, s, None))

    # 3. Process the Graph
    while pq:
        d, u, parent = heapq.heappop(pq)
        
        # If we found a shorter path to u previously, skip
        if d > shortest_dist[u]:
            continue
        
        # AUGMENTATION LOGIC:
        # If u is not in the tree yet, we attach it to its parent.
        # The parent is the node that led us here on the shortest path from S.
        if u not in nodes_in_tree:
            # Add the node and the edge connecting to the parent
            T.add_node(u)
            # Find the weight of the edge in the original graph G
            weight = G[parent][u].get('weight', 1) 
            T.add_edge(parent, u, weight=weight)
            nodes_in_tree.add(u)

        # Explore neighbors
        for v in G.neighbors(u):
            weight = G[u][v].get('weight', 1)
            new_dist = d + weight
            
            # Standard Dijkstra relaxation
            if new_dist < shortest_dist[v]:
                shortest_dist[v] = new_dist
                # Push to PQ with u as the parent
                heapq.heappush(pq, (new_dist, v, u))
                
    return T

# --- Example Usage ---

# Create a sample graph
G = nx.Graph()
G.add_edge('A', 'B', weight=4)
G.add_edge('B', 'C', weight=3)
G.add_edge('C', 'D', weight=2)
G.add_edge('A', 'C', weight=10)
G.add_edge('D', 'E', weight=3)
G.add_edge('B', 'E', weight=8)

# Define terminals S
S = {'A', 'E'}

# Pretend we have a pre-calculated Steiner Tree T_H connecting A and C
# (In this case, path A-B-C is weight 7, A-C is 10. Let's assume T_H uses A-B-C)
# T_H = nx.Graph()
# T_H.add_edge('A', 'B', weight=4)
# T_H.add_edge('B', 'C', weight=3)

T_H = steiner_tree(G, S)
see_graph(T_H)
exit()

# Run the subroutine
final_tree = construct_augmented_spanning_tree(G, S, T_H)

print("Edges in final Augmented Tree:")
for u, v in final_tree.edges():
    print(f"({u}, {v})")
    
# Expected Logic:
# D is closest to C (dist 2). E is closest to D (dist 1 -> total 3 to C).
# The tree should contain T_H edges (A-B, B-C) plus (C-D) and (D-E).