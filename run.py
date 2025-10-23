from collections import defaultdict, deque
from final_tree_T import *
from tree_center import find_tree_center
import random
import networkx as nx
import network_generator as my_ng
import argparse
import matplotlib.pyplot as plt
import os
import re
from fractions import Fraction
from collections import Counter
from draw_graph import see_graph
from random_no_consecutive_numbers_generator import random_from_set_no_consecutive
from augment_rem_nodes_with_existing_tree import augment_tree_with_remaining_nodes


request_queue = defaultdict(deque)
global link_, linkArrow_, linkNew_
link_ = None
linkArrow_ = None
linkNew_ = None


def build_parent_dict(T, root):
    """
    Perform a BFS (or DFS) from 'root' in the tree T to define
    a parent-child relationship. 
    Returns a dict 'parent' where parent[u] = node's parent in T 
    (with root having parent[root] = None).
    """
    parent = {root: None}
    queue = deque([root])
    
    while queue:
        current = queue.popleft()
        for neighbor in T.neighbors(current):
            if neighbor not in parent:  # not visited
                parent[neighbor] = current
                queue.append(neighbor)
    return parent

def build_parent_dict_arrow(mst, root):
    """
    Perform a BFS (or DFS) from 'root' in the tree T to define
    a parent-child relationship. 
    Returns a dict 'parent' where parent[u] = node's parent in T 
    (with root having parent[root] = None).
    """
    parent_arrow = {root: None}
    queue_arrow = deque([root])
    
    while queue_arrow:
        current_arrow = queue_arrow.popleft()
        for neighbor in mst.neighbors(current_arrow):
            if neighbor not in parent_arrow:  # not visited
                parent_arrow[neighbor] = current_arrow
                queue_arrow.append(neighbor)
    return parent_arrow


def publish(T, o, root, parent, link_):
    """
    Implements Algorithm 1 (Publish) from your snippet.
    
    Parameters:
    -----------
    T      : networkx.Graph (tree)
    o      : The node that currently receives/owns the resource.
    root   : The designated root of T.
    parent : dict, mapping each node to its parent in T.
    link_  : dict, mapping each node to link[node]. 
             This function modifies link_ in place.
    
    After publish(), for each node ui on the path from o up to (but not including) root,
    we set link(ui) = child, where 'child' is the node from which the publish message arrived.
    """
    u = o
    # ui = parent[u]
    ui = parent.get(u, None)  # Use .get() to avoid KeyError
    
    # Climb up the tree until we reach the root
    while ui is not None:
        link_[ui] = u
        # Move up one level
        u = ui
        # ui = parent[u]
        ui = parent.get(u, None)  # Use .get() to avoid KeyError
        # print("U->", u, "ui->",ui)
        if(u == root):
            break
    # The loop stops when ui == root or ui == None.
    # By the pseudocode, we do NOT set link(root) in the final step.


def publish_arrow(mst_g, o, root, parent_arrow, linkArrow_):
    """
    Implements Algorithm 1 (Publish) from your snippet.
    
    Parameters:
    -----------
    T      : networkx.Graph (tree)
    o      : The node that currently receives/owns the resource.
    root   : The designated root of T.
    parent : dict, mapping each node to its parent in T.
    link_  : dict, mapping each node to link[node]. 
             This function modifies link_ in place.
    
    After publish(), for each node ui on the path from o up to (but not including) root,
    we set link(ui) = child, where 'child' is the node from which the publish message arrived.
    """
    u = o
    # ui = parent[u]
    ui = parent_arrow.get(u, None)  # Use .get() to avoid KeyError
    
    # Climb up the tree until we reach the root
    while ui is not None:
        linkArrow_[ui] = u
        # Move up one level
        u = ui
        # ui = parent_arrow[u]
        ui = parent_arrow.get(u, None)  # Use .get() to avoid KeyError
        # print("U->", u, "ui->",ui)
        if(u == root):
            break
    # The loop stops when ui == root or ui == None.
    # By the pseudocode, we do NOT set link(root) in the final step.

def publish_new(T, o, root, parent, linkNew_):
    """
    Implements Algorithm 1 (Publish) from your snippet.
    
    Parameters:
    -----------
    T      : networkx.Graph (tree)
    o      : The node that currently receives/owns the resource.
    root   : The designated root of T.
    parent : dict, mapping each node to its parent in T.
    link_  : dict, mapping each node to link[node]. 
             This function modifies link_ in place.
    
    After publish(), for each node ui on the path from o up to (but not including) root,
    we set link(ui) = child, where 'child' is the node from which the publish message arrived.
    """
    u = o
    # ui = parent[u]
    ui = parent.get(u, None)  # Use .get() to avoid KeyError
    
    # Climb up the tree until we reach the root
    while ui is not None:
        linkNew_[ui] = u
        # Move up one level
        u = ui
        # ui = parent[u]
        ui = parent.get(u, None)  # Use .get() to avoid KeyError
        # print("U->", u, "ui->",ui)
        if(u == root):
            break
    # The loop stops when ui == root or ui == None.
    # By the pseudocode, we do NOT set link(root) in the final step.


def set_links_for_request(G, T, mst_g, T_new, requesting_node, parent, parent_arrow, parent_new, link_, linkArrow_, linkNew_, root):
    """
    For a requesting node r:
    1) Set link_[r] = r.
    2) Climb up from r to root, flipping pointers so that link_[p] = child,
       where p is the parent and child is the node from which the request came.
    3) After establishing these links on the path, set all other links to None.
    """
    # Keep track of the path from requesting_node to root
    path_nodes = []
    path_nodes_arrow = []
    path_nodes_new = []


    for node, value in link_.items():
        if value == node:
            owner = node

    for node, value in linkArrow_.items():
        if value == node:
            owner_arrow = node

    for node, value in linkNew_.items():
        if value == node:
            owner_new = node

    dist_u_v_in_T = nx.shortest_path_length(T, source=owner, target=requesting_node, weight='weight')

    dist_u_v_in_mst_g = nx.shortest_path_length(mst_g, source=owner_arrow, target=requesting_node, weight='weight')

    dist_u_v_in_T_new = nx.shortest_path_length(T_new, source=owner_new, target=requesting_node, weight='weight')

    dist_u_v_in_G = nx.shortest_path_length(G, source=owner, target=requesting_node, weight='weight')

    # stretch = float(dist_u_v_in_T / dist_u_v_in_G)
    
    # Step 1: requesting_node points to itself
    link_[requesting_node] = requesting_node
    path_nodes.append(requesting_node)

    linkArrow_[requesting_node] = requesting_node
    path_nodes_arrow.append(requesting_node)

    linkNew_[requesting_node] = requesting_node
    path_nodes_new.append(requesting_node)
    
    # Step 2: climb upwards until we reach the root
    current = requesting_node
    while current != root:
        p = parent[current]
        # If there's no parent (i.e., current is already root), break
        if p is None:
            break
        link_[p] = current  # flip pointer
        path_nodes.append(p)
        current = p

    
    current_arrow = requesting_node
    while current_arrow != root:
        p_arr = parent_arrow[current_arrow]
        # If there's no parent (i.e., current is already root), break
        if p_arr is None:
            break
        linkArrow_[p_arr] = current_arrow  # flip pointer
        path_nodes_arrow.append(p_arr)
        current_arrow = p_arr

    current_new = requesting_node
    while current_new != root:
        p = parent_new[current_new]
        # If there's no parent (i.e., current is already root), break
        if p is None:
            break
        linkNew_[p] = current_new  # flip pointer
        path_nodes_new.append(p)
        current_new = p
    
    # Step 3: For all other nodes NOT on this path, set link_[node] = None
    for node in link_.keys():
        if node not in path_nodes:
            link_[node] = None

    for node in linkArrow_.keys():
        if node not in path_nodes_arrow:
            linkArrow_[node] = None

    for node in linkNew_.keys():
        if node not in path_nodes_new:
            linkNew_[node] = None

    return dist_u_v_in_G, dist_u_v_in_T, dist_u_v_in_mst_g, dist_u_v_in_T_new


def set_links_for_request_for_arrow(G, T, requesting_node, parent, linkArrow_, root):
    """
    For a requesting node r:
    1) Set link_[r] = r.
    2) Climb up from r to root, flipping pointers so that link_[p] = child,
       where p is the parent and child is the node from which the request came.
    3) After establishing these links on the path, set all other links to None.
    """
    # Keep track of the path from requesting_node to root
    path_nodes = []


    for node, value in linkArrow_.items():
        if value == node:
            owner = node

    dist_u_v_in_mst_g = nx.shortest_path_length(T, source=owner, target=requesting_node, weight='weight')

    dist_u_v_in_G = nx.shortest_path_length(G, source=owner, target=requesting_node, weight='weight')

    # stretch = float(dist_u_v_in_T / dist_u_v_in_G)
    
    # Step 1: requesting_node points to itself
    linkArrow_[requesting_node] = requesting_node
    path_nodes.append(requesting_node)
    
    # Step 2: climb upwards until we reach the root
    current = requesting_node
    while current != root:
        p = parent[current]
        # If there's no parent (i.e., current is already root), break
        if p is None:
            break
        linkArrow_[p] = current  # flip pointer
        path_nodes.append(p)
        current = p
    
    # Step 3: For all other nodes NOT on this path, set link_[node] = None
    for node in linkArrow_.keys():
        if node not in path_nodes:
            linkArrow_[node] = None

    return dist_u_v_in_G, dist_u_v_in_mst_g


def load_graph(network_file_name):
    graphml_file = os.path.join('graphs_new', str(network_file_name))
    G_example = nx.read_graphml(graphml_file)
    G_example = nx.relabel_nodes(G_example, lambda x: int(x))
    return G_example

def load_mst(network_file_name):
    graphml_file = os.path.join('graphs_new', 'mst', str(network_file_name))
    mst_example = nx.read_graphml(graphml_file)
    mst_example = nx.relabel_nodes(mst_example, lambda x: int(x))
    return mst_example


def count_duplicates(input_list):
    """
    Checks for duplicate elements in a list and returns their counts.

    Args:
        input_list: The list to check for duplicates.

    Returns:
        A dictionary where keys are the duplicate elements and values are their counts.
        Returns an empty dictionary if no duplicates are found.
    """
    counts = Counter(input_list)
    duplicates = {element: count for element, count in counts.items() if count > 1}
    return duplicates

def sample_Q_within_diameter_with_overlap(G, Vp, error_cutoff, overlap, fraction, diam):
    # diam = nx.diameter(G, weight='weight')
    max_iter = 100000  # Maximum number of iterations to avoid infinite loop

    for attempt in range(1, max_iter+1):
        # 1) sample one random reachable node per v
        Q = []
        for v in Vp:
            dist_map = nx.single_source_dijkstra_path_length(G, v, cutoff=float(diam/error_cutoff), weight="weight")
            test_value = random.choice(list(dist_map.keys()))
            if(Q is not None and len(Q) > 0):
                while (test_value == Q[len(Q) - 1]):
                    dist_map = nx.single_source_dijkstra_path_length(G, v, cutoff=float(diam/error_cutoff), weight="weight")
                    test_value = random.choice(list(dist_map.keys()))
                    break
            
            Q.append(test_value)


            # if all(Q[i] != Q[i+1] for i in range(len(Q)-1)):
            #     Q.append(random.choice(list(dist_map.keys())))
            # else:
            #     continue

        # 2) compute overlap
        dup_counts = count_duplicates(Q)
        # extra dups = sum of (count - 1) for each duplicated element
        extra_dups = sum(cnt for cnt in dup_counts.values())
        current_overlap = extra_dups / len(Q) * 100

        # if dup_counts:
        #     print("Duplicate elements in Q and their counts:")
        #     for element, count in dup_counts.items():
        #         print(f"{element}: {count}")
        # else:
        #     print("No duplicate elements found.")

        # print("Extra dups: ", extra_dups)
        # print("Current overlap: ", current_overlap)

        # 3) check if within tolerance
        if current_overlap <= overlap:
            # set_Q = set(Q)
            # Q = random_from_set_no_consecutive(set_Q, n=int(fraction), rng=random.Random())
            return Q
            # while True:
            #     random.shuffle(Q)  # Shuffle Vp to ensure randomness
            #     if all(Q[i] != Q[i+1] for i in range(len(Q)-1)):
            #         return Q

    # print(f"Could not reach {overlap}% overlap after {max_iter} tries.")
    # print(f"Last overlap was {current_overlap:.2f}%, duplicates:", dup_counts)
    # while True:
    #     random.shuffle(Q)  # Shuffle Vp to ensure randomness
    #     if all(Q[i] != Q[i+1] for i in range(len(Q)-1)):
    #         return Q

    print(f"Could not reach {overlap}% overlap after {max_iter} tries.")
    # set_Q = set(Q)
    # Q = random_from_set_no_consecutive(set_Q, n=int(fraction), rng=random.Random())
    return Q


def calculate_stretch(G_example, T, T_new, mst_g, Vp, fraction, owner, error_cutoff, overlap, myNodeCount, diameter_of_G):
    # V is the set of all vertices in the graph G.
    # print("type of vp is", type(Vp))
    V = list(range(myNodeCount))
    V_new = V

    # Requesting nodes Q: randomly select 1/4th of V with the same cardinality as Vp,
    # also ensuring they do not include the owner.
    # available_for_Q = list(set(V) - {owner})
    # Q = random.sample(available_for_Q, len(Vp))

    Q = sample_Q_within_diameter_with_overlap(G_example, Vp, error_cutoff, overlap, fraction, diameter_of_G)

    print("Selected Q = ", Q)

    print("Total number of vertices (n): ", len(V))
    # print("Total vertices (V):", V)
    # print("Fraction used:", fraction)
    # print("Predicted vertices (Vp):", Vp)
    # print("Requesting nodes (Q):", Q)
    # print("\n--- Move Operations ---")

    centers = find_tree_center(T)
    # print("Center(s) of the tree:", centers)

    root = centers[0]
    print("Root node of the final tree T:", root)

    parent = build_parent_dict(T, root)
    parent_arrow = build_parent_dict_arrow(mst_g, root)
    parent_new = build_parent_dict_arrow(T_new, root)

    # Initialize link[u] = None for all nodes u
    link_ = {u: None for u in T.nodes()}
    linkArrow_ = {u: None for u in mst_g.nodes()}
    linkNew_ = {u: None for u in T_new.nodes()}
    
    # Optionally, you might set link[owner] = owner if you want
    # to indicate that the owner points to itself.
    link_[owner] = owner
    linkArrow_[owner] = owner
    linkNew_[owner] = owner
    
    
    # Run publish() from owner
    publish(T, owner, root, parent, link_)
    publish_arrow(mst_g, owner, root, parent_arrow, linkArrow_)
    publish_new(T_new, owner, root, parent_new, linkNew_)
    

    distances_in_G = []
    distances_in_T = []
    distances_in_mst_g = []
    distances_in_T_new = []
    stretches_i = []
    stretches_i_arrow = []
    stretches_i_new = []
    for r in Q:
        # print(f"\nRequest from node {r} ... ")
        d_in_G, d_in_T, d_in_mst_g, d_in_T_new = set_links_for_request(G_example, T, mst_g, T_new, r, parent, parent_arrow, parent_new, link_, linkArrow_, linkNew_, root)
        # d_in_mst_g = set_links_for_request_for_arrow(G_example, mst_g, r, parent_arrow, linkArrow_, root)
        # stretch_i = float(d_in_T) / d_in_G if d_in_G != 0.0 else float('inf')
        # stretch_i_arrow = float(d_in_mst_g) / d_in_G if d_in_G != 0.0 else float('inf')
        distances_in_G.append(d_in_G)
        distances_in_T.append(d_in_T)
        distances_in_mst_g.append(d_in_mst_g)
        distances_in_T_new.append(d_in_T_new)

        # stretches_i.append(stretch_i)
        # stretches_i_arrow.append(stretch_i_arrow)
        # print(f"\nDistance between request node {r} and owner node in T is {d_in_T}, stretch = {stretch_i:.4f}")

        # print("Updated link_ after request:")
        # for node in sorted(T.nodes()):
        #     print(link_)


    sum_of_distances_in_G = sum(distances_in_G)
    sum_of_distances_in_T = sum(distances_in_T)
    sum_of_distances_in_mst_g = sum(distances_in_mst_g)
    sum_of_distances_in_T_new = sum(distances_in_T_new)
    stretch = sum_of_distances_in_T / sum_of_distances_in_G if sum_of_distances_in_G != 0 else float('inf')
    stretch_arrow = sum_of_distances_in_mst_g / sum_of_distances_in_G if sum_of_distances_in_G != 0 else float('inf')
    stretch_new = sum_of_distances_in_T_new / sum_of_distances_in_G if sum_of_distances_in_G != 0 else float('inf')
    # stretch = max(stretches_i) if stretches_i else 0
    # stretch_arrow = max(stretches_i_arrow) if stretches_i_arrow else 0
    # print("Type of distances in G:", type(distances_in_G))
    # print("Type of distances in T:", type(distances_in_T))
    # stretch = max(stretches) if stretches else 0

    GREEN = "\033[92m"
    RESET = "\033[0m"
    SKY_BLUE = "\033[94m"
    MAGENTA = '\033[35m'
    print(f"{GREEN}\nStretch (sum_of_distance_in_T / sum_of_distance_in_G) = {stretch}{RESET}")
    print(f"{SKY_BLUE}\nStretch_Arrow (sum_of_distance_in_mst_g / sum_of_distance_in_G) = {stretch_arrow}{RESET}")
    print(f"{MAGENTA}\nStretch_New (sum_of_distance_in_T_new / sum_of_distance_in_G) = {stretch_new}{RESET}")
    return Q


def calculate_error(Q, Vp, G_example, diameter_of_G, diameter_of_T):
    errors = []
    for req, pred in zip(Q, Vp):
        # Using NetworkX to compute the shortest path length in tree T.
        dist = nx.shortest_path_length(G_example, source=req, target=pred, weight='weight')
        error = dist / diameter_of_G
        errors.append(error)
        # print(f"\nDistance between request node {req} and predicted node {pred} is {dist}, error = {error:.4f}")
    
    print("Diameter of G:", diameter_of_G)
    print("Diameter of T:", diameter_of_T)
    # print("Errors: ", errors)
    total_max_error = max(errors) if errors else 0
    total_min_error = min(errors) if errors else 0
    RED = "\033[91m"
    RESET = "\033[0m"
    print(f"{RED}\nOverall max error (max_i(distance_in_G / diameter_G)) = {total_max_error:.4f}{RESET}")
    print(f"{RED}\nOverall min error (min_i(distance_in_G / diameter_G)) = {total_min_error:.4f}{RESET}")
    return total_max_error, total_min_error

def modify_the_mst_g(mst_g, G_example, S_example):
    removed_nodes_set = set (mst_g.nodes()) - set(S_example)
    H = mst_g.copy()
    candidates = set(removed_nodes_set)
    removed_in_order = list(removed_nodes_set)


    q = deque(v for v in H.nodes if H.degree(v) <= 1 and v in candidates)

    while q:
        v = q.popleft()
        if v not in H or v not in candidates:
            continue  # might have been removed already

        # In a tree, "removable without disconnecting" ⇔ "is a leaf (degree ≤ 1)"
        if H.degree(v) <= 1:
            neighbors = list(H.neighbors(v))
            H.remove_node(v)
            candidates.remove(v)
            removed_in_order.append(v)

            # Some neighbors might have become leaves; if they are candidates, enqueue them
            for u in neighbors:
                if u in H and H.degree(u) <= 1 and u in candidates:
                    q.append(u)

    return H, removed_in_order

def make_G_sub(G_example, removed_vertices_for_subgraph):
    H = G_example.copy()
    candidates = set(removed_vertices_for_subgraph)
    removed_in_order= []

    # Initialize queue with all current leaves that are in candidates
    q = deque(v for v in H.nodes if H.degree(v) <= 1 and v in candidates)

    while q:
        v = q.popleft()
        if v not in H or v not in candidates:
            continue

        # Don't delete the last node
        if H.number_of_nodes() <= 1:
            break

        # Safe to remove leaf
        if H.degree(v) <= 1:
            neighbors = list(H.neighbors(v))
            H.remove_node(v)
            candidates.remove(v)
            removed_in_order.append(v)

            # Check neighbors — they may have become leaves
            for u in neighbors:
                if u in H and H.degree(u) <= 1 and u in candidates:
                    q.append(u)

    return H, removed_in_order

def main(fraction, network_file_name, error_cutoff, overlap):

    G_example = load_graph(network_file_name)

    match3 = re.search(r'(\d+)', network_file_name)
    if match3:
        myNodeCount = int(match3.group(1))

    mst_filename = None
    for filename in os.listdir(os.path.join('graphs_new', 'mst')):
        if str(myNodeCount) in filename:
            mst_filename = filename
            break

    mst_g = load_mst(mst_filename)

    match = re.search(r'diameter(\d+)', mst_filename)
    if match:
        diameter_of_mst_g = int(match.group(1))

    match2 = re.search(r'diameter(\d+)', network_file_name)
    if match2:
        diameter_of_G = int(match2.group(1))

    # # Contrcut MST_g of Graph G_example for Arrow protocol
    # mst_g = nx.minimum_spanning_tree(G_example, weight='weight')
    # # see_graph(mst_g)
    # diameter_of_mst_g = nx.diameter(mst_g, weight='weight')

    # diameter_of_G = nx.diameter(G_example, weight='weight')
    print("Diameter of G_example:", diameter_of_G)
    print("Diameter of MST:", diameter_of_mst_g)

    # while True:
    S_example, Vp, owner = choose_steiner_set(G_example, fraction, diameter_of_G, myNodeCount)
    print("Randomly chosen Predicted Vertices (Vp):", Vp)
    print("Owner node:", owner)
    print("Steiner set S:", S_example)

    # Select S_example, Vp, owner such that only when diameter of G_sub <= diameter of G/4
    # removed_vertices_for_subgraph = set(G_example.nodes()) - set(S_example)

    # G_sub, removed_nodes = make_G_sub(G_example, removed_vertices_for_subgraph)
    # see_graph(G_sub)
    # diameter_of_G_sub = nx.diameter(G_sub, weight='weight')
    # print("Yaha Diameter of G_sub:", diameter_of_G_sub)
    # print("Yaha Diameter of G/3:", diameter_of_G/3)
        # Compute Steiner tree
        # if diameter_of_G_sub <= diameter_of_G/3:
        #     break
    
    T_H = steiner_tree(G_example, S_example)
    # see_graph(T_H)

    modified_mst, actually_removed = modify_the_mst_g(mst_g, G_example, S_example)
    # see_graph(modified_mst)

    PINK   = "\033[95m"  # Magenta / Pink
    PURPLE = "\033[35m"  # Purple
    YELLOW = "\033[93m"  # Bright Yellow
    RESET  = "\033[0m"
    print(f"{PINK}\nDiameter of modified MST = {nx.diameter(modified_mst, weight='weight')}{RESET}")
    print(f"{YELLOW}\nDiameter of Steiner tree = {nx.diameter(T_H, weight='weight')}{RESET}")

    # print("Original nodes:", mst_g.number_of_nodes())
    # print("Modified nodes:", modified_mst.number_of_nodes())
    # print("Actually removed (leaves-only, iteratively):", sorted(actually_removed))

    # see_graph(T_H)

    # Print edges of the resulting Final tree
    # print("Final Tree edges:")
    # for (u, v, data) in T_H.edges(data=True):
    #     print(f"{u} - {v}, weight = {v['weight'] if isinstance(v, dict) else v}")
    
    # Compute Final tree T
    T = augment_steiner_tree_with_remaining_vertices(G_example, T_H, myNodeCount)
    # T = augment_tree_with_remaining_nodes(G_example, T_H, weight="weight")
    # see_graph(T)
    T_new = augment_modified_mst_with_remaining_vertices(G_example, modified_mst, myNodeCount)
    # see_graph(T_new)
    # diameter_of_T = nx.diameter(T, weight='weight')
    # diameter_of_T_new = nx.diameter(T_new, weight='weight')
    # print("Diameter of T:", diameter_of_T)
    # print("Diameter of T_new:", diameter_of_T_new)
        

    # verifying the edge weights by printing them
    # for u, v, weight in T.edges(data='weight'):
    #     print(f"Edge ({u}, {v}) has weight: {weight}")

    # see_graph(T)
    overlap = int(overlap)
    Q = calculate_stretch(G_example, T, T_new, mst_g, Vp, fraction, owner, error_cutoff, overlap, myNodeCount, diameter_of_G)
    print("Size of Q:", len(Q))
    # exit()

    # calculate_stretch(G_example, mst_g, Vp, fraction, owner, error_cutoff, overlap)


    diameter_of_T = nx.diameter(T, weight='weight')
    diameter_of_T_new = nx.diameter(T_new, weight='weight')

    total_max_error,  total_min_error = calculate_error(Q, Vp, G_example, diameter_of_G, diameter_of_T)

    return total_max_error,  total_min_error


if __name__ == "__main__":
    errros_to_plot = []
    p = argparse.ArgumentParser(description="Running the experiment with different fractions of predicted nodes ... ")
    p.add_argument(
        "--fraction",
        type=float,
        required=True,
        help="The fraction of nodes to pick as Vp (e.g. 0.0625, 0.125, 0.25, 0.5)"
    )
    p.add_argument(
        "--network",
        required=True,
        type=str,
        help="The network file name to run an algorithm on(e.g. '256random_diameter71test.edgelist')"
    )

    p.add_argument(
        "-c",
        "--cutoff",
        default=1.0,
        type=float,
        help="Cutoff parameter for the error value (implies the error value cannot go beyond this cutoff)"
    )

    p.add_argument(
        "-o",
        "--overlap",
        default=100,
        type=int,
        help="Overlap of the actual nodes requesting for the object (in percentage)"
    )
    args = p.parse_args()
    main(args.fraction, args.network, args.cutoff, args.overlap)

    


    
    

