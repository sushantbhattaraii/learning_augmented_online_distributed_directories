import networkx as nx
import copy
from collections import deque


def find_tree_center(T):
    """
    Finds the center of the tree T using iterative leaf removal.
    Returns a list of center nodes (one or two nodes).
    """
    # Work on a copy of T so the original graph is not modified.
    H = copy.deepcopy(T)
    leaves = [node for node in H.nodes() if H.degree(node) <= 1]
    
    while H.number_of_nodes() > 2:
        new_leaves = []
        for leaf in leaves:
            for neighbor in list(H.neighbors(leaf)):
                H.remove_edge(leaf, neighbor)
                if H.degree(neighbor) == 1:
                    new_leaves.append(neighbor)
            H.remove_node(leaf)
        leaves = new_leaves
    return list(H.nodes())