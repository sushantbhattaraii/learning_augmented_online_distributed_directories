import networkx as nx
from collections import deque
from typing import Iterable, Tuple, List, Set

def prune_nodes_keep_connected(mst_g: nx.Graph, removed_nodes_set: Iterable[int]):
    """
    Remove as many nodes from removed_nodes_set as possible without disconnecting the tree.
    Only leaves get removed; as leaves are pruned, new leaves may become removable.
    
    Parameters
    ----------
    mst_g : nx.Graph
        A tree (your MST). Assumes it's connected.
    removed_nodes_set : Iterable[int]
        Candidate node IDs to try removing.

    Returns
    -------
    modified_mst : nx.Graph
        The pruned tree (still connected unless it had <=1 nodes).
    removed_in_order : List[int]
        Nodes actually removed, in the order of removal.
    """
    # Work on a copy so we don't mutate the input
    H = mst_g.copy()
    candidates: Set[int] = set(removed_nodes_set)
    removed_in_order: List[int] = []

    # Initialize queue with all current leaves that are in candidates
    q = deque(v for v in H.nodes if H.degree(v) <= 1 and v in candidates)

    while q:
        v = q.popleft()
        if v not in H or v not in candidates:
            continue  # might have been removed already

        # Don't remove the last remaining node (keep graph non-empty & "connected")
        if H.number_of_nodes() <= 1:
            break

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

# --------------------------
# Example usage
# if __name__ == "__main__":
#     # Build a sample MST-like tree
#     T = nx.random_tree(128, seed=42)  # nodes 0..127
#     removed = {0, 1, 2, 3, 10, 15, 31, 64, 90, 127}  # example subset

#     modified_mst, actually_removed = prune_nodes_keep_connected(T, removed)

#     print("Original nodes:", T.number_of_nodes())
#     print("Modified nodes:", modified_mst.number_of_nodes())
#     print("Requested to remove:", sorted(removed))
#     print("Actually removed (leaves-only, iteratively):", sorted(actually_removed))
#     # Sanity check: still connected
#     assert nx.is_connected(modified_mst)
