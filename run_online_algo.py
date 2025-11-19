from collections import defaultdict, deque
import heapq, random

# -----------------------------
# Basic weighted graph
# -----------------------------
class Graph:
    def __init__(self):
        self.adj = defaultdict(dict)  # u -> {v: w}

    def add_edge(self, u, v, w=1.0):
        self.adj[u][v] = w
        self.adj[v][u] = w

    def remove_edge(self, u, v):
        if v in self.adj[u]: del self.adj[u][v]
        if u in self.adj[v]: del self.adj[v][u]

    def update_weight(self, u, v, w):
        if v in self.adj[u]:
            self.adj[u][v] = w
            self.adj[v][u] = w
        else:
            self.add_edge(u, v, w)

    def nodes(self):
        return list(self.adj.keys())

    def edges(self):
        seen = set()
        for u in self.adj:
            for v,w in self.adj[u].items():
                if (v,u) not in seen:
                    seen.add((u,v))
                    yield u,v,w

# -----------------------------
# Dijkstra (returns dist, prev)
# -----------------------------
def dijkstra(graph: Graph, src):
    dist = {src: 0.0}
    prev = {}
    pq = [(0.0, src)]
    while pq:
        d,u = heapq.heappop(pq)
        if d != dist[u]: 
            continue
        for v,w in graph.adj[u].items():
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, prev

def shortest_path(graph: Graph, s, t):
    dist, prev = dijkstra(graph, s)
    if t not in dist: 
        return float('inf'), []
    # Reconstruct
    path = [t]
    while path[-1] != s:
        path.append(prev[path[-1]])
    path.reverse()
    return dist[t], path

# -----------------------------
# Landmark-based Dynamic Distance Oracle
# -----------------------------
class LandmarkOracle:
    """
    Practical (1+epsilon-ish) approximate distances:
    dist(u,v) ~ min_L [dist(u,L) + dist(L,v)] and a small direct-Dijkstra fallback.
    Updates: on edge change, recompute SSSP from only a subset of landmarks touching endpoints,
    with an option to 'rebuild' periodically for robustness.
    """
    def __init__(self, graph: Graph, num_landmarks=16, seed=42):
        self.g = graph
        self.rng = random.Random(seed)
        self.num_landmarks = num_landmarks
        self.landmarks = []
        self.ldist = {}   # landmark -> {node: dist}
        self._choose_landmarks()
        self._build_all()

        # Simple rebuild schedule knob
        self.updates_since_full = 0
        self.full_rebuild_period = 50

    def _choose_landmarks(self):
        nodes = self.g.nodes()
        if not nodes:
            self.landmarks = []
            return
        # Choose a mix of random nodes and high-degree nodes
        high_deg = sorted(nodes, key=lambda u: len(self.g.adj[u]), reverse=True)[: self.num_landmarks//2]
        rnd = self.rng.sample(nodes, min(len(nodes), self.num_landmarks - len(high_deg)))
        lm = list(dict.fromkeys(high_deg + rnd))  # dedupe preserving order
        # pad if needed
        while len(lm) < min(len(nodes), self.num_landmarks):
            lm.append(self.rng.choice(nodes))
        self.landmarks = lm[:min(len(nodes), self.num_landmarks)]

    def _build_all(self):
        self.ldist = {}
        for L in self.landmarks:
            self.ldist[L], _ = dijkstra(self.g, L)

    def _rebuild_subset(self, candidates):
        # Recompute from a subset of landmarks (those closest to 'candidates')
        if not self.landmarks: 
            return
        # Pick k_subset landmarks nearest to candidates (by current distances)
        k_subset = max(4, len(self.landmarks)//4)
        best = set()
        for c in candidates:
            # rank landmarks by current distance to c (fallback to large if unknown)
            ranked = sorted(self.landmarks, key=lambda L: self.ldist.get(L, {}).get(c, float('inf')))
            for L in ranked[: max(1, k_subset//len(candidates)) ]:
                best.add(L)
        if not best:
            best = set(self.landmarks[:k_subset])
        for L in best:
            self.ldist[L], _ = dijkstra(self.g, L)

    def update_add_edge(self, u, v, w):
        self.g.add_edge(u, v, w)
        self.updates_since_full += 1
        self._rebuild_subset({u, v})
        if self.updates_since_full >= self.full_rebuild_period:
            self._choose_landmarks()
            self._build_all()
            self.updates_since_full = 0

    def update_remove_edge(self, u, v):
        self.g.remove_edge(u, v)
        self.updates_since_full += 1
        self._rebuild_subset({u, v})
        if self.updates_since_full >= self.full_rebuild_period:
            self._choose_landmarks()
            self._build_all()
            self.updates_since_full = 0

    def update_weight(self, u, v, w):
        self.g.update_weight(u, v, w)
        self.updates_since_full += 1
        self._rebuild_subset({u, v})
        if self.updates_since_full >= self.full_rebuild_period:
            self._choose_landmarks()
            self._build_all()
            self.updates_since_full = 0

    def query_distance(self, s, t, exact_if_close=True, exact_cutoff=8):
        if s == t: 
            return 0.0
        # quick exact for very local pairs (optional)
        if exact_if_close:
            # bounded Dijkstra by hops
            dist, prev = self._bounded_dijkstra(s, hop_limit=exact_cutoff)
            if t in dist:
                return dist[t]
        # landmark combo
        best = float('inf')
        for L in self.landmarks:
            dL = self.ldist.get(L, {})
            du = dL.get(s, float('inf'))
            dv = dL.get(t, float('inf'))
            cand = du + dv
            if cand < best:
                best = cand
        # last-resort: exact dijkstra if unreachable via landmarks (disconnected or poor coverage)
        if best == float('inf'):
            best, _path = shortest_path(self.g, s, t)
        return best

    def query_path(self, s, t):
        """Heuristic path reconstruction: try via best landmark, else exact."""
        if s == t: return [s]
        best = (float('inf'), None)
        for L in self.landmarks:
            dL = self.ldist.get(L, {})
            du = dL.get(s, float('inf'))
            dv = dL.get(t, float('inf'))
            cand = du + dv
            if cand < best[0]:
                best = (cand, L)
        if best[1] is None or best[0] == float('inf'):
            _d, path = shortest_path(self.g, s, t)
            return path
        # stitch s->L and t->L using Dijkstra trees from L
        L = best[1]
        # We don't store parents; recompute once from L for a clean path
        _dist, prev = dijkstra(self.g, L)

        def path_to(L, x):
            if x not in prev and x != L:
                return []
            p = [x]
            while p[-1] != L:
                p.append(prev[p[-1]])
            p.reverse()
            return p

        path_sL = path_to(L, s)
        path_tL = path_to(L, t)
        if not path_sL or not path_tL:
            _d, exact = shortest_path(self.g, s, t)
            return exact
        # s->L then L->t (reverse t->L)
        return path_sL + list(reversed(path_tL))[1:]

    def _bounded_dijkstra(self, src, hop_limit=8):
        dist = {src: 0.0}
        prev = {}
        pq = [(0.0, src, 0)]
        while pq:
            d,u,hops = heapq.heappop(pq)
            if d != dist[u]: 
                continue
            if hops == hop_limit:
                continue
            for v,w in self.g.adj[u].items():
                nd = d + w
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v, hops+1))
        return dist, prev

# -----------------------------
# Algorithm 1-style Steiner maintainer
# -----------------------------
class SteinerMaintainer:
    """
    Builds an approximate Steiner tree for terminal set S:
      1) metric closure over S using distance oracle,
      2) MST on S,
      3) expand each MST edge into a path in G,
      4) prune non-terminal leaves.
    """
    def __init__(self, graph: Graph, oracle: LandmarkOracle):
        self.g = graph
        self.oracle = oracle

    def build_metric_closure(self, terminals):
        # Complete graph on terminals with weights = oracle distances
        W = {}
        terms = list(terminals)
        for i in range(len(terms)):
            for j in range(i+1, len(terms)):
                u,v = terms[i], terms[j]
                d = self.oracle.query_distance(u, v)
                W[(u,v)] = d
        return W

    def mst_on_terminals(self, terminals, weights):
        # Kruskal on complete graph with given edge weights
        parent = {}
        rank = {}
        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(a,b):
            a,b = find(a), find(b)
            if a==b: return False
            ra, rb = rank.get(a,0), rank.get(b,0)
            if ra < rb:
                parent[a] = b
            elif rb < ra:
                parent[b] = a
            else:
                parent[b] = a
                rank[a] = ra+1
            return True

        edges = sorted(weights.items(), key=lambda kv: kv[1])  # ((u,v), w)
        T = []
        for (u,v), w in edges:
            if union(u, v):
                T.append((u,v,w))
        return T

    def expand_paths(self, mst_edges):
        # Union of edges along oracle-chosen paths between terminals in MST
        steiner_edges = set()
        for u, v, _w in mst_edges:
            path = self.oracle.query_path(u, v)
            for a,b in zip(path, path[1:]):
                # normalize (small,big)
                e = (a,b) if a <= b else (b,a)
                steiner_edges.add(e)
        return steiner_edges

    def prune_nonterminal_leaves(self, edges, terminals):
        # Build degree map and iteratively remove non-terminal leaves
        adj = defaultdict(set)
        for u,v in edges:
            adj[u].add(v); adj[v].add(u)
        deg = {u: len(adj[u]) for u in adj}
        q = deque([u for u in adj if deg[u]==1 and u not in terminals])

        removed = set()
        while q:
            x = q.popleft()
            if x in removed: 
                continue
            removed.add(x)
            for nb in list(adj[x]):
                adj[nb].remove(x)
                deg[nb] -= 1
                if deg[nb]==1 and nb not in terminals:
                    q.append(nb)
            adj[x].clear()
            deg[x] = 0

        pruned = set()
        for u in adj:
            for v in adj[u]:
                if u <= v:
                    pruned.add((u,v))
        return pruned

    def build_steiner_tree(self, terminals):
        if len(terminals) <= 1:
            return set()
        weights = self.build_metric_closure(terminals)
        mst = self.mst_on_terminals(terminals, weights)
        raw_edges = self.expand_paths(mst)
        pruned = self.prune_nonterminal_leaves(raw_edges, set(terminals))
        return pruned

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Build a random connected graph
    g = Graph()
    n = 60
    p = 0.06
    rng = random.Random(0)
    for i in range(n):
        g.adj[i]  # ensure node exists
    # Random Erdos-Renyi with positive weights
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < p:
                w = 1.0 + rng.random()*9.0
                g.add_edge(i, j, w)
    # Ensure connectivity: connect a random spanning tree if needed
    # (Simple union-find-free: chain connect)
    for i in range(n-1):
        if i+1 not in g.adj[i]:
            g.add_edge(i, i+1, 1.0 + rng.random()*2.0)

    # Make terminals
    terminals = rng.sample(range(n), 8)

    # Create dynamic distance oracle
    oracle = LandmarkOracle(g, num_landmarks=16, seed=1)

    # Build Steiner tree (Algorithm-1 style)
    maint = SteinerMaintainer(g, oracle)
    steiner_edges = maint.build_steiner_tree(terminals)
    total_cost = sum(min(g.adj[u][v], g.adj[v][u]) for u,v in steiner_edges)  # undirected

    print("Terminals:", sorted(terminals))
    print("Steiner edges count:", len(steiner_edges))
    print("Approx. Steiner tree cost:", round(total_cost, 3))

    # Do a dynamic update and rebuild
    # (simulate edge change; oracle updates incrementally)
    u, v = rng.randrange(n), rng.randrange(n)
    g.update_weight(u, v, 0.5 + rng.random())
    oracle.update_weight(u, v, g.adj[u][v])

    steiner_edges2 = maint.build_steiner_tree(terminals)
    total_cost2 = sum(min(g.adj[u][v], g.adj[v][u]) for u,v in steiner_edges2)
    print("After 1 update -> cost:", round(total_cost2, 3))
