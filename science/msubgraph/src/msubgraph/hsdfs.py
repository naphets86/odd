"""
HS-DFS: Hierarchical Subgraph Depth-First Search  (Algorithm 2).

Phase 1 – Meta-Graph Construction:
    For every ordered pair (G_i, G_j) with i ≠ j, the Subgraph Algorithm
    oracle S is invoked on the adjacency matrices A_i and A_j.  If
    G_i is subgraph-isomorphic to G_j, the directed edge G_i → G_j is
    added to the meta-graph M.

Phase 2 – DFS Forest Computation:
    Standard Depth-First Search is executed on M.  Every vertex receives
    discovery and finish timestamps; every edge is classified as one of
    TREE / BACK / FORWARD / CROSS.

The Subgraph Algorithm is imported from the `subgraph` package and used
as a black-box oracle — it is not re-implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from subgraph import Subgraph as _SubgraphOracle

# ---------------------------------------------------------------------------
# Oracle decision values that indicate "first argument ⊆ second argument"
# ---------------------------------------------------------------------------
# compare_graphs(A_i, A_j) returns:
#   "keep_B"       → A_j ⊃ A_i  (only G_i ⊆ G_j)
#   "keep_A"       → A_i ⊃ A_j  (only G_j ⊆ G_i, not relevant here)
#   "equal_keep_B" → mutual containment, B has more edges
#   "equal_keep_A" → mutual containment, A has more or equal edges
#   "keep_both"    → neither direction
_I_SUBGRAPH_OF_J: frozenset[str] = frozenset(
    {"keep_B", "equal_keep_B", "equal_keep_A"}
)


# ---------------------------------------------------------------------------
# EdgeClass
# ---------------------------------------------------------------------------

class EdgeClass(Enum):
    """DFS edge classification for a directed meta-graph edge."""

    TREE = "tree"
    """Discovery edge: v was WHITE when (u, v) was first explored."""

    BACK = "back"
    """Ancestor edge: v is GRAY (an ancestor of u in the DFS tree)."""

    FORWARD = "forward"
    """Descendant shortcut: v is BLACK and d[u] < d[v]."""

    CROSS = "cross"
    """Lateral edge: v is BLACK and d[u] > d[v]."""


# ---------------------------------------------------------------------------
# MetaGraph
# ---------------------------------------------------------------------------

@dataclass
class MetaGraph:
    """
    Directed graph whose nodes are graph indices 0 … n-1 and whose edges
    encode subgraph-containment relationships.
    """

    n: int
    """Number of nodes (= number of graphs in the collection)."""

    _adj: Dict[int, List[int]] = field(default_factory=dict, repr=False, init=False)

    def __post_init__(self) -> None:
        self._adj = {i: [] for i in range(self.n)}

    # ------------------------------------------------------------------
    def add_edge(self, src: int, dst: int) -> None:
        """Add directed edge src → dst (idempotent)."""
        if dst not in self._adj[src]:
            self._adj[src].append(dst)

    def neighbors(self, u: int) -> List[int]:
        """Return the list of successors of node u."""
        return list(self._adj[u])

    @property
    def edges(self) -> List[Tuple[int, int]]:
        """Return all directed edges as (src, dst) pairs."""
        return [
            (src, dst)
            for src, nbrs in self._adj.items()
            for dst in nbrs
        ]


# ---------------------------------------------------------------------------
# DFSResult
# ---------------------------------------------------------------------------

@dataclass
class DFSResult:
    """Output of the DFS Forest Computation phase."""

    discovery: Dict[int, int]
    """d[u]: discovery timestamp of node u."""

    finish: Dict[int, int]
    """f[u]: finish timestamp of node u."""

    parent: Dict[int, Optional[int]]
    """parent[u]: DFS-tree parent of u, or None for forest roots."""

    edge_class: Dict[Tuple[int, int], EdgeClass]
    """Classification of each meta-graph edge."""

    forest_roots: List[int]
    """Nodes from which a new DFS tree was started (topological order of roots)."""


# ---------------------------------------------------------------------------
# HSDFSResult
# ---------------------------------------------------------------------------

@dataclass
class HSDFSResult:
    """Complete output of the HS-DFS algorithm."""

    meta_graph: MetaGraph
    """The constructed subgraph-containment meta-graph."""

    dfs: DFSResult
    """The DFS forest computed over the meta-graph."""


# ---------------------------------------------------------------------------
# HSDFS  – the main algorithm
# ---------------------------------------------------------------------------

class HSDFS:
    """
    Hierarchical Subgraph Depth-First Search (Algorithm 2).

    Parameters
    ----------
    graphs:
        Collection G = {G_1, …, G_N} of graphs, each represented as a
        square NumPy adjacency matrix.

    Example
    -------
    >>> import numpy as np
    >>> from msubgraph import HSDFS, EdgeClass
    >>> A = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]])
    >>> B = np.array([[0,1,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]])
    >>> result = HSDFS([A, B]).run()
    >>> (0, 1) in result.meta_graph.edges   # A ⊆ B
    True
    """

    def __init__(self, graphs: List[np.ndarray]) -> None:
        self._graphs = graphs
        self._oracle = _SubgraphOracle()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> HSDFSResult:
        """
        Execute both phases and return the complete result.

        Returns
        -------
        HSDFSResult
            Contains the meta-graph and the DFS forest.
        """
        meta = self._build_meta_graph()
        dfs = self._dfs_forest(meta)
        return HSDFSResult(meta_graph=meta, dfs=dfs)

    # ------------------------------------------------------------------
    # Phase 1: Meta-Graph Construction
    # ------------------------------------------------------------------

    def _is_subgraph(self, i: int, j: int) -> bool:
        """
        Oracle call: True iff G_i is subgraph-isomorphic to G_j.

        Uses Subgraph.compare_graphs(A_i, A_j) and maps the returned
        decision string to a boolean.
        """
        decision, _ = self._oracle.compare_graphs(self._graphs[i], self._graphs[j])
        return decision in _I_SUBGRAPH_OF_J

    def _build_meta_graph(self) -> MetaGraph:
        """
        Phase 1 – Meta-Graph Construction.

        Iterates over all ordered pairs (i, j) with i ≠ j and invokes the
        Subgraph Algorithm oracle.  For every pair where G_i ⊆ G_j the
        directed edge i → j is inserted into the meta-graph.

        Time complexity: Θ(N² · T_S(n)) = O(n⁵) with T_S = O(n³), N = n.
        """
        n = len(self._graphs)
        meta = MetaGraph(n=n)
        for i in range(n):
            for j in range(n):
                if i != j and self._is_subgraph(i, j):
                    meta.add_edge(i, j)
        return meta

    # ------------------------------------------------------------------
    # Phase 2: DFS Forest Computation
    # ------------------------------------------------------------------

    @staticmethod
    def _dfs_forest(meta: MetaGraph) -> DFSResult:
        """
        Phase 2 – DFS Forest Computation.

        Runs standard DFS on the meta-graph, assigning discovery/finish
        timestamps and classifying every edge as TREE, BACK, FORWARD, or
        CROSS.

        Time complexity: Θ(N + |E|) = O(n²) in the worst case.
        """
        _WHITE, _GRAY, _BLACK = 0, 1, 2

        color: Dict[int, int] = {i: _WHITE for i in range(meta.n)}
        parent: Dict[int, Optional[int]] = {i: None for i in range(meta.n)}
        d: Dict[int, int] = {}
        f: Dict[int, int] = {}
        chi: Dict[Tuple[int, int], EdgeClass] = {}
        time = [0]
        roots: List[int] = []

        def _classify(u: int, v: int) -> EdgeClass:
            if color[v] == _WHITE:
                return EdgeClass.TREE
            if color[v] == _GRAY:
                return EdgeClass.BACK
            return EdgeClass.FORWARD if d[u] < d[v] else EdgeClass.CROSS

        def _visit(u: int) -> None:
            color[u] = _GRAY
            time[0] += 1
            d[u] = time[0]
            for v in meta.neighbors(u):
                chi[(u, v)] = _classify(u, v)
                if color[v] == _WHITE:
                    parent[v] = u
                    _visit(v)
            color[u] = _BLACK
            time[0] += 1
            f[u] = time[0]

        for i in range(meta.n):
            if color[i] == _WHITE:
                roots.append(i)
                _visit(i)

        return DFSResult(
            discovery=d,
            finish=f,
            parent=parent,
            edge_class=chi,
            forest_roots=roots,
        )
