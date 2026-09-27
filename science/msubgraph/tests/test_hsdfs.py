"""
Tests for HSDFS – Algorithm 2: Hierarchical Subgraph Depth-First Search.

Test organisation
-----------------
TestEdgeClass        – EdgeClass enum values
TestDFSForestUnit    – DFS phase tested directly via MetaGraph construction,
                       covering all four edge types and forest properties
TestHSDFSOracle      – _is_subgraph / _build_meta_graph with real adjacency
                       matrices and the Subgraph oracle
TestHSDFSRun         – full run() integration tests including the worked
                       example from the paper
"""

import numpy as np
import pytest

from msubgraph import HSDFS, HSDFSResult, MetaGraph, DFSResult, EdgeClass


# ---------------------------------------------------------------------------
# Shared adjacency matrices
# ---------------------------------------------------------------------------

# 2×2 path: 0 → 1
A2 = np.array([[0, 1], [0, 0]], dtype=int)

# 2×2 reversed edge: 1 → 0
A2_rev = np.array([[0, 0], [1, 0]], dtype=int)

# 3×3 directed path: 0 → 1 → 2
A3 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=int)

# 4×4 directed path: 0 → 1 → 2 → 3  (paper example, matrix A)
A4 = np.array(
    [[0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1],
     [0, 0, 0, 0]],
    dtype=int,
)

# 4×4 path + extra edge 0 → 2  (paper example, matrix B)
B4 = np.array(
    [[0, 1, 1, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1],
     [0, 0, 0, 0]],
    dtype=int,
)

# 3×3 graph incomparable with A2  (different row-signature patterns)
A3_incomparable = np.array(
    [[0, 0, 1],
     [0, 0, 1],
     [0, 0, 0]],
    dtype=int,
)


# ---------------------------------------------------------------------------
# TestEdgeClass
# ---------------------------------------------------------------------------

class TestEdgeClass:
    def test_all_values_exist(self):
        assert EdgeClass.TREE.value == "tree"
        assert EdgeClass.BACK.value == "back"
        assert EdgeClass.FORWARD.value == "forward"
        assert EdgeClass.CROSS.value == "cross"

    def test_enum_members(self):
        members = {e.name for e in EdgeClass}
        assert members == {"TREE", "BACK", "FORWARD", "CROSS"}


# ---------------------------------------------------------------------------
# TestDFSForestUnit  –  DFS tested independently of the oracle
# ---------------------------------------------------------------------------

class TestDFSForestUnit:
    """All DFS tests build MetaGraph directly and call HSDFS._dfs_forest."""

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _run(n: int, edge_list: list) -> DFSResult:
        meta = MetaGraph(n=n)
        for src, dst in edge_list:
            meta.add_edge(src, dst)
        return HSDFS._dfs_forest(meta)

    # ------------------------------------------------------------------ basic

    def test_single_node_no_edges(self):
        result = self._run(1, [])
        assert result.discovery[0] == 1
        assert result.finish[0] == 2
        assert result.parent[0] is None
        assert result.forest_roots == [0]
        assert result.edge_class == {}

    def test_two_nodes_no_edges(self):
        result = self._run(2, [])
        assert result.forest_roots == [0, 1]
        assert result.parent[0] is None
        assert result.parent[1] is None

    def test_three_nodes_no_edges_all_are_roots(self):
        result = self._run(3, [])
        assert len(result.forest_roots) == 3

    # ------------------------------------------------------------------ timestamps

    def test_timestamps_monotone(self):
        result = self._run(3, [(0, 1), (1, 2)])
        times = sorted(list(result.discovery.values()) + list(result.finish.values()))
        assert times == list(range(1, 7))

    def test_parenthesization_property(self):
        """For every tree edge u→v: d[u] < d[v] < f[v] < f[u]."""
        result = self._run(3, [(0, 1), (1, 2)])
        d, f = result.discovery, result.finish
        # 0 is ancestor of 1 and 2; 1 is ancestor of 2
        assert d[0] < d[1] < f[1] < f[0]
        assert d[1] < d[2] < f[2] < f[1]

    def test_discovery_before_finish(self):
        result = self._run(4, [(0, 1), (1, 2), (2, 3)])
        for i in range(4):
            assert result.discovery[i] < result.finish[i]

    # ------------------------------------------------------------------ TREE edges

    def test_linear_chain_all_tree_edges(self):
        result = self._run(3, [(0, 1), (1, 2)])
        assert result.edge_class[(0, 1)] == EdgeClass.TREE
        assert result.edge_class[(1, 2)] == EdgeClass.TREE

    def test_tree_edge_sets_parent(self):
        result = self._run(3, [(0, 1), (1, 2)])
        assert result.parent[1] == 0
        assert result.parent[2] == 1

    def test_root_has_no_parent(self):
        result = self._run(3, [(0, 1), (1, 2)])
        assert result.parent[0] is None

    # ------------------------------------------------------------------ BACK edges

    def test_back_edge_detected_in_cycle(self):
        # Meta-graph: 0→1, 1→0  (two identical graphs → mutual containment)
        result = self._run(2, [(0, 1), (1, 0)])
        assert result.edge_class[(0, 1)] == EdgeClass.TREE
        assert result.edge_class[(1, 0)] == EdgeClass.BACK

    def test_back_edge_self_loop(self):
        # Self-loop is always a back edge (node is GRAY when its own edge processed)
        result = self._run(2, [(0, 0)])
        assert result.edge_class[(0, 0)] == EdgeClass.BACK

    def test_larger_cycle_back_edge(self):
        # 0→1→2→0  – edge 2→0 must be BACK
        result = self._run(3, [(0, 1), (1, 2), (2, 0)])
        assert result.edge_class[(2, 0)] == EdgeClass.BACK

    # ------------------------------------------------------------------ FORWARD edges

    def test_forward_edge_transitive_containment(self):
        # Chain 0→1→2 plus shortcut 0→2
        # DFS from 0: visits 1, then 2 (tree). Then when processing 0's edge
        # to 2, node 2 is already BLACK and d[0]=1 < d[2]=3 → FORWARD.
        result = self._run(3, [(0, 1), (1, 2), (0, 2)])
        assert result.edge_class[(0, 2)] == EdgeClass.FORWARD

    def test_forward_edge_four_node_chain_with_shortcut(self):
        # 0→1→2→3 and 0→3
        result = self._run(4, [(0, 1), (1, 2), (2, 3), (0, 3)])
        assert result.edge_class[(0, 3)] == EdgeClass.FORWARD

    # ------------------------------------------------------------------ CROSS edges

    def test_cross_edge_between_two_dfs_trees(self):
        # Meta-graph: 0→2, 1→2 with no edge 0→1 or 1→0.
        # DFS finishes 2 in the first tree (from 0), then starts at 1.
        # 1→2: 2 is BLACK and d[1] > d[2] → CROSS.
        result = self._run(3, [(0, 2), (1, 2)])
        assert result.edge_class[(0, 2)] == EdgeClass.TREE
        assert result.edge_class[(1, 2)] == EdgeClass.CROSS
        assert result.forest_roots == [0, 1]

    def test_cross_edge_four_node_graph(self):
        # 0→1, 0→2, 2→3, 1→3
        # DFS from 0: tree 0→1, then 0→2, then 2→3.
        # When visiting 1's neighbor 3: 3 is BLACK and d[1]<d[3] … actually
        # check carefully: 0(d=1)→1(d=2), then 1→3(d=3,f=4), f[1]=5, then
        # 0→2(d=6), 2→3: 3 is BLACK, d[2]=6 > d[3]=3 → CROSS.
        result = self._run(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
        assert result.edge_class[(2, 3)] == EdgeClass.CROSS

    # ------------------------------------------------------------------ forest structure

    def test_two_independent_tree_forest(self):
        # 0→1 and 2→3, no connection
        result = self._run(4, [(0, 1), (2, 3)])
        assert set(result.forest_roots) == {0, 2}

    def test_all_nodes_have_timestamps(self):
        result = self._run(4, [(0, 1), (2, 3)])
        for i in range(4):
            assert i in result.discovery
            assert i in result.finish

    def test_topological_order_from_finish_times(self):
        # Acyclic chain 0→1→2: reverse finish order = 0,1,2
        result = self._run(3, [(0, 1), (1, 2)])
        f = result.finish
        assert f[0] > f[1] > f[2]


# ---------------------------------------------------------------------------
# TestHSDFSOracle  –  oracle-level tests with real adjacency matrices
# ---------------------------------------------------------------------------

class TestHSDFSOracle:
    """Tests for _is_subgraph and _build_meta_graph using the Subgraph oracle."""

    # ------------------------------------------------------------------ _is_subgraph

    def test_is_subgraph_a2_in_a3(self):
        # 2-node path ⊆ 3-node path
        hsdfs = HSDFS([A2, A3])
        assert hsdfs._is_subgraph(0, 1) is True

    def test_is_not_subgraph_a3_in_a2(self):
        # 3-node path is larger → not contained in 2-node path
        hsdfs = HSDFS([A2, A3])
        assert hsdfs._is_subgraph(1, 0) is False

    def test_is_subgraph_a2_in_a4(self):
        # 2-node path ⊆ 4-node path
        hsdfs = HSDFS([A2, A4])
        assert hsdfs._is_subgraph(0, 1) is True

    def test_is_subgraph_a3_in_a4(self):
        # 3-node path ⊆ 4-node path
        hsdfs = HSDFS([A3, A4])
        assert hsdfs._is_subgraph(0, 1) is True

    def test_is_subgraph_a4_in_b4_paper_example(self):
        # A4 (path) ⊆ B4 (path + extra edge)  – worked example from the paper
        hsdfs = HSDFS([A4, B4])
        assert hsdfs._is_subgraph(0, 1) is True

    def test_is_not_subgraph_incomparable_graphs(self):
        # A2 and A3_incomparable: different column-signature patterns,
        # size mismatch prevents A3 from being in A2.
        hsdfs = HSDFS([A2, A3_incomparable])
        assert hsdfs._is_subgraph(1, 0) is False

    def test_is_subgraph_identical_graphs_both_directions(self):
        # Identical graphs: both directions True (equal case)
        hsdfs = HSDFS([A3, A3.copy()])
        assert hsdfs._is_subgraph(0, 1) is True
        assert hsdfs._is_subgraph(1, 0) is True

    # ------------------------------------------------------------------ _build_meta_graph

    def test_single_graph_empty_meta(self):
        meta = HSDFS([A4])._build_meta_graph()
        assert meta.n == 1
        assert meta.edges == []

    def test_two_graphs_a_in_b_edge_added(self):
        # A2 ⊆ A3 → edge 0→1
        meta = HSDFS([A2, A3])._build_meta_graph()
        assert (0, 1) in meta.edges

    def test_two_graphs_a_in_b_no_reverse_edge(self):
        # A3 ⊄ A2 → no edge 1→0
        meta = HSDFS([A2, A3])._build_meta_graph()
        assert (1, 0) not in meta.edges

    def test_meta_graph_node_count(self):
        meta = HSDFS([A2, A3, A4])._build_meta_graph()
        assert meta.n == 3

    def test_empty_collection(self):
        meta = HSDFS([])._build_meta_graph()
        assert meta.n == 0
        assert meta.edges == []


# ---------------------------------------------------------------------------
# TestHSDFSRun  –  integration tests for the full run()
# ---------------------------------------------------------------------------

class TestHSDFSRun:
    """End-to-end tests combining both phases."""

    def test_run_returns_hsdfs_result(self):
        result = HSDFS([A4, B4]).run()
        assert isinstance(result, HSDFSResult)
        assert isinstance(result.meta_graph, MetaGraph)
        assert isinstance(result.dfs, DFSResult)

    def test_run_empty_collection(self):
        result = HSDFS([]).run()
        assert result.meta_graph.n == 0
        assert result.meta_graph.edges == []
        assert result.dfs.forest_roots == []
        assert result.dfs.edge_class == {}

    def test_run_single_graph(self):
        result = HSDFS([A4]).run()
        assert result.meta_graph.edges == []
        assert result.dfs.forest_roots == [0]
        assert 0 in result.dfs.discovery

    def test_run_paper_example_meta_edge(self):
        # A4 ⊆ B4 as documented in the paper
        result = HSDFS([A4, B4]).run()
        assert (0, 1) in result.meta_graph.edges

    def test_run_paper_example_dfs_tree_edge(self):
        # With only one meta-edge 0→1 the DFS should classify it as TREE
        result = HSDFS([A4, B4]).run()
        if (0, 1) in result.meta_graph.edges and (1, 0) not in result.meta_graph.edges:
            assert result.dfs.edge_class[(0, 1)] == EdgeClass.TREE

    def test_run_chain_three_graphs(self):
        # A2 ⊆ A3 ⊆ A4 → meta edges include 0→1, 1→2, and 0→2
        result = HSDFS([A2, A3, A4]).run()
        assert (0, 1) in result.meta_graph.edges
        assert (1, 2) in result.meta_graph.edges
        assert (0, 2) in result.meta_graph.edges

    def test_run_chain_forward_edge(self):
        # 0→2 in the chain above must be FORWARD (2 found via 0→1→2 first)
        result = HSDFS([A2, A3, A4]).run()
        if (0, 2) in result.meta_graph.edges:
            assert result.dfs.edge_class[(0, 2)] == EdgeClass.FORWARD

    def test_run_identical_graphs_back_edge(self):
        # Two copies of A3 → mutual containment → back edge expected
        result = HSDFS([A3, A3.copy()]).run()
        edges = result.meta_graph.edges
        assert (0, 1) in edges
        assert (1, 0) in edges
        # The return edge must be BACK
        assert result.dfs.edge_class[(1, 0)] == EdgeClass.BACK

    def test_run_all_nodes_receive_timestamps(self):
        result = HSDFS([A2, A3, A4]).run()
        n = 3
        for i in range(n):
            assert i in result.dfs.discovery
            assert i in result.dfs.finish

    def test_run_discovery_before_finish_for_all(self):
        result = HSDFS([A2, A3, A4]).run()
        for i in range(3):
            assert result.dfs.discovery[i] < result.dfs.finish[i]

    def test_run_forest_roots_non_empty(self):
        result = HSDFS([A2, A3]).run()
        assert len(result.dfs.forest_roots) >= 1

    def test_run_cross_edge_two_independent_containments(self):
        # G0=A2 (2×2 path), G1=3×3 identity (self-loops only), G2=A4 (4×4 path).
        # Both A2 and the identity matrix are subgraphs of A4, but they are
        # incomparable with each other (oracle returns keep_both).
        # Meta-edges: 0→2 and 1→2 only; no edge between 0 and 1.
        # DFS visits 0, discovers 2 as tree-child, finishes both, then starts
        # a new tree at 1.  Edge 1→2 lands on an already-BLACK node with
        # d[1] > d[2]  →  must be classified as CROSS.
        identity_3x3 = np.eye(3, dtype=int)
        result = HSDFS([A2, identity_3x3, A4]).run()
        edges = result.meta_graph.edges
        assert (0, 2) in edges   # A2 ⊆ A4
        assert (1, 2) in edges   # identity_3x3 ⊆ A4
        if (0, 1) not in edges and (1, 0) not in edges:
            assert result.dfs.edge_class[(1, 2)] == EdgeClass.CROSS

    def test_run_meta_graph_n_equals_number_of_graphs(self):
        graphs = [A2, A3, A4, B4]
        result = HSDFS(graphs).run()
        assert result.meta_graph.n == 4
