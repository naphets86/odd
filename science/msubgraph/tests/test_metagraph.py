"""Unit tests for MetaGraph."""

import pytest
from msubgraph.hsdfs import MetaGraph


class TestMetaGraphConstruction:
    def test_init_creates_n_nodes(self):
        meta = MetaGraph(n=4)
        assert meta.n == 4

    def test_init_with_zero_nodes(self):
        meta = MetaGraph(n=0)
        assert meta.n == 0
        assert meta.edges == []

    def test_neighbors_empty_on_construction(self):
        meta = MetaGraph(n=3)
        for i in range(3):
            assert meta.neighbors(i) == []

    def test_edges_empty_on_construction(self):
        meta = MetaGraph(n=3)
        assert meta.edges == []


class TestMetaGraphAddEdge:
    def test_add_single_edge(self):
        meta = MetaGraph(n=3)
        meta.add_edge(0, 1)
        assert 1 in meta.neighbors(0)

    def test_add_edge_does_not_affect_source(self):
        meta = MetaGraph(n=3)
        meta.add_edge(0, 1)
        assert meta.neighbors(1) == []

    def test_add_multiple_edges_from_same_node(self):
        meta = MetaGraph(n=4)
        meta.add_edge(0, 1)
        meta.add_edge(0, 2)
        meta.add_edge(0, 3)
        assert sorted(meta.neighbors(0)) == [1, 2, 3]

    def test_add_multiple_edges_to_same_node(self):
        meta = MetaGraph(n=4)
        meta.add_edge(1, 3)
        meta.add_edge(2, 3)
        assert meta.neighbors(1) == [3]
        assert meta.neighbors(2) == [3]

    def test_add_edge_idempotent(self):
        meta = MetaGraph(n=3)
        meta.add_edge(0, 1)
        meta.add_edge(0, 1)
        assert meta.neighbors(0).count(1) == 1

    def test_add_both_directions(self):
        meta = MetaGraph(n=2)
        meta.add_edge(0, 1)
        meta.add_edge(1, 0)
        assert meta.neighbors(0) == [1]
        assert meta.neighbors(1) == [0]


class TestMetaGraphEdgesProperty:
    def test_edges_reflects_added_edges(self):
        meta = MetaGraph(n=3)
        meta.add_edge(0, 1)
        meta.add_edge(1, 2)
        assert (0, 1) in meta.edges
        assert (1, 2) in meta.edges

    def test_edges_count_matches_adds(self):
        meta = MetaGraph(n=4)
        meta.add_edge(0, 1)
        meta.add_edge(0, 2)
        meta.add_edge(2, 3)
        assert len(meta.edges) == 3

    def test_edges_no_duplicates_after_idempotent_add(self):
        meta = MetaGraph(n=2)
        meta.add_edge(0, 1)
        meta.add_edge(0, 1)
        assert len(meta.edges) == 1

    def test_edges_empty_list_when_no_edges(self):
        meta = MetaGraph(n=5)
        assert meta.edges == []


class TestMetaGraphNeighbors:
    def test_neighbors_returns_copy(self):
        meta = MetaGraph(n=2)
        meta.add_edge(0, 1)
        nbrs = meta.neighbors(0)
        nbrs.clear()
        assert meta.neighbors(0) == [1]

    def test_neighbors_isolated_node(self):
        meta = MetaGraph(n=3)
        meta.add_edge(0, 1)
        assert meta.neighbors(2) == []
