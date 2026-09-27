"""
Tests für den Subgraph Algorithmus und den Bipartiten Graph.
Ablage: tests/test_subgraph.py
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from httpx import AsyncClient

# KEIN globales pytestmark=asyncio – nur async-Methoden brauchen es.
# Synchrone Testklassen bekommen KEIN asyncio-Mark.


# ─── Hilfsfunktionen: Testdaten anlegen ──────────────────────────────────────

async def _create_project(client: AsyncClient, name: str, status: str = "geplant") -> int:
    r = await client.post("/api/projects/", json={"name": name, "status": status})
    assert r.status_code == 201, r.text
    return r.json()["id"]


async def _create_person(client: AsyncClient, name: str, ptype: str = "Arbeitnehmer") -> int:
    r = await client.post("/api/persons/", json={"name": name, "person_type": ptype})
    assert r.status_code == 201, r.text
    return r.json()["id"]


async def _create_material(client: AsyncClient, name: str, mtype: str = "Stahl") -> int:
    r = await client.post("/api/materials/", json={"name": name, "material_type": mtype, "quantity": 10})
    assert r.status_code == 201, r.text
    return r.json()["id"]


async def _create_time(client: AsyncClient, name: str) -> int:
    r = await client.post("/api/time-resources/", json={"name": name, "amount": 4, "unit": "Wochen"})
    assert r.status_code == 201, r.text
    return r.json()["id"]


async def _alloc(client: AsyncClient, project_id: int, **kwargs) -> int:
    body = {"project_id": project_id, "quantity": 1.0, **kwargs}
    r = await client.post("/api/allocations/", json=body)
    assert r.status_code == 201, r.text
    return r.json()["id"]


async def _setup_two_projects(client: AsyncClient):
    """
    Projekt Alpha → Person + Material          (2 Kanten)
    Projekt Beta  → Person + Material + Zeit   (3 Kanten, Superset von Alpha)
    """
    pid_a = await _create_project(client, "Projekt Alpha")
    pid_b = await _create_project(client, "Projekt Beta")
    per1  = await _create_person(client,   "Anna")
    mat1  = await _create_material(client, "Stahl X")
    tr1   = await _create_time(client,     "Sprint 1")

    await _alloc(client, pid_a, person_id=per1)
    await _alloc(client, pid_a, material_id=mat1)
    await _alloc(client, pid_b, person_id=per1)
    await _alloc(client, pid_b, material_id=mat1)
    await _alloc(client, pid_b, time_resource_id=tr1)
    return pid_a, pid_b, per1, mat1, tr1


# ─── Mock-Hilfsfunktion ───────────────────────────────────────────────────────

def _patch_service(decision: str):
    """
    Patcht _service._algo im Router-Modul direkt.
    _service ist eine Modul-Konstante die beim Import erstellt wird –
    daher muss die Instanz selbst (nicht die Klasse Subgraph) gepatcht werden.
    """
    mock_algo = MagicMock()
    mock_algo.compare_graphs.return_value = (decision, None)
    return patch("app.routers.subgraph._service._algo", mock_algo)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Unit-Tests: app/services/bipartite.py  (synchron – kein asyncio mark)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildBipartiteGraph:

    def _proj(self, id_, name):  return {"id": id_, "name": name}
    def _pers(self, id_, name):  return {"id": id_, "name": name}
    def _mat(self,  id_, name):  return {"id": id_, "name": name}
    def _time(self, id_, name):  return {"id": id_, "name": name}

    def _alloc(self, project_id, person_id=None, material_id=None, time_resource_id=None):
        return {"project_id": project_id, "person_id": person_id,
                "material_id": material_id, "time_resource_id": time_resource_id}

    def test_empty_allocations(self):
        from app.services.bipartite import build_bipartite_graph
        g = build_bipartite_graph([], [], [], [], [])
        assert g.n_projects == 0
        assert g.n_resources == 0
        assert g.matrix.shape == (0, 0)

    def test_single_person_allocation(self):
        from app.services.bipartite import build_bipartite_graph
        g = build_bipartite_graph(
            [self._alloc(1, person_id=10)],
            [self._proj(1, "P1")], [self._pers(10, "Alice")], [], [],
        )
        assert g.n_projects == 1
        assert g.n_resources == 1
        assert g.matrix[0, 0] == 1
        assert g.projects[0].name == "P1"
        assert g.resources[0].kind == "person"
        assert g.resources[0].label == "Alice"

    def test_material_allocation(self):
        from app.services.bipartite import build_bipartite_graph
        g = build_bipartite_graph(
            [self._alloc(1, material_id=5)],
            [self._proj(1, "P1")], [], [self._mat(5, "Kupfer")], [],
        )
        assert g.resources[0].kind == "material"
        assert g.resources[0].label == "Kupfer"

    def test_time_allocation(self):
        from app.services.bipartite import build_bipartite_graph
        g = build_bipartite_graph(
            [self._alloc(1, time_resource_id=7)],
            [self._proj(1, "P1")], [], [], [self._time(7, "Sprint")],
        )
        assert g.resources[0].kind == "time"
        assert g.resources[0].label == "Sprint"

    def test_resources_sorted_by_kind_and_id(self):
        from app.services.bipartite import build_bipartite_graph
        g = build_bipartite_graph(
            [self._alloc(1, person_id=2), self._alloc(1, person_id=1)],
            [self._proj(1, "P1")],
            [self._pers(2, "Bob"), self._pers(1, "Alice")], [], [],
        )
        assert g.resources[0].id == 1
        assert g.resources[1].id == 2

    def test_unknown_person_label(self):
        from app.services.bipartite import build_bipartite_graph
        g = build_bipartite_graph(
            [self._alloc(1, person_id=99)], [self._proj(1, "P1")], [], [], [],
        )
        assert g.resources[0].label == "Person#99"

    def test_unknown_material_label(self):
        from app.services.bipartite import build_bipartite_graph
        g = build_bipartite_graph(
            [self._alloc(1, material_id=88)], [self._proj(1, "P1")], [], [], [],
        )
        assert g.resources[0].label == "Material#88"

    def test_unknown_time_label(self):
        from app.services.bipartite import build_bipartite_graph
        g = build_bipartite_graph(
            [self._alloc(1, time_resource_id=77)], [self._proj(1, "P1")], [], [], [],
        )
        assert g.resources[0].label == "Zeit#77"

    def test_unknown_project_label(self):
        from app.services.bipartite import build_bipartite_graph
        g = build_bipartite_graph(
            [self._alloc(999, person_id=1)], [], [self._pers(1, "X")], [], [],
        )
        assert g.projects[0].name == "Projekt#999"

    def test_matrix_shape_two_projects_two_resources(self):
        from app.services.bipartite import build_bipartite_graph
        g = build_bipartite_graph(
            [self._alloc(1, person_id=1), self._alloc(2, material_id=1)],
            [self._proj(1, "P1"), self._proj(2, "P2")],
            [self._pers(1, "A")], [self._mat(1, "M")], [],
        )
        assert g.matrix.shape == (2, 2)

    def test_has_edge_true(self):
        from app.services.bipartite import build_bipartite_graph
        g = build_bipartite_graph(
            [self._alloc(1, person_id=1)],
            [self._proj(1, "P")], [self._pers(1, "A")], [], [],
        )
        assert g.has_edge(1, 1) is True

    def test_has_edge_false_unknown_project(self):
        from app.services.bipartite import build_bipartite_graph
        g = build_bipartite_graph(
            [self._alloc(1, person_id=1)],
            [self._proj(1, "P")], [self._pers(1, "A")], [], [],
        )
        assert g.has_edge(999, 1) is False

    def test_has_edge_false_unknown_resource(self):
        from app.services.bipartite import build_bipartite_graph
        g = build_bipartite_graph(
            [self._alloc(1, person_id=1)],
            [self._proj(1, "P")], [self._pers(1, "A")], [], [],
        )
        assert g.has_edge(1, 999) is False

    def test_edges_returns_correct_pair(self):
        from app.services.bipartite import build_bipartite_graph
        g = build_bipartite_graph(
            [self._alloc(1, person_id=1)],
            [self._proj(1, "P")], [self._pers(1, "A")], [], [],
        )
        edges = g.edges()
        assert len(edges) == 1
        proj, res = edges[0]
        assert proj.id == 1 and res.id == 1

    def test_edges_empty_graph(self):
        from app.services.bipartite import build_bipartite_graph
        assert build_bipartite_graph([], [], [], [], []).edges() == []

    def test_same_resource_deduplicated(self):
        from app.services.bipartite import build_bipartite_graph
        g = build_bipartite_graph(
            [self._alloc(1, person_id=1), self._alloc(2, person_id=1)],
            [self._proj(1, "P1"), self._proj(2, "P2")],
            [self._pers(1, "A")], [], [],
        )
        assert g.n_resources == 1
        assert g.matrix.shape == (2, 1)
        assert g.matrix[0, 0] == 1
        assert g.matrix[1, 0] == 1

    def test_n_projects_property(self):
        from app.services.bipartite import BipartiteAllocationGraph, ProjectNode
        g = BipartiteAllocationGraph()
        g.projects = [ProjectNode(1, "A"), ProjectNode(2, "B")]
        assert g.n_projects == 2

    def test_n_resources_property(self):
        from app.services.bipartite import BipartiteAllocationGraph, ResourceNode
        g = BipartiteAllocationGraph()
        g.resources = [ResourceNode(1, "person", "Alice")]
        assert g.n_resources == 1


class TestPadToSquare:

    def test_already_square(self):
        from app.services.bipartite import _pad_to_square
        m = np.array([[1, 0], [0, 1]], dtype=int)
        np.testing.assert_array_equal(_pad_to_square(m), m)

    def test_wide_matrix_padded(self):
        from app.services.bipartite import _pad_to_square
        r = _pad_to_square(np.array([[1, 0, 1]], dtype=int))
        assert r.shape == (3, 3)
        assert r[0, 0] == 1 and r[0, 2] == 1

    def test_tall_matrix_padded(self):
        from app.services.bipartite import _pad_to_square
        r = _pad_to_square(np.array([[1], [0], [1]], dtype=int))
        assert r.shape == (3, 3)
        assert r[0, 0] == 1 and r[2, 0] == 1

    def test_empty_matrix_becomes_1x1(self):
        from app.services.bipartite import _pad_to_square
        assert _pad_to_square(np.empty((0, 0), dtype=int)).shape == (1, 1)

    def test_padding_fills_with_zeros(self):
        from app.services.bipartite import _pad_to_square
        r = _pad_to_square(np.zeros((2, 4), dtype=int))
        assert r.shape == (4, 4) and np.all(r == 0)


class TestBipartiteSubgraphService:

    def _graph(self, m: np.ndarray):
        from app.services.bipartite import BipartiteAllocationGraph, ProjectNode, ResourceNode
        g = BipartiteAllocationGraph()
        g.projects  = [ProjectNode(i + 1, f"P{i+1}") for i in range(m.shape[0])]
        g.resources = [ResourceNode(i + 1, "person", f"R{i+1}") for i in range(m.shape[1])]
        g.matrix    = m
        return g

    def _mock(self, decision: str):
        m = MagicMock()
        m.compare_graphs.return_value = (decision, None)
        return m

    def test_keep_A(self):
        from app.services.bipartite import BipartiteSubgraphService
        svc = BipartiteSubgraphService()
        svc._algo = self._mock("keep_A")
        r = svc.compare(self._graph(np.array([[1, 1]])), self._graph(np.array([[1, 0]])))
        assert r.is_redundant is True and r.is_superset is False and r.is_equal is False

    def test_keep_B(self):
        from app.services.bipartite import BipartiteSubgraphService
        svc = BipartiteSubgraphService()
        svc._algo = self._mock("keep_B")
        r = svc.compare(self._graph(np.array([[1, 0]])), self._graph(np.array([[1, 1]])))
        assert r.is_redundant is False and r.is_superset is True and r.is_equal is False

    def test_keep_both(self):
        from app.services.bipartite import BipartiteSubgraphService
        svc = BipartiteSubgraphService()
        svc._algo = self._mock("keep_both")
        r = svc.compare(self._graph(np.array([[1, 0]])), self._graph(np.array([[0, 1]])))
        assert r.is_redundant is False and r.is_superset is False and r.is_equal is False

    def test_equal_keep_A(self):
        from app.services.bipartite import BipartiteSubgraphService
        svc = BipartiteSubgraphService()
        svc._algo = self._mock("equal_keep_A")
        r = svc.compare(self._graph(np.array([[1]])), self._graph(np.array([[1]])))
        assert r.is_equal is True and r.decision == "equal_keep_A"

    def test_equal_keep_B(self):
        from app.services.bipartite import BipartiteSubgraphService
        svc = BipartiteSubgraphService()
        svc._algo = self._mock("equal_keep_B")
        r = svc.compare(self._graph(np.array([[1]])), self._graph(np.array([[1]])))
        assert r.is_equal is True and r.decision == "equal_keep_B"

    def test_unknown_decision_in_message(self):
        from app.services.bipartite import BipartiteSubgraphService
        svc = BipartiteSubgraphService()
        svc._algo = self._mock("unknown_xyz")
        r = svc.compare(self._graph(np.array([[1]])), self._graph(np.array([[1]])))
        assert "unknown_xyz" in r.message

    def test_edge_counts(self):
        from app.services.bipartite import BipartiteSubgraphService
        svc = BipartiteSubgraphService()
        svc._algo = self._mock("keep_both")
        r = svc.compare(self._graph(np.array([[1, 0, 1]])), self._graph(np.array([[1, 1, 1]])))
        assert r.existing_edges == 2 and r.proposed_edges == 3

    def test_find_redundant_empty(self):
        from app.services.bipartite import BipartiteSubgraphService
        assert BipartiteSubgraphService().find_redundant_allocations([]) == []

    def test_find_redundant_none(self):
        from app.services.bipartite import BipartiteSubgraphService
        svc = BipartiteSubgraphService()
        svc._algo = self._mock("keep_both")
        result = svc.find_redundant_allocations([
            self._graph(np.array([[1, 0]])), self._graph(np.array([[0, 1]])),
        ])
        assert result == []

    def test_find_redundant_found(self):
        from app.services.bipartite import BipartiteSubgraphService
        svc = BipartiteSubgraphService()
        svc._algo = self._mock("keep_B")
        result = svc.find_redundant_allocations([
            self._graph(np.array([[1, 0]])), self._graph(np.array([[1, 1]])),
        ])
        assert len(result) == 1
        i, j, _ = result[0]
        assert i == 0 and j == 1

    def test_suggest_optimal_empty(self):
        from app.services.bipartite import BipartiteSubgraphService
        assert BipartiteSubgraphService().suggest_optimal_graph([]) is None

    def test_suggest_optimal_single(self):
        from app.services.bipartite import BipartiteSubgraphService
        svc = BipartiteSubgraphService()
        g = self._graph(np.array([[1]]))
        assert svc.suggest_optimal_graph([g]) is g

    def test_suggest_optimal_superset_wins(self):
        from app.services.bipartite import BipartiteSubgraphService
        svc = BipartiteSubgraphService()
        svc._algo = self._mock("keep_B")
        a = self._graph(np.array([[1, 0]]))
        b = self._graph(np.array([[1, 1]]))
        assert svc.suggest_optimal_graph([a, b]) is b

    def test_suggest_optimal_keep_both_none(self):
        from app.services.bipartite import BipartiteSubgraphService
        svc = BipartiteSubgraphService()
        svc._algo = self._mock("keep_both")
        a = self._graph(np.array([[1, 0]]))
        b = self._graph(np.array([[0, 1]]))
        assert svc.suggest_optimal_graph([a, b]) is None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Integrations-Tests: /api/subgraph/...  (async)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSubgraphCompareEndpoint:
    pytestmark = pytest.mark.asyncio

    async def test_compare_keep_B(self, client: AsyncClient):
        pid_a, pid_b, *_ = await _setup_two_projects(client)
        with _patch_service("keep_B"):
            r = await client.post("/api/subgraph/compare", json={
                "existing_project_ids": [pid_a], "proposed_project_ids": [pid_b],
            })
        assert r.status_code == 200
        d = r.json()
        assert d["decision"] == "keep_B"
        assert d["is_superset"] is True
        assert d["is_redundant"] is False

    async def test_compare_keep_A(self, client: AsyncClient):
        pid_a, pid_b, *_ = await _setup_two_projects(client)
        with _patch_service("keep_A"):
            r = await client.post("/api/subgraph/compare", json={
                "existing_project_ids": [pid_a], "proposed_project_ids": [pid_b],
            })
        assert r.status_code == 200
        assert r.json()["is_redundant"] is True

    async def test_compare_keep_both(self, client: AsyncClient):
        pid_a, pid_b, *_ = await _setup_two_projects(client)
        with _patch_service("keep_both"):
            r = await client.post("/api/subgraph/compare", json={
                "existing_project_ids": [pid_a], "proposed_project_ids": [pid_b],
            })
        assert r.status_code == 200
        assert r.json()["decision"] == "keep_both"

    async def test_compare_equal_keep_A(self, client: AsyncClient):
        pid_a, pid_b, *_ = await _setup_two_projects(client)
        with _patch_service("equal_keep_A"):
            r = await client.post("/api/subgraph/compare", json={
                "existing_project_ids": [pid_a], "proposed_project_ids": [pid_b],
            })
        assert r.status_code == 200
        assert r.json()["is_equal"] is True

    async def test_compare_equal_keep_B(self, client: AsyncClient):
        pid_a, pid_b, *_ = await _setup_two_projects(client)
        with _patch_service("equal_keep_B"):
            r = await client.post("/api/subgraph/compare", json={
                "existing_project_ids": [pid_a], "proposed_project_ids": [pid_b],
            })
        assert r.status_code == 200
        assert r.json()["is_equal"] is True

    async def test_compare_matrix_is_list(self, client: AsyncClient):
        pid_a, pid_b, *_ = await _setup_two_projects(client)
        with _patch_service("keep_both"):
            r = await client.post("/api/subgraph/compare", json={
                "existing_project_ids": [pid_a], "proposed_project_ids": [pid_b],
            })
        d = r.json()
        assert isinstance(d["existing_graph"]["matrix"], list)
        assert isinstance(d["proposed_graph"]["matrix"], list)

    async def test_compare_empty_existing_422(self, client: AsyncClient):
        r = await client.post("/api/subgraph/compare", json={
            "existing_project_ids": [], "proposed_project_ids": [1],
        })
        assert r.status_code == 422

    async def test_compare_empty_proposed_422(self, client: AsyncClient):
        r = await client.post("/api/subgraph/compare", json={
            "existing_project_ids": [1], "proposed_project_ids": [],
        })
        assert r.status_code == 422

    async def test_compare_nonexistent_projects_ok(self, client: AsyncClient):
        with _patch_service("keep_both"):
            r = await client.post("/api/subgraph/compare", json={
                "existing_project_ids": [9999], "proposed_project_ids": [8888],
            })
        assert r.status_code == 200

    async def test_compare_edge_counts(self, client: AsyncClient):
        pid_a, pid_b, *_ = await _setup_two_projects(client)
        with _patch_service("keep_B"):
            r = await client.post("/api/subgraph/compare", json={
                "existing_project_ids": [pid_a], "proposed_project_ids": [pid_b],
            })
        d = r.json()
        assert d["existing_edges"] == 2
        assert d["proposed_edges"] == 3

    async def test_n_edges_matches_edges_list(self, client: AsyncClient):
        pid_a, pid_b, *_ = await _setup_two_projects(client)
        with _patch_service("keep_B"):
            r = await client.post("/api/subgraph/compare", json={
                "existing_project_ids": [pid_a], "proposed_project_ids": [pid_b],
            })
        d = r.json()
        for key in ("existing_graph", "proposed_graph"):
            assert d[key]["n_edges"] == len(d[key]["edges"])


class TestSubgraphGraphEndpoint:
    pytestmark = pytest.mark.asyncio

    async def test_single_project(self, client: AsyncClient):
        pid_a, *_ = await _setup_two_projects(client)
        r = await client.get(f"/api/subgraph/graph?project_ids={pid_a}")
        assert r.status_code == 200
        d = r.json()
        assert d["n_projects"] == 1
        assert d["n_resources"] == 2
        assert d["n_edges"] == 2

    async def test_two_projects(self, client: AsyncClient):
        pid_a, pid_b, *_ = await _setup_two_projects(client)
        r = await client.get(f"/api/subgraph/graph?project_ids={pid_a},{pid_b}")
        assert r.status_code == 200
        d = r.json()
        assert d["n_projects"] == 2
        assert d["n_resources"] == 3

    async def test_matrix_shape(self, client: AsyncClient):
        pid_a, *_ = await _setup_two_projects(client)
        r = await client.get(f"/api/subgraph/graph?project_ids={pid_a}")
        d = r.json()
        assert len(d["matrix"]) == d["n_projects"]
        assert len(d["matrix"][0]) == d["n_resources"]

    async def test_invalid_ids_422(self, client: AsyncClient):
        r = await client.get("/api/subgraph/graph?project_ids=abc,xyz")
        assert r.status_code == 422

    async def test_empty_ids_422(self, client: AsyncClient):
        r = await client.get("/api/subgraph/graph?project_ids=")
        assert r.status_code == 422

    async def test_nonexistent_project_empty(self, client: AsyncClient):
        r = await client.get("/api/subgraph/graph?project_ids=99999")
        assert r.status_code == 200
        d = r.json()
        assert d["n_projects"] == 0 and d["n_edges"] == 0

    async def test_resources_fields(self, client: AsyncClient):
        pid_a, *_ = await _setup_two_projects(client)
        r = await client.get(f"/api/subgraph/graph?project_ids={pid_a}")
        for res in r.json()["resources"]:
            assert "id" in res and "kind" in res and "label" in res

    async def test_edges_fields(self, client: AsyncClient):
        pid_a, *_ = await _setup_two_projects(client)
        r = await client.get(f"/api/subgraph/graph?project_ids={pid_a}")
        for e in r.json()["edges"]:
            for f in ("project_id", "project_name", "resource_id", "resource_kind", "resource_label"):
                assert f in e


class TestSubgraphRedundancyEndpoint:
    pytestmark = pytest.mark.asyncio

    async def test_too_few_groups_422(self, client: AsyncClient):
        r = await client.post("/api/subgraph/redundancy", json=[[1]])
        assert r.status_code == 422

    async def test_no_redundancy(self, client: AsyncClient):
        pid_a, pid_b, *_ = await _setup_two_projects(client)
        with _patch_service("keep_both"):
            r = await client.post("/api/subgraph/redundancy", json=[[pid_a], [pid_b]])
        assert r.status_code == 200
        d = r.json()
        assert d["total_graphs"] == 2
        assert d["redundant_pairs"] == []
        assert d["optimal_graph_index"] is None

    async def test_redundancy_found(self, client: AsyncClient):
        pid_a, pid_b, *_ = await _setup_two_projects(client)
        with _patch_service("keep_B"):
            r = await client.post("/api/subgraph/redundancy", json=[[pid_a], [pid_b]])
        d = r.json()
        assert len(d["redundant_pairs"]) == 1
        assert d["redundant_pairs"][0]["graph_a_index"] == 0
        assert d["redundant_pairs"][0]["graph_b_index"] == 1

    async def test_optimal_index_set(self, client: AsyncClient):
        pid_a, pid_b, *_ = await _setup_two_projects(client)
        with _patch_service("keep_B"):
            r = await client.post("/api/subgraph/redundancy", json=[[pid_a], [pid_b]])
        assert r.json()["optimal_graph_index"] is not None

    async def test_summary_group_count(self, client: AsyncClient):
        pid_a, pid_b, *_ = await _setup_two_projects(client)
        with _patch_service("keep_both"):
            r = await client.post("/api/subgraph/redundancy", json=[[pid_a], [pid_b]])
        assert "2 Projektgruppen" in r.json()["summary"]

    async def test_summary_optimal(self, client: AsyncClient):
        pid_a, pid_b, *_ = await _setup_two_projects(client)
        with _patch_service("keep_B"):
            r = await client.post("/api/subgraph/redundancy", json=[[pid_a], [pid_b]])
        assert "Optimaler Graph" in r.json()["summary"]

    async def test_three_groups(self, client: AsyncClient):
        pid_a, pid_b, *_ = await _setup_two_projects(client)
        pid_c = await _create_project(client, "Gamma")
        with _patch_service("keep_both"):
            r = await client.post("/api/subgraph/redundancy", json=[[pid_a], [pid_b], [pid_c]])
        assert r.status_code == 200
        assert r.json()["total_graphs"] == 3

    async def test_pair_fields(self, client: AsyncClient):
        pid_a, pid_b, *_ = await _setup_two_projects(client)
        with _patch_service("keep_A"):
            r = await client.post("/api/subgraph/redundancy", json=[[pid_a], [pid_b]])
        pair = r.json()["redundant_pairs"][0]
        for key in ("graph_a_index", "graph_b_index", "decision", "is_redundant", "is_superset", "message"):
            assert key in pair