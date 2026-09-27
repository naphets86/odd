"""
Bipartiter Graph – Ressourcenzuweisung via Subgraph Algorithmus
===============================================================

Modell
------
Ein bipartiter Graph B = (P ∪ R, E) beschreibt die Zuweisung von
Ressourcen zu Projekten:

    • Linke Partition  P = {p₁, p₂, …, p_m}   → Projekte
    • Rechte Partition R = {r₁, r₂, …, r_n}   → Ressourcen
      (Personen | Material | Zeitressourcen)
    • Kante (pᵢ, rⱼ) ∈ E  ↔  Ressource rⱼ ist Projekt pᵢ zugewiesen

Darstellung als Adjazenzmatrix
--------------------------------
Die Biadjazenzmatrix M hat die Form (m × n):

        r₁  r₂  …  r_n
    p₁ [ 0   1  …   0 ]
    p₂ [ 1   0  …   1 ]
    …
    p_m[ 0   1  …   0 ]

Subgraph-Anwendung
------------------
Gegeben zwei Zuteilungsgraphen (z. B. bestehende DB-Zuteilungen vs.
neu vorgeschlagene Zuteilung):

1. Baue Biadjazenzmatrizen A (bestehend) und B (vorgeschlagen).
2. Übergib an ``Subgraph.compare_graphs(A, B)`` (O(n³)).
3. Interpretiere Ergebnis:
   - ``keep_B``       → Vorschlag B enthält A vollständig → B ersetzen
   - ``keep_A``       → Bestandsgraph A enthält B → B redundant, verwerfen
   - ``keep_both``    → Kein Teilmengen-Verhältnis → beide behalten
   - ``equal_keep_B`` / ``equal_keep_A`` → Strukturell gleich, mehr Kanten gewinnt
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# Importiert aus dem GitLab-Package (installiert via requirements.txt)
from subgraph import Subgraph  # type: ignore[import]


# ── Datenklassen ──────────────────────────────────────────────────────────────

@dataclass
class ResourceNode:
    """Eine Ressource (Knoten der rechten Partition)."""
    id: int
    kind: str  # "person" | "material" | "time"
    label: str


@dataclass
class ProjectNode:
    """Ein Projekt (Knoten der linken Partition)."""
    id: int
    name: str


@dataclass
class BipartiteAllocationGraph:
    """
    Bipartiter Graph für Ressourcenzuweisungen.

    Attribute
    ---------
    projects    : geordnete Liste der Projektknoten (linke Partition)
    resources   : geordnete Liste der Ressourcenknoten (rechte Partition)
    matrix      : Biadjazenzmatrix (len(projects) × len(resources)), dtype=int
    """
    projects: list[ProjectNode] = field(default_factory=list)
    resources: list[ResourceNode] = field(default_factory=list)
    matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=int))

    # ── Zugriffshilfen ────────────────────────────────────────────────────────

    @property
    def n_projects(self) -> int:
        return len(self.projects)

    @property
    def n_resources(self) -> int:
        return len(self.resources)

    def has_edge(self, project_id: int, resource_id: int) -> bool:
        """True, wenn (Projekt, Ressource)-Kante in diesem Graph existiert."""
        try:
            pi = next(i for i, p in enumerate(self.projects)  if p.id == project_id)
            ri = next(i for i, r in enumerate(self.resources) if r.id == resource_id)
        except StopIteration:
            return False
        return bool(self.matrix[pi, ri])

    def edges(self) -> list[tuple[ProjectNode, ResourceNode]]:
        """Gibt alle Kanten (Projekt, Ressource) zurück."""
        result = []
        for pi, proj in enumerate(self.projects):
            for ri, res in enumerate(self.resources):
                if self.matrix[pi, ri]:
                    result.append((proj, res))
        return result


# ── Builder ────────────────────────────────────────────────────────────────────

def build_bipartite_graph(
    allocations: list[dict],
    projects: list[dict],
    persons: list[dict],
    materials: list[dict],
    time_resources: list[dict],
) -> BipartiteAllocationGraph:
    """
    Erzeugt einen ``BipartiteAllocationGraph`` aus rohen Daten-Dicts.

    Die Dicts entsprechen den Pydantic-Schemas ``AllocationOut``,
    ``ProjectOut``, ``PersonOut``, ``MaterialOut``, ``TimeResourceOut``
    (serialisiert via ``.model_dump()``).

    Parameter
    ---------
    allocations    : Liste aller Zuteilungs-Dicts
    projects       : Liste aller Projekt-Dicts
    persons        : Liste aller Personen-Dicts
    materials      : Liste aller Material-Dicts
    time_resources : Liste aller Zeitressourcen-Dicts

    Rückgabe
    --------
    ``BipartiteAllocationGraph`` mit befüllter Biadjazenzmatrix
    """
    # Aufbau Lookup-Maps
    proj_map  = {p["id"]: p["name"] for p in projects}
    pers_map  = {p["id"]: p["name"] for p in persons}
    mat_map   = {m["id"]: m["name"] for m in materials}
    time_map  = {t["id"]: t["name"] for t in time_resources}

    # Nur Projekte / Ressourcen, die wirklich in Zuteilungen vorkommen
    used_project_ids: set[int] = set()
    resource_nodes: dict[tuple[str, int], ResourceNode] = {}

    for alloc in allocations:
        used_project_ids.add(alloc["project_id"])

        if alloc.get("person_id") is not None:
            key = ("person", alloc["person_id"])
            if key not in resource_nodes:
                resource_nodes[key] = ResourceNode(
                    id=alloc["person_id"],
                    kind="person",
                    label=pers_map.get(alloc["person_id"], f"Person#{alloc['person_id']}"),
                )
        if alloc.get("material_id") is not None:
            key = ("material", alloc["material_id"])
            if key not in resource_nodes:
                resource_nodes[key] = ResourceNode(
                    id=alloc["material_id"],
                    kind="material",
                    label=mat_map.get(alloc["material_id"], f"Material#{alloc['material_id']}"),
                )
        if alloc.get("time_resource_id") is not None:
            key = ("time", alloc["time_resource_id"])
            if key not in resource_nodes:
                resource_nodes[key] = ResourceNode(
                    id=alloc["time_resource_id"],
                    kind="time",
                    label=time_map.get(alloc["time_resource_id"], f"Zeit#{alloc['time_resource_id']}"),
                )

    # Geordnete Listen (deterministisch nach ID)
    project_list  = sorted(
        [ProjectNode(id=pid, name=proj_map.get(pid, f"Projekt#{pid}"))
         for pid in used_project_ids],
        key=lambda x: x.id,
    )
    resource_list = sorted(resource_nodes.values(), key=lambda r: (r.kind, r.id))

    m, n = len(project_list), len(resource_list)
    matrix = np.zeros((m, n), dtype=int)

    proj_idx  = {p.id: i for i, p in enumerate(project_list)}
    res_idx   = {(r.kind, r.id): i for i, r in enumerate(resource_list)}

    for alloc in allocations:
        pi = proj_idx[alloc["project_id"]]
        for kind, fk in [("person", "person_id"), ("material", "material_id"), ("time", "time_resource_id")]:
            if alloc.get(fk) is not None:
                ri = res_idx[(kind, alloc[fk])]
                matrix[pi, ri] = 1

    return BipartiteAllocationGraph(
        projects=project_list,
        resources=resource_list,
        matrix=matrix,
    )


def _pad_to_square(matrix: np.ndarray) -> np.ndarray:
    """
    Ergänzt eine (m × n)-Biadjazenzmatrix zu einer quadratischen
    (max(m,n) × max(m,n))-Adjazenzmatrix durch Null-Padding.

    Der Subgraph Algorithmus erwartet quadratische Matrizen.
    Das Padding verändert die Kantenstruktur nicht.
    """
    m, n = matrix.shape
    size = max(m, n, 1)
    padded = np.zeros((size, size), dtype=int)
    padded[:m, :n] = matrix
    return padded


# ── Vergleichs-Service ────────────────────────────────────────────────────────

@dataclass
class SubgraphComparisonResult:
    """Ergebnis eines Subgraph-Vergleichs zweier Zuteilungsgraphen."""
    decision: str
    """
    Entscheidung des Algorithmus:
    - ``keep_B``       : Vorschlag enthält bestehenden Graph → Vorschlag verwenden
    - ``keep_A``       : Bestehender Graph enthält Vorschlag → Vorschlag redundant
    - ``keep_both``    : Keine Teilmengen-Relation → beide Graphen behalten
    - ``equal_keep_A`` : Strukturgleich, bestehender Graph bevorzugt
    - ``equal_keep_B`` : Strukturgleich, Vorschlag bevorzugt (mehr Kanten)
    """
    is_redundant: bool
    """True, wenn der vorgeschlagene Graph A vollständig in B enthalten ist (keep_A)."""
    is_superset: bool
    """True, wenn der vorgeschlagene Graph B den bestehenden Graph A enthält (keep_B)."""
    is_equal: bool
    """True, wenn beide Graphen strukturell identisch sind."""
    existing_edges: int
    """Anzahl Kanten im bestehenden Zuteilungsgraph."""
    proposed_edges: int
    """Anzahl Kanten im vorgeschlagenen Zuteilungsgraph."""
    message: str
    """Menschenlesbare Beschreibung der Entscheidung."""


class BipartiteSubgraphService:
    """
    Service-Klasse: wendet den Subgraph Algorithmus auf Ressourcenzuweisungen an.

    Verwendung
    ----------
    ::

        service = BipartiteSubgraphService()
        result  = service.compare(existing_graph, proposed_graph)

        if result.is_redundant:
            # Vorschlag ist überflüssig
            ...
        elif result.is_superset:
            # Vorschlag ersetzt bestehende Zuteilung vollständig
            ...
    """

    def __init__(self) -> None:
        self._algo = Subgraph()

    # ── Öffentliche API ───────────────────────────────────────────────────────

    def compare(
        self,
        existing: BipartiteAllocationGraph,
        proposed: BipartiteAllocationGraph,
    ) -> SubgraphComparisonResult:
        """
        Vergleicht zwei Zuteilungsgraphen mit dem Subgraph Algorithmus (O(n³)).

        Parameter
        ---------
        existing : bestehender Zuteilungsgraph (aus der Datenbank)
        proposed : neu vorgeschlagener Zuteilungsgraph

        Rückgabe
        --------
        ``SubgraphComparisonResult`` mit Entscheidung und Metadaten
        """
        A = _pad_to_square(existing.matrix)  # bestehend
        B = _pad_to_square(proposed.matrix)  # vorgeschlagen

        decision, _ = self._algo.compare_graphs(A, B)

        existing_edges = int(np.sum(existing.matrix))
        proposed_edges = int(np.sum(proposed.matrix))

        messages = {
            "keep_A":       "Der bestehende Zuteilungsgraph enthält den Vorschlag bereits vollständig – Vorschlag ist redundant.",
            "keep_B":       "Der Vorschlag erweitert den bestehenden Graphen – Vorschlag kann die bestehenden Zuteilungen ersetzen.",
            "keep_both":    "Kein Teilmengen-Verhältnis erkannt – beide Zuteilungsgraphen sind unabhängig voneinander.",
            "equal_keep_A": "Beide Graphen sind strukturell identisch – bestehender Graph wird bevorzugt.",
            "equal_keep_B": "Beide Graphen sind strukturell identisch – Vorschlag hat mehr Kanten und wird bevorzugt.",
        }

        return SubgraphComparisonResult(
            decision=decision,
            is_redundant=(decision == "keep_A"),
            is_superset=(decision == "keep_B"),
            is_equal=(decision in ("equal_keep_A", "equal_keep_B")),
            existing_edges=existing_edges,
            proposed_edges=proposed_edges,
            message=messages.get(decision, f"Unbekannte Entscheidung: {decision}"),
        )

    def find_redundant_allocations(
        self,
        graphs: list[BipartiteAllocationGraph],
    ) -> list[tuple[int, int, SubgraphComparisonResult]]:
        """
        Findet redundante Graphen in einer Liste von Zuteilungsgraphen.

        Vergleicht alle Paare (i, j) mit i < j und gibt Paare zurück,
        bei denen einer den anderen enthält.

        Parameter
        ---------
        graphs : Liste von ``BipartiteAllocationGraph``-Objekten

        Rückgabe
        --------
        Liste von (i, j, result) für alle Paare mit Teilmengenbeziehung
        """
        redundant = []
        for i in range(len(graphs)):
            for j in range(i + 1, len(graphs)):
                result = self.compare(graphs[i], graphs[j])
                if result.decision != "keep_both":
                    redundant.append((i, j, result))
        return redundant

    def suggest_optimal_graph(
        self,
        graphs: list[BipartiteAllocationGraph],
    ) -> BipartiteAllocationGraph | None:
        """
        Gibt den Graphen zurück, der alle anderen als Subgraphen enthält
        (d. h. der informatievollste Graph), falls ein solcher existiert.

        Parameter
        ---------
        graphs : Liste von Zuteilungsgraphen

        Rückgabe
        --------
        Den maximalen Graphen oder ``None``, wenn kein einzelner Graph
        alle anderen enthält.
        """
        if not graphs:
            return None
        if len(graphs) == 1:
            return graphs[0]

        candidate = graphs[0]
        for other in graphs[1:]:
            result = self.compare(candidate, other)
            if result.is_superset:
                # other enthält candidate vollständig
                candidate = other
            elif result.decision == "keep_both":
                # Kein klarer Gewinner
                return None
        return candidate
