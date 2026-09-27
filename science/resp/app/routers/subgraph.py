"""
API-Router: Bipartiter Graph & Subgraph Algorithmus
====================================================

Ablage: app/routers/subgraph.py

Endpunkte
---------
POST /api/subgraph/compare
    Vergleicht zwei Zuteilungsgraphen anhand ihrer Projekt-IDs.
    Gibt das Ergebnis des Subgraph Algorithmus (O(n³)) zurück.

GET  /api/subgraph/graph?project_ids=1,2,3
    Liefert den bipartiten Zuteilungsgraphen für eine Menge von Projekten
    als Adjazenzmatrix + Knotenlisten (für Visualisierung).

POST /api/subgraph/redundancy
    Prüft eine Liste von Projektgruppen auf Redundanz (paarweiser Vergleich).
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import Allocation, Project, Person, Material, TimeResource
from app.schemas import (
    BipartiteGraphOut,
    EdgeOut,
    ProjectNodeOut,
    ResourceNodeOut,
    SubgraphComparisonOut,
    SubgraphComparisonRequest,
    RedundancyCheckOut,
)
from app.services.bipartite import (
    BipartiteSubgraphService,
    build_bipartite_graph,
    BipartiteAllocationGraph,
)

router = APIRouter(prefix="/subgraph", tags=["Subgraph / Bipartiter Graph"])

_service = BipartiteSubgraphService()


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

async def _load_all_data(db: AsyncSession) -> tuple[list, list, list, list]:
    """Lädt alle Personen, Materialien, Zeitressourcen und Zuteilungen."""
    persons     = (await db.execute(select(Person))).scalars().all()
    materials   = (await db.execute(select(Material))).scalars().all()
    time_res    = (await db.execute(select(TimeResource))).scalars().all()
    allocations = (await db.execute(select(Allocation))).scalars().all()
    return persons, materials, time_res, allocations


def _orm_to_dicts(persons, materials, time_res, allocations):
    """Konvertiert ORM-Objekte zu Dicts für den Builder."""
    return (
        [{"id": p.id, "name": p.name} for p in persons],
        [{"id": m.id, "name": m.name} for m in materials],
        [{"id": t.id, "name": t.name} for t in time_res],
        [
            {
                "project_id":       a.project_id,
                "person_id":        a.person_id,
                "material_id":      a.material_id,
                "time_resource_id": a.time_resource_id,
            }
            for a in allocations
        ],
    )


def _filter_allocations(allocations: list[dict], project_ids: list[int]) -> list[dict]:
    """Filtert Zuteilungen nach Projekt-IDs."""
    id_set = set(project_ids)
    return [a for a in allocations if a["project_id"] in id_set]


async def _build_graph_for_projects(
    project_ids: list[int],
    db: AsyncSession,
) -> BipartiteAllocationGraph:
    """Baut BipartiteAllocationGraph für eine Projekt-ID-Menge."""
    persons, materials, time_res, allocations = await _load_all_data(db)
    p_dicts, m_dicts, t_dicts, a_dicts = _orm_to_dicts(persons, materials, time_res, allocations)

    result   = await db.execute(select(Project).where(Project.id.in_(project_ids)))
    projects = result.scalars().all()
    proj_dicts = [{"id": p.id, "name": p.name} for p in projects]

    filtered_allocs = _filter_allocations(a_dicts, project_ids)
    return build_bipartite_graph(filtered_allocs, proj_dicts, p_dicts, m_dicts, t_dicts)


def _graph_to_schema(g: BipartiteAllocationGraph) -> BipartiteGraphOut:
    """Serialisiert einen BipartiteAllocationGraph in das Pydantic-Schema."""
    edges = [
        EdgeOut(
            project_id=proj.id,
            project_name=proj.name,
            resource_id=res.id,
            resource_kind=res.kind,
            resource_label=res.label,
        )
        for proj, res in g.edges()
    ]
    return BipartiteGraphOut(
        projects=[ProjectNodeOut(id=p.id, name=p.name) for p in g.projects],
        resources=[ResourceNodeOut(id=r.id, kind=r.kind, label=r.label) for r in g.resources],
        edges=edges,
        n_projects=g.n_projects,
        n_resources=g.n_resources,
        n_edges=len(edges),
        matrix=g.matrix.tolist(),
    )


# ── Endpunkte ─────────────────────────────────────────────────────────────────

@router.post(
    "/compare",
    response_model=SubgraphComparisonOut,
    summary="Subgraph-Vergleich zweier Zuteilungsgraphen",
    description=(
        "Wendet den Subgraph Algorithmus (O(n³)) auf zwei Zuteilungsgraphen an. "
        "Liefert, ob einer den anderen als Teilgraph enthält."
    ),
)
async def compare_graphs(
    payload: SubgraphComparisonRequest,
    db: AsyncSession = Depends(get_db),
) -> SubgraphComparisonOut:
    if not payload.existing_project_ids:
        raise HTTPException(status_code=422, detail="existing_project_ids darf nicht leer sein.")
    if not payload.proposed_project_ids:
        raise HTTPException(status_code=422, detail="proposed_project_ids darf nicht leer sein.")

    existing_graph = await _build_graph_for_projects(payload.existing_project_ids, db)
    proposed_graph = await _build_graph_for_projects(payload.proposed_project_ids, db)

    result = _service.compare(existing_graph, proposed_graph)

    return SubgraphComparisonOut(
        decision=result.decision,
        is_redundant=result.is_redundant,
        is_superset=result.is_superset,
        is_equal=result.is_equal,
        existing_edges=result.existing_edges,
        proposed_edges=result.proposed_edges,
        message=result.message,
        existing_graph=_graph_to_schema(existing_graph),
        proposed_graph=_graph_to_schema(proposed_graph),
    )


@router.get(
    "/graph",
    response_model=BipartiteGraphOut,
    summary="Bipartiter Zuteilungsgraph",
    description=(
        "Liefert den bipartiten Graphen (Projekte × Ressourcen) "
        "für die angegebenen Projekt-IDs als Adjazenzmatrix und Knotenlisten."
    ),
)
async def get_bipartite_graph(
    project_ids: str,
    db: AsyncSession = Depends(get_db),
) -> BipartiteGraphOut:
    """Query-Parameter: project_ids=1,2,3  (kommagetrennte Ganzzahlen)"""
    try:
        ids = [int(x.strip()) for x in project_ids.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="project_ids muss eine kommagetrennte Liste von Ganzzahlen sein.",
        )
    if not ids:
        raise HTTPException(status_code=422, detail="Mindestens eine Projekt-ID erforderlich.")

    graph = await _build_graph_for_projects(ids, db)
    return _graph_to_schema(graph)


@router.post(
    "/redundancy",
    response_model=RedundancyCheckOut,
    summary="Redundanzprüfung mehrerer Projektgruppen",
    description=(
        "Vergleicht alle Paare der angegebenen Projektgruppen paarweise "
        "mit dem Subgraph Algorithmus und meldet redundante Graphen."
    ),
)
async def check_redundancy(
    groups: list[list[int]],
    db: AsyncSession = Depends(get_db),
) -> RedundancyCheckOut:
    if len(groups) < 2:
        raise HTTPException(status_code=422, detail="Mindestens zwei Projektgruppen erforderlich.")

    graphs = []
    for group in groups:
        g = await _build_graph_for_projects(group, db)
        graphs.append(g)

    redundant_pairs_raw = _service.find_redundant_allocations(graphs)
    optimal = _service.suggest_optimal_graph(graphs)

    redundant_pairs = []
    for i, j, result in redundant_pairs_raw:
        redundant_pairs.append({
            "graph_a_index": i,
            "graph_b_index": j,
            "decision":      result.decision,
            "is_redundant":  result.is_redundant,
            "is_superset":   result.is_superset,
            "message":       result.message,
        })

    optimal_index: int | None = None
    if optimal is not None:
        for idx, g in enumerate(graphs):
            if g is optimal:
                optimal_index = idx
                break

    n_groups = len(groups)
    n_redundant = len(redundant_pairs)
    summary = (
        f"{n_groups} Projektgruppen analysiert. "
        f"{n_redundant} redundante Paare gefunden."
        + (f" Optimaler Graph: Gruppe {optimal_index}." if optimal_index is not None else "")
    )

    return RedundancyCheckOut(
        total_graphs=n_groups,
        redundant_pairs=redundant_pairs,
        optimal_graph_index=optimal_index,
        summary=summary,
    )
