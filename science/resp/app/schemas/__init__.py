from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from app.models import PersonType, MaterialType, TimeUnit, AllocationStatus


# ── Person Schemas ────────────────────────────────────────────────────────────

class PersonBase(BaseModel):
    name: str
    person_type: PersonType
    email: Optional[str] = None
    department: Optional[str] = None
    notes: Optional[str] = None


class PersonCreate(PersonBase):
    pass


class PersonUpdate(BaseModel):
    name: Optional[str] = None
    person_type: Optional[PersonType] = None
    email: Optional[str] = None
    department: Optional[str] = None
    notes: Optional[str] = None


class PersonOut(PersonBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ── Material Schemas ──────────────────────────────────────────────────────────

class MaterialBase(BaseModel):
    name: str
    material_type: MaterialType
    quantity: float = 0.0
    unit: str = "Stück"
    location: Optional[str] = None
    notes: Optional[str] = None


class MaterialCreate(MaterialBase):
    pass


class MaterialUpdate(BaseModel):
    name: Optional[str] = None
    material_type: Optional[MaterialType] = None
    quantity: Optional[float] = None
    unit: Optional[str] = None
    location: Optional[str] = None
    notes: Optional[str] = None


class MaterialOut(MaterialBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ── Time Resource Schemas ─────────────────────────────────────────────────────

class TimeResourceBase(BaseModel):
    name: str
    amount: float
    unit: TimeUnit
    notes: Optional[str] = None


class TimeResourceCreate(TimeResourceBase):
    pass


class TimeResourceUpdate(BaseModel):
    name: Optional[str] = None
    amount: Optional[float] = None
    unit: Optional[TimeUnit] = None
    notes: Optional[str] = None


class TimeResourceOut(TimeResourceBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ── Project Schemas ───────────────────────────────────────────────────────────

class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    status: AllocationStatus = AllocationStatus.GEPLANT


class ProjectCreate(ProjectBase):
    pass


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    status: Optional[AllocationStatus] = None


class ProjectOut(ProjectBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ── Allocation Schemas ────────────────────────────────────────────────────────

class AllocationBase(BaseModel):
    project_id: int
    person_id: Optional[int] = None
    material_id: Optional[int] = None
    time_resource_id: Optional[int] = None
    quantity: float = 1.0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    status: AllocationStatus = AllocationStatus.GEPLANT
    notes: Optional[str] = None


class AllocationCreate(AllocationBase):
    pass


class AllocationUpdate(BaseModel):
    quantity: Optional[float] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    status: Optional[AllocationStatus] = None
    notes: Optional[str] = None


class AllocationOut(AllocationBase):
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Enum Lists ────────────────────────────────────────────────────────────────

class EnumsOut(BaseModel):
    person_types: list[str]
    material_types: list[str]
    time_units: list[str]
    allocation_statuses: list[str]


# ── Bipartiter Graph / Subgraph Algorithmus ───────────────────────────────────

class ResourceNodeOut(BaseModel):
    """Ressourcen-Knoten (rechte Partition des bipartiten Graphen)."""
    id: int
    kind: str    # "person" | "material" | "time"
    label: str


class ProjectNodeOut(BaseModel):
    """Projekt-Knoten (linke Partition des bipartiten Graphen)."""
    id: int
    name: str


class EdgeOut(BaseModel):
    """Eine Kante im bipartiten Graphen: (Projekt → Ressource)."""
    project_id: int
    project_name: str
    resource_id: int
    resource_kind: str
    resource_label: str


class BipartiteGraphOut(BaseModel):
    """Vollständige Repräsentation eines Zuteilungsgraphen."""
    projects: list[ProjectNodeOut]
    resources: list[ResourceNodeOut]
    edges: list[EdgeOut]
    n_projects: int
    n_resources: int
    n_edges: int
    matrix: list[list[int]]


class SubgraphComparisonRequest(BaseModel):
    """
    Anfrage für den Subgraph-Vergleich zweier Projektmengen.
    existing_project_ids : Projekte des bestehenden Graphen A
    proposed_project_ids : Projekte des vorgeschlagenen Graphen B
    """
    existing_project_ids: list[int]
    proposed_project_ids: list[int]


class SubgraphComparisonOut(BaseModel):
    """Ergebnis des Subgraph Algorithmus für zwei Zuteilungsgraphen."""
    decision: str
    is_redundant: bool
    is_superset: bool
    is_equal: bool
    existing_edges: int
    proposed_edges: int
    message: str
    existing_graph: BipartiteGraphOut
    proposed_graph: BipartiteGraphOut


class RedundancyCheckOut(BaseModel):
    """Ergebnis der Redundanzprüfung über mehrere Projektgruppen."""
    total_graphs: int
    redundant_pairs: list[dict]
    optimal_graph_index: Optional[int]
    summary: str
