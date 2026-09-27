import enum
from datetime import datetime, timezone
from sqlalchemy import String, Integer, Float, DateTime, Enum, ForeignKey, Text
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


def utc_now():
    return datetime.now(timezone.utc)


# ── Enumerations ──────────────────────────────────────────────────────────────

class PersonType(str, enum.Enum):
    ARBEITNEHMER = "Arbeitnehmer"
    LEHRER = "Lehrer"
    SCHUELER = "Schüler"
    STUDENT = "Student"
    ARBEITGEBER = "Arbeitgeber"
    PROFESSOR = "Professor"


class MaterialType(str, enum.Enum):
    GELD = "Geld"
    METALL = "Metall"
    STAHL = "Stahl"
    OEL = "Öl"
    GAS = "Gas"
    ZUCKERRUEBEN = "Zuckerrüben"
    HOLZ = "Holz"
    SILIZIUM = "Silizium"
    KUPFER = "Kupfer"
    KREIDE = "Kreide"
    SCHWAMM = "Schwamm"
    BEAMER = "Beamer"
    LAPTOP = "Laptop"
    PC_WORKSTATION = "PC/Workstation"


class TimeUnit(str, enum.Enum):
    SEKUNDEN = "Sekunden"
    MINUTEN = "Minuten"
    STUNDEN = "Stunden"
    TAGE = "Tage"
    WOCHEN = "Wochen"
    MONATE = "Monate"
    JAHRE = "Jahre"


class AllocationStatus(str, enum.Enum):
    GEPLANT = "geplant"
    AKTIV = "aktiv"
    ABGESCHLOSSEN = "abgeschlossen"
    ABGEBROCHEN = "abgebrochen"


# ── Resource Models ───────────────────────────────────────────────────────────

class Person(Base):
    __tablename__ = "persons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    person_type: Mapped[PersonType] = mapped_column(Enum(PersonType), nullable=False)
    email: Mapped[str | None] = mapped_column(String(200), unique=True, nullable=True)
    department: Mapped[str | None] = mapped_column(String(200), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=utc_now, onupdate=utc_now)

    allocations: Mapped[list["Allocation"]] = relationship(
        "Allocation", back_populates="person", cascade="all, delete-orphan"
    )


class Material(Base):
    __tablename__ = "materials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    material_type: Mapped[MaterialType] = mapped_column(Enum(MaterialType), nullable=False)
    quantity: Mapped[float] = mapped_column(Float, default=0.0)
    unit: Mapped[str] = mapped_column(String(50), default="Stück")
    location: Mapped[str | None] = mapped_column(String(200), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=utc_now, onupdate=utc_now)

    allocations: Mapped[list["Allocation"]] = relationship(
        "Allocation", back_populates="material", cascade="all, delete-orphan"
    )


class TimeResource(Base):
    __tablename__ = "time_resources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[TimeUnit] = mapped_column(Enum(TimeUnit), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=utc_now, onupdate=utc_now)

    allocations: Mapped[list["Allocation"]] = relationship(
        "Allocation", back_populates="time_resource", cascade="all, delete-orphan"
    )


# ── Project & Allocation Models ───────────────────────────────────────────────

class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    start_date: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    end_date: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    status: Mapped[AllocationStatus] = mapped_column(Enum(AllocationStatus), default=AllocationStatus.GEPLANT)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=utc_now, onupdate=utc_now)

    allocations: Mapped[list["Allocation"]] = relationship(
        "Allocation", back_populates="project", cascade="all, delete-orphan"
    )


class Allocation(Base):
    __tablename__ = "allocations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False)
    person_id: Mapped[int | None] = mapped_column(ForeignKey("persons.id"), nullable=True)
    material_id: Mapped[int | None] = mapped_column(ForeignKey("materials.id"), nullable=True)
    time_resource_id: Mapped[int | None] = mapped_column(ForeignKey("time_resources.id"), nullable=True)
    quantity: Mapped[float] = mapped_column(Float, default=1.0)
    start_date: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    end_date: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    status: Mapped[AllocationStatus] = mapped_column(Enum(AllocationStatus), default=AllocationStatus.GEPLANT)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=utc_now)

    project: Mapped["Project"] = relationship("Project", back_populates="allocations")
    person: Mapped["Person | None"] = relationship("Person", back_populates="allocations")
    material: Mapped["Material | None"] = relationship("Material", back_populates="allocations")
    time_resource: Mapped["TimeResource | None"] = relationship("TimeResource", back_populates="allocations")
