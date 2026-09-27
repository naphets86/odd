from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.models import PersonType, MaterialType, TimeUnit, AllocationStatus
from app.routers import persons, materials, time_resources, projects, allocations
from app.routers import subgraph as subgraph_router
from app.schemas import EnumsOut


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="RESP – Ressourcenverwaltung",
    description="Effiziente Verwaltung und Planung von Menschen, Material und Zeit",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routers
app.include_router(persons.router,         prefix=settings.API_PREFIX)
app.include_router(materials.router,       prefix=settings.API_PREFIX)
app.include_router(time_resources.router,  prefix=settings.API_PREFIX)
app.include_router(projects.router,        prefix=settings.API_PREFIX)
app.include_router(allocations.router,     prefix=settings.API_PREFIX)
app.include_router(subgraph_router.router, prefix=settings.API_PREFIX)


@app.get(f"{settings.API_PREFIX}/enums", response_model=EnumsOut, tags=["Meta"])
async def get_enums():
    return EnumsOut(
        person_types=[e.value for e in PersonType],
        material_types=[e.value for e in MaterialType],
        time_units=[e.value for e in TimeUnit],
        allocation_statuses=[e.value for e in AllocationStatus],
    )


@app.get("/health", tags=["Meta"])
async def health():
    return {"status": "ok"}
