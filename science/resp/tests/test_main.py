import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_health(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_enums(client: AsyncClient):
    resp = await client.get("/api/enums")
    assert resp.status_code == 200
    data = resp.json()
    assert "Arbeitnehmer" in data["person_types"]
    assert "Lehrer" in data["person_types"]
    assert "Schüler" in data["person_types"]
    assert "Student" in data["person_types"]
    assert "Arbeitgeber" in data["person_types"]
    assert "Professor" in data["person_types"]
    assert "Stahl" in data["material_types"]
    assert "Geld" in data["material_types"]
    assert "Stunden" in data["time_units"]
    assert "Wochen" in data["time_units"]
    assert "geplant" in data["allocation_statuses"]
    assert "aktiv" in data["allocation_statuses"]
    assert "abgeschlossen" in data["allocation_statuses"]
    assert "abgebrochen" in data["allocation_statuses"]


async def test_openapi_docs(client: AsyncClient):
    resp = await client.get("/docs")
    assert resp.status_code == 200


async def test_openapi_json(client: AsyncClient):
    resp = await client.get("/openapi.json")
    assert resp.status_code == 200
    data = resp.json()
    assert "paths" in data
