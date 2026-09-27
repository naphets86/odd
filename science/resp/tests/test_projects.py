import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_create_project(client: AsyncClient):
    resp = await client.post("/api/projects/", json={
        "name": "Bauprojekt Alpha",
        "description": "Neues Gebäude",
        "start_date": "2025-01-01T00:00:00",
        "end_date": "2025-12-31T00:00:00",
        "status": "geplant",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Bauprojekt Alpha"
    assert data["status"] == "geplant"


async def test_list_projects(client: AsyncClient):
    await client.post("/api/projects/", json={"name": "Proj1", "status": "aktiv"})
    resp = await client.get("/api/projects/")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


async def test_list_projects_pagination(client: AsyncClient):
    for i in range(4):
        await client.post("/api/projects/", json={"name": f"P{i}", "status": "geplant"})
    resp = await client.get("/api/projects/?skip=2&limit=2")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


async def test_get_project(client: AsyncClient):
    create = await client.post("/api/projects/", json={"name": "Test Proj", "status": "geplant"})
    pid = create.json()["id"]
    resp = await client.get(f"/api/projects/{pid}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Test Proj"


async def test_get_project_not_found(client: AsyncClient):
    resp = await client.get("/api/projects/9999")
    assert resp.status_code == 404


async def test_update_project(client: AsyncClient):
    create = await client.post("/api/projects/", json={"name": "Old Name", "status": "geplant"})
    pid = create.json()["id"]
    resp = await client.patch(f"/api/projects/{pid}", json={"name": "New Name", "status": "aktiv"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "New Name"
    assert resp.json()["status"] == "aktiv"


async def test_update_project_not_found(client: AsyncClient):
    resp = await client.patch("/api/projects/9999", json={"name": "Ghost"})
    assert resp.status_code == 404


async def test_delete_project(client: AsyncClient):
    create = await client.post("/api/projects/", json={"name": "To Delete", "status": "abgebrochen"})
    pid = create.json()["id"]
    resp = await client.delete(f"/api/projects/{pid}")
    assert resp.status_code == 204
    resp2 = await client.get(f"/api/projects/{pid}")
    assert resp2.status_code == 404


async def test_delete_project_not_found(client: AsyncClient):
    resp = await client.delete("/api/projects/9999")
    assert resp.status_code == 404


async def test_all_project_statuses(client: AsyncClient):
    statuses = ["geplant", "aktiv", "abgeschlossen", "abgebrochen"]
    for s in statuses:
        resp = await client.post("/api/projects/", json={"name": f"Proj {s}", "status": s})
        assert resp.status_code == 201
