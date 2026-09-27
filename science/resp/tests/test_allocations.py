import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def _create_project(client, name="Testprojekt"):
    resp = await client.post("/api/projects/", json={"name": name, "status": "geplant"})
    return resp.json()["id"]


async def _create_person(client, name="Anna"):
    resp = await client.post("/api/persons/", json={"name": name, "person_type": "Arbeitnehmerin"})
    return resp.json()["id"]


async def _create_material(client, name="Stahl"):
    resp = await client.post("/api/materials/", json={"name": name, "material_type": "Stahl", "quantity": 100})
    return resp.json()["id"]


async def _create_time(client, name="Sprint"):
    resp = await client.post("/api/time-resources/", json={"name": name, "amount": 2, "unit": "Wochen"})
    return resp.json()["id"]

async def test_create_allocation_material(client: AsyncClient):
    pid = await _create_project(client)
    mat_id = await _create_material(client)
    resp = await client.post("/api/allocations/", json={
        "project_id": pid,
        "material_id": mat_id,
        "quantity": 20.0,
        "status": "aktiv",
    })
    assert resp.status_code == 201
    assert resp.json()["material_id"] == mat_id

async def test_create_allocation_time(client: AsyncClient):
    pid = await _create_project(client)
    tr_id = await _create_time(client)
    resp = await client.post("/api/allocations/", json={
        "project_id": pid,
        "time_resource_id": tr_id,
        "quantity": 3.0,
        "start_date": "2025-03-01T00:00:00",
        "end_date": "2025-03-21T00:00:00",
        "notes": "3 Sprints",
    })
    assert resp.status_code == 201
    assert resp.json()["time_resource_id"] == tr_id


async def test_list_allocations(client: AsyncClient):
    pid = await _create_project(client)
    await client.post("/api/allocations/", json={"project_id": pid, "quantity": 1})
    resp = await client.get("/api/allocations/")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


async def test_list_allocations_filter_by_project(client: AsyncClient):
    pid1 = await _create_project(client, "Proj A")
    pid2 = await _create_project(client, "Proj B")
    await client.post("/api/allocations/", json={"project_id": pid1, "quantity": 1})
    await client.post("/api/allocations/", json={"project_id": pid2, "quantity": 1})
    resp = await client.get(f"/api/allocations/?project_id={pid1}")
    assert resp.status_code == 200
    for a in resp.json():
        assert a["project_id"] == pid1


async def test_list_allocations_pagination(client: AsyncClient):
    pid = await _create_project(client)
    for _ in range(5):
        await client.post("/api/allocations/", json={"project_id": pid, "quantity": 1})
    resp = await client.get("/api/allocations/?skip=2&limit=2")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


async def test_get_allocation(client: AsyncClient):
    pid = await _create_project(client)
    create = await client.post("/api/allocations/", json={"project_id": pid, "quantity": 5})
    aid = create.json()["id"]
    resp = await client.get(f"/api/allocations/{aid}")
    assert resp.status_code == 200
    assert resp.json()["id"] == aid


async def test_get_allocation_not_found(client: AsyncClient):
    resp = await client.get("/api/allocations/9999")
    assert resp.status_code == 404


async def test_update_allocation(client: AsyncClient):
    pid = await _create_project(client)
    create = await client.post("/api/allocations/", json={"project_id": pid, "quantity": 1, "status": "geplant"})
    aid = create.json()["id"]
    resp = await client.patch(f"/api/allocations/{aid}", json={"quantity": 5, "status": "aktiv"})
    assert resp.status_code == 200
    assert resp.json()["quantity"] == 5
    assert resp.json()["status"] == "aktiv"


async def test_update_allocation_not_found(client: AsyncClient):
    resp = await client.patch("/api/allocations/9999", json={"quantity": 1})
    assert resp.status_code == 404


async def test_delete_allocation(client: AsyncClient):
    pid = await _create_project(client)
    create = await client.post("/api/allocations/", json={"project_id": pid, "quantity": 1})
    aid = create.json()["id"]
    resp = await client.delete(f"/api/allocations/{aid}")
    assert resp.status_code == 204
    resp2 = await client.get(f"/api/allocations/{aid}")
    assert resp2.status_code == 404


async def test_delete_allocation_not_found(client: AsyncClient):
    resp = await client.delete("/api/allocations/9999")
    assert resp.status_code == 404
