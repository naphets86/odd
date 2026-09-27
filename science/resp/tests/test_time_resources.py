import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_create_time_resource(client: AsyncClient):
    resp = await client.post("/api/time-resources/", json={
        "name": "Sprint Dauer",
        "amount": 2.0,
        "unit": "Wochen",
        "notes": "Zwei-Wochen-Sprint",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["unit"] == "Wochen"
    assert data["amount"] == 2.0


async def test_list_time_resources(client: AsyncClient):
    await client.post("/api/time-resources/", json={"name": "T1", "amount": 1, "unit": "Stunden"})
    resp = await client.get("/api/time-resources/")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


async def test_list_time_resources_pagination(client: AsyncClient):
    for i in range(3):
        await client.post("/api/time-resources/", json={"name": f"TR{i}", "amount": i + 1, "unit": "Tage"})
    resp = await client.get("/api/time-resources/?skip=1&limit=2")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


async def test_get_time_resource(client: AsyncClient):
    create = await client.post("/api/time-resources/", json={"name": "Monat Plan", "amount": 1, "unit": "Monate"})
    tr_id = create.json()["id"]
    resp = await client.get(f"/api/time-resources/{tr_id}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Monat Plan"


async def test_get_time_resource_not_found(client: AsyncClient):
    resp = await client.get("/api/time-resources/9999")
    assert resp.status_code == 404


async def test_update_time_resource(client: AsyncClient):
    create = await client.post("/api/time-resources/", json={"name": "Old", "amount": 5, "unit": "Minuten"})
    tr_id = create.json()["id"]
    resp = await client.patch(f"/api/time-resources/{tr_id}", json={"amount": 10, "unit": "Stunden"})
    assert resp.status_code == 200
    assert resp.json()["amount"] == 10
    assert resp.json()["unit"] == "Stunden"


async def test_update_time_resource_not_found(client: AsyncClient):
    resp = await client.patch("/api/time-resources/9999", json={"amount": 1})
    assert resp.status_code == 404


async def test_delete_time_resource(client: AsyncClient):
    create = await client.post("/api/time-resources/", json={"name": "Del", "amount": 3, "unit": "Tage"})
    tr_id = create.json()["id"]
    resp = await client.delete(f"/api/time-resources/{tr_id}")
    assert resp.status_code == 204
    resp2 = await client.get(f"/api/time-resources/{tr_id}")
    assert resp2.status_code == 404


async def test_delete_time_resource_not_found(client: AsyncClient):
    resp = await client.delete("/api/time-resources/9999")
    assert resp.status_code == 404


async def test_all_time_units(client: AsyncClient):
    units = ["Sekunden", "Minuten", "Stunden", "Tage", "Wochen", "Monate", "Jahre"]
    for u in units:
        resp = await client.post("/api/time-resources/", json={"name": f"Zeit {u}", "amount": 1, "unit": u})
        assert resp.status_code == 201, f"Failed for {u}: {resp.json()}"
