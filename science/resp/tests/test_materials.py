import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_create_material(client: AsyncClient):
    resp = await client.post("/api/materials/", json={
        "name": "Stahlträger A",
        "material_type": "Stahl",
        "quantity": 50.0,
        "unit": "Tonnen",
        "location": "Lager 1",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["material_type"] == "Stahl"
    assert data["quantity"] == 50.0


async def test_list_materials(client: AsyncClient):
    await client.post("/api/materials/", json={"name": "Kupfer", "material_type": "Kupfer", "quantity": 10})
    resp = await client.get("/api/materials/")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


async def test_list_materials_pagination(client: AsyncClient):
    for i in range(3):
        await client.post("/api/materials/", json={"name": f"Mat{i}", "material_type": "Holz", "quantity": i})
    resp = await client.get("/api/materials/?skip=0&limit=2")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


async def test_get_material(client: AsyncClient):
    create = await client.post("/api/materials/", json={"name": "Gas X", "material_type": "Gas", "quantity": 100})
    mid = create.json()["id"]
    resp = await client.get(f"/api/materials/{mid}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Gas X"


async def test_get_material_not_found(client: AsyncClient):
    resp = await client.get("/api/materials/9999")
    assert resp.status_code == 404


async def test_update_material(client: AsyncClient):
    create = await client.post("/api/materials/", json={"name": "Öl", "material_type": "Öl", "quantity": 200})
    mid = create.json()["id"]
    resp = await client.patch(f"/api/materials/{mid}", json={"quantity": 300.5, "location": "Tank A"})
    assert resp.status_code == 200
    assert resp.json()["quantity"] == 300.5
    assert resp.json()["location"] == "Tank A"


async def test_update_material_not_found(client: AsyncClient):
    resp = await client.patch("/api/materials/9999", json={"quantity": 1})
    assert resp.status_code == 404


async def test_delete_material(client: AsyncClient):
    create = await client.post("/api/materials/", json={"name": "Holz2", "material_type": "Holz", "quantity": 5})
    mid = create.json()["id"]
    resp = await client.delete(f"/api/materials/{mid}")
    assert resp.status_code == 204
    resp2 = await client.get(f"/api/materials/{mid}")
    assert resp2.status_code == 404


async def test_delete_material_not_found(client: AsyncClient):
    resp = await client.delete("/api/materials/9999")
    assert resp.status_code == 404


async def test_all_material_types(client: AsyncClient):
    types = ["Geld", "Metall", "Stahl", "Öl", "Gas", "Zuckerrüben", "Holz",
             "Silizium", "Kupfer", "Kreide", "Schwamm", "Beamer", "Laptop", "PC/Workstation"]
    for t in types:
        resp = await client.post("/api/materials/", json={"name": f"Mat {t}", "material_type": t, "quantity": 1})
        assert resp.status_code == 201, f"Failed for {t}: {resp.json()}"
