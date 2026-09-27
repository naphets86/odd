import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_create_person(client: AsyncClient):
    resp = await client.post("/api/persons/", json={
        "name": "Max Mustermann",
        "person_type": "Arbeitnehmer",
        "email": "max@example.com",
        "department": "IT",
        "notes": "Testnotiz",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Max Mustermann"
    assert data["person_type"] == "Arbeitnehmer"
    assert data["id"] is not None

async def test_list_persons_pagination(client: AsyncClient):
    for i in range(3):
        await client.post("/api/persons/", json={"name": f"Person{i}", "person_type": "Student"})
    resp = await client.get("/api/persons/?skip=1&limit=1")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


async def test_get_person(client: AsyncClient):
    create = await client.post("/api/persons/", json={"name": "Bob", "person_type": "Lehrer"})
    pid = create.json()["id"]
    resp = await client.get(f"/api/persons/{pid}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Bob"


async def test_get_person_not_found(client: AsyncClient):
    resp = await client.get("/api/persons/9999")
    assert resp.status_code == 404


async def test_update_person(client: AsyncClient):
    create = await client.post("/api/persons/", json={"name": "Carol", "person_type": "Student"})
    pid = create.json()["id"]
    resp = await client.patch(f"/api/persons/{pid}", json={"name": "Carol Updated", "department": "Mathe"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "Carol Updated"
    assert resp.json()["department"] == "Mathe"


async def test_update_person_not_found(client: AsyncClient):
    resp = await client.patch("/api/persons/9999", json={"name": "Ghost"})
    assert resp.status_code == 404


async def test_delete_person(client: AsyncClient):
    create = await client.post("/api/persons/", json={"name": "Dave", "person_type": "Professor"})
    pid = create.json()["id"]
    resp = await client.delete(f"/api/persons/{pid}")
    assert resp.status_code == 204
    resp2 = await client.get(f"/api/persons/{pid}")
    assert resp2.status_code == 404


async def test_delete_person_not_found(client: AsyncClient):
    resp = await client.delete("/api/persons/9999")
    assert resp.status_code == 404


async def test_all_person_types(client: AsyncClient):
    types = ["Arbeitnehmer", "Lehrer", "Schüler", "Student", "Arbeitgeber", "Professor"]
    for t in types:
        resp = await client.post("/api/persons/", json={"name": f"Test {t}", "person_type": t})
        assert resp.status_code == 201, f"Failed for type {t}: {resp.json()}"
