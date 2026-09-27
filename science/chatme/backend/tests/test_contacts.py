"""Tests für Kontakte-Endpunkte."""

import pytest


async def _register_and_login(client, username):
    await client.post(
        "/api/auth/register",
        json={
            "username": username,
            "display_name": username,
            "email": f"{username}@ws-test.example",
            "password": "password123",
        },
    )
    r = await client.post(
        "/api/auth/login",
        json={"username": username, "password": "password123"},
    )
    return r.json()["access_token"]


@pytest.mark.asyncio
async def test_list_contacts_empty(client, auth_headers):
    resp = await client.get("/api/contacts", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_add_contact(client, auth_headers):
    # Zweiten User erstellen
    token2 = await _register_and_login(client, "contact_bob")
    _ = token2  # Bob muss nur existieren

    resp = await client.post(
        "/api/contacts",
        headers=auth_headers,
        json={"username": "contact_bob"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["contact_user"]["username"] == "contact_bob"


@pytest.mark.asyncio
async def test_add_contact_self(client, auth_headers):
    resp = await client.post(
        "/api/contacts",
        headers=auth_headers,
        json={"username": "testuser"},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_add_contact_not_found(client, auth_headers):
    resp = await client.post(
        "/api/contacts",
        headers=auth_headers,
        json={"username": "nonexistent_xyz"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_add_contact_duplicate(client, auth_headers):
    await _register_and_login(client, "contact_carol")

    await client.post(
        "/api/contacts",
        headers=auth_headers,
        json={"username": "contact_carol"},
    )
    resp = await client.post(
        "/api/contacts",
        headers=auth_headers,
        json={"username": "contact_carol"},
    )
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_list_contacts_after_add(client, auth_headers):
    await _register_and_login(client, "contact_dave")
    await client.post(
        "/api/contacts",
        headers=auth_headers,
        json={"username": "contact_dave"},
    )
    resp = await client.get("/api/contacts", headers=auth_headers)
    assert resp.status_code == 200
    usernames = [c["contact_user"]["username"] for c in resp.json()]
    assert "contact_dave" in usernames


@pytest.mark.asyncio
async def test_delete_contact(client, auth_headers):
    await _register_and_login(client, "contact_eve")
    add_resp = await client.post(
        "/api/contacts",
        headers=auth_headers,
        json={"username": "contact_eve"},
    )
    contact_id = add_resp.json()["id"]

    del_resp = await client.delete(f"/api/contacts/{contact_id}", headers=auth_headers)
    assert del_resp.status_code == 204

    list_resp = await client.get("/api/contacts", headers=auth_headers)
    ids = [c["id"] for c in list_resp.json()]
    assert contact_id not in ids


@pytest.mark.asyncio
async def test_delete_contact_not_found(client, auth_headers):
    import uuid
    resp = await client.delete(
        f"/api/contacts/{uuid.uuid4()}", headers=auth_headers
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_add_contact_with_sc_keys(client, auth_headers):
    await _register_and_login(client, "contact_frank")
    resp = await client.post(
        "/api/contacts",
        headers=auth_headers,
        json={
            "username": "contact_frank",
            "sc_key_a": "123",
            "sc_key_b": "456",
            "sc_key_p": "789",
            "sc_key_n": 4,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["sc_key_a"] == "123"
    assert data["sc_key_n"] == 4
