"""Tests für Auth-Endpunkte: Register, Login, /me."""

import pytest


@pytest.mark.asyncio
async def test_register_success(client):
    resp = await client.post(
        "/api/auth/register",
        json={
            "username": "alice",
            "display_name": "Alice",
            "email": "alice@ws-test.example",
            "password": "password123",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["username"] == "alice"
    assert "id" in data
    assert "password_hash" not in data


@pytest.mark.asyncio
async def test_register_duplicate_username(client):
    payload = {
        "username": "user_c1",
        "display_name": "User C1",
        "email": "user_c1@ws-test.example",
        "password": "password123",
    }
    r1 = await client.post("/api/auth/register", json=payload)
    assert r1.status_code == 201

    payload2 = payload.copy()
    payload2["email"] = "user_c2@ws-test.example"
    r2 = await client.post("/api/auth/register", json=payload2)
    assert r2.status_code == 400
    assert "Benutzername" in r2.json()["detail"]


@pytest.mark.asyncio
async def test_register_duplicate_email(client):
    payload = {
        "username": "user_c2",
        "display_name": "User C2",
        "email": "shared@ws-test.example",
        "password": "password123",
    }
    r1 = await client.post("/api/auth/register", json=payload)
    assert r1.status_code == 201

    payload2 = payload.copy()
    payload2["username"] = "user_c3"
    r2 = await client.post("/api/auth/register", json=payload2)
    assert r2.status_code == 400
    assert "E-Mail" in r2.json()["detail"]


@pytest.mark.asyncio
async def test_register_short_username(client):
    resp = await client.post(
        "/api/auth/register",
        json={
            "username": "ab",
            "display_name": "AB",
            "email": "ab@ws-test.example",
            "password": "password123",
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_register_short_password(client):
    resp = await client.post(
        "/api/auth/register",
        json={
            "username": "validuser",
            "display_name": "Valid",
            "email": "valid@ws-test.example",
            "password": "abc",
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_login_success(client, registered_user):
    resp = await client.post(
        "/api/auth/login",
        json={"username": "testuser", "password": "password123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_wrong_password(client, registered_user):
    resp = await client.post(
        "/api/auth/login",
        json={"username": "testuser", "password": "wrongpassword"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_login_unknown_user(client):
    resp = await client.post(
        "/api/auth/login",
        json={"username": "nobody", "password": "password123"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_me_authenticated(client, auth_headers):
    resp = await client.get("/api/auth/me", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["username"] == "testuser"


@pytest.mark.asyncio
async def test_me_unauthenticated(client):
    resp = await client.get("/api/auth/me")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_me_invalid_token(client):
    resp = await client.get(
        "/api/auth/me", headers={"Authorization": "Bearer invalidtoken"}
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_update_key(client, auth_headers):
    resp = await client.put(
        "/api/auth/me/key",
        headers=auth_headers,
        json={"sc_key_a": "111", "sc_key_b": "222", "sc_key_p": "999", "sc_key_n": 4},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["sc_key_a"] == "111"
    assert data["sc_key_n"] == 4
