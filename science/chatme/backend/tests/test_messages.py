"""Tests für Nachrichten-Endpunkte."""

import pytest


async def _create_user(client, username):
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
    token = r.json()["access_token"]
    me_resp = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    return token, me_resp.json()["id"]


@pytest.mark.asyncio
async def test_send_message(client, auth_headers, registered_user):
    token2, user2_id = await _create_user(client, "msg_bob")

    resp = await client.post(
        "/api/messages",
        headers=auth_headers,
        json={"recipient_id": user2_id, "ciphertext": '{"len":5,"blocks":[1,2,3]}'},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["ciphertext"] == '{"len":5,"blocks":[1,2,3]}'
    assert data["is_deleted"] is False
    assert data["is_read"] is False


@pytest.mark.asyncio
async def test_get_history(client, auth_headers, registered_user):
    token2, user2_id = await _create_user(client, "msg_carol")

    await client.post(
        "/api/messages",
        headers=auth_headers,
        json={"recipient_id": user2_id, "ciphertext": "hello"},
    )

    resp = await client.get(f"/api/messages/history/{user2_id}", headers=auth_headers)
    assert resp.status_code == 200
    assert len(resp.json()) == 1


@pytest.mark.asyncio
async def test_get_history_empty(client, auth_headers):
    import uuid
    resp = await client.get(
        f"/api/messages/history/{uuid.uuid4()}", headers=auth_headers
    )
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_edit_message(client, auth_headers, registered_user):
    _, user2_id = await _create_user(client, "msg_dave")

    send_resp = await client.post(
        "/api/messages",
        headers=auth_headers,
        json={"recipient_id": user2_id, "ciphertext": "original"},
    )
    msg_id = send_resp.json()["id"]

    edit_resp = await client.put(
        f"/api/messages/{msg_id}",
        headers=auth_headers,
        json={"ciphertext": "edited"},
    )
    assert edit_resp.status_code == 200
    data = edit_resp.json()
    assert data["ciphertext"] == "edited"
    assert data["edited_at"] is not None


@pytest.mark.asyncio
async def test_edit_message_not_owner(client, auth_headers, registered_user):
    token2, user2_id = await _create_user(client, "msg_eve")
    headers2 = {"Authorization": f"Bearer {token2}"}

    # Alice sendet an Bob
    send_resp = await client.post(
        "/api/messages",
        headers=auth_headers,
        json={"recipient_id": user2_id, "ciphertext": "secret"},
    )
    msg_id = send_resp.json()["id"]

    # Bob versucht zu bearbeiten
    resp = await client.put(
        f"/api/messages/{msg_id}",
        headers=headers2,
        json={"ciphertext": "hacked"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_message(client, auth_headers, registered_user):
    _, user2_id = await _create_user(client, "msg_frank")

    send_resp = await client.post(
        "/api/messages",
        headers=auth_headers,
        json={"recipient_id": user2_id, "ciphertext": "to delete"},
    )
    msg_id = send_resp.json()["id"]

    del_resp = await client.delete(f"/api/messages/{msg_id}", headers=auth_headers)
    assert del_resp.status_code == 204

    history = await client.get(
        f"/api/messages/history/{user2_id}", headers=auth_headers
    )
    assert all(m["id"] != msg_id for m in history.json())


@pytest.mark.asyncio
async def test_mark_read(client, auth_headers, registered_user):
    token2, user2_id = await _create_user(client, "msg_grace")
    headers2 = {"Authorization": f"Bearer {token2}"}

    # Alice sendet an Bob
    me_resp = await client.get("/api/auth/me", headers=auth_headers)
    alice_id = me_resp.json()["id"]

    send_resp = await client.post(
        "/api/messages",
        headers=auth_headers,
        json={"recipient_id": user2_id, "ciphertext": "read me"},
    )
    msg_id = send_resp.json()["id"]

    # Bob markiert als gelesen
    read_resp = await client.post(
        f"/api/messages/{msg_id}/read", headers=headers2
    )
    assert read_resp.status_code == 204


@pytest.mark.asyncio
async def test_delete_message_not_found(client, auth_headers):
    import uuid
    resp = await client.delete(
        f"/api/messages/{uuid.uuid4()}", headers=auth_headers
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_send_message_unauthenticated(client):
    import uuid
    resp = await client.post(
        "/api/messages",
        json={"recipient_id": str(uuid.uuid4()), "ciphertext": "test"},
    )
    assert resp.status_code == 401
