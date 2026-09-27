"""Tests für Anrufe-Endpunkte."""

import pytest


@pytest.mark.asyncio
async def test_list_calls_empty(client, auth_headers):
    resp = await client.get("/api/calls", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_list_calls_unauthenticated(client):
    resp = await client.get("/api/calls")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_list_calls_after_ws_call(client, auth_headers, db_session):
    """Direkt einen Call-Eintrag in die DB schreiben und dann via API abfragen."""
    import uuid
    from datetime import datetime, timezone
    from app.models.models import Call, CallStatus, User
    from app.core.security import hash_password

    me_resp = await client.get("/api/auth/me", headers=auth_headers)
    caller_id = uuid.UUID(me_resp.json()["id"])

    # Zweiten User direkt in DB anlegen
    callee = User(
        username="callee_test",
        display_name="Callee",
        email="callee@ws-test.example",
        password_hash=hash_password("password123"),
    )
    db_session.add(callee)
    await db_session.commit()
    await db_session.refresh(callee)

    call = Call(
        caller_id=caller_id,
        callee_id=callee.id,
        status=CallStatus.ended,
        started_at=datetime.now(timezone.utc),
    )
    db_session.add(call)
    await db_session.commit()

    resp = await client.get("/api/calls", headers=auth_headers)
    assert resp.status_code == 200
    assert len(resp.json()) == 1
    assert resp.json()[0]["status"] == "ended"
