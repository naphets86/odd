"""
WebSocket-Tests – verwenden starlette.testclient.TestClient (sync),
weil ASGITransport keine WS-Upgrades unterstützt.
"""

import json
import pytest
from tests.conftest import _sync_register, _sync_login


# ── Auth ───────────────────────────────────────────────────────────────────────


def test_ws_auth_success(ws_client):
    _sync_register(ws_client, "ws_alice")
    token = _sync_login(ws_client, "ws_alice")

    with ws_client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "auth", "payload": {"token": token}}))
        msg = json.loads(ws.receive_text())
    assert msg["type"] == "authenticated"
    assert "user_id" in msg["payload"]


def test_ws_auth_invalid_token(ws_client):
    with ws_client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "auth", "payload": {"token": "bad"}}))
        msg = json.loads(ws.receive_text())
    assert msg["type"] == "error"


def test_ws_unauthenticated_message(ws_client):
    _sync_register(ws_client, "ws_anon")
    import uuid

    with ws_client.websocket_connect("/ws") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "message",
                    "payload": {
                        "recipient_id": str(uuid.uuid4()),
                        "ciphertext": "test",
                    },
                }
            )
        )
        msg = json.loads(ws.receive_text())
    assert msg["type"] == "error"
    assert "authentifiziert" in msg["payload"]["message"].lower()


def test_ws_invalid_json(ws_client):
    with ws_client.websocket_connect("/ws") as ws:
        ws.send_text("this is not json")
        msg = json.loads(ws.receive_text())
    assert msg["type"] == "error"


# ── Chat Messages ──────────────────────────────────────────────────────────────


def test_ws_send_message(ws_client):
    _sync_register(ws_client, "ws_sender")
    _sync_register(ws_client, "ws_recvr")
    token_s = _sync_login(ws_client, "ws_sender")
    token_r = _sync_login(ws_client, "ws_recvr")

    # Empfänger-ID via REST holen
    me_resp = ws_client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {token_r}"}
    )
    recvr_id = me_resp.json()["id"]

    with ws_client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "auth", "payload": {"token": token_s}}))
        _ = ws.receive_text()  # authenticated

        ws.send_text(
            json.dumps(
                {
                    "type": "message",
                    "payload": {
                        "recipient_id": recvr_id,
                        "ciphertext": "hello world",
                        "tmp_id": "tmp-1",
                    },
                }
            )
        )
        msg = json.loads(ws.receive_text())

    assert msg["type"] == "message_new"
    assert msg["payload"]["ciphertext"] == "hello world"


def test_ws_message_edit(ws_client):
    _sync_register(ws_client, "ws_edit_u")
    _sync_register(ws_client, "ws_edit_r")
    token = _sync_login(ws_client, "ws_edit_u")
    token_r = _sync_login(ws_client, "ws_edit_r")

    me_resp = ws_client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {token_r}"}
    )
    recvr_id = me_resp.json()["id"]

    with ws_client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "auth", "payload": {"token": token}}))
        _ = ws.receive_text()

        # Nachricht senden
        ws.send_text(
            json.dumps(
                {
                    "type": "message",
                    "payload": {
                        "recipient_id": recvr_id,
                        "ciphertext": "original",
                        "tmp_id": "tmp-2",
                    },
                }
            )
        )
        new_msg = json.loads(ws.receive_text())
        msg_id = new_msg["payload"]["id"]

        # Nachricht bearbeiten
        ws.send_text(
            json.dumps(
                {
                    "type": "message_edit",
                    "payload": {"message_id": msg_id, "ciphertext": "edited"},
                }
            )
        )
        edit_msg = json.loads(ws.receive_text())

    assert edit_msg["type"] == "message_edited"
    assert edit_msg["payload"]["ciphertext"] == "edited"


def test_ws_message_delete(ws_client):
    _sync_register(ws_client, "ws_del_u")
    _sync_register(ws_client, "ws_del_r")
    token = _sync_login(ws_client, "ws_del_u")
    token_r = _sync_login(ws_client, "ws_del_r")

    me_resp = ws_client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {token_r}"}
    )
    recvr_id = me_resp.json()["id"]

    with ws_client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "auth", "payload": {"token": token}}))
        _ = ws.receive_text()

        ws.send_text(
            json.dumps(
                {
                    "type": "message",
                    "payload": {
                        "recipient_id": recvr_id,
                        "ciphertext": "delete me",
                        "tmp_id": "tmp-3",
                    },
                }
            )
        )
        new_msg = json.loads(ws.receive_text())
        msg_id = new_msg["payload"]["id"]

        ws.send_text(
            json.dumps(
                {"type": "message_delete", "payload": {"message_id": msg_id}}
            )
        )
        del_msg = json.loads(ws.receive_text())

    assert del_msg["type"] == "message_deleted"
    assert del_msg["payload"]["message_id"] == msg_id


def test_ws_typing(ws_client):
    _sync_register(ws_client, "ws_typ_u")
    _sync_register(ws_client, "ws_typ_r")
    token = _sync_login(ws_client, "ws_typ_u")
    token_r = _sync_login(ws_client, "ws_typ_r")

    me_resp = ws_client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {token_r}"}
    )
    recvr_id = me_resp.json()["id"]

    with ws_client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "auth", "payload": {"token": token}}))
        _ = ws.receive_text()

        # Typing zu nicht-verbundenem User → kein Fehler
        ws.send_text(
            json.dumps(
                {"type": "typing", "payload": {"recipient_id": recvr_id}}
            )
        )
        # kein Response erwartet, nur kein Crash


def test_ws_read_receipt(ws_client):
    import time
    _sync_register(ws_client, "ws_read_s")
    _sync_register(ws_client, "ws_read_r")
    token_s = _sync_login(ws_client, "ws_read_s")
    token_r = _sync_login(ws_client, "ws_read_r")

    me_s = ws_client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {token_s}"}
    ).json()
    me_r = ws_client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {token_r}"}
    ).json()

    with ws_client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "auth", "payload": {"token": token_s}}))
        _ = ws.receive_text()

        # Nachricht senden
        ws.send_text(
            json.dumps(
                {
                    "type": "message",
                    "payload": {
                        "recipient_id": me_r["id"],
                        "ciphertext": "read?",
                        "tmp_id": "tmp-read",
                    },
                }
            )
        )
        new_msg = json.loads(ws.receive_text())
        msg_id = new_msg["payload"]["id"]

        # Read-Receipt (Empfänger ist nicht verbunden → kein send_to, aber kein Fehler)
        ws.send_text(
            json.dumps(
                {
                    "type": "read",
                    "payload": {
                        "message_id": msg_id,
                        "sender_id": me_s["id"],
                    },
                }
            )
        )
        # sender_id == eigene ID → send_to(user_id, ...) geht an sich selbst
        read_msg = json.loads(ws.receive_text())

    assert read_msg["type"] == "read"
    assert read_msg["payload"]["message_id"] == msg_id
