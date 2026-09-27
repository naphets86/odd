"""
WebSocket Hub – Chat-Nachrichten + WebRTC-Signaling (Zweier-Gespräche)

Event-Typen (Client → Server):
  auth          { token }
  message       { recipient_id, ciphertext, tmp_id }
  message_edit  { message_id, ciphertext }
  message_delete{ message_id }
  typing        { recipient_id }
  read          { message_id, sender_id }

  # WebRTC Signaling (nur 1:1)
  call_offer    { callee_id, sdp }
  call_answer   { caller_id, call_id, sdp }
  call_ice      { peer_id, call_id, candidate }
  call_reject   { caller_id, call_id }
  call_hangup   { peer_id, call_id }

Event-Typen (Server → Client):
  authenticated { user_id }
  error         { message }
  message_new   { ... MessageOut ... }
  message_edited{ message_id, ciphertext, edited_at }
  message_deleted{ message_id }
  typing        { sender_id }
  read          { message_id }
  user_status   { user_id, is_online }

  call_incoming { caller_id, caller_name, call_id, sdp }
  call_answer   { call_id, sdp }
  call_ice      { call_id, candidate }
  call_rejected { call_id }
  call_hangup   { call_id }
"""

import json
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from app.core.database import AsyncSessionLocal
from app.core.security import decode_token
from app.core.redis_client import get_redis
from app.models.models import User, Message, Call, CallStatus

router = APIRouter()

# In-Memory Map: user_id → WebSocket  (für single-instance; Redis Pub/Sub für multi-instance)
connections: dict[str, WebSocket] = {}


async def send_to(user_id: str, event: dict):
    ws = connections.get(user_id)
    if ws:
        try:
            await ws.send_text(json.dumps(event))
        except Exception:
            connections.pop(user_id, None)


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    user_id: str | None = None

    try:
        async with AsyncSessionLocal() as db:
            while True:
                raw = await ws.receive_text()
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send_text(json.dumps({"type": "error", "payload": {"message": "Ungültiges JSON"}}))
                    continue

                etype   = event.get("type")
                payload = event.get("payload", {})

                # ── AUTH ──────────────────────────────────────────────
                if etype == "auth":
                    token = payload.get("token", "")
                    data  = decode_token(token)
                    if not data:
                        await ws.send_text(json.dumps({"type": "error", "payload": {"message": "Ungültiger Token"}}))
                        await ws.close()
                        return
                    user_id = data.get("sub")
                    connections[user_id] = ws

                    # Online-Status setzen
                    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
                    user   = result.scalar_one_or_none()
                    if user:
                        user.is_online = True
                        await db.commit()

                    await ws.send_text(json.dumps({"type": "authenticated", "payload": {"user_id": user_id}}))
                    # Alle Kontakte über Online-Status informieren
                    await _broadcast_status(user_id, True)
                    continue

                if not user_id:
                    await ws.send_text(json.dumps({"type": "error", "payload": {"message": "Nicht authentifiziert"}}))
                    continue

                # ── CHAT MESSAGE ──────────────────────────────────────
                if etype == "message":
                    recipient_id = payload.get("recipient_id")
                    ciphertext   = payload.get("ciphertext")
                    tmp_id       = payload.get("tmp_id")
                    if not recipient_id or not ciphertext:
                        continue

                    msg = Message(
                        sender_id=uuid.UUID(user_id),
                        recipient_id=uuid.UUID(recipient_id),
                        ciphertext=ciphertext,
                    )
                    db.add(msg)
                    await db.commit()
                    await db.refresh(msg)

                    out = {
                        "type": "message_new",
                        "payload": {
                            "id":           str(msg.id),
                            "tmp_id":       tmp_id,
                            "sender_id":    str(msg.sender_id),
                            "recipient_id": str(msg.recipient_id),
                            "ciphertext":   msg.ciphertext,
                            "created_at":   msg.created_at.isoformat(),
                            "edited_at":    None,
                            "is_read":      False,
                        },
                    }
                    await send_to(user_id,      out)
                    await send_to(recipient_id, out)

                elif etype == "message_edit":
                    message_id = payload.get("message_id")
                    ciphertext = payload.get("ciphertext")
                    result = await db.execute(
                        select(Message).where(
                            Message.id == uuid.UUID(message_id),
                            Message.sender_id == uuid.UUID(user_id),
                        )
                    )
                    msg = result.scalar_one_or_none()
                    if msg:
                        msg.ciphertext = ciphertext
                        msg.edited_at  = datetime.now(timezone.utc)
                        await db.commit()
                        out = {
                            "type": "message_edited",
                            "payload": {
                                "message_id": message_id,
                                "ciphertext": ciphertext,
                                "edited_at":  msg.edited_at.isoformat(),
                            },
                        }
                        await send_to(user_id,                   out)
                        await send_to(str(msg.recipient_id), out)

                elif etype == "message_delete":
                    message_id = payload.get("message_id")
                    result = await db.execute(
                        select(Message).where(
                            Message.id == uuid.UUID(message_id),
                            Message.sender_id == uuid.UUID(user_id),
                        )
                    )
                    msg = result.scalar_one_or_none()
                    if msg:
                        recipient_id = str(msg.recipient_id)
                        msg.is_deleted = True
                        await db.commit()
                        out = {"type": "message_deleted", "payload": {"message_id": message_id}}
                        await send_to(user_id,      out)
                        await send_to(recipient_id, out)

                elif etype == "typing":
                    recipient_id = payload.get("recipient_id")
                    await send_to(recipient_id, {
                        "type": "typing",
                        "payload": {"sender_id": user_id},
                    })

                elif etype == "read":
                    message_id = payload.get("message_id")
                    sender_id  = payload.get("sender_id")
                    result = await db.execute(
                        select(Message).where(Message.id == uuid.UUID(message_id))
                    )
                    msg = result.scalar_one_or_none()
                    if msg:
                        msg.is_read = True
                        await db.commit()
                    await send_to(sender_id, {"type": "read", "payload": {"message_id": message_id}})

                # ── WEBRTC SIGNALING ──────────────────────────────────
                elif etype == "call_offer":
                    callee_id = payload.get("callee_id")
                    sdp       = payload.get("sdp")

                    # Neuen Call-Datensatz anlegen
                    call = Call(
                        caller_id=uuid.UUID(user_id),
                        callee_id=uuid.UUID(callee_id),
                        status=CallStatus.ringing,
                    )
                    db.add(call)
                    await db.commit()
                    await db.refresh(call)

                    # Caller-Info holen
                    r = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
                    caller = r.scalar_one_or_none()

                    await send_to(callee_id, {
                        "type": "call_incoming",
                        "payload": {
                            "call_id":     str(call.id),
                            "caller_id":   user_id,
                            "caller_name": caller.display_name if caller else "Unbekannt",
                            "sdp":         sdp,
                        },
                    })

                elif etype == "call_answer":
                    caller_id = payload.get("caller_id")
                    call_id   = payload.get("call_id")
                    sdp       = payload.get("sdp")

                    # Call auf aktiv setzen
                    r = await db.execute(select(Call).where(Call.id == uuid.UUID(call_id)))
                    call = r.scalar_one_or_none()
                    if call:
                        call.status = CallStatus.active
                        await db.commit()

                    await send_to(caller_id, {
                        "type": "call_answer",
                        "payload": {"call_id": call_id, "sdp": sdp},
                    })

                elif etype == "call_ice":
                    peer_id   = payload.get("peer_id")
                    call_id   = payload.get("call_id")
                    candidate = payload.get("candidate")
                    await send_to(peer_id, {
                        "type": "call_ice",
                        "payload": {"call_id": call_id, "candidate": candidate},
                    })

                elif etype == "call_reject":
                    caller_id = payload.get("caller_id")
                    call_id   = payload.get("call_id")

                    r = await db.execute(select(Call).where(Call.id == uuid.UUID(call_id)))
                    call = r.scalar_one_or_none()
                    if call:
                        call.status = CallStatus.rejected
                        call.ended_at = datetime.now(timezone.utc)
                        await db.commit()

                    await send_to(caller_id, {
                        "type": "call_rejected",
                        "payload": {"call_id": call_id},
                    })

                elif etype == "call_hangup":
                    peer_id = payload.get("peer_id")
                    call_id = payload.get("call_id")

                    r = await db.execute(select(Call).where(Call.id == uuid.UUID(call_id)))
                    call = r.scalar_one_or_none()
                    if call:
                        call.status   = CallStatus.ended
                        call.ended_at = datetime.now(timezone.utc)
                        if call.started_at:
                            delta = call.ended_at - call.started_at
                            call.duration_s = int(delta.total_seconds())
                        await db.commit()

                    await send_to(peer_id, {
                        "type": "call_hangup",
                        "payload": {"call_id": call_id},
                    })

    except WebSocketDisconnect:
        pass
    finally:
        if user_id:
            connections.pop(user_id, None)
            async with AsyncSessionLocal() as db:
                result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
                user = result.scalar_one_or_none()
                if user:
                    user.is_online = False
                    user.last_seen = datetime.now(timezone.utc)
                    await db.commit()
            await _broadcast_status(user_id, False)


async def _broadcast_status(user_id: str, is_online: bool):
    """Informiert alle verbundenen Nutzer über einen Status-Wechsel."""
    event = {
        "type": "user_status",
        "payload": {"user_id": user_id, "is_online": is_online},
    }
    for uid, ws in list(connections.items()):
        if uid != user_id:
            try:
                await ws.send_text(json.dumps(event))
            except Exception:
                connections.pop(uid, None)
