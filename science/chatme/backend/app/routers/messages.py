from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, and_
from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.models import User, Message
from app.schemas.schemas import MessageSend, MessageOut, MessageEdit
from typing import List
import uuid
from datetime import datetime, timezone

router = APIRouter(prefix="/api/messages", tags=["messages"])


@router.get("/history/{other_user_id}", response_model=List[MessageOut])
async def get_history(
    other_user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Message).where(
            or_(
                and_(Message.sender_id == current_user.id, Message.recipient_id == other_user_id),
                and_(Message.sender_id == other_user_id,   Message.recipient_id == current_user.id),
            ),
            Message.is_deleted == False,
        ).order_by(Message.created_at)
    )
    return result.scalars().all()


@router.post("", response_model=MessageOut, status_code=status.HTTP_201_CREATED)
async def send_message(
    data: MessageSend,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    msg = Message(
        sender_id=current_user.id,
        recipient_id=data.recipient_id,
        ciphertext=data.ciphertext,
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return msg


@router.put("/{message_id}", response_model=MessageOut)
async def edit_message(
    message_id: uuid.UUID,
    data: MessageEdit,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Message).where(
            Message.id == message_id,
            Message.sender_id == current_user.id,
            Message.is_deleted == False,
        )
    )
    msg = result.scalar_one_or_none()
    if not msg:
        raise HTTPException(status_code=404, detail="Nachricht nicht gefunden")
    msg.ciphertext = data.ciphertext
    msg.edited_at  = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(msg)
    return msg


@router.delete("/{message_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_message(
    message_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Message).where(
            Message.id == message_id,
            Message.sender_id == current_user.id,
        )
    )
    msg = result.scalar_one_or_none()
    if not msg:
        raise HTTPException(status_code=404, detail="Nachricht nicht gefunden")
    msg.is_deleted = True
    await db.commit()


@router.post("/{message_id}/read", status_code=status.HTTP_204_NO_CONTENT)
async def mark_read(
    message_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Message).where(
            Message.id == message_id,
            Message.recipient_id == current_user.id,
        )
    )
    msg = result.scalar_one_or_none()
    if msg:
        msg.is_read = True
        await db.commit()
