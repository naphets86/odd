from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.models import User, Contact
from app.schemas.schemas import ContactCreate, ContactOut
from typing import List
import uuid

router = APIRouter(prefix="/api/contacts", tags=["contacts"])


@router.get("", response_model=List[ContactOut])
async def list_contacts(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Contact).where(Contact.owner_id == current_user.id)
    )
    contacts = result.scalars().all()
    # Eager-load contact_user
    loaded = []
    for c in contacts:
        await db.refresh(c, ["contact_user"])
        loaded.append(c)
    return loaded


@router.post("", response_model=ContactOut, status_code=status.HTTP_201_CREATED)
async def add_contact(
    data: ContactCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Ziel-User per Username suchen
    result = await db.execute(select(User).where(User.username == data.username))
    target = result.scalar_one_or_none()
    if not target:
        raise HTTPException(status_code=404, detail="Benutzer nicht gefunden")
    if target.id == current_user.id:
        raise HTTPException(status_code=400, detail="Du kannst dich nicht selbst als Kontakt hinzufügen")

    # Duplikat prüfen
    existing = await db.execute(
        select(Contact).where(
            and_(Contact.owner_id == current_user.id, Contact.contact_id == target.id)
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Kontakt bereits vorhanden")

    contact = Contact(
        owner_id=current_user.id,
        contact_id=target.id,
        nickname=data.nickname,
        sc_key_a=data.sc_key_a,
        sc_key_b=data.sc_key_b,
        sc_key_p=data.sc_key_p,
        sc_key_n=data.sc_key_n or 8,
    )
    db.add(contact)
    await db.commit()
    await db.refresh(contact, ["contact_user"])
    return contact


@router.delete("/{contact_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_contact(
    contact_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Contact).where(
            and_(Contact.id == contact_id, Contact.owner_id == current_user.id)
        )
    )
    contact = result.scalar_one_or_none()
    if not contact:
        raise HTTPException(status_code=404, detail="Kontakt nicht gefunden")
    await db.delete(contact)
    await db.commit()


@router.get("/search/{username}", response_model=dict)
async def search_user(
    username: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="Benutzer nicht gefunden")
    return {
        "id": str(user.id),
        "username": user.username,
        "display_name": user.display_name,
        "avatar_color": user.avatar_color,
        "is_online": user.is_online,
    }
