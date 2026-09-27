from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import get_db
from app.core.security import hash_password, verify_password, create_access_token
from app.core.deps import get_current_user
from app.models.models import User
from app.schemas.schemas import UserRegister, UserLogin, Token, UserOut
import random

router = APIRouter(prefix="/api/auth", tags=["auth"])

AVATAR_COLORS = [
    "#2563eb", "#7c3aed", "#db2777", "#ea580c",
    "#16a34a", "#0891b2", "#d97706", "#dc2626",
]


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(data: UserRegister, db: AsyncSession = Depends(get_db)):
    # Username prüfen
    existing = await db.execute(select(User).where(User.username == data.username))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Benutzername bereits vergeben")
    # Email prüfen
    existing_email = await db.execute(select(User).where(User.email == data.email))
    if existing_email.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="E-Mail bereits registriert")

    user = User(
        username=data.username,
        display_name=data.display_name,
        email=data.email,
        password_hash=hash_password(data.password),
        avatar_color=random.choice(AVATAR_COLORS),
        sc_key_a=data.sc_key_a,
        sc_key_b=data.sc_key_b,
        sc_key_p=data.sc_key_p,
        sc_key_n=data.sc_key_n or 8,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@router.post("/login", response_model=Token)
async def login(data: UserLogin, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.username == data.username))
    user = result.scalar_one_or_none()
    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Ungültige Anmeldedaten")
    token = create_access_token({"sub": str(user.id)})
    return {"access_token": token, "token_type": "bearer"}


@router.get("/me", response_model=UserOut)
async def me(current_user: User = Depends(get_current_user)):
    return current_user


@router.put("/me/key", response_model=UserOut)
async def update_key(
    key_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    for field in ("sc_key_a", "sc_key_b", "sc_key_p", "sc_key_n"):
        if field in key_data:
            setattr(current_user, field, key_data[field])
    await db.commit()
    await db.refresh(current_user)
    return current_user
