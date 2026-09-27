from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime
from uuid import UUID


# ── AUTH ───────────────────────────────────────────────────────

class UserRegister(BaseModel):
    username:     str = Field(..., min_length=3, max_length=40)
    display_name: str = Field(..., min_length=1, max_length=80)
    email:        EmailStr
    password:     str = Field(..., min_length=6)
    sc_key_a:     Optional[str] = None
    sc_key_b:     Optional[str] = None
    sc_key_p:     Optional[str] = None
    sc_key_n:     Optional[int] = 8


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type:   str = "bearer"


class UserOut(BaseModel):
    id:           UUID
    username:     str
    display_name: str
    email:        str
    avatar_color: str
    is_online:    bool
    last_seen:    datetime
    sc_key_a:     Optional[str] = None
    sc_key_b:     Optional[str] = None
    sc_key_p:     Optional[str] = None
    sc_key_n:     Optional[int] = 8

    model_config = {"from_attributes": True}


class UserPublic(BaseModel):
    id:           UUID
    username:     str
    display_name: str
    avatar_color: str
    is_online:    bool
    last_seen:    datetime

    model_config = {"from_attributes": True}


# ── CONTACTS ───────────────────────────────────────────────────

class ContactCreate(BaseModel):
    username:  str               # Kontakt per Username suchen
    nickname:  Optional[str] = None
    sc_key_a:  Optional[str] = None
    sc_key_b:  Optional[str] = None
    sc_key_p:  Optional[str] = None
    sc_key_n:  Optional[int] = 8


class ContactOut(BaseModel):
    id:           UUID
    contact_user: UserPublic
    nickname:     Optional[str]
    sc_key_a:     Optional[str]
    sc_key_b:     Optional[str]
    sc_key_p:     Optional[str]
    sc_key_n:     Optional[int]
    created_at:   datetime

    model_config = {"from_attributes": True}


# ── MESSAGES ───────────────────────────────────────────────────

class MessageSend(BaseModel):
    recipient_id: UUID
    ciphertext:   str            # JSON-String aus SC.encryptMessage


class MessageOut(BaseModel):
    id:           UUID
    sender_id:    UUID
    recipient_id: UUID
    ciphertext:   str
    created_at:   datetime
    edited_at:    Optional[datetime]
    is_read:      bool
    is_deleted:   bool

    model_config = {"from_attributes": True}


class MessageEdit(BaseModel):
    ciphertext: str


# ── CALLS ──────────────────────────────────────────────────────

class CallOut(BaseModel):
    id:         UUID
    caller_id:  UUID
    callee_id:  UUID
    status:     str
    started_at: datetime
    ended_at:   Optional[datetime]
    duration_s: int

    model_config = {"from_attributes": True}


# ── WEBSOCKET EVENTS ───────────────────────────────────────────

class WSEvent(BaseModel):
    type:    str
    payload: dict
