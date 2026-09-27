import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Boolean, DateTime, ForeignKey,
    Text, Integer, Enum as SAEnum
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum


def utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username    = Column(String(40), unique=True, nullable=False, index=True)
    display_name= Column(String(80), nullable=False)
    email       = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    avatar_color= Column(String(20), default="#2563eb")
    is_online   = Column(Boolean, default=False)
    created_at  = Column(DateTime(timezone=True), default=utcnow)
    last_seen   = Column(DateTime(timezone=True), default=utcnow)

    # Signatur-Chiffre Schlüssel (pro User, geteilt mit Kontakten)
    sc_key_a    = Column(Text, nullable=True)   # BigInt als String
    sc_key_b    = Column(Text, nullable=True)
    sc_key_p    = Column(Text, nullable=True)
    sc_key_n    = Column(Integer, default=8)

    contacts_as_owner  = relationship("Contact", foreign_keys="Contact.owner_id",  back_populates="owner",   cascade="all, delete-orphan")
    contacts_as_target = relationship("Contact", foreign_keys="Contact.contact_id", back_populates="contact_user")
    sent_messages      = relationship("Message", foreign_keys="Message.sender_id",   back_populates="sender")
    calls_initiated    = relationship("Call",    foreign_keys="Call.caller_id",      back_populates="caller")
    calls_received     = relationship("Call",    foreign_keys="Call.callee_id",      back_populates="callee")


class Contact(Base):
    __tablename__ = "contacts"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id    = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    contact_id  = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    nickname    = Column(String(80), nullable=True)   # optionaler Anzeigename
    created_at  = Column(DateTime(timezone=True), default=utcnow)

    owner        = relationship("User", foreign_keys=[owner_id],   back_populates="contacts_as_owner")
    contact_user = relationship("User", foreign_keys=[contact_id], back_populates="contacts_as_target")

    # Gemeinsamer Signatur-Chiffre-Schlüssel für diesen Kontakt-Kanal
    sc_key_a = Column(Text, nullable=True)
    sc_key_b = Column(Text, nullable=True)
    sc_key_p = Column(Text, nullable=True)
    sc_key_n = Column(Integer, default=8)


class Message(Base):
    __tablename__ = "messages"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sender_id   = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    recipient_id= Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    # Chiffretext (JSON-String mit 'len' + 'blocks' aus SC.encryptMessage)
    ciphertext  = Column(Text, nullable=False)
    created_at  = Column(DateTime(timezone=True), default=utcnow)
    edited_at   = Column(DateTime(timezone=True), nullable=True)
    is_deleted  = Column(Boolean, default=False)
    is_read     = Column(Boolean, default=False)

    sender    = relationship("User", foreign_keys=[sender_id],    back_populates="sent_messages")


class CallStatus(str, enum.Enum):
    initiated = "initiated"
    ringing   = "ringing"
    active    = "active"
    ended     = "ended"
    missed    = "missed"
    rejected  = "rejected"


class Call(Base):
    __tablename__ = "calls"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    caller_id   = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    callee_id   = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    status      = Column(SAEnum(CallStatus), default=CallStatus.initiated)
    started_at  = Column(DateTime(timezone=True), default=utcnow)
    ended_at    = Column(DateTime(timezone=True), nullable=True)
    duration_s  = Column(Integer, default=0)

    caller = relationship("User", foreign_keys=[caller_id], back_populates="calls_initiated")
    callee = relationship("User", foreign_keys=[callee_id], back_populates="calls_received")
