"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-03-15 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── users ────────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id",            postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("username",      sa.String(40),  nullable=False),
        sa.Column("display_name",  sa.String(80),  nullable=False),
        sa.Column("email",         sa.String(255), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("avatar_color",  sa.String(20),  nullable=True,  server_default="#2563eb"),
        sa.Column("is_online",     sa.Boolean(),   nullable=False, server_default=sa.false()),
        sa.Column("created_at",    sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_seen",     sa.DateTime(timezone=True), nullable=True),
        # Signatur-Chiffre keys
        sa.Column("sc_key_a", sa.Text(),    nullable=True),
        sa.Column("sc_key_b", sa.Text(),    nullable=True),
        sa.Column("sc_key_p", sa.Text(),    nullable=True),
        sa.Column("sc_key_n", sa.Integer(), nullable=True, server_default="8"),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)
    op.create_index("ix_users_email",    "users", ["email"],    unique=True)

    # ── contacts ─────────────────────────────────────────────────────────────
    op.create_table(
        "contacts",
        sa.Column("id",         postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("owner_id",   postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("contact_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("nickname",   sa.String(80), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        # Shared Signatur-Chiffre key for this contact channel
        sa.Column("sc_key_a", sa.Text(),    nullable=True),
        sa.Column("sc_key_b", sa.Text(),    nullable=True),
        sa.Column("sc_key_p", sa.Text(),    nullable=True),
        sa.Column("sc_key_n", sa.Integer(), nullable=True, server_default="8"),
    )

    # ── messages ─────────────────────────────────────────────────────────────
    op.create_table(
        "messages",
        sa.Column("id",           postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("sender_id",    postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("recipient_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ciphertext",   sa.Text(),    nullable=False),
        sa.Column("created_at",   sa.DateTime(timezone=True), nullable=True),
        sa.Column("edited_at",    sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_deleted",   sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_read",      sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_index("ix_messages_sender",    "messages", ["sender_id"])
    op.create_index("ix_messages_recipient", "messages", ["recipient_id"])
    op.create_index("ix_messages_created",   "messages", ["created_at"])

    # ── calls ────────────────────────────────────────────────────────────────
    call_status = postgresql.ENUM(
        "initiated", "ringing", "active", "ended", "missed", "rejected",
        name="callstatus",
    )
    call_status.create(op.get_bind())

    op.create_table(
        "calls",
        sa.Column("id",         postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("caller_id",  postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("callee_id",  postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status",     sa.Enum(
                      "initiated", "ringing", "active", "ended", "missed", "rejected",
                      name="callstatus", create_type=False,
                  ), nullable=True, server_default="initiated"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at",   sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_s", sa.Integer(), nullable=True, server_default="0"),
    )
    op.create_index("ix_calls_caller", "calls", ["caller_id"])
    op.create_index("ix_calls_callee", "calls", ["callee_id"])
    op.create_index("ix_calls_started", "calls", ["started_at"])


def downgrade() -> None:
    op.drop_table("calls")
    op.execute("DROP TYPE IF EXISTS callstatus")
    op.drop_index("ix_messages_created",   table_name="messages")
    op.drop_index("ix_messages_recipient", table_name="messages")
    op.drop_index("ix_messages_sender",    table_name="messages")
    op.drop_table("messages")
    op.drop_table("contacts")
    op.drop_index("ix_users_email",    table_name="users")
    op.drop_index("ix_users_username", table_name="users")
    op.drop_table("users")
