"""
Gemeinsame Fixtures für alle Backend-Tests.

Design-Entscheidungen:
  - Jeder Test bekommt eine eigene in-memory SQLite-Engine → vollständige Isolation
  - WebSocket-Tests nutzen starlette.testclient.TestClient (sync), da ASGITransport
    keine WebSocket-Upgrades unterstützt
  - AsyncSessionLocal wird für WS-Tests via patch umgeleitet
"""

import asyncio
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool, NullPool
from starlette.testclient import TestClient

from app.core.database import Base, get_db
from app.main import app


# ── HTTP-Fixtures ──────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def db_session() -> AsyncSession:
    """Frische SQLite-In-Memory-DB pro Test."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest_asyncio.fixture
async def client(db_session: AsyncSession):
    """Async HTTP-Client mit überschriebenem DB-Dependency."""
    from httpx import AsyncClient, ASGITransport

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def registered_user(client):
    """Registriert einen Testnutzer und gibt seine Daten zurück."""
    resp = await client.post(
        "/api/auth/register",
        json={
            "username": "testuser",
            "display_name": "Test User",
            "email": "testuser@ws-test.example",
            "password": "password123",
        },
    )
    assert resp.status_code == 201
    return resp.json()


@pytest_asyncio.fixture
async def auth_token(client, registered_user):
    """Gibt ein gültiges JWT für den registrierten Testnutzer zurück."""
    resp = await client.post(
        "/api/auth/login",
        json={"username": "testuser", "password": "password123"},
    )
    assert resp.status_code == 200
    return resp.json()["access_token"]


@pytest_asyncio.fixture
async def auth_headers(auth_token):
    return {"Authorization": f"Bearer {auth_token}"}


# ── Redis-Mock ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_redis():
    """Mockt den Redis-Client (TURN-Credentials werden nicht wirklich benötigt)."""
    redis_mock = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.set = AsyncMock(return_value=True)
    redis_mock.delete = AsyncMock(return_value=True)
    with patch("app.core.redis_client.get_redis", return_value=redis_mock):
        yield redis_mock


# ── WebSocket-Fixtures (sync) ─────────────────────────────────────────────────


def _sync_register(tc: TestClient, username: str, password: str = "password123") -> dict:
    """Registriert einen Nutzer über den synchronen TestClient."""
    resp = tc.post(
        "/api/auth/register",
        json={
            "username": username,
            "display_name": username,
            "email": f"{username}@ws-test.example",
            "password": password,
        },
    )
    assert resp.status_code == 201, f"Register fehlgeschlagen: {resp.text}"
    return resp.json()


def _sync_login(tc: TestClient, username: str, password: str = "password123") -> str:
    """Meldet einen Nutzer an und gibt den Access-Token zurück."""
    resp = tc.post(
        "/api/auth/login",
        json={"username": username, "password": password},
    )
    assert resp.status_code == 200, f"Login fehlgeschlagen: {resp.text}"
    return resp.json()["access_token"]


@pytest.fixture
def ws_client(tmp_path, mock_redis):
    """
    Synchroner TestClient mit eigener SQLite-DB für WebSocket-Tests.
    AsyncSessionLocal wird gepatcht, damit der WS-Handler dieselbe DB nutzt.
    """
    db_file = str(tmp_path / "ws_test.db")
    ws_engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_file}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
    ws_session_factory = async_sessionmaker(ws_engine, expire_on_commit=False)

    # Tabellen synchron anlegen
    loop = asyncio.new_event_loop()

    async def _init():
        async with ws_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    loop.run_until_complete(_init())
    loop.close()

    async def override_get_db():
        async with ws_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    with patch("app.routers.ws.AsyncSessionLocal", ws_session_factory), TestClient(
        app, raise_server_exceptions=True
    ) as tc:
        yield tc

    app.dependency_overrides.clear()

    loop2 = asyncio.new_event_loop()

    async def _teardown():
        await ws_engine.dispose()

    loop2.run_until_complete(_teardown())
    loop2.close()
