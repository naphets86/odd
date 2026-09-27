import io
from pathlib import Path

import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.main import app
from app.database import Base, get_db

_REPORT_FILE = Path(__file__).parent / "report.txt"


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    cov_plugin = config.pluginmanager.get_plugin("_cov")
    if not cov_plugin:
        return
    try:
        cov = cov_plugin.cov_controller.cov
    except AttributeError:
        return
    buf = io.StringIO()
    cov.report(file=buf, show_missing=True)
    _REPORT_FILE.write_text(buf.getvalue())

# PostgreSQL test database configuration
TEST_DATABASE_URL = "postgresql+asyncpg://resp_user:resp_pass@localhost:5432/resp_test_db"

engine_test = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)
AsyncTestSession = async_sessionmaker(engine_test, expire_on_commit=False, class_=AsyncSession, autoflush=False)


async def override_get_db():
    async with AsyncTestSession() as session:
        yield session


app.dependency_overrides[get_db] = override_get_db


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    # Create all tables
    async with engine_test.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield
    
    # Drop all tables
    async with engine_test.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    # Dispose of the connection pool to avoid event loop issues
    await engine_test.dispose()


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
