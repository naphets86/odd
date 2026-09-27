"""Playwright end-to-end test infrastructure.

Run UI tests in isolation (no coverage conflict with API tests):
    pytest tests/ui/ -v

Run with Python-backend coverage:
    pytest tests/ui/ --cov=app --cov-report=term-missing -v

JS and CSS coverage is collected automatically and written to tests/ui_report.txt.
"""
import asyncio
import concurrent.futures
import json
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi.staticfiles import StaticFiles
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.main import app
from app.database import Base, get_db

FRONTEND_DIR = Path(__file__).parents[2] / "frontend"
TEST_DATABASE_URL = (
    "postgresql+asyncpg://resp_user:resp_pass@localhost:5432/resp_test_db"
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _run_async(coro):
    """Execute a coroutine in a dedicated thread.

    pytest-asyncio runs its own event loop on the main thread; calling
    asyncio.run() directly from a sync fixture would raise
    'Cannot run the event loop while another loop is running'.
    Running in a fresh thread sidesteps this limitation.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class _UvicornServer(uvicorn.Server):
    """Uvicorn server that does not install signal handlers (thread-safe)."""

    def install_signal_handlers(self) -> None:  # type: ignore[override]
        pass


# ── session-scoped live server ────────────────────────────────────────────────


@pytest.fixture(scope="session")
def live_server_base_url():
    """Start a single uvicorn process for the whole test session.

    * Uses NullPool so the async engine works across multiple asyncio event
      loops (the setup loop, uvicorn's loop, and the teardown loop are all
      independent).
    * Mounts the static frontend so the browser can access index.html via
      the same origin as the API → the JS `const API = '/api'` branch is used.
    """
    # --- DB engine (NullPool avoids cross-event-loop pool reuse) --------------
    engine = create_async_engine(TEST_DATABASE_URL, echo=False, poolclass=NullPool)
    Session = async_sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession, autoflush=False
    )

    async def _override_get_db():
        async with Session() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db

    # --- Mount static frontend (once, before server starts) -------------------
    # API routes registered in app/main.py take priority; the "/" mount only
    # serves paths that don't match any API route.
    app.mount(
        "/",
        StaticFiles(directory=str(FRONTEND_DIR), html=True),
        name="ui_static",
    )

    # --- Create DB tables -----------------------------------------------------
    async def _setup() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    _run_async(_setup())

    # --- Start uvicorn in a daemon thread -------------------------------------
    port = _free_port()
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning"
    )
    server = _UvicornServer(config=config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 15
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("Test server did not start within 15 s")
        time.sleep(0.05)

    base = f"http://127.0.0.1:{port}"
    yield base

    # --- Teardown -------------------------------------------------------------
    server.should_exit = True
    thread.join(timeout=5)

    async def _teardown() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()

    _run_async(_teardown())


# ── per-test fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def base_url(live_server_base_url: str) -> str:
    """Override pytest-playwright's base_url with the live server URL."""
    return live_server_base_url


def _wipe(base: str) -> None:
    """Delete every record from every resource table via the API."""
    with httpx.Client(base_url=base, timeout=15) as c:
        for path in [
            "/api/allocations/",
            "/api/persons/",
            "/api/materials/",
            "/api/time-resources/",
            "/api/projects/",
        ]:
            items = c.get(path, params={"limit": 1000}).json()
            for item in items:
                c.delete(f"{path}{item['id']}")


@pytest.fixture(autouse=True)
def setup_db():  # noqa: PT004
    """Override the parent autouse setup_db fixture.

    The parent fixture (tests/conftest.py) drops and recreates DB tables for
    every test, which would destroy the persistent state the live server needs.
    Here we replace it with a no-op so the UI session fixture controls the DB.
    """
    yield


@pytest.fixture(autouse=True)
def clean_db(live_server_base_url: str) -> None:
    """Wipe all data before every UI test for full isolation."""
    _wipe(live_server_base_url)
    yield


@pytest.fixture()
def api(live_server_base_url: str):
    """Synchronous httpx client for programmatic data setup in tests."""
    with httpx.Client(base_url=live_server_base_url, timeout=15) as client:
        yield client


# ── JS / CSS Coverage ────────────────────────────────────────────────────────

_UI_REPORT_FILE = Path(__file__).parents[2] / "tests" / "ui_report.txt"

# Module-level stores: url -> {"ranges": list[dict]}
_js_cov_store: dict[str, dict] = {}
_css_cov_store: dict[str, dict] = {}
_ui_report_str: str = ""


def _merge_ranges(ranges: list[dict]) -> list[dict]:
    """Union-merge overlapping byte ranges."""
    if not ranges:
        return []
    sorted_r = sorted(ranges, key=lambda r: r["start"])
    merged = [dict(sorted_r[0])]
    for r in sorted_r[1:]:
        if r["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], r["end"])
        else:
            merged.append(dict(r))
    return merged


def _line_coverage(text: str, ranges: list[dict]) -> tuple[int, int, list[int]]:
    """Return (covered_lines, total_non_empty_lines, uncovered_line_numbers)."""
    merged = _merge_ranges(ranges)
    lines = text.split("\n")
    uncovered: list[int] = []
    offset = 0
    for lineno, line in enumerate(lines, start=1):
        line_start = offset
        line_end = offset + len(line)
        if line.strip():  # skip blank lines
            covered = any(
                r["start"] < line_end and r["end"] > line_start for r in merged
            )
            if not covered:
                uncovered.append(lineno)
        offset = line_end + 1  # +1 for \n
    non_empty = sum(1 for ln in lines if ln.strip())
    return non_empty - len(uncovered), non_empty, uncovered


@pytest.fixture(autouse=True)
def _js_coverage(page):
    """Collect Chromium JS and CSS coverage per test via CDP.

    playwright-python 1.x does not expose page.coverage in the sync API.
    We use a CDP session with the V8 Profiler for JS and CSS.startRuleUsageTracking
    for CSS.  Ranges are normalised to {start, end} so the existing
    _merge_ranges / _line_coverage helpers can process them.
    """
    cdp = page.context.new_cdp_session(page)

    # JS
    cdp.send("Profiler.enable")
    cdp.send("Profiler.startPreciseCoverage", {"callCount": False, "detailed": True})

    # CSS – record styleSheetId -> sourceURL before enabling tracking
    css_sheet_urls: dict[str, str] = {}

    def _on_stylesheet_added(params: dict) -> None:
        header = params.get("header", {})
        sheet_id = header.get("styleSheetId")
        source_url = header.get("sourceURL", "")
        if sheet_id and source_url:
            css_sheet_urls[sheet_id] = source_url

    cdp.on("CSS.styleSheetAdded", _on_stylesheet_added)
    cdp.send("DOM.enable")
    cdp.send("CSS.enable")  # fires styleSheetAdded for already-loaded sheets
    cdp.send("CSS.startRuleUsageTracking")

    yield

    # --- collect JS ---
    js_result = cdp.send("Profiler.takePreciseCoverage")
    cdp.send("Profiler.stopPreciseCoverage")
    cdp.send("Profiler.disable")

    # --- collect CSS ---
    css_result = cdp.send("CSS.takeCoverageDelta")
    cdp.send("CSS.stopRuleUsageTracking")
    cdp.send("CSS.disable")

    cdp.detach()

    # Process JS
    for script in js_result.get("result", []):
        url: str = script.get("url", "")
        if "127.0.0.1" not in url:
            continue
        # Convert Profiler range format {startOffset, endOffset} -> {start, end}
        ranges: list[dict] = [
            {"start": rng["startOffset"], "end": rng["endOffset"]}
            for func in script.get("functions", [])
            for rng in func.get("ranges", [])
            if rng.get("count", 0) > 0
        ]
        if url not in _js_cov_store:
            _js_cov_store[url] = {"ranges": list(ranges)}
        else:
            _js_cov_store[url]["ranges"].extend(ranges)

    # Process CSS (only used rules)
    for rule in css_result.get("coverage", []):
        if not rule.get("used", False):
            continue
        sheet_id = rule.get("styleSheetId", "")
        url = css_sheet_urls.get(sheet_id, "")
        if not url or "127.0.0.1" not in url:
            continue
        rng = {"start": rule["startOffset"], "end": rule["endOffset"]}
        if url not in _css_cov_store:
            _css_cov_store[url] = {"ranges": [rng]}
        else:
            _css_cov_store[url]["ranges"].append(rng)


def _build_section(
    title: str,
    store: dict[str, dict],
    base: str,
    client: httpx.Client,
) -> tuple[list[str], int, int]:
    """Build report rows for one coverage section; return (rows, total_stmts, total_covered)."""
    col_file = 40
    header = f"{'File':<{col_file}} {'Lines':>6} {'Cover':>6}  Missing Lines"
    sep = "-" * 80
    rows: list[str] = [title, sep, header, sep]
    total_stmts = total_covered = 0
    for url in sorted(store):
        data = store[url]
        fname = url.replace(base, "") or "index.html"
        try:
            path = "/" + fname.lstrip("/")
            text = client.get(path).text
        except Exception:
            continue
        covered, stmts, uncovered = _line_coverage(text, data["ranges"])
        pct = int(covered / stmts * 100) if stmts else 100
        missing_str = ", ".join(str(n) for n in uncovered[:15])
        if len(uncovered) > 15:
            missing_str += f" ... (+{len(uncovered) - 15} more)"
        rows.append(f"{fname:<{col_file}} {stmts:>6} {pct:>5}%  {missing_str}")
        total_stmts += stmts
        total_covered += covered
    rows.append(sep)
    total_pct = int(total_covered / total_stmts * 100) if total_stmts else 100
    rows.append(f"{'TOTAL':<{col_file}} {total_stmts:>6} {total_pct:>5}%")
    return rows, total_stmts, total_covered


@pytest.fixture(scope="session", autouse=True)
def _ui_coverage_report(live_server_base_url):
    """After all UI tests: fetch JS/CSS source, merge coverage, write report."""
    yield  # tests run here

    if not _js_cov_store and not _css_cov_store:
        return

    base = live_server_base_url.rstrip("/") + "/"
    rows: list[str] = ["", "UI Coverage Report", "=" * 80]

    with httpx.Client(base_url=live_server_base_url, timeout=10) as client:
        if _js_cov_store:
            js_rows, _, _ = _build_section("JS Coverage", _js_cov_store, base, client)
            rows.extend(js_rows)
        if _css_cov_store:
            if _js_cov_store:
                rows.append("")
            css_rows, _, _ = _build_section("CSS Coverage", _css_cov_store, base, client)
            rows.extend(css_rows)

    rows.append("")

    global _ui_report_str
    _ui_report_str = "\n".join(rows)
    _UI_REPORT_FILE.write_text(_ui_report_str)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if _ui_report_str:
        terminalreporter.write_sep("-", "UI coverage")
        for line in _ui_report_str.splitlines():
            terminalreporter.write_line(line)
