"""
Frontend-Playwright-Tests – Fixtures und Hilfsfunktionen.

Da das Produktions-Backend (Docker) das Frontend-HTML nicht serviert,
starten wir für Tests einen einfachen HTTP-Server aus app/static/.
"""

import os
import threading
import http.server
import functools
import socket
import pytest
from playwright.sync_api import Page, Browser

# Pfad zum statischen Frontend
STATIC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "backend", "app", "static")
)


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def static_server():
    """Startet einen einfachen HTTP-Server für das Frontend-HTML."""
    port = _get_free_port()
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=STATIC_DIR,
    )
    server = http.server.HTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    yield base_url
    server.shutdown()


@pytest.fixture(scope="session")
def base_url(static_server):
    return static_server


@pytest.fixture
def page_loaded(page: Page, static_server: str):
    """Lädt die index.html-Seite in Playwright."""
    page.goto(static_server)
    page.wait_for_load_state("domcontentloaded")
    return page
