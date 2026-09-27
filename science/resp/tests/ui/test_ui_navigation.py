"""Navigation and page-load tests."""
import re

import pytest
from playwright.sync_api import Page, expect


def test_page_title(page: Page, base_url: str) -> None:
    page.goto(base_url)
    expect(page).to_have_title("RESP – Ressourcenverwaltung")


def test_initial_page_is_dashboard(page: Page, base_url: str) -> None:
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    expect(page.locator("#page-dashboard")).to_have_class(re.compile(r"\bactive\b"))
    expect(page.locator("#page-dashboard")).to_be_visible()


def test_header_visible(page: Page, base_url: str) -> None:
    page.goto(base_url)
    expect(page.locator("header h1")).to_have_text("RESP")


@pytest.mark.parametrize(
    "nav_key, page_id",
    [
        ("dashboard",    "page-dashboard"),
        ("persons",      "page-persons"),
        ("materials",    "page-materials"),
        ("time",         "page-time"),
        ("projects",     "page-projects"),
        ("allocations",  "page-allocations"),
        ("graph",        "page-graph"),
    ],
)
def test_nav_shows_correct_page(
    page: Page, base_url: str, nav_key: str, page_id: str
) -> None:
    page.goto(base_url)
    page.click(f'nav a[data-page="{nav_key}"]')
    page.wait_for_load_state("networkidle")
    expect(page.locator(f"#{page_id}")).to_be_visible()


def test_active_nav_link(page: Page, base_url: str) -> None:
    page.goto(base_url)
    page.click('nav a[data-page="persons"]')
    page.wait_for_load_state("networkidle")
    active_link = page.locator('nav a[data-page="persons"]')
    expect(active_link).to_have_class(re.compile(r"\bactive\b"))


def test_sidebar_nav_groups_visible(page: Page, base_url: str) -> None:
    page.goto(base_url)
    expect(page.locator(".nav-group").first).to_be_visible()
