"""Dashboard page tests."""
import httpx
from playwright.sync_api import Page, expect


def test_dashboard_stat_cards_visible(page: Page, base_url: str) -> None:
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    for stat_id in ["stat-persons", "stat-materials", "stat-time",
                    "stat-projects", "stat-allocations"]:
        expect(page.locator(f"#{stat_id}")).to_be_visible()


def test_dashboard_stats_zero_with_empty_db(page: Page, base_url: str) -> None:
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    for stat_id in ["stat-persons", "stat-materials", "stat-time",
                    "stat-projects", "stat-allocations"]:
        expect(page.locator(f"#{stat_id}")).to_have_text("0")


def test_dashboard_empty_projects_table(page: Page, base_url: str) -> None:
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    expect(page.locator("#dash-projects")).to_contain_text("Keine Projekte")


def test_dashboard_stats_update_with_data(
    page: Page, base_url: str, api: httpx.Client
) -> None:
    api.post("/api/persons/", json={"name": "Anna Test", "person_type": "Arbeitnehmer"})
    api.post("/api/projects/", json={"name": "Proj Alpha", "status": "geplant"})

    page.goto(base_url)
    page.wait_for_load_state("networkidle")

    expect(page.locator("#stat-persons")).to_have_text("1")
    expect(page.locator("#stat-projects")).to_have_text("1")


def test_dashboard_shows_recent_projects(
    page: Page, base_url: str, api: httpx.Client
) -> None:
    api.post("/api/projects/", json={"name": "Sichtbares Projekt", "status": "aktiv"})

    page.goto(base_url)
    page.wait_for_load_state("networkidle")

    expect(page.locator("#dash-projects")).to_contain_text("Sichtbares Projekt")


def test_dashboard_all_projects_link(page: Page, base_url: str) -> None:
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.click("button:has-text('Alle anzeigen')")
    page.wait_for_load_state("networkidle")
    expect(page.locator("#page-projects")).to_be_visible()
