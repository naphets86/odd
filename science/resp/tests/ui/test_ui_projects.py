"""Projects page – CRUD via UI."""
import httpx
from playwright.sync_api import Page, expect


def _goto_projects(page: Page, base_url: str) -> None:
    page.goto(base_url)
    page.click('nav a[data-page="projects"]')
    page.wait_for_load_state("networkidle")


def test_projects_empty_state(page: Page, base_url: str) -> None:
    _goto_projects(page, base_url)
    expect(page.locator("#projects-tbody")).to_contain_text("Noch keine Projekte erfasst")


def test_add_project_appears_in_table(page: Page, base_url: str) -> None:
    _goto_projects(page, base_url)

    page.click("button:has-text('Projekt erstellen')")
    expect(page.locator("#project-modal")).to_be_visible()

    page.fill("#pj-name", "Brückenprojekt Alpha")
    page.fill("#pj-desc", "Eine wichtige Brücke")
    page.select_option("#proj-status-select", "aktiv")
    page.click("#project-modal button:has-text('Speichern')")
    page.wait_for_load_state("networkidle")

    expect(page.locator("#projects-tbody")).to_contain_text("Brückenprojekt Alpha")


def test_add_project_shows_toast(page: Page, base_url: str) -> None:
    _goto_projects(page, base_url)

    page.click("button:has-text('Projekt erstellen')")
    page.fill("#pj-name", "Toast Projekt")
    page.select_option("#proj-status-select", "geplant")
    page.click("#project-modal button:has-text('Speichern')")

    expect(page.locator("#toast-container")).to_contain_text("Projekt erstellt")


def test_edit_project(page: Page, base_url: str, api: httpx.Client) -> None:
    api.post(
        "/api/projects/",
        json={"name": "Altes Projekt", "status": "geplant"},
    )

    _goto_projects(page, base_url)
    page.click("#projects-tbody button:first-of-type")  # edit button
    expect(page.locator("#project-modal")).to_be_visible()
    expect(page.locator("#pj-name")).to_have_value("Altes Projekt")

    page.fill("#pj-name", "Umbenanntes Projekt")
    page.click("#project-modal button:has-text('Speichern')")
    page.wait_for_load_state("networkidle")

    expect(page.locator("#projects-tbody")).to_contain_text("Umbenanntes Projekt")


def test_delete_project(page: Page, base_url: str, api: httpx.Client) -> None:
    api.post(
        "/api/projects/",
        json={"name": "Projekt zum Löschen", "status": "abgebrochen"},
    )

    _goto_projects(page, base_url)
    expect(page.locator("#projects-tbody")).to_contain_text("Projekt zum Löschen")

    page.on("dialog", lambda d: d.accept())
    page.click("#projects-tbody button.danger")
    page.wait_for_load_state("networkidle")

    expect(page.locator("#projects-tbody")).not_to_contain_text("Projekt zum Löschen")


def test_view_project_allocations_link(
    page: Page, base_url: str, api: httpx.Client
) -> None:
    r = api.post(
        "/api/projects/",
        json={"name": "Link Projekt", "status": "aktiv"},
    )
    project_id = r.json()["id"]

    _goto_projects(page, base_url)
    # Click the 🔗 allocation-link button (second icon button in the row)
    page.click("#projects-tbody tr:first-child button:nth-child(2)")
    page.wait_for_load_state("networkidle")

    expect(page.locator("#page-allocations")).to_be_visible()
    # Filter select should show the project id selected
    expect(page.locator("#alloc-project-filter")).to_have_value(str(project_id))
