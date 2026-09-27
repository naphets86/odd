"""Allocations page – CRUD via UI."""
import httpx
from playwright.sync_api import Page, expect


def _goto_allocations(page: Page, base_url: str) -> None:
    page.goto(base_url)
    page.click('nav a[data-page="allocations"]')
    page.wait_for_load_state("networkidle")


def _setup_project_and_person(api: httpx.Client) -> tuple[int, int]:
    project_id = api.post(
        "/api/projects/", json={"name": "Alloc Projekt", "status": "aktiv"}
    ).json()["id"]
    person_id = api.post(
        "/api/persons/", json={"name": "Alloc Person", "person_type": "Arbeitnehmer"}
    ).json()["id"]
    return project_id, person_id


def test_allocations_empty_state(page: Page, base_url: str) -> None:
    _goto_allocations(page, base_url)
    expect(page.locator("#alloc-tbody")).to_contain_text("Keine Zuteilungen gefunden")


def test_add_allocation_to_person(
    page: Page, base_url: str, api: httpx.Client
) -> None:
    project_id, _ = _setup_project_and_person(api)

    _goto_allocations(page, base_url)
    page.click("button:has-text('Zuteilen')")
    expect(page.locator("#alloc-modal")).to_be_visible()

    # Select project and person
    page.select_option("#alloc-proj-select", str(project_id))
    page.locator("#alloc-person-select").select_option(index=1)  # first real person
    page.fill("#al-qty", "2")
    page.select_option("#alloc-status-select", "aktiv")
    page.click("#alloc-modal button:has-text('Speichern')")
    page.wait_for_load_state("networkidle")

    expect(page.locator("#alloc-tbody")).to_contain_text("Alloc Projekt")
    expect(page.locator("#alloc-tbody")).to_contain_text("Alloc Person")


def test_add_allocation_shows_toast(
    page: Page, base_url: str, api: httpx.Client
) -> None:
    project_id, _ = _setup_project_and_person(api)

    _goto_allocations(page, base_url)
    page.click("button:has-text('Zuteilen')")
    page.select_option("#alloc-proj-select", str(project_id))
    page.locator("#alloc-person-select").select_option(index=1)
    page.click("#alloc-modal button:has-text('Speichern')")

    expect(page.locator("#toast-container")).to_contain_text("Zuteilung erstellt")


def test_add_allocation_to_material(
    page: Page, base_url: str, api: httpx.Client
) -> None:
    project_id = api.post(
        "/api/projects/", json={"name": "Material Projekt", "status": "geplant"}
    ).json()["id"]
    api.post(
        "/api/materials/",
        json={"name": "Test Material", "material_type": "Holz", "quantity": 10, "unit": "m³"},
    )

    _goto_allocations(page, base_url)
    page.click("button:has-text('Zuteilen')")
    page.select_option("#alloc-proj-select", str(project_id))
    page.locator("#alloc-mat-select").select_option(index=1)  # first material
    page.fill("#al-qty", "5")
    page.click("#alloc-modal button:has-text('Speichern')")
    page.wait_for_load_state("networkidle")

    expect(page.locator("#alloc-tbody")).to_contain_text("Test Material")


def test_filter_allocations_by_project(
    page: Page, base_url: str, api: httpx.Client
) -> None:
    proj_a_id = api.post(
        "/api/projects/", json={"name": "Projekt A", "status": "aktiv"}
    ).json()["id"]
    proj_b_id = api.post(
        "/api/projects/", json={"name": "Projekt B", "status": "aktiv"}
    ).json()["id"]
    person_id = api.post(
        "/api/persons/", json={"name": "Filter Person", "person_type": "Student"}
    ).json()["id"]
    api.post(
        "/api/allocations/",
        json={"project_id": proj_a_id, "person_id": person_id, "quantity": 1, "status": "geplant"},
    )
    api.post(
        "/api/allocations/",
        json={"project_id": proj_b_id, "person_id": person_id, "quantity": 1, "status": "geplant"},
    )

    _goto_allocations(page, base_url)
    # Filter by project A
    page.select_option("#alloc-project-filter", str(proj_a_id))
    page.wait_for_load_state("networkidle")

    rows = page.locator("#alloc-tbody tr")
    expect(rows).to_have_count(1)
    expect(page.locator("#alloc-tbody")).to_contain_text("Projekt A")
    expect(page.locator("#alloc-tbody")).not_to_contain_text("Projekt B")


def test_delete_allocation(
    page: Page, base_url: str, api: httpx.Client
) -> None:
    project_id, person_id = _setup_project_and_person(api)
    api.post(
        "/api/allocations/",
        json={"project_id": project_id, "person_id": person_id, "quantity": 1, "status": "geplant"},
    )

    _goto_allocations(page, base_url)
    expect(page.locator("#alloc-tbody")).to_contain_text("Alloc Projekt")

    page.on("dialog", lambda d: d.accept())
    page.click("#alloc-tbody button.danger")
    page.wait_for_load_state("networkidle")

    expect(page.locator("#alloc-tbody")).to_contain_text("Keine Zuteilungen gefunden")


def test_edit_allocation(
    page: Page, base_url: str, api: httpx.Client
) -> None:
    project_id, person_id = _setup_project_and_person(api)
    api.post(
        "/api/allocations/",
        json={"project_id": project_id, "person_id": person_id, "quantity": 1, "status": "geplant"},
    )

    _goto_allocations(page, base_url)
    page.click("#alloc-tbody button:first-of-type")  # edit button
    expect(page.locator("#alloc-modal")).to_be_visible()
    # Wait for the async openAllocModal to populate the form before modifying it
    expect(page.locator("#al-id")).not_to_have_value("")

    page.fill("#al-qty", "7")
    page.select_option("#alloc-status-select", "aktiv")
    page.click("#alloc-modal button:has-text('Speichern')")
    page.wait_for_load_state("networkidle")

    expect(page.locator("#alloc-tbody")).to_contain_text("7")
    expect(page.locator("#toast-container")).to_contain_text("Zuteilung aktualisiert")
