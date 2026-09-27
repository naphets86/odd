"""Materials page – CRUD via UI."""
import httpx
from playwright.sync_api import Page, expect


def _goto_materials(page: Page, base_url: str) -> None:
    page.goto(base_url)
    page.click('nav a[data-page="materials"]')
    page.wait_for_load_state("networkidle")


def test_materials_empty_state(page: Page, base_url: str) -> None:
    _goto_materials(page, base_url)
    expect(page.locator("#materials-tbody")).to_contain_text("Noch kein Material erfasst")


def test_add_material_appears_in_table(page: Page, base_url: str) -> None:
    _goto_materials(page, base_url)

    page.click("button:has-text('Material hinzufügen')")
    expect(page.locator("#material-modal")).to_be_visible()

    page.fill("#mm-name", "Testkupfer")
    page.select_option("#material-type-select", "Kupfer")
    page.fill("#mm-qty", "50")
    page.fill("#mm-unit", "kg")
    page.fill("#mm-loc", "Lager B")
    page.click("#material-modal button:has-text('Speichern')")
    page.wait_for_load_state("networkidle")

    expect(page.locator("#materials-tbody")).to_contain_text("Testkupfer")
    expect(page.locator("#materials-tbody")).to_contain_text("Lager B")


def test_add_material_shows_toast(page: Page, base_url: str) -> None:
    _goto_materials(page, base_url)

    page.click("button:has-text('Material hinzufügen')")
    page.fill("#mm-name", "Toast Material")
    page.select_option("#material-type-select", "Holz")
    page.click("#material-modal button:has-text('Speichern')")

    expect(page.locator("#toast-container")).to_contain_text("Material hinzugefügt")


def test_edit_material(page: Page, base_url: str, api: httpx.Client) -> None:
    api.post(
        "/api/materials/",
        json={"name": "Altes Material", "material_type": "Stahl", "quantity": 10, "unit": "kg"},
    )

    _goto_materials(page, base_url)
    page.click("#materials-tbody button[title='Bearbeiten']")
    expect(page.locator("#material-modal")).to_be_visible()
    expect(page.locator("#mm-name")).to_have_value("Altes Material")

    page.fill("#mm-name", "Neues Material")
    page.click("#material-modal button:has-text('Speichern')")
    page.wait_for_load_state("networkidle")

    expect(page.locator("#materials-tbody")).to_contain_text("Neues Material")


def test_delete_material(page: Page, base_url: str, api: httpx.Client) -> None:
    api.post(
        "/api/materials/",
        json={"name": "Zu löschen", "material_type": "Geld", "quantity": 1, "unit": "€"},
    )

    _goto_materials(page, base_url)
    expect(page.locator("#materials-tbody")).to_contain_text("Zu löschen")

    page.on("dialog", lambda d: d.accept())
    page.click("#materials-tbody button.danger")
    page.wait_for_load_state("networkidle")

    expect(page.locator("#materials-tbody")).not_to_contain_text("Zu löschen")
