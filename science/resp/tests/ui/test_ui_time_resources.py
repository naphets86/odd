"""Time resources page – CRUD via UI."""
import httpx
from playwright.sync_api import Page, expect


def _goto_time(page: Page, base_url: str) -> None:
    page.goto(base_url)
    page.click('nav a[data-page="time"]')
    page.wait_for_load_state("networkidle")


def test_time_resources_empty_state(page: Page, base_url: str) -> None:
    _goto_time(page, base_url)
    expect(page.locator("#time-tbody")).to_contain_text("Noch keine Zeitressourcen erfasst")


def test_add_time_resource_appears_in_table(page: Page, base_url: str) -> None:
    _goto_time(page, base_url)

    page.click("button:has-text('Zeit hinzufügen')")
    expect(page.locator("#time-modal")).to_be_visible()

    page.fill("#tm-name", "Projektsprint Q1")
    page.fill("#tm-amount", "40")
    page.select_option("#time-unit-select", "Stunden")
    page.fill("#tm-notes", "Erster Sprint")
    page.click("#time-modal button:has-text('Speichern')")
    page.wait_for_load_state("networkidle")

    expect(page.locator("#time-tbody")).to_contain_text("Projektsprint Q1")
    expect(page.locator("#time-tbody")).to_contain_text("40")


def test_add_time_resource_shows_toast(page: Page, base_url: str) -> None:
    _goto_time(page, base_url)

    page.click("button:has-text('Zeit hinzufügen')")
    page.fill("#tm-name", "Toast Zeit")
    page.fill("#tm-amount", "8")
    page.select_option("#time-unit-select", "Tage")
    page.click("#time-modal button:has-text('Speichern')")

    expect(page.locator("#toast-container")).to_contain_text("Zeitressource hinzugefügt")


def test_edit_time_resource(page: Page, base_url: str, api: httpx.Client) -> None:
    api.post(
        "/api/time-resources/",
        json={"name": "Alte Zeitressource", "amount": 5, "unit": "Tage"},
    )

    _goto_time(page, base_url)
    page.click("#time-tbody button:first-of-type")  # edit button
    expect(page.locator("#time-modal")).to_be_visible()
    expect(page.locator("#tm-name")).to_have_value("Alte Zeitressource")

    page.fill("#tm-name", "Neue Zeitressource")
    page.fill("#tm-amount", "10")
    page.click("#time-modal button:has-text('Speichern')")
    page.wait_for_load_state("networkidle")

    expect(page.locator("#time-tbody")).to_contain_text("Neue Zeitressource")


def test_delete_time_resource(page: Page, base_url: str, api: httpx.Client) -> None:
    api.post(
        "/api/time-resources/",
        json={"name": "Zeit zum Löschen", "amount": 1, "unit": "Stunden"},
    )

    _goto_time(page, base_url)
    expect(page.locator("#time-tbody")).to_contain_text("Zeit zum Löschen")

    page.on("dialog", lambda d: d.accept())
    page.click("#time-tbody button.danger")
    page.wait_for_load_state("networkidle")

    expect(page.locator("#time-tbody")).not_to_contain_text("Zeit zum Löschen")
