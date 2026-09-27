"""Persons page – CRUD via UI."""
import re

import httpx
from playwright.sync_api import Page, expect


def _goto_persons(page: Page, base_url: str) -> None:
    page.goto(base_url)
    page.click('nav a[data-page="persons"]')
    page.wait_for_load_state("networkidle")


def test_persons_empty_state(page: Page, base_url: str) -> None:
    _goto_persons(page, base_url)
    expect(page.locator("#persons-tbody")).to_contain_text("Noch keine Personen erfasst")


def test_add_person_appears_in_table(page: Page, base_url: str) -> None:
    _goto_persons(page, base_url)

    page.click("button:has-text('Person hinzufügen')")
    expect(page.locator("#person-modal")).to_be_visible()

    page.fill("#pm-name", "Maria Muster")
    page.select_option("#person-type-select", "Arbeitnehmer")
    page.fill("#pm-email", "maria@example.com")
    page.fill("#pm-dept", "IT")
    page.click("#person-modal button:has-text('Speichern')")
    page.wait_for_load_state("networkidle")

    expect(page.locator("#persons-tbody")).to_contain_text("Maria Muster")
    expect(page.locator("#persons-tbody")).to_contain_text("IT")


def test_add_person_shows_success_toast(page: Page, base_url: str) -> None:
    _goto_persons(page, base_url)

    page.click("button:has-text('Person hinzufügen')")
    page.fill("#pm-name", "Toast Tester")
    page.select_option("#person-type-select", "Student")
    page.click("#person-modal button:has-text('Speichern')")

    expect(page.locator("#toast-container")).to_contain_text("Person hinzugefügt")


def test_person_modal_closes_after_save(page: Page, base_url: str) -> None:
    _goto_persons(page, base_url)

    page.click("button:has-text('Person hinzufügen')")
    page.fill("#pm-name", "Modal Test")
    page.select_option("#person-type-select", "Lehrer")
    page.click("#person-modal button:has-text('Speichern')")
    page.wait_for_load_state("networkidle")

    expect(page.locator("#person-modal")).not_to_have_class(re.compile(r"\bopen\b"))


def test_add_person_modal_cancel(page: Page, base_url: str) -> None:
    _goto_persons(page, base_url)

    page.click("button:has-text('Person hinzufügen')")
    page.fill("#pm-name", "Soll nicht gespeichert werden")
    page.click("#person-modal button:has-text('Abbrechen')")
    page.wait_for_load_state("networkidle")

    expect(page.locator("#persons-tbody")).not_to_contain_text("Soll nicht gespeichert werden")


def test_edit_person(page: Page, base_url: str, api: httpx.Client) -> None:
    api.post("/api/persons/", json={"name": "Alt Name", "person_type": "Student"})

    _goto_persons(page, base_url)
    expect(page.locator("#persons-tbody")).to_contain_text("Alt Name")

    page.click("#persons-tbody button[title='Bearbeiten']")
    expect(page.locator("#person-modal")).to_be_visible()
    expect(page.locator("#pm-name")).to_have_value("Alt Name")

    page.fill("#pm-name", "Neuer Name")
    page.click("#person-modal button:has-text('Speichern')")
    page.wait_for_load_state("networkidle")

    expect(page.locator("#persons-tbody")).to_contain_text("Neuer Name")
    expect(page.locator("#persons-tbody")).not_to_contain_text("Alt Name")


def test_delete_person(page: Page, base_url: str, api: httpx.Client) -> None:
    api.post("/api/persons/", json={"name": "Zu Löschende Person", "person_type": "Professor"})

    _goto_persons(page, base_url)
    expect(page.locator("#persons-tbody")).to_contain_text("Zu Löschende Person")

    page.on("dialog", lambda d: d.accept())
    page.click("#persons-tbody button.danger")
    page.wait_for_load_state("networkidle")

    expect(page.locator("#persons-tbody")).not_to_contain_text("Zu Löschende Person")
