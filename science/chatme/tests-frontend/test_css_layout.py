"""
CSS Layout Tests – prüfen, dass alle visuellen Elemente korrekt gerendert werden.
"""

import pytest
from playwright.sync_api import Page, expect


def test_page_title(page_loaded: Page):
    assert page_loaded.title() == "ChatMe"


def test_css_variable_bg(page_loaded: Page):
    """Prüft ob das CSS-Design-Token --bg gesetzt ist."""
    bg = page_loaded.evaluate(
        "() => getComputedStyle(document.documentElement).getPropertyValue('--bg').trim()"
    )
    assert bg == "#0f1117"


def test_css_variable_accent(page_loaded: Page):
    accent = page_loaded.evaluate(
        "() => getComputedStyle(document.documentElement).getPropertyValue('--accent').trim()"
    )
    assert accent == "#2563eb"


def test_css_variable_surface(page_loaded: Page):
    surface = page_loaded.evaluate(
        "() => getComputedStyle(document.documentElement).getPropertyValue('--surface').trim()"
    )
    assert surface == "#1a1d27"


def test_css_variable_radius(page_loaded: Page):
    radius = page_loaded.evaluate(
        "() => getComputedStyle(document.documentElement).getPropertyValue('--radius').trim()"
    )
    assert radius == "12px"


def test_css_variable_danger(page_loaded: Page):
    danger = page_loaded.evaluate(
        "() => getComputedStyle(document.documentElement).getPropertyValue('--danger').trim()"
    )
    assert danger == "#dc2626"


def test_auth_screen_visible(page_loaded: Page):
    """Auth-Screen muss initial sichtbar sein."""
    auth_screen = page_loaded.locator("#authScreen")
    expect(auth_screen).to_be_visible()


def test_app_screen_hidden(page_loaded: Page):
    """App-Screen muss initial versteckt sein (kein Token)."""
    app_div = page_loaded.locator("#app")
    # display:none oder nicht sichtbar
    display = page_loaded.evaluate(
        "() => getComputedStyle(document.getElementById('app')).display"
    )
    assert display == "none"


def test_auth_card_exists(page_loaded: Page):
    card = page_loaded.locator(".auth-card")
    expect(card).to_be_visible()


def test_login_tab_active_by_default(page_loaded: Page):
    """Login-Tab ist standardmäßig aktiv."""
    active_tab = page_loaded.locator(".tab-btn.active")
    expect(active_tab).to_have_text("Anmelden")


def test_register_tab_exists(page_loaded: Page):
    tabs = page_loaded.locator(".tab-btn")
    assert tabs.count() == 2


def test_login_form_visible(page_loaded: Page):
    login_form = page_loaded.locator("#loginForm")
    expect(login_form).to_be_visible()


def test_register_form_hidden(page_loaded: Page):
    """Register-Form initial versteckt."""
    display = page_loaded.evaluate(
        "() => document.getElementById('registerForm').style.display"
    )
    assert display == "none"


def test_switch_to_register_tab(page_loaded: Page):
    """Klick auf Register-Tab zeigt das Registrierungsformular."""
    page_loaded.locator(".tab-btn").nth(1).click()
    register_form = page_loaded.locator("#registerForm")
    expect(register_form).to_be_visible()
    login_form = page_loaded.locator("#loginForm")
    expect(login_form).to_be_hidden()


def test_switch_back_to_login_tab(page_loaded: Page):
    """Klick auf Login-Tab nach Register-Wechsel."""
    page_loaded.locator(".tab-btn").nth(1).click()
    page_loaded.locator(".tab-btn").nth(0).click()
    login_form = page_loaded.locator("#loginForm")
    expect(login_form).to_be_visible()


def test_login_inputs_exist(page_loaded: Page):
    expect(page_loaded.locator("#loginUser")).to_be_visible()
    expect(page_loaded.locator("#loginPass")).to_be_visible()


def test_register_inputs_exist(page_loaded: Page):
    page_loaded.locator(".tab-btn").nth(1).click()
    expect(page_loaded.locator("#regUser")).to_be_visible()
    expect(page_loaded.locator("#regName")).to_be_visible()
    expect(page_loaded.locator("#regEmail")).to_be_visible()
    expect(page_loaded.locator("#regPass")).to_be_visible()


def test_auth_error_element_present(page_loaded: Page):
    assert page_loaded.locator("#authError").count() == 1


def test_call_overlay_hidden(page_loaded: Page):
    display = page_loaded.evaluate(
        "() => getComputedStyle(document.getElementById('callOverlay')).display"
    )
    assert display == "none"


def test_body_background_color(page_loaded: Page):
    bg = page_loaded.evaluate(
        "() => getComputedStyle(document.body).backgroundColor"
    )
    # rgb(15, 17, 23) = #0f1117
    assert "15" in bg or "0f1117" in bg.lower()
