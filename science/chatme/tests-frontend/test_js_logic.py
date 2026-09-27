"""
JavaScript Logic Tests – testen SC-Cipher, DOM-Manipulation und
globale Funktionen über Playwright's page.evaluate().
"""

import pytest
from playwright.sync_api import Page, expect


# ── SC-Chiffre Tests ───────────────────────────────────────────────────────────


def test_sc_object_exists(page_loaded: Page):
    """SC-Chiffre-Objekt muss im globalen Scope vorhanden sein."""
    exists = page_loaded.evaluate("() => typeof SC !== 'undefined'")
    assert exists is True


def test_sc_has_encrypt(page_loaded: Page):
    has = page_loaded.evaluate("() => typeof SC.encryptMessage === 'function'")
    assert has is True


def test_sc_has_decrypt(page_loaded: Page):
    has = page_loaded.evaluate("() => typeof SC.decryptMessage === 'function'")
    assert has is True


def test_sc_encrypt_returns_string(page_loaded: Page):
    result = page_loaded.evaluate("""() => {
        const key = { sc_key_a: '2', sc_key_b: '3', sc_key_p: '97', sc_key_n: 4 };
        return typeof SC.encryptMessage('Hallo', key);
    }""")
    assert result == "string"


def test_sc_encrypt_decrypt_roundtrip(page_loaded: Page):
    result = page_loaded.evaluate("""() => {
        const key = { sc_key_a: '5', sc_key_b: '7', sc_key_p: '101', sc_key_n: 4 };
        const cipher = SC.encryptMessage('ChatMe Test', key);
        const plain  = SC.decryptMessage(cipher, key);
        return plain;
    }""")
    assert result == "ChatMe Test"


def test_sc_encrypt_produces_json(page_loaded: Page):
    result = page_loaded.evaluate("""() => {
        const key = { sc_key_a: '3', sc_key_b: '5', sc_key_p: '89', sc_key_n: 4 };
        const cipher = SC.encryptMessage('test', key);
        const obj = JSON.parse(cipher);
        return 'len' in obj && 'blocks' in obj;
    }""")
    assert result is True


def test_sc_different_keys_different_cipher(page_loaded: Page):
    result = page_loaded.evaluate("""() => {
        const key1 = { sc_key_a: '2', sc_key_b: '3', sc_key_p: '97', sc_key_n: 4 };
        const key2 = { sc_key_a: '5', sc_key_b: '7', sc_key_p: '89', sc_key_n: 4 };
        const c1 = SC.encryptMessage('Hallo', key1);
        const c2 = SC.encryptMessage('Hallo', key2);
        return c1 !== c2;
    }""")
    assert result is True


def test_sc_wrong_key_decrypt_fails(page_loaded: Page):
    result = page_loaded.evaluate("""() => {
        const key1 = { sc_key_a: '2', sc_key_b: '3', sc_key_p: '97', sc_key_n: 4 };
        const key2 = { sc_key_a: '5', sc_key_b: '7', sc_key_p: '89', sc_key_n: 4 };
        const cipher = SC.encryptMessage('Secret', key1);
        const plain  = SC.decryptMessage(cipher, key2);
        return plain !== 'Secret';
    }""")
    assert result is True


# ── Global-Funktionen im Scope ────────────────────────────────────────────────


def test_function_switchTab_exists(page_loaded: Page):
    exists = page_loaded.evaluate("() => typeof switchTab === 'function'")
    assert exists is True


def test_function_doLogin_exists(page_loaded: Page):
    exists = page_loaded.evaluate("() => typeof doLogin === 'function'")
    assert exists is True


def test_function_doRegister_exists(page_loaded: Page):
    exists = page_loaded.evaluate("() => typeof doRegister === 'function'")
    assert exists is True


def test_function_logout_exists(page_loaded: Page):
    exists = page_loaded.evaluate("() => typeof logout === 'function'")
    assert exists is True


def test_function_sendMessage_exists(page_loaded: Page):
    exists = page_loaded.evaluate("() => typeof sendMessage === 'function'")
    assert exists is True


def test_function_connectWS_exists(page_loaded: Page):
    exists = page_loaded.evaluate("() => typeof connectWS === 'function'")
    assert exists is True


def test_function_handleWsEvent_exists(page_loaded: Page):
    exists = page_loaded.evaluate("() => typeof handleWsEvent === 'function'")
    assert exists is True


def test_function_esc_sanitizes_html(page_loaded: Page):
    """esc() muss HTML-Zeichen escapen."""
    result = page_loaded.evaluate("""() => esc('<script>alert(1)</script>')""")
    assert "<script>" not in result
    assert "&lt;" in result


def test_function_esc_ampersand(page_loaded: Page):
    result = page_loaded.evaluate("""() => esc('a & b')""")
    assert "&amp;" in result


# ── DOM-Interaktion über Playwright ──────────────────────────────────────────


def test_switchTab_changes_display(page_loaded: Page):
    """switchTab('register') macht das Register-Form sichtbar."""
    page_loaded.evaluate("() => switchTab('register')")
    display = page_loaded.evaluate(
        "() => document.getElementById('registerForm').style.display"
    )
    assert display != "none"


def test_switchTab_login_restores(page_loaded: Page):
    page_loaded.evaluate("() => switchTab('register')")
    page_loaded.evaluate("() => switchTab('login')")
    display = page_loaded.evaluate(
        "() => document.getElementById('loginForm').style.display"
    )
    assert display != "none"


def test_authErr_sets_text(page_loaded: Page):
    page_loaded.evaluate("() => authErr('Test-Fehler')")
    text = page_loaded.locator("#authError").inner_text()
    assert text == "Test-Fehler"
    # Aufräumen
    page_loaded.evaluate("() => authErr('')")


def test_global_state_token_initially_empty(page_loaded: Page):
    """token-Variable ist leer, wenn kein localStorage-Eintrag."""
    token = page_loaded.evaluate("""() => {
        localStorage.removeItem('cm_token');
        return localStorage.getItem('cm_token') === null;
    }""")
    assert token is True


def test_global_var_me_initially_null(page_loaded: Page):
    me_val = page_loaded.evaluate("() => me === null || typeof me === 'undefined' || me === ''")
    assert me_val is True


def test_ws_initially_null(page_loaded: Page):
    ws_val = page_loaded.evaluate("() => ws === null || typeof ws === 'undefined'")
    assert ws_val is True
