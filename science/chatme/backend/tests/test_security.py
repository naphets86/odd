"""Tests für Security-Hilfsfunktionen."""

import pytest
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_token,
)


def test_hash_and_verify_password():
    pw = "SuperSecret42!"
    hashed = hash_password(pw)
    assert hashed != pw
    assert verify_password(pw, hashed)


def test_verify_wrong_password():
    hashed = hash_password("correct")
    assert not verify_password("wrong", hashed)


def test_create_and_decode_token():
    data = {"sub": "user-uuid-123"}
    token = create_access_token(data)
    decoded = decode_token(token)
    assert decoded is not None
    assert decoded["sub"] == "user-uuid-123"


def test_decode_invalid_token():
    result = decode_token("not.a.valid.jwt")
    assert result is None


def test_decode_empty_string():
    result = decode_token("")
    assert result is None


def test_token_expiry_field():
    import time
    data = {"sub": "abc"}
    token = create_access_token(data)
    decoded = decode_token(token)
    assert "exp" in decoded
    assert decoded["exp"] > time.time()
