"""Tests für App-Konfiguration."""

import pytest
from app.core.config import settings


def test_settings_defaults():
    assert settings.ALGORITHM == "HS256"
    assert settings.ACCESS_TOKEN_EXPIRE_MINUTES > 0


def test_settings_has_secret_key():
    assert settings.SECRET_KEY is not None
    assert len(settings.SECRET_KEY) > 0


def test_settings_database_url():
    assert "postgresql" in settings.DATABASE_URL or "sqlite" in settings.DATABASE_URL


def test_settings_redis_url():
    assert settings.REDIS_URL.startswith("redis://")


def test_settings_cors_origins():
    assert settings.CORS_ORIGINS is not None
