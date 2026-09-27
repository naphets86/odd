"""Tests für hydr.viscoelastic – 100 % Abdeckung."""

import math
import pytest

from hydr.viscoelastic import (
    storage_modulus,
    loss_modulus,
    loss_factor,
    complex_modulus_magnitude,
)


E0 = 1.58e9      # statischer Modul [Pa]
E_INF = 1.62e9   # Hochfrequenz-Modul [Pa]
TAU_R = 18e-9    # Relaxationszeit [s], Mineralöl VG 46


class TestStorageModulus:
    def test_low_frequency_approaches_E0(self):
        """Bei ω → 0: K'(ω) → E0."""
        result = storage_modulus(E0, E_INF, omega=1.0, tau_R=TAU_R)
        assert result == pytest.approx(E0, rel=1e-6)

    def test_high_frequency_approaches_E_inf(self):
        """Bei ω τ_R >> 1: K'(ω) → E_inf."""
        omega_high = 1e15  # Hochfrequenz
        result = storage_modulus(E0, E_INF, omega_high, TAU_R)
        assert result == pytest.approx(E_INF, rel=1e-4)

    def test_at_relaxation_frequency_is_midpoint(self):
        """Bei ω τ_R = 1: K' = E0 + (E_inf - E0) * 0.5."""
        omega = 1.0 / TAU_R
        expected = E0 + (E_INF - E0) * 0.5
        assert storage_modulus(E0, E_INF, omega, TAU_R) == pytest.approx(expected)

    def test_equal_moduli_returns_E0(self):
        """Wenn E0 = E_inf, ist K' = E0 für jede Frequenz."""
        result = storage_modulus(1.58e9, 1.58e9, 1e6 / TAU_R, TAU_R)
        assert result == pytest.approx(1.58e9)

    def test_zero_omega(self):
        result = storage_modulus(E0, E_INF, 0.0, TAU_R)
        assert result == pytest.approx(E0)


class TestLossModulus:
    def test_zero_at_low_frequency(self):
        """Bei ω → 0: K''(ω) → 0."""
        result = loss_modulus(E0, E_INF, omega=1.0, tau_R=TAU_R)
        assert abs(result) < 1e3  # sehr klein im Vergleich zu K'

    def test_zero_at_high_frequency(self):
        """Bei ω τ_R → ∞: K''(ω) → 0."""
        result = loss_modulus(E0, E_INF, omega=1e20, tau_R=TAU_R)
        assert abs(result) < 1.0

    def test_maximum_at_relaxation_frequency(self):
        """K'' ist maximal bei ω τ_R = 1."""
        omega_max = 1.0 / TAU_R
        k_pp_max = loss_modulus(E0, E_INF, omega_max, TAU_R)
        k_pp_low = loss_modulus(E0, E_INF, omega_max / 10, TAU_R)
        k_pp_high = loss_modulus(E0, E_INF, omega_max * 10, TAU_R)
        assert k_pp_max > k_pp_low
        assert k_pp_max > k_pp_high

    def test_equal_moduli_gives_zero(self):
        """Falls E0 = E_inf, kein Verlust."""
        result = loss_modulus(1.58e9, 1.58e9, 1e6, TAU_R)
        assert result == pytest.approx(0.0)


class TestLossFactor:
    def test_low_frequency_tan_delta_near_zero(self):
        """Bei niedriger Frequenz: tan δ → 0 (elastisch dominiert)."""
        result = loss_factor(E0, E_INF, omega=1.0, tau_R=TAU_R)
        assert result < 1e-3

    def test_high_frequency_tan_delta_near_zero(self):
        """Bei sehr hoher Frequenz: wieder tan δ → 0."""
        result = loss_factor(E0, E_INF, omega=1e20, tau_R=TAU_R)
        assert abs(result) < 1e-3

    def test_equal_moduli_zero_denominator_raises(self):
        """Wenn K' theoretisch null würde, ZeroDivisionError.
        Hier: E0 = E_inf = 0 (pathologischer Fall) indirekt über K'=0."""
        # storage_modulus bei E0=E_inf=0 → K' = 0
        with pytest.raises(ZeroDivisionError):
            loss_factor(0.0, 0.0, 1.0, TAU_R)

    def test_is_ratio(self):
        omega = 1.0 / TAU_R
        kp = storage_modulus(E0, E_INF, omega, TAU_R)
        kpp = loss_modulus(E0, E_INF, omega, TAU_R)
        assert loss_factor(E0, E_INF, omega, TAU_R) == pytest.approx(kpp / kp)


class TestComplexModulusMagnitude:
    def test_at_low_frequency_is_E0(self):
        """Bei ω → 0: |K*| → E0."""
        result = complex_modulus_magnitude(E0, E_INF, omega=1.0, tau_R=TAU_R)
        assert result == pytest.approx(E0, rel=1e-5)

    def test_at_high_frequency_is_E_inf(self):
        result = complex_modulus_magnitude(E0, E_INF, omega=1e20, tau_R=TAU_R)
        assert result == pytest.approx(E_INF, rel=1e-4)

    def test_always_geq_storage_modulus(self):
        omega = 1.0 / TAU_R
        kp = storage_modulus(E0, E_INF, omega, TAU_R)
        mag = complex_modulus_magnitude(E0, E_INF, omega, TAU_R)
        assert mag >= kp

    def test_pythagorean_relation(self):
        omega = 1.0 / TAU_R
        kp = storage_modulus(E0, E_INF, omega, TAU_R)
        kpp = loss_modulus(E0, E_INF, omega, TAU_R)
        expected = math.hypot(kp, kpp)
        assert complex_modulus_magnitude(E0, E_INF, omega, TAU_R) == pytest.approx(expected)
