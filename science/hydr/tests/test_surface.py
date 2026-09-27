"""Tests für hydr.surface – 100 % Abdeckung."""

import pytest

from hydr.surface import (
    laplace_pressure,
    cavitation_pressure,
    henry_gas_concentration,
    min_safe_pressure,
)


class TestLaplacePressure:
    def test_paper_example_1_micrometer(self):
        """Lt. Paper: r = 1 µm, γ = 0.029 N/m → ΔP_L ≈ 0.58 bar."""
        gamma = 0.029  # N/m (Mittelwert Mineralöl)
        r = 1e-6       # 1 µm
        result = laplace_pressure(gamma, r)
        assert result == pytest.approx(2 * gamma / r)
        # ≈ 58 000 Pa ≈ 0.58 bar
        assert 0.3e5 < result < 0.8e5

    def test_zero_radius_raises(self):
        with pytest.raises(ValueError):
            laplace_pressure(0.029, 0.0)

    def test_negative_radius_raises(self):
        with pytest.raises(ValueError):
            laplace_pressure(0.029, -1e-6)

    def test_negative_gamma_raises(self):
        with pytest.raises(ValueError):
            laplace_pressure(-0.1, 1e-6)

    def test_zero_gamma(self):
        # Kein Druck bei γ = 0
        assert laplace_pressure(0.0, 1e-6) == pytest.approx(0.0)

    def test_increases_with_smaller_radius(self):
        result_small = laplace_pressure(0.029, 1e-7)
        result_large = laplace_pressure(0.029, 1e-5)
        assert result_small > result_large


class TestCavitationPressure:
    def test_equals_vapor_plus_laplace(self):
        P_vapor = 100.0    # Pa
        gamma = 0.029
        r_keim = 1e-6
        result = cavitation_pressure(P_vapor, gamma, r_keim)
        expected = P_vapor + 2 * gamma / r_keim
        assert result == pytest.approx(expected)

    def test_always_greater_than_vapor_pressure(self):
        result = cavitation_pressure(100.0, 0.029, 1e-6)
        assert result > 100.0

    def test_zero_keim_radius_raises(self):
        with pytest.raises(ValueError):
            cavitation_pressure(100.0, 0.029, 0.0)


class TestHenryGasConcentration:
    def test_proportional_to_pressure(self):
        H = 9.5e-6  # vol%/Pa
        P1 = 1e5
        P2 = 2e5
        assert henry_gas_concentration(H, P2) == pytest.approx(
            2.0 * henry_gas_concentration(H, P1)
        )

    def test_zero_pressure_gives_zero(self):
        assert henry_gas_concentration(9.5e-6, 0.0) == pytest.approx(0.0)

    def test_known_value(self):
        """9.5 vol%/bar bei 1 bar = 9.5 vol% (mit H in vol%/bar, P in bar)."""
        H = 9.5   # vol%/bar
        P = 1.0   # bar
        assert henry_gas_concentration(H, P) == pytest.approx(9.5)


class TestMinSafePressure:
    def test_default_factor_3(self):
        P_vapor = 1000.0  # Pa
        result = min_safe_pressure(P_vapor)
        assert result == pytest.approx(3.0 * P_vapor)

    def test_custom_factor(self):
        result = min_safe_pressure(1000.0, safety_factor=5.0)
        assert result == pytest.approx(5000.0)

    def test_zero_vapor_pressure(self):
        assert min_safe_pressure(0.0) == pytest.approx(0.0)
