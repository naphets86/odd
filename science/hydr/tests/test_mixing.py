"""Tests für hydr.mixing – 100 % Abdeckung."""

import pytest

from hydr.mixing import (
    reuss_modulus,
    maxwell_emulsion_modulus,
    voigt_modulus,
)


K_OIL = 1.58e9   # Mineralöl VG 46 [Pa]
K_WATER = 2.19e9  # Wasser [Pa]


class TestReussModulus:
    def test_pure_phase1(self):
        """phi2 = 0 → K_eff = K1."""
        result = reuss_modulus(K_OIL, K_WATER, phi2=0.0)
        assert result == pytest.approx(K_OIL)

    def test_pure_phase2(self):
        """phi2 = 1 → K_eff = K2."""
        result = reuss_modulus(K_OIL, K_WATER, phi2=1.0)
        assert result == pytest.approx(K_WATER)

    def test_always_less_than_voigt(self):
        """Reuss ist untere Schranke, Voigt obere."""
        phi = 0.3
        r = reuss_modulus(K_OIL, K_WATER, phi)
        v = voigt_modulus(K_OIL, K_WATER, phi)
        assert r <= v

    def test_intermediate_is_between_K1_K2(self):
        result = reuss_modulus(K_OIL, K_WATER, 0.5)
        assert K_OIL <= result <= K_WATER

    def test_invalid_phi_low_raises(self):
        with pytest.raises(ValueError):
            reuss_modulus(K_OIL, K_WATER, -0.1)

    def test_invalid_phi_high_raises(self):
        with pytest.raises(ValueError):
            reuss_modulus(K_OIL, K_WATER, 1.1)

    def test_zero_K1_raises(self):
        with pytest.raises(ValueError):
            reuss_modulus(0.0, K_WATER, 0.3)

    def test_zero_K2_raises(self):
        with pytest.raises(ValueError):
            reuss_modulus(K_OIL, 0.0, 0.3)

    def test_negative_K_raises(self):
        with pytest.raises(ValueError):
            reuss_modulus(-1.0, K_WATER, 0.3)


class TestMaxwellEmulsionModulus:
    def test_pure_water_phase(self):
        """phi = 0 → K_HFAE = K_W."""
        result = maxwell_emulsion_modulus(K_WATER, K_OIL, phi=0.0)
        assert result == pytest.approx(K_WATER)

    def test_5_percent_oil_close_to_water(self):
        """HFAE mit 5 % Öl: Modul nahe Wasser."""
        result = maxwell_emulsion_modulus(K_WATER, K_OIL, phi=0.05)
        assert result > K_OIL
        assert result < K_WATER + 0.01e9  # nicht mehr als 1 % über Wasser

    def test_oil_lower_than_water_reduces_modulus(self):
        r0 = maxwell_emulsion_modulus(K_WATER, K_OIL, phi=0.0)
        r5 = maxwell_emulsion_modulus(K_WATER, K_OIL, phi=0.05)
        r10 = maxwell_emulsion_modulus(K_WATER, K_OIL, phi=0.10)
        assert r0 >= r5 >= r10

    def test_invalid_phi_raises(self):
        with pytest.raises(ValueError):
            maxwell_emulsion_modulus(K_WATER, K_OIL, phi=-0.1)
        with pytest.raises(ValueError):
            maxwell_emulsion_modulus(K_WATER, K_OIL, phi=1.1)

    def test_zero_K_W_raises(self):
        with pytest.raises(ValueError):
            maxwell_emulsion_modulus(0.0, K_OIL, phi=0.05)

    def test_zero_K_O_raises(self):
        with pytest.raises(ValueError):
            maxwell_emulsion_modulus(K_WATER, 0.0, phi=0.05)


class TestVoigtModulus:
    def test_pure_phase1(self):
        result = voigt_modulus(K_OIL, K_WATER, phi2=0.0)
        assert result == pytest.approx(K_OIL)

    def test_pure_phase2(self):
        result = voigt_modulus(K_OIL, K_WATER, phi2=1.0)
        assert result == pytest.approx(K_WATER)

    def test_linear_mixture(self):
        phi = 0.4
        expected = (1.0 - phi) * K_OIL + phi * K_WATER
        assert voigt_modulus(K_OIL, K_WATER, phi) == pytest.approx(expected)

    def test_invalid_phi_raises(self):
        with pytest.raises(ValueError):
            voigt_modulus(K_OIL, K_WATER, -0.01)
        with pytest.raises(ValueError):
            voigt_modulus(K_OIL, K_WATER, 1.01)
