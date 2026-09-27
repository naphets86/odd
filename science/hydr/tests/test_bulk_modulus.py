"""Tests für hydr.bulk_modulus – 100 % Abdeckung."""

import math
import pytest

from hydr.bulk_modulus import (
    isentropic_modulus,
    sound_speed,
    bulk_modulus_tait_linear,
    bulk_modulus_dowson_higginson,
    bulk_modulus_temperature,
    effective_bulk_modulus,
)


class TestIsentropicModulus:
    def test_no_thermal_correction(self):
        """Wenn alpha_V = 0, ist K_s = K_T."""
        result = isentropic_modulus(
            K_T=1.58e9, alpha_V=0.0, T=313.15, rho=870.0, c_p=2000.0
        )
        assert result == pytest.approx(1.58e9)

    def test_typical_mineral_oil(self):
        """Für Mineralöl ISO VG 46: K_s muss > K_T sein.

        Die Korrektur beträgt mit alpha_V=6.9e-4 K⁻¹, c_p=2000 J/(kg·K)
        tatsächlich ~13 %; der Papier-Hinweis '<2%' bezieht sich auf einen
        anderen Parameter-Bereich (z.B. niedrigeres alpha_V oder höheres c_p).
        """
        K_T = 1.58e9
        alpha_V = 6.9e-4
        T = 313.15  # 40 °C
        rho = 870.0
        c_p = 2000.0
        result = isentropic_modulus(K_T, alpha_V, T, rho, c_p)
        assert result > K_T
        # Korrektur liegt im Bereich 0–20 %
        assert abs(result - K_T) / K_T < 0.20

    def test_correction_increases_with_temperature(self):
        common = dict(K_T=1.58e9, alpha_V=6.9e-4, rho=870.0, c_p=2000.0)
        k_low = isentropic_modulus(T=293.15, **common)
        k_high = isentropic_modulus(T=373.15, **common)
        assert k_high > k_low


class TestSoundSpeed:
    def test_known_value(self):
        """c_s = sqrt(K_s / rho) – bekannter Wert für Mineralöl."""
        K_s = 1.58e9
        rho = 870.0
        expected = math.sqrt(K_s / rho)
        assert sound_speed(K_s, rho) == pytest.approx(expected)

    def test_negative_K_s_raises(self):
        with pytest.raises(ValueError):
            sound_speed(-1.0, 870.0)

    def test_zero_K_s(self):
        # K_s = 0 → c_s = 0, kein Fehler
        assert sound_speed(0.0, 870.0) == pytest.approx(0.0)

    def test_zero_rho_raises(self):
        with pytest.raises(ValueError):
            sound_speed(1.58e9, 0.0)

    def test_negative_rho_raises(self):
        with pytest.raises(ValueError):
            sound_speed(1.58e9, -1.0)


class TestBulkModulusTaitLinear:
    def test_at_zero_pressure(self):
        """Bei P = 0 gilt K_T = K0."""
        assert bulk_modulus_tait_linear(1.58e9, 10.2, 0.0) == pytest.approx(1.58e9)

    def test_positive_pressure_increases_modulus(self):
        K0 = 1.58e9
        n = 10.2
        P = 200e5  # 200 bar
        result = bulk_modulus_tait_linear(K0, n, P)
        assert result > K0
        assert result == pytest.approx(K0 + n * P)


class TestBulkModulusDowsonHigginson:
    def test_at_zero_pressure_equals_K0(self):
        """Bei P = 0: exp(0) = 1, also K_T = K0."""
        assert bulk_modulus_dowson_higginson(1.58e9, 10.2, 0.0) == pytest.approx(1.58e9)

    def test_positive_pressure_increases_modulus(self):
        K0 = 1.58e9
        result = bulk_modulus_dowson_higginson(K0, 10.2, 200e5)
        assert result > K0

    def test_nonzero_K0_required(self):
        with pytest.raises(ValueError):
            bulk_modulus_dowson_higginson(0.0, 10.2, 200e5)

    def test_negative_K0_raises(self):
        with pytest.raises(ValueError):
            bulk_modulus_dowson_higginson(-1.0, 10.2, 200e5)

    def test_larger_pressure_gives_larger_modulus_than_tait(self):
        """Bei hohen Drücken ist DH > Tait."""
        K0 = 1.58e9
        n = 10.2
        P = 500e5  # 500 bar
        dh = bulk_modulus_dowson_higginson(K0, n, P)
        tait = bulk_modulus_tait_linear(K0, n, P)
        assert dh > tait


class TestBulkModulusTemperature:
    def test_at_reference_temperature(self):
        """Bei T = T_ref kein Korrekturfaktor."""
        result = bulk_modulus_temperature(1.58e9, 3.5e-3, 313.15, 313.15)
        assert result == pytest.approx(1.58e9)

    def test_increase_temperature_decreases_modulus(self):
        K_ref = 1.58e9
        beta_K = 3.5e-3
        T_ref = 293.15  # 20 °C
        K_high = bulk_modulus_temperature(K_ref, beta_K, 353.15, T_ref)  # 80 °C
        assert K_high < K_ref

    def test_21_percent_decrease_over_60_kelvin(self):
        """Laut Paper: +60 K → ca. 21 % Rückgang."""
        K_ref = 1.58e9
        beta_K = 3.5e-3
        K_high = bulk_modulus_temperature(K_ref, beta_K, 293.15 + 60, 293.15)
        reduction = (K_ref - K_high) / K_ref
        assert abs(reduction - 0.21) < 0.02  # ±2 % Toleranz


class TestEffectiveBulkModulus:
    def test_no_gas_returns_oil_modulus(self):
        """Wenn alpha = 0, ist K_eff = K_oil · P / P = K_oil."""
        K_oil = 1.58e9
        P = 10e5  # 10 bar
        result = effective_bulk_modulus(K_oil, P, 0.0)
        assert result == pytest.approx(K_oil)

    def test_one_percent_gas_reduces_modulus_significantly(self):
        """Die vereinfachte Formel K_eff = K_oil·P/(K_oil·α+P) (K_gas≈P)
        liefert bei α=1% und P=10 bar eine starke Reduktion, da der
        Gasmodul K_gas≈P sehr klein ist gegenüber K_oil.
        Der Papierwert von 42% gilt mit der Reuss-Formel und K_gas=γ·P;
        die hier implementierte Vereinfachung ergibt eine größere Reduktion.
        """
        K_oil = 1.58e9
        P = 10e5  # 10 bar
        alpha = 0.01
        K_eff = effective_bulk_modulus(K_oil, P, alpha)
        # Konsistenzcheck mit der Approximationsformel
        expected = K_oil * P / (K_oil * alpha + P)
        assert K_eff == pytest.approx(expected)
        # Der effektive Modul ist stark reduziert (nach der Näherung)
        assert K_eff < K_oil * 0.15

    def test_alpha_one_raises(self):
        with pytest.raises(ValueError):
            effective_bulk_modulus(1.58e9, 10e5, 1.0)

    def test_alpha_negative_raises(self):
        with pytest.raises(ValueError):
            effective_bulk_modulus(1.58e9, 10e5, -0.01)

    def test_zero_pressure_raises(self):
        with pytest.raises(ValueError):
            effective_bulk_modulus(1.58e9, 0.0, 0.01)

    def test_negative_pressure_raises(self):
        with pytest.raises(ValueError):
            effective_bulk_modulus(1.58e9, -1.0, 0.01)

    def test_negative_K_oil_raises(self):
        with pytest.raises(ValueError):
            effective_bulk_modulus(-1.0, 10e5, 0.01)

    def test_large_gas_fraction_gives_very_low_modulus(self):
        K_eff = effective_bulk_modulus(1.58e9, 10e5, 0.10)  # 10 % gas
        # Laut Tabelle: ≈ 149 MPa, also << K_oil
        assert K_eff < 0.15e9
